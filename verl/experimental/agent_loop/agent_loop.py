# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Agent framework for multi-turn rollout and agentic reinforcement learning.
- AgentLoopBase: coroutine based abstract base class for agent loop.
  - SingleTurnAgentLoop: single turn agent loop.
  - ToolAgentLoop: ReAct agent loop with tool calling, with user defined tools.
- AgentLoopWorker: worker class for running agent loop coroutines in parallel.
- AgentLoopManager: manager class for running agent loop workers in parallel.

AgentLoopManager is one specific agent-framework implementation in verl,
and is designed to be fully replaceable by other agent frameworks such as:
- NVIDIA Nemo-Gym
- AWS Bedrock AgentCore
- SWE-agent
- ...
"""

import asyncio
import hashlib
import logging
import os
import random
from abc import ABC, abstractmethod
from typing import Any, Optional
from uuid import uuid4

import hydra
import numpy as np
import ray
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from pydantic import BaseModel, ConfigDict
from tensordict import TensorDict
from transformers import AutoProcessor, AutoTokenizer

from verl.trainer.distillation.omniopd_producer import build_chunk_requests
from verl.trainer.distillation.omniopd_speculation import (SpeculativeStore, propose_anchors,
                                                           seed_for_anchor, speculation_enabled)
from verl.trainer.distillation.omniopd_stage import (attach_omniopd_audit, omniopd_enabled,
                                                      resolve_omniopd_config)
from verl.experimental.agent_loop.utils import resolve_config_path
from verl.protocol import DataProto
from verl.tools.tool_registry import load_all_tools
from verl.trainer.distillation import is_distillation_enabled
from verl.utils.chat_template import apply_chat_template, initialize_system_prompt
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.dataset.rl_dataset import RLHFDataset, get_dataset_class
from verl.utils.model import compute_position_id_with_mask
from verl.utils.profiler import simple_timer
from verl.utils.ray_utils import auto_await, get_event_loop
from verl.utils.rollout_trace import (
    RolloutTraceConfig,
    rollout_trace_attr,
)
from verl.utils.tokenizer import (
    build_multimodal_processor_inputs,
    get_processor_token_id,
    normalize_token_ids,
)
from verl.workers.config import (
    HFModelConfig,
    RolloutConfig,
)
from verl.workers.rollout.llm_server import LLMServerClient

logger = logging.getLogger(__file__)

# --- OPD: substitutive-streaming support -------------------------------------------------------
# Counts whole-response teacher rescans that fired as a RECOVERY path (chunk reconstruction failed).
# A healthy OPDFlow run must end with 0. Surfaced via the [OPD_RESCAN_FALLBACK] log marker.
_OPD_RESCAN_FALLBACKS = 0


def _force_additive_rescan() -> bool:
    """Reinstate the OLD additive behaviour (chunk calls PLUS a whole-response rescan).

    Exists solely so the redundant path can be characterised as a labelled mode against the corrected
    mode in the same binary/config -- a controlled measurement rather than a comparison with
    historical runs that differ in many other ways. Default OFF. Never set this in a production run;
    check_teacher_invariants.py fails any run where rescans appear without a recorded fallback.
    """
    return os.environ.get("OPD_FORCE_ADDITIVE_RESCAN", "0") not in ("0", "", "false", "False")


def _streamed_coverage_complete(state: dict) -> tuple:
    """(complete, reason). Complete iff the published chunk spans tile [0, final_end) EXACTLY.

    A published FINAL chunk is not sufficient evidence: an earlier chunk can fail to publish while the
    final one succeeds, leaving a hole. Skipping the whole-response rescan then silently drops
    supervision for the missing span. So we verify the actual covered interval instead of trusting a
    flag: start at 0, strictly contiguous (no gap, no overlap), ending at the final chunk's end.
    """
    if state.get("failures"):
        return False, "publish_failures=%d" % state["failures"]
    if not state.get("final_emitted"):
        return False, "final_chunk_not_published"
    spans = sorted(state.get("spans") or [])
    if not spans:
        return False, "no_spans_published"
    if spans[0][0] != 0:
        return False, "coverage_starts_at_%d_not_0" % spans[0][0]
    cursor = 0
    for off, n in spans:
        if off > cursor:
            return False, "gap_at_%d_expected_%d" % (off, cursor)
        if off < cursor:
            return False, "overlap_at_%d_expected_%d" % (off, cursor)
        cursor = off + n
    final_end = state.get("final_end")
    if final_end is not None and cursor != final_end:
        return False, "coverage_end_%d_ne_final_end_%d" % (cursor, final_end)
    return True, "complete[0,%d)" % cursor


def _hybrid_span_payload_enabled_safe() -> bool:
    """True only in the mode where per-chunk labels are stitched per parent and the whole-response
    teacher tensors are stripped -- i.e. where a final rescan is provably discarded work. Imported
    lazily and fail-closed: if the gate cannot be read we keep the old (additive) behaviour rather
    than risk removing supervision a path might rely on."""
    try:
        from verl.experimental.fully_async_policy.hybrid_assembler import hybrid_span_payload_enabled

        return bool(hybrid_span_payload_enabled())
    except Exception:
        return False


def _omniopd_base_seed(session_id):
    """The trajectory's base seed. ONE definition, shared by the speculative launcher and the
    commit -- if they disagree, every proposal is a different unit of work and silently never
    matches, so speculation would run, cost teacher decode, and hide nothing.

    A sha256 digest rather than hash(): Python's str hash is PYTHONHASHSEED-salted, so the base seed
    differed between two runs of an identical config unless that variable happened to be pinned.
    """
    if session_id is None:
        return None
    return int.from_bytes(hashlib.sha256(str(session_id).encode()).digest()[:4], "little") % (2**31)


def _final_only_teacher_enabled_safe() -> bool:
    """F mode: the final chunk is the ONLY chunk carrying supervision, and the ONLY one the hybrid
    drain consumes (fully_async_trainer.py:620-638), so a whole-response rescan here is provably
    discarded work (fully_async_rollouter.py:1609-1617 returns early once a chunk was emitted).

    This is what makes OmniOPD streamable: its audit already runs once, on the final chunk, over the
    cumulative response. Without this gate the rescan re-enters _agent_loop_postprocess with
    chunk_is_final=None, whose only guard is `is False`, so the ENTIRE audit fires a second time --
    M sequential teacher generations per trajectory, discarded. That is a 2x teacher cost on the
    streaming arm alone, i.e. exactly the arm-asymmetric confound that has already invalidated a
    result in this project.

    Fail-closed like its sibling: if the gate cannot be read, keep the old additive behaviour.
    """
    try:
        from verl.experimental.fully_async_policy.hybrid_assembler import final_only_teacher_enabled

        return bool(final_only_teacher_enabled())
    except Exception:
        return False

logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

DEFAULT_ROUTING_CACHE_SIZE = 10000


class AgentLoopMetrics(BaseModel):
    """Agent loop performance metrics."""

    generate_sequences: float = 0.0
    tool_calls: float = 0.0
    compute_score: float = 0.0
    num_preempted: int = -1  # -1 means not available


class AgentLoopOutput(BaseModel):
    """Agent loop output."""

    prompt_ids: list[int]
    """Prompt token ids."""
    response_ids: list[int]
    """Response token ids including LLM generated token, tool response token."""
    response_mask: list[int]
    """Response mask, 1 for LLM generated token, 0 for tool response token."""
    response_logprobs: Optional[list[float]] = None
    """Log probabilities for the response tokens."""
    routed_experts: Optional[Any] = None
    """Routed experts for the total tokens."""
    multi_modal_data: Optional[dict[str, Any]] = None
    """Multi-modal data for multi-modal tools."""
    reward_score: Optional[float] = None
    """Reward score for the trajectory."""
    num_turns: int = 0
    """Number of chat turns, including user, assistant, tool."""
    metrics: AgentLoopMetrics
    """Auxiliary performance metrics"""
    extra_fields: dict[str, Any] = {}
    """Extra fields for dynamic addition."""
    mm_processor_kwargs: Optional[dict[str, Any]] = None
    """Processor/backend kwargs that must stay aligned across rollout and training paths."""

    def as_dict(self) -> dict[str, Any]:
        """Convert agent loop output to a dictionary."""
        output = self.model_dump(exclude_unset=True)

        output["prompts"] = torch.tensor(output.pop("prompt_ids"), dtype=torch.int64)
        output["responses"] = torch.tensor(output.pop("response_ids"), dtype=torch.int64)
        output["response_mask"] = torch.tensor(output.pop("response_mask"), dtype=torch.int64)

        response_logprobs = output.pop("response_logprobs", None)
        if response_logprobs is not None:
            output["rollout_log_probs"] = torch.tensor(response_logprobs, dtype=torch.float32)

        routed_experts = output.pop("routed_experts", None)
        if routed_experts is not None:
            output["routed_experts"] = torch.tensor(routed_experts, dtype=torch.int64)

        # rm_scores: reward score for each token
        reward_score = output.pop("reward_score", None)
        if reward_score is not None:
            rm_scores = torch.zeros_like(output["response_mask"], dtype=torch.float32)
            rm_scores[-1] = reward_score
            output["rm_scores"] = rm_scores

        teacher_ids, teacher_logprobs = (
            output["extra_fields"].pop("teacher_ids", None),
            output["extra_fields"].pop("teacher_logprobs", None),
        )
        if teacher_ids is not None:
            output["teacher_ids"] = teacher_ids
        if teacher_logprobs is not None:
            output["teacher_logprobs"] = teacher_logprobs
        return output


class _InternalAgentLoopOutput(AgentLoopOutput):
    """Internal agent loop output with padded sequences."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt_ids: torch.Tensor
    """Padded prompt token ids."""
    response_ids: torch.Tensor
    """Padded response token ids."""
    input_ids: torch.Tensor
    """Padded input ids(prompt_ids + response_ids)."""
    position_ids: torch.Tensor
    """Padded position ids."""
    response_mask: torch.Tensor
    """Padded response mask."""
    attention_mask: torch.Tensor
    """Padded attention mask."""
    response_logprobs: Optional[torch.Tensor] = None
    """Padded log probabilities for the response tokens."""
    teacher_logprobs: Optional[torch.Tensor] = None
    """Padded log probabilities from teacher model for prompt/response tokens."""
    teacher_ids: Optional[torch.Tensor] = None
    """Padded token ids corresponding to the teacher log probabilities."""
    routed_experts: Optional[torch.Tensor] = None
    """Padded routed experts for the total tokens."""
    multi_modal_inputs: Optional[dict[str, torch.Tensor]] = None
    """Multi-modal inputs for processors (e.g. pixel_values, image_grid_thw, video_grid_thw)."""
    extra_fields: dict[str, Any] = {}
    """Extra fields for dynamic addition."""


class DictConfigWrap:
    """Wrapper for DictConfig to avoid hydra.utils.instantiate recursive resolve."""

    def __init__(self, config: DictConfig):
        self.config = config


class ToolListWrap:
    """Wraps a tool list so ``hydra.utils.instantiate`` doesn't recursively
    resolve its elements (which would demote them to ``DictConfig``)."""

    def __init__(self, tools: list):
        self.tools = tools


class AgentLoopBase(ABC):
    """An agent loop takes an input message, chat with OpenAI compatible LLM server and interact with various
    environments.

    Args:
        trainer_config (DictConfig): whole config for main entrypoint.
        server_manager (LLMServerClient): OpenAI compatible LLM server manager.
        tokenizer (AutoTokenizer): Tokenizer for tokenize messages.
        processor (AutoProcessor): Processor for process messages.
        dataset_cls (type[Dataset]): Dataset class for creating dataset, Defaults to RLHFDataset.
        data_config (DictConfigWrap): Dataset config.
    """

    def __init__(
        self,
        trainer_config: DictConfigWrap,
        server_manager: LLMServerClient,
        tokenizer: AutoTokenizer,
        processor: AutoProcessor,
        dataset_cls: type[RLHFDataset],
        data_config: DictConfigWrap,
        **kwargs,
    ):
        self.config = trainer_config.config
        self.rollout_config = self.config.actor_rollout_ref.rollout
        self.server_manager = server_manager
        self.tokenizer = tokenizer
        self.processor = processor
        self.dataset_cls = dataset_cls
        self.data_config = data_config.config
        self.apply_chat_template_kwargs = self.data_config.get("apply_chat_template_kwargs", {})
        self.mm_processor_kwargs = self.data_config.get("mm_processor_kwargs", {})
        processing_class = self.processor if self.processor is not None else self.tokenizer
        self.system_prompt = initialize_system_prompt(processing_class, **self.apply_chat_template_kwargs)
        self.loop = get_event_loop()

    def _get_mm_processor_kwargs(self, audio_data: Optional[list[Any]] = None) -> dict[str, Any]:
        mm_processor_kwargs = dict(self.mm_processor_kwargs or {})
        if audio_data is not None and "sampling_rate" not in mm_processor_kwargs:
            sampling_rate = getattr(getattr(self.processor, "feature_extractor", None), "sampling_rate", None)
            if sampling_rate is not None:
                mm_processor_kwargs["sampling_rate"] = int(sampling_rate)
        return mm_processor_kwargs

    async def process_vision_info(self, messages: list[dict]) -> dict:
        """Backward-compatible wrapper for multi-modal extraction."""
        return await self.process_multi_modal_info(messages)

    async def process_multi_modal_info(self, messages: list[dict]) -> dict:
        """Extract images, videos and audios from messages.

        Args:
            messages (list[dict]): Input messages.

        Returns:
            dict: Multi-modal data with keys like "images", "videos" and "audios".
        """
        multi_modal_data = {}
        if self.processor is not None:
            image_patch_size = getattr(getattr(self.processor, "image_processor", None), "patch_size", 14)
            if hasattr(self.dataset_cls, "process_multi_modal_info"):
                images, videos, audios = await self.dataset_cls.process_multi_modal_info(
                    messages, image_patch_size=image_patch_size, config=self.data_config
                )
            else:
                images, videos = await self.dataset_cls.process_vision_info(
                    messages, image_patch_size=image_patch_size, config=self.data_config
                )
                audios = None
            if images is not None:
                multi_modal_data["images"] = images
            if videos is not None:
                multi_modal_data["videos"] = videos
            if audios is not None:
                multi_modal_data["audios"] = audios

        return multi_modal_data

    async def apply_chat_template(
        self,
        messages: list[dict],
        tools: list[dict] = None,
        images: list[Image.Image] = None,
        videos: list[tuple[torch.Tensor, dict]] = None,
        audios: list[Any] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        remove_system_prompt: bool = False,
    ):
        """Apply chat template to messages with optional tools, images, and videos.

        Args:
            messages (list[dict]): Input messages.
            tools (list[dict], optional): Tools schemas. Defaults to None.
            images (list[Image.Image], optional): Input images. Defaults to None.
            videos (list[tuple[torch.Tensor, dict]], optional): Input videos. Defaults to None.
            remove_system_prompt (bool, optional): Whether to remove system prompt. Defaults to False.

        Returns:
            list[int]: Prompt token ids.
        """
        if self.processor is not None:
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: apply_chat_template(
                    self.processor,
                    messages,
                    tools=tools,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )

            model_inputs = build_multimodal_processor_inputs(
                self.processor,
                text=[raw_prompt],
                images=images,
                videos=videos,
                audio=audios,
                mm_processor_kwargs=mm_processor_kwargs
                if mm_processor_kwargs is not None
                else self._get_mm_processor_kwargs(audios),
            )
            prompt_ids = normalize_token_ids(model_inputs.pop("input_ids"))
        else:
            tokenized_prompt = await self.loop.run_in_executor(
                None,
                lambda: apply_chat_template(
                    self.tokenizer,
                    messages,
                    tools=tools,
                    add_generation_prompt=True,
                    tokenize=True,
                    **self.apply_chat_template_kwargs,
                ),
            )
            prompt_ids = normalize_token_ids(tokenized_prompt)

        if remove_system_prompt:
            prompt_ids = prompt_ids[len(self.system_prompt) :]

        return prompt_ids

    @abstractmethod
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """Run agent loop to interact with LLM server and environment.

        Args:
            sampling_params (Dict[str, Any]): LLM sampling params.
            **kwargs: dataset fields from `verl.utils.dataset.RLHFDataset`.

        Returns:
            AgentLoopOutput: Agent loop output.
        """
        raise NotImplementedError


"""Agent loop registry: key is agent_name, value is a dict of agent loop config
used by hydra.utils.instantiate to initialize agent loop instance.

https://hydra.cc/docs/advanced/instantiate_objects/overview/
"""
_agent_loop_registry: dict[str, dict] = {}


def register(agent_name: str):
    """Register agent loop class."""

    def decorator(subclass: type[AgentLoopBase]) -> type[AgentLoopBase]:
        fqdn = f"{subclass.__module__}.{subclass.__qualname__}"
        _agent_loop_registry[agent_name] = {"_target_": fqdn}
        return subclass

    return decorator


class AgentLoopWorker:
    """Agent loop worker takes a batch of messages and run each message in an agent loop.

    Args:
        config (DictConfig): whole config for main entrypoint.
        llm_client (LLMServerClient): Client for the LLM server.
        teacher_client (dict[str, LLMServerClient]): Client for multiple teacher servers.
        reward_loop_worker_handles (List[ray.actor.ActorHandle]): Actor handles for streaming reward computation.
    """

    def __init__(
        self,
        config: DictConfig,
        llm_client: LLMServerClient,
        teacher_client: dict[str, LLMServerClient] = None,
        reward_loop_worker_handles: list[ray.actor.ActorHandle] = None,
    ):
        self.config = config
        self._omniopd_enabled = False   # set once distillation resolves; see below
        self.llm_client = llm_client
        self.teacher_client = teacher_client
        self.reward_loop_worker_handles = reward_loop_worker_handles
        self.chunk_message_queue_client = None

        rollout_config, model_config = config.actor_rollout_ref.rollout, config.actor_rollout_ref.model
        self.rollout_config: RolloutConfig = omega_conf_to_dataclass(rollout_config)
        self.model_config: HFModelConfig = omega_conf_to_dataclass(model_config)

        self.dataset_cls = get_dataset_class(config.data)
        self.tokenizer = self.model_config.tokenizer
        self.processor = self.model_config.processor
        self.mm_processor_kwargs = config.data.get("mm_processor_kwargs", {})

        # Online policy distillation
        self.distillation_enabled = is_distillation_enabled(config.distillation)
        if self.distillation_enabled:
            from verl.experimental.teacher_loop.teacher_manager import AsyncTeacherLLMServerManager

            self.teacher_key: str = config.distillation.teacher_key
            self._omniopd_enabled = omniopd_enabled(self.config)
            self.teacher_server_manager = AsyncTeacherLLMServerManager(
                config=config,
                teacher_client=teacher_client,
            )

        # Load tools once per worker; each trajectory just reuses self.tools.
        tool_config_path = self.rollout_config.multi_turn.tool_config_path
        function_tool_path = self.rollout_config.multi_turn.function_tool_path
        self.tools = load_all_tools(
            tool_config_path=resolve_config_path(tool_config_path) if tool_config_path else None,
            function_tool_path=resolve_config_path(function_tool_path) if function_tool_path else None,
        )

        # Load custom agent loop implementations from config path
        agent_loop_config_path = self.rollout_config.agent.agent_loop_config_path
        if agent_loop_config_path:
            resolved_path = resolve_config_path(agent_loop_config_path)
            agent_loop_configs = OmegaConf.load(resolved_path)
            for agent_loop_config in agent_loop_configs:
                _agent_loop_registry[agent_loop_config.name] = agent_loop_config
        if self.model_config.get("custom_chat_template", None) is not None:
            if self.model_config.processor is not None:
                self.model_config.processor.chat_template = self.model_config.custom_chat_template
            self.model_config.tokenizer.chat_template = self.model_config.custom_chat_template

        trace_config = self.rollout_config.trace
        RolloutTraceConfig.init(
            self.rollout_config.trace.project_name,
            self.rollout_config.trace.experiment_name,
            trace_config.get("backend"),
            trace_config.get("token2text", False),
            trace_config.get("max_samples_per_step_per_worker", None),
        )

    def set_chunk_message_queue_client(self, message_queue_client) -> None:
        """Install a trainer queue publisher for streaming chunk payloads."""
        self.chunk_message_queue_client = message_queue_client

    def _get_mm_processor_kwargs(self, audio_data: Optional[list[Any]] = None) -> dict[str, Any]:
        """Return multimodal processor kwargs with audio sampling-rate defaults."""
        mm_processor_kwargs = dict(self.mm_processor_kwargs or {})
        if audio_data is not None and "sampling_rate" not in mm_processor_kwargs:
            sampling_rate = getattr(getattr(self.processor, "feature_extractor", None), "sampling_rate", None)
            if sampling_rate is not None:
                mm_processor_kwargs["sampling_rate"] = int(sampling_rate)
        return mm_processor_kwargs

    async def generate_sequences(self, batch: DataProto) -> DataProto:
        """Generate sequences from agent loop.

        Args:
            batch (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
            - prompts: [bsz, prompt_length], prompt token ids from dataset.
            - responses: [bsz, response_length], output token ids include response tokens
              from LLM generation and observation tokens from tool_calls.
            - response_mask: [bsz, response_length], 1 for LLM generated tokens, 0 for observation/padding tokens.
            - input_ids: [bsz, prompt_length + response_length], whole sequence token ids, including prompt tokens
              and response tokens.
            - attention_mask: [bsz, prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
            - position_ids: [bsz, prompt_length + response_length], incremental position ids.

            For multi-turn conversations:
            responses:     |<- LLM generation ->|<- tool_calls ->|<- LLM generation ->|<- padding ->|
            response_mask: | 1, 1, 1, ..., 1, 1 | 0, 0, .., 0, 0 | 1, 1, 1, ..., 1, 1 | 0, 0, ..., 0|
        """
        config = self.rollout_config
        validate = batch.meta_info.get("validate", False)
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repetition_penalty=1.0,
            logprobs=config.calculate_log_probs,
        )

        # OmniOPD selects audit anchors by peak student entropy, computed in the patched sampler over
        # the full unpadded vocabulary. Requested only when an audit will actually run: validation
        # takes the early return in _compute_omniopd_audit, so paying the per-step reduction over B*V
        # there would buy a series nothing reads.
        if self._omniopd_enabled and not validate:
            sampling_params["return_token_entropy"] = True

        def apply_greedy_sampling_params(params: dict[str, Any]) -> None:
            params["top_p"] = 1.0
            params["top_k"] = -1
            params["temperature"] = 0

        # override sampling params for validation
        if validate:
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["top_k"] = config.val_kwargs.top_k
            sampling_params["temperature"] = config.val_kwargs.temperature

        # by default, we assume it's a single turn agent
        if "agent_name" not in batch.non_tensor_batch:
            default_agent_loop = config.agent.default_agent_loop
            batch.non_tensor_batch["agent_name"] = np.array([default_agent_loop] * len(batch), dtype=object)

        if "index" in batch.non_tensor_batch:
            index = batch.non_tensor_batch["index"]
        else:
            index = np.arange(len(batch))

        max_samples_per_worker = RolloutTraceConfig.get_instance().max_samples_per_step_per_worker

        # For n rollouts per sample, we trace all n rollouts for selected samples
        # Note: This sampling happens per-worker, so total traces = max_samples_per_worker * num_workers * n
        if max_samples_per_worker is not None:
            unique_sample_indices = np.unique(index)
            if max_samples_per_worker < len(unique_sample_indices):
                selected_samples = set(
                    np.random.choice(unique_sample_indices, max_samples_per_worker, replace=False).tolist()
                )
                traced_indices = set(i for i in range(len(batch)) if index[i] in selected_samples)
            else:
                traced_indices = set(range(len(batch)))
        else:
            traced_indices = set(range(len(batch)))

        trajectory_info = await get_trajectory_info(
            batch.meta_info.get("global_steps", -1), index.tolist(), batch.meta_info.get("validate", False)
        )

        # NOTE: __do_sample__ is an internal per-sample override used by REMAX combined rollout.
        # Do not forward it to concrete agent loops, which may reject unknown kwargs.
        per_sample_do_sample = batch.non_tensor_batch.get("__do_sample__")
        tasks = []
        for i in range(len(batch)):
            trace_this_sample = i in traced_indices
            kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items() if k != "__do_sample__"}
            sample_sampling_params = dict(sampling_params)
            if not validate and per_sample_do_sample is not None and not bool(per_sample_do_sample[i]):
                apply_greedy_sampling_params(sample_sampling_params)
            tasks.append(
                asyncio.create_task(
                    self._run_agent_loop(sample_sampling_params, trajectory_info[i], trace=trace_this_sample, **kwargs)
                )
            )
        outputs = await asyncio.gather(*tasks)

        output = self._postprocess(
            outputs, input_non_tensor_batch=batch.non_tensor_batch, validate=batch.meta_info.get("validate", False)
        )
        return output

    async def _run_agent_loop(
        self,
        sampling_params: dict[str, Any],
        trajectory: dict[str, Any],
        *,
        agent_name: str,
        trace: bool = True,
        **kwargs,
    ) -> _InternalAgentLoopOutput:
        with rollout_trace_attr(
            step=trajectory["step"],
            sample_index=trajectory["sample_index"],
            rollout_n=trajectory["rollout_n"],
            validate=trajectory["validate"],
            name="agent_loop",
            trace=trace,
        ):
            assert agent_name in _agent_loop_registry, (
                f"Agent loop {agent_name} not registered, registered agent loops: {_agent_loop_registry.keys()}"
            )

            agent_loop_config = _agent_loop_registry[agent_name]
            agent_loop = hydra.utils.instantiate(
                config=agent_loop_config,
                trainer_config=DictConfigWrap(config=self.config),
                server_manager=self.llm_client,
                tokenizer=self.tokenizer,
                processor=self.processor,
                dataset_cls=self.dataset_cls,
                data_config=DictConfigWrap(self.config.data),
                tools=ToolListWrap(self.tools),
            )
            run_kwargs = dict(kwargs)
            stream_state = None
            if self._should_stream_chunks(agent_name=agent_name, validate=trajectory["validate"]):
                stream_state = {"emitted": 0, "final_emitted": False, "failures": 0,
                                "spans": [], "final_end": None}
                run_kwargs["_chunk_callback"] = self._make_chunk_callback(
                    sample_kwargs=kwargs,
                    validate=trajectory["validate"],
                    stream_state=stream_state,
                )

            output: AgentLoopOutput = await agent_loop.run(sampling_params, **run_kwargs)
            for key in ("xiaoshuai_sample_id", "xiaoshuai_parent_sample_id", "xiaoshuai_epoch"):
                if key in kwargs:
                    output.extra_fields[key] = self._to_python_scalar(kwargs[key])

            # Chunk streaming is SUBSTITUTIVE, not additive. When the chunks already carry this
            # response's teacher labels, recomputing them over the whole response here is pure waste:
            # under span-only payloads the trainer strips these tensors and restitches the labels from
            # each chunk's new span (hybrid_assembler), and the trace confirms it -- the trainer's
            # `get` count is 0 in the streaming arm. Before this fix every response paid for exactly
            # 1.00 duplicate whole-response call (is_final=None over the final chunk's own span).
            #
            # Gated on span-payload mode: that is the configuration where the rescan's output is
            # PROVABLY discarded. Other modes keep the old behaviour rather than be changed untested.
            compute_final_teacher = True
            _final_only = _final_only_teacher_enabled_safe()
            if stream_state is not None and not _force_additive_rescan() and _final_only:
                # F mode ignores non-final chunks entirely, so coverage of the intermediate chunks is
                # irrelevant -- the only precondition is that the final chunk actually published.
                if stream_state.get("final_emitted"):
                    compute_final_teacher = False
                else:
                    global _OPD_RESCAN_FALLBACKS
                    _OPD_RESCAN_FALLBACKS += 1
                    logger.warning(
                        "[OPD_RESCAN_FALLBACK] sample_id=%s emitted=%d failures=%d "
                        "reason=final_chunk_not_published total_fallbacks=%d",
                        self._to_python_scalar(kwargs.get("xiaoshuai_sample_id")),
                        stream_state["emitted"], stream_state["failures"], _OPD_RESCAN_FALLBACKS,
                    )
            elif (stream_state is not None and _hybrid_span_payload_enabled_safe()
                    and not _force_additive_rescan()):
                _complete, _why = _streamed_coverage_complete(stream_state)
                if _complete:
                    compute_final_teacher = False
                else:
                    # Streamed labels are NOT provably complete (gap/overlap/failure/missing final):
                    # recover via whole-response scoring rather than train on a hole -- but never
                    # silently. This marker is what the run-validity invariant counts.
                    # (`global` is declared once for this function, in the F-mode branch above --
                    # a second declaration here is a SyntaxError, not merely redundant.)
                    _OPD_RESCAN_FALLBACKS += 1
                    logger.warning(
                        "[OPD_RESCAN_FALLBACK] sample_id=%s emitted=%d failures=%d "
                        "reason=%s total_fallbacks=%d",
                        self._to_python_scalar(kwargs.get("xiaoshuai_sample_id")),
                        stream_state["emitted"],
                        stream_state["failures"],
                        _why,
                        _OPD_RESCAN_FALLBACKS,
                    )
            return await self._agent_loop_postprocess(
                output,
                trajectory["validate"],
                compute_teacher_logprobs=compute_final_teacher,
                **kwargs,
            )

    def _should_stream_chunks(self, *, agent_name: str, validate: bool) -> bool:
        """Return True when this worker can publish trainer-visible chunks during generation."""
        if validate or self.chunk_message_queue_client is None or agent_name != "single_turn_agent":
            return False
        if not self.distillation_enabled:
            return False
        distillation_loss = self.config.distillation.get("distillation_loss", {})
        use_task_rewards = distillation_loss.get("use_task_rewards", False)
        if isinstance(use_task_rewards, str):
            use_task_rewards = use_task_rewards.strip().lower() in {"1", "true", "yes", "on"}
        if bool(use_task_rewards):
            return False

        from verl.experimental.fully_async_policy.detach_utils import get_chunk_token_size

        return get_chunk_token_size(self.config) > 0

    def _make_chunk_callback(self, *, sample_kwargs: dict[str, Any], validate: bool, stream_state: Optional[dict] = None):
        async def _chunk_callback(
            output: AgentLoopOutput,
            *,
            chunk_idx: int,
            token_offset: int,
            n_tokens: int,
            is_final: bool,
        ) -> bool:
            published = await self._publish_streaming_chunk(
                output,
                sample_kwargs=sample_kwargs,
                validate=validate,
                chunk_idx=chunk_idx,
                token_offset=token_offset,
                n_tokens=n_tokens,
                is_final=is_final,
            )
            # Record what streaming ACTUALLY delivered, so the caller can distinguish "labels already
            # produced by chunks" from "reconstruction failed". Without this the whole-response rescan
            # below fires unconditionally.
            if stream_state is not None:
                if published:
                    stream_state["emitted"] += 1
                    # record the ACTUAL interval this chunk got labels for, so the caller can verify
                    # exact [0, L) coverage rather than trust that a final chunk implies completeness
                    stream_state["spans"].append((int(token_offset), int(n_tokens)))
                    if is_final:
                        stream_state["final_emitted"] = True
                        stream_state["final_end"] = int(token_offset) + int(n_tokens)
                else:
                    stream_state["failures"] += 1
            return published

        return _chunk_callback

    @staticmethod
    def _to_python_scalar(value):
        if isinstance(value, np.ndarray):
            if value.shape == ():
                return value.item()
            if len(value) == 1:
                return AgentLoopWorker._to_python_scalar(value[0])
            return value.tolist()
        if hasattr(value, "item"):
            return value.item()
        return value

    @staticmethod
    def _single_item_non_tensor_batch(sample_kwargs: dict[str, Any]) -> dict[str, np.ndarray]:
        non_tensor_batch = {}
        for key, value in sample_kwargs.items():
            arr = np.empty(1, dtype=object)
            arr[0] = AgentLoopWorker._to_python_scalar(value)
            non_tensor_batch[key] = arr
        return non_tensor_batch

    @staticmethod
    def _first_non_tensor_value(batch: DataProto, key: str, default=None):
        values = batch.non_tensor_batch.get(key)
        if values is None or len(values) == 0:
            return default
        return AgentLoopWorker._to_python_scalar(values[0])

    def _teacher_keep_response(self, parent_sample_id, surprisal_agg=None) -> bool:
        """Pre-teacher response-level skip decision (default keep_frac=1.0 => always keep).

        Decided ONCE per parent_sample_id and cached, so all N responses of a parent group (and every
        chunk) get the SAME keep/skip -> parent-level all-or-nothing.
        Policies:
          'random'           : stable md5(parent_sample_id) subsample (default; teacher-free, content-free).
          'entropy_surprisal': teacher-free TIP-inspired proxy -- keep if the first-deciding response's
                               first-chunk STUDENT sampled-token surprisal aggregate (top20%-mean of
                               -log p_student, passed as surprisal_agg) >= threshold (uncertain student ->
                               keep; confident -> skip). NOT full TIP: teacher-student divergence is
                               unavailable before the teacher runs, so surprisal is a pre-teacher proxy.
                               surprisal_agg=None (signal unreachable, e.g. logprobs off) -> stable md5.
        Skipped responses are never scored or published -> teacher tokens-forwarded drop ~x keep_frac.
        """
        if not hasattr(self, "_tk_keep_frac"):
            self._tk_keep_frac = float(os.environ.get("OPD_TEACHER_RESPONSE_KEEP_FRAC", "1.0"))
            self._tk_policy = os.environ.get("OPD_TEACHER_RESPONSE_SKIP_POLICY", "random")
            _thr_env = os.environ.get("OPD_TEACHER_RESPONSE_SKIP_UNCERTAINTY_THRESHOLD", "auto")
            try:
                self._tk_threshold = float(_thr_env)  # >0: fixed threshold; <=0/"auto": sliding-window quantile
            except ValueError:
                self._tk_threshold = 0.0
            self._tk_surp_window = []  # recent surprisal aggregates -> auto-quantile threshold (targets keep_frac)
            self._tk_skipped = 0
            self._tk_kept = 0
            # Group-consistency cache: parent_sample_id -> keep/skip. Relies on the whole parent group
            # (all N GRPO responses) being dispatched to ONE AgentLoopWorker: the fully-async path uses
            # generate_sequences_single() -> _select_best_worker(), which sends the full group batch to a
            # single worker (NO .chunk()), so the N async tasks share this cache and the first arriver's
            # decision is reused by every sibling + chunk. (If that dispatch ever shards a group across
            # workers, content-policy group-consistency would need a shared registry instead.)
            self._tk_decisions = {}  # first arriver decides; siblings reuse
            self._tk_kept_surp = []
            self._tk_skip_surp = []
            if self._tk_keep_frac < 1.0:
                _thr_disp = f"{self._tk_threshold}" if self._tk_threshold > 0 else "auto-quantile"
                print(
                    f"[TEACHER-SKIP-CFG] keep_frac={self._tk_keep_frac} policy={self._tk_policy} "
                    f"threshold={_thr_disp}",
                    flush=True,
                )
        if self._tk_keep_frac >= 1.0:
            return True
        if parent_sample_id in self._tk_decisions:
            return self._tk_decisions[parent_sample_id]
        if self._tk_policy == "entropy_surprisal" and surprisal_agg is not None:
            # Keep the most-uncertain ~keep_frac of groups. Default threshold is an auto sliding-window
            # quantile at (1-keep_frac) -> targets keep_frac for a fair equal-budget comparison vs random;
            # a positive OPD_TEACHER_RESPONSE_SKIP_UNCERTAINTY_THRESHOLD overrides with a fixed value.
            if self._tk_threshold > 0:
                _thr = self._tk_threshold
            elif len(self._tk_surp_window) >= 8:
                import numpy as _np

                _thr = float(_np.quantile(self._tk_surp_window, 1.0 - self._tk_keep_frac))
            else:
                _thr = None  # cold start -> stable md5 until the window fills
            self._tk_surp_window.append(float(surprisal_agg))
            self._tk_surp_window = self._tk_surp_window[-512:]
            if _thr is not None:
                keep = surprisal_agg >= _thr
            else:
                h = int(hashlib.md5(str(parent_sample_id).encode()).hexdigest()[:8], 16) % 10000
                keep = h < int(self._tk_keep_frac * 10000)
            # record surprisal vs decision for the kept-vs-skipped separation telemetry (all decisions)
            if keep:
                self._tk_kept_surp.append(float(surprisal_agg))
                self._tk_kept_surp = self._tk_kept_surp[-2000:]
            else:
                self._tk_skip_surp.append(float(surprisal_agg))
                self._tk_skip_surp = self._tk_skip_surp[-2000:]
        else:
            h = int(hashlib.md5(str(parent_sample_id).encode()).hexdigest()[:8], 16) % 10000
            keep = h < int(self._tk_keep_frac * 10000)
        self._tk_decisions[parent_sample_id] = keep
        if len(self._tk_decisions) > 50000:
            # FIFO-evict oldest entries to bound memory on very long runs. Safe: a parent is queried only
            # during its own (short) generation, so entries this old are long-completed and never re-queried.
            for _k in list(self._tk_decisions)[:10000]:
                del self._tk_decisions[_k]
        if keep:
            self._tk_kept += 1
        else:
            self._tk_skipped += 1
        _tot = self._tk_kept + self._tk_skipped
        if _tot % 20 == 1:
            print(
                f"[TEACHER-SKIP] policy={self._tk_policy} keep_frac={self._tk_keep_frac} "
                f"kept={self._tk_kept} skipped={self._tk_skipped} keep_rate={self._tk_kept / max(1, _tot):.3f}",
                flush=True,
            )
            if self._tk_policy == "entropy_surprisal":
                import statistics as _st

                _all = self._tk_kept_surp + self._tk_skip_surp
                if _all:
                    _s = sorted(_all)
                    _p = lambda q: _s[min(len(_s) - 1, int(q * len(_s)))]
                    _km = f"{_st.mean(self._tk_kept_surp):.3f}" if self._tk_kept_surp else "na"
                    _sm = f"{_st.mean(self._tk_skip_surp):.3f}" if self._tk_skip_surp else "na"
                    print(
                        f"[TEACHER-SKIP-SIGNAL] surprisal_mean={_st.mean(_all):.3f} "
                        f"surprisal_p50={_p(0.5):.3f} surprisal_p95={_p(0.95):.3f}",
                        flush=True,
                    )
                    print(
                        f"[TEACHER-SKIP-PRIORITY] kept_surprisal_mean={_km} skipped_surprisal_mean={_sm}",
                        flush=True,
                    )
        return keep

    async def _publish_streaming_chunk(
        self,
        output: AgentLoopOutput,
        *,
        sample_kwargs: dict[str, Any],
        validate: bool,
        chunk_idx: int,
        token_offset: int,
        n_tokens: int,
        is_final: bool,
    ) -> bool:
        """Postprocess and publish a single in-flight chunk as a ChunkSample."""
        if self.chunk_message_queue_client is None or n_tokens <= 0:
            return False

        sample_id = self._to_python_scalar(sample_kwargs.get("xiaoshuai_sample_id"))
        if not sample_id:
            logger.warning("Streaming chunk requested but xiaoshuai_sample_id is missing; skipping chunk publish")
            return False
        parent_sample_id = self._to_python_scalar(sample_kwargs.get("xiaoshuai_parent_sample_id", sample_id))
        epoch = self._to_python_scalar(sample_kwargs.get("xiaoshuai_epoch", -1))

        # Pre-teacher RESPONSE-LEVEL SKIP: drop this whole response from teacher scoring + chunk publish.
        # For the entropy_surprisal policy, compute a teacher-free STUDENT-surprisal proxy from this
        # (first) chunk's sampled-token logprobs (top20%-mean of -log p); the decision is made once per
        # parent and cached, so all N group responses agree. Skips the teacher forward at
        # _agent_loop_postprocess below -> cuts teacher tokens-forwarded ~x keep_frac. keep_frac=1.0 => no-op.
        _surprisal_agg = None
        if (
            os.environ.get("OPD_TEACHER_RESPONSE_SKIP_POLICY", "random") == "entropy_surprisal"
            and parent_sample_id not in getattr(self, "_tk_decisions", {})
        ):
            _lp = getattr(output, "response_logprobs", None)
            if _lp:
                import numpy as _np

                _s = -_np.asarray(_lp, dtype=float)
                _s = _s[_np.isfinite(_s)]
                if _s.size:
                    _k = max(1, int(0.2 * _s.size))  # top20%-mean surprisal (captures hard/uncertain spans)
                    _surprisal_agg = float(_np.sort(_s)[-_k:].mean())
        _keep = self._teacher_keep_response(parent_sample_id, surprisal_agg=_surprisal_agg)
        _is_audit = False
        if not _keep:
            # Phase 2 AUDIT: a policy-SKIPPED parent is, with prob OPD_TEACHER_RESPONSE_AUDIT_FRAC, still
            # teacher-scored as an 'audit_scored' sample (reconstructed + delta/Soft-OR computed for
            # diagnostics, but EXCLUDED from the OPD loss via the is_audit tag). Decision is stable per
            # parent (group-consistent) so all N responses + chunks agree. AUDIT_FRAC=0 => true skip.
            if not hasattr(self, "_tk_audit_frac"):
                self._tk_audit_frac = float(os.environ.get("OPD_TEACHER_RESPONSE_AUDIT_FRAC", "0.0"))
                self._tk_audit = {}
                self._tk_audited = 0
            if self._tk_audit_frac > 0.0:
                if parent_sample_id not in self._tk_audit:
                    _ah = int(hashlib.md5((str(parent_sample_id) + "|audit").encode()).hexdigest()[:8], 16) % 10000
                    _ad = _ah < int(self._tk_audit_frac * 10000)
                    self._tk_audit[parent_sample_id] = _ad
                    self._tk_audited += int(_ad)
                    if _ad and self._tk_audited % 20 == 1:
                        print(
                            f"[TEACHER-AUDIT-GATE] audited_parents={self._tk_audited} "
                            f"audit_frac={self._tk_audit_frac} (of policy-skipped groups)",
                            flush=True,
                        )
                _is_audit = self._tk_audit[parent_sample_id]
            if not _is_audit:
                return False  # truly skipped: zero teacher forward, no chunks published

        from verl.experimental.fully_async_policy.chunk_sample import ChunkSample
        from verl.experimental.fully_async_policy.opd_stage0_trace import trace_chunk_event

        try:
            output.reward_score = 0.0
            # Final-only teacher scoring (F mode, gated): skip the teacher call on non-final chunks; only
            # the final chunk scores the full response once -> no per-chunk prefix amplification. Default off.
            _final_only = os.environ.get("OPD_TEACHER_FINAL_ONLY", "0") not in ("0", "", "false", "False")
            internal = await self._agent_loop_postprocess(
                output,
                validate,
                compute_score=False,
                compute_teacher_logprobs=(is_final or not _final_only),
                chunk_is_final=is_final,
                **sample_kwargs,
            )
            # F MODE: A NON-FINAL CHUNK HAS NOTHING TO DELIVER. The drain skips it outright
            # (fully_async_trainer: `if not chunk.is_final: continue`), so everything below is
            # computed and then discarded -- and it is not cheap. _postprocess pads the CUMULATIVE
            # response into a full DataProto, CPU-bound, ON THE ASYNCIO EVENT LOOP that is also
            # dispatching generation; and the ChunkSample then occupies one slot of a queue bounded
            # at 16.
            #
            # Measured cost of not skipping (jobs 44900600/601, q128, c512, ~1290-token responses,
            # so ~3 chunks/trajectory): queue depth p95 14/16 against umem's 0.2, rollouter idle
            # 6-7% against umem's 0.0, and timing_s/gen 8.79 -> 11.74 s for identical work -- the
            # whole of the arm's -20.1%.
            #
            # _agent_loop_postprocess above still runs, so the speculative proposal pass still sees
            # every chunk boundary. That is the part that has value; this part had none.
            if _final_only and not is_final:
                return True

            chunk_batch = self._postprocess(
                [internal],
                input_non_tensor_batch=self._single_item_non_tensor_batch(sample_kwargs),
                validate=validate,
            )

            batch_size = len(chunk_batch)
            chunk_batch.non_tensor_batch["uid"] = np.array([f"uid_{parent_sample_id}"] * batch_size, dtype=object)
            chunk_batch.non_tensor_batch["chunk_sample_id"] = np.array([sample_id] * batch_size, dtype=object)
            chunk_batch.non_tensor_batch["chunk_parent_sample_id"] = np.array(
                [parent_sample_id] * batch_size, dtype=object
            )
            chunk_batch.non_tensor_batch["chunk_idx"] = np.array([chunk_idx] * batch_size, dtype=np.int32)
            chunk_batch.non_tensor_batch["chunk_token_offset"] = np.array([token_offset] * batch_size, dtype=np.int32)
            chunk_batch.non_tensor_batch["chunk_n_tokens"] = np.array([n_tokens] * batch_size, dtype=np.int32)
            chunk_batch.non_tensor_batch["chunk_is_final"] = np.array([is_final] * batch_size, dtype=bool)
            # Phase 2: per-response audit tag (rides the carrier non_tensor -> assembler -> loss is_audit mask).
            chunk_batch.non_tensor_batch["is_audit"] = np.array([_is_audit] * batch_size, dtype=bool)

            min_global_steps = self._first_non_tensor_value(chunk_batch, "min_global_steps")
            max_global_steps = self._first_non_tensor_value(chunk_batch, "max_global_steps")
            policy_version = min_global_steps if min_global_steps is not None else max_global_steps
            policy_version = int(policy_version or 0)
            chunk_batch.non_tensor_batch["chunk_policy_version"] = np.array(
                [policy_version] * batch_size, dtype=np.int32
            )

            # H-ACC-SPAN: ship only THIS chunk's new-span teacher labels. Non-final chunks carry NO
            # parent_payload (the full-prefix [P+R, k] labels are never serialized); the final chunk
            # carries a structural carrier with those big tensors stripped, and the trainer rebuilds
            # them from the accumulated spans. Off -> legacy full-prefix payload per chunk.
            from verl.experimental.fully_async_policy.hybrid_assembler import hybrid_span_payload_enabled

            span_teacher_ids = None
            span_teacher_logprobs = None
            chunk_parent_payload = chunk_batch
            if hybrid_span_payload_enabled() and "teacher_ids" in chunk_batch.batch.keys():
                P = int(chunk_batch.batch["prompts"].shape[1])
                # -1: index i holds the prediction for token i+1, so response token j lives at
                # P+j-1. Must match `ss` below, which writes the span at the same offset.
                lo, hi = P + token_offset - 1, P + token_offset - 1 + n_tokens
                span_teacher_ids = chunk_batch.batch["teacher_ids"][0, lo:hi, :].detach().cpu().clone()
                span_teacher_logprobs = chunk_batch.batch["teacher_logprobs"][0, lo:hi, :].detach().cpu().clone()
                if is_final:
                    for _k in ("teacher_ids", "teacher_logprobs"):
                        if _k in chunk_batch.batch.keys():
                            del chunk_batch.batch[_k]
                    chunk_parent_payload = chunk_batch  # structural carrier (no big teacher tensors)
                else:
                    chunk_parent_payload = None
            elif (_final_only_teacher_enabled_safe() and not is_final
                  and os.environ.get("OPD_HYBRID_FULL_SAMPLE", "0") not in ("0", "", "false", "False")):
                # F mode: the drain skips non-final chunks outright (fully_async_trainer.py:624-625),
                # so serialising a complete [1, P+R] DataProto for each is pure waste -- roughly
                # response_length/chunk_tokens discarded payloads per trajectory, and all of it
                # charged to the streaming arm, which is the arm under measurement. parent_payload
                # None is already an exercised production state (see the span-payload branch above).
                # Conjoined with OPD_HYBRID_FULL_SAMPLE so the legacy chunk-training path, which
                # dereferences parent_payload.batch unconditionally, stays out of reach.
                chunk_parent_payload = None

            chunk = ChunkSample(
                sample_id=str(sample_id),
                chunk_idx=int(chunk_idx),
                token_offset=int(token_offset),
                n_tokens=int(n_tokens),
                tokens=list(output.response_ids[token_offset : token_offset + n_tokens]),
                is_final=bool(is_final),
                policy_version=policy_version,
                parent_payload=chunk_parent_payload,
                span_teacher_ids=span_teacher_ids,
                span_teacher_logprobs=span_teacher_logprobs,
                meta={
                    "epoch": epoch,
                    "parent_sample_id": parent_sample_id,
                    "row_id": sample_id,
                    "chunk_id": f"{sample_id}:{chunk_idx}",
                    "source": "streaming",
                    "response_end": token_offset + n_tokens,
                    "response_width": self.rollout_config.response_length,
                    # Teacher KV-reuse telemetry for this chunk (cached/total tokens, replica, latency).
                    "teacher_telemetry": output.extra_fields.get("teacher_telemetry"),
                },
            )
            success = await self.chunk_message_queue_client.put_sample(ray.cloudpickle.dumps(chunk))
            trace_chunk_event(
                "chunk_emit",
                sample_id=chunk.sample_id,
                chunk_idx=chunk.chunk_idx,
                role="rollouter",
                n_tokens=chunk.n_tokens,
                token_offset=chunk.token_offset,
                policy_version=chunk.policy_version,
                is_final=chunk.is_final,
                row_id=chunk.sample_id,
                source="streaming",
                streaming=True,
                success=bool(success),
                parent_sample_id=parent_sample_id,
                finish_reason=output.extra_fields.get("finish_reason"),
                # Q3 per-replica attribution: which engine replica decoded this slice.
                replica_rank=output.extra_fields.get("replica_rank"),
                # Which side of the split-vs-continuous A/B produced this chunk. `streaming=True`
                # above is set for BOTH, so without this the two arms are indistinguishable from
                # trace data alone -- the exact blind spot that let six acc cells run A/A.
                continuous_stream=bool(output.extra_fields.get("continuous_stream", False)),
            )
            return bool(success)
        except Exception:
            logger.exception("Failed to publish streaming chunk sample_id=%s chunk_idx=%s", sample_id, chunk_idx)
            return False

    async def _agent_loop_postprocess(
        self,
        output,
        validate,
        compute_score: bool = True,
        compute_teacher_logprobs: bool = True,
        chunk_is_final: Optional[bool] = None,
        **kwargs,
    ) -> _InternalAgentLoopOutput:
        """Perform post-processing operations on the output of each individual agent loop."""
        output.extra_fields["raw_prompt"] = kwargs["raw_prompt"]

        # Some AgentLoop may have already computed the reward score, e.g SWE-agent.

        # NOTE: consistent with the legacy batch version of generate_sequences that existed in the
        # deprecated vLLM SPMD rollout implementation.
        # prompt_ids: left padded with zeros (e.g., [0,0,0,0,1,2,3,4])
        # response_ids: right padded with zeros (e.g., [5,6,7,8,0,0,0,0])
        # input_ids: concatenation of prompt + response
        # Mask:
        # For example, if the prompt is [1,2,3,4] and the response is [5,6,7,(tool start)8,9(tool end),10,11,12]
        # - prompt_attention_mask: 0s for padding, 1s for tokens
        #   e.g., [0,0,0,0,1,1,1,1]
        # - response_attention_mask: 0s for padding, 1s for tokens
        #   e.g., [1,1,1,1,1,1,1,1,1,1,1,0,0,0,0]
        # attention_mask: concatenation of prompt_attention_mask and response_attention_mask
        #   e.g., [0,0,0,0,1,1,1,1(prompt),1,1,1,1,1,1,1,1,1,1,1,0,0,0,0(response)]
        # - response_mask: 1s for LLM generated tokens, 0 for tool response/padding tokens
        #   e.g., [1,1,1,1,1,1,1,(tool start),0,0(tool end),1,1,0,0,0,0]
        # - position_ids: sequential positions for tokens, starting at 0
        #   e.g., [0,0,0,0,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,0,0,0,0]

        # TODO(wuxibin): remove padding and use tensordict.
        self.tokenizer.padding_side = "left"
        prompt_output = self.tokenizer.pad(
            {"input_ids": output.prompt_ids},
            padding="max_length",
            max_length=self.rollout_config.prompt_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        if prompt_output["input_ids"].dim() == 1:
            prompt_output["input_ids"] = prompt_output["input_ids"].unsqueeze(0)
            prompt_output["attention_mask"] = prompt_output["attention_mask"].unsqueeze(0)

        self.tokenizer.padding_side = "right"
        response_output = self.tokenizer.pad(
            {"input_ids": output.response_ids},
            padding="max_length",
            max_length=self.rollout_config.response_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        if response_output["input_ids"].dim() == 1:
            response_output["input_ids"] = response_output["input_ids"].unsqueeze(0)
            response_output["attention_mask"] = response_output["attention_mask"].unsqueeze(0)

        response_mask_output = self.tokenizer.pad(
            {"input_ids": output.response_mask},
            padding="max_length",
            max_length=self.rollout_config.response_length,
            return_tensors="pt",
            return_attention_mask=False,
        )
        if response_mask_output["input_ids"].dim() == 1:
            response_mask_output["input_ids"] = response_mask_output["input_ids"].unsqueeze(0)

        response_logprobs = None
        if output.response_logprobs is not None:
            pad_size = self.rollout_config.response_length - len(output.response_logprobs)
            response_logprobs = torch.tensor(output.response_logprobs + [0.0] * pad_size).unsqueeze(0)

        response_mask = response_mask_output["input_ids"] * response_output["attention_mask"]
        attention_mask = torch.cat([prompt_output["attention_mask"], response_output["attention_mask"]], dim=1)
        input_ids = torch.cat([prompt_output["input_ids"], response_output["input_ids"]], dim=1)

        routed_experts = None
        if output.routed_experts is not None:
            total_length = input_ids.shape[1]
            length, layer_num, topk_num = output.routed_experts.shape
            if isinstance(output.routed_experts, np.ndarray):
                routed_experts_array = output.routed_experts
                if not routed_experts_array.flags.writeable:
                    routed_experts_array = routed_experts_array.copy()
                experts_tensor = torch.from_numpy(routed_experts_array)
            elif isinstance(output.routed_experts, torch.Tensor):
                experts_tensor = output.routed_experts
            else:
                raise TypeError(f"Unsupported type for routed_experts: {type(output.routed_experts)}")
            routed_experts = torch.zeros(1, total_length, layer_num, topk_num, dtype=experts_tensor.dtype)

            # Calculate start position: left padding means original prompt starts at the end
            start_pos = prompt_output["input_ids"].shape[1] - len(output.prompt_ids)
            end_pos = min(start_pos + length, total_length)

            # Add boundary checks for robustness
            if start_pos < 0 or end_pos > total_length:
                raise ValueError(
                    f"Invalid position range: start_pos={start_pos}, end_pos={end_pos}, total_length={total_length}"
                )

            routed_experts[:, start_pos:end_pos] = experts_tensor.unsqueeze(0)

        multi_modal_inputs = self._compute_multi_modal_inputs(output, input_ids)
        position_ids = self._compute_position_ids(
            input_ids,
            attention_mask,
            multi_modal_inputs,
            output.mm_processor_kwargs
            if output.mm_processor_kwargs is not None
            else self._get_mm_processor_kwargs(
                output.multi_modal_data.get("audios") if output.multi_modal_data else None
            ),
        )
        if compute_score:
            await self._compute_score([output], kwargs=kwargs)
        # SPECULATION RUNS OUTSIDE compute_teacher_logprobs, and must.
        #
        # It targets NON-FINAL chunks, and in F mode -- the only mode omniopd streams in --
        # compute_teacher_logprobs is False for exactly those chunks (see the call site: it passes
        # `is_final or not _final_only`). Gating speculation behind that flag therefore meant it
        # could never fire on any chunk it was written for: the controller was live, the manifest
        # said speculate=1, and 96/96 trajectories reported spec=off/off. The first speculation A/B
        # was an A/A and its +1.3% measured nothing.
        if (self._omniopd_enabled and speculation_enabled() and chunk_is_final is False
                and self.distillation_enabled and not validate):
            if not getattr(self, "_omniopd_spec_announced", False):
                self._omniopd_spec_announced = True
                print("[OMNIOPD-SPEC] speculative early teacher launch ENABLED", flush=True)
            try:
                await self._omniopd_speculate(output, sample_kwargs=kwargs)
            except Exception:
                # An optimisation must never cost the trajectory its audit, which still runs in
                # full on the final chunk.
                logger.warning("[OMNIOPD-SPEC] proposal pass failed; commit is unaffected",
                               exc_info=True)

        if compute_teacher_logprobs:
            # OmniOPD replaces teacher SCORING with a teacher AUDIT: its objective reads k_sem and an
            # exact KL against the frozen initial policy, and never a teacher logprob. Running both
            # would pay for a full-sequence teacher forward whose result nothing consumes.
            if self._omniopd_enabled:
                await self._compute_omniopd_audit(
                    output,
                    prompt_ids=output.prompt_ids,
                    response_ids=output.response_ids,
                    validate=validate,
                    sample_kwargs=kwargs,
                    chunk_is_final=chunk_is_final,
                )
            else:
                await self._compute_teacher_logprobs(
                    output,
                    prompt_ids=output.prompt_ids,
                    response_ids=output.response_ids,
                    validate=validate,
                    sample_kwargs=kwargs,
                    chunk_is_final=chunk_is_final,
                )
        teacher_ids, teacher_logprobs = (
            output.extra_fields.pop("teacher_ids", None),
            output.extra_fields.pop("teacher_logprobs", None),
        )
        if teacher_ids is not None and teacher_logprobs is not None:
            # TODO(wuxibin): remove padding and use tensordict.
            from verl.experimental.teacher_loop.teacher_manager import _pad_teacher_outputs

            teacher_ids, teacher_logprobs = _pad_teacher_outputs(
                teacher_ids,
                teacher_logprobs,
                prompt_width=prompt_output["input_ids"].shape[1],
                response_width=response_output["input_ids"].shape[1],
                prompt_length=len(output.prompt_ids),
                response_length=len(output.response_ids),
                pad_token_id=self.tokenizer.pad_token_id,
            )

        return _InternalAgentLoopOutput(
            prompt_ids=prompt_output["input_ids"],
            response_ids=response_output["input_ids"],
            input_ids=input_ids,
            position_ids=position_ids,
            response_mask=response_mask,
            attention_mask=attention_mask,
            response_logprobs=response_logprobs,
            routed_experts=routed_experts,
            multi_modal_inputs=multi_modal_inputs,
            multi_modal_data=output.multi_modal_data,
            mm_processor_kwargs=output.mm_processor_kwargs,
            teacher_logprobs=teacher_logprobs,
            teacher_ids=teacher_ids,
            reward_score=output.reward_score,
            num_turns=output.num_turns,
            metrics=output.metrics,
            extra_fields=output.extra_fields,
        )

    def _compute_multi_modal_inputs(self, output, input_ids) -> dict[str, torch.Tensor]:
        """Compute multi-modal inputs with image, video and audio."""
        multi_modal_inputs = {}
        if self.processor is None:
            return multi_modal_inputs

        multi_modal_data = output.multi_modal_data or {}
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")
        audios = multi_modal_data.get("audios")
        current_text = self.tokenizer.decode(input_ids.squeeze(0), skip_special_tokens=True)

        multi_modal_inputs = build_multimodal_processor_inputs(
            self.processor,
            text=[current_text],
            images=images,
            videos=videos,
            audio=audios,
            mm_processor_kwargs=output.mm_processor_kwargs
            if output.mm_processor_kwargs is not None
            else self._get_mm_processor_kwargs(audios),
        )
        multi_modal_inputs.pop("input_ids", None)
        multi_modal_inputs.pop("attention_mask", None)

        # We must use dict(multi_modal_inputs) to convert BatchFeature values to a new dict
        # because np.array() only keeps the keys for BatchFeature.
        multi_modal_inputs = dict(multi_modal_inputs.convert_to_tensors("pt"))
        image_grid_thw = multi_modal_inputs.get("image_grid_thw")
        if image_grid_thw is not None:
            images_seqlens = torch.repeat_interleave(image_grid_thw[:, 1] * image_grid_thw[:, 2], image_grid_thw[:, 0])
            multi_modal_inputs["images_seqlens"] = images_seqlens
        return multi_modal_inputs

    def _compute_position_ids(
        self,
        input_ids,
        attention_mask,
        multi_modal_inputs,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Compute position ids for multi-modal inputs."""
        if self.processor is None:
            return compute_position_id_with_mask(attention_mask)  # (1, seq_len)

        multi_modal_kwargs = {
            "image_grid_thw": multi_modal_inputs.get("image_grid_thw"),
            "video_grid_thw": multi_modal_inputs.get("video_grid_thw"),
        }
        # For transformers>=5.3.0, mm_token_type_ids is only used to calculate position ids.
        if multi_modal_inputs.pop("mm_token_type_ids", None) is not None:
            mm_token_type_ids = torch.zeros_like(input_ids)
            image_token_id = get_processor_token_id(self.processor, "image")
            video_token_id = get_processor_token_id(self.processor, "video")
            if image_token_id is not None:
                mm_token_type_ids[0][input_ids[0] == image_token_id] = 1
            if video_token_id is not None:
                mm_token_type_ids[0][input_ids[0] == video_token_id] = 2
            multi_modal_kwargs["mm_token_type_ids"] = mm_token_type_ids

        # Model's get_rope_index has been dynamically bind to the processor.
        vision_position_ids, _ = self.processor.get_rope_index(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **multi_modal_kwargs,
        )
        vision_position_ids = vision_position_ids.transpose(0, 1)  # (3, 1, seq_len) => (1, 3, seq_len)

        valid_mask = attention_mask[0].bool()
        text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
        text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
        text_position_ids = text_position_ids.unsqueeze(0)
        position_ids = torch.cat((text_position_ids, vision_position_ids), dim=1)  # (1, 4, seq_length)
        return position_ids

    async def _compute_score(self, outputs: list[AgentLoopOutput], kwargs: dict) -> None:
        """Compute reward score for all outputs in a trajectory; assigns result to outputs[-1]."""
        enable_async_reward = self.reward_loop_worker_handles is not None

        final_output = outputs[-1]
        if final_output.reward_score is None and enable_async_reward:
            timing = {}
            with simple_timer("compute_score", timing):
                all_prompts, all_responses, all_input_ids, all_attention_mask, all_position_ids = [], [], [], [], []
                for output in outputs:
                    prompts = torch.tensor(output.prompt_ids, dtype=torch.int64)
                    responses = torch.tensor(output.response_ids, dtype=torch.int64)
                    input_ids = torch.cat([prompts, responses], dim=0)
                    attention_mask = torch.ones_like(input_ids, dtype=torch.int64)
                    multi_modal_inputs = self._compute_multi_modal_inputs(output, input_ids)
                    position_ids = self._compute_position_ids(
                        input_ids.unsqueeze(0),
                        attention_mask.unsqueeze(0),
                        multi_modal_inputs,
                        output.mm_processor_kwargs
                        if output.mm_processor_kwargs is not None
                        else self._get_mm_processor_kwargs(
                            output.multi_modal_data.get("audios") if output.multi_modal_data else None
                        ),
                    ).squeeze(0)
                    all_prompts.append(prompts)
                    all_responses.append(responses)
                    all_input_ids.append(input_ids)
                    all_attention_mask.append(attention_mask)
                    all_position_ids.append(position_ids)

                n = len(outputs)
                batch = TensorDict(
                    {
                        "prompts": torch.nn.utils.rnn.pad_sequence(all_prompts, batch_first=True, padding_value=0),
                        "responses": torch.nn.utils.rnn.pad_sequence(all_responses, batch_first=True, padding_value=0),
                        "attention_mask": torch.nn.utils.rnn.pad_sequence(
                            all_attention_mask, batch_first=True, padding_value=0
                        ),
                        "input_ids": torch.nn.utils.rnn.pad_sequence(all_input_ids, batch_first=True, padding_value=0),
                        "position_ids": torch.nn.utils.rnn.pad_sequence(
                            all_position_ids, batch_first=True, padding_value=0
                        ),
                    },
                    batch_size=n,
                )
                non_tensor_batch = {
                    **{k: np.array([v] * n) for k, v in kwargs.items()},
                    "__num_turns__": np.array([o.num_turns for o in outputs]),
                    "tool_extra_fields": np.array([o.extra_fields for o in outputs], dtype=object),
                    "prompt_len": np.array([len(o.prompt_ids) for o in outputs]),
                    "response_len": np.array([len(o.response_ids) for o in outputs]),
                }

                data = DataProto(
                    batch=batch,
                    non_tensor_batch=non_tensor_batch,
                )
                selected_reward_loop_worker_handle = random.choice(self.reward_loop_worker_handles)
                result = await selected_reward_loop_worker_handle.compute_score.remote(data)
                final_output.reward_score = result["reward_score"]
                final_output.extra_fields["reward_extra_info"] = result["reward_extra_info"]
            final_output.metrics.compute_score = timing["compute_score"]

    def _omniopd_store(self, session_id, base_seed, om):
        """One store per trajectory, created on first use and handed to the commit."""
        if not hasattr(self, "_omniopd_spec_stores"):
            self._omniopd_spec_stores = {}
        key = str(session_id)
        st = self._omniopd_spec_stores.get(key)
        if st is None:
            st = SpeculativeStore(base_seed, int(om.N), int(om.C))
            self._omniopd_spec_stores[key] = st
        return st

    async def _omniopd_speculate(self, output, *, sample_kwargs=None) -> None:
        """Propose anchors from the entropy prefix and launch their teacher work early.

        Writes nothing to output. The commit on the final chunk selects from the COMPLETE series,
        exactly as it always has, and consults the store only for anchors it has already chosen.
        """
        ents = output.extra_fields.get("token_entropies")
        resp = output.response_ids
        if not ents or not resp or len(ents) != len(resp):
            return                                   # nothing to propose from; commit is unaffected
        om = resolve_omniopd_config(self.config)
        routing_key = session_id = None
        if sample_kwargs is not None:
            rv = sample_kwargs.get(self.teacher_key)
            if rv is not None:
                routing_key = rv.item() if hasattr(rv, "item") else rv
            sid = sample_kwargs.get("xiaoshuai_sample_id")
            if sid is not None:
                session_id = sid.item() if hasattr(sid, "item") else sid
        if session_id is None:
            return
        base_seed = _omniopd_base_seed(session_id)
        store = self._omniopd_store(session_id, base_seed, om)

        margin = float(os.environ.get("OPD_OMNIOPD_SPEC_MARGIN", "0") or 0.0)
        proposed = propose_anchors(ents, len(output.prompt_ids), len(resp),
                                   int(om.M), int(om.C), margin=margin)
        fresh = store.pending(proposed)
        if not fresh:
            return
        prefixes = build_chunk_requests(output.prompt_ids, resp, fresh)
        for t0, prefix in zip(fresh, prefixes, strict=True):
            def _factory(_p=prefix, _t=int(t0)):
                # is_final is ALWAYS False here. The sticky-parent release belongs to the commit's
                # last call, which is why the commit never reuses its final anchor.
                return self.teacher_server_manager.generate_chunk_continuations(
                    prefix_ids=_p, n=int(om.N), max_tokens=int(om.C),
                    routing_key=routing_key, session_id=session_id,
                    seed=seed_for_anchor(base_seed, _t, int(om.N)), is_final=False,
                )
            store.launch(int(t0), _factory)

    async def _compute_omniopd_audit(
        self,
        output: AgentLoopOutput,
        prompt_ids: list[int],
        response_ids: list[int],
        validate: bool,
        sample_kwargs: Optional[dict[str, Any]] = None,
        chunk_is_final: Optional[bool] = None,
    ) -> None:
        """Select audit anchors from student entropy, have the teacher rewrite them, score k_sem.

        The audit is a whole-response operation -- anchors are a global argmax over the FINISHED
        response -- so unlike incremental scoring it has no per-chunk form and runs exactly once.

        THAT IS WHY chunk_is_final IS LOAD-BEARING. Under chunk streaming this postprocess runs once
        per chunk; without the gate the audit would fire on every partial response, spending M teacher
        generations per chunk and then overwriting its own anchors with those of a prefix. The last
        writer would win, so the trajectory would train against anchors chosen from a truncated
        response -- expensive, wrong, and completely silent. None means the caller is not streaming.
        """
        if not (self.distillation_enabled and not validate):
            return
        if chunk_is_final is False:
            # Never the audit on a non-final chunk. Speculation is driven from
            # _agent_loop_postprocess instead, because it must run on chunks whose
            # compute_teacher_logprobs is False -- which is all of them in F mode.
            return
        routing_key = session_id = None
        if sample_kwargs is not None:
            routing_value = sample_kwargs.get(self.teacher_key)
            if routing_value is not None:
                routing_key = routing_value.item() if hasattr(routing_value, "item") else routing_value
            # Same per-response id the scoring path uses: it pins this trajectory's M nested chunk
            # prefixes to one replica, which is what makes each prefill reuse the previous one's KV.
            sid = sample_kwargs.get("xiaoshuai_sample_id")
            if sid is not None:
                session_id = sid.item() if hasattr(sid, "item") else sid
        _store = None
        if speculation_enabled() and session_id is not None:
            _store = getattr(self, "_omniopd_spec_stores", {}).pop(str(session_id), None)
        await attach_omniopd_audit(
            output,
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            teacher_manager=self.teacher_server_manager,
            tokenizer=self.tokenizer,
            omniopd_config=resolve_omniopd_config(self.config),
            session_id=session_id,
            routing_key=routing_key,
            store=_store,
            seed=_omniopd_base_seed(session_id),
        )

    async def _compute_teacher_logprobs(
        self,
        output: AgentLoopOutput,
        prompt_ids: list[int],
        response_ids: list[int],
        validate: bool,
        sample_kwargs: Optional[dict[str, Any]] = None,
        chunk_is_final: Optional[bool] = None,
    ) -> None:
        """Compute teacher logprobs for single sample."""
        if self.distillation_enabled and not validate:
            routing_key = None
            session_id = None
            if sample_kwargs is not None:
                routing_value = sample_kwargs.get(self.teacher_key)
                if routing_value is not None:
                    # Non-tensor batch values arrive as 0-d numpy objects / arrays; normalize to Python.
                    routing_key = routing_value.item() if hasattr(routing_value, "item") else routing_value
                # Per-RESPONSE id pins this response's chunks to one teacher replica (KV reuse, gated).
                sid = sample_kwargs.get("xiaoshuai_sample_id")
                if sid is not None:
                    session_id = sid.item() if hasattr(sid, "item") else sid
            # Incremental-scoring span coordinates, derived from this chunk's response_mask
            # ([0]*token_offset + [1]*n_new_tokens, single_turn_agent_loop.py:390): span_start = first 1,
            # n_tokens = sum. The teacher then re-scores only this span (gated OPD_TEACHER_INCREMENTAL_SCORE).
            span_start = span_end = prompt_width = None
            rm = output.response_mask
            if rm is not None and any(rm):
                span_start = list(rm).index(1)
                n_tokens = int(sum(rm))
                span_end = span_start + n_tokens
                prompt_width = len(prompt_ids)
            teacher_ids, teacher_logprobs, teacher_telemetry = await self.teacher_server_manager.compute_teacher_logprobs_single(
                sequence_ids=prompt_ids + response_ids,
                multi_modal_data=output.multi_modal_data,
                mm_processor_kwargs=output.mm_processor_kwargs,
                routing_key=routing_key,
                session_id=session_id,
                span_start=span_start,
                span_end=span_end,
                prompt_width=prompt_width,
                is_final=chunk_is_final,
            )
            if teacher_telemetry.get("incremental"):
                # Teacher returned ONLY the new span [n, k]. Re-place it into a full-prefix-aligned [S, k]
                # tensor (zeros for the cached prefix we did not re-score), so the existing span-only slice
                # and every downstream consumer are byte-for-byte unchanged. The span-only emission strips
                # the zeros back to [n, k] on the wire; compute was still incremental.
                S = len(prompt_ids) + len(response_ids)
                k = teacher_ids.shape[1]
                full_ids = torch.zeros(S, k, dtype=torch.int32)
                full_lps = torch.zeros(S, k, dtype=torch.float32)
                # -1: see the convention note in teacher_manager. A chunk owning response tokens
                # [s, e) fills teacher indices [P+s-1, P+e-1).
                ss = prompt_width + span_start - 1
                full_ids[ss:ss + (span_end - span_start)] = teacher_ids
                full_lps[ss:ss + (span_end - span_start)] = teacher_logprobs
                teacher_ids, teacher_logprobs = full_ids, full_lps
            output.extra_fields["teacher_ids"] = teacher_ids
            output.extra_fields["teacher_logprobs"] = teacher_logprobs
            output.extra_fields["teacher_telemetry"] = teacher_telemetry

    def _postprocess(
        self,
        inputs: list[_InternalAgentLoopOutput],
        input_non_tensor_batch: dict | None = None,
        validate: bool = False,
    ) -> DataProto:
        """Process the padded outputs from _run_agent_loop and combine them into a batch."""
        # Convert lists back to tensors and stack them to create a batch.
        prompt_ids = torch.cat([input.prompt_ids for input in inputs], dim=0)
        response_ids = torch.cat([input.response_ids for input in inputs], dim=0)
        response_mask = torch.cat([input.response_mask for input in inputs], dim=0)
        attention_mask = torch.cat([input.attention_mask for input in inputs], dim=0)
        input_ids = torch.cat([input.input_ids for input in inputs], dim=0)
        position_ids = torch.cat([input.position_ids for input in inputs], dim=0)
        optional_outputs = {}
        if inputs[0].response_logprobs is not None:
            optional_outputs["rollout_log_probs"] = torch.cat([input.response_logprobs for input in inputs], dim=0)
        if inputs[0].routed_experts is not None:
            optional_outputs["routed_experts"] = torch.cat([input.routed_experts for input in inputs], dim=0)
        if inputs[0].teacher_logprobs is not None and inputs[0].teacher_ids is not None:
            # A per-sample teacher call can time out under teacher SATURATION and return None (its
            # scoring latency exceeds the FIFO window; TAIL shows is_final=None, latency_s>>timeout).
            # The old guard checked only inputs[0], so a None in a LATER input crashed torch.cat with a
            # cryptic "expected Tensor ... got NoneType". Fail LOUDLY and actionably instead -- this is a
            # provisioning signal (rollout:teacher chunk-arrival rate too high), not a code bug. The
            # healthy path (all inputs labelled) is unchanged.
            _missing = [i for i, x in enumerate(inputs) if x.teacher_logprobs is None or x.teacher_ids is None]
            if _missing:
                raise RuntimeError(
                    f"[agent_loop._postprocess] teacher saturation: {len(_missing)}/{len(inputs)} samples have "
                    f"None teacher labels (idx {_missing[:8]}); a teacher call exceeded its window and returned "
                    f"None. Lower the rollout:teacher chunk rate (fewer rollout GPUs, larger c_stream, or more "
                    f"teacher replicas).")
            optional_outputs["teacher_logprobs"] = torch.cat([input.teacher_logprobs for input in inputs], dim=0)
            optional_outputs["teacher_ids"] = torch.cat([input.teacher_ids for input in inputs], dim=0)
        batch = TensorDict(
            {
                "prompts": prompt_ids,  # [bsz, prompt_length]
                "responses": response_ids,  # [bsz, response_length]
                "response_mask": response_mask,  # [bsz, response_length]
                "input_ids": input_ids,  # [bsz, prompt_length + response_length]
                "attention_mask": attention_mask,  # [bsz, prompt_length + response_length]
                # position_ids: [bsz, 3, prompt_length + response_length] or [bsz, prompt_length + response_length]
                "position_ids": position_ids,
                **optional_outputs,
            },
            batch_size=len(inputs),
        )

        scores = [input.reward_score for input in inputs]
        if all(score is not None for score in scores):
            prompt_length = prompt_ids.size(1)
            response_length = attention_mask[:, prompt_length:].sum(dim=1) - 1
            rm_scores = torch.zeros_like(response_mask, dtype=torch.float32)
            rm_scores[torch.arange(response_mask.size(0)), response_length] = torch.tensor(scores, dtype=torch.float32)
            batch["rm_scores"] = rm_scores

        non_tensor_batch = {
            "__num_turns__": np.array([input.num_turns for input in inputs], dtype=np.int32),
        }
        if self.reward_loop_worker_handles is None and input_non_tensor_batch:
            non_tensor_batch.update(input_non_tensor_batch)

        # add reward_extra_info to non_tensor_batch
        reward_extra_infos = [input.extra_fields.get("reward_extra_info", {}) for input in inputs]
        reward_extra_keys = list(reward_extra_infos[0].keys())
        for key in reward_extra_keys:
            non_tensor_batch[key] = np.array([info[key] for info in reward_extra_infos])

        # Add multi_modal_inputs to non_tensor_batch if any samples have them
        multi_modal_inputs_list = [input.multi_modal_inputs for input in inputs]
        if any(mmi is not None for mmi in multi_modal_inputs_list):
            non_tensor_batch["multi_modal_inputs"] = np.array(multi_modal_inputs_list, dtype=object)

        metrics = [input.metrics.model_dump() for input in inputs]
        # Collect extra fields from all inputs and convert them to np.ndarray
        # Keep a stable set of keys so downstream batch concat stays consistent across agent loops.
        extra_fields = {}
        default_extra_keys = {
            "turn_scores",
            "tool_rewards",
            "min_global_steps",
            "max_global_steps",
            "extras",
        }
        all_keys = set(key for input_item in inputs for key in input_item.extra_fields) | default_extra_keys
        for key in all_keys:
            temp_arr = np.empty(len(inputs), dtype=object)
            temp_arr[:] = [input.extra_fields.get(key) for input in inputs]
            extra_fields[key] = temp_arr

        non_tensor_batch.update(extra_fields)

        # Only include reward_extra_keys in meta_info if rm_scores is in batch
        # This avoids conflicts when reward_tensor is merged later in ray_trainer.py
        if "rm_scores" in batch.keys():
            meta_info = {"metrics": metrics, "reward_extra_keys": reward_extra_keys}
        else:
            meta_info = {"metrics": metrics}

        return DataProto(
            batch=batch,
            non_tensor_batch=non_tensor_batch,
            meta_info=meta_info,
        )


async def get_trajectory_info(step, index, validate):
    """Get trajectory info.

    Args:
        step (int): global steps in the trainer.
        index (list): form datastore extra_info.index column.
        validate (bool): whether is a validate step.

    Returns:
        list: trajectory.
    """
    trajectory_info = []
    rollout_n = 0
    for i in range(len(index)):
        if i > 0 and index[i - 1] == index[i]:
            rollout_n += 1
        else:
            rollout_n = 0
        trajectory_info.append({"step": step, "sample_index": index[i], "rollout_n": rollout_n, "validate": validate})
    return trajectory_info


class AgentLoopManager:
    """Agent loop manager that manages a group of agent loop workers.

    Args:
        config (DictConfig): whole config for main entrypoint.
        llm_client (LLMServerClient): Client for the LLM server.
        teacher_client (dict[str, LLMServerClient]): Client for multiple teacher servers.
        reward_loop_worker_handles (List[ray.actor.ActorHandle]): Actor handles for streaming reward computation.
    """

    def __init__(
        self,
        config: DictConfig,
        llm_client: LLMServerClient,
        teacher_client: dict[str, LLMServerClient] = None,
        reward_loop_worker_handles: list[ray.actor.ActorHandle] = None,
    ):
        self.config = config
        self.rollout_config = config.actor_rollout_ref.rollout
        self.model_config = config.actor_rollout_ref.model
        self.llm_client = llm_client
        self.teacher_client = teacher_client
        self.reward_loop_worker_handles = reward_loop_worker_handles

        if not hasattr(self, "agent_loop_workers_class"):
            self.agent_loop_workers_class = ray.remote(AgentLoopWorker)

    @classmethod
    @auto_await
    async def create(cls, *args, **kwargs):
        """Create agent loop manager."""
        instance = cls(*args, **kwargs)
        await instance._init_agent_loop_workers()
        return instance

    async def _init_agent_loop_workers(self):
        self.agent_loop_workers = []
        num_workers = self.rollout_config.agent.num_workers

        node_ids = [node["NodeID"] for node in ray.nodes() if node["Alive"] and node["Resources"].get("CPU", 0) > 0]
        for i in range(num_workers):
            # Round-robin scheduling over the all nodes
            node_id = node_ids[i % len(node_ids)]
            self.agent_loop_workers.append(
                self.agent_loop_workers_class.options(
                    name=f"agent_loop_worker_{i}" + f"_{uuid4().hex[:8]}",
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=node_id, soft=True
                    ),
                ).remote(
                    self.config,
                    self.llm_client,
                    self.teacher_client,
                    self.reward_loop_worker_handles,
                )
            )

    @auto_await
    async def set_chunk_message_queue_client(self, message_queue_client) -> None:
        """Forward the fully-async chunk queue client to all agent loop workers."""
        await asyncio.gather(
            *(worker.set_chunk_message_queue_client.remote(message_queue_client) for worker in self.agent_loop_workers)
        )

    @auto_await
    async def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Split input batch and dispatch to agent loop workers.

        Args:
            prompts (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
        """
        chunkes = prompts.chunk(len(self.agent_loop_workers))
        outputs = await asyncio.gather(
            *[
                worker.generate_sequences.remote(chunk)
                for worker, chunk in zip(self.agent_loop_workers, chunkes, strict=True)
            ]
        )
        output = DataProto.concat(outputs)

        # calculate performance metrics
        metrics = [output.meta_info.pop("metrics") for output in outputs]  # List[List[Dict[str, str]]]
        timing = self._performance_metrics(metrics, output)

        output.meta_info = {"timing": timing, **outputs[0].meta_info}
        return output

    def _performance_metrics(self, metrics: list[list[dict[str, str]]], output: DataProto) -> dict[str, float]:
        timing = {}
        t_generate_sequences = np.array([metric["generate_sequences"] for chunk in metrics for metric in chunk])
        t_tool_calls = np.array([metric["tool_calls"] for chunk in metrics for metric in chunk])
        t_compute_score = np.array([metric["compute_score"] for chunk in metrics for metric in chunk])
        num_preempted = np.array([metric["num_preempted"] for chunk in metrics for metric in chunk])
        timing["agent_loop/num_preempted/min"] = num_preempted.min()
        timing["agent_loop/num_preempted/max"] = num_preempted.max()
        timing["agent_loop/num_preempted/mean"] = num_preempted.mean()
        timing["agent_loop/generate_sequences/min"] = t_generate_sequences.min()
        timing["agent_loop/generate_sequences/max"] = t_generate_sequences.max()
        timing["agent_loop/generate_sequences/mean"] = t_generate_sequences.mean()
        timing["agent_loop/tool_calls/min"] = t_tool_calls.min()
        timing["agent_loop/tool_calls/max"] = t_tool_calls.max()
        timing["agent_loop/tool_calls/mean"] = t_tool_calls.mean()
        timing["agent_loop/compute_score/min"] = t_compute_score.min()
        timing["agent_loop/compute_score/max"] = t_compute_score.max()
        timing["agent_loop/compute_score/mean"] = t_compute_score.mean()

        # batch sequence generation is bounded by the slowest sample
        slowest = np.argmax(t_generate_sequences + t_tool_calls + t_compute_score)
        prompt_length = output.batch["prompts"].shape[1]
        timing["agent_loop/slowest/generate_sequences"] = t_generate_sequences[slowest]
        timing["agent_loop/slowest/tool_calls"] = t_tool_calls[slowest]
        timing["agent_loop/slowest/compute_score"] = t_compute_score[slowest]
        timing["agent_loop/slowest/num_preempted"] = num_preempted[slowest]

        if "attention_mask" in output.batch:
            attention_mask = output.batch["attention_mask"][slowest]
            timing["agent_loop/slowest/prompt_length"] = attention_mask[:prompt_length].sum().item()
            timing["agent_loop/slowest/response_length"] = attention_mask[prompt_length:].sum().item()

        return timing
