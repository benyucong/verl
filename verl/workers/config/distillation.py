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

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from verl.base_config import BaseConfig
from verl.utils.config import omega_conf_to_dataclass

from .rollout import RolloutConfig

__all__ = ["DistillationLossConfig", "DistillationTeacherModelConfig", "DistillationConfig"]

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@dataclass
class DistillationLossConfig(BaseConfig):
    """Configuration for distillation loss settings.

    loss_mode (str):
        Distillation loss function to use.
    topk (int, optional):
        Number of top tokens to consider for top-k distillation losses.
    use_task_rewards (bool):
        Whether to include task rewards alongside distillation loss.
    distillation_loss_coef (float):
        Coefficient for distillation loss when combined with task rewards.
    loss_max_clamp (float, optional):
        Maximum value to clamp distillation loss. If None, no clamping is applied.
    log_prob_min_clamp (float, optional):
        Minimum value to clamp log probabilities for stability, e.g., log q - log p where p or q are
        very close to zero. If None, no clamping is applied.
    use_policy_gradient (bool):
        Whether to incorporate distillation loss as a reward, as done
        by https://thinkingmachines.ai/blog/on-policy-distillation/. Recommended to use loss_mode=k1.
        Otherwise, distillation loss is directly backpropagated as a supervised loss,
        as in https://arxiv.org/abs/2306.13649. Recommended to use loss_mode=k3 or forward_kl_topk.
    policy_loss_mode (str):
        Name of the policy loss to use when use_policy_gradient is true.
    clip_ratio (float):
        PPO clipping ratio for policy loss.
    clip_ratio_low (float):
        Lower bound for PPO clipping ratio.
    clip_ratio_high (float):
        Upper bound for PPO clipping ratio.
    loss_settings (DistillationLossSettings, optional):
        Runtime-populated settings based on loss_mode. Not set by user.
    """

    loss_mode: str = "k3"
    topk: Optional[int] = 128
    use_task_rewards: bool = True
    distillation_loss_coef: float = 1.0
    loss_max_clamp: Optional[float] = 10.0
    log_prob_min_clamp: Optional[float] = -10.0

    # --- TIP-style post-teacher token selection (loss masking). Default OFF => byte-equivalent. ---
    # token_select_mode: "none" | "entropy" | "soft_or"; token_retention in (0,1]; token_select_scope:
    # "response" (per-rollout top-rho, paper-faithful) | "batch" (per-micro-batch global top-rho).
    # Env-overridable via OPD_TOKEN_SELECT_MODE / OPD_TOKEN_RETENTION / OPD_TOKEN_SELECT_SCOPE /
    # OPD_TOKEN_SELECT_LOG (read in __post_init__).
    token_select_mode: str = "none"
    token_retention: float = 1.0
    token_select_scope: str = "response"
    token_select_log: bool = False

    use_policy_gradient: bool = True
    policy_loss_mode: str = "vanilla"
    clip_ratio: float = 0.2
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.2

    # Store global batch info for loss aggregation:
    # dp_size: data parallel size
    # batch_num_tokens: number of valid tokens in global batch
    # global_batch_size: global batch size
    global_batch_info: dict = field(default_factory=dict)

    # Store distillation loss settings for computing the specified loss_mode
    # Not set by user, populated at runtime
    loss_settings: Optional[dict] = None

    def __post_init__(self):
        self._mutable_fields.add("loss_settings")
        from verl.trainer.distillation.losses import DistillationLossSettings, get_distillation_loss_settings

        self.loss_settings: DistillationLossSettings = get_distillation_loss_settings(self.loss_mode)

        # --- TIP-style token-selection env overrides (default OFF => byte-equivalent) ---
        for _f in ("token_select_mode", "token_retention", "token_select_scope", "token_select_log"):
            self._mutable_fields.add(_f)
        self.token_select_mode = os.environ.get("OPD_TOKEN_SELECT_MODE", self.token_select_mode)
        self.token_retention = float(os.environ.get("OPD_TOKEN_RETENTION", self.token_retention))
        self.token_select_scope = os.environ.get("OPD_TOKEN_SELECT_SCOPE", self.token_select_scope)
        if "OPD_TOKEN_SELECT_LOG" in os.environ:
            self.token_select_log = bool(int(os.environ["OPD_TOKEN_SELECT_LOG"]))
        if self.token_select_mode not in ("none", "random", "entropy", "soft_or"):
            raise ValueError(
                f"OPD_TOKEN_SELECT_MODE must be none|random|entropy|soft_or, got {self.token_select_mode!r}"
            )
        if not (0.0 < self.token_retention <= 1.0):
            raise ValueError(f"OPD_TOKEN_RETENTION must be in (0, 1], got {self.token_retention}")
        if self.token_select_scope not in ("response", "batch"):
            raise ValueError(f"OPD_TOKEN_SELECT_SCOPE must be response|batch, got {self.token_select_scope!r}")
        if self.token_select_mode != "none":
            print(
                f"[TOKEN-SELECT-CFG] resolved mode={self.token_select_mode} retention={self.token_retention} "
                f"scope={self.token_select_scope} log={self.token_select_log}",
                flush=True,
            )

        if self.policy_loss_mode != "vanilla":
            raise NotImplementedError(
                f"Only vanilla policy loss is currently supported when use_policy_gradient is True, "
                f"but got {self.policy_loss_mode}."
            )

        if self.use_policy_gradient and self.loss_mode == "forward_kl_topk":
            print(
                "WARNING: forward_kl_topk is most effective as a supervised distillation loss "
                "(use_policy_gradient=False). With policy gradient, the update uses only the sampled"
                " token's logprob ∇logπ(a), so the top-k distributional signal (how non-sampled logits "
                "should move) is largely unused."
            )

        if not self.use_policy_gradient and self.loss_mode == "k1":
            raise ValueError(
                "Directly backpropagating k1 loss is incorrect since gradient of k1 loss"
                " wrt model weights does not depend on teacher log probabilities."
            )


@dataclass
class DistillationTeacherModelConfig(BaseConfig):
    """Configuration for on-policy distillation teacher.

    key (str, optional):
        Identifier to route examples to the teacher model in multi-teacher setting.
    model_path (str, optional):
        Model path for the teacher model. Can be a local path or a Hugging Face model
    inference (RolloutConfig):
        Rollout configuration for the teacher model inference during distillation.
    num_replicas (int):
        Number of inference replicas of this teacher to launch. Each replica occupies
        `per_replica_world_size` GPUs (= inference.data_parallel_size *
        inference.tensor_model_parallel_size * inference.pipeline_model_parallel_size),
        so the teacher's total GPU footprint is
        `num_replicas * per_replica_world_size`.
    """

    _mutable_fields = BaseConfig._mutable_fields | {"num_replicas", "key"}

    key: Optional[str] = None
    model_path: Optional[str] = None
    inference: RolloutConfig = field(default_factory=RolloutConfig)
    num_replicas: Optional[int] = 0

    @property
    def per_replica_world_size(self) -> int:
        return (
            self.inference.tensor_model_parallel_size
            * self.inference.data_parallel_size
            * self.inference.pipeline_model_parallel_size
        )

    @property
    def world_size(self) -> int:
        return self.num_replicas * self.per_replica_world_size

    def check_configured(self):
        if self.model_path is None:
            raise ValueError("model_path must be specified for distillation teacher model config.")
        if self.key is None:
            raise ValueError("key must be specified for distillation teacher model config.")
        if self.num_replicas is None:
            raise ValueError("num_replicas must be specified for distillation teacher model config.")

    def validate_and_prepare_for_distillation(
        self, use_topk: bool, topk: Optional[int], generation_tokens: Optional[int] = None
    ) -> None:
        """Dimension the teacher engine for how this objective actually uses it.

        A SCORING teacher (top-k or estimator objectives) reads prompt+response and emits one token,
        so its whole context is prompt, and response_length collapses to 1.

        A GENERATING teacher (OmniOPD) is the opposite: it reads a PREFIX that stops before the
        audited chunk and then writes `generation_tokens` of its own. Leaving response_length at 1
        here is what makes the teacher structurally unable to generate -- the engine would be built
        with room for a single token, and every continuation would come back empty or truncated with
        no indication that the configuration, not the model, was responsible.

        The prefix is at most prompt + response - C, because a chunk of C tokens must fit inside the
        response after its anchor. So generation costs the engine NOTHING extra over scoring: the
        same prompt+response total is merely split differently.
        """
        max_model_len = self.inference.max_model_len
        student_prompt_length = self.inference.prompt_length
        student_response_length = self.inference.response_length
        required_context_len = student_prompt_length + student_response_length + 1
        if max_model_len is not None and required_context_len > max_model_len:
            raise ValueError(
                "Distillation teacher inference requires room for the student prompt, the full student "
                f"response, and one generated token, but got {student_prompt_length=}, "
                f"{student_response_length=}, {required_context_len=}, {max_model_len=}."
            )

        if generation_tokens is not None:
            if generation_tokens < 1:
                raise ValueError(f"generation_tokens must be >= 1, got {generation_tokens}")
            if generation_tokens >= student_response_length:
                raise ValueError(
                    f"Generative teaching needs a chunk of {generation_tokens} tokens to fit inside "
                    f"the student response ({student_response_length}); with C >= the response "
                    f"length no anchor can be placed and the teacher would have nothing to continue."
                )
            # prefix is at most prompt + response - C; the teacher then writes C
            self.inference.prompt_length = student_prompt_length + student_response_length - generation_tokens
            self.inference.response_length = generation_tokens
        else:
            self.inference.prompt_length = self.inference.prompt_length + self.inference.response_length
            self.inference.response_length = 1
        self._validate_topk_logprobs(use_topk=use_topk, topk=topk)

    def _validate_topk_logprobs(self, use_topk: bool, topk: Optional[int]) -> None:
        if not use_topk:
            return
        if topk is None:
            raise ValueError("topk must be specified when use_topk is True.")

        engine_name = self.inference.name
        engine_kwargs = self.inference.engine_kwargs
        match engine_name:
            case "vllm":
                vllm_engine_kwargs = dict(engine_kwargs.get("vllm", {}))
                max_logprobs = vllm_engine_kwargs.get("max_logprobs")
                if max_logprobs is None:
                    vllm_engine_kwargs["max_logprobs"] = topk
                    max_logprobs = topk
                if max_logprobs < topk:
                    raise ValueError(
                        f"VLLM max_logprobs ({max_logprobs}) must be >= distillation_loss topk "
                        f"({topk}) to enable distillation loss computation."
                    )
                engine_kwargs["vllm"] = vllm_engine_kwargs
            case "sglang":
                # SGLang's top_logprobs_num is a per-request parameter, so there is no
                # engine-boot cap to align (unlike vLLM's max_logprobs). The async
                # server translates sampling_params["prompt_logprobs"] into
                # return_logprob + logprob_start_len=0 + top_logprobs_num at call time.
                pass
            case _:
                raise NotImplementedError(
                    f"DistillationTeacherModelConfig does not support inference engine {engine_name}"
                )


@dataclass
class OmniOPDConfig(BaseConfig):
    """Parameters of the OmniOPD generative-critic objective.

    M (int): audited chunks per trajectory, taken at the highest full-vocabulary entropy positions.
    N (int): teacher continuations sampled per chunk.
    C (int): tokens per chunk, and per teacher continuation.
    alpha (float): Dirichlet-Multinomial smoothing of the Bayesian target.
    beta (float): weight of the trust-region anchor on unaudited positions.
    phi (str): semantic metric, "ned" or "rouge1". REQUIRED to be explicit -- a silently defaulted
        metric changes k_sem and therefore the objective (FID-3).
    entropy_topk (int): 0 means exact full-vocabulary entropy for chunk selection. FID-1: entropy
        from a truncated tail systematically under-estimates H and selects DIFFERENT chunks, which
        changes the algorithm rather than its cost. Non-zero is an explicit, recorded deviation.
    """

    M: int = 10
    N: int = 10
    C: int = 50
    alpha: float = 1.0
    beta: float = 0.1
    phi: str = "ned"
    entropy_topk: int = 0

    def __post_init__(self):
        if self.phi not in ("ned", "rouge1"):
            raise ValueError(f"omniopd.phi must be 'ned' or 'rouge1', got {self.phi!r}")
        for name in ("M", "N", "C"):
            if getattr(self, name) < 1:
                raise ValueError(f"omniopd.{name} must be >= 1, got {getattr(self, name)}")
        if self.alpha <= 0:
            raise ValueError(f"omniopd.alpha must be > 0, got {self.alpha}")
        if self.beta < 0:
            raise ValueError(f"omniopd.beta must be >= 0, got {self.beta}")
        if self.entropy_topk != 0:
            logger.warning(
                "omniopd.entropy_topk=%d deviates from FID-1: chunk selection is argmax_M over the "
                "student's FULL-vocabulary entropy, and a truncated tail under-estimates H and "
                "selects different chunks. This changes the algorithm, not its speed.",
                self.entropy_topk,
            )


@dataclass
class DistillationConfig(BaseConfig):
    """Configuration for on-policy distillation.

    enabled (bool):
        Whether on-policy distillation is enabled.
    n_gpus_per_node (int):
        Number of GPUs per node in the teacher resource pool.
    nnodes (int):
        Number of nodes in the teacher resource pool.
    teacher_models (dict[str, TeacherModelConfig]):
        Configurations for teacher models used for multi-teacher distillation.
    teacher_key (str):
        Key to route examples to the appropriate teacher model in multi-teacher setups. Should correspond to a field in
        the data proto, e.g., data_source.
    distillation_loss (DistillationLossConfig):
    Configuration for distillation loss settings.

    NOTE: The `teacher_model` entry is in the `teacher_models` dict by default.
    Since it is popped when other teacher entries are added, using `teacher_model` as
    one of several keys silently drops it. For example, the following CLI overrides result
    in ONLY `teacher_model2` being used:

    ```bash
    distillation.teacher_models.teacher_model.key=openai/gsm8k
    distillation.teacher_models.teacher_model.model_path=Qwen/Qwen3-4B
    +distillation.teacher_models.teacher_model2.key=hiyouga/geometry3k
    +distillation.teacher_models.teacher_model2.model_path=Qwen/Qwen3-VL-4B-Instruct
    ```
    Instead, give the first teacher a different name:

    ```bash
    +distillation.teacher_models.teacher_model1.key=openai/gsm8k
    +distillation.teacher_models.teacher_model1.model_path=Qwen/Qwen3-4B
    +distillation.teacher_models.teacher_model2.key=hiyouga/geometry3k
    +distillation.teacher_models.teacher_model2.model_path=Qwen/Qwen3-VL-4B-Instruct
    ```
    """

    _mutable_fields = BaseConfig._mutable_fields | {"teacher_models", "n_gpus_per_node", "nnodes"}

    enabled: bool = False
    n_gpus_per_node: int = 0
    nnodes: int = 0
    teacher_models: dict[str, DistillationTeacherModelConfig] = field(default_factory=dict)
    teacher_key: str = "data_source"
    distillation_loss: DistillationLossConfig = field(default_factory=DistillationLossConfig)
    omniopd: "OmniOPDConfig" = field(default_factory=lambda: OmniOPDConfig())

    def __post_init__(self):
        if not self.enabled:
            return

        self.teacher_models = self._resolve_teacher_models()
        teacher_world_size_sum = 0
        for teacher_model in self.teacher_models.values():
            teacher_model.validate_and_prepare_for_distillation(
                use_topk=self.distillation_loss.loss_settings.use_topk,
                topk=self.distillation_loss.topk,
                generation_tokens=(
                    self.omniopd.C if self.distillation_loss.loss_settings.use_teacher_generation else None
                ),
            )
            teacher_world_size_sum += teacher_model.world_size
        total_pool_size = self.n_gpus_per_node * self.nnodes
        if teacher_world_size_sum != total_pool_size:
            raise ValueError(
                f"Sum of teacher (num_replicas * per_replica_world_size) ({teacher_world_size_sum}) must match "
                f"the distillation resource pool size "
                f"({self.n_gpus_per_node=} * {self.nnodes=} = {total_pool_size})."
            )

    def _resolve_teacher_models(self) -> dict[str, DistillationTeacherModelConfig]:
        assert "teacher_model" in self.teacher_models
        if len(self.teacher_models) == 1:
            # Single teacher occupies the entire teacher resource pool.
            teacher_model = self.teacher_models["teacher_model"]
            inference = teacher_model.inference
            per_replica = (
                inference.tensor_model_parallel_size
                * inference.data_parallel_size
                * inference.pipeline_model_parallel_size
            )
            pool_size = self.n_gpus_per_node * self.nnodes
            if pool_size % per_replica != 0:
                raise ValueError(
                    f"Single teacher's per_replica_world_size ({per_replica}) must divide the distillation "
                    f"resource pool size ({self.n_gpus_per_node=} * {self.nnodes=} = {pool_size})."
                )
            teacher_model.num_replicas = pool_size // per_replica
            teacher_model.key = "default"
        else:
            # Multiple teachers: remove default single teacher config
            self.teacher_models.pop("teacher_model")

        # Teacher models dict is keyed by teacher_key instead of YAML entry name
        teacher_models = {}
        for teacher_config in self.teacher_models.values():
            teacher_config = omega_conf_to_dataclass(teacher_config, dataclass_type=DistillationTeacherModelConfig)
            teacher_config.check_configured()
            if teacher_config.key in teacher_models:
                raise ValueError(f"Duplicate teacher key {teacher_config.key} found in teacher models.")
            teacher_models[teacher_config.key] = teacher_config
        return teacher_models
