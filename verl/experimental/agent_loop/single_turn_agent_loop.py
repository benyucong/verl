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
import asyncio
import logging
import os
import time
from typing import Any
from uuid import uuid4

import numpy as np
import torch

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopMetrics, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _resolve_streaming_chunk_tokens(config) -> int:
    """Resolve chunk size without importing the fully-async stack from the generic agent loop."""
    async_training = config.get("async_training", {}) if hasattr(config, "get") else {}
    chunk_tokens = 0
    try:
        chunk_tokens = int(async_training.get("chunk_tokens", 0) or 0)
    except (TypeError, ValueError):
        chunk_tokens = 0

    real_chunks_env = os.environ.get("OPD_STAGE1_REAL_CHUNKS", "0").strip().lower() in {"1", "true", "yes", "on"}
    if chunk_tokens <= 0 and real_chunks_env:
        try:
            chunk_tokens = int(os.environ.get("OPD_STAGE1_CHUNK_TOKENS", "0"))
        except ValueError:
            chunk_tokens = 0
    return max(0, chunk_tokens)


def _rollout_lease_max_tokens() -> int:
    """Rollout-lease refresh knob (prototype for the Q3 'pinned suffix' problem).

    Max response tokens a streaming request may decode under one sticky engine session
    before its request_id is rotated. 0 disables (default — no behavior change).

    When a request crosses the lease, the next slice gets a fresh request_id, which
    forces it to re-establish a session (re-prefill prompt + response-so-far) instead
    of riding a stale engine snapshot to completion. Each slice is still stamped with
    the engine version that actually decoded it, so chunk provenance/labels are
    unchanged; the lease only affects WHICH serving path decodes the suffix.
    """
    try:
        return max(0, int(os.environ.get("OPD_ROLLOUT_LEASE_MAX_TOKENS", "0")))
    except ValueError:
        return 0


def _rollout_lease_force_reprefill() -> bool:
    """Experimental forced-re-prefill refresh (default off).

    When OPD_ROLLOUT_LEASE_FORCE_REPREFILL=1 AND a lease is set, lease expiry does NOT
    merely rotate request_id (validated ineffective: vLLM reuses old-weight prefix KV by
    token content). Instead it invalidates the serving replica's prefix cache
    (reset_prefix_cache) so the next slice RE-PREFILLS the accumulated prompt+response
    under the engine's currently-loaded weights, then continues suffix decoding. Chunks
    are still stamped with the actual decode-time version (no relabeling); only WHICH
    weights decode the suffix changes. NOTE: reset_prefix_cache is engine-WIDE on the
    replica, so this also forces concurrent requests on that replica to re-prefill -- the
    cost this flag exists to measure.
    """
    return os.environ.get("OPD_ROLLOUT_LEASE_FORCE_REPREFILL", "0").strip().lower() in {"1", "true", "yes", "on"}


def _finish_reason(output: TokenOutput) -> str | None:
    finish_reason = output.extra_fields.get("finish_reason") if output.extra_fields else None
    if finish_reason is not None:
        return finish_reason
    return output.stop_reason


def _should_continue_chunked_generation(
    output: TokenOutput,
    *,
    chunk_limit: int,
    total_response_tokens: int,
    response_length: int,
) -> bool:
    """Return True when a chunk-sized request should be resumed for the next chunk."""
    if total_response_tokens >= response_length:
        return False

    finish_reason = _finish_reason(output)
    if finish_reason == "length":
        return True
    if finish_reason is None or finish_reason == "completed":
        # Older rollout backends may not preserve finish_reason. In that case,
        # continue only when the request saturated the chunk cap.
        return len(output.token_ids or []) >= chunk_limit
    return False


def _append_routed_experts(existing, routed_experts, n_new_tokens: int):
    if routed_experts is None or n_new_tokens <= 0:
        return existing
    if existing is None:
        return routed_experts

    routed_slice = routed_experts[-n_new_tokens:]
    if isinstance(existing, torch.Tensor) or isinstance(routed_slice, torch.Tensor):
        existing_tensor = existing if isinstance(existing, torch.Tensor) else torch.as_tensor(existing)
        routed_tensor = routed_slice if isinstance(routed_slice, torch.Tensor) else torch.as_tensor(routed_slice)
        return torch.cat([existing_tensor, routed_tensor], dim=0)
    return np.concatenate([existing, routed_slice], axis=0)


@register("single_turn_agent")
class SingleTurnAgentLoop(AgentLoopBase):
    """Naive agent loop that only do single turn chat completion."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        chunk_callback = kwargs.pop("_chunk_callback", None)
        messages = list(kwargs["raw_prompt"])

        # 1. extract multimodal inputs from messages
        multi_modal_data = await self.process_multi_modal_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")
        audios = multi_modal_data.get("audios")
        mm_processor_kwargs = self._get_mm_processor_kwargs(audios)

        # 2. apply chat template and tokenize
        prompt_ids = await self.apply_chat_template(
            messages,
            images=images,
            videos=videos,
            audios=audios,
            mm_processor_kwargs=mm_processor_kwargs,
        )

        # 3. generate sequences
        chunk_tokens = _resolve_streaming_chunk_tokens(self.config) if chunk_callback is not None else 0
        if chunk_tokens > 0:
            output = await self._generate_streaming_chunks(
                sampling_params=sampling_params,
                prompt_ids=prompt_ids,
                multi_modal_data=multi_modal_data,
                images=images,
                videos=videos,
                audios=audios,
                mm_processor_kwargs=mm_processor_kwargs,
                chunk_tokens=chunk_tokens,
                chunk_callback=chunk_callback,
            )
        else:
            output = await self._generate_full_response(
                sampling_params=sampling_params,
                prompt_ids=prompt_ids,
                multi_modal_data=multi_modal_data,
                images=images,
                videos=videos,
                audios=audios,
                mm_processor_kwargs=mm_processor_kwargs,
            )

        # keeping the schema consistent with tool_agent_loop
        output.extra_fields.update({"turn_scores": [], "tool_rewards": []})

        return output

    async def _generate_full_response(
        self,
        *,
        sampling_params: dict[str, Any],
        prompt_ids: list[int],
        multi_modal_data: dict[str, Any],
        images,
        videos,
        audios,
        mm_processor_kwargs: dict[str, Any],
    ) -> AgentLoopOutput:
        metrics = {}
        with simple_timer("generate_sequences", metrics):
            output: TokenOutput = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=images,
                video_data=videos,
                audio_data=audios,
                mm_processor_kwargs=mm_processor_kwargs,
            )
        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        response_mask = [1] * len(output.token_ids)

        output: AgentLoopOutput = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=output.token_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=output.log_probs[: self.response_length] if output.log_probs else None,
            routed_experts=(
                output.routed_experts[: len(prompt_ids) + self.response_length]
                if output.routed_experts is not None
                else None
            ),
            multi_modal_data=multi_modal_data,
            mm_processor_kwargs=mm_processor_kwargs,
            num_turns=2,
            metrics=metrics,
            extra_fields=output.extra_fields,
        )

        return output

    async def _generate_streaming_chunks(
        self,
        *,
        sampling_params: dict[str, Any],
        prompt_ids: list[int],
        multi_modal_data: dict[str, Any],
        images,
        videos,
        audios,
        mm_processor_kwargs: dict[str, Any],
        chunk_tokens: int,
        chunk_callback,
    ) -> AgentLoopOutput:
        request_id = uuid4().hex
        response_ids: list[int] = []
        response_logprobs: list[float] = []
        routed_experts = None
        total_generate_time = 0.0
        total_num_preempted = 0
        last_extra_fields: dict[str, Any] = {}
        last_stop_reason = None
        emitted_chunks = 0
        chunk_idx = 0
        chunk_emit_tasks = []

        # Rollout-lease refresh state (no-op when lease_tokens == 0).
        lease_tokens = _rollout_lease_max_tokens()
        force_reprefill = _rollout_lease_force_reprefill()
        lease_anchor = 0
        lease_refreshes = 0

        while len(response_ids) < self.response_length:
            token_offset = len(response_ids)
            chunk_limit = min(chunk_tokens, self.response_length - token_offset)
            if chunk_limit <= 0:
                break

            # Rollout-lease refresh. No-op unless lease_tokens > 0. Two modes:
            #   - default (cheap): rotate request_id only. Validated INEFFECTIVE (vLLM
            #     reuses old-weight prefix KV by token content); kept for A/B.
            #   - force_reprefill: invalidate the serving replica's prefix cache so the
            #     next slice RE-PREFILLS the accumulated prefix under current weights.
            #     reset_prefix_cache is engine-WIDE on that replica, so concurrent
            #     requests also re-prefill (the cost being measured). Provenance is
            #     preserved: each slice is stamped by whatever version decodes it.
            pending_refresh = None
            if lease_tokens and token_offset - lease_anchor >= lease_tokens:
                _prev_version = last_extra_fields.get("global_steps")
                _prev_replica = last_extra_fields.get("replica_rank")
                _refresh_start = time.time()
                _kv_invalidated = False
                _reset_latency = 0.0
                if force_reprefill:
                    try:
                        server_id, server = await self.server_manager._acquire_server(request_id)
                        try:
                            _t0 = time.time()
                            await server.clear_kv_cache.remote()
                            _reset_latency = time.time() - _t0
                            _kv_invalidated = True
                        finally:
                            self.server_manager._release_server(server_id)
                    except Exception as e:  # noqa: BLE001
                        logger.warning(f"[rollout-lease] forced re-prefill cache reset failed: {e}")
                old_request_id = request_id
                request_id = uuid4().hex
                lease_anchor = token_offset
                lease_refreshes += 1
                pending_refresh = {
                    "old_request_id": old_request_id,
                    "new_request_id": request_id,
                    "token_offset": int(token_offset),
                    "chunk_idx": int(chunk_idx),
                    "lease_tokens": int(lease_tokens),
                    "refresh_seq": int(lease_refreshes),
                    "old_chunk_version": int(_prev_version) if _prev_version is not None else None,
                    "prev_replica_rank": int(_prev_replica) if _prev_replica is not None else None,
                    "refresh_start_ts": _refresh_start,
                    "kv_cache_invalidated": _kv_invalidated,
                    "reset_prefix_cache_latency_s": float(_reset_latency),
                    "prefix_tokens_replayed": int(token_offset),
                }

            chunk_sampling_params = dict(sampling_params)
            limit_key = "max_new_tokens" if "max_new_tokens" in chunk_sampling_params else "max_tokens"
            chunk_sampling_params[limit_key] = chunk_limit

            chunk_metrics = {}
            with simple_timer("generate_sequences", chunk_metrics):
                chunk_output: TokenOutput = await self.server_manager.generate(
                    request_id=request_id,
                    prompt_ids=prompt_ids + response_ids,
                    sampling_params=chunk_sampling_params,
                    image_data=images,
                    video_data=videos,
                    audio_data=audios,
                    mm_processor_kwargs=mm_processor_kwargs,
                )

            total_generate_time += float(chunk_metrics.get("generate_sequences", 0.0))

            # Forced-re-prefill: the slice we just ran re-prefilled the accumulated prefix
            # under current weights. Trace cost + whether the suffix adopted a fresher
            # version (first_post_refresh_chunk_version > old_chunk_version == it worked).
            if pending_refresh is not None:
                try:
                    from verl.experimental.fully_async_policy.opd_stage0_trace import trace_event as _opd_trace

                    _post_ver = chunk_output.extra_fields.get("global_steps") if chunk_output.extra_fields else None
                    _post_rep = chunk_output.extra_fields.get("replica_rank") if chunk_output.extra_fields else None
                    _now = time.time()
                    _opd_trace(
                        "rollout_lease_force_reprefill" if pending_refresh["kv_cache_invalidated"] else "rollout_lease_refresh",
                        pending_refresh["new_request_id"],
                        role="rollouter",
                        prefix_replay_start_ts=pending_refresh["refresh_start_ts"],
                        prefix_replay_end_ts=_now,
                        prefix_replay_latency_s=float(chunk_metrics.get("generate_sequences", 0.0)),
                        suffix_resume_ts=_now,
                        first_post_refresh_chunk_version=int(_post_ver) if _post_ver is not None else None,
                        first_post_refresh_replica_rank=int(_post_rep) if _post_rep is not None else None,
                        **pending_refresh,
                    )
                except Exception:
                    pass

            new_token_ids = list(chunk_output.token_ids or [])
            if not new_token_ids:
                last_extra_fields = dict(chunk_output.extra_fields or {})
                last_stop_reason = chunk_output.stop_reason
                break

            remaining_slots = self.response_length - len(response_ids)
            if len(new_token_ids) > remaining_slots:
                new_token_ids = new_token_ids[:remaining_slots]
            n_new_tokens = len(new_token_ids)

            response_ids.extend(new_token_ids)
            if chunk_output.log_probs is not None:
                response_logprobs.extend(list(chunk_output.log_probs[:n_new_tokens]))
            routed_experts = _append_routed_experts(routed_experts, chunk_output.routed_experts, n_new_tokens)
            if chunk_output.num_preempted is not None:
                total_num_preempted += int(chunk_output.num_preempted)
            last_extra_fields = dict(chunk_output.extra_fields or {})
            last_stop_reason = chunk_output.stop_reason

            continue_generating = _should_continue_chunked_generation(
                chunk_output,
                chunk_limit=chunk_limit,
                total_response_tokens=len(response_ids),
                response_length=self.response_length,
            )
            chunk_metrics["num_preempted"] = (
                chunk_output.num_preempted if chunk_output.num_preempted is not None else -1
            )
            chunk_agent_output = AgentLoopOutput(
                prompt_ids=prompt_ids,
                response_ids=response_ids[: self.response_length],
                response_mask=([0] * token_offset + [1] * n_new_tokens)[: self.response_length],
                response_logprobs=response_logprobs[: self.response_length] if response_logprobs else None,
                routed_experts=(
                    routed_experts[: len(prompt_ids) + self.response_length] if routed_experts is not None else None
                ),
                multi_modal_data=multi_modal_data,
                mm_processor_kwargs=mm_processor_kwargs,
                num_turns=2,
                metrics=AgentLoopMetrics(**chunk_metrics),
                reward_score=0.0,
                extra_fields={**last_extra_fields, "turn_scores": [], "tool_rewards": []},
            )
            chunk_emit_tasks.append(
                asyncio.create_task(
                    chunk_callback(
                        chunk_agent_output,
                        chunk_idx=chunk_idx,
                        token_offset=token_offset,
                        n_tokens=n_new_tokens,
                        is_final=not continue_generating,
                    )
                )
            )
            chunk_idx += 1

            if not continue_generating:
                break

        if chunk_emit_tasks:
            emit_results = await asyncio.gather(*chunk_emit_tasks, return_exceptions=True)
            for result in emit_results:
                if isinstance(result, BaseException):
                    logger.error(
                        "Streaming chunk callback failed",
                        exc_info=(type(result), result, result.__traceback__),
                    )
                elif result:
                    emitted_chunks += 1

        metrics = {"generate_sequences": total_generate_time, "num_preempted": total_num_preempted}
        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=([1] * len(response_ids))[: self.response_length],
            response_logprobs=response_logprobs[: self.response_length] if response_logprobs else None,
            routed_experts=(
                routed_experts[: len(prompt_ids) + self.response_length] if routed_experts is not None else None
            ),
            multi_modal_data=multi_modal_data,
            mm_processor_kwargs=mm_processor_kwargs,
            num_turns=2,
            metrics=AgentLoopMetrics(**metrics),
            extra_fields={
                **last_extra_fields,
                "stop_reason": last_stop_reason,
                "streaming_chunks_emitted": emitted_chunks,
            },
        )

        return output
