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
    # Same struct-mode trap as detach_utils.get_chunk_token_size: .get(key, default) RAISES on a
    # missing key rather than returning the default, and the synchronous trainer's config has no
    # async_training node.
    try:
        async_training = config.get("async_training", {}) if hasattr(config, "get") else {}
    except Exception:
        async_training = {}
    async_training = async_training or {}
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


def _boundaries_only_enabled() -> bool:
    """State-credit's boundaries-only streaming: chunks are OBSERVED, never published.

    Read from the env rather than imported from the agent-loop worker, to keep this generic loop
    free of a dependency on the fully-async stack (the same reason
    _resolve_streaming_chunk_tokens exists here rather than being imported).
    """
    return os.environ.get("OPD_STATE_CREDIT_EARLY_SYNC", "0").strip().lower() in {"1", "true", "yes", "on"}


def _continuous_stream_enabled() -> bool:
    """OPD_CONTINUOUS_STREAM=1: stream chunks out of ONE engine request (default off).

    The default path below obtains chunk boundaries by ENDING the vLLM request every chunk_tokens
    tokens (see chunk_sampling_params[limit_key] = chunk_limit) and resubmitting prompt +
    everything-so-far. That was never necessary for streaming, and it costs: N engine requests per
    response instead of 1, each resubmitting a prefix growing toward prompt+response_length, plus
    the loss of whatever the engine had decoded when an abort lands.

    Off by default so the published split-path arms stay byte-identical and the two can be A/B'd.
    """
    return os.environ.get("OPD_CONTINUOUS_STREAM", "0").strip().lower() in {"1", "true", "yes", "on"}


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
        if chunk_tokens > 0 and _continuous_stream_enabled():
            output = await self._generate_continuous_stream(
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
        elif chunk_tokens > 0:
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
        # Cap the request at response_length. No caller sets max_tokens (agent_loop builds
        # sampling_params without it), so the server falls back to
        #   min(response_length, prompt_length + response_length - len(prompt_ids))
        # and on a partial-rollout resume prompt_ids already contains everything generated so far
        # -- so that fallback is computed against the MODEL budget rather than the REMAINING one.
        # Measured on the veRL baseline arm: 17.0-17.4% of resumes ran past response_length, peak
        # tokens_so_far 10,148 against a cap of 8,192, all of it discarded below by the
        # [:self.response_length] slice. The chunk-streaming path is immune only because it sets
        # the key explicitly per chunk, so leaving this unfixed handicaps the BASELINE and inflates
        # every OPDFlow-vs-veRL number. Setting it also revives the rollouter's resume budget,
        # which is dead code while limit_key resolves to None.
        sampling_params = dict(sampling_params)
        _limit_key = "max_new_tokens" if "max_new_tokens" in sampling_params else "max_tokens"
        sampling_params[_limit_key] = self.response_length

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

    async def _generate_continuous_stream(
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
        """OPD_CONTINUOUS_STREAM: the same chunk stream, out of ONE engine request.

        The callback contract is identical to _generate_streaming_chunks -- same chunk_agent_output
        shape, same (chunk_idx, token_offset, n_tokens, is_final) -- so the trainer's per-parent
        [0, L) coverage check in _get_hybrid_full_samples_from_chunks is unaffected. Only the
        rollout side changes: no max_tokens injection, so nothing ends the request at a chunk
        boundary; the engine is left to decode the whole response and deltas are sliced off it.

        ONE-DELTA LOOKAHEAD. `is_final` has to be set when the chunk is emitted, but whether a
        delta is the last one is only known once the stream ends. So each delta is buffered and
        emitted when its successor arrives; whatever is left over at the end is emitted final.
        Empty deltas (an abort can yield one) are never buffered, so they cannot become a zero-token
        final chunk and break the coverage check.
        """
        request_id = uuid4().hex
        response_ids: list[int] = []
        response_logprobs: list[float] = []
        # CUMULATIVE, like response_ids -- not the per-delta slice. _delta_token_output slices
        # token_entropies to its own delta, and _emit pairs CUMULATIVE response_ids with the LAST
        # delta's extra_fields, so the final chunk (the one the OmniOPD audit runs on) would carry
        # the whole response against one chunk's worth of entropy. Anchor selection indexes that
        # series against the response, so every anchor would move.
        response_entropies: list[float] = []
        total_num_preempted = 0
        emitted_chunks = 0
        chunk_emit_tasks: list[asyncio.Task] = []
        last_extra_fields: dict[str, Any] = {}
        last_stop_reason = None
        final_emitted = False
        chunk_idx = 0
        stream_start = time.time()
        last_delta_ts = stream_start
        # Tokens received but not yet cut into a chunk. This buffer is what makes chunk boundaries
        # survive a weight-sync abort -- see the comment on the cutting loop below.
        buf_ids: list[int] = []
        buf_lps: list[float] = []
        buf_ents: list[float] = []
        # Version window accumulated since the last chunk cut; consumed and reset by _stage.
        cut_min_gs: int | None = None
        cut_max_gs: int | None = None

        def _emit(entry: tuple, *, is_final: bool) -> None:
            entry_idx, token_offset, n_new_tokens, snapshot_len, extra_fields, gen_s = entry
            chunk_agent_output = AgentLoopOutput(
                prompt_ids=prompt_ids,
                response_ids=response_ids[:snapshot_len][: self.response_length],
                response_mask=([0] * token_offset + [1] * n_new_tokens)[: self.response_length],
                response_logprobs=(
                    response_logprobs[:snapshot_len][: self.response_length] if response_logprobs else None
                ),
                routed_experts=None,
                multi_modal_data=multi_modal_data,
                mm_processor_kwargs=mm_processor_kwargs,
                num_turns=2,
                metrics=AgentLoopMetrics(**{"generate_sequences": gen_s, "num_preempted": -1}),
                reward_score=0.0,
                extra_fields={
                    **extra_fields,
                    "turn_scores": [],
                    "tool_rewards": [],
                    # Sliced exactly like response_ids above, so the two always describe the same
                    # tokens. Omitted entirely when entropy was never requested.
                    **(
                        {"token_entropies": response_entropies[:snapshot_len][: self.response_length]}
                        if response_entropies
                        else {}
                    ),
                },
            )
            chunk_emit_tasks.append(
                asyncio.create_task(
                    chunk_callback(
                        chunk_agent_output,
                        chunk_idx=entry_idx,
                        token_offset=token_offset,
                        n_tokens=n_new_tokens,
                        is_final=is_final,
                    )
                )
            )

        def _stage(tok_ids: list, lps: list, is_final: bool, ents: list | None = None) -> bool:
            """Emit one cut chunk IMMEDIATELY. Returns False once response_length is reached.

            This used to buffer a chunk and release it only when its successor arrived, because
            is_final has to be set at emit time. That one-delta lookahead was catastrophic at large
            chunk sizes: with 2 chunks per response, chunk 0 was held until chunk 1 staged, and
            chunk 1 only stages at the post-stream flush -- so BOTH fired at the end. Measured at
            chunk 4096, a parent emitted its whole response within 3.5 s of a 177 s generation
            (emit-span / gen-span = 0.021, against the split path's 0.502). That is not streaming,
            it is a batch dump, which is exactly veRL-baseline behaviour -- and it is why the arm
            reproduced veRL's throughput to 0.5%.

            No lookahead is needed: the server marks its terminal delta with a non-abort
            stop_reason, and only an abort is followed by a resume, so finality is known when the
            chunk is cut. The response_length cap is also terminal, hence the promotion below.
            """
            nonlocal chunk_idx, last_delta_ts, cut_min_gs, cut_max_gs
            remaining = self.response_length - len(response_ids)
            if remaining <= 0:
                return False
            if len(tok_ids) > remaining:
                tok_ids, lps = tok_ids[:remaining], lps[:remaining]
                if ents is not None:
                    ents = ents[:remaining]
            token_offset = len(response_ids)
            n = len(tok_ids)
            response_ids.extend(tok_ids)
            if len(lps) >= n:
                response_logprobs.extend(lps[:n])
            if ents is not None and len(ents) >= n:
                response_entropies.extend(ents[:n])
            _now = time.time()
            # Weight-version window scoped to THIS CHUNK, matching split -- where each chunk is its
            # own request, so generate() naturally reports the versions that decoded that chunk.
            # Scoping it to the response instead made every continuous chunk carry the
            # response-START version: measured spread of 0 across all 3732 multi-chunk parents,
            # against split's {0: 713, 1: 2967, 2: 35}. That is not cosmetic -- chunk_policy_version
            # feeds ChunkSample.is_stale and the trainer's _drop_stale_chunk gate, so it biases the
            # staleness accounting one-sidedly against continuous.
            chunk_extra = dict(last_extra_fields)
            if cut_min_gs is not None:
                chunk_extra["min_global_steps"] = cut_min_gs
                chunk_extra["max_global_steps"] = cut_max_gs
            cut_min_gs, cut_max_gs = None, None
            entry = (chunk_idx, token_offset, n, len(response_ids), chunk_extra, _now - last_delta_ts)
            last_delta_ts = _now
            chunk_idx += 1
            # Hitting response_length ends the response just as surely as a terminal stop_reason,
            # and the trainer's coverage check requires exactly one is_final per parent.
            final = is_final or len(response_ids) >= self.response_length
            if final:
                nonlocal final_emitted
                final_emitted = True
            _emit(entry, is_final=final)
            return True

        # Cap the request at response_length, exactly as the split path caps each chunk at
        # chunk_limit. No caller sets max_tokens for the rollout, so leaving it unset makes the
        # server fall back to min(response_length, prompt_length + response_length - len(prompt_ids))
        # -- and on a partial-rollout resume prompt_ids already contains everything generated so
        # far, so that fallback is computed against the MODEL budget rather than the REMAINING one.
        # Measured: 26-27% of resumed parents overran response_length by ~970 tokens (peak 10,153
        # against a cap of 8,192), all of it decoded at the longest, most expensive KV lengths and
        # then discarded by the response_length clamp below. The split arm overshoots on 0% of
        # parents because it always sets the key. Setting it here also revives the rollouter's
        # resume budget, which was dead code while limit_key resolved to None.
        stream_sampling_params = dict(sampling_params)
        _limit_key = "max_new_tokens" if "max_new_tokens" in stream_sampling_params else "max_tokens"
        stream_sampling_params[_limit_key] = self.response_length

        async for delta in self.server_manager.generate_stream(
            request_id=request_id,
            prompt_ids=prompt_ids,
            sampling_params=stream_sampling_params,
            chunk_tokens=chunk_tokens,
            image_data=images,
            video_data=videos,
            audio_data=audios,
            mm_processor_kwargs=mm_processor_kwargs,
        ):
            last_extra_fields = dict(delta.extra_fields or {})
            last_stop_reason = delta.stop_reason
            if delta.num_preempted is not None:
                total_num_preempted += int(delta.num_preempted)

            _gs = last_extra_fields.get("global_steps")
            if _gs is not None:
                _gs = int(_gs)
                cut_min_gs = _gs if cut_min_gs is None else min(cut_min_gs, _gs)
                cut_max_gs = _gs if cut_max_gs is None else max(cut_max_gs, _gs)

            # The param-version fields are the ASYNC trainer's staleness bookkeeping, supplied by
            # the fully-async rollouter. Boundaries-only streaming publishes nothing and assembles
            # nothing, so assemble_batch_from_rollout_samples -- the path whose unhelpful NoneType
            # error this assertion exists to pre-empt -- is never reached, and the synchronous
            # trainer has no notion of a param version to supply.
            if chunk_idx == 0 and not _boundaries_only_enabled():
                # Fail loudly, by name, on the first chunk. A version field missing from a delta does
                # not degrade gracefully: it surfaces ~7 minutes later as "unsupported operand
                # type(s) for -: 'NoneType' and 'NoneType'" out of
                # detach_utils.assemble_batch_from_rollout_samples, which names neither the field nor
                # this code path. (That is exactly how job 43924137 died.)
                _missing = [
                    k for k in ("global_steps", "min_global_steps", "max_global_steps")
                    if last_extra_fields.get(k) is None
                ]
                assert not _missing, (
                    f"continuous-stream delta missing required extra_fields {_missing}; the trainer's "
                    f"param-version accounting needs them on every chunk (present: {sorted(last_extra_fields)})"
                )

            # Evaluate finality BEFORE the empty-delta shortcut. A terminal delta legitimately
            # carries zero tokens when the response ends on an exact chunk boundary -- and the
            # median response here is exactly response_length -- so skipping it would lose the only
            # signal that the stream is over.
            stream_done = delta.stop_reason not in (None, "aborted", "abort")
            new_token_ids = list(delta.token_ids or [])
            if not new_token_ids and not stream_done:
                continue
            if new_token_ids:
                buf_ids.extend(new_token_ids)
                if delta.log_probs is not None:
                    buf_lps.extend(list(delta.log_probs[: len(new_token_ids)]))
                _dent = last_extra_fields.get("token_entropies")
                if _dent is not None:
                    if len(_dent) < len(new_token_ids):
                        raise RuntimeError(
                            f"continuous-stream delta carried {len(new_token_ids)} tokens but "
                            f"{len(_dent)} entropy values; anchor selection would be shifted."
                        )
                    buf_ents.extend(list(_dent[: len(new_token_ids)]))

            # Cut chunks at EXACT chunk_tokens boundaries, ACROSS aborts.
            #
            # The server sees one engine call at a time, so on an abort its terminal delta is a
            # partial. Emitting that partial as a chunk in its own right is what fragmented the
            # stream: at chunk 4096 the mean emitted chunk came out 2437 tokens with 44% of them
            # under half size, and the arm issued 64% more teacher calls than the split arm for
            # identical token totals. The split path never had this problem because its abort is
            # absorbed inside the chunk request's own resume loop, so a boundary only ever lands
            # at max_tokens. Buffering here spans the resume, so a weight sync can no longer move
            # a chunk boundary.
            # Hold back the tail so the LAST emission carries is_final -- and hold back ENOUGH.
            #
            # This was `chunk_tokens if stream_done else chunk_tokens - 1`, which cuts as soon as a
            # full chunk exists (len > 1023 cuts 1024, leaving 0). A response whose length is an
            # exact multiple of chunk_tokens therefore drains the buffer on its last cut, and when
            # the terminal delta arrives `if stream_done and buf_ids` finds nothing to stage, so NO
            # chunk carries is_final. Downstream that is fatal, not cosmetic: the assembler never
            # finalizes the parent, the substitutive gate falls back to the additive whole-response
            # rescan, that rescan re-enters the per-parent FIFO with the same session_id and
            # span_start=0 -- which the FIFO still holds from chunk 0, because parent state is only
            # cleaned up on is_final -- and the resulting FifoDuplicateError kills the rollouter.
            # It killed every k1.0cont job in three consecutive campaigns after 8-16 minutes.
            #
            # `keep = chunk_tokens` cuts only when MORE than a full chunk is buffered, so at least
            # one token always survives to be staged as final. It costs one token of emit latency
            # per chunk and makes the boundary case impossible by construction rather than caught.
            #
            # BOUNDARIES-ONLY keeps nothing back. The hold-back exists to guarantee a final chunk
            # for the ASSEMBLER; the failure it prevents runs assembler -> additive rescan -> FIFO
            # duplicate, and every step of that chain requires the chunk to be PUBLISHED. This mode
            # publishes nothing and skips is_final entirely, so none of it is reachable.
            #
            # It costs a full chunk of latency on the only thing this mode exists for: with
            # keep=chunk_tokens a cut needs MORE than a chunk buffered, so depth d is first observed
            # at d + chunk_tokens generated tokens -- 3072 for d=2048, c=1024. Any response shorter
            # than that could never launch early at all, and every longer one launched a full chunk
            # late.
            keep = 0 if _boundaries_only_enabled() else chunk_tokens
            while len(buf_ids) > keep:
                if not _stage(buf_ids[:chunk_tokens], buf_lps[:chunk_tokens], False,
                              ents=buf_ents[:chunk_tokens] if buf_ents else None):
                    break  # response_length reached; drop the rest
                del buf_ids[:chunk_tokens]
                del buf_lps[: min(chunk_tokens, len(buf_lps))]
                del buf_ents[: min(chunk_tokens, len(buf_ents))]
            if stream_done and buf_ids:
                # ents MUST be passed here: this is the terminal cut, and the final chunk is exactly
                # the one _compute_omniopd_audit runs on.
                _stage(buf_ids, buf_lps, True, ents=buf_ents if buf_ents else None)
                buf_ids.clear()
                buf_lps.clear()
                buf_ents.clear()

        # Defensive tail flush: the generator can in principle end without ever yielding a terminal
        # stop_reason. Losing this would leave the parent with no is_final and stall its assembly.
        if buf_ids:
            _stage(buf_ids, buf_lps, True, ents=buf_ents if buf_ents else None)
        if chunk_idx > 0 and not final_emitted:
            # A response ending exactly on a chunk boundary leaves the buffer empty when the
            # terminal delta arrives, so no cut chunk can carry is_final. Log it rather than let
            # the parent stall in the assembler waiting for a final chunk that never comes. If this
            # ever fires at scale the emit granularity needs decoupling from the chunk size.
            logger.error(
                "continuous-stream parent produced %d chunks with no is_final (response_ids=%d, "
                "chunk_tokens=%d); the assembler will not finalize it",
                chunk_idx, len(response_ids), chunk_tokens,
            )

        if chunk_emit_tasks:
            emit_results = await asyncio.gather(*chunk_emit_tasks, return_exceptions=True)
            for result in emit_results:
                if isinstance(result, BaseException):
                    logger.error(
                        "Continuous-stream chunk callback failed",
                        exc_info=(type(result), result, result.__traceback__),
                    )
                elif result:
                    emitted_chunks += 1

        metrics = {"generate_sequences": time.time() - stream_start, "num_preempted": total_num_preempted}
        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=([1] * len(response_ids))[: self.response_length],
            response_logprobs=response_logprobs[: self.response_length] if response_logprobs else None,
            routed_experts=None,
            multi_modal_data=multi_modal_data,
            mm_processor_kwargs=mm_processor_kwargs,
            num_turns=2,
            metrics=AgentLoopMetrics(**metrics),
            extra_fields={
                **last_extra_fields,
                # cumulative, sliced exactly like response_ids above
                **({"token_entropies": response_entropies[: self.response_length]}
                   if response_entropies else {}),
                "stop_reason": last_stop_reason,
                "streaming_chunks_emitted": emitted_chunks,
                "continuous_stream": True,
            },
        )

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
        # CUMULATIVE, like response_ids. Each chunk here is a separate request, so its extra_fields
        # carry only that chunk's entropies, while every emitted output pairs them with the
        # cumulative response -- and the final chunk is the one the OmniOPD audit runs on.
        response_entropies: list[float] = []
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
            _cent = (chunk_output.extra_fields or {}).get("token_entropies")
            if _cent is not None:
                if len(_cent) < n_new_tokens:
                    raise RuntimeError(
                        f"split-stream chunk carried {n_new_tokens} tokens but {len(_cent)} entropy "
                        f"values; anchor selection would be shifted against the response."
                    )
                response_entropies.extend(list(_cent[:n_new_tokens]))
            routed_experts = _append_routed_experts(routed_experts, chunk_output.routed_experts, n_new_tokens)
            if chunk_output.num_preempted is not None:
                total_num_preempted += int(chunk_output.num_preempted)
            last_extra_fields = dict(chunk_output.extra_fields or {})
            if response_entropies:
                last_extra_fields["token_entropies"] = response_entropies[: self.response_length]
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
                # cumulative, sliced exactly like response_ids above
                **({"token_entropies": response_entropies[: self.response_length]}
                   if response_entropies else {}),
                "stop_reason": last_stop_reason,
                "streaming_chunks_emitted": emitted_chunks,
            },
        )

        return output
