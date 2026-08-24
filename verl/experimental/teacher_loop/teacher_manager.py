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
import math
import os
import time
from typing import Any, Optional
from uuid import uuid4

import torch
from omegaconf import DictConfig
from torch.nn import functional as F

from verl.experimental.teacher_loop.fifo import PerParentFifo, fifo_enabled
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import (
    DistillationConfig,
    DistillationLossConfig,
    DistillationTeacherModelConfig,
)
from verl.workers.rollout.llm_server import LLMServerClient


def _sampling_params_supports_window() -> bool:
    """Does the running vLLM have the prompt_logprobs_range patch?

    Cached, because this is consulted per teacher call. If the engine is stock, passing the key would
    raise TypeError inside SamplingParams(**sampling_params) and take down every teacher request --
    so an unpatched engine must degrade to the old full-materialisation behaviour, not crash.
    """
    if _WINDOW_SUPPORTED[0] is None:
        from vllm import SamplingParams

        _WINDOW_SUPPORTED[0] = "prompt_logprobs_range" in getattr(SamplingParams, "__struct_fields__", ())
        if not _WINDOW_SUPPORTED[0]:
            # FAIL, do not warn. This is only ever reached when the window was EXPLICITLY requested,
            # and the sweep encodes that request in the run label (_win1). Degrading to unwindowed
            # would produce a run labelled as the treatment arm that silently received the control --
            # an arm-asymmetric flag that survives into the analysis unnoticed. That exact failure
            # mode (detokenize=False on the streaming arm only) manufactured a +67% result here that
            # collapsed to +0.3% once corrected. A warning in a Ray worker's stdout is not a control:
            # this project has already lost measurements to log lines nobody read.
            raise RuntimeError(
                "OPD_TEACHER_LOGPROBS_WINDOW=1 but this vLLM has no SamplingParams.prompt_logprobs_range. "
                "Apply patches/vllm-0.15.1-prompt_logprobs_range.patch to the vLLM in use (or put a "
                "patched source checkout on PYTHONPATH). Refusing to run: the job would be labelled as "
                "the windowed arm while executing the unwindowed one."
            )
    return _WINDOW_SUPPORTED[0]


_WINDOW_SUPPORTED: list[Optional[bool]] = [None]


def _get_teacher_sampling_params(
    teacher_model_config: DistillationTeacherModelConfig,
    distillation_loss_config: DistillationLossConfig,
    incremental: bool = False,
    logprobs_window: Optional[tuple[int, int]] = None,
) -> dict[str, Any]:
    """Get sampling parameters for teacher model when computing log probabilities for distillation."""
    if teacher_model_config.inference.temperature != 1.0:
        raise NotImplementedError("vLLM does not support temperature for prompt_logprobs.")

    num_logprobs = distillation_loss_config.topk if distillation_loss_config.loss_settings.use_topk else 0
    params = {
        "max_tokens": 1,
        "temperature": teacher_model_config.inference.temperature,
        "prompt_logprobs": num_logprobs,
        # BOTH ARMS, and it must stay that way. We consume only .logprob/.rank/the token-id keys
        # (workers/rollout/vllm_rollout/utils.py extract_prompt_logprobs) -- the decoded strings are
        # never read. But detokenize defaults to True, and in vLLM v1 that flag also decides whether
        # the LOGPROBS processor gets a tokenizer (v1/engine/output_processor.py: "if not
        # sampling_params.detokenize: tokenizer = None"), which in turn drives
        # convert_ids_list_to_tokens over num_prompt_tokens * K ids plus a per-position UTF-8
        # correction pass. At K=64 over an 8192-token response that is ~533k string conversions per
        # request -- pure Python, on the engine's output thread.
        #
        # This used to be set only on the incremental path, which quietly handicapped the baseline:
        # at Gt=2 the SAME two teacher GPUs delivered 3224 uncached tok/s/GPU for the one-big-request
        # baseline versus 7389 for chunk streaming, and baseline teacher latency was 156 s against
        # 1.13 s. That gap read as an OPDFlow win (+67%) when it was really an arm-specific flag.
        "detokenize": False,
    }
    if incremental:
        # Incremental-only: read the cached prefix KV (the default skips it for prompt_logprobs), so
        # the server returns just the recomputed suffix and we slice the exact new span below. This
        # is also why the arm CANNOT detokenize -- the cached-prefix rows carry out-of-range ids that
        # crash the detokenizer -- but that is a correctness requirement here, not a speed knob.
        params["skip_reading_prefix_cache"] = False
        # INCREMENTAL ONLY. Build Logprob objects for just the rows this chunk will actually slice.
        # Stock vLLM builds K per position over the WHOLE submitted prefix regardless of cache hits --
        # 8192 x K=64 = 524,288 objects per request -- which is what makes the teacher's cost track
        # SUBMITTED context (measured 105 us/submitted token, of which ~92% is this construction) and
        # is therefore what makes chunk streaming pay an (N+1)/2 tax for context it re-sends but never
        # uses. Windowed, that marginal cost drops to ~8 us/token.
        #
        # This must NEVER be set on the clean/fallback arm below: that path is parsed by the STRICT
        # extract_prompt_logprobs, which iterates prompt_logprobs[1:] and asserts a fixed row width, so
        # the None rows outside the window would make it raise. The fallback needs every row anyway --
        # it exists precisely for when the incremental span is not recoverable.
        if logprobs_window is not None and _sampling_params_supports_window():
            params["prompt_logprobs_range"] = logprobs_window
    return params


def _finalize_span_tensors(new_ids, new_lps, n: int, K: int):
    """Validate exactly-n top-k rows (no silent padding/repair) and build [n, K] int32/float32 tensors."""
    if len(new_ids) != n or len(new_lps) != n:
        raise AssertionError(f"span retained {len(new_ids)} rows != requested span length {n}")
    if n == 0:  # empty span (chunk added no new tokens): well-formed [0, K] tensors
        return torch.zeros(0, K, dtype=torch.int32), torch.zeros(0, K, dtype=torch.float32)
    for r, (ids_row, lp_row) in enumerate(zip(new_ids, new_lps)):
        if len(ids_row) != K or len(lp_row) != K:
            raise AssertionError(f"span row {r}: top-k width {len(ids_row)}/{len(lp_row)} != {K}")
        if any((tid is None or int(tid) < 0) for tid in ids_row):
            raise AssertionError(f"span row {r}: invalid token id")
        if any((lp is None or not math.isfinite(lp)) for lp in lp_row):
            raise AssertionError(f"span row {r}: non-finite logprob")
    return torch.tensor(new_ids, dtype=torch.int32), torch.tensor(new_lps, dtype=torch.float32)


def _pad_teacher_outputs(
    teacher_ids: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    prompt_width: int,
    response_width: int,
    prompt_length: int,
    response_length: int,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # TODO(wuxibin): remove padding and use tensordict.
    left_pad_size = prompt_width - prompt_length
    right_pad_size = response_width - response_length
    padding = (0, 0, left_pad_size, right_pad_size)
    return (
        F.pad(teacher_ids, padding, value=pad_token_id).unsqueeze(0),
        F.pad(teacher_logprobs, padding, value=0.0).unsqueeze(0),
    )


class AsyncTeacherLLMServerManager:
    """Teacher-specific async client used for distillation logprob computation."""

    def __init__(
        self,
        config: DictConfig,
        teacher_client: dict[str, LLMServerClient],
    ):
        self.distillation_config: DistillationConfig = omega_conf_to_dataclass(config.distillation)
        self.distillation_loss_config: DistillationLossConfig = self.distillation_config.distillation_loss
        self.teacher_key: str = self.distillation_config.teacher_key

        self.teacher_model_configs: dict[str, DistillationTeacherModelConfig] = self.distillation_config.teacher_models
        expected = set(self.teacher_model_configs)
        if set(teacher_client.keys()) != expected:
            raise ValueError(
                f"teacher client keys {sorted(teacher_client.keys())} "
                f"do not match teacher routing keys {sorted(expected)}."
            )
        self.teacher_client: dict[str, LLMServerClient] = teacher_client
        # Stage 2: per-parent FIFO sequencer for incremental scoring (gated OPD_TEACHER_PER_PARENT_FIFO).
        # One sequencer per worker (== one asyncio event loop), shared across this worker's chunk tasks.
        _to = float(os.environ.get("OPD_TEACHER_FIFO_TIMEOUT_S", "120") or 120)
        self._fifo = PerParentFifo(timeout_s=_to)
        self._fifo_warned = False

    def release_parent(self, session_id) -> None:
        """Release per-parent FIFO ordering state for a session that will issue no further calls.

        Only meaningful when OPD_TEACHER_PER_PARENT_FIFO is on; a no-op otherwise, so callers do not
        have to know which mode they are in.
        """
        f = getattr(self, "_fifo", None)
        if f is None or session_id is None:
            return
        try:
            f.release(str(session_id))
        except Exception:          # releasing a parent must never fail a trajectory
            logging.getLogger(__name__).debug("release_parent(%s) failed", session_id, exc_info=True)

    def _resolve_teacher_key(self, routing_key: Optional[str]) -> str:
        if len(self.teacher_model_configs) == 1:
            # Single-teacher path: route everything to the one teacher regardless of the sample's key.
            return next(iter(self.teacher_model_configs))
        if routing_key is None:
            raise ValueError(
                f"Routing key is required for multi-teacher distillation "
                f"(configured via distillation.teacher_key={self.teacher_key!r})."
            )
        if routing_key not in self.teacher_model_configs:
            raise ValueError(
                f"No teacher configured for routing key {routing_key!r}. "
                f"Configured teachers: {sorted(self.teacher_model_configs)}."
            )
        return routing_key

    async def generate_chunk_continuations(
        self,
        prefix_ids: list[int],
        n: int,
        max_tokens: int,
        routing_key: Optional[str] = None,
        session_id: Optional[str] = None,
        seed: Optional[int] = None,
        is_final: bool = False,
    ) -> tuple[list[list[int]], dict]:
        """N teacher continuations of ONE audited chunk's prefix (generative teaching).

        The counterpart to compute_teacher_logprobs_single, and its opposite: that method hands the
        teacher a COMPLETE sequence and asks how likely the student's tokens were; this hands it a
        prefix that stops dead before the audited chunk and asks what IT would have written. The
        teacher must never see the tokens it is being asked to independently produce -- that is
        checked server-side, where the submitted prompt length is compared against what the engine
        actually ingested.

        Returns (sequences, telemetry). Sequences are raw token ids; decoding happens where phi is
        computed, not here, so the hot path never pays for text it may not use.

        STICKY ROUTING IS NOT OPTIONAL HERE. A trajectory's M chunk prefixes are NESTED -- each
        extends the previous -- so pinning them to one replica lets chunk k+1's prefill reuse chunk
        k's KV. Scattered across replicas, every chunk re-ingests a prefix that grows with the
        response, which is the dominant cost of generative teaching.
        """
        teacher_key = self._resolve_teacher_key(routing_key)
        client = self.teacher_client[teacher_key]
        use_stable_routing = session_id is not None
        routing_request_id = f"teacher::{session_id}" if use_stable_routing else uuid4().hex

        t0 = time.perf_counter()
        out = await client.generate_n(
            request_id=routing_request_id,
            prompt_ids=prefix_ids,
            n=n,
            max_tokens=max_tokens,
            seed=seed,
            is_final=is_final,
            track_parent=bool(use_stable_routing),
        )
        dt = time.perf_counter() - t0

        # An empty result means the request was aborted mid-flight. Returning it silently would
        # leave the chunk with k_sem computed from zero continuations -- indistinguishable from a
        # teacher that disagreed with the student everywhere, which is a real value the objective
        # would happily train on.
        if not out.sequences:
            raise RuntimeError(
                f"teacher returned no continuations for a chunk (prefix {len(prefix_ids)} tokens); "
                f"k_sem from zero rollouts is indistinguishable from total teacher disagreement."
            )
        telemetry = {
            "teacher_gen_seconds": dt,
            "teacher_gen_tokens": sum(len(x) for x in out.sequences),
            "teacher_prefix_tokens": len(prefix_ids),
            "teacher_cached_tokens": out.num_cached_tokens,
            "teacher_n": len(out.sequences),
            # Consumed by state-credit's truncation gate. Dropping it did not make the gate fail --
            # it made the gate report 0.000 truncation unconditionally, which reads as an
            # affirmative all-clear on the one check that distinguishes "Phi is meaningless because
            # B is too small" from "the method does not work".
            "finish_reasons": list(getattr(out, "finish_reasons", None) or []),
        }
        return out.sequences, telemetry

    async def compute_teacher_logprobs_single(
        self,
        sequence_ids: list[int],
        multi_modal_data: Optional[dict[str, Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        routing_key: Optional[str] = None,
        session_id: Optional[str] = None,
        span_start: Optional[int] = None,
        span_end: Optional[int] = None,
        prompt_width: Optional[int] = None,
        is_final: Optional[bool] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Compute teacher log probabilities for a single unpadded sequence.

        Returns (teacher_ids, teacher_logprobs, telemetry). In incremental mode (Stage 1, gated
        OPD_TEACHER_INCREMENTAL_SCORE) the result is ONLY the new span's [n, k] top-k labels, sliced from
        the server's valid recomputed suffix; otherwise it is the full-sequence [S, k] labels.
        """
        multi_modal_data = multi_modal_data or {}
        teacher_key = self._resolve_teacher_key(routing_key)
        teacher_model_config = self.teacher_model_configs[teacher_key]
        client = self.teacher_client[teacher_key]
        incremental = (
            os.environ.get("OPD_TEACHER_INCREMENTAL_SCORE", "0") not in ("0", "", "false", "False")
            and span_start is not None and span_end is not None and prompt_width is not None
        )
        # KV-reuse: a STABLE per-response ROUTING id pins all of a response's chunks to one teacher
        # replica (sticky session), so that replica's content-keyed vLLM prefix cache can reuse the prior
        # chunk's prefix KV. Incremental scoring REQUIRES this (it is what populates the cache it reads).
        # The SERVER request id stays unique per call (llm_server.py:209), so concurrent calls never collide.
        kv_reuse = os.environ.get("OPD_TEACHER_KV_REUSE", "0") not in ("0", "", "false", "False")
        use_stable_routing = (kv_reuse or incremental) and session_id is not None
        routing_request_id = f"teacher::{session_id}" if use_stable_routing else uuid4().hex

        # Stage 2: per-parent FIFO. Serialize this parent's chunk-score calls so the prefix KV is populated
        # in span order (chunk k completes before k+1 begins); different parents stay concurrent. The
        # SERVER request id stays unique per call so concurrent CROSS-parent calls never collide.
        # The exact rows this call will slice, in ABSOLUTE prompt positions: the shifted span starts at
        # prompt_width + span_start (see the next-token derivation below, where
        # shift_start_abs == teacher_span_start_abs), and this request's prompt IS prompt+response[:e],
        # so its last row is len(sequence_ids) - 1. Asking for anything narrower than
        # [shift_start, len(sequence_ids)) would strand rows the server still intends to hand back:
        # it extracts [max(num_cached+1, 1, window_start), full_prefix_len), and a None inside that
        # range is a hard ValueError in extract_incremental_prompt_logprobs. Widening at the START is
        # what the server-side clamp does; the END must cover the full submitted length.
        logprobs_window = None
        if incremental and os.environ.get("OPD_TEACHER_LOGPROBS_WINDOW", "0") not in ("0", "", "false", "False"):
            logprobs_window = (int(prompt_width) + int(span_start), len(sequence_ids))

        def _generate():
            return client.generate(
                request_id=routing_request_id,
                prompt_ids=sequence_ids,
                sampling_params=_get_teacher_sampling_params(
                    teacher_model_config,
                    self.distillation_loss_config,
                    incremental=incremental,
                    logprobs_window=logprobs_window,
                ),
                image_data=multi_modal_data.get("images"),
                video_data=multi_modal_data.get("videos"),
                audio_data=multi_modal_data.get("audios"),
                mm_processor_kwargs=mm_processor_kwargs,
                # Parent-aware routing: account this sticky parent (only when stable routing is on)
                # and release its parent-debt on the final chunk. No-op under the default policy.
                track_parent=bool(use_stable_routing),
                is_final=bool(is_final),
            )

        fifo_on = (
            fifo_enabled(incremental) and session_id is not None and span_start is not None and span_end is not None
        )
        if os.environ.get("OPD_TEACHER_PER_PARENT_FIFO", "0") not in ("0", "", "false", "False") and not incremental:
            if not self._fifo_warned:
                logging.getLogger(__name__).warning(
                    "OPD_TEACHER_PER_PARENT_FIFO=1 but incremental scoring is off; FIFO is a no-op "
                    "(clean scoring has no ordering precondition) -- leaving behavior unchanged."
                )
                self._fifo_warned = True
        fifo_wait_s = fifo_score_s = None
        t0 = time.monotonic()
        if fifo_on:
            teacher_output, fifo_wait_s, fifo_score_s = await self._fifo.run(
                str(session_id), int(span_start), int(span_end), bool(is_final), _generate
            )
        else:
            teacher_output = await _generate()
        latency_s = time.monotonic() - t0
        ef = getattr(teacher_output, "extra_fields", {}) or {}
        cached = ef.get("num_cached_tokens")
        telemetry = {
            "cached_tokens": cached,
            "uncached_tokens": (len(sequence_ids) - cached) if cached is not None else None,
            "total_tokens": len(sequence_ids),
            "replica_rank": ef.get("replica_rank"),
            "latency_s": latency_s,
            "queue_wait_s": ef.get("queue_wait_s"),
            "kv_reuse": bool(use_stable_routing),
            "incremental": bool(incremental),
            "fifo": bool(fifo_on),
            "fifo_wait_s": fifo_wait_s,
            "fifo_score_s": fifo_score_s,
            "fifo_snapshot": self._fifo.snapshot() if fifo_on else None,
        }
        if os.environ.get("OPD_TAIL_DEBUG", "0") not in ("0", "", "false", "False"):
            # Tail-latency probe (observability only): per teacher call, the post-generation teacher work.
            # U-mem path = ONE call/response (incremental=False) -> latency_s is the full-response teacher
            # tail. OPDFlow = per-chunk; the is_final call's (fifo_wait_s + latency_s) is its tail, earlier
            # chunks having overlapped with generation. Aligned by wall-clock + session_id offline.
            # print(), NOT logging: this runs inside a Ray actor, and only STDOUT is
            # forwarded to the driver log. The logging call went to stderr and was
            # silently dropped on BSC MN5 -- zero [TAIL] lines in any acc run, which
            # is why teacher tail latency (and therefore alpha) could not be measured
            # there at all. Same "[TAIL] " prefix and field names, so existing LUMI
            # extractors keep matching.
            print(
                "[TAIL] ts=%.3f sid=%s is_final=%s incr=%s total_tok=%s cached_tok=%s uncached_tok=%s "
                "latency_s=%.4f fifo_wait_s=%s queue_wait_s=%s span=%s"
                % (
                    time.time(), str(session_id)[-16:], is_final, bool(incremental), len(sequence_ids),
                    telemetry["cached_tokens"], telemetry["uncached_tokens"],
                    latency_s, (round(fifo_wait_s, 4) if fifo_wait_s is not None else None),
                    ef.get("queue_wait_s"),
                    f"[{span_start},{span_end})" if span_start is not None else "full",
                ),
                flush=True,
            )

        if not incremental:
            # Full-sequence labels (shape [S, (1 or K)]).
            teacher_ids = torch.tensor(ef["prompt_ids"], dtype=torch.int32)
            teacher_logprobs = torch.tensor(ef["prompt_logprobs"])
            assert teacher_ids.shape[0] == teacher_logprobs.shape[0] == len(sequence_ids)
            return teacher_ids, teacher_logprobs, telemetry

        # Incremental: server returned ONLY the valid recomputed suffix rows + its absolute start. Slice
        # the exact desired teacher span with explicit offsets; HARD-fail rather than silently pad/repair.
        suffix_ids = ef["prompt_ids"]
        suffix_lps = ef["prompt_logprobs"]
        valid_suffix_start_abs = int(ef["valid_suffix_start_abs"])
        valid_suffix_end_abs = int(ef["valid_suffix_end_abs"])
        full_prefix_len = int(ef.get("full_prefix_len", len(sequence_ids)))
        teacher_span_start_abs = prompt_width + span_start
        teacher_span_end_abs = prompt_width + span_end
        n = span_end - span_start
        K = self.distillation_loss_config.topk if self.distillation_loss_config.loss_settings.use_topk else 1
        # Next-token convention: teacher tensor index i holds the prediction for token i+1 (the strict
        # parser gets this by iterating prompt_logprobs[1:]). So the label for response token t sits at
        # index P+t-1, and a chunk owning response tokens [s, e) fills indices [P+s-1, P+e-1) with the
        # predictions for tokens [P+s, P+e) -- i.e. prompt_logprobs rows [P+s, P+e), every one of which
        # THIS request already has (its prompt is prompt + response[:e], length P+e).
        #
        # This previously read [P+s+1, P+e+1) and wrote at [P+s, P+e), which made a chunk responsible for
        # the prediction of token P+e -- the FIRST TOKEN OF THE NEXT CHUNK, not yet generated when this
        # request was scored. So last_is_dummy (which compares against valid_suffix_end_abs = P+span_end)
        # was unconditionally TRUE for every chunk, not just the final one as its comment claimed, and a
        # [0]*K row was appended each time. Under the student's left shift that dummy landed on a real
        # trained position: one all-zero teacher row per chunk, which exp() reads as uniform mass K.
        # Confirmed by actor/distillation/teacher_mass tracking 1 + (K-1)*ceil(R/c)/R exactly --
        # umem 0.99986, c4096 1.0155, c1024 1.0613, c256 1.2512.
        #
        # With the window realigned no dummy is ever needed: index P+R-1 (the prediction for the token
        # after the response) is simply never written, exactly as the strict parser's trailing dummy is
        # never read.
        shift_start_abs = teacher_span_start_abs
        shift_end_abs = teacher_span_end_abs
        last_is_dummy = False
        covered_end_abs = min(shift_end_abs, valid_suffix_end_abs)
        if os.environ.get("OPD_TEACHER_FIFO_DEBUG", "0") not in ("0", "", "false", "False"):
            logging.getLogger(__name__).warning(
                "[FIFO-DBG] sid=%s span=[%s,%s) n=%s cached=%s valid_suffix=[%s,%s) shifted=[%s,%s) "
                "fifo=%s wait=%s score=%s",
                str(session_id)[-14:], span_start, span_end, n, ef.get("num_cached_tokens"),
                valid_suffix_start_abs, valid_suffix_end_abs, shift_start_abs, shift_end_abs,
                fifo_on, None if fifo_wait_s is None else round(fifo_wait_s, 3),
                None if fifo_score_s is None else round(fifo_score_s, 3),
            )
        # Cross-response cache sharing: sibling responses to the SAME prompt share the replica's
        # content-keyed prefix cache, so when siblings share a few early RESPONSE tokens too, num_cached
        # for this response's chunk can land PAST its span_start -- the span's leading rows are then cached
        # (not recomputed) and unrecoverable from this request. FIFO cannot fix this (it is cross-response).
        # Fall back to a CLEAN recompute (skip_reading=True -> APC bypass -> every row recomputed) for this
        # chunk: correct, and rare (mostly cheap chunk-0s where siblings share a prefix).
        # Fall back to a clean recompute whenever the valid suffix does not fully cover the shifted span --
        # cross-response cache interference (num_cached past span_start) OR a fully-cached sequence (empty
        # suffix). The clean recompute (skip_reading=True) returns every row, so the span is always correct.
        needs_fallback = not (valid_suffix_start_abs <= shift_start_abs and covered_end_abs <= valid_suffix_end_abs)
        if needs_fallback:
            clean_out = await client.generate(
                request_id=uuid4().hex,
                prompt_ids=sequence_ids,
                sampling_params=_get_teacher_sampling_params(
                    teacher_model_config, self.distillation_loss_config, incremental=False
                ),
                image_data=multi_modal_data.get("images"),
                video_data=multi_modal_data.get("videos"),
                audio_data=multi_modal_data.get("audios"),
                mm_processor_kwargs=mm_processor_kwargs,
            )
            cef = getattr(clean_out, "extra_fields", {}) or {}
            full_ids = cef["prompt_ids"]  # strict full [S, K]: row j == prompt_logprobs[j+1]
            full_lps = cef["prompt_logprobs"]
            # -1: the strict parser is SHIFTED (row j == prompt_logprobs[j+1], trailing all-zero dummy
            # at row S-1), while the incremental suffix above is UNSHIFTED (row p scores token p). The
            # predictions for tokens [P+s, P+e) therefore sit at strict rows [P+s-1, P+e-1). Slicing
            # [P+s, P+e) here returned every label one token late and ended on the dummy, so a fallback
            # chunk trained each position against the NEXT token's distribution and its last position
            # against an all-zero row (uniform mass K) -- the residual teacher_mass_max == K that
            # appeared in exactly the metric windows where teacher/fallback_clean_count > 0.
            new_ids = list(full_ids[teacher_span_start_abs - 1:teacher_span_end_abs - 1])
            new_lps = list(full_lps[teacher_span_start_abs - 1:teacher_span_end_abs - 1])
            teacher_ids, teacher_logprobs = _finalize_span_tensors(new_ids, new_lps, n, K)
            telemetry.update({
                "fallback_clean": True,
                "teacher_span_start_abs": teacher_span_start_abs,
                "teacher_span_end_abs": teacher_span_end_abs,
                "retained_span_rows": n,
            })
            return teacher_ids, teacher_logprobs, telemetry

        local_start = shift_start_abs - valid_suffix_start_abs
        local_end = covered_end_abs - valid_suffix_start_abs
        new_ids = list(suffix_ids[local_start:local_end])
        new_lps = list(suffix_lps[local_start:local_end])
        if last_is_dummy:  # final chunk: pad the last position's missing next-token label with a dummy
            new_ids.append([0] * K)
            new_lps.append([0.0] * K)
        teacher_ids, teacher_logprobs = _finalize_span_tensors(new_ids, new_lps, n, K)
        telemetry.update({
            "full_prefix_len": full_prefix_len,
            "valid_suffix_start_abs": valid_suffix_start_abs,
            "valid_suffix_end_abs": valid_suffix_end_abs,
            "teacher_span_start_abs": teacher_span_start_abs,
            "teacher_span_end_abs": teacher_span_end_abs,
            "raw_rows": full_prefix_len,
            "valid_suffix_rows": len(suffix_ids),
            "retained_span_rows": n,
        })
        return teacher_ids, teacher_logprobs, telemetry
