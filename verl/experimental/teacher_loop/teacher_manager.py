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
import math
import os
import time
from typing import Any, Optional
from uuid import uuid4

import torch
from omegaconf import DictConfig
from torch.nn import functional as F

from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import (
    DistillationConfig,
    DistillationLossConfig,
    DistillationTeacherModelConfig,
)
from verl.workers.rollout.llm_server import LLMServerClient


def _get_teacher_sampling_params(
    teacher_model_config: DistillationTeacherModelConfig,
    distillation_loss_config: DistillationLossConfig,
    incremental: bool = False,
) -> dict[str, Any]:
    """Get sampling parameters for teacher model when computing log probabilities for distillation."""
    if teacher_model_config.inference.temperature != 1.0:
        raise NotImplementedError("vLLM does not support temperature for prompt_logprobs.")

    num_logprobs = distillation_loss_config.topk if distillation_loss_config.loss_settings.use_topk else 0
    params = {
        "max_tokens": 1,
        "temperature": teacher_model_config.inference.temperature,
        "prompt_logprobs": num_logprobs,
    }
    if incremental:
        # Read the cached prefix KV (default would skip it for prompt_logprobs) and skip detokenization
        # (the cached-prefix rows hold out-of-range garbage ids that crash the detokenizer). The server
        # then returns only the valid recomputed suffix; we slice the exact new span below.
        params["skip_reading_prefix_cache"] = False
        params["detokenize"] = False
    return params


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
        t0 = time.monotonic()
        teacher_output = await client.generate(
            request_id=routing_request_id,
            prompt_ids=sequence_ids,
            sampling_params=_get_teacher_sampling_params(
                teacher_model_config, self.distillation_loss_config, incremental=incremental
            ),
            image_data=multi_modal_data.get("images"),
            video_data=multi_modal_data.get("videos"),
            audio_data=multi_modal_data.get("audios"),
            mm_processor_kwargs=mm_processor_kwargs,
        )
        latency_s = time.monotonic() - t0
        ef = getattr(teacher_output, "extra_fields", {}) or {}
        telemetry = {
            "cached_tokens": ef.get("num_cached_tokens"),
            "total_tokens": len(sequence_ids),
            "replica_rank": ef.get("replica_rank"),
            "latency_s": latency_s,
            "queue_wait_s": ef.get("queue_wait_s"),
            "kv_reuse": bool(use_stable_routing),
            "incremental": bool(incremental),
        }

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
        # Next-token convention: the teacher label supervising student position p is the prediction for
        # token p+1 = prompt_logprobs[p+1] (the strict parser achieves this by iterating prompt_logprobs[1:]).
        # So shift the extraction window +1. The FINAL response position has no in-sequence next token, so
        # its label is a dummy (exactly as the strict parser pads a dummy last row).
        shift_start_abs = teacher_span_start_abs + 1
        shift_end_abs = teacher_span_end_abs + 1
        last_is_dummy = shift_end_abs > valid_suffix_end_abs
        covered_end_abs = min(shift_end_abs, valid_suffix_end_abs)
        # coverage guards (the shifted span must lie inside the returned valid suffix)
        if not (valid_suffix_start_abs <= shift_start_abs and covered_end_abs <= valid_suffix_end_abs):
            raise AssertionError(
                f"incremental span not covered: shifted [{shift_start_abs},{shift_end_abs}) "
                f"not within valid suffix [{valid_suffix_start_abs},{valid_suffix_end_abs})"
            )
        local_start = shift_start_abs - valid_suffix_start_abs
        local_end = covered_end_abs - valid_suffix_start_abs
        new_ids = list(suffix_ids[local_start:local_end])
        new_lps = list(suffix_lps[local_start:local_end])
        if last_is_dummy:  # final chunk: pad the last position's missing next-token label with a dummy
            new_ids.append([0] * K)
            new_lps.append([0.0] * K)
        # hard guards: exact row count, valid top-k width, valid token ids, finite logprobs (no repair)
        if len(new_ids) != n or len(new_lps) != n:
            raise AssertionError(f"incremental retained {len(new_ids)} rows != requested span length {n}")
        for r, (ids_row, lp_row) in enumerate(zip(new_ids, new_lps)):
            if len(ids_row) != K or len(lp_row) != K:
                raise AssertionError(f"incremental row {r}: top-k width {len(ids_row)}/{len(lp_row)} != {K}")
            if any((tid is None or int(tid) < 0) for tid in ids_row):
                raise AssertionError(f"incremental row {r}: invalid token id")
            if any((lp is None or not math.isfinite(lp)) for lp in lp_row):
                raise AssertionError(f"incremental row {r}: non-finite logprob")
        teacher_ids = torch.tensor(new_ids, dtype=torch.int32)
        teacher_logprobs = torch.tensor(new_lps, dtype=torch.float32)
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
