# Copyright 2025 Meituan Ltd. and/or its affiliates
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
import bisect
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.experimental.fully_async_policy.chunk_sample import ChunkSample
from verl.trainer.ppo.ray_trainer import compute_response_mask


@dataclass
class RolloutSample:
    """Enhanced rollout sample containing both original batch info and AgentLoopOutput"""

    # Original batch information
    full_batch: Any

    # Metadata
    sample_id: str
    epoch: int

    # Processing metadata
    rollout_status: dict[str, Any]


def prepare_single_generation_data(batch_dict, config) -> DataProto:
    """
    Similar to the logic of ray_trainer._prepare_generate_batch, but for a single sample.
    Separate the data used for generation from the original data.

    Returns:
        tuple: (original_batch_dict, gen_data_for_single_sample)
    """

    full_batch = DataProto.from_single_dict(batch_dict)

    batch_keys_to_pop = []
    non_tensor_batch_keys_to_pop = []

    existing_batch_keys = [k for k in batch_keys_to_pop if k in full_batch.batch.keys()]
    existing_non_tensor_keys = [k for k in non_tensor_batch_keys_to_pop if k in full_batch.non_tensor_batch.keys()]

    if existing_batch_keys or existing_non_tensor_keys:
        full_batch.pop(
            batch_keys=existing_batch_keys,
            non_tensor_batch_keys=existing_non_tensor_keys,
        )

    # Setting selected agent, that supports partial
    if not config.actor_rollout_ref.rollout.multi_turn.enable:
        full_batch.non_tensor_batch["agent_name"] = np.array(["single_turn_agent"] * len(full_batch), dtype=object)

    # Add global step count to generated data
    full_batch = full_batch.repeat(repeat_times=config.actor_rollout_ref.rollout.n, interleave=True)
    return full_batch


def addition_process(output: DataProto):
    """collect metirics"""
    metrics = output.meta_info.pop("metrics")  # List[Dict[str, str]]
    processing_times_list = [item["generate_sequences"] for item in metrics]
    tool_calls_times_list = [item["tool_calls"] for item in metrics]
    output.non_tensor_batch["processing_times"] = processing_times_list
    output.non_tensor_batch["tool_calls_times"] = tool_calls_times_list
    return output


def get_chunk_token_size(config) -> int:
    """Resolve the real chunk data-path token size.

    `async_training.chunk_tokens` is the primary knob. For compatibility with
    the profiling scripts, `OPD_STAGE1_CHUNK_TOKENS` is honored only when
    `OPD_STAGE1_REAL_CHUNKS=1` is set, so the old synthetic A/B mode remains
    opt-in and non-invasive.
    """
    chunk_tokens = int(config.async_training.get("chunk_tokens", 0) or 0)
    real_chunks_env = os.environ.get("OPD_STAGE1_REAL_CHUNKS", "0").strip().lower() in {"1", "true", "yes", "on"}
    if chunk_tokens <= 0 and real_chunks_env:
        try:
            chunk_tokens = int(os.environ.get("OPD_STAGE1_CHUNK_TOKENS", "0"))
        except ValueError:
            chunk_tokens = 0
    return max(0, chunk_tokens)


def is_chunk_data_path_enabled(config) -> bool:
    """Return True when rollouter/trainer should exchange `ChunkSample`s."""
    return get_chunk_token_size(config) > 0


def get_chunk_token_budget(config, chunk_tokens: int | None = None) -> int:
    """Resolve trainer-side token budget for one chunk batch."""
    explicit_budget = config.async_training.get("chunk_token_budget", None)
    if explicit_budget is not None:
        return max(1, int(explicit_budget))

    if chunk_tokens is None:
        chunk_tokens = get_chunk_token_size(config)
    required_units = int(config.actor_rollout_ref.actor.ppo_mini_batch_size) * int(config.async_training.require_batches)
    rollout_n = int(config.actor_rollout_ref.rollout.get("n", 1))
    return max(1, required_units * max(1, rollout_n) * max(1, int(chunk_tokens)))


def get_optimizer_step_token_budget(config) -> int:
    """Resolve the per-optimizer-step token budget for the chunk trainer.

    When > 0, the trainer accumulates streamed-chunk supervision across multiple
    memory-safe ``fit_step``s until the accumulated train-token count reaches this
    budget, then performs ONE ``optimizer.step()`` + policy-version increment +
    weight sync. This decouples the control plane (how often the student changes)
    from the data plane (how early chunks arrive), so fewer policy versions elapse
    during a long decode -- WITHOUT the cosmetic effect of merely raising
    ``trigger_parameter_sync_step`` (which keeps the student stepping every chunk
    batch and only lets the rollouter lag more).

    Returns ``0`` to DISABLE (default -> exact current per-fit-step behavior).

    Resolution precedence (first match wins), then floored at one chunk batch:
      1. env  ``OPD_OPTIMIZER_STEP_TOKEN_BUDGET``  (absolute tokens)
      2. cfg  ``async_training.optimizer_step_token_budget``  (absolute tokens)
      3. env  ``OPD_OPTIMIZER_STEP_BUDGET_MULT``   (multiplier of get_chunk_token_budget)
    The "moderate" setting is mult == trigger_parameter_sync_step, i.e. one optimizer
    step per ``trigger_parameter_sync_step`` worth of chunk-token supervision. A
    resolved value is floored to ``max(value, get_chunk_token_budget(config))`` so one
    optimizer step always covers at least one memory-safe chunk batch.
    """
    chunk_budget = get_chunk_token_budget(config)

    abs_env = _positive_int_env("OPD_OPTIMIZER_STEP_TOKEN_BUDGET")
    if abs_env is not None:
        return max(abs_env, chunk_budget)

    cfg_val = config.async_training.get("optimizer_step_token_budget", None)
    if cfg_val is not None and int(cfg_val) > 0:
        return max(int(cfg_val), chunk_budget)

    mult_raw = os.environ.get("OPD_OPTIMIZER_STEP_BUDGET_MULT", "").strip()
    if mult_raw:
        try:
            mult = float(mult_raw)
        except ValueError:
            mult = 0.0
        if mult > 0:
            return max(int(round(mult * chunk_budget)), chunk_budget)

    return 0


def get_chunk_staleness_threshold(config) -> float:
    """Resolve the trainer-side staleness threshold for chunk payloads."""
    sigma = config.async_training.get("chunk_staleness_threshold", None)
    if sigma is None:
        sigma = config.async_training.staleness_threshold
    return float(sigma)


@dataclass
class ChunkBatchSelection:
    train_chunks: list[ChunkSample]
    deferred_chunks: list[ChunkSample]
    collected_rows: int
    usable_rows: int
    deferred_rows: int
    world_size: int
    train_row_divisor: int
    max_effective_seq_len: int
    estimated_train_tokens: int
    max_chunk_rows_budget: int | None
    max_train_tokens_budget: int | None
    max_effective_seq_len_budget: int | None
    trimmed_by_dp_divisibility: bool
    trimmed_by_memory_budget: bool
    budget_prefix_rows: int
    budget_prefix_chunks: int

    def as_metrics(self) -> dict[str, int | bool | None]:
        return {
            "collected_rows": self.collected_rows,
            "usable_rows": self.usable_rows,
            "deferred_rows": self.deferred_rows,
            "world_size": self.world_size,
            "train_row_divisor": self.train_row_divisor,
            "max_effective_seq_len": self.max_effective_seq_len,
            "estimated_train_tokens": self.estimated_train_tokens,
            "max_chunk_rows_budget": self.max_chunk_rows_budget,
            "max_train_tokens_budget": self.max_train_tokens_budget,
            "max_effective_seq_len_budget": self.max_effective_seq_len_budget,
            "trimmed_by_dp_divisibility": self.trimmed_by_dp_divisibility,
            "trimmed_by_memory_budget": self.trimmed_by_memory_budget,
            "budget_prefix_rows": self.budget_prefix_rows,
            "budget_prefix_chunks": self.budget_prefix_chunks,
        }


@dataclass
class ChunkCoalescingConfig:
    enabled: bool
    max_coalesced_chunks: int | None
    max_coalesced_effective_seq_len: int | None
    lookahead: int = 0


@dataclass
class ChunkCoalescingResult:
    chunks: list[ChunkSample]
    metrics: dict[str, int | float]


def _positive_int_env(name: str) -> int | None:
    value = os.environ.get(name, "").strip()
    if not value:
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def _env_flag(name: str) -> bool | None:
    value = os.environ.get(name, "").strip().lower()
    if not value:
        return None
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return None


def get_chunk_coalescing_config(config) -> ChunkCoalescingConfig:
    """Resolve optional same-parent contiguous chunk coalescing knobs."""
    enabled = bool(config.async_training.get("coalesce_contiguous_chunks", False))
    env_enabled = _env_flag("OPD_STAGE1_COALESCE_CONTIGUOUS")
    if env_enabled is not None:
        enabled = env_enabled

    max_chunks = config.async_training.get("max_coalesced_chunks", None)
    env_max_chunks = _positive_int_env("OPD_STAGE1_MAX_COALESCED_CHUNKS")
    if env_max_chunks is not None:
        max_chunks = env_max_chunks
    max_chunks = int(max_chunks) if max_chunks is not None and int(max_chunks) > 0 else None

    max_seq_len = config.async_training.get("max_coalesced_effective_seq_len", None)
    env_max_seq_len = _positive_int_env("OPD_STAGE1_MAX_COALESCED_EFFECTIVE_SEQ_LEN")
    if env_max_seq_len is not None:
        max_seq_len = env_max_seq_len
    max_seq_len = int(max_seq_len) if max_seq_len is not None and int(max_seq_len) > 0 else None

    lookahead = config.async_training.get("coalesce_lookahead", 0)
    env_lookahead = os.environ.get("OPD_STAGE1_COALESCE_LOOKAHEAD", "").strip()
    if env_lookahead:
        try:
            lookahead = int(env_lookahead)
        except ValueError:
            pass
    lookahead = int(lookahead) if lookahead is not None and int(lookahead) > 0 else 0

    return ChunkCoalescingConfig(
        enabled=enabled,
        max_coalesced_chunks=max_chunks,
        max_coalesced_effective_seq_len=max_seq_len,
        lookahead=lookahead,
    )


def get_chunk_coalescing_drain_multiplier(config) -> float:
    """Resolve how far past the train token budget the trainer may opportunistically
    drain *already-available* queued chunks before coalescing.

    A value of 1.0 (default) reproduces the legacy behavior: stop collecting as soon
    as the train token budget is met. Values > 1.0 let the collection loop pull up to
    ``multiplier * token_budget`` worth of chunks that are *already sitting* in the
    message queue (it never blocks waiting for new arrivals), giving the same-parent
    coalescer a larger pool of sibling chunks. Leftover rows beyond the train budget
    are deferred to the next batch via the existing pending-chunk buffer, so this only
    changes how many ready chunks are *visible* to coalescing, not the train batch size.
    """
    multiplier = config.async_training.get("coalesce_drain_multiplier", 1.0)
    env_multiplier = os.environ.get("OPD_STAGE1_COALESCE_DRAIN_MULTIPLIER", "").strip()
    if env_multiplier:
        try:
            multiplier = float(env_multiplier)
        except ValueError:
            pass
    try:
        multiplier = float(multiplier)
    except (TypeError, ValueError):
        multiplier = 1.0
    return multiplier if multiplier > 1.0 else 1.0


def get_chunk_batch_memory_limits_from_env() -> dict[str, int | None]:
    """Return optional Xiaoshuai trainer batch memory limits from env knobs."""
    return {
        "max_chunk_rows": _positive_int_env("OPD_STAGE1_MAX_CHUNK_ROWS"),
        "max_train_tokens": _positive_int_env("OPD_STAGE1_MAX_TRAIN_TOKENS"),
        "max_effective_seq_len": _positive_int_env("OPD_STAGE1_MAX_EFFECTIVE_SEQ_LEN"),
    }


def iter_chunk_constituents(chunk: ChunkSample) -> list[ChunkSample]:
    """Return original chunks represented by a possibly coalesced training row."""
    meta = chunk.meta if isinstance(chunk.meta, dict) else {}
    originals = meta.get("_merged_original_chunks")
    if isinstance(originals, list) and originals:
        return originals
    return [chunk]


def flatten_chunk_constituents(chunks: list[ChunkSample]) -> list[ChunkSample]:
    originals: list[ChunkSample] = []
    for chunk in chunks:
        originals.extend(iter_chunk_constituents(chunk))
    return originals


def get_original_chunk_count(chunk: ChunkSample) -> int:
    return len(iter_chunk_constituents(chunk))


def get_chunk_row_count(chunk: ChunkSample) -> int:
    payload = getattr(chunk, "parent_payload", None)
    if payload is not None:
        try:
            return max(1, int(len(payload)))
        except Exception:
            pass
    return 1


def choose_chunk_actor_mini_batch_size(
    selected_rows: int,
    world_size: int,
    configured_actor_mini_batch_size: int,
) -> int:
    """Pick a chunk actor mini-batch that divides every DP worker's local rows.

    The normal configured actor mini-batch is preferred.  When a memory-safe
    FIFO prefix is DP-divisible but not divisible by the configured per-GPU
    mini-batch, fall back to the largest safe per-GPU divisor no larger than
    the configured value.  This avoids dropping/defering otherwise trainable
    chunks purely to satisfy the worker dataloader assertion.
    """
    selected_rows = max(1, int(selected_rows))
    world_size = max(1, int(world_size))
    configured = max(world_size, int(configured_actor_mini_batch_size))

    if selected_rows % world_size != 0:
        return configured

    if selected_rows <= configured:
        return selected_rows

    configured_per_gpu = max(1, configured // world_size)
    local_rows = max(1, selected_rows // world_size)
    local_mini_batch = math.gcd(local_rows, configured_per_gpu)
    return max(world_size, world_size * max(1, local_mini_batch))


def estimate_chunk_effective_seq_len(chunk: ChunkSample) -> int:
    """Estimate the sequence length that actor update must process for a chunk."""
    payload = getattr(chunk, "parent_payload", None)
    batch = getattr(payload, "batch", None)
    if batch is not None and "attention_mask" in batch:
        attention_mask = batch["attention_mask"]
        if attention_mask.dim() >= 2:
            return max(1, int(attention_mask.reshape(attention_mask.shape[0], -1).sum(-1).max().item()))

    if batch is not None and "prompts" in batch:
        prompt_width = int(batch["prompts"].shape[1])
        response_end = None
        if isinstance(chunk.meta, dict):
            response_end = chunk.meta.get("response_end")
        if response_end is None:
            response_end = int(chunk.token_offset) + int(chunk.n_tokens)
        return max(1, prompt_width + int(response_end))

    if batch is not None and "input_ids" in batch:
        input_ids = batch["input_ids"]
        if input_ids.dim() >= 2:
            return max(1, int(input_ids.shape[1]))

    return max(1, int(chunk.token_offset) + int(chunk.n_tokens))


def _first_non_tensor_value_from_payload(chunk: ChunkSample, key: str):
    payload = getattr(chunk, "parent_payload", None)
    non_tensor_batch = getattr(payload, "non_tensor_batch", None)
    if non_tensor_batch is None or key not in non_tensor_batch:
        return None
    values = non_tensor_batch.get(key)
    if values is None or len(values) == 0:
        return None
    value = values[0]
    if hasattr(value, "item"):
        value = value.item()
    return value


def _chunk_parent_key(chunk: ChunkSample) -> tuple[str, str, str | None]:
    meta = chunk.meta if isinstance(chunk.meta, dict) else {}
    row_id = str(meta.get("row_id", chunk.sample_id))
    parent_sample_id = str(meta.get("parent_sample_id", chunk.sample_id))
    uid = _first_non_tensor_value_from_payload(chunk, "uid")
    return row_id, parent_sample_id, str(uid) if uid is not None else None


def _chunk_response_span(chunk: ChunkSample) -> tuple[int, int]:
    start = int(chunk.token_offset)
    meta = chunk.meta if isinstance(chunk.meta, dict) else {}
    end = meta.get("response_end")
    if end is None:
        end = start + int(chunk.n_tokens)
    return start, int(end)


def _has_teacher_payload(chunk: ChunkSample) -> bool:
    payload = getattr(chunk, "parent_payload", None)
    batch = getattr(payload, "batch", None)
    return batch is not None and "teacher_ids" in batch and "teacher_logprobs" in batch


def _can_start_coalesced_group(chunk: ChunkSample) -> bool:
    meta = chunk.meta if isinstance(chunk.meta, dict) else {}
    # Keep completed-sample fallback behavior unchanged in this first version.
    if meta.get("source") != "streaming":
        return False
    return _has_teacher_payload(chunk)


def _can_extend_coalesced_group(previous: ChunkSample, candidate: ChunkSample) -> bool:
    if not _can_start_coalesced_group(candidate):
        return False
    if _chunk_parent_key(previous) != _chunk_parent_key(candidate):
        return False
    if int(previous.policy_version) != int(candidate.policy_version):
        return False
    previous_start, previous_end = _chunk_response_span(previous)
    candidate_start, candidate_end = _chunk_response_span(candidate)
    if previous_end != candidate_start or candidate_end <= candidate_start:
        return False
    return int(previous.chunk_idx) + 1 == int(candidate.chunk_idx)


# Reasons a same-parent successor candidate is rejected for merging. Used for
# bounded-lookahead coalescing diagnostics so we can tell whether the coalescer
# is failing to activate and why.
MERGE_REJECT_REASONS = (
    "different_parent",
    "noncontiguous_span",
    "policy_version",
    "effective_len_cap",
    "missing_teacher_payload",
    "fallback",
    "outside_lookahead",
)


def _extend_rejection_reason(
    previous: ChunkSample,
    candidate: ChunkSample,
    effective_cap: int | None,
) -> str | None:
    """Classify why ``candidate`` cannot extend the group ending at ``previous``.

    The caller has already established that ``candidate`` is the earliest
    same-parent forward chunk with ``chunk_idx == previous.chunk_idx + 1`` and
    that it lies inside the lookahead window.  Returns ``None`` when the merge
    is allowed, otherwise one of ``MERGE_REJECT_REASONS``.
    """
    meta = candidate.meta if isinstance(candidate.meta, dict) else {}
    if meta.get("source") != "streaming":
        return "fallback"
    if not _has_teacher_payload(candidate):
        return "missing_teacher_payload"
    if _chunk_parent_key(previous) != _chunk_parent_key(candidate):
        return "different_parent"
    if int(previous.policy_version) != int(candidate.policy_version):
        return "policy_version"
    previous_start, previous_end = _chunk_response_span(previous)
    candidate_start, candidate_end = _chunk_response_span(candidate)
    if previous_end != candidate_start or candidate_end <= candidate_start:
        return "noncontiguous_span"
    if int(previous.chunk_idx) + 1 != int(candidate.chunk_idx):
        return "noncontiguous_span"
    if effective_cap is not None and estimate_chunk_effective_seq_len(candidate) > int(effective_cap):
        return "effective_len_cap"
    return None


def _clone_dataproto(data: DataProto) -> DataProto:
    return DataProto(
        batch=data.batch.clone(),
        non_tensor_batch=_clone_numpy_dict(data.non_tensor_batch),
        meta_info=dict(data.meta_info),
    )


def _overlay_response_tensor(target: torch.Tensor, source: torch.Tensor, start: int, end: int) -> None:
    if target.dim() < 2 or source.dim() < 2:
        return
    src_end = min(end, int(source.shape[1]))
    dst_end = min(end, int(target.shape[1]))
    if start >= src_end or start >= dst_end:
        return
    width = min(src_end - start, dst_end - start)
    target[:, start : start + width].copy_(source[:, start : start + width])


def _overlay_sequence_tensor(
    target: torch.Tensor,
    source: torch.Tensor,
    target_prompt_width: int,
    source_prompt_width: int,
    start: int,
    end: int,
) -> None:
    if target.dim() < 2 or source.dim() < 2:
        return
    source_start = source_prompt_width + start
    target_start = target_prompt_width + start
    src_end = min(source_prompt_width + end, int(source.shape[1]))
    dst_end = min(target_prompt_width + end, int(target.shape[1]))
    if source_start >= src_end or target_start >= dst_end:
        return
    width = min(src_end - source_start, dst_end - target_start)
    target[:, target_start : target_start + width].copy_(source[:, source_start : source_start + width])


def _concat_chunk_tokens(chunks: list[ChunkSample]):
    values = [chunk.tokens for chunk in chunks]
    if all(isinstance(value, torch.Tensor) for value in values):
        first = values[0]
        dim = 1 if first.dim() >= 2 else 0
        try:
            return torch.cat(values, dim=dim)
        except Exception:
            return values[-1]
    if all(isinstance(value, list) for value in values):
        out = []
        for value in values:
            out.extend(value)
        return out
    return values[-1]


def _merge_contiguous_chunk_group(chunks: list[ChunkSample]) -> ChunkSample:
    if len(chunks) == 1:
        return chunks[0]

    base = chunks[-1]
    merged_payload = _clone_dataproto(base.parent_payload)
    target_batch = merged_payload.batch
    prompt_width = int(target_batch["prompts"].shape[1])

    response_aligned_keys = {
        "response_mask",
        "rollout_log_probs",
        "rm_scores",
        "token_level_scores",
        "advantages",
        "old_log_probs",
    }
    for key in response_aligned_keys:
        if key in target_batch and target_batch[key].dim() >= 2:
            target_batch[key].zero_()

    sequence_aligned_keys = {"teacher_ids", "teacher_logprobs"}
    for chunk in chunks:
        start, end = _chunk_response_span(chunk)
        source_batch = chunk.parent_payload.batch
        source_prompt_width = int(source_batch["prompts"].shape[1])
        for key in response_aligned_keys:
            if key in target_batch and key in source_batch:
                _overlay_response_tensor(target_batch[key], source_batch[key], start, end)
        for key in sequence_aligned_keys:
            if key in target_batch and key in source_batch:
                _overlay_sequence_tensor(
                    target_batch[key],
                    source_batch[key],
                    prompt_width,
                    source_prompt_width,
                    start,
                    end,
                )

    first = chunks[0]
    _, response_end = _chunk_response_span(base)
    total_tokens = sum(int(chunk.n_tokens) for chunk in chunks)
    merged_ids = [
        (chunk.meta if isinstance(chunk.meta, dict) else {}).get("chunk_id", f"{chunk.sample_id}:{chunk.chunk_idx}")
        for chunk in chunks
    ]
    merged_indices = [int(chunk.chunk_idx) for chunk in chunks]
    merged_offsets = [int(chunk.token_offset) for chunk in chunks]
    merged_ends = [_chunk_response_span(chunk)[1] for chunk in chunks]

    batch_size = len(merged_payload)
    merged_payload.non_tensor_batch["chunk_idx"] = np.array([first.chunk_idx] * batch_size, dtype=np.int32)
    merged_payload.non_tensor_batch["chunk_token_offset"] = np.array([first.token_offset] * batch_size, dtype=np.int32)
    merged_payload.non_tensor_batch["chunk_n_tokens"] = np.array([total_tokens] * batch_size, dtype=np.int32)
    merged_payload.non_tensor_batch["chunk_is_final"] = np.array([base.is_final] * batch_size, dtype=bool)
    merged_payload.non_tensor_batch["chunk_policy_version"] = np.array(
        [first.policy_version] * batch_size, dtype=np.int32
    )

    meta = dict(first.meta if isinstance(first.meta, dict) else {})
    meta.update(
        {
            "coalesced": True,
            "merged_chunk_count": len(chunks),
            "merged_chunk_ids": merged_ids,
            "merged_chunk_indices": merged_indices,
            "merged_chunk_offsets": merged_offsets,
            "merged_chunk_ends": merged_ends,
            "response_end": response_end,
            "_merged_original_chunks": chunks,
        }
    )
    return ChunkSample(
        sample_id=first.sample_id,
        chunk_idx=int(first.chunk_idx),
        token_offset=int(first.token_offset),
        n_tokens=int(total_tokens),
        tokens=_concat_chunk_tokens(chunks),
        is_final=bool(base.is_final),
        policy_version=int(first.policy_version),
        parent_payload=merged_payload,
        meta=meta,
    )


def _coalescing_metrics(
    before_chunks: list[ChunkSample],
    after_chunks: list[ChunkSample],
    enabled: bool,
    *,
    lookahead: int = 0,
    rejections: dict[str, int] | None = None,
    opportunities: int = 0,
) -> dict[str, int | float]:
    before_prefix_tokens = sum(get_chunk_row_count(chunk) * estimate_chunk_effective_seq_len(chunk) for chunk in before_chunks)
    after_prefix_tokens = sum(get_chunk_row_count(chunk) * estimate_chunk_effective_seq_len(chunk) for chunk in after_chunks)
    chunks_per_row = [get_original_chunk_count(chunk) for chunk in after_chunks] or [0]
    merged_group_sizes = [count for count in chunks_per_row if count > 1]
    chunks_merged_total = sum(merged_group_sizes)
    reduction = max(0, before_prefix_tokens - after_prefix_tokens)
    rows_before = sum(get_chunk_row_count(chunk) for chunk in before_chunks)
    rows_after = sum(get_chunk_row_count(chunk) for chunk in after_chunks)
    rejections = rejections or {}
    metrics: dict[str, int | float] = {
        "enabled": int(enabled),
        "coalesce_lookahead": int(lookahead),
        "pending_chunks_before_coalesce": len(before_chunks),
        "pending_rows_before_coalesce": rows_before,
        "rows_before_coalesce": rows_before,
        "coalescible_chunks": chunks_merged_total,
        "coalescing_opportunities_visible_in_window": int(opportunities),
        "coalesced_groups": len(merged_group_sizes),
        "chunks_merged": chunks_merged_total,
        "chunks_merged_total": chunks_merged_total,
        "rows_after_coalesce": rows_after,
        "chunks_per_merged_row_median": float(np.median(chunks_per_row)),
        "chunks_per_merged_row_p95": float(np.percentile(chunks_per_row, 95)),
        "chunks_per_merged_row_max": int(max(chunks_per_row)),
        "estimated_prefix_tokens_before_coalesce": int(before_prefix_tokens),
        "estimated_prefix_tokens_after_coalesce": int(after_prefix_tokens),
        "estimated_prefix_recompute_reduction": int(reduction),
        "estimated_prefix_recompute_reduction_fraction": float(reduction / before_prefix_tokens)
        if before_prefix_tokens
        else 0.0,
    }
    for reason in MERGE_REJECT_REASONS:
        metrics[f"merge_reject_{reason}"] = int(rejections.get(reason, 0))
    return metrics


def coalesce_contiguous_chunk_samples(
    chunk_samples: list[ChunkSample],
    config: ChunkCoalescingConfig,
    max_effective_seq_len: int | None = None,
) -> ChunkCoalescingResult:
    """Bounded-lookahead FIFO coalescing of same-parent streaming chunks.

    The function inspects a sliding window of already-pending chunks.  Starting
    from the earliest FIFO chunk it merges that chunk's same-parent contiguous
    successors that fall within ``config.lookahead`` pending positions ahead of
    the current frontier.  The merged row is placed at the FIFO slot of its
    earliest constituent; remaining chunks keep their relative order.

    When ``config.lookahead`` is ``0`` the window collapses to ``1`` and the
    behavior matches the original adjacent-only coalescer (a successor must be
    the immediately next pending chunk).

    The function never waits for future chunks, never reorders beyond placing a
    merged row at its earliest constituent position, and never drops chunks.
    Original chunks are carried in the merged row's metadata for accounting and
    staleness rechecks.
    """
    rejections: dict[str, int] = {reason: 0 for reason in MERGE_REJECT_REASONS}
    if not config.enabled or not chunk_samples:
        return ChunkCoalescingResult(
            chunks=chunk_samples,
            metrics=_coalescing_metrics(
                chunk_samples,
                chunk_samples,
                enabled=config.enabled,
                lookahead=config.lookahead,
                rejections=rejections,
                opportunities=0,
            ),
        )

    max_group_size = config.max_coalesced_chunks
    effective_cap = config.max_coalesced_effective_seq_len
    if max_effective_seq_len is not None:
        effective_cap = min(effective_cap, max_effective_seq_len) if effective_cap is not None else max_effective_seq_len

    window = config.lookahead if config.lookahead > 0 else 1
    n = len(chunk_samples)

    use_index = n >= _coalesce_index_threshold()
    if use_index:
        coalesced, opportunities = _coalesce_indexed(
            chunk_samples,
            window=window,
            max_group_size=max_group_size,
            effective_cap=effective_cap,
            rejections=rejections,
        )
    else:
        coalesced, opportunities = _coalesce_scan(
            chunk_samples,
            window=window,
            max_group_size=max_group_size,
            effective_cap=effective_cap,
            rejections=rejections,
        )

    return ChunkCoalescingResult(
        chunks=coalesced,
        metrics=_coalescing_metrics(
            chunk_samples,
            coalesced,
            enabled=config.enabled,
            lookahead=config.lookahead,
            rejections=rejections,
            opportunities=opportunities,
        ),
    )


def _coalesce_scan(
    chunk_samples: list[ChunkSample],
    *,
    window: int,
    max_group_size: int | None,
    effective_cap: int | None,
    rejections: dict[str, int],
) -> tuple[list[ChunkSample], int]:
    """Reference O(n^2) bounded-lookahead coalescer used for small pools.

    Returns the coalesced chunk list and the static opportunity count. This is
    the original, simplest implementation; the indexed variant must reproduce
    its output exactly.
    """
    n = len(chunk_samples)
    consumed = [False] * n

    opportunities = _count_coalescing_opportunities(chunk_samples, window)

    coalesced: list[ChunkSample] = []
    for i in range(n):
        if consumed[i]:
            continue
        head = chunk_samples[i]
        consumed[i] = True
        if not _can_start_coalesced_group(head):
            coalesced.append(head)
            continue

        parent_key = _chunk_parent_key(head)
        group = [head]
        prev_pos = i
        while max_group_size is None or len(group) < int(max_group_size):
            prev = group[-1]
            best_j = None
            best_rank = 0
            rank = 0
            for j in range(prev_pos + 1, n):
                if consumed[j]:
                    continue
                rank += 1
                candidate = chunk_samples[j]
                if _chunk_parent_key(candidate) == parent_key and int(candidate.chunk_idx) > int(prev.chunk_idx):
                    best_j = j
                    best_rank = rank
                    break
            if best_j is None:
                rejections["different_parent"] += 1
                break
            candidate = chunk_samples[best_j]
            if int(candidate.chunk_idx) != int(prev.chunk_idx) + 1:
                rejections["noncontiguous_span"] += 1
                break
            if best_rank > window:
                rejections["outside_lookahead"] += 1
                break
            reason = _extend_rejection_reason(prev, candidate, effective_cap)
            if reason is not None:
                rejections[reason] += 1
                break
            consumed[best_j] = True
            group.append(candidate)
            prev_pos = best_j

        coalesced.append(_merge_contiguous_chunk_group(group))

    return coalesced, opportunities


class _FenwickTree:
    """Minimal Fenwick (binary-indexed) tree over unconsumed-position flags.

    Supports O(log n) point updates and prefix sums so the indexed coalescer can
    compute the lookahead ``rank`` (number of still-unconsumed positions between
    the current frontier and a candidate) without re-scanning the pool.
    """

    __slots__ = ("_n", "_tree")

    def __init__(self, n: int):
        self._n = n
        self._tree = [0] * (n + 1)

    def add(self, index: int, delta: int) -> None:
        i = index + 1
        while i <= self._n:
            self._tree[i] += delta
            i += i & (-i)

    def prefix_sum(self, index: int) -> int:
        """Sum of flags over positions [0, index] inclusive."""
        i = index + 1
        total = 0
        while i > 0:
            total += self._tree[i]
            i -= i & (-i)
        return total


def _coalesce_indexed(
    chunk_samples: list[ChunkSample],
    *,
    window: int,
    max_group_size: int | None,
    effective_cap: int | None,
    rejections: dict[str, int],
) -> tuple[list[ChunkSample], int]:
    """Hash-index + Fenwick fast path for large pools.

    Builds a per-parent index of pool positions so a same-parent successor is
    found by scanning only same-parent entries (not the whole pool), and uses a
    Fenwick tree of unconsumed positions to evaluate the lookahead ``rank`` in
    O(log n). Produces output identical to :func:`_coalesce_scan`.
    """
    n = len(chunk_samples)
    consumed = [False] * n

    # Per-parent ascending pool positions for constant-ish successor lookup.
    parent_positions: dict[Any, list[int]] = defaultdict(list)
    for pos, chunk in enumerate(chunk_samples):
        parent_positions[_chunk_parent_key(chunk)].append(pos)

    fenwick = _FenwickTree(n)
    for pos in range(n):
        fenwick.add(pos, 1)

    def _consume(pos: int) -> None:
        consumed[pos] = True
        fenwick.add(pos, -1)

    def _next_same_parent(parent_key: Any, prev_pos: int, prev_chunk_idx: int) -> int | None:
        positions = parent_positions.get(parent_key)
        if not positions:
            return None
        start = bisect.bisect_right(positions, prev_pos)
        for k in range(start, len(positions)):
            pos = positions[k]
            if consumed[pos]:
                continue
            if int(chunk_samples[pos].chunk_idx) > prev_chunk_idx:
                return pos
        return None

    opportunities = _count_coalescing_opportunities_indexed(chunk_samples, window, parent_positions)

    coalesced: list[ChunkSample] = []
    for i in range(n):
        if consumed[i]:
            continue
        head = chunk_samples[i]
        _consume(i)
        if not _can_start_coalesced_group(head):
            coalesced.append(head)
            continue

        parent_key = _chunk_parent_key(head)
        group = [head]
        prev_pos = i
        while max_group_size is None or len(group) < int(max_group_size):
            prev = group[-1]
            best_j = _next_same_parent(parent_key, prev_pos, int(prev.chunk_idx))
            if best_j is None:
                rejections["different_parent"] += 1
                break
            candidate = chunk_samples[best_j]
            if int(candidate.chunk_idx) != int(prev.chunk_idx) + 1:
                rejections["noncontiguous_span"] += 1
                break
            best_rank = fenwick.prefix_sum(best_j) - fenwick.prefix_sum(prev_pos)
            if best_rank > window:
                rejections["outside_lookahead"] += 1
                break
            reason = _extend_rejection_reason(prev, candidate, effective_cap)
            if reason is not None:
                rejections[reason] += 1
                break
            _consume(best_j)
            group.append(candidate)
            prev_pos = best_j

        coalesced.append(_merge_contiguous_chunk_group(group))

    return coalesced, opportunities


def _coalesce_index_threshold() -> int:
    """Pool size at/above which the hash-indexed coalescer fast path is used.

    Defaults to 256 so small pools keep the allocation-free reference scan and
    only large drained pools pay for the index. Overridable via
    ``OPD_STAGE1_COALESCE_INDEX_THRESHOLD`` (values <= 1 disable the fast path).
    """
    raw = os.environ.get("OPD_STAGE1_COALESCE_INDEX_THRESHOLD", "").strip()
    if raw:
        try:
            parsed = int(raw)
        except ValueError:
            parsed = 256
        if parsed <= 1:
            return 1 << 62
        return parsed
    return 256


def _count_coalescing_opportunities(chunk_samples: list[ChunkSample], window: int) -> int:
    """Count same-parent contiguous successor pairs visible within ``window``.

    This is a static property of the input pending buffer (independent of the
    merge order) used to gauge how much coalescing the queue interleaving
    permits at the configured lookahead.
    """
    n = len(chunk_samples)
    opportunities = 0
    for p in range(n):
        head = chunk_samples[p]
        if not _can_start_coalesced_group(head):
            continue
        parent_key = _chunk_parent_key(head)
        rank = 0
        for q in range(p + 1, n):
            rank += 1
            candidate = chunk_samples[q]
            if _chunk_parent_key(candidate) == parent_key and int(candidate.chunk_idx) > int(head.chunk_idx):
                if (
                    int(candidate.chunk_idx) == int(head.chunk_idx) + 1
                    and rank <= window
                    and _can_extend_coalesced_group(head, candidate)
                ):
                    opportunities += 1
                break
    return opportunities


def _count_coalescing_opportunities_indexed(
    chunk_samples: list[ChunkSample],
    window: int,
    parent_positions: dict[Any, list[int]],
) -> int:
    """Index-accelerated equivalent of :func:`_count_coalescing_opportunities`.

    Uses the per-parent position index so each head inspects only same-parent
    entries. Since this is a static pass (nothing consumed) the lookahead rank
    of a successor at pool position ``q`` for a head at ``p`` is exactly
    ``q - p``. Produces the same count as the linear-scan version.
    """
    opportunities = 0
    for p, head in enumerate(chunk_samples):
        if not _can_start_coalesced_group(head):
            continue
        parent_key = _chunk_parent_key(head)
        positions = parent_positions.get(parent_key)
        if not positions:
            continue
        head_chunk_idx = int(head.chunk_idx)
        start = bisect.bisect_right(positions, p)
        for k in range(start, len(positions)):
            q = positions[k]
            if int(chunk_samples[q].chunk_idx) > head_chunk_idx:
                if (
                    int(chunk_samples[q].chunk_idx) == head_chunk_idx + 1
                    and (q - p) <= window
                    and _can_extend_coalesced_group(head, chunk_samples[q])
                ):
                    opportunities += 1
                break
    return opportunities


def select_chunk_samples_for_train_batch(
    chunk_samples: list[ChunkSample],
    batch_divisor: int,
    min_rows: int,
    train_row_divisor: int | None = None,
    max_chunk_rows: int | None = None,
    max_train_tokens: int | None = None,
    max_effective_seq_len: int | None = None,
) -> ChunkBatchSelection:
    """Select a memory-safe DP-divisible FIFO prefix and defer the suffix."""
    world_size = max(1, int(batch_divisor))
    divisor = max(world_size, int(train_row_divisor) if train_row_divisor is not None else world_size)
    min_rows = max(1, int(min_rows))
    collected_rows = sum(get_chunk_row_count(chunk) for chunk in chunk_samples)
    selected_chunk_count = 0
    selected_rows = 0
    selected_tokens = 0
    selected_max_seq_len = 0

    current_rows = 0
    current_tokens = 0
    current_max_seq_len = 0
    budget_prefix_chunks = 0
    budget_prefix_rows = 0
    trimmed_by_memory_budget = False

    for index, chunk in enumerate(chunk_samples):
        row_count = get_chunk_row_count(chunk)
        effective_seq_len = estimate_chunk_effective_seq_len(chunk)
        next_rows = current_rows + row_count
        next_tokens = current_tokens + row_count * effective_seq_len
        next_max_seq_len = max(current_max_seq_len, effective_seq_len)

        exceeds_budget = (
            (max_chunk_rows is not None and next_rows > int(max_chunk_rows))
            or (max_train_tokens is not None and next_tokens > int(max_train_tokens))
            or (max_effective_seq_len is not None and next_max_seq_len > int(max_effective_seq_len))
        )
        if exceeds_budget:
            trimmed_by_memory_budget = True
            break

        current_rows = next_rows
        current_tokens = next_tokens
        current_max_seq_len = next_max_seq_len
        budget_prefix_chunks = index + 1
        budget_prefix_rows = current_rows
        if current_rows >= min_rows and current_rows % divisor == 0:
            selected_chunk_count = index + 1
            selected_rows = current_rows
            selected_tokens = current_tokens
            selected_max_seq_len = current_max_seq_len

    if budget_prefix_chunks == 0 and not chunk_samples:
        budget_prefix_rows = 0

    train_chunks = chunk_samples[:selected_chunk_count]
    deferred_chunks = chunk_samples[selected_chunk_count:]
    deferred_rows = collected_rows - selected_rows
    trimmed_by_dp_divisibility = selected_chunk_count < budget_prefix_chunks
    return ChunkBatchSelection(
        train_chunks=train_chunks,
        deferred_chunks=deferred_chunks,
        collected_rows=collected_rows,
        usable_rows=selected_rows,
        deferred_rows=deferred_rows,
        world_size=world_size,
        train_row_divisor=divisor,
        max_effective_seq_len=selected_max_seq_len,
        estimated_train_tokens=selected_tokens,
        max_chunk_rows_budget=max_chunk_rows,
        max_train_tokens_budget=max_train_tokens,
        max_effective_seq_len_budget=max_effective_seq_len,
        trimmed_by_dp_divisibility=trimmed_by_dp_divisibility,
        trimmed_by_memory_budget=trimmed_by_memory_budget,
        budget_prefix_rows=budget_prefix_rows,
        budget_prefix_chunks=budget_prefix_chunks,
    )


def split_chunk_samples_for_balanced_batch(
    chunk_samples: list[ChunkSample], batch_divisor: int, min_chunks: int
) -> tuple[list[ChunkSample], list[ChunkSample]]:
    """Return a DP-divisible FIFO prefix and defer the remaining suffix."""
    divisor = max(1, int(batch_divisor))
    if divisor <= 1:
        return chunk_samples, []

    selection = select_chunk_samples_for_train_batch(
        chunk_samples=chunk_samples,
        batch_divisor=divisor,
        min_rows=min_chunks,
    )
    return selection.train_chunks, selection.deferred_chunks


def _clone_numpy_dict(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: np.array(value, copy=True) for key, value in data.items()}


def _as_int_list(values: Any) -> list[int]:
    if values is None:
        return []
    if isinstance(values, np.ndarray):
        values = values.tolist()
    if not isinstance(values, list | tuple):
        values = [values]
    out = []
    for value in values:
        if value is None:
            continue
        if hasattr(value, "item"):
            value = value.item()
        if value is not None:
            out.append(int(value))
    return out


def _extract_policy_version(batch: DataProto, default: int = 0) -> int:
    """Use the oldest version represented in this batch as the chunk version."""
    for key in ("min_global_steps", "max_global_steps"):
        versions = _as_int_list(batch.non_tensor_batch.get(key))
        if versions:
            return min(versions)
    return int(default)


def _pad_tensor_dim(tensor: torch.Tensor, dim: int, target: int, value: int | float) -> torch.Tensor:
    current = tensor.shape[dim]
    if current == target:
        return tensor.clone()
    if current > target:
        index = [slice(None)] * tensor.dim()
        index[dim] = slice(0, target)
        return tensor[tuple(index)].clone()

    pad_shape = list(tensor.shape)
    pad_shape[dim] = target - current
    pad = torch.full(pad_shape, value, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, pad], dim=dim)


def _crop_sequence_dim(
    tensor: torch.Tensor,
    prompt_width: int,
    response_width: int,
    response_end: int,
    dim: int,
) -> torch.Tensor:
    total_width = prompt_width + response_width
    if tensor.shape[dim] != total_width:
        return tensor.clone()
    prompt_index = [slice(None)] * tensor.dim()
    prompt_index[dim] = slice(0, prompt_width)
    response_index = [slice(None)] * tensor.dim()
    response_index[dim] = slice(prompt_width, prompt_width + response_end)
    return torch.cat([tensor[tuple(prompt_index)], tensor[tuple(response_index)]], dim=dim).clone()


def _crop_position_ids(tensor: torch.Tensor, prompt_width: int, response_width: int, response_end: int) -> torch.Tensor:
    total_width = prompt_width + response_width
    if tensor.shape[-1] != total_width:
        return tensor.clone()
    prompt = tensor[..., :prompt_width]
    response = tensor[..., prompt_width : prompt_width + response_end]
    return torch.cat([prompt, response], dim=-1).clone()


def _build_chunk_dataproto(
    full_batch: DataProto,
    token_offset: int,
    response_end: int,
) -> DataProto:
    """Build a trainable prompt+prefix chunk from a full rollout batch."""
    prompt_width = full_batch.batch["prompts"].shape[1]
    response_width = full_batch.batch["responses"].shape[1]
    total_width = prompt_width + response_width

    chunk_tensors = {}
    for key, tensor in full_batch.batch.items():
        if key == "prompts":
            chunk_tensors[key] = tensor.clone()
        elif key == "responses":
            chunk_tensors[key] = tensor[:, :response_end].clone()
        elif key == "response_mask":
            response_mask = tensor[:, :response_end].clone()
            if token_offset > 0:
                response_mask[:, :token_offset] = 0
            chunk_tensors[key] = response_mask
        elif key in {"rollout_log_probs", "rm_scores", "token_level_scores", "advantages", "old_log_probs"}:
            value = tensor[:, :response_end].clone()
            if key in {"rm_scores", "token_level_scores", "advantages"} and token_offset > 0:
                value[:, :token_offset] = 0
            chunk_tensors[key] = value
        elif key in {"input_ids", "attention_mask", "teacher_ids", "teacher_logprobs", "routed_experts"}:
            if tensor.dim() >= 2 and tensor.shape[1] == total_width:
                chunk_tensors[key] = _crop_sequence_dim(tensor, prompt_width, response_width, response_end, dim=1)
            else:
                chunk_tensors[key] = tensor.clone()
        elif key == "position_ids":
            chunk_tensors[key] = _crop_position_ids(tensor, prompt_width, response_width, response_end)
        elif tensor.dim() >= 2 and tensor.shape[1] == response_width:
            chunk_tensors[key] = tensor[:, :response_end].clone()
        elif tensor.dim() >= 2 and tensor.shape[1] == total_width:
            chunk_tensors[key] = _crop_sequence_dim(tensor, prompt_width, response_width, response_end, dim=1)
        else:
            chunk_tensors[key] = tensor.clone()

    chunk_batch = TensorDict(source=chunk_tensors, batch_size=full_batch.batch.batch_size)
    return DataProto(
        batch=chunk_batch,
        non_tensor_batch=_clone_numpy_dict(full_batch.non_tensor_batch),
        meta_info=dict(full_batch.meta_info),
    )


def pad_chunk_dataproto_to_response_width(
    chunk_batch: DataProto,
    target_response_width: int,
    pad_token_id: int,
) -> DataProto:
    """Pad a chunk batch so `DataProto.concat` can combine mixed chunk widths."""
    response_width = chunk_batch.batch["responses"].shape[1]
    if response_width == target_response_width:
        return DataProto(
            batch=chunk_batch.batch.clone(),
            non_tensor_batch=_clone_numpy_dict(chunk_batch.non_tensor_batch),
            meta_info=dict(chunk_batch.meta_info),
        )

    prompt_width = chunk_batch.batch["prompts"].shape[1]
    sequence_width = prompt_width + response_width
    target_sequence_width = prompt_width + target_response_width
    padded_tensors = {}
    for key, tensor in chunk_batch.batch.items():
        if key == "prompts":
            padded_tensors[key] = tensor.clone()
        elif key == "responses":
            padded_tensors[key] = _pad_tensor_dim(tensor, dim=1, target=target_response_width, value=pad_token_id)
        elif key in {
            "response_mask",
            "rollout_log_probs",
            "rm_scores",
            "token_level_scores",
            "advantages",
            "old_log_probs",
        }:
            padded_tensors[key] = _pad_tensor_dim(tensor, dim=1, target=target_response_width, value=0)
        elif key == "input_ids":
            padded_tensors[key] = _pad_tensor_dim(tensor, dim=1, target=target_sequence_width, value=pad_token_id)
        elif key in {"attention_mask", "teacher_logprobs", "routed_experts"}:
            padded_tensors[key] = _pad_tensor_dim(tensor, dim=1, target=target_sequence_width, value=0)
        elif key == "teacher_ids":
            padded_tensors[key] = _pad_tensor_dim(tensor, dim=1, target=target_sequence_width, value=pad_token_id)
        elif key == "position_ids" and tensor.shape[-1] == sequence_width:
            padded_tensors[key] = _pad_tensor_dim(tensor, dim=tensor.dim() - 1, target=target_sequence_width, value=0)
        elif tensor.dim() >= 2 and tensor.shape[1] == response_width:
            padded_tensors[key] = _pad_tensor_dim(tensor, dim=1, target=target_response_width, value=0)
        elif tensor.dim() >= 2 and tensor.shape[1] == sequence_width:
            padded_tensors[key] = _pad_tensor_dim(tensor, dim=1, target=target_sequence_width, value=0)
        else:
            padded_tensors[key] = tensor.clone()

    return DataProto(
        batch=TensorDict(source=padded_tensors, batch_size=chunk_batch.batch.batch_size),
        non_tensor_batch=_clone_numpy_dict(chunk_batch.non_tensor_batch),
        meta_info=dict(chunk_batch.meta_info),
    )


def _front_pad_dim(tensor: torch.Tensor, dim: int, target: int, value) -> torch.Tensor:
    """Left-pad (prepend) `tensor` along `dim` to width `target` (mirror of _pad_tensor_dim)."""
    cur = tensor.shape[dim]
    if cur == target:
        return tensor.clone()
    if cur > target:  # not expected (target is the batch max); keep the rightmost `target`
        idx = [slice(None)] * tensor.dim()
        idx[dim] = slice(cur - target, cur)
        return tensor[tuple(idx)].clone()
    pad_shape = list(tensor.shape)
    pad_shape[dim] = target - cur
    pad = torch.full(pad_shape, value, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([pad, tensor], dim=dim)


def _split_pad_seq(tensor: torch.Tensor, p: int, target_p: int, target_r: int, value) -> torch.Tensor:
    """For a [B, p+r, ...] sequence tensor: LEFT-pad the prompt span (first p) to target_p and
    RIGHT-pad the response span (remaining r) to target_r, preserving the prompt|response split."""
    prompt_part = _front_pad_dim(tensor[:, :p], 1, target_p, value)
    resp_part = _pad_tensor_dim(tensor[:, p:], 1, target_r, value)
    return torch.cat([prompt_part, resp_part], dim=1)


_PAD_PROMPT_KEYS = {"prompts"}
_PAD_RESPONSE_KEYS = {
    "responses", "response_mask", "rollout_log_probs", "rm_scores",
    "token_level_scores", "advantages", "old_log_probs",
}
_PAD_SEQ_KEYS = {"input_ids", "attention_mask", "teacher_logprobs", "routed_experts", "teacher_ids"}
_PAD_TOKEN_VALUE_KEYS = {"prompts", "responses", "input_ids", "teacher_ids"}


def pad_dataproto_to_prompt_response_width(
    batch: DataProto, target_prompt_width: int, target_response_width: int, pad_token_id: int
) -> DataProto:
    """Pad one DataProto so its prompt span (LEFT-padded) reaches `target_prompt_width` and its
    response span (RIGHT-padded) reaches `target_response_width`, so DataProto.concat can combine
    samples/chunks with different prompt AND response lengths (the prior helper unified responses
    only -> mixed prompt widths crashed torch.cat). attention_mask is padded with 0 in both spans
    and position_ids is recomputed from the padded mask (standard left-pad convention), so padded
    positions are masked and never affect the loss. Token tensors pad with pad_token_id; masks /
    log-probs / scores pad with 0."""
    P, R = int(target_prompt_width), int(target_response_width)
    p = int(batch.batch["prompts"].shape[1])
    r = int(batch.batch["responses"].shape[1])
    S = p + r
    if p == P and r == R:
        return DataProto(
            batch=batch.batch.clone(),
            non_tensor_batch=_clone_numpy_dict(batch.non_tensor_batch),
            meta_info=dict(batch.meta_info),
        )
    out = {}
    for key, t in batch.batch.items():
        if key == "position_ids":
            out[key] = t  # recomputed from the padded attention_mask below
            continue
        val = pad_token_id if key in _PAD_TOKEN_VALUE_KEYS else 0
        if key in _PAD_PROMPT_KEYS:
            out[key] = _front_pad_dim(t, 1, P, val)
        elif key in _PAD_RESPONSE_KEYS:
            out[key] = _pad_tensor_dim(t, 1, R, val)
        elif key in _PAD_SEQ_KEYS:
            out[key] = _split_pad_seq(t, p, P, R, val)
        else:
            # unknown key: classify by width (sequence > response > prompt); clone if none match
            w = t.shape[1] if t.dim() >= 2 else None
            if w == S:
                out[key] = _split_pad_seq(t, p, P, R, val)
            elif w == r:
                out[key] = _pad_tensor_dim(t, 1, R, val)
            elif w == p and p != r:
                out[key] = _front_pad_dim(t, 1, P, val)
            else:
                out[key] = t.clone()
    # Recompute position_ids from the padded attention_mask (left-pad convention) so real tokens
    # keep contiguous positions and pad positions are harmless (masked out).
    if "position_ids" in batch.batch:
        am = out.get("attention_mask")
        orig_pos = batch.batch["position_ids"]
        if am is not None and orig_pos.dim() == 2:
            out["position_ids"] = (am.cumsum(dim=-1) - 1).clamp_(min=0).to(orig_pos.dtype)
        elif orig_pos.dim() >= 2 and orig_pos.shape[1] == S:
            out["position_ids"] = _split_pad_seq(orig_pos, p, P, R, 0)  # uncommon (e.g. mrope)
        else:
            out["position_ids"] = orig_pos.clone()
    return DataProto(
        batch=TensorDict(source=out, batch_size=batch.batch.batch_size),
        non_tensor_batch=_clone_numpy_dict(batch.non_tensor_batch),
        meta_info=dict(batch.meta_info),
    )


def create_chunk_samples_from_rollout_sample(
    rollout_sample: RolloutSample,
    chunk_tokens: int,
    policy_version: int | None = None,
) -> list[ChunkSample]:
    """Split a completed rollout batch into trainer-visible chunk samples."""
    if chunk_tokens <= 0:
        raise ValueError(f"chunk_tokens must be positive, got {chunk_tokens}")

    full_batch = rollout_sample.full_batch
    if full_batch.batch is None:
        raise ValueError("rollout_sample.full_batch must contain tensor batch data")
    if "responses" not in full_batch.batch or "response_mask" not in full_batch.batch:
        raise ValueError("chunked OPD requires responses and response_mask in rollout batch")

    prompt_width = full_batch.batch["prompts"].shape[1]
    response_width = full_batch.batch["responses"].shape[1]
    if "attention_mask" in full_batch.batch:
        response_attention = full_batch.batch["attention_mask"][:, prompt_width : prompt_width + response_width]
        max_response_len = int(response_attention.sum(dim=1).max().item())
    else:
        max_response_len = int(full_batch.batch["response_mask"].sum(dim=1).max().item())
    max_response_len = min(max_response_len, response_width)
    if max_response_len <= 0:
        return []

    if policy_version is None:
        policy_version = _extract_policy_version(full_batch)

    chunks = []
    for chunk_idx, token_offset in enumerate(range(0, max_response_len, chunk_tokens)):
        response_end = min(token_offset + chunk_tokens, max_response_len)
        response_mask_slice = full_batch.batch["response_mask"][:, token_offset:response_end]
        n_tokens = int(response_mask_slice.sum().item())
        if n_tokens <= 0:
            continue

        chunk_batch = _build_chunk_dataproto(full_batch, token_offset=token_offset, response_end=response_end)
        batch_size = len(chunk_batch)
        chunk_batch.non_tensor_batch["chunk_sample_id"] = np.array(
            [rollout_sample.sample_id] * batch_size, dtype=object
        )
        chunk_batch.non_tensor_batch["chunk_idx"] = np.array([chunk_idx] * batch_size, dtype=np.int32)
        chunk_batch.non_tensor_batch["chunk_token_offset"] = np.array([token_offset] * batch_size, dtype=np.int32)
        chunk_batch.non_tensor_batch["chunk_n_tokens"] = np.array([n_tokens] * batch_size, dtype=np.int32)
        chunk_batch.non_tensor_batch["chunk_is_final"] = np.array(
            [response_end >= max_response_len] * batch_size, dtype=bool
        )
        chunk_batch.non_tensor_batch["chunk_policy_version"] = np.array([policy_version] * batch_size, dtype=np.int32)

        chunks.append(
            ChunkSample(
                sample_id=rollout_sample.sample_id,
                chunk_idx=chunk_idx,
                token_offset=token_offset,
                n_tokens=n_tokens,
                tokens=full_batch.batch["responses"][:, token_offset:response_end].clone(),
                is_final=response_end >= max_response_len,
                policy_version=int(policy_version),
                parent_payload=chunk_batch,
                meta={
                    "epoch": rollout_sample.epoch,
                    "rollout_status": dict(rollout_sample.rollout_status),
                    "source": "fallback",
                    "parent_sample_id": rollout_sample.sample_id,
                    "response_end": response_end,
                    "response_width": response_width,
                },
            )
        )

    return chunks


def assemble_batch_from_chunk_samples(
    chunk_samples: list[ChunkSample], tokenizer, config, balance_batch=None
) -> DataProto:
    """Assemble a trainer batch from chunk-level rollout payloads."""
    start_time = time.time()

    if not chunk_samples:
        raise ValueError("Empty chunk_samples provided for batch assembly")

    print(f"[BatchUtils] Assembling batch from {len(chunk_samples)} ChunkSample objects")

    max_prompt_width = max(int(chunk.parent_payload.batch["prompts"].shape[1]) for chunk in chunk_samples)
    max_response_width = max(int(chunk.parent_payload.batch["responses"].shape[1]) for chunk in chunk_samples)
    pad_token_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
    chunk_batches = []
    for chunk in chunk_samples:
        # Unify BOTH prompt and response widths (the prior response-only padding crashed
        # torch.cat when parents had different prompt lengths).
        padded = pad_dataproto_to_prompt_response_width(
            chunk.parent_payload, max_prompt_width, max_response_width, pad_token_id
        )
        chunk_batches.append(addition_process(padded))

    final_batch = DataProto.concat(chunk_batches)

    if "response_mask" not in final_batch.batch.keys():
        final_batch.batch["response_mask"] = compute_response_mask(final_batch)

    if balance_batch:
        balance_batch(final_batch, metrics={})

    if "attention_mask" in final_batch.batch:
        final_batch.meta_info["global_token_num"] = torch.sum(final_batch.batch["attention_mask"], dim=-1).tolist()

    processing_times = final_batch.non_tensor_batch["processing_times"]
    tool_calls = final_batch.non_tensor_batch["tool_calls_times"]
    processing_time_stats = {
        "processing_time/avg": np.mean(processing_times),
        "processing_time/max": np.max(processing_times),
        "processing_time/min": np.min(processing_times),
        "processing_time/tp50": np.percentile(processing_times, 50),
        "processing_time/tp99": np.percentile(processing_times, 99),
        "processing_time/tp95": np.percentile(processing_times, 95),
    }
    tool_calls_stats = {}
    if len(tool_calls) > 0:
        tool_calls_stats = {
            "timing_s/agent_loop/tool_calls/max": np.max(tool_calls),
            "timing_s/agent_loop/tool_calls/min": np.min(tool_calls),
            "timing_s/agent_loop/tool_calls/mean": np.mean(tool_calls),
        }
    processing_time_stats = {f"fully_async/{key}": value for key, value in processing_time_stats.items()}

    original_chunks = flatten_chunk_constituents(chunk_samples)

    param_version_start = _as_int_list(final_batch.non_tensor_batch.get("min_global_steps", []))
    param_version_end = _as_int_list(final_batch.non_tensor_batch.get("max_global_steps", []))
    param_version_diff = [abs(a - b) for a, b in zip(param_version_end, param_version_start, strict=False)]
    if param_version_diff:
        num_diff0 = param_version_diff.count(0)
        partial_stats = {
            "fully_async/partial/total_partial_num": len(param_version_diff) - num_diff0,
            "fully_async/partial/partial_ratio": (len(param_version_diff) - num_diff0) / len(param_version_diff),
            "fully_async/partial/max_partial_span": max(param_version_diff),
        }
    else:
        partial_stats = {
            "fully_async/partial/total_partial_num": 0,
            "fully_async/partial/partial_ratio": 0.0,
            "fully_async/partial/max_partial_span": 0,
        }

    trajectory_param_versions = np.array(param_version_end, dtype=np.int32)
    rollout_status = chunk_samples[0].meta.get("rollout_status", {})
    rollout_status = {f"fully_async/{key}": value for key, value in rollout_status.items()}
    chunk_token_counts = [chunk.n_tokens for chunk in original_chunks]
    coalesced_rows = sum(1 for chunk in chunk_samples if get_original_chunk_count(chunk) > 1)
    chunk_stats = {
        "fully_async/chunk/enabled": 1,
        "fully_async/chunk/count": len(original_chunks),
        "fully_async/chunk/total_tokens": int(sum(chunk_token_counts)),
        "fully_async/chunk/avg_tokens": float(np.mean(chunk_token_counts)),
        "fully_async/chunk/max_response_width": int(max_response_width),
        "fully_async/chunk/coalesced_rows": coalesced_rows,
        "fully_async/chunk/train_rows": len(chunk_samples),
    }

    final_batch.meta_info.update(
        {
            "param_version_diversity": len(set(trajectory_param_versions.tolist())),
            "trajectory_param_versions": trajectory_param_versions,
            "chunk_samples": [
                {
                    "sample_id": chunk.sample_id,
                    "parent_sample_id": chunk.meta.get("parent_sample_id", chunk.sample_id),
                    "row_id": chunk.meta.get("row_id", chunk.sample_id),
                    "chunk_id": chunk.meta.get("chunk_id", f"{chunk.sample_id}:{chunk.chunk_idx}"),
                    "merged_chunk_ids": chunk.meta.get("merged_chunk_ids"),
                    "source": chunk.meta.get("source", "unknown"),
                    "chunk_idx": chunk.chunk_idx,
                    "n_tokens": chunk.n_tokens,
                    "token_offset": chunk.token_offset,
                    "policy_version": chunk.policy_version,
                    "is_final": chunk.is_final,
                }
                for chunk in original_chunks
            ],
            **processing_time_stats,
            **rollout_status,
            **partial_stats,
            **tool_calls_stats,
            **chunk_stats,
        }
    )

    print(f"[BatchUtils] Chunk batch assembly completed in {time.time() - start_time:.2f}s")

    return final_batch


def assemble_batch_from_rollout_samples(
    rollout_samples: list[RolloutSample], tokenizer, config, balance_batch=None
) -> DataProto:
    """
    Assemble gen_batch_output from RolloutSample objects
    Assembles batches from RolloutSample objects, similar to the _post_generate_batch logic in ray_trainer.

    Args:
        rollout_samples: List of RolloutSample objects
        tokenizer: Tokenizer instance
        config: Configuration object containing trainer settings
        balance_batch: Whether to balance the batch (simplified version)

    Returns:
        DataProto: Assembled gen_batch_output

    Raises:
        ValueError: If rollout_samples is empty
    """
    start_time = time.time()

    if not rollout_samples:
        raise ValueError("Empty rollout_samples provided for batch assembly")

    print(f"[BatchUtils] Assembling batch from {len(rollout_samples)} RolloutSample objects")

    rollout_status = rollout_samples[0].rollout_status
    # Add a prefix to all rollout_status keys
    rollout_status = {f"fully_async/{key}": value for key, value in rollout_status.items()}

    processed = [addition_process(rs.full_batch) for rs in rollout_samples]
    # Pad prompts (left) + responses (right) to a common width before concat. Upstream's
    # assembly concatenated unpadded samples, which crashes torch.cat on variable-length
    # batches (the long-reasoning regime) -- an infra fix needed to run completed-sample async
    # here at all; applied identically to all arms so it does not bias the comparison.
    target_prompt_width = max(int(b.batch["prompts"].shape[1]) for b in processed)
    target_response_width = max(int(b.batch["responses"].shape[1]) for b in processed)
    pad_token_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
    rollout_samples_batch = [
        pad_dataproto_to_prompt_response_width(b, target_prompt_width, target_response_width, pad_token_id)
        for b in processed
    ]
    final_batch = DataProto.concat(rollout_samples_batch)

    # Calculate response_mask (if not present)
    if "response_mask" not in final_batch.batch.keys():
        final_batch.batch["response_mask"] = compute_response_mask(final_batch)

    if balance_batch:
        balance_batch(final_batch, metrics={})

    # Calculate the global valid token number
    if "attention_mask" in final_batch.batch:
        final_batch.meta_info["global_token_num"] = torch.sum(final_batch.batch["attention_mask"], dim=-1).tolist()

    processing_times = final_batch.non_tensor_batch["processing_times"]
    tool_calls = final_batch.non_tensor_batch["tool_calls_times"]
    # Collect statistics
    processing_time_stats = {
        "processing_time/avg": np.mean(processing_times),
        "processing_time/max": np.max(processing_times),
        "processing_time/min": np.min(processing_times),
        "processing_time/tp50": np.percentile(processing_times, 50),
        "processing_time/tp99": np.percentile(processing_times, 99),
        "processing_time/tp95": np.percentile(processing_times, 95),
    }
    tool_calls_stats = {}
    if len(tool_calls) > 0:
        tool_calls_stats = {
            "timing_s/agent_loop/tool_calls/max": np.max(tool_calls),
            "timing_s/agent_loop/tool_calls/min": np.min(tool_calls),
            "timing_s/agent_loop/tool_calls/mean": np.mean(tool_calls),
        }
    processing_time_stats = {f"fully_async/{key}": value for key, value in processing_time_stats.items()}

    param_version_start = final_batch.non_tensor_batch["min_global_steps"]
    param_version_end = final_batch.non_tensor_batch["max_global_steps"]
    param_version_diff = [abs(a - b) for a, b in zip(param_version_end, param_version_start, strict=False)]
    num_diff0 = param_version_diff.count(0)
    partial_stats = {
        "fully_async/partial/total_partial_num": len(param_version_diff) - num_diff0,
        "fully_async/partial/partial_ratio": (len(param_version_diff) - num_diff0) / len(param_version_diff),
        "fully_async/partial/max_partial_span": max(param_version_diff),
    }
    # add meta_info
    trajectory_param_versions = final_batch.non_tensor_batch["max_global_steps"]

    final_batch.meta_info.update(
        {
            "param_version_diversity": len(set(trajectory_param_versions)),
            "trajectory_param_versions": trajectory_param_versions,
            **processing_time_stats,
            **rollout_status,
            **partial_stats,
            **tool_calls_stats,
        }
    )

    print(f"[BatchUtils] Batch assembly completed in {time.time() - start_time:.2f}s")

    return final_batch


class MetricsAggregator:
    """Metrics aggregator, used to combine metrics from multiple training steps"""

    def __init__(self, total_gpus: int):
        # Store all values ​​for each metric
        self.metric_values: dict[str, list[float]] = defaultdict(list)
        # Store the number of samples at each step for weighted averaging
        self.sample_counts: list[int] = []
        # Store the timestamp of each step for time-related calculations
        self.timestamps: list[float] = []
        # Step Count
        self.step_count = 0
        # total num gpus used
        self.total_gpus = total_gpus

        # Metric aggregation rule configuration
        self.aggregation_rules = self._init_aggregation_rules()

    def _init_aggregation_rules(self) -> dict[str, dict[str, list[str]]]:
        """Initialize metrics aggregation rules"""
        return {
            # Time-Based metrics, can add metrics here
            "time_sum": ["perf/time_per_step"],
            "min": ["timing_s/agent_loop/tool_calls/min"],
            "avg": ["timing_s/agent_loop/tool_calls/mean"],
            "max": ["timing_s/agent_loop/tool_calls/max"],
            "last": [
                "fully_async/count/total_generated_samples",
                "fully_async/count/stale_samples_processed",
                "fully_async/count/stale_trajectory_processed",
                "fully_async/count/current_param_version",
                "fully_async/count/dropped_stale_samples",
                "fully_async/count/dropped_stale_chunks",
                "fully_async/count/processed_chunks",
                "training/global_step",  # TODO change name to: total_step
            ],
        }

    def add_step_metrics(self, metrics: dict[str, Any], sample_count: int, timestamp: float = None):
        """Adding a single-step metrics"""
        if timestamp is None:
            timestamp = time.time()

        self.sample_counts.append(sample_count)
        self.timestamps.append(timestamp)
        self.step_count += 1

        # Store all metrics values
        for key, value in metrics.items():
            if isinstance(value, int | float | np.number):
                self.metric_values[key].append(float(value))
            elif isinstance(value, torch.Tensor):
                self.metric_values[key].append(float(value.item()))

    def _get_aggregation_type(self, metric_name: str) -> str:
        """Determine the aggregation type based on the metric name"""
        for agg_type, metric_list in self.aggregation_rules.items():
            if metric_name in metric_list:
                return agg_type

        metric_lower = metric_name.lower()
        if any(keyword in metric_lower for keyword in ["timing_s/"]):
            return "time_sum"
        if any(keyword in metric_lower for keyword in ["mean", "avg", "average"]):
            return "avg"
        if any(keyword in metric_lower for keyword in ["max", "maximum"]):
            return "max"
        if any(keyword in metric_lower for keyword in ["min", "minimum"]):
            return "min"
        if any(keyword in metric_lower for keyword in ["sum", "total"]):
            return "sum"
        if any(keyword in metric_lower for keyword in ["weighted_avg"]):
            return "weighted_avg"

        return "avg"

    def _aggregate_single_metric(self, metric_name: str, values: list[float]) -> float:
        """Aggregating a single metric"""
        if not values:
            return 0.0

        agg_type = self._get_aggregation_type(metric_name)

        if agg_type == "last":
            return values[-1]

        elif agg_type == "weighted_avg":
            # Weighted average
            if len(values) != len(self.sample_counts):
                # If the lengths do not match, use a simple average
                return sum(values) / len(values)

            total_samples = sum(self.sample_counts)
            if total_samples == 0:
                return sum(values) / len(values)

            weighted_sum = sum(v * c for v, c in zip(values, self.sample_counts, strict=False))
            return weighted_sum / total_samples

        elif agg_type == "sum" or agg_type == "time_sum":
            return sum(values)

        elif agg_type == "avg":
            return sum(values) / len(values)

        elif agg_type == "max":
            return max(values)

        elif agg_type == "min":
            return min(values)

        else:
            # Default average
            return sum(values) / len(values)

    def get_aggregated_metrics(self) -> dict[str, Any]:
        """aggregated metrics"""
        t = time.time()
        if self.step_count == 0:
            return {}

        aggregated = {}

        # Aggregate all metrics
        for metric_name, values in self.metric_values.items():
            aggregated[metric_name] = self._aggregate_single_metric(metric_name, values)

        # Aggregate special metrics
        aggregated = self._special_metrics_aggergate(aggregated)

        print(f"aggregated metrics done. cost {time.time() - t:.4f} seconds.")

        return aggregated

    def _special_metrics_aggergate(self, aggregated: dict[str, Any]) -> dict[str, Any]:
        """calculate special metrics"""

        # global_seqlen/minmax_diff
        if "global_seqlen/minmax_diff" in aggregated.keys():
            aggregated["global_seqlen/minmax_diff"] = aggregated["global_seqlen/max"] - aggregated["global_seqlen/min"]

        # perf/throughput
        REQUIRED_PERF_KEYS = {"perf/throughput", "perf/total_num_tokens", "perf/time_per_step"}
        if REQUIRED_PERF_KEYS.issubset(aggregated):
            aggregated["perf/throughput"] = aggregated["perf/total_num_tokens"] / (
                aggregated["perf/time_per_step"] * self.total_gpus
            )

        # trainer/idle_ratio
        if "timing_s/gen" in aggregated.keys() and "timing_s/step" in aggregated.keys():
            aggregated["fully_async/trainer/idle_ratio"] = aggregated["timing_s/gen"] / aggregated["timing_s/step"]

        return aggregated

    def reset(self):
        """Reset Aggregator"""
        self.metric_values.clear()
        self.sample_counts.clear()
        self.timestamps.clear()
        self.step_count = 0

    def get_current_stats(self) -> dict[str, Any]:
        """Get statistics about the current aggregation state (for debugging)"""
        return {
            "step_count": self.step_count,
            "metric_count": len(self.metric_values),
            "total_samples": sum(self.sample_counts),
            "metric_names": list(self.metric_values.keys()),
        }


def task_exception_handler(task: asyncio.Task):
    """Handle task exceptions and log them"""
    try:
        task.result()
    except asyncio.CancelledError:
        pass  # Task was cancelled, this is expected
    except Exception as e:
        print(f"Task {task.get_name()} failed with exception: {e}")
        raise e


def safe_create_task(coro, name: str, task_set: set = None):
    """Safely create a task with exception handling

    Args:
        coro: The coroutine to run
        name: Name for the task
        task_set: Optional set to add the task to

    Returns:
        The created asyncio.Task
    """
    task = asyncio.create_task(coro, name=name)
    task.add_done_callback(task_exception_handler)
    if task_set is not None:
        task_set.add(task)
    return task
