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


def _positive_int_env(name: str) -> int | None:
    value = os.environ.get(name, "").strip()
    if not value:
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def get_chunk_batch_memory_limits_from_env() -> dict[str, int | None]:
    """Return optional Xiaoshuai trainer batch memory limits from env knobs."""
    return {
        "max_chunk_rows": _positive_int_env("OPD_STAGE1_MAX_CHUNK_ROWS"),
        "max_train_tokens": _positive_int_env("OPD_STAGE1_MAX_TRAIN_TOKENS"),
        "max_effective_seq_len": _positive_int_env("OPD_STAGE1_MAX_EFFECTIVE_SEQ_LEN"),
    }


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

    max_response_width = max(int(chunk.parent_payload.batch["responses"].shape[1]) for chunk in chunk_samples)
    pad_token_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
    chunk_batches = []
    for chunk in chunk_samples:
        padded = pad_chunk_dataproto_to_response_width(chunk.parent_payload, max_response_width, pad_token_id)
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
    chunk_token_counts = [chunk.n_tokens for chunk in chunk_samples]
    chunk_stats = {
        "fully_async/chunk/enabled": 1,
        "fully_async/chunk/count": len(chunk_samples),
        "fully_async/chunk/total_tokens": int(sum(chunk_token_counts)),
        "fully_async/chunk/avg_tokens": float(np.mean(chunk_token_counts)),
        "fully_async/chunk/max_response_width": int(max_response_width),
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
                    "source": chunk.meta.get("source", "unknown"),
                    "chunk_idx": chunk.chunk_idx,
                    "n_tokens": chunk.n_tokens,
                    "token_offset": chunk.token_offset,
                    "policy_version": chunk.policy_version,
                    "is_final": chunk.is_final,
                }
                for chunk in chunk_samples
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

    rollout_samples_batch = []
    rollout_status = rollout_samples[0].rollout_status
    # Add a prefix to all rollout_status keys
    rollout_status = {f"fully_async/{key}": value for key, value in rollout_status.items()}

    for rs in rollout_samples:
        batch = addition_process(rs.full_batch)
        rollout_samples_batch.append(batch)
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
