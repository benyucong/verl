# Copyright 2026 The opdflow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl.experimental.agent_loop.single_turn_agent_loop import _should_continue_chunked_generation
from verl.experimental.fully_async_policy.chunk_sample import ChunkSample
from verl.experimental.fully_async_policy.detach_utils import (
    RolloutSample,
    assemble_batch_from_chunk_samples,
    choose_chunk_actor_mini_batch_size,
    create_chunk_samples_from_rollout_sample,
    estimate_chunk_effective_seq_len,
    get_chunk_token_budget,
    select_chunk_samples_for_train_batch,
    split_chunk_samples_for_balanced_batch,
)
from verl.protocol import DataProto
from verl.workers.rollout.replica import TokenOutput


class _Tokenizer:
    pad_token_id = 0


def _config():
    return OmegaConf.create(
        {
            "async_training": {
                "require_batches": 1,
                "chunk_tokens": 3,
                "chunk_token_budget": None,
                "staleness_threshold": 1,
            },
            "actor_rollout_ref": {
                "actor": {"ppo_mini_batch_size": 2},
                "rollout": {"n": 1},
            },
            "trainer": {"balance_batch": False},
        }
    )


def _rollout_sample():
    batch_size = 2
    prompt_width = 3
    response_width = 8
    topk = 2
    total_width = prompt_width + response_width

    prompts = torch.tensor([[11, 12, 13], [21, 22, 23]], dtype=torch.long)
    responses = torch.tensor(
        [
            [31, 32, 33, 34, 35, 0, 0, 0],
            [41, 42, 43, 44, 45, 46, 47, 0],
        ],
        dtype=torch.long,
    )
    response_attention = torch.zeros(batch_size, response_width, dtype=torch.long)
    response_attention[0, :5] = 1
    response_attention[1, :7] = 1
    response_mask = response_attention.clone()
    attention_mask = torch.cat([torch.ones(batch_size, prompt_width, dtype=torch.long), response_attention], dim=1)
    input_ids = torch.cat([prompts, responses], dim=1)
    position_ids = torch.arange(total_width, dtype=torch.long).repeat(batch_size, 1)
    rollout_log_probs = torch.zeros(batch_size, response_width, dtype=torch.float32)
    teacher_ids = torch.arange(batch_size * total_width * topk, dtype=torch.int32).view(batch_size, total_width, topk)
    teacher_logprobs = torch.zeros(batch_size, total_width, topk, dtype=torch.float32)

    data = DataProto(
        batch=TensorDict(
            {
                "prompts": prompts,
                "responses": responses,
                "response_mask": response_mask,
                "attention_mask": attention_mask,
                "input_ids": input_ids,
                "position_ids": position_ids,
                "rollout_log_probs": rollout_log_probs,
                "teacher_ids": teacher_ids,
                "teacher_logprobs": teacher_logprobs,
            },
            batch_size=batch_size,
        ),
        non_tensor_batch={
            "min_global_steps": np.array([2, 2], dtype=object),
            "max_global_steps": np.array([2, 2], dtype=object),
            "extras": np.array([None, None], dtype=object),
            "turn_scores": np.array([[], []], dtype=object),
            "tool_rewards": np.array([[], []], dtype=object),
        },
        meta_info={
            "metrics": [
                {"generate_sequences": 1.0, "tool_calls": 0.0},
                {"generate_sequences": 2.0, "tool_calls": 0.0},
            ]
        },
    )
    return RolloutSample(full_batch=data, sample_id="sample_0_1", epoch=0, rollout_status={"count/foo": 3})


def _chunk_with_effective_len(idx: int, effective_len: int, row_count: int = 1) -> ChunkSample:
    payload = DataProto(
        batch=TensorDict(
            {"attention_mask": torch.ones(row_count, effective_len, dtype=torch.long)},
            batch_size=row_count,
        ),
        non_tensor_batch={},
        meta_info={},
    )
    return ChunkSample(
        sample_id=f"s{idx}",
        chunk_idx=idx,
        token_offset=idx * 256,
        n_tokens=256,
        tokens=[idx],
        is_final=False,
        policy_version=0,
        parent_payload=payload,
        meta={"row_id": f"s{idx}", "response_end": (idx + 1) * 256},
    )


def test_create_chunk_samples_masks_only_chunk_tokens():
    chunks = create_chunk_samples_from_rollout_sample(_rollout_sample(), chunk_tokens=3, policy_version=2)

    assert [chunk.chunk_idx for chunk in chunks] == [0, 1, 2]
    assert [chunk.token_offset for chunk in chunks] == [0, 3, 6]
    assert [chunk.n_tokens for chunk in chunks] == [6, 5, 1]
    assert chunks[-1].is_final

    first = chunks[0].parent_payload
    assert first.batch["responses"].shape == (2, 3)
    assert first.batch["input_ids"].shape == (2, 6)
    assert first.batch["teacher_ids"].shape == (2, 6, 2)
    assert int(first.batch["response_mask"].sum().item()) == 6

    second = chunks[1].parent_payload
    assert second.batch["responses"].shape == (2, 6)
    assert second.batch["input_ids"].shape == (2, 9)
    assert int(second.batch["response_mask"][:, :3].sum().item()) == 0
    assert int(second.batch["response_mask"][:, 3:].sum().item()) == 5


def test_assemble_batch_from_chunk_samples_pads_mixed_widths():
    chunks = create_chunk_samples_from_rollout_sample(_rollout_sample(), chunk_tokens=3, policy_version=2)

    batch = assemble_batch_from_chunk_samples(chunks[:2], _Tokenizer(), _config(), balance_batch=None)

    assert batch.batch["responses"].shape == (4, 6)
    assert batch.batch["input_ids"].shape == (4, 9)
    assert batch.batch["teacher_logprobs"].shape == (4, 9, 2)
    assert batch.non_tensor_batch["chunk_idx"].tolist() == [0, 0, 1, 1]
    assert int(batch.batch["response_mask"][:2, 3:].sum().item()) == 0
    assert int(batch.batch["response_mask"][2:, :3].sum().item()) == 0
    assert int(batch.batch["response_mask"].sum().item()) == 11
    assert batch.meta_info["fully_async/chunk/count"] == 2
    assert batch.meta_info["fully_async/chunk/total_tokens"] == 11


def test_chunk_sample_staleness_allows_float_sigma():
    chunk = ChunkSample(
        sample_id="s0",
        chunk_idx=0,
        token_offset=0,
        n_tokens=1,
        tokens=[1],
        is_final=False,
        policy_version=3,
    )

    assert not chunk.is_stale(current_version=4, sigma=1.0)
    assert chunk.is_stale(current_version=4, sigma=0.5)


def test_chunk_token_budget_accounts_for_rollout_responses():
    config = _config()
    config.actor_rollout_ref.actor.ppo_mini_batch_size = 16
    config.actor_rollout_ref.rollout.n = 4
    config.async_training.chunk_tokens = 256

    assert get_chunk_token_budget(config) == 16 * 4 * 256


def test_split_chunk_samples_for_balanced_batch_defers_fifo_suffix():
    chunks = [_chunk_with_effective_len(idx, 100 + idx) for idx in range(3)]

    train_chunks, deferred_chunks = split_chunk_samples_for_balanced_batch(chunks, batch_divisor=2, min_chunks=2)

    assert [chunk.chunk_idx for chunk in train_chunks] == [0, 1]
    assert [chunk.chunk_idx for chunk in deferred_chunks] == [2]


def test_split_chunk_samples_for_balanced_batch_keeps_all_without_divisor():
    chunks = [_chunk_with_effective_len(idx, 100 + idx) for idx in range(3)]

    train_chunks, deferred_chunks = split_chunk_samples_for_balanced_batch(chunks, batch_divisor=1, min_chunks=2)

    assert train_chunks == chunks
    assert deferred_chunks == []


def test_select_chunk_samples_for_train_batch_respects_token_budget_and_fifo():
    chunks = [_chunk_with_effective_len(idx, effective_len) for idx, effective_len in enumerate([100, 150, 200, 250])]

    selection = select_chunk_samples_for_train_batch(
        chunks,
        batch_divisor=2,
        min_rows=2,
        max_train_tokens=500,
    )

    assert [chunk.chunk_idx for chunk in selection.train_chunks] == [0, 1]
    assert [chunk.chunk_idx for chunk in selection.deferred_chunks] == [2, 3]
    assert selection.usable_rows == 2
    assert selection.deferred_rows == 2
    assert selection.estimated_train_tokens == 250
    assert selection.max_effective_seq_len == 150
    assert selection.trimmed_by_dp_divisibility
    assert selection.trimmed_by_memory_budget


def test_select_chunk_samples_for_train_batch_respects_row_budget():
    chunks = [_chunk_with_effective_len(idx, 100 + idx) for idx in range(6)]

    selection = select_chunk_samples_for_train_batch(
        chunks,
        batch_divisor=2,
        min_rows=2,
        max_chunk_rows=4,
    )

    assert [chunk.chunk_idx for chunk in selection.train_chunks] == [0, 1, 2, 3]
    assert [chunk.chunk_idx for chunk in selection.deferred_chunks] == [4, 5]
    assert selection.usable_rows == 4
    assert selection.deferred_rows == 2
    assert selection.trimmed_by_memory_budget


def test_select_chunk_samples_for_train_batch_respects_actor_row_divisor():
    chunks = [_chunk_with_effective_len(idx, 100 + idx) for idx in range(65)]

    selection = select_chunk_samples_for_train_batch(
        chunks,
        batch_divisor=8,
        train_row_divisor=32,
        min_rows=32,
        max_chunk_rows=65,
    )

    assert len(selection.train_chunks) == 64
    assert len(selection.deferred_chunks) == 1
    assert selection.world_size == 8
    assert selection.train_row_divisor == 32
    assert selection.usable_rows == 64
    assert selection.deferred_rows == 1
    assert selection.trimmed_by_dp_divisibility


def test_choose_chunk_actor_mini_batch_size_prefers_configured_when_valid():
    assert choose_chunk_actor_mini_batch_size(
        selected_rows=64,
        world_size=8,
        configured_actor_mini_batch_size=32,
    ) == 32


def test_choose_chunk_actor_mini_batch_size_falls_back_to_local_divisor():
    assert choose_chunk_actor_mini_batch_size(
        selected_rows=56,
        world_size=8,
        configured_actor_mini_batch_size=32,
    ) == 8


def test_choose_chunk_actor_mini_batch_size_uses_full_small_batch():
    assert choose_chunk_actor_mini_batch_size(
        selected_rows=24,
        world_size=8,
        configured_actor_mini_batch_size=32,
    ) == 24


def test_select_chunk_samples_for_train_batch_respects_effective_length_budget():
    chunks = [_chunk_with_effective_len(idx, effective_len) for idx, effective_len in enumerate([100, 120, 400, 130])]

    selection = select_chunk_samples_for_train_batch(
        chunks,
        batch_divisor=2,
        min_rows=2,
        max_effective_seq_len=200,
    )

    assert [chunk.chunk_idx for chunk in selection.train_chunks] == [0, 1]
    assert [chunk.chunk_idx for chunk in selection.deferred_chunks] == [2, 3]
    assert selection.max_effective_seq_len == 120
    assert selection.trimmed_by_memory_budget


def test_select_chunk_samples_for_train_batch_defers_without_dropping_or_reordering():
    chunks = [_chunk_with_effective_len(idx, effective_len) for idx, effective_len in enumerate([100, 150, 200, 250, 300])]
    chunks[0].policy_version = -10

    selection = select_chunk_samples_for_train_batch(
        chunks,
        batch_divisor=2,
        min_rows=2,
        max_train_tokens=500,
    )
    selected_and_deferred = selection.train_chunks + selection.deferred_chunks

    assert selected_and_deferred == chunks
    assert len({id(chunk) for chunk in selected_and_deferred}) == len(chunks)
    assert chunks[0].is_stale(current_version=0, sigma=1)
    assert estimate_chunk_effective_seq_len(chunks[-1]) == 300


def test_chunked_single_turn_resumes_only_on_length_finish_reason():
    length_output = TokenOutput(
        token_ids=[1, 2, 3],
        stop_reason="completed",
        extra_fields={"finish_reason": "length"},
    )
    stop_output = TokenOutput(
        token_ids=[1, 2, 3],
        stop_reason="completed",
        extra_fields={"finish_reason": "stop"},
    )

    assert _should_continue_chunked_generation(
        length_output,
        chunk_limit=3,
        total_response_tokens=3,
        response_length=9,
    )
    assert not _should_continue_chunked_generation(
        stop_output,
        chunk_limit=3,
        total_response_tokens=3,
        response_length=9,
    )
