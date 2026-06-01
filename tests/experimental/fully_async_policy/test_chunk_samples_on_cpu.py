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
    ChunkCoalescingConfig,
    RolloutSample,
    assemble_batch_from_chunk_samples,
    choose_chunk_actor_mini_batch_size,
    coalesce_contiguous_chunk_samples,
    create_chunk_samples_from_rollout_sample,
    estimate_chunk_effective_seq_len,
    flatten_chunk_constituents,
    get_original_chunk_count,
    get_chunk_token_budget,
    select_chunk_samples_for_train_batch,
    split_chunk_samples_for_balanced_batch,
)
from verl.experimental.fully_async_policy.detach_utils import (
    get_chunk_coalescing_drain_multiplier,
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


def _streaming_chunk(
    parent: str,
    idx: int,
    start: int,
    end: int,
    *,
    policy_version: int = 0,
    source: str = "streaming",
) -> ChunkSample:
    prompt_width = 2
    topk = 2
    responses = torch.arange(100, 100 + end, dtype=torch.long).view(1, end)
    response_mask = torch.zeros(1, end, dtype=torch.long)
    response_mask[:, start:end] = 1
    prompts = torch.tensor([[11, 12]], dtype=torch.long)
    input_ids = torch.cat([prompts, responses], dim=1)
    attention_mask = torch.ones(1, prompt_width + end, dtype=torch.long)
    position_ids = torch.arange(prompt_width + end, dtype=torch.long).view(1, prompt_width + end)
    teacher_ids = torch.full((1, prompt_width + end, topk), idx, dtype=torch.int32)
    teacher_logprobs = torch.full((1, prompt_width + end, topk), float(idx), dtype=torch.float32)
    rollout_log_probs = torch.full((1, end), float(idx), dtype=torch.float32)
    payload = DataProto(
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
            batch_size=1,
        ),
        non_tensor_batch={
            "uid": np.array([f"uid_{parent}"], dtype=object),
            "min_global_steps": np.array([policy_version], dtype=object),
            "max_global_steps": np.array([policy_version], dtype=object),
            "extras": np.array([None], dtype=object),
            "turn_scores": np.array([[]], dtype=object),
            "tool_rewards": np.array([[]], dtype=object),
            "chunk_sample_id": np.array([parent], dtype=object),
            "chunk_parent_sample_id": np.array([parent], dtype=object),
            "chunk_idx": np.array([idx], dtype=np.int32),
            "chunk_token_offset": np.array([start], dtype=np.int32),
            "chunk_n_tokens": np.array([end - start], dtype=np.int32),
            "chunk_is_final": np.array([False], dtype=bool),
            "chunk_policy_version": np.array([policy_version], dtype=np.int32),
        },
        meta_info={"metrics": [{"generate_sequences": 1.0, "tool_calls": 0.0}]},
    )
    return ChunkSample(
        sample_id=parent,
        chunk_idx=idx,
        token_offset=start,
        n_tokens=end - start,
        tokens=list(range(start, end)),
        is_final=False,
        policy_version=policy_version,
        parent_payload=payload,
        meta={
            "row_id": parent,
            "parent_sample_id": parent,
            "chunk_id": f"{parent}:{idx}",
            "source": source,
            "response_end": end,
            "response_width": 16,
        },
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


def test_coalesce_contiguous_streaming_chunks_unions_masks_and_preserves_fifo():
    chunks = [
        _streaming_chunk("parent", 0, 0, 3),
        _streaming_chunk("parent", 1, 3, 6),
        _streaming_chunk("parent", 2, 6, 9),
    ]

    result = coalesce_contiguous_chunk_samples(
        chunks,
        ChunkCoalescingConfig(
            enabled=True,
            max_coalesced_chunks=2,
            max_coalesced_effective_seq_len=32,
        ),
    )

    assert len(result.chunks) == 2
    merged, tail = result.chunks
    assert get_original_chunk_count(merged) == 2
    assert tail is chunks[2]
    assert merged.chunk_idx == 0
    assert merged.n_tokens == 6
    assert merged.meta["merged_chunk_ids"] == ["parent:0", "parent:1"]
    assert int(merged.parent_payload.batch["response_mask"].sum().item()) == 6
    assert merged.parent_payload.batch["response_mask"].tolist() == [[1, 1, 1, 1, 1, 1]]
    assert result.metrics["coalesced_groups"] == 1
    assert result.metrics["chunks_merged_total"] == 2
    assert result.metrics["rows_after_coalesce"] == 2
    assert result.metrics["estimated_prefix_recompute_reduction"] > 0

    batch = assemble_batch_from_chunk_samples(result.chunks, _Tokenizer(), _config(), balance_batch=None)
    assert batch.meta_info["fully_async/chunk/count"] == 3
    assert batch.meta_info["fully_async/chunk/train_rows"] == 2
    assert len(batch.meta_info["chunk_samples"]) == 3
    assert int(batch.batch["response_mask"].sum().item()) == 9


def test_coalesce_contiguous_streaming_chunks_does_not_cross_interleaved_fifo_parent():
    chunks = [
        _streaming_chunk("a", 0, 0, 3),
        _streaming_chunk("b", 0, 0, 3),
        _streaming_chunk("a", 1, 3, 6),
    ]

    result = coalesce_contiguous_chunk_samples(
        chunks,
        ChunkCoalescingConfig(enabled=True, max_coalesced_chunks=4, max_coalesced_effective_seq_len=32),
    )

    assert result.chunks == chunks
    assert result.metrics["coalesced_groups"] == 0


def test_coalesce_contiguous_streaming_chunks_requires_matching_policy_version():
    chunks = [
        _streaming_chunk("parent", 0, 0, 3, policy_version=1),
        _streaming_chunk("parent", 1, 3, 6, policy_version=2),
    ]

    result = coalesce_contiguous_chunk_samples(
        chunks,
        ChunkCoalescingConfig(enabled=True, max_coalesced_chunks=4, max_coalesced_effective_seq_len=32),
    )

    assert result.chunks == chunks
    assert result.metrics["coalesced_groups"] == 0


def test_coalesce_contiguous_streaming_chunks_respects_effective_length_cap():
    chunks = [
        _streaming_chunk("parent", 0, 0, 3),
        _streaming_chunk("parent", 1, 3, 6),
    ]

    result = coalesce_contiguous_chunk_samples(
        chunks,
        ChunkCoalescingConfig(enabled=True, max_coalesced_chunks=4, max_coalesced_effective_seq_len=5),
    )

    assert result.chunks == chunks
    assert result.metrics["coalesced_groups"] == 0


def test_coalesce_contiguous_streaming_chunks_leaves_fallback_unchanged():
    chunks = [
        _streaming_chunk("parent", 0, 0, 3, source="fallback"),
        _streaming_chunk("parent", 1, 3, 6, source="fallback"),
    ]

    result = coalesce_contiguous_chunk_samples(
        chunks,
        ChunkCoalescingConfig(enabled=True, max_coalesced_chunks=4, max_coalesced_effective_seq_len=32),
    )

    assert result.chunks == chunks
    assert result.metrics["coalesced_groups"] == 0


def _interleaved_three_parent_queue():
    # Globally interleaved FIFO order across three concurrent rollouts. Same-parent
    # contiguous chunks are never directly adjacent in the queue.
    return [
        _streaming_chunk("a", 0, 0, 3),  # 0
        _streaming_chunk("b", 0, 0, 3),  # 1
        _streaming_chunk("c", 0, 0, 3),  # 2
        _streaming_chunk("a", 1, 3, 6),  # 3
        _streaming_chunk("b", 1, 3, 6),  # 4
        _streaming_chunk("a", 2, 6, 9),  # 5
    ]


def test_lookahead_coalesces_interleaved_same_parent_and_preserves_fifo():
    chunks = _interleaved_three_parent_queue()

    result = coalesce_contiguous_chunk_samples(
        chunks,
        ChunkCoalescingConfig(
            enabled=True,
            max_coalesced_chunks=4,
            max_coalesced_effective_seq_len=64,
            lookahead=128,
        ),
    )

    merged = result.chunks
    assert len(merged) == 3
    # Merged A occupies the earliest FIFO slot, then B, then C.
    assert get_original_chunk_count(merged[0]) == 3
    assert merged[0].meta["merged_chunk_ids"] == ["a:0", "a:1", "a:2"]
    assert get_original_chunk_count(merged[1]) == 2
    assert merged[1].meta["merged_chunk_ids"] == ["b:0", "b:1"]
    assert merged[2] is chunks[2]  # C:c0 unchanged

    assert result.metrics["coalesced_groups"] == 2
    assert result.metrics["chunks_merged"] == 5
    assert result.metrics["rows_before_coalesce"] == 6
    assert result.metrics["rows_after_coalesce"] == 3
    assert result.metrics["coalescing_opportunities_visible_in_window"] == 3
    assert result.metrics["estimated_prefix_recompute_reduction"] > 0
    assert result.metrics["coalesce_lookahead"] == 128

    # Merged A response mask covers all three chunk spans exactly once.
    assert int(merged[0].parent_payload.batch["response_mask"].sum().item()) == 9


def test_lookahead_too_small_does_not_merge_interleaved_parents():
    chunks = _interleaved_three_parent_queue()

    result = coalesce_contiguous_chunk_samples(
        chunks,
        ChunkCoalescingConfig(
            enabled=True,
            max_coalesced_chunks=4,
            max_coalesced_effective_seq_len=64,
            lookahead=1,
        ),
    )

    assert result.chunks == chunks
    assert result.metrics["coalesced_groups"] == 0
    assert result.metrics["chunks_merged"] == 0
    # The same-parent successors exist but are outside the 1-chunk window.
    assert result.metrics["merge_reject_outside_lookahead"] > 0


def test_lookahead_preserves_every_original_chunk_exactly_once():
    chunks = _interleaved_three_parent_queue()

    result = coalesce_contiguous_chunk_samples(
        chunks,
        ChunkCoalescingConfig(
            enabled=True,
            max_coalesced_chunks=4,
            max_coalesced_effective_seq_len=64,
            lookahead=128,
        ),
    )

    originals = flatten_chunk_constituents(result.chunks)
    assert {id(chunk) for chunk in originals} == {id(chunk) for chunk in chunks}
    assert len(originals) == len(chunks)


def test_lookahead_does_not_merge_different_policy_versions():
    chunks = [
        _streaming_chunk("a", 0, 0, 3, policy_version=1),
        _streaming_chunk("b", 0, 0, 3, policy_version=1),
        _streaming_chunk("a", 1, 3, 6, policy_version=2),
    ]

    result = coalesce_contiguous_chunk_samples(
        chunks,
        ChunkCoalescingConfig(
            enabled=True,
            max_coalesced_chunks=4,
            max_coalesced_effective_seq_len=64,
            lookahead=128,
        ),
    )

    assert result.chunks == chunks
    assert result.metrics["coalesced_groups"] == 0
    assert result.metrics["merge_reject_policy_version"] == 1


def test_lookahead_respects_effective_length_cap():
    chunks = [
        _streaming_chunk("a", 0, 0, 3),
        _streaming_chunk("b", 0, 0, 3),
        _streaming_chunk("a", 1, 3, 6),
    ]

    result = coalesce_contiguous_chunk_samples(
        chunks,
        ChunkCoalescingConfig(
            enabled=True,
            max_coalesced_chunks=4,
            max_coalesced_effective_seq_len=5,
            lookahead=128,
        ),
    )

    assert result.chunks == chunks
    assert result.metrics["coalesced_groups"] == 0
    assert result.metrics["merge_reject_effective_len_cap"] == 1


def test_lookahead_respects_max_coalesced_chunks():
    chunks = _interleaved_three_parent_queue()

    result = coalesce_contiguous_chunk_samples(
        chunks,
        ChunkCoalescingConfig(
            enabled=True,
            max_coalesced_chunks=2,
            max_coalesced_effective_seq_len=64,
            lookahead=128,
        ),
    )

    # A merges only its first two chunks; A:c2 stays separate.
    assert get_original_chunk_count(result.chunks[0]) == 2
    assert result.chunks[0].meta["merged_chunk_ids"] == ["a:0", "a:1"]
    originals = flatten_chunk_constituents(result.chunks)
    assert len(originals) == len(chunks)


def test_lookahead_leaves_fallback_chunks_unmerged():
    chunks = [
        _streaming_chunk("a", 0, 0, 3, source="fallback"),
        _streaming_chunk("b", 0, 0, 3),
        _streaming_chunk("a", 1, 3, 6, source="fallback"),
    ]

    result = coalesce_contiguous_chunk_samples(
        chunks,
        ChunkCoalescingConfig(
            enabled=True,
            max_coalesced_chunks=4,
            max_coalesced_effective_seq_len=64,
            lookahead=128,
        ),
    )

    assert result.chunks == chunks
    assert result.metrics["coalesced_groups"] == 0


def test_lookahead_disabled_preserves_input_and_reports_metrics():
    chunks = _interleaved_three_parent_queue()

    result = coalesce_contiguous_chunk_samples(
        chunks,
        ChunkCoalescingConfig(
            enabled=False,
            max_coalesced_chunks=4,
            max_coalesced_effective_seq_len=64,
            lookahead=128,
        ),
    )

    assert result.chunks is chunks
    assert result.metrics["coalesced_groups"] == 0
    assert result.metrics["rows_after_coalesce"] == 6


def test_lookahead_merged_rows_obey_token_budget_and_defer_original_chunks():
    chunks = _interleaved_three_parent_queue()

    result = coalesce_contiguous_chunk_samples(
        chunks,
        ChunkCoalescingConfig(
            enabled=True,
            max_coalesced_chunks=4,
            max_coalesced_effective_seq_len=64,
            lookahead=128,
        ),
    )
    # Merged rows still flow through the memory-safe FIFO selector without
    # reordering or dropping, and any deferred suffix flattens to original chunks.
    selection = select_chunk_samples_for_train_batch(
        result.chunks,
        batch_divisor=2,
        min_rows=2,
        max_chunk_rows=2,
    )
    kept_and_deferred = flatten_chunk_constituents(selection.train_chunks) + flatten_chunk_constituents(
        selection.deferred_chunks
    )
    assert len(kept_and_deferred) == len(chunks)
    assert {id(chunk) for chunk in kept_and_deferred} == {id(chunk) for chunk in chunks}


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


def _drain_config(multiplier=None):
    async_training = {}
    if multiplier is not None:
        async_training["coalesce_drain_multiplier"] = multiplier
    return OmegaConf.create({"async_training": async_training})


def test_get_chunk_coalescing_drain_multiplier_defaults_to_one(monkeypatch):
    monkeypatch.delenv("OPD_STAGE1_COALESCE_DRAIN_MULTIPLIER", raising=False)
    assert get_chunk_coalescing_drain_multiplier(_drain_config()) == 1.0


def test_get_chunk_coalescing_drain_multiplier_reads_config(monkeypatch):
    monkeypatch.delenv("OPD_STAGE1_COALESCE_DRAIN_MULTIPLIER", raising=False)
    assert get_chunk_coalescing_drain_multiplier(_drain_config(2.5)) == 2.5


def test_get_chunk_coalescing_drain_multiplier_env_overrides_config(monkeypatch):
    monkeypatch.setenv("OPD_STAGE1_COALESCE_DRAIN_MULTIPLIER", "3")
    assert get_chunk_coalescing_drain_multiplier(_drain_config(2.0)) == 3.0


def test_get_chunk_coalescing_drain_multiplier_floors_below_one(monkeypatch):
    # Values <= 1 (or invalid) must collapse to 1.0 so the legacy cut-at-budget
    # behavior is preserved and the drain phase stays disabled.
    monkeypatch.setenv("OPD_STAGE1_COALESCE_DRAIN_MULTIPLIER", "0.5")
    assert get_chunk_coalescing_drain_multiplier(_drain_config(2.0)) == 1.0
    monkeypatch.setenv("OPD_STAGE1_COALESCE_DRAIN_MULTIPLIER", "not-a-number")
    assert get_chunk_coalescing_drain_multiplier(_drain_config(2.0)) == 2.0
    monkeypatch.delenv("OPD_STAGE1_COALESCE_DRAIN_MULTIPLIER", raising=False)
    assert get_chunk_coalescing_drain_multiplier(_drain_config(0.25)) == 1.0


def _build_interleaved_pool(seed: int, num_parents: int, chunks_per_parent: int):
    import random

    rng = random.Random(seed)
    # Build each parent's chunks in contiguous chunk_idx order, then interleave
    # the per-parent streams while preserving each parent's relative order (the
    # realistic FIFO arrival pattern under async rollout).
    streams = []
    for p in range(num_parents):
        parent = f"p{p}"
        length = rng.randint(1, chunks_per_parent)
        streams.append(
            [_streaming_chunk(parent, idx, idx * 3, (idx + 1) * 3) for idx in range(length)]
        )
    pool = []
    cursors = [0] * num_parents
    remaining = sum(len(s) for s in streams)
    while remaining:
        p = rng.randrange(num_parents)
        if cursors[p] < len(streams[p]):
            pool.append(streams[p][cursors[p]])
            cursors[p] += 1
            remaining -= 1
    return pool


def _coalesced_signature(result):
    chunks_sig = [
        (
            chunk.sample_id,
            int(chunk.chunk_idx),
            int(chunk.n_tokens),
            get_original_chunk_count(chunk),
            tuple((chunk.meta or {}).get("merged_chunk_ids", ())),
        )
        for chunk in result.chunks
    ]
    metric_keys = [
        "coalesced_groups",
        "chunks_merged_total",
        "rows_after_coalesce",
        "coalescing_opportunities_visible_in_window",
        "estimated_prefix_recompute_reduction",
        "merge_reject_different_parent",
        "merge_reject_noncontiguous_span",
        "merge_reject_policy_version",
        "merge_reject_effective_len_cap",
        "merge_reject_missing_teacher_payload",
        "merge_reject_fallback",
        "merge_reject_outside_lookahead",
    ]
    metrics_sig = {key: result.metrics[key] for key in metric_keys}
    return chunks_sig, metrics_sig


def test_indexed_coalescer_matches_scan_path_on_large_pools(monkeypatch):
    # The hash-indexed fast path (used at/above the pool-size threshold) must
    # produce byte-identical merges, ordering, and metrics as the reference
    # linear scan across a range of lookahead/group/cap settings.
    for seed in range(12):
        pool = _build_interleaved_pool(seed, num_parents=6, chunks_per_parent=5)
        for lookahead, max_chunks, cap in [
            (0, 4, 10_000),
            (2, 3, 10_000),
            (8, 4, 10_000),
            (4, 2, 8),
            (16, None, 10_000),
        ]:
            config = ChunkCoalescingConfig(
                enabled=True,
                max_coalesced_chunks=max_chunks,
                max_coalesced_effective_seq_len=cap,
                lookahead=lookahead,
            )

            # Force the reference scan path.
            monkeypatch.setenv("OPD_STAGE1_COALESCE_INDEX_THRESHOLD", "1000000")
            scan_result = coalesce_contiguous_chunk_samples(list(pool), config)

            # Force the indexed fast path (threshold below the pool size).
            monkeypatch.setenv("OPD_STAGE1_COALESCE_INDEX_THRESHOLD", "2")
            indexed_result = coalesce_contiguous_chunk_samples(list(pool), config)

            assert _coalesced_signature(scan_result) == _coalesced_signature(indexed_result), (
                f"mismatch at seed={seed}, lookahead={lookahead}, max_chunks={max_chunks}, cap={cap}"
            )


