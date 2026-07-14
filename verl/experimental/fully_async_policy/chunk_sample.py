# Copyright 2026 The opdflow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Stage-1 (Xiaoshuai) chunk-level rollout unit.

This module defines `ChunkSample`, the trainer-visible unit for chunk-level
async OPD. A `ChunkSample` represents a contiguous slice of response tokens
from an in-flight (or completed) rollout, tagged with the rollouter policy
version at emission time so the trainer can enforce a staleness bound.

The dataclass is intentionally minimal in this first pass: it carries only
what teacher scoring and the per-token distillation loss require, plus the
provenance fields used by the staleness gate and the analyzer.

The rollouter publishes these objects through the fully-async message queue
when chunked training is enabled. The trainer consumes them, applies a
staleness gate, pads compatible chunk batches, and runs the usual OPD loss
with `response_mask` narrowed to the chunk's trainable response positions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ChunkSample:
    """One trainer-visible chunk of an in-flight rollout.

    Fields
    ------
    sample_id:
        Parent rollout sample identifier; matches the Stage-0 `sample_id`
        used by `gen_start` / `gen_end` / `put` / `get` events so chunk
        events join cleanly with sample events in the analyzer.
    chunk_idx:
        0-based chunk index within the parent sample. Chunk 0 is the
        first emitted slice; the chunk with `is_final=True` contains EOS.
    token_offset:
        Offset (in response tokens) of this chunk's first token within
        the parent sample's response.
    n_tokens:
        Number of response tokens carried by this chunk (== `len(tokens)`).
    tokens:
        Response token ids for this chunk. The teacher receives
        `(prompt + previously_emitted_response + tokens)` and scores
        the slice corresponding to `tokens`.
    is_final:
        True when this chunk is the last one of its sample (EOS or
        length cap reached). The trainer uses this to retire per-sample
        bookkeeping.
    policy_version:
        Rollouter policy version at chunk emission time. The trainer
        drops chunks with `current_version - policy_version > sigma`.
    parent_payload:
        Opaque handle that teacher + trainer need to score and supervise this
        chunk. In the first real data-path wiring this is a chunk-shaped
        `DataProto`: prompt plus response prefix through this chunk, with
        `response_mask` non-zero only for the chunk's trainable tokens.
    meta:
        Free-form metadata (e.g. epoch, request_id) for diagnostics.
    """

    sample_id: str
    chunk_idx: int
    token_offset: int
    n_tokens: int
    tokens: Any  # token id list / tensor; concrete type TBD when plumbing
    is_final: bool
    policy_version: int
    parent_payload: Any = None
    meta: dict[str, Any] = field(default_factory=dict)
    # H-ACC-SPAN (span-only payload mode): teacher top-k labels for ONLY this chunk's new span
    # [token_offset : token_offset+n_tokens], shape [n_tokens, k]. When set, non-final chunks carry
    # parent_payload=None (the full-prefix labels are NOT serialized onto the queue); the final chunk
    # keeps a structural parent_payload with its full-width teacher tensors stripped, and the trainer
    # writes the stitched labels back in. None = legacy full-payload mode.
    span_teacher_ids: Any = None
    span_teacher_logprobs: Any = None

    def is_stale(self, current_version: int, sigma: int | float) -> bool:
        """Return True iff this chunk should be dropped by the staleness gate."""
        return (current_version - self.policy_version) > max(0.0, float(sigma))
