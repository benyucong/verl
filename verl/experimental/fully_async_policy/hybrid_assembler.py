# Hybrid pipeline (chunk-scored teacher labels -> full-sample actor training): per-parent
# label assembler. Teacher targets are collected incrementally per chunk (bounded memory,
# overlappable), buffered per parent, and stitched into ONE ordinary completed-sample OPD
# example when the response finishes. The actor then trains on the upstream completed-sample
# path -- big efficient batches, zero born-stale drops -- and cannot tell the labels arrived
# incrementally.
#
# MVP scope (Phase A1): in-order, contiguous spans only; exact [0, L) coverage required;
# any gap/overlap/shape-mismatch is an assertion failure. No retries/duplicates/spilling/
# backpressure yet (partial-rollout abort/resume must be OFF for the MVP). Pure stdlib so it
# is unit-testable without torch; a torch tensor-builder lives in build_sample_tensors().
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


def hybrid_span_payload_enabled() -> bool:
    """H-ACC-SPAN gate: emit span-only teacher labels (no full-prefix labels on the wire).

    Requires the hybrid full-sample trainer (OPD_HYBRID_FULL_SAMPLE) to also be on: the legacy
    chunk-training path needs full-width teacher tensors in every payload, so span-only payloads
    are only valid when the trainer stitches spans per parent. Default OFF.
    """
    span = os.environ.get("OPD_HYBRID_SPAN_PAYLOAD", "0") not in ("0", "", "false", "False")
    full = os.environ.get("OPD_HYBRID_FULL_SAMPLE", "0") not in ("0", "", "false", "False")
    return span and full

# Canonical per-sample fields an upstream completed-sample OPD example carries, so the
# reconstructed sample is schema-indistinguishable to the existing actor-training path.
CANONICAL_SAMPLE_KEYS = (
    "prompts",          # (P,) left-padded prompt token ids
    "responses",        # (R,) right-padded response token ids
    "input_ids",        # (P+R,) prompt|response
    "attention_mask",   # (P+R,)
    "position_ids",     # (P+R,)
    "response_mask",    # (R,) 1 over real response tokens
    "teacher_logprobs",  # (R, k) teacher top-k log-probs aligned to response positions
    "teacher_ids",       # (R, k) teacher top-k ids
)


@dataclass
class ChunkLabelSpan:
    """Teacher labels for one contiguous response span [span_start, span_end)."""
    span_start: int
    span_end: int
    response_token_ids: list[int]               # len == span_end - span_start
    teacher_topk_ids: list[list[int]]           # [span_len][k]
    teacher_topk_log_probs: list[list[float]]   # [span_len][k]
    policy_version: int = -1                     # diagnostics only (never an input to the loss)


@dataclass
class ParentLabelAccumulator:
    """Buffers + stitches one in-flight response's chunk-scored teacher labels."""
    parent_id: str
    prompt_token_ids: list[int]
    topk: int
    response_token_ids: list[int] = field(default_factory=list)
    teacher_topk_ids: list[list[int]] = field(default_factory=list)
    teacher_topk_log_probs: list[list[float]] = field(default_factory=list)
    rollout_policy_versions: list[int] = field(default_factory=list)  # per-token, diagnostics
    next_expected_offset: int = 0
    is_final_known: bool = False
    _pending: dict = field(default_factory=dict)  # reorder buffer: span_start -> ChunkLabelSpan
    is_finished: bool = False
    final_length: Optional[int] = None
    finish_reason: Optional[str] = None

    def add_span(self, span: ChunkLabelSpan) -> None:
        if self.is_finished:
            raise AssertionError(f"parent {self.parent_id} already finalized")
        span_len = span.span_end - span.span_start
        if span_len <= 0:
            raise AssertionError(f"parent {self.parent_id}: empty/negative span [{span.span_start},{span.span_end})")
        if len(span.response_token_ids) != span_len:
            raise AssertionError(f"parent {self.parent_id}: token len {len(span.response_token_ids)} != span len {span_len}")
        if len(span.teacher_topk_ids) != span_len or len(span.teacher_topk_log_probs) != span_len:
            raise AssertionError(f"parent {self.parent_id}: teacher label rows != span len {span_len}")
        for ids, lps in zip(span.teacher_topk_ids, span.teacher_topk_log_probs):
            if len(ids) != self.topk or len(lps) != self.topk:
                raise AssertionError(f"parent {self.parent_id}: teacher top-k width != {self.topk}")
        # Reorder-tolerant: streaming chunks are emitted as concurrent asyncio tasks
        # (single_turn_agent_loop.py:402) and teacher-scoring latency reorders them, so a later span can
        # arrive first. Buffer by start offset and stitch the contiguous prefix as gaps fill. A span at/below
        # what we already consumed, or a second span at the same start, is a real overlap/duplicate -> error.
        if span.span_start < self.next_expected_offset:
            raise AssertionError(
                f"parent {self.parent_id}: overlap -- span_start {span.span_start} < already-consumed {self.next_expected_offset}"
            )
        if span.span_start in self._pending:
            raise AssertionError(f"parent {self.parent_id}: duplicate span at start {span.span_start}")
        self._pending[span.span_start] = span
        while self.next_expected_offset in self._pending:
            s = self._pending.pop(self.next_expected_offset)
            self.response_token_ids.extend(s.response_token_ids)
            self.teacher_topk_ids.extend(s.teacher_topk_ids)
            self.teacher_topk_log_probs.extend(s.teacher_topk_log_probs)
            self.rollout_policy_versions.extend([s.policy_version] * (s.span_end - s.span_start))
            self.next_expected_offset = s.span_end

    def mark_final(self, final_length: int, reason: str = "eos") -> None:
        """Record the response's known final length. The is_final chunk may arrive BEFORE earlier chunks
        (concurrent emission), so this only records the target -- emission waits for is_complete."""
        self.is_final_known = True
        self.final_length = final_length
        self.finish_reason = reason

    @property
    def is_complete(self) -> bool:
        """True once the final length is known AND contiguous coverage [0, final_length) has reached it."""
        return self.is_final_known and self.final_length is not None and self.next_expected_offset == self.final_length

    def finalize(self, final_length: int, reason: str = "eos") -> None:
        """Mark the response complete (EOS / max-length / truncation). Requires exact coverage.
        Idempotence guard: a parent finalizes (and is enqueued for the actor) EXACTLY once."""
        if self.is_finished:
            raise AssertionError(f"parent {self.parent_id}: finalize() called twice")
        if self.next_expected_offset != final_length:
            raise AssertionError(
                f"parent {self.parent_id}: coverage gap -- have [0,{self.next_expected_offset}), "
                f"final length {final_length}; refusing to train a partially-labeled sample"
            )
        self.is_finished = True
        self.final_length = final_length
        self.finish_reason = reason

    def assemble(self) -> dict:
        """Return the stitched per-position labels for one completed sample (stdlib types)."""
        if not self.is_finished:
            raise AssertionError(f"parent {self.parent_id}: assemble() before finalize()")
        L = self.final_length
        assert len(self.response_token_ids) == L == len(self.teacher_topk_ids) == len(self.teacher_topk_log_probs)
        return {
            "parent_id": self.parent_id,
            "prompt_token_ids": list(self.prompt_token_ids),
            "response_token_ids": list(self.response_token_ids),
            "teacher_topk_ids": [list(r) for r in self.teacher_topk_ids],
            "teacher_topk_log_probs": [list(r) for r in self.teacher_topk_log_probs],
            "response_mask": [1] * L,
            "finish_reason": self.finish_reason,
            "rollout_policy_versions": list(self.rollout_policy_versions),
        }


def span_from_chunk(chunk) -> ChunkLabelSpan:
    """Build a ChunkLabelSpan from a ChunkSample in EITHER payload format.

    Span-only format (H-ACC-SPAN): the chunk carries `span_teacher_ids`/`span_teacher_logprobs`
    tensors of shape [n_tokens, k] plus `tokens` (the new-span response ids) -- no parent_payload
    needed (non-final chunks ship none). Legacy format: slice the new span out of the full-prefix
    parent_payload (span_from_chunk_payload).
    """
    if getattr(chunk, "span_teacher_ids", None) is not None:
        o = int(chunk.token_offset)
        n = int(chunk.n_tokens)
        return ChunkLabelSpan(
            span_start=o,
            span_end=o + n,
            response_token_ids=[int(t) for t in chunk.tokens],
            teacher_topk_ids=chunk.span_teacher_ids.tolist(),
            teacher_topk_log_probs=chunk.span_teacher_logprobs.tolist(),
            policy_version=int(getattr(chunk, "policy_version", 0) or 0),
        )
    return span_from_chunk_payload(chunk)


def span_from_chunk_payload(chunk, prompt_width=None) -> ChunkLabelSpan:
    """Slice ONE chunk's NEW-span labels [token_offset : token_offset+n_tokens] out of its
    `parent_payload` DataProto into a ChunkLabelSpan (stdlib lists, so the accumulator validates them).

    Layout (verified): teacher tensors in parent_payload are FULL-SEQUENCE aligned, shape
    [1, prompt_width + response_width, k] (left-padded prompt + right-padded response), so response
    token j lives at absolute index prompt_width + j. `responses` is response-only width
    [1, response_width]. prompt_width is constant per parent (= rollout.prompt_length).
    """
    b = chunk.parent_payload.batch
    o = int(chunk.token_offset)
    n = int(chunk.n_tokens)
    P = int(b["prompts"].shape[1]) if prompt_width is None else int(prompt_width)
    return ChunkLabelSpan(
        span_start=o,
        span_end=o + n,
        response_token_ids=b["responses"][0, o:o + n].tolist(),
        teacher_topk_ids=b["teacher_ids"][0, P + o:P + o + n, :].tolist(),
        teacher_topk_log_probs=b["teacher_logprobs"][0, P + o:P + o + n, :].tolist(),
        policy_version=int(getattr(chunk, "policy_version", 0) or 0),
    )


def fill_carrier_teacher_tensors(carrier, assembled: dict) -> None:
    """H-ACC-SPAN: rebuild full-sequence teacher tensors from the accumulated stitch and write them
    INTO the final chunk's structural carrier (whose big full-prefix teacher tensors were stripped
    before serialization). Produces [1, prompt_width + response_width, k] with response token j at
    index prompt_width + j -- the legacy full-prefix layout the actor path + loss expect."""
    import torch

    b = carrier.batch
    P = int(b["prompts"].shape[1])
    RW = int(b["responses"].shape[1])
    ids = assembled["teacher_topk_ids"]
    lps = assembled["teacher_topk_log_probs"]
    L = len(ids)
    k = len(ids[0]) if L else 0
    # Match the native teacher tensor dtype (teacher_manager.py:124 -> int32) so the rebuilt carrier is
    # bit-for-bit schema-identical to the upstream path (no torch.cat/dtype-assert divergence downstream).
    t_ids = torch.zeros(1, P + RW, k, dtype=torch.int32)
    t_lps = torch.zeros(1, P + RW, k, dtype=torch.float32)
    if L:
        t_ids[0, P:P + L] = torch.tensor(ids, dtype=torch.int32)
        t_lps[0, P:P + L] = torch.tensor(lps, dtype=torch.float32)
    b["teacher_ids"] = t_ids
    b["teacher_logprobs"] = t_lps


def build_sample_tensors(assembled: dict, pad_token_id: int = 0):
    """Build the canonical padded per-sample tensors from an assembled dict (needs torch).

    Unpadded single sample (P=len(prompt), R=len(response)); returns a dict over
    CANONICAL_SAMPLE_KEYS so the result is schema-compatible with the upstream
    completed-sample path. position_ids use the standard left-pad convention.
    """
    import torch

    prompt = assembled["prompt_token_ids"]
    resp = assembled["response_token_ids"]
    P, R = len(prompt), len(resp)
    prompts = torch.tensor(prompt, dtype=torch.long)
    responses = torch.tensor(resp, dtype=torch.long)
    input_ids = torch.cat([prompts, responses])
    attention_mask = torch.ones(P + R, dtype=torch.long)
    position_ids = (attention_mask.cumsum(-1) - 1).clamp_min(0)
    response_mask = torch.tensor(assembled["response_mask"], dtype=torch.long)
    teacher_ids = torch.tensor(assembled["teacher_topk_ids"], dtype=torch.long)            # (R, k)
    teacher_logprobs = torch.tensor(assembled["teacher_topk_log_probs"], dtype=torch.float)  # (R, k)
    return {
        "prompts": prompts,
        "responses": responses,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "response_mask": response_mask,
        "teacher_logprobs": teacher_logprobs,
        "teacher_ids": teacher_ids,
    }


def aggregate_teacher_telemetry(records: list) -> dict:
    """Aggregate per-chunk teacher telemetry into teacher/* metrics -- the cache-hit PROOF.

    records: list of (parent_id, chunk_idx, n_tokens, telemetry), telemetry having cached_tokens /
    total_tokens / replica_rank / latency_s / queue_wait_s. Pure stdlib (unit-testable without torch).

    Key signal: prefix_amplification_ratio = processed(uncached) tokens per new-span token. ~1 means
    full reuse; the no-cache baseline (total/span) is what the teacher processes WITHOUT reuse, so their
    ratio is the reduction factor. cache_hits_by_chunk_idx should be ~0 at idx 0 and rise after.
    """
    out: dict = {}
    if not records:
        return out

    def _pct(xs, p):
        if not xs:
            return 0.0
        s = sorted(xs)
        return float(s[min(len(s) - 1, int(p * len(s)))])

    _lat = [t.get("latency_s") for *_, t in records if t.get("latency_s") is not None]
    _qw = [t.get("queue_wait_s") for *_, t in records if t.get("queue_wait_s") is not None]
    sum_cached = sum(t.get("cached_tokens") for *_, t in records if t.get("cached_tokens") is not None)
    sum_total = sum(t.get("total_tokens") for *_, t in records if t.get("total_tokens") is not None)
    sum_span = sum(n for _, _, n, _ in records)
    sum_uncached = max(0, sum_total - sum_cached)
    out["teacher/cached_tokens"] = sum_cached
    out["teacher/uncached_tokens"] = sum_uncached
    out["teacher/cache_hit_ratio"] = (sum_cached / sum_total) if sum_total else 0.0
    out["teacher/prefix_amplification_ratio"] = (sum_uncached / sum_span) if sum_span else 0.0
    out["teacher/prefix_amplification_no_cache"] = (sum_total / sum_span) if sum_span else 0.0

    by_idx: dict = {}
    per_parent_reqs: dict = {}
    per_parent_replicas: dict = {}
    replica_loads: dict = {}
    for pid, cidx, n, t in records:
        c = t.get("cached_tokens")
        if c is not None:
            by_idx.setdefault(cidx, []).append(c)
        per_parent_reqs[pid] = per_parent_reqs.get(pid, 0) + 1
        rr = t.get("replica_rank")
        if rr is not None:
            per_parent_replicas.setdefault(pid, set()).add(rr)
            replica_loads[rr] = replica_loads.get(rr, 0) + 1
    out["teacher/cache_hits_by_chunk_idx"] = {k: (sum(v) / len(v)) for k, v in sorted(by_idx.items())}
    out["teacher/request_latency_p50"] = _pct(_lat, 0.5)
    out["teacher/request_latency_p95"] = _pct(_lat, 0.95)
    out["teacher/queue_wait_p50"] = _pct(_qw, 0.5)
    out["teacher/queue_wait_p95"] = _pct(_qw, 0.95)
    nparents = len(per_parent_reqs)
    out["teacher/requests_per_parent"] = (sum(per_parent_reqs.values()) / nparents) if nparents else 0.0
    if per_parent_replicas:
        out["teacher/unique_replicas_per_parent"] = sum(len(s) for s in per_parent_replicas.values()) / len(per_parent_replicas)
    out["teacher/replica_load_distribution"] = {k: replica_loads[k] for k in sorted(replica_loads)}
    if replica_loads:
        loads = list(replica_loads.values())
        out["teacher/replica_load_skew_max_over_mean"] = max(loads) / (sum(loads) / len(loads))
    return out
