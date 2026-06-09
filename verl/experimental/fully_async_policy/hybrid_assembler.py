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

from dataclasses import dataclass, field
from typing import Optional

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
    is_finished: bool = False
    final_length: Optional[int] = None
    finish_reason: Optional[str] = None

    def add_span(self, span: ChunkLabelSpan) -> None:
        if self.is_finished:
            raise AssertionError(f"parent {self.parent_id} already finalized")
        # MVP invariant: spans arrive strictly in order and contiguous -> exactly-once coverage.
        if span.span_start != self.next_expected_offset:
            raise AssertionError(
                f"parent {self.parent_id}: non-contiguous span (gap/overlap): "
                f"expected start {self.next_expected_offset}, got {span.span_start}"
            )
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
        self.response_token_ids.extend(span.response_token_ids)
        self.teacher_topk_ids.extend(span.teacher_topk_ids)
        self.teacher_topk_log_probs.extend(span.teacher_topk_log_probs)
        self.rollout_policy_versions.extend([span.policy_version] * span_len)
        self.next_expected_offset = span.span_end

    def finalize(self, final_length: int, reason: str = "eos") -> None:
        """Mark the response complete (EOS / max-length / truncation). Requires exact coverage."""
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
