# Phase A1 (CPU): hybrid per-parent label assembler unit test.
#
# Load-bearing check: splitting a whole completed response into in-order chunk spans and
# stitching them back via ParentLabelAccumulator reproduces the whole-response labels
# EXACTLY (response ids, teacher top-k ids, teacher top-k log-probs, response mask), with
# exact [0, L) coverage, no duplicate positions, correct per-position (causal) alignment,
# and a reconstructed sample whose tensor schema matches an upstream completed sample.
#
# Pure-stdlib assembler; build_sample_tensors needs torch (run in the container).
import random

from verl.experimental.fully_async_policy.hybrid_assembler import (
    CANONICAL_SAMPLE_KEYS,
    ChunkLabelSpan,
    ParentLabelAccumulator,
)


def _synthetic_whole_response(L, k, seed=0):
    rng = random.Random(seed)
    resp = [rng.randint(0, 50000) for _ in range(L)]
    tk_ids = [[rng.randint(0, 50000) for _ in range(k)] for _ in range(L)]
    tk_lps = [[-rng.random() for _ in range(k)] for _ in range(L)]
    return resp, tk_ids, tk_lps


def _spans(resp, tk_ids, tk_lps, chunk, version_of=lambda i: 0):
    out = []
    L = len(resp)
    for s in range(0, L, chunk):
        e = min(s + chunk, L)
        out.append(ChunkLabelSpan(
            span_start=s, span_end=e,
            response_token_ids=resp[s:e],
            teacher_topk_ids=tk_ids[s:e],
            teacher_topk_log_probs=tk_lps[s:e],
            policy_version=version_of(s),
        ))
    return out


def test_stitch_equals_whole_response():
    L, k, chunk = 1000, 8, 256  # last span is partial (1000 % 256 != 0)
    resp, tk_ids, tk_lps = _synthetic_whole_response(L, k)
    acc = ParentLabelAccumulator(parent_id="p0", prompt_token_ids=[1, 2, 3, 4], topk=k)
    # chunks generated across drifting policy versions -- must not affect the stitched labels
    for span in _spans(resp, tk_ids, tk_lps, chunk, version_of=lambda s: s // chunk):
        acc.add_span(span)
    acc.finalize(L, reason="eos")
    out = acc.assemble()
    # exact stitch equivalence
    assert out["response_token_ids"] == resp
    assert out["teacher_topk_ids"] == tk_ids
    assert out["teacher_topk_log_probs"] == tk_lps
    assert out["response_mask"] == [1] * L
    # exact coverage, no dup positions, per-position alignment
    assert len(out["teacher_topk_ids"]) == L
    for i in range(L):
        assert out["teacher_topk_ids"][i] == tk_ids[i], f"position {i} misaligned"
    # version is per-token diagnostic, length-aligned, never folded into labels
    assert len(out["rollout_policy_versions"]) == L


def test_gap_overlap_shape_and_coverage_are_hard_errors():
    k = 8
    resp, tk_ids, tk_lps = _synthetic_whole_response(512, k)
    spans = _spans(resp, tk_ids, tk_lps, 256)

    # out-of-order / gap: feeding span[1] before span[0]
    acc = ParentLabelAccumulator(parent_id="g", prompt_token_ids=[1], topk=k)
    _raised(lambda: acc.add_span(spans[1]))  # starts at 256, expected 0

    # overlap: re-add span[0]
    acc2 = ParentLabelAccumulator(parent_id="o", prompt_token_ids=[1], topk=k)
    acc2.add_span(spans[0])
    _raised(lambda: acc2.add_span(spans[0]))  # starts at 0, expected 256

    # wrong top-k width
    acc3 = ParentLabelAccumulator(parent_id="w", prompt_token_ids=[1], topk=k)
    bad = ChunkLabelSpan(0, 2, [1, 2], [[1] * (k - 1), [1] * (k - 1)], [[0.0] * (k - 1)] * 2)
    _raised(lambda: acc3.add_span(bad))

    # coverage gap at finalize: only first span present, finalize to full length
    acc4 = ParentLabelAccumulator(parent_id="c", prompt_token_ids=[1], topk=k)
    acc4.add_span(spans[0])
    _raised(lambda: acc4.finalize(512))


def test_reconstructed_tensor_schema_matches_upstream():
    from verl.experimental.fully_async_policy.hybrid_assembler import build_sample_tensors

    L, k = 300, 8
    resp, tk_ids, tk_lps = _synthetic_whole_response(L, k)
    acc = ParentLabelAccumulator(parent_id="t", prompt_token_ids=[5, 6, 7], topk=k)
    for span in _spans(resp, tk_ids, tk_lps, 256):
        acc.add_span(span)
    acc.finalize(L)
    t = build_sample_tensors(acc.assemble(), pad_token_id=0)
    assert set(t.keys()) == set(CANONICAL_SAMPLE_KEYS), (set(t.keys()), set(CANONICAL_SAMPLE_KEYS))
    P, R = 3, L
    assert tuple(t["input_ids"].shape) == (P + R,)
    assert tuple(t["teacher_ids"].shape) == (R, k)
    assert tuple(t["teacher_logprobs"].shape) == (R, k)
    assert t["response_mask"].sum().item() == R
    # standard left-pad position_ids convention (no left padding here -> 0..P+R-1)
    assert t["position_ids"][0].item() == 0 and t["position_ids"][-1].item() == P + R - 1
    # input_ids = prompt | response
    assert t["input_ids"][:P].tolist() == [5, 6, 7]
    assert t["input_ids"][P:].tolist() == resp


def test_span_from_chunk_payload_stitches_to_carrier():
    """H-ACC: slicing each chunk's NEW span out of its (full-sequence-aligned) parent_payload and
    accumulating reproduces the carrier's response-region labels bit-identically -- the equivalence
    the runtime relies on when it reuses the final chunk's payload as the actor row."""
    import torch
    from types import SimpleNamespace

    from verl.experimental.fully_async_policy.hybrid_assembler import build_sample_tensors, span_from_chunk_payload

    P, R, k = 4, 12, 8  # prompt_width, response_width(=L), topk; teacher tensors are [1, P+R, k]
    g = torch.Generator().manual_seed(11)
    responses = torch.randint(0, 50000, (1, R), generator=g)
    teacher_ids = torch.randint(0, 50000, (1, P + R, k), generator=g)
    teacher_lps = -torch.rand(1, P + R, k, generator=g)
    payload = SimpleNamespace(batch={
        "prompts": torch.zeros(1, P, dtype=torch.long),
        "responses": responses,
        "teacher_ids": teacher_ids,
        "teacher_logprobs": teacher_lps,
    })

    acc = ParentLabelAccumulator(parent_id="p", prompt_token_ids=[0] * P, topk=k)
    chunk_size = 4
    for o in range(0, R, chunk_size):
        n = min(chunk_size, R - o)
        chunk = SimpleNamespace(parent_payload=payload, token_offset=o, n_tokens=n,
                                policy_version=0, is_final=(o + n == R))
        acc.add_span(span_from_chunk_payload(chunk))
    acc.finalize(R)
    out = acc.assemble()

    # stitched labels == the carrier's response region (response token j is at teacher index P+j)
    assert out["response_token_ids"] == responses[0].tolist()
    assert out["teacher_topk_ids"] == teacher_ids[0, P:P + R].tolist()
    assert out["teacher_topk_log_probs"] == teacher_lps[0, P:P + R].tolist()
    assert out["response_mask"] == [1] * R

    # built tensors match the carrier's response region exactly (carrier-overwrite would be a no-op)
    t = build_sample_tensors(out, pad_token_id=0)
    assert torch.equal(t["teacher_ids"], teacher_ids[0, P:P + R])
    assert torch.allclose(t["teacher_logprobs"], teacher_lps[0, P:P + R], atol=1e-6)
    assert torch.equal(t["responses"], responses[0])


def _raised(fn):
    try:
        fn()
    except AssertionError:
        return
    raise AssertionError("expected an AssertionError but none was raised")


if __name__ == "__main__":
    test_stitch_equals_whole_response()
    test_gap_overlap_shape_and_coverage_are_hard_errors()
    test_reconstructed_tensor_schema_matches_upstream()
    test_span_from_chunk_payload_stitches_to_carrier()
    print("HYBRID ASSEMBLER (A1+H-ACC) PASS: stitch==whole; coverage guarded; schema matches; span-slice==carrier")
