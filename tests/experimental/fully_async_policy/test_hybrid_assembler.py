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

    # out-of-order is NO LONGER an error (reorder-tolerant): span[1] before span[0] buffers, then completes
    acc = ParentLabelAccumulator(parent_id="g", prompt_token_ids=[1], topk=k)
    acc.add_span(spans[1])                      # starts at 256 -> buffered, nothing drained yet
    assert acc.next_expected_offset == 0
    acc.add_span(spans[0])                      # fills the gap -> drains both contiguously
    assert acc.next_expected_offset == 512

    # overlap: re-add span[0] (start 0 < already-consumed 256) is a hard error
    acc2 = ParentLabelAccumulator(parent_id="o", prompt_token_ids=[1], topk=k)
    acc2.add_span(spans[0])
    _raised(lambda: acc2.add_span(spans[0]))  # starts at 0, already consumed -> overlap

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


def test_span_only_payload_reconstructs_full_sample():
    """H-ACC-SPAN: span-only ChunkSamples (carrying only their new-span teacher labels, no full-prefix
    payload) accumulate into the SAME full sample as the whole-payload path, and fill_carrier_teacher_
    tensors rebuilds the full-sequence [1, P+RW, k] layout the actor path expects."""
    import torch

    from verl.experimental.fully_async_policy.chunk_sample import ChunkSample
    from verl.experimental.fully_async_policy.hybrid_assembler import fill_carrier_teacher_tensors, span_from_chunk

    R, k, P, RW = 12, 8, 4, 16  # L=R=12, response_width RW=16 (4 pad slots)
    g = torch.Generator().manual_seed(7)
    gt_resp = torch.randint(0, 50000, (R,), generator=g)
    gt_ids = torch.randint(0, 50000, (R, k), generator=g)
    gt_lps = -torch.rand(R, k, generator=g)

    acc = ParentLabelAccumulator(parent_id="s", prompt_token_ids=[], topk=k)
    cs = 4
    for o in range(0, R, cs):
        n = min(cs, R - o)
        ch = ChunkSample(
            sample_id="s", chunk_idx=o // cs, token_offset=o, n_tokens=n,
            tokens=gt_resp[o:o + n].tolist(), is_final=(o + n == R), policy_version=0,
            parent_payload=None,  # span-only: non-final chunks ship no payload
            span_teacher_ids=gt_ids[o:o + n].clone(),
            span_teacher_logprobs=gt_lps[o:o + n].clone(),
        )
        acc.add_span(span_from_chunk(ch))
    acc.finalize(R)
    out = acc.assemble()
    assert out["response_token_ids"] == gt_resp.tolist()
    assert out["teacher_topk_ids"] == gt_ids.tolist()
    assert out["teacher_topk_log_probs"] == gt_lps.tolist()

    # fill_carrier rebuilds full-sequence teacher tensors: response token j at index P+j; pad elsewhere
    from types import SimpleNamespace
    carrier = SimpleNamespace(batch={
        "prompts": torch.zeros(1, P, dtype=torch.long),
        "responses": torch.zeros(1, RW, dtype=torch.long),
    })
    fill_carrier_teacher_tensors(carrier, out)
    assert tuple(carrier.batch["teacher_ids"].shape) == (1, P + RW, k)
    assert carrier.batch["teacher_ids"].dtype == torch.int32  # matches native teacher tensor dtype
    assert torch.equal(carrier.batch["teacher_ids"][0, P:P + R], gt_ids.to(torch.int32))
    assert torch.allclose(carrier.batch["teacher_logprobs"][0, P:P + R], gt_lps, atol=1e-6)
    assert int(torch.count_nonzero(carrier.batch["teacher_ids"][0, :P])) == 0          # prompt region pad
    assert int(torch.count_nonzero(carrier.batch["teacher_ids"][0, P + R:])) == 0      # response pad region


def test_reorder_tolerant_reassembly():
    """Chunks are emitted as concurrent asyncio tasks (single_turn_agent_loop.py:402) and arrive out of
    order -- including the final chunk before earlier ones. The accumulator must buffer and stitch the
    SAME whole response, and only report is_complete once contiguous coverage [0,L) reaches final_length."""
    L, k, chunk = 1024, 8, 256
    resp, tk_ids, tk_lps = _synthetic_whole_response(L, k)
    spans = _spans(resp, tk_ids, tk_lps, chunk)  # 4 spans: [0,256) [256,512) [512,768) [768,1024)
    acc = ParentLabelAccumulator(parent_id="r", prompt_token_ids=[1, 2], topk=k)
    # worst case: final span (idx 3) arrives FIRST, then 1, 2, and offset-0 (idx 0) LAST
    for idx in (3, 1, 2, 0):
        if idx == 3:
            acc.mark_final(L)               # final chunk seen before any earlier chunk
        acc.add_span(spans[idx])
        if idx != 0:
            assert not acc.is_complete       # gap at offset 0 keeps it incomplete until idx 0 lands
    assert acc.is_complete                   # offset-0 span closed the contiguous chain to L
    acc.finalize(L)
    out = acc.assemble()
    assert out["response_token_ids"] == resp
    assert out["teacher_topk_ids"] == tk_ids
    assert out["teacher_topk_log_probs"] == tk_lps
    assert out["response_mask"] == [1] * L


def test_finalize_is_exactly_once():
    """A parent finalizes (and is enqueued for the actor) EXACTLY once -> a second finalize raises."""
    k = 4
    acc = ParentLabelAccumulator(parent_id="x", prompt_token_ids=[], topk=k)
    acc.add_span(ChunkLabelSpan(0, 2, [1, 2], [[0] * k, [0] * k], [[0.0] * k, [0.0] * k]))
    acc.finalize(2)
    _raised(lambda: acc.finalize(2))


def test_aggregate_teacher_telemetry():
    """The cache-hit proof math: full KV reuse -> only the new span is processed (amplification ~1) and
    the parent is pinned to one replica; no reuse -> full prefix each chunk (ratio 0) and scattered."""
    from verl.experimental.fully_async_policy.hybrid_assembler import aggregate_teacher_telemetry

    P, span = 4, 256
    recs_reuse, recs_noreuse = [], []
    for i in range(4):  # one response, 4 chunks; prefix grows by `span` each chunk
        total = P + span * (i + 1)
        recs_reuse.append(("p_r0", i, span,
                           {"cached_tokens": total - span, "total_tokens": total, "replica_rank": 2,
                            "latency_s": 0.1, "queue_wait_s": 0.01}))
        recs_noreuse.append(("p_r0", i, span,
                             {"cached_tokens": 0, "total_tokens": total, "replica_rank": i % 4,
                              "latency_s": 0.2, "queue_wait_s": 0.02}))
    a = aggregate_teacher_telemetry(recs_reuse)
    b = aggregate_teacher_telemetry(recs_noreuse)

    assert abs(a["teacher/prefix_amplification_ratio"] - 1.0) < 1e-9     # reuse -> ~1 span processed/span
    assert a["teacher/cache_hit_ratio"] > 0.5
    assert a["teacher/unique_replicas_per_parent"] == 1.0                # pinned to replica 2
    assert a["teacher/replica_load_distribution"] == {2: 4}
    assert a["teacher/cache_hits_by_chunk_idx"][0] < a["teacher/cache_hits_by_chunk_idx"][3]

    assert b["teacher/cache_hit_ratio"] == 0.0                            # no reuse -> 0 cached
    assert b["teacher/prefix_amplification_ratio"] > a["teacher/prefix_amplification_ratio"]
    assert b["teacher/unique_replicas_per_parent"] == 4.0                 # scattered across 4 replicas

    assert aggregate_teacher_telemetry([]) == {}


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
    test_span_only_payload_reconstructs_full_sample()
    test_reorder_tolerant_reassembly()
    test_finalize_is_exactly_once()
    test_aggregate_teacher_telemetry()
    print("HYBRID ASSEMBLER (A1+H-ACC+SPAN+REORDER+KVTELEM) PASS: stitch==whole; coverage guarded; "
          "span-slice==carrier; span-only==full; reorder-tolerant; finalize-once; teacher-telemetry-agg")
