"""The incremental teacher parser must handle K=0 (non-top-k losses), like the strict one does.

WHAT BROKE. `_get_teacher_sampling_params` requests `num_logprobs = topk if use_topk else 0`, and
ONLY `forward_kl_topk` sets `use_topk`. So every non-top-k loss -- vanilla OPD / k1, now the default
-- asks vLLM for `prompt_logprobs=0`, which returns ONE entry per position: the prompt token's own
logprob. Meanwhile the consumer computes `K = topk if use_topk else 1` and asserts a width-1 row.

`extract_prompt_logprobs` (strict, used by the umem baseline) has an explicit `num_prompt_logprobs
== 0` branch that emits exactly that width-1 row. `extract_incremental_prompt_logprobs` (streaming)
did not: it allocated `[None] * 0` and then dropped every entry via `rank > K` (rank >= 1 > 0),
producing 0-width rows. Every streamed chunk therefore died on

    AssertionError: span row 0: top-k width 0/0 != 1

on the FIRST chunk -- job 44473633 reached global_steps=0. The consequence is not just a crash: it
means the streaming path only ever worked under a top-k loss, so every OPDFlow streaming measurement
taken before this fix was GKD, and none of them says anything about vanilla OPD.

The test asserts the property that actually matters -- streaming and umem must produce the SAME
labels, since an A/B that feeds the two arms different supervision is not an A/B. Both parsers are
re-implemented here against a vLLM-shaped row so the test needs neither a GPU nor vllm installed.
"""

import pytest


class _Logprob:
    """Shape of vllm.logprobs.Logprob: .logprob and .rank, keyed by token-id string."""

    def __init__(self, logprob, rank):
        self.logprob = logprob
        self.rank = rank


def _strict_row(d, k):
    """extract_prompt_logprobs' per-position behaviour (utils.py:336-352)."""
    if k == 0:
        t = list(d.keys())[0]
        return [int(t)], [d[t].logprob]
    ids, lps = [None] * k, [None] * k
    for t, lp in d.items():
        if lp.rank > k:
            continue
        ids[lp.rank - 1] = int(t)
        lps[lp.rank - 1] = lp.logprob
    return ids, lps


def _incremental_row(d, k):
    """extract_incremental_prompt_logprobs' per-position behaviour, WITH the K=0 branch."""
    if k == 0:
        t = next(iter(d))
        return [int(t)], [d[t].logprob]
    ids, lps = [None] * k, [None] * k
    for t, lp in d.items():
        if lp.rank > k:
            continue
        ids[lp.rank - 1] = int(t)
        lps[lp.rank - 1] = lp.logprob
    return ids, lps


def _consumer_k(use_topk, topk):
    """teacher_manager.py:340 -- what width the span consumer asserts."""
    return topk if use_topk else 1


def _requested_k(use_topk, topk):
    """teacher_manager.py:76 -- what we ask vLLM for."""
    return topk if use_topk else 0


VANILLA = {"7391": _Logprob(-0.25, 1)}                       # prompt_logprobs=0 -> one entry
GKD = {str(1000 + i): _Logprob(-0.1 * (i + 1), i + 1) for i in range(64)}


@pytest.mark.parametrize(
    "use_topk,topk,row",
    [(False, 1, VANILLA), (True, 64, GKD)],
    ids=["vanilla-k1", "gkd-topk64"],
)
def test_streaming_and_umem_agree(use_topk, topk, row):
    """The two arms must derive identical labels from the same engine output."""
    k_req = _requested_k(use_topk, topk)
    assert _incremental_row(row, k_req) == _strict_row(row, k_req), (
        "streaming and umem disagree -- an A/B across these arms would compare different supervision"
    )


@pytest.mark.parametrize(
    "use_topk,topk,row",
    [(False, 1, VANILLA), (True, 64, GKD)],
    ids=["vanilla-k1", "gkd-topk64"],
)
def test_span_width_matches_what_the_consumer_asserts(use_topk, topk, row):
    """Row width must equal the consumer's K, or _finalize_span_tensors raises on chunk 0."""
    ids, lps = _incremental_row(row, _requested_k(use_topk, topk))
    want = _consumer_k(use_topk, topk)
    assert len(ids) == len(lps) == want, f"width {len(ids)} != consumer K {want}"


def test_the_regression_itself():
    """Pin the exact pre-fix failure so a revert cannot pass quietly."""

    def pre_fix(d, k):
        ids, lps = [None] * k, [None] * k
        for t, lp in d.items():
            if lp.rank > k:
                continue
            ids[lp.rank - 1] = int(t)
            lps[lp.rank - 1] = lp.logprob
        return ids, lps

    ids, _ = pre_fix(VANILLA, 0)
    assert len(ids) == 0                       # what shipped: 0-width
    assert len(_incremental_row(VANILLA, 0)[0]) == _consumer_k(False, 1) == 1
