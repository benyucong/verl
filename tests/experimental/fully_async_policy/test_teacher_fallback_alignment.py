# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""Clean-recompute FALLBACK alignment for chunk-streamed teacher scoring.

TWO INDEXING CONVENTIONS feed the same span pipeline (teacher_manager.compute_teacher_logprobs_single):

  * incremental suffix (extract_incremental_prompt_logprobs): UNSHIFTED -- ``prompt_logprobs[p]
    scores absolute token p``. The desired predictions for tokens [P+s, P+e) are rows [P+s, P+e).
  * clean recompute  (extract_prompt_logprobs, the strict parser): SHIFTED -- it iterates
    ``prompt_logprobs[1:]``, so row j holds the prediction for token j+1, and it appends a trailing
    ALL-ZERO dummy row at index S-1. The same predictions sit at strict rows [P+s-1, P+e-1).

Both return "row i = the prediction for token P+s+i", which agent_loop re-places at full-tensor
index P+s-1+i (index x holds the prediction for token x+1) and the loss reads back with the left
shift (response position t reads index P+t-1).

THE BUG THIS GUARDS AGAINST. The span-alignment fix (d28750e8) moved the incremental window from
[P+s+1, P+e+1) to [P+s, P+e) together with the write offset -- but the FALLBACK slice (taken when
cross-response cache interference makes the incremental suffix unusable) kept slicing the STRICT
output at the same absolute offsets [P+s, P+e). In strict indexing that window is one row late:
every position of a fallback chunk was supervised by the NEXT token's teacher distribution, and the
chunk's last position by the strict trailing dummy -- an all-zero row that ``exp()`` reads as
uniform mass K instead of a distribution.

Observed in production (post-d28750e8 k1.0cont runs, jobs 43978992/43976609): the metric windows
with ``teacher/fallback_clean_count > 0`` showed ``actor/distillation/teacher_mass_max == 64.0``
(== K) EXACTLY, and every window with fallback_clean_count == 0 showed max ~= 1.0000005. Perfect
binary separation over ~45 windows -- the same teacher_mass tell that found the original bug.
"""

import math

try:
    import pytest
except ImportError:  # runnable standalone: the cluster venv has no pytest
    class _Approx:
        def __init__(self, v, abs=0.0):
            self.v, self.abs = v, abs

        def __eq__(self, other):
            return math.isclose(other, self.v, abs_tol=self.abs or 1e-12)

    class _Pytest:
        approx = staticmethod(lambda v, abs=0.0: _Approx(v, abs))

        class mark:
            @staticmethod
            def parametrize(_names, _values):
                def deco(fn):
                    fn._params = (_names, _values)
                    return fn
                return deco

    pytest = _Pytest()

# A row's value encodes WHICH token it predicts; DUMMY marks the strict parser's trailing all-zero
# row, UNWRITTEN a full-tensor slot no span ever filled. The production tensor is torch.zeros, so
# the loss cannot tell the two apart -- both exponentiate to uniform mass K.
UNWRITTEN = -1
DUMMY = 0


def strict_rows(seq_len):
    """The strict parser's output for a request of seq_len tokens: row j predicts token j+1
    (it iterates prompt_logprobs[1:]), plus a trailing all-zero dummy row at index seq_len-1."""
    return [j + 1 for j in range(seq_len - 1)] + [DUMMY]


def incremental_rows(seq_len, valid_start):
    """The incremental parser's valid suffix, in UNSHIFTED absolute indexing: row p (for p in
    [valid_start, seq_len)) scores absolute token p. Returned as {abs_index: predicted_token}."""
    return {p: p for p in range(valid_start, seq_len)}


def manager_span(prompt_len, span_start, span_end, *, fallback, legacy):
    """The [n] span rows compute_teacher_logprobs_single returns for a chunk owning response
    tokens [span_start, span_end). The request's sequence is prompt + response[:span_end].

    fallback=False: slice the UNSHIFTED incremental suffix at [P+s, P+e)  (both eras, post-fix).
    fallback=True : slice the STRICT output; legacy=True reproduces the pre-fix window [P+s, P+e),
                    the fixed window is [P+s-1, P+e-1).
    """
    seq_len = prompt_len + span_end
    if not fallback:
        rows = incremental_rows(seq_len, valid_start=prompt_len + span_start)
        return [rows[p] for p in range(prompt_len + span_start, prompt_len + span_end)]
    rows = strict_rows(seq_len)
    if legacy:
        return rows[prompt_len + span_start:prompt_len + span_end]
    return rows[prompt_len + span_start - 1:prompt_len + span_end - 1]


def assemble_full_tensor(prompt_len, response_len, chunk, fallback_chunks, *, legacy):
    """Run every chunk through the manager (fallback for the chunk indices in fallback_chunks),
    re-place each span at index P+s-1 (agent_loop's ``ss``), and return the full tensor of
    'predicts token N' markers."""
    tensor = [UNWRITTEN] * (prompt_len + response_len)
    for ci, start in enumerate(range(0, response_len, chunk)):
        end = min(start + chunk, response_len)
        span = manager_span(prompt_len, start, end, fallback=(ci in fallback_chunks), legacy=legacy)
        assert len(span) == end - start
        ss = prompt_len + start - 1
        for i, v in enumerate(span):
            tensor[ss + i] = v
    return tensor


def trained_positions(tensor, prompt_len, response_len):
    """What the loss actually sees: response position t reads index P+t-1 (the left shift)."""
    return [tensor[prompt_len + t - 1] for t in range(response_len)]


def zero_rows(seen):
    """Rows the loss sees as all-zero: explicit dummies plus never-written positions."""
    return sum(1 for v in seen if v in (DUMMY, UNWRITTEN))


def test_incremental_and_fallback_paths_return_identical_spans():
    """The invariant the bug violated: for the same chunk, the clean-recompute fallback must return
    exactly the rows the incremental path returns -- it is a recovery path, not a different labeling."""
    P = 153
    for s, e in [(0, 1024), (1024, 2048), (4096, 7855)]:
        inc = manager_span(P, s, e, fallback=False, legacy=False)
        fb = manager_span(P, s, e, fallback=True, legacy=False)
        assert fb == inc, f"span [{s},{e}): fallback rows differ from incremental rows"
        assert inc == [P + t for t in range(s, e)], f"span [{s},{e}): rows are not the span's own tokens"


@pytest.mark.parametrize("fallback_chunks", [set(), {0}, {1}, {0, 1}, {0, 3}])
def test_fixed_fallback_trains_every_position_on_its_own_label(fallback_chunks):
    P, R, chunk = 153, 8192, 1024  # 8 chunks
    tensor = assemble_full_tensor(P, R, chunk, fallback_chunks, legacy=False)
    seen = trained_positions(tensor, P, R)
    for t, got in enumerate(seen):
        assert got == P + t, (
            f"response token {t} supervised by the prediction for token {got}, expected {P + t} "
            f"(fallback_chunks={sorted(fallback_chunks)})"
        )
    assert zero_rows(seen) == 0


def test_legacy_fallback_poisons_the_chunk_end_and_shifts_every_label():
    """Regression witness: with the pre-fix strict-window slice, a fallback chunk [s, e) trains
    position e-1 on the all-zero dummy and every other position on the NEXT token's prediction."""
    P, R, chunk = 153, 8192, 4096
    fb = {0}  # chunk 0 falls back (the common case: a sibling response cached the prompt prefix)
    seen = trained_positions(assemble_full_tensor(P, R, chunk, fb, legacy=True), P, R)
    s, e = 0, chunk
    assert seen[e - 1] == DUMMY, "the fallback chunk's last position must hit the strict trailing dummy"
    for t in range(s, e - 1):
        assert seen[t] == P + t + 1, f"position {t}: expected the off-by-one label {P + t + 1}, got {seen[t]}"
    # every non-fallback position stays correct
    for t in range(e, R):
        assert seen[t] == P + t


@pytest.mark.parametrize("n_fallback,chunk", [(1, 4096), (2, 4096), (5, 1024)])
def test_legacy_reproduces_the_observed_teacher_mass_signature(n_fallback, chunk):
    """teacher_mass = 1 + (K-1)*f with f = one all-zero row per fallback chunk / R; max mass == K
    exactly when any chunk fell back -- the binary fallback_clean_count <-> teacher_mass_max == 64
    correlation observed across every k1.0cont metric window (jobs 43978992/43976609)."""
    P, R, K = 153, 8192, 64
    n_chunks = math.ceil(R / chunk)
    assert n_fallback <= n_chunks
    fb = set(range(n_fallback))
    legacy_seen = trained_positions(assemble_full_tensor(P, R, chunk, fb, legacy=True), P, R)
    fixed_seen = trained_positions(assemble_full_tensor(P, R, chunk, fb, legacy=False), P, R)
    assert zero_rows(legacy_seen) == n_fallback
    f = zero_rows(legacy_seen) / R
    assert 1.0 + (K - 1) * f == pytest.approx(1.0 + (K - 1) * n_fallback / R, abs=1e-12)
    max_mass_legacy = K if zero_rows(legacy_seen) else 1
    assert max_mass_legacy == K
    assert zero_rows(fixed_seen) == 0, "fixed path must leave teacher_mass_max at ~1"


def test_fallback_window_needs_no_dummy_and_no_out_of_range_row():
    """The fixed strict window [P+s-1, P+e-1) exists entirely inside the request's REAL rows:
    the request is prompt + response[:e] (S = P+e tokens), the strict output has real rows
    [0, S-1) and the dummy at S-1. Lowest read row P+s-1 >= 0 (P >= 1); highest P+e-2 == S-2."""
    P, R, chunk = 1, 8192, 1024  # P=1 is the tightest legal prompt (strict parser needs >= 1 row)
    for start in range(0, R, chunk):
        end = min(start + chunk, R)
        span = manager_span(P, start, end, fallback=True, legacy=False)
        assert DUMMY not in span
        assert span == [P + t for t in range(start, end)]


def test_non_multiple_response_length_with_fallback_final_chunk():
    """Real responses stop on EOS, so the final chunk is short -- and it too can fall back."""
    P, R, chunk = 153, 7855, 4096
    fb = {math.ceil(R / chunk) - 1}  # only the (short) final chunk falls back
    seen = trained_positions(assemble_full_tensor(P, R, chunk, fb, legacy=False), P, R)
    assert seen == [P + t for t in range(R)]
    assert zero_rows(seen) == 0


if __name__ == "__main__":
    import sys

    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        names, values = getattr(fn, "_params", (None, [()]))
        for v in values:
            args = v if isinstance(v, tuple) else (v,)
            label = f"{name}{args if names else ''}"
            try:
                fn(*args)
                print(f"  PASS  {label}")
            except AssertionError as e:
                failures += 1
                print(f"  FAIL  {label}\n        {e}")
    print(f"\n{'ALL PASS' if not failures else str(failures) + ' FAILURES'}")
    sys.exit(1 if failures else 0)
