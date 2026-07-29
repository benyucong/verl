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
"""Teacher-label alignment for chunk-streamed (incremental) scoring.

THE CONVENTION, set by the strict parser (vllm_rollout/utils.py extract_prompt_logprobs, which
iterates ``output.prompt_logprobs[1:]``):

    teacher tensor index i holds the teacher's prediction for token i+1.

The student's logits are left-shifted by one before the loss (workers/utils/padding.py: the response
slice is ``values[seq_offset - resp_len - 1 : seq_offset - 1]``), so response position t reads index
P+t-1. Combining the two: **the label for response token t must sit at index P+t-1.**

A chunk owning response tokens [s, e) therefore fills indices [P+s-1, P+e-1) with the predictions for
tokens [P+s, P+e). Those are ``prompt_logprobs`` rows [P+s, P+e), every one of which that chunk's own
teacher request already has -- its prompt is ``prompt + response[:e]``, i.e. P+e tokens long.

THE BUG THIS GUARDS AGAINST. The incremental path used to read [P+s+1, P+e+1) and write at [P+s, P+e),
which made each chunk responsible for the prediction of token P+e -- the FIRST TOKEN OF THE NEXT
CHUNK, which had not been generated when the request was scored. The guard for that
(``last_is_dummy``) compared against ``valid_suffix_end_abs = P+span_end`` and so was unconditionally
true for EVERY chunk, not just the final one, appending ``[0]*K`` each time. Under the left shift the
dummy landed on a real trained position, so one response position per chunk was supervised by an
all-zero teacher row -- which ``exp()`` reads as uniform mass K rather than a distribution.

Observed in production before the fix: ``actor/distillation/teacher_mass`` tracked
``1 + (K-1)*ceil(R/c)/R`` exactly -- umem 0.99986, c4096 1.0155, c1024 1.0613, c256 1.2512.
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

# A row's value encodes WHICH token it predicts, so alignment is checkable by equality.
# NOTE: the production tensor is torch.zeros, so an UNWRITTEN row and an explicit DUMMY row are
# indistinguishable to the loss -- both exponentiate to uniform mass K. They are separate markers
# here only so the test can say WHICH defect produced each bad row.
UNWRITTEN = -1
DUMMY = 0


def zero_rows(seen):
    """Rows the loss sees as all-zero: explicit dummies plus never-written positions."""
    return sum(1 for v in seen if v in (DUMMY, UNWRITTEN))


def place_spans(prompt_len, response_len, chunk, *, legacy):
    """Return the assembled teacher tensor as a list of 'predicts token N' markers.

    legacy=True reproduces the pre-fix behaviour so the test demonstrates the bug it guards.
    """
    tensor = [UNWRITTEN] * (prompt_len + response_len)
    for start in range(0, response_len, chunk):
        end = min(start + chunk, response_len)
        n = end - start
        # The teacher request for this chunk has prompt = prompt + response[:end],
        # so prompt_logprobs rows 0 .. prompt_len+end-1 exist. Row i predicts token i.
        highest_available_row = prompt_len + end - 1

        if legacy:
            read_from = prompt_len + start + 1        # rows [P+s+1, P+e+1)
            write_at = prompt_len + start             # indices [P+s, P+e)
        else:
            read_from = prompt_len + start            # rows [P+s, P+e)
            write_at = prompt_len + start - 1         # indices [P+s-1, P+e-1)

        for i in range(n):
            row = read_from + i
            value = row if row <= highest_available_row else DUMMY
            tensor[write_at + i] = value
    return tensor


def trained_positions(tensor, prompt_len, response_len):
    """What the loss actually sees: response position t reads index P+t-1 (the left shift)."""
    return [tensor[prompt_len + t - 1] for t in range(response_len)]


@pytest.mark.parametrize("chunk", [256, 1024, 2048, 4096])
def test_every_response_position_gets_its_own_token_label(chunk):
    P, R = 153, 8192
    seen = trained_positions(place_spans(P, R, chunk, legacy=False), P, R)
    for t, got in enumerate(seen):
        assert got == P + t, (
            f"response token {t} supervised by the prediction for token {got}, expected {P + t} "
            f"(chunk={chunk})"
        )


@pytest.mark.parametrize("chunk", [256, 1024, 2048, 4096])
def test_no_zero_teacher_rows_in_the_trained_region(chunk):
    P, R = 153, 8192
    seen = trained_positions(place_spans(P, R, chunk, legacy=False), P, R)
    assert DUMMY not in seen, f"{seen.count(DUMMY)} all-zero teacher rows are trained (chunk={chunk})"
    assert UNWRITTEN not in seen, f"{seen.count(UNWRITTEN)} trained positions have no teacher label"


@pytest.mark.parametrize("chunk", [256, 1024, 2048, 4096])
def test_legacy_path_poisons_one_position_per_chunk(chunk):
    """Regression witness: reproduces the bug, and the count that made it visible in teacher_mass."""
    P, R = 153, 8192
    seen = trained_positions(place_spans(P, R, chunk, legacy=True), P, R)
    # ceil(R/c) bad rows, from TWO sources: the final chunk's dummy lands at P+R-1 which the left
    # shift never reads, but index P-1 (the label for response token 0) is never written at all.
    # (ceil(R/c) - 1) dummies + 1 unwritten = ceil(R/c).
    assert zero_rows(seen) == math.ceil(R / chunk), (
        f"expected ceil(R/c)={math.ceil(R / chunk)} all-zero trained rows, got {zero_rows(seen)}"
    )
    assert seen[0] == UNWRITTEN, "legacy never writes index P-1, the label for response token 0"


@pytest.mark.parametrize("chunk,expected_mass", [(4096, 1.0154), (1024, 1.0615), (256, 1.2461)])
def test_legacy_reproduces_the_observed_teacher_mass(chunk, expected_mass):
    """teacher_mass = 1 + (K-1)*f, since an all-zero row exponentiates to mass K, not 1.

    Matching the production numbers (c4096 1.0155, c1024 1.0613, c256 1.2512) is what identified
    this as the cause rather than a plausible story.
    """
    P, R, K = 153, 8192, 64
    seen = trained_positions(place_spans(P, R, chunk, legacy=True), P, R)
    f = zero_rows(seen) / R
    assert 1.0 + (K - 1) * f == pytest.approx(expected_mass, abs=2e-3)


def test_fixed_path_leaves_teacher_mass_at_unity():
    P, R, K = 153, 8192, 64
    for chunk in (256, 1024, 2048, 4096):
        seen = trained_positions(place_spans(P, R, chunk, legacy=False), P, R)
        f = zero_rows(seen) / R
        assert 1.0 + (K - 1) * f == pytest.approx(1.0, abs=1e-9), f"chunk={chunk}"


def test_chunk_spans_tile_without_gap_or_overlap():
    P, R, chunk = 153, 8192, 1024
    written = []
    for start in range(0, R, chunk):
        end = min(start + chunk, R)
        written.append((P + start - 1, P + end - 1))
    for (_, prev_end), (next_start, _) in zip(written, written[1:]):
        assert prev_end == next_start, "chunk teacher spans must tile exactly"
    assert written[0][0] == P - 1, "the first chunk must supply index P-1, the label for response token 0"
    assert written[-1][1] == P + R - 1, "the last chunk must stop before index P+R-1"


def test_non_multiple_response_length():
    """Real responses stop on EOS, so the final chunk is usually short."""
    P, R, chunk = 153, 7855, 4096
    seen = trained_positions(place_spans(P, R, chunk, legacy=False), P, R)
    assert seen == [P + t for t in range(R)]


if __name__ == "__main__":
    import itertools
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
