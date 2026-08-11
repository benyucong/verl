"""The windowed-teacher contract: what the engine MATERIALISES must cover what the server PARSES.

Three components have to agree, and they are in three different files:

  * the engine   (vllm/v1/engine/logprobs.py, patched) builds Logprob rows only inside
    ``prompt_logprobs_range`` and leaves every other position ``None``
  * the server   (vllm_async_server.py) extracts ``[max(num_cached+1, 1, window_start), full_len)``
    and ``extract_incremental_prompt_logprobs`` raises on a ``None`` anywhere in that range
  * the caller   (teacher_manager.py) requests the window, then slices its span out of whatever
    the server returned using ``shift_start_abs - valid_suffix_start_abs``

The trap that makes this worth a test: ``num_cached_tokens`` is block-aligned DOWNWARD and is only
known AFTER the request runs, so in the normal case it lands slightly BELOW the window start. Without
the server-side clamp, every single windowed call would try to parse rows the engine deliberately did
not build -- a hard ValueError on the happy path, not a rare edge case.

This simulates all three in plain Python so the invariant is checked without a GPU. It asserts the
thing that actually matters: the teacher labels a chunk receives are the rows for absolute positions
[P+span_start, P+span_end), whether the call went down the windowed path or fell back -- and that
turning the window off reproduces the old behaviour exactly.
"""

import pytest


class _Engine:
    """prompt_logprobs[p] scores absolute token p; index 0 is None by convention."""

    def __init__(self, seq_len, window=None):
        self.rows = [None if p == 0 else ("row", p) for p in range(seq_len)]
        if window is not None:
            lo, hi = window
            for p in range(seq_len):
                if not (lo <= p < hi):
                    self.rows[p] = None


def _server_extract(engine, seq_len, num_cached, window):
    """vllm_async_server.py's incremental branch, including the clamp under test."""
    start = max(num_cached + 1, 1)
    if window is not None:
        start = max(start, int(window[0]))
    if start >= seq_len:
        return [], seq_len, seq_len            # fully cached -> empty suffix, caller falls back
    rows = []
    for p in range(start, seq_len):
        if engine.rows[p] is None:
            raise ValueError(f"prompt position {p}: None (uncomputed) -- incremental span not covered")
        rows.append(engine.rows[p])
    return rows, start, seq_len


def _client_span(prompt_width, span_start, span_end, num_cached, use_window):
    """teacher_manager.py: request the window, then slice the span (or fall back)."""
    seq_len = prompt_width + span_end                      # request prompt is prompt+response[:e]
    shift_start_abs = prompt_width + span_start
    shift_end_abs = prompt_width + span_end
    window = (shift_start_abs, seq_len) if use_window else None

    engine = _Engine(seq_len, window)
    suffix, valid_start, valid_end = _server_extract(engine, seq_len, num_cached, window)

    covered_end = min(shift_end_abs, valid_end)
    if not (valid_start <= shift_start_abs and covered_end <= valid_end):
        # Clean recompute: no window is ever set on this arm, so every row exists.
        full = _Engine(seq_len, None)
        return [full.rows[p] for p in range(shift_start_abs, shift_end_abs)], "fallback"

    lo = shift_start_abs - valid_start
    hi = covered_end - valid_start
    return suffix[lo:hi], "windowed" if use_window else "incremental"


P, SPAN = 512, 512
CASES = [
    # (num_cached, what it represents)
    (0, "cold: nothing cached"),
    (P - 1, "prompt cached, response not"),
    (P + SPAN - 1, "cached right up to the window start"),
    (P + SPAN, "cached exactly AT the window start"),
    (P + SPAN - 7, "block-aligned DOWN below the window start -- the normal case"),
    (P + SPAN + 3, "cross-response interference: cached PAST the span start"),
]


@pytest.mark.parametrize("num_cached,label", CASES)
@pytest.mark.parametrize("chunk", [0, 1, 3])
def test_window_never_strands_a_row(num_cached, label, chunk):
    """Whatever the cache state, the chunk gets exactly its own span and nothing raises."""
    span_start, span_end = chunk * SPAN, (chunk + 1) * SPAN
    expected = [("row", p) for p in range(P + span_start, P + span_end)]

    windowed, how = _client_span(P, span_start, span_end, num_cached, use_window=True)
    assert windowed == expected, f"{label}: windowed path returned the wrong rows via {how}"

    # And the window must not change WHICH rows the chunk trains on -- only how many were built.
    plain, _ = _client_span(P, span_start, span_end, num_cached, use_window=False)
    assert windowed == plain, f"{label}: window changed the labels (chunk {chunk})"


def test_clamp_is_what_prevents_the_crash():
    """Guard the guard: without the clamp, the normal cache state parses unbuilt rows."""
    span_start, span_end = SPAN, 2 * SPAN
    seq_len = P + span_end
    num_cached = P + span_start - 7          # block-aligned just below the window start
    window = (P + span_start, seq_len)
    engine = _Engine(seq_len, window)

    with pytest.raises(ValueError, match="None \\(uncomputed\\)"):
        # the pre-clamp server: start = max(num_cached+1, 1), window ignored
        start = max(num_cached + 1, 1)
        for p in range(start, seq_len):
            if engine.rows[p] is None:
                raise ValueError(f"prompt position {p}: None (uncomputed)")

    # with the clamp, the same request is fine
    rows, start, _ = _server_extract(engine, seq_len, num_cached, window)
    assert start == P + span_start and len(rows) == span_end - span_start


def test_fallback_still_fires_on_cross_response_interference():
    """The window must not mask the case the clean-recompute fallback exists for."""
    rows, how = _client_span(P, 0, SPAN, num_cached=P + 3, use_window=True)
    assert how == "fallback"
    assert rows == [("row", p) for p in range(P, P + SPAN)]
