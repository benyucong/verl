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
"""The two arms must ask the teacher for the SAME work.

The baseline (incremental=False, one teacher call per finished response) and chunk streaming
(incremental=True, one call per chunk) are compared on throughput. That comparison is only
meaningful if the teacher request differs solely in what is *semantically* required by
streaming. Any performance-relevant flag set on one arm and not the other is a handicap wearing
the costume of a result.

WHAT WENT WRONG. ``detokenize=False`` was set only on the incremental path. It looks like a
detail -- we read only ``.logprob``, ``.rank`` and the token-id keys out of ``prompt_logprobs``
(vllm_rollout/utils.py ``extract_prompt_logprobs``), never the decoded strings, so detokenizing
is wasted work for BOTH arms. But ``detokenize`` defaults to True, and in vLLM v1 it also decides
whether the *logprobs* processor receives a tokenizer::

    # vllm/v1/engine/output_processor.py
    if not sampling_params.detokenize:
        tokenizer = None
    logprobs_processor = LogprobsProcessor.from_new_request(tokenizer=tokenizer, ...)

With a tokenizer present, ``LogprobsProcessor._update_prompt_logprobs`` runs
``convert_ids_list_to_tokens`` across ``num_prompt_tokens * K`` ids and then a per-position UTF-8
correction pass. At K=64 over an 8192-token response that is ~533k string conversions per
request, in Python, on the engine's output path -- so the baseline was CPU-bound while streaming
was not.

Measured on BSC MN5 (acc), 2 teacher GPUs, 8B teacher, c4096:

    arm        uncached tok/s/GPU   teacher latency p50   in-flight median
    baseline                 3224               156.70 s               123
    streaming                7389                 1.13 s                 4

2.3x throughput from the same two GPUs, while the baseline was doing 35% LESS uncached work.
That asymmetry read as a +67% OPDFlow win under teacher scarcity.

THE INVARIANT: every performance-relevant teacher sampling parameter is identical across arms.
``skip_reading_prefix_cache`` is the one allowed difference -- streaming needs the cached prefix
KV so the server returns only the recomputed suffix, which is a correctness requirement of
chunked scoring, not a speed knob.
"""

import math
from types import SimpleNamespace

try:
    import pytest
except ImportError:  # runnable standalone: the cluster venv has no pytest
    class _Pytest:
        approx = staticmethod(lambda v, abs=0.0: v)

        class mark:
            @staticmethod
            def parametrize(_names, _values):
                def deco(fn):
                    fn._params = (_names, _values)
                    return fn
                return deco

    pytest = _Pytest()


# The one flag streaming is allowed to set on its own, because chunked scoring cannot work
# without it. Anything else appearing on one arm only is a handicap.
INCREMENTAL_ONLY = {"skip_reading_prefix_cache"}


def _params(incremental):
    """Call the real production builder -- no reimplementation, or the test guards nothing."""
    from verl.experimental.teacher_loop.teacher_manager import _get_teacher_sampling_params

    teacher_cfg = SimpleNamespace(inference=SimpleNamespace(temperature=1.0))
    loss_cfg = SimpleNamespace(topk=64, loss_settings=SimpleNamespace(use_topk=True))
    return _get_teacher_sampling_params(teacher_cfg, loss_cfg, incremental=incremental)


def test_detokenize_is_disabled_on_both_arms():
    """The regression itself: detokenize must not be an arm-specific flag."""
    for incremental in (False, True):
        p = _params(incremental)
        assert p.get("detokenize") is False, (
            f"incremental={incremental} would detokenize {p.get('prompt_logprobs')} logprobs per "
            f"position; with detokenize unset vLLM hands the LogprobsProcessor a tokenizer and the "
            f"arm pays ~num_prompt_tokens*K string conversions the other arm does not"
        )


def test_arms_differ_only_by_the_incremental_only_flag():
    base, incr = _params(False), _params(True)
    extra_on_incremental = set(incr) - set(base)
    extra_on_baseline = set(base) - set(incr)

    assert extra_on_incremental <= INCREMENTAL_ONLY, (
        f"streaming sets {sorted(extra_on_incremental - INCREMENTAL_ONLY)} that the baseline does "
        f"not; a throughput comparison across these arms would measure the flag, not the mechanism"
    )
    assert not extra_on_baseline, (
        f"baseline sets {sorted(extra_on_baseline)} that streaming does not"
    )

    for key in set(base) & set(incr):
        assert base[key] == incr[key], (
            f"shared parameter {key!r} differs across arms: baseline={base[key]!r} "
            f"streaming={incr[key]!r}"
        )


def test_streaming_still_reads_the_cached_prefix():
    """The allowed difference must actually still be there -- it is load-bearing for correctness."""
    assert _params(True).get("skip_reading_prefix_cache") is False, (
        "chunk streaming must read the cached prefix KV, otherwise the server does not return the "
        "recomputed suffix that span slicing depends on"
    )
    assert "skip_reading_prefix_cache" not in _params(False), (
        "the baseline scores one full response per call and has no prefix to reuse beyond the "
        "prompt; leaving the flag unset keeps it on vLLM's default path"
    )


def test_topk_and_max_tokens_are_shared():
    base, incr = _params(False), _params(True)
    for key, want in (("max_tokens", 1), ("prompt_logprobs", 64), ("temperature", 1.0)):
        assert base[key] == want and incr[key] == want, (
            f"{key} must be {want} on both arms, got baseline={base[key]!r} streaming={incr[key]!r}"
        )


def test_detokenization_cost_that_the_flag_avoids():
    """Documents the magnitude, so the reason the flag matters is not lost again.

    Per request the skipped work is num_prompt_tokens * K id->string conversions.
    """
    prompt_len, response_len, k = 136, 8192, 64
    per_request = (prompt_len + response_len) * k
    per_run = per_request * 3840  # parents per run in the acc configuration
    assert per_request == 532_992, f"{per_request} id->string conversions per request"
    assert math.isclose(per_run / 1e9, 2.047, abs_tol=0.01), (
        f"{per_run / 1e9:.3f}e9 string conversions per run avoided per arm"
    )


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
