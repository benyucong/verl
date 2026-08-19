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
"""OmniOPD's producer side: a finished trajectory -> (anchors, k_sem) for the objective.

Three steps, and each is a place the algorithm can be changed by accident:

    1  SELECT   M non-overlapping C-token chunks at the highest-entropy anchors
    2  GENERATE the teacher writes N continuations of C tokens from each chunk's prefix
    3  SCORE    phi compares each continuation against the student's own chunk -> k_sem

The loss (`compute_distillation_loss_omniopd`) consumes only the (anchors, k_sem) this returns, so
everything the objective depends on upstream of the trainer is decided here.

PHI IS BYTE-IDENTICAL TO THE OFFLINE IMPLEMENTATION, NOT MERELY EQUIVALENT. k_sem is a sum of phi
over N continuations and multiplies the chunk log-likelihood, so a phi that differs anywhere in the
fourth decimal changes the target, which changes the gradient, silently. scripts/omniopd_a0.py,
omniopd_smoke.py and the offline planners all use the function below verbatim, and
scripts/test_omniopd_producer.py asserts they agree character for character.

WHICH ENTROPY SELECTS is a policy choice, not a detail, and this module does not make it. The caller
passes a signal. Two providers exist and they DO NOT agree:

    post_eos_actor_forward  the published selector; exact full-vocabulary entropy from an actor
                            forward over the finished trajectory (OMNIOPD_PUBLISHED)
    rollout_exact_scalar    entropy emitted by the patched vLLM during decode
                            (OMNIOPD_ONLINE_VARIANT, its own selector_hash)

Gate 6 measured 9 of 16 responses selecting the same chunk set. Anchor 150 and anchor 151 are
different chunks, never a partial match, because their teacher prefixes end at different tokens. A
run using the online signal is therefore a declared VARIANT and must be reported as such.
"""

from typing import Any, Callable, Optional, Sequence

def _chunks():
    """Imported lazily. omniopd_chunks pulls in torch at module level for the entropy helper, but
    select_chunks and chunk_prefix are pure Python -- and the selection and phi logic here is
    exactly what most benefits from being testable without a GPU stack installed."""
    try:                                       # as a package member
        from . import omniopd_chunks as m
    except ImportError:                        # or with the distillation dir on sys.path
        import omniopd_chunks as m
    return m


def ned_char_maxlen(a: str, b: str, cap: int = 4000) -> float:
    """FID-7. Normalised character edit distance, 1.0 = identical.

    Verbatim from the offline implementations. Do not "optimise": the cap, the max-length
    normalisation and the empty-string cases all enter k_sem.
    """
    a, b = a[:cap], b[:cap]
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return 1.0 - prev[-1] / max(len(a), len(b))


PHI = {"ned": ned_char_maxlen}


def get_phi(name: str) -> Callable[[str, str], float]:
    if name not in PHI:
        raise ValueError(
            f"omniopd.phi={name!r} has no implementation. Available: {sorted(PHI)}. A missing phi "
            f"must raise rather than fall back -- k_sem computed by a different metric is a "
            f"different objective, and nothing downstream would reveal it.")
    return PHI[name]


def select_anchors(signal: Sequence[float], prompt_len: int, resp_len: int, M: int,
                   C: int) -> list[int]:
    """M non-overlapping C-token anchors, highest signal first.

    `signal` is indexed over the WHOLE sequence: response position t is scored by
    signal[prompt_len + t - 1], the distribution that produced y_t rather than the one it
    conditions. Returns anchors sorted ascending; fewer than M means fewer than M fit, which is
    recorded rather than padded with positions the algorithm would not have chosen.
    """
    return [t for t, _ in _chunks().select_chunks(signal, prompt_len, resp_len, M, C)]


def k_sem_for_chunk(student_chunk_text: str, continuation_texts: Sequence[str], N: int,
                    phi: Callable[[str, str], float]) -> float:
    """k_sem = sum of phi over the N continuations.

    A SHORT candidate list is refused, not tolerated. k_sem from fewer than N continuations is
    numerically indistinguishable from a teacher that disagreed on the missing ones, and it feeds
    straight into pi_hat -- so an aborted teacher request would quietly weaken exactly the chunks
    it failed on.
    """
    if len(continuation_texts) != N:
        raise ValueError(
            f"k_sem needs exactly N={N} continuations, got {len(continuation_texts)}. A truncated "
            f"candidate set is indistinguishable from teacher disagreement once summed.")
    return float(sum(phi(student_chunk_text, c) for c in continuation_texts))


def build_chunk_requests(prompt_ids: Sequence[int], response_ids: Sequence[int],
                         anchors: Sequence[int]) -> list[list[int]]:
    """The exact prefixes the teacher is given, one per anchor.

    Each stops one token BEFORE its audited span: the teacher may never see the tokens it is being
    asked to produce independently (FID-8). Emitted in ascending anchor order because the prefixes
    are NESTED -- chunk k+1 extends chunk k -- so issuing them in order lets each prefill reuse the
    previous one's KV. Scattered, every chunk re-ingests a prefix that grows with the response,
    which is the dominant cost of generative teaching.
    """
    cp = _chunks().chunk_prefix
    return [cp(list(prompt_ids), list(response_ids), t0) for t0 in anchors]


def assemble_omniopd_record(prompt_ids: Sequence[int], response_ids: Sequence[int],
                            anchors: Sequence[int], continuations: Sequence[Sequence[str]],
                            tokenizer: Any, N: int, C: int, phi_name: str = "ned",
                            decode_cache: Optional[dict] = None) -> dict:
    """(anchors, k_sem) for one trajectory, ready for the loss.

    `continuations[i]` are the N texts the teacher wrote from `anchors[i]`'s prefix. The student's
    own chunk is decoded with the SAME tokenizer the teacher used -- phi compares characters, so a
    tokenizer mismatch would change k_sem without changing anything visible.
    """
    phi = get_phi(phi_name)
    if len(continuations) != len(anchors):
        raise ValueError(f"{len(anchors)} anchors but {len(continuations)} continuation sets")
    k_sem = []
    for t0, cont in zip(anchors, continuations, strict=True):
        if t0 + C > len(response_ids):
            raise ValueError(f"anchor {t0} + C {C} runs past response {len(response_ids)}")
        key = (t0, C)
        if decode_cache is not None and key in decode_cache:
            truth = decode_cache[key]
        else:
            truth = tokenizer.decode(list(response_ids[t0 : t0 + C]), skip_special_tokens=True)
            if decode_cache is not None:
                decode_cache[key] = truth
        k_sem.append(k_sem_for_chunk(truth, cont, N, phi))
    return {"omniopd_anchors": list(anchors), "omniopd_k_sem": k_sem}
