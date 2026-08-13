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
"""Chunk selection for OmniOPD: which spans of a trajectory the teacher is asked to re-write.

FID-1: selection is argmax_M over the student's EXACT FULL-VOCABULARY entropy. Entropy computed
from a truncated top-k tail is systematically under-estimated, and by a different amount at
different positions, so it selects DIFFERENT chunks. That changes which spans get audited -- the
algorithm -- rather than how fast the same algorithm runs. Any speed-up bought by truncating here
would be measuring a different method.

Full-vocabulary entropy at every position is a (T, V) tensor, ~1.3 GiB in fp32 for one 2178-token
trajectory at V=151936, and it is needed for the whole batch. This computes it in position slices
from hidden states and keeps only the (T,) result, so peak memory is set by `slice_size` rather
than by trajectory length. Selection is inference-only, so nothing is retained for backward.

`select_chunks` is a VERBATIM port of scripts/omniopd_plan.py, which produced the manifests that
milestones 1 and 2 validated against. It is not re-derived here: the tie-breaking is load-bearing
(`sort(reverse=True)` on `(H, t)` tuples resolves equal entropy toward the LARGER t), and a
plausible rewrite that broke ties the other way would silently audit different spans. The offline
implementation is the oracle, and scripts/test_omniopd_chunks.py asserts the two agree.
"""

from typing import Callable, Optional

import torch


def exact_entropy_from_hidden(
    hidden: torch.Tensor,
    lm_head: Callable[[torch.Tensor], torch.Tensor],
    slice_size: int = 512,
) -> torch.Tensor:
    """Exact full-vocabulary per-position entropy, computed in slices.

    Args:
        hidden: (T, H) hidden states for ONE sequence, from the model body.
        lm_head: maps (S, H) -> (S, V). Applied per slice so (T, V) is never materialised.
        slice_size: positions per slice; sets peak memory, not the result.

    Returns:
        (T,) float32 entropy in nats. H[i] is the entropy of the distribution at position i, i.e.
        the uncertainty the model had when it emitted token i+1.
    """
    out = torch.empty(hidden.shape[0], dtype=torch.float32, device=hidden.device)
    with torch.no_grad():
        for i in range(0, hidden.shape[0], slice_size):
            logits = lm_head(hidden[i : i + slice_size]).float()
            logp = torch.log_softmax(logits, dim=-1)
            out[i : i + slice_size] = -(logp.exp() * logp).sum(-1)
            del logits, logp
    return out


def select_chunks(
    H,
    prompt_len: int,
    resp_len: int,
    M: int,
    C: int,
) -> list[tuple[int, float]]:
    """M non-overlapping C-token chunks anchored at the highest-entropy response positions.

    VERBATIM port of scripts/omniopd_plan.py:select_chunks. Do not "clean up": the ordering and
    tie-breaking decide which spans are audited, and the offline manifests that milestones 1 and 2
    validated were produced by exactly this logic.

    H is indexed over the whole sequence; response position t corresponds to H[prompt_len + t - 1]
    (the logits that produced y_t). Anchors are taken greedily by descending entropy, skipping any
    that would overlap an already-chosen chunk or run past the end of the response.

    Returns [(anchor_token, entropy), ...] sorted by anchor. Fewer than M entries means fewer than M
    non-overlapping chunks fit; that is recorded rather than padded with low-entropy positions the
    algorithm would not have chosen.
    """
    scored = []
    for t in range(resp_len):
        h_idx = prompt_len + t - 1
        if 0 <= h_idx < len(H):
            scored.append((H[h_idx], t))
    scored.sort(reverse=True)
    chosen = []
    for h, t in scored:
        if t + C > resp_len:
            continue
        if any(abs(t - u) < C for u, _ in chosen):
            continue
        chosen.append((t, h))
        if len(chosen) == M:
            break
    return sorted(chosen)


def select_chunks_for_sequence(
    hidden: torch.Tensor,
    lm_head: Callable[[torch.Tensor], torch.Tensor],
    prompt_len: int,
    resp_len: int,
    M: int,
    C: int,
    slice_size: int = 512,
    entropy_topk: int = 0,
) -> tuple[list[tuple[int, float]], torch.Tensor]:
    """Entropy + selection for one sequence. Returns (anchors, entropy).

    entropy_topk != 0 is an explicit, recorded deviation from FID-1 and is refused here rather than
    silently honoured: a caller that wants truncated entropy is choosing a different algorithm and
    must say so somewhere it will be seen, not by passing an argument.
    """
    if entropy_topk != 0:
        raise NotImplementedError(
            f"entropy_topk={entropy_topk}: chunk selection is argmax_M over FULL-vocabulary entropy "
            f"(FID-1). A truncated tail under-estimates entropy unevenly and selects different "
            f"chunks, which changes the algorithm rather than its cost."
        )
    if prompt_len + resp_len > hidden.shape[0]:
        raise ValueError(
            f"prompt_len({prompt_len}) + resp_len({resp_len}) exceeds the {hidden.shape[0]} "
            f"positions available; the hidden states do not cover this sequence."
        )
    H = exact_entropy_from_hidden(hidden, lm_head, slice_size)
    return select_chunks(H.tolist(), prompt_len, resp_len, M, C), H


def chunk_prefix(prompt_ids: list[int], response_ids: list[int], anchor: int) -> list[int]:
    """The exact prefix the teacher is given for a chunk anchored at `anchor`.

    Stops one token BEFORE the audited span. The teacher may never see the tokens it is being asked
    to produce independently -- that is the whole basis of comparing its continuation against the
    student's (FID-8, chunk_placement="anchor_starts").
    """
    return list(prompt_ids) + list(response_ids[:anchor])
