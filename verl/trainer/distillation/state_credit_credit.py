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
"""Turn per-depth Phi into a dense per-token credit weight.

    Delta_j      = Phi(s_{k_j}) - Phi(s_{k_{j-1}})          progress made by chunk j
    Delta~_j     = Delta_j - mean_{other rollouts} Delta_j   leave-one-out centering
    W[b,t]       = Delta~_j / (m_b * l_j)                    for t in chunk j of row b

so that sum_t W[b,t]*(-log pi) reproduces -(1/m_b) sum_j Delta~_j * logbar_j, with logbar the
chunk's MEAN log-prob.

WHY THIS LIVES IN THE DRIVER AND NOT THE LOSS. Centering is across the rollouts OF ONE PROBLEM, and
by the time a micro-batch reaches the loss those siblings are gone: balance_batch reorders rows onto
DP ranks BY LENGTH -- so two rollouts of one problem with different lengths land on different GPUs
-- then mini-batching slices, then dynamic-bsz repartitions again by token workload. A micro-batch
is an arbitrary length-selected subset of one rank's slice. Here the whole batch is present and
grouping is by key, and the dense weight that comes out is row-local, hence exactly invariant to all
three later reorderings.

WHY A MISSING BASELINE DROPS THE CHUNK RATHER THAN CENTERING AGAINST ZERO. With too few surviving
siblings there is no estimate of "what this problem's chunks usually achieve". Substituting 0 does
not degrade gracefully -- it converts an unknown baseline into a maximal-magnitude target, so the
rows with the least information would exert the most pull.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _phi_terminal(reward_row) -> float:
    """Phi(s_T) is the task reward: the trajectory already ran to an answer, so continuations from a
    finished state would estimate something already known exactly."""
    return float(reward_row)


def build_state_credit_weights(
    *,
    uids,                     # (n_rows,) problem id per row -- the LOO grouping key
    phis,                     # (n_rows,) list of per-depth Phi, ragged
    depths,                   # (n_rows,) list of the depths each row actually reached
    terminal,                 # (n_rows,) task reward = Phi(s_T)
    response_lengths,         # (n_rows,) true token count, NOT response_length
    max_response_len: int,
    min_survivors: int = 3,
):
    """Returns (W, stats). W is (n_rows, max_response_len) float32, detached by construction."""
    n = len(uids)
    W = np.zeros((n, max_response_len), dtype=np.float32)
    stats = {"rows": n, "rows_credited": 0, "chunks_total": 0, "chunks_dropped_small_group": 0,
             "chunks_dropped_nan": 0, "groups": 0, "groups_too_small": 0}

    # ---- per row: the chunk boundaries and the RAW progress of each chunk ----------------------
    # Chunk j spans (k_{j-1}, k_j]; the last chunk runs from the deepest reached depth to the end,
    # and its Phi target is the terminal reward.
    raw = []          # list of (row, [(lo, hi, delta_raw)])
    for b in range(n):
        d = [int(x) for x in (depths[b] or [])]
        p = [float(x) for x in (phis[b] or [])]
        T = int(response_lengths[b])
        if len(d) != len(p):
            raise ValueError(f"row {b}: {len(d)} depths but {len(p)} Phi values")
        # Phi(s_0) is not needed: it is identical across a problem's rollouts and cancels under LOO.
        # So the FIRST chunk has no defined progress and is not credited.
        bounds, prev_phi, prev_k = [], None, 0
        for k, ph in zip(d, p):
            k = min(k, T)
            if prev_phi is not None and k > prev_k:
                bounds.append((prev_k, k, ph - prev_phi))
            prev_phi, prev_k = ph, k
        if prev_phi is not None and T > prev_k:
            bounds.append((prev_k, T, _phi_terminal(terminal[b]) - prev_phi))
        raw.append(bounds)
        stats["chunks_total"] += len(bounds)

    # ---- leave-one-out per (problem, chunk index) ----------------------------------------------
    groups: dict = {}
    for b, u in enumerate(uids):
        groups.setdefault(str(u), []).append(b)
    stats["groups"] = len(groups)

    for _u, rows in groups.items():
        depth_count = max((len(raw[b]) for b in rows), default=0)
        for j in range(depth_count):
            vals = [(b, raw[b][j][2]) for b in rows
                    if j < len(raw[b]) and raw[b][j][2] == raw[b][j][2]]   # drop NaN
            if len(vals) < min_survivors:
                stats["chunks_dropped_small_group"] += len(vals)
                if len(vals):
                    stats["groups_too_small"] += 1
                continue
            total = sum(v for _, v in vals)
            k = len(vals)
            for b, v in vals:
                # leave-one-out: this row's delta against the mean of the OTHERS
                baseline = (total - v) / (k - 1)
                centered = v - baseline
                lo, hi, _ = raw[b][j]
                m_b = len(raw[b])
                l_j = max(1, hi - lo)
                hi_c = min(hi, max_response_len)
                if hi_c > lo:
                    W[b, lo:hi_c] = centered / (m_b * l_j)
    stats["rows_credited"] = int((np.abs(W).sum(axis=1) > 0).sum())
    return W, stats
