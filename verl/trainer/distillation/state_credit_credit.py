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

import math

import numpy as np

logger = logging.getLogger(__name__)


def _phi_terminal(reward_row) -> float:
    """Phi(s_T) is the task reward: the trajectory already ran to an answer, so continuations from a
    finished state would estimate something already known exactly."""
    return float(reward_row)


def _noise_floor(phi_obs, group_sizes, n_sampled, M=None) -> float:
    """E|centered| if Phi carried nothing but its own sampling noise.

    Phi-hat is a mean of M Bernoulli continuations, so a single state's estimate has sampling
    variance p(1-p)/M at the observed base rate. A credited chunk is a DIFFERENCE of Phi terms:
    interior chunks difference two Phi-hats (variance 2p(1-p)/M), while the terminal chunk
    differences one Phi-hat against the verifier's exact reward (variance p(1-p)/M). Leave-one-out
    against k-1 others inflates by sqrt(1 + 1/(k-1)), and E|X| = sd*sqrt(2/pi) for centered normal.

    Compare credit_abs_mean against this. At or below it the credit is the teacher's sampling noise
    dressed as supervision -- rows_credited can be perfectly healthy and still mean nothing. This is
    the whole reason the diagnostic exists: at M=4 the floor is ~0.2, which is the same order as a
    real effect, so the comparison is not optional.

    Takes the Phi values themselves, NOT the deltas -- a delta is signed and would drive p(1-p)
    negative. M is recovered from Phi's lattice rather than passed in, so this stays honest if M
    changes.
    """
    if not phi_obs or not group_sizes or not n_sampled:
        return 0.0
    p = float(np.mean(phi_obs))
    if not 0.0 <= p <= 1.0:
        # Clamping here would silently return 0.0 and read as "no noise floor", which is exactly
        # backwards. Phi is an accuracy; anything outside [0,1] means deltas reached this function.
        raise ValueError(
            f"noise floor needs Phi values in [0,1], got mean {p:.4f}. Deltas were probably passed "
            f"instead of Phi -- the floor would silently collapse to 0 and every credit would look "
            f"like signal.")
    if M is None:
        # Fallback only. The lattice cannot identify M when Phi happens to take extreme values --
        # an all-{0,1} group is consistent with M=1 and would inflate the floor several-fold. Pass
        # the configured M whenever it is known.
        M = 4
        probe = phi_obs[:64]
        for cand in (1, 2, 4, 8, 16, 32, 64):
            if max(abs(v * cand - round(v * cand)) for v in probe) < 1e-6:
                M = cand
                break
    M = max(int(M), 1)
    sd = math.sqrt(float(np.mean(n_sampled)) * p * (1.0 - p) / M)
    sd *= math.sqrt(1.0 + 1.0 / max(float(np.mean(group_sizes)) - 1.0, 1.0))
    return sd * math.sqrt(2.0 / math.pi)


def build_state_credit_weights(
    *,
    uids,                     # (n_rows,) problem id per row -- the LOO grouping key
    phis,                     # (n_rows,) list of per-depth Phi, ragged
    depths,                   # (n_rows,) list of the depths each row actually reached
    terminal,                 # (n_rows,) task reward = Phi(s_T)
    response_lengths,         # (n_rows,) true token count, NOT response_length
    max_response_len: int,
    min_survivors: int = 3,
    M: int | None = None,     # continuations per state; None => infer from Phi's lattice
):
    """Returns (W, stats). W is (n_rows, max_response_len) float32, detached by construction."""
    n = len(uids)
    W = np.zeros((n, max_response_len), dtype=np.float32)
    stats = {"rows": n, "rows_credited": 0, "chunks_total": 0, "chunks_dropped_small_group": 0,
             "chunks_dropped_nan": 0, "groups": 0, "groups_too_small": 0}

    # ---- per row: the chunk boundaries and the RAW progress of each chunk ----------------------
    # Chunk j spans (k_{j-1}, k_j]; the last chunk runs from the deepest reached depth to the end,
    # and its Phi target is the terminal reward.
    raw = []          # list of (row, [(lo, hi, delta_raw, n_sampled)])
    phi_obs: list[float] = []   # the Phi values themselves, for the sampling-noise floor
    for b in range(n):
        d = [int(x) for x in (depths[b] or [])]
        p = [float(x) for x in (phis[b] or [])]
        phi_obs.extend(v for v in p if v == v)
        T = int(response_lengths[b])
        if len(d) != len(p):
            raise ValueError(f"row {b}: {len(d)} depths but {len(p)} Phi values")
        # Phi(s_0) is not needed: it is identical across a problem's rollouts and cancels under LOO.
        # So the FIRST chunk has no defined progress and is not credited.
        bounds, prev_phi, prev_k = [], None, 0
        for k, ph in zip(d, p):
            k = min(k, T)
            if prev_phi is not None and k > prev_k:
                # interior: BOTH ends are Phi-hats, so this delta carries two draws of noise
                bounds.append((prev_k, k, ph - prev_phi, 2))
            prev_phi, prev_k = ph, k
        if prev_phi is not None and T > prev_k:
            # terminal: the far end is the verifier's exact reward, so only ONE sampled term
            bounds.append((prev_k, T, _phi_terminal(terminal[b]) - prev_phi, 1))
        raw.append(bounds)
        stats["chunks_total"] += len(bounds)

    # ---- leave-one-out per (problem, chunk index) ----------------------------------------------
    groups: dict = {}
    for b, u in enumerate(uids):
        groups.setdefault(str(u), []).append(b)
    stats["groups"] = len(groups)

    centered_all: list[float] = []
    n_sampled: list[int] = []
    group_sizes: list[int] = []
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
            group_sizes.append(k)
            n_sampled.extend(raw[b][j][3] for b, _ in vals)
            for b, v in vals:
                # leave-one-out: this row's delta against the mean of the OTHERS
                baseline = (total - v) / (k - 1)
                centered = v - baseline
                centered_all.append(centered)
                lo, hi, _, _ = raw[b][j]
                m_b = len(raw[b])
                l_j = max(1, hi - lo)
                hi_c = min(hi, max_response_len)
                if hi_c > lo:
                    W[b, lo:hi_c] = centered / (m_b * l_j)
    stats["rows_credited"] = int((np.abs(W).sum(axis=1) > 0).sum())
    # Magnitude of the centered credit, BEFORE the 1/(m_b*l_j) spread over tokens. Without this the
    # objective cannot be told apart from its own sampling noise: Phi is a mean over M continuations,
    # so at M=4 a single state's Phi-hat has sd ~= sqrt(p(1-p)/4) ~= 0.24 at p=0.65, and
    # leave-one-out against k-1 others gives sd(centered) ~= sd*sqrt(1 + 1/(k-1)) -- about 0.26 at
    # k=8. A credit_abs_mean near 0.8*that is pure noise dressed as supervision; real between-rollout
    # variation in Phi has to clear it. Reported unweighted so it stays comparable across chunk
    # lengths and across M.
    if centered_all:
        arr = np.asarray(centered_all, dtype=np.float64)
        stats["credit_abs_mean"] = float(np.abs(arr).mean())
        stats["credit_sd"] = float(arr.std())
        stats["credit_noise_floor"] = float(_noise_floor(phi_obs, group_sizes, n_sampled, M))
    return W, stats

# ---------------------------------------------------------------------------------------------
# DRIVER-SIDE ENTRY POINT
#
# Shared by BOTH trainers. The async path calls it from separation/ray_trainer.py; the
# synchronous path (trainer/ppo/ray_trainer.py) calls it after token_level_scores lands. It has
# to live in the DRIVER either way: leave-one-out centres a rollout against its siblings, and
# once a batch is dispatched to workers each rank holds only its own slice, so the siblings are
# gone. Row ORDER does not matter here -- grouping is by uid, and W is stored on the batch so
# any later reordering carries it along.
# ---------------------------------------------------------------------------------------------

def attach_state_credit_weights(batch, config, metrics: dict) -> None:
    """Turn per-depth Phi into the dense weight the State-Credit loss consumes.

    No-op unless the loss actually asks for teacher continuations, so every other objective is
    byte-identical.
    """
    try:
        from verl.trainer.distillation.losses import get_distillation_loss_settings
        lm = config.distillation.distillation_loss.loss_mode
        if not get_distillation_loss_settings(str(lm)).use_teacher_continuations:
            return
    except Exception:
        return

    import torch as _torch

    ntb = batch.non_tensor_batch
    need = ("state_credit_phi", "state_credit_depths")
    missing = [k for k in need if k not in ntb]
    if missing:
        raise KeyError(
            f"state-credit loss is configured but {missing} is absent from the batch. Phi is "
            f"produced in the agent loop (state_credit_stage.attach_state_credit); if that did "
            f"not run, there is no credit signal and training must not proceed without one.")

    resp_mask = batch.batch["response_mask"]
    lengths = resp_mask.sum(dim=-1).tolist()
    # Phi(s_T) is the task reward, already on the batch as the per-token score
    terminal = batch.batch["token_level_scores"].sum(dim=-1).tolist()
    uids = ntb["uid"] if "uid" in ntb else np.arange(len(lengths))
    sc = config.distillation.state_credit

    W, stats = build_state_credit_weights(
        uids=list(uids),
        phis=list(ntb["state_credit_phi"]),
        depths=list(ntb["state_credit_depths"]),
        terminal=terminal,
        response_lengths=lengths,
        max_response_len=int(resp_mask.shape[1]),
        min_survivors=int(sc.min_survivors),
        M=int(sc.M),   # known; the noise floor must not have to guess it
    )
    batch.batch["state_credit_weights"] = _torch.as_tensor(
        W, dtype=_torch.float32, device=resp_mask.device)
    for k, v in stats.items():
        metrics[f"state_credit/{k}"] = v
    print(f"[STATE-CREDIT] rows={stats['rows']} credited={stats['rows_credited']} "
          f"chunks={stats['chunks_total']} dropped_small_group="
          f"{stats['chunks_dropped_small_group']} groups={stats['groups']}", flush=True)

