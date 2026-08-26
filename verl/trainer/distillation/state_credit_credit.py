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
    Delta~_j     = Delta_j - mean_{other rollouts of the SAME transition} Delta_j    (leave-one-out)
    W[b,t]       = Delta~_{j(t)}                             for t in chunk j of row b

W is the state term of the combined advantage u_t = a_t^OPD + beta * W[b,t] (Sec 7). It is
broadcast to the chunk's tokens unchanged; the 1/T normalisation belongs to the loss aggregator.

credit_norm="per_chunk" instead spreads it as Delta~_j / (m_b * l_j), so summing over a row gives
-(1/m_b) sum_j Delta~_j * logbar_j with logbar the chunk's MEAN log-prob. That is the Sec 7.1
equal-per-chunk objective: it weights a short terminal chunk's tokens more heavily than a full
fixed-length chunk's, which is why it is the ablation and not the default.

CENTERING IS PER TRANSITION, not per ordinal chunk index -- see the comment at the grouping loop.

Phi(s_0) is never computed. It is one constant per problem, so it cancels exactly out of the
leave-one-out centering of the first chunk -- which means chunk 1 is credited for FREE, not skipped.
See the note at the boundary construction.

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
    credit_norm: str = "broadcast",
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
    raw = []          # list of (row, [(lo, hi, delta_raw, n_sampled, transition_key)])
    phi_obs: list[float] = []   # the Phi values themselves, for the sampling-noise floor
    for b in range(n):
        d = [int(x) for x in (depths[b] or [])]
        p = [float(x) for x in (phis[b] or [])]
        phi_obs.extend(v for v in p if v == v)
        T = int(response_lengths[b])
        if len(d) != len(p):
            raise ValueError(f"row {b}: {len(d)} depths but {len(p)} Phi values")
        # THE FIRST CHUNK (0 -> d_1). Its progress is Phi(s_{d_1}) - Phi(s_0), and s_0 = (x, empty)
        # is the SAME state for every rollout of a problem -- so Phi(s_0) is one constant per group.
        # Leave-one-out subtracts a mean of the same quantity over siblings, and adding a constant to
        # every member of a group leaves the centered value untouched:
        #
        #   G~_i = [Phi_i(d_1) - Phi(s_0)] - mean_{n!=i}[Phi_n(d_1) - Phi(s_0)]
        #        =  Phi_i(d_1)            - mean_{n!=i} Phi_n(d_1)
        #
        # So the centered credit for chunk 1 is available WITHOUT ever running continuations from
        # s_0. That is why the raw delta stored here is Phi_i(d_1) alone. It is not the true Delta_1
        # -- it is offset by the unknown constant Phi(s_0) -- which matters only for the uncentered
        # diagnostic, never for W.
        #
        # Dropping this chunk instead (what the earlier version did, on the same cancellation
        # argument) left the first d_1 tokens of EVERY trajectory with zero state credit: a quarter
        # of a median response here, and for a trajectory that reaches only one depth, half its
        # chunks. Cancellation is the reason chunk 1 is FREE, not the reason to skip it.
        #
        # n_sampled=1: Phi(s_0) contributes no variance to the centered value because it cancels, so
        # this delta carries one sampled term, not two.
        bounds, prev_phi, prev_k = [], None, 0
        for k, ph in zip(d, p):
            k = min(k, T)
            if prev_phi is None:
                if k > 0:
                    bounds.append((0, k, ph, 1, ("i", 0, k)))
            elif k > prev_k:
                # interior: BOTH ends are Phi-hats, so this delta carries two draws of noise.
                # Keyed by the DEPTH PAIR, which comes from the fixed schedule and so means the same
                # transition ("2560 -> 5120") in every trajectory that reached it.
                bounds.append((prev_k, k, ph - prev_phi, 2, ("i", prev_k, k)))
            prev_phi, prev_k = ph, k
        if prev_phi is not None and T > prev_k:
            # terminal: the far end is the verifier's exact reward, so only ONE sampled term. T is
            # not on the fixed schedule, so the key carries only the PRECEDING anchor -- two rows
            # are comparable here if they left the same fixed depth, whatever length they ran to.
            bounds.append((prev_k, T, _phi_terminal(terminal[b]) - prev_phi, 1, ("t", prev_k)))
        raw.append(bounds)
        stats["chunks_total"] += len(bounds)

    # ---- leave-one-out per (problem, TRANSITION) ------------------------------------------------
    # Keyed by the transition, NOT by ordinal chunk index. Ordinal index is not a comparable label
    # once responses differ in length: with a fixed 2560 schedule, chunk 2 of a 6k response is its
    # terminal chunk while chunk 2 of a 17k response is the interior 5120->7680. Centering those
    # against each other subtracts a baseline drawn from a different quantity, which shows up as
    # credit that tracks response length rather than progress.
    groups: dict = {}
    for b, u in enumerate(uids):
        for idx, ent in enumerate(raw[b]):
            groups.setdefault((str(u), ent[4]), []).append((b, idx))
    stats["groups"] = len({g[0] for g in groups})
    stats["transitions"] = len(groups)

    centered_all: list[float] = []
    n_sampled: list[int] = []
    group_sizes: list[int] = []
    for _key, members in groups.items():
        vals = [(b, idx) for b, idx in members if raw[b][idx][2] == raw[b][idx][2]]  # drop NaN
        stats["chunks_dropped_nan"] += len(members) - len(vals)
        if len(vals) < min_survivors:
            stats["chunks_dropped_small_group"] += len(vals)
            if len(vals):
                stats["groups_too_small"] += 1
            continue
        total = sum(raw[b][idx][2] for b, idx in vals)
        k = len(vals)
        group_sizes.append(k)
        n_sampled.extend(raw[b][idx][3] for b, idx in vals)
        for b, idx in vals:
            lo, hi, v, _, _ = raw[b][idx]
            # leave-one-out: this row's delta against the mean of the OTHERS
            baseline = (total - v) / (k - 1)
            centered = v - baseline
            centered_all.append(centered)
            hi_c = min(hi, max_response_len)
            if hi_c > lo:
                if credit_norm == "per_chunk":
                    # Sec 7.1 ablation: equal TOTAL weight per chunk, which also hands a token in a
                    # short terminal chunk more weight than one in a full fixed-length chunk.
                    m_b = max(1, len(raw[b]))
                    l_j = max(1, hi - lo)
                    W[b, lo:hi_c] = centered / (m_b * l_j)
                else:
                    # Sec 7 default: broadcast the chunk credit to each of its tokens unchanged, so
                    # u_t = a_t^OPD + beta * G_tilde_{j(t)}. The 1/T normalisation is the loss
                    # aggregator's, not this function's.
                    W[b, lo:hi_c] = centered
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
        credit_norm=str(getattr(sc, "credit_norm", "broadcast")),
        min_survivors=int(sc.min_survivors),
        M=int(sc.M),   # known; the noise floor must not have to guess it
    )
    batch.batch["state_credit_weights"] = _torch.as_tensor(
        W, dtype=_torch.float32, device=resp_mask.device)
    for k, v in stats.items():
        metrics[f"state_credit/{k}"] = v
    print(f"[STATE-CREDIT] rows={stats['rows']} credited={stats['rows_credited']} "
          f"chunks={stats['chunks_total']} dropped_small_group="
          f"{stats['chunks_dropped_small_group']} groups={stats['groups']} "
          f"transitions={stats['transitions']} norm={getattr(sc, 'credit_norm', 'broadcast')}",
          flush=True)

