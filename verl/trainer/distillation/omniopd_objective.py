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
"""The OmniOPD objective, as a per-token loss matrix.

    L = L_chunk + beta * L_KL
    L_chunk = - sum_{c audited} pi_hat^(c) * sum_{t in c} log pi_theta(y_t)
    pi_bar^(c)  = exp( mean_{t in c} log pi_theta(y_t) )              geometric mean
    pi_hat^(c)  = (k_sem^(c) + alpha * pi_bar^(c)) / (N + alpha)      DETACHED (FID-5)
    L_KL        = sum_{t unaudited} KL( pi_ref(.|s_t) || pi_theta(.|s_t) )   exact, full vocab (FID-2)

verl consumes a distillation loss as a (bsz, response_len) matrix, so the objective is expressed
per token and SUMS to the scalar above:

    audited t in chunk c : -pi_hat^(c) * log pi_theta(y_t)
    unaudited t          : beta * KL(pi_ref || pi_theta)[t]

AGGREGATION IS NOT NEUTRAL. The reference objective is a SUM. `loss_agg_mode="token-mean"` divides
by the number of valid tokens in the batch, which varies batch to batch, so the effective learning
rate would move with batch composition -- and OmniOPD's audited fraction (M*C tokens out of T) is
itself variable. "seq-mean-token-sum" applies a constant 1/B instead. The mode is recorded in the
metrics so a run can be checked rather than assumed.

MEMORY. The KL is exact over the full vocabulary, so a naive implementation materialises B*T*V for
the student AND the reference -- ~4 GiB each at B=3, T=2178, V=151936 before autograd retains the
log_softmax intermediates, which OOMs a 64 GiB card (observed in milestone 2). Nothing here forms a
full logits tensor: it works from hidden states (B*T*H, four orders of magnitude smaller), applies
lm_head only where the objective reads, and takes the KL in gradient-checkpointed slices so peak is
set by kl_slice rather than by sequence length.

The validated oracle is scripts/omniopd_step.py; scripts/test_omniopd_objective.py asserts this
agrees with it term by term.
"""

from typing import Any, Callable, Optional

import torch


def bayes_target(k_sem: torch.Tensor, pi_bar: torch.Tensor, N: int, alpha: float = 1.0) -> torch.Tensor:
    """Dirichlet-Multinomial smoothed target, DETACHED.

    FID-5: pi_hat multiplies the chunk log-likelihood. If gradient flowed through pi_bar the update
    would gain a second term the method does not have, and the bound that holds because the
    estimator is a CONSTANT multiplier in [0, 1] would no longer apply.
    """
    return ((k_sem + alpha * pi_bar) / (N + alpha)).detach()


def omniopd_token_losses(
    h_cur: torch.Tensor,
    h_ref: torch.Tensor,
    lm_cur: Callable[[torch.Tensor], torch.Tensor],
    lm_ref: Callable[[torch.Tensor], torch.Tensor],
    rows: list[dict[str, Any]],
    N: int,
    C: int,
    alpha: float = 1.0,
    beta: float = 0.1,
    kl_slice: int = 512,
    fixed_target: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Per-token OmniOPD loss for a padded batch.

    Args:
        h_cur, h_ref: (B, L, H) hidden states of the student and the frozen reference.
        lm_cur, lm_ref: their language-model heads, applied only where the objective reads.
        rows: one dict per batch row with
            prompt_len (int), response_ids (list[int]), anchors (list[int]), k_sem (list[float]).
        fixed_target: freeze pi_hat at pre-update parameters. The target is detached but depends on
            pi_bar(theta), so re-evaluating after a step recomputes it and compares a DIFFERENT
            function -- a descent check that does that is not a descent check.

    Returns:
        (B, T_resp) loss matrix summing to L_chunk + beta*L_KL, and metrics.
    """
    from torch.utils.checkpoint import checkpoint

    B = len(rows)
    T_resp = max(len(r["response_ids"]) for r in rows)
    device = h_cur.device
    losses = h_cur.new_zeros((B, T_resp), dtype=torch.float32)

    aud_r, aud_c, realized, k_list, owner, tpos = [], [], [], [], [], []
    un_r, un_c, un_tpos, un_owner = [], [], [], []
    for b, r in enumerate(rows):
        P, T = r["prompt_len"], len(r["response_ids"])
        audited = torch.zeros(T, dtype=torch.bool, device=device)
        for ci, t0 in enumerate(r["anchors"]):
            if t0 + C > T:
                raise ValueError(f"row {b} chunk {ci}: anchor {t0} + C {C} runs past response {T}")
            if bool(audited[t0 : t0 + C].any()):
                raise ValueError(f"row {b} chunk {ci}: overlaps an earlier chunk")
            audited[t0 : t0 + C] = True
            idx = torch.arange(t0, t0 + C, device=device)
            aud_r.append(torch.full((C,), b, device=device, dtype=torch.long))
            aud_c.append(idx + (P - 1))            # position P+t-1 produces response token t
            tpos.append(idx)
            realized.append(torch.tensor(r["response_ids"][t0 : t0 + C], device=device))
            k_list.append(r["k_sem"][ci])
            owner.append(b)
        un = (~audited).nonzero(as_tuple=True)[0]  # real response positions only; padding unreachable
        un_r.append(torch.full_like(un, b))
        un_c.append(un + (P - 1))
        un_tpos.append(un)
        un_owner.append(torch.full_like(un, b))

    # ---- L_chunk ---------------------------------------------------------------------------------
    n_chunks = len(k_list)
    if n_chunks == 0:
        raise ValueError("no audited chunks in this batch; the objective has nothing to reinforce")
    ar, ac = torch.cat(aud_r), torch.cat(aud_c)
    lg = lm_cur(h_cur[ar, ac]).float()                                   # (n_chunks*C, V)
    logp_tok = torch.log_softmax(lg, dim=-1).gather(-1, torch.cat(realized)[:, None]).squeeze(-1)
    del lg
    logp_chunks = logp_tok.view(n_chunks, C)
    pi_bar = torch.exp(logp_chunks.mean(dim=-1))                          # geometric mean, log-space
    k_sem = torch.tensor(k_list, device=device, dtype=torch.float32)
    target = bayes_target(k_sem, pi_bar, N, alpha) if fixed_target is None else fixed_target
    # per token: -pi_hat^(c) * log pi(y_t); summing over t in c and over c gives L_chunk
    per_tok = -(target[:, None] * logp_chunks)                            # (n_chunks, C)
    losses[torch.cat(aud_r), torch.cat(tpos)] = per_tok.reshape(-1)

    # ---- L_KL, checkpointed slice by slice --------------------------------------------------------
    rr, cc = torch.cat(un_r), torch.cat(un_c)
    tt, oo = torch.cat(un_tpos), torch.cat(un_owner)

    def _kl(hc, hr):
        cl = torch.log_softmax(lm_cur(hc).float(), dim=-1)
        with torch.no_grad():
            rl = torch.log_softmax(lm_ref(hr).float(), dim=-1)
        return (rl.exp() * (rl - cl)).sum(-1)                             # (S,) per position

    kl_parts = []
    for i in range(0, rr.numel(), kl_slice):
        hc = h_cur[rr[i : i + kl_slice], cc[i : i + kl_slice]]
        hr = h_ref[rr[i : i + kl_slice], cc[i : i + kl_slice]]
        if torch.is_grad_enabled() and hc.requires_grad:
            kl_parts.append(checkpoint(_kl, hc, hr, use_reentrant=False))
        else:
            kl_parts.append(_kl(hc, hr))
    kl_tok = torch.cat(kl_parts) if kl_parts else h_cur.new_zeros(0)
    if kl_tok.numel():
        losses[oo, tt] = beta * kl_tok

    L_chunk = per_tok.sum()
    L_kl = kl_tok.sum()
    metrics = {
        "omniopd/L_chunk": L_chunk.detach(),
        "omniopd/L_kl": L_kl.detach(),
        "omniopd/n_chunks": float(n_chunks),
        "omniopd/n_unaudited": float(rr.numel()),
        "omniopd/target_mean": target.mean().detach(),
        "omniopd/target_min": target.min().detach(),
        "omniopd/target_max": target.max().detach(),
        "omniopd/k_sem_mean": k_sem.mean(),
        "omniopd/pi_bar_mean": pi_bar.mean().detach(),
    }
    # FID-9: the objective only ever reinforces. pi_hat is a probability, so a target outside [0, 1]
    # means k_sem or N is wrong upstream and the update would be pushing log-likelihood DOWN.
    tmin, tmax = float(metrics["omniopd/target_min"]), float(metrics["omniopd/target_max"])
    if not (0.0 <= tmin and tmax <= 1.0):
        raise ValueError(
            f"OmniOPD target outside [0,1]: [{tmin}, {tmax}]. pi_hat is a probability; outside that "
            f"range the objective is no longer purely reinforcing (FID-9), which means k_sem or N "
            f"is wrong rather than that the loss is merely large."
        )
    if target.requires_grad:
        raise ValueError("OmniOPD target carries gradient; it must be detached (FID-5)")
    return losses, metrics
