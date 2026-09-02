# Copyright 2025
#
# Teacher-Guided Block Preconditioning (TGBP).
#
#   a_m          = <g_D^(m), g_R^(m)> / (||g_R^(m)||^2 + eps)
#   g_train^(m)  = (1 + lambda * clip(a_m, 0, a_max)) * g_R^(m)
#
# The teacher never contributes a direction. It can only SCALE the verifier's update inside a
# block, and only upward, and only when its projection onto that update is positive.
#
# THE COEFFICIENT APPLIED IS NEVER THE ONE MEASURED ON THE SAME BATCH.
# ---------------------------------------------------------------------------------------------
# g_R and g_D are built from the SAME rollouts, so they share sampling noise and <g_D,g_R> is
# biased positive even with no real alignment. Measured on this project's own data
# (results/admission/, job 21652581): median a_m is 0.33 same-batch but only 0.17 cross-fitted,
# i.e. roughly half the raw coefficient is shared-batch noise. Applying a same-batch coefficient
# would therefore amplify a block partly BECAUSE its noise happened to line up this step, which is
# a positive feedback loop on noise. So the estimate is an EMA over PREVIOUS batches and step t
# applies weights that never saw batch t. Batch t only updates the estimate afterwards.
#
# WHAT THE DIAGNOSTIC ACTUALLY FOUND, stated here because it bounds what this arm can show:
# at the base policy no block's alignment was distinguishable from zero (0 of 59 sub-blocks
# significant at FDR<0.10, every one below its own detection threshold). The coefficients are big
# enough to matter -- 16/59 blocks would exceed +20% at lambda=1 -- but they were not shown to
# carry signal. This arm may therefore be measuring what a random per-layer learning-rate schedule
# does, which is exactly the "random norm-matched weights" control of the source document. That is
# a legitimate thing to measure; it is not a legitimate thing to describe as a win.

from __future__ import annotations

import json
import math
import os
from typing import Dict, Iterable, Optional, Tuple

import torch


def block_of(name: str) -> str:
    """Map a parameter name to its block. One transformer layer is one block.

    Embeddings and the LM head are kept separate and, in the safest configuration, are left at
    weight 1.0 rather than preconditioned: they are shared across every position and a bad
    coefficient there perturbs the whole model rather than one depth.
    """
    if ".layers." in name:
        return "layer_" + name.split(".layers.")[1].split(".")[0]
    if "embed_tokens" in name:
        return "embed"
    if "lm_head" in name:
        return "lm_head"
    return "other"


class TGBPState:
    """EMA of per-block projection coefficients, plus the weights to apply.

    Kept deliberately small and picklable (plain floats) so it can ride in a checkpoint next to
    the optimizer state. A cold start after a resume would silently revert the arm to plain RLVR
    for the EMA warm-up window, which is the kind of thing that shows up as an unexplained kink
    in a training curve weeks later.
    """

    def __init__(self, ema_beta: float = 0.9, lam: float = 1.0, a_max: float = 1.0,
                 tau_rel: float = 0.02, eps: float = 1e-12,
                 precondition_embed: bool = False):
        self.ema_beta = float(ema_beta)
        self.lam = float(lam)
        self.a_max = float(a_max)
        self.tau_rel = float(tau_rel)
        self.eps = float(eps)
        self.precondition_embed = bool(precondition_embed)
        self.ema: Dict[str, float] = {}
        self.n_updates: int = 0

    # ---------------------------------------------------------------- state dict
    def state_dict(self) -> dict:
        return {"ema": dict(self.ema), "n_updates": self.n_updates,
                "ema_beta": self.ema_beta, "lam": self.lam, "a_max": self.a_max,
                "tau_rel": self.tau_rel, "precondition_embed": self.precondition_embed}

    def load_state_dict(self, sd: dict) -> None:
        if not sd:
            return
        self.ema = dict(sd.get("ema", {}))
        self.n_updates = int(sd.get("n_updates", 0))

    # ------------------------------------------------------------ persistence
    # WHY THIS EXISTS. The EMA lives on the worker object, and verl's checkpoint carries model,
    # optimizer and dataloader state but nothing custom. On a resume the state would therefore
    # reconstruct EMPTY, weights() would return all 1.0, and the arm would silently run as plain
    # RLVR for the whole warm-up window before the projection re-engaged. That is a discontinuity
    # in exactly the quantity this arm exists to measure, and it would surface much later as an
    # unexplained kink in the curve with no way to attribute it.
    #
    # Deliberately a small JSON beside the checkpoint tree rather than a hook into verl's
    # checkpoint manager: a few dozen floats, written atomically, with no coupling to a save path
    # that four backends share.
    def save(self, path: Optional[str]) -> None:
        if not path:
            return
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
                return          # one writer only
        except Exception:
            pass
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self.state_dict(), f)
            os.replace(tmp, path)   # atomic: a torn file would cold-start the EMA silently
        except Exception as e:
            print(f"[tgbp] WARNING: could not persist EMA to {path}: {e}", flush=True)

    def load(self, path: Optional[str]) -> bool:
        if not path or not os.path.exists(path):
            return False
        try:
            with open(path) as f:
                self.load_state_dict(json.load(f))
            print(f"[tgbp] resumed EMA from {path}: {len(self.ema)} blocks, "
                  f"{self.n_updates} prior updates", flush=True)
            return True
        except Exception as e:
            print(f"[tgbp] WARNING: could not load EMA from {path}: {e}", flush=True)
            return False

    # ---------------------------------------------------------------- weights
    def weights(self) -> Dict[str, float]:
        """Weights to APPLY this step, from the EMA of previous batches only.

        Before any EMA exists (step 0, or a genuinely cold start) every weight is 1.0, i.e. the
        arm is exactly RLVR. That is the correct degenerate case: with no out-of-sample estimate
        there is nothing to precondition with.
        """
        w = {}
        for b, a in self.ema.items():
            if b in ("embed", "lm_head", "other") and not self.precondition_embed:
                w[b] = 1.0
                continue
            w[b] = 1.0 + self.lam * min(max(a, 0.0), self.a_max)
        return w

    # ---------------------------------------------------------------- update
    def update(self, dots: Dict[str, float], nR2: Dict[str, float]) -> Dict[str, float]:
        """Fold this batch's measurement into the EMA. Returns the RAW a_m actually measured.

        `dots` and `nR2` must already be reduced over every process group that shards the
        gradient, i.e. they must be the GLOBAL <g_D,g_R> and ||g_R||^2 for the block.
        """
        if not nR2:
            return {}
        max_n = max(nR2.values()) or 0.0
        tau2 = (self.tau_rel ** 2) * max_n
        raw: Dict[str, float] = {}
        for b, n2 in nR2.items():
            if n2 < tau2 or n2 <= 0.0:
                # Below threshold: a_m = dot/||g_R||^2 divides by a vanishing denominator and
                # explodes. An earlier uncorrected analysis reported a=5.53 on such a block with
                # the sign flipping between data halves. Those are not measurements; the block is
                # left at its previous estimate rather than being fed a garbage one.
                continue
            a = dots.get(b, 0.0) / (n2 + self.eps)
            if not math.isfinite(a):
                continue
            raw[b] = a
            prev = self.ema.get(b)
            self.ema[b] = a if prev is None else (self.ema_beta * prev + (1.0 - self.ema_beta) * a)
        self.n_updates += 1
        return raw


@torch.no_grad()
def block_stats(named_grads_R: Iterable[Tuple[str, torch.Tensor]],
                grads_D: Dict[str, torch.Tensor],
                reduce_group=None) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Per-block <g_D, g_R> and ||g_R||^2, reduced over `reduce_group`.

    Both quantities are accumulated in float64 on the device and reduced ONCE as a single stacked
    tensor: a per-block all_reduce would be hundreds of tiny collectives per step.
    """
    dots: Dict[str, float] = {}
    nR2: Dict[str, float] = {}
    for name, gR in named_grads_R:
        if gR is None:
            continue
        gD = grads_D.get(name)
        b = block_of(name)
        r = gR.detach()
        if hasattr(r, "to_local"):
            r = r.to_local()
        r = r.float()
        nR2[b] = nR2.get(b, 0.0) + float((r * r).sum())
        if gD is None:
            dots.setdefault(b, 0.0)
            continue
        d = gD.detach()
        if hasattr(d, "to_local"):
            d = d.to_local()
        dots[b] = dots.get(b, 0.0) + float((d.float() * r).sum())

    if reduce_group is not None and torch.distributed.is_initialized():
        keys = sorted(set(dots) | set(nR2))
        if keys:
            buf = torch.tensor([[dots.get(k, 0.0) for k in keys],
                                [nR2.get(k, 0.0) for k in keys]],
                               dtype=torch.float64,
                               device=torch.cuda.current_device() if torch.cuda.is_available() else "cpu")
            torch.distributed.all_reduce(buf, op=torch.distributed.ReduceOp.SUM, group=reduce_group)
            dots = {k: float(buf[0, i]) for i, k in enumerate(keys)}
            nR2 = {k: float(buf[1, i]) for i, k in enumerate(keys)}
    return dots, nR2


@torch.no_grad()
def apply_weights(named_params, weights: Dict[str, float], default: float = 1.0) -> None:
    """Scale each parameter's .grad in place by its block weight."""
    for name, p in named_params:
        if p.grad is None:
            continue
        w = weights.get(block_of(name), default)
        if w != 1.0:
            p.grad.mul_(w)
