# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import os
import torch
from tensordict import TensorDict

from verl.base_config import BaseConfig
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.metric import AggregationType, Metric
from verl.workers.config import ActorConfig, DistillationConfig, DistillationLossConfig
from verl.workers.utils.losses import ppo_loss
from verl.workers.utils.padding import no_padding_2_padding

_TOKEN_SELECT_LOG_COUNTER = 0
_AUDIT_LOG_COUNTER = 0
# Running (whole-run) accumulators for the Phase-2 audit so the kept-vs-audit aggregate is meaningful even
# when per-micro-batch audit n is tiny (AUDIT_FRAC << 1). Per-response masses; capped. Per-process (per FSDP rank).
_AUDIT_KEPT_D: list = []
_AUDIT_AUD_D: list = []
_AUDIT_KEPT_S: list = []
_AUDIT_AUD_S: list = []

DistillationLossFn = Callable[
    [
        ActorConfig,  # actor_config
        DistillationConfig,  # distillation_config
        dict,  # model_output
        TensorDict,  # micro batch input
    ],
    tuple[torch.Tensor, dict[str, Any]],
]


def is_distillation_enabled(config: Optional[DistillationConfig]) -> bool:
    """Check if distillation is enabled based on the provided configuration."""
    if config is None:
        return False
    return config.enabled


@dataclass
class DistillationLossSettings(BaseConfig):
    """
    Settings for a distillation loss function to be registered.

    Args:
        names (str | list[str]): Name(s) to register the distillation loss function under.
        use_topk (bool): Whether the loss function uses top-k log probabilities.
        use_estimator (bool): Whether the loss function uses single-sample KL estimators.
    """

    names: str | list[str] = field(default_factory=list)
    use_topk: bool = False
    use_estimator: bool = False
    use_teacher_generation: bool = False
    use_teacher_continuations: bool = False

    _mutable_fields = {"names"}

    def __post_init__(self):
        self.names = [self.names] if isinstance(self.names, str) else self.names
        # A distillation objective consumes the teacher in exactly one of three ways: top-k
        # logprobs, a sampled-token KL estimator, or -- for generative teaching -- text the teacher
        # WROTE. The third is not a variant of the first two: it needs the teacher to generate
        # rather than score, which changes how its engine must be dimensioned, so it has to be
        # declared here where the teacher config can see it.
        # A fourth way: teacher CONTINUATIONS run to a verifiable answer (State-Credit). It is not a
        # variant of use_teacher_generation -- that writes a fixed C-token chunk to compare wording,
        # this writes a whole completion so an external verifier can say whether the state was
        # solvable. The engine dimensioning differs (prompt + depth + B, which can exceed
        # prompt + response), so it must be declared where the teacher config can see it.
        if sum([self.use_topk, self.use_estimator, self.use_teacher_generation,
                self.use_teacher_continuations]) != 1:
            raise ValueError(
                f"Expected exactly one of use_estimator, use_topk, use_teacher_generation, "
                f"use_teacher_continuations, but got {self.use_estimator=}, {self.use_topk=}, "
                f"{self.use_teacher_generation=}, {self.use_teacher_continuations=}."
            )


DISTILLATION_LOSS_REGISTRY: dict[str, DistillationLossFn] = {}
DISTILLATION_SETTINGS_REGISTRY: dict[str, DistillationLossSettings] = {}


def register_distillation_loss(
    loss_settings: DistillationLossSettings,
) -> Callable[[DistillationLossFn], DistillationLossFn]:
    """Register a distillation loss function with the given name."""

    def decorator(func: DistillationLossFn) -> DistillationLossFn:
        for name in loss_settings.names:
            if name in DISTILLATION_LOSS_REGISTRY:
                raise ValueError(f"Distillation loss function with name '{name}' is already registered.")
            DISTILLATION_LOSS_REGISTRY[name] = func
            DISTILLATION_SETTINGS_REGISTRY[name] = loss_settings
        return func

    return decorator


def get_distillation_loss_fn(loss_name: str) -> DistillationLossFn:
    """Get the distillation loss function with a given name."""
    if loss_name not in DISTILLATION_LOSS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(DISTILLATION_LOSS_REGISTRY.keys())}"
        )
    return DISTILLATION_LOSS_REGISTRY[loss_name]


def get_distillation_loss_settings(loss_name: str) -> DistillationLossSettings:
    """Get the distillation loss settings with a given name."""
    if loss_name not in DISTILLATION_SETTINGS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(DISTILLATION_SETTINGS_REGISTRY.keys())}"
        )
    return DISTILLATION_SETTINGS_REGISTRY[loss_name]


def compute_distillation_loss_range(
    distillation_losses: torch.Tensor, response_mask: torch.Tensor
) -> dict[str, Metric]:
    """Compute min and max distillation loss over valid response tokens."""
    if response_mask.is_nested:
        distillation_losses_response = distillation_losses[response_mask.bool().to_padded_tensor(False)]
    else:
        distillation_losses_response = distillation_losses[response_mask.bool()]
    return {
        "distillation/loss_min": Metric(AggregationType.MIN, distillation_losses_response.min()),
        "distillation/loss_max": Metric(AggregationType.MAX, distillation_losses_response.max()),
    }


def compute_topk_loss(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    data: TensorDict,
    student_logits: torch.Tensor,
    data_format: str,
) -> torch.Tensor:
    """Compute the topk loss in logit processor.

    Returns:
    - distillation_losses: (bsz, seqlen/cp_size)
    - student_mass: (bsz, seqlen/cp_size)
    - teacher_mass: (bsz, seqlen/cp_size)
    """
    match config.strategy:
        # VeOmni uses FSDP2 internally, so its loss computation is identical to FSDP.
        case "fsdp" | "veomni":
            import verl.trainer.distillation.fsdp.losses as fsdp_losses

            distillation_loss_fn = fsdp_losses.compute_forward_kl_topk
        case "megatron":
            import verl.trainer.distillation.megatron.losses as megatron_losses

            distillation_loss_fn = megatron_losses.compute_forward_kl_topk
        case _:
            raise NotImplementedError(f"Unsupported strategy: {config.strategy=}")

    outputs = distillation_loss_fn(
        student_logits=student_logits,
        teacher_topk_log_probs=data["teacher_logprobs"],
        teacher_topk_ids=data["teacher_ids"],
        config=distillation_config,
        data_format=data_format,
    )

    expected_shape = student_logits.shape[:2]
    for k, v in outputs.items():
        assert v.shape == expected_shape, f"Expected shape {expected_shape}, but got {v.shape} for {k=}."

    return outputs


def distillation_ppo_loss(
    config: ActorConfig,
    distillation_config: Optional[DistillationConfig],
    model_output: dict = None,
    data: TensorDict = None,
    dp_group=None,
    student_logits: torch.Tensor = None,
    data_format: str = "thd",
):
    """Loss function used both for logit processor and final policy loss.
    - student_logits is not None, compute the topk loss in logit processor.
    - student_logits is None, compute final policy loss.

    [split sequence across sp/cp groups]
                   |
    [model forward and output logits: (bsz, seqlen/cp_size, vocab_size/tp_size)]
                   |
    [logits processor compute topk loss: (bsz, seqlen/cp_size)]
                   |
    [all gather topk loss across sp/cp groups: (bsz, seqlen)]
                   |
    [combine topk loss with policy loss]

    Args:
        config: Actor configuration.
        distillation_config: Distillation configuration.
        model_output: Model output, including log_probs, entropy.
        data: Micro input batch, contains
          - teacher_logprobs: (bsz, seqlen, topk)
          - teacher_ids: (bsz, seqlen, topk)
        student_logits: (bsz, seqlen/cp_size, vocab_size/tp_size).
        data_format: "thd" or "bshd", models not support THD format, e.g GPT-OSS, Qwen3.5

    Returns:
    - student_logits is not None, return the topk loss tensor (bsz, seqlen/cp_size).
    - student_logits is None, return the final policy loss scalar and metrics.
    """

    # Called as logits processor
    if student_logits is not None:
        return compute_topk_loss(config, distillation_config, data, student_logits, data_format)

    # Called as final policy loss
    distillation_loss_config = distillation_config.distillation_loss
    # ORDER IS LOAD-BEARING. ppo_loss is the ONLY writer of config.global_batch_info
    # (workers/utils/losses.py:65-68), and distillation_loss READS it (:461) to normalise. Called the
    # other way round, the very first micro-batch of a run sees an empty dict, so agg_loss falls back
    # to dp_size=1 and to that micro-batch's own sequence count instead of ppo_mini_batch_size x
    # dp_size -- silently over-weighting the first accumulated gradient by (global/n)/dp_size.
    # Nothing raises: the guard in core_algos only fires for dp_size > 1, and dp_size is exactly what
    # defaulted to 1. It is inert for k1 (use_policy_gradient=True takes the other branch at :456),
    # which is why it has never mattered -- but omniopd FORCES use_policy_gradient=False.
    # ppo_loss does not read the distillation result; that is combined below at :236.
    policy_loss, policy_metrics = ppo_loss(config, model_output, data, dp_group)
    distill_loss, distill_metrics = distillation_loss(config, distillation_config, model_output, data)
    if not distillation_loss_config.use_task_rewards:
        policy_loss = 0.0

    # Combine distillation with policy loss
    policy_metrics.update(distill_metrics)
    distillation_loss_coef = (
        distillation_loss_config.distillation_loss_coef if distillation_loss_config.use_task_rewards else 1.0
    )
    policy_loss += distill_loss * distillation_loss_coef
    policy_metrics["distillation/loss"] = Metric(value=distill_loss, aggregation=AggregationType.SUM)

    return policy_loss, policy_metrics


def _compute_rollout_drift_surrogate(model_output: dict, data: TensorDict) -> dict[str, Any]:
    """Detached, analysis-only rollout-drift surrogate D^roll_hat (FROST mechanism diagnostic).

    Returns metrics {distillation/d_roll_hat_(mean|absmean|p95)} computed as the masked per-response-
    token log-ratio logp_current - logp_stale, where logp_stale := data["rollout_log_probs"] (the
    generate-time student) and logp_current := model_output["log_probs"] (the current student, same
    sampled token). Never enters the gradient. Returns {} if the required tensors are absent or
    mis-shaped (e.g. the supervised batch does not carry rollout_log_probs) so it is always safe.
    """
    try:
        if "rollout_log_probs" not in data.keys():
            return {}
        rm = data["response_mask"]
        rm = rm.to_padded_tensor(False) if rm.is_nested else rm
        rm = rm.bool()
        cur = no_padding_2_padding(model_output["log_probs"], data)
        stale = data["rollout_log_probs"]
        stale = stale.to_padded_tensor(0.0) if stale.is_nested else stale
        if not (cur.shape == stale.shape == rm.shape):
            return {}
        d = (cur - stale).detach()[rm]
        if d.numel() == 0:
            return {}
        d_abs = d.abs().float()
        return {
            "distillation/d_roll_hat_mean": Metric(AggregationType.MEAN, d.mean()),
            "distillation/d_roll_hat_absmean": Metric(AggregationType.MEAN, d_abs.mean()),
            "distillation/d_roll_hat_p95": Metric(AggregationType.MAX, torch.quantile(d_abs, 0.95)),
        }
    except Exception:
        return {}


def _minmax01_clip(x: torch.Tensor, valid: torch.Tensor, clip_q: float = 0.98) -> torch.Tensor:
    """Clip the top (1-clip_q) at the batch percentile over valid tokens, then min-max -> [0,1] (0 outside valid).

    Matches TIP's "clip at 98th batch percentile, then min-max normalize" for the Soft-OR inputs.
    """
    xv = x[valid].float()
    if xv.numel() == 0:
        return torch.zeros_like(x, dtype=torch.float32)
    hi = torch.quantile(xv, clip_q)
    xc = torch.minimum(x.float(), hi)
    lo = xv.min()
    denom = (torch.minimum(xv, hi).max() - lo).clamp(min=1e-8)
    out = ((xc - lo) / denom).clamp(0.0, 1.0)
    return torch.where(valid, out, torch.zeros_like(out))


def _token_select_log_metrics(h, delta, score, valid, select) -> dict[str, Any]:
    """Diagnostic metrics: retained-vs-dropped means + Q1-Q4 retained fractions (median split of h x delta)."""
    m: dict[str, Any] = {}
    dropped = valid & (~select)

    def _mean(t, msk):
        msk = msk & valid
        return t[msk].float().mean().item() if msk.any() else 0.0

    if h is not None:
        m["token_select/entropy_retained"] = _mean(h, select)
        m["token_select/entropy_dropped"] = _mean(h, dropped)
    m["token_select/divergence_retained"] = _mean(delta, select)
    m["token_select/divergence_dropped"] = _mean(delta, dropped)
    m["token_select/score_retained"] = _mean(score, select)
    m["token_select/score_dropped"] = _mean(score, dropped)
    if h is not None and valid.any():
        hmed = h[valid].float().median()
        dmed = delta[valid].float().median()
        hi_h, hi_d = (h >= hmed), (delta >= dmed)
        # Q1 high-h/high-d, Q2 high-h/low-d, Q3 low-h/high-d (TIP blind spot), Q4 low-h/low-d
        quads = {"Q1": hi_h & hi_d, "Q2": hi_h & (~hi_d), "Q3": (~hi_h) & hi_d, "Q4": (~hi_h) & (~hi_d)}
        n_valid = valid.sum().clamp(min=1)
        for name, q in quads.items():
            qv = q & valid
            qc = qv.sum()
            kept = (qv & select).sum()
            m[f"token_select/{name}_frac_of_valid"] = (qc / n_valid).item()
            m[f"token_select/{name}_retained_frac"] = (kept / qc.clamp(min=1)).item() if qc > 0 else 0.0
    return m


def _compute_token_selection(distillation_losses, student_entropy, response_mask, loss_config):
    """TIP-style post-teacher token selection. Returns (select_mask [B,T] float in {0,1}, metrics).

    select_mask is 1 exactly where the token is retained, always 0 outside response_mask. When
    mode=='none' or retention>=1.0 it returns response_mask itself (=> byte-equivalent loss). The
    selection signal is detached (never backpropagated through).

    Axes: h_t = full-vocab student entropy (exact, from the processor); delta_t = per-token
    distillation_losses (forward-KL over teacher top-k support, the in-system divergence analog).
    """
    mode = loss_config.token_select_mode
    rho = float(loss_config.token_retention)
    scope = loss_config.token_select_scope
    valid = response_mask.bool()
    metrics: dict[str, Any] = {}
    if mode == "none" or rho >= 1.0:
        return valid.float(), metrics

    delta = distillation_losses.detach().float()
    if mode == "random":
        # Uniform-random selection among valid response tokens (baseline). No teacher/student signal.
        score = torch.rand_like(delta)
    elif mode == "entropy":
        if student_entropy is None:
            raise ValueError(
                "OPD_TOKEN_SELECT_MODE=entropy requires per-token student_entropy in model_output "
                "(computed in the forward_kl_topk logits processor); none found. Is the loss_mode "
                "forward_kl_topk on the FSDP path?"
            )
        score = student_entropy.detach().float()
    else:  # soft_or
        if student_entropy is None:
            raise ValueError("OPD_TOKEN_SELECT_MODE=soft_or requires per-token student_entropy in model_output; none found.")
        h_hat = _minmax01_clip(student_entropy.detach(), valid)
        d_hat = _minmax01_clip(delta, valid)
        score = h_hat + d_hat - h_hat * d_hat

    neg = torch.finfo(score.dtype).min
    score_v = torch.where(valid, score, torch.full_like(score, neg))

    if scope == "response":
        n_i = valid.sum(dim=1)  # [B]
        k_i = torch.clamp((rho * n_i.float()).floor().long(), min=1)
        k_i = torch.minimum(k_i, n_i.long())  # cannot exceed row's valid count
        order = score_v.argsort(dim=1, descending=True)  # [B,T]
        ranks = torch.empty_like(order)
        ranks.scatter_(1, order, torch.arange(score_v.shape[1], device=score_v.device).expand_as(order))
        select = (ranks < k_i.unsqueeze(1)) & valid
    else:  # batch (per-micro-batch global top-rho)
        flat = score_v[valid]
        n = int(valid.sum().item())
        if n == 0:
            return valid.float(), metrics
        k = max(1, int(rho * n))
        thresh = torch.topk(flat, k).values.min()
        select = (score_v >= thresh) & valid

    if loss_config.token_select_log:
        metrics.update(_token_select_log_metrics(h=student_entropy, delta=delta, score=score, valid=valid, select=select))
    n_sel, n_val = select.sum(), valid.sum()
    metrics["token_select/retained_frac"] = (n_sel / n_val.clamp(min=1)).item()
    metrics["token_select/retained_tokens"] = float(n_sel.item())
    metrics["token_select/valid_tokens"] = float(n_val.item())
    metrics["token_select/retention_target"] = rho
    return select.float(), metrics


def distillation_loss(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the distillation loss and related metrics.

    Returns:
    - distillation_loss: Aggregated distillation loss scalar.
    - distillation_metrics: Dictionary of metrics.
    """
    assert distillation_config is not None
    loss_config: DistillationLossConfig = distillation_config.distillation_loss
    distillation_loss_fn = get_distillation_loss_fn(loss_config.loss_mode)
    distillation_losses, distillation_metrics = distillation_loss_fn(
        config=config,
        distillation_config=distillation_config,
        model_output=model_output,
        data=data,
    )
    response_mask = data["response_mask"]
    loss_agg_mode = config.loss_agg_mode

    distillation_metrics.update(
        compute_distillation_loss_range(distillation_losses=distillation_losses, response_mask=response_mask)
    )

    # --- FROST rollout-drift surrogate diagnostic (detached, analysis-only; never enters the gradient) ---
    # D^roll_hat = logp_current(sampled token) - logp_stale(sampled token): a k1 log-ratio SURROGATE
    # for the per-token rollout drift KL(pi_stale || pi_current). `rollout_log_probs` holds the stale
    # (generate-time) student's sampled-token logprob; model_output["log_probs"] is the current
    # student's logprob of the same token (computed in the forward -> zero extra model pass). This is a
    # surrogate, NOT the paper's full top-k KL (rollout_log_probs is a per-token scalar, not a
    # distribution); calibrate offline against the full KL before trusting it. Logged in ALL arms.
    distillation_metrics.update(
        _compute_rollout_drift_surrogate(model_output=model_output, data=data)
    )

    if loss_config.loss_max_clamp is not None:
        # clamping min is for k1 loss which can be negative
        distillation_losses = distillation_losses.clamp(min=-loss_config.loss_max_clamp, max=loss_config.loss_max_clamp)

    if loss_config.use_policy_gradient:
        # Use negative distillation loss as reward, as done by https://thinkingmachines.ai/blog/on-policy-distillation/.
        policy_loss_fn = get_policy_loss_fn(loss_config.policy_loss_mode)
        for k, v in config.global_batch_info.items():
            loss_config.global_batch_info[k] = v
        log_prob = no_padding_2_padding(model_output["log_probs"], data)
        old_log_prob = data["old_log_probs"]
        if old_log_prob.is_nested:
            old_log_prob = data["old_log_probs"].to_padded_tensor(0.0)
        if response_mask.is_nested:
            response_mask = response_mask.to_padded_tensor(False)
        rollout_is_weights = data.get("rollout_is_weights", None)
        distillation_loss, pg_metrics = policy_loss_fn(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=-distillation_losses.detach(),
            response_mask=response_mask,
            loss_agg_mode=loss_agg_mode,
            config=loss_config,
            rollout_is_weights=rollout_is_weights,
        )
        pg_metrics = {f"distillation/{k[len('actor/') :]}": v for k, v in pg_metrics.items()}
        distillation_metrics.update(pg_metrics)
    else:
        # Directly backpropagate distillation loss as a supervised loss, as in https://arxiv.org/abs/2306.13649.
        if response_mask.is_nested:
            response_mask = response_mask.to_padded_tensor(False)
        loss_mask = response_mask
        gbi = config.global_batch_info
        # --- TIP-style post-teacher token selection (default OFF => the call below is byte-identical) ---
        if loss_config.token_select_mode != "none" and loss_config.token_retention < 1.0:
            student_entropy = model_output.get("student_entropy")
            if student_entropy is not None:
                student_entropy = no_padding_2_padding(student_entropy, data)
            select_mask, sel_metrics = _compute_token_selection(
                distillation_losses=distillation_losses,
                student_entropy=student_entropy,
                response_mask=response_mask,
                loss_config=loss_config,
            )
            distillation_metrics.update(sel_metrics)
            global _TOKEN_SELECT_LOG_COUNTER
            _TOKEN_SELECT_LOG_COUNTER += 1
            if _TOKEN_SELECT_LOG_COUNTER % 20 == 1:
                print(
                    f"[TOKEN-SELECT] mode={loss_config.token_select_mode} ret={loss_config.token_retention} "
                    f"scope={loss_config.token_select_scope} "
                    f"retained_frac={sel_metrics.get('token_select/retained_frac')} "
                    f"retained_tok={sel_metrics.get('token_select/retained_tokens')} "
                    f"ent_ret={sel_metrics.get('token_select/entropy_retained')} "
                    f"ent_drop={sel_metrics.get('token_select/entropy_dropped')} "
                    f"div_ret={sel_metrics.get('token_select/divergence_retained')} "
                    f"div_drop={sel_metrics.get('token_select/divergence_dropped')} "
                    f"Q3_ret={sel_metrics.get('token_select/Q3_retained_frac')}",
                    flush=True,
                )
            loss_mask = response_mask.bool().float() * select_mask
            # Faithful TIP loss = MEAN over SELECTED tokens (not full-token down-weighting). agg_loss
            # token-mean divides by the DP-global batch_num_tokens; scale it by the per-rank retained
            # fraction so the denominator becomes ~the global selected-token count (exact when the
            # retained fraction is uniform across DP ranks, which holds to high precision for large
            # micro-batches). Byte-equivalent when retention=1.0 (n_sel==n_val => ratio 1).
            n_val = response_mask.bool().sum()
            n_sel = loss_mask.sum()
            if n_sel > 0 and gbi.get("batch_num_tokens"):
                gbi = dict(gbi)
                gbi["batch_num_tokens"] = gbi["batch_num_tokens"] * (n_sel / n_val.clamp(min=1)).item()
        # --- Phase 2 AUDIT: post-hoc TIP/Soft-OR diagnostics on policy would-skip (audit_scored) samples,
        #     + exclude them from the gradient. Default OFF: data has no 'is_audit' => byte-identical. ---
        is_audit = data.get("is_audit", None)
        if is_audit is not None:
            is_audit = is_audit.bool()
            if is_audit.dim() == 1:
                is_audit = is_audit.unsqueeze(1).expand_as(response_mask)
            valid = response_mask.bool()
            delta = distillation_losses.detach()
            n_tok = valid.sum(dim=1).clamp(min=1)
            delta_resp = (delta * valid).sum(dim=1) / n_tok  # per-response mean divergence (teacher value)
            has_tok = valid.sum(dim=1) > 0
            aud = is_audit[:, 0] & has_tok
            kep = (~is_audit[:, 0]) & has_tok
            n_aud, n_kep = int(aud.sum()), int(kep.sum())
            so_resp = None
            ent = model_output.get("student_entropy")
            if ent is not None:
                ent = no_padding_2_padding(ent, data)
                h_hat = _minmax01_clip(ent.detach(), valid)
                d_hat = _minmax01_clip(delta, valid)
                so = h_hat + d_hat - h_hat * d_hat  # Soft-OR per token
                so_resp = (so * valid).sum(dim=1) / n_tok
            # PERSIST each response's (label, delta, Soft-OR) to a per-PID CSV under the run trace dir, so the
            # audit aggregate SURVIVES the FSDP/Ray per-call module reload (module globals reset between loss
            # calls -> in-memory accumulation does NOT persist). Aggregated post-hoc across all audit_*.csv.
            import os as _os

            _trace = _os.environ.get("OPD_STAGE0_TRACE_DIR", "")
            if _trace:
                try:
                    _ad = _os.path.join(_trace, "audit")
                    _os.makedirs(_ad, exist_ok=True)
                    _dl = delta_resp.tolist()
                    _sl = so_resp.tolist() if so_resp is not None else None
                    _isa = is_audit[:, 0].tolist()
                    _ht = has_tok.tolist()
                    _rows = [
                        f"{'a' if _isa[_i] else 'k'},{_dl[_i]:.6f},{(f'{_sl[_i]:.6f}' if _sl is not None else 'nan')}\n"
                        for _i in range(len(_dl))
                        if _ht[_i]
                    ]
                    if _rows:
                        with open(_os.path.join(_ad, f"audit_{_os.getpid()}.csv"), "a") as _f:
                            _f.write("".join(_rows))
                except Exception:
                    pass
            distillation_metrics["audit/n_kept_mb"] = float(n_kep)
            distillation_metrics["audit/n_audit_mb"] = float(n_aud)
            # exclude audit responses from the gradient (+ rescale token-mean denom to kept tokens)
            loss_mask = loss_mask.float() * (~is_audit).float()
            n_keep_tok = (valid & (~is_audit)).sum()
            if n_keep_tok > 0 and gbi.get("batch_num_tokens"):
                gbi = dict(gbi)
                gbi["batch_num_tokens"] = gbi["batch_num_tokens"] * (n_keep_tok / valid.sum().clamp(min=1)).item()
            global _AUDIT_LOG_COUNTER
            _AUDIT_LOG_COUNTER += 1
            if _AUDIT_LOG_COUNTER % 20 == 1:
                _kdm = float(delta_resp[kep].mean()) if n_kep else float("nan")
                _adm = float(delta_resp[aud].mean()) if n_aud else float("nan")
                print(
                    f"[TEACHER-AUDIT] this_mb n_kept={n_kep} n_audit={n_aud} "
                    f"kept_delta_mass={_kdm:.4f} audit_skipped_delta_mass={_adm:.4f} "
                    f"(full aggregate via <trace>/audit/*.csv post-hoc)",
                    flush=True,
                )
        distillation_loss = agg_loss(
            loss_mat=distillation_losses,
            loss_mask=loss_mask,
            loss_agg_mode=loss_agg_mode,
            **gbi,
        )

    return distillation_loss, distillation_metrics


@register_distillation_loss(DistillationLossSettings(names=["forward_kl_topk"], use_topk=True))  # type: ignore[arg-type]
def compute_forward_kl_topk(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute forward KL distillation loss and related metrics using top-k log probabilities.

    Returns:
    - distillation_losses: (bsz, resp_len)
    - distillation_metrics: Dictionary of metrics.
    """
    # topk loss has been computed in logits processor
    distillation_losses = no_padding_2_padding(model_output["distillation_losses"], data)
    student_mass = no_padding_2_padding(model_output["student_mass"], data)
    teacher_mass = no_padding_2_padding(model_output["teacher_mass"], data)
    if data["response_mask"].is_nested:
        response_mask_bool = data["response_mask"].bool().to_padded_tensor(False)
    else:
        response_mask_bool = data["response_mask"].bool()
    assert distillation_losses.shape == student_mass.shape == teacher_mass.shape == response_mask_bool.shape

    # Log amount of mass in the top-k log probabilities for both student and teacher.
    student_mass = student_mass[response_mask_bool]
    teacher_mass = teacher_mass[response_mask_bool]
    distillation_metrics = {
        "distillation/student_mass": student_mass.mean().item(),
        "distillation/student_mass_min": Metric(AggregationType.MIN, student_mass.min()),
        "distillation/student_mass_max": Metric(AggregationType.MAX, student_mass.max()),
        "distillation/teacher_mass": teacher_mass.mean().item(),
        "distillation/teacher_mass_min": Metric(AggregationType.MIN, teacher_mass.min()),
        "distillation/teacher_mass_max": Metric(AggregationType.MAX, teacher_mass.max()),
    }

    # Due to use of top-k, student and teacher distributions don't sum to 1 -> divergences can be negative.
    distillation_losses = distillation_losses.clamp_min(0.0)

    return distillation_losses, distillation_metrics


def _state_credit_column(data, key):
    """Same container problem as _omniopd_column: the trainer hands the loss a TensorDict, where
    non-tensor columns live as NonTensorStacks reached by data[key], while a DataProto exposes
    .non_tensor_batch. state_credit_weights is a real tensor, so it rides the tensor batch -- but
    the lookup still has to work on both containers."""
    ntb = getattr(data, "non_tensor_batch", None)
    if ntb is not None and key in ntb:
        return ntb[key]
    try:
        return data[key]
    except (KeyError, IndexError) as e:
        raise KeyError(
            f"State-Credit loss needs {key!r} and it is absent from this batch. It is built in the "
            f"driver (_fit_compute_advantage) from the per-depth Phi columns; if that stage did not "
            f"run, the objective has no credit signal and must not silently train without one."
        ) from e


def _omniopd_column(data, key):
    """Read a per-sample OmniOPD column from a DataProto OR a bare TensorDict.

    Both appear on the path: the rollout assembles a DataProto, and the trainer micro-batches it
    into TensorDicts (protocol.py to_tensordict wraps each non-tensor column in a NonTensorStack, so
    the values survive rearrange_micro_batches and make_iterator in row order).
    """
    ntb = getattr(data, "non_tensor_batch", None)
    if ntb is not None and key in ntb:
        return ntb[key]
    try:
        return data[key]
    except (KeyError, IndexError) as e:
        raise KeyError(
            f"OmniOPD loss needs the per-sample column {key!r} and it is absent from this batch. "
            f"It is produced by the rollout-side audit (omniopd_stage.attach_omniopd_audit); if the "
            f"audit did not run, the objective has no targets and must not silently train without "
            f"them."
        ) from e


@register_distillation_loss(
    DistillationLossSettings(names=["state_credit"], use_teacher_continuations=True)
)  # type: ignore[arg-type]
def compute_distillation_loss_state_credit(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output,
    data: TensorDict,
):
    """State-Credit: reinforce a chunk in proportion to the progress it made.

        S[b,t] = -W[b,t] * log pi_theta(y_t | s_<t)

    where W is a DENSE per-token weight the driver has already built:

        W[b,t] = Delta_tilde_j / (m_b * l_j)     for t in chunk j of row b

    with Delta_tilde the leave-one-out-centered progress of that chunk, m_b the number of chunks
    the row actually has, and l_j the chunk's true token count. Summing S over a row therefore
    reproduces -(1/m_b) * sum_j Delta_tilde_j * logbar_j exactly.

    WHY THE WEIGHT ARRIVES PRE-BUILT. Centering is leave-one-out ACROSS the rollouts of one problem,
    and by the time a micro-batch reaches this function those siblings are gone: balance_batch
    reorders rows onto DP ranks by length, mini-batching slices, and dynamic-bsz repartitions again.
    A micro-batch is an arbitrary length-selected subset of one rank's slice. So the centering is
    done in the driver, where the whole batch is present and grouping is by key, and this function
    receives a row-local tensor that is exactly invariant to all three reorderings.

    W IS DETACHED BY CONSTRUCTION -- it is built from Phi values produced by a FROZEN continuation
    model and an external verifier, neither of which is on the autograd graph. Asserted rather than
    assumed: a W that carried gradient would add a term the method does not have.
    """
    log_probs = no_padding_2_padding(model_output["log_probs"], data)
    W = _state_credit_column(data, "state_credit_weights")
    if not torch.is_tensor(W):
        W = torch.as_tensor(W, device=log_probs.device, dtype=log_probs.dtype)
    W = W.to(device=log_probs.device, dtype=log_probs.dtype)
    if W.requires_grad:
        raise ValueError(
            "state_credit_weights carries gradient; Phi comes from a frozen continuation model and "
            "a verifier, so the weight must be a constant multiplier (cf. FID-5 for the OmniOPD "
            "target).")

    response_mask = data["response_mask"]
    if response_mask.is_nested:
        response_mask = response_mask.to_padded_tensor(False)
    response_mask = response_mask.bool()
    if W.shape != log_probs.shape:
        raise ValueError(
            f"state_credit_weights {tuple(W.shape)} does not match log_probs "
            f"{tuple(log_probs.shape)}; the driver must emit one weight per response position.")

    losses = -(W * log_probs)
    losses = torch.where(response_mask, losses, torch.zeros_like(losses))

    nz = (W != 0) & response_mask
    metrics = {
        "state_credit/L_state": Metric(AggregationType.MEAN, losses.sum()),
        "state_credit/weighted_tokens": Metric(AggregationType.MEAN, nz.sum().float()),
        "state_credit/w_absmean": Metric(
            AggregationType.MEAN,
            (W[nz].abs().mean() if bool(nz.any()) else torch.zeros((), device=W.device))),
        "state_credit/w_pos_frac": Metric(
            AggregationType.MEAN,
            ((W[nz] > 0).float().mean() if bool(nz.any()) else torch.zeros((), device=W.device))),
        "state_credit/rows_without_credit": Metric(
            AggregationType.MEAN,
            (~nz.any(dim=-1)).float().sum()),
    }
    return losses, metrics


@register_distillation_loss(
    DistillationLossSettings(names=["omniopd"], use_teacher_generation=True)
)  # type: ignore[arg-type]


def compute_distillation_loss_omniopd(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """OmniOPD objective as a (bsz, resp_len) per-token loss matrix.

        audited t in chunk c : -pi_hat^(c) * log pi_theta(y_t)
        unaudited t          : beta * KL(pi_ref || pi_theta)[t]

        pi_bar^(c) = exp( mean_{t in c} log pi_theta(y_t) )            geometric mean, log-space
        pi_hat^(c) = (k_sem^(c) + alpha * pi_bar^(c)) / (N + alpha)    DETACHED (FID-5)

    WHERE THE TWO INPUTS COME FROM, and why they are split this way:

      log pi_theta(y_t)  model_output["log_probs"], which verl already computes for every loss.
      KL per token       model_output["omniopd_kl"], computed in the LOGITS PROCESSOR, because the
                         anchor is an exact full-vocabulary KL (FID-2) and the full student
                         distribution exists only there. This mirrors forward_kl_topk, which also
                         does its real work in the processor and reads the result back here.

    The processor -- not this function -- owns the reference model. A sampled-token log-ratio would
    be cheaper and is NOT acceptable: unaudited positions are the majority of every trajectory, so a
    sampled surrogate changes the gradient on most of the tokens being trained.

    FID-5 and FID-9 are enforced here rather than assumed, because both failures are silent: a target
    carrying gradient adds a term the method does not have, and a target outside [0, 1] means the
    update is pushing log-likelihood DOWN on chunks the teacher agreed with.
    """
    log_probs = no_padding_2_padding(model_output["log_probs"], data)
    if "omniopd_kl" not in model_output:
        raise ValueError(
            "loss_mode=omniopd needs model_output['omniopd_kl'], the exact full-vocabulary "
            "KL(pi_ref || pi_theta) per response token, produced by the logits processor. Its "
            "absence means the engine is not running the omniopd path -- refusing to fall back to a "
            "sampled-token surrogate, which would train a different objective and look healthy."
        )
    kl = no_padding_2_padding(model_output["omniopd_kl"], data)
    response_mask = data["response_mask"]
    if response_mask.is_nested:
        response_mask = response_mask.to_padded_tensor(False)
    response_mask = response_mask.bool()
    assert log_probs.shape == kl.shape == response_mask.shape, (
        f"log_probs {log_probs.shape}, omniopd_kl {kl.shape}, mask {response_mask.shape}")

    om = distillation_config.omniopd
    N, C, alpha, beta = om.N, om.C, om.alpha, om.beta
    # The trainer hands this a TensorDict, not a DataProto: DataProto.to_tensordict turns each
    # non-tensor column into a NonTensorStack indexed by batch dim, so the columns survive every
    # split and reorder -- but they are reached with data[key], and .non_tensor_batch does not exist
    # on a TensorDict at all. Reading it there is an AttributeError on the first backward.
    anchors_all = _omniopd_column(data, "omniopd_anchors")
    k_sem_all = _omniopd_column(data, "omniopd_k_sem")

    losses = torch.zeros_like(log_probs, dtype=torch.float32)
    audited = torch.zeros_like(response_mask)
    chunk_logp, k_sem_flat, rows_flat, pos_flat = [], [], [], []
    for b, (anchors, k_sem) in enumerate(zip(anchors_all, k_sem_all, strict=True)):
        if len(anchors) != len(k_sem):
            raise ValueError(f"row {b}: {len(anchors)} anchors but {len(k_sem)} k_sem values")
        T = int(response_mask[b].sum())
        for ci, t0 in enumerate(anchors):
            if t0 + C > T:
                raise ValueError(f"row {b} chunk {ci}: anchor {t0} + C {C} runs past response {T}")
            idx = torch.arange(t0, t0 + C, device=log_probs.device)
            if bool(audited[b, idx].any()):
                raise ValueError(f"row {b} chunk {ci}: overlaps an earlier chunk")
            audited[b, idx] = True
            chunk_logp.append(log_probs[b, idx])
            k_sem_flat.append(float(k_sem[ci]))
            rows_flat.append(b)
            pos_flat.append(idx)

    if not chunk_logp:
        # Not an error. A micro-batch can legitimately contain only responses shorter than one chunk,
        # and the objective is DEFINED there: with an empty audited set every position is unaudited,
        # so the loss is beta*KL everywhere. Raising instead would abort the optimizer step over a
        # batch whose composition is a property of the data, not of the configuration.
        # THE KEY SET MUST MATCH THE NORMAL PATH EXACTLY. Metrics are reduced across DP ranks, and a
        # rank emitting a different set either raises in Metric.aggregate_dp (when the empty
        # micro-batches land unevenly) or, when they land evenly, silently averages the shared keys
        # over a SUBSET of micro-batches with nothing to indicate it -- dropping L_kl on precisely
        # the batches where beta*KL is the entire loss.
        _empty_kl = torch.where(response_mask, beta * kl, torch.zeros_like(log_probs))
        _zero = torch.zeros((), device=log_probs.device, dtype=torch.float32)
        return _empty_kl, {
            "omniopd/L_chunk": Metric(AggregationType.MEAN, _zero),
            "omniopd/L_kl": Metric(AggregationType.MEAN, kl[response_mask].sum()),
            "omniopd/n_chunks": Metric(AggregationType.MEAN, _zero),
            "omniopd/n_unaudited": Metric(AggregationType.MEAN, response_mask.sum().float()),
            "omniopd/target_mean": Metric(AggregationType.MEAN, _zero),
            "omniopd/k_sem_mean": Metric(AggregationType.MEAN, _zero),
            "omniopd/pi_bar_mean": Metric(AggregationType.MEAN, _zero),
            "omniopd/rows_without_audit": Metric(AggregationType.MEAN,
                                                 torch.tensor(float(response_mask.shape[0]))),
        }

    lp = torch.stack(chunk_logp)                                   # (n_chunks, C)
    pi_bar = torch.exp(lp.mean(dim=-1))
    k_sem_t = torch.tensor(k_sem_flat, device=lp.device, dtype=torch.float32)
    target = ((k_sem_t + alpha * pi_bar) / (N + alpha)).detach()    # FID-5
    if target.requires_grad:
        raise ValueError("OmniOPD target carries gradient; it must be detached (FID-5)")
    tmin, tmax = float(target.min()), float(target.max())
    if not (0.0 <= tmin and tmax <= 1.0):
        raise ValueError(
            f"OmniOPD target outside [0,1]: [{tmin}, {tmax}]. pi_hat is a probability; outside that "
            f"range the objective is no longer purely reinforcing (FID-9), which means k_sem or N is "
            f"wrong upstream rather than that the loss is merely large.")

    per_tok = -(target[:, None] * lp)                               # (n_chunks, C)
    for i, (b, idx) in enumerate(zip(rows_flat, pos_flat, strict=True)):
        losses[b, idx] = per_tok[i]
    unaudited = response_mask & ~audited
    losses = torch.where(unaudited, beta * kl, losses)

    metrics = {
        "omniopd/L_chunk": Metric(AggregationType.MEAN, per_tok.sum()),
        # WITHOUT beta, matching the oracle (scripts/omniopd_step.py reports kl_tok.sum()). Reporting
        # beta*KL under the same name made the two artifacts differ by exactly 1/beta = 10x.
        "omniopd/L_kl": Metric(AggregationType.MEAN, kl[unaudited].sum()),
        "omniopd/n_chunks": Metric(AggregationType.MEAN, torch.tensor(float(len(k_sem_flat)))),
        "omniopd/n_unaudited": Metric(AggregationType.MEAN, unaudited.sum().float()),
        "omniopd/target_mean": Metric(AggregationType.MEAN, target.mean()),
        "omniopd/k_sem_mean": Metric(AggregationType.MEAN, k_sem_t.mean()),
        "omniopd/pi_bar_mean": Metric(AggregationType.MEAN, pi_bar.mean().detach()),
        "omniopd/rows_without_audit": Metric(
            AggregationType.MEAN,
            torch.tensor(float(sum(1 for a in anchors_all if len(a) == 0)))),
    }

    # EMITTED DIRECTLY, not only as Metric objects. Two smoke runs completed with three and four
    # optimizer steps and logged NO omniopd metric at all: _update_actor does return them under an
    # actor/ prefix, but the fully-async trainer's logged dict carried only fully_async/rollouter/*.
    # Until that routing is fixed, a run can train end to end and leave no evidence of what the
    # objective actually computed -- which is the difference between "it ran" and "it works". This
    # is the one place both terms and the target are known together.
    if os.environ.get("OPD_OMNIOPD_QUIET", "0") in ("0", "", "false", "False"):
        _lc = float(per_tok.sum())
        _lk = float(kl[unaudited].sum())
        _tm = float(target.mean())
        _km = float(k_sem_t.mean())
        _pm = float(pi_bar.mean())
        print(
            "[OMNIOPD-LOSS] L_chunk=%.4f L_kl=%.4f beta*L_kl=%.4f n_chunks=%d n_unaudited=%d "
            "target_mean=%.4f k_sem_mean=%.4f pi_bar_mean=%.4f identity=%.4f"
            % (_lc, _lk, beta * _lk, len(k_sem_flat), int(unaudited.sum()), _tm, _km, _pm,
               (_km + alpha * _pm) / (N + alpha)),
            flush=True,
        )
    return losses, metrics


@register_distillation_loss(
    DistillationLossSettings(names=["kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3"], use_estimator=True)
)  # type: ignore[arg-type]
def compute_distillation_loss_reverse_kl_estimator(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the distillation loss and related metrics using single-sample KL estimators.

    Uses the kl_penalty function from core_algos which supports various KL divergence
    estimators: "kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3".

    Returns:
    - distillation_losses: (bsz, resp_len)
    - distillation_metrics: Dictionary of metrics.
    """
    student_log_probs = no_padding_2_padding(model_output["log_probs"], data)
    teacher_log_probs = no_padding_2_padding(data["teacher_logprobs"], data).squeeze(-1)
    if data["response_mask"].is_nested:
        response_mask_bool = data["response_mask"].bool().to_padded_tensor(False)
    else:
        response_mask_bool = data["response_mask"].bool()
    assert teacher_log_probs.shape == student_log_probs.shape == response_mask_bool.shape

    loss_config: DistillationLossConfig = distillation_config.distillation_loss
    distillation_losses = kl_penalty(
        logprob=student_log_probs, ref_logprob=teacher_log_probs, kl_penalty=loss_config.loss_mode
    )
    # Since k1 can be negative, log the mean absolute loss.
    metrics = {
        "distillation/abs_loss": Metric(AggregationType.MEAN, distillation_losses[response_mask_bool].abs().mean()),
    }
    return distillation_losses, metrics
