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

import torch
from tensordict import TensorDict

from verl.base_config import BaseConfig
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.metric import AggregationType, Metric
from verl.workers.config import ActorConfig, DistillationConfig, DistillationLossConfig
from verl.workers.utils.losses import ppo_loss
from verl.workers.utils.padding import no_padding_2_padding

_TOKEN_SELECT_LOG_COUNTER = 0

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

    _mutable_fields = {"names"}

    def __post_init__(self):
        self.names = [self.names] if isinstance(self.names, str) else self.names
        if sum([self.use_topk, self.use_estimator]) != 1:
            raise ValueError(
                f"Expected only one of use_estimator, use_topk, but got {self.use_estimator=}, {self.use_topk=}."
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
    distill_loss, distill_metrics = distillation_loss(config, distillation_config, model_output, data)
    policy_loss, policy_metrics = ppo_loss(config, model_output, data, dp_group)
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
