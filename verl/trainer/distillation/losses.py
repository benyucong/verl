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

import os
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
    if not distillation_loss_config.use_task_rewards and not distillation_loss_config.use_policy_gradient:
        # no need to compute policy loss
        policy_loss = 0.0
        policy_metrics = {}
    else:
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


# --- DENSE PREFIX WARM-UP CURRICULUM -------------------------------------------------------
# (end_step, rollout horizon). The stage is a PURE FUNCTION of global_step, deliberately: any
# stage held as state -- an env var, a checkpointed counter -- goes missing on resubmission,
# and a chained curriculum then silently runs the wrong objective while every log line still
# looks healthy. global_step is already reconstructed from the checkpoint directory name on
# resume, so deriving the stage from it survives arbitrary job chaining for free.
OPD_DENSE_STAGES = ((20, 128), (40, 256), (60, 512), (80, 1024), (100, 2048), (120, 4096), (140, 8192))


def opd_dense_enabled() -> bool:
    """Opt-in. Unset for every other arm, so their rollout length is untouched."""
    return (os.environ.get("OPD_DENSE_CURRICULUM", "") or "").strip() not in ("", "0", "false", "False")


def opd_stage_horizon(step: int) -> int:
    """Rollout horizon for this optimizer step; 0 means 'use the full configured horizon'.

    Returns 0 for every step past the dense curriculum (141+), which is the vanilla PG phase and
    must use the unmodified baseline rollout length.
    """
    if not opd_dense_enabled():
        return 0
    for end, h in OPD_DENSE_STAGES:
        if step <= end:
            return h
    return 0


def opd_stage_name(step: int) -> str:
    h = opd_stage_horizon(step)
    return f"dense_H{h}" if h else "vanilla_pg"


def apply_forward_horizon(response_mask: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Forward-Horizon curriculum: grade only the first H tokens the policy generated.

    GENERATION IS NOT TRUNCATED -- only the graded span is. Cutting generation at H instead would
    supervise a span containing no termination, which is exactly the failure this project already
    measured in prefix-only SFT: 50 steps of loss on text that never ends taught the model not to
    end, truncation went 3.5% -> 29% within ten steps. Grading a prefix of a COMPLETE rollout is
    safe; producing only a prefix is not.

    Position is counted by cumsum over the mask, not by column index, so left/right padding and
    ragged batches all count real generated tokens only.

    The horizon arrives by environment variable because the curriculum runs as a sequence of
    separate verl invocations, each resuming the previous checkpoint. A per-stage constant needs
    no hydra schema change and no per-step plumbing into the FSDP workers, which inherit the
    environment. H <= 0 (or unset) means "grade everything", i.e. vanilla OPD.
    """
    raw = os.environ.get("OPD_FH_HORIZON", "")
    if not raw:
        return response_mask, 0
    try:
        horizon = int(raw)
    except ValueError:
        return response_mask, 0
    if horizon <= 0:
        return response_mask, 0
    # cumsum is NOT available on every tensor type this mask arrives as. In the FSDP path it can
    # be a nested tensor, where aten.cumsum.default is unimplemented and raises at the first
    # batch -- which is exactly how the first Forward-Horizon launch died. Densify first, and if
    # the type still refuses cumsum, fall back to a column index.
    dense = response_mask
    if getattr(dense, "is_nested", False):
        dense = dense.to_padded_tensor(0)
    try:
        position = torch.cumsum(dense.to(torch.int32), dim=-1)
    except (NotImplementedError, RuntimeError):
        # Column index counts padding as if it were generated, so it is only equivalent when the
        # response is right-padded -- which it is here, responses being written from index 0.
        # Preferred second, not first, because cumsum is correct under either padding.
        idx = torch.arange(1, dense.shape[-1] + 1, device=dense.device)
        position = idx.expand_as(dense)
    return dense * (position <= horizon).to(dense.dtype), horizon



def teacher_length_prior(response_mask: torch.Tensor, ref_len: float, lam: float) -> tuple:
    """Trajectory-level length prior calibrated to the TEACHER'S OWN rollout lengths.

    WHY THIS TERM EXISTS
    --------------------
    The diagnosis (results/length_explosion/DIAGNOSIS.md) establishes that teacher-forced local
    scoring carries no instruction to close earlier:

      * in 0 of 125 trajectories does the teacher put >1e-3 on `</think>` anywhere before the
        student's own close, though it endorses that close at p=1.0 once reached;
      * teacher surprisal and student-teacher disagreement both FALL over the redundant tail, so
        no per-token quantity distinguishes productive reasoning from re-derivation;
      * the k1 advantage is 25-35% less negative in the tail, so the objective discourages
        redundant continuation least.

    The information that IS missing sits in the teacher's own trajectory-length distribution:
    29,505 correct teacher traces close at a median of 6,621 think-tokens (~8,624 full response),
    implying a per-token closing hazard of ~1.9e-4 past 6k. The student assigns ~1e-12 there.
    That is 8 orders of magnitude, and it cannot be recovered from any teacher score on student
    text -- which is exactly why this term is trajectory-level rather than token-level.

    WHY NOT THE PER-TOKEN HAZARD FORM. The principled version supervises p_S(</think>|c_t) toward
    the empirical hazard directly, but that needs the `</think>` logit column inside the loss.
    These runs use use_remove_padding with ulysses_sequence_parallel_size=2, so a new per-token
    tensor would have to be threaded through the rmpad + SP gather + fused-kernel paths. That is
    the highest-risk edit available and this file has already produced several silent, plausible
    failures. This form tests the same hypothesis with a dense, post-gather, padded input.

    Returns (per_sequence_penalty[bsz], metrics). Penalty is 0 for sequences at or under ref_len,
    so a model that is already short is not pushed shorter.
    """
    lengths = response_mask.sum(dim=-1).to(torch.float32)
    excess = torch.clamp(lengths / max(ref_len, 1.0) - 1.0, min=0.0)
    penalty = lam * excess
    return penalty, {
        "distillation/len_prior_mean_excess": excess.mean().detach().item(),
        "distillation/len_prior_frac_over": (excess > 0).float().mean().detach().item(),
        "distillation/len_prior_mean_penalty": penalty.mean().detach().item(),
        "distillation/len_prior_mean_length": lengths.mean().detach().item(),
    }


def mixed_sft_term(log_prob: torch.Tensor, response_mask: torch.Tensor) -> tuple:
    """Per-sequence-normalised SFT loss over the offline prefix inside the response region.

    TAKES ALREADY-DENSE TENSORS. Both arguments must be the [bsz, max_resp] tensors the policy
    loss itself consumes. Earlier versions re-derived log_prob from model_output and called
    no_padding_2_padding themselves, which is correct only outside the packed path: with
    use_remove_padding and Ulysses sequence parallelism the re-derivation produced a different
    width and the update died with "size of tensor a (31632) must match tensor b (18068)".
    Reusing the tensors the PG branch already built removes an entire class of shape bug.

    THE PREFIX REGION is the leading run of zeros in response_mask before the first sampled
    token. That mask is exactly "1 iff the policy sampled it", so the prefix (supplied, not
    sampled) reads 0 and so does right padding -- but padding comes AFTER the last 1, so the
    leading run identifies the prefix unambiguously and no side-channel field is needed.

    NORMALISED SEPARATELY FROM OPD, deliberately. Averaging prefix and suffix tokens together
    would make the relative weight of the two terms a function of K, so shrinking the prefix
    across the curriculum would silently re-weight the objective and no stage would be comparable
    to another.

    Returns (None, {}) when no example carries a prefix, which is the K = 0 stage: the mixed loss
    then reduces to vanilla PG-OPD exactly.
    """
    rm = response_mask.to(torch.bool)
    any_sampled = rm.any(dim=-1)
    first_sampled = torch.argmax(rm.to(torch.int32), dim=-1)
    n_pref = torch.where(any_sampled, first_sampled, torch.zeros_like(first_sampled))
    if int(n_pref.sum().item()) == 0:
        return None, {}

    n_pref = n_pref.to(log_prob.device).reshape(-1, 1)
    pos = torch.arange(log_prob.shape[-1], device=log_prob.device).unsqueeze(0)
    sft_mask = (pos < n_pref)

    k = sft_mask.sum(-1)
    if int(k.sum().item()) == 0:
        return None, {}
    per_seq = -(log_prob * sft_mask).sum(-1) / k.clamp(min=1)
    has = k > 0
    loss = (per_seq * has).sum() / has.sum().clamp(min=1)
    return loss, {
        "distillation/sft_loss": loss.detach().item(),
        "distillation/sft_prefix_tokens": float(k.sum().item()),
        "distillation/sft_examples_with_prefix": float(has.sum().item()),
    }


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

    # TRUE GLOBAL TOKEN MEAN -- fixes two bugs at once.
    #
    # (1) config.global_batch_info is populated in exactly ONE place, inside ppo_loss
    #     (workers/utils/losses.py:65-68). The supervised branch below skips ppo_loss entirely
    #     when use_task_rewards and use_policy_gradient are both False -- which is precisely the
    #     distillation-only configuration -- so the dict is empty forever, agg_loss falls back to
    #     dp_size=1 and batch_num_tokens=loss_mask.sum(), and "token-mean" silently degrades to a
    #     LOCAL micro-batch mean whose scale rides on how many micro-batches dynamic batching
    #     happened to produce.
    # (2) In the policy-gradient branch, distillation_loss() is called BEFORE ppo_loss(), so even
    #     when ppo_loss does run, the value read here is one call stale -- and empty on the very
    #     first backward of every process, fresh run and every resume alike.
    #
    # The engine already computed the globally all-reduced count and put it on the batch
    # (workers/engine/fsdp/transformer_impl.py:691-696). Read that, with the same accessor
    # ppo_loss uses, and fall back silently only if a call path lacks the keys.
    # OPT-IN, deliberately. This is a genuine bug fix, but switching it on mid-flight would
    # change the loss normalisation of an experiment already in progress -- the four R_max clip
    # arms pick up this file on their next chained stage. Changing the objective under a running
    # comparison is worse than carrying a small known bug through it consistently, so arms opt in.
    if (os.environ.get("OPD_FIX_GLOBAL_TOKEN_MEAN", "") or "").strip() not in ("", "0", "false", "False"):
        try:
            config.global_batch_info["dp_size"] = data["dp_size"]
            config.global_batch_info["batch_num_tokens"] = data["batch_num_tokens"]
            config.global_batch_info["global_batch_size"] = data["global_batch_size"]
            config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor
        except (KeyError, TypeError):
            pass

    distillation_metrics.update(
        compute_distillation_loss_range(distillation_losses=distillation_losses, response_mask=response_mask)
    )
    if loss_config.loss_max_clamp is not None:
        # HOW OFTEN DOES THE CLAMP ACTUALLY BIND?
        # Without this, a null result from tightening loss_max_clamp is uninterpretable: it
        # cannot be told apart from a threshold that never fired.
        #
        # CAREFUL: the dataclass default is 10.0 but config/distillation/distillation.yaml
        # ships `loss_max_clamp: null`, and the YAML wins through hydra. So unless a run passes
        # it EXPLICITLY, this whole branch is dead and there is no clip at all -- which was the
        # case for every arm in this study before 2026-08-22. Setting it turns clamping ON
        # against an unclamped baseline; it does not "tighten" an existing 10.0.
        # Note compute_distillation_loss_range above is deliberately computed PRE-clamp, so
        # loss_min/loss_max keep reporting the raw range while these report the clipping.
        _c = loss_config.loss_max_clamp
        _m = response_mask.bool().to_padded_tensor(False) if response_mask.is_nested else response_mask.bool()
        _resp = distillation_losses[_m]
        if _resp.numel() > 0:
            distillation_metrics["distillation/clamp_frac_hi"] = Metric(
                aggregation=AggregationType.MEAN, value=(_resp > _c).float().mean())
            distillation_metrics["distillation/clamp_frac_lo"] = Metric(
                aggregation=AggregationType.MEAN, value=(_resp < -_c).float().mean())
            distillation_metrics["distillation/clamp_frac"] = Metric(
                aggregation=AggregationType.MEAN, value=(_resp.abs() > _c).float().mean())
            distillation_metrics["distillation/clamp_threshold"] = Metric(
                aggregation=AggregationType.MEAN, value=torch.tensor(float(_c)))
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
        _adv = -distillation_losses.detach()
        # INTERVENTION D: subtract a trajectory-level penalty for exceeding the teacher's own
        # typical length. Broadcast over tokens so every token in an over-long sequence carries
        # it, which is what makes it a TRAJECTORY signal rather than another local one.
        _lam_len = float(os.environ.get("OPD_LAMBDA_LEN", "0") or 0)
        if _lam_len:
            _ref = float(os.environ.get("OPD_REF_LEN", "8624") or 8624)
            _pen, _lm = teacher_length_prior(response_mask, _ref, _lam_len)
            _adv = _adv - _pen.unsqueeze(-1)
            distillation_metrics.update(_lm)
            distillation_metrics["distillation/len_prior_lambda"] = _lam_len
            distillation_metrics["distillation/len_prior_ref"] = _ref
        distillation_loss, pg_metrics = policy_loss_fn(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=_adv,
            response_mask=response_mask,
            loss_agg_mode=loss_agg_mode,
            config=loss_config,
            rollout_is_weights=rollout_is_weights,
        )
        pg_metrics = {f"distillation/{k[len('actor/') :]}": v for k, v in pg_metrics.items()}
        distillation_metrics.update(pg_metrics)
        _dense_log_prob, _dense_resp_mask = log_prob, response_mask
    else:
        # Directly backpropagate distillation loss as a supervised loss, as in https://arxiv.org/abs/2306.13649.
        if response_mask.is_nested:
            response_mask = response_mask.to_padded_tensor(False)
        distillation_loss = agg_loss(
            loss_mat=distillation_losses,
            loss_mask=response_mask,
            loss_agg_mode=loss_agg_mode,
            **config.global_batch_info,
        )

    # Mixed SFT+OPD: add the prefix SFT term to the UNCHANGED OPD loss. Everything above ran
    # exactly as it does for Reverse-Handoff, so lambda_SFT = 0 recovers RH-OPD identically.
    lam = float(os.environ.get("OPD_LAMBDA_SFT", "0") or 0)
    _dense_log_prob = locals().get("_dense_log_prob", None)
    if lam:
        sft, sft_metrics = (mixed_sft_term(_dense_log_prob, _dense_resp_mask)
                            if _dense_log_prob is not None else (None, {}))
        if sft is not None:
            distillation_metrics.update(sft_metrics)
            distillation_metrics["distillation/opd_loss"] = distillation_loss.detach().item()
            distillation_metrics["distillation/lambda_sft"] = lam
            distillation_loss = distillation_loss + lam * sft

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
    overlap_count = model_output.get("overlap_count")
    overlap_token_advantage = model_output.get("overlap_token_advantage")
    if overlap_count is not None and overlap_token_advantage is not None:
        overlap_count = no_padding_2_padding(overlap_count, data)
        overlap_token_advantage = no_padding_2_padding(overlap_token_advantage, data)
    if data["response_mask"].is_nested:
        response_mask_bool = data["response_mask"].bool().to_padded_tensor(False)
    else:
        response_mask_bool = data["response_mask"].bool()
    assert distillation_losses.shape == student_mass.shape == teacher_mass.shape == response_mask_bool.shape

    overlap_metrics = {}
    if overlap_count is not None and overlap_token_advantage is not None:
        assert overlap_count.shape == overlap_token_advantage.shape == response_mask_bool.shape
        valid_overlap_count = overlap_count[response_mask_bool]
        k = distillation_config.distillation_loss.topk
        assert k is not None
        # Diagnostics for tracking teacher/student top-k overlap in OPD, following
        # "Rethinking On-Policy Distillation of Large Language Models" (arXiv:2604.13016):
        # overlap ratio and average teacher-token KL contribution on overlapped tokens.
        overlap_metrics["distillation/overlap_ratio"] = (valid_overlap_count.float().mean() / k).item()
        overlap_position_mask = response_mask_bool & (overlap_count > 0)
        if overlap_position_mask.any():
            overlap_metrics["distillation/overlap_token_advantage"] = (
                overlap_token_advantage[overlap_position_mask].mean().item()
            )
        else:
            overlap_metrics["distillation/overlap_token_advantage"] = 0.0

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
        **overlap_metrics,
    }

    # CLOSING-TOKEN COVERAGE: what fraction of supervised positions had </think> / EOS inside
    # the teacher's retained top-k at all. Positions where they are absent receive ZERO gradient
    # about closing, however large K is.
    for _key, _name in (("has_close_token", "think_close"), ("has_eos_token", "eos")):
        _v = model_output.get(_key)
        if _v is None:
            continue
        _v = no_padding_2_padding(_v, data)
        _v = _v[response_mask_bool]
        if _v.numel() > 0:
            distillation_metrics[f"distillation/topk_covers_{_name}"] = _v.float().mean().item()

    # RETAINED TEACHER MASS, the quantity that says how good an approximation top-k is.
    # mean/min/max alone hide the tail; the fractions below 0.95 and 0.99 are what tell you
    # whether a position was effectively supervised by a truncated target.
    _tm = teacher_mass.float()
    if _tm.numel() > 0:
        distillation_metrics["distillation/teacher_mass_frac_lt_0.95"] = (_tm < 0.95).float().mean().item()
        distillation_metrics["distillation/teacher_mass_frac_lt_0.99"] = (_tm < 0.99).float().mean().item()
        # torch.quantile refuses inputs above ~16M elements; sort is exact and cheap here.
        _sorted = torch.sort(_tm).values
        _n = _sorted.numel()
        distillation_metrics["distillation/teacher_mass_median"] = _sorted[_n // 2].item()
        distillation_metrics["distillation/teacher_mass_p10"] = _sorted[max(0, int(0.10 * _n) - 1)].item()

    _renorm = bool(getattr(distillation_config.distillation_loss, "renormalize_teacher_topk", False))
    distillation_metrics["distillation/teacher_renormalized"] = float(_renorm)
    # Curriculum stage and horizon, so the objective in force at each step is visible in wandb
    # rather than inferred from the step number by a reader months later.
    try:
        _gs = int(data["global_steps"])
        distillation_metrics["distillation/curriculum_step"] = float(_gs)
        distillation_metrics["distillation/curriculum_horizon"] = float(opd_stage_horizon(_gs))
    except (KeyError, TypeError, ValueError):
        pass
    if _renorm:
        # With a renormalised target the divergence is a genuine KL and is non-negative up to
        # float error, so clamping should never bind. Record it if it ever does rather than
        # silently hiding a bug behind the clamp.
        _neg = (distillation_losses < 0)
        distillation_metrics["distillation/kl_negative_frac"] = _neg.float().mean().item()
        distillation_metrics["distillation/kl_most_negative"] = distillation_losses.min().item()
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
    # All of the following are computed on the RAW estimator, before loss_max_clamp, so they
    # describe the teacher-student relationship rather than the clipped training signal.
    _d = distillation_losses[response_mask_bool].float()      # log p_S - log p_T, per token
    _t = teacher_log_probs[response_mask_bool].float()
    _s = student_log_probs[response_mask_bool].float()

    # Since k1 can be negative, log the mean absolute loss.
    metrics = {
        "distillation/abs_loss": Metric(AggregationType.MEAN, _d.abs().mean()),
    }
    if _d.numel() > 0:
        # THE ACTUAL KL, WHICH abs_loss IS NOT.
        # Tokens are sampled from the student, so the SIGNED mean of (log p_S - log p_T) is the
        # k1 single-sample estimator of the reverse KL, KL(pi_S || pi_T). abs_loss throws the
        # sign away, and therefore cannot distinguish "teacher and student disagree a lot, in
        # both directions" from "the student is systematically more confident than the teacher"
        # -- which are opposite situations for distillation.
        metrics["distillation/kl_k1"] = Metric(AggregationType.MEAN, _d.mean())

        # k3 = (r - 1) - log r  with  r = p_T/p_S.  Non-negative by construction and much lower
        # variance than k1, so it is the better convergence read even though k1 is what the
        # gradient actually uses. The exponent is clamped only to keep the METRIC finite: on our
        # runs the raw k1 reaches -15, and exp(15) is already 3e6.
        _r = torch.exp((-_d).clamp(max=20.0))
        metrics["distillation/kl_k3"] = Metric(AggregationType.MEAN, ((_r - 1.0) - (-_d)).mean())

        # The two halves separately, so a moving KL can be attributed to the student drifting
        # or to the teacher scoring the student's new tokens differently.
        metrics["distillation/teacher_logprob_mean"] = Metric(AggregationType.MEAN, _t.mean())
        metrics["distillation/student_logprob_mean"] = Metric(AggregationType.MEAN, _s.mean())
        metrics["distillation/teacher_ppl"] = Metric(AggregationType.MEAN, torch.exp(-_t.mean()))
    return distillation_losses, metrics
