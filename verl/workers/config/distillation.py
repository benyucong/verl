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

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from omegaconf import OmegaConf

from verl.base_config import BaseConfig
from verl.utils.config import omega_conf_to_dataclass

from .rollout import RolloutConfig

__all__ = ["DistillationLossConfig", "DistillationTeacherModelConfig", "DistillationConfig"]

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@dataclass
class DistillationLossConfig(BaseConfig):
    """Configuration for distillation loss settings.

    loss_mode (str):
        Distillation loss function to use.
    topk (int, optional):
        Number of top tokens to consider for top-k distillation losses.
    use_task_rewards (bool):
        Whether to include task rewards alongside distillation loss.
    distillation_loss_coef (float):
        Coefficient for distillation loss when combined with task rewards.
    loss_max_clamp (float, optional):
        Maximum value to clamp distillation loss. If None, no clamping is applied.
    log_prob_min_clamp (float, optional):
        Minimum value to clamp log probabilities for stability, e.g., log q - log p where p or q are
        very close to zero. If None, no clamping is applied.
    use_policy_gradient (bool):
        Whether to incorporate distillation loss as a reward, as done
        by https://thinkingmachines.ai/blog/on-policy-distillation/. Recommended to use loss_mode=k1.
        Otherwise, distillation loss is directly backpropagated as a supervised loss,
        as in https://arxiv.org/abs/2306.13649. Recommended to use loss_mode=k3 or forward_kl_topk.
    policy_loss_mode (str):
        Name of the policy loss to use when use_policy_gradient is true.
    clip_ratio (float):
        PPO clipping ratio for policy loss.
    clip_ratio_low (float):
        Lower bound for PPO clipping ratio.
    clip_ratio_high (float):
        Upper bound for PPO clipping ratio.
    loss_settings (DistillationLossSettings, optional):
        Runtime-populated settings based on loss_mode. Not set by user.
    """

    loss_mode: str = "k3"
    topk: Optional[int] = 128
    use_task_rewards: bool = True
    distillation_loss_coef: float = 1.0
    loss_max_clamp: Optional[float] = 10.0
    log_prob_min_clamp: Optional[float] = -10.0

    # --- TIP-style post-teacher token selection (loss masking). Default OFF => byte-equivalent. ---
    # token_select_mode: "none" | "entropy" | "soft_or"; token_retention in (0,1]; token_select_scope:
    # "response" (per-rollout top-rho, paper-faithful) | "batch" (per-micro-batch global top-rho).
    # Env-overridable via OPD_TOKEN_SELECT_MODE / OPD_TOKEN_RETENTION / OPD_TOKEN_SELECT_SCOPE /
    # OPD_TOKEN_SELECT_LOG (read in __post_init__).
    token_select_mode: str = "none"
    token_retention: float = 1.0
    token_select_scope: str = "response"
    token_select_log: bool = False

    use_policy_gradient: bool = True
    policy_loss_mode: str = "vanilla"
    clip_ratio: float = 0.2
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.2

    # Store global batch info for loss aggregation:
    # dp_size: data parallel size
    # batch_num_tokens: number of valid tokens in global batch
    # global_batch_size: global batch size
    global_batch_info: dict = field(default_factory=dict)

    # Store distillation loss settings for computing the specified loss_mode
    # Not set by user, populated at runtime
    loss_settings: Optional[dict] = None

    def __post_init__(self):
        self._mutable_fields.add("loss_settings")
        from verl.trainer.distillation.losses import DistillationLossSettings, get_distillation_loss_settings

        self.loss_settings: DistillationLossSettings = get_distillation_loss_settings(self.loss_mode)

        # --- TIP-style token-selection env overrides (default OFF => byte-equivalent) ---
        for _f in ("token_select_mode", "token_retention", "token_select_scope", "token_select_log"):
            self._mutable_fields.add(_f)
        self.token_select_mode = os.environ.get("OPD_TOKEN_SELECT_MODE", self.token_select_mode)
        self.token_retention = float(os.environ.get("OPD_TOKEN_RETENTION", self.token_retention))
        self.token_select_scope = os.environ.get("OPD_TOKEN_SELECT_SCOPE", self.token_select_scope)
        if "OPD_TOKEN_SELECT_LOG" in os.environ:
            self.token_select_log = bool(int(os.environ["OPD_TOKEN_SELECT_LOG"]))
        if self.token_select_mode not in ("none", "random", "entropy", "soft_or"):
            raise ValueError(
                f"OPD_TOKEN_SELECT_MODE must be none|random|entropy|soft_or, got {self.token_select_mode!r}"
            )
        if not (0.0 < self.token_retention <= 1.0):
            raise ValueError(f"OPD_TOKEN_RETENTION must be in (0, 1], got {self.token_retention}")
        if self.token_select_scope not in ("response", "batch"):
            raise ValueError(f"OPD_TOKEN_SELECT_SCOPE must be response|batch, got {self.token_select_scope!r}")
        if self.token_select_mode != "none":
            print(
                f"[TOKEN-SELECT-CFG] resolved mode={self.token_select_mode} retention={self.token_retention} "
                f"scope={self.token_select_scope} log={self.token_select_log}",
                flush=True,
            )

        if self.policy_loss_mode != "vanilla":
            raise NotImplementedError(
                f"Only vanilla policy loss is currently supported when use_policy_gradient is True, "
                f"but got {self.policy_loss_mode}."
            )

        if self.use_policy_gradient and self.loss_mode == "forward_kl_topk":
            print(
                "WARNING: forward_kl_topk is most effective as a supervised distillation loss "
                "(use_policy_gradient=False). With policy gradient, the update uses only the sampled"
                " token's logprob ∇logπ(a), so the top-k distributional signal (how non-sampled logits "
                "should move) is largely unused."
            )

        if not self.use_policy_gradient and self.loss_mode == "k1":
            raise ValueError(
                "Directly backpropagating k1 loss is incorrect since gradient of k1 loss"
                " wrt model weights does not depend on teacher log probabilities."
            )


@dataclass
class DistillationTeacherModelConfig(BaseConfig):
    """Configuration for on-policy distillation teacher.

    key (str, optional):
        Identifier to route examples to the teacher model in multi-teacher setting.
    model_path (str, optional):
        Model path for the teacher model. Can be a local path or a Hugging Face model
    inference (RolloutConfig):
        Rollout configuration for the teacher model inference during distillation.
    num_replicas (int):
        Number of inference replicas of this teacher to launch. Each replica occupies
        `per_replica_world_size` GPUs (= inference.data_parallel_size *
        inference.tensor_model_parallel_size * inference.pipeline_model_parallel_size),
        so the teacher's total GPU footprint is
        `num_replicas * per_replica_world_size`.
    """

    _mutable_fields = BaseConfig._mutable_fields | {"num_replicas", "key"}

    key: Optional[str] = None
    model_path: Optional[str] = None
    inference: RolloutConfig = field(default_factory=RolloutConfig)
    num_replicas: Optional[int] = 0

    @property
    def per_replica_world_size(self) -> int:
        return (
            self.inference.tensor_model_parallel_size
            * self.inference.data_parallel_size
            * self.inference.pipeline_model_parallel_size
        )

    @property
    def world_size(self) -> int:
        return self.num_replicas * self.per_replica_world_size

    def check_configured(self):
        if self.model_path is None:
            raise ValueError("model_path must be specified for distillation teacher model config.")
        if self.key is None:
            raise ValueError("key must be specified for distillation teacher model config.")
        if self.num_replicas is None:
            raise ValueError("num_replicas must be specified for distillation teacher model config.")

    def validate_and_prepare_for_distillation(
        self, use_topk: bool, topk: Optional[int], generation_tokens: Optional[int] = None,
        prefix_length: Optional[int] = None, total_horizon: Optional[int] = None,
    ) -> None:
        """Dimension the teacher engine for how this objective actually uses it.

        A SCORING teacher (top-k or estimator objectives) reads prompt+response and emits one token,
        so its whole context is prompt, and response_length collapses to 1.

        A GENERATING teacher (OmniOPD) is the opposite: it reads a PREFIX that stops before the
        audited chunk and then writes `generation_tokens` of its own. Leaving response_length at 1
        here is what makes the teacher structurally unable to generate -- the engine would be built
        with room for a single token, and every continuation would come back empty or truncated with
        no indication that the configuration, not the model, was responsible.

        The prefix is at most prompt + response - C, because a chunk of C tokens must fit inside the
        response after its anchor. So generation costs the engine NOTHING extra over scoring: the
        same prompt+response total is merely split differently.
        """
        max_model_len = self.inference.max_model_len
        student_prompt_length = self.inference.prompt_length
        student_response_length = self.inference.response_length
        required_context_len = student_prompt_length + student_response_length + 1
        if max_model_len is not None and required_context_len > max_model_len:
            raise ValueError(
                "Distillation teacher inference requires room for the student prompt, the full student "
                f"response, and one generated token, but got {student_prompt_length=}, "
                f"{student_response_length=}, {required_context_len=}, {max_model_len=}."
            )

        if prefix_length is not None:
            # STATE-CREDIT. Unlike OmniOPD, whose prefix + generation always sums back to
            # prompt + response, this reads prompt + response[:max_depth] and then writes a FULL
            # continuation of B tokens -- so the engine needs prompt + max_depth + B, which can
            # exceed the scoring total above. Asserted here, at boot: generate_n RAISES rather than
            # clamping when the budget does not fit (vllm_async_server), and discovering that
            # mid-rollout means the whole trajectory has already been paid for.
            if generation_tokens is None or generation_tokens < 1:
                raise ValueError(f"state-credit needs a continuation budget B >= 1, got {generation_tokens}")
            if prefix_length >= student_response_length:
                raise ValueError(
                    f"state_credit depth {prefix_length} must be strictly inside the student "
                    f"response ({student_response_length}); a depth at or past the end has no "
                    f"interior state to evaluate.")
            if total_horizon is not None:
                # budget_mode='remaining': depth d writes B - d tokens, so EVERY sequence is
                # prompt + d + (B - d) = prompt + B regardless of depth. The context is flat in
                # depth, which is why this mode is far cheaper to serve than the sum below.
                need = student_prompt_length + total_horizon
                detail = (f"prompt {student_prompt_length} + horizon B {total_horizon}, flat in "
                          f"depth under budget_mode='remaining'")
                remedy = "Raise max_model_len or lower B."
            else:
                # budget_mode='fixed': the deepest prefix still gets a full B on top of it.
                need = student_prompt_length + prefix_length + generation_tokens
                detail = (f"prompt {student_prompt_length} + deepest depth {prefix_length} + B "
                          f"{generation_tokens}")
                remedy = ("Raise max_model_len, lower the deepest depth, lower B, or switch to "
                          "budget_mode='remaining', whose context does not grow with depth.")
            if max_model_len is not None and need > max_model_len:
                raise ValueError(
                    f"state-credit continuations need {need} tokens of context ({detail}) but the "
                    f"teacher engine has max_model_len={max_model_len}. {remedy}")
            self.inference.prompt_length = student_prompt_length + prefix_length
            self.inference.response_length = generation_tokens
            self._validate_topk_logprobs(use_topk=use_topk, topk=topk)
            return

        if generation_tokens is not None:
            if generation_tokens < 1:
                raise ValueError(f"generation_tokens must be >= 1, got {generation_tokens}")
            if generation_tokens >= student_response_length:
                raise ValueError(
                    f"Generative teaching needs a chunk of {generation_tokens} tokens to fit inside "
                    f"the student response ({student_response_length}); with C >= the response "
                    f"length no anchor can be placed and the teacher would have nothing to continue."
                )
            # prefix is at most prompt + response - C; the teacher then writes C
            self.inference.prompt_length = student_prompt_length + student_response_length - generation_tokens
            self.inference.response_length = generation_tokens
        else:
            self.inference.prompt_length = self.inference.prompt_length + self.inference.response_length
            self.inference.response_length = 1
        self._validate_topk_logprobs(use_topk=use_topk, topk=topk)

    def _validate_topk_logprobs(self, use_topk: bool, topk: Optional[int]) -> None:
        if not use_topk:
            return
        if topk is None:
            raise ValueError("topk must be specified when use_topk is True.")

        engine_name = self.inference.name
        engine_kwargs = self.inference.engine_kwargs
        match engine_name:
            case "vllm":
                vllm_engine_kwargs = dict(engine_kwargs.get("vllm", {}))
                max_logprobs = vllm_engine_kwargs.get("max_logprobs")
                if max_logprobs is None:
                    vllm_engine_kwargs["max_logprobs"] = topk
                    max_logprobs = topk
                if max_logprobs < topk:
                    raise ValueError(
                        f"VLLM max_logprobs ({max_logprobs}) must be >= distillation_loss topk "
                        f"({topk}) to enable distillation loss computation."
                    )
                engine_kwargs["vllm"] = vllm_engine_kwargs
            case "sglang":
                # SGLang's top_logprobs_num is a per-request parameter, so there is no
                # engine-boot cap to align (unlike vLLM's max_logprobs). The async
                # server translates sampling_params["prompt_logprobs"] into
                # return_logprob + logprob_start_len=0 + top_logprobs_num at call time.
                pass
            case _:
                raise NotImplementedError(
                    f"DistillationTeacherModelConfig does not support inference engine {engine_name}"
                )


@dataclass
class OmniOPDConfig(BaseConfig):
    """Parameters of the OmniOPD generative-critic objective.

    M (int): audited chunks per trajectory, taken at the highest full-vocabulary entropy positions.
    N (int): teacher continuations sampled per chunk.
    C (int): tokens per chunk, and per teacher continuation.
    alpha (float): Dirichlet-Multinomial smoothing of the Bayesian target.
    beta (float): weight of the trust-region anchor on unaudited positions.
    phi (str): semantic metric, "ned" or "rouge1". REQUIRED to be explicit -- a silently defaulted
        metric changes k_sem and therefore the objective (FID-3).
    entropy_topk (int): 0 means exact full-vocabulary entropy for chunk selection. FID-1: entropy
        from a truncated tail systematically under-estimates H and selects DIFFERENT chunks, which
        changes the algorithm rather than its cost. Non-zero is an explicit, recorded deviation.
    """

    M: int = 10
    N: int = 10
    C: int = 50
    alpha: float = 1.0
    beta: float = 0.1
    phi: str = "ned"
    entropy_topk: int = 0

    def __post_init__(self):
        if self.phi not in ("ned", "rouge1"):
            raise ValueError(f"omniopd.phi must be 'ned' or 'rouge1', got {self.phi!r}")
        for name in ("M", "N", "C"):
            if getattr(self, name) < 1:
                raise ValueError(f"omniopd.{name} must be >= 1, got {getattr(self, name)}")
        if self.alpha <= 0:
            raise ValueError(f"omniopd.alpha must be > 0, got {self.alpha}")
        if self.beta < 0:
            raise ValueError(f"omniopd.beta must be >= 0, got {self.beta}")
        if self.entropy_topk != 0:
            logger.warning(
                "omniopd.entropy_topk=%d deviates from FID-1: chunk selection is argmax_M over the "
                "student's FULL-vocabulary entropy, and a truncated tail under-estimates H and "
                "selects different chunks. This changes the algorithm, not its speed.",
                self.entropy_topk,
            )


# Under budget_mode='remaining' the deepest state must still have room to reach an answer.
# Below this, a Phi of 0 means "did not fit", not "not solvable".
_MIN_REMAINING_BUDGET = 1024


@dataclass
class StateCreditConfig(BaseConfig):
    """State-Credit OPD: credit a reasoning chunk by the change in how solvable the state becomes.

        Phi(s) = E_{c ~ pi_C(.|s)} [ Q(x, s||c) ]      estimated with M continuations
        Delta_j = Phi(s_{k_j}) - Phi(s_{k_{j-1}})      progress made by chunk j

    depths: INTERIOR cut points, ascending. The terminal state is not listed -- Phi(s_T) is the
        task reward itself and needs no continuations. Phi(s_0) is not computed either: it cancels
        exactly under leave-one-out centering.
    M: continuations per Phi estimate. The dominant recurring cost -- it multiplies by depths, by
        rollouts per problem, and by every optimizer step. Measured separation at M=4 was +0.104
        against a per-problem sampling sd of ~0.35, so this is the floor rather than a default.
    B: the total response horizon the continuation is scored against, in tokens. Must be large
        enough that continuations reach an answer: at 20k thinking-on traces, a probe at B=1024
        truncated 99.3% of continuations and measured nothing, while B=20480 truncated 6.9%.
    budget_mode: how B converts into a per-depth continuation budget.
        "remaining" (default) gives depth d exactly B - d tokens, so prefix + continuation obeys the
            SAME horizon the student trains under. Phi is then one functional everywhere -- "is this
            state still solvable within the task's budget" -- and Delta measures progress toward
            finishing on time. It also makes the teacher's context constant at prompt + B.
        "fixed" gives every depth a full B tokens on top of its prefix. Phi is then a different,
            budget-free question, and a deep state is scored with more total room than the task ever
            allows: prompt + max(depths) + B of teacher context, and an optimistic Phi at depth.
        Neither is "unbiased" in the abstract -- they estimate different Phi. "remaining" is the one
        that matches the training objective, so it is the default.
    beta: weight of the state term against the token-level OPD loss, in
        u_t = a_t^OPD + beta * G_tilde_{j(t)}.
    cont_cap: hard ceiling on a single continuation, in tokens. 0 disables it and the budget is
        whatever budget_mode says. Non-zero takes min(budget, cont_cap).

        THIS CHANGES WHAT PHI MEANS and is not a free speedup. Measured at B=16384 with
        budget_mode=remaining: 27-31% of continuations ran to their cap, and working back from the
        mean length (5127 tokens) those truncated ones account for ~64% of ALL teacher decode
        tokens. A truncated continuation scores 0 whether or not the state was solvable, so that
        majority of the teacher's work buys a degenerate measurement.

        Capping at 4096 cuts mean continuation length to ~3050 -- about 40% less teacher work --
        but Phi then answers "solvable within cont_cap tokens from here" instead of "within the
        task's remaining budget". Both are legitimate operational definitions; they are not the
        same one, and Phi values measured under different caps must not be pooled. The cap rides
        the record (sc_fp_cont_cap) for exactly that reason.
    credit_norm: how a chunk's centered credit reaches its tokens.
        "broadcast" (default) gives every token in the chunk the same G_tilde, leaving the 1/T
            normalisation to the loss aggregator.
        "per_chunk" divides by (chunks in the row * chunk length), giving every chunk equal TOTAL
            weight. That also gives a token in a short terminal chunk more weight than one in a full
            fixed-length chunk, so it is an ablation rather than the default.
    base_loss_mode: the token-level OPD estimator that supplies a_t^OPD. "k1" is vanilla OPD and
        keeps the dense base signal. "none" drops it, leaving state credit alone -- a diagnostic
        arm, not the method.
    verifier_fast: grade interior continuations with the FAST verifier. Default False so Phi and the
        terminal reward come from the same Q, as Sec 4 requires. Setting it True is a measured
        speedup with a known bias: the fast grader skips the is_latex_equal recall pass, so it can
        only mark a correct continuation wrong, which inflates every terminal delta by a
        problem-dependent amount that leave-one-out does not remove.
    pi_c_key: which teacher_models entry serves as the FROZEN continuation model.
        ONLY meaningful with a single teacher. Continuations route on the SAMPLE's teacher_key, the
        same way scoring does, so pi_C is whatever teacher that sample routes to. With one teacher
        that is unambiguous and equals pi_T, which is the configuration Sec 8's dual-use probe
        assumes. With several, this key would name one model while routing picked another per
        sample -- so it is refused rather than silently ignored.
    min_survivors: per-(group, depth) floor for leave-one-out. Below this the chunk is DROPPED, not
        centered against zero -- an absent baseline would otherwise become a maximal-magnitude
        target.
    """

    depths: list[int] = field(default_factory=list)
    M: int = 4
    B: int = 20480
    budget_mode: str = "remaining"
    cont_cap: int = 0
    beta: float = 1.0
    credit_norm: str = "broadcast"
    base_loss_mode: str = "k1"
    verifier_fast: bool = False
    pi_c_key: str = "math"
    min_survivors: int = 3

    _mutable_fields = {"depths"}

    def __post_init__(self):
        if not self.depths:
            return
        self.depths = [int(d) for d in self.depths]
        if any(d <= 0 for d in self.depths):
            raise ValueError(f"state_credit.depths must be positive, got {self.depths}")
        if self.depths != sorted(self.depths) or len(set(self.depths)) != len(self.depths):
            raise ValueError(f"state_credit.depths must be strictly ascending, got {self.depths}")
        if self.M < 1:
            raise ValueError(f"state_credit.M must be >= 1, got {self.M}")
        if self.M == 1:
            # M=1 makes Phi a single Bernoulli draw per state, so Delta lands in {-1, 0, +1}. It is
            # still UNBIASED -- the guard used to refuse it at M<2 with no stated reason, and there
            # is no mathematical bar. What it costs is measurable, and was measured (2026-08-29,
            # n=3517 Phi estimates from the M=4 run):
            #
            #   the latent per-state solvability p is nearly BIMODAL -- 61.1% of states scored 0/4
            #   and 23.3% scored 4/4 -- so E[p(1-p)] is only 0.0434 and the binomial term shrinks
            #   fast. SE(Delta) = sqrt(2*0.0434/M): 0.295 at M=1, 0.208 at M=2, 0.147 at M=4.
            #   Against the documented 0.104 separation that is a per-transition signal/noise of
            #   0.35 / 0.50 / 0.71. Leave-one-out centering over a problem's rollouts recovers
            #   roughly sqrt(group size) of it.
            #
            # Allowed so the M sweep can be run, NOT recommended: at M=1 a single verifier failure
            # or truncated draw decides the state outright, with no other sample to average against.
            logger.warning(
                "[STATE-CREDIT] M=1: Phi is a single Bernoulli draw per state (SE(Delta)~0.295 "
                "against a ~0.104 separation). Unbiased but the noisiest setting that runs; "
                "credit quality depends entirely on leave-one-out group averaging.")
        if self.B < 1:
            raise ValueError(f"state_credit.B must be >= 1, got {self.B}")
        if self.cont_cap < 0:
            raise ValueError(f"state_credit.cont_cap must be >= 0 (0 disables), got {self.cont_cap}")
        if 0 < self.cont_cap < _MIN_REMAINING_BUDGET:
            raise ValueError(
                f"state_credit.cont_cap={self.cont_cap} is under the {_MIN_REMAINING_BUDGET}-token "
                f"floor. Below it Phi measures whether the continuation fits, not whether the state "
                f"is solvable -- the same failure the budget_mode floor exists to prevent.")
        if self.credit_norm not in ("broadcast", "per_chunk"):
            raise ValueError(
                f"state_credit.credit_norm must be 'broadcast' or 'per_chunk', got "
                f"{self.credit_norm!r}")
        if self.base_loss_mode not in ("k1", "kl", "abs", "mse", "k2", "low_var_kl", "k3", "none"):
            raise ValueError(
                f"state_credit.base_loss_mode must be a KL estimator name or 'none', got "
                f"{self.base_loss_mode!r}")
        if self.budget_mode not in ("remaining", "fixed"):
            raise ValueError(
                f"state_credit.budget_mode must be 'remaining' or 'fixed', got {self.budget_mode!r}")
        if self.budget_mode == "remaining":
            # B is the shared horizon here, so it has to leave the DEEPEST state something to write
            # with. A budget of a few hundred tokens measures truncation, not solvability, so this
            # refuses at boot rather than returning a Phi of ~0 that looks like a hard state.
            deepest = max(self.depths)
            if self.B - deepest < _MIN_REMAINING_BUDGET:
                raise ValueError(
                    f"state_credit.budget_mode='remaining' leaves depth {deepest} only "
                    f"{self.B - deepest} tokens (B={self.B}), under the {_MIN_REMAINING_BUDGET}-token "
                    f"floor. At that budget Phi measures whether the continuation fits, not whether "
                    f"the state is solvable. Raise B, or drop the deepest depth.")
        if self.min_survivors < 2:
            raise ValueError(
                f"state_credit.min_survivors must be >= 2: leave-one-out needs at least one other "
                f"trajectory to center against, got {self.min_survivors}")


@dataclass
class DistillationConfig(BaseConfig):
    """Configuration for on-policy distillation.

    enabled (bool):
        Whether on-policy distillation is enabled.
    n_gpus_per_node (int):
        Number of GPUs per node in the teacher resource pool.
    nnodes (int):
        Number of nodes in the teacher resource pool.
    teacher_models (dict[str, TeacherModelConfig]):
        Configurations for teacher models used for multi-teacher distillation.
    teacher_key (str):
        Key to route examples to the appropriate teacher model in multi-teacher setups. Should correspond to a field in
        the data proto, e.g., data_source.
    distillation_loss (DistillationLossConfig):
    Configuration for distillation loss settings.

    NOTE: The `teacher_model` entry is in the `teacher_models` dict by default.
    Since it is popped when other teacher entries are added, using `teacher_model` as
    one of several keys silently drops it. For example, the following CLI overrides result
    in ONLY `teacher_model2` being used:

    ```bash
    distillation.teacher_models.teacher_model.key=openai/gsm8k
    distillation.teacher_models.teacher_model.model_path=Qwen/Qwen3-4B
    +distillation.teacher_models.teacher_model2.key=hiyouga/geometry3k
    +distillation.teacher_models.teacher_model2.model_path=Qwen/Qwen3-VL-4B-Instruct
    ```
    Instead, give the first teacher a different name:

    ```bash
    +distillation.teacher_models.teacher_model1.key=openai/gsm8k
    +distillation.teacher_models.teacher_model1.model_path=Qwen/Qwen3-4B
    +distillation.teacher_models.teacher_model2.key=hiyouga/geometry3k
    +distillation.teacher_models.teacher_model2.model_path=Qwen/Qwen3-VL-4B-Instruct
    ```
    """

    _mutable_fields = BaseConfig._mutable_fields | {"teacher_models", "n_gpus_per_node", "nnodes"}

    enabled: bool = False
    n_gpus_per_node: int = 0
    nnodes: int = 0
    teacher_models: dict[str, DistillationTeacherModelConfig] = field(default_factory=dict)
    teacher_key: str = "data_source"
    distillation_loss: DistillationLossConfig = field(default_factory=DistillationLossConfig)
    omniopd: "OmniOPDConfig" = field(default_factory=lambda: OmniOPDConfig())
    state_credit: "StateCreditConfig" = field(default_factory=lambda: StateCreditConfig())

    def _teacher_generation_tokens(self):
        """How many tokens the teacher must be able to GENERATE, or None if it only scores."""
        ls = self.distillation_loss.loss_settings
        if getattr(ls, "use_teacher_continuations", False):
            sc = self.state_credit
            if sc.budget_mode == "remaining" and sc.depths:
                # The SHALLOWEST depth gets the most room, so that is what the engine's generation
                # limit has to cover. Sizing it from the deepest instead would clamp the shallow
                # continuations, and a clamped Phi at depth 0 is exactly the baseline every Delta
                # is measured against.
                return sc.B - min(int(d) for d in sc.depths)
            return sc.B
        if ls.use_teacher_generation:
            return self.omniopd.C
        return None

    def _teacher_total_horizon(self):
        """The flat prompt+B context bound under budget_mode='remaining', else None.

        None means the caller falls back to prompt + deepest depth + B, which is the correct (and
        strictly larger) bound for budget_mode='fixed'.
        """
        ls = self.distillation_loss.loss_settings
        sc = self.state_credit
        if (getattr(ls, "use_teacher_continuations", False) and sc.depths
                and sc.budget_mode == "remaining"):
            return sc.B
        return None

    def _teacher_prefix_length(self):
        """Longest prompt the teacher will be handed, or None to keep the rollout's default.

        State-Credit continues from prompt + response[:max(depths)], so the engine must be
        dimensioned for that prefix PLUS B. Asserting it here fails at boot; discovering it at
        runtime means generate_n raises mid-rollout after the work is already paid for.
        """
        ls = self.distillation_loss.loss_settings
        if getattr(ls, "use_teacher_continuations", False) and self.state_credit.depths:
            return max(self.state_credit.depths)
        return None

    def __post_init__(self):
        if not self.enabled:
            return
        ls = self.distillation_loss.loss_settings
        if getattr(ls, "use_teacher_continuations", False) and not self.state_credit.depths:
            raise ValueError(
                "loss_mode requires teacher continuations but state_credit.depths is empty: there "
                "are no interior states to evaluate, so the objective would be identical to the "
                "token-level loss while paying for a continuation engine.")
        if getattr(ls, "use_teacher_continuations", False) and not self.distillation_loss.use_policy_gradient:
            # The state term enters as an ADVANTAGE (u_t = a^OPD + beta*G~) and only becomes an
            # objective through the clipped ratio. Backpropagated as a supervised loss it is a
            # constant with zero gradient: the run trains on the token term alone and reports
            # perfectly healthy state_credit/* metrics the whole time.
            raise ValueError(
                "state_credit requires actor_rollout_ref.actor.distillation_loss.use_policy_gradient"
                "=True. The state term is an advantage, not a supervised target, and with the "
                "policy-gradient wrap off it would contribute exactly zero gradient while every "
                "state_credit metric still looked correct.")

        self.teacher_models = self._resolve_teacher_models()
        if getattr(ls, "use_teacher_continuations", False) and len(self.teacher_models) > 1:
            # AFTER _resolve_teacher_models, which pops the unused `teacher_model` template entry --
            # before it, a perfectly ordinary single-teacher config still counts two.
            #
            # pi_c_key cannot select the continuation model: generate_chunk_continuations resolves
            # the teacher from the SAMPLE's routing key, exactly like scoring. With one teacher that
            # is the same thing; with several, pi_C would vary per sample while pi_c_key claimed
            # otherwise -- and Phi is only a well-defined functional if pi_C is one frozen policy.
            # Sec 8's dual-use assumption also requires pi_C == pi_T.
            raise ValueError(
                f"state_credit is configured with {len(self.teacher_models)} teacher models "
                f"({sorted(self.teacher_models)}). Continuations route per-sample like scoring "
                f"does, so pi_C would differ between samples and Phi would not be one functional. "
                f"state_credit.pi_c_key does NOT select the continuation model -- it is not read at "
                f"runtime. Use a single teacher.")
        teacher_world_size_sum = 0
        for teacher_model in self.teacher_models.values():
            teacher_model.validate_and_prepare_for_distillation(
                use_topk=self.distillation_loss.loss_settings.use_topk,
                topk=self.distillation_loss.topk,
                generation_tokens=self._teacher_generation_tokens(),
                prefix_length=self._teacher_prefix_length(),
                total_horizon=self._teacher_total_horizon(),
            )
            teacher_world_size_sum += teacher_model.world_size
        total_pool_size = self.n_gpus_per_node * self.nnodes
        if teacher_world_size_sum != total_pool_size:
            raise ValueError(
                f"Sum of teacher (num_replicas * per_replica_world_size) ({teacher_world_size_sum}) must match "
                f"the distillation resource pool size "
                f"({self.n_gpus_per_node=} * {self.nnodes=} = {total_pool_size})."
            )

    #: The only `inference` fields that describe the STUDENT rather than the teacher's own engine.
    #: validate_and_prepare_for_distillation reads them as INPUT and then OVERWRITES them with the
    #: engine's budget, so a wrong value here cannot be caught downstream -- it can only make the
    #: boot check meaningless.
    _STUDENT_DERIVED_INFERENCE_FIELDS = ("prompt_length", "response_length")

    @staticmethod
    def _node_get(node, key, default=None):
        """Read `key` off a node that may be a dict, a DictConfig, or a BaseConfig."""
        if isinstance(node, dict):
            return node.get(key, default)
        return getattr(node, key, default)

    def _seed_student_lengths(self, template) -> None:
        """Give every '+'-added teacher the student's prompt/response lengths.

        Only the YAML `teacher_model` entry interpolates

            prompt_length:   ${oc.select:actor_rollout_ref.rollout.prompt_length}
            response_length: ${oc.select:actor_rollout_ref.rollout.response_length}

        A teacher added as `+distillation.teacher_models.<name>.*` -- the recipe this class's own
        docstring recommends, and what every launcher here uses -- is a NEW sibling node that
        inherits nothing from that template. Both lengths therefore fell back to RolloutConfig's
        dataclass default of 512.

        That was never cosmetic. validate_and_prepare_for_distillation reads the STUDENT's response
        length out of exactly this field, and the scoring branch then collapses response_length to
        1 -- so its boot check evaluated 512 + 512 + 1 <= max_model_len and passed against every
        real engine. The 512 stayed invisible for the whole campaign and only surfaced when
        state-credit compared a 2048-token depth against it.

        Seeded on the RAW node, BEFORE omega_conf_to_dataclass: after the structured merge a 512
        that came from the dataclass default is indistinguishable from one the user asked for, and
        an explicit teacher-side length could no longer win.

        ONLY these two fields. `temperature` is a student interpolation on the template too and is
        deliberately NOT seeded: it would turn every run with a non-1.0 rollout temperature into a
        hard NotImplementedError in the teacher sampling params -- a behaviour change wearing a bug
        fix's clothes.
        """
        template_inference = self._node_get(template, "inference")
        for name, teacher in self.teacher_models.items():
            if isinstance(teacher, BaseConfig):
                continue  # already-structured entry: hydra resolved its interpolations
            inference = self._node_get(teacher, "inference")
            if inference is None:
                continue
            if not isinstance(inference, dict):  # DictConfig: allow adding the absent keys
                OmegaConf.set_struct(inference, False)
            for fld in self._STUDENT_DERIVED_INFERENCE_FIELDS:
                if self._node_get(inference, fld) is not None:
                    continue  # explicitly configured on this teacher -- never override
                seed = self._node_get(template_inference, fld)
                if seed is None:
                    raise ValueError(
                        f"teacher {name!r} does not set inference.{fld}, and the `teacher_model` "
                        f"template could not supply it. Refusing to fall back to RolloutConfig's "
                        f"512: teacher dimensioning reads the STUDENT's lengths out of this field, "
                        f"and a 512 makes the prompt + response + 1 <= max_model_len boot check "
                        f"vacuous."
                    )
                inference[fld] = int(seed)

    def _resolve_teacher_models(self) -> dict[str, DistillationTeacherModelConfig]:
        assert "teacher_model" in self.teacher_models
        if len(self.teacher_models) == 1:
            # Single teacher occupies the entire teacher resource pool.
            teacher_model = self.teacher_models["teacher_model"]
            inference = teacher_model.inference
            per_replica = (
                inference.tensor_model_parallel_size
                * inference.data_parallel_size
                * inference.pipeline_model_parallel_size
            )
            pool_size = self.n_gpus_per_node * self.nnodes
            if pool_size % per_replica != 0:
                raise ValueError(
                    f"Single teacher's per_replica_world_size ({per_replica}) must divide the distillation "
                    f"resource pool size ({self.n_gpus_per_node=} * {self.nnodes=} = {pool_size})."
                )
            teacher_model.num_replicas = pool_size // per_replica
            teacher_model.key = "default"
        else:
            # Multiple teachers: `teacher_model` stops being a teacher and becomes the TEMPLATE that
            # the explicitly-added entries inherit their student-derived lengths from. Popping it
            # without reading it first is what silently reverted those lengths to 512.
            self._seed_student_lengths(self.teacher_models.pop("teacher_model"))

        # Teacher models dict is keyed by teacher_key instead of YAML entry name
        teacher_models = {}
        for teacher_config in self.teacher_models.values():
            teacher_config = omega_conf_to_dataclass(teacher_config, dataclass_type=DistillationTeacherModelConfig)
            teacher_config.check_configured()
            if teacher_config.key in teacher_models:
                raise ValueError(f"Duplicate teacher key {teacher_config.key} found in teacher models.")
            teacher_models[teacher_config.key] = teacher_config
        return teacher_models
