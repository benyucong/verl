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
"""Phi: how solvable is this reasoning state?

    Phi(s) = E_{c ~ pi_C(.|s)} [ Q(x, s||c) ]     estimated with M continuations from a FROZEN pi_C

This produces one Phi per interior depth per trajectory. It does NOT produce the credit: the
progress Delta and its leave-one-out centering are computed in the driver, where a problem's
sibling rollouts are all present. Here we only measure states.

WHY THE TERMINAL STATE IS NOT MEASURED HERE. Phi(s_T) is the task reward itself -- the trajectory
already ran to an answer, so sampling continuations from a finished state would be a strictly worse
estimate of something already known exactly. The driver reads it off the reward column.

WHY Phi(s_0) IS NEVER COMPUTED. It is identical for every rollout of a problem, so it cancels
exactly under leave-one-out centering. Computing it would cost M continuations per problem to
produce a constant that subtracts out.

TRUNCATION IS NOT A ZERO. A continuation that runs out of budget scores 0 from the verifier, which
is indistinguishable from one that reasoned to a wrong answer -- and those mean opposite things
about the state. The probe measured this directly: at B=1024 against 20k thinking-on traces, 99.3%
of continuations truncated and Phi collapsed to 0.003 everywhere, which reads as "state value does
not exist" when it actually meant "the budget was 20x too small". So the truncated fraction is
recorded per depth and surfaced; the caller decides whether Phi is interpretable.
"""

import logging
import os
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)


def state_credit_enabled(config) -> bool:
    """Resolved from the loss NAME through the registry the trainer uses -- never from a
    loss_settings attribute, which is populated at runtime and absent from every composed YAML."""
    try:
        loss_mode = config.distillation.distillation_loss.loss_mode
    except Exception:
        return False
    try:
        from .losses import get_distillation_loss_settings
    except ImportError:
        from losses import get_distillation_loss_settings
    try:
        return bool(get_distillation_loss_settings(str(loss_mode)).use_teacher_continuations)
    except Exception:
        return False


def resolve_state_credit_config(config):
    """The state_credit node from the raw DictConfig, or the dataclass defaults if absent.

    The agent loop holds the RAW Hydra config, where a missing key is a struct-mode error raised
    mid-rollout -- after the whole response has been generated. distillation.yaml carries the node,
    so this is a fallback rather than the normal path.
    """
    try:
        sc = config.distillation.state_credit
        if sc is not None:
            return sc
    except Exception:
        pass
    try:
        from verl.workers.config.distillation import StateCreditConfig
    except ImportError:
        from ...workers.config.distillation import StateCreditConfig
    logger.warning("distillation.state_credit absent from the config; falling back to defaults")
    return StateCreditConfig()


def score_answer(text: str, ground_truth) -> float:
    """Q(x, y) in {0, 1}.

    fast=True: grade_answer_mathd (a normalising string compare) runs first and short-circuits, so
    the sympy path -- which forks a process for its timeout -- is reached only when the cheap check
    fails. That matters at M continuations per depth per row.
    """
    try:
        from custom_reward.ttrl_math import compute_score
    except ImportError:
        logger.error("[STATE-CREDIT] verifier unavailable; Phi cannot be computed")
        raise
    try:
        s = compute_score(text, ground_truth, fast=True)
        return float(s["score"] if isinstance(s, dict) else s)
    except Exception as e:
        # A verifier failure is NOT evidence the state was bad. Counted separately so it cannot be
        # silently folded into Phi as a zero.
        logger.warning("[STATE-CREDIT] verifier raised on one continuation: %s", e)
        return float("nan")


async def run_state_credit(
    *,
    prompt_ids: list[int],
    response_ids: list[int],
    ground_truth,
    teacher_manager,
    tokenizer,
    sc_config,
    session_id: Optional[str] = None,
    routing_key: Optional[str] = None,
    seed: Optional[int] = None,
) -> dict:
    """Phi at every interior depth this trajectory actually reaches."""
    depths = [int(d) for d in (sc_config.depths or [])]
    M, B = int(sc_config.M), int(sc_config.B)
    T = len(response_ids)

    reached = [d for d in depths if d < T]
    tele = {"sc_depths_requested": len(depths), "sc_depths_reached": len(reached),
            "sc_response_len": T, "sc_gen_seconds": 0.0, "sc_gen_tokens": 0,
            "sc_truncated": 0, "sc_scored": 0, "sc_verifier_failed": 0}
    if not reached:
        # Not an error: a short trajectory has no interior state to evaluate. The driver drops it
        # from every LOO group rather than crediting it against a baseline it never joined.
        tele["sc_skipped"] = "no_depth_reached"
        return {"state_credit_phi": [], "state_credit_depths": [], "state_credit_telemetry": tele}

    phis: list[float] = []
    for i, d in enumerate(reached):
        prefix = list(prompt_ids) + list(response_ids[:d])
        seqs, t = await teacher_manager.generate_chunk_continuations(
            prefix_ids=prefix,
            n=M,
            max_tokens=B,
            routing_key=routing_key,
            session_id=session_id,
            # keyed on DEPTH, stride M, so the M children of one depth never share a stream with
            # another depth's (vLLM seeds children parent+0..parent+M-1)
            seed=None if seed is None else (seed + M * d) % (2**31),
            is_final=(i == len(reached) - 1),      # releases this parent's sticky routing debt
        )
        if len(seqs) != M:
            raise RuntimeError(
                f"state-credit: depth {d} returned {len(seqs)} continuations, asked for {M}. Phi "
                f"from a short sample is a different estimator, not a noisier one.")
        fr = (t or {}).get("finish_reasons") or []
        prefix_text = tokenizer.decode(response_ids[:d], skip_special_tokens=True)
        qs = []
        for j, sq in enumerate(seqs):
            cont = tokenizer.decode(sq, skip_special_tokens=True)
            q = score_answer(prefix_text + cont, ground_truth)
            if q != q:                              # NaN -> verifier failed, not a zero
                tele["sc_verifier_failed"] += 1
                continue
            qs.append(q)
            if j < len(fr) and fr[j] == "length":
                tele["sc_truncated"] += 1
        tele["sc_scored"] += len(qs)
        tele["sc_gen_seconds"] += (t or {}).get("teacher_gen_seconds") or 0.0
        tele["sc_gen_tokens"] += (t or {}).get("teacher_gen_tokens") or 0
        phis.append(sum(qs) / len(qs) if qs else float("nan"))

    n_cont = len(reached) * M
    tele["sc_trunc_frac"] = tele["sc_truncated"] / max(1, n_cont)
    return {"state_credit_phi": phis, "state_credit_depths": reached,
            "state_credit_telemetry": tele}


async def attach_state_credit(output, *, prompt_ids, response_ids, ground_truth, teacher_manager,
                              tokenizer, sc_config, session_id=None, routing_key=None,
                              seed=None) -> None:
    """Run the measurement and hang it on the agent-loop output.

    Everything in extra_fields becomes a non_tensor_batch column, which is where the driver reads
    the per-depth Phi to build the credit.
    """
    t0 = time.time()
    rec = await run_state_credit(
        prompt_ids=prompt_ids, response_ids=response_ids, ground_truth=ground_truth,
        teacher_manager=teacher_manager, tokenizer=tokenizer, sc_config=sc_config,
        session_id=session_id, routing_key=routing_key, seed=seed)
    output.extra_fields["state_credit_phi"] = rec["state_credit_phi"]
    output.extra_fields["state_credit_depths"] = rec["state_credit_depths"]
    output.extra_fields["state_credit_telemetry"] = rec["state_credit_telemetry"]

    if os.environ.get("OPD_STATE_CREDIT_QUIET", "0") in ("0", "", "false", "False"):
        te = rec["state_credit_telemetry"]
        print("[STATE-CREDIT] sid=%s T=%d depths=%s phi=%s trunc=%.3f gen_s=%.1f vfail=%d wall=%.1f"
              % (session_id, te["sc_response_len"], rec["state_credit_depths"],
                 ["%.3f" % p for p in rec["state_credit_phi"]],
                 te.get("sc_trunc_frac", 0.0), te["sc_gen_seconds"],
                 te["sc_verifier_failed"], time.time() - t0), flush=True)
