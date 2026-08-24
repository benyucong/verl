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
    store=None,
) -> dict:
    """Phi at every interior depth this trajectory actually reaches.

    `store` holds continuations launched EARLY, as the stream crossed each depth. Reusing them is
    what makes the teacher's work overlap generation instead of following it.
    """
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
    n_reused = 0
    for i, d in enumerate(reached):
        prefix = list(prompt_ids) + list(response_ids[:d])
        # keyed on DEPTH, stride M, so the M children of one depth never share a stream with
        # another depth's (vLLM seeds children parent+0..parent+M-1)
        d_seed = sc_seed_for_depth(seed, M, d)
        got = None
        if store is not None:
            # Awaits an in-flight early launch rather than racing it: that work is already paid
            # for. Returns None when the key does not match what the commit is asking for -- a
            # partial-rollout rewind is the only way that happens -- and we then issue it fresh
            # rather than substituting a prefix nobody asked for.
            got = await store.take(d, expect_key=store.request_key(prefix, M, B, d_seed))
        if got is not None:
            seqs, t = got
            n_reused += 1
        else:
            seqs, t = await teacher_manager.generate_chunk_continuations(
                prefix_ids=prefix,
                n=M,
                max_tokens=B,
                routing_key=routing_key,
                session_id=session_id,
                seed=d_seed,
                # An EARLY launch cannot know which depth is last: the set is only settled once the
                # true response length is known. So no call carries is_final and the parent's FIFO
                # state is released explicitly once every depth is in hand.
                is_final=False,
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
    # How much of the teacher's work was already done by the time the response finished. This is
    # THE metric for the mechanism: sc_early_frac near 1.0 means the continuations overlapped
    # generation, near 0.0 means they followed it and the run is the sequential arm wearing the
    # streaming label. Reported even when the store is absent, so the two arms stay comparable.
    tele["sc_reused_early"] = n_reused
    tele["sc_early_frac"] = n_reused / max(1, len(reached))
    if store is not None:
        # Every depth is accounted for; no call carried is_final, so release the parent's ordering
        # state here rather than leaving it for the stale reaper.
        await store.drain_unused()
        try:
            teacher_manager.release_parent(session_id)
        except AttributeError:
            pass       # older manager without the explicit release; the reaper still collects it
    return {"state_credit_phi": phis, "state_credit_depths": reached,
            "state_credit_telemetry": tele}


async def attach_state_credit(output, *, prompt_ids, response_ids, ground_truth, teacher_manager,
                              tokenizer, sc_config, session_id=None, routing_key=None,
                              seed=None, store=None) -> None:
    """Run the measurement and hang it on the agent-loop output.

    Everything in extra_fields becomes a non_tensor_batch column, which is where the driver reads
    the per-depth Phi to build the credit.
    """
    t0 = time.time()
    rec = await run_state_credit(
        prompt_ids=prompt_ids, response_ids=response_ids, ground_truth=ground_truth,
        teacher_manager=teacher_manager, tokenizer=tokenizer, sc_config=sc_config,
        session_id=session_id, routing_key=routing_key, seed=seed, store=store)
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


# ---------------------------------------------------------------------------------------------
# EARLY ASYNC CONTINUATION
#
# Phi(s_d) depends on prompt + response[:d] and nothing else. Once the student has emitted d
# tokens that prefix is FINAL -- it cannot change, because generation only appends. So the
# continuation for depth d can be issued the moment the stream crosses d, and by the time the
# response finishes ~10x later the answer is already sitting in the store.
#
# This is not speculation. OmniOPD's anchors are a global argmax over the FINISHED response, so an
# early launch there is a guess that can miss (top_k=4 of 10, mism>0 possible). State-credit's
# depths are configured constants, so every early launch is a request the commit is guaranteed to
# make, and every launched depth is one the trajectory actually reaches (we only launch after
# crossing it). Nothing is wasted and nothing is guessed -- the key check below exists only to
# catch a rewind, not an ordinary miss.
# ---------------------------------------------------------------------------------------------

def sc_seed_for_depth(base_seed, M: int, depth: int):
    """Seed for a depth's M children. Must match run_state_credit's exactly or the key check fires."""
    return None if base_seed is None else (base_seed + int(M) * int(depth)) % (2**31)


def launch_state_credit_early(store, *, prompt_ids, response_ids, sc_config, teacher_manager,
                              session_id=None, routing_key=None, seed=None) -> int:
    """Issue continuations for every depth the stream has just passed. Returns how many started.

    THE GENERATION LOOP PAYS ONE COPY PER DEPTH, ONCE. `response_ids` is referenced rather than
    snapshotted, and the factory's slice runs inside the coroutine -- which SpeculativeStore.launch
    starts after an `await asyncio.sleep(0)` and, with OPD_OMNIOPD_SPEC_THREAD=1, on a separate
    dispatch loop entirely. What remains here is the request key, which needs the prefix bytes and
    so must materialise prompt + response[:d] synchronously. That is d+|prompt| elements once per
    depth per trajectory -- not the whole response at every chunk boundary, which is the cost that
    capped the OmniOPD version at +5.2%.

    Referencing the live list is safe because response[:d] is immutable once len(response) > d:
    generation appends. A partial-rollout rewind is the one case that could violate it, and that is
    exactly what the request key catches at commit -- take() relaunches rather than substituting a
    prefix the commit never asked for.
    """
    depths = sorted({int(d) for d in (sc_config.depths or [])})
    if not depths:
        return 0
    M, B = int(sc_config.M), int(sc_config.B)
    n_launched = 0
    resp_len = len(response_ids)
    for d in depths:
        if d >= resp_len or d in store.tasks:
            continue

        def _factory(_d=int(d)):
            # Runs on the dispatch loop, after the first yield -- see the note above.
            _p = list(prompt_ids) + list(response_ids[:_d])
            return teacher_manager.generate_chunk_continuations(
                prefix_ids=_p, n=M, max_tokens=B,
                routing_key=routing_key, session_id=session_id,
                seed=sc_seed_for_depth(seed, M, _d),
                # No early call can know it is the last: the depth set is only settled once the
                # true response length is known. The commit releases the parent explicitly instead.
                is_final=False,
            )

        # Same identity the commit recomputes, built from the same inputs, so the two agree by
        # construction. The prefix IS materialised here -- unavoidable, the hash needs the bytes --
        # but only for the first _d tokens, not the whole response.
        key = store.request_key(list(prompt_ids) + list(response_ids[:d]), M, B,
                                sc_seed_for_depth(seed, M, d))
        store.launch(d, _factory, key=key)
        n_launched += 1
    return n_launched
