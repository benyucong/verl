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


def state_credit_needs_teacher_scores(config) -> bool:
    """True when the state-credit objective also needs token-level teacher scores.

    Sec 7's advantage is u_t = a_t^OPD + beta * G~, so arm C consumes BOTH halves of the Sec 8
    probe: the prefill that scores the completed chunk AND the continuations that value the state.
    Only base_loss_mode='none' -- the credit-only diagnostic, arm D -- can do without the first.

    Read off the RAW config rather than a resolved dataclass: the agent loop runs before the
    trainer's dataclass conversion, and this decides whether a teacher RPC is issued at all.
    """
    if not state_credit_enabled(config):
        return False
    try:
        mode = config.distillation.state_credit.base_loss_mode
    except Exception:
        # The key is in the schema; its absence means an older composed config, where the objective
        # had no base term at all. Defaulting to False there would silently reproduce arm D.
        return True
    return str(mode) != "none"


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


def score_answer(text: str, ground_truth, fast: bool = False) -> float:
    """Q(x, y) in {0, 1}.

    `fast` MUST match the grader that produces the terminal reward, and it defaults to False for
    that reason. Sec 4 makes Phi an operational quantity -- "solvable by this policy under this
    budget, as judged by THIS verifier" -- and Sec 5's terminal delta subtracts the two directly:

        Delta_m = Phi(s_T) - Phi(s_{k_{m-1}}) = Q_terminal(x, y) - Phi_hat_interior

    fast=True is not a cheaper approximation of the same verdict, it is a strictly smaller one:
    grade() runs the mathd/sympy pair either way and only the slow path adds the is_latex_equal
    recall pass, so fast=True can turn a correct answer into a wrong one and never the reverse.
    Grading the interior end fast and the terminal end fully therefore biases every terminal chunk
    upward by the recall gap -- and biases it by a DIFFERENT amount per problem, so leave-one-out
    does not remove it.

    The cost this was avoiding is real and is paid only on continuations the cheap path already
    rejected (roughly 1 - Phi of them), so it scales with how badly the student is doing.
    """
    try:
        from custom_reward.ttrl_math import compute_score
    except ImportError:
        logger.error("[STATE-CREDIT] verifier unavailable; Phi cannot be computed")
        raise
    try:
        # str(), because the TERMINAL reward does: custom_reward.ttrl_math.reward_func calls
        # compute_score(solution_str, str(ground_truth)). compute_score branches on the type --
        # a list ground truth is graded with OR-over-elements, a string literally -- so passing the
        # raw object here and a stringified one there would make the two ends of every terminal
        # delta different functions of the same answer. Latent on DAPO-Math (17917/17917 ground
        # truths are already str) and NOT latent on the AIME/AMC sets, which carry lists.
        s = compute_score(text, str(ground_truth), fast=fast)
        return float(s["score"] if isinstance(s, dict) else s)
    except Exception as e:
        # A verifier failure is NOT evidence the state was bad. Counted separately so it cannot be
        # silently folded into Phi as a zero.
        logger.warning("[STATE-CREDIT] verifier raised on one continuation: %s", e)
        return float("nan")


def sc_budget_for_depth(sc_config, d: int) -> int:
    """Continuation budget for a probe launched at depth `d`.

    budget_mode='remaining' (default): B is the SHARED horizon, so the continuation gets B - d and
    prefix + continuation obeys the same total budget the student trains under. Phi then answers one
    question at every depth -- "is this state still solvable in the budget the task allows" -- and
    the teacher's context is flat at prompt + B instead of growing with depth.

    budget_mode='fixed': every depth gets a full B on top of its prefix. That is a different Phi
    (budget-free), and a deep state is scored with more room than the task ever grants it.

    Both call sites -- the early launch and the commit -- MUST route through here. The store's
    request key includes the budget, so a disagreement between them would not corrupt anything
    silently: it would simply miss every early launch and quietly re-issue the work.
    """
    B = int(sc_config.B)
    if getattr(sc_config, "budget_mode", "remaining") != "remaining":
        return B
    return max(1, B - int(d))


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
    M = int(sc_config.M)
    T = len(response_ids)

    reached = [d for d in depths if d < T]
    tele = {"sc_depths_requested": len(depths), "sc_depths_reached": len(reached),
            "sc_response_len": T, "sc_gen_seconds": 0.0, "sc_gen_tokens": 0,
            "sc_truncated": 0, "sc_scored": 0, "sc_verifier_failed": 0,
            "sc_trunc_unmeasured": 0, "sc_cont_max_tokens": 0,
            "sc_prefix_tokens_logical": 0, "sc_prefix_tokens_cached": 0,
            # Sec 12.3 measurement-policy fingerprint. Phi means nothing without the settings it was
            # measured under, and "read them off the config afterwards" is exactly how this repo lost
            # a whole campaign to a teacher response_length of 512 that nobody had recorded. These
            # ride the RECORD, so two Phi values measured under different budgets or graders cannot
            # be pooled by accident.
            "sc_fp_M": int(sc_config.M),
            "sc_fp_B": int(sc_config.B),
            "sc_fp_budget_mode": str(getattr(sc_config, "budget_mode", "remaining")),
            "sc_fp_verifier_fast": bool(getattr(sc_config, "verifier_fast", False)),
            "sc_fp_temperature": None,
            "sc_fp_top_p": None,
            "sc_fp_teacher": None,
            # The verifier's identity, not just its fast/slow flag: Sec 12.3 lists the verifier and
            # the answer extractor separately, and both live in this module.
            "sc_fp_verifier": "custom_reward.ttrl_math.compute_score"}
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
        B = sc_budget_for_depth(sc_config, d)
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
        # Sec 10 / Sec 14.4: logical vs PHYSICAL prefix work. The teacher ingests prompt + y[:d] at
        # every boundary and those prefixes are nested, so most of it should come back from the
        # prefix cache -- "high logical-prefix volume is not equivalent to high physical prefill
        # work when cache reuse succeeds". Both numbers were already being measured per request and
        # then dropped on the floor, which left the one quantity that distinguishes the two
        # unobservable and the Sec 8 sharing claim unfalsifiable.
        # First probe to report them wins; they are constant across depths by construction, and a
        # LATER value silently overwriting an earlier one is how a mid-run change would hide.
        if tele["sc_fp_temperature"] is None:
            tele["sc_fp_temperature"] = (t or {}).get("teacher_temperature")
            tele["sc_fp_top_p"] = (t or {}).get("teacher_top_p")
            tele["sc_fp_teacher"] = (t or {}).get("teacher_model")
        tele["sc_prefix_tokens_logical"] += int((t or {}).get("teacher_prefix_tokens") or 0)
        tele["sc_prefix_tokens_cached"] += int((t or {}).get("teacher_cached_tokens") or 0)
        fr = (t or {}).get("finish_reasons") or []
        if len(fr) < len(seqs):
            # No finish reasons for these rows: truncation is UNMEASURED here, not zero.
            tele["sc_trunc_unmeasured"] += len(seqs) - len(fr)
        prefix_text = tokenizer.decode(response_ids[:d], skip_special_tokens=True)
        qs = []
        for j, sq in enumerate(seqs):
            cont = tokenizer.decode(sq, skip_special_tokens=True)
            q = score_answer(prefix_text + cont, ground_truth,
                             fast=bool(getattr(sc_config, "verifier_fast", False)))
            if q != q:                              # NaN -> verifier failed, not a zero
                tele["sc_verifier_failed"] += 1
                continue
            qs.append(q)
            if j < len(fr) and fr[j] == "length":
                tele["sc_truncated"] += 1
        tele["sc_scored"] += len(qs)
        tele["sc_gen_seconds"] += (t or {}).get("teacher_gen_seconds") or 0.0
        tele["sc_gen_tokens"] += (t or {}).get("teacher_gen_tokens") or 0
        # LONGEST single continuation, which is what actually bounds B -- and through B the teacher's
        # max_model_len, its KV per sequence, and therefore how many sequences the pool can hold.
        # sc_gen_tokens is a SUM over the M continuations and cannot answer that. With trunc=0 the
        # cap is never reached, so B is pure KV reservation: at B=20480 an 8B teacher reserves
        # ~3.5 GB per sequence and the pool fits ~12 against max_num_seqs=16. Knowing the real
        # maximum is what licenses lowering B, which costs no generated tokens at all.
        tele["sc_cont_max_tokens"] = max(tele["sc_cont_max_tokens"],
                                         max((len(x) for x in seqs), default=0))
        phis.append(sum(qs) / len(qs) if qs else float("nan"))

    n_cont = len(reached) * M
    # NaN, not 0.0, when any continuation came back without a finish reason. A gate that cannot see
    # truncation must say so: reporting 0.000 is how this silently passed for a whole campaign.
    tele["sc_trunc_frac"] = (float("nan") if tele["sc_trunc_unmeasured"]
                             else tele["sc_truncated"] / max(1, n_cont))
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
        # `early` is the arm's metric and the ONLY place it surfaces: state_credit_telemetry has no
        # aggregator consumer, so anything absent from this line is invisible in the run.
        # reused/reached near 1.0 means the continuations overlapped generation; 0/n means they
        # followed it and the run is the sequential arm wearing a streaming label. `wall` minus
        # `gen_s` is the part the overlap actually removes.
        print("[STATE-CREDIT] sid=%s T=%d depths=%s phi=%s trunc=%.3f gen_s=%.1f vfail=%d "
              "early=%d/%d maxcont=%d prefix=%d/%d wall=%.1f"
              % (session_id, te["sc_response_len"], rec["state_credit_depths"],
                 ["%.3f" % p for p in rec["state_credit_phi"]],
                 te.get("sc_trunc_frac", 0.0), te["sc_gen_seconds"],
                 te["sc_verifier_failed"], te.get("sc_reused_early", 0),
                 len(rec["state_credit_depths"]), te.get("sc_cont_max_tokens", 0),
                 # cached/logical -- Sec 14.4. Nested boundary prefixes SHOULD make these nearly
                 # equal; a cached count well below logical is the cache-contention hazard of
                 # Sec 16.7 showing up, not a scheduling problem.
                 te.get("sc_prefix_tokens_cached", 0), te.get("sc_prefix_tokens_logical", 0),
                 time.time() - t0), flush=True)


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
    M = int(sc_config.M)
    n_launched = 0
    resp_len = len(response_ids)
    for d in depths:
        # `>=`, not `>`. response[:d] is COMPLETE the instant the stream reaches d tokens, so the
        # boundary at exactly d is launchable. Requiring resp_len > d slipped every launch a full
        # chunk later (2048 -> 3072 at CHUNK=1024) and, worse, meant a response landing in [d, d+CHUNK)
        # never launched early at all -- its next boundary is the final one, where the hook does not
        # run. That is what produced reused=0/2 on the first trajectories of job 45007334.
        #
        # A depth exactly equal to the FINAL length is still not credited (run_state_credit keeps
        # d < T, since the chunk from d to T would be empty); such a launch is simply dropped by
        # drain_unused. Launching it is the cheaper error.
        if resp_len < d or d in store.tasks:
            continue

        B = sc_budget_for_depth(sc_config, d)

        def _factory(_d=int(d), _B=int(B)):
            # Runs on the dispatch loop, after the first yield -- see the note above.
            _p = list(prompt_ids) + list(response_ids[:_d])
            return teacher_manager.generate_chunk_continuations(
                prefix_ids=_p, n=M, max_tokens=_B,
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
