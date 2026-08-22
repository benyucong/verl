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
"""The OmniOPD audit, as one await inside the agent loop.

Between a finished student response and the loss there is a stage nothing in verl performed: pick M
audit anchors from the student's own entropy, ask the teacher to independently write each audited
span N times, and score the agreement into k_sem. This is that stage. It returns the two fields
compute_distillation_loss_omniopd reads off non_tensor_batch -- omniopd_anchors and omniopd_k_sem --
and nothing else; the objective, the KL and the aggregation all live downstream and are unchanged.

WHY THE TEACHER CALLS ARE SEQUENTIAL WITHIN A TRAJECTORY. A trajectory's M chunk prefixes are
NESTED: each is the previous plus the tokens between two anchors. Issued in ascending order against
one sticky replica, chunk k+1's prefill reads chunk k's KV out of the prefix cache and pays only for
the tokens that are new. Issued concurrently they race -- every call arrives before any has
populated the cache, so each re-ingests a prefix that grows with the response, which is the dominant
cost of generative teaching. The concurrency that matters here is ACROSS trajectories, and the agent
loop already provides it: every trajectory is its own coroutine, so M sequential awaits here overlap
with every other trajectory in flight. This is also forced rather than chosen -- selection is a
global argmax over the whole response, so no anchor is known until generation ends, and the audit of
one trajectory cannot be streamed against itself.

WHAT IS DELIBERATELY NOT HERE. No selection from teacher signal (the selector is student-entropy
only, and reading teacher output to choose what to audit would make the audit self-confirming), no
retry of a failed chunk, and no partial result: a trajectory either carries a complete audit or
raises. k_sem computed from fewer than N continuations is not a degraded measurement, it is a
smaller number, and the objective cannot tell that apart from a teacher that disagreed.
"""

import logging
import os
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:  # as a package member
    from .omniopd_speculation import seed_for_anchor
    from .omniopd_producer import assemble_omniopd_record, build_chunk_requests, select_anchors
    from .selector_spec import OMNIOPD_ONLINE_VARIANT
    from .token_entropy import to_signal_series
except ImportError:  # or with the distillation dir on sys.path, which is how the tests import it
    from omniopd_speculation import seed_for_anchor
    from omniopd_producer import assemble_omniopd_record, build_chunk_requests, select_anchors
    from selector_spec import OMNIOPD_ONLINE_VARIANT
    from token_entropy import to_signal_series


def omniopd_enabled(config: Any) -> bool:
    """OmniOPD runs only when the loss asks for teacher generation.

    Resolved from the loss NAME through the same registry the trainer uses, so a run cannot be
    configured into a state where the trainer expects an audit the rollout never produced.

    NOT from config.distillation.distillation_loss.loss_settings. That field is populated at runtime
    on the dataclass and exists in NO composed YAML, so on the raw DictConfig the read raises
    ConfigAttributeError -- a subclass of AttributeError, which an `except AttributeError` swallows.
    The gate then returns False on every real run: no audit, no teacher generation, no anchors, and
    a loss that raises much later complaining the engine gave it no KL. Nothing named the true cause.
    """
    try:
        loss_mode = config.distillation.distillation_loss.loss_mode
    except Exception:
        return False
    try:
        from .losses import get_distillation_loss_settings
    except ImportError:
        from losses import get_distillation_loss_settings
    try:
        return bool(get_distillation_loss_settings(str(loss_mode)).use_teacher_generation)
    except Exception:
        return False


def resolve_omniopd_config(config):
    """The omniopd node, from the raw DictConfig, or the dataclass defaults if it is absent.

    Two different objects carry this config. The trainer gets the instantiated DistillationConfig,
    where `omniopd` always exists via default_factory. The AGENT LOOP holds the raw Hydra
    DictConfig, and there a missing key is a struct-mode ConfigAttributeError -- raised mid-rollout,
    after the whole response has been generated. distillation.yaml now carries the node, so this is
    a fallback rather than the normal path; it exists because the failure it prevents is expensive
    and arrives late, and because the values it substitutes are the same defaults the dataclass
    would have used anyway.
    """
    try:
        om = config.distillation.omniopd
        if om is not None:
            return om
    except Exception:
        pass
    try:
        from verl.workers.config.distillation import OmniOPDConfig
    except ImportError:
        from ...workers.config.distillation import OmniOPDConfig
    logger.warning(
        "distillation.omniopd absent from the config; falling back to OmniOPDConfig() defaults "
        "(M=10 N=10 C=50 alpha=1.0 beta=0.1 phi=ned). Add the node to distillation.yaml to set them."
    )
    return OmniOPDConfig()


async def run_omniopd_audit(
    *,
    prompt_ids: list[int],
    response_ids: list[int],
    entropies: Optional[list[float]],
    teacher_manager,
    tokenizer,
    omniopd_config,
    session_id: Optional[str] = None,
    routing_key: Optional[str] = None,
    seed: Optional[int] = None,
    store=None,
) -> dict:
    """Audit one finished trajectory. Returns the two loss fields plus telemetry.

    `entropies` is one value per RESPONSE token, in emission order, as the patched sampler produced
    them. Absence is fatal: with no entropy there is no selector, and any fallback would be a
    different rule under the same name.
    """
    M, N, C = int(omniopd_config.M), int(omniopd_config.N), int(omniopd_config.C)
    phi_name = str(omniopd_config.phi)

    if entropies is None:
        raise RuntimeError(
            "OmniOPD audit: the rollout returned no token_entropies. Anchor selection is argmax over "
            "the student's full-vocabulary entropy (FID-1); there is no defined selector without it. "
            "Check that the rollout ran on the patched vLLM fork and requested return_token_entropy."
        )
    if len(entropies) != len(response_ids):
        raise RuntimeError(
            f"OmniOPD audit: {len(response_ids)} response tokens but {len(entropies)} entropy values. "
            f"Every anchor would be shifted against the tokens it indexes."
        )

    # A response shorter than one chunk has nothing to audit. Returning empty is correct and is NOT
    # the same as a failure: the loss falls back to beta*KL on every position, which is exactly the
    # objective's own definition when the audited set is empty.
    if len(response_ids) < C:
        return {"omniopd_anchors": [], "omniopd_k_sem": [], "omniopd_telemetry": {"skipped": "short_response"}}

    t_eos = time.time()          # the audit begins the moment the response is complete
    signal = to_signal_series(entropies, prompt_len=len(prompt_ids))
    anchors = select_anchors(signal, len(prompt_ids), len(response_ids), M, C)
    if not anchors:
        return {"omniopd_anchors": [], "omniopd_k_sem": [], "omniopd_telemetry": {"skipped": "no_anchor"}}

    prefixes = build_chunk_requests(prompt_ids, response_ids, anchors)

    # RECONCILIATION. `store` holds continuations launched speculatively at earlier chunk
    # boundaries. Each committed anchor is taken from it when a proposal matched, and executed here
    # when none did. The commit's selection above is untouched -- the store can only ever answer for
    # an anchor this function already chose, which is why speculation cannot move the objective.
    #
    # THE LAST ANCHOR IS ALWAYS EXECUTED HERE, never reused. generate_chunk_continuations releases
    # the sticky parent on is_final=True, and that release must happen exactly once per session; if
    # every anchor were served from the store, no call would be issued and the parent would leak --
    # the same class of defect as the stranding hang. Costing one anchor's overlap buys an
    # accounting invariant that holds by construction rather than by bookkeeping.
    last_i = len(prefixes) - 1

    continuations: list[list[str]] = []
    tele = {"teacher_gen_seconds": 0.0, "teacher_gen_tokens": 0, "teacher_prefix_tokens": 0,
            "teacher_cached_tokens": 0, "chunks": len(anchors)}
    for i, (t0, prefix) in enumerate(zip(anchors, prefixes, strict=True)):   # ascending: KV reuse
        if store is not None and i != last_i:
            taken = await store.take(int(t0))
            if taken is not None:
                seqs, t = taken
                if len(seqs) != N:
                    raise RuntimeError(
                        f"OmniOPD audit: reused proposal for anchor {t0} carried {len(seqs)} "
                        f"continuations, expected {N}."
                    )
                continuations.append([tokenizer.decode(x, skip_special_tokens=True) for x in seqs])
                for k in ("teacher_gen_seconds", "teacher_gen_tokens", "teacher_prefix_tokens"):
                    tele[k] += t.get(k) or 0
                tele["teacher_cached_tokens"] += t.get("teacher_cached_tokens") or 0
                continue
            store.relaunched += 1
        # SEED KEYED ON ANCHOR POSITION, STRIDE N -- not on the chunk's rank i.
        #
        # Two things were wrong with `seed + i`. First, vLLM expands an n=N request into N children
        # with seeds parent+0..parent+N-1, so consecutive chunks one apart in seed OVERLAP: chunk i's
        # child j and chunk i+1's child j-1 draw the same stream. At M=N=10 that makes the per-chunk
        # k_sem estimates correlated when the objective assumes N independent samples -- a live
        # defect, independent of anything speculative. Stride N removes the overlap.
        #
        # Second, i is the anchor's RANK in the committed set, which is unknowable before the set
        # exists. Keying on the position t0 makes the request a pure function of the anchor, which is
        # what would let work launched early be reused by the commit rather than recomputed.
        seqs, t = await teacher_manager.generate_chunk_continuations(
            prefix_ids=prefix,
            n=N,
            max_tokens=C,
            routing_key=routing_key,
            session_id=session_id,
            # ONE definition, shared with the speculative launcher. If these two ever compute the
            # seed differently a proposal stops being the same unit of work and silently never
            # matches -- speculation would appear to run and hide nothing.
            seed=seed_for_anchor(seed, int(t0), N),
            is_final=(i == len(prefixes) - 1),                 # releases this parent's sticky debt
        )
        if len(seqs) != N:
            raise RuntimeError(
                f"OmniOPD audit: chunk {i} got {len(seqs)} continuations, asked for {N}. k_sem from a "
                f"short sample is a smaller number, not a noisier one, and the objective cannot tell "
                f"it apart from teacher disagreement."
            )
        continuations.append([tokenizer.decode(s, skip_special_tokens=True) for s in seqs])
        for k in ("teacher_gen_seconds", "teacher_gen_tokens", "teacher_prefix_tokens"):
            tele[k] += t.get(k) or 0
        tele["teacher_cached_tokens"] += t.get("teacher_cached_tokens") or 0

    record = assemble_omniopd_record(
        prompt_ids, response_ids, anchors, continuations, tokenizer, N=N, C=C, phi_name=phi_name
    )
    # DOES NOT MEASURE ORDERING, despite what the previous comment here claimed. num_cached_tokens
    # is reported per REQUEST while an n=N request expands into N children sharing one prefix, so the
    # ratio is dominated by intra-request sharing and sits at 0.97-0.98 whatever order the calls go
    # out in (FID-11 measured the probe reporting 1488 cached where 992 were physically read). It
    # therefore cannot support the claim it was being used for -- that sequential issue is what earns
    # the KV reuse -- and I reported it several times as if it could. Renamed so nothing reads it as
    # evidence of ordering; a real ordering metric has to count physical uncached prefill.
    if tele["teacher_prefix_tokens"]:
        tele["cached_token_ratio_unreliable"] = (
            tele["teacher_cached_tokens"] / tele["teacher_prefix_tokens"])
    if store is not None:
        await store.drain_unused()
        tele.update(store.telemetry(t_eos=t_eos))
    record["omniopd_telemetry"] = tele
    return record


async def attach_omniopd_audit(output, *, prompt_ids, response_ids, teacher_manager, tokenizer,
                               omniopd_config, session_id=None, routing_key=None, seed=None,
                               store=None) -> None:
    """Run the audit and hang it on the agent-loop output.

    Everything in AgentLoopOutput.extra_fields becomes a non_tensor_batch column (agent_loop.py
    _postprocess), which is where compute_distillation_loss_omniopd reads omniopd_anchors and
    omniopd_k_sem from. That is the whole reason this writes there and not into a tensor.
    """
    rec = await run_omniopd_audit(
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        entropies=output.extra_fields.get("token_entropies"),
        teacher_manager=teacher_manager,
        tokenizer=tokenizer,
        omniopd_config=omniopd_config,
        session_id=session_id,
        routing_key=routing_key,
        seed=seed,
        store=store,
    )
    output.extra_fields["omniopd_anchors"] = rec["omniopd_anchors"]
    output.extra_fields["omniopd_k_sem"] = rec["omniopd_k_sem"]
    output.extra_fields["omniopd_telemetry"] = rec.get("omniopd_telemetry")
    # STAMP THE SELECTOR THAT ACTUALLY RAN. Anchors here come from entropy emitted by the rollout
    # engine during decode, not from the canonical post-EOS actor forward. Gate 6 measured the two
    # picking the same chunk set in only 9 of 16 responses, so this trains on DIFFERENT spans -- it
    # is a declared variant with its own selector_hash, and an artifact that did not say so could be
    # read as published OmniOPD. Recorded per trajectory so the label travels with the data.
    # Anchor INDICES appear in no other artifact -- the per-trajectory line below carries only the
    # count and k_sem. Without them the central claim of the streaming arm ("the selector saw the
    # whole finished response, so it picked what umem would have picked for that response") cannot
    # be checked against anything. Default off: one line per trajectory carrying up to M indices.
    if os.environ.get("OPD_OMNIOPD_DUMP_ANCHORS", "0") not in ("0", "", "false", "False"):
        print("[OMNIOPD-ANCHORS] sid=%s T=%d anchors=%s" % (
            session_id, len(response_ids), rec.get("omniopd_anchors")), flush=True)
    output.extra_fields["selector_hash"] = OMNIOPD_ONLINE_VARIANT.selector_hash()
    output.extra_fields["selector_variant"] = OMNIOPD_ONLINE_VARIANT.selector_variant

    # EMIT PER TRAJECTORY. aggregate_omniopd_telemetry() consumes these fields, but it runs in the
    # trainer and the non-streaming arm never reaches that path -- so without this line an entire run
    # can complete and leave no evidence that the audit did anything. One line per trajectory is
    # affordable (there are as many as there are samples) and it is the only place k_sem, the anchor
    # count and the prefix-cache hit rate can be read back on any arm.
    if os.environ.get("OPD_OMNIOPD_QUIET", "0") in ("0", "", "false", "False"):
        t = rec.get("omniopd_telemetry") or {}
        ks = rec.get("omniopd_k_sem") or []
        pf = t.get("cached_token_ratio_unreliable")
        logger.info(
            "[OMNIOPD] sid=%s chunks=%d k_sem[min=%.3f mean=%.3f max=%.3f] cachratio=%s gen_tok=%s "
            "gen_s=%.2f variant=%s",
            session_id, len(rec.get("omniopd_anchors") or []),
            min(ks) if ks else float("nan"),
            (sum(ks) / len(ks)) if ks else float("nan"),
            max(ks) if ks else float("nan"),
            ("%.3f" % pf) if pf is not None else "n/a",
            t.get("teacher_gen_tokens"), t.get("teacher_gen_seconds") or 0.0,
            OMNIOPD_ONLINE_VARIANT.selector_variant,
        )
        print(
            # spec=... is UNCONDITIONAL. Without it there is no way to tell from a log whether
            # speculation was actually applied: the controller writes its counters into
            # non_tensor_batch, which nothing prints, and its only stdout line fires on FAILURE.
            # A speculation A/B whose treatment cannot be verified is an A/A, and job 44900605 was
            # exactly that -- the manifest said speculate=1 and the log could not confirm it.
            "[OMNIOPD] sid=%s chunks=%d k_sem_mean=%s cachratio=%s gen_tok=%s gen_s=%.2f "
            "spec=%s/%s skipped=%s"
            % (session_id, len(rec.get("omniopd_anchors") or []),
               ("%.3f" % (sum(ks) / len(ks))) if ks else "n/a",
               ("%.3f" % pf) if pf is not None else "n/a",
               t.get("teacher_gen_tokens"), t.get("teacher_gen_seconds") or 0.0,
               t.get("spec_reused", "off"), t.get("spec_launched", "off"),
               t.get("skipped", "-")),
            flush=True,
        )
    # The entropy series has done its job and is large (one float per response token). Dropping it
    # keeps it out of the object array _postprocess builds for every extra_fields key.
    if os.environ.get("OPD_KEEP_TOKEN_ENTROPIES", "0") in ("0", "", "false", "False"):
        output.extra_fields.pop("token_entropies", None)


def aggregate_omniopd_telemetry(records: list) -> dict:
    """Per-trajectory audit records into omniopd/* metrics. Pure stdlib, no torch.

    `records` are the dicts attach_omniopd_audit wrote, one per trajectory, each optionally carrying
    omniopd_k_sem, omniopd_anchors and omniopd_telemetry.

    THE POINT OF THIS IS TO MAKE TWO SILENT FAILURES LOUD.

    k_sem DEGENERACY. k_sem is a sum of phi over N teacher continuations. If the teacher decodes
    greedily, all N continuations are IDENTICAL, so every chunk scores either ~N (teacher reproduced
    the student) or ~0 (it did not), with nothing in between -- and the objective still trains, on a
    target that has collapsed to a near-binary signal carrying a fraction of the intended
    information. Nothing raises. `k_sem_interior_frac` is the fraction of chunks strictly inside
    (0.05N, 0.95N); a value near zero means the teacher is not sampling, whatever the config says.

    PREFIX REUSE. The M chunk requests per trajectory are issued sequentially precisely so each
    prefill reuses the previous one's KV. If sticky routing breaks, they scatter across replicas and
    every chunk re-ingests a prefix that grows with the response. The only symptom is that the run is
    slow, so the hit fraction is reported rather than assumed.
    """
    out: dict = {}
    if not records:
        return out

    def _pct(xs, p):
        if not xs:
            return 0.0
        s = sorted(xs)
        return float(s[min(len(s) - 1, int(p * len(s)))])

    audited = [r for r in records if r.get("omniopd_anchors")]
    skipped = [r for r in records if not r.get("omniopd_anchors")]
    out["omniopd/trajectories"] = len(records)
    out["omniopd/trajectories_audited"] = len(audited)
    out["omniopd/trajectories_skipped"] = len(skipped)
    for why in ("short_response", "no_anchor"):
        out[f"omniopd/skipped_{why}"] = sum(
            1 for r in skipped if (r.get("omniopd_telemetry") or {}).get("skipped") == why
        )

    n_chunks = [len(r["omniopd_anchors"]) for r in audited]
    out["omniopd/chunks_total"] = sum(n_chunks)
    out["omniopd/chunks_per_trajectory"] = (sum(n_chunks) / len(n_chunks)) if n_chunks else 0.0

    ks = [float(k) for r in audited for k in (r.get("omniopd_k_sem") or [])]
    if ks:
        N = max(1.0, max(ks))                      # k_sem is bounded above by N
        out["omniopd/k_sem_mean"] = sum(ks) / len(ks)
        out["omniopd/k_sem_p05"] = _pct(ks, 0.05)
        out["omniopd/k_sem_p50"] = _pct(ks, 0.50)
        out["omniopd/k_sem_p95"] = _pct(ks, 0.95)
        out["omniopd/k_sem_interior_frac"] = sum(1 for k in ks if 0.05 * N < k < 0.95 * N) / len(ks)
        out["omniopd/k_sem_zero_frac"] = sum(1 for k in ks if k <= 0.05 * N) / len(ks)

    tels = [r.get("omniopd_telemetry") or {} for r in audited]
    pref = sum(t.get("teacher_prefix_tokens") or 0 for t in tels)
    cach = sum(t.get("teacher_cached_tokens") or 0 for t in tels)
    out["omniopd/teacher_prefix_tokens"] = pref
    out["omniopd/teacher_cached_tokens"] = cach
    out["omniopd/cached_token_ratio_unreliable"] = (cach / pref) if pref else 0.0
    out["omniopd/teacher_gen_tokens"] = sum(t.get("teacher_gen_tokens") or 0 for t in tels)
    out["omniopd/teacher_gen_seconds"] = sum(t.get("teacher_gen_seconds") or 0.0 for t in tels)

    variants = {r.get("selector_variant") for r in records if r.get("selector_variant")}
    hashes = {r.get("selector_hash") for r in records if r.get("selector_hash")}
    # More than one selector in a single batch means two different rules produced these rows and the
    # aggregate is a blend of two systems. Surfaced rather than silently averaged.
    out["omniopd/selector_variants"] = sorted(v for v in variants if v)
    out["omniopd/selector_hashes"] = sorted(h for h in hashes if h)
    return out
