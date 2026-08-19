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
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:  # as a package member
    from .omniopd_producer import assemble_omniopd_record, build_chunk_requests, select_anchors
    from .token_entropy import to_signal_series
except ImportError:  # or with the distillation dir on sys.path, which is how the tests import it
    from omniopd_producer import assemble_omniopd_record, build_chunk_requests, select_anchors
    from token_entropy import to_signal_series


def omniopd_enabled(config: Any) -> bool:
    """OmniOPD runs only when the loss asks for teacher generation.

    Read off the same flag the loss registry keys on rather than a separate env switch, so a run
    cannot be configured into a state where the trainer expects an audit the rollout never produced.
    """
    try:
        return bool(config.distillation.distillation_loss.loss_settings.use_teacher_generation)
    except AttributeError:
        return False


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

    signal = to_signal_series(entropies, prompt_len=len(prompt_ids))
    anchors = select_anchors(signal, len(prompt_ids), len(response_ids), M, C)
    if not anchors:
        return {"omniopd_anchors": [], "omniopd_k_sem": [], "omniopd_telemetry": {"skipped": "no_anchor"}}

    prefixes = build_chunk_requests(prompt_ids, response_ids, anchors)

    continuations: list[list[str]] = []
    tele = {"teacher_gen_seconds": 0.0, "teacher_gen_tokens": 0, "teacher_prefix_tokens": 0,
            "teacher_cached_tokens": 0, "chunks": len(anchors)}
    for i, prefix in enumerate(prefixes):                      # ascending: chunk k+1 reuses chunk k's KV
        seqs, t = await teacher_manager.generate_chunk_continuations(
            prefix_ids=prefix,
            n=N,
            max_tokens=C,
            routing_key=routing_key,
            session_id=session_id,
            seed=None if seed is None else seed + i,
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
    # The cache-hit fraction is the whole reason the calls are ordered rather than concurrent, so it
    # is recorded per trajectory: if it collapses, sticky routing has broken and the audit silently
    # costs several times what it should.
    if tele["teacher_prefix_tokens"]:
        tele["prefix_cache_frac"] = tele["teacher_cached_tokens"] / tele["teacher_prefix_tokens"]
    record["omniopd_telemetry"] = tele
    return record


async def attach_omniopd_audit(output, *, prompt_ids, response_ids, teacher_manager, tokenizer,
                               omniopd_config, session_id=None, routing_key=None, seed=None) -> None:
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
    )
    output.extra_fields["omniopd_anchors"] = rec["omniopd_anchors"]
    output.extra_fields["omniopd_k_sem"] = rec["omniopd_k_sem"]
    output.extra_fields["omniopd_telemetry"] = rec.get("omniopd_telemetry")
    # The entropy series has done its job and is large (one float per response token). Dropping it
    # keeps it out of the object array _postprocess builds for every extra_fields key.
    if os.environ.get("OPD_KEEP_TOKEN_ENTROPIES", "0") in ("0", "", "false", "False"):
        output.extra_fields.pop("token_entropies", None)
