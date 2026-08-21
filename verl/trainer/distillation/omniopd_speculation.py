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
"""Speculative early launch of OmniOPD teacher work, over OPDFlow's chunk stream.

THE PROBLEM. OmniOPD picks its M audit anchors by argmax over the entropy of the WHOLE finished
response, so nothing is knowable until EOS and the teacher -- 87% of the per-trajectory critical
path -- cannot start until generation is over. Chunk streaming moves the DATA earlier but not the
WORK.

WHAT THIS DOES. At each chunk boundary it runs the same selection rule over the entropy PREFIX,
and launches teacher continuations for the anchors it proposes, concurrently with ongoing
generation. At EOS the audit computes the true anchor set exactly as before and takes each
committed anchor from this store when a proposal matches it, or runs it then when none does.

WHY THIS CANNOT CHANGE THE OBJECTIVE. Fidelity here is structural rather than statistical: nothing
in this module writes supervision. `omniopd_anchors` and `omniopd_k_sem` are written at exactly one
site, in attach_omniopd_audit, behind the final-chunk gate, from the unchanged commit-time
selection. This module only ever hands back continuations for an anchor the commit ALREADY chose.
A proposal rule with 0% recall would therefore train precisely the same spans -- it would simply
hide no work. That is what makes this a systems change with no quality-validation burden.

WHAT MAKES A PROPOSAL REUSABLE. The teacher request is (prefix, n, max_tokens, temperature, top_p,
seed). The prefix is prompt + response[:anchor], a pure function of the anchor, and the response is
fixed once generated. n, max_tokens and the sampling parameters are config constants. The seed is
base + N*anchor -- keyed on POSITION, not on the anchor's rank in the committed set, which is
unknowable while generation is still running. So a proposal at position t is the identical unit of
work the commit would have issued for t, and reuse is exact at the level of the request. It is NOT
a claim that the returned tokens are bitwise reproducible: at temperature 1.0 under continuous
batching the composition of the running batch changes reduction order, and this is not claimed.

V1 IS LAUNCH-ONLY. There is no abort. An unmatched proposal simply runs to completion and its
result is dropped. Cancellation is a much larger piece of machinery (see speculative_attempt.py)
and it is only worth building once the waste is measured and shown to matter.
"""

import asyncio
import logging
import os
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:  # as a package member
    from .omniopd_producer import build_chunk_requests, select_anchors
    from .token_entropy import to_signal_series
except ImportError:  # or with the distillation dir on sys.path (tests)
    from omniopd_producer import build_chunk_requests, select_anchors
    from token_entropy import to_signal_series


def speculation_enabled() -> bool:
    """Default OFF. Speculation adds teacher decode, which on a queue-bound teacher lands on the
    binding resource -- so it must be an explicit choice, and it must appear in the run label."""
    return os.environ.get("OPD_OMNIOPD_SPECULATE", "0") not in ("0", "", "false", "False")


def seed_for_anchor(base_seed: Optional[int], anchor: int, N: int) -> Optional[int]:
    """The single definition of a chunk's seed. Must match the commit path exactly, or a proposal is
    a different unit of work and can never be reused."""
    if base_seed is None:
        return None
    return (int(base_seed) + int(N) * int(anchor)) % (2**31)


def propose_anchors(entropies, prompt_len: int, resp_len: int, M: int, C: int,
                    margin: float = 0.0) -> list[int]:
    """Anchors the commit would pick if the response ended here, optionally thinned by a margin.

    Same selection rule as the commit (select_anchors over the signal series), applied to the
    prefix. A position proposed now survives to the committed set only if fewer than M positions in
    the REST of the response outscore it, which is unknowable -- so `margin` trades recall for
    waste: a proposal is kept only if its entropy exceeds the prefix's M-th best by that margin,
    which is a cheap proxy for "unlikely to be displaced". margin=0 proposes the plain prefix top-M.
    """
    if resp_len < C:
        return []
    signal = to_signal_series(list(entropies)[:resp_len], prompt_len=prompt_len)
    anchors = select_anchors(signal, prompt_len, resp_len, M, C)
    if margin <= 0.0 or not anchors:
        return anchors
    scores = sorted((float(entropies[t]) for t in anchors), reverse=True)
    cutoff = scores[-1] + margin
    return [t for t in anchors if float(entropies[t]) >= cutoff]


class SpeculativeStore:
    """Per-trajectory record of launched proposals. Not shared across trajectories."""

    def __init__(self, base_seed: Optional[int], N: int, C: int):
        self.base_seed, self.N, self.C = base_seed, N, C
        self.tasks: dict[int, asyncio.Task] = {}       # anchor -> in-flight/finished launch
        self.launched_at: dict[int, float] = {}
        self.t_first_launch: Optional[float] = None
        self.reused = 0
        self.relaunched = 0
        self.wasted = 0
        self.launch_calls = 0

    def pending(self, anchors) -> list[int]:
        return [t for t in anchors if t not in self.tasks]

    def launch(self, anchor: int, coro_factory) -> None:
        """Start one proposal. Never awaited here -- that is the entire point."""
        if anchor in self.tasks:
            return
        self.tasks[anchor] = asyncio.ensure_future(coro_factory())
        now = time.time()
        self.launched_at[anchor] = now
        self.launch_calls += 1
        if self.t_first_launch is None:
            self.t_first_launch = now

    async def take(self, anchor: int):
        """The committed set is asking for this anchor. Returns (seqs, telemetry) or None.

        Awaits a proposal that is still in flight rather than abandoning it: the work is already
        paid for, and racing it with a fresh launch would double the cost of the very anchor
        speculation was supposed to make cheaper.
        """
        task = self.tasks.pop(anchor, None)
        if task is None:
            return None
        try:
            seqs, tele = await task
        except Exception as e:                       # a failed proposal is not a failed trajectory
            logger.warning("[OMNIOPD-SPEC] proposal for anchor %d failed, relaunching: %s", anchor, e)
            return None
        self.reused += 1
        return seqs, tele

    async def drain_unused(self) -> None:
        """Let unmatched proposals finish and drop their results. v1 does not abort.

        They are NOT cancelled: the decode has already been issued to the engine, so cancelling the
        awaitable frees no GPU work, and dropping the reference without awaiting produces
        'task exception was never retrieved' noise that hides real failures.
        """
        for anchor, task in list(self.tasks.items()):
            try:
                await task
                self.wasted += 1
            except Exception:
                pass
            self.tasks.pop(anchor, None)

    def telemetry(self, t_eos: Optional[float] = None) -> dict:
        out = {
            "spec_launched": self.launch_calls,
            "spec_reused": self.reused,
            "spec_relaunched": self.relaunched,
            "spec_wasted": self.wasted,
        }
        committed = self.reused + self.relaunched
        out["spec_hit_rate"] = (self.reused / committed) if committed else 0.0
        out["spec_waste_multiplier"] = (self.launch_calls / committed) if committed else 0.0
        if t_eos is not None and self.t_first_launch is not None:
            out["spec_lead_s"] = max(0.0, t_eos - self.t_first_launch)
        return out
