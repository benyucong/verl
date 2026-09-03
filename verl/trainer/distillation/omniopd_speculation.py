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

import array
import asyncio
import hashlib
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


def spec_top_k() -> int:
    """How many anchors to speculate on per trajectory. 0 = all M.

    Every proposal costs a teacher RPC whose completion callback lands on the generation event
    loop, so the RPC count -- not the GPU work -- is what competes with token streaming. Speculating
    on the highest-entropy few keeps most of the hideable work while cutting that traffic
    proportionally.
    """
    return int(os.environ.get("OPD_OMNIOPD_SPEC_TOP_K", "0") or 0)


def propose_anchors(entropies, prompt_len: int, resp_len: int, M: int, C: int,
                    margin: float = 0.0, top_k: int = 0) -> list[int]:
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
    if not anchors:
        return anchors
    if margin > 0.0:
        scores = sorted((float(entropies[t]) for t in anchors), reverse=True)
        cutoff = scores[-1] + margin
        anchors = [t for t in anchors if float(entropies[t]) >= cutoff]
    if top_k and len(anchors) > top_k:
        # keep the highest-entropy top_k, in ascending position order (prefix nesting)
        keep = sorted(anchors, key=lambda t: float(entropies[t]), reverse=True)[:top_k]
        anchors = sorted(keep)
    return anchors


_SPEC_SEM: dict = {}
_SPEC_LOOP: dict = {}


def _dispatch_loop():
    """A dedicated event loop, on its own thread, for speculative teacher RPCs.

    THIS IS THE CEILING ON THE WHOLE MECHANISM. Perfect overlap of a 22.5 s audit behind a 15.4 s
    generation would be +68%; we measure +5.2%, and the gap is that only 4 of 10 anchors can be
    speculated. Not because the teacher cannot take them -- because each proposal's RPC and its
    completion callback are serviced on the SAME loop that drains the token stream, so proposal
    count trades directly against generation speed. That is why top_k=4 beats top_k=0 (parity) and
    top_k=2 is worse than both: it is an optimum of a tradeoff that should not exist.

    Moving dispatch to its own loop breaks the coupling. The generation thread does not wait on it,
    and RPC completions no longer interleave with token processing, so top_k can go to M.

    Off by default. Enabled with OPD_OMNIOPD_SPEC_THREAD=1.
    """
    import threading
    key = "dispatch"
    ent = _SPEC_LOOP.get(key)
    if ent is not None:
        return ent
    loop = asyncio.new_event_loop()
    t = threading.Thread(target=loop.run_forever, name="omniopd-spec", daemon=True)
    t.start()
    _SPEC_LOOP[key] = (loop, t)
    return _SPEC_LOOP[key]


def dispatch_thread_enabled() -> bool:
    return os.environ.get("OPD_OMNIOPD_SPEC_THREAD", "0") not in ("0", "", "false", "False")


def _launch_jitter_s() -> float:
    """Max seconds to spread one trajectory's early launches over. 0 = off (fire at the crossing).

    Sized against the BOUNDARY INTERVAL, not the step: at spacing 1024 with ~8000-token responses a
    boundary passes every ~1/8th of generation, so a jitter of a few seconds de-synchronises the
    batch while costing almost none of the head start the mechanism exists to buy.
    """
    try:
        return max(0.0, float(os.environ.get("OPD_STATE_CREDIT_JITTER_S", "0") or 0))
    except ValueError:
        return 0.0


def _stable_hash(s) -> int:
    """Deterministic across processes -- Python's hash() is salted per interpreter, so using it here
    would give the two arms different launch schedules and make them incomparable."""
    return int(hashlib.sha256(str(s).encode()).hexdigest()[:8], 16)


def _spec_semaphore(size: Optional[int] = None):
    """Bound in-flight SPECULATIVE teacher requests, per event loop.

    `size` is the caller's own budget. Without it this falls back to the OmniOPD speculation
    default, which is correct for speculation and WRONG for anyone else: state-credit's store sets
    max_inflight from OPD_STATE_CREDIT_MAX_INFLIGHT, and that value used to select this semaphore
    without sizing it -- so asking for 8 applied 4 while the banner printed 8. A cap that reports
    one number and enforces another is worse than no cap: it makes a throttled A/B look like an
    unthrottled one.

    Without this a proposal is fired the instant a chunk closes, for every trajectory in flight. At
    staleness=1 the in-flight budget is ppo_mini x (staleness+1) x sync = 32 trajectories, each
    proposing up to M=10 anchors -- up to ~320 concurrent teacher requests against an engine that
    admits max_num_seqs/N ~= 12 at a time. The proposals do not merely wait: they occupy the very
    admission slots the COMMITS need, and a commit is what training is blocked on. Speculation then
    makes the critical path longer, which is the -25.4% measured at q128 (job 44912520).

    Speculation is only ever worth spare capacity, so it takes a small fixed budget and commits are
    never gated. Default deliberately well under the engine's concurrent-request ceiling.
    """
    loop = asyncio.get_event_loop()
    key = (loop, int(size) if size else 0)   # a distinct budget gets a distinct semaphore
    sem = _SPEC_SEM.get(key)   # per-loop: the dispatch loop gets its own budget
    if sem is None:
        n = int(size) if size else int(os.environ.get("OPD_OMNIOPD_SPEC_MAX_INFLIGHT", "4") or 4)
        sem = asyncio.Semaphore(max(1, n))
        _SPEC_SEM[key] = sem
    return sem


class SpeculativeStore:
    """Per-trajectory record of launched proposals. Not shared across trajectories."""

    def __init__(self, base_seed: Optional[int], N: int, C: int, label: str = "OMNIOPD-SPEC",
                 max_inflight: Optional[int] = None):
        # Tag on every log line. The store is shared with state-credit's early launch, and a
        # mismatch there reported as [OMNIOPD-SPEC] sends you reading the wrong subsystem.
        self.label = label
        # None => the shared speculative budget (small, deliberately). 0 => NO throttle.
        #
        # The shared budget exists because a SPECULATIVE proposal can miss, so it must never occupy
        # an admission slot a commit needs. That argument does not transfer to work the commit is
        # guaranteed to ask for: throttling it does not protect the commit, it IS the commit,
        # issued early and then serialised. And the arm it is compared against -- issuing the same
        # calls at commit time -- has no throttle at all, so a bound here handicaps the early arm
        # against its own baseline rather than measuring the mechanism.
        self.max_inflight = max_inflight
        self.base_seed, self.N, self.C = base_seed, N, C
        self.tasks: dict[int, asyncio.Task] = {}       # anchor -> in-flight/finished launch
        self.keys: dict[int, str] = {}                # anchor -> request identity at launch
        self.launched_at: dict[int, float] = {}
        self.key_mismatch = 0
        self.t_first_launch: Optional[float] = None
        self.reused = 0
        self.relaunched = 0
        self.wasted = 0
        self.launch_calls = 0

    def pending(self, anchors) -> list[int]:
        return [t for t in anchors if t not in self.tasks]

    @staticmethod
    def request_key(prefix_ids, n: int, max_tokens: int, seed) -> str:
        """Identity of one unit of teacher work.

        Reuse is only fidelity-preserving if the stored result answers the SAME request the commit
        would have issued. Matching on anchor position alone assumes that, and the assumption is
        load-bearing: a prefix or seed that differed would substitute a different sample into k_sem
        with nothing raising. Hashed rather than compared element-wise because the prefix is up to
        a few thousand ids and this runs per anchor.
        """
        h = hashlib.sha256()
        h.update(str(len(prefix_ids)).encode())
        h.update(memoryview(array.array("l", prefix_ids)).cast("B"))
        h.update(b"|%d|%d|%s" % (int(n), int(max_tokens), str(seed).encode()))
        return h.hexdigest()[:16]

    def launch(self, anchor: int, coro_factory, key: Optional[str] = None,
               key_factory=None) -> None:
        """Start one proposal. Never awaited here -- that is the entire point.

        ensure_future makes the teacher CALL concurrent, but it does not move any of this off the
        event loop that is also draining the generation stream. Everything the factory does before
        its first await -- notably materialising the prefix -- runs synchronously on that loop, so
        the factory is written to do its work INSIDE the coroutine, after the first yield point,
        rather than in the caller.

        `key_factory` extends that same rule to the request KEY, which used to break it. Passing
        `key=` means the caller has already built the prefix and hashed it ON the generation loop:
        a list concat of `anchor` ids, an 8-byte-per-token array copy and a sha256 over it, per
        depth per trajectory -- work that is QUADRATIC in depth count, since depth k hashes k
        chunks. One event loop serves every concurrent trajectory in the batch, so that cost is
        subtracted from token draining for all of them, and it exists ONLY in the arm that launches
        early. It is therefore invisible to the STREAM_ONLY control, which was built to make the
        two arms differ in release time alone.

        Pass `key_factory` instead: a zero-argument callable evaluated inside the coroutine, on the
        dispatch loop, after the first yield. `take()` reads the key only after awaiting the task,
        so it is always set by the time it is compared.
        """
        if anchor in self.tasks:
            return

        # JITTER. Every trajectory in a batch decodes at roughly the same rate, so they all cross
        # boundary d at roughly the same MOMENT -- and "release at the earliest causally valid
        # moment" turns out to mean "release at the same instant as everyone else". The teacher then
        # receives batch_size x M requests in one burst per boundary instead of a stream, and
        # everything behind the burst waits.
        #
        # Measured (roihu, spacing 1024, 64 trajectories => 256-wide bursts): the early arm's teacher
        # RPC time was slightly LOWER than the sequential arm's (1064s vs 1138s) while its commit
        # waited LONGER (145s vs 116s). Same work, started sooner, finished later -- which only
        # happens in a queue. EOS-triggered release is naturally staggered because response lengths
        # vary hugely (sd ~5000 tokens), so the baseline got its smoothing for free.
        #
        # A per-in-flight CAP throttles the queue; it does not stop it forming (measured: halved the
        # penalty, did not remove it). Spreading each trajectory's launches does. The offset is
        # DETERMINISTIC in the session id -- a random one would make the schedule unreproducible
        # across arms, and reproducibility is what lets the two arms be compared at all.
        _j = _launch_jitter_s()
        # base_seed is derived from the session id, so it is unique per TRAJECTORY and stable
        # across processes -- which is what de-synchronises trajectories from each other while
        # keeping each one's schedule identical between the two arms.
        _off = ((_stable_hash(self.base_seed) % 1000) / 1000.0 * _j) if _j > 0 else 0.0

        def _set_key():
            # On the dispatch loop, not the caller's. Ordered BEFORE the jitter sleep so the key is
            # in place as early as possible; take() does not read it until the task completes.
            if key_factory is not None:
                self.keys[anchor] = key_factory()

        if self.max_inflight == 0:
            async def _gated():
                await asyncio.sleep(0)      # yield first: let the generation stream advance
                _set_key()
                if _off:
                    await asyncio.sleep(_off)
                return await coro_factory()
        else:
            async def _gated():
                await asyncio.sleep(0)      # yield first: let the generation stream advance
                _set_key()
                if _off:
                    await asyncio.sleep(_off)
                async with _spec_semaphore(self.max_inflight):
                    return await coro_factory()

        if dispatch_thread_enabled():
            # Runs on the dispatch loop; the generation thread only hands over a coroutine. The
            # returned future is bound to THIS loop so the commit can await it normally.
            loop, _ = _dispatch_loop()
            cfut = asyncio.run_coroutine_threadsafe(_gated(), loop)
            self.tasks[anchor] = asyncio.wrap_future(cfut)
        else:
            self.tasks[anchor] = asyncio.ensure_future(_gated())
        if key is not None:
            self.keys[anchor] = key
        now = time.time()
        self.launched_at[anchor] = now
        self.launch_calls += 1
        if self.t_first_launch is None:
            self.t_first_launch = now

    async def take(self, anchor: int, expect_key: Optional[str] = None):
        """The committed set is asking for this anchor. Returns (seqs, telemetry) or None.

        Awaits a proposal that is still in flight rather than abandoning it: the work is already
        paid for, and racing it with a fresh launch would double the cost of the very anchor
        speculation was supposed to make cheaper.
        """
        task = self.tasks.pop(anchor, None)
        if task is None:
            return None
        # AWAIT FIRST, then compare. With key_factory the key is computed on the dispatch loop
        # inside the task, so it does not exist until the task has run. Reading self.keys before
        # awaiting would see None and silently accept ANY result as matching -- the exact
        # substitution the key was added to prevent. The old order was observationally identical
        # (the mismatch branch awaited the task anyway before returning None), so nothing else
        # changes here.
        try:
            seqs, tele = await task
        except asyncio.CancelledError:
            # CancelledError derives from BaseException, so `except Exception` below does NOT catch
            # it. Left unhandled it escapes take() -> _fetch -> the commit's gather and kills the
            # trajectory -- and only in the arm that has a store, which is the asymmetry this whole
            # pass exists to remove.
            #
            # Distinguish WHOSE cancellation it is. If the proposal itself was cancelled, it is just
            # an unusable proposal: drop it and let the caller issue the depth fresh. If WE are
            # being cancelled, swallowing it would break cooperative cancellation and leave the
            # commit issuing fresh teacher work during shutdown, so re-raise.
            if task.cancelled():
                logger.warning("[%s] proposal for anchor %d was cancelled, relaunching",
                               self.label, anchor)
                self.keys.pop(anchor, None)
                return None
            raise
        except Exception as e:                       # a failed proposal is not a failed trajectory
            logger.warning("[%s] proposal for anchor %d failed, relaunching: %s", self.label, anchor, e)
            self.keys.pop(anchor, None)
            return None
        launched_key = self.keys.pop(anchor, None)
        if expect_key is not None and launched_key is not None and launched_key != expect_key:
            # NOT reusable. The stored result answers a different request than the commit is
            # asking for, so using it would substitute a different sample into k_sem silently.
            # Drop it and let the caller run the anchor fresh; the wasted work is already spent.
            self.key_mismatch += 1
            logger.warning("[%s] anchor %d key mismatch (launched %s, commit %s); relaunching "
                           "rather than reusing", self.label, anchor, launched_key, expect_key)
            return None
        self.reused += 1
        return seqs, tele

    async def drain_unused(self) -> None:
        """Release unmatched proposals WITHOUT blocking the caller. v1 does not abort.

        They are NOT cancelled: the decode has already been issued to the engine, so cancelling the
        awaitable frees no GPU work, and dropping the reference without any handler produces
        'task exception was never retrieved' noise that hides real failures.

        But they are no longer AWAITED either, and that distinction is the whole point. This runs on
        the commit path, so awaiting an orphan put the teacher's slowest discarded request directly
        in series with the trajectory's critical path -- in the ARM THAT LAUNCHES EARLY ONLY, since
        the control has no store and never calls this. An orphan is produced whenever a response
        ends exactly on a depth (launch fires at resp_len >= d, the commit keeps d < T), so the
        penalty lands on a subset of trajectories in one arm and on none in the other.

        A done-callback retrieves the result instead, which suppresses the warning without putting
        the wait on the critical path. Coroutine only for signature compatibility with callers that
        await it.
        """
        def _reap(fut, _self=self):
            try:
                if fut.cancelled():
                    return
                fut.exception()          # retrieve, so it is not reported as never-retrieved
            except Exception:
                pass                # a future that cannot even report its own state is not our problem

        for anchor, task in list(self.tasks.items()):
            # Counted HERE, synchronously: an orphan is wasted the moment the commit releases it
            # without taking it, which is knowable now. Counting it in the done-callback instead
            # made spec_wasted a permanent 0 for every reader -- including omniopd_stage.py, which
            # calls telemetry() and had been getting a real number before.
            self.wasted += 1
            try:
                if not task.done():
                    task.add_done_callback(_reap)
                else:
                    _reap(task)
            except Exception:
                pass
            self.tasks.pop(anchor, None)

    def telemetry(self, t_eos: Optional[float] = None) -> dict:
        out = {
            "spec_launched": self.launch_calls,
            "spec_reused": self.reused,
            "spec_relaunched": self.relaunched,
            "spec_wasted": self.wasted,
            "spec_key_mismatch": self.key_mismatch,
        }
        committed = self.reused + self.relaunched
        out["spec_hit_rate"] = (self.reused / committed) if committed else 0.0
        out["spec_waste_multiplier"] = (self.launch_calls / committed) if committed else 0.0
        if t_eos is not None and self.launched_at:
            leads = [t_eos - t for t in self.launched_at.values()]
            # THE BINDING LEAD IS THE SMALLEST ONE. The commit gathers EVERY depth, so the
            # trajectory is gated by whichever proposal finishes last -- normally the one launched
            # last (the deepest), which had the least head start. Reporting t_first_launch instead
            # gives the SHALLOWEST depth's lead, i.e. the maximum over depths, and the doc's own
            # criterion ("a lead shorter than the teacher's service time cannot hide anything")
            # then reads the most flattering number available. On a [2560,5120,7680] trajectory at
            # ~1 tok/ms that is 5.4s reported against a 0.3s lead that actually binds.
            out["spec_lead_s"] = max(0.0, min(leads))
            out["spec_lead_first_s"] = max(0.0, max(leads))
            # STILL AN UPPER BOUND, both ends. launched_at is stamped when launch() creates the
            # task -- before the first yield, the key hash, the jitter sleep and the semaphore --
            # not when the RPC is submitted; and t_eos is taken at commit entry, after scoring. Both
            # errors inflate the lead, and the control prints nothing to cancel them. Treat a large
            # lead as "not yet ruled out", never as "the work was hidden".
        return out
