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
"""Per-parent FIFO sequencer for incremental teacher scoring (Stage 2).

The incremental teacher scorer (Stage 1) reuses the prefix KV that an EARLIER chunk of the same parent
populated. In production the rollouter launches a parent's chunk-score calls concurrently
(single_turn_agent_loop create_task), so a later chunk can populate the prefix cache before an earlier
chunk is scored; the earlier chunk then requests labels for positions the server now treats as cached
(num_cached > span_start) and the Stage-1 coverage guard correctly fails.

This sequencer serializes teacher scoring WITHIN one parent response (chunk k completes before chunk k+1
begins) while DIFFERENT parents run concurrently. Ordering follows the contiguous span layout -- each
chunk is [span_start, span_end) and chunk k+1 starts exactly at chunk k's span_end -- via an explicit
ordered gate chain, NOT lock-acquisition race order:

    gate[p] is an asyncio.Event that is .set() when the chunk *ending* at position p completes.
    A chunk [s, e) waits for gate[s], scores, then .set()s gate[e] (releasing its successor).
    gate[0] is pre-seeded set, so the first chunk (offset 0) never waits.

One AgentLoopWorker == one asyncio event loop, so the dict bookkeeping below is race-free between awaits
(no global lock; the only awaits are the per-chunk gate wait and the scoring coroutine itself).
"""
import asyncio
import os
import time
from collections import deque
from typing import Awaitable, Callable, Optional

_OFF = ("0", "", "false", "False")


def fifo_enabled(incremental: bool) -> bool:
    """FIFO activates only WITH incremental scoring. If the flag is set without incremental scoring we
    leave behavior unchanged (the caller logs a warning) -- the safer choice, since FIFO is a no-op for
    non-incremental (clean) scoring which has no ordering precondition."""
    return incremental and os.environ.get("OPD_TEACHER_PER_PARENT_FIFO", "0") not in _OFF


class FifoTimeoutError(Exception):
    """A chunk's predecessor did not complete within the FIFO timeout (missing/stuck predecessor)."""


class FifoParentError(Exception):
    """An earlier chunk of this parent failed, so this chunk cannot be scored."""


class FifoDuplicateError(ValueError):
    """A chunk for a span_start that has already begun scoring (duplicate chunk_idx)."""


class _ParentState:
    __slots__ = ("gates", "started", "failed", "last_touch", "max_buffered")

    def __init__(self):
        self.gates: dict[int, asyncio.Event] = {}
        self.started: set[int] = set()
        self.failed: Optional[BaseException] = None
        self.last_touch: float = 0.0
        self.max_buffered: int = 0


class PerParentFifo:
    """Serializes per-parent teacher scoring; concurrent across parents. See module docstring."""

    def __init__(self, timeout_s: float = 120.0, reap_after_s: float = 900.0, clock: Callable[[], float] = time.monotonic):
        self._parents: dict[str, _ParentState] = {}
        self._timeout_s = timeout_s
        self._reap_after_s = reap_after_s
        self._clock = clock
        self.cleanup_count = 0
        self.error_count = 0
        self.timeout_count = 0
        self._wait_times: deque = deque(maxlen=8192)
        self._score_times: deque = deque(maxlen=8192)
        self._max_buffered_per_parent = 0

    def _gate(self, st: _ParentState, pos: int) -> asyncio.Event:
        ev = st.gates.get(pos)
        if ev is None:
            ev = asyncio.Event()
            st.gates[pos] = ev
        return ev

    def _cleanup(self, session_id: str) -> None:
        if self._parents.pop(session_id, None) is not None:
            self.cleanup_count += 1

    def _maybe_cleanup(self, session_id: str, st: _ParentState) -> None:
        # Remove the parent only if nothing is still buffered waiting on it (bounded memory on timeout).
        if not any(not e.is_set() for e in st.gates.values()):
            self._cleanup(session_id)

    def _reap_stale(self, now: float) -> None:
        # Defense in depth against leaks (aborted parent whose final chunk never arrives): sweep parents
        # untouched for reap_after_s. Cheap O(parents) scan amortized over run() calls; no background task.
        if not self._parents:
            return
        stale = [sid for sid, st in self._parents.items() if now - st.last_touch > self._reap_after_s]
        for sid in stale:
            self._cleanup(sid)

    async def run(
        self,
        session_id: str,
        span_start: int,
        span_end: int,
        is_final: bool,
        score: Callable[[], Awaitable],
    ):
        """Score one chunk under per-parent ordering. Returns (result, wait_s, score_s).

        Invariant: for one parent, the chunk ending at `span_start` has fully completed before this chunk's
        `score()` begins; this chunk releases the chunk starting at `span_end` on completion.
        """
        now = self._clock()
        self._reap_stale(now)
        st = self._parents.get(session_id)
        if st is None:
            st = _ParentState()
            g0 = asyncio.Event()
            g0.set()  # the response starts at offset 0: first chunk never waits
            st.gates[0] = g0
            self._parents[session_id] = st
        st.last_touch = now

        # Duplicate detection (race-free: no await between the check and the add).
        if span_start in st.started:
            raise FifoDuplicateError(
                f"FIFO: duplicate chunk span_start={span_start} for parent {session_id}"
            )
        st.started.add(span_start)

        ev = self._gate(st, span_start)
        buffered = sum(1 for e in st.gates.values() if not e.is_set())
        st.max_buffered = max(st.max_buffered, buffered)
        self._max_buffered_per_parent = max(self._max_buffered_per_parent, st.max_buffered)

        t_arrive = now
        if not ev.is_set():
            try:
                await asyncio.wait_for(ev.wait(), timeout=self._timeout_s)
            except asyncio.TimeoutError:
                self.timeout_count += 1
                st.started.discard(span_start)
                st.gates.pop(span_start, None)
                self._maybe_cleanup(session_id, st)
                raise FifoTimeoutError(
                    f"FIFO: predecessor of span@{span_start} for parent {session_id} did not "
                    f"complete within {self._timeout_s}s"
                )
        if st.failed is not None:
            raise FifoParentError(
                f"FIFO: parent {session_id} aborted by an earlier chunk failure"
            ) from st.failed
        wait_s = self._clock() - t_arrive
        self._wait_times.append(wait_s)

        t_score = self._clock()
        try:
            result = await score()
        except BaseException as exc:
            if not isinstance(exc, asyncio.CancelledError):
                self.error_count += 1
            st.failed = exc
            for e in st.gates.values():  # wake every waiter so they fail fast instead of timing out
                if not e.is_set():
                    e.set()
            self._cleanup(session_id)
            raise
        score_s = self._clock() - t_score
        self._score_times.append(score_s)

        self._gate(st, span_end).set()  # release the successor
        st.last_touch = self._clock()
        if is_final:
            self._cleanup(session_id)
        return result, wait_s, score_s

    @staticmethod
    def _pct(xs, q: float) -> float:
        if not xs:
            return 0.0
        s = sorted(xs)
        return s[min(len(s) - 1, int(q * len(s)))]

    def snapshot(self) -> dict:
        """Current FIFO metrics (per-worker)."""
        return {
            "active_parents": len(self._parents),
            "buffered_chunks": sum(
                sum(1 for e in st.gates.values() if not e.is_set()) for st in self._parents.values()
            ),
            "max_buffered_chunks_per_parent": self._max_buffered_per_parent,
            "wait_time_s_p50": self._pct(self._wait_times, 0.50),
            "wait_time_s_p95": self._pct(self._wait_times, 0.95),
            "score_time_s_p50": self._pct(self._score_times, 0.50),
            "score_time_s_p95": self._pct(self._score_times, 0.95),
            "cleanup_count": self.cleanup_count,
            "error_count": self.error_count,
            "timeout_count": self.timeout_count,
        }
