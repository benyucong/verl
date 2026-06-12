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
"""CPU tests for the Stage-2 per-parent FIFO teacher-scoring sequencer.

Spans are contiguous 256-token chunks: A0=[0,256), A1=[256,512), A2=[512,768).
"""
import asyncio

import pytest

from verl.experimental.teacher_loop.fifo import (
    FifoDuplicateError,
    FifoParentError,
    FifoTimeoutError,
    PerParentFifo,
    fifo_enabled,
)


def test_1_same_parent_orders_by_span():
    """A2, A0, A1 submitted concurrently -> score order must be A0, A1, A2."""

    async def body():
        fifo = PerParentFifo(timeout_s=5.0)
        order = []

        async def score(tag):
            await asyncio.sleep(0.01)
            order.append(tag)
            return tag

        tasks = [
            asyncio.create_task(fifo.run("A", 512, 768, True, lambda: score("A2"))),
            asyncio.create_task(fifo.run("A", 0, 256, False, lambda: score("A0"))),
            asyncio.create_task(fifo.run("A", 256, 512, False, lambda: score("A1"))),
        ]
        await asyncio.gather(*tasks)
        assert order == ["A0", "A1", "A2"]
        assert "A" not in fifo._parents  # cleaned on is_final

    asyncio.run(body())


def test_2_cross_parent_concurrency_with_overlap():
    """A and B each preserve internal order AND overlap in wall-clock (no global lock)."""

    async def body():
        fifo = PerParentFifo(timeout_s=5.0)
        evA0, evB0 = asyncio.Event(), asyncio.Event()
        orderA, orderB = [], []

        async def scoreA0():
            evA0.set()
            await asyncio.wait_for(evB0.wait(), 2.0)  # only completes if B0 runs concurrently
            orderA.append("A0")

        async def scoreB0():
            evB0.set()
            await asyncio.wait_for(evA0.wait(), 2.0)
            orderB.append("B0")

        async def scoreA1():
            orderA.append("A1")

        async def scoreB1():
            orderB.append("B1")

        tasks = [
            asyncio.create_task(fifo.run("A", 0, 256, False, scoreA0)),
            asyncio.create_task(fifo.run("A", 256, 512, True, scoreA1)),
            asyncio.create_task(fifo.run("B", 0, 256, False, scoreB0)),
            asyncio.create_task(fifo.run("B", 256, 512, True, scoreB1)),
        ]
        await asyncio.gather(*tasks)  # would deadlock/timeout if A and B could not overlap
        assert orderA == ["A0", "A1"]
        assert orderB == ["B0", "B1"]

    asyncio.run(body())


def test_3_final_chunk_cleanup():
    """parent state removed after the final chunk completes."""

    async def body():
        fifo = PerParentFifo()

        async def score():
            return None

        await fifo.run("A", 0, 256, True, score)
        assert "A" not in fifo._parents
        assert fifo.cleanup_count >= 1

    asyncio.run(body())


def test_4_teacher_exception_fails_waiters_and_cleans():
    """A0 score raises -> A0 propagates it, waiting A1 fails with FifoParentError, state cleaned."""

    async def body():
        fifo = PerParentFifo(timeout_s=5.0)

        async def boom():
            raise RuntimeError("teacher failed")

        async def ok():
            return None

        tA1 = asyncio.create_task(fifo.run("A", 256, 512, True, ok))
        await asyncio.sleep(0.02)  # let A1 register and block on gate[256]
        with pytest.raises(RuntimeError):
            await fifo.run("A", 0, 256, False, boom)
        with pytest.raises(FifoParentError):
            await tA1
        assert "A" not in fifo._parents
        assert fifo.error_count >= 1

    asyncio.run(body())


def test_5_cancellation_cleans_state():
    """Cancelling an in-flight score cleans parent state, no hanging futures."""

    async def body():
        fifo = PerParentFifo(timeout_s=5.0)
        started = asyncio.Event()

        async def slow():
            started.set()
            await asyncio.sleep(10)

        tA0 = asyncio.create_task(fifo.run("A", 0, 256, False, slow))
        await started.wait()
        tA0.cancel()
        with pytest.raises(asyncio.CancelledError):
            await tA0
        assert "A" not in fifo._parents

    asyncio.run(body())


def test_6_duplicate_span_fails_loudly():
    """A duplicate chunk for an already-started span_start raises FifoDuplicateError."""

    async def body():
        fifo = PerParentFifo(timeout_s=5.0)
        started = asyncio.Event()

        async def slow():
            started.set()
            await asyncio.sleep(0.1)

        async def ok():
            return None

        tA0 = asyncio.create_task(fifo.run("A", 0, 256, False, slow))
        await started.wait()
        with pytest.raises(FifoDuplicateError):
            await fifo.run("A", 0, 256, False, ok)
        await tA0

    asyncio.run(body())


def test_7_missing_predecessor_times_out():
    """A1 with no A0 -> the predecessor gate never opens -> explicit timeout, no deadlock."""

    async def body():
        fifo = PerParentFifo(timeout_s=0.2)

        async def ok():
            return None

        with pytest.raises(FifoTimeoutError):
            await fifo.run("A", 256, 512, False, ok)
        assert fifo.timeout_count >= 1

    asyncio.run(body())


def test_8_fifo_gating_disabled(monkeypatch):
    """fifo_enabled requires BOTH the flag and incremental scoring."""
    monkeypatch.delenv("OPD_TEACHER_PER_PARENT_FIFO", raising=False)
    assert fifo_enabled(incremental=True) is False
    monkeypatch.setenv("OPD_TEACHER_PER_PARENT_FIFO", "1")
    assert fifo_enabled(incremental=True) is True
    assert fifo_enabled(incremental=False) is False  # FIFO needs incremental scoring
    monkeypatch.setenv("OPD_TEACHER_PER_PARENT_FIFO", "0")
    assert fifo_enabled(incremental=True) is False


def test_9_snapshot_metrics_shape():
    """snapshot() exposes the required teacher_fifo/* fields."""

    async def body():
        fifo = PerParentFifo()

        async def score():
            return None

        await fifo.run("A", 0, 256, True, score)
        snap = fifo.snapshot()
        for k in (
            "active_parents", "buffered_chunks", "max_buffered_chunks_per_parent",
            "wait_time_s_p50", "wait_time_s_p95", "score_time_s_p50", "score_time_s_p95",
            "cleanup_count", "error_count", "timeout_count",
        ):
            assert k in snap

    asyncio.run(body())
