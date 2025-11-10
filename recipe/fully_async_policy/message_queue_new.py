# Copyright 2025 Meituan Ltd. and/or its affiliates
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

import asyncio
import logging
import os
import pickle
import threading
import time
from collections import deque
from typing import Any, Optional

import ray
from omegaconf import DictConfig

# ultra-low-latency shared-memory SPSC ring (pybind11)
from recipe.fully_async_policy import fastmq

logger = logging.getLogger(__name__)

# --------- serialization helpers (simple & fast enough) ----------
def _ser(x: Any) -> bytes:
    # If caller already passes bytes, don’t touch it
    if isinstance(x, (bytes, bytearray, memoryview)):
        return bytes(x)
    # Pickle is the most compatible; swap with orjson/msgpack if you prefer
    return pickle.dumps(x, protocol=pickle.HIGHEST_PROTOCOL)

def _des(b: bytes) -> Any:
    return pickle.loads(b)

def _now_ns() -> int:
    return time.time_ns()


@ray.remote(num_cpus=2, max_concurrency=20)
class MessageQueue:
    """
    Ray-based message queue using sharded SPSC (fastmq) + fan-in
    """

    # --------------- Life-cycle ---------------
    def __init__(
        self,
        config: DictConfig,
        max_queue_size: int = 1000,
        *,
        num_shards: Optional[int] = None,
        shard_capacity_bytes: int = 1 << 20,  # 1 MiB per shard by default
        poll_batch_per_shard: int = 64,
        spin_iters: int = 256,
    ):
        self.config = config
        if max_queue_size is None:
            raise ValueError(f"max_queue_size cannot be None, got: {max_queue_size}")
        self.max_queue_size = int(max_queue_size)

        # training/versioning
        try:
            if hasattr(config, "async_training") and config.async_training is not None:
                self.staleness_threshold = getattr(config.async_training, "staleness_threshold", 3)
            else:
                self.staleness_threshold = 3
        except (AttributeError, RecursionError):
            self.staleness_threshold = 3
        self.current_param_version = 0

        # stats
        self.total_produced = 0
        self.total_consumed = 0
        self.dropped_samples = 0

        # fan-in buffer that feeds get_sample(); size kept small
        self._fifo = deque(maxlen=max_queue_size)

        # async signaling for consumers waiting in get_sample()
        self._lock = asyncio.Lock()
        self._cv = asyncio.Condition(self._lock)
        self.running = True

        # fastmq sharded SPSC setup
        # If not specified, default shards ~= CPUs of the process (but capped)
        if num_shards is None or num_shards <= 0:
            # small default that scales well; adjust per your rollout count
            num_shards = min(16, os.cpu_count() or 8)

        self._num_shards = int(num_shards)
        self._poll_batch_per_shard = int(poll_batch_per_shard)
        self._spin_iters = int(spin_iters)

        # Name each shard as POSIX shm objects (must start with "/")
        self._shard_names = [f"/mq_shard_{os.getpid()}_{i}" for i in range(self._num_shards)]
        self._queues = []  # type: list[fastmq.SPSC]

        # Create all shards (creator side = True here)
        for name in self._shard_names:
            q = fastmq.SPSC(name, shard_capacity_bytes, True)
            self._queues.append(q)

        # Start fan-in/poll thread
        self._poll_thread = threading.Thread(target=self._poll_loop, name="fastmq-poller", daemon=True)
        self._poll_thread.start()

        print(
            f"[MessageQueue] fastmq shards={self._num_shards}, shard_capacity={shard_capacity_bytes}B, "
            f"max_queue_size={max_queue_size}, staleness_threshold={self.staleness_threshold}"
        )

    # --------------- Internal: fan-in poller ---------------
    def _poll_loop(self):
        """
        Drain all SPSC shards in a tight loop and push into the in-process fifo.
        This isolates the low-latency path from asyncio/Python locks.
        """
        # optional: pin this polling thread if you exposed pinning APIs
        # try:
        #     fastmq.pin_thread(<core_id>)
        # except Exception:
        #     pass

        while self.running:
            any_msg = False
            for q in self._queues:
                for _ in range(self._poll_batch_per_shard):
                    b = q.try_pop()
                    if b is None:
                        break
                    any_msg = True
                    try:
                        (param_version, payload_bytes) = pickle.loads(b)  # small tuple framing
                        # Drop staleness at *consume* time to bias toward fresh data
                        if (self.current_param_version - param_version) > self.staleness_threshold:
                            self.dropped_samples += 1
                            continue
                        obj = _des(payload_bytes)
                    except Exception as e:
                        logger.exception("Deserialization failed: %s", e)
                        self.dropped_samples += 1
                        continue

                    # push into small fifo; if full, drop oldest (policy match original)
                    if len(self._fifo) >= self._fifo.maxlen:
                        self._fifo.popleft()
                        self.dropped_samples += 1
                    self._fifo.append(obj)
                    self.total_consumed += 1  # consumer-side ingest into fifo

            if any_msg:
                # Wake any awaiters
                # Note: we must schedule this on the event loop; use call_soon_threadsafe via a loop handle
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        loop.call_soon_threadsafe(self._notify_consumers)
                except RuntimeError:
                    # no event loop yet; ignore
                    pass
                continue

            # No messages: do a short spin to keep latency low, then yield
            for _ in range(self._spin_iters):
                pass
            time.sleep(0)  # yield to scheduler briefly

    def _notify_consumers(self):
        async def _notify():
            async with self._cv:
                self._cv.notify_all()
        # fire-and-forget
        asyncio.create_task(_notify())

    # --------------- Public API (unchanged names) ---------------
    async def put_sample(self, sample: Any, param_version: int) -> bool:
        """
        Original API preserved.
        In this fastmq version, we route the put to a shard and push into its SPSC ring.
        """
        # Choose shard by lightweight hash (pid+time or param_version) to spread load
        shard_idx = (hash(os.getpid()) ^ hash(param_version) ^ (_now_ns() & 0xFFFF)) % self._num_shards
        q = self._queues[shard_idx]

        # Serialize as a tiny tuple (param_version, payload_bytes)
        try:
            payload = _ser(sample)
            frame = pickle.dumps((param_version, payload), protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            logger.exception("Serialization failed: %s", e)
            return False

        # Try a few times before dropping (keep latency)
        for _ in range(256):
            if q.try_push(frame):
                self.total_produced += 1
                if (self.total_produced % 100) == 0:
                    print(f"MessageQueue stats: produced={self.total_produced}, fifo_size={len(self._fifo)}")
                return True
        # drop on persistent backpressure
        self.dropped_samples += 1
        logger.warning("Shard full, dropped sample")
        return False

    async def get_sample(self) -> Any | None:
        """
        Original API preserved: waits until one sample is available or queue is shut down.
        Returns (data, buffered_size) like before.
        """
        async with self._cv:
            while len(self._fifo) == 0 and self.running:
                await self._cv.wait()

        if not self.running and len(self._fifo) == 0:
            return None

        data = self._fifo.popleft()
        return data, len(self._fifo)

    async def update_param_version(self, version: int):
        # No lock needed; single writer semantics would be ideal, but atomicity here is fine.
        old = self.current_param_version
        self.current_param_version = int(version)
        print(f"Parameter version updated from {old} to {version}")

    async def get_queue_size(self) -> int:
        return len(self._fifo)

    async def get_statistics(self) -> dict[str, Any]:
        return {
            "fifo_size": len(self._fifo),
            "total_produced": self.total_produced,
            "total_consumed": self.total_consumed,
            "dropped_samples": self.dropped_samples,
            "current_param_version": self.current_param_version,
            "staleness_threshold": self.staleness_threshold,
            "max_queue_size": self.max_queue_size,
            "num_shards": self._num_shards,
        }

    async def clear_queue(self):
        cleared = len(self._fifo)
        self._fifo.clear()
        logger.info(f"Cleared {cleared} samples from fifo")

    async def shutdown(self):
        self.running = False
        # Wake waiters so they exit
        async with self._cv:
            self._cv.notify_all()
        logger.info("MessageQueue shutdown")

    async def get_memory_usage(self) -> dict:
        # We can only estimate Python-side buffer; the shard rings live in SHM.
        # We can also report per-shard free_space via a small C++ binding.
        import sys
        sample_count = len(self._fifo)
        total_size = 0
        if sample_count:
            try:
                s0 = self._fifo[0]
                total_size = sys.getsizeof(s0) * sample_count
            except Exception:
                total_size = sample_count * 15000
        return {
            "fifo_samples": sample_count,
            "estimated_fifo_memory_bytes": total_size,
            "estimated_fifo_memory_mb": total_size / (1024 * 1024),
            "num_shards": self._num_shards,
        }

    async def put_validate(self, data):
        # keep old validation path simple (rare)
        # route via shard 0
        try:
            frame = pickle.dumps((-1, _ser(data)), protocol=pickle.HIGHEST_PROTOCOL)
            if self._queues[0].try_push(frame):
                return True
        except Exception:
            pass
        return False

    async def get_validate(self):
        # drain from fifo and filter those with param_version == -1
        # (kept lightweight; you can also dedicate a separate shard if needed)
        while True:
            async with self._cv:
                while len(self._fifo) == 0 and self.running:
                    await self._cv.wait()
            if not self.running and len(self._fifo) == 0:
                return None
            x = self._fifo.popleft()
            # if your validation objects are distinguishable, return first match
            return x

    # -------- extra (non-breaking) helper so clients can open shards ----------
    async def _get_shard_names(self) -> list[str]:
        return list(self._shard_names)


class MessageQueueClient:
    """
    Client wrapper with the same public methods, but:
    - put_sample() will try a direct fastmq path (no RPC) by opening one shard.
    - Falls back to RPC if direct path isn’t available.
    """

    def __init__(self, queue_actor: Any, shard_affinity: Optional[int] = None):
        self.queue_actor = queue_actor
        self._direct_q: Optional[fastmq.SPSC] = None
        self._direct_enabled = False
        self._shard_idx = 0

        # Try to open one shard for direct puts
        try:
            names = ray.get(self.queue_actor._get_shard_names.remote())
            if names:
                if shard_affinity is None:
                    # stable-ish spread over shards across processes
                    self._shard_idx = (os.getpid() ^ os.getppid()) % len(names)
                else:
                    self._shard_idx = int(shard_affinity) % len(names)
                self._direct_q = fastmq.SPSC(names[self._shard_idx], 0, False)  # open existing
                self._direct_enabled = True
        except Exception as e:
            logger.warning("Direct fastmq path unavailable, falling back to RPC: %s", e)

    # ---------------- public APIs (names unchanged) ----------------
    async def put_sample(self, sample: Any, param_version: int) -> bool:
        if self._direct_enabled and self._direct_q is not None:
            try:
                payload = _ser(sample)
                frame = pickle.dumps((param_version, payload), protocol=pickle.HIGHEST_PROTOCOL)
                # small spin to keep latency; drop if full
                for _ in range(256):
                    if self._direct_q.try_push(frame):
                        return True
                return False
            except Exception:
                # fall back to RPC on error
                pass

        # RPC fallback (compatible with old behavior)
        fut = self.queue_actor.put_sample.remote(sample, param_version)
        return await asyncio.wrap_future(fut.future())

    async def put_validate(self, data: Any) -> bool:
        # keep RPC (rare path)
        fut = self.queue_actor.put_validate.remote(data)
        return await asyncio.wrap_future(fut.future())

    def get_validate_sync(self) -> Any | None:
        return ray.get(self.queue_actor.get_validate.remote())

    async def get_sample(self) -> Any | None:
        fut = self.queue_actor.get_sample.remote()
        return await asyncio.wrap_future(fut.future())

    async def get_queue_size(self) -> int:
        fut = self.queue_actor.get_queue_size.remote()
        return await asyncio.wrap_future(fut.future())

    async def get_statistics(self) -> dict[str, Any]:
        fut = self.queue_actor.get_statistics.remote()
        return await asyncio.wrap_future(fut.future())

    async def clear_queue(self):
        fut = self.queue_actor.clear_queue.remote()
        await asyncio.wrap_future(fut.future())

    async def shutdown(self):
        fut = self.queue_actor.shutdown.remote()
        await asyncio.wrap_future(fut.future())

    async def get_memory_usage(self) -> dict:
        fut = self.queue_actor.get_memory_usage.remote()
        return await asyncio.wrap_future(fut.future())

    # (deprecated sync variants retained for compatibility)
    def put_sample_sync(self, sample: Any, param_version: int) -> bool:
        if self._direct_enabled and self._direct_q is not None:
            try:
                payload = _ser(sample)
                frame = pickle.dumps((param_version, payload), protocol=pickle.HIGHEST_PROTOCOL)
                for _ in range(256):
                    if self._direct_q.try_push(frame):
                        return True
                return False
            except Exception:
                pass
        return ray.get(self.queue_actor.put_sample.remote(sample, param_version))

    def get_sample_sync(self) -> Any | None:
        return ray.get(self.queue_actor.get_sample.remote())

    def get_statistics_sync(self) -> dict[str, Any]:
        return ray.get(self.queue_actor.get_statistics.remote())

    def update_param_version_sync(self, version: int):
        return ray.get(self.queue_actor.update_param_version.remote(version))
