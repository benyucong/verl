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
import time
from collections import deque
from typing import Any

import ray
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


# max_concurrency must exceed the max number of concurrently-blocked producers (in-flight agent
# loops awaiting put_sample under non-dropping backpressure) PLUS consumer headroom: a blocked
# put_sample holds an asyncio-actor slot while it waits, so if every slot were a blocked producer
# the consumer's get_sample could never be scheduled to drain -> deadlock. 256 >> rollout in-flight.
@ray.remote(num_cpus=2, max_concurrency=256)
class MessageQueue:
    """
    Simplified Ray-based asynchronous message queue for communication between Rollouter and Trainer
    """

    def __init__(self, config: DictConfig, max_queue_size: int = 1000):
        self.config = config
        if max_queue_size is None:
            raise ValueError(f"max_queue_size cannot be None, got: {max_queue_size}")
        self.max_queue_size = int(max_queue_size)
        self.queue = deque(maxlen=self.max_queue_size)

        self.val_queue = deque()

        # Asyncio for message handling
        self.running = True

        # async safe
        self._lock = asyncio.Lock()
        self._consumer_condition = asyncio.Condition(self._lock)
        self._producer_condition = asyncio.Condition(self._lock)

        # statistic message
        self.total_produced = 0
        self.total_consumed = 0
        self.dropped_samples = 0

        # H-ACC-SPAN bounded NON-DROPPING backpressure: when enabled, a full queue BLOCKS the
        # producer until the consumer frees space instead of silently evicting the oldest span.
        # A generous timeout preserves liveness (e.g. dead consumer) with an EXPLICIT, counted drop
        # rather than a silent one. Off by default -> legacy popleft-drop behavior is unchanged.
        self.block_on_full = os.environ.get("OPD_QUEUE_BLOCK_ON_FULL", "0") not in ("0", "", "false", "False")
        self.block_timeout_s = float(os.environ.get("OPD_QUEUE_BLOCK_TIMEOUT_S", "600"))
        self.producer_blocked_time_s = 0.0
        self.producer_block_events = 0
        self.blocked_timeout_drops = 0
        self.max_queue_depth = 0
        self._depth_samples: deque = deque(maxlen=4096)  # for p50/p95 depth

        print(f"[MessageQueue] initialized max_queue_size={max_queue_size} block_on_full={self.block_on_full} timeout={self.block_timeout_s}s")

    async def put_sample(self, sample: Any) -> bool:
        """
        Put a batch sample into the queue

        Args:
            sample: Sample data

        Returns:
            bool: Whether the sample was successfully put into the queue
        """
        async with self._lock:
            is_drop = False
            if len(self.queue) >= self.max_queue_size:
                if self.block_on_full:
                    # Bounded NON-dropping backpressure: wait for the consumer to free space.
                    blocked_start = time.monotonic()
                    self.producer_block_events += 1
                    while len(self.queue) >= self.max_queue_size and self.running:
                        try:
                            await asyncio.wait_for(self._producer_condition.wait(), timeout=self.block_timeout_s)
                        except asyncio.TimeoutError:
                            # Liveness guard: explicit, counted drop (NOT silent) after a long stall.
                            self.queue.popleft()
                            self.dropped_samples += 1
                            self.blocked_timeout_drops += 1
                            logger.error(
                                "Queue block timed out after %.0fs (consumer stalled?); EXPLICIT drop #%d",
                                self.block_timeout_s, self.blocked_timeout_drops,
                            )
                            is_drop = True
                            break
                    self.producer_blocked_time_s += time.monotonic() - blocked_start
                else:
                    # Legacy behavior: silently evict the oldest sample.
                    self.queue.popleft()
                    self.dropped_samples += 1
                    is_drop = True
                    logger.warning("Queue full, dropped sample")
            self.queue.append(sample)
            self.total_produced += 1
            depth = len(self.queue)
            self.max_queue_depth = max(self.max_queue_depth, depth)
            self._depth_samples.append(depth)

            # Notify waiting consumers
            self._consumer_condition.notify_all()

            if self.total_produced % 100 == 0:
                print(f"MessageQueue stats: produced={self.total_produced}, queue_size={depth}, "
                      f"blocked_s={self.producer_blocked_time_s:.1f}, timeout_drops={self.blocked_timeout_drops}")
            if is_drop:
                return False
            return True

    async def get_sample(self) -> Any | None:
        """
        Get a single sample from the queue, wait until one is available

        Returns:
            Any: Single sample data or None if queue is closed
        """
        async with self._lock:
            while len(self.queue) == 0 and self.running:
                await self._consumer_condition.wait()

            # If queue is closed and empty, return None
            if not self.running and len(self.queue) == 0:
                return None

            # Get one sample
            data = self.queue.popleft()
            self.total_consumed += 1
            # Space freed -> wake a blocked producer (non-dropping backpressure).
            self._producer_condition.notify(1)
            return data, len(self.queue)

    async def get_queue_size(self) -> int:
        """Get current queue length"""
        async with self._lock:
            return len(self.queue)

    async def get_statistics(self) -> dict[str, Any]:
        """Get queue statistics"""
        async with self._lock:
            d = sorted(self._depth_samples)
            def _pct(p):
                return d[min(len(d) - 1, int(p * len(d)))] if d else 0
            return {
                "queue_size": len(self.queue),
                "total_produced": self.total_produced,
                "total_consumed": self.total_consumed,
                "dropped_samples": self.dropped_samples,
                "max_queue_size": self.max_queue_size,
                # H-ACC-SPAN backpressure metrics
                "block_on_full": self.block_on_full,
                "producer_blocked_time_s": self.producer_blocked_time_s,
                "producer_block_events": self.producer_block_events,
                "blocked_timeout_drops": self.blocked_timeout_drops,
                "queue_depth_p50": _pct(0.50),
                "queue_depth_p95": _pct(0.95),
                "queue_depth_max": self.max_queue_depth,
            }

    async def clear_queue(self):
        """Clear the queue"""
        async with self._lock:
            cleared_count = len(self.queue)
            self.queue.clear()
            logger.info(f"Cleared {cleared_count} samples from queue")

    async def shutdown(self):
        """Shutdown the message queue"""
        async with self._lock:
            self.running = False
            # Notify all waiting coroutines (consumers AND blocked producers) so they can exit.
            self._consumer_condition.notify_all()
            self._producer_condition.notify_all()
        logger.info("MessageQueue shutdown")

    async def get_memory_usage(self) -> dict:
        """Get memory usage statistics"""
        async with self._lock:
            # Estimate memory usage of samples in queue
            import sys

            total_size = 0
            sample_count = len(self.queue)

            if sample_count > 0:
                # Estimate size of a single sample (simplified estimation)
                sample = list(self.queue)[0]
                try:
                    sample_size = sys.getsizeof(sample)
                    # Since we now store RolloutSample directly, estimate based on its components
                    if hasattr(sample, "original_batch_dict") and sample.original_batch_dict:
                        # Estimate batch data size
                        batch_data = sample.original_batch_dict.get("batch", {})
                        sample_size += len(batch_data) * 1000  # Roughly estimate 1KB per batch entry
                    if hasattr(sample, "agent_loop_output"):
                        # Estimate AgentLoopOutput size
                        sample_size += 5000  # Roughly estimate 5KB for AgentLoopOutput
                    total_size = sample_size * sample_count
                except Exception:
                    total_size = sample_count * 15000  # Roughly estimate 15KB per RolloutSample

            return {
                "queue_samples": sample_count,
                "estimated_memory_bytes": total_size,
                "estimated_memory_mb": total_size / (1024 * 1024),
            }

    async def put_validate(self, data):
        async with self._lock:
            self.val_queue.append(data)

    async def get_validate(self):
        async with self._lock:
            if self.val_queue:
                return self.val_queue.popleft()
            else:
                return None


class MessageQueueClient:
    """Asyncio-compatible MessageQueue client for communicating with MessageQueue Actor"""

    def __init__(self, queue_actor: Any):
        self.queue_actor = queue_actor

    async def put_sample(self, sample: Any) -> bool:
        """Put batch into queue (async)"""
        future = self.queue_actor.put_sample.remote(sample)
        return await asyncio.wrap_future(future.future())

    async def put_validate(self, data: Any) -> bool:
        future = self.queue_actor.put_validate.remote(data)
        return await asyncio.wrap_future(future.future())

    def get_validate_sync(self) -> Any | None:
        return ray.get(self.queue_actor.get_validate.remote())

    async def get_sample(self) -> Any | None:
        """Get single sample from queue, wait until one is available (async)"""
        future = self.queue_actor.get_sample.remote()
        return await asyncio.wrap_future(future.future())

    async def get_queue_size(self) -> int:
        """Get queue size (async)"""
        future = self.queue_actor.get_queue_size.remote()
        return await asyncio.wrap_future(future.future())

    async def get_statistics(self) -> dict[str, Any]:
        """Get statistics (async)"""
        future = self.queue_actor.get_statistics.remote()
        return await asyncio.wrap_future(future.future())

    async def clear_queue(self):
        """Clear queue (async)"""
        future = self.queue_actor.clear_queue.remote()
        await asyncio.wrap_future(future.future())

    async def shutdown(self):
        """Shutdown queue (async)"""
        future = self.queue_actor.shutdown.remote()
        await asyncio.wrap_future(future.future())

    async def get_memory_usage(self) -> dict:
        """Get memory usage statistics (async)"""
        future = self.queue_actor.get_memory_usage.remote()
        return await asyncio.wrap_future(future.future())

    def get_sample_sync(self) -> Any | None:
        """Get single sample from queue (sync - deprecated, use get_sample instead)"""
        return ray.get(self.queue_actor.get_sample.remote())

    def get_statistics_sync(self) -> dict[str, Any]:
        """Get statistics (sync - deprecated, use get_statistics instead)"""
        return ray.get(self.queue_actor.get_statistics.remote())
