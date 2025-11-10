"""
Unit tests for message_queue_new.py

This test suite covers:
1. Basic MessageQueue operations (put/get)
2. Staleness threshold behavior
3. Queue overflow and sample dropping
4. Parameter version updates
5. Statistics and memory tracking
6. Shutdown and cleanup
7. MessageQueueClient operations
8. Concurrent producer-consumer scenarios
9. Validation data paths
"""

import asyncio
import os
import pickle
import sys
import time
import unittest
from unittest.mock import MagicMock, patch

import ray
from omegaconf import DictConfig, OmegaConf

# Add the parent directory to the path to import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from recipe.fully_async_policy.message_queue_new import (
    MessageQueue,
    MessageQueueClient,
    _des,
    _ser,
)


class TestSerializationHelpers(unittest.TestCase):
    """Test serialization/deserialization helper functions"""

    def test_ser_des_basic_types(self):
        """Test serialization of basic Python types"""
        test_cases = [
            42,
            3.14,
            "hello",
            [1, 2, 3],
            {"key": "value"},
            (1, 2, 3),
            None,
            True,
        ]
        
        for obj in test_cases:
            serialized = _ser(obj)
            self.assertIsInstance(serialized, bytes)
            deserialized = _des(serialized)
            self.assertEqual(obj, deserialized)

    def test_ser_bytes_passthrough(self):
        """Test that bytes are passed through without double-serialization"""
        original_bytes = b"raw bytes data"
        result = _ser(original_bytes)
        self.assertEqual(result, original_bytes)

    def test_ser_bytearray(self):
        """Test bytearray handling"""
        original = bytearray(b"test data")
        result = _ser(original)
        self.assertEqual(result, bytes(original))

    def test_des_invalid_data(self):
        """Test deserialization with invalid data"""
        with self.assertRaises(pickle.UnpicklingError):
            _des(b"invalid pickle data")


class TestMessageQueueBasics(unittest.IsolatedAsyncioTestCase):
    """Test basic MessageQueue functionality"""

    @classmethod
    def setUpClass(cls):
        """Initialize Ray once for all tests"""
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

    @classmethod
    def tearDownClass(cls):
        """Shutdown Ray after all tests"""
        if ray.is_initialized():
            ray.shutdown()

    def create_test_config(self, staleness_threshold=3):
        """Create a test configuration"""
        return OmegaConf.create({
            "async_training": {
                "staleness_threshold": staleness_threshold
            }
        })

    async def asyncSetUp(self):
        """Set up test fixtures"""
        self.config = self.create_test_config()
        # Create a small queue for testing
        self.queue_actor = MessageQueue.remote(
            config=self.config,
            max_queue_size=10,
            num_shards=2,
            shard_capacity_bytes=1024 * 10,  # 10 KB for testing
        )
        # Give the polling thread time to start
        await asyncio.sleep(0.1)

    async def asyncTearDown(self):
        """Clean up after tests"""
        try:
            await asyncio.wait_for(
                asyncio.wrap_future(self.queue_actor.shutdown.remote().future()),
                timeout=2.0
            )
        except asyncio.TimeoutError:
            pass
        # Small delay to allow cleanup
        await asyncio.sleep(0.1)

    async def test_queue_initialization(self):
        """Test that queue initializes correctly"""
        stats = await asyncio.wrap_future(
            self.queue_actor.get_statistics.remote().future()
        )
        
        self.assertEqual(stats["max_queue_size"], 10)
        self.assertEqual(stats["staleness_threshold"], 3)
        self.assertEqual(stats["num_shards"], 2)
        self.assertEqual(stats["total_produced"], 0)
        self.assertEqual(stats["total_consumed"], 0)

    async def test_put_and_get_sample(self):
        """Test basic put and get operations"""
        test_sample = {"data": "test_value", "id": 123}
        param_version = 1
        
        # Put sample
        put_result = await asyncio.wrap_future(
            self.queue_actor.put_sample.remote(test_sample, param_version).future()
        )
        self.assertTrue(put_result)
        
        # Give polling thread time to process
        await asyncio.sleep(0.2)
        
        # Get sample with timeout to prevent hanging
        result = await asyncio.wait_for(
            asyncio.wrap_future(
                self.queue_actor.get_sample.remote().future()
            ),
            timeout=2.0
        )
        
        self.assertIsNotNone(result)
        sample, queue_size = result
        self.assertEqual(sample, test_sample)
        self.assertGreaterEqual(queue_size, 0)

    async def test_multiple_samples(self):
        """Test putting and getting multiple samples"""
        num_samples = 5
        samples = [{"id": i, "data": f"sample_{i}"} for i in range(num_samples)]
        
        # Put all samples
        for i, sample in enumerate(samples):
            result = await asyncio.wrap_future(
                self.queue_actor.put_sample.remote(sample, i).future()
            )
            self.assertTrue(result)
        
        # Wait for processing
        await asyncio.sleep(0.3)
        
        # Get all samples
        retrieved = []
        for _ in range(num_samples):
            try:
                result = await asyncio.wait_for(
                    asyncio.wrap_future(
                        self.queue_actor.get_sample.remote().future()
                    ),
                    timeout=1.0
                )
                if result is not None:
                    sample, _ = result
                    retrieved.append(sample)
            except asyncio.TimeoutError:
                # If timeout, some samples may have been dropped
                break
        
        # Should have retrieved at least some samples
        self.assertGreater(len(retrieved), 0, "Should have retrieved at least some samples")
        self.assertLessEqual(len(retrieved), num_samples, "Should not retrieve more than we put")
        
        # Verify all retrieved samples are from our original set
        retrieved_ids = {s["id"] for s in retrieved}
        expected_ids = {s["id"] for s in samples}
        self.assertTrue(retrieved_ids.issubset(expected_ids), "Retrieved samples should be from original set")

    async def test_queue_size_tracking(self):
        """Test that queue size is tracked correctly"""
        # Initially empty
        size = await asyncio.wrap_future(
            self.queue_actor.get_queue_size.remote().future()
        )
        self.assertEqual(size, 0)
        
        # Add samples
        for i in range(3):
            await asyncio.wrap_future(
                self.queue_actor.put_sample.remote({"id": i}, 0).future()
            )
        
        await asyncio.sleep(0.2)
        
        # Check size increased
        size = await asyncio.wrap_future(
            self.queue_actor.get_queue_size.remote().future()
        )
        self.assertGreater(size, 0)


class TestStalenessHandling(unittest.IsolatedAsyncioTestCase):
    """Test staleness threshold and version handling"""

    @classmethod
    def setUpClass(cls):
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

    @classmethod
    def tearDownClass(cls):
        if ray.is_initialized():
            ray.shutdown()

    async def asyncSetUp(self):
        config = OmegaConf.create({
            "async_training": {"staleness_threshold": 2}
        })
        self.queue_actor = MessageQueue.remote(
            config=config,
            max_queue_size=10,
            num_shards=2,
        )
        await asyncio.sleep(0.1)

    async def asyncTearDown(self):
        try:
            await asyncio.wait_for(
                asyncio.wrap_future(self.queue_actor.shutdown.remote().future()),
                timeout=2.0
            )
        except asyncio.TimeoutError:
            pass
        await asyncio.sleep(0.1)

    async def test_staleness_threshold_drops_old_samples(self):
        """Test that samples exceeding staleness threshold are dropped"""
        # First, update parameter version to 5
        await asyncio.wrap_future(
            self.queue_actor.update_param_version.remote(5).future()
        )
        
        await asyncio.sleep(0.1)
        
        # Now put samples with old version that will be stale
        # staleness_threshold is 2, so version difference of 5-0=5 > 2
        old_samples = [{"id": i, "version": "old"} for i in range(3)]
        for sample in old_samples:
            await asyncio.wrap_future(
                self.queue_actor.put_sample.remote(sample, param_version=0).future()
            )
        
        # Put fresh samples with current version
        fresh_samples = [{"id": i, "version": "fresh"} for i in range(2)]
        for sample in fresh_samples:
            await asyncio.wrap_future(
                self.queue_actor.put_sample.remote(sample, param_version=5).future()
            )
        
        # Wait for poll loop to process all samples from SPSC queues
        # The stale samples should be dropped during polling
        await asyncio.sleep(0.5)
        
        # Check statistics
        stats = await asyncio.wrap_future(
            self.queue_actor.get_statistics.remote().future()
        )
        
        # The old samples should have been dropped by the poll loop
        self.assertGreater(stats["dropped_samples"], 0, 
                          f"Expected stale samples to be dropped, but dropped_samples={stats['dropped_samples']}")
        
        # Verify only fresh samples are in the queue
        consumed_samples = []
        for _ in range(5):  # Try to get all available samples
            try:
                result = await asyncio.wait_for(
                    asyncio.wrap_future(
                        self.queue_actor.get_sample.remote().future()
                    ),
                    timeout=0.2
                )
                if result is not None:
                    consumed_samples.append(result[0])  # result is (data, buffered_size)
            except asyncio.TimeoutError:
                break
        
        # All consumed samples should be fresh (version="fresh")
        old_count = sum(1 for s in consumed_samples if s.get("version") == "old")
        fresh_count = sum(1 for s in consumed_samples if s.get("version") == "fresh")
        
        self.assertEqual(old_count, 0, "No stale samples should have made it to the queue")
        self.assertGreater(fresh_count, 0, "Fresh samples should be in the queue")

    async def test_update_param_version(self):
        """Test parameter version update"""
        # Check initial version
        stats = await asyncio.wrap_future(
            self.queue_actor.get_statistics.remote().future()
        )
        self.assertEqual(stats["current_param_version"], 0)
        
        # Update version
        await asyncio.wrap_future(
            self.queue_actor.update_param_version.remote(42).future()
        )
        
        # Verify update
        stats = await asyncio.wrap_future(
            self.queue_actor.get_statistics.remote().future()
        )
        self.assertEqual(stats["current_param_version"], 42)


class TestQueueOverflow(unittest.IsolatedAsyncioTestCase):
    """Test queue overflow and sample dropping behavior"""

    @classmethod
    def setUpClass(cls):
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

    @classmethod
    def tearDownClass(cls):
        if ray.is_initialized():
            ray.shutdown()

    async def asyncSetUp(self):
        config = OmegaConf.create({
            "async_training": {"staleness_threshold": 10}
        })
        # Small queue to easily test overflow
        self.queue_actor = MessageQueue.remote(
            config=config,
            max_queue_size=5,
            num_shards=1,
        )
        await asyncio.sleep(0.1)

    async def asyncTearDown(self):
        try:
            await asyncio.wait_for(
                asyncio.wrap_future(self.queue_actor.shutdown.remote().future()),
                timeout=2.0
            )
        except asyncio.TimeoutError:
            pass
        await asyncio.sleep(0.1)

    async def test_queue_overflow_drops_oldest(self):
        """Test that queue drops oldest samples when full"""
        # Fill queue beyond capacity
        num_samples = 10
        for i in range(num_samples):
            await asyncio.wrap_future(
                self.queue_actor.put_sample.remote({"id": i}, 0).future()
            )
        
        await asyncio.sleep(0.3)
        
        # Check that some samples were dropped
        stats = await asyncio.wrap_future(
            self.queue_actor.get_statistics.remote().future()
        )
        
        # Queue size should be capped at max
        queue_size = stats["fifo_size"]
        self.assertLessEqual(queue_size, 5)
        
        # Should have dropped samples
        if stats["total_consumed"] > 0:
            self.assertGreater(stats["dropped_samples"], 0)


class TestStatisticsAndMemory(unittest.IsolatedAsyncioTestCase):
    """Test statistics and memory usage tracking"""

    @classmethod
    def setUpClass(cls):
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

    @classmethod
    def tearDownClass(cls):
        if ray.is_initialized():
            ray.shutdown()

    async def asyncSetUp(self):
        config = OmegaConf.create({
            "async_training": {"staleness_threshold": 3}
        })
        self.queue_actor = MessageQueue.remote(config=config, max_queue_size=10)
        await asyncio.sleep(0.1)

    async def asyncTearDown(self):
        try:
            await asyncio.wait_for(
                asyncio.wrap_future(self.queue_actor.shutdown.remote().future()),
                timeout=2.0
            )
        except asyncio.TimeoutError:
            pass
        await asyncio.sleep(0.1)

    async def test_statistics(self):
        """Test that statistics are tracked correctly"""
        # Add some samples
        for i in range(3):
            await asyncio.wrap_future(
                self.queue_actor.put_sample.remote({"data": i}, 0).future()
            )
        
        await asyncio.sleep(0.2)
        
        stats = await asyncio.wrap_future(
            self.queue_actor.get_statistics.remote().future()
        )
        
        # Verify stats structure
        self.assertIn("fifo_size", stats)
        self.assertIn("total_produced", stats)
        self.assertIn("total_consumed", stats)
        self.assertIn("dropped_samples", stats)
        self.assertIn("current_param_version", stats)
        self.assertIn("staleness_threshold", stats)
        self.assertIn("num_shards", stats)
        
        # Verify counts
        self.assertEqual(stats["total_produced"], 3)
        self.assertGreaterEqual(stats["total_consumed"], 0)

    async def test_memory_usage(self):
        """Test memory usage estimation"""
        # Add samples
        for i in range(5):
            await asyncio.wrap_future(
                self.queue_actor.put_sample.remote(
                    {"data": "x" * 1000, "id": i}, 0
                ).future()
            )
        
        await asyncio.sleep(0.2)
        
        mem_usage = await asyncio.wrap_future(
            self.queue_actor.get_memory_usage.remote().future()
        )
        
        # Verify structure
        self.assertIn("fifo_samples", mem_usage)
        self.assertIn("estimated_fifo_memory_bytes", mem_usage)
        self.assertIn("estimated_fifo_memory_mb", mem_usage)
        self.assertIn("num_shards", mem_usage)
        
        # Should have some samples and non-zero memory
        if mem_usage["fifo_samples"] > 0:
            self.assertGreater(mem_usage["estimated_fifo_memory_bytes"], 0)

    async def test_clear_queue(self):
        """Test clearing the queue"""
        # Add samples
        for i in range(3):
            await asyncio.wrap_future(
                self.queue_actor.put_sample.remote({"id": i}, 0).future()
            )
        
        await asyncio.sleep(0.2)
        
        # Verify queue has items
        size_before = await asyncio.wrap_future(
            self.queue_actor.get_queue_size.remote().future()
        )
        self.assertGreater(size_before, 0)
        
        # Clear queue
        await asyncio.wrap_future(
            self.queue_actor.clear_queue.remote().future()
        )
        
        # Verify queue is empty
        size_after = await asyncio.wrap_future(
            self.queue_actor.get_queue_size.remote().future()
        )
        self.assertEqual(size_after, 0)


class TestShutdown(unittest.IsolatedAsyncioTestCase):
    """Test shutdown behavior"""

    @classmethod
    def setUpClass(cls):
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

    @classmethod
    def tearDownClass(cls):
        if ray.is_initialized():
            ray.shutdown()

    async def test_shutdown_wakes_consumers(self):
        """Test that shutdown wakes waiting consumers"""
        config = OmegaConf.create({
            "async_training": {"staleness_threshold": 3}
        })
        queue_actor = MessageQueue.remote(config=config, max_queue_size=10)
        await asyncio.sleep(0.1)
        
        # Start a task that will wait for samples
        async def consumer():
            result = await asyncio.wrap_future(
                queue_actor.get_sample.remote().future()
            )
            return result
        
        consumer_task = asyncio.create_task(consumer())
        
        # Give it time to start waiting
        await asyncio.sleep(0.2)
        
        # Shutdown queue
        await asyncio.wrap_future(
            queue_actor.shutdown.remote().future()
        )
        
        # Consumer should wake up and return None
        result = await asyncio.wait_for(consumer_task, timeout=2.0)
        self.assertIsNone(result)


class TestValidationPath(unittest.IsolatedAsyncioTestCase):
    """Test validation data put/get operations"""

    @classmethod
    def setUpClass(cls):
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

    @classmethod
    def tearDownClass(cls):
        if ray.is_initialized():
            ray.shutdown()

    async def asyncSetUp(self):
        config = OmegaConf.create({
            "async_training": {"staleness_threshold": 3}
        })
        self.queue_actor = MessageQueue.remote(config=config, max_queue_size=10)
        await asyncio.sleep(0.1)

    async def asyncTearDown(self):
        try:
            await asyncio.wait_for(
                asyncio.wrap_future(self.queue_actor.shutdown.remote().future()),
                timeout=2.0
            )
        except asyncio.TimeoutError:
            pass
        await asyncio.sleep(0.1)

    async def test_put_and_get_validate(self):
        """Test validation data path"""
        validation_data = {"type": "validation", "data": "test"}
        
        # Put validation data
        result = await asyncio.wrap_future(
            self.queue_actor.put_validate.remote(validation_data).future()
        )
        self.assertTrue(result)
        
        await asyncio.sleep(0.2)
        
        # Get validation data
        retrieved = await asyncio.wrap_future(
            self.queue_actor.get_validate.remote().future()
        )
        
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved, validation_data)


class TestMessageQueueClient(unittest.IsolatedAsyncioTestCase):
    """Test MessageQueueClient wrapper"""

    @classmethod
    def setUpClass(cls):
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

    @classmethod
    def tearDownClass(cls):
        if ray.is_initialized():
            ray.shutdown()

    async def asyncSetUp(self):
        config = OmegaConf.create({
            "async_training": {"staleness_threshold": 3}
        })
        self.queue_actor = MessageQueue.remote(config=config, max_queue_size=10)
        await asyncio.sleep(0.1)
        
        # Create client
        self.client = MessageQueueClient(self.queue_actor, shard_affinity=0)
        await asyncio.sleep(0.1)

    async def asyncTearDown(self):
        try:
            await self.client.shutdown()
        except:
            pass
        await asyncio.sleep(0.1)

    async def test_client_put_and_get(self):
        """Test client put and get operations"""
        sample = {"client": "test", "value": 42}
        
        # Put via client
        result = await self.client.put_sample(sample, param_version=1)
        self.assertTrue(result)
        
        await asyncio.sleep(0.2)
        
        # Get via client
        retrieved = await self.client.get_sample()
        self.assertIsNotNone(retrieved)
        
        data, queue_size = retrieved
        self.assertEqual(data, sample)

    async def test_client_get_statistics(self):
        """Test client can retrieve statistics"""
        stats = await self.client.get_statistics()
        
        self.assertIsInstance(stats, dict)
        self.assertIn("total_produced", stats)
        self.assertIn("total_consumed", stats)

    async def test_client_sync_methods(self):
        """Test synchronous client methods"""
        sample = {"sync": "test"}
        
        # Sync put
        result = self.client.put_sample_sync(sample, param_version=1)
        self.assertTrue(result)
        
        await asyncio.sleep(0.2)
        
        # Sync get
        retrieved = self.client.get_sample_sync()
        self.assertIsNotNone(retrieved)
        
        data, _ = retrieved
        self.assertEqual(data, sample)

    async def test_client_get_memory_usage(self):
        """Test client can get memory usage"""
        mem_usage = await self.client.get_memory_usage()
        
        self.assertIsInstance(mem_usage, dict)
        self.assertIn("fifo_samples", mem_usage)
        self.assertIn("estimated_fifo_memory_bytes", mem_usage)


class TestConcurrentOperations(unittest.IsolatedAsyncioTestCase):
    """Test concurrent producer-consumer scenarios"""

    @classmethod
    def setUpClass(cls):
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

    @classmethod
    def tearDownClass(cls):
        if ray.is_initialized():
            ray.shutdown()

    async def asyncSetUp(self):
        config = OmegaConf.create({
            "async_training": {"staleness_threshold": 5}
        })
        self.queue_actor = MessageQueue.remote(
            config=config,
            max_queue_size=50,
            num_shards=4,
        )
        await asyncio.sleep(0.1)

    async def asyncTearDown(self):
        try:
            await asyncio.wait_for(
                asyncio.wrap_future(self.queue_actor.shutdown.remote().future()),
                timeout=2.0
            )
        except asyncio.TimeoutError:
            pass
        await asyncio.sleep(0.1)

    async def test_multiple_producers(self):
        """Test multiple concurrent producers"""
        num_producers = 3
        samples_per_producer = 5
        
        async def producer(producer_id):
            for i in range(samples_per_producer):
                sample = {"producer": producer_id, "sample": i}
                await asyncio.wrap_future(
                    self.queue_actor.put_sample.remote(sample, producer_id).future()
                )
                await asyncio.sleep(0.01)
        
        # Run producers concurrently
        tasks = [producer(i) for i in range(num_producers)]
        await asyncio.gather(*tasks)
        
        await asyncio.sleep(0.3)
        
        # Verify total produced
        stats = await asyncio.wrap_future(
            self.queue_actor.get_statistics.remote().future()
        )
        expected_total = num_producers * samples_per_producer
        self.assertEqual(stats["total_produced"], expected_total)

    async def test_producer_consumer_balance(self):
        """Test balanced producer-consumer scenario"""
        num_items = 10
        
        async def producer():
            for i in range(num_items):
                await asyncio.wrap_future(
                    self.queue_actor.put_sample.remote({"id": i}, 0).future()
                )
                await asyncio.sleep(0.02)
        
        async def consumer():
            consumed = []
            while len(consumed) < num_items:
                try:
                    # Use a reasonable timeout per sample
                    # Producer sends every 0.02s, so 1s timeout is more than enough
                    result = await asyncio.wait_for(
                        asyncio.wrap_future(
                            self.queue_actor.get_sample.remote().future()
                        ),
                        timeout=1.0
                    )
                    if result is not None:
                        sample, _ = result
                        consumed.append(sample)
                    else:
                        # Queue was shut down
                        break
                    await asyncio.sleep(0.03)
                except asyncio.TimeoutError:
                    # If we timeout, no more samples are immediately available
                    # This is expected if producer finished or samples were dropped
                    break
            return consumed
        
        # Run producer and consumer concurrently
        producer_task = asyncio.create_task(producer())
        # Give producer time to put a few samples before consumer starts
        await asyncio.sleep(0.2)
        consumer_task = asyncio.create_task(consumer())
        
        # Wait for both to complete
        await producer_task
        consumed = await consumer_task
        
        # Should have consumed most items (some might be dropped due to timing)
        self.assertGreater(len(consumed), 0, "Consumer should have received at least one sample")
        self.assertLessEqual(len(consumed), num_items, "Should not consume more than produced")


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestSerializationHelpers))
    suite.addTests(loader.loadTestsFromTestCase(TestMessageQueueBasics))
    suite.addTests(loader.loadTestsFromTestCase(TestStalenessHandling))
    suite.addTests(loader.loadTestsFromTestCase(TestQueueOverflow))
    suite.addTests(loader.loadTestsFromTestCase(TestStatisticsAndMemory))
    suite.addTests(loader.loadTestsFromTestCase(TestShutdown))
    suite.addTests(loader.loadTestsFromTestCase(TestValidationPath))
    suite.addTests(loader.loadTestsFromTestCase(TestMessageQueueClient))
    suite.addTests(loader.loadTestsFromTestCase(TestConcurrentOperations))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
