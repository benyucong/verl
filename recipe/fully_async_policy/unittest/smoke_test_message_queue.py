#!/usr/bin/env python3
"""
Quick smoke test for message_queue_new.py

This is a simplified test that can be run quickly to verify basic functionality.
For comprehensive testing, use test_message_queue_new.py
"""

import asyncio
import sys
import time

import ray
from omegaconf import OmegaConf

# Add parent directory to path
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from recipe.fully_async_policy.message_queue_new import MessageQueue, MessageQueueClient


async def smoke_test():
    """Run a quick smoke test of the message queue"""
    print("=" * 70)
    print("Message Queue Smoke Test")
    print("=" * 70)
    
    # Initialize Ray
    print("\n[1/7] Initializing Ray...")
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    print("✓ Ray initialized")
    
    # Create config
    print("\n[2/7] Creating configuration...")
    config = OmegaConf.create({
        "async_training": {
            "staleness_threshold": 3
        }
    })
    print("✓ Config created")
    
    # Create message queue
    print("\n[3/7] Creating MessageQueue actor...")
    queue_actor = MessageQueue.remote(
        config=config,
        max_queue_size=10,
        num_shards=2,
        shard_capacity_bytes=1024 * 10,
    )
    await asyncio.sleep(0.2)
    print("✓ MessageQueue created")
    
    # Create client
    print("\n[4/7] Creating MessageQueueClient...")
    client = MessageQueueClient(queue_actor, shard_affinity=0)
    await asyncio.sleep(0.1)
    print("✓ Client created")
    
    # Test put and get
    print("\n[5/7] Testing put/get operations...")
    test_samples = [
        {"id": 0, "data": "first"},
        {"id": 1, "data": "second"},
        {"id": 2, "data": "third"},
    ]
    
    # Put samples
    for i, sample in enumerate(test_samples):
        result = await client.put_sample(sample, param_version=i)
        if not result:
            print(f"✗ Failed to put sample {i}")
            return False
        print(f"  ✓ Put sample {i}")
    
    # Wait for processing
    await asyncio.sleep(0.3)
    
    # Get samples
    retrieved = []
    for i in range(len(test_samples)):
        result = await client.get_sample()
        if result is None:
            print(f"✗ Failed to get sample {i}")
            return False
        sample, queue_size = result
        retrieved.append(sample)
        print(f"  ✓ Got sample {i}: {sample}")
    
    print(f"✓ Put/get test passed ({len(retrieved)}/{len(test_samples)} samples)")
    
    # Test statistics
    print("\n[6/7] Testing statistics...")
    stats = await client.get_statistics()
    print(f"  Total produced: {stats['total_produced']}")
    print(f"  Total consumed: {stats['total_consumed']}")
    print(f"  Dropped samples: {stats['dropped_samples']}")
    print(f"  Queue size: {stats['fifo_size']}")
    print(f"  Num shards: {stats['num_shards']}")
    print("✓ Statistics retrieved")
    
    # Test memory usage
    print("\n[7/7] Testing memory usage...")
    mem_usage = await client.get_memory_usage()
    print(f"  FIFO samples: {mem_usage['fifo_samples']}")
    print(f"  Estimated memory: {mem_usage['estimated_fifo_memory_bytes']} bytes")
    print("✓ Memory usage retrieved")
    
    # Cleanup
    print("\n[Cleanup] Shutting down...")
    await client.shutdown()
    await asyncio.sleep(0.2)
    print("✓ Shutdown complete")
    
    return True


async def stress_test():
    """Quick stress test with multiple samples"""
    print("\n" + "=" * 70)
    print("Quick Stress Test")
    print("=" * 70)
    
    config = OmegaConf.create({"async_training": {"staleness_threshold": 5}})
    queue_actor = MessageQueue.remote(
        config=config,
        max_queue_size=100,
        num_shards=4,
    )
    await asyncio.sleep(0.2)
    
    client = MessageQueueClient(queue_actor)
    await asyncio.sleep(0.1)
    
    # Produce many samples
    num_samples = 50
    print(f"\nProducing {num_samples} samples...")
    start_time = time.time()
    
    for i in range(num_samples):
        await client.put_sample({"id": i, "data": f"sample_{i}"}, param_version=i % 10)
    
    produce_time = time.time() - start_time
    print(f"✓ Produced {num_samples} samples in {produce_time:.3f}s")
    print(f"  Throughput: {num_samples/produce_time:.1f} samples/sec")
    
    # Wait for processing
    await asyncio.sleep(0.5)
    
    # Consume samples
    print(f"\nConsuming samples...")
    start_time = time.time()
    consumed = 0
    
    for _ in range(num_samples):
        result = await client.get_sample()
        if result is not None:
            consumed += 1
    
    consume_time = time.time() - start_time
    print(f"✓ Consumed {consumed} samples in {consume_time:.3f}s")
    print(f"  Throughput: {consumed/consume_time:.1f} samples/sec")
    
    # Final stats
    stats = await client.get_statistics()
    print(f"\nFinal Statistics:")
    print(f"  Produced: {stats['total_produced']}")
    print(f"  Consumed: {stats['total_consumed']}")
    print(f"  Dropped: {stats['dropped_samples']}")
    
    await client.shutdown()
    await asyncio.sleep(0.2)
    
    return consumed > 0


def main():
    """Main entry point"""
    try:
        # Run smoke test
        success = asyncio.run(smoke_test())
        
        if success:
            print("\n" + "=" * 70)
            print("✓ Smoke test PASSED")
            print("=" * 70)
            
            # Run stress test
            stress_success = asyncio.run(stress_test())
            
            if stress_success:
                print("\n" + "=" * 70)
                print("✓ All tests PASSED")
                print("=" * 70)
                return 0
            else:
                print("\n✗ Stress test FAILED")
                return 1
        else:
            print("\n✗ Smoke test FAILED")
            return 1
            
    except Exception as e:
        print(f"\n✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Cleanup Ray
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    sys.exit(main())
