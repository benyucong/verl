#!/usr/bin/env python3
"""
Quick benchmark: Fast comparison of message queue implementations

This is a simplified version that runs in ~30 seconds vs the full benchmark.
Good for quick checks during development.
"""

import asyncio
import sys
import time
from pathlib import Path

import ray
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from recipe.fully_async_policy.message_queue import MessageQueue as QueueLock
from recipe.fully_async_policy.message_queue import MessageQueueClient as ClientLock

try:
    from recipe.fully_async_policy.message_queue_new import MessageQueue as QueueFast
    from recipe.fully_async_policy.message_queue_new import MessageQueueClient as ClientFast
    FASTMQ_OK = True
except ImportError:
    FASTMQ_OK = False


async def quick_test(queue_cls, client_cls, name: str, num_samples: int = 500):
    """Run a quick benchmark"""
    config = OmegaConf.create({"async_training": {"staleness_threshold": 10}})
    
    # Setup
    queue = queue_cls.remote(config=config, max_queue_size=200)
    await asyncio.sleep(0.15)
    client = client_cls(queue)
    await asyncio.sleep(0.1)
    
    # Test 1: Put latency
    put_times = []
    for i in range(num_samples):
        start = time.perf_counter_ns()
        await client.put_sample({"id": i}, 0)
        put_times.append((time.perf_counter_ns() - start) / 1000.0)
    
    await asyncio.sleep(0.2)
    
    # Test 2: Get latency
    get_times = []
    for _ in range(min(num_samples, 200)):
        start = time.perf_counter_ns()
        result = await client.get_sample()
        if result:
            get_times.append((time.perf_counter_ns() - start) / 1000.0)
    
    # Test 3: Throughput
    count = 0
    start = time.perf_counter()
    end_time = start + 1.0
    while time.perf_counter() < end_time:
        await client.put_sample({"id": count}, 0)
        count += 1
    throughput = count / (time.perf_counter() - start)
    
    # Cleanup
    try:
        await asyncio.wait_for(client.shutdown(), timeout=1.0)
    except:
        pass
    
    return {
        "name": name,
        "put_mean_us": sum(put_times) / len(put_times),
        "get_mean_us": sum(get_times) / len(get_times) if get_times else 0,
        "throughput_sps": throughput,
    }


async def main():
    print("🚀 Quick Message Queue Benchmark\n")
    
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    results = []
    
    # Test lock-based
    print("Testing lock-based queue...", end=" ", flush=True)
    r1 = await quick_test(QueueLock, ClientLock, "Lock-based")
    results.append(r1)
    print("✓")
    
    # Test fastmq
    if FASTMQ_OK:
        print("Testing FastMQ queue...", end=" ", flush=True)
        r2 = await quick_test(QueueFast, ClientFast, "FastMQ")
        results.append(r2)
        print("✓")
    
    # Results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    if len(results) == 2:
        r1, r2 = results
        print(f"\n{'Metric':<30} {'Lock-based':<15} {'FastMQ':<15} {'Speedup'}")
        print("-"*60)
        
        put_speedup = r1["put_mean_us"] / r2["put_mean_us"]
        print(f"{'Put latency (µs)':<30} {r1['put_mean_us']:>8.2f}{' '*6} "
              f"{r2['put_mean_us']:>8.2f}{' '*6} {put_speedup:.2f}x")
        
        if r2["get_mean_us"] > 0:
            get_speedup = r1["get_mean_us"] / r2["get_mean_us"]
            print(f"{'Get latency (µs)':<30} {r1['get_mean_us']:>8.2f}{' '*6} "
                  f"{r2['get_mean_us']:>8.2f}{' '*6} {get_speedup:.2f}x")
        
        tput_speedup = r2["throughput_sps"] / r1["throughput_sps"]
        print(f"{'Throughput (samples/s)':<30} {r1['throughput_sps']:>8,.0f}{' '*6} "
              f"{r2['throughput_sps']:>8,.0f}{' '*6} {tput_speedup:.2f}x")
        
        print("\n" + "="*60)
        avg_speedup = (put_speedup + get_speedup + tput_speedup) / 3 if r2["get_mean_us"] > 0 else (put_speedup + tput_speedup) / 2
        print(f"Average speedup: {avg_speedup:.2f}x faster")
        print("="*60)
    else:
        for r in results:
            print(f"\n{r['name']}:")
            print(f"  Put latency: {r['put_mean_us']:.2f} µs")
            print(f"  Get latency: {r['get_mean_us']:.2f} µs")
            print(f"  Throughput: {r['throughput_sps']:,.0f} samples/s")
    
    ray.shutdown()
    print("\n✅ Done!")


if __name__ == "__main__":
    asyncio.run(main())
