#!/usr/bin/env python3
"""
Quick examples of running benchmarks with different configurations
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from recipe.fully_async_policy.unittest.benchmark_message_queues import (
    main,
    BenchmarkConfig,
)


async def example_small():
    """Small benchmark for quick testing"""
    print("Running SMALL benchmark (fast, ~30 seconds)\n")
    config = BenchmarkConfig.small()
    await main(config)


async def example_production_4_workers():
    """Simulate 4 rollout workers"""
    print("Running benchmark simulating 4 rollout workers\n")
    config = BenchmarkConfig.production_like(num_rollout_workers=4)
    await main(config)


async def example_production_8_workers():
    """Simulate 8 rollout workers (matches your training script)"""
    print("Running benchmark simulating 8 rollout workers\n")
    config = BenchmarkConfig.production_like(num_rollout_workers=8)
    await main(config)


async def example_high_concurrency():
    """Test with many concurrent producers"""
    print("Running HIGH CONCURRENCY benchmark (16 producers)\n")
    config = BenchmarkConfig(
        num_producers=16,
        samples_per_producer=500,
        num_shards=16,
        latency_num_samples=500,
    )
    await main(config)


async def example_custom():
    """Custom configuration"""
    print("Running CUSTOM benchmark\n")
    config = BenchmarkConfig(
        # Latency tests
        latency_num_samples=2000,
        roundtrip_num_samples=1000,
        
        # Throughput tests
        put_throughput_duration=3.0,
        get_throughput_num_samples=10000,
        
        # Concurrent test - matches your training setup
        num_producers=4,  # 4 rollout GPUs
        samples_per_producer=2000,
        
        # Queue config
        max_queue_size=1000,
        num_shards=4,  # Match num_producers
        shard_capacity_bytes=2 * 1024 * 1024,  # 2MB per shard
    )
    await main(config)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run benchmark examples")
    parser.add_argument(
        'example',
        choices=['small', 'prod4', 'prod8', 'high', 'custom'],
        help='Which example to run'
    )
    
    args = parser.parse_args()
    
    examples = {
        'small': example_small,
        'prod4': example_production_4_workers,
        'prod8': example_production_8_workers,
        'high': example_high_concurrency,
        'custom': example_custom,
    }
    
    asyncio.run(examples[args.example]())
