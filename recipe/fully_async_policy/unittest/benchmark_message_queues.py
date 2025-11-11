#!/usr/bin/env python3
"""
Benchmark comparison: message_queue.py vs message_queue_new.py

This script compares the latency and throughput of:
1. message_queue.py - Lock-based implementation with asyncio.Lock
2. message_queue_new.py - Lock-free fastmq SPSC implementation

Metrics measured:
- Put latency (single sample)
- Get latency (single sample)
- Round-trip latency (put + wait + get)
- Throughput (samples/second)
- Concurrent producer throughput
- Memory usage
"""

import asyncio
import os
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Any, List

import ray
from omegaconf import OmegaConf

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# Import both implementations
from recipe.fully_async_policy.message_queue import MessageQueue as MessageQueueLock
from recipe.fully_async_policy.message_queue import MessageQueueClient as ClientLock

try:
    from recipe.fully_async_policy.message_queue_new import MessageQueue as MessageQueueFast
    from recipe.fully_async_policy.message_queue_new import MessageQueueClient as ClientFast
    FASTMQ_AVAILABLE = True
except ImportError as e:
    print(f"Warning: fastmq not available: {e}")
    FASTMQ_AVAILABLE = False


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark parameters"""
    # Latency test parameters
    latency_num_samples: int = 1000
    roundtrip_num_samples: int = 500
    
    # Throughput test parameters
    put_throughput_duration: float = 2.0
    get_throughput_num_samples: int = 5000
    
    # Concurrent test parameters
    num_producers: int = 4  # Simulates multiple rollout workers
    samples_per_producer: int = 1000
    
    # Queue configuration
    max_queue_size: int = 1000
    
    # FastMQ specific (ignored for lock-based)
    num_shards: int = 4  # Should match number of rollout workers
    shard_capacity_bytes: int = 1024 * 1024  # 1MB per shard
    
    def __post_init__(self):
        """Validate configuration"""
        assert self.num_producers > 0, "num_producers must be > 0"
        assert self.samples_per_producer > 0, "samples_per_producer must be > 0"
        assert self.num_shards > 0, "num_shards must be > 0"
    
    @classmethod
    def small(cls):
        """Small benchmark for quick testing"""
        return cls(
            latency_num_samples=100,
            roundtrip_num_samples=50,
            put_throughput_duration=1.0,
            get_throughput_num_samples=500,
            num_producers=2,
            samples_per_producer=100,
        )
    
    @classmethod
    def large(cls):
        """Large benchmark for thorough testing"""
        return cls(
            latency_num_samples=5000,
            roundtrip_num_samples=2000,
            put_throughput_duration=5.0,
            get_throughput_num_samples=20000,
            num_producers=8,
            samples_per_producer=2000,
        )
    
    @classmethod
    def production_like(cls, num_rollout_workers: int = 8):
        """Configuration that mimics production workload"""
        return cls(
            latency_num_samples=2000,
            roundtrip_num_samples=1000,
            put_throughput_duration=3.0,
            get_throughput_num_samples=10000,
            num_producers=num_rollout_workers,  # Match your rollout workers
            samples_per_producer=1500,
            num_shards=num_rollout_workers,  # One shard per rollout worker
            shard_capacity_bytes=2 * 1024 * 1024,  # 2MB per shard
        )


@dataclass
class BenchmarkResult:
    """Results from a benchmark run"""
    name: str
    implementation: str
    
    # Latency metrics (microseconds)
    put_latency_mean: float
    put_latency_median: float
    put_latency_p95: float
    put_latency_p99: float
    
    get_latency_mean: float
    get_latency_median: float
    get_latency_p95: float
    get_latency_p99: float
    
    roundtrip_latency_mean: float
    roundtrip_latency_median: float
    roundtrip_latency_p95: float
    roundtrip_latency_p99: float
    
    # Throughput metrics (samples/second)
    put_throughput: float
    get_throughput: float
    concurrent_throughput: float
    
    # Memory metrics
    peak_memory_mb: float
    avg_queue_size: float
    
    # Additional stats
    total_samples: int
    dropped_samples: int
    duration_seconds: float


class MessageQueueBenchmark:
    """Benchmark harness for message queues"""
    
    def __init__(self, queue_class, client_class, name: str, config: BenchmarkConfig):
        self.queue_class = queue_class
        self.client_class = client_class
        self.name = name
        self.config = config
        self.queue_actor = None
        self.client = None
        
    async def setup(self, **kwargs):
        """Initialize the queue"""
        queue_config = OmegaConf.create({
            "async_training": {"staleness_threshold": 10}
        })
        
        print(f"\n[{self.name}] Setting up queue...")
        print(f"  Max queue size: {self.config.max_queue_size}")
        if "num_shards" in kwargs:
            print(f"  Num shards: {kwargs['num_shards']}")
            print(f"  Shard capacity: {kwargs['shard_capacity_bytes'] / 1024:.0f} KB")
        
        self.queue_actor = self.queue_class.remote(
            config=queue_config,
            max_queue_size=self.config.max_queue_size,
            **kwargs
        )
        await asyncio.sleep(0.2)  # Let polling thread start
        
        self.client = self.client_class(self.queue_actor)
        await asyncio.sleep(0.1)
        print(f"[{self.name}] Setup complete")
    
    async def teardown(self):
        """Cleanup"""
        if self.client:
            try:
                await asyncio.wait_for(self.client.shutdown(), timeout=2.0)
            except asyncio.TimeoutError:
                pass
        await asyncio.sleep(0.2)
    
    async def benchmark_put_latency(self, num_samples: int = 1000) -> List[float]:
        """Measure put operation latency"""
        print(f"[{self.name}] Benchmarking put latency ({num_samples} samples)...")
        latencies = []
        
        for i in range(num_samples):
            sample = {"id": i, "data": f"sample_{i}"}
            
            start = time.perf_counter_ns()
            await self.client.put_sample(sample, param_version=0)
            end = time.perf_counter_ns()
            
            latencies.append((end - start) / 1000.0)  # Convert to microseconds
            
            if i % 100 == 0:
                await asyncio.sleep(0.001)  # Let queue drain a bit
        
        return latencies
    
    async def benchmark_get_latency(self, num_samples: int = 1000) -> List[float]:
        """Measure get operation latency"""
        print(f"[{self.name}] Benchmarking get latency ({num_samples} samples)...")
        
        # Pre-fill queue
        for i in range(num_samples):
            await self.client.put_sample({"id": i}, param_version=0)
        
        await asyncio.sleep(0.3)  # Let queue fill
        
        latencies = []
        for _ in range(num_samples):
            start = time.perf_counter_ns()
            result = await self.client.get_sample()
            end = time.perf_counter_ns()
            
            if result is not None:
                latencies.append((end - start) / 1000.0)  # Microseconds
        
        return latencies
    
    async def benchmark_roundtrip_latency(self, num_samples: int = 500) -> List[float]:
        """Measure round-trip latency (put + get)"""
        print(f"[{self.name}] Benchmarking round-trip latency ({num_samples} samples)...")
        latencies = []
        
        for i in range(num_samples):
            sample = {"id": i, "timestamp": time.perf_counter_ns()}
            
            start = time.perf_counter_ns()
            await self.client.put_sample(sample, param_version=0)
            
            # Wait a tiny bit for processing
            await asyncio.sleep(0.0001)
            
            result = await self.client.get_sample()
            end = time.perf_counter_ns()
            
            if result is not None:
                latencies.append((end - start) / 1000.0)  # Microseconds
        
        return latencies
    
    async def benchmark_put_throughput(self, duration_seconds: float = 2.0) -> float:
        """Measure put throughput (samples/second)"""
        print(f"[{self.name}] Benchmarking put throughput ({duration_seconds}s)...")
        
        count = 0
        start = time.perf_counter()
        end_time = start + duration_seconds
        
        while time.perf_counter() < end_time:
            sample = {"id": count, "data": f"data_{count}"}
            await self.client.put_sample(sample, param_version=0)
            count += 1
        
        elapsed = time.perf_counter() - start
        throughput = count / elapsed
        
        print(f"[{self.name}] Put throughput: {throughput:.2f} samples/sec")
        return throughput
    
    async def benchmark_get_throughput(self, num_samples: int = 5000) -> float:
        """Measure get throughput (samples/second)"""
        print(f"[{self.name}] Benchmarking get throughput ({num_samples} samples)...")
        
        # Pre-fill queue
        for i in range(num_samples):
            await self.client.put_sample({"id": i}, param_version=0)
        
        await asyncio.sleep(0.5)  # Let queue fill
        
        # Measure get throughput
        count = 0
        start = time.perf_counter()
        
        for _ in range(num_samples):
            result = await self.client.get_sample()
            if result is not None:
                count += 1
        
        elapsed = time.perf_counter() - start
        throughput = count / elapsed
        
        print(f"[{self.name}] Get throughput: {throughput:.2f} samples/sec")
        return throughput
    
    async def benchmark_concurrent_throughput(self, 
                                             num_producers: int = 4,
                                             samples_per_producer: int = 1000) -> float:
        """Measure concurrent producer throughput"""
        print(f"[{self.name}] Benchmarking concurrent throughput "
              f"({num_producers} producers, {samples_per_producer} samples each)...")
        
        async def producer(producer_id: int):
            for i in range(samples_per_producer):
                sample = {"producer": producer_id, "id": i}
                await self.client.put_sample(sample, param_version=producer_id)
        
        start = time.perf_counter()
        await asyncio.gather(*[producer(i) for i in range(num_producers)])
        elapsed = time.perf_counter() - start
        
        total_samples = num_producers * samples_per_producer
        throughput = total_samples / elapsed
        
        print(f"[{self.name}] Concurrent throughput: {throughput:.2f} samples/sec")
        return throughput
    
    async def get_memory_usage(self) -> float:
        """Get current memory usage"""
        mem_info = await self.client.get_memory_usage()
        return mem_info.get("estimated_memory_mb", 0)
    
    async def run_full_benchmark(self) -> BenchmarkResult:
        """Run complete benchmark suite"""
        print(f"\n{'='*70}")
        print(f"Running full benchmark: {self.name}")
        print(f"{'='*70}")
        
        # Latency tests
        put_latencies = await self.benchmark_put_latency(
            num_samples=self.config.latency_num_samples
        )
        get_latencies = await self.benchmark_get_latency(
            num_samples=self.config.latency_num_samples
        )
        roundtrip_latencies = await self.benchmark_roundtrip_latency(
            num_samples=self.config.roundtrip_num_samples
        )
        
        # Throughput tests
        put_throughput = await self.benchmark_put_throughput(
            duration_seconds=self.config.put_throughput_duration
        )
        get_throughput = await self.benchmark_get_throughput(
            num_samples=self.config.get_throughput_num_samples
        )
        concurrent_throughput = await self.benchmark_concurrent_throughput(
            num_producers=self.config.num_producers,
            samples_per_producer=self.config.samples_per_producer
        )
        
        # Memory usage
        peak_memory = await self.get_memory_usage()
        
        # Get statistics
        stats = await self.client.get_statistics()
        
        def percentile(data: List[float], p: float) -> float:
            return statistics.quantiles(data, n=100)[int(p) - 1] if data else 0
        
        result = BenchmarkResult(
            name=self.name,
            implementation=self.queue_class.__name__,
            
            # Put latency
            put_latency_mean=statistics.mean(put_latencies) if put_latencies else 0,
            put_latency_median=statistics.median(put_latencies) if put_latencies else 0,
            put_latency_p95=percentile(put_latencies, 95),
            put_latency_p99=percentile(put_latencies, 99),
            
            # Get latency
            get_latency_mean=statistics.mean(get_latencies) if get_latencies else 0,
            get_latency_median=statistics.median(get_latencies) if get_latencies else 0,
            get_latency_p95=percentile(get_latencies, 95),
            get_latency_p99=percentile(get_latencies, 99),
            
            # Roundtrip latency
            roundtrip_latency_mean=statistics.mean(roundtrip_latencies) if roundtrip_latencies else 0,
            roundtrip_latency_median=statistics.median(roundtrip_latencies) if roundtrip_latencies else 0,
            roundtrip_latency_p95=percentile(roundtrip_latencies, 95),
            roundtrip_latency_p99=percentile(roundtrip_latencies, 99),
            
            # Throughput
            put_throughput=put_throughput,
            get_throughput=get_throughput,
            concurrent_throughput=concurrent_throughput,
            
            # Memory
            peak_memory_mb=peak_memory,
            avg_queue_size=stats.get("fifo_size", 0),
            
            # Stats
            total_samples=stats.get("total_produced", 0),
            dropped_samples=stats.get("dropped_samples", 0),
            duration_seconds=0,  # Not tracked in this version
        )
        
        return result


def print_comparison_table(results: List[BenchmarkResult]):
    """Print a comparison table of benchmark results"""
    if len(results) < 2:
        print("Need at least 2 results to compare")
        return
    
    lock_result = results[0]
    fast_result = results[1]
    
    print("\n" + "="*100)
    print("BENCHMARK COMPARISON RESULTS")
    print("="*100)
    
    # Latency comparison
    print("\n📊 LATENCY COMPARISON (microseconds - lower is better)")
    print("-"*100)
    print(f"{'Metric':<30} {'Lock-based':<20} {'FastMQ':<20} {'Speedup':<20}")
    print("-"*100)
    
    def compare_metric(name: str, lock_val: float, fast_val: float):
        speedup = lock_val / fast_val if fast_val > 0 else 0
        speedup_str = f"{speedup:.2f}x faster" if speedup > 1 else f"{1/speedup:.2f}x slower"
        print(f"{name:<30} {lock_val:>8.2f} µs{' '*10} {fast_val:>8.2f} µs{' '*10} {speedup_str}")
    
    compare_metric("Put - Mean", lock_result.put_latency_mean, fast_result.put_latency_mean)
    compare_metric("Put - Median", lock_result.put_latency_median, fast_result.put_latency_median)
    compare_metric("Put - P95", lock_result.put_latency_p95, fast_result.put_latency_p95)
    compare_metric("Put - P99", lock_result.put_latency_p99, fast_result.put_latency_p99)
    print()
    compare_metric("Get - Mean", lock_result.get_latency_mean, fast_result.get_latency_mean)
    compare_metric("Get - Median", lock_result.get_latency_median, fast_result.get_latency_median)
    compare_metric("Get - P95", lock_result.get_latency_p95, fast_result.get_latency_p95)
    compare_metric("Get - P99", lock_result.get_latency_p99, fast_result.get_latency_p99)
    print()
    compare_metric("Round-trip - Mean", lock_result.roundtrip_latency_mean, fast_result.roundtrip_latency_mean)
    compare_metric("Round-trip - Median", lock_result.roundtrip_latency_median, fast_result.roundtrip_latency_median)
    compare_metric("Round-trip - P95", lock_result.roundtrip_latency_p95, fast_result.roundtrip_latency_p95)
    compare_metric("Round-trip - P99", lock_result.roundtrip_latency_p99, fast_result.roundtrip_latency_p99)
    
    # Throughput comparison
    print("\n" + "="*100)
    print("🚀 THROUGHPUT COMPARISON (samples/second - higher is better)")
    print("-"*100)
    print(f"{'Metric':<30} {'Lock-based':<20} {'FastMQ':<20} {'Improvement':<20}")
    print("-"*100)
    
    def compare_throughput(name: str, lock_val: float, fast_val: float):
        improvement = fast_val / lock_val if lock_val > 0 else 0
        improvement_str = f"{improvement:.2f}x faster" if improvement > 1 else f"{1/improvement:.2f}x slower"
        print(f"{name:<30} {lock_val:>12,.2f}/s{' '*5} {fast_val:>12,.2f}/s{' '*5} {improvement_str}")
    
    compare_throughput("Put throughput", lock_result.put_throughput, fast_result.put_throughput)
    compare_throughput("Get throughput", lock_result.get_throughput, fast_result.get_throughput)
    compare_throughput("Concurrent (4 producers)", lock_result.concurrent_throughput, fast_result.concurrent_throughput)
    
    # Memory and stats
    print("\n" + "="*100)
    print("💾 MEMORY & STATISTICS")
    print("-"*100)
    print(f"{'Metric':<30} {'Lock-based':<20} {'FastMQ':<20} {'Difference':<20}")
    print("-"*100)
    print(f"{'Peak Memory (MB)':<30} {lock_result.peak_memory_mb:>12.2f} MB{' '*5} "
          f"{fast_result.peak_memory_mb:>12.2f} MB{' '*5} "
          f"{((fast_result.peak_memory_mb - lock_result.peak_memory_mb) / lock_result.peak_memory_mb * 100):+.1f}%")
    print(f"{'Total Samples':<30} {lock_result.total_samples:>12,}{' '*8} "
          f"{fast_result.total_samples:>12,}")
    print(f"{'Dropped Samples':<30} {lock_result.dropped_samples:>12,}{' '*8} "
          f"{fast_result.dropped_samples:>12,}")
    
    # Summary
    print("\n" + "="*100)
    print("📈 SUMMARY")
    print("-"*100)
    avg_latency_speedup = (
        (lock_result.put_latency_mean + lock_result.get_latency_mean) /
        (fast_result.put_latency_mean + fast_result.get_latency_mean)
    )
    avg_throughput_improvement = (
        (fast_result.put_throughput + fast_result.get_throughput) /
        (lock_result.put_throughput + lock_result.get_throughput)
    )
    
    print(f"Average Latency Speedup:       {avg_latency_speedup:.2f}x")
    print(f"Average Throughput Improvement: {avg_throughput_improvement:.2f}x")
    print(f"Concurrent Throughput Boost:    {fast_result.concurrent_throughput / lock_result.concurrent_throughput:.2f}x")
    print("="*100)


async def main(benchmark_config: BenchmarkConfig = None):
    """Main benchmark runner
    
    Args:
        benchmark_config: Configuration for benchmark parameters.
                         If None, uses default config.
    """
    if benchmark_config is None:
        benchmark_config = BenchmarkConfig()
    
    print("="*70)
    print("MESSAGE QUEUE BENCHMARK SUITE")
    print("="*70)
    print("\nComparing:")
    print("1. message_queue.py (Lock-based with asyncio.Lock)")
    print("2. message_queue_new.py (Lock-free with fastmq SPSC)")
    print()
    print("Benchmark Configuration:")
    print(f"  Latency samples: {benchmark_config.latency_num_samples}")
    print(f"  Throughput duration: {benchmark_config.put_throughput_duration}s")
    print(f"  Concurrent producers: {benchmark_config.num_producers}")
    print(f"  Samples per producer: {benchmark_config.samples_per_producer}")
    print(f"  FastMQ shards: {benchmark_config.num_shards}")
    print()
    
    # Initialize Ray
    if not ray.is_initialized():
        print("Initializing Ray...")
        ray.init(ignore_reinit_error=True)
        print("Ray initialized\n")
    
    results = []
    
    # Benchmark 1: Lock-based implementation
    print("\n" + "="*70)
    print("BENCHMARK 1: Lock-based Implementation (message_queue.py)")
    print("="*70)
    
    bench_lock = MessageQueueBenchmark(
        MessageQueueLock,
        ClientLock,
        "Lock-based Queue",
        benchmark_config
    )
    await bench_lock.setup()
    
    try:
        result_lock = await bench_lock.run_full_benchmark()
        results.append(result_lock)
    finally:
        await bench_lock.teardown()
    
    # Benchmark 2: FastMQ implementation
    if FASTMQ_AVAILABLE:
        print("\n" + "="*70)
        print("BENCHMARK 2: Lock-free FastMQ Implementation (message_queue_new.py)")
        print("="*70)
        
        bench_fast = MessageQueueBenchmark(
            MessageQueueFast,
            ClientFast,
            "FastMQ Queue",
            benchmark_config
        )
        await bench_fast.setup(
            num_shards=benchmark_config.num_shards,
            shard_capacity_bytes=benchmark_config.shard_capacity_bytes
        )
        
        try:
            result_fast = await bench_fast.run_full_benchmark()
            results.append(result_fast)
        finally:
            await bench_fast.teardown()
    else:
        print("\n⚠️  FastMQ not available - skipping lock-free benchmark")
        print("Make sure fastmq C++ extension is compiled")
    
    # Print comparison
    if len(results) >= 2:
        print_comparison_table(results)
    
    # Cleanup
    print("\nCleaning up...")
    if ray.is_initialized():
        ray.shutdown()
    
    print("\n✅ Benchmark complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Benchmark message queue implementations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default benchmark
  python benchmark_message_queues.py
  
  # Small/quick benchmark
  python benchmark_message_queues.py --preset small
  
  # Large/thorough benchmark
  python benchmark_message_queues.py --preset large
  
  # Production-like with 8 rollout workers
  python benchmark_message_queues.py --preset production --num-rollout-workers 8
  
  # Custom configuration
  python benchmark_message_queues.py --num-producers 6 --samples-per-producer 2000
        """
    )
    
    parser.add_argument(
        '--preset',
        choices=['default', 'small', 'large', 'production'],
        default='default',
        help='Use a preset configuration (default: default)'
    )
    parser.add_argument(
        '--num-rollout-workers',
        type=int,
        help='Number of rollout workers to simulate (for production preset)'
    )
    parser.add_argument(
        '--num-producers',
        type=int,
        help='Number of concurrent producers'
    )
    parser.add_argument(
        '--samples-per-producer',
        type=int,
        help='Number of samples each producer sends'
    )
    parser.add_argument(
        '--num-shards',
        type=int,
        help='Number of FastMQ shards (should match rollout workers)'
    )
    parser.add_argument(
        '--latency-samples',
        type=int,
        help='Number of samples for latency tests'
    )
    
    args = parser.parse_args()
    
    # Select configuration based on preset
    if args.preset == 'small':
        config = BenchmarkConfig.small()
    elif args.preset == 'large':
        config = BenchmarkConfig.large()
    elif args.preset == 'production':
        num_workers = args.num_rollout_workers or 8
        config = BenchmarkConfig.production_like(num_rollout_workers=num_workers)
    else:
        config = BenchmarkConfig()
    
    # Override with command-line arguments
    if args.num_producers is not None:
        config.num_producers = args.num_producers
    if args.samples_per_producer is not None:
        config.samples_per_producer = args.samples_per_producer
    if args.num_shards is not None:
        config.num_shards = args.num_shards
    if args.latency_samples is not None:
        config.latency_num_samples = args.latency_samples
    
    try:
        asyncio.run(main(config))
    except KeyboardInterrupt:
        print("\n\n⚠️  Benchmark interrupted by user")
        if ray.is_initialized():
            ray.shutdown()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        if ray.is_initialized():
            ray.shutdown()
        sys.exit(1)
