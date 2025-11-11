#!/usr/bin/env python3
"""
Benchmark: Realistic Training Loop Simulation

This benchmark mimics the actual training loop pattern from fully_async_main.py:
- Multiple async rollout workers (producers) continuously generating samples
- Single trainer (consumer) collecting batches and training
- Measures end-to-end latency and throughput under realistic conditions

Pattern simulated:
1. Rollouter: Multiple workers continuously put samples (streaming)
2. Trainer: Collects batches of samples, processes them, repeats
3. Parameter updates: Periodic version updates that trigger staleness dropping

Metrics:
- Producer throughput (samples/sec per worker)
- Consumer batch collection latency
- End-to-end sample latency (put to get)
- Queue utilization (fullness over time)
- Staleness drops (samples dropped due to old param version)
- Training iteration throughput (batches/sec)
"""

import asyncio
import os
import pickle
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

import ray
from omegaconf import OmegaConf

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# Import both implementations
from recipe.fully_async_policy.message_queue import \
    MessageQueue as MessageQueueLock
from recipe.fully_async_policy.message_queue import \
    MessageQueueClient as ClientLock

try:
    from recipe.fully_async_policy.message_queue_new import \
        MessageQueue as MessageQueueFast
    from recipe.fully_async_policy.message_queue_new import \
        MessageQueueClient as ClientFast
    FASTMQ_AVAILABLE = True
except ImportError as e:
    print(f"Warning: fastmq not available: {e}")
    FASTMQ_AVAILABLE = False


@dataclass
class TrainingLoopConfig:
    """Configuration mimicking real training setup"""
    # Rollout configuration
    num_rollout_workers: int = 8  # Number of parallel rollout workers
    samples_per_worker: int = 500  # Total samples each worker generates
    sample_generation_delay_ms: float = 10.0  # Simulate generation time per sample
    
    # Trainer configuration
    batch_size: int = 256  # Samples per training batch
    training_iterations: int = 10  # Number of training iterations to run
    training_delay_ms: float = 50.0  # Simulate training time per batch
    
    # Parameter update configuration
    param_update_interval: int = 2  # Update param version every N iterations
    staleness_threshold: int = 3  # Drop samples older than this many versions
    
    # Queue configuration
    max_queue_size: int = 2000
    
    # FastMQ specific
    num_shards: int = 8  # Should match num_rollout_workers
    shard_capacity_bytes: int = 2 * 1024 * 1024  # 2MB per shard
    
    # Benchmark control
    warmup_samples: int = 100  # Samples to generate before starting measurement
    
    def __post_init__(self):
        assert self.num_rollout_workers > 0
        assert self.batch_size > 0
        assert self.num_shards > 0
        if self.num_shards != self.num_rollout_workers:
            print(f"Warning: num_shards ({self.num_shards}) != num_rollout_workers ({self.num_rollout_workers})")
    
    @classmethod
    def small(cls):
        """Small benchmark for quick testing"""
        return cls(
            num_rollout_workers=2,
            samples_per_worker=200,
            sample_generation_delay_ms=5.0,
            batch_size=64,
            training_iterations=5,
            training_delay_ms=20.0,
            num_shards=2,
            warmup_samples=20,
        )
    
    @classmethod
    def realistic(cls, num_rollout_workers: int = 8):
        """Realistic production-like configuration"""
        return cls(
            num_rollout_workers=num_rollout_workers,
            samples_per_worker=1000,
            sample_generation_delay_ms=15.0,
            batch_size=256,
            training_iterations=20,
            training_delay_ms=100.0,
            num_shards=num_rollout_workers,
            max_queue_size=3000,
            warmup_samples=200,
        )
    
    @classmethod
    def stress(cls):
        """Stress test with high throughput"""
        return cls(
            num_rollout_workers=16,
            samples_per_worker=2000,
            sample_generation_delay_ms=2.0,  # Very fast generation
            batch_size=512,
            training_iterations=30,
            training_delay_ms=150.0,
            num_shards=16,
            max_queue_size=5000,
            warmup_samples=500,
        )


@dataclass
class TrainingLoopMetrics:
    """Results from training loop benchmark"""
    implementation: str
    
    # Producer metrics
    total_samples_generated: int
    total_samples_put: int
    samples_dropped: int
    producer_throughput: float  # samples/sec across all workers
    avg_put_latency_us: float
    
    # Consumer metrics
    total_batches_collected: int
    total_samples_consumed: int
    consumer_throughput: float  # samples/sec
    avg_batch_collection_time_ms: float
    
    # End-to-end metrics
    avg_e2e_latency_ms: float  # Time from sample generation to consumption
    p50_e2e_latency_ms: float
    p95_e2e_latency_ms: float
    p99_e2e_latency_ms: float
    
    # Training metrics
    training_iterations_completed: int
    avg_iteration_time_s: float
    total_training_time_s: float
    
    # Queue metrics
    avg_queue_size: float
    max_queue_size: int
    queue_utilization: float  # avg_queue_size / max_queue_size
    
    # Staleness metrics
    param_version_updates: int
    staleness_drops: int
    
    # Fields with defaults must come last
    batch_collection_times: List[float] = field(default_factory=list)
    e2e_latencies: List[float] = field(default_factory=list)


class MockRolloutSample:
    """Mock rollout sample mimicking real data structure"""
    def __init__(self, worker_id: int, sample_id: int, param_version: int):
        self.worker_id = worker_id
        self.sample_id = sample_id
        self.param_version = param_version
        self.timestamp = time.time()
        # Simulate realistic data size (tokens, logprobs, etc.)
        self.data = {
            "prompts": ["sample prompt"] * 10,
            "responses": ["sample response"] * 10,
            "old_log_probs": [0.1] * 100,
            "values": [0.5] * 100,
        }


class TrainingLoopSimulator:
    """Simulates the fully async training loop pattern"""
    
    def __init__(self, queue_class, client_class, name: str, config: TrainingLoopConfig):
        self.queue_class = queue_class
        self.client_class = client_class
        self.name = name
        self.config = config
        self.queue_actor = None
        self.client = None
        
        # Tracking metrics
        self.current_param_version = 0
        self.samples_generated = 0
        self.samples_put = 0
        self.samples_consumed = 0
        self.put_latencies = []
        self.e2e_latencies = []
        self.batch_collection_times = []
        self.queue_sizes = []
        
        # Control flags
        self.running = False
        self.warmup_complete = False
        
    async def setup(self, **kwargs):
        """Initialize queue with realistic settings"""
        queue_config = OmegaConf.create({
            "async_training": {"staleness_threshold": self.config.staleness_threshold}
        })
        
        # Ensure queue is large enough
        actual_queue_size = max(
            self.config.max_queue_size,
            self.config.batch_size * 2,
            self.config.num_rollout_workers * 100
        )
        
        print(f"\n[{self.name}] Setting up training loop simulation...")
        print(f"  Rollout workers: {self.config.num_rollout_workers}")
        print(f"  Samples per worker: {self.config.samples_per_worker}")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Training iterations: {self.config.training_iterations}")
        print(f"  Queue size: {actual_queue_size}")
        print(f"  Staleness threshold: {self.config.staleness_threshold}")
        if "num_shards" in kwargs:
            print(f"  Num shards: {kwargs['num_shards']}")
        
        self.queue_actor = self.queue_class.remote(
            config=queue_config,
            max_queue_size=actual_queue_size,
            **kwargs
        )
        await asyncio.sleep(0.2)
        
        self.client = self.client_class(self.queue_actor)
        await asyncio.sleep(0.1)
        print(f"[{self.name}] Setup complete")
    
    async def teardown(self):
        """Cleanup"""
        print(f"[{self.name}] Tearing down...")
        self.running = False
        
        if self.client:
            try:
                await asyncio.wait_for(self.client.shutdown(), timeout=5.0)
            except Exception as e:
                print(f"Warning: client shutdown failed: {e}")
        
        if self.queue_actor:
            try:
                ray.kill(self.queue_actor)
            except Exception as e:
                print(f"Warning: failed to kill queue actor: {e}")
        
        await asyncio.sleep(0.2)
    
    async def rollout_worker(self, worker_id: int) -> Dict[str, Any]:
        """
        Simulates a single rollout worker continuously generating samples.
        This mimics the behavior in fully_async_rollouter.py
        """
        samples_generated = 0
        samples_put_success = 0
        put_latencies = []
        
        print(f"[{self.name}] Rollout worker {worker_id} started")
        
        for sample_id in range(self.config.samples_per_worker):
            # Simulate sample generation time
            await asyncio.sleep(self.config.sample_generation_delay_ms / 1000.0)
            
            # Create mock sample
            sample = MockRolloutSample(
                worker_id=worker_id,
                sample_id=sample_id,
                param_version=self.current_param_version
            )
            samples_generated += 1
            
            # Serialize like real code does
            serialized_sample = pickle.dumps(sample, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Put sample in queue
            start = time.perf_counter_ns()
            try:
                success = await self.client.put_sample(
                    sample=serialized_sample,
                    param_version=sample.param_version
                )
                end = time.perf_counter_ns()
                
                if success:
                    samples_put_success += 1
                    if self.warmup_complete:
                        put_latencies.append((end - start) / 1000.0)  # microseconds
                
            except Exception as e:
                print(f"Worker {worker_id} put_sample failed: {e}")
            
            # Progress reporting
            if sample_id > 0 and sample_id % 100 == 0:
                print(f"[{self.name}] Worker {worker_id}: {sample_id}/{self.config.samples_per_worker} samples")
        
        print(f"[{self.name}] Rollout worker {worker_id} completed: "
              f"{samples_put_success}/{samples_generated} samples successfully put")
        
        return {
            "worker_id": worker_id,
            "generated": samples_generated,
            "put_success": samples_put_success,
            "put_latencies": put_latencies,
        }
    
    async def trainer_worker(self) -> Dict[str, Any]:
        """
        Simulates the trainer consuming batches and training.
        This mimics the behavior in fully_async_trainer.py
        """
        iteration = 0
        batches_collected = 0
        samples_consumed = 0
        batch_collection_times = []
        e2e_latencies = []
        queue_sizes_sampled = []
        
        print(f"[{self.name}] Trainer started")
        
        # Warmup phase
        print(f"[{self.name}] Trainer warmup: collecting {self.config.warmup_samples} samples...")
        warmup_start = time.time()
        for _ in range(self.config.warmup_samples):
            try:
                result = await asyncio.wait_for(self.client.get_sample(), timeout=10.0)
                if result is None:
                    break
            except asyncio.TimeoutError:
                break
        
        warmup_time = time.time() - warmup_start
        print(f"[{self.name}] Trainer warmup complete ({warmup_time:.2f}s)")
        self.warmup_complete = True
        
        training_start_time = time.time()
        
        while iteration < self.config.training_iterations and self.running:
            # Collect a batch of samples
            batch_start = time.time()
            batch = []
            
            print(f"[{self.name}] Trainer iteration {iteration+1}: collecting batch of {self.config.batch_size}")
            
            for _ in range(self.config.batch_size):
                try:
                    result = await asyncio.wait_for(self.client.get_sample(), timeout=10.0)
                    if result is None:
                        print(f"[{self.name}] Received None, stopping collection")
                        break
                    
                    sample_data, queue_len = result
                    queue_sizes_sampled.append(queue_len)
                    
                    # Deserialize
                    sample = pickle.loads(sample_data)
                    
                    # Calculate end-to-end latency
                    e2e_latency_ms = (time.time() - sample.timestamp) * 1000.0
                    e2e_latencies.append(e2e_latency_ms)
                    
                    batch.append(sample)
                    
                except asyncio.TimeoutError:
                    print(f"[{self.name}] Timeout collecting sample {len(batch)}/{self.config.batch_size}")
                    break
                except Exception as e:
                    print(f"[{self.name}] Error collecting sample: {e}")
                    break
            
            batch_end = time.time()
            batch_collection_time_ms = (batch_end - batch_start) * 1000.0
            
            if len(batch) < self.config.batch_size:
                print(f"[{self.name}] Only collected {len(batch)}/{self.config.batch_size} samples")
                if len(batch) == 0:
                    print(f"[{self.name}] No samples collected, stopping training")
                    break
            
            batch_collection_times.append(batch_collection_time_ms)
            batches_collected += 1
            samples_consumed += len(batch)
            
            # Simulate training on the batch
            print(f"[{self.name}] Training on batch {iteration+1} ({len(batch)} samples)...")
            await asyncio.sleep(self.config.training_delay_ms / 1000.0)
            
            iteration += 1
            
            # Simulate parameter update
            if iteration % self.config.param_update_interval == 0:
                self.current_param_version += 1
                await self.client.update_param_version(self.current_param_version)
                print(f"[{self.name}] Parameter version updated to {self.current_param_version}")
            
            print(f"[{self.name}] Iteration {iteration} complete. "
                  f"Collected {len(batch)} samples in {batch_collection_time_ms:.2f}ms")
        
        training_end_time = time.time()
        total_training_time = training_end_time - training_start_time
        
        print(f"[{self.name}] Trainer completed: {iteration} iterations, "
              f"{batches_collected} batches, {samples_consumed} samples")
        
        return {
            "iterations": iteration,
            "batches": batches_collected,
            "samples": samples_consumed,
            "batch_collection_times": batch_collection_times,
            "e2e_latencies": e2e_latencies,
            "queue_sizes": queue_sizes_sampled,
            "total_time": total_training_time,
        }
    
    async def run_training_loop(self) -> TrainingLoopMetrics:
        """
        Run the full training loop simulation with concurrent producers and consumer
        """
        print(f"\n{'='*70}")
        print(f"Running Training Loop Simulation: {self.name}")
        print(f"{'='*70}")
        
        self.running = True
        self.warmup_complete = False
        self.current_param_version = 0
        
        start_time = time.time()
        
        # Start trainer first (consumer)
        trainer_task = asyncio.create_task(self.trainer_worker())
        
        # Give trainer a moment to get ready
        await asyncio.sleep(0.5)
        
        # Start all rollout workers (producers)
        rollout_tasks = [
            asyncio.create_task(self.rollout_worker(worker_id))
            for worker_id in range(self.config.num_rollout_workers)
        ]
        
        # Wait for rollout workers to complete
        rollout_results = await asyncio.gather(*rollout_tasks, return_exceptions=True)
        print(f"[{self.name}] All rollout workers completed")
        
        # Send termination signal (like real training loop does)
        await self.client.put_sample(sample=None, param_version=self.current_param_version)
        print(f"[{self.name}] Sent termination signal")
        
        # Wait for trainer to complete
        trainer_result = await trainer_task
        print(f"[{self.name}] Trainer completed")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Aggregate results
        total_generated = sum(r["generated"] for r in rollout_results if isinstance(r, dict))
        total_put_success = sum(r["put_success"] for r in rollout_results if isinstance(r, dict))
        all_put_latencies = []
        for r in rollout_results:
            if isinstance(r, dict):
                all_put_latencies.extend(r["put_latencies"])
        
        # Get queue statistics
        stats = await self.client.get_statistics()
        
        # Calculate metrics
        e2e_latencies = trainer_result["e2e_latencies"]
        batch_times = trainer_result["batch_collection_times"]
        queue_sizes = trainer_result["queue_sizes"]
        
        metrics = TrainingLoopMetrics(
            implementation=self.name,
            
            # Producer metrics
            total_samples_generated=total_generated,
            total_samples_put=total_put_success,
            samples_dropped=stats.get("dropped_samples", 0),
            producer_throughput=total_put_success / total_time if total_time > 0 else 0,
            avg_put_latency_us=statistics.mean(all_put_latencies) if all_put_latencies else 0,
            
            # Consumer metrics
            total_batches_collected=trainer_result["batches"],
            total_samples_consumed=trainer_result["samples"],
            consumer_throughput=trainer_result["samples"] / total_time if total_time > 0 else 0,
            avg_batch_collection_time_ms=statistics.mean(batch_times) if batch_times else 0,
            batch_collection_times=batch_times,
            
            # End-to-end metrics
            avg_e2e_latency_ms=statistics.mean(e2e_latencies) if e2e_latencies else 0,
            p50_e2e_latency_ms=statistics.median(e2e_latencies) if e2e_latencies else 0,
            p95_e2e_latency_ms=statistics.quantiles(e2e_latencies, n=20)[18] if len(e2e_latencies) > 20 else 0,
            p99_e2e_latency_ms=statistics.quantiles(e2e_latencies, n=100)[98] if len(e2e_latencies) > 100 else 0,
            e2e_latencies=e2e_latencies,
            
            # Training metrics
            training_iterations_completed=trainer_result["iterations"],
            avg_iteration_time_s=total_time / trainer_result["iterations"] if trainer_result["iterations"] > 0 else 0,
            total_training_time_s=total_time,
            
            # Queue metrics
            avg_queue_size=statistics.mean(queue_sizes) if queue_sizes else 0,
            max_queue_size=max(queue_sizes) if queue_sizes else 0,
            queue_utilization=statistics.mean(queue_sizes) / self.config.max_queue_size if queue_sizes else 0,
            
            # Staleness metrics
            param_version_updates=self.current_param_version,
            staleness_drops=stats.get("dropped_samples", 0),
        )
        
        return metrics


def print_training_loop_results(results: List[TrainingLoopMetrics]):
    """Print detailed comparison of training loop results"""
    if not results:
        print("No results to display")
        return
    
    print("\n" + "="*100)
    print("TRAINING LOOP BENCHMARK RESULTS")
    print("="*100)
    
    if len(results) == 1:
        result = results[0]
        print(f"\nImplementation: {result.implementation}")
        print("-"*100)
        print(f"\n📊 PRODUCTION METRICS")
        print(f"  Total training time: {result.total_training_time_s:.2f}s")
        print(f"  Training iterations: {result.training_iterations_completed}")
        print(f"  Avg iteration time: {result.avg_iteration_time_s:.2f}s")
        print(f"\n🔄 PRODUCER METRICS (Rollout Workers)")
        print(f"  Samples generated: {result.total_samples_generated:,}")
        print(f"  Samples put: {result.total_samples_put:,}")
        print(f"  Samples dropped: {result.samples_dropped:,}")
        print(f"  Producer throughput: {result.producer_throughput:.2f} samples/sec")
        print(f"  Avg put latency: {result.avg_put_latency_us:.2f} µs")
        print(f"\n📥 CONSUMER METRICS (Trainer)")
        print(f"  Batches collected: {result.total_batches_collected}")
        print(f"  Samples consumed: {result.total_samples_consumed:,}")
        print(f"  Consumer throughput: {result.consumer_throughput:.2f} samples/sec")
        print(f"  Avg batch collection time: {result.avg_batch_collection_time_ms:.2f} ms")
        print(f"\n⏱️  END-TO-END LATENCY")
        print(f"  Mean: {result.avg_e2e_latency_ms:.2f} ms")
        print(f"  P50:  {result.p50_e2e_latency_ms:.2f} ms")
        print(f"  P95:  {result.p95_e2e_latency_ms:.2f} ms")
        print(f"  P99:  {result.p99_e2e_latency_ms:.2f} ms")
        print(f"\n📊 QUEUE METRICS")
        print(f"  Avg queue size: {result.avg_queue_size:.1f}")
        print(f"  Max queue size: {result.max_queue_size}")
        print(f"  Queue utilization: {result.queue_utilization*100:.1f}%")
        print(f"\n⚠️  STALENESS METRICS")
        print(f"  Parameter updates: {result.param_version_updates}")
        print(f"  Staleness drops: {result.staleness_drops}")
        return
    
    # Comparison mode
    lock_result = results[0]
    fast_result = results[1]
    
    print("\n" + "="*100)
    print("📊 PRODUCTION METRICS COMPARISON")
    print("-"*100)
    print(f"{'Metric':<40} {'Lock-based':<25} {'FastMQ':<25} {'Improvement':<20}")
    print("-"*100)
    
    def compare(name: str, lock_val: float, fast_val: float, unit: str = "", inverse: bool = False):
        if inverse:  # For latency (lower is better)
            improvement = lock_val / fast_val if fast_val > 0 else 0
            improvement_str = f"{improvement:.2f}x faster" if improvement > 1 else f"{1/improvement:.2f}x slower"
        else:  # For throughput (higher is better)
            improvement = fast_val / lock_val if lock_val > 0 else 0
            improvement_str = f"{improvement:.2f}x better" if improvement > 1 else f"{1/improvement:.2f}x worse"
        
        print(f"{name:<40} {lock_val:>12,.2f} {unit:<10} {fast_val:>12,.2f} {unit:<10} {improvement_str}")
    
    compare("Total training time", lock_result.total_training_time_s, fast_result.total_training_time_s, "s", inverse=True)
    compare("Avg iteration time", lock_result.avg_iteration_time_s, fast_result.avg_iteration_time_s, "s", inverse=True)
    
    print("\n" + "="*100)
    print("🔄 PRODUCER THROUGHPUT & LATENCY")
    print("-"*100)
    compare("Producer throughput", lock_result.producer_throughput, fast_result.producer_throughput, "samples/s")
    compare("Avg put latency", lock_result.avg_put_latency_us, fast_result.avg_put_latency_us, "µs", inverse=True)
    print(f"{'Samples dropped (Lock)':<40} {lock_result.samples_dropped:>12,}")
    print(f"{'Samples dropped (FastMQ)':<40} {fast_result.samples_dropped:>12,}")
    
    print("\n" + "="*100)
    print("📥 CONSUMER THROUGHPUT & BATCH COLLECTION")
    print("-"*100)
    compare("Consumer throughput", lock_result.consumer_throughput, fast_result.consumer_throughput, "samples/s")
    compare("Avg batch collection time", lock_result.avg_batch_collection_time_ms, fast_result.avg_batch_collection_time_ms, "ms", inverse=True)
    
    print("\n" + "="*100)
    print("⏱️  END-TO-END LATENCY (lower is better)")
    print("-"*100)
    compare("Mean E2E latency", lock_result.avg_e2e_latency_ms, fast_result.avg_e2e_latency_ms, "ms", inverse=True)
    compare("P50 E2E latency", lock_result.p50_e2e_latency_ms, fast_result.p50_e2e_latency_ms, "ms", inverse=True)
    compare("P95 E2E latency", lock_result.p95_e2e_latency_ms, fast_result.p95_e2e_latency_ms, "ms", inverse=True)
    compare("P99 E2E latency", lock_result.p99_e2e_latency_ms, fast_result.p99_e2e_latency_ms, "ms", inverse=True)
    
    print("\n" + "="*100)
    print("📊 QUEUE UTILIZATION")
    print("-"*100)
    print(f"{'Avg queue size (Lock)':<40} {lock_result.avg_queue_size:>12,.1f}")
    print(f"{'Avg queue size (FastMQ)':<40} {fast_result.avg_queue_size:>12,.1f}")
    print(f"{'Queue utilization (Lock)':<40} {lock_result.queue_utilization*100:>12,.1f} %")
    print(f"{'Queue utilization (FastMQ)':<40} {fast_result.queue_utilization*100:>12,.1f} %")
    
    print("\n" + "="*100)
    print("📈 SUMMARY")
    print("-"*100)
    training_speedup = lock_result.total_training_time_s / fast_result.total_training_time_s
    throughput_improvement = fast_result.producer_throughput / lock_result.producer_throughput
    latency_improvement = lock_result.avg_e2e_latency_ms / fast_result.avg_e2e_latency_ms
    
    print(f"Training Time Speedup:       {training_speedup:.2f}x")
    print(f"Producer Throughput Boost:   {throughput_improvement:.2f}x")
    print(f"Consumer Throughput Boost:   {fast_result.consumer_throughput / lock_result.consumer_throughput:.2f}x")
    print(f"E2E Latency Improvement:     {latency_improvement:.2f}x")
    print("="*100)


async def main(config: TrainingLoopConfig = None):
    """Main benchmark runner"""
    if config is None:
        config = TrainingLoopConfig()
    
    print("="*70)
    print("TRAINING LOOP BENCHMARK")
    print("="*70)
    print("\nSimulating fully async training loop pattern:")
    print("- Multiple rollout workers (producers) generating samples continuously")
    print("- Single trainer (consumer) collecting batches and training")
    print("- Periodic parameter updates triggering staleness drops")
    print()
    print("Configuration:")
    print(f"  Rollout workers: {config.num_rollout_workers}")
    print(f"  Samples per worker: {config.samples_per_worker}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Training iterations: {config.training_iterations}")
    print(f"  Sample generation delay: {config.sample_generation_delay_ms} ms")
    print(f"  Training delay per batch: {config.training_delay_ms} ms")
    print(f"  Staleness threshold: {config.staleness_threshold}")
    print()
    
    # Initialize Ray
    if not ray.is_initialized():
        print("Initializing Ray...")
        ray.init(ignore_reinit_error=True)
        print("Ray initialized\n")
    
    results = []
    
    # Benchmark 1: Lock-based implementation
    print("\n" + "="*70)
    print("BENCHMARK 1: Lock-based Implementation")
    print("="*70)
    
    sim_lock = TrainingLoopSimulator(
        MessageQueueLock,
        ClientLock,
        "Lock-based Queue",
        config
    )
    await sim_lock.setup()
    
    try:
        result_lock = await asyncio.wait_for(
            sim_lock.run_training_loop(),
            timeout=600.0  # 10 minutes max
        )
        results.append(result_lock)
    except asyncio.TimeoutError:
        print(f"\n❌ Lock-based training loop timed out!")
    except Exception as e:
        print(f"\n❌ Lock-based training loop failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await sim_lock.teardown()
    
    # Benchmark 2: FastMQ implementation
    if FASTMQ_AVAILABLE:
        print("\n" + "="*70)
        print("BENCHMARK 2: FastMQ Implementation")
        print("="*70)
        
        sim_fast = TrainingLoopSimulator(
            MessageQueueFast,
            ClientFast,
            "FastMQ Queue",
            config
        )
        await sim_fast.setup(
            num_shards=config.num_shards,
            shard_capacity_bytes=config.shard_capacity_bytes
        )
        
        try:
            result_fast = await asyncio.wait_for(
                sim_fast.run_training_loop(),
                timeout=600.0  # 10 minutes max
            )
            results.append(result_fast)
        except asyncio.TimeoutError:
            print(f"\n❌ FastMQ training loop timed out!")
        except Exception as e:
            print(f"\n❌ FastMQ training loop failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            await sim_fast.teardown()
    else:
        print("\n⚠️  FastMQ not available - skipping FastMQ benchmark")
    
    # Print results
    print_training_loop_results(results)
    
    # Cleanup
    print("\nCleaning up...")
    if ray.is_initialized():
        ray.shutdown()
    
    print("\n✅ Training loop benchmark complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Benchmark message queues with realistic training loop simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Small/quick benchmark
  python benchmark_training_loop.py --preset small
  
  # Realistic production-like benchmark
  python benchmark_training_loop.py --preset realistic
  
  # Realistic with specific number of workers
  python benchmark_training_loop.py --preset realistic --num-rollout-workers 16
  
  # Stress test
  python benchmark_training_loop.py --preset stress
  
  # Custom configuration
  python benchmark_training_loop.py --num-rollout-workers 8 --batch-size 512 --training-iterations 15
        """
    )
    
    parser.add_argument(
        '--preset',
        choices=['default', 'small', 'realistic', 'stress'],
        default='default',
        help='Use a preset configuration'
    )
    parser.add_argument(
        '--num-rollout-workers',
        type=int,
        help='Number of rollout workers (producers)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Batch size for training'
    )
    parser.add_argument(
        '--training-iterations',
        type=int,
        help='Number of training iterations'
    )
    parser.add_argument(
        '--samples-per-worker',
        type=int,
        help='Total samples each worker generates'
    )
    parser.add_argument(
        '--num-shards',
        type=int,
        help='Number of FastMQ shards (should match rollout workers)'
    )
    
    args = parser.parse_args()
    
    # Select configuration
    if args.preset == 'small':
        config = TrainingLoopConfig.small()
    elif args.preset == 'realistic':
        num_workers = args.num_rollout_workers or 8
        config = TrainingLoopConfig.realistic(num_rollout_workers=num_workers)
    elif args.preset == 'stress':
        config = TrainingLoopConfig.stress()
    else:
        config = TrainingLoopConfig()
    
    # Override with CLI arguments
    if args.num_rollout_workers is not None:
        config.num_rollout_workers = args.num_rollout_workers
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.training_iterations is not None:
        config.training_iterations = args.training_iterations
    if args.samples_per_worker is not None:
        config.samples_per_worker = args.samples_per_worker
    if args.num_shards is not None:
        config.num_shards = args.num_shards
    
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
