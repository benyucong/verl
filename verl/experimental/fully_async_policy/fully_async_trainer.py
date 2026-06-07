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
from datetime import datetime
from typing import Any

import ray
from omegaconf import OmegaConf, open_dict
from tqdm import tqdm

from verl import DataProto
from verl.checkpoint_engine import CheckpointEngineManager
from verl.experimental.fully_async_policy.detach_utils import (
    MetricsAggregator,
    assemble_batch_from_chunk_samples,
    assemble_batch_from_rollout_samples,
    choose_chunk_actor_mini_batch_size,
    coalesce_contiguous_chunk_samples,
    flatten_chunk_constituents,
    get_chunk_coalescing_config,
    get_chunk_coalescing_drain_multiplier,
    get_chunk_batch_memory_limits_from_env,
    get_chunk_row_count,
    get_chunk_staleness_threshold,
    get_chunk_token_budget,
    get_chunk_token_size,
    get_optimizer_step_token_budget,
    iter_chunk_constituents,
    is_chunk_data_path_enabled,
    pad_chunk_dataproto_to_response_width,
    select_chunk_samples_for_train_batch,
)
from verl.experimental.fully_async_policy.message_queue import MessageQueueClient
from verl.experimental.fully_async_policy.starvation import (
    get_max_starvation_resets,
    get_optimizer_step_max_fit_steps,
    get_starvation_escape_enabled,
    get_validate_every_flush,
    starvation_stall_detected,
)
from verl.experimental.fully_async_policy.opd_stage0_trace import (
    is_enabled as _stage0_is_enabled,
    trace_chunk_event as _stage0_trace_chunk,
    trace_event as _stage0_trace,
)
from verl.experimental.fully_async_policy.chunk_sample import ChunkSample
from verl.experimental.separation.ray_trainer import SeparateRayPPOTrainer
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.tracking import Tracking

logger = logging.getLogger(__name__)


class TrainingStopException(Exception):
    """Exception raised to signal training should stop"""

    pass


class _StarvationFlush(Exception):
    """Internal signal raised ONLY in optimizer-step-token-budget mode when the rollouter
    is fully stalled (paused, nothing in flight, queue drained) while the trainer holds
    accumulated optimizer-step supervision. Caught in fit_step to force an early flush that
    bumps the policy version + resets staleness, breaking the budget-flush <-> staleness-
    pause deadlock. Never raised on the control path (budget<=0)."""

    pass


@ray.remote(num_cpus=10)
class FullyAsyncTrainer(SeparateRayPPOTrainer):
    """
    A fully asynchronous PPO trainer that obtains samples from a MessageQueue for training.
    Based on an improved implementation of OneStepOffRayTrainer
    """

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        device_name=None,
    ):
        # ==================== RayPPOTrainer config ====================

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.config = config

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert not self.hybrid_engine

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.config)

        self.use_rm = need_reward_model(self.config)

        # distillation config needed by _update_actor in ray_trainer.py
        from verl.trainer.distillation.losses import is_distillation_enabled

        if is_distillation_enabled(self.config.get("distillation")):
            self.distillation_config = omega_conf_to_dataclass(self.config.distillation)
        else:
            self.distillation_config = None

        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        self.ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self.use_prefix_grouper = self.config.actor_rollout_ref.actor.get("use_prefix_grouper", False)

        # ==================== SeparateRayPPOTrainer config ====================
        self.global_steps = 0
        self.epoch = 0
        self.max_steps_duration = 0
        self.progress_bar = None
        self.is_last_step = False
        self.prev_step_profile = False
        self.curr_step_profile = False
        self.next_step_profile = False
        self.last_val_metrics = {}
        self.metrics = {}
        self.timing_raw = {}
        # reward message
        self.future_reward = None
        self.reward_tensor = None
        self.reward_extra_infos_dict = {}

        self.logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        # ==================== fully async config ====================

        self.message_queue_client = None

        # Statistics
        self.local_trigger_step = 1
        self.processed_samples = 0
        self.processed_chunks = 0
        self.stale_trajectory_processed = 0
        self.stale_chunks_dropped = 0
        self._pending_chunk_samples: list[ChunkSample] = []
        self._chunk_queue_terminated = False
        self.current_param_version = 0
        self.total_train_steps = None
        self.progress_bar = None
        self.trigger_parameter_sync_step = config.async_training.trigger_parameter_sync_step
        self.last_ckpt_version = 0

        # ---- Control-plane decoupling: upstream-like token budget per optimizer step ----
        # When > 0, accumulate streamed-chunk supervision across multiple memory-safe
        # fit_steps until the accumulated train-token count reaches this budget, then do
        # ONE optimizer.step() + version increment + weight sync. Default 0 == OFF
        # (exact current per-fit-step behavior). See get_optimizer_step_token_budget().
        self._optimizer_step_token_budget = get_optimizer_step_token_budget(config)
        self._accumulated_train_batches: list[DataProto] = []
        self._accumulated_train_tokens = 0
        # In budget mode every (budget-gated) optimizer step is a sync boundary, so the
        # version bumps + reset_staleness fire once per accumulated step. We use a
        # trainer-LOCAL sync cadence and deliberately leave the rollouter-visible
        # config.async_training.trigger_parameter_sync_step untouched (it sizes the
        # queue / staleness budget -- a data-plane concern).
        self._sync_every = 1 if self._optimizer_step_token_budget > 0 else self.trigger_parameter_sync_step
        # Budget-mode deadlock-breaker + opt-in cadence knobs. All inert when budget<=0
        # (control arm); the starvation escape is additionally inert outside the exact
        # stall fingerprint, so validated healthy-m12 behavior is preserved. See starvation.py.
        self._starvation_escape_enabled = get_starvation_escape_enabled(config)
        self._max_fit_steps_per_flush = get_optimizer_step_max_fit_steps(config)
        self._validate_every_flush = get_validate_every_flush(config)
        # Bound for the empty-accumulation branch of the escape (direct rollouter resume
        # with no version bump). Reset to 0 on any forward progress; raises on the cap.
        self._max_starvation_resets = get_max_starvation_resets(config)
        self._consecutive_starvation_resets = 0
        if self._optimizer_step_token_budget > 0:
            _ppo_epochs = int(config.actor_rollout_ref.actor.get("ppo_epochs", 1))
            if _ppo_epochs != 1:
                print(
                    "[FullyAsyncTrainer][OptStepBudget] WARNING: ppo_epochs="
                    f"{_ppo_epochs} != 1 with optimizer_step_token_budget > 0. Each accumulated "
                    "mini-batch will run ppo_epochs optimizer steps, partially defeating the "
                    "one-step-per-budget intent. Set ppo_epochs=1 for the intended cadence.",
                    flush=True,
                )
            print(
                "[FullyAsyncTrainer][OptStepBudget] ENABLED: one optimizer step + version "
                f"bump + weight sync per {self._optimizer_step_token_budget} accumulated train "
                "tokens (data plane / chunk streaming unchanged).",
                flush=True,
            )
        self.train_role = Role.ActorRollout if config.async_training.use_trainer_do_validate else Role.Actor

        # required_samples use ppo_mini_batch_size*require_batches as the minimum number of samples.
        self.require_batches = config.async_training.require_batches
        self.required_samples = config.actor_rollout_ref.actor.ppo_mini_batch_size * self.require_batches
        total_gpus = (
            config.trainer.nnodes * config.trainer.n_gpus_per_node
            + config.rollout.nnodes * config.rollout.n_gpus_per_node
        )
        self.metrics_aggregator = MetricsAggregator(total_gpus=total_gpus)

        # Reference to rollouter for parameter synchronization
        self.rollouter = None
        self.checkpoint_manager = None

        # Hybrid checkpoint manager for trainer-side validation (use_trainer_do_validate)
        # Uses naive backend to sync weights from trainer to hybrid rollout replicas.
        # Initialized in _setup_hybrid_checkpoint_manager_and_sleep() via set_rollouter().
        self.hybrid_checkpoint_manager = None

    async def _setup_checkpoint_manager(self):
        """Setup checkpoint manager after rollouter is initialized"""
        replicas = await self.rollouter.get_replicas.remote()
        checkpoint_engine_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)
        self.checkpoint_manager = CheckpointEngineManager(
            config=checkpoint_engine_config, trainer=self.actor_wg, replicas=replicas
        )
        print("[FullyAsyncTrainer] Checkpoint manager initialized")

    async def _setup_hybrid_checkpoint_manager(self):
        """Setup hybrid checkpoint manager and perform initial sleep of hybrid replicas.

        When use_trainer_do_validate is enabled:
          1. Creates a CheckpointEngineManager with naive backend for trainer-side
             weight sync to hybrid rollout replicas.
          2. Fetches hybrid replicas from the rollouter's ALM (created during
             rollouter.init_workers()).
          3. Registers them with the hybrid CP manager and calls sleep_replicas()
             to release GPU memory for training.

        Must be called AFTER set_rollouter() so that self.rollouter is available,
        and AFTER rollouter.init_workers() so that hybrid replicas exist.
        This mirrors the colocate pattern in ray_trainer.py:882-889 but fetches
        replicas from the rollouter's ALM via RPC since they live on the rollout side.
        """
        if not self.config.async_training.use_trainer_do_validate:
            return

        # --- Part 1: Create hybrid CheckpointEngineManager with naive backend ---
        print("[FullyAsyncTrainer] Setting up hybrid checkpoint manager (naive backend)")

        # Create hybrid CheckpointEngineManager with naive backend.
        checkpoint_engine_cfg = self.config.actor_rollout_ref.rollout.checkpoint_engine
        original_backend = checkpoint_engine_cfg.backend
        with open_dict(checkpoint_engine_cfg):
            checkpoint_engine_cfg.backend = "naive"
        checkpoint_engine_config = omega_conf_to_dataclass(checkpoint_engine_cfg)

        self.hybrid_checkpoint_manager = CheckpointEngineManager(
            config=checkpoint_engine_config,
            trainer=self.actor_rollout_wg,
            replicas=[],  # Start empty; will be populated below
        )

        # Restore original backend value
        with open_dict(checkpoint_engine_cfg):
            checkpoint_engine_cfg.backend = original_backend

        print("[FullyAsyncTrainer] Hybrid checkpoint manager initialized (naive backend)")

        # --- Part 2: Fetch hybrid replicas from rollouter's ALM ---
        print("[FullyAsyncTrainer] Fetching hybrid replicas from rollouter...")
        hybrid_replicas_dict = ray.get(self.rollouter.get_all_hybrid_replicas.remote())
        print(
            f"[FullyAsyncTrainer] Got {len(hybrid_replicas_dict)} hybrid replicas: {list(hybrid_replicas_dict.keys())}"
        )

        if not hybrid_replicas_dict:
            print("[FullyAsyncTrainer] No hybrid replicas found, skipping initial sleep")
            return

        # --- Part 3: Register replicas and perform initial sleep ---
        for resource_id, replica in hybrid_replicas_dict.items():
            self.hybrid_checkpoint_manager.replicas.append(replica)
            print(
                f"[FullyAsyncTrainer] Registered '{resource_id}' "
                f"(mode={getattr(replica, 'rollout_mode', '?')}, "
                f"addr={getattr(replica, '_server_address', '?')})"
            )

        # Step 3: Sleep all hybrid replicas
        print(
            f"[FullyAsyncTrainer] Calling sleep_replicas() on "
            f"{len(self.hybrid_checkpoint_manager.replicas)} replicas..."
        )
        await self.hybrid_checkpoint_manager.sleep_replicas()
        print("[FullyAsyncTrainer] Initial sleep complete, GPU memory now owned by training engine")

    def set_message_queue_client(self, message_queue_client: MessageQueueClient):
        """Set message queue client"""
        self.message_queue_client = message_queue_client

    async def set_rollouter(self, rollouter):
        """Set rollouter reference and initialize all checkpoint managers."""
        self.rollouter = rollouter
        # Setup checkpoint manager after rollouter is set
        await self._setup_checkpoint_manager()
        await self._setup_hybrid_checkpoint_manager()

    def set_total_train_steps(self, total_training_steps):
        self.total_train_steps = total_training_steps

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

        self.progress_bar = tqdm(total=self.total_train_steps, initial=0, desc="Training Progress")

    def get_actor_wg(self):
        """Get actor worker group"""
        return self.actor_wg

    async def _get_samples_from_queue(self) -> tuple[None, None] | tuple[int, Any]:
        """
        Get samples from message queue and compose gen_batch_output
        Uses a loop to continuously collect samples until enough are gathered

        Returns:
            tuple: (epoch, batch_dict, gen_batch_output)
        """
        if is_chunk_data_path_enabled(self.config):
            return await self._get_chunks_from_queue()

        print(
            f"[FullyAsyncTrainer] Requesting {self.required_samples} samples from queue",
            flush=True,
        )

        # Collect samples using a simple loop calling get_sample
        consumer_start = time.time()
        queue_samples = []
        queue_len = 0
        while len(queue_samples) < self.required_samples:
            # Get a single sample and wait until there is a sample or None is received
            sample, queue_len = await self.message_queue_client.get_sample()

            if sample is None:
                print(
                    f"[FullyAsyncTrainer] Detected termination signal (None), stopping sample collection. "
                    f"Collected {len(queue_samples)}/{self.required_samples} samples"
                )
                break

            # Stage-0 profiling: record per-sample arrival time at trainer.
            # Peek the sample_id by unpickling only when tracing is enabled.
            if _stage0_is_enabled():
                try:
                    _peeked = ray.cloudpickle.loads(sample)
                    _sid = getattr(_peeked, "sample_id", "unknown")
                    _stage0_trace(
                        "get",
                        _sid,
                        role="trainer",
                        queue_len=int(queue_len) if queue_len is not None else -1,
                    )
                except Exception:
                    pass

            queue_samples.append(sample)

            if len(queue_samples) % 64 == 0:
                print(
                    f"[FullyAsyncTrainer] Collected {len(queue_samples)}/{self.required_samples} samples. "
                    f"mq_len: {queue_len}"
                )

        consumer_end = time.time()

        if not queue_samples or len(queue_samples) < self.required_samples:
            print("[FullyAsyncTrainer] not enough samples collected after loop")
            return None, None
        total_wait_time = consumer_end - consumer_start

        print(
            f"[FullyAsyncTrainer] Loop collection completed: {len(queue_samples)}/{self.required_samples} samples, "
            f"total wait time: {total_wait_time:.2f} seconds. "
            f"mq_len: {queue_len}"
        )

        queue_samples = [ray.cloudpickle.loads(x) for x in queue_samples]
        # Assemble batch - now working directly with RolloutSample objects
        if self.config.trainer.balance_batch:
            batch = assemble_batch_from_rollout_samples(queue_samples, self.tokenizer, self.config, self._balance_batch)
        else:
            batch = assemble_batch_from_rollout_samples(queue_samples, self.tokenizer, self.config, None)

        batch.meta_info["fully_async/total_wait_time"] = total_wait_time
        return 0, batch

    def _get_chunk_batch_divisor(self) -> int:
        """Return the row-count divisor required by trainer-side batch balancing."""
        if not self.config.trainer.balance_batch:
            return 1
        try:
            return max(1, int(self._get_dp_size(self.actor_rollout_wg, "actor")))
        except Exception as exc:
            fallback = max(1, int(self.config.trainer.nnodes) * int(self.config.trainer.n_gpus_per_node))
            print(
                "[FullyAsyncTrainer] Could not query actor DP size for chunk batching; "
                f"falling back to trainer GPU count {fallback}. Error: {exc}",
                flush=True,
            )
            return fallback

    def _trace_chunk_get_event(self, chunk: ChunkSample, queue_len: int | None, accepted: bool, drop_reason=None):
        meta = chunk.meta if isinstance(chunk.meta, dict) else {}
        source = meta.get("source", "unknown")
        row_id = meta.get("row_id", chunk.sample_id)
        chunk_get_ts = meta.get("_trainer_chunk_get_ts")
        if chunk_get_ts is None:
            chunk_get_ts = time.time()
            meta["_trainer_chunk_get_ts"] = chunk_get_ts
        policy_version_lag = self.current_param_version - int(chunk.policy_version)
        _stage0_trace_chunk(
            "chunk_get",
            chunk.sample_id,
            chunk.chunk_idx,
            role="trainer",
            n_tokens=chunk.n_tokens,
            token_offset=chunk.token_offset,
            policy_version=chunk.policy_version,
            is_final=chunk.is_final,
            row_id=row_id,
            source=source,
            parent_sample_id=meta.get("parent_sample_id", chunk.sample_id),
            trainer_policy_version=self.current_param_version,
            policy_version_lag=policy_version_lag,
            accepted=accepted,
            drop_reason=drop_reason,
            queue_len=int(queue_len) if queue_len is not None else -1,
            chunk_get_ts=float(chunk_get_ts),
        )
        if accepted and chunk.is_final:
            _stage0_trace(
                "get",
                chunk.sample_id,
                role="trainer",
                queue_len=int(queue_len) if queue_len is not None else -1,
            )

    def _drop_stale_chunk(self, chunk: ChunkSample, sigma: float, queue_len: int | None):
        meta = chunk.meta if isinstance(chunk.meta, dict) else {}
        source = meta.get("source", "unknown")
        row_id = meta.get("row_id", chunk.sample_id)
        policy_version_lag = self.current_param_version - int(chunk.policy_version)
        self.stale_chunks_dropped += 1
        self._trace_chunk_get_event(chunk, queue_len=queue_len, accepted=False, drop_reason="stale")
        _stage0_trace_chunk(
            "chunk_drop_stale",
            chunk.sample_id,
            chunk.chunk_idx,
            role="trainer",
            n_tokens=chunk.n_tokens,
            token_offset=chunk.token_offset,
            policy_version=chunk.policy_version,
            is_final=chunk.is_final,
            row_id=row_id,
            source=source,
            parent_sample_id=meta.get("parent_sample_id", chunk.sample_id),
            trainer_policy_version=self.current_param_version,
            current_version=self.current_param_version,
            policy_version_lag=policy_version_lag,
            accepted=False,
            drop_reason="stale",
            sigma=sigma,
        )

    def _min_train_chunks_for_divisor(self, batch_divisor: int) -> int:
        divisor = max(1, int(batch_divisor))
        return ((int(self.required_samples) + divisor - 1) // divisor) * divisor

    async def _get_chunks_from_queue(self) -> tuple[None, None] | tuple[int, Any]:
        """Collect trainer-visible chunks by token budget and assemble a batch."""
        chunk_tokens = get_chunk_token_size(self.config)
        token_budget = get_chunk_token_budget(self.config, chunk_tokens=chunk_tokens)
        sigma = get_chunk_staleness_threshold(self.config)
        batch_divisor = self._get_chunk_batch_divisor()
        configured_actor_mini_batch_size = int(self.config.actor_rollout_ref.actor.ppo_mini_batch_size) * int(
            self.config.actor_rollout_ref.rollout.get("n", 1)
        )
        train_row_divisor = batch_divisor
        min_train_chunks = self._min_train_chunks_for_divisor(batch_divisor)
        memory_limits = get_chunk_batch_memory_limits_from_env()
        coalescing_config = get_chunk_coalescing_config(self.config)
        drain_multiplier = get_chunk_coalescing_drain_multiplier(self.config)
        # Pool-level coalescing: when coalescing is on and a drain multiplier > 1 is
        # configured, keep pulling chunks that are *already sitting* in the queue past
        # the train token budget (never blocking on new arrivals) so the same-parent
        # coalescer sees more sibling chunks. Rows beyond the train budget are deferred.
        drain_token_budget = (
            int(token_budget * drain_multiplier)
            if coalescing_config.enabled and drain_multiplier > 1.0
            else token_budget
        )
        has_memory_budget = any(value is not None for value in memory_limits.values())
        selection_min_rows = batch_divisor if has_memory_budget else min_train_chunks
        print(
            f"[FullyAsyncTrainer] Requesting chunks from queue "
            f"(token_budget={token_budget}, min_chunks={min_train_chunks}, "
            f"batch_divisor={batch_divisor}, train_row_divisor={train_row_divisor}, sigma={sigma}, "
            f"max_chunk_rows={memory_limits['max_chunk_rows']}, "
            f"max_train_tokens={memory_limits['max_train_tokens']}, "
            f"max_effective_seq_len={memory_limits['max_effective_seq_len']}, "
            f"coalesce_contiguous={coalescing_config.enabled}, "
            f"max_coalesced_chunks={coalescing_config.max_coalesced_chunks}, "
            f"max_coalesced_effective_seq_len={coalescing_config.max_coalesced_effective_seq_len}, "
            f"coalesce_lookahead={coalescing_config.lookahead}, "
            f"drain_multiplier={drain_multiplier}, drain_token_budget={drain_token_budget})",
            flush=True,
        )

        consumer_start = time.time()
        queue_chunks: list[ChunkSample] = []
        queue_len = 0
        token_count = 0
        row_count = 0
        pending_chunks_before = len(self._pending_chunk_samples)
        pending_rows_before = sum(get_chunk_row_count(chunk) for chunk in self._pending_chunk_samples)
        if self._pending_chunk_samples:
            print(
                f"[FullyAsyncTrainer] Reusing {len(self._pending_chunk_samples)} deferred chunks "
                "from the previous batch",
                flush=True,
            )
        pending_chunks = self._pending_chunk_samples
        self._pending_chunk_samples = []
        for chunk in pending_chunks:
            queue_len = int(chunk.meta.get("_trainer_queue_len", -1)) if isinstance(chunk.meta, dict) else -1
            if chunk.is_stale(current_version=self.current_param_version, sigma=sigma):
                self._drop_stale_chunk(chunk, sigma=sigma, queue_len=queue_len)
                continue
            queue_chunks.append(chunk)
            token_count += int(chunk.n_tokens)
            row_count += get_chunk_row_count(chunk)

        terminated = self._chunk_queue_terminated
        # ``observed_queue_len`` tracks the residual queue depth reported by the most
        # recent pop (or an explicit size probe). It is only used to gate the optional
        # post-budget drain phase so we never block waiting for new chunks to arrive.
        observed_queue_len: int | None = None
        pool_at_minimum: int | None = None
        while not self._chunk_queue_terminated:
            minimum_met = token_count >= token_budget and row_count >= min_train_chunks
            if minimum_met:
                if pool_at_minimum is None:
                    pool_at_minimum = len(queue_chunks)
                # Train batch minimum is satisfied. Stop unless a drain multiplier > 1
                # lets us opportunistically pull already-queued residual chunks (so the
                # coalescer sees more siblings). Never block on an empty queue.
                if drain_token_budget <= token_budget or token_count >= drain_token_budget:
                    break
                if observed_queue_len is None:
                    observed_queue_len = await self.message_queue_client.get_queue_size()
                if observed_queue_len is None or observed_queue_len <= 0:
                    break

            # Starvation escape (budget mode): if we are about to block for a new chunk but
            # the rollouter is fully stalled and we already hold accumulated supervision,
            # unwind so fit_step can flush it early -> version bump + reset_staleness ->
            # rollouter resumes (breaks the budget-flush <-> staleness-pause deadlock). Cheap
            # gates first (no RPC); the authoritative queue/rollouter probes run only when we
            # are actually about to block. A (rare) false positive only yields a safe,
            # slightly-early flush of real accumulated data. Fully inert when budget<=0.
            if (
                self._starvation_escape_enabled
                and self._optimizer_step_token_budget > 0
                and not minimum_met
                and observed_queue_len is not None
                and observed_queue_len <= 0
            ):
                confirmed_qsize = await self.message_queue_client.get_queue_size()
                rollouter_paused, rollouter_active = await self._rollouter_pause_state()
                if starvation_stall_detected(
                    confirmed_queue_size=confirmed_qsize,
                    rollouter_paused=rollouter_paused,
                    rollouter_active_tasks=rollouter_active,
                ):
                    if self._accumulated_train_batches:
                        # We hold accumulated supervision: flush it early. The version bump +
                        # reset_staleness (in fit_step's handler) resumes the rollouter. Every
                        # such escape bumps the version => guaranteed forward progress.
                        # Preserve survivors (already passed the stale check) for the next
                        # round; mirrors the deferred-chunk reuse path (originals, re-coalesced).
                        if queue_chunks:
                            self._pending_chunk_samples = list(queue_chunks) + self._pending_chunk_samples
                        assert self._optimizer_step_token_budget > 0, "starvation flush only in budget mode"
                        print(
                            "[FullyAsyncTrainer][OptStepBudget] starvation detected (queue empty, "
                            f"rollouter paused, active_tasks={rollouter_active}); escaping to flush "
                            f"{len(self._accumulated_train_batches)} accumulated sub-batches early.",
                            flush=True,
                        )
                        raise _StarvationFlush()
                    # No accumulated supervision yet (e.g. mid-collection right after a flush,
                    # when reset_staleness pinned staleness to a now-stale active+queue value
                    # and the monitor won't self-resume). Nothing to train, so directly resume
                    # the rollouter (no version bump) so it dispatches its pending samples and
                    # chunks flow again. Bounded: the counter resets on any forward progress
                    # (see _maybe_flush_optimizer_step / _force_flush_accumulated); on the cap
                    # we raise rather than spin.
                    if self._consecutive_starvation_resets >= self._max_starvation_resets:
                        raise TrainingStopException(
                            "[FullyAsyncTrainer][OptStepBudget] starvation: rollouter failed to "
                            f"resume after {self._max_starvation_resets} direct resets with no "
                            "accumulated supervision to flush"
                        )
                    self._consecutive_starvation_resets += 1
                    print(
                        "[FullyAsyncTrainer][OptStepBudget] starvation with empty accumulation "
                        f"(queue empty, rollouter paused, active_tasks={rollouter_active}); directly "
                        f"resuming rollouter (reset {self._consecutive_starvation_resets}/"
                        f"{self._max_starvation_resets}).",
                        flush=True,
                    )
                    await self._reset_rollouter_staleness()
                    # Fall through to get_sample(): the rollouter is now dispatching, so a
                    # chunk will arrive and unblock us.

            result = await self.message_queue_client.get_sample()
            if result is None:
                payload, queue_len = None, 0
            else:
                payload, queue_len = result

            if payload is None:
                terminated = True
                self._chunk_queue_terminated = True
                print(
                    f"[FullyAsyncTrainer] Detected termination signal (None), stopping chunk collection. "
                    f"Collected {len(queue_chunks)} chunks / {token_count} tokens"
                )
                break

            observed_queue_len = int(queue_len) if queue_len is not None else 0

            chunk = ray.cloudpickle.loads(payload)
            if not isinstance(chunk, ChunkSample):
                raise TypeError(
                    "Chunk data path expected ChunkSample queue payloads, "
                    f"got {type(chunk).__name__}"
                )

            if isinstance(chunk.meta, dict):
                chunk.meta.setdefault("_trainer_chunk_get_ts", time.time())
                chunk.meta["_trainer_queue_len"] = int(queue_len) if queue_len is not None else -1
            is_stale = chunk.is_stale(current_version=self.current_param_version, sigma=sigma)
            if is_stale:
                self._drop_stale_chunk(chunk, sigma=sigma, queue_len=queue_len)
                continue

            queue_chunks.append(chunk)
            token_count += int(chunk.n_tokens)
            row_count += get_chunk_row_count(chunk)

            if len(queue_chunks) % 64 == 0:
                print(
                    f"[FullyAsyncTrainer] Collected {len(queue_chunks)} chunks / {row_count} rows / "
                    f"{token_count} chunk tokens. "
                    f"mq_len: {queue_len}"
                )

        consumer_end = time.time()

        if not queue_chunks or row_count < selection_min_rows:
            print(
                "[FullyAsyncTrainer] not enough chunks collected after loop "
                f"(rows={row_count}, min_rows={selection_min_rows}, terminated={terminated})"
            )
            return None, None
        total_wait_time = consumer_end - consumer_start

        coalescing_result = coalesce_contiguous_chunk_samples(
            queue_chunks,
            coalescing_config,
            max_effective_seq_len=memory_limits["max_effective_seq_len"],
        )
        candidate_chunks = coalescing_result.chunks
        coalescing_metrics = coalescing_result.metrics
        if coalescing_config.enabled:
            print(
                f"[FullyAsyncTrainer] Coalescing (lookahead={coalescing_config.lookahead}): "
                f"rows {coalescing_metrics['rows_before_coalesce']} -> "
                f"{coalescing_metrics['rows_after_coalesce']}, "
                f"groups={coalescing_metrics['coalesced_groups']}, "
                f"chunks_merged={coalescing_metrics['chunks_merged']}, "
                f"opportunities={coalescing_metrics['coalescing_opportunities_visible_in_window']}, "
                f"pool_coalesced={len(queue_chunks)}, "
                f"pool_residual_in_queue={max(int(queue_len), 0)}, "
                f"pool_drained_extra={max(len(queue_chunks) - pool_at_minimum, 0) if pool_at_minimum is not None else 0}, "
                f"prefix_reduction_frac="
                f"{coalescing_metrics['estimated_prefix_recompute_reduction_fraction']:.4f}, "
                f"rejects={{'diff_parent': {coalescing_metrics['merge_reject_different_parent']}, "
                f"'noncontig': {coalescing_metrics['merge_reject_noncontiguous_span']}, "
                f"'policy': {coalescing_metrics['merge_reject_policy_version']}, "
                f"'eff_len': {coalescing_metrics['merge_reject_effective_len_cap']}, "
                f"'no_teacher': {coalescing_metrics['merge_reject_missing_teacher_payload']}, "
                f"'fallback': {coalescing_metrics['merge_reject_fallback']}, "
                f"'outside_window': {coalescing_metrics['merge_reject_outside_lookahead']}}}",
                flush=True,
            )

        selection = select_chunk_samples_for_train_batch(
            candidate_chunks,
            batch_divisor=batch_divisor,
            min_rows=selection_min_rows,
            train_row_divisor=train_row_divisor,
            **memory_limits,
        )
        train_chunks = selection.train_chunks
        deferred_chunks = selection.deferred_chunks
        train_original_chunks = flatten_chunk_constituents(train_chunks)
        deferred_original_chunks = flatten_chunk_constituents(deferred_chunks)
        if deferred_chunks:
            self._pending_chunk_samples.extend(deferred_original_chunks)
        train_original_chunk_count = len(train_original_chunks)
        deferred_original_chunk_count = len(deferred_original_chunks)
        train_token_count = sum(int(chunk.n_tokens) for chunk in train_original_chunks)
        pending_chunks_after = len(self._pending_chunk_samples)
        pending_rows_after = sum(get_chunk_row_count(chunk) for chunk in self._pending_chunk_samples)
        actor_mini_batch_size = choose_chunk_actor_mini_batch_size(
            selection.usable_rows,
            selection.world_size,
            configured_actor_mini_batch_size,
        )
        batch_selection_metrics = {
            **selection.as_metrics(),
            "collected_chunks": len(queue_chunks),
            "usable_chunks": len(train_chunks),
            "deferred_chunks": len(deferred_chunks),
            "train_chunk_tokens": train_token_count,
            "selection_min_rows": selection_min_rows,
            "has_memory_budget": has_memory_budget,
            "configured_actor_mini_batch_size": configured_actor_mini_batch_size,
            "actor_mini_batch_size": actor_mini_batch_size,
            "pending_buffer_chunks_before": pending_chunks_before,
            "pending_buffer_rows_before": pending_rows_before,
            "pending_buffer_chunks_after": pending_chunks_after,
            "pending_buffer_rows_after": pending_rows_after,
            "selected_rows_after_coalesce": selection.usable_rows,
            "selected_original_chunks_after_coalesce": train_original_chunk_count,
            "deferred_rows_after_coalesce": selection.deferred_rows,
            "deferred_original_chunks_after_coalesce": deferred_original_chunk_count,
            "memory_budget_trimmed": selection.trimmed_by_memory_budget,
            "dp_divisibility_trimmed": selection.trimmed_by_dp_divisibility,
            "dynamic_actor_mini_batch_size": actor_mini_batch_size,
            # Pool-size diagnostic: quantify how much larger the coalescing pool
            # could be if the token-budget cut were moved after coalescing.
            # pool_size_coalesced is what the coalescer actually saw this round;
            # pool_residual_in_queue_at_cut is the MQ depth still available when
            # collection stopped (0 when the queue was drained / terminated).
            "pool_size_coalesced": len(queue_chunks),
            "pool_residual_in_queue_at_cut": max(int(queue_len), 0),
            "pool_size_available_estimate": len(queue_chunks) + max(int(queue_len), 0),
            # pool_drained_extra_chunks: residual chunks pulled past the train token
            # budget by the optional pool-level drain (0 when drain disabled). Quantifies
            # how many extra sibling chunks the coalescer got to consider this round.
            "pool_drained_extra_chunks": (
                max(len(queue_chunks) - pool_at_minimum, 0) if pool_at_minimum is not None else 0
            ),
        }
        trace_metrics = {**batch_selection_metrics, **coalescing_metrics}
        _stage0_trace("chunk_batch_select", "chunk_batch", role="trainer", **trace_metrics)
        if not train_chunks or selection.usable_rows < selection_min_rows:
            print(
                "[FullyAsyncTrainer] not enough DP-divisible chunks collected after loop "
                f"(collected_chunks={len(queue_chunks)}, candidate_rows={selection.collected_rows}, "
                f"usable_rows={selection.usable_rows}, usable_original_chunks={train_original_chunk_count}, "
                f"min_rows={selection_min_rows}, deferred_rows={selection.deferred_rows}, "
                f"deferred_original_chunks={deferred_original_chunk_count}, "
                f"trimmed_by_dp={selection.trimmed_by_dp_divisibility}, "
                f"trimmed_by_budget={selection.trimmed_by_memory_budget}, terminated={terminated})"
            )
            return None, None
        self.processed_chunks += train_original_chunk_count

        for chunk in train_chunks:
            for original in iter_chunk_constituents(chunk):
                q_len = int(original.meta.get("_trainer_queue_len", -1)) if isinstance(original.meta, dict) else -1
                self._trace_chunk_get_event(original, queue_len=q_len, accepted=True, drop_reason=None)

        print(
            f"[FullyAsyncTrainer] Chunk collection completed: {train_original_chunk_count} train chunks, "
            f"{selection.usable_rows} train rows, {train_token_count} chunk tokens, "
            f"estimated_train_tokens={selection.estimated_train_tokens}, "
            f"max_effective_seq_len={selection.max_effective_seq_len}, "
            f"deferred={deferred_original_chunk_count} chunks/{selection.deferred_rows} rows, "
            f"collected={len(queue_chunks)} chunks/{selection.collected_rows} candidate rows, "
            f"world_size={selection.world_size}, "
            f"train_row_divisor={selection.train_row_divisor}, "
            f"max_chunk_rows_budget={selection.max_chunk_rows_budget}, "
            f"max_train_tokens_budget={selection.max_train_tokens_budget}, "
            f"max_effective_seq_len_budget={selection.max_effective_seq_len_budget}, "
            f"coalesced_groups={coalescing_metrics['coalesced_groups']}, "
            f"chunks_merged_total={coalescing_metrics['chunks_merged_total']}, "
            f"rows_after_coalesce={coalescing_metrics['rows_after_coalesce']}, "
            f"estimated_prefix_recompute_reduction="
            f"{coalescing_metrics['estimated_prefix_recompute_reduction']}, "
            f"actor_mini_batch_size={actor_mini_batch_size}, "
            f"trimmed_by_dp={selection.trimmed_by_dp_divisibility}, "
            f"trimmed_by_budget={selection.trimmed_by_memory_budget}, "
            f"pending_before={pending_chunks_before}/{pending_rows_before}, "
            f"pending_after={pending_chunks_after}/{pending_rows_after}, "
            f"total wait time: {total_wait_time:.2f} seconds. "
            f"mq_len: {queue_len}"
        )

        if self.config.trainer.balance_batch:
            batch = assemble_batch_from_chunk_samples(train_chunks, self.tokenizer, self.config, self._balance_batch)
        else:
            batch = assemble_batch_from_chunk_samples(train_chunks, self.tokenizer, self.config, None)

        batch.meta_info["fully_async/total_wait_time"] = total_wait_time
        batch.meta_info["fully_async/count/dropped_stale_chunks"] = self.stale_chunks_dropped
        batch.meta_info["fully_async/count/processed_chunks"] = self.processed_chunks
        batch.meta_info["fully_async/chunk/deferred_chunks"] = deferred_original_chunk_count
        if actor_mini_batch_size != configured_actor_mini_batch_size:
            batch.meta_info["fully_async/chunk_batch/actor_mini_batch_size"] = actor_mini_batch_size
        for key, value in batch_selection_metrics.items():
            if value is not None:
                batch.meta_info[f"fully_async/chunk_batch/{key}"] = value
        for key, value in coalescing_metrics.items():
            if value is not None:
                batch.meta_info[f"fully_async/chunk_coalesce/{key}"] = value
        return 0, batch

    def _create_actor_rollout_classes(self):
        # create actor — always use Role.Actor (not ActorRollout) even when
        # use_trainer_do_validate is enabled. Rollout capability on trainer GPUs
        # is handled by ElasticAgentLoopManager's hybrid replicas.
        for role in [self.train_role]:
            resource_pool = self.resource_pool_manager.get_resource_pool(role)
            role_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[role],
                config=self.config.actor_rollout_ref,
                distillation_config=self.config.get("distillation"),
                role=str(role),
            )
            self.resource_pool_to_cls[resource_pool][str(role)] = role_cls

    def _create_reward_model_class(self):
        # In fully async mode, RM is managed by RewardLoopManager (standalone). Skip worker group creation for RM.
        pass

    def _init_models(self):
        if self.use_critic:
            self.critic_wg = self.all_wg[str(Role.Critic)]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = self.all_wg[str(Role.RefPolicy)]
            self.ref_policy_wg.init_model()

        self.actor_wg = self.all_wg[str(self.train_role)]
        self.actor_wg.init_model()
        self.actor_rollout_wg = self.actor_wg  # to be compatible with the functions that not be modified

    async def init_workers(self):
        """Initialize distributed training workers using Ray backend.
        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self._init_resource_pools()
        self._create_worker_classes()
        self._init_worker_groups()
        self._init_models()

    async def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        print("[FullyAsyncTrainer] Starting FullyAsyncTrainer...")
        if self.message_queue_client is None:
            raise ValueError("MessageQueue client not set. Call set_message_queue_client() first.")
        if self.rollouter is None:
            raise ValueError("rollouter not set. Call set_rollouter() first.")

        self.max_steps_duration = 0

        self.global_steps += 1

        self.prev_step_profile = False
        self.curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        self.next_step_profile = False

        # Use queue mode, no need for traditional dataloader iterator
        # Initialize to get the first batch of data
        while True:
            try:
                await self.fit_step()
            except TrainingStopException:
                print("[FullyAsyncTrainer] Training stopped by queue termination signal")
                break

        # Flush any trailing accumulated supervision (optimizer-step-budget mode) so the
        # last partial budget still trains before shutdown.
        if self._optimizer_step_token_budget > 0 and self._accumulated_train_batches:
            subs = self._accumulated_train_batches
            self._accumulated_train_batches = []
            self._accumulated_train_tokens = 0
            print(
                f"[FullyAsyncTrainer][OptStepBudget] shutdown flush: final optimizer step "
                f"over {len(subs)} accumulated sub-batches",
                flush=True,
            )
            self._run_accumulated_optimizer_step(subs)
            self._fit_update_local_step()
            await self._fit_update_weights()

        self.progress_bar.close()
        if self.current_param_version % self.config.trainer.test_freq != 0 or self.local_trigger_step > 1:
            await self._fit_update_weights()
            await self._fit_validate()
        self._fit_save_checkpoint(force=True)

    async def fit_step(self, batch_dict: dict = None):
        """
        Single-step training template method. Handles all logic for one training step.

        Flow:
        1. Pre-step processing -> 2. Get batch -> 3. Generate sequences ->
        4. Compute reward -> 5. Compute log_prob -> 6. Compute reward ->
        7. Compute advantage -> 8. Update critic -> 9. Update actor -> 10. Post-step processing

        Args:
            batch_dict: Raw data dictionary
        """
        self.metrics = {"training/global_step": self.global_steps, "training/epoch": self.epoch}
        self.timing_raw = {}
        # reward message
        self.future_reward = None
        self.reward_tensor = None
        self.reward_extra_infos_dict = {}

        self._fit_start_profile()

        did_step = True
        starvation_flush = False
        with marked_timer("step", self.timing_raw):
            try:
                batch = await self._fit_generate(None)
            except _StarvationFlush:
                # Budget mode only (asserted at the raise site): the rollouter starved while
                # accumulated supervision was pending. Skip the data plane for this step and
                # flush the accumulation now so the version bumps + reset_staleness un-pauses
                # the rollouter. The OFF/control path can never raise this -> unchanged.
                starvation_flush = True
            if starvation_flush:
                step_batch = self._force_flush_accumulated()  # subs guaranteed non-empty
                self._fit_update_local_step()
                await self._fit_update_weights()
                self._fit_dump_data(step_batch)
                batch = step_batch
                did_step = True
            else:
                batch = self._fit_compute_reward(batch)
                batch = self._fit_compute_log_prob(batch)
                batch = self._fit_compute_ref_log_prob(batch)
                batch = self._fit_compute_critic(batch)
                batch = self._fit_compute_advantage(batch)
                batch = self._fit_update_critic(batch)
                train_batch = batch
                self._trace_chunk_train_events(train_batch, "chunk_train_start")
                if self._optimizer_step_token_budget <= 0:
                    # Control plane OFF (default): one optimizer step + version bump + sync
                    # per fit_step, exactly as before.
                    batch = self._fit_update_actor(batch)
                    self._trace_chunk_train_events(train_batch, "chunk_train_end")
                    self._fit_update_local_step()
                    await self._fit_update_weights()
                    self._fit_dump_data(batch)
                else:
                    # Control plane decoupled: accumulate streamed-chunk supervision and do
                    # ONE optimizer step + version bump + sync per upstream-like token budget.
                    # The data plane above (generate/score/logprob/advantage) still ran this
                    # fit_step, so chunks keep arriving early and the queue is not backpressured.
                    did_step, step_batch = self._maybe_flush_optimizer_step(train_batch)
                    self._trace_chunk_train_events(train_batch, "chunk_train_end")
                    if did_step:
                        self._fit_update_local_step()
                        await self._fit_update_weights()
                        self._fit_dump_data(step_batch)
                        batch = step_batch

        # Validation / checkpointing align with an actual optimizer step + version bump
        # (every fit_step when OFF; only on a flush when the budget mode is active).
        if did_step:
            await self._fit_validate()
            self._fit_save_checkpoint()
        self._fit_stop_profile()
        self._fit_collect_metrics(batch)
        self._fit_postprocess_step()

    @staticmethod
    def _count_train_tokens(batch: DataProto) -> int:
        """Trainable response tokens in an assembled chunk batch (the optimizer-step budget unit)."""
        try:
            if batch.batch is not None and "response_mask" in batch.batch:
                return int(batch.batch["response_mask"].sum().item())
        except Exception:
            pass
        return 0

    def _run_accumulated_optimizer_step(self, subs: list[DataProto]):
        """Fork 2: perform ONE optimizer step over `subs` (memory-safe per-fit-step chunk
        batches) via gradient accumulation, keeping peak memory at a single sub-batch.

        Each sub-batch is forwarded/backwarded in its own update_actor call (peak = one
        sub-batch, exactly the proven memory-safe size). We zero-grad only on the first
        sub-batch and optimizer.step() only on the last, accumulating grads in between.
        Each sub-batch's loss is scaled by its token share (N_i / N_total): the engine
        normalizes each sub-batch's loss by its own token count (token-mean), so the scale
        cancels that and the summed gradient equals the global token-mean over all
        sub-batches -- upstream-equivalent, no loss/teacher/mask semantic change. Each
        sub-batch is run as ONE mini-batch (actor_mini_batch_size = its row count) so the
        zero/step gating is per-call rather than split across internal mini-batches."""
        n = len(subs)
        token_counts = [max(1, self._count_train_tokens(b)) for b in subs]
        total_tokens = max(1, sum(token_counts))
        step_batch = None
        for i, b in enumerate(subs):
            b.meta_info["acc_zero_grad"] = i == 0
            b.meta_info["acc_do_step"] = i == n - 1
            b.meta_info["acc_loss_scale"] = float(token_counts[i] / total_tokens)
            # One mini-batch per sub-batch (its rows are DP-divisible by construction) so
            # train_mini_batch issues exactly one engine.train_batch call for it.
            b.meta_info["fully_async/chunk_batch/actor_mini_batch_size"] = int(len(b))
            step_batch = self._fit_update_actor(b)
        return step_batch

    def _maybe_flush_optimizer_step(self, train_batch: DataProto):
        """Budget mode: accumulate `train_batch`; when accumulated trainable tokens reach
        `self._optimizer_step_token_budget`, do ONE optimizer step over the accumulated
        sub-batches via gradient accumulation (memory-safe). Returns (did_step,
        step_metrics_batch_or_None). Accumulation defers only the optimizer step -- the
        queue was already drained for these batches, so the rollouter is not backpressured."""
        # A chunk batch was successfully assembled this fit_step -> forward progress; clear the
        # empty-accumulation starvation-reset counter.
        self._consecutive_starvation_resets = 0
        self._accumulated_train_batches.append(train_batch)
        self._accumulated_train_tokens += self._count_train_tokens(train_batch)
        budget_met = self._accumulated_train_tokens >= self._optimizer_step_token_budget
        # Opt-in safety cap (default 0 = OFF): bound version lag by forcing a flush after
        # at most _max_fit_steps_per_flush accumulated chunk-batch fit_steps. Default leaves
        # the validated pure-token-budget cadence untouched.
        cap_met = (
            self._max_fit_steps_per_flush > 0
            and len(self._accumulated_train_batches) >= self._max_fit_steps_per_flush
        )
        if not budget_met and not cap_met:
            return False, None
        subs = self._accumulated_train_batches
        acc_tokens = self._accumulated_train_tokens
        self._accumulated_train_batches = []
        self._accumulated_train_tokens = 0
        trigger = "budget" if budget_met else f"fit-step cap={self._max_fit_steps_per_flush}"
        print(
            f"[FullyAsyncTrainer][OptStepBudget] gradient-accumulating ONE optimizer step over "
            f"{len(subs)} sub-batches ({acc_tokens} train tokens, budget={self._optimizer_step_token_budget}, "
            f"trigger={trigger})",
            flush=True,
        )
        step_batch = self._run_accumulated_optimizer_step(subs)
        return True, step_batch

    async def _rollouter_pause_state(self) -> tuple[bool, int]:
        """RPC: atomic (paused, in-flight active_tasks) snapshot from the rollouter for the
        starvation-escape check. Read-only."""
        state = await asyncio.wrap_future(self.rollouter.get_generation_state.remote().future())
        return bool(state.get("paused")), int(state.get("active_tasks", 0))

    async def _reset_rollouter_staleness(self):
        """Directly reset the rollouter's staleness counter (RPC) to resume generation
        WITHOUT a version bump. Used by the starvation escape only when there is no
        accumulated supervision to flush, so the rollouter dispatches its pending samples and
        chunks flow again. Safe: it does not advance the policy version, so newly generated
        chunks stay fresh (not stale-dropped)."""
        timing_raw = await asyncio.wrap_future(self.rollouter.reset_staleness.remote().future())
        try:
            self.logger.log(data=timing_raw, step=self.current_param_version)
        except Exception:
            pass

    def _force_flush_accumulated(self) -> DataProto:
        """Starvation escape: run ONE optimizer step over the currently accumulated sub-
        batches regardless of whether the token budget is met (it is not -- the rollouter
        starved first). Caller guarantees a non-empty accumulation, so the returned step
        batch is always a real DataProto (never None). Uses the same proven token-weighted,
        memory-safe gradient accumulation as a normal budget flush, so it is numerically a
        valid (smaller) optimizer step."""
        assert self._accumulated_train_batches, "force-flush invoked with no accumulated batches"
        self._consecutive_starvation_resets = 0  # a flush is forward progress
        subs = self._accumulated_train_batches
        acc_tokens = self._accumulated_train_tokens
        self._accumulated_train_batches = []
        self._accumulated_train_tokens = 0
        print(
            f"[FullyAsyncTrainer][OptStepBudget] STARVATION FLUSH: early optimizer step over "
            f"{len(subs)} sub-batches ({acc_tokens}/{self._optimizer_step_token_budget} train "
            "tokens) -- rollouter starved; bumping version + reset_staleness to resume it.",
            flush=True,
        )
        return self._run_accumulated_optimizer_step(subs)

    async def _fit_generate(self, batch: DataProto = None) -> DataProto | None:
        metrics = self.metrics
        timing_raw = self.timing_raw
        with marked_timer("gen", timing_raw, color="red"):
            epoch, batch = await self._get_samples_from_queue()
            if batch is None:
                raise TrainingStopException("Training terminated: queue returned None")
            self._collect_metrics_from_samples(batch, metrics)
        batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
        return batch

    def _trace_chunk_train_events(self, batch: DataProto, event: str):
        """Trace chunk train start/end for Stage-1 analyzer compatibility."""
        chunk_samples = batch.meta_info.get("chunk_samples") if hasattr(batch, "meta_info") else None
        if not chunk_samples:
            return
        for chunk in chunk_samples:
            policy_version = int(chunk["policy_version"])
            policy_version_lag = self.current_param_version - policy_version
            _stage0_trace_chunk(
                event,
                chunk["sample_id"],
                chunk["chunk_idx"],
                role="trainer",
                n_tokens=chunk["n_tokens"],
                token_offset=chunk["token_offset"],
                policy_version=policy_version,
                is_final=chunk["is_final"],
                row_id=chunk.get("row_id", chunk["sample_id"]),
                source=chunk.get("source", "unknown"),
                parent_sample_id=chunk.get("parent_sample_id", chunk["sample_id"]),
                trainer_policy_version=self.current_param_version,
                policy_version_lag=policy_version_lag,
                accepted=True,
            )

    def _compute_old_log_prob(self, batch: DataProto):
        """
        If algorithm.rollout_correction.bypass_mode is False,
        use model engine and first version model params to re-calculate old_log_prob.

        If local_trigger_step == 1, load the training engine's parameters to the CPU
          and save a copy for subsequent MIS use.

        If local_trigger_step == 2, 3, ..., restore the parameters of version 1 to calculate the old_log_prob,
        then restore the parameters of the current version.
        """
        if self.local_trigger_step == 1:
            self.actor_rollout_wg.save_model_to_cpu(1)
            old_log_prob, old_log_prob_mfu = super()._compute_old_log_prob(batch)
        else:
            self.actor_rollout_wg.save_model_to_cpu(self.local_trigger_step)
            self.actor_rollout_wg.restore_model_from_cpu(1)
            old_log_prob, old_log_prob_mfu = super()._compute_old_log_prob(batch)
            self.actor_rollout_wg.restore_model_from_cpu(self.local_trigger_step)
            self.actor_rollout_wg.clear_cpu_model(self.local_trigger_step)
        return old_log_prob, old_log_prob_mfu

    def _fit_update_local_step(self):
        time_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(
            f"[FullyAsyncTrainer] global_steps: {self.global_steps} "
            f"local_trigger_step: {self.local_trigger_step} "
            f"sync_every: {self._sync_every} "
            f"trigger_parameter_sync_step: {self.trigger_parameter_sync_step} "
            f"{time_str}"
        )
        # In optimizer-step-budget mode self._sync_every == 1, so each (budget-gated)
        # optimizer step bumps the version + triggers one weight sync. Otherwise this is
        # the unchanged per-fit-step cadence keyed on trigger_parameter_sync_step.
        if self.local_trigger_step < self._sync_every:
            self.local_trigger_step += 1
        else:
            self.current_param_version += 1
            self.local_trigger_step = 1

    async def _fit_update_weights(self):
        if self.local_trigger_step != 1:
            return

        _sync_start_ts = time.time()
        _stage0_trace(
            "param_sync_start",
            f"sync_{self.current_param_version}",
            role="trainer",
            param_version=int(self.current_param_version),
            global_steps=int(self.global_steps),
            param_sync_start_ts=_sync_start_ts,
            _trace_ts=_sync_start_ts,
        )
        with marked_timer("timing_s/param_sync", self.timing_raw):
            await self.checkpoint_manager.update_weights(global_steps=self.current_param_version)
        _sync_end_ts = time.time()
        _stage0_trace(
            "param_sync_end",
            f"sync_{self.current_param_version}",
            role="trainer",
            param_version=int(self.current_param_version),
            global_steps=int(self.global_steps),
            param_sync_end_ts=_sync_end_ts,
            param_sync_latency_s=float(self.timing_raw.get("timing_s/param_sync", _sync_end_ts - _sync_start_ts)),
            _trace_ts=_sync_end_ts,
        )
        print(
            f"[FullyAsyncTrainer] _fit_update_weights, "
            f"timing_s/param_sync: {self.timing_raw['timing_s/param_sync']:.4f} seconds "
            f"self.current_param_version: {self.current_param_version}"
        )

        # Reset staleness in rollouter
        timing_raw = await asyncio.wrap_future(self.rollouter.reset_staleness.remote().future())
        self.logger.log(
            data=timing_raw,
            step=self.current_param_version,
        )

        # Log aggregated training metrics
        self.logger.log(
            data=self.metrics_aggregator.get_aggregated_metrics(),
            step=self.current_param_version,
        )
        self.metrics_aggregator.reset()

    async def _fit_validate(self, val_before_train=False):
        if self.local_trigger_step != 1:
            return

        # Check if validation is needed
        need_validate = (
            self.config.trainer.test_freq > 0
            and self.current_param_version % self.config.trainer.test_freq == 0
            and self.current_param_version > 0
        )
        # Opt-in (default OFF): in budget mode, validate on every version-bumping flush
        # regardless of test_freq so budget arms produce dense val-vs-time/tokens curves.
        # Default leaves the existing test_freq cadence untouched.
        if (
            self._validate_every_flush
            and self._optimizer_step_token_budget > 0
            and self.current_param_version > 0
        ):
            need_validate = True
        # Skip validation if not needed and not validation before training
        if not need_validate and not val_before_train:
            return
        # Execute validation
        if self.config.async_training.use_trainer_do_validate:
            await self._trainer_side_validate()
        else:
            val_metrics = await self.rollouter.do_validate.remote()
            self.logger.log(data=val_metrics, step=self.current_param_version)

    async def _trainer_side_validate(self):
        """Run trainer-side validation using hybrid rollout replicas."""
        print("[FullyAsyncTrainer] _trainer_side_validate === START ===")
        validate_start = time.time()
        # ================================================================
        # Phase 1: Switch ALL trainer GPUs to ROLLOUT mode
        # ================================================================
        phase_1_start = time.time()
        print("[FullyAsyncTrainer] Phase 1: Switching all GPUs to ROLLOUT mode")
        await self.hybrid_checkpoint_manager.update_weights(global_steps=self.current_param_version)
        await self.checkpoint_manager.abort_replicas()
        await self.hybrid_checkpoint_manager.abort_replicas()
        hybrid_replicas_dict = await self.rollouter.get_all_hybrid_replicas.remote()
        hybrid_resource_ids = list(hybrid_replicas_dict.keys())
        await self.rollouter.add_replicas.remote(hybrid_resource_ids)
        await self.checkpoint_manager.resume_generation_replicas()
        await self.hybrid_checkpoint_manager.resume_generation_replicas()
        print(f"[FullyAsyncTrainer] Phase 1 done ({time.time() - phase_1_start:.2f}s)")

        # ================================================================
        # Phase 2: Run validation via RPC to rollouter
        # ================================================================
        print("[FullyAsyncTrainer] Phase 2: Running validation")
        val_metrics = await self.rollouter.do_validate.remote()
        self.logger.log(data=val_metrics, step=self.current_param_version)

        # ================================================================
        # Phase 3: Switch hybrid GPUs back to TRAIN mode
        # ================================================================
        print("[FullyAsyncTrainer] Phase 3: Switching hybrid GPUs back to TRAIN mode")
        await self.checkpoint_manager.abort_replicas()
        await self.hybrid_checkpoint_manager.abort_replicas()
        # Batch remove all hybrid replicas from the load balancer in a single RPC.
        await self.rollouter.remove_replicas.remote(hybrid_resource_ids)
        await self.hybrid_checkpoint_manager.sleep_replicas()
        await self.checkpoint_manager.resume_generation_replicas()
        await self.hybrid_checkpoint_manager.resume_generation_replicas()

        total_time = time.time() - validate_start
        print(f"[FullyAsyncTrainer] _trainer_side_validate === END === (total: {total_time:.2f}s)")

    def _fit_save_checkpoint(self, force=False):
        if self.current_param_version == self.last_ckpt_version:
            return

        timing_raw = self.timing_raw
        # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
        esi_close_to_expiration = should_save_ckpt_esi(
            max_steps_duration=self.max_steps_duration,
            redundant_time=self.config.trainer.esi_redundant_time,
        )
        # Check if the conditions for saving a checkpoint are met.
        # The conditions include a mandatory condition (1) and
        # one of the following optional conditions (2/3/4):
        # 1. The save frequency is set to a positive value.
        # 2. It's the last training step.
        # 3. The current step number is a multiple of the save frequency.
        # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
        if self.config.trainer.save_freq > 0 and (
            force or self.current_param_version % self.config.trainer.save_freq == 0 or esi_close_to_expiration
        ):
            if esi_close_to_expiration:
                print("Force saving checkpoint: ESI instance expiration approaching.")
            with marked_timer("save_checkpoint", timing_raw, color="green"):
                # sleep replicas to avoid OOM during checkpoint saving
                self._save_checkpoint()
                self.last_ckpt_version = self.current_param_version

    def _fit_postprocess_step(self):
        self.global_steps += 1

        self.metrics_aggregator.add_step_metrics(
            metrics=self.metrics, sample_count=self.required_samples, timestamp=time.time()
        )

        if self.local_trigger_step == 1:
            self.progress_bar.update(1)

    def _save_checkpoint(self):
        # Warning: Currently, to align the training process and metrics of colocate,
        # we use current_param_version instead of global step.
        # This can be logically aligned with the original self.global_steps of colocate
        # and is used for metrics and ckpt. which means that the parameter synchronization
        # from trainer to rollouter will increase by 1 each time.

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.current_param_version}"
        )

        print(f"[FullyAsyncTrainer] local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(
                self.config.trainer.default_hdfs_dir, f"global_step_{self.current_param_version}", "actor"
            )
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "[FullyAsyncTrainer] Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.current_param_version, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.current_param_version}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path,
                critic_remote_path,
                self.current_param_version,
                max_ckpt_to_keep=max_critic_ckpt_to_keep,
            )
        ray.get(self.rollouter.save_checkpoint.remote(local_global_step_folder))
        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.current_param_version))

    async def load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"[FullyAsyncTrainer] Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.current_param_version = int(global_step_folder.split("global_step_")[-1])
        self.global_steps = self.current_param_version * self.trigger_parameter_sync_step + 1
        self.last_ckpt_version = self.current_param_version
        print(
            f"[FullyAsyncTrainer] Setting global step to {self.global_steps}, "
            f"current_param_version to {self.current_param_version}"
        )
        print(f"[FullyAsyncTrainer] Resuming from  {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        return self.current_param_version

    def _collect_metrics_from_samples(self, batch, metrics):
        """
        Collect metrics from samples
        """
        if hasattr(batch, "meta_info") and batch.meta_info:
            trajectory_param_versions = batch.meta_info["trajectory_param_versions"]
            stale_traj_count = sum(1 for v in trajectory_param_versions if self.current_param_version - v >= 1)
            self.stale_trajectory_processed += stale_traj_count
            metrics.update(
                {
                    "fully_async/count/stale_trajectory_processed": self.stale_trajectory_processed,
                    "fully_async/count/current_param_version": self.current_param_version,
                }
            )
            for key, value in batch.meta_info.items():
                if key.startswith("fully_async") or key.startswith("timing_s"):
                    metrics[key] = value
