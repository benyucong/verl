# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""Pure, dependency-free decision logic for the optimizer-step-token-budget
*starvation escape* (the fix for the m12 budget-flush <-> staleness-pause deadlock).

Kept stdlib-only (no torch / ray / verl imports) so the decision and config resolvers
are unit-testable on any CPU without the training stack. The async RPC plumbing that
*supplies* the inputs lives in fully_async_trainer.py / fully_async_rollouter.py.

Background: in optimizer-step-token-budget mode the trainer bumps the policy version
(and calls rollouter.reset_staleness) only on a budget flush. With a large budget and
long decodes the rollouter hits its staleness pause *before* the budget fills, stops
emitting chunks, and the trainer blocks in get_sample() waiting for a full batch -> the
budget never fills -> no version bump -> no reset_staleness -> circular wait. The escape:
when the rollouter is fully stalled (paused, nothing in flight) and the trainer holds
accumulated supervision pending an optimizer step, flush it early; that bumps the version
and resets staleness, un-pausing the rollouter.
"""
import os


def _int_env(name: str) -> int | None:
    """Parse an int env var; None if unset/blank/invalid (negative kept; caller clamps)."""
    value = os.environ.get(name, "").strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _flag_env(name: str) -> bool | None:
    """Parse a boolean env var; None if unset/unrecognised."""
    value = os.environ.get(name, "").strip().lower()
    if not value:
        return None
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return None


def get_starvation_escape_enabled(config=None) -> bool:
    """Whether the budget-mode starvation escape is active. Default ON: this is a
    deadlock fix, and it is fully inert outside budget mode (budget<=0) and outside the
    exact stall fingerprint, so it never perturbs the control arm or healthy m12. Set
    ``OPD_STARVATION_ESCAPE=0`` (or async_training.starvation_escape=false) to restore
    the old (deadlocking) behavior for A/B reproduction."""
    env = _flag_env("OPD_STARVATION_ESCAPE")
    if env is not None:
        return env
    if config is not None:
        return bool(config.async_training.get("starvation_escape", True))
    return True


def get_optimizer_step_max_fit_steps(config=None) -> int:
    """Optional safety cap (default 0 = OFF): force a budget-mode optimizer-step flush
    after at most this many accumulated chunk-batch fit_steps even if the token budget is
    not yet met. Bounds version lag deterministically (defense-in-depth on top of the
    starvation escape). 0 preserves the validated pure-token-budget m12 cadence.
    Resolution: env ``OPD_OPTIMIZER_STEP_MAX_FIT_STEPS`` then
    async_training.optimizer_step_max_fit_steps."""
    env = _int_env("OPD_OPTIMIZER_STEP_MAX_FIT_STEPS")
    if env is not None:
        return max(0, env)
    if config is not None:
        cfg = config.async_training.get("optimizer_step_max_fit_steps", 0)
        # Accept only a genuine non-negative int from config (YAML ints are native). Reject
        # float/str/None/bool/etc. -> off, so a misconfigured value never silently applies a
        # truncated or surprising cap (off preserves the validated pure-token-budget cadence).
        if isinstance(cfg, bool) or not isinstance(cfg, int):
            return 0
        return cfg if cfg > 0 else 0
    return 0


def get_validate_every_flush(config=None) -> bool:
    """When True (budget mode only), validate on every version-bumping flush regardless
    of test_freq -> dense val-vs-time/tokens curves for budget arms. Default False
    (unchanged cadence). Resolution: env ``OPD_VALIDATE_EVERY_FLUSH`` then
    async_training.validate_every_flush."""
    env = _flag_env("OPD_VALIDATE_EVERY_FLUSH")
    if env is not None:
        return env
    if config is not None:
        return bool(config.async_training.get("validate_every_flush", False))
    return False


def starvation_escape_decision(
    *,
    budget_enabled: bool,
    escape_enabled: bool,
    minimum_met: bool,
    observed_queue_len: int | None,
    accumulated_nonempty: bool,
    confirmed_queue_size: int | None,
    rollouter_paused: bool,
    rollouter_active_tasks: int,
) -> bool:
    """Return True iff the trainer should raise _StarvationFlush to force an early
    optimizer-step flush instead of blocking on the next chunk.

    The predicate is intentionally conservative AND only ever *gated on real accumulated
    work* (``accumulated_nonempty``), so the worst case of any cross-actor staleness in
    the inputs is a safe, slightly-early flush of genuine data (never a no-op reset, never
    data loss). Conditions:
      - budget mode on and escape enabled;
      - the current chunk batch is NOT already satisfiable (else just assemble it);
      - we hold accumulated supervision to flush (so the flush bumps the version ->
        guaranteed forward progress, no livelock);
      - we are about to block: the last observed queue depth was empty;
      - a fresh queue-size probe confirms the queue is still empty; and
      - the rollouter is fully stalled: paused with zero in-flight tasks (so no chunk can
        possibly arrive until we bump the version).
    """
    if not budget_enabled or not escape_enabled:
        return False
    if minimum_met or not accumulated_nonempty:
        return False
    if observed_queue_len is None or observed_queue_len > 0:
        return False
    if confirmed_queue_size is not None and confirmed_queue_size > 0:
        return False
    return bool(rollouter_paused) and int(rollouter_active_tasks) == 0
