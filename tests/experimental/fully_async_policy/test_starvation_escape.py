# CPU unit tests for the optimizer-step-token-budget *starvation escape* — the fix for the
# m12 budget-flush <-> staleness-pause deadlock.
#
# Covers the pure, dependency-free decision + config-resolution logic in
# verl.experimental.fully_async_policy.starvation. The live actor behavior (RPC plumbing,
# version bump + reset_staleness un-pausing the rollouter) requires the training stack and
# is exercised by the debug run, not here.
import importlib.util
import os
import types

import pytest

# Load the module by file path rather than `import verl.experimental...` so the test never
# triggers verl/__init__.py (which imports torch/tensordict/etc.). starvation.py is
# deliberately stdlib-only, so this exercises the real file on any CPU / bare container.
_STARV_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "verl", "experimental", "fully_async_policy", "starvation.py")
)
_spec = importlib.util.spec_from_file_location("verl_starvation_under_test", _STARV_PATH)
stv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stv)


# --------------------------------------------------------------------------- helpers
def _cfg(**async_training):
    class _AT:
        def __init__(self, d):
            self._d = d

        def get(self, k, default=None):
            return self._d.get(k, default)

    return types.SimpleNamespace(async_training=_AT(dict(async_training)))


# The exact deadlock fingerprint: budget on, escape on, current chunk batch not yet
# satisfiable, accumulated work pending, queue observed+confirmed empty, rollouter paused
# with nothing in flight. This MUST trigger the escape.
STARVED = dict(
    budget_enabled=True,
    escape_enabled=True,
    minimum_met=False,
    observed_queue_len=0,
    accumulated_nonempty=True,
    confirmed_queue_size=0,
    rollouter_paused=True,
    rollouter_active_tasks=0,
)


# --------------------------------------------------------------------------- decision
def test_decision_fires_on_exact_deadlock_fingerprint():
    assert stv.starvation_escape_decision(**STARVED) is True


def test_decision_confirmed_queue_none_still_fires():
    # confirmed_queue_size None means "unknown" -> rely on observed empty + rollouter stall.
    assert stv.starvation_escape_decision(**{**STARVED, "confirmed_queue_size": None}) is True


@pytest.mark.parametrize(
    "override",
    [
        {"budget_enabled": False},        # control arm (budget<=0) is immune
        {"escape_enabled": False},        # A/B toggle off -> old (deadlocking) behavior
        {"minimum_met": True},            # batch already satisfiable -> just assemble it
        {"accumulated_nonempty": False},  # nothing to flush -> never no-op reset (livelock guard)
        {"observed_queue_len": None},     # not yet about to block
        {"observed_queue_len": 7},        # queue had items on last pop
        {"confirmed_queue_size": 4},      # fresh probe: a chunk arrived (race) -> consume it
        {"confirmed_queue_size": 1},      # boundary: any positive confirmed depth suppresses
        {"rollouter_paused": False},      # rollouter still running -> chunks will come
        {"rollouter_active_tasks": 3},    # in-flight decodes can still emit chunks -> wait
        {"rollouter_active_tasks": 1},    # boundary: a single in-flight task suppresses
    ],
)
def test_decision_suppressed_when_any_condition_missing(override):
    assert stv.starvation_escape_decision(**{**STARVED, **override}) is False


def test_decision_positive_boundaries_fire():
    # The exact zero-boundaries (queue drained, nothing in flight) MUST allow the escape.
    assert stv.starvation_escape_decision(**{**STARVED, "confirmed_queue_size": 0}) is True
    assert stv.starvation_escape_decision(**{**STARVED, "rollouter_active_tasks": 0}) is True


# ----------------------------------------------------------- starvation_escape_enabled
def test_escape_enabled_default_on(monkeypatch):
    monkeypatch.delenv("OPD_STARVATION_ESCAPE", raising=False)
    assert stv.get_starvation_escape_enabled(_cfg()) is True
    assert stv.get_starvation_escape_enabled(None) is True


@pytest.mark.parametrize("val,expected", [("0", False), ("off", False), ("1", True), ("on", True)])
def test_escape_enabled_env(monkeypatch, val, expected):
    monkeypatch.setenv("OPD_STARVATION_ESCAPE", val)
    assert stv.get_starvation_escape_enabled(_cfg()) is expected


def test_escape_enabled_config_fallback(monkeypatch):
    monkeypatch.delenv("OPD_STARVATION_ESCAPE", raising=False)
    assert stv.get_starvation_escape_enabled(_cfg(starvation_escape=False)) is False


def test_escape_enabled_env_overrides_config(monkeypatch):
    monkeypatch.setenv("OPD_STARVATION_ESCAPE", "1")
    assert stv.get_starvation_escape_enabled(_cfg(starvation_escape=False)) is True


# --------------------------------------------------------- optimizer_step_max_fit_steps
@pytest.mark.parametrize(
    "val,expected", [(None, 0), ("0", 0), ("8", 8), ("-3", 0), ("abc", 0), ("  12 ", 12), ("   ", 0)]
)
def test_max_fit_steps_env(monkeypatch, val, expected):
    if val is None:
        monkeypatch.delenv("OPD_OPTIMIZER_STEP_MAX_FIT_STEPS", raising=False)
    else:
        monkeypatch.setenv("OPD_OPTIMIZER_STEP_MAX_FIT_STEPS", val)
    # invalid/blank env falls through to config; here config is empty -> default 0
    assert stv.get_optimizer_step_max_fit_steps(_cfg()) == expected


def test_max_fit_steps_config_fallback(monkeypatch):
    monkeypatch.delenv("OPD_OPTIMIZER_STEP_MAX_FIT_STEPS", raising=False)
    assert stv.get_optimizer_step_max_fit_steps(_cfg(optimizer_step_max_fit_steps=5)) == 5
    assert stv.get_optimizer_step_max_fit_steps(_cfg(optimizer_step_max_fit_steps=0)) == 0


def test_max_fit_steps_env_overrides_config(monkeypatch):
    monkeypatch.setenv("OPD_OPTIMIZER_STEP_MAX_FIT_STEPS", "8")
    assert stv.get_optimizer_step_max_fit_steps(_cfg(optimizer_step_max_fit_steps=5)) == 8


# -------------------------------------------------------------- validate_every_flush
def test_validate_every_flush_default_off(monkeypatch):
    monkeypatch.delenv("OPD_VALIDATE_EVERY_FLUSH", raising=False)
    assert stv.get_validate_every_flush(_cfg()) is False
    assert stv.get_validate_every_flush(None) is False


@pytest.mark.parametrize("val,expected", [("1", True), ("on", True), ("0", False), ("no", False)])
def test_validate_every_flush_env(monkeypatch, val, expected):
    monkeypatch.setenv("OPD_VALIDATE_EVERY_FLUSH", val)
    assert stv.get_validate_every_flush(_cfg()) is expected


def test_validate_every_flush_config_fallback(monkeypatch):
    monkeypatch.delenv("OPD_VALIDATE_EVERY_FLUSH", raising=False)
    assert stv.get_validate_every_flush(_cfg(validate_every_flush=True)) is True


def test_validate_every_flush_env_overrides_config(monkeypatch):
    monkeypatch.setenv("OPD_VALIDATE_EVERY_FLUSH", "0")
    assert stv.get_validate_every_flush(_cfg(validate_every_flush=True)) is False


# ------------------------------------------------- unrecognized env -> config fallback
def test_unrecognized_env_falls_back_to_config(monkeypatch):
    # An env value the parsers don't recognize must be ignored, deferring to config.
    monkeypatch.setenv("OPD_STARVATION_ESCAPE", "maybe")
    monkeypatch.setenv("OPD_OPTIMIZER_STEP_MAX_FIT_STEPS", "garbage")
    monkeypatch.setenv("OPD_VALIDATE_EVERY_FLUSH", "perhaps")
    assert stv.get_starvation_escape_enabled(_cfg(starvation_escape=False)) is False
    assert stv.get_optimizer_step_max_fit_steps(_cfg(optimizer_step_max_fit_steps=7)) == 7
    assert stv.get_validate_every_flush(_cfg(validate_every_flush=True)) is True


# -------------------------------------------------- config robustness (non-int value)
@pytest.mark.parametrize("bad", [3.9, "lots", None, [1]])
def test_max_fit_steps_config_non_int_defaults_to_zero(monkeypatch, bad):
    monkeypatch.delenv("OPD_OPTIMIZER_STEP_MAX_FIT_STEPS", raising=False)
    assert stv.get_optimizer_step_max_fit_steps(_cfg(optimizer_step_max_fit_steps=bad)) == 0


# ------------------------------------------------------ all resolvers coexist cleanly
def test_all_resolvers_coexist_in_one_config(monkeypatch):
    for k in ("OPD_STARVATION_ESCAPE", "OPD_OPTIMIZER_STEP_MAX_FIT_STEPS", "OPD_VALIDATE_EVERY_FLUSH"):
        monkeypatch.delenv(k, raising=False)
    cfg = _cfg(starvation_escape=False, optimizer_step_max_fit_steps=10, validate_every_flush=True)
    assert stv.get_starvation_escape_enabled(cfg) is False
    assert stv.get_optimizer_step_max_fit_steps(cfg) == 10
    assert stv.get_validate_every_flush(cfg) is True
