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


# ----------------------------------------------------------- stall fingerprint detection
# The full stall: chunk queue drained AND rollouter paused AND nothing in flight -> no chunk
# can arrive until the trainer acts (flush if accumulated, else direct resume). The caller
# checks the cheap pre-gates (budget on, escape enabled, batch not satisfiable, observed
# queue empty) inline before the confirming RPCs fed here.
def test_stall_fires_on_fingerprint():
    assert stv.starvation_stall_detected(confirmed_queue_size=0, rollouter_paused=True, rollouter_active_tasks=0) is True


def test_stall_confirmed_queue_none_still_fires():
    # None means "unknown / treat as empty" -> rely on the rollouter being fully stalled.
    assert stv.starvation_stall_detected(confirmed_queue_size=None, rollouter_paused=True, rollouter_active_tasks=0) is True


@pytest.mark.parametrize(
    "q,paused,active,expected",
    [
        (0, True, 0, True),     # exact fingerprint -> stalled
        (None, True, 0, True),  # unknown queue + fully stalled -> stalled
        (1, True, 0, False),    # boundary: any positive confirmed depth -> consume it
        (5, True, 0, False),    # queue has chunks -> not stalled
        (0, False, 0, False),   # not paused -> chunks coming
        (0, True, 1, False),    # boundary: a single in-flight task -> wait, may still emit
        (0, True, 3, False),    # in-flight decodes -> wait
    ],
)
def test_stall_truth_table(q, paused, active, expected):
    assert (
        stv.starvation_stall_detected(
            confirmed_queue_size=q, rollouter_paused=paused, rollouter_active_tasks=active
        )
        is expected
    )


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


# ----------------------------------------------------------- max_starvation_resets cap
def test_max_resets_default(monkeypatch):
    monkeypatch.delenv("OPD_STARVATION_MAX_RESETS", raising=False)
    assert stv.get_max_starvation_resets(_cfg()) == 8
    assert stv.get_max_starvation_resets(None) == 8


@pytest.mark.parametrize("val,expected", [("3", 3), ("0", 8), ("-1", 8), ("abc", 8), ("  5 ", 5)])
def test_max_resets_env(monkeypatch, val, expected):
    monkeypatch.setenv("OPD_STARVATION_MAX_RESETS", val)
    assert stv.get_max_starvation_resets(_cfg()) == expected


@pytest.mark.parametrize("cfgval,expected", [(4, 4), (0, 8), (3.5, 8), ("x", 8), (None, 8), (True, 8)])
def test_max_resets_config(monkeypatch, cfgval, expected):
    monkeypatch.delenv("OPD_STARVATION_MAX_RESETS", raising=False)
    assert stv.get_max_starvation_resets(_cfg(starvation_max_resets=cfgval)) == expected


def test_max_resets_env_overrides_config(monkeypatch):
    monkeypatch.setenv("OPD_STARVATION_MAX_RESETS", "6")
    assert stv.get_max_starvation_resets(_cfg(starvation_max_resets=4)) == 6


# ------------------------------------------------------ all resolvers coexist cleanly
def test_all_resolvers_coexist_in_one_config(monkeypatch):
    for k in (
        "OPD_STARVATION_ESCAPE",
        "OPD_OPTIMIZER_STEP_MAX_FIT_STEPS",
        "OPD_VALIDATE_EVERY_FLUSH",
        "OPD_STARVATION_MAX_RESETS",
    ):
        monkeypatch.delenv(k, raising=False)
    cfg = _cfg(
        starvation_escape=False,
        optimizer_step_max_fit_steps=10,
        validate_every_flush=True,
        starvation_max_resets=4,
    )
    assert stv.get_starvation_escape_enabled(cfg) is False
    assert stv.get_optimizer_step_max_fit_steps(cfg) == 10
    assert stv.get_validate_every_flush(cfg) is True
    assert stv.get_max_starvation_resets(cfg) == 4
