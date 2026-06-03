# CPU unit tests for the rollout-lease refresh knobs (Q3).
#
# Covers the pure decision logic:
#   - OPD_ROLLOUT_LEASE_MAX_TOKENS env parsing (_rollout_lease_max_tokens)
#   - OPD_ROLLOUT_LEASE_FORCE_REPREFILL env parsing (_rollout_lease_force_reprefill)
#   - lease-trigger cadence (when a refresh fires across a streamed decode)
#
# The engine-side effect of forced re-prefill (server.clear_kv_cache / reset_prefix_cache)
# requires a live vLLM engine and is exercised by the training-stack debug run, not here.
import importlib

import pytest

stal = importlib.import_module("verl.experimental.agent_loop.single_turn_agent_loop")


@pytest.mark.parametrize(
    "val,expected",
    [(None, 0), ("0", 0), ("1024", 1024), ("-5", 0), ("abc", 0), ("  2048 ", 2048)],
)
def test_lease_max_tokens_env(monkeypatch, val, expected):
    if val is None:
        monkeypatch.delenv("OPD_ROLLOUT_LEASE_MAX_TOKENS", raising=False)
    else:
        monkeypatch.setenv("OPD_ROLLOUT_LEASE_MAX_TOKENS", val)
    assert stal._rollout_lease_max_tokens() == expected


@pytest.mark.parametrize(
    "val,expected",
    [(None, False), ("0", False), ("", False), ("1", True), ("true", True),
     ("TRUE", True), ("on", True), ("yes", True), ("nope", False)],
)
def test_force_reprefill_env(monkeypatch, val, expected):
    if val is None:
        monkeypatch.delenv("OPD_ROLLOUT_LEASE_FORCE_REPREFILL", raising=False)
    else:
        monkeypatch.setenv("OPD_ROLLOUT_LEASE_FORCE_REPREFILL", val)
    assert stal._rollout_lease_force_reprefill() is expected


def _lease_fire_offsets(response_length: int, chunk_tokens: int, lease_tokens: int) -> list[int]:
    """Reference model of the loop's lease-trigger logic (single_turn_agent_loop):
    a refresh fires when (token_offset - lease_anchor) >= lease_tokens, anchor resets
    to token_offset on fire. Returns the token_offsets at which a refresh fires."""
    fires, anchor, off = [], 0, 0
    while off < response_length:
        if lease_tokens and off - anchor >= lease_tokens:
            fires.append(off)
            anchor = off
        off += min(chunk_tokens, response_length - off)
    return fires


def test_lease_trigger_disabled():
    assert _lease_fire_offsets(4096, 256, 0) == []


def test_lease_trigger_1024():
    # chunk=256, lease=1024 -> fire every 4 chunks: 1024, 2048, 3072 (not at 0; not past end)
    assert _lease_fire_offsets(4096, 256, 1024) == [1024, 2048, 3072]


def test_lease_trigger_512():
    assert _lease_fire_offsets(4096, 256, 512) == [512, 1024, 1536, 2048, 2560, 3072, 3584]


def test_lease_trigger_ge_response_never_fires():
    assert _lease_fire_offsets(4096, 256, 8192) == []
