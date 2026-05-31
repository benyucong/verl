# Stage-0 profiling helper for Xiaoshuai (chunk-level async OPD) opportunity sizing.
#
# When the OPD_STAGE0_TRACE_DIR environment variable is set, `trace_event()`
# appends a JSON line per call to a per-process file inside that directory.
# Otherwise it is a fast no-op (single env-var lookup cached at import time).
#
# Schema of each line:
#   {"event": "<str>", "sample_id": "<str>", "ts": <float seconds>,
#    "pid": <int>, "role": "<rollouter|trainer|...>",
#    "extra": {...optional extra fields...}}
#
# Designed to be safe to import from any Ray actor; each actor gets its own
# file based on PID so there are no cross-process write conflicts.

from __future__ import annotations

import json
import os
import socket
import threading
import time
from typing import Any

_TRACE_DIR = os.environ.get("OPD_STAGE0_TRACE_DIR", "").strip() or None
_ENABLED = _TRACE_DIR is not None

_lock = threading.Lock()
_file = None
_path = None


def is_enabled() -> bool:
    return _ENABLED


def _open_file_if_needed(role: str) -> None:
    global _file, _path
    if _file is not None:
        return
    os.makedirs(_TRACE_DIR, exist_ok=True)
    host = socket.gethostname().split(".")[0]
    pid = os.getpid()
    _path = os.path.join(_TRACE_DIR, f"trace_{role}_{host}_{pid}.jsonl")
    _file = open(_path, "a", buffering=1)  # line-buffered


def trace_event(event: str, sample_id: str, role: str = "unknown", **extra: Any) -> None:
    """Append one JSONL event. No-op when OPD_STAGE0_TRACE_DIR is unset."""
    if not _ENABLED:
        return
    trace_ts = extra.pop("_trace_ts", None)
    rec = {
        "event": event,
        "sample_id": str(sample_id),
        "ts": float(trace_ts) if trace_ts is not None else time.time(),
        "pid": os.getpid(),
        "role": role,
    }
    if extra:
        rec["extra"] = extra
    try:
        with _lock:
            _open_file_if_needed(role)
            _file.write(json.dumps(rec) + "\n")
    except Exception:
        # Never let profiling break training.
        pass


# ---------------------------------------------------------------------------
# Stage-1 (chunk-level) event vocabulary.
#
# All chunk events share the same `sample_id` as their parent sample so that
# the analyzer can join them with Stage-0 sample-level events (`gen_start`,
# `gen_end`, `put`, `get`) for A/B comparison. Per-chunk identity is carried
# in the `extra` field:
#
#   chunk_idx     : 0-based index of the chunk within the sample
#   n_tokens      : tokens published in this chunk
#   token_offset  : offset (in response tokens) of the chunk start
#   policy_version: rollouter policy version at chunk emission time
#   is_final      : True for the chunk that contains the EOS
#
# Events:
#   chunk_emit         (role=rollouter)  chunk is published to queue
#   chunk_get          (role=trainer)    chunk is dequeued by trainer
#   chunk_score_start  (role=trainer)    teacher scoring of chunk begins
#   chunk_score_end    (role=trainer)    teacher scoring of chunk ends
#   chunk_train_start  (role=trainer)    optimizer step using chunk begins
#   chunk_train_end    (role=trainer)    optimizer step using chunk ends
#   chunk_drop_stale   (role=trainer)    chunk dropped due to staleness > sigma
# ---------------------------------------------------------------------------

CHUNK_EMIT = "chunk_emit"
CHUNK_GET = "chunk_get"
CHUNK_SCORE_START = "chunk_score_start"
CHUNK_SCORE_END = "chunk_score_end"
CHUNK_TRAIN_START = "chunk_train_start"
CHUNK_TRAIN_END = "chunk_train_end"
CHUNK_DROP_STALE = "chunk_drop_stale"


def trace_chunk_event(
    event: str,
    sample_id: str,
    chunk_idx: int,
    role: str = "unknown",
    n_tokens: int | None = None,
    token_offset: int | None = None,
    policy_version: int | None = None,
    is_final: bool | None = None,
    **extra: Any,
) -> None:
    """Convenience wrapper for chunk-level events; no-op when tracing disabled."""
    if not _ENABLED:
        return
    event_ts = time.time()
    row_id = extra.pop("row_id", sample_id)
    chunk_id = extra.pop("chunk_id", f"{row_id}:{int(chunk_idx)}")
    payload: dict[str, Any] = {
        "chunk_idx": int(chunk_idx),
        "row_id": str(row_id),
        "chunk_id": str(chunk_id),
        f"{event}_ts": event_ts,
    }
    if n_tokens is not None:
        payload["n_tokens"] = int(n_tokens)
    if token_offset is not None:
        payload["token_offset"] = int(token_offset)
    if policy_version is not None:
        chunk_policy_version = int(policy_version)
        payload["policy_version"] = chunk_policy_version
        payload["chunk_policy_version"] = chunk_policy_version
    if is_final is not None:
        payload["is_final"] = bool(is_final)
    if extra:
        payload.update(extra)
    trace_event(event, sample_id, role=role, _trace_ts=event_ts, **payload)
