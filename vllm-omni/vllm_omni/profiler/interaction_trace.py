# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in, bounded CPU timeline for an interacting multi-stage request.

Set OMNI_INTERACTION_TRACE_DIR *before process startup*. Each process writes a
separate JSONL file. Events are CPU observations, never GPU completion fences.
No tensor is inspected, copied, stringified or synchronized by this module.
The writer flushes every 250 ms; abrupt death can lose queued/buffered events.
Use the sequence gaps and dropped_total/trace_summary to detect incomplete traces.
"""

from __future__ import annotations

import atexit
import itertools
import json
import math
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any

_DIRECTORY = os.environ.get("OMNI_INTERACTION_TRACE_DIR", "").strip()
_writer: _TraceWriter | None = None
_init_lock = threading.Lock()


def enabled() -> bool:
    """Configuration is fixed at import, keeping the disabled path cheap."""
    return bool(_DIRECTORY)


def _small_value(value: Any, depth: int = 0) -> Any:
    # Exact built-in types avoid calling user/tensor conversion methods.
    kind = type(value)
    if value is None or kind in (bool, int):
        return value
    if kind is float:
        return value if math.isfinite(value) else None
    if kind is str:
        return value[:512]
    if depth < 2 and kind in (list, tuple):
        return [_small_value(item, depth + 1) for item in value[:32]]
    if depth < 2 and kind is dict:
        return {
            key[:128]: _small_value(item, depth + 1)
            for key, item in itertools.islice(value.items(), 32)
            if type(key) is str
        }
    return "<omitted>"


class _TraceWriter:
    def __init__(self, directory: str, capacity: int):
        self.pid = os.getpid()
        self.epoch_ns = time.monotonic_ns()
        self.path = Path(directory) / f"interaction-{self.pid}-{self.epoch_ns}.jsonl"
        self.queue: queue.Queue = queue.Queue(maxsize=capacity)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.sequence = 0
        self.dropped = 0
        self.written = 0
        self.error: str | None = None
        self.thread = threading.Thread(target=self._run, name="omni-interaction-trace", daemon=True)
        self.thread.start()

    def emit(self, event: str, fields: dict[str, Any]) -> None:
        clock_ns = time.monotonic_ns()
        tid = threading.get_native_id()
        row = {key[:128]: _small_value(value) for key, value in itertools.islice(fields.items(), 32)}
        with self.lock:
            self.sequence += 1
            if self.stop.is_set() or self.error is not None:
                self.dropped += 1
                return
            row.update(
                event=event[:128], monotonic_ns=clock_ns, pid=self.pid, tid=tid,
                worker_epoch=self.epoch_ns, seq=self.sequence, dropped_total=self.dropped,
            )
            try:
                self.queue.put_nowait(row)
            except queue.Full:
                self.dropped += 1

    def _run(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            try:
                boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
            except OSError:
                boot_id = None
            with self.path.open("x", encoding="utf-8") as stream:
                stream.write(json.dumps({
                    "event": "trace_start", "pid": self.pid, "worker_epoch": self.epoch_ns,
                    "monotonic_ns": time.monotonic_ns(), "wall_time_ns": time.time_ns(),
                    "boot_id": boot_id, "queue_capacity": self.queue.maxsize,
                    "clock": "monotonic_ns", "schema": 1,
                }) + "\n")
                last_flush = time.monotonic()
                last_reported_drop = 0
                while not self.stop.is_set() or not self.queue.empty():
                    try:
                        item = self.queue.get(timeout=0.25)
                    except queue.Empty:
                        item = None
                    if isinstance(item, threading.Event):
                        stream.flush()
                        item.set()
                    elif item is not None:
                        stream.write(json.dumps(item, separators=(",", ":"), allow_nan=False) + "\n")
                        self.written += 1
                    now = time.monotonic()
                    if now - last_flush >= 0.25:
                        if self.dropped != last_reported_drop:
                            stream.write(json.dumps({
                                "event": "trace_drop", "pid": self.pid, "worker_epoch": self.epoch_ns,
                                "monotonic_ns": time.monotonic_ns(), "dropped_total": self.dropped,
                            }) + "\n")
                            last_reported_drop = self.dropped
                        stream.flush()
                        last_flush = now
                stream.write(json.dumps({
                    "event": "trace_summary", "pid": self.pid, "worker_epoch": self.epoch_ns,
                    "monotonic_ns": time.monotonic_ns(), "attempted": self.sequence,
                    "written": self.written, "dropped_total": self.dropped,
                }) + "\n")
        except Exception as exc:
            # Tracing must never fail inference or introduce fallback sync I/O.
            with self.lock:
                self.error = type(exc).__name__
                self.dropped = self.sequence - self.written

    def flush(self, timeout: float) -> bool:
        barrier = threading.Event()
        deadline = time.monotonic() + timeout
        while self.thread.is_alive() and self.error is None:
            try:
                self.queue.put(barrier, timeout=max(0.0, deadline - time.monotonic()))
                return barrier.wait(max(0.0, deadline - time.monotonic()))
            except queue.Full:
                return False
        return False


def emit(event: str, **fields: Any) -> None:
    """Enqueue a small CPU event without waiting for disk or queue capacity.

    Prefer request_id, external_req_id, stage_id, replica_id, chunk_id and job_id.
    Reserved envelope fields cannot be overwritten. Unsupported/large values are
    omitted or truncated; callers should pass only small scalar metadata.
    """
    if not _DIRECTORY:
        return
    global _writer
    try:
        if _writer is None:
            with _init_lock:
                if _writer is None:
                    try:
                        capacity = int(os.environ.get("OMNI_INTERACTION_TRACE_BUFFER_EVENTS", "65536"))
                    except ValueError:
                        capacity = 65536
                    _writer = _TraceWriter(_DIRECTORY, max(1, min(capacity, 1_000_000)))
        _writer.emit(event, fields)
    except Exception:
        # Invalid diagnostic metadata must not affect request execution.
        return


def flush(timeout: float = 2.0) -> bool:
    """Explicit test/operator boundary; never called by the inference hooks."""
    return _writer is None or _writer.flush(timeout)


def shutdown(timeout: float = 2.0) -> None:
    if _writer is not None:
        _writer.stop.set()
        _writer.thread.join(timeout)


def status() -> dict[str, Any]:
    if _writer is None:
        return {"enabled": enabled(), "started": False}
    return {
        "enabled": True, "started": True, "path": str(_writer.path),
        "attempted": _writer.sequence, "written": _writer.written,
        "dropped_total": _writer.dropped, "error": _writer.error,
    }


def _after_fork() -> None:
    # Parent threads/locks cannot be used by a forked engine process.
    global _writer, _init_lock
    _writer = None
    _init_lock = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)
atexit.register(shutdown)
