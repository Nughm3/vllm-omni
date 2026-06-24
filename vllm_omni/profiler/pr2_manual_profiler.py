# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Low-overhead PR #2677 manual profiler.

This profiler is meant for Python/control-plane paths where torch profiler is
too noisy or misses background threads. It records aggregate timings in memory
and optionally emits small Chrome/Perfetto trace events.
"""

from __future__ import annotations

import atexit
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

_TRUE_VALUES = {"1", "true", "yes", "on"}


def _env_true(*names: str) -> bool:
    for name in names:
        if os.environ.get(name, "").strip().lower() in _TRUE_VALUES:
            return True
    return False


class _NoopSpan:
    bytes_count: int | None = None

    def __enter__(self) -> _NoopSpan:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return False


class _ManualSpan:
    def __init__(self, profiler: _ManualProfiler, name: str, args: dict[str, Any]):
        self._profiler = profiler
        self._name = name
        self._args = args
        self._start_ns = 0
        self.bytes_count: int | None = None

    def __enter__(self) -> _ManualSpan:
        self._start_ns = time.perf_counter_ns()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        self._profiler.record(
            self._name,
            self._start_ns,
            time.perf_counter_ns(),
            self.bytes_count,
            self._args,
        )
        return False


class _ManualProfiler:
    def __init__(self) -> None:
        self.profile_enabled = _env_true("PR2_MANUAL_PROFILE", "PR2_BG_IO_PROFILE") or _env_true(
            "PR2_MANUAL_TRACE",
            "PR2_BG_IO_TRACE",
        )
        self.trace_enabled = _env_true("PR2_MANUAL_TRACE", "PR2_BG_IO_TRACE")
        self._lock = threading.Lock()
        self._stats: dict[str, dict[str, int]] = {}
        self._events: list[dict[str, Any]] = []
        self._origin_ns = time.time_ns() - time.perf_counter_ns()
        self._pid = os.getpid()
        self._output_path = self._resolve_output_path()
        if self.profile_enabled:
            atexit.register(self.dump)

    def _refresh_process_state_if_needed(self) -> None:
        current_pid = os.getpid()
        if current_pid == self._pid:
            return
        self._pid = current_pid
        self._origin_ns = time.time_ns() - time.perf_counter_ns()
        self._output_path = self._resolve_output_path()
        self._stats = {}
        self._events = []

    def _resolve_output_path(self) -> Path:
        output_file = os.environ.get("PR2_MANUAL_OUTPUT") or os.environ.get("PR2_BG_IO_OUTPUT")
        if output_file:
            return Path(output_file)
        output_dir = Path(
            os.environ.get("PR2_MANUAL_DIR")
            or os.environ.get("PR2_BG_IO_DIR")
            or "/tmp"
        )
        return output_dir / f"pr2_manual_profile_pid{self._pid}.json"

    def span(self, name: str, **args: Any) -> _ManualSpan | _NoopSpan:
        if not self.profile_enabled:
            return _NoopSpan()
        self._refresh_process_state_if_needed()
        return _ManualSpan(self, name, args)

    def record(
        self,
        name: str,
        start_ns: int,
        end_ns: int,
        bytes_count: int | None,
        args: dict[str, Any],
    ) -> None:
        dur_ns = max(0, end_ns - start_ns)
        with self._lock:
            stats = self._stats.setdefault(
                name,
                {
                    "count": 0,
                    "total_ns": 0,
                    "max_ns": 0,
                    "total_bytes": 0,
                },
            )
            stats["count"] += 1
            stats["total_ns"] += dur_ns
            stats["max_ns"] = max(stats["max_ns"], dur_ns)
            if bytes_count is not None:
                stats["total_bytes"] += int(bytes_count)

            if self.trace_enabled:
                event_args = dict(args)
                if bytes_count is not None:
                    event_args["bytes"] = int(bytes_count)
                self._events.append(
                    {
                        "ph": "X",
                        "name": name,
                        "ts": (self._origin_ns + start_ns) / 1000.0,
                        "dur": dur_ns / 1000.0,
                        "pid": self._pid,
                        "tid": threading.get_ident(),
                        "args": event_args,
                    }
                )

    def dump(self) -> None:
        if not self.profile_enabled:
            return
        self._refresh_process_state_if_needed()
        with self._lock:
            stats = {
                name: {
                    **values,
                    "total_ms": values["total_ns"] / 1_000_000.0,
                    "max_ms": values["max_ns"] / 1_000_000.0,
                }
                for name, values in sorted(self._stats.items())
            }
            events = list(self._events)
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._output_path.with_suffix(self._output_path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(
                {
                    "traceEvents": events,
                    "pr2_manual_stats": stats,
                    "metadata": {
                        "pid": self._pid,
                        "trace_enabled": self.trace_enabled,
                    },
                },
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        tmp_path.replace(self._output_path)


_PROFILER = _ManualProfiler()


def manual_span(name: str, **args: Any) -> _ManualSpan | _NoopSpan:
    return _PROFILER.span(name, **args)


def dump_manual_profile() -> None:
    _PROFILER.dump()
