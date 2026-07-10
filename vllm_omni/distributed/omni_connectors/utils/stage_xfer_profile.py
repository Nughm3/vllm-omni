# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Manual stage-transfer timing logs (MTE-style).

Emits one INFO line per span in the same format as Mooncake's
``[RDMA GET] key: a=X.Xms, b=Y.Yms, total=Z.Zms`` logs so transfer
overhead can be grepped without PyTorch profiler noise.

Enable/disable with ``OMNI_STAGE_XFER_PROFILE`` (default: on).

Key tags (do not conflate the two queues):
- ``STAGE XFER ENQUEUE.send_queue_wait``: stage send path, enqueue → put()
- ``MTE SENDER TPE.tpe_queue_wait``: Mooncake sender ThreadPoolExecutor,
  listener submit → handler thread start
- ``RDMA GET.pull_wait``: receiver blocked on ZMQ pull reply (includes
  sender TPE + RDMA write + reply drain — not pure wire RDMA)
"""

from __future__ import annotations

import os
import time
from collections.abc import Mapping
from typing import Any

from .logging import get_connector_logger

logger = get_connector_logger(__name__)

_ENABLED = os.environ.get("OMNI_STAGE_XFER_PROFILE", "1").strip().lower() not in {
    "0",
    "false",
    "off",
    "no",
}


def stage_xfer_profile_enabled() -> bool:
    return _ENABLED


def now() -> float:
    return time.perf_counter()


def ms_since(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


def log_stage_xfer(tag: str, key: str, parts: Mapping[str, Any], *, extra: str = "") -> None:
    """Log ``[TAG] key: a=X.Xms, b=Y.Yms, ...`` (numeric values treated as ms)."""
    if not _ENABLED:
        return
    fields: list[str] = []
    for name, value in parts.items():
        if value is None:
            continue
        if isinstance(value, float):
            fields.append(f"{name}={value:.1f}ms")
        elif isinstance(value, bool):
            fields.append(f"{name}={int(value)}")
        else:
            fields.append(f"{name}={value}")
    suffix = f" {extra}" if extra else ""
    logger.info("[%s] %s: %s%s", tag, key, ", ".join(fields), suffix)


class StageXferSpan:
    """Accumulate named sub-timings and emit one MTE-style log line."""

    def __init__(self, tag: str, key: str):
        self.tag = tag
        self.key = key
        self._t0 = time.perf_counter()
        self._parts: dict[str, Any] = {}
        self._marks: dict[str, float] = {}

    def mark(self, name: str) -> None:
        self._marks[name] = time.perf_counter()

    def lap(self, name: str, since: str | None = None) -> float:
        """Record ms since ``since`` mark (or span start) under ``name``."""
        t_end = time.perf_counter()
        t_start = self._marks[since] if since is not None else self._t0
        elapsed = (t_end - t_start) * 1000.0
        self._parts[name] = elapsed
        self._marks[name] = t_end
        return elapsed

    def set(self, name: str, value: Any) -> None:
        self._parts[name] = value

    def finish(self, *, extra: str = "", include_total: bool = True) -> float:
        total = (time.perf_counter() - self._t0) * 1000.0
        if include_total:
            self._parts["total"] = total
        log_stage_xfer(self.tag, self.key, self._parts, extra=extra)
        return total
