# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass

from .connectors.base import OmniConnectorBase
from .utils.initialization import ConnectorOwner
from .utils.logging import get_connector_logger

logger = get_connector_logger(__name__)


@dataclass(frozen=True, slots=True)
class ConnectorRuntimeContext:
    """Process-local context supplied when materializing a :class:`StageConnectorPlan`.

    Must be built *inside* the owning process after that process's distributed
    context (if any) exists. ``tp_rank`` is unused for ``STAGE_REPLICA`` (chunk
    transfer adapter / scheduler) — materialization forces local_rank=0 so it
    never consults the worker TP group.
    """

    stage_id: int
    owner_scope: ConnectorOwner
    replica_id: int = 0
    tp_rank: int | None = None
    tp_size: int | None = None

    def __post_init__(self) -> None:
        if self.stage_id < 0:
            raise ValueError(f"stage_id must be non-negative, got {self.stage_id}")
        if self.replica_id < 0:
            raise ValueError(f"replica_id must be non-negative, got {self.replica_id}")
        if self.tp_rank is not None and self.tp_rank < 0:
            raise ValueError(f"tp_rank must be non-negative, got {self.tp_rank}")
        if self.tp_size is not None and self.tp_size <= 0:
            raise ValueError(f"tp_size must be positive, got {self.tp_size}")
        if self.tp_rank is not None and self.tp_size is not None and self.tp_rank >= self.tp_size:
            raise ValueError(f"tp_rank={self.tp_rank} must be smaller than tp_size={self.tp_size}")


@dataclass
class StageConnectorSet:
    """Lifecycle container for a stage's receive/send connectors.

    ``close()`` de-duplicates when both refs are the same dual instance.
    """

    receive: OmniConnectorBase | None
    send: OmniConnectorBase | None

    @property
    def connector(self) -> OmniConnectorBase | None:
        """Send-path compatibility view used by legacy payload builders."""
        return self.send or self.receive

    def close(self) -> None:
        seen: set[int] = set()
        for conn in (self.receive, self.send):
            if conn is None or id(conn) in seen:
                continue
            seen.add(id(conn))
            try:
                conn.close()
            except Exception:
                logger.warning("Failed to close connector %s", type(conn).__name__, exc_info=True)
