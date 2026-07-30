# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

from ..utils.logging import get_connector_logger

logger = get_connector_logger(__name__)


@dataclass(frozen=True, slots=True)
class ConnectorCapabilities:
    """Capability flags consulted by dual-collapse / materialization."""

    # Whether the connector can handle raw bytes/torch.Tensor natively
    # without going through OmniSerializer.  Connectors that copy raw
    # payloads directly (e.g. RDMA) should override this to True.
    supports_raw_data: bool = False

    # Whether one physical connector instance can safely serve the inbound
    # receiver edge and outbound sender edge of a middle stage.
    supports_shared_dual: bool = False


class OmniConnectorBase(ABC):
    """Base class for all OmniConnectors."""

    # Conservative defaults: serialized payloads, separate directional
    # instances, and no assumption that concurrent get/put is safe.
    capabilities: ClassVar[ConnectorCapabilities] = ConnectorCapabilities()

    # Fields that legitimately differ per edge/direction — excluded when
    # checking whether two same-type edges describe a compatible backend.
    _DIRECTIONAL_EXTRA_KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "zmq_port",
            "sender_host",
            "sender_zmq_port",
            "role",
            "rank_mapping",
            "from_stage",
            "to_stage",
            "stage_id",
        }
    )

    @abstractmethod
    def put(self, from_stage: str, to_stage: str, put_key: str, data: Any) -> tuple[bool, int, dict[str, Any] | None]:
        """Store Python object, internal serialization handled by connector.

        Args:
            from_stage: Source stage identifier
            to_stage: Destination stage identifier
            put_key: Unique request identifier
            data: Python object to store

        Returns:
            tuple: (success: bool, serialized_size: int, metadata: Optional[dict])
                   Metadata may contain transport-specific handles or inline data.
        """
        pass

    @abstractmethod
    def get(
        self, from_stage: str, to_stage: str, get_key: str, metadata: dict[str, Any] | None = None
    ) -> tuple[Any, int] | None:
        """Retrieve Python object and payload size (bytes).

        Args:
            from_stage: Source stage identifier
            to_stage: Destination stage identifier
            get_key: Unique request identifier
            metadata: Optional transport-specific metadata.  When provided,
                the connector uses it directly (e.g. source_host, source_port,
                data_size) instead of querying the sender.  For heterogeneous
                TP the manager may supply partial metadata (host/port only);
                the connector will query the sender at that address to fill
                in data_size.

        Returns:
            Tuple of (Python object, serialized byte size) if found, None otherwise
        """
        pass

    @abstractmethod
    def cleanup(self, request_id: str) -> None:
        """Clean up resources for a request."""
        pass

    @abstractmethod
    def health(self) -> dict[str, Any]:
        """Return health status and metrics."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Release resources held by this connector.

        Subclasses must implement this to clean up transport-specific
        resources (connections, memory pools, threads, etc.).
        Implementations should be idempotent (safe to call multiple times).
        """
        pass

    # --- Configuration and helpers for connector creation ---

    @classmethod
    def can_share(cls, recv_extra: dict[str, Any], send_extra: dict[str, Any]) -> bool:
        """Whether inbound/outbound extras describe one shareable backend.

        Same connector name is not enough: two edges can use the same class
        against different backends (distinct metadata servers, hosts,
        protocols, devices). Only non-directional keys must match.
        """
        shared_keys = (set(recv_extra) | set(send_extra)) - cls._DIRECTIONAL_EXTRA_KEYS
        return all(recv_extra.get(key) == send_extra.get(key) for key in shared_keys)

    @classmethod
    def merge_dual_specs(cls, recv_extra: dict[str, Any], send_extra: dict[str, Any]) -> dict[str, Any]:
        """Merge inbound + outbound extras into one ``role="dual"`` config.

        Shared backend knobs come from the outbound extra; direction-specific
        fields are taken from the edge that owns them: the outbound listen
        ``zmq_port`` and the inbound upstream ``sender_host`` /
        ``sender_zmq_port``.
        """
        merged = dict(recv_extra)
        merged.update(send_extra)  # outbound wins on shared knobs
        merged["role"] = "dual"
        if "zmq_port" in send_extra:
            merged["zmq_port"] = send_extra["zmq_port"]
        for key in ("sender_host", "sender_zmq_port"):
            if recv_extra.get(key) is not None:
                merged[key] = recv_extra[key]
        recv_rank_mapping = recv_extra.get("rank_mapping")
        send_rank_mapping = send_extra.get("rank_mapping")
        if recv_rank_mapping != send_rank_mapping:
            # A dual connector has one shared config dict, but TP topology is
            # directional. Avoid presenting the outbound mapping as if it
            # described both edges.
            merged.pop("rank_mapping", None)
            if recv_rank_mapping is not None:
                merged["recv_rank_mapping"] = recv_rank_mapping
            if send_rank_mapping is not None:
                merged["send_rank_mapping"] = send_rank_mapping
        return merged

    @staticmethod
    def resolve_role(role: str, *, connector_name: str) -> tuple[bool, bool]:
        """Resolve a ``role`` config value into ``(can_put, can_get)``.

        - ``"sender"``: bind a listener, accept ``put()`` calls only.
        - ``"receiver"``: skip the listener bind, accept ``get()`` calls only.
        - ``"dual"``: bind a listener AND accept ``get()`` — a middle stage
          whose two edges use the same connector type shares one instance.

        Shared by connectors (e.g. Mooncake/Mori transfer engines) whose
        listener bind and ``put()``/``get()`` gating depend on role.
        """
        role = role.lower()
        if role not in {"sender", "receiver", "dual"}:
            raise ValueError(f"Invalid role={role!r} for {connector_name}. Expected 'sender', 'receiver', or 'dual'.")
        return role in {"sender", "dual"}, role in {"receiver", "dual"}

    # --- Default resource-management protocol ---
    # Subclasses get context-manager and destructor support for free;
    # they only need to implement close().

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @staticmethod
    def serialize_obj(obj: Any) -> bytes:
        """Serialize a Python object to bytes using centralized serializer."""
        from ..utils.serialization import OmniSerializer

        return OmniSerializer.serialize(obj)

    @staticmethod
    def deserialize_obj(data: bytes) -> Any:
        """Deserialize bytes to Python object using centralized serializer."""
        from ..utils.serialization import OmniSerializer

        return OmniSerializer.deserialize(data)

    @staticmethod
    def _make_key(key: str, from_stage: str, to_stage: str, separator: str = "@") -> str:
        """Generate internal key with stage routing info.

        Default format: ``{key}@{from_stage}_{to_stage}``.
        Connectors with different key conventions can override this method.
        """
        return f"{key}{separator}{from_stage}_{to_stage}"
