# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .utils.logging import get_connector_logger

try:
    from .connectors.base import OmniConnectorBase
    from .utils.config import (
        TRANSFER_ENGINE_CONNECTOR_NAMES,
        ConnectorSpec,
        StageConnectorPlan,
        StageConnectorSpec,
    )
except ImportError:
    # Fallback for direct execution
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from omni_connectors.connectors.base import OmniConnectorBase
    from omni_connectors.utils.config import (
        TRANSFER_ENGINE_CONNECTOR_NAMES,
        ConnectorSpec,
        StageConnectorPlan,
        StageConnectorSpec,
    )

logger = get_connector_logger(__name__)


DualConfigMerger = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any] | None]


@dataclass(frozen=True)
class _ConnectorRegistration:
    constructor: Callable[[dict[str, Any]], OmniConnectorBase]
    merge_dual: DualConfigMerger | None = None


@dataclass
class StageConnectorSet:
    receive: OmniConnectorBase | None = None
    send: OmniConnectorBase | None = None

    @property
    def connector(self) -> OmniConnectorBase | None:
        """Backward-compatible single-connector view."""
        return self.send or self.receive

    def close(self) -> None:
        if self.receive is not None:
            self.receive.close()
        if self.send is not None and self.send is not self.receive:
            self.send.close()


class OmniConnectorFactory:
    """Factory for creating OmniConnectors."""

    _registry: dict[str, _ConnectorRegistration] = {}

    @classmethod
    def register_connector(
        cls,
        name: str,
        constructor: Callable[[dict[str, Any]], OmniConnectorBase],
        *,
        merge_dual: DualConfigMerger | None = None,
    ) -> None:
        """Register a connector constructor."""
        if name in cls._registry:
            raise ValueError(f"Connector '{name}' is already registered.")
        cls._registry[name] = _ConnectorRegistration(constructor, merge_dual)
        logger.debug(f"Registered connector: {name}")

    @classmethod
    def create_connector(cls, spec: ConnectorSpec) -> OmniConnectorBase:
        """Create a connector from specification."""
        if spec.name not in cls._registry:
            raise ValueError(f"Unknown connector: {spec.name}. Available: {list(cls._registry.keys())}")

        constructor = cls._registry[spec.name].constructor
        try:
            connector = constructor(spec.extra)
            logger.info(f"Created connector: {spec.name}")
            return connector
        except Exception as e:
            logger.error(f"Failed to create connector {spec.name}: {e}")
            raise ValueError(f"Failed to create connector {spec.name}: {e}")

    @classmethod
    def create_stage_connectors(
        cls,
        plan: StageConnectorPlan,
        *,
        stage_id: int,
        local_rank: int = 0,
        replica_id: int = 0,
    ) -> StageConnectorSet:
        receive = cls.materialize_connector_spec(plan.inbound, "receiver", stage_id, local_rank, replica_id)
        send = cls.materialize_connector_spec(plan.outbound, "sender", stage_id, local_rank, replica_id)

        if receive is not None and send is not None and receive.name == send.name:
            registration = cls._registry.get(receive.name)
            merger = registration.merge_dual if registration is not None else None
            if merger is not None:
                dual = merger(receive.extra, send.extra)
                if dual is not None:
                    connector = cls.create_connector(ConnectorSpec(receive.name, dual))
                    return StageConnectorSet(receive=connector, send=connector)

        return StageConnectorSet(
            receive=cls.create_connector(receive) if receive is not None else None,
            send=cls.create_connector(send) if send is not None else None,
        )

    @staticmethod
    def materialize_connector_spec(
        edge: StageConnectorSpec | None,
        role: str,
        stage_id: int,
        local_rank: int,
        replica_id: int,
    ) -> ConnectorSpec | None:
        if edge is None:
            return None

        extra = dict(edge.spec.extra)
        extra["stage_id"] = stage_id
        extra["role"] = role
        if edge.spec.name in TRANSFER_ENGINE_CONNECTOR_NAMES:
            from .utils.initialization import KV_RANK_PORT_STRIDE, KV_REPLICA_PORT_STRIDE

            offset = replica_id * KV_REPLICA_PORT_STRIDE + local_rank * KV_RANK_PORT_STRIDE
            extra["zmq_port"] = int(os.path.expandvars(str(extra.get("zmq_port", 50051)))) + offset
            if extra.get("sender_zmq_port") is not None:
                extra["sender_zmq_port"] = int(os.path.expandvars(str(extra["sender_zmq_port"]))) + offset
        return ConnectorSpec(edge.spec.name, extra)

    @classmethod
    def list_registered_connectors(cls) -> list[str]:
        """List all registered connector names."""
        return list(cls._registry.keys())


# Register built-in connectors with lazy imports
def _create_mooncake_store_connector(config: dict[str, Any]) -> OmniConnectorBase:
    try:
        from .connectors.mooncake_store_connector import MooncakeStoreConnector
    except ImportError:
        # Fallback import
        import sys

        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from omni_connectors.connectors.mooncake_store_connector import MooncakeStoreConnector
    return MooncakeStoreConnector(config)


def _create_shm_connector(config: dict[str, Any]) -> OmniConnectorBase:
    try:
        from .connectors.shm_connector import SharedMemoryConnector
    except ImportError:
        # Fallback import
        import sys

        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from omni_connectors.connectors.shm_connector import SharedMemoryConnector
    return SharedMemoryConnector(config)


def _create_yuanrong_connector(config: dict[str, Any]) -> OmniConnectorBase:
    try:
        from .connectors.yuanrong_connector import YuanrongConnector
    except ImportError:
        import sys

        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from omni_connectors.connectors.yuanrong_connector import YuanrongConnector
    return YuanrongConnector(config)


def _create_yuanrong_transfer_engine_connector(config: dict[str, Any]) -> OmniConnectorBase:
    try:
        from vllm_omni.platforms.npu.omni_connectors import YuanrongTransferEngineConnector
    except ImportError as exc:
        raise ImportError(
            "YuanrongTransferEngineConnector is only available in the NPU platform "
            "environment. Install the Ascend/Yuanrong runtime dependencies before "
            "using this connector."
        ) from exc
    return YuanrongTransferEngineConnector(config)


def _create_mooncake_transfer_engine_connector(config: dict[str, Any]) -> OmniConnectorBase:
    try:
        from .connectors.mooncake_transfer_engine_connector import MooncakeTransferEngineConnector
    except ImportError:
        import sys

        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from omni_connectors.connectors.mooncake_transfer_engine_connector import MooncakeTransferEngineConnector
    return MooncakeTransferEngineConnector(config)


def _create_mori_transfer_engine_connector(config: dict[str, Any]) -> OmniConnectorBase:
    try:
        from .connectors.mori_transfer_engine_connector import MoriTransferEngineConnector
    except ImportError:
        import sys

        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from omni_connectors.connectors.mori_transfer_engine_connector import MoriTransferEngineConnector
    return MoriTransferEngineConnector(config)


_DIRECTIONAL_KEYS = {
    "from_stage",
    "rank_mapping",
    "role",
    "sender_host",
    "sender_zmq_port",
    "stage_id",
    "to_stage",
    "zmq_port",
}


def _merge_dual_config(receive: dict[str, Any], send: dict[str, Any]) -> dict[str, Any] | None:
    """Merge compatible configs for a connector that can send and receive."""
    shared_keys = (receive.keys() | send.keys()) - _DIRECTIONAL_KEYS
    if any(receive.get(key) != send.get(key) for key in shared_keys):
        return None

    merged = dict(send)
    for key in ("sender_host", "sender_zmq_port"):
        if receive.get(key) is not None:
            merged[key] = receive[key]
    recv_mapping = receive.get("rank_mapping")
    send_mapping = send.get("rank_mapping")
    if recv_mapping != send_mapping:
        merged.pop("rank_mapping", None)
        if recv_mapping is not None:
            merged["recv_rank_mapping"] = recv_mapping
        if send_mapping is not None:
            merged["send_rank_mapping"] = send_mapping
    merged["role"] = "dual"
    return merged


# Register connectors
OmniConnectorFactory.register_connector(
    "MooncakeStoreConnector",
    _create_mooncake_store_connector,
    merge_dual=_merge_dual_config,
)
OmniConnectorFactory.register_connector(
    "MooncakeTransferEngineConnector",
    _create_mooncake_transfer_engine_connector,
    merge_dual=_merge_dual_config,
)
OmniConnectorFactory.register_connector(
    "SharedMemoryConnector",
    _create_shm_connector,
    merge_dual=_merge_dual_config,
)
OmniConnectorFactory.register_connector(
    "YuanrongConnector",
    _create_yuanrong_connector,
    merge_dual=_merge_dual_config,
)
OmniConnectorFactory.register_connector(
    "YuanrongTransferEngineConnector",
    _create_yuanrong_transfer_engine_connector,
    merge_dual=_merge_dual_config,
)
OmniConnectorFactory.register_connector(
    "MoriTransferEngineConnector",
    _create_mori_transfer_engine_connector,
    merge_dual=_merge_dual_config,
)
# Backward-compatible aliases – will be removed in the future
OmniConnectorFactory.register_connector(
    "MooncakeConnector",
    _create_mooncake_store_connector,
    merge_dual=_merge_dual_config,
)
