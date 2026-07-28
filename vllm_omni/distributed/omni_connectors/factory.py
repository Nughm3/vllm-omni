# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .stage_connector import ConnectorRuntimeContext, StageConnectorSet
from .utils.logging import get_connector_logger

if TYPE_CHECKING:
    from .utils.initialization import ResolvedConnectorSpec, StageConnectorPlan

from .connectors.base import OmniConnectorBase
from .utils.config import TRANSFER_ENGINE_CONNECTOR_NAMES, ConnectorSpec

logger = get_connector_logger(__name__)


class OmniConnectorFactory:
    """Factory for creating OmniConnectors."""

    _registry: dict[str, Callable[[dict[str, Any]], OmniConnectorBase]] = {}
    _class_registry: dict[str, type[OmniConnectorBase]] = {}

    @classmethod
    def register_connector(
        cls,
        name: str,
        constructor: Callable[[dict[str, Any]], OmniConnectorBase],
        *,
        connector_cls: type[OmniConnectorBase] | None = None,
    ) -> None:
        """Register a connector constructor (and optionally its class)."""
        if name in cls._registry:
            raise ValueError(f"Connector '{name}' is already registered.")
        cls._registry[name] = constructor
        if connector_cls is not None:
            cls._class_registry[name] = connector_cls
        logger.debug(f"Registered connector: {name}")

    @classmethod
    def create_connector(cls, spec: ConnectorSpec) -> OmniConnectorBase:
        """Create a connector from specification."""
        if spec.name not in cls._registry:
            raise ValueError(f"Unknown connector: {spec.name}. Available: {list(cls._registry.keys())}")

        constructor = cls._registry[spec.name]
        try:
            connector = constructor(spec.extra)
            logger.info(f"Created connector: {spec.name}")
            return connector
        except Exception as e:
            logger.error(f"Failed to create connector {spec.name}: {e}")
            raise ValueError(f"Failed to create connector {spec.name}: {e}") from e

    @classmethod
    def create_stage_connectors(
        cls,
        plan: StageConnectorPlan,
        runtime_context: ConnectorRuntimeContext,
    ) -> StageConnectorSet:
        """Materialize runtime connectors from a typed stage plan.

        Called only inside the owning process after that process's distributed
        context (if any) exists:

        1. Skip edges whose ``owner_scope`` does not match ``runtime_context``
           (chunk transfer adapter vs mixin collision guard).
        2. Apply TP/replica port offset for transfer-engine connectors
           (``STAGE_REPLICA`` forces local_rank=0 — never queries the TP group).
        3. Dual-collapse same-type compatible edges via ``can_share`` /
           ``merge_duplex_specs`` when shared, thread-safe dual operation is
           supported.
        4. Explicit inbound-only → ``send is None``; outbound-only →
           ``receive is None``. Legacy default SHM serves both directions.
        """
        from .utils.initialization import StageConnectorPlan

        if not isinstance(plan, StageConnectorPlan):
            raise TypeError(f"Expected StageConnectorPlan, got {type(plan)!r}")

        if len(plan.inbound) > 1 or len(plan.outbound) > 1:
            raise ValueError("Fan-in and fan-out are not currently supported")

        if plan.uses_legacy_default:
            return cls._materialize_legacy_default(runtime_context)

        recv_spec = cls.resolve_connector_spec(plan.input_spec, runtime_context)
        send_spec = cls.resolve_connector_spec(plan.output_spec, runtime_context)
        if recv_spec is None and send_spec is None:
            return StageConnectorSet(receive=None, send=None)

        if recv_spec is not None and send_spec is not None:
            return cls._build_duplex_or_hybrid(recv_spec, send_spec)

        if recv_spec is not None:
            # Explicit inbound-only: do not reuse the receiver as a sender.
            return StageConnectorSet(receive=cls.create_connector(recv_spec), send=None)

        if send_spec is not None:
            return StageConnectorSet(receive=None, send=cls.create_connector(send_spec))

        return StageConnectorSet(receive=None, send=None)

    @classmethod
    def _materialize_legacy_default(cls, runtime_context: ConnectorRuntimeContext) -> StageConnectorSet:
        """No transfer config → one SHM instance serving both directions."""
        spec = ConnectorSpec(
            name="SharedMemoryConnector",
            extra={"stage_id": runtime_context.stage_id, "role": "dual"},
        )
        conn = cls.create_connector(spec)
        return StageConnectorSet(receive=conn, send=conn)

    @classmethod
    def resolve_connector_spec(
        cls,
        resolved: ResolvedConnectorSpec | None,
        runtime_context: ConnectorRuntimeContext,
    ) -> ConnectorSpec | None:
        if resolved is None:
            return None
        if resolved.owner_scope != runtime_context.owner_scope:
            logger.debug(
                "Skipping %s edge %s->%s: owner_scope=%s != caller=%s",
                resolved.direction.value,
                resolved.edge.from_stage,
                resolved.edge.to_stage,
                resolved.owner_scope.value,
                runtime_context.owner_scope.value,
            )
            return None

        from .utils.initialization import ConnectorOwnerScope
        from .utils.kv_utils import worker_rank_port_offset

        if runtime_context.owner_scope == ConnectorOwnerScope.STAGE_REPLICA:
            port_offset = worker_rank_port_offset(local_rank=0, replica_id=runtime_context.replica_id)
        else:
            if runtime_context.tp_rank is None:
                raise ValueError("TP_WORKER connector resolution requires an explicit tp_rank")
            port_offset = worker_rank_port_offset(
                local_rank=runtime_context.tp_rank,
                replica_id=runtime_context.replica_id,
            )

        extra = dict(resolved.spec.extra)
        extra.setdefault("stage_id", runtime_context.stage_id)
        if resolved.spec.name in TRANSFER_ENGINE_CONNECTOR_NAMES and port_offset:
            if "zmq_port" in extra:
                extra["zmq_port"] = int(extra["zmq_port"]) + port_offset
            if extra.get("sender_zmq_port") is not None:
                extra["sender_zmq_port"] = int(extra["sender_zmq_port"]) + port_offset
        return ConnectorSpec(name=resolved.spec.name, extra=extra)

    @classmethod
    def _build_duplex_or_hybrid(
        cls,
        recv_spec: ConnectorSpec,
        send_spec: ConnectorSpec,
    ) -> StageConnectorSet:
        connector_cls = cls._resolve_connector_class(recv_spec.name)
        if (
            recv_spec.name == send_spec.name
            and connector_cls.capabilities.supports_shared_dual
            and connector_cls.capabilities.thread_safe_dual
            and connector_cls.can_share(recv_spec.extra, send_spec.extra)
        ):
            merged = connector_cls.merge_duplex_specs(recv_spec.extra, send_spec.extra)
            conn = cls.create_connector(ConnectorSpec(name=recv_spec.name, extra=merged))
            logger.info(
                "Dual-collapsed %s inbound/outbound into one shared instance (role=dual)",
                recv_spec.name,
            )
            return StageConnectorSet(receive=conn, send=conn)

        receive = cls.create_connector(recv_spec)
        send = cls.create_connector(send_spec)
        logger.info(
            "Hybrid connectors: recv=%s send=%s",
            recv_spec.name,
            send_spec.name,
        )
        return StageConnectorSet(receive=receive, send=send)

    @classmethod
    def _resolve_connector_class(cls, name: str) -> type[OmniConnectorBase]:
        if name in cls._class_registry:
            return cls._class_registry[name]
        return OmniConnectorBase

    @classmethod
    def list_registered_connectors(cls) -> list[str]:
        """List all registered connector names."""
        return list(cls._registry.keys())


# Register built-in connectors with lazy imports
def _create_mooncake_store_connector(config: dict[str, Any]) -> OmniConnectorBase:
    from .connectors.mooncake_store_connector import MooncakeStoreConnector

    return MooncakeStoreConnector(config)


def _create_shm_connector(config: dict[str, Any]) -> OmniConnectorBase:
    from .connectors.shm_connector import SharedMemoryConnector

    return SharedMemoryConnector(config)


def _create_yuanrong_connector(config: dict[str, Any]) -> OmniConnectorBase:
    from .connectors.yuanrong_connector import YuanrongConnector

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
    from .connectors.mooncake_transfer_engine_connector import MooncakeTransferEngineConnector

    return MooncakeTransferEngineConnector(config)


def _create_mori_transfer_engine_connector(config: dict[str, Any]) -> OmniConnectorBase:
    from .connectors.mori_transfer_engine_connector import MoriTransferEngineConnector

    return MoriTransferEngineConnector(config)


def _register_builtins() -> None:
    from .connectors.shm_connector import SharedMemoryConnector

    OmniConnectorFactory.register_connector(
        "SharedMemoryConnector",
        _create_shm_connector,
        connector_cls=SharedMemoryConnector,
    )
    OmniConnectorFactory.register_connector("MooncakeStoreConnector", _create_mooncake_store_connector)
    OmniConnectorFactory.register_connector("YuanrongConnector", _create_yuanrong_connector)

    try:
        from .connectors.mooncake_transfer_engine_connector import MooncakeTransferEngineConnector

        OmniConnectorFactory.register_connector(
            "MooncakeTransferEngineConnector",
            _create_mooncake_transfer_engine_connector,
            connector_cls=MooncakeTransferEngineConnector,
        )
    except ImportError:
        OmniConnectorFactory.register_connector(
            "MooncakeTransferEngineConnector",
            _create_mooncake_transfer_engine_connector,
        )

    try:
        from .connectors.mori_transfer_engine_connector import MoriTransferEngineConnector

        OmniConnectorFactory.register_connector(
            "MoriTransferEngineConnector",
            _create_mori_transfer_engine_connector,
            connector_cls=MoriTransferEngineConnector,
        )
    except ImportError:
        OmniConnectorFactory.register_connector(
            "MoriTransferEngineConnector",
            _create_mori_transfer_engine_connector,
        )

    try:
        from vllm_omni.platforms.npu.omni_connectors import YuanrongTransferEngineConnector

        OmniConnectorFactory.register_connector(
            "YuanrongTransferEngineConnector",
            _create_yuanrong_transfer_engine_connector,
            connector_cls=YuanrongTransferEngineConnector,
        )
    except ImportError:
        OmniConnectorFactory.register_connector(
            "YuanrongTransferEngineConnector",
            _create_yuanrong_transfer_engine_connector,
        )

    # Backward-compatible aliases – will be removed in the future
    OmniConnectorFactory.register_connector("MooncakeConnector", _create_mooncake_store_connector)


_register_builtins()
