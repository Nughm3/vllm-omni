# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm_omni.engine import ConnectorEndpoint

from .connectors.base import ConnectorCapabilities, OmniConnectorBase
from .connectors.mooncake_store_connector import MooncakeStoreConnector
from .connectors.shm_connector import SharedMemoryConnector
from .connectors.yuanrong_connector import YuanrongConnector
from .stage_connector import ConnectorRuntimeContext, StageConnectorSet

try:
    from vllm_omni.platforms.npu.omni_connectors.yuanrong_transfer_engine_connector import (
        YuanrongTransferEngineConnector,
    )
except ImportError:
    YuanrongTransferEngineConnector = None

try:
    from .connectors.mooncake_transfer_engine_connector import MooncakeTransferEngineConnector
except ImportError:
    MooncakeTransferEngineConnector = None  # RDMA deps (msgspec/zmq/mooncake) not installed

try:
    from .connectors.mori_transfer_engine_connector import MoriTransferEngineConnector
except ImportError:
    MoriTransferEngineConnector = None  # RDMA deps (msgspec/zmq/mori) not installed
from .factory import OmniConnectorFactory
from .utils.config import ConnectorSpec, OmniTransferConfig
from .utils.initialization import (
    ConnectorOwner,
    ResolvedConnectorSpec,
    StageConnectorPlan,
    StageEdge,
    get_connectors_config_for_stage,
    load_omni_transfer_config,
    resolve_stage_connector_plan,
)

# Backward-compatible alias: MooncakeConnector was renamed to MooncakeStoreConnector.
# Keep this alias for at least one release cycle.
MooncakeConnector = MooncakeStoreConnector

__all__ = [
    # Config
    "ConnectorSpec",
    "OmniTransferConfig",
    # Planning
    "ConnectorOwner",
    "StageEdge",
    "ResolvedConnectorSpec",
    "StageConnectorPlan",
    "resolve_stage_connector_plan",
    "ConnectorEndpoint",
    # Construction / runtime
    "ConnectorCapabilities",
    "ConnectorRuntimeContext",
    "StageConnectorSet",
    # Base classes and implementations
    "OmniConnectorBase",
    # Factory
    "OmniConnectorFactory",
    # Specific implementations
    "MooncakeConnector",  # compat alias → MooncakeStoreConnector
    "MooncakeStoreConnector",
    "MooncakeTransferEngineConnector",
    "MoriTransferEngineConnector",
    "SharedMemoryConnector",
    "YuanrongConnector",
    "YuanrongTransferEngineConnector",
    # Utilities
    "load_omni_transfer_config",
    "get_connectors_config_for_stage",
]
