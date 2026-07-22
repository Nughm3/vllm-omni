# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Utilities for OmniConnector configuration and validation."""

import json
import sys
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..factory import OmniConnectorFactory
from .config import TRANSFER_ENGINE_CONNECTOR_NAMES, ConnectorSpec, OmniTransferConfig
from .env import expand_env_int
from .local_rank import get_connector_local_rank, get_omni_replica_id
from .logging import get_connector_logger

if TYPE_CHECKING:
    from ..connectors.base import OmniConnectorBase
else:
    OmniConnectorBase = Any

logger = get_connector_logger(__name__)

# Reserve a separate port range for KV-transfer sockets so they do not
# collide with request-forwarding endpoints that share the same base port.
KV_TRANSFER_PORT_OFFSET = 100

# Port stride between TP ranks so each worker binds a unique ZMQ port
# when TP > 1.  Must be larger than the maximum number of pipeline stages.
# Formula:
#   zmq_port = base + KV_TRANSFER_PORT_OFFSET
#            + replica * KV_REPLICA_PORT_STRIDE
#            + rank * KV_RANK_PORT_STRIDE
#            + stage
KV_RANK_PORT_STRIDE = 16

# Port stride between Omni replicas of the same stage, so co-located replicas
# each bind a distinct port range. This reserves a comfortably sized block
# per replica for TP-rank and stage offsets.
KV_REPLICA_PORT_STRIDE = 1024


def initialize_connectors_from_config(
    config_path: str | Path | None = None,
    default_shm_threshold: int = 65536,
    purpose: str = "request_forwarding",
    caller_stage_id: int | str | None = None,
    is_sender: bool | None = None,
) -> tuple[OmniTransferConfig | None, dict[tuple[str, str], OmniConnectorBase]]:
    """
    Initialize connectors from configuration file.

    Returns:
        tuple: (OmniTransferConfig, dict of {(from, to): connector_instance})
    """
    transfer_config = load_omni_transfer_config(config_path, default_shm_threshold=default_shm_threshold)

    if not transfer_config:
        logger.info("No OmniTransferConfig provided")
        return None, {}

    # create connectors from config
    connectors = create_connectors_from_config(
        transfer_config.connectors,
        purpose=purpose,
        caller_stage_id=caller_stage_id,
        is_sender=is_sender,
    )
    return transfer_config, connectors


def create_connectors_from_config(
    connectors_config: dict[tuple[str, str], ConnectorSpec],
    purpose: str = "request_forwarding",
    caller_stage_id: int | str | None = None,
    is_sender: bool | None = None,
) -> dict[tuple[str, str], OmniConnectorBase]:
    """
    Create connectors from config.

    Args:
        connectors_config: A dictionary of connector configurations.

    Returns:
        A dictionary of connectors.
    """
    purpose_port_offsets = {
        "request_forwarding": 0,
        "kv_transfer": KV_TRANSFER_PORT_OFFSET,
    }
    port_offset = purpose_port_offsets.get(purpose, 0)
    orchestrator_port_offset = 200

    connectors = {}
    for edge_key, connector_spec in connectors_config.items():
        from_stage, to_stage = edge_key
        try:
            if connector_spec.name in TRANSFER_ENGINE_CONNECTOR_NAMES:
                extra = dict(connector_spec.extra) if connector_spec.extra else {}
                base_port = expand_env_int(extra.get("zmq_port", 50051), "zmq_port")
                try:
                    stage_offset = int(from_stage)
                except (TypeError, ValueError):
                    stage_offset = 0

                is_orchestrator = str(caller_stage_id) == "orchestrator"
                local_rank = 0 if is_orchestrator else get_connector_local_rank()
                replica_offset = 0 if is_orchestrator else get_omni_replica_id() * KV_REPLICA_PORT_STRIDE
                if is_orchestrator:
                    rank0_port = base_port + orchestrator_port_offset + stage_offset
                else:
                    rank0_port = base_port + port_offset + stage_offset
                adjusted_port = rank0_port + replica_offset + local_rank * KV_RANK_PORT_STRIDE
                extra["zmq_port"] = adjusted_port

                if is_sender is not None:
                    extra["role"] = "sender" if is_sender else "receiver"
                    if not is_sender:
                        extra.setdefault("sender_host", extra.get("host", "127.0.0.1"))
                        extra.setdefault("sender_zmq_port", adjusted_port)
                elif caller_stage_id is not None:
                    caller_str = str(caller_stage_id)
                    if caller_str == from_stage:
                        extra["role"] = "sender"
                    elif caller_str == to_stage:
                        extra["role"] = "receiver"
                        extra.setdefault("sender_host", extra.get("host", "127.0.0.1"))
                        extra.setdefault("sender_zmq_port", adjusted_port)
                    else:
                        extra["role"] = "sender"
                else:
                    extra["role"] = extra.get("role", "auto")

                connector = OmniConnectorFactory.create_connector(ConnectorSpec(name=connector_spec.name, extra=extra))
            else:
                connector = OmniConnectorFactory.create_connector(connector_spec)
            connectors[edge_key] = connector
            logger.info(
                "Created connector for %s -> %s: %s",
                from_stage,
                to_stage,
                type(connector).__name__,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize connector for edge {edge_key}: {e}") from e

    return connectors


def _stage_port_offset(stage_id: str | int) -> int:
    try:
        return int(stage_id)
    except (TypeError, ValueError):
        return 0


def _apply_transfer_engine_ports(
    extra: dict[str, Any],
    *,
    from_stage: str,
    is_put: bool,
    is_get: bool,
) -> None:
    """Resolve listen / query ZMQ ports for transfer-engine connectors.

    ``zmq_port`` in YAML is treated as a *base* port. The local listen port
    for the outbound edge is ``base + from_stage``. The upstream query port
    for the inbound edge is ``base + upstream_stage`` unless
    ``sender_zmq_port`` is already set.
    """
    base_port = expand_env_int(extra.get("zmq_port", 50051), "zmq_port")
    local_rank = get_connector_local_rank()
    port_offset = get_omni_replica_id() * KV_REPLICA_PORT_STRIDE + local_rank * KV_RANK_PORT_STRIDE

    if is_put:
        # Outbound edge: from_stage is this stage — bind base + this_stage (+ replica/rank).
        extra["zmq_port"] = base_port + _stage_port_offset(from_stage) + port_offset
    if is_get:
        # Inbound edge: query the upstream sender's listen port.
        computed_sender_port = base_port + _stage_port_offset(from_stage) + port_offset
        if extra.get("sender_zmq_port") is None:
            extra["sender_zmq_port"] = computed_sender_port
        else:
            extra["sender_zmq_port"] = expand_env_int(extra["sender_zmq_port"], "sender_zmq_port")
        extra.setdefault("sender_host", extra.get("host", "127.0.0.1"))


def get_connectors_config_for_stage(transfer_config: OmniTransferConfig | None, stage_id: str | int) -> dict[str, Any]:
    """
    Extract connector configurations relevant for a specific stage worker.

    Returns a dict keyed by edge direction for edges that touch this stage:
    {
        "from_stage_X": {"spec": {"name": ..., "extra": {...}}},  # inbound
        "to_stage_Y":   {"spec": {"name": ..., "extra": {...}}},  # outbound
    }

    Each edge entry is role-aware for this stage:
    - inbound  → role=receiver
    - outbound → role=sender

    Middle stages receive both keys; ``resolve_stage_connector_specs`` splits
    them into ``(input_spec, output_spec)``, which the mixin builds as one
    dual instance (same type) or two instances (hybrid).
    """
    if not transfer_config:
        return {}

    stage_connectors_config: dict[str, Any] = {}
    target_stage = str(stage_id)

    for (from_stage, to_stage), spec in transfer_config.connectors.items():
        is_put = from_stage == target_stage
        is_get = to_stage == target_stage
        if not is_put and not is_get:
            continue

        extra = dict(spec.extra) if spec.extra else {}

        # Role for single-direction connectors (Mori, Yuanrong, Mooncake).
        if is_put and not is_get:
            extra.setdefault("role", "sender")
        elif is_get and not is_put:
            extra.setdefault("role", "receiver")

        if spec.name in TRANSFER_ENGINE_CONNECTOR_NAMES:
            _apply_transfer_engine_ports(
                extra,
                from_stage=from_stage,
                is_put=is_put,
                is_get=is_get,
            )

        if is_get:
            stage_connectors_config[f"from_stage_{from_stage}"] = {"spec": {"name": spec.name, "extra": extra}}
        if is_put:
            # Outbound edges for every stage (not only stage 0) so middle-stage
            # dual merge and stage-0 sender spec extraction both work.
            stage_connectors_config[f"to_stage_{to_stage}"] = {"spec": {"name": spec.name, "extra": dict(extra)}}

    return stage_connectors_config


def resolve_stage_connector_specs(
    stage_connectors_cfg: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Split per-edge stage connector entries into ``(input_spec, output_spec)``.

    ``input_spec`` drives the inbound (recv) connector, ``output_spec`` the
    outbound (send) connector. Each is a flat single-direction spec
    (``{"name", "extra"}``) carrying its own ``role`` / ports / ``rank_mapping``.
    Either may be ``None`` — stage 0 has no input, the last stage no output.

    The mixin decides whether both specs collapse to one dual instance
    (same connector type) or stay as two (hybrid, e.g. Mooncake in / SHM out).
    """
    input_spec: dict[str, Any] | None = None
    output_spec: dict[str, Any] | None = None

    for edge_key, cfg in stage_connectors_cfg.items():
        spec = cfg.get("spec") or {}
        name = spec.get("name")
        if not name:
            continue
        flat = {"name": name, "extra": dict(spec.get("extra") or {})}
        if edge_key.startswith("from_stage_"):
            if input_spec is not None:
                raise ValueError(
                    "Fan-in (multiple inbound edges) is not supported: got a second "
                    f"inbound edge {edge_key!r} in addition to an already-resolved one."
                )
            input_spec = flat
        elif edge_key.startswith("to_stage_"):
            if output_spec is not None:
                raise ValueError(
                    "Fan-out (multiple outbound edges) is not supported: got a second "
                    f"outbound edge {edge_key!r} in addition to an already-resolved one."
                )
            output_spec = flat

    return input_spec, output_spec


def load_omni_transfer_config(
    config_path: str | Path | None = None,
    config_dict: dict[str, Any] | None = None,
    default_shm_threshold: int = 65536,
) -> OmniTransferConfig | None:
    """Load OmniTransferConfig from file or dict."""
    if config_path is None and config_dict is None:
        # Even if no config provided, we might want to return a default config with SHM connectors
        # But without stage info we can't do much.
        return None

    if config_path is not None:
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, encoding="utf-8") as f:
            if config_path.suffix.lower() == ".json":
                config_dict = json.load(f)
            elif config_path.suffix.lower() in [".yaml", ".yml"]:
                try:
                    import yaml

                    config_dict = yaml.safe_load(f)
                except ImportError:
                    raise ImportError("PyYAML required for YAML config files")
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")

    if config_dict is None:
        return None

    # Normalize new-schema (top-level ``connectors`` + ``stages``) into the
    # legacy ``runtime.connectors`` + ``stage_args`` shape the parser reads.
    if "stages" in config_dict and "stage_args" not in config_dict:
        normalized: dict[str, Any] = dict(config_dict)
        runtime = dict(normalized.get("runtime") or {})
        if "connectors" in normalized and "connectors" not in runtime:
            runtime["connectors"] = normalized["connectors"]
        if "edges" in normalized and "edges" not in runtime:
            runtime["edges"] = normalized["edges"]
        normalized["runtime"] = runtime
        normalized["stage_args"] = normalized["stages"]
        config_dict = normalized

    # Parse connectors
    connectors = {}
    runtime_config = config_dict.get("runtime", {})

    # Parse global connectors (from runtime.connectors)
    global_connectors = runtime_config.get("connectors", {})

    # Parse stage-level connectors
    stage_args = config_dict.get("stage_args", [])
    expected_edges: set[tuple[str, str]] = set()
    for stage_config in stage_args:
        stage_id = str(stage_config["stage_id"])

        # Input connectors (this stage is the receiver)
        # NOTE: role is NOT injected here — the shared edge-level ConnectorSpec
        # must remain role-neutral.  Role is injected per-stage in
        # get_connectors_config_for_stage() / resolve_omni_kv_config_for_stage().
        for input_key, conn_ref in stage_config.get("input_connectors", {}).items():
            if isinstance(conn_ref, str):
                # Reference to global connector
                if conn_ref in global_connectors:
                    conn_config = global_connectors[conn_ref]
                    extra = dict(conn_config.get("extra", {}))
                else:
                    raise ValueError(f"Undefined connector reference: {conn_ref}")
                connector = ConnectorSpec(name=conn_config["name"], extra=extra)
            else:
                # Inline connector definition
                extra = dict(conn_ref.get("extra", {}))
                connector = ConnectorSpec(name=conn_ref["name"], extra=extra)

            # Parse from_stage from key (e.g., "from_stage_0" -> "0")
            from_stage = input_key.replace("from_stage_", "")
            edge_key = (from_stage, stage_id)
            # Both sides of an edge may define the same connector reference;
            # verify consistency if already registered.
            if edge_key in connectors:
                existing = connectors[edge_key]
                if existing.name != connector.name:
                    raise ValueError(
                        f"Connector type mismatch for edge {edge_key[0]}->{edge_key[1]}: "
                        f"previously registered as '{existing.name}', "
                        f"but input_connectors of stage {stage_id} specifies '{connector.name}'"
                    )
            else:
                connectors[edge_key] = connector
            expected_edges.add(edge_key)

        # Output connectors (this stage is the sender)
        for output_key, conn_ref in stage_config.get("output_connectors", {}).items():
            if isinstance(conn_ref, str):
                # Reference to global connector
                if conn_ref in global_connectors:
                    conn_config = global_connectors[conn_ref]
                    extra = dict(conn_config.get("extra", {}))
                else:
                    raise ValueError(f"Undefined connector reference: {conn_ref}")
                connector = ConnectorSpec(name=conn_config["name"], extra=extra)
            else:
                # Inline connector definition
                extra = dict(conn_ref.get("extra", {}))
                connector = ConnectorSpec(name=conn_ref["name"], extra=extra)

            # Parse to_stage from key (e.g., "to_stage_1" -> "1")
            to_stage = output_key.replace("to_stage_", "")
            edge_key = (stage_id, to_stage)
            if edge_key in connectors:
                existing = connectors[edge_key]
                if existing.name != connector.name:
                    raise ValueError(
                        f"Connector type mismatch for edge {edge_key[0]}->{edge_key[1]}: "
                        f"previously registered as '{existing.name}', "
                        f"but output_connectors of stage {stage_id} specifies '{connector.name}'"
                    )
            else:
                connectors[edge_key] = connector
            expected_edges.add(edge_key)

    # Auto-configure SharedMemoryConnector for missing edges based on runtime edges / engine_input_source
    if stage_args:
        try:
            # Prefer explicit runtime edges if provided
            runtime_edges = runtime_config.get("edges", [])
            if isinstance(runtime_edges, list) and runtime_edges:
                for edge in runtime_edges:
                    from_stage = edge.get("from")
                    to_stage = edge.get("to")
                    if from_stage is None or to_stage is None:
                        continue
                    edge_key = (str(from_stage), str(to_stage))
                    expected_edges.add(edge_key)
                    if edge_key not in connectors:
                        logger.info(f"Auto-configuring SharedMemoryConnector for edge {edge_key}")
                        connectors[edge_key] = ConnectorSpec(
                            name="SharedMemoryConnector",
                            extra={"shm_threshold_bytes": default_shm_threshold},
                        )

            # Fallback: infer edges from engine_input_source for each stage
            for stage_config in stage_args:
                to_stage = str(stage_config["stage_id"])
                # Check explicit input sources
                sources = stage_config.get("engine_input_source", [])

                for from_stage in sources:
                    from_stage_str = str(from_stage)
                    edge_key = (from_stage_str, to_stage)
                    expected_edges.add(edge_key)

                    if edge_key not in connectors:
                        logger.info(f"Auto-configuring SharedMemoryConnector for edge {edge_key}")
                        connectors[edge_key] = ConnectorSpec(
                            name="SharedMemoryConnector", extra={"shm_threshold_bytes": default_shm_threshold}
                        )

        except Exception as e:
            logger.warning(f"Failed to auto-configure SHM connectors: {e}")

    # Fail fast if any expected edge is still missing a connector
    missing_edges = [edge for edge in expected_edges if edge not in connectors]
    if missing_edges:
        missing_str = ", ".join([f"{f}->{t}" for f, t in missing_edges])
        raise ValueError(
            "Connector configuration missing for edges: "
            f"{missing_str}. Define connectors or allow auto SHM creation for these edges."
        )

    config = OmniTransferConfig(connectors=connectors)

    logger.info(f"Loaded OmniTransferConfig with {len(connectors)} connector configurations")
    return config


# High-level management functions


def initialize_orchestrator_connectors(
    config_path: str | None, worker_backend: str | None = "multi_process", shm_threshold_bytes: int = 65536
) -> tuple[OmniTransferConfig | None, dict[tuple[str, str], OmniConnectorBase]]:
    """Initialize connectors shared at orchestrator level.
    Args:
        config_path: The path to the configuration file.
        worker_backend: The backend to use for the worker.
    Returns:
        A tuple containing the OmniTransferConfig and a dictionary of connectors.
    """
    if worker_backend == "ray":
        default_shm_threshold = sys.maxsize
    else:
        default_shm_threshold = max(0, shm_threshold_bytes)
    transfer_config, connectors = initialize_connectors_from_config(
        config_path,
        default_shm_threshold=default_shm_threshold,
        purpose="request_forwarding",
        caller_stage_id="orchestrator",
        is_sender=True,
    )
    return transfer_config, connectors


def get_stage_connector_config(
    transfer_config: OmniTransferConfig | None,
    stage_id: int,
) -> dict[str, Any]:
    """Return the serialized connector config payload for a specific stage."""
    if transfer_config is None:
        return {}

    try:
        return get_connectors_config_for_stage(transfer_config, stage_id)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning(
            "Failed to build connector config for stage %s: %s. Using IPC fallback.",
            stage_id,
            exc,
        )
        return {}


def build_stage_connectors(
    stage_id: int,
    connectors_config: dict[str, Any],
    purpose: str = "request_forwarding",
) -> dict[tuple[str, str], Any] | None:
    """Instantiate OmniConnectors for a stage based on config.

    Deprecated: prefer ``get_stage_connector_config`` plus the unified
    connector factory. Kept as a thin shim so legacy callers keep working.
    """
    warnings.warn(
        "build_stage_connectors is deprecated; use get_stage_connector_config and OmniConnectorFactory instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if not connectors_config:
        return {}

    logger.info(
        "[Stage-%s] Initializing OmniConnectors with config keys: %s",
        stage_id,
        list(connectors_config.keys()),
    )

    from .config import ConnectorSpec

    connectors: dict[tuple[str, str], Any] = {}
    # Convert dictionary-formatted config to ConnectorSpec objects.
    # Only instantiate INPUT connectors ("from_stage_X") — the stage worker
    # only receives via connectors.  Output connectors are handled at the
    # orchestrator level (try_send_via_connector uses orchestrator connectors).
    stage_connector_specs = {}
    for key, config in connectors_config.items():
        if not key.startswith("from_stage_"):
            continue

        from_stage = key.replace("from_stage_", "")
        spec_dict = config.get("spec", {})
        if not spec_dict:
            continue

        connector_spec = ConnectorSpec(
            name=spec_dict.get("name", "SharedMemoryConnector"),
            extra=spec_dict.get("extra", {}),
        )
        stage_connector_specs[(str(from_stage), str(stage_id))] = connector_spec

    try:
        # Use unified connector creation logic
        connectors = create_connectors_from_config(
            stage_connector_specs,
            purpose=purpose,
            caller_stage_id=stage_id,
            is_sender=False,
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        # Fail fast so the stage does not start with missing connectors.
        logger.exception("[Stage-%s] Failed to initialize connectors: %s", stage_id, exc)
        raise

    return connectors


def resolve_omni_kv_config_for_stage(
    transfer_cfg: OmniTransferConfig | None, stage_id: int | str
) -> tuple[dict[str, Any] | None, str | None, str | None]:
    """Resolve connector configuration for a specific stage (Sender/Receiver).

    This determines the primary connector configuration to be injected into the
    engine arguments, prioritizing outgoing edges (Sender role).
    """
    if not transfer_cfg or not getattr(transfer_cfg, "connectors", None):
        return None, None, None

    stage_id_str = str(stage_id)

    # Find outgoing edges (Sender logic)
    outgoing = [
        (to_stage, spec)
        for (from_stage, to_stage), spec in transfer_cfg.connectors.items()
        if from_stage == stage_id_str
    ]

    # Find incoming edges (Receiver logic)
    incoming = [
        (from_stage, spec)
        for (from_stage, to_stage), spec in transfer_cfg.connectors.items()
        if to_stage == stage_id_str
    ]

    omni_conn_cfg = None
    omni_from = None
    omni_to = None

    # Prioritize outgoing (Sender) if exists, else check incoming (Receiver).
    # Inject direction-specific role so the connector initializes correctly.
    if outgoing:
        if len(outgoing) > 1:
            logger.debug(
                "Stage-%s has %d outgoing edges; using the smallest to_stage",
                stage_id,
                len(outgoing),
            )
        outgoing.sort(key=lambda x: int(x[0]) if str(x[0]).isdigit() else str(x[0]))
        to_s, spec = outgoing[0]
        omni_conn_cfg = {"type": spec.name, **(spec.extra or {})}
        omni_conn_cfg.setdefault("role", "sender")
        omni_from = stage_id_str
        omni_to = str(to_s)
    elif incoming:
        # For receiver, pick one incoming edge to configure the connector
        incoming.sort(key=lambda x: int(x[0]) if str(x[0]).isdigit() else str(x[0]))
        from_s, spec = incoming[0]
        omni_conn_cfg = {"type": spec.name, **(spec.extra or {})}
        omni_conn_cfg.setdefault("role", "receiver")
        omni_from = str(from_s)
        omni_to = stage_id_str

    return omni_conn_cfg, omni_from, omni_to
