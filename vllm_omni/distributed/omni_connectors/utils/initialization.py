# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Utilities for OmniConnector configuration and validation."""

import enum
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import TRANSFER_ENGINE_CONNECTOR_NAMES, ConnectorSpec, OmniTransferConfig
from .env import expand_env_int
from .kv_utils import KVTPTopology, validate_kv_tp_topology
from .logging import get_connector_logger

logger = get_connector_logger(__name__)


class ConnectorOwner(enum.Enum):
    """Which process owns bind/materialization for an edge.

    Materialization skips edges whose ``owner_scope`` does not match the
    caller's scope so chunk transfer adapter and the mixin cannot bind the same
    endpoint.
    """

    CHUNK_TRANSFER_ADAPTER = "CHUNK_TRANSFER_ADAPTER"
    CONNECTOR_MIXIN = "CONNECTOR_MIXIN"


@dataclass(frozen=True, slots=True)
class StageEdge:
    """Directed pipeline edge between two stage ids."""

    from_stage: int
    to_stage: int


@dataclass(frozen=True, slots=True)
class ResolvedConnectorSpec:
    """Rank/replica-agnostic connector template for one edge direction.

    Ports in ``spec.extra`` are stage-level *base* ports only. Per-TP-rank /
    per-replica offsets are applied later by ``materialize_stage_connectors``
    inside the owning process.
    """

    edge: StageEdge
    spec: ConnectorSpec
    owner: ConnectorOwner
    kv_topology: KVTPTopology | None = None


@dataclass(frozen=True, slots=True)
class StageConnectorPlan:
    """Typed, edge-preserving connector plan for one stage.

    ``inbound`` / ``outbound`` are tuples so fan-in/fan-out can be represented
    later; today resolution still rejects more than one edge per direction.
    ``uses_legacy_default`` is True only when *no* transfer config was provided
    (legacy single-node SHM fallback). An explicit empty config yields empty
    tuples with ``uses_legacy_default=False``.
    """

    inbound: tuple[ResolvedConnectorSpec, ...] = ()
    outbound: tuple[ResolvedConnectorSpec, ...] = ()
    uses_legacy_default: bool = False

    @property
    def input_spec(self) -> ResolvedConnectorSpec | None:
        return self.inbound[0] if self.inbound else None

    @property
    def output_spec(self) -> ResolvedConnectorSpec | None:
        return self.outbound[0] if self.outbound else None

    def to_model_connector_configs(self, stage_id: int) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """Dict view for model processors / mixin until they read the plan.

        Mirrors ``OmniEngineArgs.create_model_config`` legacy SHM behavior:
        ``uses_legacy_default`` fabricates an inbound SharedMemoryConnector.
        """
        if self.uses_legacy_default:
            return (
                {"name": "SharedMemoryConnector", "extra": {"stage_id": stage_id}},
                None,
            )

        def _flat(resolved: ResolvedConnectorSpec | None) -> dict[str, Any] | None:
            if resolved is None:
                return None
            extra = dict(resolved.spec.extra)
            extra["stage_id"] = stage_id
            extra["from_stage"] = resolved.edge.from_stage
            extra["to_stage"] = resolved.edge.to_stage
            return {"name": resolved.spec.name, "extra": extra}

        return _flat(self.input_spec), _flat(self.output_spec)


def _parse_stage_id(stage_id: str | int) -> int:
    try:
        return int(stage_id)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Stage id must be an integer, got {stage_id!r}") from exc


def _kv_topology_from_spec(spec: ConnectorSpec) -> KVTPTopology | None:
    rank_mapping = spec.extra.get("rank_mapping")
    if not isinstance(rank_mapping, dict):
        return None
    topology = KVTPTopology(
        source_tp_size=int(rank_mapping.get("from_tp", 1)),
        target_tp_size=int(rank_mapping.get("to_tp", 1)),
        # Planning runs before a worker-local TP rank exists. Materialization
        # replaces this template rank inside the owning worker process.
        local_rank=0,
    )
    validate_kv_tp_topology(topology)
    return topology


def stage_connector_plan_from_model_config(
    model_config: Any,
    *,
    owner_scope: ConnectorOwner | None = None,
) -> StageConnectorPlan:
    """Return the typed plan carried by ``model_config`` or derive a legacy one."""
    plan = getattr(model_config, "stage_connector_plan", None)
    if isinstance(plan, StageConnectorPlan):
        return plan

    stage_id = _parse_stage_id(getattr(model_config, "stage_id", 0))
    if owner_scope is None:
        owner_scope = (
            ConnectorOwner.CHUNK_TRANSFER_ADAPTER
            if bool(getattr(model_config, "async_chunk", False))
            else ConnectorOwner.CONNECTOR_MIXIN
        )

    def _config(attr: str) -> ConnectorSpec | None:
        raw = getattr(model_config, attr, None)
        if raw is None:
            return None
        if isinstance(raw, dict):
            name = raw.get("name")
            extra = raw.get("extra")
        else:
            name = getattr(raw, "name", None)
            extra = getattr(raw, "extra", None)
        if extra is None:
            extra = {}
        if not isinstance(name, str) or not name:
            raise ValueError(f"Invalid {attr}: missing connector name")
        if not isinstance(extra, dict):
            raise TypeError(f"Invalid extra config for connector {name}: expected dict, got {type(extra).__name__}")
        return ConnectorSpec(name=name, extra=dict(extra))

    input_connector = _config("stage_input_connector_config")
    output_connector = _config("stage_output_connector_config")
    if input_connector is None and output_connector is None:
        has_explicit_directional_config = hasattr(model_config, "stage_input_connector_config") or hasattr(
            model_config, "stage_output_connector_config"
        )
        return StageConnectorPlan(uses_legacy_default=not has_explicit_directional_config)

    inbound: tuple[ResolvedConnectorSpec, ...] = ()
    if input_connector is not None:
        from_stage = int(input_connector.extra.get("from_stage", max(0, stage_id - 1)))
        inbound = (
            ResolvedConnectorSpec(
                edge=StageEdge(from_stage=from_stage, to_stage=stage_id),
                spec=input_connector,
                owner=owner_scope,
                kv_topology=_kv_topology_from_spec(input_connector),
            ),
        )

    outbound: tuple[ResolvedConnectorSpec, ...] = ()
    if output_connector is not None:
        to_stage = int(output_connector.extra.get("to_stage", stage_id + 1))
        outbound = (
            ResolvedConnectorSpec(
                edge=StageEdge(from_stage=stage_id, to_stage=to_stage),
                spec=output_connector,
                owner=owner_scope,
                kv_topology=_kv_topology_from_spec(output_connector),
            ),
        )

    return StageConnectorPlan(inbound=inbound, outbound=outbound)


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
    """Resolve listen / query ZMQ *base* ports for transfer-engine connectors.

    ``zmq_port`` in YAML is a *base* port; the listen port is ``base +
    from_stage``. This deliberately excludes the per-TP-rank / per-replica
    offset — this runs once per stage before per-rank worker processes
    exist, so baking a rank-resolved port in here would copy the same port
    to every rank. The factory adds that offset inside the owning process.
    """
    base_port = expand_env_int(extra.get("zmq_port", 50051), "zmq_port")
    stage_base_port = base_port + _stage_port_offset(from_stage)

    if is_put:
        # Outbound edge: from_stage is this stage — bind base + this_stage.
        extra["zmq_port"] = stage_base_port
    if is_get:
        # Inbound edge: query the upstream sender's listen port.
        if extra.get("sender_zmq_port") is None:
            extra["sender_zmq_port"] = stage_base_port
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

    Middle stages receive both keys; ``resolve_stage_connector_plan`` preserves
    them as typed inbound/outbound edge specifications.
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


def resolve_stage_connector_plan(
    transfer_config: OmniTransferConfig | None,
    stage_id: int | str,
    *,
    async_chunk: bool = False,
) -> StageConnectorPlan:
    """Resolve a typed, rank/replica-agnostic connector plan for ``stage_id``.

    Reuses :func:`get_connectors_config_for_stage` for edge collection and
    transfer-engine *base* port math. Does **not** apply TP/replica port
    offsets — those belong in ``materialize_stage_connectors`` inside the
    owning process.

    ``async_chunk=True`` marks every edge ``STAGE_REPLICA``-owned (chunk
    transfer adapter); otherwise edges are ``TP_WORKER``-owned (mixin).
    """
    if transfer_config is None:
        return StageConnectorPlan(inbound=(), outbound=(), uses_legacy_default=True)

    stage_connectors_cfg = get_connectors_config_for_stage(transfer_config, stage_id)
    owner_scope = ConnectorOwner.CHUNK_TRANSFER_ADAPTER if async_chunk else ConnectorOwner.CONNECTOR_MIXIN
    this_stage = _parse_stage_id(stage_id)

    inbound: list[ResolvedConnectorSpec] = []
    outbound: list[ResolvedConnectorSpec] = []

    for edge_key, cfg in stage_connectors_cfg.items():
        spec_dict = cfg.get("spec") or {}
        name = spec_dict.get("name")
        if not name:
            continue
        connector_spec = ConnectorSpec(name=name, extra=dict(spec_dict.get("extra") or {}))

        if edge_key.startswith("from_stage_"):
            if inbound:
                raise ValueError(
                    "Fan-in (multiple inbound edges) is not supported: got a second "
                    f"inbound edge {edge_key!r} in addition to an already-resolved one."
                )
            from_stage = _parse_stage_id(edge_key.removeprefix("from_stage_"))
            inbound.append(
                ResolvedConnectorSpec(
                    edge=StageEdge(from_stage=from_stage, to_stage=this_stage),
                    spec=connector_spec,
                    owner=owner_scope,
                    kv_topology=_kv_topology_from_spec(connector_spec),
                )
            )
        elif edge_key.startswith("to_stage_"):
            if outbound:
                raise ValueError(
                    "Fan-out (multiple outbound edges) is not supported: got a second "
                    f"outbound edge {edge_key!r} in addition to an already-resolved one."
                )
            to_stage = _parse_stage_id(edge_key.removeprefix("to_stage_"))
            outbound.append(
                ResolvedConnectorSpec(
                    edge=StageEdge(from_stage=this_stage, to_stage=to_stage),
                    spec=connector_spec,
                    owner=owner_scope,
                    kv_topology=_kv_topology_from_spec(connector_spec),
                )
            )

    return StageConnectorPlan(
        inbound=tuple(inbound),
        outbound=tuple(outbound),
        uses_legacy_default=False,
    )


def load_omni_transfer_config(
    config_path: str | Path | None = None,
    config_dict: dict[str, Any] | None = None,
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
                        connectors[edge_key] = ConnectorSpec(name="SharedMemoryConnector")

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
                        connectors[edge_key] = ConnectorSpec(name="SharedMemoryConnector")

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
