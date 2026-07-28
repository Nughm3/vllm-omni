# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for OmniConnectorFactory.create_stage_connectors materialization."""

from __future__ import annotations

import pytest

from vllm_omni.distributed.omni_connectors.connectors.base import OmniConnectorBase
from vllm_omni.distributed.omni_connectors.factory import OmniConnectorFactory
from vllm_omni.distributed.omni_connectors.stage_connector import (
    ConnectorRuntimeContext,
    StageConnectorSet,
)
from vllm_omni.distributed.omni_connectors.utils.config import ConnectorSpec, OmniTransferConfig
from vllm_omni.distributed.omni_connectors.utils.initialization import (
    ConnectorOwnerScope,
    StageConnectorPlan,
    resolve_stage_connector_plan,
)
from vllm_omni.distributed.omni_connectors.utils.kv_utils import (
    KV_RANK_PORT_STRIDE,
    KV_REPLICA_PORT_STRIDE,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _tp_worker_ctx(stage_id: int = 1, *, tp_rank: int = 0, replica_id: int = 0) -> ConnectorRuntimeContext:
    return ConnectorRuntimeContext(
        stage_id=stage_id,
        owner_scope=ConnectorOwnerScope.TP_WORKER,
        tp_rank=tp_rank,
        replica_id=replica_id,
    )


def _cta_ctx(stage_id: int = 1, *, replica_id: int = 0) -> ConnectorRuntimeContext:
    return ConnectorRuntimeContext(
        stage_id=stage_id,
        owner_scope=ConnectorOwnerScope.STAGE_REPLICA,
        replica_id=replica_id,
    )


def test_legacy_default_shares_one_shm_instance():
    plan = StageConnectorPlan(uses_legacy_default=True)
    connectors = OmniConnectorFactory.create_stage_connectors(plan, _tp_worker_ctx(0))
    assert connectors.receive is not None
    assert connectors.send is connectors.receive
    assert type(connectors.receive).__name__ == "SharedMemoryConnector"
    connectors.close()


def test_same_type_mooncake_dual_collapse(mocker):
    config = OmniTransferConfig(
        connectors={
            ("0", "1"): ConnectorSpec(
                name="MooncakeTransferEngineConnector",
                extra={"host": "auto", "zmq_port": 50051, "protocol": "rdma"},
            ),
            ("1", "2"): ConnectorSpec(
                name="MooncakeTransferEngineConnector",
                extra={"host": "auto", "zmq_port": 50051, "protocol": "rdma"},
            ),
        }
    )
    plan = resolve_stage_connector_plan(config, stage_id=1)
    created: list[ConnectorSpec] = []

    def _fake_create(*args):
        spec = args[-1]
        created.append(spec)

        class _Stub:
            def close(self):
                pass

        return _Stub()

    mocker.patch.object(OmniConnectorFactory, "create_connector", side_effect=_fake_create)
    connectors = OmniConnectorFactory.create_stage_connectors(plan, _tp_worker_ctx(1))
    assert connectors.receive is connectors.send
    assert len(created) == 1
    assert created[0].extra["role"] == "dual"
    assert created[0].extra["zmq_port"] == 50052
    assert created[0].extra["sender_zmq_port"] == 50051


def test_incompatible_same_type_stays_hybrid(mocker):
    config = OmniTransferConfig(
        connectors={
            ("0", "1"): ConnectorSpec(
                name="MooncakeTransferEngineConnector",
                extra={"host": "10.0.0.1", "zmq_port": 50051, "protocol": "rdma"},
            ),
            ("1", "2"): ConnectorSpec(
                name="MooncakeTransferEngineConnector",
                # Different non-directional host → must not share one instance.
                extra={"host": "10.0.0.2", "zmq_port": 50051, "protocol": "rdma"},
            ),
        }
    )
    plan = resolve_stage_connector_plan(config, stage_id=1)
    created: list[ConnectorSpec] = []

    def _fake_create(*args):
        spec = args[-1]
        created.append(spec)

        class _Stub:
            def close(self):
                pass

        return _Stub()

    mocker.patch.object(OmniConnectorFactory, "create_connector", side_effect=_fake_create)
    connectors = OmniConnectorFactory.create_stage_connectors(plan, _tp_worker_ctx(1))
    assert connectors.receive is not connectors.send
    assert len(created) == 2
    assert created[0].extra["host"] == "10.0.0.1"
    assert created[1].extra["host"] == "10.0.0.2"


def test_hybrid_mooncake_shm_two_instances(mocker):
    config = OmniTransferConfig(
        connectors={
            ("0", "1"): ConnectorSpec(
                name="MooncakeTransferEngineConnector",
                extra={"host": "auto", "zmq_port": 50051, "protocol": "rdma"},
            ),
            ("1", "2"): ConnectorSpec(name="SharedMemoryConnector", extra={"codec_chunk_frames": 25}),
        }
    )
    plan = resolve_stage_connector_plan(config, stage_id=1)
    created: list[ConnectorSpec] = []

    def _fake_create(*args):
        spec = args[-1]
        created.append(spec)

        class _Stub:
            def close(self):
                pass

        return _Stub()

    mocker.patch.object(OmniConnectorFactory, "create_connector", side_effect=_fake_create)
    connectors = OmniConnectorFactory.create_stage_connectors(plan, _tp_worker_ctx(1))
    assert connectors.receive is not connectors.send
    assert [s.name for s in created] == ["MooncakeTransferEngineConnector", "SharedMemoryConnector"]


def test_inbound_only_send_is_none():
    config = OmniTransferConfig(connectors={("0", "1"): ConnectorSpec(name="SharedMemoryConnector")})
    plan = resolve_stage_connector_plan(config, stage_id=1)
    connectors = OmniConnectorFactory.create_stage_connectors(plan, _tp_worker_ctx(1))
    assert connectors.receive is not None
    assert connectors.send is None
    connectors.close()


def test_outbound_only_receive_is_none():
    config = OmniTransferConfig(connectors={("0", "1"): ConnectorSpec(name="SharedMemoryConnector")})
    plan = resolve_stage_connector_plan(config, stage_id=0)
    connectors = OmniConnectorFactory.create_stage_connectors(plan, _tp_worker_ctx(0))
    assert connectors.receive is None
    assert connectors.send is not None
    connectors.close()


def test_owner_scope_filters_cta_vs_mixin():
    config = OmniTransferConfig(
        connectors={
            ("0", "1"): ConnectorSpec(name="SharedMemoryConnector"),
            ("1", "2"): ConnectorSpec(name="SharedMemoryConnector"),
        }
    )
    # Plan owned by TP_WORKER (async_chunk=False).
    plan = resolve_stage_connector_plan(config, stage_id=1, async_chunk=False)

    mixin_set = OmniConnectorFactory.create_stage_connectors(plan, _tp_worker_ctx(1))
    assert mixin_set.receive is not None and mixin_set.send is not None

    # Chunk transfer adapter calling with STAGE_REPLICA must skip every edge → empty set.
    cta_set = OmniConnectorFactory.create_stage_connectors(plan, _cta_ctx(1))
    assert cta_set.receive is None and cta_set.send is None

    # Flip ownership for async_chunk.
    cta_plan = resolve_stage_connector_plan(config, stage_id=1, async_chunk=True)
    cta_owned = OmniConnectorFactory.create_stage_connectors(cta_plan, _cta_ctx(1))
    assert cta_owned.receive is not None and cta_owned.send is not None
    mixin_skipped = OmniConnectorFactory.create_stage_connectors(cta_plan, _tp_worker_ctx(1))
    assert mixin_skipped.receive is None and mixin_skipped.send is None

    mixin_set.close()
    cta_owned.close()


def test_tp_and_replica_port_offset_applied_once(mocker):
    config = OmniTransferConfig(
        connectors={
            ("0", "1"): ConnectorSpec(
                name="MooncakeTransferEngineConnector",
                extra={"host": "auto", "zmq_port": 50051, "protocol": "rdma"},
            ),
            ("1", "2"): ConnectorSpec(
                name="MooncakeTransferEngineConnector",
                extra={"host": "auto", "zmq_port": 50051, "protocol": "rdma"},
            ),
        }
    )
    plan = resolve_stage_connector_plan(config, stage_id=1)
    # Avoid constructing a real Mooncake engine — capture the spec passed in.
    created: list[ConnectorSpec] = []

    def _fake_create(*args):
        spec = args[-1]
        created.append(spec)

        class _Stub:
            def close(self):
                pass

        return _Stub()

    mocker.patch.object(OmniConnectorFactory, "create_connector", side_effect=_fake_create)

    OmniConnectorFactory.create_stage_connectors(
        plan,
        _tp_worker_ctx(1, tp_rank=2, replica_id=1),
    )
    assert len(created) == 1  # dual-collapsed
    offset = 1 * KV_REPLICA_PORT_STRIDE + 2 * KV_RANK_PORT_STRIDE
    assert created[0].extra["zmq_port"] == 50052 + offset
    assert created[0].extra["sender_zmq_port"] == 50051 + offset
    assert created[0].extra["role"] == "dual"


def test_stage_replica_never_uses_tp_rank_for_ports(mocker):
    config = OmniTransferConfig(
        connectors={
            ("0", "1"): ConnectorSpec(
                name="MooncakeTransferEngineConnector",
                extra={"host": "auto", "zmq_port": 50051, "protocol": "rdma"},
            ),
        }
    )
    plan = resolve_stage_connector_plan(config, stage_id=1, async_chunk=True)
    created: list[ConnectorSpec] = []

    def _fake_create(*args):
        spec = args[-1]
        created.append(spec)

        class _Stub:
            def close(self):
                pass

        return _Stub()

    mocker.patch.object(OmniConnectorFactory, "create_connector", side_effect=_fake_create)
    # Even if a bogus tp_rank sneaks in, STAGE_REPLICA must force local_rank=0.
    ctx = ConnectorRuntimeContext(
        stage_id=1,
        owner_scope=ConnectorOwnerScope.STAGE_REPLICA,
        replica_id=3,
        tp_rank=7,
    )
    OmniConnectorFactory.create_stage_connectors(plan, ctx)
    assert len(created) == 1
    offset = 3 * KV_REPLICA_PORT_STRIDE  # rank contribution forced to 0
    assert created[0].extra["sender_zmq_port"] == 50051 + offset


def test_stage_connector_set_close_dedupes_dual():
    closes = []

    class _Conn:
        def close(self):
            closes.append(id(self))

    conn = _Conn()
    StageConnectorSet(receive=conn, send=conn).close()
    assert closes == [id(conn)]

    a, b = _Conn(), _Conn()
    StageConnectorSet(receive=a, send=b).close()
    assert closes == [id(conn), id(a), id(b)]


def test_dual_merge_preserves_distinct_directional_rank_mappings():
    merged = OmniConnectorBase.merge_duplex_specs(
        {"role": "receiver", "rank_mapping": {"from_tp": 4, "to_tp": 2}},
        {"role": "sender", "rank_mapping": {"from_tp": 2, "to_tp": 1}},
    )

    assert "rank_mapping" not in merged
    assert merged["recv_rank_mapping"] == {"from_tp": 4, "to_tp": 2}
    assert merged["send_rank_mapping"] == {"from_tp": 2, "to_tp": 1}
