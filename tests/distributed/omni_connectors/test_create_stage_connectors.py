# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for OmniConnectorFactory.create_stage_connectors materialization."""

from __future__ import annotations

import pytest

from vllm_omni.distributed.omni_connectors.connectors.base import OmniConnectorBase
from vllm_omni.distributed.omni_connectors.factory import OmniConnectorFactory
from vllm_omni.distributed.omni_connectors.utils.config import ConnectorSpec, OmniTransferConfig
from vllm_omni.distributed.omni_connectors.utils.initialization import (
    ConnectorOwner,
    StageConnectorPlan,
    resolve_stage_connector_plan,
)
from vllm_omni.distributed.omni_connectors.utils.kv_utils import (
    KV_RANK_PORT_STRIDE,
    KV_REPLICA_PORT_STRIDE,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _tp_worker_kwargs(stage_id: int = 1, *, tp_rank: int = 0, replica_id: int = 0) -> dict:
    return {
        "stage_id": stage_id,
        "owner": ConnectorOwner.CONNECTOR_MIXIN,
        "tp_rank": tp_rank,
        "replica_id": replica_id,
    }


def _cta_kwargs(stage_id: int = 1, *, replica_id: int = 0) -> dict:
    return {
        "stage_id": stage_id,
        "owner": ConnectorOwner.CHUNK_TRANSFER_ADAPTER,
        "replica_id": replica_id,
    }


@pytest.fixture
def create_recording_connector(mocker):
    """Patch OmniConnectorFactory.create_connector to return stub instances
    while recording every ConnectorSpec it was called with."""
    created: list[ConnectorSpec] = []

    def _fake_create(*args):
        spec = args[-1]
        created.append(spec)

        class _Stub:
            def close(self):
                pass

        return _Stub()

    mocker.patch.object(OmniConnectorFactory, "create_connector", side_effect=_fake_create)
    return created


def test_legacy_default_shares_one_shm_instance():
    plan = StageConnectorPlan(uses_legacy_default=True)
    recv, send = OmniConnectorFactory.create_stage_connectors(plan, **_tp_worker_kwargs(0))
    assert recv is not None
    assert send is recv
    assert type(recv).__name__ == "SharedMemoryConnector"
    recv.close()


def test_same_type_mooncake_dual_collapse(create_recording_connector):
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
    created = create_recording_connector
    recv, send = OmniConnectorFactory.create_stage_connectors(plan, **_tp_worker_kwargs(1))
    assert recv is send
    assert len(created) == 1
    assert created[0].extra["role"] == "dual"
    assert created[0].extra["zmq_port"] == 50052
    assert created[0].extra["sender_zmq_port"] == 50051


def test_incompatible_same_type_stays_hybrid(create_recording_connector):
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
    created = create_recording_connector
    recv, send = OmniConnectorFactory.create_stage_connectors(plan, **_tp_worker_kwargs(1))
    assert recv is not send
    assert len(created) == 2
    assert created[0].extra["host"] == "10.0.0.1"
    assert created[1].extra["host"] == "10.0.0.2"


def test_hybrid_mooncake_shm_two_instances(create_recording_connector):
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
    created = create_recording_connector
    recv, send = OmniConnectorFactory.create_stage_connectors(plan, **_tp_worker_kwargs(1))
    assert recv is not send
    assert [s.name for s in created] == ["MooncakeTransferEngineConnector", "SharedMemoryConnector"]


def test_inbound_only_send_is_none():
    config = OmniTransferConfig(connectors={("0", "1"): ConnectorSpec(name="SharedMemoryConnector")})
    plan = resolve_stage_connector_plan(config, stage_id=1)
    recv, send = OmniConnectorFactory.create_stage_connectors(plan, **_tp_worker_kwargs(1))
    assert recv is not None
    assert send is None
    recv.close()


def test_outbound_only_receive_is_none():
    config = OmniTransferConfig(connectors={("0", "1"): ConnectorSpec(name="SharedMemoryConnector")})
    plan = resolve_stage_connector_plan(config, stage_id=0)
    recv, send = OmniConnectorFactory.create_stage_connectors(plan, **_tp_worker_kwargs(0))
    assert recv is None
    assert send is not None
    send.close()


def test_owner_filters_cta_vs_mixin():
    config = OmniTransferConfig(
        connectors={
            ("0", "1"): ConnectorSpec(name="SharedMemoryConnector"),
            ("1", "2"): ConnectorSpec(name="SharedMemoryConnector"),
        }
    )
    # Plan owned by CONNECTOR_MIXIN (async_chunk=False).
    plan = resolve_stage_connector_plan(config, stage_id=1, async_chunk=False)

    mixin_recv, mixin_send = OmniConnectorFactory.create_stage_connectors(plan, **_tp_worker_kwargs(1))
    assert mixin_recv is not None and mixin_send is not None

    # Chunk transfer adapter calling with CHUNK_TRANSFER_ADAPTER must skip every edge → both None.
    cta_recv, cta_send = OmniConnectorFactory.create_stage_connectors(plan, **_cta_kwargs(1))
    assert cta_recv is None and cta_send is None

    # Flip ownership for async_chunk.
    cta_plan = resolve_stage_connector_plan(config, stage_id=1, async_chunk=True)
    cta_owned_recv, cta_owned_send = OmniConnectorFactory.create_stage_connectors(cta_plan, **_cta_kwargs(1))
    assert cta_owned_recv is not None and cta_owned_send is not None
    mixin_skipped_recv, mixin_skipped_send = OmniConnectorFactory.create_stage_connectors(
        cta_plan, **_tp_worker_kwargs(1)
    )
    assert mixin_skipped_recv is None and mixin_skipped_send is None

    mixin_recv.close()
    cta_owned_recv.close()


def test_tp_and_replica_port_offset_applied_once(create_recording_connector):
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
    created = create_recording_connector

    OmniConnectorFactory.create_stage_connectors(
        plan,
        **_tp_worker_kwargs(1, tp_rank=2, replica_id=1),
    )
    assert len(created) == 1  # dual-collapsed
    offset = 1 * KV_REPLICA_PORT_STRIDE + 2 * KV_RANK_PORT_STRIDE
    assert created[0].extra["zmq_port"] == 50052 + offset
    assert created[0].extra["sender_zmq_port"] == 50051 + offset
    assert created[0].extra["role"] == "dual"


def test_stage_replica_never_uses_tp_rank_for_ports(create_recording_connector):
    config = OmniTransferConfig(
        connectors={
            ("0", "1"): ConnectorSpec(
                name="MooncakeTransferEngineConnector",
                extra={"host": "auto", "zmq_port": 50051, "protocol": "rdma"},
            ),
        }
    )
    plan = resolve_stage_connector_plan(config, stage_id=1, async_chunk=True)
    created = create_recording_connector
    # Even if a bogus tp_rank sneaks in, CHUNK_TRANSFER_ADAPTER must force local_rank=0.
    OmniConnectorFactory.create_stage_connectors(
        plan,
        stage_id=1,
        owner=ConnectorOwner.CHUNK_TRANSFER_ADAPTER,
        replica_id=3,
        tp_rank=7,
    )
    assert len(created) == 1
    offset = 3 * KV_REPLICA_PORT_STRIDE  # rank contribution forced to 0
    assert created[0].extra["sender_zmq_port"] == 50051 + offset


def test_dual_merge_preserves_distinct_directional_rank_mappings():
    merged = OmniConnectorBase.merge_dual_specs(
        {"role": "receiver", "rank_mapping": {"from_tp": 4, "to_tp": 2}},
        {"role": "sender", "rank_mapping": {"from_tp": 2, "to_tp": 1}},
    )

    assert "rank_mapping" not in merged
    assert merged["recv_rank_mapping"] == {"from_tp": 4, "to_tp": 2}
    assert merged["send_rank_mapping"] == {"from_tp": 2, "to_tp": 1}
