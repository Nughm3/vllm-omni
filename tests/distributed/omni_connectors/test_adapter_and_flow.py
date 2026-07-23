# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from pytest_mock import MockerFixture

from vllm_omni.distributed.omni_connectors.adapter import try_recv_via_connector, try_send_via_connector
from vllm_omni.distributed.omni_connectors.connectors.shm_connector import SharedMemoryConnector
from vllm_omni.distributed.omni_connectors.utils.config import ConnectorSpec, OmniTransferConfig
from vllm_omni.distributed.omni_connectors.utils.initialization import get_connectors_config_for_stage
from vllm_omni.engine.stage_init_utils import get_stage_connector_spec

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def mock_objects(mocker: MockerFixture):
    return {"connector": mocker.MagicMock(), "metrics": mocker.MagicMock(), "queue_fn": mocker.MagicMock()}


def test_send_success(mock_objects):
    """Test try_send_via_connector success path."""
    # Setup
    mock_connector = mock_objects["connector"]
    mock_metrics = mock_objects["metrics"]
    mock_queue_fn = mock_objects["queue_fn"]

    stage_id = 0
    next_stage_id = 1
    req_id = "req_123"
    inputs = {"input_ids": [1, 2, 3]}
    sampling_params = {"temperature": 0.7}
    prompt = "test prompt"

    # Mock connector.put return
    # Returns: (success, size, metadata)
    mock_metadata = {"handle": "xyz"}
    mock_connector.put.return_value = (True, 100, mock_metadata)

    # Execute
    result = try_send_via_connector(
        connector=mock_connector,
        stage_id=stage_id,
        next_stage_id=next_stage_id,
        req_id=req_id,
        next_inputs=inputs,
        sampling_params=sampling_params,
        original_prompt=prompt,
        next_stage_queue_submit_fn=mock_queue_fn,
        metrics=mock_metrics,
    )

    # Verify
    assert result is True

    # 1. Verify connector.put called correctly
    mock_connector.put.assert_called_once()
    args, _ = mock_connector.put.call_args
    assert args[0] == "0"  # from_stage
    assert args[1] == "1"  # to_stage
    assert args[2] == req_id
    # Verify payload structure in put
    payload = args[3]
    assert payload["engine_inputs"] == inputs
    assert payload["sampling_params"] == sampling_params

    # 2. Verify queue notification submitted
    mock_queue_fn.assert_called_once()
    notify_payload = mock_queue_fn.call_args[0][0]
    assert notify_payload["request_id"] == req_id
    assert notify_payload["from_connector"] is True
    assert notify_payload["connector_metadata"] == mock_metadata

    # 3. Verify metrics recorded
    mock_metrics.on_forward.assert_called_once()


def test_send_fail(mock_objects):
    """Test try_send_via_connector when connector fails."""
    mock_connector = mock_objects["connector"]
    mock_metrics = mock_objects["metrics"]
    mock_queue_fn = mock_objects["queue_fn"]

    mock_connector.put.return_value = (False, 0, None)

    result = try_send_via_connector(
        connector=mock_connector,
        stage_id=0,
        next_stage_id=1,
        req_id="req_fail",
        next_inputs={},
        sampling_params={},
        original_prompt="",
        next_stage_queue_submit_fn=mock_queue_fn,
        metrics=mock_metrics,
    )

    assert result is False
    mock_queue_fn.assert_not_called()


def test_recv_success(mock_objects):
    """Test try_recv_via_connector success path."""
    mock_connector = mock_objects["connector"]

    # Setup task received from queue
    task = {
        "request_id": "req_recv",
        "from_connector": True,
        "from_stage": "0",
        "connector_metadata": {"handle": "xyz"},
    }

    # Setup connectors dict
    connectors = {("0", "1"): mock_connector}

    # Mock connector.get return
    expected_data = {"engine_inputs": {"ids": [1]}}
    # get returns: (data_obj, size)
    mock_connector.get.return_value = (expected_data, 50)
    # serialize_obj needed for metrics calculation if size not returned directly
    mock_connector.serialize_obj.return_value = b"bytes"

    # Execute
    # We are stage 1 receiving from stage 0
    inputs, rx_metrics = try_recv_via_connector(task, connectors, stage_id=1)

    # Verify
    assert inputs == expected_data["engine_inputs"]
    assert rx_metrics is not None
    mock_connector.get.assert_called_once_with("0", "1", "req_recv", metadata={"handle": "xyz"})


def test_recv_no_connector():
    """Test recv fails when no connector exists for edge."""
    task = {"request_id": "req_missing", "from_connector": True, "from_stage": "0"}
    connectors = {}  # Empty connectors

    inputs, _ = try_recv_via_connector(task, connectors, stage_id=1)
    assert inputs is None


def test_shm_connector_flow(mocker: MockerFixture):
    """
    Verify the full flow: Send -> Adapter -> Connector -> Adapter -> Recv.
    Using real SharedMemoryConnector and key-addressed SHM metadata.
    """
    # 1. Setup Connector
    connector = SharedMemoryConnector({})
    connectors_map = {("0", "1"): connector}

    # 2. Setup Data
    stage_id = 0
    next_stage_id = 1
    req_id = "flow_req"
    inputs = {"tokens": [10, 20, 30]}
    sampling_params = {"n": 1}

    # Queue capture mechanism
    queue_capture = []

    def mock_submit(payload):
        queue_capture.append(payload)

    mock_metrics = mocker.MagicMock()

    # 3. Send
    success = try_send_via_connector(
        connector=connector,
        stage_id=stage_id,
        next_stage_id=next_stage_id,
        req_id=req_id,
        next_inputs=inputs,
        sampling_params=sampling_params,
        original_prompt="prompt",
        next_stage_queue_submit_fn=mock_submit,
        metrics=mock_metrics,
    )
    assert success is True
    assert len(queue_capture) == 1

    # 4. Recv
    # The 'task' is what would be popped from the queue
    received_task = queue_capture[0]

    # Verify queue payload contains what we expect
    assert received_task["from_connector"] is True
    assert received_task["from_stage"] == "0"

    # Decode
    decoded_inputs, _ = try_recv_via_connector(received_task, connectors_map, stage_id=1)

    # 5. Verify Data Integrity
    assert decoded_inputs == inputs


def test_get_connectors_for_stage():
    """Test filtering logic for stage config."""
    # Config has edges: 0->1, 1->2
    config = OmniTransferConfig(connectors={("0", "1"): ConnectorSpec(name="C1"), ("1", "2"): ConnectorSpec(name="C2")})

    # Stage 1 receives from 0 and sends to 2 — both edges are returned so
    # resolve_stage_connector_specs can split them into (input, output).
    stage_config = get_connectors_config_for_stage(config, stage_id=1)

    assert "from_stage_0" in stage_config
    assert stage_config["from_stage_0"]["spec"]["name"] == "C1"

    assert "to_stage_2" in stage_config
    assert stage_config["to_stage_2"]["spec"]["name"] == "C2"

    # Unrelated edges must not appear.
    assert "from_stage_1" not in stage_config

    stage_2_config = get_connectors_config_for_stage(config, stage_id=2)
    assert "from_stage_1" in stage_2_config
    assert stage_2_config["from_stage_1"]["spec"]["name"] == "C2"
    assert "to_stage_2" not in stage_2_config

    stage_0_config = get_connectors_config_for_stage(config, stage_id=0)
    assert "to_stage_1" in stage_0_config
    assert "from_stage_0" not in stage_0_config


def test_resolve_stage_connector_specs_same_type_mooncake():
    """A same-type middle stage yields two Mooncake specs (input + output),
    each with its own role and ports; the mixin collapses them into one dual
    instance. Stage 0 has only an output spec; the last stage only an input."""
    from vllm_omni.distributed.omni_connectors.utils.initialization import (
        resolve_stage_connector_specs,
    )

    config = OmniTransferConfig(
        connectors={
            ("0", "1"): ConnectorSpec(
                name="MooncakeTransferEngineConnector",
                extra={
                    "host": "auto",
                    "zmq_port": 50051,
                    "sender_host": "10.248.12.80",
                    "protocol": "rdma",
                },
            ),
            ("1", "2"): ConnectorSpec(
                name="MooncakeTransferEngineConnector",
                extra={
                    "host": "auto",
                    "zmq_port": 50051,
                    "sender_host": "10.248.12.86",
                    "protocol": "rdma",
                },
            ),
        }
    )

    # Middle stage: both edges present, same type.
    recv_spec, send_spec = resolve_stage_connector_specs(get_connectors_config_for_stage(config, stage_id=1))
    assert recv_spec["name"] == "MooncakeTransferEngineConnector"
    assert recv_spec["extra"]["role"] == "receiver"
    assert recv_spec["extra"]["sender_host"] == "10.248.12.80"
    assert recv_spec["extra"]["sender_zmq_port"] == 50051
    assert send_spec["name"] == "MooncakeTransferEngineConnector"
    assert send_spec["extra"]["role"] == "sender"
    # Listen on base+stage1 (outbound edge from stage 1).
    assert send_spec["extra"]["zmq_port"] == 50052

    # Stage 0: outbound only.
    recv0, send0 = resolve_stage_connector_specs(get_connectors_config_for_stage(config, stage_id=0))
    assert recv0 is None
    assert send0["extra"]["role"] == "sender"
    assert send0["extra"]["zmq_port"] == 50051

    # Last stage: inbound only.
    recv2, send2 = resolve_stage_connector_specs(get_connectors_config_for_stage(config, stage_id=2))
    assert send2 is None
    assert recv2["extra"]["role"] == "receiver"
    assert recv2["extra"]["sender_host"] == "10.248.12.86"
    assert recv2["extra"]["sender_zmq_port"] == 50052


def test_resolve_stage_connector_specs_ignores_rank_and_replica_at_config_build_time(mocker: MockerFixture):
    """``get_connectors_config_for_stage``/``resolve_stage_connector_specs``
    run once per stage while the shared engine config is being built, before
    any per-rank worker process (and its real local_rank / replica_id)
    exists. They must therefore resolve only the stage-level *base* port —
    baking a rank/replica-resolved port in here would copy the same value to
    every worker, and only one could ever bind it. The per-worker offset is
    applied later, by the mixin, at actual connector-construction time (see
    test_omni_connector_mixin.py's TestWorkerPortOffset)."""
    from vllm_omni.distributed.omni_connectors.utils.initialization import resolve_stage_connector_specs

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

    # Even with a non-zero rank/replica in the environment at config-build
    # time (e.g. leftover from a previous process), the resolved spec must
    # not vary — this function no longer consults them at all.
    mocker.patch(
        "vllm_omni.distributed.omni_connectors.utils.initialization.get_connector_local_rank",
        return_value=3,
    )
    mocker.patch(
        "vllm_omni.distributed.omni_connectors.utils.initialization.get_omni_replica_id",
        return_value=2,
    )
    recv, send = resolve_stage_connector_specs(get_connectors_config_for_stage(config, stage_id=1))
    assert send["extra"]["zmq_port"] == 50052
    assert recv["extra"]["sender_zmq_port"] == 50051


def test_resolve_stage_connector_specs_hybrid_mooncake_shm():
    """A hybrid middle stage (Mooncake inbound + SHM outbound) yields two
    different-type specs; the mixin builds two distinct instances."""
    from vllm_omni.distributed.omni_connectors.utils.initialization import (
        resolve_stage_connector_specs,
    )

    config = OmniTransferConfig(
        connectors={
            ("0", "1"): ConnectorSpec(
                name="MooncakeTransferEngineConnector",
                extra={
                    "host": "auto",
                    "zmq_port": 50051,
                    "sender_host": "10.248.12.80",
                    "protocol": "rdma",
                },
            ),
            ("1", "2"): ConnectorSpec(
                name="SharedMemoryConnector",
                extra={
                    "initial_codec_chunk_frames": 4,
                    "codec_chunk_frames": 25,
                    "codec_left_context_frames": 25,
                },
            ),
        }
    )

    recv_spec, send_spec = resolve_stage_connector_specs(get_connectors_config_for_stage(config, stage_id=1))
    assert recv_spec["name"] == "MooncakeTransferEngineConnector"
    assert recv_spec["extra"]["sender_host"] == "10.248.12.80"
    assert recv_spec["extra"]["sender_zmq_port"] == 50051
    assert send_spec["name"] == "SharedMemoryConnector"
    assert send_spec["extra"]["codec_chunk_frames"] == 25

    recv0, send0 = resolve_stage_connector_specs(get_connectors_config_for_stage(config, stage_id=0))
    assert recv0 is None
    assert send0["name"] == "MooncakeTransferEngineConnector"

    recv2, send2 = resolve_stage_connector_specs(get_connectors_config_for_stage(config, stage_id=2))
    assert send2 is None
    assert recv2["name"] == "SharedMemoryConnector"


def test_resolve_stage_connector_specs_empty():
    """No transfer config resolves to (None, None)."""
    from vllm_omni.distributed.omni_connectors.utils.initialization import (
        resolve_stage_connector_specs,
    )

    assert resolve_stage_connector_specs({}) == (None, None)


@pytest.mark.parametrize("async_chunk", [False, True])
def test_outgoing_stage_uses_sender_connector_spec(async_chunk: bool):
    config = OmniTransferConfig(
        connectors={
            ("1", "2"): ConnectorSpec(
                name="SharedMemoryConnector",
                extra={"codec_chunk_frames": 25},
            )
        }
    )

    spec = get_stage_connector_spec(config, stage_id=1, async_chunk=async_chunk)

    assert spec["name"] == "SharedMemoryConnector"
    assert spec["extra"] == {"codec_chunk_frames": 25, "role": "sender"}


def test_full_payload_consumer_uses_receiver_connector_spec():
    config = OmniTransferConfig(
        connectors={
            ("1", "2"): ConnectorSpec(
                name="SharedMemoryConnector",
                extra={"codec_chunk_frames": 25},
            )
        }
    )

    spec = get_stage_connector_spec(config, stage_id=2, async_chunk=False)

    assert spec["name"] == "SharedMemoryConnector"
    assert spec["extra"] == {"codec_chunk_frames": 25, "role": "receiver"}


def test_recv_with_missing_metadata(mocker: MockerFixture):
    """Test recv when queue payload is malformed (missing metadata)."""
    # Connector expects metadata but task doesn't have it
    task = {
        "request_id": "req_bad",
        "from_connector": True,
        "from_stage": "0",
        # Missing "connector_metadata"
    }
    mock_conn = mocker.MagicMock()
    # If get is called with None metadata, connector usually handles it or adapter handles exception
    mock_conn.get.side_effect = Exception("Get failed")

    connectors = {("0", "1"): mock_conn}

    inputs, _ = try_recv_via_connector(task, connectors, stage_id=1)
    assert inputs is None
