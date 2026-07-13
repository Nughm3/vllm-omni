# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from pytest_mock import MockerFixture

from vllm_omni.distributed.omni_connectors.adapter import try_recv_via_connector, try_send_via_connector
from vllm_omni.distributed.omni_connectors.connectors.shm_connector import SharedMemoryConnector
from vllm_omni.distributed.omni_connectors.utils.config import ConnectorSpec, OmniTransferConfig
from vllm_omni.distributed.omni_connectors.utils.initialization import get_connectors_config_for_stage

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
    Using real SharedMemoryConnector (inline mode for simplicity).
    """
    # 1. Setup Connector
    config = {"shm_threshold_bytes": 1024, "inline_small_payloads": True}
    connector = SharedMemoryConnector(config)
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
    # get_stage_connector_spec can merge them into a duplex connector.
    stage_config = get_connectors_config_for_stage(config, stage_id=1)

    assert "from_stage_0" in stage_config
    assert stage_config["from_stage_0"]["spec"]["name"] == "C1"
    assert stage_config["from_stage_0"]["spec"]["extra"]["can_get"] is True
    assert stage_config["from_stage_0"]["spec"]["extra"]["can_put"] is False

    assert "to_stage_2" in stage_config
    assert stage_config["to_stage_2"]["spec"]["name"] == "C2"
    assert stage_config["to_stage_2"]["spec"]["extra"]["can_put"] is True
    assert stage_config["to_stage_2"]["spec"]["extra"]["can_get"] is False

    # Unrelated edges must not appear.
    assert "from_stage_1" not in stage_config

    stage_2_config = get_connectors_config_for_stage(config, stage_id=2)
    assert "from_stage_1" in stage_2_config
    assert stage_2_config["from_stage_1"]["spec"]["name"] == "C2"
    assert "to_stage_2" not in stage_2_config

    stage_0_config = get_connectors_config_for_stage(config, stage_id=0)
    assert "to_stage_1" in stage_0_config
    assert stage_0_config["to_stage_1"]["spec"]["extra"]["can_put"] is True
    assert "from_stage_0" not in stage_0_config


def test_merge_stage_connector_specs_duplex_mooncake():
    """Middle stage merges inbound+outbound Mooncake edges into one duplex spec."""
    from vllm_omni.distributed.omni_connectors.utils.initialization import (
        merge_stage_connector_specs,
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

    stage_1 = get_connectors_config_for_stage(config, stage_id=1)
    merged = merge_stage_connector_specs(stage_1)

    assert merged["name"] == "MooncakeTransferEngineConnector"
    extra = merged["extra"]
    assert extra["can_put"] is True
    assert extra["can_get"] is True
    assert "role" not in extra
    # Listen on base+stage1; query stage0's listen port.
    assert extra["zmq_port"] == 50052
    assert extra["sender_host"] == "10.248.12.80"
    assert extra["sender_zmq_port"] == 50051

    stage_0 = merge_stage_connector_specs(get_connectors_config_for_stage(config, stage_id=0))
    assert stage_0["extra"]["can_put"] is True
    assert stage_0["extra"]["can_get"] is False
    assert stage_0["extra"]["zmq_port"] == 50051
    assert stage_0["extra"]["role"] == "sender"

    stage_2 = merge_stage_connector_specs(get_connectors_config_for_stage(config, stage_id=2))
    assert stage_2["extra"]["can_put"] is False
    assert stage_2["extra"]["can_get"] is True
    assert stage_2["extra"]["sender_host"] == "10.248.12.86"
    assert stage_2["extra"]["sender_zmq_port"] == 50052
    assert stage_2["extra"]["role"] == "receiver"


def test_merge_stage_connector_specs_heterogeneous_mooncake_shm():
    """Middle stage with Mooncake inbound + SHM outbound becomes a composite."""
    from vllm_omni.distributed.omni_connectors.utils.initialization import (
        merge_stage_connector_specs,
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

    stage_1 = merge_stage_connector_specs(get_connectors_config_for_stage(config, stage_id=1))
    assert stage_1["name"] == "CompositeOmniConnector"
    extra = stage_1["extra"]
    assert extra["can_put"] is True
    assert extra["can_get"] is True
    assert extra["get_connector"]["name"] == "MooncakeTransferEngineConnector"
    assert extra["get_connector"]["extra"]["can_get"] is True
    assert extra["get_connector"]["extra"]["can_put"] is False
    assert extra["get_connector"]["extra"]["sender_host"] == "10.248.12.80"
    assert extra["get_connector"]["extra"]["sender_zmq_port"] == 50051
    assert extra["put_connector"]["name"] == "SharedMemoryConnector"
    assert extra["put_connector"]["extra"]["can_put"] is True
    assert extra["put_connector"]["extra"]["can_get"] is False
    assert extra["put_connector"]["extra"]["codec_chunk_frames"] == 25

    stage_0 = merge_stage_connector_specs(get_connectors_config_for_stage(config, stage_id=0))
    assert stage_0["name"] == "MooncakeTransferEngineConnector"
    assert stage_0["extra"]["role"] == "sender"

    stage_2 = merge_stage_connector_specs(get_connectors_config_for_stage(config, stage_id=2))
    assert stage_2["name"] == "SharedMemoryConnector"
    assert stage_2["extra"]["can_get"] is True
    assert stage_2["extra"]["can_put"] is False


def test_composite_omni_connector_delegates_put_get(mocker: MockerFixture):
    """Composite routes get to inbound connector and put to outbound."""
    from vllm_omni.distributed.omni_connectors.connectors.composite_connector import (
        CompositeOmniConnector,
    )
    from vllm_omni.distributed.omni_connectors.utils.config import ConnectorSpec

    get_conn = mocker.Mock()
    get_conn.get.return_value = ({"ok": True}, 4)
    get_conn.supports_raw_data = False
    put_conn = mocker.Mock()
    put_conn.put.return_value = (True, 8, {"shm": {"name": "k"}})
    put_conn.supports_raw_data = False

    def _create(spec: ConnectorSpec):
        if spec.name == "MooncakeTransferEngineConnector":
            return get_conn
        if spec.name == "SharedMemoryConnector":
            return put_conn
        raise AssertionError(spec.name)

    mocker.patch(
        "vllm_omni.distributed.omni_connectors.factory.OmniConnectorFactory.create_connector",
        side_effect=_create,
    )

    composite = CompositeOmniConnector(
        {
            "stage_id": 1,
            "get_connector": {
                "name": "MooncakeTransferEngineConnector",
                "extra": {"can_get": True, "can_put": False},
            },
            "put_connector": {
                "name": "SharedMemoryConnector",
                "extra": {"can_put": True, "can_get": False},
            },
        }
    )

    assert composite.get("0", "1", "req") == ({"ok": True}, 4)
    get_conn.get.assert_called_once_with("0", "1", "req", None)
    put_conn.get.assert_not_called()

    assert composite.put("1", "2", "req", {"data": 1}) == (True, 8, {"shm": {"name": "k"}})
    put_conn.put.assert_called_once_with("1", "2", "req", {"data": 1})
    get_conn.put.assert_not_called()

    composite.cleanup("req")
    get_conn.cleanup.assert_called_once_with("req")
    put_conn.cleanup.assert_called_once_with("req")

    composite.close()
    get_conn.close.assert_called_once()
    put_conn.close.assert_called_once()


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
