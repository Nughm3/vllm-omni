# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading

import pytest
import torch
from pytest_mock import MockerFixture

from vllm_omni.distributed.omni_connectors.connectors.mooncake_transfer_engine_connector import (
    MooncakeTransferEngineConnector,
)
from vllm_omni.distributed.omni_connectors.connectors.shm_connector import SharedMemoryConnector
from vllm_omni.distributed.omni_connectors.factory import OmniConnectorFactory
from vllm_omni.distributed.omni_connectors.utils.config import ConnectorSpec
from vllm_omni.distributed.omni_connectors.utils.memory_pool import (
    MANAGED_BUFFER_LEASES_KEY,
    BufferAllocator,
    ManagedBuffer,
)
from vllm_omni.distributed.omni_connectors.utils.serialization import OmniSerializer

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_basic_serialization():
    """Test basic msgpack serialization."""
    data = {"key": "value", "list": [1, 2, 3]}
    serialized = OmniSerializer.serialize(data)
    assert isinstance(serialized, bytes)

    deserialized = OmniSerializer.deserialize(serialized)
    assert data == deserialized


def test_tensor_serialization():
    """Test torch.Tensor serialization."""
    import torch

    tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    serialized = OmniSerializer.serialize(tensor)
    deserialized = OmniSerializer.deserialize(serialized)

    assert torch.equal(tensor, deserialized)


def test_noncontiguous_tensor_serialization():
    """Test torch.Tensor serialization for strided views."""
    import torch

    tensor = torch.arange(40, dtype=torch.long).reshape(4, 10).transpose(0, 1)
    assert not tensor.is_contiguous()
    assert tensor.stride(-1) != 1

    serialized = OmniSerializer.serialize({"codes": tensor})
    deserialized = OmniSerializer.deserialize(serialized)

    assert torch.equal(tensor, deserialized["codes"])


def test_ndarray_serialization():
    """Test numpy.ndarray serialization."""
    import numpy as np

    arr = np.array([[1, 2, 3], [4, 5, 6]])
    serialized = OmniSerializer.serialize(arr)
    deserialized = OmniSerializer.deserialize(serialized)

    assert np.array_equal(arr, deserialized)


def test_create_shm_connector():
    """Test creating SharedMemoryConnector via Factory."""
    spec = ConnectorSpec(name="SharedMemoryConnector", extra={"shm_threshold_bytes": 1024})
    connector = OmniConnectorFactory.create_connector(spec)
    assert isinstance(connector, SharedMemoryConnector)
    assert connector.threshold == 1024


def test_create_unknown_connector():
    """Test error when creating unknown connector."""
    spec = ConnectorSpec(name="UnknownConnector")
    with pytest.raises(ValueError):
        OmniConnectorFactory.create_connector(spec)


@pytest.fixture
def shm_connector():
    config = {"shm_threshold_bytes": 100, "inline_small_payloads": True}
    return SharedMemoryConnector(config)


def test_put_get_inline(shm_connector):
    """Test inline transfer for small data."""
    data = {"small": "data"}

    success, size, metadata = shm_connector.put("stage_0", "stage_1", "req_1", data)
    assert success is True
    assert "inline_bytes" in metadata
    assert "shm" not in metadata
    assert "size" in metadata
    assert shm_connector._metrics["inline_writes"] == 1
    assert shm_connector._metrics["shm_writes"] == 0

    # Retrieve
    retrieved_data, ret_size = shm_connector.get("stage_0", "stage_1", "req_1", metadata)
    assert data == retrieved_data
    assert size == ret_size


def test_put_get_shm(mocker: MockerFixture, shm_connector, monkeypatch: pytest.MonkeyPatch):
    """Test SHM transfer logic for large data (Mocked)."""
    # Create data larger than 100 bytes
    data = {"large": "x" * 200}

    # Mock SHM return values
    mock_handle = {"name": "req_2", "size": 200}
    mock_write = mocker.MagicMock(return_value=mock_handle)
    monkeypatch.setattr("vllm_omni.distributed.omni_connectors.connectors.shm_connector.shm_write_bytes", mock_write)

    # When reading, return the serialized bytes of the data
    serialized_data = shm_connector.serialize_obj(data)
    mock_read = mocker.MagicMock(return_value=serialized_data)
    monkeypatch.setattr("vllm_omni.distributed.omni_connectors.connectors.shm_connector.shm_read_bytes", mock_read)

    # Put
    success, size, metadata = shm_connector.put("stage_0", "stage_1", "req_2", data)

    assert success is True
    # Should use SHM because data > threshold
    assert "shm" in metadata
    assert metadata["shm"] == mock_handle
    assert "inline_bytes" not in metadata

    mock_write.assert_called_once()

    # Get
    retrieved_data, ret_size = shm_connector.get("stage_0", "stage_1", "req_2", metadata)

    assert data == retrieved_data
    mock_read.assert_called_once_with(mock_handle)


def test_get_invalid_metadata(shm_connector):
    """Test get with invalid metadata."""
    result = shm_connector.get("stage_0", "stage_1", "req_3", {})
    assert result is None

    result = shm_connector.get("stage_0", "stage_1", "req_3", {"unknown": "format"})
    assert result is None


def test_mooncake_connector_defaults_missing_host_to_detected_ip(monkeypatch: pytest.MonkeyPatch):
    import vllm_omni.distributed.omni_connectors.connectors.mooncake_transfer_engine_connector as mooncake_module

    class _FakePool:
        is_cuda = False

        def pin_memory(self):
            return self

        def data_ptr(self):
            return 1234

    class _FakeTransferEngine:
        def initialize(self, host, mode, protocol, device_name):
            self.host = host
            self.mode = mode
            self.protocol = protocol
            self.device_name = device_name
            return 0

        def get_rpc_port(self):
            return 23456

        def register_memory(self, base_ptr, pool_size):
            del base_ptr, pool_size
            return 0

        def unregister_memory(self, base_ptr):
            del base_ptr
            return 0

    monkeypatch.setattr(mooncake_module, "TransferEngine", _FakeTransferEngine)
    monkeypatch.setattr(mooncake_module.torch, "empty", lambda *args, **kwargs: _FakePool())
    monkeypatch.setattr(
        mooncake_module.MooncakeTransferEngineConnector,
        "_get_local_ip",
        lambda self: "10.20.30.40",
    )
    monkeypatch.setattr(
        mooncake_module.MooncakeTransferEngineConnector,
        "_zmq_listener_loop",
        lambda self: self._listener_ready.set(),
    )

    connector = mooncake_module.MooncakeTransferEngineConnector(
        {
            "zmq_port": 50051,
            "memory_pool_size": 4096,
        }
    )
    try:
        assert connector.host == "10.20.30.40"
        assert connector.engine.host == "10.20.30.40"
        assert connector.get_connection_info()["host"] == "10.20.30.40"
    finally:
        connector.close()


def test_mooncake_cuda_copy_uses_transfer_stream_events(mocker: MockerFixture):
    import vllm_omni.distributed.omni_connectors.connectors.mooncake_transfer_engine_connector as mooncake_module

    connector = object.__new__(mooncake_module.MooncakeTransferEngineConnector)
    connector._cuda_copy_streams = {}
    connector._cuda_copy_streams_lock = threading.Lock()

    producer_stream = mocker.MagicMock()
    transfer_stream = mocker.MagicMock()
    ready_event = mocker.MagicMock()
    complete_event = mocker.MagicMock()
    src = mocker.MagicMock(is_cuda=True, device="cuda:0")
    dst = mocker.MagicMock(is_cuda=False, device="cpu")

    mocker.patch.object(mooncake_module.torch.cuda, "Stream", return_value=transfer_stream)
    mocker.patch.object(mooncake_module.torch.cuda, "current_stream", return_value=producer_stream)
    mocker.patch.object(mooncake_module.torch.cuda, "Event", side_effect=[ready_event, complete_event])
    mocker.patch.object(mooncake_module.torch.cuda, "device", return_value=mocker.MagicMock())
    mocker.patch.object(mooncake_module.torch.cuda, "stream", return_value=mocker.MagicMock())

    connector._copy_tensor_via_cuda_stream(dst, src)

    ready_event.record.assert_called_once_with(producer_stream)
    transfer_stream.wait_event.assert_called_once_with(ready_event)
    dst.copy_.assert_called_once_with(src, non_blocking=True)
    src.record_stream.assert_called_once_with(transfer_stream)
    complete_event.record.assert_called_once_with(transfer_stream)
    complete_event.synchronize.assert_called_once_with()
    producer_stream.synchronize.assert_not_called()
    transfer_stream.synchronize.assert_not_called()


def test_mooncake_cpu_copy_does_not_create_cuda_events(mocker: MockerFixture):
    import vllm_omni.distributed.omni_connectors.connectors.mooncake_transfer_engine_connector as mooncake_module

    connector = object.__new__(mooncake_module.MooncakeTransferEngineConnector)
    connector._cuda_copy_streams = {}
    connector._cuda_copy_streams_lock = threading.Lock()
    src = mocker.MagicMock(is_cuda=False, device="cpu")
    dst = mocker.MagicMock(is_cuda=False, device="cpu")
    event = mocker.patch.object(mooncake_module.torch.cuda, "Event")

    connector._copy_tensor_via_cuda_stream(dst, src)

    dst.copy_.assert_called_once_with(src)
    event.assert_not_called()


def _make_uninitialized_mooncake_connector(pool_size: int = 64 * 1024):
    connector = object.__new__(MooncakeTransferEngineConnector)
    connector.pool = torch.empty(pool_size, dtype=torch.uint8)
    connector.base_ptr = connector.pool.data_ptr()
    connector.allocator = BufferAllocator(pool_size, alignment=8)
    connector._local_buffers = {}
    connector._local_buffers_lock = threading.Lock()
    connector._metrics = {
        "puts": 0,
        "gets": 0,
        "bytes_transferred": 0,
        "errors": 0,
        "timeouts": 0,
    }
    connector.host = "127.0.0.1"
    connector.zmq_port = 50051
    return connector


def test_mooncake_flat_packet_uses_one_multisegment_put():
    connector = _make_uninitialized_mooncake_connector()
    first = connector.allocate_buf(16, dtype=torch.float32, shape=(2, 2))
    second = connector.allocate_buf(8, dtype=torch.int64, shape=(1,))
    first.as_tensor(torch.float32, (2, 2)).copy_(torch.arange(4, dtype=torch.float32).reshape(2, 2))
    second.as_tensor(torch.int64, (1,)).fill_(7)
    packet = {
        "__qwen3_flat__": True,
        "packet_version": 1,
        "payload_kind": "thinker_to_talker_full_payload",
        "payload": {
            "embed.prefill": first,
            "hidden_states.output": second,
            "ids.all": [1, 2],
        },
    }

    success, total_size, metadata = connector._put_impl("request@0_1", packet)

    assert success
    assert metadata["is_segmented"] is True
    assert metadata["segment_lengths"][1:] == [16, 8]
    assert total_size == sum(metadata["segment_lengths"])
    src_addrs, lengths, holders, should_release, is_fast_path, _ = connector._local_buffers["request@0_1"]
    assert len(src_addrs) == len(lengths) == len(holders) == 3
    assert should_release is True
    assert is_fast_path is True


def test_mooncake_segmented_payload_restores_typed_tensor_views():
    connector = _make_uninitialized_mooncake_connector()
    first = connector.allocate_buf(16, dtype=torch.float32, shape=(2, 2))
    second = connector.allocate_buf(8, dtype=torch.int64, shape=(1,))
    first.as_tensor(torch.float32, (2, 2)).copy_(torch.arange(4, dtype=torch.float32).reshape(2, 2))
    second.as_tensor(torch.int64, (1,)).fill_(7)
    packet = {
        "__qwen3_flat__": True,
        "packet_version": 1,
        "payload_kind": "thinker_to_talker_full_payload",
        "payload": {
            "embed.prefill": first,
            "hidden_states.output": second,
            "ids.all": [1, 2],
        },
    }
    success, _, _ = connector._put_impl("request@0_1", packet)
    assert success
    _, _, holders, _, _, _ = connector._local_buffers["request@0_1"]
    sidecar = holders[0].to_bytes()

    restored = connector._restore_segmented_payload(sidecar, holders[1:])

    assert restored["payload"]["ids.all"] == [1, 2]
    assert torch.equal(
        restored["payload"]["embed.prefill"],
        torch.arange(4, dtype=torch.float32).reshape(2, 2),
    )
    assert restored["payload"]["hidden_states.output"].dtype == torch.int64
    assert restored["payload"]["hidden_states.output"].tolist() == [7]
    assert restored[MANAGED_BUFFER_LEASES_KEY] == holders[1:]


def test_managed_buffer_slices_keep_shared_allocation_alive():
    allocator = BufferAllocator(32, alignment=8)
    pool = torch.empty(32, dtype=torch.uint8)
    parent = ManagedBuffer(
        allocator,
        allocator.alloc(32),
        32,
        pool,
        dtype=torch.float32,
        shape=(8,),
    )
    first = parent.slice(0, 16, dtype=torch.float32, shape=(4,))
    second = parent.slice(16, 16, dtype=torch.float32, shape=(4,))

    parent.release()
    first.release()
    with pytest.raises(MemoryError):
        allocator.alloc(32)

    second.release()
    assert allocator.alloc(32) == 0
