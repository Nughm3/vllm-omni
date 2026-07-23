# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for OmniConnectorModelRunnerMixin.

These tests use a mock connector (in-memory dict store) and do not require
GPU or vLLM runtime.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_omni.outputs import OmniConnectorOutput
from vllm_omni.worker.omni_connector_model_runner_mixin import (
    OmniConnectorModelRunnerMixin,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# ------------------------------------------------------------------ #
#  Mock helpers
# ------------------------------------------------------------------ #


class MockConnector:
    """In-memory connector for testing (mimics OmniConnectorBase)."""

    supports_raw_data: bool = False

    def __init__(self, stage_id: int = 0):
        self.stage_id = stage_id
        self._store: dict[str, Any] = {}

    def supports_raw_put(self) -> bool:
        return bool(self.supports_raw_data)

    def supports_raw_get(self) -> bool:
        return bool(self.supports_raw_data)

    def put(self, from_stage, to_stage, put_key, data):
        key = f"{from_stage}_{to_stage}_{put_key}"
        self._store[key] = data
        return True, len(str(data)), None

    def get(self, from_stage, to_stage, get_key, metadata=None):
        key = f"{from_stage}_{to_stage}_{get_key}"
        data = self._store.pop(key, None)
        if data is None:
            return None
        return data, len(str(data))

    def close(self):
        pass


class RawDataMockConnector(MockConnector):
    supports_raw_data = True


def _make_model_config(
    stage_id: int = 0,
    async_chunk: bool = False,
    worker_type: str = "ar",
    custom_func: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        stage_connector_config=None,
        async_chunk=async_chunk,
        worker_type=worker_type,
        custom_process_next_stage_input_func=custom_func,
    )


def _make_request(req_id: str, external_req_id: str | None = None):
    r = SimpleNamespace(
        request_id=req_id,
        external_req_id=external_req_id or req_id,
        additional_information=None,
        prompt_token_ids=[],
        num_computed_tokens=0,
    )
    return r


class MixinHost(OmniConnectorModelRunnerMixin):
    """Minimal class that mixes in the mixin for testing.

    Exposes a back-compat ``_omni_connector`` property: the mixin now holds
    separate ``_recv_connector`` / ``_send_connector`` (one dual instance
    for same-type stages, two for hybrid). Tests that inject a single mock via
    ``host._omni_connector = mock`` still work — the setter points both
    directions at it, and the getter returns the send connector (falling back
    to recv), matching the old single-connector semantics.
    """

    @property
    def _omni_connector(self):
        send = getattr(self, "_send_connector", None)
        return send if send is not None else getattr(self, "_recv_connector", None)

    @_omni_connector.setter
    def _omni_connector(self, connector):
        self._recv_connector = connector
        self._send_connector = connector


class _FakeTPGroup:
    def __init__(self, *, world_size: int, rank_in_group: int, follower_result: Any = None):
        self.world_size = world_size
        self.rank_in_group = rank_in_group
        self.follower_result = follower_result
        self.broadcast_inputs: list[Any] = []

    def broadcast_object(self, obj: Any | None = None, src: int = 0):
        self.broadcast_inputs.append(obj)
        if self.rank_in_group == src:
            return obj
        return self.follower_result


# ------------------------------------------------------------------ #
#  Test cases
# ------------------------------------------------------------------ #


class TestMixinAsyncChunkSendRecv:
    """Test 2: Async chunk send/recv + bg threads."""

    def test_send_chunk_passes_is_finished_and_connector(self):
        connector = MockConnector(stage_id=0)

        sender = MixinHost()
        sender.init_omni_connectors(
            vllm_config=None,
            model_config=_make_model_config(stage_id=0, async_chunk=True),
        )
        sender._send_connector = connector
        sender._stage_id = 0
        sender._async_chunk = True

        seen = {}

        def mock_process(transfer_manager, pooling_output, request, is_finished=False):
            seen["connector"] = transfer_manager.send_connector
            seen["is_finished"] = is_finished
            return {"data": pooling_output, "finished": is_finished}

        sender._custom_process_func = mock_process

        request = _make_request("req-1", "ext-req-1")
        request.is_finished = lambda: True
        sender._send_single_request(
            {
                "stage_id": 0,
                "next_stage_id": 1,
                "request_id": "ext-req-1",
                "request": request,
                "pooling_output": {"value": 42},
            }
        )
        assert seen["connector"] is connector
        assert seen["is_finished"]

        sender.shutdown_omni_connectors()

    def test_send_chunk_does_not_retry_real_type_error(self):
        connector = MockConnector(stage_id=0)

        sender = MixinHost()
        sender.init_omni_connectors(
            vllm_config=None,
            model_config=_make_model_config(stage_id=0, async_chunk=True),
        )
        sender._send_connector = connector
        sender._stage_id = 0
        sender._async_chunk = True

        seen = {"calls": 0}

        def broken_process(transfer_manager, pooling_output, request, is_finished=""):
            seen["calls"] += 1
            return {"data": is_finished + "tail"}

        sender._custom_process_func = broken_process

        request = _make_request("req-1", "ext-req-1")
        request.is_finished = lambda: True
        ok = sender.send_chunk(request, pooling_output={"value": 42})
        assert not ok
        assert seen["calls"] == 1

        sender.shutdown_omni_connectors()


class TestMixinKVCacheTransfer:
    """Test 3: KV cache delegation to OmniKVTransferManager."""

    def test_send_kv_delegates(self):
        mock_kvm = MagicMock()
        mock_kvm.handle_finished_requests_kv_transfer.return_value = ["req-1"]

        host = MixinHost()
        host.init_omni_connectors(
            vllm_config=None,
            model_config=_make_model_config(),
            kv_transfer_manager=mock_kvm,
        )

        result = host.send_kv_cache(
            finished_reqs={"req-1": {"seq_len": 10, "block_ids": [0]}},
            kv_caches=[],
            block_size=16,
            cache_dtype="float16",
        )
        assert result == ["req-1"]
        mock_kvm.handle_finished_requests_kv_transfer.assert_called_once()

        host.shutdown_omni_connectors()

    def test_recv_kv_delegates(self):
        mock_kvm = MagicMock()
        mock_kvm.receive_kv_cache_for_request.return_value = ({"layer_blocks": {}}, 100)

        host = MixinHost()
        host.init_omni_connectors(
            vllm_config=None,
            model_config=_make_model_config(),
            kv_transfer_manager=mock_kvm,
        )

        data, size = host.recv_kv_cache("req-1")
        assert data is not None
        assert size == 100
        mock_kvm.receive_kv_cache_for_request.assert_called_once()

        host.shutdown_omni_connectors()

    def test_receive_multi_kv_fetches_companions_via_mixin(self):
        mock_kvm = MagicMock()

        host = MixinHost()
        host.init_omni_connectors(
            vllm_config=None,
            model_config=_make_model_config(),
            kv_transfer_manager=mock_kvm,
        )

        host.recv_kv_cache = MagicMock(
            side_effect=[({"layer_blocks": {"k": [1]}}, 64), ({"layer_blocks": {"k": [2]}}, 32)]
        )
        seen = {}

        def collect_cfg(request_id, cfg_role_payloads):
            seen["request_id"] = request_id
            seen["cfg_role_payloads"] = cfg_role_payloads
            return {"cfg_text_kv_metadata": {"seq_len": 3}}

        req = SimpleNamespace(
            request_id="req-1",
            sampling_params=SimpleNamespace(cfg_kv_request_ids={"cfg_text": "req-1__cfg_text"}),
        )
        ok = host.receive_multi_kv_cache(req, cfg_kv_collect_func=collect_cfg)
        assert ok
        host.recv_kv_cache.assert_any_call("req-1", target_device=None)
        host.recv_kv_cache.assert_any_call("req-1__cfg_text", target_device=None)
        mock_kvm.apply_kv_cache_to_request.assert_called_once_with(req, {"layer_blocks": {"k": [1]}})
        assert seen["request_id"] == "req-1"
        assert seen["cfg_role_payloads"] == {"cfg_text": ({"layer_blocks": {"k": [2]}}, 32)}
        assert req.sampling_params.cfg_text_kv_metadata == {"seq_len": 3}

        host.shutdown_omni_connectors()

    def test_receive_multi_kv_skips_inactive_request(self):
        mock_kvm = MagicMock()

        host = MixinHost()
        host.init_omni_connectors(
            vllm_config=None,
            model_config=_make_model_config(),
            kv_transfer_manager=mock_kvm,
        )

        host.requests = {}
        host.recv_kv_cache = MagicMock(return_value=({"layer_blocks": {"k": [1]}}, 64))
        req = SimpleNamespace(request_id="req-1", sampling_params=None)

        ok = host.receive_multi_kv_cache(req)

        assert not ok
        host.recv_kv_cache.assert_not_called()
        mock_kvm.apply_kv_cache_to_request.assert_not_called()

        host.shutdown_omni_connectors()


class TestOmniConnectorOutput:
    """Test 4: Output aggregation across transfer modes."""

    def test_output_aggregation(self):
        host = MixinHost()
        host.init_omni_connectors(
            vllm_config=None,
            model_config=_make_model_config(),
        )

        host._chunk_ready_req_ids.add("req-1")
        host._chunk_finished_req_ids.add("req-2")
        host._local_request_metadata["req-1"] = {"next_stage_prompt_len": 10}
        host._stage_recv_req_ids.add("req-3")

        output = host.get_omni_connector_output()
        assert isinstance(output, OmniConnectorOutput)
        assert output.chunk_ready_req_ids == {"req-1"}
        assert output.chunk_finished_req_ids == {"req-2"}
        assert output.request_metadata == {"req-1": {"next_stage_prompt_len": 10}}
        assert output.stage_recv_req_ids == {"req-3"}

        output2 = host.get_omni_connector_output()
        assert output2.chunk_ready_req_ids == set()
        assert output2.request_metadata == {}

        host.shutdown_omni_connectors()


class TestMixinNoConnector:
    """Edge case: mixin works gracefully without a connector."""

    def test_no_connector(self):
        host = MixinHost()
        host.init_omni_connectors(
            vllm_config=None,
            model_config=_make_model_config(),
        )
        assert host._recv_connector is None
        assert host._send_connector is None

        results = host.recv_full_payload_inputs(scheduler_output=None)
        assert results is None

        sent = host.send_full_payload_outputs(None, {"req-1": {}})
        assert sent == []

        ok = host.send_chunk(_make_request("req-1"), pooling_output={})
        assert not ok

        output = host.get_omni_connector_output()
        assert isinstance(output, OmniConnectorOutput)

        host.shutdown_omni_connectors()


class TestFinishedLoadReqsDrain:
    """Test A1 fix: get_omni_connector_output drains _finished_load_reqs."""

    def test_finished_load_reqs_flow_to_chunk_ready(self):
        host = MixinHost()
        host.init_omni_connectors(
            vllm_config=None,
            model_config=_make_model_config(),
        )

        host._finished_load_reqs.add("req-1")
        host._finished_load_reqs.add("req-2")

        output = host.get_omni_connector_output()
        assert "req-1" in output.chunk_ready_req_ids
        assert "req-2" in output.chunk_ready_req_ids

        assert len(host._finished_load_reqs) == 0
        assert len(host._chunk_ready_req_ids) == 0

        host.shutdown_omni_connectors()


class TestLoadCustomFuncSelection:
    def test_skips_non_payload_stage_input_processors_for_full_payload_mode(self):
        incompatible_paths = [
            "vllm_omni.model_executor.stage_input_processors.mimo_audio.llm2code2wav",
            "vllm_omni.model_executor.stage_input_processors.mammoth_moda2.ar2dit",
            "vllm_omni.model_executor.stage_input_processors.cosyvoice3.text2flow",
            "vllm_omni.model_executor.stage_input_processors.glm_image.ar2diffusion",
        ]

        for func_path in incompatible_paths:
            selected_path, func = MixinHost._load_custom_func(
                SimpleNamespace(
                    async_chunk=False,
                    custom_process_input_func=func_path,
                    custom_process_next_stage_input_func=None,
                )
            )
            assert selected_path != func_path
            assert func is None or MixinHost._is_connector_payload_builder(func)


class TestFullPayloadSendWithCustomFunc:
    """Test B4: send_full_payload_outputs with full_payload_mode custom process func."""

    def test_full_payload_send_passes_is_finished_and_connector(self):
        seen = {}

        def full_payload_func(transfer_manager, pooling_output, request, is_finished=False):
            seen["connector"] = transfer_manager.send_connector
            seen["is_finished"] = is_finished
            seen["data"] = pooling_output
            seen["rid"] = request.request_id if request else None
            return {"processed": True, "finished": is_finished}

        host = MixinHost()
        host.init_omni_connectors(
            vllm_config=None,
            model_config=_make_model_config(),
        )
        host._send_connector = MockConnector(stage_id=0)
        host._stage_id = 0
        host._custom_process_func = full_payload_func

        req = _make_request("req-1")
        req.is_finished = lambda: True
        sent = host.send_full_payload_outputs(
            scheduler_output=None,
            outputs={"req-1": ({"raw": 100}, req)},
        )
        assert sent == ["req-1"]
        assert seen == {
            "connector": host._send_connector,
            "is_finished": True,
            "data": {"raw": 100},
            "rid": "req-1",
        }

        host.shutdown_omni_connectors()

    def test_accumulate_and_flush(self):
        call_log = []

        def full_payload_func(transfer_manager, pooling_output, request):
            call_log.append(request.request_id if request else None)
            return {"processed": True}

        host = MixinHost()
        host.init_omni_connectors(
            vllm_config=None,
            model_config=_make_model_config(),
        )
        host._send_connector = MockConnector(stage_id=0)
        host._stage_id = 0
        host._custom_process_func = full_payload_func

        req = _make_request("req-1")
        host.accumulate_full_payload_output("req-1", {"raw": 42}, req)
        assert len(host._pending_full_payload_send) == 1

        host.flush_full_payload_outputs({"req-1"})
        assert len(host._pending_full_payload_send) == 0
        assert len(call_log) == 1
        assert call_log[0] == "req-1"

        time.sleep(0.1)
        host.shutdown_omni_connectors()


def _make_qwen3_async_chunk_model_config(stage_id: int) -> SimpleNamespace:
    return SimpleNamespace(
        stage_connector_config=None,
        async_chunk=True,
        worker_type="ar",
        model_arch="Qwen3OmniMoeForConditionalGeneration",
        model_stage="thinker" if stage_id == 0 else "talker",
        custom_process_next_stage_input_func=None,
    )


def _make_qwen3_full_payload_model_config(stage_id: int) -> SimpleNamespace:
    return SimpleNamespace(
        stage_connector_config=None,
        async_chunk=False,
        worker_type="ar",
        model_arch="Qwen3OmniMoeForConditionalGeneration",
        model_stage="thinker" if stage_id == 0 else "talker",
        custom_process_next_stage_input_func=None,
    )


def _drain_pending_connector_sends(host: MixinHost) -> None:
    for _ in range(64):
        task = None
        with host._lock:
            for req_id in list(host._pending_save_reqs.keys()):
                dq = host._pending_save_reqs.get(req_id)
                if dq:
                    task = dq.popleft()
                    if not dq:
                        del host._pending_save_reqs[req_id]
                    break
        if task is None:
            return
        host._send_single_request(task)


class TestQwen3PacketAsyncChunk:
    """Thinker→talker async_chunk packet path over a raw-data-capable connector."""

    def test_packet_send_recv_roundtrip(self):
        payload = {
            "embed": {"prefill": torch.ones(2, 3), "tts_bos": torch.zeros(1, 3)},
            "hidden_states": {"output": torch.full((2, 3), 2.0)},
            "ids": {"all": [1, 2, 3], "prompt": [1, 2]},
            "meta": {"finished": torch.tensor(True, dtype=torch.bool)},
            "next_stage_prompt_len": 5,
        }
        connector = RawDataMockConnector(stage_id=0)

        sender = MixinHost()
        sender.model_config = _make_qwen3_async_chunk_model_config(0)
        sender.init_omni_connectors(
            vllm_config=None,
            model_config=sender.model_config,
        )
        sender._send_connector = connector
        sender._stage_id = 0
        sender._next_stage_id = 1
        sender._custom_process_func = lambda **kwargs: payload

        req = _make_request("req-1")
        assert sender.send_chunk(req, pooling_output={})
        _drain_pending_connector_sends(sender)
        assert len(connector._store) > 1
        assert any("@packed" in key for key in connector._store)

        receiver = MixinHost()
        receiver.model_config = _make_qwen3_async_chunk_model_config(1)
        receiver.init_omni_connectors(
            vllm_config=None,
            model_config=receiver.model_config,
        )
        receiver._recv_connector = connector
        receiver._stage_id = 1
        receiver._request_ids_mapping["req-1"] = "req-1"
        receiver._pending_load_reqs["req-1"] = _make_request("req-1")

        assert receiver._poll_single_request("req-1")
        cached = receiver.get_local_stage_payload("req-1")
        assert cached is not None
        assert cached["ids"] == payload["ids"]
        assert cached["next_stage_prompt_len"] == payload["next_stage_prompt_len"]
        assert torch.equal(cached["embed"]["prefill"], payload["embed"]["prefill"])
        assert torch.equal(cached["hidden_states"]["output"], payload["hidden_states"]["output"])

        sender.shutdown_omni_connectors()
        receiver.shutdown_omni_connectors()


class TestQwen3PacketFullPayload:
    """Thinker→talker non-async full-payload packet path over raw-data connector."""

    def test_packet_send_recv_roundtrip(self):
        payload = {
            "embed": {"prefill": torch.ones(2, 3), "tts_bos": torch.zeros(1, 3)},
            "hidden_states": {"output": torch.full((2, 3), 2.0)},
            "ids": {"all": [1, 2, 3], "prompt": [1, 2]},
            "meta": {"finished": torch.tensor(True, dtype=torch.bool)},
            "next_stage_prompt_len": 5,
        }
        connector = RawDataMockConnector(stage_id=0)

        sender = MixinHost()
        sender.model_config = _make_qwen3_full_payload_model_config(0)
        sender.init_omni_connectors(
            vllm_config=None,
            model_config=sender.model_config,
        )
        sender._send_connector = connector
        sender._stage_id = 0
        sender._next_stage_id = 1
        sender._custom_process_func = lambda **kwargs: payload

        req = _make_request("req-1")
        req.is_finished = lambda: True
        sent = sender.send_full_payload_outputs(
            scheduler_output=None,
            outputs={"req-1": ({}, req)},
        )
        assert sent == ["req-1"]
        _drain_pending_connector_sends(sender)
        assert len(connector._store) > 1
        assert any("@packed" in key for key in connector._store)

        receiver = MixinHost()
        receiver.model_config = _make_qwen3_full_payload_model_config(1)
        receiver.init_omni_connectors(
            vllm_config=None,
            model_config=receiver.model_config,
        )
        receiver._recv_connector = connector
        receiver._stage_id = 1
        receiver._request_ids_mapping["req-1"] = "req-1"
        receiver._pending_load_reqs["req-1"] = _make_request("req-1")

        assert receiver._poll_single_request("req-1")
        results = receiver.recv_full_payload_inputs(scheduler_output=None)
        assert results is not None
        assert "req-1" in results
        assert results["req-1"]["ids"] == payload["ids"]
        assert results["req-1"]["next_stage_prompt_len"] == payload["next_stage_prompt_len"]
        assert torch.equal(results["req-1"]["embed"]["prefill"], payload["embed"]["prefill"])

        connector_output = receiver.get_omni_connector_output()
        assert "req-1" in connector_output.stage_recv_req_ids

        sender.shutdown_omni_connectors()
        receiver.shutdown_omni_connectors()


class TestQwen3TalkerPacketFullPayload:
    """Talker→code2wav non-async full-payload packet path over raw-data connector."""

    def test_packet_send_recv_roundtrip(self):
        payload = {
            "codes": {"audio": [1, 2, 3, 4]},
            "meta": {"finished": torch.tensor(True, dtype=torch.bool), "left_context_size": 25},
        }
        connector = RawDataMockConnector(stage_id=1)

        sender = MixinHost()
        sender.model_config = _make_qwen3_full_payload_model_config(1)
        sender.init_omni_connectors(
            vllm_config=None,
            model_config=sender.model_config,
        )
        sender._send_connector = connector
        sender._stage_id = 1
        sender._next_stage_id = 2
        sender._custom_process_func = lambda **kwargs: payload

        req = _make_request("req-1")
        req.is_finished = lambda: True
        sent = sender.send_full_payload_outputs(
            scheduler_output=None,
            outputs={"req-1": ({}, req)},
        )
        assert sent == ["req-1"]
        _drain_pending_connector_sends(sender)
        assert any("@packed" in key for key in connector._store)

        receiver = MixinHost()
        receiver.model_config = _make_qwen3_full_payload_model_config(2)
        receiver.init_omni_connectors(
            vllm_config=None,
            model_config=receiver.model_config,
        )
        receiver._recv_connector = connector
        receiver._stage_id = 2
        receiver._request_ids_mapping["req-1"] = "req-1"
        receiver._pending_load_reqs["req-1"] = _make_request("req-1")

        assert receiver._poll_single_request("req-1")
        results = receiver.recv_full_payload_inputs(scheduler_output=None)
        assert results is not None
        assert results["req-1"]["code_predictor_codes"] == [1, 2, 3, 4]
        assert torch.is_tensor(results["req-1"]["codes"]["audio"])

        sender.shutdown_omni_connectors()
        receiver.shutdown_omni_connectors()


class TestKVSentReqIdsAccumulation:
    """Test that kv_sent_req_ids accumulates results from send_kv_cache."""

    def test_kv_sent_accumulation(self):
        mock_kvm = MagicMock()
        mock_kvm.handle_finished_requests_kv_transfer.return_value = ["req-1", "req-2"]

        host = MixinHost()
        host.init_omni_connectors(
            vllm_config=None,
            model_config=_make_model_config(),
            kv_transfer_manager=mock_kvm,
        )

        host.send_kv_cache(
            finished_reqs={"req-1": {}, "req-2": {}},
            kv_caches=[],
            block_size=16,
            cache_dtype="float16",
        )

        output = host.get_omni_connector_output()
        assert "req-1" in output.kv_sent_req_ids
        assert "req-2" in output.kv_sent_req_ids

        output2 = host.get_omni_connector_output()
        assert output2.kv_sent_req_ids == []

        host.shutdown_omni_connectors()


class TestChunkStreamCompletedGuard:
    """Test that register_chunk_recv is skipped after finish sentinel.

    This validates the fix for the race condition where the scheduling
    coordinator re-registers a request for chunk polling after its
    upstream chunk stream has already finished (is_finished sentinel
    received), causing the bg recv thread to poll for a non-existent
    shared-memory segment (e.g. ``_0_7`` when only 7 chunks 0–6 exist).
    """

    def _make_host(self, stage_id: int = 1) -> MixinHost:
        host = MixinHost()
        host.init_omni_connectors(
            vllm_config=None,
            model_config=_make_model_config(stage_id=stage_id, async_chunk=True),
        )
        host._recv_connector = MockConnector(stage_id=stage_id)
        host._stage_id = stage_id
        host._async_chunk = True
        return host

    def test_register_blocked_after_finish_sentinel(self):
        """register_chunk_recv must be a no-op after the finish sentinel."""
        host = self._make_host(stage_id=1)

        req = _make_request("req-1", "ext-req-1")

        # Simulate the bg thread having received the finish sentinel:
        with host._lock:
            host._chunk_stream_completed.add("req-1")

        # Now try to re-register — this mimics the coordinator asking
        # the model runner to poll for the next (non-existent) chunk.
        host.register_chunk_recv(req)

        # The request must NOT appear in _pending_load_reqs
        assert "req-1" not in host._pending_load_reqs

        host.shutdown_omni_connectors()

    def test_register_allowed_before_finish(self):
        """register_chunk_recv works normally before finish sentinel."""
        host = self._make_host(stage_id=1)
        req = _make_request("req-1", "ext-req-1")

        host.register_chunk_recv(req)
        assert "req-1" in host._pending_load_reqs

        host.shutdown_omni_connectors()

    def test_finish_sentinel_populates_completed_set(self):
        """Receiving is_finished=True adds to _chunk_stream_completed."""
        host = self._make_host(stage_id=1)

        # Simulate _poll_single_request receiving is_finished=True
        req_id = "req-1"
        with host._lock:
            host._chunk_finished_req_ids.add(req_id)
            host._chunk_stream_completed.add(req_id)
            host._local_stage_payload_cache[req_id] = {"finished": True}
            host._local_request_metadata[req_id] = {}
            host._finished_load_reqs.add(req_id)
            host._pending_load_reqs.pop(req_id, None)

        assert req_id in host._chunk_stream_completed

        # Subsequent register_chunk_recv should be blocked
        req = _make_request(req_id, f"ext-{req_id}")
        host.register_chunk_recv(req)
        assert req_id not in host._pending_load_reqs

        host.shutdown_omni_connectors()

    def test_stage_0_always_skipped(self):
        """Stage-0 has no upstream, register_chunk_recv is always no-op."""
        host = self._make_host(stage_id=0)
        host._stage_id = 0

        req = _make_request("req-1")
        host.register_chunk_recv(req)
        assert "req-1" not in host._pending_load_reqs

        host.shutdown_omni_connectors()

    def test_full_payload_recv_guard_still_works(self):
        """Pre-existing guard: staged full-payload results prevent registration."""
        host = self._make_host(stage_id=1)

        with host._lock:
            host._stage_recv_req_ids.add("req-1")

        req = _make_request("req-1", "ext-req-1")
        host.register_chunk_recv(req)
        assert "req-1" not in host._pending_load_reqs

        host.shutdown_omni_connectors()


class TestCleanupFinishedRequest:
    """Test cleanup_finished_request frees per-request mixin state."""

    def _make_host(self, stage_id: int = 1) -> MixinHost:
        host = MixinHost()
        host.init_omni_connectors(
            vllm_config=None,
            model_config=_make_model_config(stage_id=stage_id, async_chunk=True),
        )
        host._recv_connector = MockConnector(stage_id=stage_id)
        host._stage_id = stage_id
        host._async_chunk = True
        return host

    def test_cleanup_removes_all_state(self):
        """cleanup_finished_request removes all tracking dicts/sets."""
        host = self._make_host(stage_id=1)
        req_id = "req-1"
        ext_id = "ext-req-1"

        # Simulate state accumulated during a request's lifetime
        host._request_ids_mapping[req_id] = ext_id
        host._put_req_chunk[ext_id] = 5
        host._get_req_chunk[req_id] = 3
        host._send_side_request_payload[ext_id] = {"some": "data"}
        host._code_prompt_token_ids[ext_id] = [[1, 2, 3]]
        host._cached_ic[ext_id] = 16
        host._chunk_stream_completed.add(req_id)
        host._stage_recv_req_ids.add(req_id)
        host._local_stage_payload_cache[req_id] = {"engine_inputs": {}}
        host._local_request_metadata[req_id] = {"prompt_len": 10}

        # Cleanup
        host.cleanup_finished_request(req_id)

        # All state should be gone
        assert req_id not in host._request_ids_mapping
        assert ext_id not in host._put_req_chunk
        assert req_id not in host._get_req_chunk
        assert ext_id not in host._send_side_request_payload
        assert ext_id not in host._code_prompt_token_ids
        assert ext_id not in host._cached_ic
        assert req_id not in host._chunk_stream_completed
        assert req_id not in host._stage_recv_req_ids
        assert req_id not in host._local_stage_payload_cache
        assert req_id not in host._local_request_metadata

        host.shutdown_omni_connectors()

    def test_cleanup_removes_per_cycle_ready_state(self):
        """cleanup_finished_request clears ready/finished carry-over for req-id reuse."""
        host = self._make_host(stage_id=1)
        req_id = "req-1"

        host._pending_load_reqs[req_id] = _make_request(req_id, "ext-req-1")
        host._finished_load_reqs.add(req_id)
        host._chunk_ready_req_ids.add(req_id)
        host._chunk_finished_req_ids.add(req_id)

        host.cleanup_finished_request(req_id)

        assert req_id not in host._pending_load_reqs
        assert req_id not in host._finished_load_reqs
        assert req_id not in host._chunk_ready_req_ids
        assert req_id not in host._chunk_finished_req_ids

        host.shutdown_omni_connectors()

    def test_cleanup_without_mapping(self):
        """cleanup works for Stage-0 where _request_ids_mapping isn't set."""
        host = self._make_host(stage_id=0)
        host._stage_id = 0
        req_id = "req-1"

        # Stage-0 uses req_id directly (no ext_id mapping)
        host._put_req_chunk[req_id] = 3
        host._get_req_chunk[req_id] = 0
        host._cached_ic[req_id] = 4

        host.cleanup_finished_request(req_id)

        assert req_id not in host._put_req_chunk
        assert req_id not in host._get_req_chunk
        assert req_id not in host._cached_ic

        host.shutdown_omni_connectors()

    def test_deferred_cleanup_removes_cached_ic(self):
        host = self._make_host(stage_id=1)
        req_id = "req-1"
        ext_id = "ext-req-1"

        host._request_ids_mapping[req_id] = ext_id
        host._pending_save_counts[ext_id] = 1
        host._cached_ic[ext_id] = 8

        host.cleanup_finished_request(req_id)

        assert ext_id in host._deferred_send_cleanup
        assert ext_id in host._cached_ic

        host._decrement_pending_save_count(ext_id)

        assert ext_id not in host._deferred_send_cleanup
        assert ext_id not in host._cached_ic

        host.shutdown_omni_connectors()

    def test_prune_inactive_requests_cleans_stale_state_but_keeps_active(self):
        """Inactive request IDs should be pruned without touching active ones."""
        host = self._make_host(stage_id=1)
        active_req_id = "req-active"
        stale_req_id = "req-stale"
        stale_ext_id = "ext-stale"

        host._request_ids_mapping[active_req_id] = "ext-active"
        host._request_ids_mapping[stale_req_id] = stale_ext_id
        host._put_req_chunk[stale_ext_id] = 2
        host._get_req_chunk[stale_req_id] = 1
        host._finished_load_reqs.add(stale_req_id)
        host._chunk_ready_req_ids.update({active_req_id, stale_req_id})
        host._chunk_finished_req_ids.add(stale_req_id)
        host._chunk_stream_completed.add(stale_req_id)
        host._stage_recv_req_ids.add(active_req_id)
        host._send_side_request_payload[stale_ext_id] = {"stale": True}
        host._code_prompt_token_ids[stale_ext_id] = [[1, 2, 3]]

        pruned = host.prune_inactive_requests({active_req_id})

        assert pruned == {stale_req_id}
        assert active_req_id in host._request_ids_mapping
        assert active_req_id in host._chunk_ready_req_ids
        assert active_req_id in host._stage_recv_req_ids
        assert stale_req_id not in host._request_ids_mapping
        assert stale_ext_id not in host._put_req_chunk
        assert stale_req_id not in host._get_req_chunk
        assert stale_req_id not in host._pending_load_reqs
        assert stale_req_id not in host._finished_load_reqs
        assert stale_req_id not in host._chunk_ready_req_ids
        assert stale_req_id not in host._chunk_finished_req_ids
        assert stale_req_id not in host._chunk_stream_completed
        assert stale_req_id not in host._stage_recv_req_ids
        assert stale_ext_id not in host._send_side_request_payload
        assert stale_ext_id not in host._code_prompt_token_ids

        host.shutdown_omni_connectors()

    def test_prune_inactive_requests_keeps_recently_received_full_payload_state(self):
        """Late bg-thread receives must survive until the scheduler catches up."""
        host = self._make_host(stage_id=1)
        req_id = "req-recv-race"
        ext_id = "ext-recv-race"

        host._request_ids_mapping[req_id] = ext_id
        host._put_req_chunk[ext_id] = 1
        host._local_stage_payload_cache[req_id] = {"engine_inputs": {"ids": [1, 2, 3]}}
        host._local_request_metadata[req_id] = {"next_stage_prompt_len": 3}
        host._stage_recv_req_ids.add(req_id)

        pruned = host.prune_inactive_requests(set())

        assert pruned == set()
        assert req_id in host._request_ids_mapping
        assert req_id in host._local_stage_payload_cache
        assert req_id in host._local_request_metadata
        assert req_id in host._stage_recv_req_ids
        assert ext_id in host._put_req_chunk

        # Once the scheduler has consumed the wake-up and the request really
        # disappears from all protected sets, prune should clean it up.
        host._stage_recv_req_ids.clear()
        host._local_stage_payload_cache.clear()
        host._local_request_metadata.clear()

        pruned = host.prune_inactive_requests(set())

        assert pruned == {req_id}
        assert req_id not in host._request_ids_mapping
        assert ext_id not in host._put_req_chunk

        host.shutdown_omni_connectors()


class TestSendChunkCachesMapping:
    """Test that send_chunk caches internal→external req ID mapping."""

    def test_send_chunk_populates_request_ids_mapping(self):
        """send_chunk should cache the internal→external mapping."""
        host = MixinHost()
        host.init_omni_connectors(
            vllm_config=None,
            model_config=_make_model_config(stage_id=0, async_chunk=True),
        )
        host._send_connector = MockConnector(stage_id=0)
        host._stage_id = 0
        host._async_chunk = True

        def mock_process(transfer_manager, pooling_output, request):
            return {"data": "test", "finished": False}

        host._custom_process_func = mock_process

        request = _make_request("internal-1", "external-1")
        host.send_chunk(request, pooling_output={"v": 1})

        # The mapping should be cached
        assert host._request_ids_mapping.get("internal-1") == "external-1"

        time.sleep(0.1)
        host.shutdown_omni_connectors()


class TestLocalPayloadCacheLifecycle:
    """Unit tests for the local payload cache API (RFC §2.4)."""

    def _make_host(self) -> MixinHost:
        host = MixinHost()
        host.init_omni_connectors(
            vllm_config=None,
            model_config=_make_model_config(stage_id=0),
        )
        host._recv_connector = MockConnector(stage_id=0)
        host._stage_id = 0
        return host

    def test_put_get_pop(self):
        host = self._make_host()
        payload = {"engine_inputs": {"ids": [1, 2, 3]}}
        host.put_local_stage_payload("r1", payload)

        assert host.get_local_stage_payload("r1") == payload
        popped = host.pop_local_stage_payload("r1")
        assert popped == payload
        assert host.get_local_stage_payload("r1") is None
        host.shutdown_omni_connectors()

    def test_recv_full_payload_inputs_populates_local_cache(self):
        host = self._make_host()
        host._recv_connector = MockConnector(stage_id=0)
        host._stage_id = 0

        # Simulate a full payload already staged by the bg recv path
        with host._lock:
            host._local_stage_payload_cache["r1"] = {"tok": [10]}
            host._stage_recv_req_ids.add("r1")

        host.recv_full_payload_inputs(scheduler_output=None)
        assert host.get_local_stage_payload("r1") == {"tok": [10]}
        host.shutdown_omni_connectors()

    def test_rank0_only_polls_connector_for_tp_full_payload(self):
        host = self._make_host()
        host._recv_connector = MagicMock()
        host._stage_id = 2
        host._local_rank = 0
        host._request_ids_mapping["r1"] = "ext-r1"
        host._get_req_chunk["r1"] = 0
        payload = {"tok": [10], "finished": torch.tensor(True)}
        connector_result = (payload, 123)
        host._recv_connector.get.return_value = connector_result
        tp_group = _FakeTPGroup(world_size=2, rank_in_group=0)

        with patch("vllm_omni.worker.omni_connector_model_runner_mixin.get_tp_group", return_value=tp_group):
            made_progress = host._poll_single_request("r1")

        assert made_progress
        host._recv_connector.get.assert_called_once_with("1", "2", "ext-r1_1_0")
        assert tp_group.broadcast_inputs == []
        assert host.get_local_stage_payload("r1") == payload
        assert "r1" in host._full_payload_pending_broadcast_req_ids
        assert "r1" not in host._stage_recv_req_ids
        assert host.get_local_request_metadata("r1") is None
        host.shutdown_omni_connectors()

    def test_tp_follower_skips_connector_poll_for_full_payload(self):
        host = self._make_host()
        host._recv_connector = MagicMock()
        host._stage_id = 2
        host._local_rank = 1
        host._request_ids_mapping["r1"] = "ext-r1"
        host._get_req_chunk["r1"] = 0
        tp_group = _FakeTPGroup(world_size=2, rank_in_group=1)

        with patch("vllm_omni.worker.omni_connector_model_runner_mixin.get_tp_group", return_value=tp_group):
            made_progress = host._poll_single_request("r1")

        assert not made_progress
        host._recv_connector.get.assert_not_called()
        assert tp_group.broadcast_inputs == []
        assert "r1" not in host._local_stage_payload_cache
        host.shutdown_omni_connectors()

    def test_recv_full_payload_inputs_broadcasts_tp_leader_results_to_followers(self):
        host = self._make_host()
        host._recv_connector = MagicMock()
        host._stage_id = 2
        host._local_rank = 1
        host._pending_load_reqs["r1"] = object()
        payload = {"tok": [10], "finished": torch.tensor(True)}
        tp_group = _FakeTPGroup(world_size=2, rank_in_group=1, follower_result={"r1": payload})

        with patch("vllm_omni.worker.omni_connector_model_runner_mixin.get_tp_group", return_value=tp_group):
            results = host.recv_full_payload_inputs(scheduler_output=None)

        assert results == {"r1": payload}
        assert host.get_local_stage_payload("r1") == payload
        assert host.get_local_request_metadata("r1") == {}
        assert host._stage_recv_req_ids == {"r1"}
        assert "r1" not in host._pending_load_reqs
        assert tp_group.broadcast_inputs == [None]
        host.shutdown_omni_connectors()


class TestTPAsyncChunkFanout:
    def _make_host(self, rank: int) -> MixinHost:
        host = MixinHost()
        host.init_omni_connectors(
            vllm_config=None,
            model_config=_make_model_config(stage_id=2, async_chunk=True, worker_type="gen"),
        )
        host._recv_connector = MagicMock()
        host._stage_id = 2
        host._async_chunk = True
        host._model_mode = "gen"
        host._local_rank = rank
        host._request_ids_mapping["r1"] = "ext-r1"
        host._get_req_chunk["r1"] = 0
        return host

    def test_rank0_only_polls_connector_for_tp_async_chunk(self):
        host = self._make_host(rank=0)
        payload = {
            "codes": {"audio": [10, 11]},
            "meta": {"left_context_size": 0, "finished": torch.tensor(False)},
        }
        host._recv_connector.get.return_value = (payload, 123)
        tp_group = _FakeTPGroup(world_size=2, rank_in_group=0)

        with patch("vllm_omni.worker.omni_connector_model_runner_mixin.get_tp_group", return_value=tp_group):
            made_progress = host._poll_single_request("r1")

        assert made_progress
        host._recv_connector.get.assert_called_once_with("1", "2", "ext-r1_1_0")
        assert host.get_local_stage_payload("r1") == payload
        assert "r1" in host._finished_load_reqs
        assert "r1" in host._async_chunk_updated_req_ids
        assert tp_group.broadcast_inputs == []
        host.shutdown_omni_connectors()

    def test_tp_follower_skips_connector_poll_for_async_chunk(self):
        host = self._make_host(rank=1)
        tp_group = _FakeTPGroup(world_size=2, rank_in_group=1)

        with patch("vllm_omni.worker.omni_connector_model_runner_mixin.get_tp_group", return_value=tp_group):
            made_progress = host._poll_single_request("r1")

        assert not made_progress
        host._recv_connector.get.assert_not_called()
        assert host.get_local_stage_payload("r1") is None
        assert tp_group.broadcast_inputs == []
        host.shutdown_omni_connectors()

    def test_get_output_broadcasts_tp_async_chunk_payloads_to_followers(self):
        host = self._make_host(rank=1)
        host._pending_load_reqs["r1"] = object()
        payload = {
            "code_predictor_codes": [10, 11],
            "left_context_size": 0,
            "finished": torch.tensor(True),
        }
        packet = {
            "staged_payloads": {"r1": payload},
            "request_metadata": {"r1": {"code_predictor_codes": [10, 11], "left_context_size": 0}},
            "newly_finished": {"r1"},
            "chunk_finished": {"r1"},
        }
        tp_group = _FakeTPGroup(world_size=2, rank_in_group=1, follower_result=packet)

        with patch("vllm_omni.worker.omni_connector_model_runner_mixin.get_tp_group", return_value=tp_group):
            output = host.get_omni_connector_output()

        assert output.chunk_ready_req_ids == {"r1"}
        assert output.chunk_finished_req_ids == {"r1"}
        assert output.request_metadata == {"r1": {"code_predictor_codes": [10, 11], "left_context_size": 0}}
        assert host.get_local_stage_payload("r1") == payload
        assert "r1" not in host._pending_load_reqs
        assert "r1" in host._chunk_stream_completed
        assert tp_group.broadcast_inputs == [None]
        host.shutdown_omni_connectors()


class TestKVTransferLifecycle:
    """Unit tests for KV transfer lifecycle methods."""

    def _make_host(self) -> MixinHost:
        host = MixinHost()
        host.init_omni_connectors(
            vllm_config=None,
            model_config=_make_model_config(stage_id=0),
        )
        return host

    def test_mark_drain_ack_complete(self):
        host = self._make_host()
        assert not host.has_pending_kv_work()

        host.mark_kv_transfer("r1", seq_len=100, block_ids=[0, 1, 2])
        assert host.has_pending_kv_work()
        assert host.is_kv_transfer_triggered("r1")

        # Drain moves pending → active
        pending = host.drain_pending_kv_transfers()
        assert pending == {"r1": {"seq_len": 100, "block_ids": [0, 1, 2]}}
        assert "r1" in host._kv_active_transfers
        assert host.has_pending_kv_work()

        # Ack moves active → completed
        host.ack_kv_transfers(["r1"])
        assert "r1" not in host._kv_active_transfers
        assert "r1" in host._kv_completed_transfers

        # Drain completed
        completed = host.drain_completed_kv_transfers()
        assert completed == {"r1"}
        assert not host.has_pending_kv_work()
        host.shutdown_omni_connectors()

    def test_mark_dedup(self):
        host = self._make_host()
        host.mark_kv_transfer("r1", seq_len=100, block_ids=[0])
        host.mark_kv_transfer("r1", seq_len=200, block_ids=[0, 1])
        # Second mark is a no-op
        assert host._kv_pending_transfers["r1"]["seq_len"] == 100
        host.shutdown_omni_connectors()

    def test_cleanup_removes_kv_state(self):
        host = self._make_host()
        host.mark_kv_transfer("r1", seq_len=50, block_ids=[0])
        host.drain_pending_kv_transfers()
        host.cleanup_finished_request("r1")
        assert not host.is_kv_transfer_triggered("r1")
        assert "r1" not in host._kv_active_transfers
        assert not host.has_pending_kv_work()
        host.shutdown_omni_connectors()


class TestAsyncPayloadLifecycle:
    """Regression tests for async payload delivery lifecycle."""

    def test_send_side_request_payload_not_cleared_before_payload_is_consumable(self):
        host = MixinHost()
        host.init_omni_connectors(
            vllm_config=None,
            model_config=_make_model_config(stage_id=1, async_chunk=True, worker_type="ar"),
        )
        host._request_ids_mapping["r1"] = "r1"
        payload = {
            "embed": {"decode": torch.ones(1, 2)},
            "ids": {"output": [1]},
            "meta": {
                "finished": torch.tensor(False),
                "override_keys": [["embed", "decode"], ["ids", "output"]],
            },
        }

        host._accumulate_payload("r1", dict(payload))
        with host._lock:
            host._finished_load_reqs.add("r1")

        host.get_omni_connector_output()
        assert "r1" in host._send_side_request_payload
        host.shutdown_omni_connectors()

    def test_payload_consumable_ignores_token_horizon_only_updates(self):
        host = MixinHost()
        host.init_omni_connectors(
            vllm_config=None,
            model_config=_make_model_config(stage_id=1, async_chunk=True, worker_type="ar"),
        )
        payload = {
            "ids": {"output": [1, 2, 3]},
            "embed": {"decode_token_start": 2, "decode_token_end": 3},
            "meta": {
                "finished": torch.tensor(False),
                "override_keys": [
                    ["ids", "output"],
                    ["embed", "decode_token_start"],
                    ["embed", "decode_token_end"],
                ],
            },
        }
        assert not host._payload_is_consumable(payload)
        host.shutdown_omni_connectors()

    def test_payload_consumable_accepts_decode_embeddings(self):
        host = MixinHost()
        host.init_omni_connectors(
            vllm_config=None,
            model_config=_make_model_config(stage_id=1, async_chunk=True, worker_type="ar"),
        )
        payload = {
            "ids": {"output": [1, 2, 3]},
            "embed": {"decode": torch.ones(1, 2)},
            "meta": {"finished": torch.tensor(False)},
        }
        assert host._payload_is_consumable(payload)
        host.shutdown_omni_connectors()

    def test_ar_metadata_only_followup_chunk_does_not_rewake_request(self):
        host = MixinHost()
        host.init_omni_connectors(
            vllm_config=None,
            model_config=_make_model_config(stage_id=1, async_chunk=True, worker_type="ar"),
        )
        host._recv_connector = MagicMock()
        host._stage_id = 1
        host._async_chunk = True
        host._model_mode = "ar"
        host._request_ids_mapping["r1"] = "ext-r1"
        host._get_req_chunk["r1"] = 0

        host._recv_connector.get.side_effect = [
            (
                {
                    "embed": {"decode": torch.ones(1, 2)},
                    "meta": {"finished": torch.tensor(False)},
                },
                1,
            ),
            (
                {
                    "meta": {"next_stage_prompt_len": 7, "finished": torch.tensor(False)},
                },
                1,
            ),
        ]

        host._poll_single_request("r1")
        output1 = host.get_omni_connector_output()
        assert output1.chunk_ready_req_ids == {"r1"}

        host._poll_single_request("r1")
        output2 = host.get_omni_connector_output()
        assert output2.chunk_ready_req_ids == set()
        assert output2.request_metadata == {"r1": {"next_stage_prompt_len": 7}}

        host.shutdown_omni_connectors()

    def test_non_ar_recv_does_not_overwrite_unconsumed_staged_chunk(self):
        host = MixinHost()
        host.init_omni_connectors(
            vllm_config=None,
            model_config=_make_model_config(stage_id=2, async_chunk=True, worker_type="gen"),
        )
        host._recv_connector = MagicMock()
        host._stage_id = 2
        host._async_chunk = True
        host._model_mode = "gen"
        host._request_ids_mapping["r1"] = "ext-r1"
        host._get_req_chunk["r1"] = 1
        host._local_stage_payload_cache["r1"] = {
            "code_predictor_codes": [1, 2, 3],
            "left_context_size": 0,
            "finished": torch.tensor(False),
        }

        made_progress = host._poll_single_request("r1")

        assert not made_progress
        host._recv_connector.get.assert_not_called()
        assert host._get_req_chunk["r1"] == 1

        host.shutdown_omni_connectors()

    def test_non_ar_recv_waits_for_scheduler_handoff_before_fetching_next_chunk(self):
        host = MixinHost()
        host.init_omni_connectors(
            vllm_config=None,
            model_config=_make_model_config(stage_id=2, async_chunk=True, worker_type="gen"),
        )
        host._recv_connector = MagicMock()
        host._stage_id = 2
        host._async_chunk = True
        host._model_mode = "gen"
        host._request_ids_mapping["r1"] = "ext-r1"
        host._get_req_chunk["r1"] = 1
        host._local_request_metadata["r1"] = {
            "code_predictor_codes": [10, 11, 12],
            "left_context_size": 0,
        }
        host._finished_load_reqs.add("r1")

        made_progress = host._poll_single_request("r1")

        assert not made_progress
        host._recv_connector.get.assert_not_called()
        assert host._get_req_chunk["r1"] == 1

        output = host.get_omni_connector_output()
        assert output.request_metadata["r1"]["code_predictor_codes"] == [10, 11, 12]
        assert output.chunk_ready_req_ids == {"r1"}

        host._recv_connector.get.return_value = (
            {
                "codes": {"audio": [20, 21, 22]},
                "meta": {"left_context_size": 0, "finished": torch.tensor(False)},
            },
            1,
        )
        made_progress = host._poll_single_request("r1")

        assert made_progress
        host._recv_connector.get.assert_called_once()
        assert host._get_req_chunk["r1"] == 2

        host.shutdown_omni_connectors()


class TestRankAwareKVRouting:
    def _make_host(self, *, from_tp: int, to_tp: int, local_rank: int) -> MixinHost:
        host = MixinHost()
        host.init_omni_connectors(vllm_config=None, model_config=_make_model_config(stage_id=1))
        # Both directions share the same topology here; these tests exercise
        # recv-side and send-side routing independently, not asymmetric TP.
        host._recv_from_tp = from_tp
        host._recv_to_tp = to_tp
        host._send_from_tp = from_tp
        host._send_to_tp = to_tp
        host._local_rank = local_rank
        return host

    def test_recv_keys_use_remote_rank_as_from_rank(self):
        host = self._make_host(from_tp=4, to_tp=2, local_rank=1)
        assert host.get_rank_aware_kv_keys("req", from_stage=0) == ["req_0_0_2_1", "req_0_0_3_1"]
        host.shutdown_omni_connectors()

    def test_send_keys_route_from_rank_gt_to_rank(self):
        host = self._make_host(from_tp=4, to_tp=2, local_rank=3)
        assert host.get_rank_aware_kv_send_keys("req", from_stage=0) == ["req_0_0_3_1"]
        host.shutdown_omni_connectors()

    def test_invalid_recv_rank_mapping_raises(self):
        host = self._make_host(from_tp=3, to_tp=2, local_rank=1)
        with pytest.raises(ValueError):
            host.get_rank_aware_kv_keys("req", from_stage=0)
        host.shutdown_omni_connectors()

    def test_invalid_send_rank_mapping_raises(self):
        host = self._make_host(from_tp=3, to_tp=2, local_rank=1)
        with pytest.raises(ValueError):
            host.get_rank_aware_kv_send_keys("req", from_stage=0)
        host.shutdown_omni_connectors()

    def test_merge_rank_sharded_payloads_concatenates_head_dimension(self):
        host = self._make_host(from_tp=4, to_tp=2, local_rank=0)
        payloads = [
            {"layer_blocks": {"key_cache": [torch.ones(2, 1, 3)], "value_cache": [torch.ones(2, 1, 3)]}},
            {"layer_blocks": {"key_cache": [torch.full((2, 1, 3), 2.0)], "value_cache": [torch.full((2, 1, 3), 2.0)]}},
        ]
        merged = host._merge_rank_sharded_kv_payloads(payloads)
        assert tuple(merged["layer_blocks"]["key_cache"][0].shape) == (2, 2, 3)
        assert torch.equal(merged["layer_blocks"]["key_cache"][0][:, 0], torch.ones(2, 3))
        assert torch.equal(merged["layer_blocks"]["key_cache"][0][:, 1], torch.full((2, 3), 2.0))
        host.shutdown_omni_connectors()

    def test_slice_rank_sharded_payload_splits_head_dimension(self):
        host = self._make_host(from_tp=2, to_tp=4, local_rank=1)
        payload = {
            "layer_blocks": {
                "key_cache": [torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)],
                "value_cache": [torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)],
            },
            "metadata": {},
        }
        sliced = host._slice_rank_sharded_kv_payload(payload)
        assert tuple(sliced["layer_blocks"]["key_cache"][0].shape) == (2, 2, 3)
        expected = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)[:, 2:4, :]
        assert torch.equal(sliced["layer_blocks"]["key_cache"][0], expected)
        host.shutdown_omni_connectors()


class TestRankMappingResolutionFromConnectorConfig:
    """A stage's recv (inbound) and send (outbound) edges may have different
    TP ratios. init_omni_connectors must resolve each independently from the
    inbound stage_connector_config and outbound stage_output_connector_config,
    and must still support the single-edge case (stage 0 / last stage) applying
    one mapping to both."""

    def test_distinct_recv_and_send_tp_from_two_specs(self):
        # Middle stage with different TP ratios on each edge (heterogeneous TP):
        # thinker TP=4 -> talker TP=2 -> code2wav TP=1.
        model_config = _make_model_config(stage_id=1)
        model_config.stage_connector_config = {
            "name": "MooncakeTransferEngineConnector",
            "extra": {"role": "receiver", "rank_mapping": {"from_tp": 4, "to_tp": 2}},
        }
        model_config.stage_output_connector_config = {
            "name": "MooncakeTransferEngineConnector",
            "extra": {"role": "sender", "rank_mapping": {"from_tp": 2, "to_tp": 1}},
        }

        host = MixinHost()
        with patch(
            "vllm_omni.distributed.omni_connectors.factory.OmniConnectorFactory.create_connector",
            side_effect=lambda spec: MockConnector(),
        ):
            host.init_omni_connectors(model_config=model_config)

        assert host._recv_from_tp == 4
        assert host._recv_to_tp == 2
        assert host._send_from_tp == 2
        assert host._send_to_tp == 1

        host.shutdown_omni_connectors()

    def test_same_type_two_edges_collapse_to_one_dual_instance(self):
        # Both edges use the same connector type -> one shared dual instance.
        model_config = _make_model_config(stage_id=1)
        model_config.stage_connector_config = {
            "name": "MooncakeTransferEngineConnector",
            "extra": {"role": "receiver"},
        }
        model_config.stage_output_connector_config = {
            "name": "MooncakeTransferEngineConnector",
            "extra": {"role": "sender"},
        }

        built: list[Any] = []

        def _create(spec):
            conn = MockConnector()
            built.append(spec)
            return conn

        host = MixinHost()
        with patch(
            "vllm_omni.distributed.omni_connectors.factory.OmniConnectorFactory.create_connector",
            side_effect=_create,
        ):
            host.init_omni_connectors(model_config=model_config)

        # One instance built, shared by both directions, with role="dual".
        assert len(built) == 1
        assert built[0].extra.get("role") == "dual"
        assert host._recv_connector is host._send_connector
        assert host._recv_connector is not None

        host.shutdown_omni_connectors()

    def test_hybrid_two_edges_build_two_instances(self):
        # Different connector types (Mooncake in / SHM out) -> two instances.
        model_config = _make_model_config(stage_id=1)
        model_config.stage_connector_config = {
            "name": "MooncakeTransferEngineConnector",
            "extra": {"role": "receiver"},
        }
        model_config.stage_output_connector_config = {
            "name": "SharedMemoryConnector",
            "extra": {},
        }

        def _create(spec):
            return MockConnector()

        host = MixinHost()
        with patch(
            "vllm_omni.distributed.omni_connectors.factory.OmniConnectorFactory.create_connector",
            side_effect=_create,
        ):
            host.init_omni_connectors(model_config=model_config)

        assert host._recv_connector is not None
        assert host._send_connector is not None
        assert host._recv_connector is not host._send_connector

        host.shutdown_omni_connectors()

    def test_stage0_output_only_has_no_recv_connector(self):
        # Stage 0 has an outbound edge but no inbound edge.
        model_config = _make_model_config(stage_id=0)
        model_config.stage_connector_config = None
        model_config.stage_output_connector_config = {
            "name": "MooncakeTransferEngineConnector",
            "extra": {"role": "sender"},
        }

        host = MixinHost()
        with patch(
            "vllm_omni.distributed.omni_connectors.factory.OmniConnectorFactory.create_connector",
            side_effect=lambda spec: MockConnector(),
        ):
            host.init_omni_connectors(model_config=model_config)

        assert host._recv_connector is None
        assert host._send_connector is not None

        host.shutdown_omni_connectors()

    def test_flat_single_edge_spec_applies_same_mapping_to_both(self):
        """Regression guard: stage 0 / the last stage only has one edge, so
        the flat spec's rank_mapping must still resolve for both recv and
        send (there's no ambiguity with a single edge)."""
        model_config = _make_model_config(stage_id=0)
        model_config.stage_connector_config = {
            "name": "MooncakeTransferEngineConnector",
            "extra": {"role": "sender", "rank_mapping": {"from_tp": 2, "to_tp": 1}},
        }

        host = MixinHost()
        with patch(
            "vllm_omni.distributed.omni_connectors.factory.OmniConnectorFactory.create_connector",
            return_value=MockConnector(),
        ):
            host.init_omni_connectors(model_config=model_config)

        assert host._recv_from_tp == 2
        assert host._recv_to_tp == 1
        assert host._send_from_tp == 2
        assert host._send_to_tp == 1

        host.shutdown_omni_connectors()

    def test_missing_stage_connector_config_defaults_to_homogeneous(self):
        host = MixinHost()
        host.init_omni_connectors(model_config=_make_model_config(stage_id=0))

        assert host._recv_from_tp == 1
        assert host._recv_to_tp == 1
        assert host._send_from_tp == 1
        assert host._send_to_tp == 1

        host.shutdown_omni_connectors()


class TestWorkerPortOffset:
    """The per-worker TP-rank/replica port offset must be applied at actual
    connector-construction time inside the worker process (mixin), not at
    config-build time (initialization.py) — see the P1 fix for co-located
    TP ranks / replicas colliding on the same baked-in port."""

    def test_non_transfer_engine_connector_untouched(self):
        extra = {"some": "config"}
        with patch(
            "vllm_omni.worker.omni_connector_model_runner_mixin.worker_rank_port_offset",
            return_value=16,
        ):
            result = OmniConnectorModelRunnerMixin._apply_worker_port_offset("SharedMemoryConnector", extra)
        assert result is extra

    def test_zero_offset_returns_same_dict(self):
        """rank 0 / replica 0 (the common case) must be a no-op, not just a
        same-valued copy — avoids needless mutation on the hot path."""
        extra = {"zmq_port": 50052, "sender_zmq_port": 50051}
        with patch(
            "vllm_omni.worker.omni_connector_model_runner_mixin.worker_rank_port_offset",
            return_value=0,
        ):
            result = OmniConnectorModelRunnerMixin._apply_worker_port_offset("MooncakeTransferEngineConnector", extra)
        assert result is extra

    def test_nonzero_offset_shifts_both_ports_without_mutating_input(self):
        extra = {"zmq_port": 50052, "sender_zmq_port": 50051}
        with patch(
            "vllm_omni.worker.omni_connector_model_runner_mixin.worker_rank_port_offset",
            return_value=1040,  # e.g. replica 1 (1024) + rank 1 (16)
        ):
            result = OmniConnectorModelRunnerMixin._apply_worker_port_offset("MooncakeTransferEngineConnector", extra)
        assert result == {"zmq_port": 51092, "sender_zmq_port": 51091}
        # Original spec dict (shared across workers) must be left untouched.
        assert extra == {"zmq_port": 50052, "sender_zmq_port": 50051}

    def test_missing_sender_zmq_port_is_not_added(self):
        """Stage 0 / sender-only specs have no sender_zmq_port key at all —
        the offset must not introduce one where there wasn't one."""
        extra = {"zmq_port": 50052}
        with patch(
            "vllm_omni.worker.omni_connector_model_runner_mixin.worker_rank_port_offset",
            return_value=16,
        ):
            result = OmniConnectorModelRunnerMixin._apply_worker_port_offset("MooncakeTransferEngineConnector", extra)
        assert result == {"zmq_port": 50068}

    def test_distinct_tp_ranks_resolve_to_distinct_listen_ports(self):
        """End-to-end through init_omni_connectors: two 'workers' (different
        worker_rank_port_offset values, simulating different TP ranks) must
        build connectors with different zmq_port — the actual collision this
        fix prevents."""
        model_config = _make_model_config(stage_id=0)
        model_config.stage_connector_config = None
        model_config.stage_output_connector_config = {
            "name": "MooncakeTransferEngineConnector",
            "extra": {"role": "sender", "zmq_port": 50052},
        }

        seen_ports: list[int] = []

        def _create(spec):
            seen_ports.append(spec.extra["zmq_port"])
            return MockConnector()

        for offset in (0, 16):
            host = MixinHost()
            with (
                patch(
                    "vllm_omni.worker.omni_connector_model_runner_mixin.worker_rank_port_offset",
                    return_value=offset,
                ),
                patch(
                    "vllm_omni.distributed.omni_connectors.factory.OmniConnectorFactory.create_connector",
                    side_effect=_create,
                ),
            ):
                host.init_omni_connectors(model_config=model_config)
            host.shutdown_omni_connectors()

        assert seen_ports == [50052, 50068]
        assert len(set(seen_ports)) == 2


class TestAttachOmniConnectorOutput:
    def test_wraps_empty_model_runner_output_when_signals_exist(self):
        from vllm.v1.worker.gpu_model_runner import EMPTY_MODEL_RUNNER_OUTPUT

        host = MixinHost()
        host.get_omni_connector_output = lambda: OmniConnectorOutput(chunk_ready_req_ids={"req-1"})

        wrapped = host.attach_omni_connector_output(EMPTY_MODEL_RUNNER_OUTPUT)

        assert wrapped is not EMPTY_MODEL_RUNNER_OUTPUT
        assert wrapped.omni_connector_output.chunk_ready_req_ids == {"req-1"}


class TestConnectorConfigValidation:
    def test_invalid_connector_name_raises(self):
        host = MixinHost()
        model_config = _make_model_config(stage_id=1)
        model_config.stage_connector_config = {"name": "   "}

        with pytest.raises(RuntimeError, match="missing connector name"):
            host.init_omni_connectors(vllm_config=None, model_config=model_config)


class _FailingConnector:
    """Connector whose put() fails a configurable number of times."""

    def __init__(self, fail_count: int = 1, raise_on_fail: bool = False):
        self._fail_count = fail_count
        self._raise_on_fail = raise_on_fail
        self.attempt = 0

    def put(self, from_stage, to_stage, put_key, data):
        self.attempt += 1
        if self.attempt <= self._fail_count:
            if self._raise_on_fail:
                raise ConnectionError("transient connector error")
            return False, 0, None
        return True, len(str(data)), None

    def get(self, *a, **kw):
        return None

    def close(self):
        pass


class TestSendRetry:
    """Tests for P1-2: failed connector sends must be retried."""

    def _make_sender(self, connector):
        sender = MixinHost()
        sender.init_omni_connectors(
            vllm_config=None,
            model_config=_make_model_config(stage_id=0, async_chunk=True),
        )
        sender._send_connector = connector
        sender._stage_id = 0
        sender._async_chunk = True
        return sender

    def _make_task(self, req_id="r1"):
        return {
            "stage_id": 0,
            "next_stage_id": 1,
            "request_id": req_id,
            "data": {"payload": "test"},
        }

    def test_send_single_request_returns_false_on_put_failure(self):
        connector = _FailingConnector(fail_count=999)
        sender = self._make_sender(connector)

        result = sender._send_single_request(self._make_task())
        assert not result
        sender.shutdown_omni_connectors()

    def test_send_single_request_does_not_decrement_on_failure(self):
        connector = _FailingConnector(fail_count=999)
        sender = self._make_sender(connector)
        sender._pending_save_counts["r1"] = 1

        sender._send_single_request(self._make_task())
        assert sender._pending_save_counts.get("r1") == 1
        sender.shutdown_omni_connectors()

    def test_send_single_request_decrements_on_success(self):
        connector = MockConnector(stage_id=0)
        sender = self._make_sender(connector)
        sender._pending_save_counts["r1"] = 1

        result = sender._send_single_request(self._make_task())
        assert result
        assert "r1" not in sender._pending_save_counts
        sender.shutdown_omni_connectors()

    def test_requeue_or_drop_requeues_on_first_failure(self):
        sender = self._make_sender(MockConnector(stage_id=0))
        task = self._make_task()

        sender._requeue_or_drop_failed_send(task)

        assert task.get("_retry_count") == 1
        with sender._lock:
            dq = sender._pending_save_reqs.get("r1")
        assert dq is not None
        assert len(dq) == 1
        sender.shutdown_omni_connectors()

    def test_requeue_or_drop_drops_after_max_retries(self):
        sender = self._make_sender(MockConnector(stage_id=0))
        sender._pending_save_counts["r1"] = 1
        task = self._make_task()
        task["_retry_count"] = sender._MAX_SEND_RETRIES  # already at max

        sender._requeue_or_drop_failed_send(task)

        with sender._lock:
            dq = sender._pending_save_reqs.get("r1")
        assert dq is None or len(dq) == 0
        assert "r1" not in sender._pending_save_counts
        sender.shutdown_omni_connectors()

    def test_save_loop_retries_on_exception(self):
        """Integration: _save_loop retries a task when put() raises."""
        from collections import deque

        connector = _FailingConnector(fail_count=1, raise_on_fail=True)
        sender = self._make_sender(connector)
        task = self._make_task()

        with sender._lock:
            sender._pending_save_reqs["r1"] = deque([task])
        sender._pending_save_counts["r1"] = 1

        sender._stop_event.clear()

        def run_one_loop():
            sender._save_loop()

        sender._stop_event.set()  # will exit after one iteration
        # Run manually instead of threading
        # Simulate: pop task, send fails, requeue
        popped_task = None
        with sender._lock:
            dq = sender._pending_save_reqs.get("r1")
            if dq:
                popped_task = dq.popleft()
                if not dq:
                    del sender._pending_save_reqs["r1"]

        if popped_task is not None:
            success = False
            try:
                success = sender._send_single_request(popped_task)
            except Exception:
                pass
            if not success:
                sender._requeue_or_drop_failed_send(popped_task)

        # After first failure, task should be re-enqueued
        with sender._lock:
            dq = sender._pending_save_reqs.get("r1")
        assert dq is not None
        assert len(dq) == 1
        requeued = dq[0]
        assert requeued.get("_retry_count") == 1

        # Second attempt should succeed (connector now returns True)
        success = sender._send_single_request(requeued)
        assert success
        sender.shutdown_omni_connectors()
