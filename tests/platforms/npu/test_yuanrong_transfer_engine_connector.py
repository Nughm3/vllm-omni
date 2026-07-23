# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for YuanrongTransferEngineConnector role handling.

Loaded directly from source (bypassing ``vllm_omni.platforms.npu``'s package
``__init__``, which requires ``vllm_ascend``) so these run without NPU deps.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _repo_root() -> Path:
    marker = Path("vllm_omni") / "platforms" / "npu" / "omni_connectors"
    for parent in Path(__file__).resolve().parents:
        if (parent / marker).is_dir():
            return parent
    raise FileNotFoundError(f"could not locate repo root containing {marker}")


def _load_yuanrong_module():
    path = (
        _repo_root() / "vllm_omni" / "platforms" / "npu" / "omni_connectors" / "yuanrong_transfer_engine_connector.py"
    )
    spec = importlib.util.spec_from_file_location("vllm_omni_test_yuanrong_transfer_engine_connector", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeResult:
    def is_error(self) -> bool:
        return False

    def to_string(self) -> str:
        return ""


class _FakePool:
    def data_ptr(self) -> int:
        return 1234


class _FakeTransferEngine:
    def initialize(self, endpoint, protocol, device_name):
        return _FakeResult()

    def get_rpc_port(self) -> int:
        return 23456

    def register_memory(self, base_ptr, pool_size):
        return _FakeResult()


def _patch_fake_yuanrong(monkeypatch: pytest.MonkeyPatch):
    """Monkeypatch the Yuanrong engine/pool so the connector builds without real NPU hardware."""
    yuanrong_module = _load_yuanrong_module()

    monkeypatch.setattr(yuanrong_module, "TransferEngine", _FakeTransferEngine)
    monkeypatch.setattr(yuanrong_module.torch, "empty", lambda *args, **kwargs: _FakePool())
    monkeypatch.setattr(
        yuanrong_module.YuanrongTransferEngineConnector,
        "_zmq_listener_loop",
        lambda self: self._listener_ready.set(),
    )
    return yuanrong_module


def test_yuanrong_transfer_engine_connector_dual_role_binds_and_gets(monkeypatch: pytest.MonkeyPatch):
    """role='dual' binds the outbound listener (can_put) and also allows
    get() from upstream on a single instance — same contract as the
    Mooncake/Mori transfer-engine connectors."""
    yuanrong_module = _patch_fake_yuanrong(monkeypatch)

    connector = yuanrong_module.YuanrongTransferEngineConnector(
        {
            "host": "10.20.30.40",
            "zmq_port": 50052,
            "rpc_port": 50053,
            "memory_pool_size": 4096,
            "role": "dual",
            "sender_host": "10.20.30.41",
            "sender_zmq_port": 50051,
        }
    )
    try:
        assert connector.can_put is True
        assert connector.sender_host == "10.20.30.41"
        assert connector.sender_zmq_port == 50051
    finally:
        connector.close()


def test_yuanrong_transfer_engine_connector_receiver_role_skips_bind(monkeypatch: pytest.MonkeyPatch):
    yuanrong_module = _patch_fake_yuanrong(monkeypatch)

    connector = yuanrong_module.YuanrongTransferEngineConnector(
        {
            "host": "10.20.30.40",
            "zmq_port": 50052,
            "rpc_port": 50053,
            "memory_pool_size": 4096,
            "role": "receiver",
            "sender_host": "10.20.30.41",
            "sender_zmq_port": 50051,
        }
    )
    try:
        assert connector.can_put is False
    finally:
        connector.close()


def test_yuanrong_transfer_engine_connector_rejects_unknown_role(monkeypatch: pytest.MonkeyPatch):
    yuanrong_module = _patch_fake_yuanrong(monkeypatch)
    with pytest.raises(ValueError, match="Invalid role"):
        yuanrong_module.YuanrongTransferEngineConnector(
            {
                "host": "10.20.30.40",
                "zmq_port": 50052,
                "rpc_port": 50053,
                "memory_pool_size": 4096,
                "role": "bogus",
            }
        )
