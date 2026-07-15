# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3-Omni stage-transfer packet helpers.

Flattens nested stage payloads into a single-level dict so that
``payload["a"]["b"]`` becomes ``payload["a.b"]`` with no nesting. Connectors
that do not support raw data fall back to the legacy single-dict put/get path.
"""

from __future__ import annotations

from typing import Any

import torch

PACKET_VERSION = 1
MODE_ASYNC_CHUNK = "async_chunk"
MODE_NON_ASYNC_FULL_PAYLOAD = "non_async_full_payload"
PAYLOAD_KIND_THINKER_TO_TALKER_FULL = "thinker_to_talker_full_payload"
PAYLOAD_KIND_TALKER_TO_CODE2WAV_FULL = "talker_to_code2wav_full_payload"

# Key separator used when flattening nested payload dicts into a single level.
_FLAT_SEP = "."

# Tensor field names for thinker→talker full payload (dot-separated paths).
_THINKER_TO_TALKER_TENSOR_PATHS: tuple[str, ...] = (
    "embed.prefill",
    "embed.tts_bos",
    "embed.tts_eos",
    "embed.tts_pad",
    "hidden_states.output",
)

_TALKER_TO_CODE2WAV_TENSOR_PATHS: tuple[str, ...] = ("codes.audio",)


def is_qwen3_flat_packet(data: Any) -> bool:
    """True when a connector result is a flattened Qwen3 packet."""
    return (
        isinstance(data, dict) and data.get("packet_version") == PACKET_VERSION and data.get("__qwen3_flat__") is True
    )


def should_use_qwen3_packet_path(
    *,
    async_chunk: bool,
    supports_raw_data: bool,
    model_arch: str | None,
    from_stage_id: int,
    to_stage_id: int,
    transfer_mode: str | None = None,
) -> bool:
    if not supports_raw_data:
        return False
    if model_arch != "Qwen3OmniMoeForConditionalGeneration":
        return False
    if (from_stage_id, to_stage_id) not in {(0, 1), (1, 2)}:
        return False

    mode = transfer_mode or MODE_ASYNC_CHUNK
    if mode == MODE_ASYNC_CHUNK:
        return bool(async_chunk)
    if mode == MODE_NON_ASYNC_FULL_PAYLOAD:
        return not bool(async_chunk)
    return False


def _get_nested(payload: dict[str, Any], path: str) -> Any:
    cur: Any = payload
    for part in path.split(_FLAT_SEP):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _resolve_payload_kind(from_stage_id: int, to_stage_id: int) -> str:
    """Return the payload kind for a stage edge."""
    if from_stage_id == 1 and to_stage_id == 2:
        return PAYLOAD_KIND_TALKER_TO_CODE2WAV_FULL
    return PAYLOAD_KIND_THINKER_TO_TALKER_FULL


def _flatten_payload(payload: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten a nested payload dict so ``payload["a"]["b"]`` -> ``["a.b"]``.

    Only ``dict`` values are recursed into; every other value (tensors,
    lists, scalars) is copied to the flat output under its dotted key. Empty
    dicts collapse away entirely (no leaf is emitted).
    """
    flat: dict[str, Any] = {}
    for key, value in payload.items():
        flat_key = f"{prefix}{_FLAT_SEP}{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten_payload(value, flat_key))
        else:
            flat[flat_key] = value
    return flat


def _unflatten_payload(flat: dict[str, Any]) -> dict[str, Any]:
    """Rebuild a nested payload dict from a flattened one (inverse of flatten)."""
    payload: dict[str, Any] = {}
    for flat_key, value in flat.items():
        parts = flat_key.split(_FLAT_SEP)
        cur = payload
        for part in parts[:-1]:
            nxt = cur.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                cur[part] = nxt
            cur = nxt
        cur[parts[-1]] = value
    return payload


def _coerce_tensor_payload_value(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return value.detach().contiguous()
    if isinstance(value, list):
        if not value:
            return None
        try:
            tensor = torch.as_tensor(value)
        except Exception:
            return None
        if tensor.numel() == 0:
            return None
        return tensor.detach().contiguous()
    return None


def payload_has_packet_tensors(payload: dict[str, Any]) -> bool:
    """True when the payload carries at least one non-empty packet tensor field."""
    for path in _THINKER_TO_TALKER_TENSOR_PATHS + _TALKER_TO_CODE2WAV_TENSOR_PATHS:
        value = _get_nested(payload, path)
        if _coerce_tensor_payload_value(value) is not None:
            return True
    return False


def flatten_qwen3_payload(
    payload: dict[str, Any],
    *,
    from_stage_id: int,
    to_stage_id: int,
) -> dict[str, Any]:
    """Flatten a Qwen3 payload into a single-level, self-describing packet.

    Every nested path ``a.b.c`` collapses to a single key ``"a.b.c"``. The
    result is tagged with a small versioned header so the receiver can detect
    it (``is_qwen3_flat_packet``) and rebuild the nested structure
    (``reconstruct_qwen3_full_payload``). It is transferred with a single
    connector put().
    """
    payload_kind = _resolve_payload_kind(from_stage_id, to_stage_id)
    tensor_paths = (
        _TALKER_TO_CODE2WAV_TENSOR_PATHS
        if payload_kind == PAYLOAD_KIND_TALKER_TO_CODE2WAV_FULL
        else _THINKER_TO_TALKER_TENSOR_PATHS
    )

    flat = _flatten_payload(payload)
    for key, value in list(flat.items()):
        if key in tensor_paths:
            # Known tensor-path fields (e.g. codes.audio) must arrive as
            # tensors for the downstream stage; coerce list payloads too.
            coerced = _coerce_tensor_payload_value(value)
            if coerced is not None:
                flat[key] = coerced
        elif isinstance(value, torch.Tensor):
            # Detach/contiguous-ify other tensor leaves so raw-data connectors
            # can transfer them without aliasing runtime state.
            flat[key] = value.detach().contiguous()

    return {
        "__qwen3_flat__": True,
        "packet_version": PACKET_VERSION,
        "payload_kind": payload_kind,
        "payload": flat,
    }


# Back-compat aliases for older call sites / tests.
should_use_thinker_to_talker_packet_path = should_use_qwen3_packet_path


def reconstruct_qwen3_full_payload(packet: dict[str, Any]) -> dict[str, Any]:
    """Rebuild the nested semantic payload dict from a flat packet."""
    if packet.get("packet_version") != PACKET_VERSION:
        raise ValueError(f"Unsupported packet_version: {packet.get('packet_version')!r}")
    payload_kind = packet.get("payload_kind")
    if payload_kind not in {
        PAYLOAD_KIND_THINKER_TO_TALKER_FULL,
        PAYLOAD_KIND_TALKER_TO_CODE2WAV_FULL,
    }:
        raise ValueError(f"Unsupported payload_kind: {payload_kind!r}")

    flat = packet.get("payload")
    if not isinstance(flat, dict):
        raise ValueError("Qwen3 flat packet missing 'payload' dict")

    payload = _unflatten_payload(flat)

    # Normalize the finish sentinel that was flattened from meta.finished.
    meta = payload.get("meta")
    if isinstance(meta, dict) and "finished" in meta:
        finished = meta["finished"]
        if not isinstance(finished, torch.Tensor):
            meta["finished"] = torch.tensor(bool(finished), dtype=torch.bool)

    if payload_kind == PAYLOAD_KIND_TALKER_TO_CODE2WAV_FULL:
        audio_codes = _get_nested(payload, "codes.audio")
        if isinstance(audio_codes, torch.Tensor):
            payload["code_predictor_codes"] = audio_codes.reshape(-1).tolist()
        elif isinstance(audio_codes, list):
            payload["code_predictor_codes"] = list(audio_codes)

    return payload
