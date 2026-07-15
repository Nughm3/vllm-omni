# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import torch

from vllm_omni.distributed.omni_connectors.utils import qwen3_transfer_packet as pkt
from vllm_omni.model_executor.stage_input_processors import qwen3_omni as q3


def _sample_thinker_payload() -> dict:
    return {
        "embed": {
            "prefill": torch.ones(2, 4),
            "tts_bos": torch.zeros(1, 4),
        },
        "hidden_states": {"output": torch.full((2, 4), 2.0)},
        "ids": {"all": [1, 2, 3], "prompt": [1, 2]},
        "meta": {"finished": torch.tensor(True, dtype=torch.bool)},
        "next_stage_prompt_len": 7,
        "speaker": "alice",
    }


def test_flatten_and_reconstruct_roundtrip() -> None:
    payload = _sample_thinker_payload()
    packet = pkt.flatten_qwen3_payload(
        payload,
        from_stage_id=0,
        to_stage_id=1,
    )
    assert pkt.is_qwen3_flat_packet(packet)
    assert packet["payload_kind"] == pkt.PAYLOAD_KIND_THINKER_TO_TALKER_FULL
    # Payload is flattened into a single-level dict: nested a.b -> "a.b".
    flat = packet["payload"]
    assert "embed.prefill" in flat
    assert "embed.tts_bos" in flat
    assert "hidden_states.output" in flat
    assert "ids.all" in flat
    assert "meta.finished" in flat
    # No nesting remains in the flat payload.
    assert not any(isinstance(v, dict) for v in flat.values())

    rebuilt = pkt.reconstruct_qwen3_full_payload(packet)
    assert rebuilt["ids"] == payload["ids"]
    assert rebuilt["next_stage_prompt_len"] == 7
    assert rebuilt["speaker"] == "alice"
    assert torch.equal(rebuilt["embed"]["prefill"], payload["embed"]["prefill"])
    assert torch.equal(rebuilt["hidden_states"]["output"], payload["hidden_states"]["output"])
    assert bool(rebuilt["meta"]["finished"].item()) is True


def test_flatten_unflatten_roundtrip() -> None:
    payload = _sample_thinker_payload()
    flat = pkt._flatten_payload(payload)
    # Every leaf reachable via a dotted key, no nested dicts.
    assert flat["ids.all"] == [1, 2, 3]
    assert flat["ids.prompt"] == [1, 2]
    assert flat["next_stage_prompt_len"] == 7
    assert not any(isinstance(v, dict) for v in flat.values())

    rebuilt = pkt._unflatten_payload(flat)
    assert rebuilt["ids"]["all"] == [1, 2, 3]
    assert torch.equal(rebuilt["embed"]["prefill"], payload["embed"]["prefill"])
    assert rebuilt["speaker"] == "alice"


def test_should_use_packet_path_gate() -> None:
    assert pkt.should_use_thinker_to_talker_packet_path(
        async_chunk=True,
        supports_raw_data=True,
        model_arch="Qwen3OmniMoeForConditionalGeneration",
        from_stage_id=0,
        to_stage_id=1,
    )
    assert not pkt.should_use_thinker_to_talker_packet_path(
        async_chunk=False,
        supports_raw_data=True,
        model_arch="Qwen3OmniMoeForConditionalGeneration",
        from_stage_id=0,
        to_stage_id=1,
    )
    assert pkt.should_use_thinker_to_talker_packet_path(
        async_chunk=False,
        supports_raw_data=True,
        model_arch="Qwen3OmniMoeForConditionalGeneration",
        from_stage_id=0,
        to_stage_id=1,
        transfer_mode=pkt.MODE_NON_ASYNC_FULL_PAYLOAD,
    )
    assert not pkt.should_use_thinker_to_talker_packet_path(
        async_chunk=True,
        supports_raw_data=False,
        model_arch="Qwen3OmniMoeForConditionalGeneration",
        from_stage_id=0,
        to_stage_id=1,
    )
    assert not pkt.should_use_thinker_to_talker_packet_path(
        async_chunk=True,
        supports_raw_data=True,
        model_arch="Qwen3OmniMoeForConditionalGeneration",
        from_stage_id=0,
        to_stage_id=2,
    )
    assert pkt.should_use_thinker_to_talker_packet_path(
        async_chunk=False,
        supports_raw_data=True,
        model_arch="Qwen3OmniMoeForConditionalGeneration",
        from_stage_id=1,
        to_stage_id=2,
        transfer_mode=pkt.MODE_NON_ASYNC_FULL_PAYLOAD,
    )


def test_payload_has_packet_tensors() -> None:
    assert pkt.payload_has_packet_tensors(_sample_thinker_payload())
    assert pkt.payload_has_packet_tensors({"codes": {"audio": [1, 2, 3]}})
    assert not pkt.payload_has_packet_tensors({"ids": {"all": [1]}, "meta": {"finished": True}})


def test_talker_to_code2wav_split_and_reconstruct_roundtrip() -> None:
    payload = {
        "codes": {"audio": [1, 2, 3, 4]},
        "meta": {"finished": torch.tensor(True, dtype=torch.bool), "left_context_size": 25},
    }
    packet = pkt.flatten_qwen3_payload(
        payload,
        from_stage_id=1,
        to_stage_id=2,
    )
    assert packet["payload_kind"] == pkt.PAYLOAD_KIND_TALKER_TO_CODE2WAV_FULL
    # Known tensor-path fields are coerced to tensors even from list input.
    assert torch.is_tensor(packet["payload"]["codes.audio"])
    assert packet["payload"]["codes.audio"].tolist() == [1, 2, 3, 4]
    rebuilt = pkt.reconstruct_qwen3_full_payload(packet)
    assert "codes" in rebuilt and "audio" in rebuilt["codes"]
    assert rebuilt["code_predictor_codes"] == [1, 2, 3, 4]


def test_thinker2talker_full_payload_processor_roundtrip() -> None:
    request = SimpleNamespace(
        request_id="thinker",
        prompt_token_ids=[151644, 872],
        output_token_ids=[3],
        all_token_ids=[151644, 872, 3],
    )
    pooling_output = {
        "hidden_states.layer_0": torch.ones(3, 2),
        "hidden_states.layer_24": torch.full((3, 2), 2.0),
        "embed.tts_bos": torch.zeros(1, 2),
    }
    payload = q3.thinker2talker_full_payload(None, pooling_output, request)
    assert payload is not None

    packet = pkt.flatten_qwen3_payload(
        payload,
        from_stage_id=0,
        to_stage_id=1,
    )
    rebuilt = pkt.reconstruct_qwen3_full_payload(packet)
    assert rebuilt["ids"]["all"] == payload["ids"]["all"]
    assert rebuilt["embed"]["prefill"].shape == payload["embed"]["prefill"].shape
    assert rebuilt["hidden_states"]["output"].shape == payload["hidden_states"]["output"].shape
    if "next_stage_prompt_len" in payload:
        assert rebuilt["next_stage_prompt_len"] == payload["next_stage_prompt_len"]
