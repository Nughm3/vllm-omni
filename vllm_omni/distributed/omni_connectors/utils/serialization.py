# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import threading
import time
from collections.abc import Buffer, Mapping
from dataclasses import asdict, is_dataclass
from typing import Any

import msgspec
import numpy as np
import torch
from msgspec import msgpack
from PIL import Image
from vllm.outputs import CompletionOutput, RequestOutput

PROFILE_ENV = os.getenv("SHM_PROFILE", "0")
PROFILE = PROFILE_ENV != "0"
_PID = os.getpid()

# Keys that identify a RequestOutput dict (for reconstruction)
_REQUEST_OUTPUT_KEYS = frozenset({"request_id", "prompt", "prompt_token_ids", "outputs", "finished"})

# Keys that identify a CompletionOutput dict (for reconstruction)
_COMPLETION_OUTPUT_KEYS = frozenset({"index", "text", "token_ids", "finish_reason"})

# Keys that identify an OmniRequestOutput dict (for reconstruction)
# OmniRequestOutput has 'final_output_type' which is unique, or can be identified by
# having 'finished' and ('images' or 'final_output_type')
_OMNI_REQUEST_OUTPUT_KEYS = frozenset({"finished", "final_output_type"})


class ScalarTensorPayload(msgspec.Struct, tag=True):
    """PyTorch tensor payload specialized to avoid overhead for scalar tensors."""

    dtype: str
    shape: list[int]
    value: float | int | bool  # tensor.item() returns bool/int/float depending on dtype


class TensorPayload(msgspec.Struct, tag=True):
    """PyTorch tensor payload."""

    dtype: str
    shape: list[int]
    data: memoryview


class NdarrayPayload(msgspec.Struct, tag=True):
    """NumPy ndarray payload."""

    dtype: str
    shape: list[int]
    data: memoryview


class ImagePayload(msgspec.Struct, tag=True):
    """PIL image payload."""

    mode: str
    shape: list[int]
    data: memoryview


class SlicePayload(msgspec.Struct, tag=True):
    """Slice object payload."""

    start: int | None
    stop: int | None
    step: int | None


class BytesPayload(msgspec.Struct, tag=True):
    """Bytes payload."""

    payload: bytes


# TODO: add type hint
class DictPayload(msgspec.Struct, tag=True):
    """Fallback payload for encoding arbitrary dict-shaped data."""

    # Need to wrap dict in a tagged Struct to enable tagged union optimizations
    payload: dict[str, Any]


OmniConnectorPayload = (
    ScalarTensorPayload | TensorPayload | NdarrayPayload | ImagePayload | BytesPayload | SlicePayload | DictPayload
)


class OmniMsgpackEncoder:
    """
    This implementation is adapted from vLLM’s MsgpackEncoder.
    Handles torch.Tensor, numpy.ndarray, PIL.Image, RequestOutput and
    CompletionOutput by converting them to serializable dict representations.

    Encoding is zero-copy for the caller: encode() returns a memoryview into a
    thread-local bytearray that is reused across calls, avoiding a Python bytes
    allocation per encode.  Callers that need to persist the result beyond the
    next encode() call on the same thread must copy (bytes(view)).
    """

    # Initial per-thread encode buffer size; grows automatically via encode_into.
    _INITIAL_BUF = 65536

    def __init__(self):
        self._local = threading.local()

    def _get_encoder_and_buf(self) -> tuple[msgpack.Encoder, bytearray]:
        """Return (or lazily create) the thread-local encoder + write buffer."""
        loc = self._local
        if not hasattr(loc, "encoder"):
            loc.encoder = msgpack.Encoder(enc_hook=self._enc_hook)
            loc.buf = bytearray(self._INITIAL_BUF)
        return loc.encoder, loc.buf

    def encode(self, obj: Any) -> memoryview:
        """Encode an object.

        Returns a memoryview into the thread-local buffer — zero allocation.
        encode_into() truncates buf to the encoded length, so memoryview(buf)
        is exactly the serialized bytes.  The view is valid until the next
        encode() call on the same thread.
        """
        encoder, buf = self._get_encoder_and_buf()
        # Native msgspec types bypass enc_hook and produce untagged roots.
        # The typed decoder requires a tagged root.
        #
        # - Plain dict: wrap directly.
        # - Untagged msgspec.Struct (e.g. OmniPayloadStruct, _StructBase subclasses):
        #   extract fields into a plain dict preserving the original Python values,
        #   then wrap.  Field values that are tensors/ndarrays will go through
        #   enc_hook when msgspec encodes the DictPayload, so no special handling
        #   is needed.  We cannot use msgspec.to_builtins() here because it cannot
        #   convert torch.Tensor or numpy.ndarray.
        if isinstance(obj, dict):
            obj = DictPayload(obj)
        elif isinstance(obj, msgspec.Struct) and not isinstance(obj, OmniConnectorPayload.__args__):
            fields_dict = {fi.name: getattr(obj, fi.name) for fi in msgspec.structs.fields(obj)}
            obj = DictPayload(fields_dict)
        encoder.encode_into(obj, buf)
        return memoryview(buf)

    def _enc_hook(self, obj: Any) -> OmniConnectorPayload:
        """Custom encoding hook for non-standard types."""
        # torch.Tensor — single-element tensors skip the heavy encode path
        if isinstance(obj, torch.Tensor):
            if obj.numel() == 1:
                return self._encode_scalar_tensor(obj)
            return self._encode_tensor(obj)

        # numpy.ndarray (exclude object/void dtypes)
        if isinstance(obj, np.ndarray) and obj.dtype.kind not in ("O", "V"):
            return self._encode_ndarray(obj)

        # PIL.Image
        if isinstance(obj, Image.Image):
            return self._encode_pil_image(obj)

        # byte-like objects — modern msgspec encodes memoryview/bytearray natively
        # as msgpack bin without reaching this hook; this is a compatibility fallback.
        if isinstance(obj, (memoryview, bytearray)):
            return BytesPayload(bytes(obj))

        # slice
        if isinstance(obj, slice):
            return SlicePayload(start=obj.start, stop=obj.stop, step=obj.step)

        # RequestOutput (not a dataclass, needs special handling)
        if isinstance(obj, RequestOutput):
            return self._encode_request_output(obj)

        # CompletionOutput (dataclass)
        if isinstance(obj, CompletionOutput):
            return self._encode_completion_output(obj)

        # Other dataclasses
        if is_dataclass(obj) and not isinstance(obj, type):
            return DictPayload(asdict(obj))

        raise TypeError(
            f"Object of type {type(obj).__name__} is not serializable. "
            "Supported types: torch.Tensor, np.ndarray, PIL.Image, dataclass, "
            "RequestOutput, and standard Python types (dict, list, str, int, float, bool, None, bytes)."
        )

    def _encode_scalar_tensor(self, tensor: torch.Tensor) -> ScalarTensorPayload:
        """Encode a single-element tensor as a Python scalar — avoids detach/cpu/view/numpy overhead."""
        tensor = tensor.detach().cpu()
        return ScalarTensorPayload(
            dtype=str(tensor.dtype).removeprefix("torch."), shape=list(tensor.shape), value=tensor.item()
        )

    def _encode_tensor(self, tensor: torch.Tensor) -> TensorPayload:
        """Encode torch.Tensor to dict."""
        called_reshape = False
        called_contiguous = False

        start = time.perf_counter()

        t = tensor.detach()

        # Perform contiguous on GPU if possible
        if not t.is_contiguous():
            t = t.contiguous()
            called_contiguous = True

        t = t.cpu()

        # Handle 0-dimensional (scalar) tensors by reshaping to 1D first
        if t.dim() == 0:
            called_reshape = True
            t = t.reshape(1)

        t = t.view(torch.uint8)
        data = memoryview(t.numpy())

        end = time.perf_counter()

        if PROFILE:
            print(
                "SHM_PROFILE",
                json.dumps(
                    {
                        "name": "_encode_tensor",
                        "ph": "X",
                        "ts": start * 1_000_000,
                        "dur": (end - start) * 1_000_000,
                        "pid": _PID,
                        "tid": threading.get_ident(),
                        "args": {
                            "dtype": str(tensor.dtype).removeprefix("torch."),
                            "shape": list(tensor.shape),
                            "nbytes": tensor.nbytes,
                            "device": str(tensor.device),
                            "called_reshape": called_reshape,
                            "called_contiguous": called_contiguous,
                        },
                    }
                ),
            )

        return TensorPayload(
            dtype=str(tensor.dtype).removeprefix("torch."),
            shape=list(tensor.shape),
            data=data,
        )

    def _encode_ndarray(self, arr: np.ndarray) -> NdarrayPayload:
        """Encode numpy.ndarray to dict."""
        called_contiguous = False

        start = time.perf_counter()

        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)
            called_contiguous = True

        data = memoryview(arr)

        end = time.perf_counter()
        if PROFILE:
            print(
                "SHM_PROFILE",
                json.dumps(
                    {
                        "name": "_encode_ndarray",
                        "ph": "X",
                        "ts": start * 1_000_000,
                        "dur": (end - start) * 1_000_000,
                        "pid": _PID,
                        "tid": threading.get_ident(),
                        "args": {
                            "dtype": arr.dtype.str,
                            "shape": list(arr.shape),
                            "nbytes": arr.nbytes,
                            "called_contiguous": called_contiguous,
                        },
                    }
                ),
            )

        return NdarrayPayload(
            dtype=arr.dtype.str,
            shape=list(arr.shape),
            data=data,
        )

    def _encode_pil_image(self, img: Image.Image) -> ImagePayload:
        """Encode PIL.Image to dict."""
        called_contiguous = False

        start = time.perf_counter()

        arr = np.asarray(img, dtype=np.uint8)
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)
            called_contiguous = True

        data = memoryview(arr)

        end = time.perf_counter()

        if PROFILE:
            print(
                "SHM_PROFILE",
                json.dumps(
                    {
                        "name": "_encode_pil_image",
                        "ph": "X",
                        "ts": start * 1_000_000,
                        "dur": (end - start) * 1_000_000,
                        "pid": _PID,
                        "tid": threading.get_ident(),
                        "args": {
                            "mode": img.mode,
                            "shape": list(arr.shape),
                            "nbytes": arr.nbytes,
                            "called_contiguous": called_contiguous,
                        },
                    }
                ),
            )

        return ImagePayload(
            mode=img.mode,
            shape=list(arr.shape),
            data=data,
        )

    def _encode_request_output(self, obj: RequestOutput) -> DictPayload:
        """Encode RequestOutput to dict.

        RequestOutput is not a dataclass, so we manually extract its attributes.
        Also handles dynamically added 'multimodal_output' attribute.
        """
        start = time.perf_counter()
        # msgspec can serialize CompletionOutput dataclasses directly, but it
        # drops dynamic fields such as multimodal_output. Encode them manually
        # to preserve multimodal payloads across IPC.
        encoded_outputs = []
        for o in obj.outputs:
            if isinstance(o, CompletionOutput):
                encoded_outputs.append(self._encode_completion_output(o))
            else:
                encoded_outputs.append(o)

        result = {
            "request_id": obj.request_id,
            "prompt": obj.prompt,
            "prompt_token_ids": obj.prompt_token_ids,
            "prompt_logprobs": obj.prompt_logprobs,
            "outputs": encoded_outputs,
            "finished": obj.finished,
            "metrics": obj.metrics,
            "lora_request": obj.lora_request,
            "encoder_prompt": obj.encoder_prompt,
            "encoder_prompt_token_ids": obj.encoder_prompt_token_ids,
            "num_cached_tokens": obj.num_cached_tokens,
            "multi_modal_placeholders": getattr(obj, "multi_modal_placeholders", None),
            "kv_transfer_params": obj.kv_transfer_params,
        }
        # Handle multimodal_output attribute (MultimodalPayload or dict)
        mm_output = getattr(obj, "multimodal_output", None)
        if mm_output is not None:
            if isinstance(mm_output, Mapping):
                result["multimodal_output"] = DictPayload(dict(mm_output))
            else:
                result["multimodal_output"] = mm_output

        end = time.perf_counter()
        if PROFILE:
            print(
                "SHM_PROFILE",
                json.dumps(
                    {
                        "name": "_encode_request_output",
                        "ph": "X",
                        "ts": start * 1_000_000,
                        "dur": (end - start) * 1_000_000,
                        "pid": _PID,
                        "tid": threading.get_ident(),
                        "args": {
                            "request_id": obj.request_id,
                            "num_outputs": len(obj.outputs),
                            "has_multimodal_output": mm_output is not None,
                        },
                    }
                ),
            )

        return DictPayload(result)

    def _encode_completion_output(self, obj: CompletionOutput) -> DictPayload:
        """Encode CompletionOutput to dict, preserving multimodal payloads."""
        start = time.perf_counter()

        result = asdict(obj)
        mm_output = getattr(obj, "multimodal_output", None)
        if mm_output is not None:
            # Convert MultimodalPayload to plain dict for wire format
            if isinstance(mm_output, Mapping):
                result["multimodal_output"] = DictPayload(dict(mm_output))
            else:
                result["multimodal_output"] = mm_output

        end = time.perf_counter()
        if PROFILE:
            print(
                "SHM_PROFILE",
                json.dumps(
                    {
                        "name": "_encode_completion_output",
                        "ph": "X",
                        "ts": start * 1_000_000,
                        "dur": (end - start) * 1_000_000,
                        "pid": _PID,
                        "tid": threading.get_ident(),
                        "args": {
                            "has_multimodal_output": mm_output is not None,
                        },
                    }
                ),
            )

        return DictPayload(result)


class OmniMsgpackDecoder:
    """
    This implementation is adapted from vLLM’s MsgpackDecoder.
    Automatically reconstructs torch.Tensor, numpy.ndarray, PIL.Image,
    RequestOutput and CompletionOutput from their dict representations.

    decode() accepts bytes, bytearray, or memoryview — pass a memoryview of the
    SHM buffer directly to avoid copying the full payload onto the Python heap.
    Binary fields (tensor/ndarray data) are copied into bytes by msgspec during
    decoding, so the input buffer can be released immediately after decode().
    """

    def __init__(self):
        # Typed decoder: the root is always an OmniConnectorPayload (the encoder
        # wraps plain dicts in DictPayload).  Binary fields (TensorPayload.data
        # etc.) are decoded as memoryview objects that ALIAS the input buffer.
        # When decoding from SHM, the connector must not munmap the SHM segment
        # while objects derived from the payload (tensors, arrays) are still alive.
        # See SharedMemoryConnector._get_data_with_lock for the lifetime contract.
        self.decoder = msgpack.Decoder(OmniConnectorPayload)

    def decode(self, data: Buffer) -> Any:
        """Decode bytes to object."""
        try:
            return self._post_process(self.decoder.decode(data))
        except msgspec.DecodeError:
            # Root is not a tagged OmniConnectorPayload — most likely a bare
            # msgspec.Struct (e.g. OmniPayloadStruct) which msgspec encodes
            # natively as an untagged map without calling enc_hook.  Fall back
            # to a generic decode and let _post_process handle the plain dict.
            return self._post_process(msgpack.decode(data))

    def _post_process(self, obj: Any) -> Any:
        """Recursively restore typed objects from their wire representations."""
        # Root-level Struct instances from the typed decoder — direct dispatch.
        if isinstance(obj, ScalarTensorPayload):
            return self._decode_scalar_tensor(obj)
        if isinstance(obj, TensorPayload):
            return self._decode_tensor(obj)
        if isinstance(obj, NdarrayPayload):
            return self._decode_ndarray(obj)
        if isinstance(obj, ImagePayload):
            return self._decode_pil_image(obj)
        if isinstance(obj, SlicePayload):
            return slice(obj.start, obj.stop, obj.step)
        if isinstance(obj, BytesPayload):
            return obj.payload
        if isinstance(obj, DictPayload):
            return self._post_process_dict(obj.payload)

        # Nested raw values from DictPayload.payload (typed as Any by msgspec).
        if isinstance(obj, dict):
            return self._post_process_dict(obj)
        if isinstance(obj, list):
            return [self._post_process(item) for item in obj]
        if isinstance(obj, tuple):
            return tuple(self._post_process(item) for item in obj)
        return obj

    def _post_process_dict(self, d: dict) -> Any:
        """Process the values of a plain dict and reconstruct typed objects."""
        processed = {k: self._convert_value(v) for k, v in d.items()}

        # Heuristic reconstruction of complex types not represented as tagged structs.
        if self._is_omni_request_output(processed):
            return self._decode_omni_request_output(processed)
        if _REQUEST_OUTPUT_KEYS.issubset(processed.keys()):
            return self._decode_request_output(processed)
        if _COMPLETION_OUTPUT_KEYS.issubset(processed.keys()):
            return self._decode_completion_output(processed)
        return processed

    def _convert_value(self, v: Any) -> Any:
        """Convert a single value from DictPayload.payload.

        Uses msgspec.convert for tagged dicts so nested payloads (tensors,
        images, nested DictPayloads, etc.) are reconstructed with type
        validation rather than heuristic string matching.
        """
        if isinstance(v, dict):
            if "type" in v:
                try:
                    return self._post_process(msgspec.convert(v, OmniConnectorPayload, strict=False))
                except msgspec.ValidationError:
                    pass
            return self._post_process_dict(v)
        if isinstance(v, list):
            return [self._convert_value(item) for item in v]
        return v

    # --- Struct-typed decode methods (primary path with typed decoder) ---

    def _decode_scalar_tensor(self, payload: ScalarTensorPayload) -> torch.Tensor:
        """Decode a scalar-encoded tensor back to a torch.Tensor."""
        return torch.tensor(payload.value, dtype=getattr(torch, payload.dtype)).reshape(payload.shape)

    def _decode_tensor(self, payload: TensorPayload) -> torch.Tensor:
        """Decode dict to torch.Tensor."""
        start = time.perf_counter()

        torch_dtype = getattr(torch, payload.dtype)
        if not payload.data:
            result = torch.empty(payload.shape, dtype=torch_dtype)
        else:
            arr = torch.frombuffer(payload.data, dtype=torch.uint8)
            result = arr.view(torch_dtype).reshape(payload.shape)

        end = time.perf_counter()
        if PROFILE:
            print(
                "SHM_PROFILE",
                json.dumps(
                    {
                        "name": "_decode_tensor",
                        "ph": "X",
                        "ts": start * 1_000_000,
                        "dur": (end - start) * 1_000_000,
                        "pid": _PID,
                        "tid": threading.get_ident(),
                        "args": {
                            "dtype": payload.dtype,
                            "shape": payload.shape,
                            "nbytes": len(payload.data) if payload.data else 0,
                        },
                    }
                ),
            )

        return result

    def _decode_ndarray(self, payload: NdarrayPayload) -> np.ndarray:
        """Decode dict to numpy.ndarray."""
        start = time.perf_counter()

        result = np.frombuffer(payload.data, dtype=payload.dtype).reshape(payload.shape)

        end = time.perf_counter()
        if PROFILE:
            print(
                "SHM_PROFILE",
                json.dumps(
                    {
                        "name": "_decode_ndarray",
                        "ph": "X",
                        "ts": start * 1_000_000,
                        "dur": (end - start) * 1_000_000,
                        "pid": _PID,
                        "tid": threading.get_ident(),
                        "args": {
                            "dtype": payload.dtype,
                            "shape": payload.shape,
                            "nbytes": len(payload.data) if payload.data else 0,
                        },
                    }
                ),
            )

        return result

    def _decode_pil_image(self, payload: ImagePayload) -> Image.Image:
        """Decode dict to PIL.Image."""
        start = time.perf_counter()

        arr = np.frombuffer(payload.data, dtype=np.uint8).reshape(payload.shape)
        result = Image.fromarray(arr, mode=payload.mode)

        end = time.perf_counter()
        if PROFILE:
            print(
                "SHM_PROFILE",
                json.dumps(
                    {
                        "name": "_decode_pil_image",
                        "ph": "X",
                        "ts": start * 1_000_000,
                        "dur": (end - start) * 1_000_000,
                        "pid": _PID,
                        "tid": threading.get_ident(),
                        "args": {
                            "mode": payload.mode,
                            "shape": payload.shape,
                            "nbytes": len(payload.data) if payload.data else 0,
                        },
                    }
                ),
            )

        return result

    def _decode_completion_output(self, obj: dict[str, Any]) -> CompletionOutput:
        """Decode dict to CompletionOutput using msgspec.convert."""
        mm_output = obj.pop("multimodal_output", None)
        start = time.perf_counter()
        co = msgspec.convert(obj, CompletionOutput)
        if mm_output is not None:
            setattr(co, "multimodal_output", mm_output)
        end = time.perf_counter()
        if PROFILE:
            print(
                "SHM_PROFILE",
                json.dumps(
                    {
                        "name": "_decode_completion_output",
                        "ph": "X",
                        "ts": start * 1_000_000,
                        "dur": (end - start) * 1_000_000,
                        "pid": _PID,
                        "tid": threading.get_ident(),
                        "args": {
                            "has_multimodal_output": mm_output is not None,
                        },
                    }
                ),
            )
        return co

    def _decode_request_output(self, obj: dict[str, Any]) -> RequestOutput:
        """Decode dict to RequestOutput.

        RequestOutput is not a dataclass, so msgspec.convert doesn't work.
        We construct it manually using only the known __init__ parameters to
        avoid triggering the "Ignoring extra arguments" warning in vllm.
        Fields that are not part of RequestOutput.__init__ (e.g.
        multi_modal_placeholders, multimodal_output) are extracted first and
        then restored as dynamic attributes after construction.
        """
        # Extract dynamically-added / non-init fields before constructing so
        # they are not passed as unknown **kwargs to RequestOutput.__init__.
        mm_output = obj.pop("multimodal_output", None)
        multi_modal_placeholders = obj.pop("multi_modal_placeholders", None)

        start = time.perf_counter()
        ro = RequestOutput(**obj)

        # Restore dynamic attributes that are not part of __init__.
        if multi_modal_placeholders is not None:
            setattr(ro, "multi_modal_placeholders", multi_modal_placeholders)
        if mm_output is not None:
            setattr(ro, "multimodal_output", mm_output)
        end = time.perf_counter()
        if PROFILE:
            print(
                "SHM_PROFILE",
                json.dumps(
                    {
                        "name": "_decode_request_output",
                        "ph": "X",
                        "ts": start * 1_000_000,
                        "dur": (end - start) * 1_000_000,
                        "pid": _PID,
                        "tid": threading.get_ident(),
                        "args": {
                            "request_id": obj.get("request_id"),
                            "num_outputs": len(obj.get("outputs", [])),
                            "has_multimodal_output": mm_output is not None,
                        },
                    }
                ),
            )
        return ro

    def _is_omni_request_output(self, obj: dict[str, Any]) -> bool:
        """Check if a dict looks like an OmniRequestOutput.

        OmniRequestOutput can be identified by:
        - Having 'finished' and 'final_output_type' fields (unique to OmniRequestOutput)
        - OR having 'finished' and 'images' fields (diffusion mode)
        """
        # Must have 'finished' field
        if "finished" not in obj:
            return False

        # Check for unique identifier: 'final_output_type'
        if "final_output_type" in obj:
            return True

        # Alternative: check for 'images' field (diffusion mode)
        if "images" in obj:
            return True

        return False

    def _decode_omni_request_output(self, obj: dict[str, Any]) -> Any:
        """Decode dict to OmniRequestOutput.

        OmniRequestOutput is a dataclass, so we can use msgspec.convert
        or construct it directly.
        """
        from vllm_omni.outputs import OmniRequestOutput

        start = time.perf_counter()
        try:
            # The dict contains already-reconstructed Python objects (tensors,
            # ndarrays, etc.) so msgspec.convert won't work here — go straight
            # to direct construction.
            result = OmniRequestOutput(**obj)
        except Exception:
            # If construction fails, return dict as-is (defensive fallback)
            result = obj
        end = time.perf_counter()
        if PROFILE:
            print(
                "SHM_PROFILE",
                json.dumps(
                    {
                        "name": "_decode_omni_request_output",
                        "ph": "X",
                        "ts": start * 1_000_000,
                        "dur": (end - start) * 1_000_000,
                        "pid": _PID,
                        "tid": threading.get_ident(),
                        "args": {
                            "finished": obj.get("finished"),
                            "final_output_type": obj.get("final_output_type"),
                        },
                    }
                ),
            )
        return result


class OmniSerde:
    """Serialization/deserialization handler for Omni IPC."""

    def __init__(self):
        self.encoder = OmniMsgpackEncoder()
        self.decoder = OmniMsgpackDecoder()

    def serialize(self, obj: Any) -> bytes:
        """Serialize an object.

        This method allocates a bytes object from the buffer that the serializer
        writes to, making it safe for general use.
        """
        return bytes(self.encoder.encode(obj))

    def serialize_view(self, obj: Any) -> memoryview:
        """Serialize an object into a memoryview.

        Returns a memoryview into a thread-local buffer — zero allocation.
        Valid until the next serialize operation on the same thread.

        If you need the object beyond this lifetime, use the serialize() method
        instead.
        """
        return self.encoder.encode(obj)

    def deserialize(self, data: Buffer) -> Any:
        """Deserialize bytes to an object."""
        return self.decoder.decode(data)


# Global instance for simple interface
OmniSerializer = OmniSerde()
