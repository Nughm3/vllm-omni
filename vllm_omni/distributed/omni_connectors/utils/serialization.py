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

# Type markers for custom serialization
_TENSOR_MARKER = "__tensor__"
_NDARRAY_MARKER = "__ndarray__"
_PIL_IMAGE_MARKER = "__pil_image__"

# Keys that identify a RequestOutput dict (for reconstruction)
_REQUEST_OUTPUT_KEYS = frozenset({"request_id", "prompt", "prompt_token_ids", "outputs", "finished"})

# Keys that identify a CompletionOutput dict (for reconstruction)
_COMPLETION_OUTPUT_KEYS = frozenset({"index", "text", "token_ids", "finish_reason"})


class OmniMsgpackEncoder:
    """
    This implementation is adapted from vLLM’s MsgpackEncoder.
    Handles torch.Tensor, numpy.ndarray, PIL.Image, RequestOutput and
    CompletionOutput by converting them to serializable dict representations.

    The encoder instance itself is shared; buffer ownership is left to the
    caller.  Use encode_into() to write into a caller-owned buffer (e.g. a
    connector’s thread-local bytearray) for zero-copy paths, or encode() for
    a one-shot bytearray when convenience matters more than allocation count.
    """

    def __init__(self):
        self.encoder = msgpack.Encoder(enc_hook=self._enc_hook)

    def encode(self, obj: Any) -> bytes:
        """Encode obj and return a new bytes object containing the result.

        Allocates a new bytes object. For a hot path use encode_into() with a
        pre-allocated caller-owned buffer instead.
        """
        return self.encoder.encode(obj)

    def encode_into(self, obj: Any, buf: bytearray) -> None:
        """Encode obj into buf in-place.

        buf is grown automatically if the encoded output exceeds its current
        length, and truncated to the encoded length on success.  The caller
        owns buf and controls its lifetime; a memoryview(buf) taken after this
        call is valid as long as no further encode_into call on the same buf
        occurs (which would resize it and invalidate open exports).

        Thread safety: encode_into() must not be called concurrently on the
        same OmniMsgpackEncoder instance from multiple threads — the underlying
        msgpack.Encoder may have internal scratch state that is not protected by
        the GIL when encoding large binary payloads.  encode() is safe to call
        concurrently because each call allocates its own output buffer.  The
        caller is responsible for ensuring at most one thread calls encode_into()
        on a given instance at a time; SharedMemoryConnector upholds this by
        holding one OmniSerde per connector instance and calling put() from a
        single thread.
        """
        self.encoder.encode_into(obj, buf)

    def _enc_hook(self, obj: Any) -> Any:
        """Custom encoding hook for non-standard types."""
        # torch.Tensor — single-element tensors skip the heavy encode path
        if isinstance(obj, torch.Tensor):
            return self._encode_tensor(obj)

        # numpy.ndarray (exclude object/void dtypes)
        if isinstance(obj, np.ndarray) and obj.dtype.kind not in ("O", "V"):
            return self._encode_ndarray(obj)

        # PIL.Image
        if isinstance(obj, Image.Image):
            return self._encode_pil_image(obj)

        # RequestOutput (not a dataclass, needs special handling)
        if isinstance(obj, RequestOutput):
            return self._encode_request_output(obj)

        # CompletionOutput (dataclass)
        if isinstance(obj, CompletionOutput):
            return self._encode_completion_output(obj)

        # Other dataclasses
        if is_dataclass(obj) and not isinstance(obj, type):
            return asdict(obj)

        # slice
        if isinstance(obj, slice):
            return (obj.start, obj.stop, obj.step)

        # byte-like objects — modern msgspec encodes memoryview/bytearray natively
        # as msgpack bin without reaching this hook; this is a compatibility fallback.
        if isinstance(obj, (memoryview, bytearray)):
            return bytes(obj)

        raise TypeError(
            f"Object of type {type(obj).__name__} is not serializable. "
            "Supported types: torch.Tensor, np.ndarray, PIL.Image, dataclass, "
            "RequestOutput, and standard Python types (dict, list, str, int, float, bool, None, bytes)."
        )

    def _encode_tensor(self, tensor: torch.Tensor) -> dict[str, Any]:
        """Encode torch.Tensor to dict."""
        called_reshape = False
        called_contiguous = False

        start = time.perf_counter()

        t = tensor.detach().cpu()

        # Handle 0-dimensional (scalar) tensors by reshaping to 1D first
        if t.dim() == 0:
            called_reshape = True
            t = t.reshape(1)
        if not t.is_contiguous():
            t = t.contiguous()
            called_contiguous = True

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

        return {
            _TENSOR_MARKER: True,
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": list(tensor.shape),
            "data": data,
        }

    def _encode_ndarray(self, arr: np.ndarray) -> dict[str, Any]:
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

        return {
            _NDARRAY_MARKER: True,
            "dtype": arr.dtype.str,
            "shape": list(arr.shape),
            "data": data,
        }

    def _encode_pil_image(self, img: Image.Image) -> dict[str, Any]:
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
        return {
            _PIL_IMAGE_MARKER: True,
            "mode": img.mode,
            "shape": list(arr.shape),
            "data": data,
        }

    def _encode_request_output(self, obj: RequestOutput) -> dict[str, Any]:
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
                result["multimodal_output"] = dict(mm_output)
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
        return result

    def _encode_completion_output(self, obj: CompletionOutput) -> dict[str, Any]:
        """Encode CompletionOutput to dict, preserving multimodal payloads."""
        start = time.perf_counter()
        result = asdict(obj)
        mm_output = getattr(obj, "multimodal_output", None)
        if mm_output is not None:
            # Convert MultimodalPayload to plain dict for wire format
            if isinstance(mm_output, Mapping):
                result["multimodal_output"] = dict(mm_output)
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
        return result


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
        self.decoder = msgpack.Decoder()

    def decode(self, data: Buffer) -> Any:
        """Decode bytes to object."""
        result = self.decoder.decode(data)
        return self._post_process(result)

    def _post_process(self, obj: Any) -> Any:
        """Recursively restore tensor/ndarray/image/RequestOutput/OmniRequestOutput from their dict representations."""
        if isinstance(obj, dict):
            # Check for type markers first
            if obj.get(_TENSOR_MARKER):
                return self._decode_tensor(obj)
            if obj.get(_NDARRAY_MARKER):
                return self._decode_ndarray(obj)
            if obj.get(_PIL_IMAGE_MARKER):
                return self._decode_pil_image(obj)

            # Process values recursively first
            processed = {k: self._post_process(v) for k, v in obj.items()}

            # Check if this looks like an OmniRequestOutput (check before RequestOutput
            # since OmniRequestOutput may also have some RequestOutput-like fields)
            if self._is_omni_request_output(processed):
                return self._decode_omni_request_output(processed)

            # Check if this looks like a RequestOutput
            if _REQUEST_OUTPUT_KEYS.issubset(processed.keys()):
                return self._decode_request_output(processed)

            # Check if this looks like a CompletionOutput
            if _COMPLETION_OUTPUT_KEYS.issubset(processed.keys()):
                return self._decode_completion_output(processed)

            return processed

        if isinstance(obj, list):
            return [self._post_process(item) for item in obj]

        if isinstance(obj, tuple):
            return tuple(self._post_process(item) for item in obj)

        return obj

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
            # Use msgspec.convert for dataclass reconstruction
            result = msgspec.convert(obj, OmniRequestOutput)
        except Exception:
            try:
                # Fallback: construct directly if msgspec.convert fails
                # (e.g., if some fields are missing or have wrong types)
                result = OmniRequestOutput(**obj)
            except Exception:
                # If both attempts fail, return dict as-is (defensive fallback)
                # This should rarely happen if _is_omni_request_output is correct
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

    def _decode_tensor(self, obj: dict[str, Any]) -> torch.Tensor:
        """Decode dict to torch.Tensor."""
        dtype_str = obj["dtype"]
        shape = obj["shape"]
        data = obj["data"]

        start = time.perf_counter()
        torch_dtype = getattr(torch, dtype_str)
        if not data:
            result = torch.empty(shape, dtype=torch_dtype)
        else:
            # To prevent undefined behavior when downstream stages mutate the tensor, a copy is necessary.
            buffer = bytearray(data) if isinstance(data, (bytes, memoryview)) else data
            arr = torch.frombuffer(buffer, dtype=torch.uint8)
            result = arr.view(torch_dtype).reshape(shape)
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
                            "dtype": dtype_str,
                            "shape": shape,
                            "nbytes": len(data) if data else 0,
                        },
                    }
                ),
            )
        return result

    def _decode_ndarray(self, obj: dict[str, Any]) -> np.ndarray:
        """Decode dict to numpy.ndarray."""
        dtype = obj["dtype"]
        shape = obj["shape"]
        data = obj["data"]
        start = time.perf_counter()
        result = np.frombuffer(data, dtype=dtype).reshape(shape)
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
                            "dtype": dtype,
                            "shape": shape,
                            "nbytes": len(data) if data else 0,
                        },
                    }
                ),
            )
        return result

    def _decode_pil_image(self, obj: dict[str, Any]) -> Image.Image:
        """Decode dict to PIL.Image."""
        mode = obj["mode"]
        shape = obj["shape"]
        data = obj["data"]
        start = time.perf_counter()
        arr = np.frombuffer(data, dtype=np.uint8).reshape(shape)
        result = Image.fromarray(arr, mode=mode)
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
                            "mode": mode,
                            "shape": shape,
                            "nbytes": len(data) if data else 0,
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


class OmniSerde:
    """Serialization/deserialization handler for Omni IPC."""

    def __init__(self):
        self.encoder = OmniMsgpackEncoder()
        self.decoder = OmniMsgpackDecoder()

    def serialize(self, obj: Any) -> bytes:
        """Serialize obj and return an owned bytes object.

        Allocates a bytearray, encodes into it, then copies to bytes.  Safe for
        general use: the result can be stored indefinitely.  One allocation per
        call; prefer serialize_into() with a caller-owned buffer on hot paths.
        """
        return bytes(self.encoder.encode(obj))

    def serialize_into(self, obj: Any, buf: bytearray) -> None:
        """Serialize obj into caller-owned buf in-place.

        buf is grown automatically and truncated to the encoded length on
        success.  The caller is responsible for buf's lifetime.  Taking a
        memoryview(buf) after this call gives a zero-copy view of the encoded
        bytes; that view is only valid until the next serialize_into call on
        the same buf.
        """
        self.encoder.encode_into(obj, buf)

    def deserialize(self, data: Buffer) -> Any:
        """Deserialize bytes to an object."""
        return self.decoder.decode(data)


# Global instance for simple interface
OmniSerializer = OmniSerde()
