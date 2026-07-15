# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import queue
import socket
import threading
import time as _time_mod
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import msgspec
import torch
import zmq

from ..utils.logging import get_connector_logger
from ..utils.memory_pool import (
    MANAGED_BUFFER_LEASES_KEY,
    BufferAllocator,
    ManagedBuffer,
)
from ..utils.serialization import OmniSerializer
from .base import OmniConnectorBase

logger = get_connector_logger(__name__)

try:
    from mooncake.engine import TransferEngine
except ImportError:
    TransferEngine = None

# Stale buffer TTL: buffers older than this are automatically reclaimed
# to prevent memory leaks when receiver crashes or gives up.
_BUFFER_TTL_SECONDS = 300  # 5 minutes

# ZMQ Message constants
TRANS_DONE = b"trans_done"
TRANS_ERROR = b"trans_error"
QUERY_INFO = b"query_info"
INFO_NOT_FOUND = b"info_not_found"
_SEGMENT_MARKER = "__mte_segment__"
_SEGMENTED_PROTOCOL_VERSION = 2


@dataclass
class QueryRequest:
    """Request to query metadata for a specific key."""

    request_id: str


@dataclass
class QueryResponse:
    """Response containing metadata for a request."""

    request_id: str
    data_size: int
    is_fast_path: bool
    segment_lengths: list[int] | None = None
    sidecar: bytes | None = None
    generation: str | None = None
    protocol_version: int = 1


@dataclass
class MooncakeAgentMetadata:
    """
    Metadata exchanged via ZMQ Handshake.
    """

    remote_hostname: str
    remote_port: int  # RDMA Port
    request_id: str
    dst_addrs: list[int]
    lengths: list[int]
    generation: str | None = None
    protocol_version: int = 1


class MooncakeTransferEngineConnector(OmniConnectorBase):
    """
    OmniConnector implementation using Mooncake Transfer Engine with a managed memory pool.
    Supports both CPU (Pinned) and GPU memory pools, and both RDMA and TCP protocols.
    Topology limitations (current implementation):
        Current design focuses on peer-to-peer communication between stages.
        - **1 sender → 1 receiver per key**: After a successful RDMA write the
          sender immediately cleans up the buffer (``cleanup()``), so only the
          first receiver to pull a given key will succeed.  Broadcast / multicast
          (1 sender → N receivers sharing the same data) is not yet supported.
        - **1 receiver → N senders**: Supported via partial metadata.  The
          manager constructs metadata with the target sender's
          ``source_host`` / ``source_port`` (computed from ``from_rank``)
          and passes it to ``get(metadata=...)``.  The connector detects
          that ``data_size`` is missing, queries the specified sender at
          the given address to fill it in, then performs the RDMA pull.
          This enables heterogeneous TP (sender TP > receiver TP) where a
          single receiver must pull KV shards from multiple sender ranks.

    Future work:
        - Support 1 sender → N receivers (e.g. reference-counted buffers, or
          explicit ``retain()`` / ``release()`` semantics so the buffer survives
          multiple pulls).
    """

    # RDMA connector copies raw bytes/tensor directly to the memory pool
    # without going through OmniSerializer, so the sender can use
    # to_gpu_tensor() / to_bytes() fast-paths.
    supports_raw_data: bool = True

    @staticmethod
    def _resolve_pool_config(
        config: dict[str, Any],
        *,
        can_put: bool,
        can_get: bool,
    ) -> tuple[int, str]:
        default_size = int(config.get("memory_pool_size", 1024**3))
        default_device = str(config.get("memory_pool_device", "cpu"))
        if can_put and not can_get:
            return (
                int(config.get("sender_memory_pool_size", default_size)),
                str(config.get("sender_memory_pool_device", default_device)),
            )
        if can_get and not can_put:
            return (
                int(config.get("receiver_memory_pool_size", default_size)),
                str(config.get("receiver_memory_pool_device", default_device)),
            )
        return default_size, default_device

    def __init__(self, config: dict[str, Any]):
        if TransferEngine is None:
            raise ImportError("Mooncake not available")

        self._closed = False
        self._bind_error: Exception | None = None  # fatal ZMQ bind error from listener thread

        # --- Early init of all teardown-related fields ---
        # If __init__ fails later (engine init, pool alloc, etc.), __del__ → close()
        # must not crash on missing attributes.  Safe defaults here ensure close()
        # can always run without AttributeError.
        self._stop_event = threading.Event()
        self._sender_executor: ThreadPoolExecutor | None = None
        self._listener_thread: threading.Thread | None = None
        self._listener_ready = threading.Event()
        self._local_buffers: dict[str, Any] = {}
        self._local_sidecars: dict[str, bytes] = {}
        self._local_generations: dict[str, str] = {}
        self._claimed_generations: dict[str, str | None] = {}
        self._cancelled_claims: set[tuple[str, str | None]] = set()
        self._local_buffers_lock = threading.Lock()
        self._put_lock = threading.Lock()
        self._req_local = threading.local()
        self._worker_local = threading.local()
        self._last_ttl_check: float = _time_mod.monotonic()
        self._sender_endpoints: dict[int, tuple[str, int]] = {}

        self._metrics = {
            "puts": 0,
            "gets": 0,
            "bytes_transferred": 0,
            "errors": 0,
            "timeouts": 0,
        }

        self.config = config
        host_config = config.get("host")
        host_value = "auto" if host_config is None else str(host_config)
        # Default sender/receiver bootstrap to a routable local IP so the
        # advertised endpoint matches the interface Mooncake binds.
        if host_value.lower() == "auto" or host_value in {"", "*", "0.0.0.0", "::"}:
            self.host = self._get_local_ip()
            logger.info(f"Auto-detected local IP for RDMA: {self.host}")
        else:
            self.host = host_value
        self.zmq_port = config.get("zmq_port", 50051)
        self.protocol = config.get("protocol", "rdma")

        # --- RDMA Device Configuration ---
        # Specify device names to filter (comma-separated), or empty for all devices.
        # Example: "mlx5_0,mlx5_1" to use only specific NICs.
        # This is important for environments with mixed InfiniBand/RoCE NICs.
        self.device_name = config.get("device_name", "")
        if not self.device_name:
            env_device = os.environ.get("RDMA_DEVICE_NAME", "")
            if env_device:
                self.device_name = env_device
                logger.info(f"Using RDMA_DEVICE_NAME from env: {self.device_name}")

        self.gpu_segment_min_bytes = int(config.get("gpu_segment_min_bytes", 4096))

        # --- Sender Configuration (for receiver to query without metadata) ---
        # Full-payload mixin calls get() without metadata, so receivers must
        # resolve sender_host here. "auto"/missing => same-node local IP
        # (cross-node deploys set an explicit remote sender_host).
        raw_sender_host = config.get("sender_host", None)
        if raw_sender_host is None or str(raw_sender_host).lower() in {
            "",
            "auto",
            "*",
            "0.0.0.0",
            "::",
        }:
            self.sender_host = self.host
        else:
            self.sender_host = str(raw_sender_host)
        self.sender_zmq_port = config.get("sender_zmq_port", None)

        # --- Role ---
        # Prefer explicit can_put / can_get (duplex-capable). Fall back to legacy
        # role=sender|receiver for callers that have not been migrated yet.
        self.can_put = bool(config.get("can_put", False))
        self.can_get = bool(config.get("can_get", False))
        if not self.can_put and not self.can_get:
            role = config.get("role")
            if role == "sender":
                self.can_put = True
            elif role == "receiver":
                self.can_get = True
            else:
                raise ValueError(
                    "MooncakeTransferEngineConnector must have can_put and/or can_get "
                    f"set to True (or legacy role=sender|receiver); got role={role!r}."
                )

        self.pool_size, self.pool_device = self._resolve_pool_config(
            config,
            can_put=self.can_put,
            can_get=self.can_get,
        )
        self.engine_id = str(uuid.uuid4())

        # --- Mooncake Engine Init ---
        self.engine = TransferEngine()
        # Note: For P2P handshake mode, local_hostname should be just the IP address.
        # Mooncake will auto-assign an RPC port, retrievable via get_rpc_port().
        ret = self.engine.initialize(self.host, "P2PHANDSHAKE", self.protocol, self.device_name)
        if ret != 0:
            raise RuntimeError(f"Mooncake Engine initialization failed with code {ret}")

        self.rpc_port = self.engine.get_rpc_port()
        logger.info(f"MooncakeTransferEngineConnector initialized at {self.host}:{self.rpc_port}")

        # --- Pool Allocation & Registration ---
        logger.info(f"Allocating RDMA Memory Pool: {self.pool_size / 1024**2:.2f} MB on {self.pool_device}")
        try:
            if self.pool_device == "cpu":
                self.pool = torch.empty(self.pool_size, dtype=torch.uint8).pin_memory()
                self.base_ptr = self.pool.data_ptr()
            else:
                self.pool = torch.empty(self.pool_size, dtype=torch.uint8, device=self.pool_device)
                self.base_ptr = self.pool.data_ptr()

            # Register the entire pool
            ret = self.engine.register_memory(self.base_ptr, self.pool_size)
            if ret != 0:
                raise RuntimeError("Failed to register memory pool with Mooncake Engine")

        except Exception as e:
            logger.error(f"Failed to allocate/register memory pool: {e}")
            raise

        self.allocator = BufferAllocator(self.pool_size, alignment=4096)  # 4KB alignment for safety

        # --- State Management & Background Threads ---
        # Most fields already initialized at the top of __init__ for teardown
        # safety.  Only create the real ZMQ context and thread pool here.
        self.zmq_ctx = zmq.Context()
        self._sender_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="mooncake-sender")

        # Log complete connector configuration for debugging
        logger.info(
            f"MooncakeTransferEngineConnector config summary:\n"
            f"  Local: host={self.host}, zmq_port={self.zmq_port}, rpc_port={self.rpc_port}\n"
            f"  Remote: sender_host={self.sender_host}, sender_zmq_port={self.sender_zmq_port}\n"
            f"  Role: can_put={self.can_put}, can_get={self.can_get}"
        )

        # Only sender needs ZMQ listener to handle pull requests
        if self.can_put:
            self._last_ttl_check = _time_mod.monotonic()  # reset after slow init
            self._listener_thread = threading.Thread(target=self._zmq_listener_loop, daemon=True)
            self._listener_thread.start()
            # Wait briefly for listener to bind (or fail)
            self._listener_ready.wait(timeout=1.0)
            # Propagate any bind error to the caller — sender must bind successfully.
            if self._bind_error is not None:
                raise RuntimeError(
                    f"MooncakeTransferEngineConnector failed to bind ZMQ on "
                    f"{self.host}:{self.zmq_port}: {self._bind_error}"
                ) from self._bind_error
            logger.info(
                f"MooncakeTransferEngineConnector started as SENDER (ZMQ listener on {self.host}:{self.zmq_port})"
            )
        else:
            # Receiver mode — sender address is provided per-request via
            # metadata from put() through the queue, not pre-configured.
            if not self.sender_host or self.sender_host.lower() == "auto":
                logger.info(
                    "MooncakeTransferEngineConnector receiver: sender_host='auto', "
                    "awaiting sender info via task from orchestrator."
                )
            else:
                logger.info(
                    f"MooncakeTransferEngineConnector started as RECEIVER "
                    f"(will query sender at {self.sender_host}:{self.sender_zmq_port})"
                )

    def _allocate_pool_buffer(
        self,
        size: int,
        *,
        dtype: torch.dtype | None = None,
        shape: tuple[int, ...] | None = None,
    ) -> ManagedBuffer:
        offset = self.allocator.alloc(size)
        return ManagedBuffer(
            self.allocator,
            offset,
            size,
            self.pool,
            dtype=dtype,
            shape=shape,
        )

    def allocate_buffer(
        self,
        size: int,
        *,
        dtype: torch.dtype | None = None,
        shape: tuple[int, ...] | None = None,
    ) -> ManagedBuffer | None:
        if size < self.gpu_segment_min_bytes:
            return None
        return self._allocate_pool_buffer(size, dtype=dtype, shape=shape)

    def supports_device_buffers(self) -> bool:
        return bool(self.pool.is_cuda)

    def allocate_buf(
        self,
        size: int,
        *,
        dtype: torch.dtype | None = None,
        shape: tuple[int, ...] | None = None,
    ) -> ManagedBuffer:
        return self._allocate_pool_buffer(size, dtype=dtype, shape=shape)

    def get_connection_info(self) -> dict[str, Any]:
        """Get connection info for this connector (useful for sender to share with receivers)."""
        return {
            "host": self.host,
            "zmq_port": self.zmq_port,
            "rpc_port": self.rpc_port,
            "can_put": self.can_put,
        }

    def update_sender_info(
        self,
        sender_host: str,
        sender_zmq_port: int,
        sender_rank: int | None = None,
    ) -> None:
        """Inject a sender's ZMQ endpoint into the receiver connector.

        When ``sender_rank`` is ``None`` (default), sets the single default
        sender used by ``get()`` when no rank is specified — this preserves
        backward-compatible 1:1 semantics.

        When ``sender_rank`` is an integer, the endpoint is stored in a
        per-rank registry for internal use (e.g. by
        ``_query_metadata_from_sender(sender_rank=R)``).
        """
        if sender_rank is not None:
            self._sender_endpoints[sender_rank] = (sender_host, sender_zmq_port)
            logger.info(
                "Sender info updated for rank %s: host=%r, zmq_port=%s",
                sender_rank,
                sender_host,
                sender_zmq_port,
            )
        else:
            self.sender_host = sender_host
            self.sender_zmq_port = sender_zmq_port
            logger.info(
                "Sender info updated (default): host=%r, zmq_port=%s",
                sender_host,
                sender_zmq_port,
            )

    def _get_local_ip(self) -> str:
        """
        Auto-detect the local IP address that can be used for RDMA communication.
        This tries to get the IP that would be used to communicate externally,
        not the loopback address.
        """
        try:
            # Create a socket to determine the local IP used for external communication
            # We don't actually connect, just use the socket to get routing info
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                # Use Google's DNS as a target (doesn't actually connect)
                s.connect(("8.8.8.8", 80))
                local_ip = s.getsockname()[0]
            return local_ip
        except Exception as e:
            logger.warning(f"Failed to auto-detect local IP: {e}, falling back to hostname lookup")
            try:
                # Fallback: use hostname resolution
                hostname = socket.gethostname()
                local_ip = socket.gethostbyname(hostname)
                return local_ip
            except Exception as e2:
                logger.error(f"Failed to get local IP via hostname: {e2}, using 127.0.0.1")
                return "127.0.0.1"

    def _get_req_socket(self, zmq_addr: str, timeout_ms: int) -> zmq.Socket:
        """Get or create a thread-local cached ZMQ REQ socket.

        Each calling thread maintains its own ``{zmq_addr: socket}`` map
        stored in ``self._req_local``, so concurrent ``get()`` /
        ``_query_metadata_from_sender()`` calls from different threads
        never share a REQ socket — which would violate ZMQ's strict
        send→recv ordering constraint.

        If a call fails, use ``_invalidate_req_socket()`` to discard the
        broken socket; the next call will transparently create a fresh one.
        """
        cache: dict[str, zmq.Socket] = getattr(self._req_local, "cache", None)  # type: ignore[assignment]
        if cache is None:
            cache = {}
            self._req_local.cache = cache

        sock = cache.get(zmq_addr)
        if sock is None:
            sock = self.zmq_ctx.socket(zmq.REQ)
            sock.connect(zmq_addr)
            cache[zmq_addr] = sock
        # Set timeouts per-call (safe to change between send/recv cycles).
        # SNDTIMEO guards against send() blocking when the peer is unreachable
        # or its receive buffer is full.
        sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
        sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        return sock

    def _invalidate_req_socket(self, zmq_addr: str) -> None:
        """Close and remove a thread-local cached REQ socket (e.g., after recv timeout)."""
        cache: dict[str, zmq.Socket] = getattr(self._req_local, "cache", None)  # type: ignore[assignment]
        if cache is None:
            return
        sock = cache.pop(zmq_addr, None)
        if sock is not None:
            try:
                sock.close(linger=0)
            except Exception:
                pass  # Best-effort; zmq_ctx.term() will reclaim if needed

    def put(self, from_stage: str, to_stage: str, put_key: str, data: Any) -> tuple[bool, int, dict[str, Any] | None]:
        """
        Producer Side.
        Exposes data for RDMA transfer.

        Behavior by data type:
        - ManagedBuffer (from this pool): Zero-copy, is_fast_path=True
        - ManagedBuffer (different pool): Fallback to tensor copy path
        - torch.Tensor / bytes: Copy to pool, is_fast_path=True
        - Other (dict, etc.): Serialize to bytes, copy to pool, is_fast_path=False
          (receiver will deserialize automatically)
        """
        if self._closed:
            raise RuntimeError("Cannot put data: MooncakeTransferEngineConnector is closed")

        if not self.can_put:
            logger.warning(
                f"Rejecting put for {put_key}: connector is in receiver mode "
                f"(sender fallback occurred, ZMQ port likely occupied by another instance)"
            )
            return False, 0, None

        put_key = self._make_key(put_key, from_stage, to_stage)

        # Serialize concurrent put() calls on the same connector to prevent
        # races in the alloc -> copy -> store-metadata flow at the Mooncake
        # C++ engine level.
        with self._put_lock:
            return self._put_impl(put_key, data)

    @staticmethod
    def _copy_tensor_to_pool(dst_tensor: torch.Tensor, src_tensor: torch.Tensor) -> None:
        if not src_tensor.is_cuda and not dst_tensor.is_cuda:
            dst_tensor.copy_(src_tensor)
            return
        dst_tensor.copy_(src_tensor, non_blocking=True)
        device = src_tensor.device if src_tensor.is_cuda else dst_tensor.device
        with torch.cuda.device(device):
            torch.cuda.current_stream(device).synchronize()

    def stage_tensor(
        self,
        buffer: ManagedBuffer,
        tensor: torch.Tensor,
        *,
        ready_event: torch.cuda.Event | None = None,
    ) -> None:
        self.stage_tensors([(buffer, tensor)], ready_event=ready_event)

    def stage_tensors(
        self,
        pairs: list[tuple[ManagedBuffer, torch.Tensor]],
        *,
        ready_event: torch.cuda.Event | None = None,
    ) -> None:
        if not pairs:
            return
        if ready_event is not None:
            ready_event.synchronize()
        cuda_device: torch.device | None = None
        for buffer, tensor in pairs:
            if buffer.pool_tensor.data_ptr() != self.pool.data_ptr():
                raise ValueError("Cannot stage into a ManagedBuffer from a different pool")
            if buffer.dtype != tensor.dtype or buffer.shape != tuple(tensor.shape):
                raise ValueError(
                    f"Buffer metadata {buffer.dtype}/{buffer.shape} does not match "
                    f"tensor {tensor.dtype}/{tuple(tensor.shape)}"
                )
            destination = buffer.as_tensor(tensor.dtype, tuple(tensor.shape))
            destination.copy_(tensor, non_blocking=True)
            if tensor.is_cuda:
                cuda_device = tensor.device
            elif destination.is_cuda:
                cuda_device = destination.device
            buffer.ready_event = None
        if cuda_device is not None:
            with torch.cuda.device(cuda_device):
                torch.cuda.current_stream(cuda_device).synchronize()

    @staticmethod
    def _contains_managed_buffer(value: Any) -> bool:
        if isinstance(value, ManagedBuffer):
            return True
        if isinstance(value, dict):
            return any(MooncakeTransferEngineConnector._contains_managed_buffer(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return any(MooncakeTransferEngineConnector._contains_managed_buffer(item) for item in value)
        return False

    def _build_segmented_sidecar(
        self,
        value: Any,
        buffers: list[ManagedBuffer],
    ) -> Any:
        if isinstance(value, ManagedBuffer):
            if value.pool_tensor.data_ptr() != self.pool.data_ptr():
                raise ValueError("Segmented packet contains a ManagedBuffer from a different pool")
            if value.dtype is None or value.shape is None:
                raise ValueError("Segmented tensor buffers require dtype and shape metadata")
            buffers.append(value)
            return {
                _SEGMENT_MARKER: len(buffers),
                "dtype": str(value.dtype),
                "shape": list(value.shape),
            }
        if isinstance(value, dict):
            return {key: self._build_segmented_sidecar(item, buffers) for key, item in value.items()}
        if isinstance(value, list):
            return [self._build_segmented_sidecar(item, buffers) for item in value]
        if isinstance(value, tuple):
            return tuple(self._build_segmented_sidecar(item, buffers) for item in value)
        return value

    @staticmethod
    def _release_holders(holder: Any, should_release: bool) -> None:
        if not should_release:
            return
        holders = holder if isinstance(holder, (list, tuple)) else [holder]
        for item in holders:
            if isinstance(item, ManagedBuffer):
                item.release()

    def _finish_claimed_transfer(
        self,
        request_id: str,
        item: Any,
        sidecar: bytes | None,
        generation: str | None,
        *,
        restore: bool,
    ) -> bool:
        with self._local_buffers_lock:
            self._claimed_generations.pop(request_id, None)
            cancelled_key = (request_id, generation)
            cancelled = cancelled_key in self._cancelled_claims
            self._cancelled_claims.discard(cancelled_key)
            if restore and not cancelled and request_id not in self._local_buffers:
                self._local_buffers[request_id] = item
                if sidecar is not None:
                    self._local_sidecars[request_id] = sidecar
                if generation is not None:
                    self._local_generations[request_id] = generation
                return True
        return False

    def _put_segmented_impl(
        self,
        put_key: str,
        data: Any,
    ) -> tuple[bool, int, dict[str, Any] | None]:
        data_buffers: list[ManagedBuffer] = []
        sidecar_value = self._build_segmented_sidecar(data, data_buffers)
        sidecar_bytes = OmniSerializer.serialize(sidecar_value)
        if not sidecar_bytes or not data_buffers:
            return False, 0, None

        holders = data_buffers
        for buffer in data_buffers:
            if buffer.ready_event is not None:
                buffer.ready_event.synchronize()
        src_addrs = [self.base_ptr + holder.offset for holder in holders]
        lengths = [holder.size for holder in holders]
        total_size = sum(lengths)
        generation = uuid.uuid4().hex

        with self._local_buffers_lock:
            if put_key in self._claimed_generations:
                logger.warning(
                    "Rejecting duplicate put for %s while its previous generation is in flight",
                    put_key,
                )
                return False, 0, None
            old_item = self._local_buffers.pop(put_key, None)
            self._local_sidecars.pop(put_key, None)
            self._local_generations.pop(put_key, None)
            if old_item:
                _, _, old_holder, old_should_release, _, _ = old_item
                self._release_holders(old_holder, old_should_release)
                logger.warning(f"Released old buffers for duplicate put_key: {put_key}")
            self._local_buffers[put_key] = (
                src_addrs,
                lengths,
                holders,
                True,
                True,
                _time_mod.monotonic(),
            )
            self._local_sidecars[put_key] = sidecar_bytes
            self._local_generations[put_key] = generation

        self._metrics["puts"] += 1
        self._metrics["bytes_transferred"] += total_size
        return (
            True,
            total_size,
            {
                "source_host": self.host,
                "source_port": self.zmq_port,
                "data_size": total_size,
                "segment_lengths": lengths,
                "sidecar": sidecar_bytes,
                "generation": generation,
                "protocol_version": _SEGMENTED_PROTOCOL_VERSION,
                "is_segmented": True,
                "is_fast_path": True,
            },
        )

    @staticmethod
    def _parse_torch_dtype(dtype_name: str) -> torch.dtype:
        name = dtype_name.removeprefix("torch.")
        dtype = getattr(torch, name, None)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"Unsupported tensor dtype in segmented payload: {dtype_name}")
        return dtype

    def _restore_segmented_payload(
        self,
        sidecar_bytes: bytes,
        data_buffers: list[ManagedBuffer],
    ) -> Any:
        sidecar = OmniSerializer.deserialize(sidecar_bytes)
        tensor_cache: dict[int, torch.Tensor] = {}

        def restore(value: Any) -> Any:
            if isinstance(value, dict) and _SEGMENT_MARKER in value:
                segment_index = int(value[_SEGMENT_MARKER])
                if segment_index < 1 or segment_index > len(data_buffers):
                    raise ValueError(f"Invalid segmented payload index: {segment_index}")
                cached = tensor_cache.get(segment_index)
                if cached is not None:
                    return cached
                buffer = data_buffers[segment_index - 1]
                dtype = self._parse_torch_dtype(str(value["dtype"]))
                shape = tuple(int(dim) for dim in value["shape"])
                tensor = buffer.as_tensor(dtype, shape)
                tensor_cache[segment_index] = tensor
                return tensor
            if isinstance(value, dict):
                return {key: restore(item) for key, item in value.items()}
            if isinstance(value, list):
                return [restore(item) for item in value]
            if isinstance(value, tuple):
                return tuple(restore(item) for item in value)
            return value

        restored = restore(sidecar)
        if not isinstance(restored, dict):
            raise TypeError("Segmented payload root must be a dict")
        restored[MANAGED_BUFFER_LEASES_KEY] = data_buffers
        return restored

    def _put_impl(self, put_key: str, data: Any) -> tuple[bool, int, dict[str, Any] | None]:
        """Internal put implementation, called under _put_lock."""
        try:
            if self._contains_managed_buffer(data):
                return self._put_segmented_impl(put_key, data)

            src_addr = 0
            size = 0
            holder = None
            is_fast_path = True
            should_release_holder = False

            # Reject empty data early — alloc(0) has undefined behaviour and
            # would leave a zombie entry in _local_buffers that the receiver
            # side can never consume (get() returns None for data_size==0).
            if isinstance(data, bytes) and len(data) == 0:
                logger.warning(f"Rejecting put for {put_key}: empty bytes payload")
                return False, 0, None
            if isinstance(data, torch.Tensor) and data.nbytes == 0:
                logger.warning(f"Rejecting put for {put_key}: zero-size tensor")
                return False, 0, None

            # Pre-process: serialize non-raw types before entering the copy path.
            # This avoids the previous recursive put() pattern and keeps
            # _local_buffers writes atomic (single write, no override needed).
            if not isinstance(data, (ManagedBuffer, torch.Tensor, bytes)):
                data = OmniSerializer.serialize(data)
                is_fast_path = False  # Receiver must deserialize

            if isinstance(data, ManagedBuffer):
                # Zero-Copy Path
                # Validate that the buffer belongs to this connector's pool.
                if data.pool_tensor.data_ptr() != self.pool.data_ptr():
                    # Fallback to copy path: extract tensor and let the
                    # torch.Tensor branch below handle the copy.
                    logger.warning("ManagedBuffer from different pool detected. Falling back to copy path.")
                    data = data.tensor.contiguous()
                    # Fall through to torch.Tensor branch
                else:
                    src_addr = self.base_ptr + data.offset
                    size = data.size
                    holder = data  # Keep the buffer alive
                    should_release_holder = False  # Caller owns it

            # Use 'if' (not 'elif') so the ManagedBuffer fallback above can
            # convert to Tensor and flow into this branch seamlessly.
            if isinstance(data, (torch.Tensor, bytes)):
                # Copy Path
                # 1. Determine size
                if isinstance(data, torch.Tensor):
                    size = data.nbytes
                    tensor_data = data
                else:
                    size = len(data)
                    # Convert bytes to tensor for copy
                    tensor_data = torch.frombuffer(data, dtype=torch.uint8)

                # 2. Alloc from pool
                try:
                    offset = self.allocator.alloc(size)
                    holder = ManagedBuffer(self.allocator, offset, size, self.pool)
                    should_release_holder = True  # We created it, we release it
                except MemoryError:
                    logger.error(f"Pool exhausted, cannot put data size {size}")
                    return False, 0, None

                # 3. Copy data to pool
                # Handle device mismatch for copy
                try:
                    dst_tensor = holder.tensor
                    if isinstance(data, torch.Tensor):
                        if not data.is_contiguous():
                            data = data.contiguous()

                        # View as flat uint8
                        src_view = data.view(torch.uint8).flatten()
                        self._copy_tensor_to_pool(dst_tensor, src_view)
                    else:
                        # bytes -> tensor copy
                        # torch.frombuffer creates CPU tensor. If pool is GPU, copy_ handles H2D.
                        self._copy_tensor_to_pool(dst_tensor, tensor_data)
                except Exception as e:
                    # Copy failed, release the allocated buffer to prevent leak
                    holder.release()
                    logger.error(f"Failed to copy data to pool: {e}")
                    return False, 0, None

                src_addr = self.base_ptr + offset

            # Final guard: reject zero-size payloads regardless of how they
            # got here (ManagedBuffer(size=0), empty serialization output, etc.).
            # This is the unified catch-all that complements the early checks
            # for bytes/Tensor above.
            if size <= 0:
                if should_release_holder and isinstance(holder, ManagedBuffer):
                    holder.release()
                logger.warning(f"Rejecting put for {put_key}: final payload size is {size}")
                return False, 0, None

            with self._local_buffers_lock:
                if put_key in self._claimed_generations:
                    if should_release_holder:
                        self._release_holders(holder, True)
                    logger.warning(
                        "Rejecting duplicate put for %s while its previous generation is in flight",
                        put_key,
                    )
                    return False, 0, None
                # Release old buffer if put_key already exists (prevents pool leak)
                old_item = self._local_buffers.pop(put_key, None)
                self._local_sidecars.pop(put_key, None)
                self._local_generations.pop(put_key, None)
                if old_item:
                    _, _, old_holder, old_should_release, _, _ = old_item
                    self._release_holders(old_holder, old_should_release)
                    logger.warning(f"Released old buffer for duplicate put_key: {put_key}")

                # Store: (src_addrs, lengths, holder, should_release, is_fast_path, created_at)
                self._local_buffers[put_key] = (
                    [src_addr],
                    [size],
                    holder,
                    should_release_holder,
                    is_fast_path,
                    _time_mod.monotonic(),
                )

            # Metadata for Consumer
            metadata = {
                "source_host": self.host,
                "source_port": self.zmq_port,
                "data_size": size,
                "is_fast_path": is_fast_path,  # Hint: True = return ManagedBuffer, False = deserialize
            }

            self._metrics["puts"] += 1
            self._metrics["bytes_transferred"] += size

            return True, size, metadata

        except Exception as e:
            self._metrics["errors"] += 1
            logger.error(f"RDMA Put failed for {put_key}: {e}", exc_info=True)
            return False, 0, None

    def _resolve_sender_endpoint(self, sender_rank: int | None = None) -> tuple[str, int] | None:
        """Return ``(host, zmq_port)`` for *sender_rank*.

        Resolution order:
        1. Per-rank registry (``_sender_endpoints[sender_rank]``)
        2. Default sender (``sender_host`` / ``sender_zmq_port``)
        3. ``None`` if nothing is configured.
        """
        if sender_rank is not None and sender_rank in self._sender_endpoints:
            return self._sender_endpoints[sender_rank]
        host = getattr(self, "sender_host", None)
        port = getattr(self, "sender_zmq_port", None)
        if host and port and str(host).lower() != "auto":
            return (host, int(port))
        return None

    def _query_metadata_at(self, get_key: str, host: str, port: int) -> dict[str, Any] | None:
        """Query metadata from a sender endpoint via ZMQ.

        Returns ``{source_host, source_port, data_size, is_fast_path}``
        or ``None`` when the key is not found / the query fails.
        """
        zmq_addr = f"tcp://{host}:{port}"
        req_socket = self._get_req_socket(zmq_addr, timeout_ms=5000)
        try:
            req_socket.send(QUERY_INFO + msgspec.msgpack.encode(QueryRequest(request_id=get_key)))
            resp = req_socket.recv()
            if resp == INFO_NOT_FOUND:
                return None
            query_resp = msgspec.msgpack.decode(resp, type=QueryResponse)
            return {
                "source_host": host,
                "source_port": port,
                "data_size": query_resp.data_size,
                "is_fast_path": query_resp.is_fast_path,
                "segment_lengths": query_resp.segment_lengths,
                "sidecar": query_resp.sidecar,
                "generation": query_resp.generation,
                "protocol_version": query_resp.protocol_version,
                "is_segmented": bool(query_resp.segment_lengths),
            }
        except Exception as e:
            self._invalidate_req_socket(zmq_addr)
            logger.debug("Failed to query metadata at %s for %s: %s", zmq_addr, get_key, e)
            return None

    def _query_metadata_from_sender(self, get_key: str, sender_rank: int | None = None) -> dict[str, Any] | None:
        """Query metadata from sender via ZMQ (fallback when ``metadata=None``).

        ``get()`` supports three metadata resolution paths::

            get(metadata=?)
            ├── Path 1: metadata has data_size (adapter path)
            │     → use metadata directly → RDMA pull
            ├── Path 2: metadata has source_host/port but no data_size
            │     → _query_metadata_at(host, port) → get data_size → RDMA pull
            └── Path 3: metadata=None (KV-transfer polling path)
                  → _query_metadata_from_sender(get_key)   ← this method
                  │
                  ├── sender endpoint resolved (via update_sender_info)
                  │     → ZMQ query → get data_size/is_fast_path
                  │     → construct metadata → RDMA pull
                  └── sender endpoint unresolved
                        → return None → caller retries or times out

        When *sender_rank* is provided, the query is routed to that
        rank's endpoint (registered via ``update_sender_info(rank=...)``).
        Otherwise the default sender is used.
        """
        endpoint = self._resolve_sender_endpoint(sender_rank)
        if endpoint is None:
            return None
        return self._query_metadata_at(get_key, *endpoint)

    def get(
        self,
        from_stage: str,
        to_stage: str,
        get_key: str,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[Any, int] | None:
        """Consumer Side.  Allocates from local pool and pulls data via RDMA.

        Metadata resolution:

        1. ``metadata`` provided **with** ``data_size`` → use directly (RDMA pull).
        2. ``metadata`` provided with ``source_host``/``source_port`` but
           **without** ``data_size`` → query that specific sender for
           ``data_size`` / ``is_fast_path``, then RDMA pull.  This is the
           heterogeneous-TP path where the manager knows the target sender
           endpoint but not the payload size.
        3. ``metadata=None`` → query the default sender (set via
           ``update_sender_info()``) for the full metadata.

        Returns:
            ``(data, size)`` on success, ``None`` on failure.

            - **is_fast_path=True** (tensor *or* bytes payload):
                Returns ``(ManagedBuffer, size)``.
                **CALLER MUST call ``ManagedBuffer.release()`` after consuming.**
            - **is_fast_path=False** (serialized Python object):
                Returns ``(DeserializedObject, size)``.
                Buffer is auto-released internally after deserialization.
        """
        if self._closed:
            raise RuntimeError("Cannot get data: MooncakeTransferEngineConnector is closed")

        get_key = self._make_key(get_key, from_stage, to_stage)

        _t0 = _time_mod.perf_counter()

        if not metadata:
            # Path 3: no metadata at all — query default sender
            if not self.sender_host or not self.sender_zmq_port or str(self.sender_host).lower() == "auto":
                raise RuntimeError(
                    f"get(metadata=None) requires sender info to be resolved, "
                    f"but sender_host={self.sender_host!r}, sender_zmq_port={self.sender_zmq_port!r}. "
                    f"Call update_sender_info(host, port) before using get() without metadata."
                )
            metadata = self._query_metadata_from_sender(get_key)
            if not metadata:
                return None
        elif "data_size" not in metadata:
            # Path 2: partial metadata (host/port only) — query that sender
            partial_host = metadata.get("source_host")
            partial_port = metadata.get("source_port")
            if not partial_host or not partial_port:
                logger.warning(
                    "get(%s): partial metadata missing source_host/source_port, cannot resolve data_size. metadata=%s",
                    get_key,
                    metadata,
                )
                return None
            queried = self._query_metadata_at(get_key, str(partial_host), int(partial_port))
            if not queried:
                return None
            metadata = queried

        _t1 = _time_mod.perf_counter()
        _query_ms = (_t1 - _t0) * 1000

        src_host = metadata.get("source_host")
        src_port = metadata.get("source_port")
        data_size = metadata.get("data_size", 0)
        is_fast_path = metadata.get("is_fast_path", False)
        segment_lengths = metadata.get("segment_lengths")
        sidecar = metadata.get("sidecar")
        generation = metadata.get("generation")
        protocol_version = int(metadata.get("protocol_version", 1))
        is_segmented = bool(metadata.get("is_segmented") or segment_lengths)

        if not src_host or not src_port or str(src_host).lower() == "auto":
            logger.error(
                f"Invalid metadata for {get_key}: source_host={src_host!r}, "
                f"source_port={src_port!r}. Cannot establish RDMA connection."
            )
            return None

        if data_size == 0:
            logger.warning(f"Skipping get for {get_key}: data_size is 0 (metadata={metadata})")
            return None

        # 1. Allocate Destination Buffer(s) from Pool
        recv_buffers: list[ManagedBuffer] = []
        try:
            if is_segmented:
                if not isinstance(segment_lengths, list) or not segment_lengths:
                    raise ValueError(f"Invalid segmented transfer lengths: {segment_lengths!r}")
                if not isinstance(sidecar, bytes) or not sidecar:
                    raise ValueError("Segmented transfer metadata is missing its sidecar")
                if protocol_version != _SEGMENTED_PROTOCOL_VERSION or not generation:
                    raise ValueError("Segmented transfer requires a coordinated protocol-v2 sender and receiver")
                if sum(int(length) for length in segment_lengths) != data_size:
                    raise ValueError(f"Segment lengths sum to {sum(segment_lengths)}, expected data_size={data_size}")
                for length in segment_lengths:
                    recv_buffers.append(self._allocate_pool_buffer(int(length)))
            else:
                # Mooncake's TCP/memcpy transport may prepend a protocol byte
                # for serialized objects, so retain the legacy padding.
                padding = 4 if not is_fast_path else 0
                recv_buffers.append(self._allocate_pool_buffer(data_size + padding))
        except MemoryError:
            self._release_holders(recv_buffers, True)
            logger.error(f"Failed to allocate {data_size} bytes in receive pool")
            return None
        except Exception as e:
            self._release_holders(recv_buffers, True)
            logger.error(f"Invalid segmented transfer metadata for {get_key}: {e}")
            return None

        _t2 = _time_mod.perf_counter()
        _alloc_ms = (_t2 - _t1) * 1000

        # 2. Prepare Handshake
        transfer_lengths = [int(length) for length in segment_lengths] if is_segmented else [data_size]
        agent_meta = MooncakeAgentMetadata(
            remote_hostname=self.host,
            remote_port=self.rpc_port,
            request_id=get_key,
            dst_addrs=[self.base_ptr + buffer.offset for buffer in recv_buffers],
            lengths=transfer_lengths,
            generation=str(generation) if is_segmented else None,
            protocol_version=protocol_version if is_segmented else 1,
        )

        # 3. ZMQ Transaction (uses cached socket to avoid TCP reconnection)
        # Timeout scales with data size: base 30s + 5s per 100MB.
        # Large RDMA transfers (esp. loopback or cross-NIC) can take tens of
        # seconds; if we time out too early the receiver closes and deregisters
        # its memory, causing the sender's in-flight write to fail with ret=-1.
        _base_timeout_ms = 30000
        _size_timeout_ms = max(0, (data_size // (100 * 1024 * 1024))) * 5000
        _total_timeout_ms = _base_timeout_ms + _size_timeout_ms
        zmq_addr = f"tcp://{src_host}:{src_port}"
        req_socket = self._get_req_socket(zmq_addr, timeout_ms=_total_timeout_ms)

        try:
            req_socket.send(msgspec.msgpack.encode(agent_meta))
            resp = req_socket.recv()

            _t3 = _time_mod.perf_counter()
            _rdma_ms = (_t3 - _t2) * 1000

            if resp == TRANS_DONE:
                # batch_transfer_sync_write has completed before TRANS_DONE is
                # sent. Do not synchronize an arbitrary CUDA stream here: that
                # measures unrelated model work and caused large latency tails.
                # A transport-provided GPUDirect visibility fence should replace
                # this completion contract if Mooncake exposes one.
                _t4 = _time_mod.perf_counter()
                _sync_ms = (_t4 - _t3) * 1000

                if is_segmented:
                    try:
                        value = self._restore_segmented_payload(
                            sidecar,
                            recv_buffers,
                        )
                    except Exception:
                        raise
                    _t5 = _time_mod.perf_counter()
                    _total_ms = (_t5 - _t0) * 1000
                    _mbps = (data_size / 1024 / 1024) / (_total_ms / 1000) if _total_ms > 0 else 0
                    logger.info(
                        f"[RDMA GET] {get_key}: query={_query_ms:.1f}ms, alloc={_alloc_ms:.1f}ms, "
                        f"rdma={_rdma_ms:.1f}ms, sync={_sync_ms:.1f}ms, "
                        f"segments={len(recv_buffers)}, total={_total_ms:.1f}ms, {_mbps:.1f} MB/s"
                    )
                    self._metrics["gets"] += 1
                    self._metrics["bytes_transferred"] += data_size
                    return value, data_size

                recv_buffer = recv_buffers[0]
                if is_fast_path:
                    # Return ManagedBuffer directly for ALL fast_path data
                    # (both bytes and tensor payloads).  Caller is responsible
                    # for consuming the data and calling release() afterwards.
                    # This avoids the expensive to_bytes() copy (~54ms for 115MB).
                    #
                    # Usage patterns for callers:
                    #   - Zero-copy read: mv = memoryview(buf.tensor.numpy())
                    #   - Copy to GPU:    buf.tensor.to(device); buf.release()
                    #   - Fallback:       data = buf.to_bytes(); buf.release()
                    _t5 = _time_mod.perf_counter()
                    _total_ms = (_t5 - _t0) * 1000
                    _mbps = (data_size / 1024 / 1024) / (_total_ms / 1000) if _total_ms > 0 else 0
                    logger.info(
                        f"[RDMA GET] {get_key}: query={_query_ms:.1f}ms, alloc={_alloc_ms:.1f}ms, "
                        f"rdma={_rdma_ms:.1f}ms, sync={_sync_ms:.1f}ms, "
                        f"total={_total_ms:.1f}ms, {_mbps:.1f} MB/s (fast_path, zero-copy)"
                    )
                    self._metrics["gets"] += 1
                    self._metrics["bytes_transferred"] += data_size
                    return recv_buffer, data_size
                else:
                    # If it was a serialized object or generic bytes, we assume standard Omni behavior:
                    # Deserialize and return object. This requires a copy (to_bytes).
                    # We MUST release the buffer after deserialization.
                    try:
                        _t_copy_start = _time_mod.perf_counter()
                        raw_bytes = recv_buffer.to_bytes()
                        _t_copy_end = _time_mod.perf_counter()
                        _copy_ms = (_t_copy_end - _t_copy_start) * 1000

                        # Try deserializing the first data_size bytes.  Mooncake's
                        # TCP/memcpy transport may prepend a leading protocol byte
                        # (0x01) before the payload — handle this by retrying from
                        # offset 1 if the first attempt raises DecodeError.
                        _t_deser_start = _time_mod.perf_counter()
                        payload = raw_bytes[:data_size]
                        try:
                            val = OmniSerializer.deserialize(payload)
                        except msgspec.DecodeError:
                            # Likely a leading Mooncake TCP protocol byte — retry
                            # from offset 1, using the extra padding bytes if needed.
                            payload = raw_bytes[1 : data_size + 1]
                            val = OmniSerializer.deserialize(payload)
                        _t_deser_end = _time_mod.perf_counter()
                        _deser_ms = (_t_deser_end - _t_deser_start) * 1000

                        _total_ms = (_t_deser_end - _t0) * 1000
                        _mbps = (data_size / 1024 / 1024) / (_total_ms / 1000) if _total_ms > 0 else 0
                        logger.info(
                            f"[RDMA GET] {get_key}: query={_query_ms:.1f}ms, alloc={_alloc_ms:.1f}ms, "
                            f"rdma={_rdma_ms:.1f}ms, sync={_sync_ms:.1f}ms, copy={_copy_ms:.1f}ms, "
                            f"deser={_deser_ms:.1f}ms, total={_total_ms:.1f}ms, {_mbps:.1f} MB/s"
                        )
                        self._metrics["gets"] += 1
                        self._metrics["bytes_transferred"] += data_size
                        return val, data_size
                    finally:
                        recv_buffer.release()
            else:
                self._metrics["errors"] += 1
                logger.error(f"RDMA Get failed: received {resp} instead of TRANS_DONE")
                self._release_holders(recv_buffers, True)
                return None
        except Exception as e:
            # Socket may be stuck after timeout; discard it
            self._invalidate_req_socket(zmq_addr)
            self._metrics["timeouts"] += 1
            logger.error(f"RDMA Get error: {e}", exc_info=True)
            self._release_holders(recv_buffers, True)
            return None

    def cleanup(self, request_id: str, from_stage: str | None = None, to_stage: str | None = None) -> None:
        """Release the producer-side buffer associated with the request.

        Args:
            request_id: The key used in ``put()``.
            from_stage: Optional source stage.  When both *from_stage* and
                *to_stage* are provided the method applies ``_make_key()``
                internally, so callers can pass the same raw key they used
                in ``put()`` / ``get()`` without knowing the internal format.
                When omitted the *request_id* is used as-is (suitable for
                internal callers that already hold the transformed key).
            to_stage: Optional destination stage (see *from_stage*).
        """
        if (from_stage is None) != (to_stage is None):
            raise ValueError(
                f"cleanup() requires both from_stage and to_stage, or neither. "
                f"Got from_stage={from_stage!r}, to_stage={to_stage!r}"
            )
        if from_stage is not None and to_stage is not None:
            request_id = self._make_key(request_id, from_stage, to_stage)
        with self._local_buffers_lock:
            item = self._local_buffers.pop(request_id, None)
            self._local_sidecars.pop(request_id, None)
            self._local_generations.pop(request_id, None)
            if request_id in self._claimed_generations:
                self._cancelled_claims.add((request_id, self._claimed_generations[request_id]))
            if item:
                # item is (src_addrs, lengths, holder, should_release, is_fast_path, created_at)
                _, _, holder, should_release, _, _ = item
                self._release_holders(holder, should_release)
                # If holder was externally owned (should_release=False), we do nothing.
                # If holder was something else (e.g. Tensor), GC handles it.

    def health(self) -> dict[str, Any]:
        if self._closed:
            return {"status": "unhealthy", "error": "Connector is closed"}

        return {
            "status": "healthy",
            "host": self.host,
            "metadata_server": None,
            "master": None,
            "protocol": self.protocol,
            "pool_device": self.pool_device,
            "pool_size": self.pool_size,
            **self._metrics,
        }

    def close(self) -> None:
        """
        Gracefully shutdown the connector and release all resources.
        This method should be called when the connector is no longer needed.
        Idempotent: safe to call multiple times.
        """
        # Idempotent guard — use getattr for safety: if __init__ raised before
        # setting _closed, __del__ → close() would hit AttributeError.
        if getattr(self, "_closed", True):
            return
        self._closed = True

        logger.info("Closing MooncakeTransferEngineConnector...")

        # 1. Signal listener thread to stop
        self._stop_event.set()

        # 2. Wait for listener thread to finish (only if sender mode)
        if self._listener_thread is not None and self._listener_thread.is_alive():
            self._listener_thread.join(timeout=2.0)
            if self._listener_thread.is_alive():
                logger.warning("Listener thread did not stop gracefully")

        # 3. Shutdown sender executor (may be None if __init__ failed early)
        if self._sender_executor is not None:
            self._sender_executor.shutdown(wait=True, cancel_futures=False)

        # 4. Release all pending buffers
        with self._local_buffers_lock:
            for req_id, item in list(self._local_buffers.items()):
                _, _, holder, should_release, _, _ = item
                self._release_holders(holder, should_release)
            self._local_buffers.clear()
            self._local_sidecars.clear()
            self._local_generations.clear()
            self._claimed_generations.clear()
            self._cancelled_claims.clear()

        # 5. Close thread-local cached REQ sockets (for the calling thread).
        #    Sockets cached in *other* threads are not directly accessible here
        #    because they live in per-thread ``threading.local()`` storage.
        #    They will be forcefully closed when ``zmq_ctx.term()`` is called
        #    in step 7 below, which is the intended cleanup path.
        cache: dict[str, zmq.Socket] = getattr(self._req_local, "cache", None)  # type: ignore[assignment]
        if cache:
            for addr, sock in cache.items():
                try:
                    sock.close(linger=0)
                except Exception:
                    pass  # Best-effort; zmq_ctx.term() below will reclaim
            cache.clear()

        # 6. Unregister memory from engine (if supported)
        try:
            if hasattr(self, "engine") and hasattr(self.engine, "unregister_memory"):
                # Mooncake API only takes address, not size
                self.engine.unregister_memory(self.base_ptr)
        except Exception as e:
            logger.warning(f"Failed to unregister memory: {e}")

        # 7. Close ZMQ contexts (also reclaims sockets cached in other threads)
        try:
            if hasattr(self, "zmq_ctx"):
                self.zmq_ctx.term()
        except Exception as e:
            logger.warning(f"Failed to terminate ZMQ context: {e}")

        # 8. Release pool tensor reference (let GC handle actual deallocation)
        # Note: We set to None instead of del to avoid AttributeError on repeated access
        self.pool = None

        logger.info("MooncakeTransferEngineConnector closed.")

    # -------------------------------------------------------
    # Listener Logic
    # -------------------------------------------------------

    def _cleanup_stale_buffers(self) -> None:
        """Reclaim buffers older than ``_BUFFER_TTL_SECONDS``.
        Prevents permanent memory leaks when a receiver crashes or times out
        without ever pulling the data.
        TODO(wzliu): In extreme rare case, long transfer time, there might exist
        TTL cleanup vs in-flight transfer conflict, which will be handled in the next PR.
        """
        now = _time_mod.monotonic()
        with self._local_buffers_lock:
            stale_keys = [k for k, v in self._local_buffers.items() if now - v[5] > _BUFFER_TTL_SECONDS]
            for k in stale_keys:
                item = self._local_buffers.pop(k)
                self._local_sidecars.pop(k, None)
                self._local_generations.pop(k, None)
                _, _, holder, should_release, _, _ = item
                self._release_holders(holder, should_release)
                logger.warning(f"TTL expired ({_BUFFER_TTL_SECONDS}s): cleaned up stale buffer for {k}")

    def _zmq_listener_loop(self):
        socket = self.zmq_ctx.socket(zmq.ROUTER)
        try:
            socket.bind(f"tcp://{self.host}:{self.zmq_port}")
        except zmq.ZMQError as exc:
            # Any bind failure (EADDRINUSE, EADDRNOTAVAIL, EACCES, etc.)
            # is fatal for a sender — fail fast so __init__ propagates the error.
            logger.error(f"ZMQ bind failed on {self.host}:{self.zmq_port}: {exc} (errno={exc.errno})")
            self.can_put = False
            self._bind_error = exc
            self._listener_ready.set()
            return  # let __init__ propagate the error

        # Successfully bound - signal ready
        self._listener_ready.set()

        # Create inproc socket pair for worker thread notifications
        # This allows workers to wake up the listener immediately when done
        notify_addr = f"inproc://notify-{id(self)}"
        notify_recv = self.zmq_ctx.socket(zmq.PULL)
        notify_recv.bind(notify_addr)

        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)
        poller.register(notify_recv, zmq.POLLIN)

        # Response queue for thread-safe socket operations
        response_queue: queue.Queue = queue.Queue()

        try:
            while not self._stop_event.is_set():
                try:
                    # Poll for incoming requests or notifications (with timeout)
                    events = dict(poller.poll(1000))

                    # Process notifications (drain all)
                    if notify_recv in events:
                        while True:
                            try:
                                notify_recv.recv(zmq.NOBLOCK)
                            except zmq.Again:
                                break

                    # Process any pending responses (non-blocking)
                    while True:
                        try:
                            identity, response = response_queue.get_nowait()
                            socket.send_multipart([identity, b"", response])
                        except queue.Empty:
                            break

                    # Periodic TTL cleanup (~every 10s)
                    now = _time_mod.monotonic()
                    if now - self._last_ttl_check >= 10.0:
                        self._last_ttl_check = now
                        self._cleanup_stale_buffers()

                    # Process incoming requests
                    if socket in events:
                        frames = socket.recv_multipart()
                        if len(frames) >= 2:
                            # Submit to thread pool
                            self._sender_executor.submit(
                                self._handle_pull_request,
                                response_queue,
                                notify_addr,
                                frames[0],
                                frames[-1],
                            )
                except zmq.ContextTerminated:
                    break
                except Exception:
                    logger.debug("Listener loop error", exc_info=True)
        finally:
            try:
                notify_recv.close(linger=0)
                socket.close(linger=0)
            except Exception:
                pass  # Best-effort cleanup during listener shutdown

    def _handle_pull_request(self, response_queue: queue.Queue, notify_addr: str, identity, payload):
        """
        Handle pull request or query request in worker thread.
        Results are put into response_queue and listener is notified via inproc.
        """
        claimed_item = None
        claimed_sidecar = None
        claimed_generation = None
        try:
            # Check if this is a query request
            if payload.startswith(QUERY_INFO):
                self._handle_query_request(response_queue, notify_addr, identity, payload[len(QUERY_INFO) :])
                return

            # Normal RDMA transfer request
            meta = msgspec.msgpack.decode(payload, type=MooncakeAgentMetadata)

            with self._local_buffers_lock:
                item = self._local_buffers.get(meta.request_id)
                sidecar = self._local_sidecars.get(meta.request_id)
                generation = self._local_generations.get(meta.request_id)
                validation_error = None
                if not item:
                    validation_error = "buffer not found"
                else:
                    current_segmented = sidecar is not None
                    requested_segmented = (
                        meta.protocol_version == _SEGMENTED_PROTOCOL_VERSION or meta.generation is not None
                    )
                    if current_segmented != requested_segmented:
                        validation_error = "segmented/nonsegmented protocol mismatch"
                    elif current_segmented and (
                        meta.protocol_version != _SEGMENTED_PROTOCOL_VERSION or meta.generation != generation
                    ):
                        validation_error = f"stale generation requested={meta.generation}, current={generation}"
                    elif len(meta.dst_addrs) != len(item[0]) or meta.lengths != item[1]:
                        validation_error = (
                            f"segment mismatch source={item[1]}, destination={meta.lengths}, "
                            f"addresses={len(meta.dst_addrs)}"
                        )
                if validation_error is None:
                    claimed_item = self._local_buffers.pop(meta.request_id)
                    claimed_sidecar = self._local_sidecars.pop(meta.request_id, None)
                    claimed_generation = self._local_generations.pop(meta.request_id, None)
                    self._claimed_generations[meta.request_id] = claimed_generation

            if validation_error is not None:
                logger.error(
                    "Rejected pull for %s: %s",
                    meta.request_id,
                    validation_error,
                )
                response_queue.put((identity, TRANS_ERROR))
                self._notify_listener(notify_addr)
                return

            # RDMA Write
            src_addrs, src_lengths, holder, should_release, _, _ = claimed_item
            remote_session = f"{meta.remote_hostname}:{meta.remote_port}"
            ret = self.engine.batch_transfer_sync_write(remote_session, src_addrs, meta.dst_addrs, src_lengths)

            if ret == 0:
                self._finish_claimed_transfer(
                    meta.request_id,
                    claimed_item,
                    claimed_sidecar,
                    claimed_generation,
                    restore=False,
                )
                self._release_holders(holder, should_release)
                claimed_item = None
                response_queue.put((identity, TRANS_DONE))
            else:
                restored = self._finish_claimed_transfer(
                    meta.request_id,
                    claimed_item,
                    claimed_sidecar,
                    claimed_generation,
                    restore=True,
                )
                if not restored:
                    self._release_holders(holder, should_release)
                claimed_item = None
                logger.warning(
                    f"RDMA write failed for {meta.request_id} to {remote_session} "
                    f"(ret={ret}). Buffer {'retained for retry' if restored else 'superseded by a newer put'}."
                )
                response_queue.put((identity, TRANS_ERROR))

        except Exception as e:
            if claimed_item is not None:
                _, _, holder, should_release, _, _ = claimed_item
                restored = self._finish_claimed_transfer(
                    meta.request_id,
                    claimed_item,
                    claimed_sidecar,
                    claimed_generation,
                    restore=True,
                )
                if not restored:
                    self._release_holders(holder, should_release)
            logger.error(f"Push failed: {e}")
            response_queue.put((identity, TRANS_ERROR))

        # Notify listener thread that response is ready
        self._notify_listener(notify_addr)

    def _handle_query_request(self, response_queue: queue.Queue, notify_addr: str, identity, payload):
        """Handle metadata query request."""
        try:
            query = msgspec.msgpack.decode(payload, type=QueryRequest)

            with self._local_buffers_lock:
                item = self._local_buffers.get(query.request_id)
                sidecar = self._local_sidecars.get(query.request_id)
                generation = self._local_generations.get(query.request_id)

            if not item:
                response_queue.put((identity, INFO_NOT_FOUND))
            else:
                src_addrs, src_lengths, _, _, is_fast_path, _ = item

                resp = QueryResponse(
                    request_id=query.request_id,
                    data_size=sum(src_lengths),
                    is_fast_path=is_fast_path,
                    segment_lengths=src_lengths if sidecar is not None else None,
                    sidecar=sidecar,
                    generation=generation,
                    protocol_version=(_SEGMENTED_PROTOCOL_VERSION if sidecar is not None else 1),
                )
                response_queue.put((identity, msgspec.msgpack.encode(resp)))

        except Exception as e:
            logger.error(f"Query request failed: {e}")
            response_queue.put((identity, INFO_NOT_FOUND))

        self._notify_listener(notify_addr)

    def _notify_listener(self, notify_addr: str):
        """Send notification to wake up listener thread.

        Uses a per-worker-thread cached PUSH socket to avoid the overhead
        of creating/connecting/closing a socket on every call.
        """
        try:
            local = self._worker_local
            sock = getattr(local, "notify_socket", None)
            cached_addr = getattr(local, "notify_addr", None)
            if sock is None or cached_addr != notify_addr:
                # First call on this thread, or address changed – create socket
                if sock is not None:
                    sock.close(linger=0)
                sock = self.zmq_ctx.socket(zmq.PUSH)
                sock.connect(notify_addr)
                local.notify_socket = sock
                local.notify_addr = notify_addr
            sock.send(b"", zmq.NOBLOCK)
        except Exception:
            # Socket may be broken; clear cache so next call recreates it
            local.notify_socket = None
            local.notify_addr = None
