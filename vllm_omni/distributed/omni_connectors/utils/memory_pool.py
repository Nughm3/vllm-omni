# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading

import torch

from .logging import get_connector_logger

logger = get_connector_logger(__name__)
MANAGED_BUFFER_LEASES_KEY = "__mte_managed_buffer_leases__"


class BufferAllocator:
    """
    Manages the allocation of memory segments within a registered pool.
    Thread-safe implementation using a simple free list.
    """

    def __init__(self, total_size: int, alignment: int = 4096):
        self.total_size = total_size
        self.alignment = alignment
        self.lock = threading.Lock()
        self.free_blocks = [(0, total_size)]

    def alloc(self, size: int) -> int:
        """
        Allocates a block of *size* bytes and returns the starting offset.
        """
        aligned_size = (size + self.alignment - 1) // self.alignment * self.alignment

        with self.lock:
            for i, (start, block_size) in enumerate(self.free_blocks):
                if block_size >= aligned_size:
                    new_start = start + aligned_size
                    new_size = block_size - aligned_size

                    if new_size > 0:
                        self.free_blocks[i] = (new_start, new_size)
                    else:
                        self.free_blocks.pop(i)
                    return start

        raise MemoryError(f"Out of memory in buffer pool. Requested {size} bytes (aligned {aligned_size}).")

    def free(self, offset: int, size: int) -> None:
        """
        Frees a previously allocated block.
        """
        aligned_size = (size + self.alignment - 1) // self.alignment * self.alignment

        with self.lock:
            for start, length in self.free_blocks:
                if offset == start and aligned_size == length:
                    logger.warning("Double free detected at offset %s, size %s. Ignoring.", offset, aligned_size)
                    return
                if offset >= start and offset + aligned_size <= start + length:
                    logger.warning(
                        "Double free detected: block %s-%s is already within free block %s-%s. Ignoring.",
                        offset,
                        offset + aligned_size,
                        start,
                        start + length,
                    )
                    return
                if not (offset + aligned_size <= start or start + length <= offset):
                    raise RuntimeError(
                        f"Memory corruption detected: freeing {offset}-{offset + aligned_size} "
                        f"partially overlaps with free block {start}-{start + length}"
                    )

            self.free_blocks.append((offset, aligned_size))
            self.free_blocks.sort()

            i = 0
            while i < len(self.free_blocks) - 1:
                curr_start, curr_size = self.free_blocks[i]
                next_start, next_size = self.free_blocks[i + 1]

                if curr_start + curr_size == next_start:
                    self.free_blocks[i] = (curr_start, curr_size + next_size)
                    self.free_blocks.pop(i + 1)
                else:
                    i += 1


class _BufferLease:
    def __init__(self, allocator: BufferAllocator, offset: int, size: int):
        self.allocator = allocator
        self.offset = offset
        self.size = size
        self._references = 1
        self._released = False
        self._lock = threading.Lock()

    def retain(self) -> None:
        with self._lock:
            if self._released:
                raise RuntimeError("Cannot retain a released buffer allocation")
            self._references += 1

    def release(self) -> None:
        should_free = False
        with self._lock:
            if self._released:
                return
            self._references -= 1
            if self._references == 0:
                self._released = True
                should_free = True
        if should_free:
            self.allocator.free(self.offset, self.size)


class ManagedBuffer:
    """
    A temporary view into a global memory pool.
    Must be kept alive while the data view is being used.
    """

    def __init__(
        self,
        allocator: BufferAllocator,
        offset: int,
        size: int,
        pool_tensor: torch.Tensor,
        *,
        dtype: torch.dtype | None = None,
        shape: tuple[int, ...] | None = None,
        lease: _BufferLease | None = None,
    ):
        self.allocator = allocator
        self.offset = offset
        self.size = size
        self.pool_tensor = pool_tensor
        self.dtype = dtype
        self.shape = shape
        self._lease = lease or _BufferLease(allocator, offset, size)
        if lease is not None:
            self._lease.retain()
        self.ready_event: torch.cuda.Event | None = None
        self._released = False

    def release(self) -> None:
        """Explicitly release the buffer back to the pool."""
        if not self._released:
            self._lease.release()
            self._released = True

    def __del__(self):
        self.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    @property
    def tensor(self) -> torch.Tensor:
        """
        Returns a 1D uint8 zero-copy view of the buffer.
        """
        return self.pool_tensor[self.offset : self.offset + self.size]

    def as_tensor(self, dtype: torch.dtype, shape: tuple) -> torch.Tensor:
        """
        Returns a typed, shaped zero-copy view.
        Validates size, shape, and alignment.
        """
        itemsize = torch.tensor([], dtype=dtype).element_size()

        expected_bytes = itemsize
        for dim in shape:
            if dim < 0:
                raise ValueError("Dynamic dimension (-1) is not supported in as_tensor")
            expected_bytes *= dim

        if expected_bytes != self.size:
            raise ValueError(
                f"Shape {shape} with dtype {dtype} requires {expected_bytes} bytes, but buffer size is {self.size}"
            )

        if self.offset % itemsize != 0:
            raise RuntimeError(f"Buffer offset {self.offset} is not aligned for dtype {dtype} (itemsize {itemsize})")

        typed_view = self.tensor.view(dtype)
        return typed_view.reshape(shape)

    def slice(
        self,
        byte_offset: int,
        size: int,
        *,
        dtype: torch.dtype,
        shape: tuple[int, ...],
    ) -> "ManagedBuffer":
        if byte_offset < 0 or size <= 0 or byte_offset + size > self.size:
            raise ValueError(f"Invalid ManagedBuffer slice offset={byte_offset}, size={size}, buffer_size={self.size}")
        return ManagedBuffer(
            self.allocator,
            self.offset + byte_offset,
            size,
            self.pool_tensor,
            dtype=dtype,
            shape=shape,
            lease=self._lease,
        )

    def to_bytes(self) -> bytes:
        """
        Returns a copy of the data as Python bytes.
        Performs D2H copy if the pool is on a device.
        """
        t = self.tensor
        if getattr(t.device, "type", "cpu") != "cpu":
            t = t.cpu()
        return t.numpy().tobytes()
