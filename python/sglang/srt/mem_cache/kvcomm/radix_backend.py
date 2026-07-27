from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Protocol, Sequence

import torch

from sglang.srt.layers.rotary_embedding.utils import apply_rotary_emb
from sglang.srt.mem_cache.kvcomm.types import (
    KVSegmentHandle,
    KVSegmentKey,
    ResidencyTier,
    SegmentKind,
    token_ids_hash,
)
from sglang.srt.mem_cache.kvcomm.store import ResidencyLoadResult

if TYPE_CHECKING:
    from sglang.srt.mem_cache.kvcomm.manager import KVCommManager


@dataclass(frozen=True)
class DeviceKVRef:
    indices: torch.Tensor


@dataclass(frozen=True)
class HostKVRef:
    payload: Any


@dataclass(frozen=True)
class RoPEConfig:
    rotary_dim: int
    base: float
    is_neox_style: bool

    def __post_init__(self) -> None:
        if self.rotary_dim < 0 or self.rotary_dim % 2:
            raise ValueError("rotary_dim must be a non-negative even number")
        if self.base <= 0:
            raise ValueError("RoPE base must be positive")


class KVPoolAllocator(Protocol):
    def alloc(self, need_size: int) -> torch.Tensor | None: ...

    def free(self, indices: torch.Tensor) -> None: ...

    def get_kvcache(self) -> Any: ...

    def get_cpu_copy(self, indices: torch.Tensor) -> Any: ...

    def load_cpu_copy(self, payload: Any, indices: torch.Tensor) -> None: ...


class AllocatorResidencyLoader:
    """Loads a host KV payload into newly allocated device slots."""

    def __init__(self, allocator: KVPoolAllocator) -> None:
        self._allocator = allocator

    def load(
        self,
        handle: KVSegmentHandle,
        target_tier: ResidencyTier,
    ) -> ResidencyLoadResult:
        if target_tier != ResidencyTier.DEVICE:
            raise NotImplementedError(
                "the Radix allocator adapter currently loads only to device"
            )
        if not isinstance(handle.backend_ref, HostKVRef):
            raise TypeError("host-resident handle must carry HostKVRef")
        indices = self._allocator.alloc(len(handle.token_ids))
        if indices is None or len(indices) != len(handle.token_ids):
            if indices is not None:
                self._allocator.free(indices)
            raise MemoryError("unable to allocate device slots for prefetched KV")
        try:
            self._allocator.load_cpu_copy(handle.backend_ref.payload, indices)
        except Exception:
            self._allocator.free(indices)
            raise
        return ResidencyLoadResult(
            backend_ref=DeviceKVRef(indices=indices),
            release_backend=self._release_device_ref,
        )

    def _release_device_ref(
        self, backend_ref: object, residency: ResidencyTier
    ) -> None:
        if residency != ResidencyTier.DEVICE or not isinstance(
            backend_ref, DeviceKVRef
        ):
            raise TypeError("allocator releaser received a non-device KV ref")
        self._allocator.free(backend_ref.indices)


class TargetSlotTransaction:
    """Own newly allocated KV slots until they are published to a request."""

    def __init__(self, allocator: KVPoolAllocator, length: int) -> None:
        if length <= 0:
            raise ValueError("target transaction length must be positive")
        self._allocator = allocator
        self._indices = allocator.alloc(length)
        if self._indices is None or len(self._indices) != length:
            if self._indices is not None:
                allocator.free(self._indices)
            raise MemoryError("unable to allocate transactional target KV slots")
        self._committed = False
        self._closed = False

    @property
    def indices(self) -> torch.Tensor:
        if self._closed:
            raise RuntimeError("target slot transaction is closed")
        return self._indices

    def commit(self) -> torch.Tensor:
        if self._closed:
            raise RuntimeError("target slot transaction is closed")
        self._committed = True
        self._closed = True
        return self._indices

    def rollback(self) -> bool:
        if self._closed:
            return False
        self._closed = True
        self._allocator.free(self._indices)
        return True

    def __enter__(self) -> "TargetSlotTransaction":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if not self._committed:
            self.rollback()


class DeviceSegmentMaterializer:
    """Create an allocator-owned device copy of a completed request span."""

    def __init__(
        self,
        *,
        manager: "KVCommManager",
        allocator: KVPoolAllocator,
        model_id: str,
        cache_dtype: str,
    ) -> None:
        if not manager.config.core_enabled:
            raise ValueError("KVCOMM core must be enabled")
        if not model_id or not cache_dtype:
            raise ValueError("model_id and cache_dtype must be non-empty")
        self._manager = manager
        self._allocator = allocator
        self._model_id = model_id
        self._cache_dtype = cache_dtype
        self._accounting_lock = threading.Lock()
        self._owned_device_tokens = 0

    @property
    def owned_device_tokens(self) -> int:
        with self._accounting_lock:
            return self._owned_device_tokens

    def materialize(
        self,
        *,
        token_ids: Sequence[int],
        source_indices: torch.Tensor,
        source_start: int,
        content_hash: str | None = None,
    ) -> KVSegmentHandle:
        tokens = tuple(int(token) for token in token_ids)
        if not tokens:
            raise ValueError("cannot materialize an empty device segment")
        if source_start < 0:
            raise ValueError("source_start must be non-negative")
        if source_indices.ndim != 1 or len(source_indices) != len(tokens):
            raise ValueError(
                "source_indices must be 1-D and match the segment token count"
            )

        destination = self._allocator.alloc(len(tokens))
        if destination is None or len(destination) != len(tokens):
            if destination is not None:
                self._allocator.free(destination)
            raise MemoryError("unable to allocate owned device segment")

        released = False
        accounted = False

        def release_owned(ref: object, residency: ResidencyTier) -> None:
            nonlocal released
            if released:
                return
            if residency != ResidencyTier.DEVICE or not isinstance(ref, DeviceKVRef):
                raise TypeError("owned device segment has an invalid backend ref")
            self._allocator.free(ref.indices)
            if accounted:
                with self._accounting_lock:
                    self._owned_device_tokens -= len(tokens)
            released = True

        backend_ref = DeviceKVRef(indices=destination)
        try:
            self._allocator.get_kvcache().move_kv_cache(
                destination, source_indices
            )
            identity = token_ids_hash(tokens)
            with self._accounting_lock:
                self._owned_device_tokens += len(tokens)
            accounted = True
            handle = self._manager.register_segment(
                key=KVSegmentKey(
                    content_hash=content_hash or identity,
                    token_hash=identity,
                    token_count=len(tokens),
                    model_id=self._model_id,
                    cache_dtype=self._cache_dtype,
                    kind=SegmentKind.MIDDLE,
                ),
                token_ids=tokens,
                source_start=source_start,
                residency=ResidencyTier.DEVICE,
                backend_ref=backend_ref,
                release_backend=release_owned,
            )
            if handle is None:
                raise RuntimeError("KVCOMM core rejected device materialization")
            return handle
        except Exception:
            release_owned(backend_ref, ResidencyTier.DEVICE)
            raise

    def materialize_host(
        self,
        *,
        token_ids: Sequence[int],
        source_indices: torch.Tensor,
        source_start: int,
        content_hash: str | None = None,
    ) -> KVSegmentHandle:
        """Store a synchronous host copy when no owned device slots fit."""

        tokens = tuple(int(token) for token in token_ids)
        if not tokens:
            raise ValueError("cannot materialize an empty host segment")
        if source_start < 0:
            raise ValueError("source_start must be non-negative")
        if source_indices.ndim != 1 or len(source_indices) != len(tokens):
            raise ValueError(
                "source_indices must be 1-D and match the segment token count"
            )
        payload = self._allocator.get_cpu_copy(source_indices)
        identity = token_ids_hash(tokens)
        handle = self._manager.register_segment(
            key=KVSegmentKey(
                content_hash=content_hash or identity,
                token_hash=identity,
                token_count=len(tokens),
                model_id=self._model_id,
                cache_dtype=self._cache_dtype,
                kind=SegmentKind.MIDDLE,
            ),
            token_ids=tokens,
            source_start=source_start,
            residency=ResidencyTier.HOST,
            backend_ref=HostKVRef(payload=payload),
        )
        if handle is None:
            raise RuntimeError("KVCOMM core rejected host materialization")
        return handle


class RadixKVTransferBackend:
    """Production adapter for SGLang's all-layer KV cache copy primitive."""

    def __init__(
        self,
        *,
        allocator: KVPoolAllocator,
        target_indices: Callable[[int, int], torch.Tensor],
        dense_prefill: Callable[[int, int, str], None],
        rope: RoPEConfig,
    ) -> None:
        self._allocator = allocator
        self._target_indices = target_indices
        self._dense_prefill = dense_prefill
        self._rope = rope

    def dense_prefill(
        self, *, target_start: int, length: int, reason: str
    ) -> None:
        self._dense_prefill(target_start, length, reason)

    def copy_and_rotate(
        self,
        *,
        source_ref: object,
        source_offset: int,
        target_start: int,
        length: int,
        rope_delta: int,
    ) -> tuple[int, int, int]:
        target_indices = self._target_indices(target_start, length)
        if len(target_indices) != length:
            raise ValueError("physical target KV index slice length mismatch")

        kvcache = self._allocator.get_kvcache()
        if isinstance(source_ref, DeviceKVRef):
            source_indices = source_ref.indices[
                source_offset : source_offset + length
            ]
            if len(source_indices) != length:
                raise ValueError("physical source KV index slice length mismatch")
            kvcache.move_kv_cache(target_indices, source_indices)
        elif isinstance(source_ref, HostKVRef):
            if source_offset != 0:
                raise ValueError("host KV copy does not support a source offset")
            self._allocator.load_cpu_copy(source_ref.payload, target_indices)
        else:
            raise TypeError("KV transfer requires a device or host KV reference")
        self._rotate_all_copied_keys(
            kvcache=kvcache,
            target_indices=target_indices,
            rope_delta=rope_delta,
        )
        return length, length, length

    def _rotate_all_copied_keys(
        self,
        *,
        kvcache: Any,
        target_indices: torch.Tensor,
        rope_delta: int,
    ) -> None:
        rotary_dim = self._rope.rotary_dim
        if rotary_dim == 0 or rope_delta == 0 or len(target_indices) == 0:
            return
        device = target_indices.device
        inverse_frequency = 1.0 / (
            self._rope.base
            ** (
                torch.arange(
                    0,
                    rotary_dim,
                    2,
                    dtype=torch.float32,
                    device=device,
                )
                / rotary_dim
            )
        )
        delta = torch.full(
            (len(target_indices),),
            int(rope_delta),
            dtype=torch.float32,
            device=device,
        )
        frequencies = torch.einsum("i,j->ij", delta, inverse_frequency)
        cosine = frequencies.cos()
        sine = frequencies.sin()
        flat_indices = target_indices.reshape(-1).long()

        for layer_id in range(kvcache.layer_num):
            key_buffer = kvcache.get_key_buffer(layer_id)
            selected = key_buffer[flat_indices]
            if selected.shape[-1] < rotary_dim:
                raise ValueError("KV head dimension is smaller than rotary_dim")
            rotary = selected[..., :rotary_dim]
            rotated = apply_rotary_emb(
                rotary,
                cosine,
                sine,
                self._rope.is_neox_style,
            )
            if rotary_dim == selected.shape[-1]:
                key_buffer[flat_indices] = rotated
            else:
                key_buffer[flat_indices, ..., :rotary_dim] = rotated
