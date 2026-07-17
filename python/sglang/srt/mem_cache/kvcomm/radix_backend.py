from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

import torch

from sglang.srt.layers.rotary_embedding.utils import apply_rotary_emb
from sglang.srt.mem_cache.kvcomm.types import (
    KVSegmentHandle,
    ResidencyTier,
)


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

    def load_cpu_copy(self, payload: Any, indices: torch.Tensor) -> None: ...


class AllocatorResidencyLoader:
    """Loads a host KV payload into newly allocated device slots."""

    def __init__(self, allocator: KVPoolAllocator) -> None:
        self._allocator = allocator

    def load(
        self,
        handle: KVSegmentHandle,
        target_tier: ResidencyTier,
    ) -> DeviceKVRef:
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
        return DeviceKVRef(indices=indices)


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
        if not isinstance(source_ref, DeviceKVRef):
            raise TypeError("device transfer requires DeviceKVRef")
        source_indices = source_ref.indices[
            source_offset : source_offset + length
        ]
        target_indices = self._target_indices(target_start, length)
        if len(source_indices) != length or len(target_indices) != length:
            raise ValueError("physical KV index slice length mismatch")

        kvcache = self._allocator.get_kvcache()
        kvcache.move_kv_cache(target_indices, source_indices)
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
