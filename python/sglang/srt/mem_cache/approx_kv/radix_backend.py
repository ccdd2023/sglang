from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

import torch

from sglang.srt.layers.rotary_embedding.utils import apply_rotary_emb

from .store import ResidencyLoadResult
from .types import KVSegmentHandle, ResidencyTier


@dataclass(frozen=True)
class DeviceKVRef:
    indices: torch.Tensor


@dataclass(frozen=True)
class HostKVRef:
    payload: Any


@dataclass(frozen=True)
class AnchorDeltaRef:
    key_deltas: tuple[torch.Tensor, ...]
    value_deltas: tuple[torch.Tensor, ...]

    def __post_init__(self) -> None:
        if not self.key_deltas or not self.value_deltas:
            raise ValueError("anchor deltas must contain every model layer")
        if len(self.key_deltas) != len(self.value_deltas):
            raise ValueError("anchor K/V layer counts must match")


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

    def load_cpu_copy(
        self,
        payload: Any,
        indices: torch.Tensor,
    ) -> None: ...


class AllocatorResidencyLoader:
    def __init__(self, allocator: KVPoolAllocator) -> None:
        self._allocator = allocator

    def load(
        self,
        handle: KVSegmentHandle,
        target_tier: ResidencyTier,
    ) -> ResidencyLoadResult:
        if target_tier != ResidencyTier.DEVICE:
            raise NotImplementedError(
                "the Radix allocator adapter loads only to device"
            )
        if not isinstance(handle.backend_ref, HostKVRef):
            raise TypeError("host-resident handle must carry HostKVRef")
        indices = self._allocator.alloc(len(handle.token_ids))
        if indices is None or len(indices) != len(handle.token_ids):
            if indices is not None:
                self._allocator.free(indices)
            raise MemoryError("unable to allocate device slots for approximate KV")
        try:
            self._allocator.load_cpu_copy(
                handle.backend_ref.payload,
                indices,
            )
        except Exception:
            self._allocator.free(indices)
            raise
        return ResidencyLoadResult(
            backend_ref=DeviceKVRef(indices=indices),
            release_backend=self._release_device_ref,
        )

    def _release_device_ref(
        self,
        backend_ref: object,
        residency: ResidencyTier,
    ) -> None:
        if residency != ResidencyTier.DEVICE or not isinstance(
            backend_ref,
            DeviceKVRef,
        ):
            raise TypeError("allocator releaser received a non-device KV ref")
        self._allocator.free(backend_ref.indices)


class RadixKVTransferBackend:
    def __init__(
        self,
        *,
        allocator: KVPoolAllocator,
        target_indices: Callable[[int, int], torch.Tensor],
        dense_prefill: Callable[[int, int, str], None],
        rope: RoPEConfig,
        use_native_copy: bool = True,
    ) -> None:
        self._allocator = allocator
        self._target_indices = target_indices
        self._dense_prefill = dense_prefill
        self._rope = rope
        self._use_native_copy = use_native_copy

    def dense_prefill(
        self,
        *,
        target_start: int,
        length: int,
        reason: str,
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
        source_indices = source_ref.indices[source_offset : source_offset + length]
        target_indices = self._target_indices(target_start, length)
        self._validate_indices(source_indices, target_indices, length)

        kvcache = self._allocator.get_kvcache()
        self._copy_all_layers(
            kvcache=kvcache,
            target_indices=target_indices,
            source_indices=source_indices,
        )
        self._rotate_all_copied_keys(
            kvcache=kvcache,
            target_indices=target_indices,
            rope_delta=rope_delta,
        )
        return length, length, length

    def reconstruct_and_rotate(
        self,
        *,
        base_ref: object,
        anchor_refs: tuple[object, ...],
        weights: tuple[float, ...],
        source_offset: int,
        target_start: int,
        length: int,
        rope_delta: int,
    ) -> tuple[int, int, int]:
        if not isinstance(base_ref, DeviceKVRef):
            raise TypeError("anchor reconstruction requires a device base")
        if len(anchor_refs) != len(weights):
            raise ValueError("anchor reference and weight counts must match")
        if not all(
            isinstance(anchor_ref, AnchorDeltaRef) for anchor_ref in anchor_refs
        ):
            raise TypeError("anchor reconstruction requires AnchorDeltaRef entries")

        source_indices = base_ref.indices[source_offset : source_offset + length]
        target_indices = self._target_indices(target_start, length)
        self._validate_indices(source_indices, target_indices, length)
        kvcache = self._allocator.get_kvcache()
        self._copy_all_layers(
            kvcache=kvcache,
            target_indices=target_indices,
            source_indices=source_indices,
        )

        flat_target = target_indices.reshape(-1).long()
        for layer_id in range(kvcache.layer_num):
            key_buffer = kvcache.get_key_buffer(layer_id)
            value_buffer = kvcache.get_value_buffer(layer_id)
            for anchor_ref, weight in zip(anchor_refs, weights):
                key_delta = anchor_ref.key_deltas[layer_id][
                    source_offset : source_offset + length
                ]
                value_delta = anchor_ref.value_deltas[layer_id][
                    source_offset : source_offset + length
                ]
                if len(key_delta) != length or len(value_delta) != length:
                    raise ValueError("anchor delta slice length does not match target")
                key_buffer[flat_target] += key_delta * weight
                value_buffer[flat_target] += value_delta * weight

        self._rotate_all_copied_keys(
            kvcache=kvcache,
            target_indices=target_indices,
            rope_delta=rope_delta,
        )
        return length, length, length

    @staticmethod
    def _validate_indices(
        source_indices: torch.Tensor,
        target_indices: torch.Tensor,
        length: int,
    ) -> None:
        if len(source_indices) != length or len(target_indices) != length:
            raise ValueError("physical KV index slice length mismatch")

    def _copy_all_layers(
        self,
        *,
        kvcache: Any,
        target_indices: torch.Tensor,
        source_indices: torch.Tensor,
    ) -> None:
        if not self._use_native_copy:
            kvcache.move_kv_cache(target_indices, source_indices)
            return
        target = target_indices.reshape(-1).long()
        source = source_indices.reshape(-1).long()
        for layer_id in range(kvcache.layer_num):
            key_buffer = kvcache.get_key_buffer(layer_id)
            value_buffer = kvcache.get_value_buffer(layer_id)
            key_buffer[target] = key_buffer[source]
            value_buffer[target] = value_buffer[source]

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
