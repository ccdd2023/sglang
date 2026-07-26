from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol

import torch

from sglang.srt.layers.rotary_embedding.utils import apply_rotary_emb

from .store import ResidencyLoadResult
from .types import (
    KVLayerTransferResult,
    KVSegmentHandle,
    ResidencyTier,
)


@dataclass(frozen=True)
class DeviceKVRef:
    indices: torch.Tensor


@dataclass(frozen=True)
class CPUKVRef:
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

    def get_cpu_copy(
        self,
        indices: torch.Tensor,
        mamba_indices: torch.Tensor | None = None,
    ) -> Any: ...

    def load_cpu_copy(
        self,
        payload: Any,
        indices: torch.Tensor,
        mamba_indices: torch.Tensor | None = None,
    ) -> None: ...


class AllocatorCPUResidencyBackend:
    def __init__(
        self,
        allocator: KVPoolAllocator,
        allocate_slots: Callable[[int], Any] | None = None,
    ) -> None:
        self._allocator = allocator
        self._allocate_slots = allocate_slots or allocator.alloc

    def export_to_host(self, device_ref: DeviceKVRef) -> ResidencyLoadResult:
        started = time.perf_counter()
        payload = self._allocator.get_cpu_copy(device_ref.indices)
        return ResidencyLoadResult(
            backend_ref=CPUKVRef(payload),
            release_backend=self.release_host,
            num_tokens=len(device_ref.indices),
            bytes_transferred=_payload_nbytes(payload),
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )

    def load(
        self,
        handle: KVSegmentHandle,
        target_tier: ResidencyTier,
    ) -> ResidencyLoadResult:
        return self._load(
            handle,
            target_tier,
            allocate_slots=self._allocate_slots,
        )

    def load_for_rollback(
        self,
        handle: KVSegmentHandle,
        target_tier: ResidencyTier,
    ) -> ResidencyLoadResult:
        return self._load(
            handle,
            target_tier,
            allocate_slots=self._allocator.alloc,
        )

    def _load(
        self,
        handle: KVSegmentHandle,
        target_tier: ResidencyTier,
        *,
        allocate_slots: Callable[[int], Any],
    ) -> ResidencyLoadResult:
        if target_tier != ResidencyTier.DEVICE:
            raise NotImplementedError("allocator backend loads only to device")
        if not isinstance(handle.backend_ref, CPUKVRef):
            raise TypeError("host-resident handle must carry CPUKVRef")
        indices = allocate_slots(len(handle.token_ids))
        if indices is None or len(indices) != len(handle.token_ids):
            if indices is not None:
                self._allocator.free(indices)
            raise MemoryError("unable to allocate device slots for approximate KV")
        started = time.perf_counter()
        try:
            self._allocator.load_cpu_copy(
                handle.backend_ref.payload,
                indices,
            )
        except Exception:
            self._allocator.free(indices)
            raise
        return ResidencyLoadResult(
            backend_ref=DeviceKVRef(indices),
            release_backend=self.release_device,
            num_tokens=len(handle.token_ids),
            bytes_transferred=_payload_nbytes(handle.backend_ref.payload),
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )

    def release_device(
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

    @staticmethod
    def release_host(
        backend_ref: object,
        residency: ResidencyTier,
    ) -> None:
        if residency != ResidencyTier.HOST or not isinstance(
            backend_ref,
            CPUKVRef,
        ):
            raise TypeError("allocator releaser received a non-host KV ref")


def _payload_nbytes(payload: Any) -> int:
    if isinstance(payload, torch.Tensor):
        return payload.numel() * payload.element_size()
    if isinstance(payload, dict):
        return sum(_payload_nbytes(value) for value in payload.values())
    if isinstance(payload, (list, tuple)):
        return sum(_payload_nbytes(value) for value in payload)
    nbytes = getattr(payload, "nbytes", None)
    return int(nbytes) if nbytes is not None else 0


class RadixKVTransferBackend:
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
    ) -> KVLayerTransferResult:
        if not isinstance(source_ref, DeviceKVRef):
            raise TypeError("device transfer requires DeviceKVRef")
        source_indices = source_ref.indices[source_offset : source_offset + length]
        target_indices = self._target_indices(target_start, length)
        self._validate_indices(source_indices, target_indices, length)

        kvcache = self._allocator.get_kvcache()
        copy_start = time.perf_counter()
        kvcache.move_kv_cache(target_indices, source_indices)
        copy_ms = (time.perf_counter() - copy_start) * 1000

        rope_start = time.perf_counter()
        self._rotate_all_copied_keys(
            kvcache=kvcache,
            target_indices=target_indices,
            rope_delta=rope_delta,
        )
        rope_ms = (time.perf_counter() - rope_start) * 1000
        return KVLayerTransferResult(
            copied_k_tokens=length,
            rotated_k_tokens=length,
            copied_v_tokens=length,
            copy_ms=copy_ms,
            rope_ms=rope_ms,
        )

    @staticmethod
    def _validate_indices(
        source_indices: torch.Tensor,
        target_indices: torch.Tensor,
        length: int,
    ) -> None:
        if len(source_indices) != length or len(target_indices) != length:
            raise ValueError("physical KV index slice length mismatch")

    def copy_and_rotate_layer(
        self,
        *,
        layer_id: int,
        source_ref: object,
        source_offset: int,
        target_start: int,
        length: int,
        rope_delta: int,
    ) -> KVLayerTransferResult:
        """Copy+RoPE-correct exactly one layer's worth of body KV.

        ``copy_and_rotate`` fuses ``kvcache.move_kv_cache`` (itself an
        all-layers-at-once physical move) with a RoPE-only correction loop
        that also walks every layer before returning. That shape is correct
        for the plain R0 reuse path, where the whole body is committed in
        one step, but it is unusable for EPIC's per-layer interleaving: the
        leading-k tokens for layer L must be genuinely recomputed *before*
        layer L's body KV is written, and layer L+1's body must not be
        touched until layer L+1's leading-k recompute has run. This method
        therefore only ever touches ``layer_id``, using the same
        ``get_key_buffer``/``get_value_buffer`` accessors already used by
        ``_rotate_all_copied_keys`` (no new physical primitive is
        introduced; only the calling shape is per-layer).
        """
        if not isinstance(source_ref, DeviceKVRef):
            raise TypeError("device transfer requires DeviceKVRef")
        source_indices = source_ref.indices[source_offset : source_offset + length]
        target_indices = self._target_indices(target_start, length)
        self._validate_indices(source_indices, target_indices, length)

        kvcache = self._allocator.get_kvcache()
        copy_start = time.perf_counter()
        key_buffer = kvcache.get_key_buffer(layer_id)
        value_buffer = kvcache.get_value_buffer(layer_id)
        key_buffer[target_indices] = key_buffer[source_indices]
        value_buffer[target_indices] = value_buffer[source_indices]
        copy_ms = (time.perf_counter() - copy_start) * 1000

        rope_start = time.perf_counter()
        self._rotate_layer_copied_keys(
            kvcache=kvcache,
            layer_id=layer_id,
            target_indices=target_indices,
            rope_delta=rope_delta,
        )
        rope_ms = (time.perf_counter() - rope_start) * 1000
        return KVLayerTransferResult(
            copied_k_tokens=length,
            rotated_k_tokens=length,
            copied_v_tokens=length,
            copy_ms=copy_ms,
            rope_ms=rope_ms,
        )

    def _rope_cos_sin(
        self,
        *,
        rope_delta: int,
        num_indices: int,
        device: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rotary_dim = self._rope.rotary_dim
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
            (num_indices,),
            int(rope_delta),
            dtype=torch.float32,
            device=device,
        )
        frequencies = torch.einsum("i,j->ij", delta, inverse_frequency)
        return frequencies.cos(), frequencies.sin()

    def _rotate_one_layer_keys(
        self,
        *,
        kvcache: Any,
        layer_id: int,
        flat_indices: torch.Tensor,
        cosine: torch.Tensor,
        sine: torch.Tensor,
    ) -> None:
        rotary_dim = self._rope.rotary_dim
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

    def _rotate_layer_copied_keys(
        self,
        *,
        kvcache: Any,
        layer_id: int,
        target_indices: torch.Tensor,
        rope_delta: int,
    ) -> None:
        rotary_dim = self._rope.rotary_dim
        if rotary_dim == 0 or rope_delta == 0 or len(target_indices) == 0:
            return
        flat_indices = target_indices.reshape(-1).long()
        cosine, sine = self._rope_cos_sin(
            rope_delta=rope_delta,
            num_indices=len(flat_indices),
            device=(
                flat_indices.device
                if flat_indices.device.type != "meta"
                else kvcache.get_key_buffer(layer_id).device
            ),
        )
        self._rotate_one_layer_keys(
            kvcache=kvcache,
            layer_id=layer_id,
            flat_indices=flat_indices,
            cosine=cosine,
            sine=sine,
        )

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
        flat_indices = target_indices.reshape(-1).long()
        first_key_buffer = kvcache.get_key_buffer(0)
        device = first_key_buffer.device
        cosine, sine = self._rope_cos_sin(
            rope_delta=rope_delta,
            num_indices=len(flat_indices),
            device=device,
        )
        for layer_id in range(kvcache.layer_num):
            self._rotate_one_layer_keys(
                kvcache=kvcache,
                layer_id=layer_id,
                flat_indices=flat_indices,
                cosine=cosine,
                sine=sine,
            )
