from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

import torch

from sglang.srt.mem_cache.kvcomm.manager import KVCommManager
from sglang.srt.mem_cache.kvcomm.radix_backend import (
    AllocatorResidencyLoader,
    DeviceKVRef,
    HostKVRef,
)
from sglang.srt.mem_cache.kvcomm.types import (
    KVPrefetchHint,
    KVSegmentHandle,
    KVSegmentKey,
    ResidencyTier,
    SegmentKind,
    token_ids_hash,
)
from sglang.srt.mem_cache.kvcomm_prefetch.coordinator import (
    KVPrefetchCoordinator,
    PrefetchResult,
)


class MiddleKVAllocator(Protocol):
    def alloc(self, need_size: int) -> torch.Tensor | None: ...

    def free(self, indices: torch.Tensor) -> None: ...

    def get_kvcache(self) -> Any: ...

    def get_cpu_copy(self, indices: torch.Tensor) -> Any: ...

    def load_cpu_copy(self, payload: Any, indices: torch.Tensor) -> None: ...


class MiddleKVPrefetchError(RuntimeError):
    pass


@dataclass
class PrefetchTicket:
    """Synchronous ticket whose API can later be backed by CUDA events."""

    key: KVSegmentKey
    result: PrefetchResult
    _manager: KVCommManager = field(repr=False)
    _coordinator: KVPrefetchCoordinator = field(repr=False)
    _released: bool = field(default=False, init=False, repr=False)

    @property
    def done(self) -> bool:
        return True

    @property
    def successful(self) -> bool:
        handle = self._manager.store.lookup(self.key)
        return (
            not self.result.disabled
            and self.result.failed == 0
            and self.result.store_misses == 0
            and handle is not None
            and handle.residency == ResidencyTier.DEVICE
        )

    def wait(self, timeout_s: float | None = None) -> KVSegmentHandle:
        if timeout_s is not None and timeout_s < 0:
            raise ValueError("timeout_s must be non-negative")
        if self.result.disabled:
            raise MiddleKVPrefetchError("KV prefetch is disabled")
        if self.result.store_misses:
            raise MiddleKVPrefetchError(f"middle KV segment not found: {self.key}")
        if self.result.failed:
            details = "; ".join(self.result.failure_reasons)
            raise MiddleKVPrefetchError(f"middle KV prefetch failed: {details}")
        handle = self._manager.store.lookup(self.key)
        if handle is None or handle.residency != ResidencyTier.DEVICE:
            raise MiddleKVPrefetchError(
                "prefetch completed without a device-resident segment"
            )
        return handle

    def device_indices(self) -> torch.Tensor:
        handle = self.wait()
        if not isinstance(handle.backend_ref, DeviceKVRef):
            raise MiddleKVPrefetchError(
                "device-resident segment does not carry DeviceKVRef"
            )
        return handle.backend_ref.indices

    def release(self) -> bool:
        if self._released:
            return False
        self._released = True
        return self._coordinator.release(self.result) > 0

    def __enter__(self) -> "PrefetchTicket":
        self.wait()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()


class MiddleKVPrefetchAPI:
    """High-level export and prefetch interface for middle-of-request KV."""

    def __init__(
        self,
        *,
        manager: KVCommManager,
        allocator: MiddleKVAllocator,
        model_id: str,
        cache_dtype: str,
        lease_ttl_s: float = 60.0,
    ) -> None:
        if not manager.config.core_enabled:
            raise ValueError("KVCOMM core must be enabled")
        if not manager.config.prefetch_enabled:
            raise ValueError("KV prefetch must be enabled")
        if not model_id or not cache_dtype:
            raise ValueError("model_id and cache_dtype must be non-empty")
        self.manager = manager
        self.allocator = allocator
        self.model_id = model_id
        self.cache_dtype = cache_dtype
        self.coordinator = KVPrefetchCoordinator(
            manager=manager,
            loader=AllocatorResidencyLoader(allocator),
            lease_ttl_s=lease_ttl_s,
        )

    def export_middle_kv(
        self,
        *,
        token_ids: Sequence[int],
        kv_indices: torch.Tensor,
        source_start: int,
        content_hash: str | None = None,
    ) -> KVSegmentHandle:
        """Copy a device KV slice to host and register it as a middle segment.

        The source device slots remain owned by the request/RadixCache. This
        method does not free or pin them.
        """

        tokens = tuple(int(token) for token in token_ids)
        if not tokens:
            raise ValueError("cannot export an empty middle KV segment")
        if kv_indices.ndim != 1 or len(kv_indices) != len(tokens):
            raise ValueError("kv_indices must be 1-D and match token_ids length")
        key = self._key(tokens=tokens, content_hash=content_hash)
        host_payload = self.allocator.get_cpu_copy(kv_indices)
        handle = self.manager.register_segment(
            key=key,
            token_ids=tokens,
            source_start=source_start,
            residency=ResidencyTier.HOST,
            backend_ref=HostKVRef(host_payload),
        )
        if handle is None:
            raise MiddleKVPrefetchError("KVCOMM core rejected middle KV export")
        return handle

    def register_host_middle_kv(
        self,
        *,
        token_ids: Sequence[int],
        host_payload: Any,
        source_start: int,
        content_hash: str | None = None,
    ) -> KVSegmentHandle:
        """Register a caller-precomputed host payload without a device export."""

        tokens = tuple(int(token) for token in token_ids)
        if not tokens:
            raise ValueError("cannot register an empty middle KV segment")
        handle = self.manager.register_segment(
            key=self._key(tokens=tokens, content_hash=content_hash),
            token_ids=tokens,
            source_start=source_start,
            residency=ResidencyTier.HOST,
            backend_ref=HostKVRef(host_payload),
        )
        if handle is None:
            raise MiddleKVPrefetchError("KVCOMM core rejected middle KV payload")
        return handle

    def prefetch(
        self,
        key: KVSegmentKey,
        *,
        deadline_s: float | None = None,
        priority: int = 0,
    ) -> PrefetchTicket:
        if key.kind != SegmentKind.MIDDLE:
            raise ValueError("MiddleKVPrefetchAPI accepts only middle segments")
        result = self.coordinator.prefetch(
            (
                KVPrefetchHint(
                    key=key,
                    target_tier=ResidencyTier.DEVICE,
                    deadline_s=deadline_s,
                    priority=priority,
                ),
            )
        )
        return PrefetchTicket(
            key=key,
            result=result,
            _manager=self.manager,
            _coordinator=self.coordinator,
        )

    def drop(self, handle_or_key: KVSegmentHandle | KVSegmentKey) -> bool:
        handle = (
            handle_or_key
            if isinstance(handle_or_key, KVSegmentHandle)
            else self.manager.store.lookup(handle_or_key)
        )
        return False if handle is None else self.manager.store.release(handle)

    def _key(
        self, *, tokens: tuple[int, ...], content_hash: str | None
    ) -> KVSegmentKey:
        identity = token_ids_hash(tokens)
        return KVSegmentKey(
            content_hash=content_hash or identity,
            token_hash=identity,
            token_count=len(tokens),
            model_id=self.model_id,
            cache_dtype=self.cache_dtype,
            kind=SegmentKind.MIDDLE,
        )
