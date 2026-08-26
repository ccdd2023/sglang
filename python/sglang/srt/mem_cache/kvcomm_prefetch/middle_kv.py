from __future__ import annotations

from typing import Any, Protocol, Sequence

import torch

from sglang.srt.mem_cache.kvcomm.manager import KVCommManager
from sglang.srt.mem_cache.kvcomm.radix_backend import (
    AllocatorResidencyLoader,
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
)
from sglang.srt.mem_cache.kvcomm_prefetch.scheduler import (
    AsyncKVPrefetchScheduler,
    MiddleKVPrefetchError,
    PrefetchTicket,
)


class MiddleKVAllocator(Protocol):
    def alloc(self, need_size: int) -> torch.Tensor | None: ...

    def free(self, indices: torch.Tensor) -> None: ...

    def get_kvcache(self) -> Any: ...

    def get_cpu_copy(self, indices: torch.Tensor) -> Any: ...

    def load_cpu_copy(self, payload: Any, indices: torch.Tensor) -> None: ...


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
        worker_count: int = 1,
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
        self.scheduler = AsyncKVPrefetchScheduler(
            manager=manager,
            coordinator=self.coordinator,
            worker_count=worker_count,
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
        return self.scheduler.submit(
            KVPrefetchHint(
                key=key,
                target_tier=ResidencyTier.DEVICE,
                deadline_s=deadline_s,
                priority=priority,
            )
        )

    def close(self, *, wait: bool = True, cancel_pending: bool = False) -> None:
        self.scheduler.close(wait=wait, cancel_pending=cancel_pending)

    def __enter__(self) -> "MiddleKVPrefetchAPI":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

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
