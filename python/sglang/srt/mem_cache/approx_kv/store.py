from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from .types import (
    KVSegmentHandle,
    KVSegmentKey,
    ResidencyTier,
    token_ids_hash,
)

ReleaseBackend = Callable[[Any, ResidencyTier], None]


@dataclass(frozen=True)
class ResidencyLoadResult:
    backend_ref: Any
    release_backend: ReleaseBackend | None = None


class ResidencyLoader(Protocol):
    def load(
        self,
        handle: KVSegmentHandle,
        target_tier: ResidencyTier,
    ) -> Any | ResidencyLoadResult: ...


@dataclass
class _Record:
    key: KVSegmentKey
    generation: int
    residency: ResidencyTier
    source_start: int
    token_ids: tuple[int, ...]
    backend_ref: Any
    release_backend: ReleaseBackend | None
    last_access_s: float


@dataclass(frozen=True)
class ApproxKVLease:
    lease_id: str
    key: KVSegmentKey
    generation: int
    expires_at_s: float


class ApproxKVSegmentStore:
    def __init__(self, max_records: int = 4096) -> None:
        if max_records <= 0:
            raise ValueError("max_records must be positive")
        self._max_records = max_records
        self._records: OrderedDict[KVSegmentKey, _Record] = OrderedDict()
        self._generation: dict[KVSegmentKey, int] = {}
        self._leases: dict[str, ApproxKVLease] = {}
        self._lock = threading.RLock()

    def _handle(self, record: _Record) -> KVSegmentHandle:
        return KVSegmentHandle(
            key=record.key,
            generation=record.generation,
            residency=record.residency,
            source_start=record.source_start,
            token_ids=record.token_ids,
            backend_ref=record.backend_ref,
        )

    def register(
        self,
        *,
        key: KVSegmentKey,
        token_ids: tuple[int, ...] | list[int],
        source_start: int,
        residency: ResidencyTier,
        backend_ref: Any,
        release_backend: ReleaseBackend | None = None,
    ) -> KVSegmentHandle:
        tokens = tuple(int(token) for token in token_ids)
        if source_start < 0:
            raise ValueError("source_start must be non-negative")
        if len(tokens) != key.token_count:
            raise ValueError("token_count does not match registered tokens")
        if token_ids_hash(tokens) != key.token_hash:
            raise ValueError("token_hash does not match registered tokens")

        with self._lock:
            previous = self._records.get(key)
            if previous is not None and self._is_leased(
                key,
                previous.generation,
            ):
                raise RuntimeError("cannot replace a leased KV segment")
            generation = self._generation.get(key, 0) + 1
            self._generation[key] = generation
            record = _Record(
                key=key,
                generation=generation,
                residency=residency,
                source_start=source_start,
                token_ids=tokens,
                backend_ref=backend_ref,
                release_backend=release_backend,
                last_access_s=time.monotonic(),
            )
            if previous is not None:
                self._dispose_record(previous)
            self._records[key] = record
            self._records.move_to_end(key)
            try:
                self._evict_unleased_if_needed(protected_key=key)
            except RuntimeError:
                del self._records[key]
                self._dispose_record(record)
                raise
            return self._handle(record)

    def lookup(self, key: KVSegmentKey) -> KVSegmentHandle | None:
        with self._lock:
            record = self._records.get(key)
            if record is None:
                return None
            record.last_access_s = time.monotonic()
            self._records.move_to_end(key)
            return self._handle(record)

    def find_by_content_hash(
        self,
        content_hash: str,
    ) -> KVSegmentHandle | None:
        if not content_hash:
            raise ValueError("content_hash must be non-empty")
        with self._lock:
            matches = [
                record
                for key, record in self._records.items()
                if key.content_hash == content_hash
            ]
            if not matches:
                return None
            record = max(
                matches,
                key=lambda item: (
                    item.generation,
                    item.last_access_s,
                ),
            )
            record.last_access_s = time.monotonic()
            self._records.move_to_end(record.key)
            return self._handle(record)

    def is_current(self, handle: KVSegmentHandle) -> bool:
        with self._lock:
            record = self._records.get(handle.key)
            return record is not None and record.generation == handle.generation

    def pin(self, handle: KVSegmentHandle, ttl_s: float) -> ApproxKVLease:
        if ttl_s <= 0:
            raise ValueError("ttl_s must be positive")
        with self._lock:
            if not self.is_current(handle):
                raise KeyError("cannot pin a stale or missing segment")
            lease = ApproxKVLease(
                lease_id=uuid.uuid4().hex,
                key=handle.key,
                generation=handle.generation,
                expires_at_s=time.monotonic() + ttl_s,
            )
            self._leases[lease.lease_id] = lease
            return lease

    def unpin(self, lease: ApproxKVLease | str) -> bool:
        lease_id = lease if isinstance(lease, str) else lease.lease_id
        with self._lock:
            return self._leases.pop(lease_id, None) is not None

    def gc_expired_leases(self, now_s: float | None = None) -> int:
        now_s = time.monotonic() if now_s is None else now_s
        with self._lock:
            expired = [
                lease_id
                for lease_id, lease in self._leases.items()
                if lease.expires_at_s <= now_s
            ]
            for lease_id in expired:
                del self._leases[lease_id]
            self._evict_unleased_if_needed()
            return len(expired)

    def ensure_resident(
        self,
        handle: KVSegmentHandle,
        target_tier: ResidencyTier,
        loader: ResidencyLoader,
    ) -> KVSegmentHandle:
        with self._lock:
            if not self.is_current(handle):
                raise KeyError("cannot load a stale or missing segment")
            record = self._records[handle.key]
            if record.residency == target_tier:
                return self._handle(record)

        loaded = loader.load(handle, target_tier)
        if isinstance(loaded, ResidencyLoadResult):
            backend_ref = loaded.backend_ref
            release_backend = loaded.release_backend
        else:
            backend_ref = loaded
            release_backend = None

        with self._lock:
            record = self._records.get(handle.key)
            if record is None or record.generation != handle.generation:
                if release_backend is not None:
                    release_backend(backend_ref, target_tier)
                raise KeyError("segment changed while residency load was in flight")
            old_backend_ref = record.backend_ref
            old_residency = record.residency
            old_release_backend = record.release_backend
            record.backend_ref = backend_ref
            record.residency = target_tier
            record.release_backend = release_backend
            record.last_access_s = time.monotonic()
            updated = self._handle(record)

        if old_release_backend is not None:
            old_release_backend(old_backend_ref, old_residency)
        return updated

    def release(self, handle: KVSegmentHandle) -> bool:
        with self._lock:
            if not self.is_current(handle) or self._is_leased(
                handle.key,
                handle.generation,
            ):
                return False
            record = self._records.pop(handle.key)
            self._dispose_record(record)
            return True

    def reset(self) -> None:
        with self._lock:
            for record in self._records.values():
                self._dispose_record(record)
            self._records.clear()
            self._leases.clear()

    @property
    def record_count(self) -> int:
        with self._lock:
            return len(self._records)

    @property
    def lease_count(self) -> int:
        with self._lock:
            return len(self._leases)

    def _is_leased(self, key: KVSegmentKey, generation: int) -> bool:
        return any(
            lease.key == key and lease.generation == generation
            for lease in self._leases.values()
        )

    def _evict_unleased_if_needed(
        self,
        protected_key: KVSegmentKey | None = None,
    ) -> None:
        while len(self._records) > self._max_records:
            victim = next(
                (
                    key
                    for key, record in self._records.items()
                    if key != protected_key
                    and not self._is_leased(key, record.generation)
                ),
                None,
            )
            if victim is None:
                raise RuntimeError(
                    "approximate KV segment store capacity is fully pinned"
                )
            record = self._records.pop(victim)
            self._dispose_record(record)

    @staticmethod
    def _dispose_record(record: _Record) -> None:
        if record.release_backend is not None:
            record.release_backend(record.backend_ref, record.residency)
