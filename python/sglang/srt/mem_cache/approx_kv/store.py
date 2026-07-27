from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from sglang.srt.mem_cache.cross_store.allocator import CrossStoreResource
from sglang.srt.mem_cache.cross_store.event_clock import global_event_clock
from sglang.srt.mem_cache.cross_store.types import (
    CrossStoreKind,
    CrossStoreObject,
    CrossStoreTier,
    ObjectProvenance,
)

from .types import (
    KVSegmentHandle,
    KVSegmentKey,
    ResidencyTier,
    token_ids_hash,
)

ReleaseBackend = Callable[[Any, ResidencyTier], None]


class ApproxKVStoreCapacityError(RuntimeError):
    pass


@dataclass(frozen=True)
class ResidencyLoadResult:
    backend_ref: Any
    release_backend: ReleaseBackend | None = None
    release_previous: bool = True
    release_on_stale: ReleaseBackend | None = None
    num_tokens: int = 0
    bytes_transferred: int = 0
    duration_ms: float = 0.0


class ResidencyLoader(Protocol):
    def load(
        self,
        handle: KVSegmentHandle,
        target_tier: ResidencyTier,
    ) -> Any | ResidencyLoadResult: ...


class AsyncResidencyTransfer(Protocol):
    @property
    def done(self) -> bool: ...

    def wait(self, timeout_s: float | None = None) -> ResidencyLoadResult: ...

    def cancel(self) -> None: ...


class AsyncResidencyLoader(Protocol):
    def begin_load(
        self,
        handle: KVSegmentHandle,
        target_tier: ResidencyTier,
    ) -> AsyncResidencyTransfer: ...


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
    event_ordinal: int
    resident_bytes: int
    object_id: str
    object_kind: CrossStoreKind
    dependencies: frozenset[str]
    dense_cost_ms: float | None
    recovery_cost_ms: float | None
    next_use_ordinal: int | None
    retired: bool


@dataclass(frozen=True)
class ApproxKVLease:
    lease_id: str
    key: KVSegmentKey
    generation: int
    expires_at_s: float


class ApproxKVSegmentStore:
    def __init__(
        self,
        max_records: int = 4096,
        *,
        max_device_bytes: int | None = None,
        max_host_bytes: int | None = None,
        bytes_per_token: int = 1,
    ) -> None:
        if max_records <= 0 or bytes_per_token <= 0:
            raise ValueError("max_records and bytes_per_token must be positive")
        self._max_records = max_records
        self._max_device_bytes = max_device_bytes
        self._max_host_bytes = max_host_bytes
        self.bytes_per_token = bytes_per_token
        self._records: OrderedDict[KVSegmentKey, _Record] = OrderedDict()
        self._generation: dict[KVSegmentKey, int] = {}
        self._leases: dict[str, ApproxKVLease] = {}
        self._object_keys: dict[str, KVSegmentKey] = {}
        self._dependents: dict[str, set[str]] = {}
        self._device_owned_tokens = 0
        self._device_owned_bytes = 0
        self._host_owned_bytes = 0
        self._lock = threading.RLock()

    def configure_byte_limits(
        self,
        *,
        max_device_bytes: int | None = None,
        max_host_bytes: int | None = None,
    ) -> None:
        with self._lock:
            if self._records:
                raise RuntimeError("cannot reconfigure a non-empty KV store")
            self._max_device_bytes = max_device_bytes
            self._max_host_bytes = max_host_bytes

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
        resident_bytes: int | None = None,
        object_id: str | None = None,
        object_kind: CrossStoreKind = CrossStoreKind.PRECOMPUTED_ADAPTER,
        dependencies: frozenset[str] = frozenset(),
        dense_cost_ms: float | None = None,
        recovery_cost_ms: float | None = None,
        next_use_ordinal: int | None = None,
        retired: bool = False,
    ) -> KVSegmentHandle:
        tokens = tuple(int(token) for token in token_ids)
        if source_start < 0:
            raise ValueError("source_start must be non-negative")
        if len(tokens) != key.token_count:
            raise ValueError("token_count does not match registered tokens")
        if token_ids_hash(tokens) != key.token_hash:
            raise ValueError("token_hash does not match registered tokens")
        bytes_owned = (
            key.token_count * self.bytes_per_token
            if resident_bytes is None
            else resident_bytes
        )
        if bytes_owned <= 0:
            raise ValueError("resident_bytes must be positive")
        resolved_object_id = object_id or f"approx:{key.content_hash}"

        with self._lock:
            previous = self._records.get(key)
            if previous is not None and self._is_leased(
                key,
                previous.generation,
            ):
                if release_backend is not None:
                    release_backend(backend_ref, residency)
                raise RuntimeError("cannot replace a leased KV segment")
            if (
                previous is not None
                and previous.object_id != resolved_object_id
                and self._dependents.get(previous.object_id)
            ):
                if release_backend is not None:
                    release_backend(backend_ref, residency)
                raise RuntimeError(
                    "cannot replace an approximate object that still has " "dependents"
                )
            available_object_ids = set(self._object_keys)
            if previous is not None:
                available_object_ids.discard(previous.object_id)
            missing_dependencies = dependencies.difference(available_object_ids)
            if missing_dependencies:
                if release_backend is not None:
                    release_backend(backend_ref, residency)
                raise KeyError(
                    f"missing approximate dependencies: "
                    f"{sorted(missing_dependencies)}"
                )
            duplicate_key = self._object_keys.get(resolved_object_id)
            if duplicate_key is not None and duplicate_key != key:
                if release_backend is not None:
                    release_backend(backend_ref, residency)
                raise ValueError(
                    f"duplicate approximate object_id {resolved_object_id!r}"
                )
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
                event_ordinal=global_event_clock().tick(),
                resident_bytes=bytes_owned,
                object_id=resolved_object_id,
                object_kind=object_kind,
                dependencies=dependencies,
                dense_cost_ms=dense_cost_ms,
                recovery_cost_ms=recovery_cost_ms,
                next_use_ordinal=next_use_ordinal,
                retired=retired,
            )
            preserved_dependents = (
                set(self._dependents.get(previous.object_id, ()))
                if previous is not None and previous.object_id == resolved_object_id
                else set()
            )
            if previous is not None:
                self._remove_record(key, dispose=False)
            self._add_record(record)
            if preserved_dependents:
                self._dependents[record.object_id] = preserved_dependents
            try:
                self._evict_unleased_if_needed(protected_key=key)
            except RuntimeError:
                self._remove_record(key, dispose=True)
                if previous is not None:
                    self._add_record(previous)
                    if preserved_dependents:
                        self._dependents[previous.object_id] = preserved_dependents
                raise
            if previous is not None:
                self._dispose_record(previous)
            return self._handle(record)

    def lookup(self, key: KVSegmentKey) -> KVSegmentHandle | None:
        with self._lock:
            record = self._records.get(key)
            if record is None:
                return None
            record.last_access_s = time.monotonic()
            record.event_ordinal = global_event_clock().tick()
            self._records.move_to_end(key)
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
        updated, _ = self.load_resident(handle, target_tier, loader)
        return updated

    def load_resident(
        self,
        handle: KVSegmentHandle,
        target_tier: ResidencyTier,
        loader: ResidencyLoader,
    ) -> tuple[KVSegmentHandle, ResidencyLoadResult]:
        with self._lock:
            if not self.is_current(handle):
                raise KeyError("cannot load a stale or missing segment")
            record = self._records[handle.key]
            if record.residency == target_tier:
                return self._handle(record), ResidencyLoadResult(
                    backend_ref=record.backend_ref,
                    release_backend=record.release_backend,
                    release_previous=False,
                )

        loaded = loader.load(handle, target_tier)
        if isinstance(loaded, ResidencyLoadResult):
            result = loaded
        else:
            result = ResidencyLoadResult(backend_ref=loaded)

        return (
            self.commit_residency(
                handle,
                target_tier=target_tier,
                result=result,
            ),
            result,
        )

    def commit_residency(
        self,
        handle: KVSegmentHandle,
        *,
        target_tier: ResidencyTier,
        result: ResidencyLoadResult,
    ) -> KVSegmentHandle:
        with self._lock:
            record = self._records.get(handle.key)
            if (
                record is None
                or record.generation != handle.generation
                or record.residency != handle.residency
                or record.backend_ref is not handle.backend_ref
            ):
                release = result.release_on_stale or result.release_backend
                if release is not None:
                    release(result.backend_ref, target_tier)
                raise KeyError("segment or residency changed while load was in flight")
            old_backend_ref = record.backend_ref
            old_residency = record.residency
            old_release_backend = record.release_backend
            if old_residency != target_tier:
                if old_residency == ResidencyTier.DEVICE:
                    self._device_owned_tokens -= record.key.token_count
                    self._device_owned_bytes -= record.resident_bytes
                    self._host_owned_bytes += record.resident_bytes
                else:
                    self._host_owned_bytes -= record.resident_bytes
                    self._device_owned_tokens += record.key.token_count
                    self._device_owned_bytes += record.resident_bytes
            record.backend_ref = result.backend_ref
            record.residency = target_tier
            record.release_backend = result.release_backend
            record.last_access_s = time.monotonic()
            record.event_ordinal = global_event_clock().tick()
            updated = self._handle(record)

        if result.release_previous and old_release_backend is not None:
            old_release_backend(old_backend_ref, old_residency)
        return updated

    def release(self, handle: KVSegmentHandle) -> bool:
        with self._lock:
            if not self.is_current(handle) or self._is_leased(
                handle.key,
                handle.generation,
            ):
                return False
            object_id = self._records[handle.key].object_id
            if self._dependents.get(object_id):
                return False
            self._remove_record(handle.key, dispose=True)
            return True

    def reset(self) -> None:
        with self._lock:
            for record in self._records.values():
                self._dispose_record(record)
            self._records.clear()
            self._leases.clear()
            self._object_keys.clear()
            self._dependents.clear()
            self._device_owned_tokens = 0
            self._device_owned_bytes = 0
            self._host_owned_bytes = 0

    def handles(self) -> tuple[KVSegmentHandle, ...]:
        with self._lock:
            return tuple(self._handle(record) for record in self._records.values())

    def handle_for_object_id(self, object_id: str) -> KVSegmentHandle:
        with self._lock:
            key = self._object_keys.get(object_id)
            if key is None:
                raise KeyError(f"missing approximate object {object_id!r}")
            return self._handle(self._records[key])

    @property
    def device_owned_tokens(self) -> int:
        with self._lock:
            return self._device_owned_tokens

    @property
    def device_owned_bytes(self) -> int:
        with self._lock:
            return self._device_owned_bytes

    @property
    def host_owned_bytes(self) -> int:
        with self._lock:
            return self._host_owned_bytes

    @property
    def record_count(self) -> int:
        with self._lock:
            return len(self._records)

    @property
    def lease_count(self) -> int:
        with self._lock:
            return len(self._leases)

    @property
    def orphan_count(self) -> int:
        with self._lock:
            available = set(self._object_keys)
            return sum(
                bool(record.dependencies.difference(available))
                for record in self._records.values()
            )

    def cross_store_resources(self) -> tuple[CrossStoreResource, ...]:
        with self._lock:
            records = tuple(self._records.values())
        return tuple(self._cross_store_resource(record) for record in records)

    def _is_leased(self, key: KVSegmentKey, generation: int) -> bool:
        return any(
            lease.key == key and lease.generation == generation
            for lease in self._leases.values()
        )

    def _cross_store_resource(self, record: _Record) -> CrossStoreResource:
        handle = self._handle(record)
        item = CrossStoreObject(
            object_id=record.object_id,
            kind=record.object_kind,
            tier=(
                CrossStoreTier.DEVICE
                if record.residency == ResidencyTier.DEVICE
                else CrossStoreTier.HOST
            ),
            provenance=ObjectProvenance.APPROXIMATE,
            token_count=record.key.token_count,
            resident_bytes=record.resident_bytes,
            event_ordinal=record.event_ordinal,
            generation=record.generation,
            dependencies=record.dependencies,
            dense_cost_ms=record.dense_cost_ms,
            recovery_cost_ms=record.recovery_cost_ms,
            next_use_ordinal=record.next_use_ordinal,
            retired=record.retired,
            leased=self._is_leased(record.key, record.generation),
            evictable=True,
            demotable=False,
        )

        def evict() -> None:
            if not self.release(handle):
                raise KeyError("approximate victim changed before eviction")
            return None

        return CrossStoreResource(item=item, evict=evict)

    def _evict_unleased_if_needed(
        self,
        protected_key: KVSegmentKey | None = None,
    ) -> None:
        for victim in self._eviction_plan(protected_key):
            self._remove_record(victim, dispose=True)

    def _eviction_plan(
        self,
        protected_key: KVSegmentKey | None,
    ) -> tuple[KVSegmentKey, ...]:
        if not self._over_budget():
            return ()
        simulated = OrderedDict(self._records)
        plan: list[KVSegmentKey] = []

        def usage(tier: ResidencyTier) -> int:
            return sum(
                record.resident_bytes
                for record in simulated.values()
                if record.residency == tier
            )

        while True:
            constrained_tier = None
            if (
                self._max_device_bytes is not None
                and usage(ResidencyTier.DEVICE) > self._max_device_bytes
            ):
                constrained_tier = ResidencyTier.DEVICE
            elif (
                self._max_host_bytes is not None
                and usage(ResidencyTier.HOST) > self._max_host_bytes
            ):
                constrained_tier = ResidencyTier.HOST
            elif len(simulated) <= self._max_records:
                break
            victim = next(
                (
                    key
                    for key, record in simulated.items()
                    if key != protected_key
                    and (
                        constrained_tier is None or record.residency == constrained_tier
                    )
                    and not self._is_leased(key, record.generation)
                    and not any(
                        record.object_id in candidate.dependencies
                        for other_key, candidate in simulated.items()
                        if other_key != key
                    )
                ),
                None,
            )
            if victim is None:
                raise ApproxKVStoreCapacityError(
                    "approximate KV segment store capacity is fully pinned"
                )
            simulated.pop(victim)
            plan.append(victim)
        return tuple(plan)

    def _over_budget(self) -> bool:
        if len(self._records) > self._max_records:
            return True
        if (
            self._max_device_bytes is not None
            and self._device_owned_bytes > self._max_device_bytes
        ):
            return True
        return (
            self._max_host_bytes is not None
            and self._host_owned_bytes > self._max_host_bytes
        )

    def _constrained_tier(self) -> ResidencyTier | None:
        if (
            self._max_device_bytes is not None
            and self._device_owned_bytes > self._max_device_bytes
        ):
            return ResidencyTier.DEVICE
        if (
            self._max_host_bytes is not None
            and self._host_owned_bytes > self._max_host_bytes
        ):
            return ResidencyTier.HOST
        return None

    @staticmethod
    def _dispose_record(record: _Record) -> None:
        if record.release_backend is not None:
            record.release_backend(record.backend_ref, record.residency)

    def _add_record(self, record: _Record) -> None:
        self._records[record.key] = record
        self._records.move_to_end(record.key)
        self._object_keys[record.object_id] = record.key
        for dependency in record.dependencies:
            self._dependents.setdefault(dependency, set()).add(record.object_id)
        if record.residency == ResidencyTier.DEVICE:
            self._device_owned_tokens += record.key.token_count
            self._device_owned_bytes += record.resident_bytes
        else:
            self._host_owned_bytes += record.resident_bytes

    def _remove_record(
        self,
        key: KVSegmentKey,
        *,
        dispose: bool,
    ) -> _Record:
        record = self._records.pop(key)
        self._object_keys.pop(record.object_id, None)
        for dependency in record.dependencies:
            dependents = self._dependents.get(dependency)
            if dependents is not None:
                dependents.discard(record.object_id)
                if not dependents:
                    self._dependents.pop(dependency, None)
        self._dependents.pop(record.object_id, None)
        if record.residency == ResidencyTier.DEVICE:
            self._device_owned_tokens -= record.key.token_count
            self._device_owned_bytes -= record.resident_bytes
        else:
            self._host_owned_bytes -= record.resident_bytes
        if dispose:
            self._dispose_record(record)
        return record
