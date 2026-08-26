from __future__ import annotations

from dataclasses import dataclass, field
from math import inf
from typing import Iterable

from sglang.srt.mem_cache.kvcomm.manager import KVCommManager
from sglang.srt.mem_cache.kvcomm.store import KVLease, ResidencyLoader
from sglang.srt.mem_cache.kvcomm.types import KVPrefetchHint, KVSegmentKey


@dataclass
class PrefetchResult:
    requested: int = 0
    unique_requested: int = 0
    store_misses: int = 0
    already_resident: int = 0
    loaded: int = 0
    failed: int = 0
    expired: int = 0
    disabled: bool = False
    queue_wait_s: float = 0.0
    load_elapsed_s: float = 0.0
    leases: list[KVLease] = field(default_factory=list)
    failure_reasons: list[str] = field(default_factory=list)


class KVPrefetchCoordinator:
    """Chooses when to load segments without choosing lossy reuse policy."""

    def __init__(
        self,
        *,
        manager: KVCommManager,
        loader: ResidencyLoader,
        lease_ttl_s: float = 60.0,
    ) -> None:
        if lease_ttl_s <= 0:
            raise ValueError("lease_ttl_s must be positive")
        self._manager = manager
        self._loader = loader
        self._lease_ttl_s = lease_ttl_s

    @staticmethod
    def _order(hint: KVPrefetchHint) -> tuple[float, int, KVSegmentKey]:
        deadline = inf if hint.deadline_s is None else hint.deadline_s
        return (deadline, -hint.priority, hint.key)

    def prefetch(self, hints: Iterable[KVPrefetchHint]) -> PrefetchResult:
        hint_list = list(hints)
        result = PrefetchResult(requested=len(hint_list))
        config = self._manager.config
        if not config.core_enabled or not config.prefetch_enabled:
            result.disabled = True
            return result

        deduplicated: dict[KVSegmentKey, KVPrefetchHint] = {}
        for hint in sorted(hint_list, key=self._order):
            deduplicated.setdefault(hint.key, hint)
        ordered = sorted(deduplicated.values(), key=self._order)
        result.unique_requested = len(ordered)

        for hint in ordered:
            handle = self._manager.store.lookup(hint.key)
            if handle is None:
                result.store_misses += 1
                continue
            try:
                if handle.residency == hint.target_tier:
                    resident = handle
                    result.already_resident += 1
                else:
                    resident = self._manager.ensure_resident(
                        handle, hint.target_tier, self._loader
                    )
                    result.loaded += 1
                result.leases.append(
                    self._manager.store.pin(resident, ttl_s=self._lease_ttl_s)
                )
            except Exception as exc:
                result.failed += 1
                result.failure_reasons.append(
                    f"{hint.key.content_hash}:{type(exc).__name__}:{exc}"
                )
        return result

    def release(self, result: PrefetchResult) -> int:
        released = 0
        for lease in result.leases:
            released += int(self._manager.store.unpin(lease))
        result.leases.clear()
        return released
