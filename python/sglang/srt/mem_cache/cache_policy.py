from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from sglang.srt.mem_cache.cross_store.class_order import (
    s4_class,
    s4_next_use_key,
)


class CacheObjectKind(str, Enum):
    EXACT_VARIANT = "exact_variant"
    CANONICAL_BASE = "canonical_base"
    ANCHOR = "anchor"
    REPAIR_METADATA = "repair_metadata"
    FILLER = "filler"


class PrefetchMode(str, Enum):
    OFF = "p0"
    FREE_SPACE_ONLY = "p1"
    DEAD_OBJECT_ONLY = "p2"
    ORACLE_NEXT_STAGE = "p3"


_OBJECT_KIND_ALIASES = {
    "stage_variant": CacheObjectKind.EXACT_VARIANT,
    "anchor_delta": CacheObjectKind.ANCHOR,
}


@dataclass(frozen=True)
class CacheProtectionMetadata:
    object_id: str
    protected_tokens: int | None = None
    resident_bytes: int = 0
    dense_cost_ms: float | None = None
    recovery_cost_ms: float | None = None
    current_step: int | None = None
    next_use_step: int | None = None
    next_use_request_step: int | None = None
    next_use_distance: int | None = None
    workflow_stage: str | None = None
    object_kind: CacheObjectKind = CacheObjectKind.EXACT_VARIANT
    recoverable_from_lower_tier: bool = False
    retired: bool = False

    def __post_init__(self) -> None:
        if not self.object_id:
            raise ValueError("object_id must be non-empty")
        if self.protected_tokens is not None and self.protected_tokens <= 0:
            raise ValueError("protected_tokens must be positive")
        if self.resident_bytes < 0:
            raise ValueError("resident_bytes must be non-negative")
        if self.dense_cost_ms is not None and self.dense_cost_ms < 0:
            raise ValueError("dense_cost_ms must be non-negative")
        if self.recovery_cost_ms is not None and self.recovery_cost_ms < 0:
            raise ValueError("recovery_cost_ms must be non-negative")
        if self.current_step is not None and self.current_step < 0:
            raise ValueError("current_step must be non-negative")
        if self.next_use_step is not None and self.next_use_step < 0:
            raise ValueError("next_use_step must be non-negative")
        if self.next_use_request_step is not None and self.next_use_request_step < 0:
            raise ValueError("next_use_request_step must be non-negative")
        if self.next_use_distance is not None and self.next_use_distance < 0:
            raise ValueError("next_use_distance must be non-negative")

    @property
    def has_future_use(self) -> bool:
        return not self.retired and (
            self.next_use_step is not None
            or self.next_use_request_step is not None
            or self.next_use_distance is not None
        )

    @property
    def saved_ms(self) -> float:
        if self.dense_cost_ms is None:
            return 0.0
        recovery_cost = self.recovery_cost_ms or 0.0
        return max(0.0, self.dense_cost_ms - recovery_cost)

    @property
    def value_density(self) -> float:
        return self.saved_ms / max(1, self.resident_bytes)


@dataclass
class CacheProtectionState:
    objects: dict[str, CacheProtectionMetadata] = field(default_factory=dict)

    def update(self, metadata: Iterable[CacheProtectionMetadata]) -> None:
        for item in metadata:
            self.objects[item.object_id] = item

    def clone(self) -> CacheProtectionState:
        return CacheProtectionState(objects=dict(self.objects))

    def discard(self, object_id: str) -> None:
        self.objects.pop(object_id, None)

    def values(self) -> tuple[CacheProtectionMetadata, ...]:
        return tuple(self.objects.values())


@dataclass(frozen=True)
class CachePrefetchHint:
    object_id: str
    next_use_step: int | None = None

    def __post_init__(self) -> None:
        if not self.object_id:
            raise ValueError("prefetch object_id must be non-empty")
        if self.next_use_step is not None and self.next_use_step < 0:
            raise ValueError("prefetch next_use_step must be non-negative")


@dataclass(frozen=True)
class PrefetchCandidate:
    candidate_id: int
    token_count: int
    protection: CacheProtectionState
    last_access_time: float

    def __post_init__(self) -> None:
        if self.token_count <= 0:
            raise ValueError("prefetch candidate token_count must be positive")


@dataclass(frozen=True)
class PrefetchDecision:
    mode: PrefetchMode
    admitted: bool
    required_tokens: int
    available_tokens: int
    victim_ids: tuple[int, ...] = ()
    victim_tokens: int = 0
    loaded_tokens: int = 0
    reason: str = ""


def _optional_int(raw: Mapping[str, Any], key: str) -> int | None:
    value = raw.get(key)
    return None if value is None else int(value)


def _optional_float(raw: Mapping[str, Any], key: str) -> float | None:
    value = raw.get(key)
    return None if value is None else float(value)


def _bool_value(raw: Mapping[str, Any], key: str, default: bool = False) -> bool:
    value = raw.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"cache_protection.{key} must be a boolean")
    return value


def _object_kind(value: Any) -> CacheObjectKind:
    normalized = str(value)
    alias = _OBJECT_KIND_ALIASES.get(normalized)
    if alias is not None:
        return alias
    try:
        return CacheObjectKind(normalized)
    except ValueError as exc:
        supported = ", ".join(kind.value for kind in CacheObjectKind)
        raise ValueError(
            f"unknown cache object kind {normalized!r}; expected one of {supported}"
        ) from exc


def _parse_metadata(raw: Mapping[str, Any]) -> CacheProtectionMetadata:
    try:
        object_id = str(raw["object_id"])
    except KeyError as exc:
        raise ValueError("cache_protection.object_id is required") from exc
    return CacheProtectionMetadata(
        object_id=object_id,
        protected_tokens=_optional_int(raw, "protected_tokens"),
        resident_bytes=int(raw.get("resident_bytes", 0)),
        dense_cost_ms=_optional_float(raw, "dense_cost_ms"),
        recovery_cost_ms=_optional_float(raw, "recovery_cost_ms"),
        current_step=_optional_int(raw, "current_step"),
        next_use_step=_optional_int(raw, "next_use_step"),
        next_use_request_step=_optional_int(raw, "next_use_request_step"),
        next_use_distance=_optional_int(raw, "next_use_distance"),
        workflow_stage=(
            None if raw.get("workflow_stage") is None else str(raw["workflow_stage"])
        ),
        object_kind=_object_kind(
            raw.get("object_kind", CacheObjectKind.EXACT_VARIANT.value)
        ),
        recoverable_from_lower_tier=_bool_value(raw, "recoverable_from_lower_tier"),
        retired=_bool_value(raw, "retired"),
    )


def parse_cache_protection_metadata(
    custom_params: Mapping[str, Any] | None,
) -> tuple[CacheProtectionMetadata, ...]:
    if not custom_params:
        return ()
    raw = custom_params.get("cache_protection")
    if raw is None:
        return ()
    if isinstance(raw, Mapping):
        raw_objects = raw.get("objects")
        if raw_objects is None:
            return (_parse_metadata(raw),)
    else:
        raw_objects = raw
    if not isinstance(raw_objects, Sequence) or isinstance(raw_objects, (str, bytes)):
        raise ValueError("custom_params.cache_protection must be an object or array")
    metadata = []
    for item in raw_objects:
        if not isinstance(item, Mapping):
            raise ValueError("cache_protection entries must be objects")
        metadata.append(_parse_metadata(item))
    if not metadata:
        raise ValueError("cache_protection entries must not be empty")
    return tuple(metadata)


def parse_cache_prefetch_hints(
    custom_params: Mapping[str, Any] | None,
) -> tuple[CachePrefetchHint, ...]:
    if not custom_params:
        return ()
    raw = custom_params.get("cache_prefetch")
    if raw is None:
        return ()
    if not isinstance(raw, Mapping):
        raise ValueError("custom_params.cache_prefetch must be an object")
    try:
        object_id = str(raw["object_id"])
    except KeyError as exc:
        raise ValueError("cache_prefetch.object_id is required") from exc
    return (
        CachePrefetchHint(
            object_id=object_id,
            next_use_step=_optional_int(raw, "next_use_step"),
        ),
    )


def future_entries(state: CacheProtectionState) -> tuple[CacheProtectionMetadata, ...]:
    return tuple(item for item in state.values() if item.has_future_use)


def steps_eviction_key(
    state: CacheProtectionState, last_access_time: float
) -> tuple[int, int, float]:
    steps = [
        item.next_use_step
        for item in future_entries(state)
        if item.next_use_step is not None
    ]
    if not steps:
        return (0, 0, last_access_time)
    return (1, -min(steps), last_access_time)


def belady_eviction_key(
    state: CacheProtectionState, last_access_time: float
) -> tuple[int, int, float]:
    next_steps = [
        (
            item.next_use_request_step
            if item.next_use_request_step is not None
            else item.next_use_distance
        )
        for item in future_entries(state)
        if item.next_use_request_step is not None or item.next_use_distance is not None
    ]
    if not next_steps:
        return (0, 0, last_access_time)
    return (1, -min(next_steps), last_access_time)


def value_density_eviction_key(
    state: CacheProtectionState,
    last_access_time: float,
    current_step: int,
) -> tuple[int, float, int, float]:
    active = future_entries(state)
    if not active:
        return (0, 0.0, 0, last_access_time)
    score = max(
        item.value_density
        / (
            1
            + max(
                0,
                (
                    item.next_use_request_step
                    if item.next_use_request_step is not None
                    else current_step + (item.next_use_distance or 0)
                )
                - current_step,
            )
        )
        for item in active
    )
    next_steps = [
        (
            item.next_use_request_step
            if item.next_use_request_step is not None
            else current_step + item.next_use_distance
        )
        for item in active
        if item.next_use_request_step is not None or item.next_use_distance is not None
    ]
    return (1, score, -min(next_steps) if next_steps else 0, last_access_time)


def _hierarchical_class(item: CacheProtectionMetadata) -> int:
    return s4_class(
        item.object_kind.value,
        retired=not item.has_future_use,
        recoverable_from_lower_tier=item.recoverable_from_lower_tier,
    )


def hierarchical_eviction_key(
    state: CacheProtectionState, last_access_time: float
) -> tuple[int, float, int, float]:
    if not state.objects:
        return (0, 0.0, 0, last_access_time)
    protection = max(
        (
            _hierarchical_class(item),
            item.value_density,
            s4_next_use_key(
                item.next_use_request_step,
                item.next_use_distance,
            ),
        )
        for item in state.values()
    )
    return (*protection, last_access_time)


def _known_dead(candidate: PrefetchCandidate) -> bool:
    return bool(candidate.protection.objects) and not future_entries(
        candidate.protection
    )


def _next_use_step(candidate: PrefetchCandidate) -> int | None:
    next_steps = [
        (
            item.next_use_request_step
            if item.next_use_request_step is not None
            else item.next_use_distance
        )
        for item in future_entries(candidate.protection)
        if item.next_use_request_step is not None or item.next_use_distance is not None
    ]
    return None if not next_steps else min(next_steps)


def plan_prefetch(
    *,
    mode: PrefetchMode,
    required_tokens: int,
    available_tokens: int,
    candidates: Iterable[PrefetchCandidate],
    target_next_use_step: int | None = None,
) -> PrefetchDecision:
    if required_tokens <= 0:
        raise ValueError("required_tokens must be positive")
    if available_tokens < 0:
        raise ValueError("available_tokens must be non-negative")
    if mode == PrefetchMode.OFF:
        return PrefetchDecision(
            mode=mode,
            admitted=False,
            required_tokens=required_tokens,
            available_tokens=available_tokens,
            reason="prefetch_disabled",
        )
    if available_tokens >= required_tokens:
        return PrefetchDecision(
            mode=mode,
            admitted=True,
            required_tokens=required_tokens,
            available_tokens=available_tokens,
            reason="free_space",
        )
    if mode == PrefetchMode.FREE_SPACE_ONLY:
        return PrefetchDecision(
            mode=mode,
            admitted=False,
            required_tokens=required_tokens,
            available_tokens=available_tokens,
            reason="insufficient_free_space",
        )

    eligible = []
    for candidate in candidates:
        if _known_dead(candidate):
            eligible.append((0, 0, candidate.last_access_time, candidate))
            continue
        if mode != PrefetchMode.ORACLE_NEXT_STAGE:
            continue
        next_step = _next_use_step(candidate)
        active = future_entries(candidate.protection)
        if (
            target_next_use_step is not None
            and next_step is not None
            and next_step > target_next_use_step
            and active
            and all(item.recoverable_from_lower_tier for item in active)
        ):
            eligible.append((1, -next_step, candidate.last_access_time, candidate))

    eligible.sort(key=lambda row: row[:3])
    needed = required_tokens - available_tokens
    victims = []
    victim_tokens = 0
    for _, _, _, candidate in eligible:
        victims.append(candidate.candidate_id)
        victim_tokens += candidate.token_count
        if victim_tokens >= needed:
            return PrefetchDecision(
                mode=mode,
                admitted=True,
                required_tokens=required_tokens,
                available_tokens=available_tokens,
                victim_ids=tuple(victims),
                victim_tokens=victim_tokens,
                reason=(
                    "dead_object_eviction"
                    if mode == PrefetchMode.DEAD_OBJECT_ONLY
                    else "oracle_farther_use_eviction"
                ),
            )
    return PrefetchDecision(
        mode=mode,
        admitted=False,
        required_tokens=required_tokens,
        available_tokens=available_tokens,
        victim_ids=tuple(victims),
        victim_tokens=victim_tokens,
        reason="insufficient_admissible_victims",
    )
