from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

from sglang.srt.mem_cache.cross_store.types import CrossStoreKind

from .types import ResidencyTier

logger = logging.getLogger(__name__)
_WARNED_OBJECT_KINDS: set[str] = set()
_OBJECT_KIND_ALIASES = {
    "stage_variant": CrossStoreKind.EXACT_VARIANT,
    "anchor_delta": CrossStoreKind.ANCHOR,
    "repair_metadata": CrossStoreKind.REPAIR_STATE,
}


def _object_kind(value: Any) -> CrossStoreKind:
    normalized = str(value)
    alias = _OBJECT_KIND_ALIASES.get(normalized)
    if alias is not None:
        return alias
    try:
        return CrossStoreKind(normalized)
    except ValueError:
        if normalized not in _WARNED_OBJECT_KINDS:
            logger.warning(
                "Unknown approximate KV object_kind %r; using " "precomputed_adapter",
                normalized,
            )
            _WARNED_OBJECT_KINDS.add(normalized)
        return CrossStoreKind.PRECOMPUTED_ADAPTER


class ApproxKVRequestOperation(str, Enum):
    REGISTER = "register"
    REUSE = "reuse"


@dataclass(frozen=True)
class ApproxKVRequestSegment:
    content_hash: str
    target_start: int
    length: int
    source_offset: int = 0
    object_id: str | None = None
    object_kind: CrossStoreKind = CrossStoreKind.PRECOMPUTED_ADAPTER
    dense_cost_ms: float | None = None
    recovery_cost_ms: float | None = None
    next_use_ordinal: int | None = None
    retired: bool = False
    residency: ResidencyTier | None = None
    dependencies: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not self.content_hash:
            raise ValueError("content_hash must be non-empty")
        if self.target_start < 0 or self.source_offset < 0:
            raise ValueError("segment offsets must be non-negative")
        if self.length <= 0:
            raise ValueError("segment length must be positive")
        if self.object_id is not None and not self.object_id:
            raise ValueError("object_id must be non-empty when provided")
        if self.dense_cost_ms is not None and self.dense_cost_ms < 0:
            raise ValueError("dense_cost_ms must be non-negative")
        if self.recovery_cost_ms is not None and self.recovery_cost_ms < 0:
            raise ValueError("recovery_cost_ms must be non-negative")
        if self.next_use_ordinal is not None and self.next_use_ordinal < 0:
            raise ValueError("next_use_ordinal must be non-negative")
        if self.object_id is not None and self.object_id in self.dependencies:
            raise ValueError("a segment cannot depend on itself")

    @property
    def target_end(self) -> int:
        return self.target_start + self.length


@dataclass(frozen=True)
class ApproxKVRequestMetadata:
    operation: ApproxKVRequestOperation
    segments: tuple[ApproxKVRequestSegment, ...]
    model_fingerprint: str
    cache_dtype: str
    plugin: str | None = None
    pin_until_reset: bool = False

    def __post_init__(self) -> None:
        if not self.segments:
            raise ValueError("segments must not be empty")
        if not self.model_fingerprint or not self.cache_dtype:
            raise ValueError("model_fingerprint and cache_dtype must be non-empty")
        if not isinstance(self.pin_until_reset, bool):
            raise ValueError("pin_until_reset must be a boolean")
        if self.pin_until_reset and self.operation != ApproxKVRequestOperation.REGISTER:
            raise ValueError("pin_until_reset is only valid for registration")

    def validate_prompt_length(self, prompt_length: int) -> None:
        if prompt_length <= 0:
            raise ValueError("prompt_length must be positive")
        reusable_limit = prompt_length - 1
        occupied: set[int] = set()
        for segment in self.segments:
            if segment.target_end > reusable_limit:
                raise ValueError(
                    "approximate KV segments must leave the final prompt "
                    "token for a real forward pass"
                )
            positions = set(range(segment.target_start, segment.target_end))
            if occupied & positions:
                raise ValueError("approximate KV segments must not overlap")
            occupied |= positions


def parse_request_metadata(
    custom_params: Mapping[str, Any] | None,
) -> ApproxKVRequestMetadata | None:
    if not custom_params:
        return None
    raw = custom_params.get("approx_kv")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("custom_params.approx_kv must be an object")

    try:
        operation = ApproxKVRequestOperation(str(raw["operation"]))
    except KeyError as exc:
        raise ValueError("approx_kv.operation is required") from exc
    raw_segments = raw.get("segments")
    if not isinstance(raw_segments, Sequence) or isinstance(
        raw_segments,
        (str, bytes),
    ):
        raise ValueError("approx_kv.segments must be an array")
    if "pin_ttl_s" in raw:
        raise ValueError("approx_kv.pin_ttl_s is unsupported; use pin_until_reset")
    pin_until_reset = raw.get("pin_until_reset", False)
    if not isinstance(pin_until_reset, bool):
        raise ValueError("approx_kv.pin_until_reset must be a boolean")
    segments = tuple(
        ApproxKVRequestSegment(
            content_hash=str(segment["content_hash"]),
            target_start=int(segment["target_start"]),
            length=int(segment["length"]),
            source_offset=int(segment.get("source_offset", 0)),
            object_id=(
                None if segment.get("object_id") is None else str(segment["object_id"])
            ),
            object_kind=_object_kind(
                segment.get(
                    "object_kind",
                    CrossStoreKind.PRECOMPUTED_ADAPTER.value,
                )
            ),
            dense_cost_ms=(
                None
                if segment.get("dense_cost_ms") is None
                else float(segment["dense_cost_ms"])
            ),
            recovery_cost_ms=(
                None
                if segment.get("recovery_cost_ms") is None
                else float(segment["recovery_cost_ms"])
            ),
            next_use_ordinal=(
                None
                if segment.get("next_use_ordinal") is None
                else int(segment["next_use_ordinal"])
            ),
            retired=bool(segment.get("retired", False)),
            residency=(
                None
                if segment.get("residency") is None
                else ResidencyTier(str(segment["residency"]))
            ),
            dependencies=frozenset(
                str(value) for value in segment.get("dependencies", ())
            ),
        )
        for segment in raw_segments
    )
    return ApproxKVRequestMetadata(
        operation=operation,
        segments=segments,
        model_fingerprint=str(raw.get("model_fingerprint", "runtime")),
        cache_dtype=str(raw.get("cache_dtype", "auto")),
        plugin=(None if raw.get("plugin") is None else str(raw["plugin"])),
        pin_until_reset=pin_until_reset,
    )
