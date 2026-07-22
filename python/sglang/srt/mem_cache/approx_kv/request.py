from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence


class ApproxKVRequestOperation(str, Enum):
    REGISTER = "register"
    REUSE = "reuse"


@dataclass(frozen=True)
class ApproxKVRequestSegment:
    content_hash: str
    target_start: int
    length: int
    source_offset: int = 0

    def __post_init__(self) -> None:
        if not self.content_hash:
            raise ValueError("content_hash must be non-empty")
        if self.target_start < 0 or self.source_offset < 0:
            raise ValueError("segment offsets must be non-negative")
        if self.length <= 0:
            raise ValueError("segment length must be positive")

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
    plugin_params: Mapping[str, Any] = field(
        default_factory=dict,
        compare=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if not self.segments:
            raise ValueError("segments must not be empty")
        if not self.model_fingerprint or not self.cache_dtype:
            raise ValueError("model_fingerprint and cache_dtype must be non-empty")
        if not isinstance(self.plugin_params, Mapping):
            raise ValueError("plugin_params must be an object")

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
    segments = tuple(
        ApproxKVRequestSegment(
            content_hash=str(segment["content_hash"]),
            target_start=int(segment["target_start"]),
            length=int(segment["length"]),
            source_offset=int(segment.get("source_offset", 0)),
        )
        for segment in raw_segments
    )
    plugin_params = raw.get("plugin_params", {})
    if not isinstance(plugin_params, Mapping):
        raise ValueError("approx_kv.plugin_params must be an object")
    return ApproxKVRequestMetadata(
        operation=operation,
        segments=segments,
        model_fingerprint=str(raw.get("model_fingerprint", "runtime")),
        cache_dtype=str(raw.get("cache_dtype", "auto")),
        plugin=(None if raw.get("plugin") is None else str(raw["plugin"])),
        plugin_params=dict(plugin_params),
    )
