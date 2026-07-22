from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .types import RecoveryMode


@dataclass(frozen=True)
class ApproxKVRequestSegment:
    source_content_hash: str
    target_start: int
    length: int
    source_offset: int = 0

    def __post_init__(self) -> None:
        if not self.source_content_hash:
            raise ValueError("source_content_hash must be non-empty")
        if self.target_start < 0 or self.source_offset < 0:
            raise ValueError("segment offsets must be non-negative")
        if self.length <= 0:
            raise ValueError("segment length must be positive")

    @property
    def target_end(self) -> int:
        return self.target_start + self.length


@dataclass(frozen=True)
class ApproxKVRequestMetadata:
    recovery_mode: RecoveryMode
    segments: tuple[ApproxKVRequestSegment, ...]
    speed_only: bool = False

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
        mode = RecoveryMode(str(raw["recovery_mode"]))
    except KeyError as exc:
        raise ValueError("approx_kv.recovery_mode is required") from exc
    raw_segments = raw.get("segments")
    if not isinstance(raw_segments, Sequence) or isinstance(
        raw_segments,
        (str, bytes),
    ):
        raise ValueError("approx_kv.segments must be an array")
    segments = tuple(
        ApproxKVRequestSegment(
            source_content_hash=str(segment["source_content_hash"]),
            target_start=int(segment["target_start"]),
            length=int(segment["length"]),
            source_offset=int(segment.get("source_offset", 0)),
        )
        for segment in raw_segments
    )
    if not segments:
        raise ValueError("approx_kv.segments must not be empty")
    return ApproxKVRequestMetadata(
        recovery_mode=mode,
        segments=segments,
        speed_only=bool(raw.get("speed_only", False)),
    )
