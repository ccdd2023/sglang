from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence


class ResidencyTier(str, Enum):
    DEVICE = "device"
    HOST = "host"
    STORAGE = "storage"


class SegmentKind(str, Enum):
    PREFIX = "prefix"
    MIDDLE = "middle"


def token_ids_hash(token_ids: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for token_id in token_ids:
        value = int(token_id)
        digest.update(value.to_bytes(8, byteorder="little", signed=True))
    return digest.hexdigest()


@dataclass(frozen=True, order=True)
class KVSegmentKey:
    content_hash: str
    token_hash: str
    token_count: int
    model_id: str
    cache_dtype: str
    kind: SegmentKind = SegmentKind.MIDDLE

    def __post_init__(self) -> None:
        if not self.content_hash:
            raise ValueError("content_hash must be non-empty")
        if not self.token_hash:
            raise ValueError("token_hash must be non-empty")
        if self.token_count <= 0:
            raise ValueError("token_count must be positive")
        if not self.model_id or not self.cache_dtype:
            raise ValueError("model_id and cache_dtype must be non-empty")


@dataclass(frozen=True)
class KVSegmentHandle:
    key: KVSegmentKey
    generation: int
    residency: ResidencyTier
    source_start: int
    token_ids: tuple[int, ...]
    backend_ref: Any = field(compare=False, repr=False)


@dataclass(frozen=True)
class DenseRange:
    target_start: int
    length: int
    reason: str = "planned"

    def __post_init__(self) -> None:
        if self.target_start < 0 or self.length <= 0:
            raise ValueError("dense range must have a non-negative start and positive length")


@dataclass(frozen=True)
class TransferSpan:
    source: KVSegmentHandle
    source_offset: int
    target_start: int
    length: int
    rope_delta: int
    chunk_start: int
    chunk_length: int

    def __post_init__(self) -> None:
        if self.source_offset < 0 or self.target_start < 0 or self.length <= 0:
            raise ValueError("transfer span has invalid bounds")
        if self.chunk_start < 0 or self.chunk_length <= 0:
            raise ValueError("chunk bounds must be valid")
        if not (
            self.chunk_start
            <= self.target_start
            < self.target_start + self.length
            <= self.chunk_start + self.chunk_length
        ):
            raise ValueError("transfer span must be contained in its chunk")


@dataclass(frozen=True)
class KVReusePlan:
    target_token_ids: tuple[int, ...]
    copied_spans: tuple[TransferSpan, ...] = ()
    dense_ranges: tuple[DenseRange, ...] = ()
    require_full_coverage: bool = False


@dataclass(frozen=True)
class KVPrefetchHint:
    key: KVSegmentKey
    target_tier: ResidencyTier = ResidencyTier.DEVICE
    deadline_s: float | None = None
    priority: int = 0


@dataclass
class KVTransferStats:
    target_tokens: int = 0
    copied_k_tokens: int = 0
    rotated_k_tokens: int = 0
    copied_v_tokens: int = 0
    recomputed_tokens: int = 0
    source_slice_mismatch: int = 0
    stale_handle: int = 0
    residency_miss: int = 0
    zeroed_gap_tokens: int = 0
    fallback_reasons: list[str] = field(default_factory=list)

    @property
    def mechanically_valid(self) -> bool:
        return (
            self.copied_k_tokens == self.rotated_k_tokens
            and self.copied_k_tokens == self.copied_v_tokens
            and self.source_slice_mismatch == 0
            and self.stale_handle == 0
            and self.zeroed_gap_tokens == 0
        )
