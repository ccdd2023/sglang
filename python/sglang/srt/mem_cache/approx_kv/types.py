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
    EXACT_PREFIX = "exact_prefix"
    CANONICAL_BASE = "canonical_base"
    CONTEXT_ANCHOR = "context_anchor"
    MIDDLE = "middle"


class RecoveryMode(str, Enum):
    DENSE = "dense"
    RAW_ROPE = "raw_rope"
    EPIC_FIXED_K = "epic_fixed_k"
    SELECTIVE_REPAIR = "selective_repair"
    KVCOMM_ANCHOR = "kvcomm_anchor"


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

    def __post_init__(self) -> None:
        if self.generation <= 0:
            raise ValueError("generation must be positive")
        if self.source_start < 0:
            raise ValueError("source_start must be non-negative")
        if len(self.token_ids) != self.key.token_count:
            raise ValueError("handle token count does not match key")


@dataclass(frozen=True)
class DenseRange:
    target_start: int
    length: int
    reason: str = "planned"

    def __post_init__(self) -> None:
        if self.target_start < 0 or self.length <= 0:
            raise ValueError(
                "dense range must have a non-negative start and positive length"
            )
        if not self.reason:
            raise ValueError("dense range reason must be non-empty")


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
    recovery_mode: RecoveryMode = RecoveryMode.DENSE
    copied_spans: tuple[TransferSpan, ...] = ()
    dense_ranges: tuple[DenseRange, ...] = ()
    require_full_coverage: bool = False


@dataclass
class KVTransferStats:
    recovery_mode: RecoveryMode = RecoveryMode.DENSE
    target_tokens: int = 0
    copied_k_tokens: int = 0
    rotated_k_tokens: int = 0
    copied_v_tokens: int = 0
    recomputed_tokens: int = 0
    source_slice_mismatch: int = 0
    stale_handle: int = 0
    residency_miss: int = 0
    zeroed_gap_tokens: int = 0
    h2d_bytes: int = 0
    copy_ms: float = 0.0
    rope_ms: float = 0.0
    repair_ms: float = 0.0
    fallback_reasons: list[str] = field(default_factory=list)

    @property
    def accounted_tokens(self) -> int:
        return self.copied_k_tokens + self.recomputed_tokens

    @property
    def mechanically_valid(self) -> bool:
        return (
            self.copied_k_tokens == self.rotated_k_tokens
            and self.copied_k_tokens == self.copied_v_tokens
            and self.zeroed_gap_tokens == 0
            and self.accounted_tokens == self.target_tokens
        )
