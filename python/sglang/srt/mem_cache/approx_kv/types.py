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
    ARTIFACT = "artifact"
    KVCOMM_BASE = "kvcomm_base"
    KVCOMM_PLACEHOLDER_DELTA = "kvcomm_placeholder_delta"
    KVCOMM_NEIGHBOR_DELTA = "kvcomm_neighbor_delta"


class RecoveryMode(str, Enum):
    DENSE = "dense"
    COPY = "copy"
    KVCOMM = "kvcomm"


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
    model_fingerprint: str
    cache_dtype: str
    kind: SegmentKind = SegmentKind.ARTIFACT

    def __post_init__(self) -> None:
        if not self.content_hash:
            raise ValueError("content_hash must be non-empty")
        if not self.token_hash:
            raise ValueError("token_hash must be non-empty")
        if self.token_count <= 0:
            raise ValueError("token_count must be positive")
        if not self.model_fingerprint or not self.cache_dtype:
            raise ValueError("model_fingerprint and cache_dtype must be non-empty")


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
    plugin_data: Any = field(default=None, compare=False, repr=False)


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
    h2d_tokens: int = 0
    h2d_bytes: int = 0
    h2d_ms: float = 0.0
    copy_ms: float = 0.0
    rope_ms: float = 0.0
    fallback_reasons: list[str] = field(default_factory=list)
    layer_count: int = 0
    copied_k_layer_tokens: int = 0
    rotated_k_layer_tokens: int = 0
    copied_v_layer_tokens: int = 0

    @property
    def accounted_tokens(self) -> int:
        return self.copied_k_tokens + self.recomputed_tokens

    @property
    def mechanically_valid(self) -> bool:
        token_accounting_valid = (
            self.copied_k_tokens == self.rotated_k_tokens
            and self.copied_k_tokens == self.copied_v_tokens
            and self.zeroed_gap_tokens == 0
            and self.accounted_tokens == self.target_tokens
        )
        if not token_accounting_valid:
            return False
        if self.layer_count == 0:
            return True
        expected = self.copied_k_tokens * self.layer_count
        return (
            self.copied_k_layer_tokens == expected
            and self.rotated_k_layer_tokens == expected
            and self.copied_v_layer_tokens == expected
        )


@dataclass(frozen=True)
class KVLayerTransferResult:
    copied_k_tokens: int
    rotated_k_tokens: int
    copied_v_tokens: int
    copy_ms: float = 0.0
    rope_ms: float = 0.0
    layer_count: int = 0
    copied_k_layer_tokens: int = 0
    rotated_k_layer_tokens: int = 0
    copied_v_layer_tokens: int = 0


@dataclass(frozen=True)
class SchedulerMetadata:
    object_id: str
    resident_bytes: int
    dense_cost_ms: float | None = None
    recovery_cost_ms: float | None = None
    next_use_step: int | None = None
    workflow_stage: str | None = None
    retired: bool = False

    def __post_init__(self) -> None:
        if not self.object_id:
            raise ValueError("object_id must be non-empty")
        if self.resident_bytes < 0:
            raise ValueError("resident_bytes must be non-negative")
        if self.dense_cost_ms is not None and self.dense_cost_ms < 0:
            raise ValueError("dense_cost_ms must be non-negative")
        if self.recovery_cost_ms is not None and self.recovery_cost_ms < 0:
            raise ValueError("recovery_cost_ms must be non-negative")
