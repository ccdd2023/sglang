"""Policy-neutral KVCOMM data-plane primitives.

The package deliberately contains no coding-aware selection policy and no
prefetch scheduling policy.  Callers may build either feature on the same
segment store and transfer contract.
"""

from sglang.srt.mem_cache.kvcomm.config import KVCommFeatureConfig
from sglang.srt.mem_cache.kvcomm.manager import KVCommManager
from sglang.srt.mem_cache.kvcomm.store import KVSegmentStore
from sglang.srt.mem_cache.kvcomm.types import (
    DenseRange,
    KVPrefetchHint,
    KVReusePlan,
    KVSegmentHandle,
    KVSegmentKey,
    KVTransferStats,
    ResidencyTier,
    SegmentKind,
    TransferSpan,
    token_ids_hash,
)

__all__ = [
    "DenseRange",
    "KVCommFeatureConfig",
    "KVCommManager",
    "KVPrefetchHint",
    "KVReusePlan",
    "KVSegmentHandle",
    "KVSegmentKey",
    "KVSegmentStore",
    "KVTransferStats",
    "ResidencyTier",
    "SegmentKind",
    "TransferSpan",
    "token_ids_hash",
]
