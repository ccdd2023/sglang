from __future__ import annotations

from sglang.srt.mem_cache.coding_aware.policy import (
    CodingRisk,
    CodingSegment,
    build_coding_reuse_plan,
)
from sglang.srt.mem_cache.kvcomm.config import KVCommFeatureConfig
from sglang.srt.mem_cache.kvcomm.manager import KVCommManager
from sglang.srt.mem_cache.kvcomm.types import (
    KVPrefetchHint,
    KVSegmentKey,
    ResidencyTier,
    SegmentKind,
    token_ids_hash,
)
from sglang.srt.mem_cache.kvcomm_prefetch.coordinator import (
    KVPrefetchCoordinator,
)


class Loader:
    def load(self, handle, target_tier):
        assert target_tier == ResidencyTier.DEVICE
        return f"device:{handle.key.content_hash}"


class Backend:
    def __init__(self):
        self.copies = []
        self.dense = []

    def copy_and_rotate(
        self,
        *,
        source_ref,
        source_offset,
        target_start,
        length,
        rope_delta,
    ):
        self.copies.append(
            (source_ref, source_offset, target_start, length, rope_delta)
        )
        return length, length, length

    def dense_prefill(self, *, target_start, length, reason):
        self.dense.append((target_start, length, reason))


def _key(tokens, name):
    return KVSegmentKey(
        content_hash=name,
        token_hash=token_ids_hash(tokens),
        token_count=len(tokens),
        model_id="test",
        cache_dtype="bf16",
        kind=SegmentKind.MIDDLE,
    )


def test_coding_plan_and_prefetch_compose_through_shared_core():
    stable_tokens = tuple(range(10))
    critical_tokens = tuple(range(10, 15))
    manager = KVCommManager(
        KVCommFeatureConfig(
            core_enabled=True,
            coding_aware_lossy_enabled=True,
            prefetch_enabled=True,
        )
    )
    stable_key = _key(stable_tokens, "stable")
    critical_key = _key(critical_tokens, "critical")
    manager.register_segment(
        key=stable_key,
        token_ids=stable_tokens,
        source_start=100,
        residency=ResidencyTier.HOST,
        backend_ref="host:stable",
    )
    critical_handle = manager.register_segment(
        key=critical_key,
        token_ids=critical_tokens,
        source_start=200,
        residency=ResidencyTier.DEVICE,
        backend_ref="device:critical",
    )

    coordinator = KVPrefetchCoordinator(manager=manager, loader=Loader())
    prefetch_result = coordinator.prefetch((KVPrefetchHint(stable_key),))
    assert prefetch_result.loaded == 1
    stable_handle = manager.store.lookup(stable_key)
    assert stable_handle is not None
    assert stable_handle.residency == ResidencyTier.DEVICE

    plan = build_coding_reuse_plan(
        target_token_ids=stable_tokens + critical_tokens,
        segments=(
            CodingSegment(
                slot_id="stable.py:f",
                target_start=0,
                token_ids=stable_tokens,
                risk=CodingRisk.STABLE,
                source=stable_handle,
                head_tokens=2,
            ),
            CodingSegment(
                slot_id="target.py:f",
                target_start=10,
                token_ids=critical_tokens,
                risk=CodingRisk.CRITICAL,
                source=critical_handle,
            ),
        ),
    )
    backend = Backend()
    stats = manager.execute(plan, backend)
    assert stats.mechanically_valid
    assert stats.recomputed_tokens == 7
    assert stats.copied_k_tokens == 8
    assert backend.copies == [("device:stable", 2, 2, 8, -100)]
    assert {(start, length) for start, length, _ in backend.dense} == {
        (0, 2),
        (10, 5),
    }
    assert coordinator.release(prefetch_result) == 1
