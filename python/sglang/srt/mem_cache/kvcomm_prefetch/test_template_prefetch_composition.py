"""Template hints compose with prefetch+copy without changing the admit set."""

from __future__ import annotations

from sglang.srt.mem_cache.kvcomm.config import KVCommFeatureConfig
from sglang.srt.mem_cache.kvcomm.manager import KVCommManager
from sglang.srt.mem_cache.kvcomm.types import (
    DenseRange,
    KVReusePlan,
    KVSegmentKey,
    ResidencyTier,
    SegmentKind,
    TransferSpan,
    token_ids_hash,
)
from sglang.srt.mem_cache.kvcomm_prefetch.coordinator import (
    KVPrefetchCoordinator,
)
from sglang.srt.mem_cache.kvcomm_prefetch.scheduler import (
    AsyncKVPrefetchScheduler,
)
from sglang.srt.mem_cache.kvcomm_prefetch.template_hints import (
    TemplatePrefetchIsland,
    compile_template_prefetch_hints,
)


class Loader:
    def load(self, handle, target_tier):
        assert handle.residency == ResidencyTier.HOST
        assert target_tier == ResidencyTier.DEVICE
        return "device:file-module"


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


def test_template_prefetch_then_copy_does_not_add_tokens():
    observation = tuple(range(100, 108))
    target = (1, 2) + observation + (3, 4, 5)
    key = KVSegmentKey(
        content_hash="repo-file-v3",
        token_hash=token_ids_hash(observation),
        token_count=len(observation),
        model_id="test",
        cache_dtype="bf16",
        kind=SegmentKind.MIDDLE,
    )
    ineligible = KVSegmentKey(
        content_hash="tool-log",
        token_hash=token_ids_hash((9, 9)),
        token_count=2,
        model_id="test",
        cache_dtype="bf16",
        kind=SegmentKind.MIDDLE,
    )
    plan = compile_template_prefetch_hints(
        (
            TemplatePrefetchIsland(
                source_id="read:mod.py@v3",
                key=key,
                remaining_uses=4,
                next_group_index=0,
            ),
            TemplatePrefetchIsland(
                source_id="tool-log",
                key=ineligible,
                remaining_uses=4,
                eligible=False,
            ),
        ),
        group_eta_s={0: 5.0},
        now_s=0.0,
    )
    assert [hint.key.content_hash for hint in plan.hints] == ["repo-file-v3"]
    assert plan.hints[0].priority == 4

    manager = KVCommManager(
        KVCommFeatureConfig(
            core_enabled=True,
            coding_aware_lossy_enabled=True,
            prefetch_enabled=True,
        )
    )
    source = manager.register_segment(
        key=key,
        token_ids=observation,
        source_start=40,
        residency=ResidencyTier.HOST,
        backend_ref="host:file-module",
    )
    assert source is not None
    coordinator = KVPrefetchCoordinator(
        manager=manager, loader=Loader(), lease_ttl_s=30
    )
    with AsyncKVPrefetchScheduler(
        manager=manager, coordinator=coordinator
    ) as scheduler:
        ticket = scheduler.submit(plan.hints[0])
        resident = ticket.wait(timeout_s=5)
        reuse = KVReusePlan(
            target_token_ids=target,
            dense_ranges=(
                DenseRange(0, 2, "dense_prefix"),
                DenseRange(10, 3, "dense_suffix"),
            ),
            copied_spans=(
                TransferSpan(
                    source=resident,
                    source_offset=0,
                    target_start=2,
                    length=8,
                    rope_delta=0,
                    chunk_start=2,
                    chunk_length=8,
                ),
            ),
            require_full_coverage=True,
        )
        backend = Backend()
        stats = manager.execute(reuse, backend)
        assert stats.mechanically_valid
        assert stats.copied_k_tokens == 8
        assert stats.recomputed_tokens == 5
        assert backend.copies == [("device:file-module", 0, 2, 8, 0)]
        assert ticket.release()
    assert manager.store.lease_count == 0


def test_prefetch_store_miss_does_not_copy():
    key = KVSegmentKey(
        content_hash="missing-island",
        token_hash=token_ids_hash((1, 2, 3)),
        token_count=3,
        model_id="test",
        cache_dtype="bf16",
        kind=SegmentKind.MIDDLE,
    )
    manager = KVCommManager(
        KVCommFeatureConfig(core_enabled=True, prefetch_enabled=True)
    )
    coordinator = KVPrefetchCoordinator(manager=manager, loader=Loader())
    result = coordinator.prefetch(
        compile_template_prefetch_hints(
            (
                TemplatePrefetchIsland(
                    source_id="missing",
                    key=key,
                    remaining_uses=1,
                ),
            )
        ).hints
    )
    assert result.store_misses == 1
    assert result.loaded == 0
