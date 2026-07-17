from __future__ import annotations

import inspect

from sglang.srt.mem_cache.kvcomm.config import KVCommFeatureConfig
from sglang.srt.mem_cache.kvcomm.manager import KVCommManager
from sglang.srt.mem_cache.kvcomm.types import (
    KVPrefetchHint,
    KVSegmentKey,
    ResidencyTier,
    SegmentKind,
    token_ids_hash,
)
from sglang.srt.mem_cache.kvcomm_prefetch import coordinator
from sglang.srt.mem_cache.kvcomm_prefetch.coordinator import (
    KVPrefetchCoordinator,
)


def _key(tokens, name, kind):
    return KVSegmentKey(
        content_hash=name,
        token_hash=token_ids_hash(tokens),
        token_count=len(tokens),
        model_id="test",
        cache_dtype="bf16",
        kind=kind,
    )


class Loader:
    def __init__(self):
        self.calls = []

    def load(self, handle, target_tier):
        self.calls.append((handle.key.kind, handle.residency, target_tier))
        return f"loaded-{handle.key.content_hash}"


def _manager(enabled=True):
    return KVCommManager(
        KVCommFeatureConfig(
            core_enabled=enabled,
            prefetch_enabled=enabled,
        )
    )


def test_disabled_prefetch_has_no_store_or_loader_side_effect():
    manager = _manager(enabled=False)
    loader = Loader()
    coordinator_instance = KVPrefetchCoordinator(manager=manager, loader=loader)
    key = _key((1, 2), "prefix", SegmentKind.PREFIX)
    result = coordinator_instance.prefetch((KVPrefetchHint(key),))
    assert result.disabled
    assert not loader.calls
    assert manager.store.record_count == 0


def test_prefix_and_middle_segments_share_real_residency_load_path():
    manager = _manager()
    prefix_tokens = (1, 2, 3)
    middle_tokens = (4, 5, 6, 7)
    prefix_key = _key(prefix_tokens, "prefix", SegmentKind.PREFIX)
    middle_key = _key(middle_tokens, "middle", SegmentKind.MIDDLE)
    manager.register_segment(
        key=prefix_key,
        token_ids=prefix_tokens,
        source_start=0,
        residency=ResidencyTier.HOST,
        backend_ref="prefix-host",
    )
    manager.register_segment(
        key=middle_key,
        token_ids=middle_tokens,
        source_start=100,
        residency=ResidencyTier.HOST,
        backend_ref="middle-host",
    )
    loader = Loader()
    coordinator_instance = KVPrefetchCoordinator(
        manager=manager, loader=loader, lease_ttl_s=10
    )
    result = coordinator_instance.prefetch(
        (
            KVPrefetchHint(prefix_key, priority=1),
            KVPrefetchHint(middle_key, priority=2),
        )
    )
    assert result.loaded == 2
    assert result.failed == 0
    assert len(loader.calls) == 2
    assert {call[0] for call in loader.calls} == {
        SegmentKind.PREFIX,
        SegmentKind.MIDDLE,
    }
    assert manager.store.lookup(prefix_key).residency == ResidencyTier.DEVICE
    assert manager.store.lookup(middle_key).residency == ResidencyTier.DEVICE
    assert coordinator_instance.release(result) == 2
    assert manager.store.lease_count == 0


def test_already_device_is_not_reported_as_a_transfer():
    manager = _manager()
    tokens = (1, 2)
    key = _key(tokens, "device", SegmentKind.MIDDLE)
    manager.register_segment(
        key=key,
        token_ids=tokens,
        source_start=0,
        residency=ResidencyTier.DEVICE,
        backend_ref="device",
    )
    loader = Loader()
    result = KVPrefetchCoordinator(manager=manager, loader=loader).prefetch(
        (KVPrefetchHint(key),)
    )
    assert result.already_resident == 1
    assert result.loaded == 0
    assert not loader.calls


def test_duplicate_hints_are_deduplicated_and_missing_is_visible():
    manager = _manager()
    loader = Loader()
    key = _key((8, 9), "missing", SegmentKind.MIDDLE)
    result = KVPrefetchCoordinator(manager=manager, loader=loader).prefetch(
        (
            KVPrefetchHint(key, priority=1),
            KVPrefetchHint(key, priority=9),
        )
    )
    assert result.requested == 2
    assert result.unique_requested == 1
    assert result.store_misses == 1


def test_prefetch_module_has_no_coding_policy_dependency():
    source = inspect.getsource(coordinator)
    assert "coding_aware" not in source
    assert "ast_chunker" not in source
