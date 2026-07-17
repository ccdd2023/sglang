from __future__ import annotations

from dataclasses import replace

import pytest

from sglang.srt.mem_cache.kvcomm.config import KVCommFeatureConfig
from sglang.srt.mem_cache.kvcomm.manager import KVCommManager
from sglang.srt.mem_cache.kvcomm.store import KVSegmentStore
from sglang.srt.mem_cache.kvcomm.transfer import KVTransferInvariantError
from sglang.srt.mem_cache.kvcomm.types import (
    DenseRange,
    KVReusePlan,
    KVSegmentKey,
    ResidencyTier,
    SegmentKind,
    TransferSpan,
    token_ids_hash,
)


def _key(tokens, name="segment", kind=SegmentKind.MIDDLE):
    return KVSegmentKey(
        content_hash=name,
        token_hash=token_ids_hash(tokens),
        token_count=len(tokens),
        model_id="test-model",
        cache_dtype="bf16",
        kind=kind,
    )


class RecordingBackend:
    def __init__(self, rotate_all=True):
        self.rotate_all = rotate_all
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
        rotated = length if self.rotate_all else min(2, length)
        return length, rotated, length

    def dense_prefill(self, *, target_start, length, reason):
        self.dense.append((target_start, length, reason))


class RecordingLoader:
    def __init__(self):
        self.calls = []

    def load(self, handle, target_tier):
        self.calls.append((handle.key, handle.residency, target_tier))
        return f"{target_tier.value}-ref"


def test_feature_gates_default_off_and_independent():
    assert KVCommFeatureConfig.from_env({}) == KVCommFeatureConfig()
    config = KVCommFeatureConfig.from_env(
        {
            "SGLANG_KVCOMM_CORE": "1",
            "SGLANG_CODING_AWARE_LOSSY": "1",
            "SGLANG_KV_PREFETCH": "0",
        }
    )
    assert config.core_enabled
    assert config.coding_aware_lossy_enabled
    assert not config.prefetch_enabled


def test_dependent_feature_requires_core():
    with pytest.raises(ValueError, match="SGLANG_KVCOMM_CORE"):
        KVCommFeatureConfig.from_env({"SGLANG_KV_PREFETCH": "1"})


def test_legacy_flag_requires_explicit_compatibility_mode():
    assert not KVCommFeatureConfig.from_env(
        {"SGLANG_LOSSY_ENABLED": "1"}
    ).core_enabled
    config = KVCommFeatureConfig.from_env(
        {
            "SGLANG_KVFLOW_LEGACY_FLAGS": "1",
            "SGLANG_LOSSY_ENABLED": "1",
        }
    )
    assert config.legacy_flags_used
    assert config.core_enabled
    assert config.coding_aware_lossy_enabled
    assert config.prefetch_enabled


def test_register_validates_token_identity_and_stales_old_handle():
    store = KVSegmentStore()
    tokens = (1, 2, 3)
    key = _key(tokens)
    old = store.register(
        key=key,
        token_ids=tokens,
        source_start=4,
        residency=ResidencyTier.HOST,
        backend_ref="old",
    )
    new = store.register(
        key=key,
        token_ids=tokens,
        source_start=8,
        residency=ResidencyTier.DEVICE,
        backend_ref="new",
    )
    assert not store.is_current(old)
    assert store.is_current(new)
    with pytest.raises(ValueError, match="token_hash"):
        store.register(
            key=key,
            token_ids=(1, 2, 4),
            source_start=0,
            residency=ResidencyTier.DEVICE,
            backend_ref=None,
        )


def test_residency_transition_is_a_real_loader_call():
    store = KVSegmentStore()
    tokens = (4, 5, 6)
    handle = store.register(
        key=_key(tokens),
        token_ids=tokens,
        source_start=0,
        residency=ResidencyTier.HOST,
        backend_ref="host-ref",
    )
    loader = RecordingLoader()
    loaded = store.ensure_resident(handle, ResidencyTier.DEVICE, loader)
    assert len(loader.calls) == 1
    assert loaded.residency == ResidencyTier.DEVICE
    assert loaded.backend_ref == "device-ref"


def test_lease_prevents_eviction_then_expires():
    store = KVSegmentStore(max_records=1)
    first_tokens = (1, 2)
    first = store.register(
        key=_key(first_tokens, "first"),
        token_ids=first_tokens,
        source_start=0,
        residency=ResidencyTier.DEVICE,
        backend_ref="first",
    )
    lease = store.pin(first, ttl_s=1)
    second_tokens = (3, 4)
    with pytest.raises(RuntimeError, match="fully pinned"):
        store.register(
            key=_key(second_tokens, "second"),
            token_ids=second_tokens,
            source_start=0,
            residency=ResidencyTier.DEVICE,
            backend_ref="second",
        )
    assert store.record_count == 1
    assert store.gc_expired_leases(now_s=lease.expires_at_s + 1) == 1
    assert store.release(first)


def test_ten_thousand_register_pin_release_cycles_do_not_leak():
    store = KVSegmentStore(max_records=2)
    for index in range(10_000):
        tokens = (index, index + 1)
        handle = store.register(
            key=_key(tokens, f"segment-{index}"),
            token_ids=tokens,
            source_start=0,
            residency=ResidencyTier.DEVICE,
            backend_ref=index,
        )
        lease = store.pin(handle, ttl_s=1)
        assert store.unpin(lease)
        assert store.release(handle)
    assert store.record_count == 0
    assert store.lease_count == 0


def test_l100_k20_copies_source_body_to_target_body():
    tokens = tuple(range(100))
    manager = KVCommManager(KVCommFeatureConfig(core_enabled=True))
    handle = manager.register_segment(
        key=_key(tokens),
        token_ids=tokens,
        source_start=0,
        residency=ResidencyTier.DEVICE,
        backend_ref="kv",
    )
    assert handle is not None
    plan = KVReusePlan(
        target_token_ids=tokens,
        dense_ranges=(DenseRange(0, 20, "head"),),
        copied_spans=(
            TransferSpan(
                source=handle,
                source_offset=20,
                target_start=20,
                length=80,
                rope_delta=0,
                chunk_start=0,
                chunk_length=100,
            ),
        ),
        require_full_coverage=True,
    )
    backend = RecordingBackend()
    stats = manager.execute(plan, backend)
    assert backend.copies == [("kv", 20, 20, 80, 0)]
    assert backend.dense == [(0, 20, "head")]
    assert stats.recomputed_tokens == 20
    assert stats.copied_k_tokens == stats.rotated_k_tokens == 80
    assert stats.mechanically_valid


@pytest.mark.parametrize("delta", [-17, 0, 23])
def test_full_rope_required_for_every_copied_k(delta):
    tokens = tuple(range(8))
    manager = KVCommManager(KVCommFeatureConfig(core_enabled=True))
    handle = manager.register_segment(
        key=_key(tokens),
        token_ids=tokens,
        source_start=0,
        residency=ResidencyTier.DEVICE,
        backend_ref="kv",
    )
    plan = KVReusePlan(
        target_token_ids=tokens,
        copied_spans=(
            TransferSpan(handle, 0, 0, 8, delta, 0, 8),
        ),
        require_full_coverage=True,
    )
    with pytest.raises(KVTransferInvariantError, match="full RoPE"):
        manager.execute(plan, RecordingBackend(rotate_all=False))


def test_source_mismatch_falls_back_to_complete_chunk_once():
    source = tuple(range(10))
    target = tuple(range(9)) + (99,)
    manager = KVCommManager(KVCommFeatureConfig(core_enabled=True))
    handle = manager.register_segment(
        key=_key(source),
        token_ids=source,
        source_start=0,
        residency=ResidencyTier.DEVICE,
        backend_ref="kv",
    )
    plan = KVReusePlan(
        target_token_ids=target,
        dense_ranges=(DenseRange(0, 2, "head"),),
        copied_spans=(
            TransferSpan(handle, 2, 2, 8, 0, 0, 10),
        ),
        require_full_coverage=True,
    )
    backend = RecordingBackend()
    stats = manager.execute(plan, backend)
    assert backend.copies == []
    assert backend.dense == [(0, 10, "source_slice_mismatch")]
    assert stats.recomputed_tokens == 10
    assert stats.source_slice_mismatch == 1


def test_stale_or_nonresident_source_falls_back_dense():
    tokens = tuple(range(5))
    store = KVSegmentStore()
    old = store.register(
        key=_key(tokens),
        token_ids=tokens,
        source_start=0,
        residency=ResidencyTier.DEVICE,
        backend_ref="old",
    )
    store.register(
        key=_key(tokens),
        token_ids=tokens,
        source_start=0,
        residency=ResidencyTier.HOST,
        backend_ref="new",
    )
    plan = KVReusePlan(
        target_token_ids=tokens,
        copied_spans=(TransferSpan(old, 0, 0, 5, 0, 0, 5),),
        require_full_coverage=True,
    )
    backend = RecordingBackend()
    stats = KVCommManager(
        KVCommFeatureConfig(core_enabled=True), store
    ).execute(plan, backend)
    assert stats.stale_handle == 1
    assert backend.dense == [(0, 5, "stale_handle")]


def test_disabled_manager_is_dense_and_has_no_store_side_effect():
    tokens = (1, 2, 3)
    manager = KVCommManager()
    assert (
        manager.register_segment(
            key=_key(tokens),
            token_ids=tokens,
            source_start=0,
            residency=ResidencyTier.DEVICE,
            backend_ref="kv",
        )
        is None
    )
    backend = RecordingBackend()
    stats = manager.execute(KVReusePlan(target_token_ids=tokens), backend)
    assert backend.dense == [(0, 3, "kvcomm_core_disabled")]
    assert stats.recomputed_tokens == 3
