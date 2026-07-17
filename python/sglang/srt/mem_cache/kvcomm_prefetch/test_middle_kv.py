from __future__ import annotations

import pytest
import torch

from sglang.srt.mem_cache.kvcomm.config import KVCommFeatureConfig
from sglang.srt.mem_cache.kvcomm.manager import KVCommManager
from sglang.srt.mem_cache.kvcomm.radix_backend import DeviceKVRef
from sglang.srt.mem_cache.kvcomm.types import (
    DenseRange,
    KVReusePlan,
    KVSegmentKey,
    ResidencyTier,
    SegmentKind,
    TransferSpan,
    token_ids_hash,
)
from sglang.srt.mem_cache.kvcomm_prefetch.middle_kv import (
    MiddleKVPrefetchAPI,
    MiddleKVPrefetchError,
)


class FakeAllocator:
    def __init__(self):
        self.next_slot = 100
        self.exported = []
        self.loaded = []
        self.freed = []

    def alloc(self, size):
        indices = torch.arange(self.next_slot, self.next_slot + size)
        self.next_slot += size
        return indices

    def free(self, indices):
        self.freed.extend(indices.tolist())

    def get_kvcache(self):
        raise AssertionError("prefetch-only test must not copy into a target")

    def get_cpu_copy(self, indices):
        payload = {"source_indices": tuple(indices.tolist())}
        self.exported.append(payload)
        return payload

    def load_cpu_copy(self, payload, indices):
        self.loaded.append((payload, tuple(indices.tolist())))


class RecordingTransferBackend:
    def __init__(self):
        self.dense = []
        self.copies = []

    def dense_prefill(self, *, target_start, length, reason):
        self.dense.append((target_start, length, reason))

    def copy_and_rotate(
        self,
        *,
        source_ref,
        source_offset,
        target_start,
        length,
        rope_delta,
    ):
        assert isinstance(source_ref, DeviceKVRef)
        self.copies.append(
            (
                tuple(source_ref.indices.tolist()),
                source_offset,
                target_start,
                length,
                rope_delta,
            )
        )
        return length, length, length


def _api(enabled=True):
    manager = KVCommManager(
        KVCommFeatureConfig(
            core_enabled=enabled,
            prefetch_enabled=enabled,
        )
    )
    allocator = FakeAllocator()
    if not enabled:
        return manager, allocator, None
    return (
        manager,
        allocator,
        MiddleKVPrefetchAPI(
            manager=manager,
            allocator=allocator,
            model_id="Qwen2.5-Coder-7B",
            cache_dtype="bf16",
            lease_ttl_s=10,
        ),
    )


def test_export_prefetch_wait_release_and_drop_middle_kv():
    manager, allocator, api = _api()
    exported = api.export_middle_kv(
        token_ids=(11, 12, 13),
        kv_indices=torch.tensor([4, 5, 6]),
        source_start=40,
        content_hash="function:parse_config",
    )
    assert exported.key.kind == SegmentKind.MIDDLE
    assert exported.residency == ResidencyTier.HOST
    assert allocator.exported == [{"source_indices": (4, 5, 6)}]

    ticket = api.prefetch(exported.key, deadline_s=0.25, priority=7)
    assert ticket.done
    assert ticket.successful
    resident = ticket.wait()
    assert resident.residency == ResidencyTier.DEVICE
    assert tuple(ticket.device_indices().tolist()) == (100, 101, 102)
    assert allocator.loaded == [
        ({"source_indices": (4, 5, 6)}, (100, 101, 102))
    ]

    assert ticket.release()
    assert not ticket.release()
    assert manager.store.lease_count == 0
    assert api.drop(resident)
    assert allocator.freed == [100, 101, 102]


def test_context_manager_releases_prefetch_lease():
    manager, _, api = _api()
    handle = api.register_host_middle_kv(
        token_ids=(1, 2),
        host_payload="precomputed-host-kv",
        source_start=8,
    )
    with api.prefetch(handle.key) as ticket:
        assert ticket.successful
        assert manager.store.lease_count == 1
    assert manager.store.lease_count == 0


def test_prefetched_middle_handle_is_consumable_by_shared_transfer_interface():
    manager, _, api = _api()
    exported = api.register_host_middle_kv(
        token_ids=(11, 12, 13),
        host_payload="all-layer-host-kv",
        source_start=20,
    )
    ticket = api.prefetch(exported.key)
    resident = ticket.wait()
    backend = RecordingTransferBackend()
    stats = manager.execute(
        KVReusePlan(
            target_token_ids=(0, 11, 12, 13, 99),
            copied_spans=(
                TransferSpan(
                    source=resident,
                    source_offset=0,
                    target_start=1,
                    length=3,
                    rope_delta=-19,
                    chunk_start=1,
                    chunk_length=3,
                ),
            ),
            dense_ranges=(
                DenseRange(0, 1, "prefix"),
                DenseRange(4, 1, "suffix"),
            ),
            require_full_coverage=True,
        ),
        backend,
    )
    assert backend.dense == [(0, 1, "prefix"), (4, 1, "suffix")]
    assert backend.copies == [((100, 101, 102), 0, 1, 3, -19)]
    assert stats.recomputed_tokens == 2
    assert stats.copied_k_tokens == 3
    assert stats.mechanically_valid
    assert ticket.release()
    assert api.drop(resident)


def test_missing_middle_key_returns_ticket_with_visible_error():
    _, _, api = _api()
    tokens = (9, 10)
    key = KVSegmentKey(
        content_hash="missing",
        token_hash=token_ids_hash(tokens),
        token_count=len(tokens),
        model_id="Qwen2.5-Coder-7B",
        cache_dtype="bf16",
        kind=SegmentKind.MIDDLE,
    )
    ticket = api.prefetch(key)
    assert not ticket.successful
    with pytest.raises(MiddleKVPrefetchError, match="not found"):
        ticket.wait()


def test_export_rejects_shape_mismatch_and_non_middle_prefetch():
    _, _, api = _api()
    with pytest.raises(ValueError, match="match token_ids"):
        api.export_middle_kv(
            token_ids=(1, 2),
            kv_indices=torch.tensor([4]),
            source_start=0,
        )
    prefix_key = KVSegmentKey(
        content_hash="prefix",
        token_hash=token_ids_hash((1,)),
        token_count=1,
        model_id="Qwen2.5-Coder-7B",
        cache_dtype="bf16",
        kind=SegmentKind.PREFIX,
    )
    with pytest.raises(ValueError, match="only middle"):
        api.prefetch(prefix_key)


def test_api_requires_explicit_core_and_prefetch_gates():
    manager, allocator, _ = _api(enabled=False)
    with pytest.raises(ValueError, match="core"):
        MiddleKVPrefetchAPI(
            manager=manager,
            allocator=allocator,
            model_id="test",
            cache_dtype="bf16",
        )
