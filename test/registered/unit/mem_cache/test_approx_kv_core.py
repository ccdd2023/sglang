from __future__ import annotations

import importlib
import sys
import types as python_types
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
PACKAGE_DIR = REPO_ROOT / "python/sglang/srt/mem_cache/approx_kv"
PACKAGE_NAME = "approx_kv_under_test"

package = python_types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(PACKAGE_DIR)]
sys.modules[PACKAGE_NAME] = package

config_module = importlib.import_module(f"{PACKAGE_NAME}.config")
manager_module = importlib.import_module(f"{PACKAGE_NAME}.manager")
store_module = importlib.import_module(f"{PACKAGE_NAME}.store")
transfer_module = importlib.import_module(f"{PACKAGE_NAME}.transfer")
types_module = importlib.import_module(f"{PACKAGE_NAME}.types")

ApproxKVFeatureConfig = config_module.ApproxKVFeatureConfig
ApproxKVManager = manager_module.ApproxKVManager
ApproxKVSegmentStore = store_module.ApproxKVSegmentStore
DenseRange = types_module.DenseRange
KVReusePlan = types_module.KVReusePlan
KVSegmentKey = types_module.KVSegmentKey
KVTransferInvariantError = transfer_module.KVTransferInvariantError
RecoveryMode = types_module.RecoveryMode
ResidencyTier = types_module.ResidencyTier
SegmentKind = types_module.SegmentKind
TransferSpan = types_module.TransferSpan
token_ids_hash = types_module.token_ids_hash


def make_key(
    tokens: tuple[int, ...],
    name: str = "segment",
) -> KVSegmentKey:
    return KVSegmentKey(
        content_hash=name,
        token_hash=token_ids_hash(tokens),
        token_count=len(tokens),
        model_id="test-model",
        cache_dtype="bf16",
        kind=SegmentKind.CANONICAL_BASE,
    )


class RecordingBackend:
    def __init__(self, rotate_all: bool = True) -> None:
        self.rotate_all = rotate_all
        self.copies: list[tuple[object, int, int, int, int]] = []
        self.dense: list[tuple[int, int, str]] = []

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
            (
                source_ref,
                source_offset,
                target_start,
                length,
                rope_delta,
            )
        )
        rotated = length if self.rotate_all else max(0, length - 1)
        return length, rotated, length

    def dense_prefill(self, *, target_start, length, reason):
        self.dense.append((target_start, length, reason))


class RecordingLoader:
    def __init__(self) -> None:
        self.calls = []

    def load(self, handle, target_tier):
        self.calls.append((handle.key, handle.residency, target_tier))
        return f"{target_tier.value}-ref"


class TestApproxKVCore(unittest.TestCase):
    def test_feature_gates(self):
        self.assertEqual(
            ApproxKVFeatureConfig.from_env({}),
            ApproxKVFeatureConfig(),
        )
        enabled = ApproxKVFeatureConfig.from_env(
            {
                "SGLANG_APPROX_KV_CORE": "1",
                "SGLANG_APPROX_KV_LOSSY": "true",
                "SGLANG_APPROX_KV_PREFETCH": "0",
            }
        )
        self.assertTrue(enabled.core_enabled)
        self.assertTrue(enabled.lossy_recovery_enabled)
        self.assertFalse(enabled.prefetch_enabled)
        with self.assertRaisesRegex(
            ValueError,
            "SGLANG_APPROX_KV_CORE",
        ):
            ApproxKVFeatureConfig.from_env({"SGLANG_APPROX_KV_PREFETCH": "1"})

    def test_store_generation_identity_and_residency(self):
        store = ApproxKVSegmentStore()
        tokens = (1, 2, 3)
        key = make_key(tokens)
        old = store.register(
            key=key,
            token_ids=tokens,
            source_start=4,
            residency=ResidencyTier.HOST,
            backend_ref="host",
        )
        new = store.register(
            key=key,
            token_ids=tokens,
            source_start=8,
            residency=ResidencyTier.DEVICE,
            backend_ref="device",
        )
        self.assertFalse(store.is_current(old))
        self.assertTrue(store.is_current(new))
        with self.assertRaisesRegex(ValueError, "token_hash"):
            store.register(
                key=key,
                token_ids=(1, 2, 4),
                source_start=0,
                residency=ResidencyTier.DEVICE,
                backend_ref=None,
            )

        loader = RecordingLoader()
        host = store.register(
            key=make_key((4, 5, 6), "host"),
            token_ids=(4, 5, 6),
            source_start=0,
            residency=ResidencyTier.HOST,
            backend_ref="host-ref",
        )
        loaded = store.ensure_resident(
            host,
            ResidencyTier.DEVICE,
            loader,
        )
        self.assertEqual(len(loader.calls), 1)
        self.assertEqual(loaded.backend_ref, "device-ref")

    def test_leases_capacity_and_disposal(self):
        disposed = []

        def release_backend(ref, tier):
            disposed.append((ref, tier))

        store = ApproxKVSegmentStore(max_records=1)
        first_tokens = (1, 2)
        first = store.register(
            key=make_key(first_tokens, "first"),
            token_ids=first_tokens,
            source_start=0,
            residency=ResidencyTier.DEVICE,
            backend_ref="first",
            release_backend=release_backend,
        )
        lease = store.pin(first, ttl_s=1)
        with self.assertRaisesRegex(RuntimeError, "fully pinned"):
            store.register(
                key=make_key((3, 4), "second"),
                token_ids=(3, 4),
                source_start=0,
                residency=ResidencyTier.DEVICE,
                backend_ref="second",
            )
        self.assertEqual(
            store.gc_expired_leases(lease.expires_at_s + 1),
            1,
        )
        self.assertTrue(store.release(first))
        self.assertEqual(
            disposed,
            [("first", ResidencyTier.DEVICE)],
        )

    def test_complete_copy_and_dense_head(self):
        tokens = tuple(range(10))
        manager = ApproxKVManager(
            ApproxKVFeatureConfig(
                core_enabled=True,
                lossy_recovery_enabled=True,
            )
        )
        handle = manager.register_segment(
            key=make_key(tokens),
            token_ids=tokens,
            source_start=0,
            residency=ResidencyTier.DEVICE,
            backend_ref="kv",
        )
        self.assertIsNotNone(handle)
        plan = KVReusePlan(
            target_token_ids=tokens,
            recovery_mode=RecoveryMode.EPIC_FIXED_K,
            dense_ranges=(DenseRange(0, 2, "head"),),
            copied_spans=(
                TransferSpan(
                    source=handle,
                    source_offset=2,
                    target_start=2,
                    length=8,
                    rope_delta=0,
                    chunk_start=0,
                    chunk_length=10,
                ),
            ),
            require_full_coverage=True,
        )
        backend = RecordingBackend()
        stats = manager.execute(plan, backend)
        self.assertEqual(backend.dense, [(0, 2, "head")])
        self.assertEqual(backend.copies, [("kv", 2, 2, 8, 0)])
        self.assertEqual(stats.recomputed_tokens, 2)
        self.assertTrue(stats.mechanically_valid)

    def test_source_mismatch_falls_back_complete_chunk(self):
        source = tuple(range(10))
        target = tuple(range(9)) + (99,)
        manager = ApproxKVManager(
            ApproxKVFeatureConfig(
                core_enabled=True,
                lossy_recovery_enabled=True,
            )
        )
        handle = manager.register_segment(
            key=make_key(source),
            token_ids=source,
            source_start=0,
            residency=ResidencyTier.DEVICE,
            backend_ref="kv",
        )
        plan = KVReusePlan(
            target_token_ids=target,
            recovery_mode=RecoveryMode.RAW_ROPE,
            dense_ranges=(DenseRange(0, 2, "head"),),
            copied_spans=(TransferSpan(handle, 2, 2, 8, 0, 0, 10),),
            require_full_coverage=True,
        )
        backend = RecordingBackend()
        stats = manager.execute(plan, backend)
        self.assertEqual(backend.copies, [])
        self.assertEqual(
            backend.dense,
            [(0, 10, "source_slice_mismatch")],
        )
        self.assertEqual(stats.recomputed_tokens, 10)
        self.assertTrue(stats.mechanically_valid)

    def test_speed_only_plan_allows_token_mismatch(self):
        source = tuple(range(4))
        target = (9, 8, 7, 6)
        manager = ApproxKVManager(
            ApproxKVFeatureConfig(
                core_enabled=True,
                lossy_recovery_enabled=True,
            )
        )
        handle = manager.register_segment(
            key=make_key(source),
            token_ids=source,
            source_start=0,
            residency=ResidencyTier.DEVICE,
            backend_ref="kv",
        )
        plan = KVReusePlan(
            target_token_ids=target,
            recovery_mode=RecoveryMode.RAW_ROPE,
            copied_spans=(TransferSpan(handle, 0, 0, 4, 0, 0, 4),),
            require_full_coverage=True,
            allow_token_mismatch=True,
        )
        backend = RecordingBackend()
        stats = manager.execute(plan, backend)
        self.assertEqual(backend.copies, [("kv", 0, 0, 4, 0)])
        self.assertEqual(stats.source_slice_mismatch, 0)
        self.assertTrue(stats.mechanically_valid)

    def test_gap_and_partial_rope_are_rejected(self):
        tokens = tuple(range(4))
        manager = ApproxKVManager(ApproxKVFeatureConfig(core_enabled=True))
        handle = manager.register_segment(
            key=make_key(tokens),
            token_ids=tokens,
            source_start=0,
            residency=ResidencyTier.DEVICE,
            backend_ref="kv",
        )
        gap_plan = KVReusePlan(
            target_token_ids=tokens,
            copied_spans=(TransferSpan(handle, 0, 1, 3, 1, 1, 3),),
            require_full_coverage=True,
        )
        with self.assertRaisesRegex(ValueError, "unowned target gap"):
            manager.execute(gap_plan, RecordingBackend())

        full_plan = KVReusePlan(
            target_token_ids=tokens,
            copied_spans=(TransferSpan(handle, 0, 0, 4, -7, 0, 4),),
            require_full_coverage=True,
        )
        with self.assertRaisesRegex(
            KVTransferInvariantError,
            "full RoPE",
        ):
            manager.execute(
                full_plan,
                RecordingBackend(rotate_all=False),
            )


if __name__ == "__main__":
    unittest.main()
