from __future__ import annotations

import importlib
import math
import sys
import types as python_types
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[4]
PACKAGE_DIR = REPO_ROOT / "python/sglang/srt/mem_cache/approx_kv"
PACKAGE_NAME = "approx_kv_under_test"

package = python_types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(PACKAGE_DIR)]
sys.modules[PACKAGE_NAME] = package

config_module = importlib.import_module(f"{PACKAGE_NAME}.config")
async_module = importlib.import_module(f"{PACKAGE_NAME}.async_transfer")
manager_module = importlib.import_module(f"{PACKAGE_NAME}.manager")
plugins_module = importlib.import_module(f"{PACKAGE_NAME}.plugins")
store_module = importlib.import_module(f"{PACKAGE_NAME}.store")
transfer_module = importlib.import_module(f"{PACKAGE_NAME}.transfer")
types_module = importlib.import_module(f"{PACKAGE_NAME}.types")

ApproxKVFeatureConfig = config_module.ApproxKVFeatureConfig
ApproxKVManager = manager_module.ApproxKVManager
ApproxKVSegmentStore = store_module.ApproxKVSegmentStore
ResidencyLoadResult = store_module.ResidencyLoadResult
DenseRange = types_module.DenseRange
KVLayerTransferResult = types_module.KVLayerTransferResult
KVReusePlan = types_module.KVReusePlan
KVSegmentKey = types_module.KVSegmentKey
KVTransferInvariantError = transfer_module.KVTransferInvariantError
RecoveryMode = types_module.RecoveryMode
ResidencyTier = types_module.ResidencyTier
SegmentKind = types_module.SegmentKind
TransferSpan = types_module.TransferSpan
token_ids_hash = types_module.token_ids_hash
RecoveryPluginRegistry = plugins_module.RecoveryPluginRegistry
AsyncTransferState = async_module.AsyncTransferState


def make_key(
    tokens: tuple[int, ...],
    name: str = "segment",
) -> KVSegmentKey:
    return KVSegmentKey(
        content_hash=name,
        token_hash=token_ids_hash(tokens),
        token_count=len(tokens),
        model_fingerprint="test-model",
        cache_dtype="bf16",
        kind=SegmentKind.ARTIFACT,
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
        return KVLayerTransferResult(
            copied_k_tokens=length,
            rotated_k_tokens=rotated,
            copied_v_tokens=length,
            copy_ms=1.5,
            rope_ms=0.5,
        )

    def dense_prefill(self, *, target_start, length, reason):
        self.dense.append((target_start, length, reason))


class RecordingLoader:
    def __init__(self) -> None:
        self.calls = []

    def load(self, handle, target_tier):
        self.calls.append((handle.key, handle.residency, target_tier))
        return f"{target_tier.value}-ref"


class RecordingAsyncTransfer:
    def __init__(self, result, done=True):
        self.result = result
        self.cancelled = False
        self.ready = done

    @property
    def done(self):
        return self.ready or self.cancelled

    def wait(self, timeout_s=None):
        del timeout_s
        if self.cancelled:
            raise RuntimeError("cancelled")
        if not self.ready:
            raise TimeoutError("not ready")
        return self.result

    def cancel(self):
        self.cancelled = True


class RecordingAsyncLoader:
    def __init__(self, result, done=True):
        self.result = result
        self.done = done
        self.transfers = []

    def begin_load(self, handle, target_tier):
        self.transfers.append((handle, target_tier))
        return RecordingAsyncTransfer(self.result, done=self.done)


class TestApproxKVCore(unittest.TestCase):
    def test_feature_gates(self):
        self.assertEqual(
            ApproxKVFeatureConfig.from_env({}),
            ApproxKVFeatureConfig(),
        )
        enabled = ApproxKVFeatureConfig.from_env(
            {
                "SGLANG_APPROX_KV_CORE": "1",
                "SGLANG_APPROX_KV_HOST": "true",
                "SGLANG_APPROX_KV_PREFETCH": "true",
            }
        )
        self.assertTrue(enabled.core_enabled)
        self.assertTrue(enabled.host_residency_enabled)
        self.assertTrue(enabled.async_prefetch_enabled)
        with self.assertRaisesRegex(
            ValueError,
            "SGLANG_APPROX_KV_CORE",
        ):
            ApproxKVFeatureConfig.from_env({"SGLANG_APPROX_KV_PREFETCH": "1"})
        with self.assertRaisesRegex(
            ValueError,
            "SGLANG_APPROX_KV_HOST",
        ):
            ApproxKVFeatureConfig.from_env(
                {
                    "SGLANG_APPROX_KV_CORE": "1",
                    "SGLANG_APPROX_KV_PREFETCH": "1",
                }
            )

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
        self.assertEqual(store.device_owned_tokens, 6)

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
                release_backend=release_backend,
            )
        self.assertEqual(
            disposed,
            [("second", ResidencyTier.DEVICE)],
        )
        self.assertEqual(
            store.gc_expired_leases(lease.expires_at_s + 1),
            1,
        )
        self.assertTrue(store.release(first))
        self.assertEqual(
            disposed,
            [
                ("second", ResidencyTier.DEVICE),
                ("first", ResidencyTier.DEVICE),
            ],
        )

    def test_finite_lease_expires_without_explicit_gc(self):
        store = ApproxKVSegmentStore()
        handle = store.register(
            key=make_key((1,), "finite"),
            token_ids=(1,),
            source_start=0,
            residency=ResidencyTier.DEVICE,
            backend_ref="finite",
        )
        with patch.object(store_module.time, "monotonic", return_value=10.0):
            lease = store.pin(handle, ttl_s=1.0)
        self.assertEqual(lease.expires_at_s, 11.0)

        with patch.object(store_module.time, "monotonic", return_value=12.0):
            self.assertTrue(store.release(handle))
        self.assertEqual(store.lease_count, 0)

    def test_pin_rejects_non_finite_ttl_and_persistent_gc_is_ignored(self):
        store = ApproxKVSegmentStore()
        handle = store.register(
            key=make_key((1,), "persistent"),
            token_ids=(1,),
            source_start=0,
            residency=ResidencyTier.DEVICE,
            backend_ref="persistent",
        )
        for ttl_s in (0.0, -1.0, math.nan, math.inf, -math.inf):
            with self.subTest(ttl_s=ttl_s):
                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    store.pin(handle, ttl_s=ttl_s)

        lease = store.pin(handle, ttl_s=None)
        self.assertIsNone(lease.expires_at_s)
        self.assertEqual(store.gc_expired_leases(now_s=math.inf), 0)
        self.assertFalse(store.release(handle))
        self.assertTrue(store.unpin(lease))
        self.assertTrue(store.release(handle))

    def test_leased_replacement_releases_new_backend_once(self):
        disposed = []

        def release_backend(ref, tier):
            disposed.append((ref, tier))

        store = ApproxKVSegmentStore()
        tokens = (7, 8)
        key = make_key(tokens, "leased")
        first = store.register(
            key=key,
            token_ids=tokens,
            source_start=0,
            residency=ResidencyTier.HOST,
            backend_ref="first",
            release_backend=release_backend,
        )
        store.pin(first, ttl_s=10)
        with self.assertRaisesRegex(RuntimeError, "leased"):
            store.register(
                key=key,
                token_ids=tokens,
                source_start=1,
                residency=ResidencyTier.HOST,
                backend_ref="replacement",
                release_backend=release_backend,
            )
        self.assertEqual(
            disposed,
            [("replacement", ResidencyTier.HOST)],
        )

    def test_complete_copy_and_dense_head(self):
        tokens = tuple(range(10))
        manager = ApproxKVManager(ApproxKVFeatureConfig(core_enabled=True))
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
            recovery_mode=RecoveryMode.COPY,
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
        self.assertEqual(stats.copy_ms, 1.5)
        self.assertEqual(stats.rope_ms, 0.5)
        self.assertTrue(stats.mechanically_valid)

    def test_source_mismatch_falls_back_complete_chunk(self):
        source = tuple(range(10))
        target = tuple(range(9)) + (99,)
        manager = ApproxKVManager(ApproxKVFeatureConfig(core_enabled=True))
        handle = manager.register_segment(
            key=make_key(source),
            token_ids=source,
            source_start=0,
            residency=ResidencyTier.DEVICE,
            backend_ref="kv",
        )
        plan = KVReusePlan(
            target_token_ids=target,
            recovery_mode=RecoveryMode.COPY,
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

    def test_async_prefetch_commits_and_stale_load_is_released(self):
        released = []

        def release_backend(ref, tier):
            released.append((ref, tier))

        config = ApproxKVFeatureConfig(
            core_enabled=True,
            host_residency_enabled=True,
            async_prefetch_enabled=True,
        )
        manager = ApproxKVManager(config)
        tokens = (10, 11, 12)
        key = make_key(tokens, "async")
        handle = manager.register_segment(
            key=key,
            token_ids=tokens,
            source_start=0,
            residency=ResidencyTier.HOST,
            backend_ref="host",
            release_backend=release_backend,
        )
        loader = RecordingAsyncLoader(
            ResidencyLoadResult(
                backend_ref="device",
                release_backend=release_backend,
                release_on_stale=release_backend,
            )
        )
        manager.bind_async_loader(loader)
        ticket = manager.begin_prefetch(handle)
        loaded = ticket.wait()
        self.assertEqual(loaded.residency, ResidencyTier.DEVICE)
        self.assertEqual(loaded.backend_ref, "device")
        self.assertEqual(manager.active_ticket_count, 0)
        self.assertEqual(released, [("host", ResidencyTier.HOST)])

        stale = manager.store.register(
            key=key,
            token_ids=tokens,
            source_start=1,
            residency=ResidencyTier.HOST,
            backend_ref="new-host",
            release_backend=release_backend,
        )
        stale_ticket = manager.begin_prefetch(stale)
        manager.store.register(
            key=key,
            token_ids=tokens,
            source_start=2,
            residency=ResidencyTier.HOST,
            backend_ref="replacement",
            release_backend=release_backend,
        )
        with self.assertRaisesRegex(
            KeyError,
            "changed while",
        ):
            stale_ticket.wait()
        self.assertIn(("device", ResidencyTier.DEVICE), released)

    def test_concurrent_residency_commit_keeps_first_device_buffer(self):
        released = []

        def release_backend(ref, tier):
            released.append((ref, tier))

        store = ApproxKVSegmentStore()
        tokens = (30, 31)
        handle = store.register(
            key=make_key(tokens, "residency-race"),
            token_ids=tokens,
            source_start=0,
            residency=ResidencyTier.HOST,
            backend_ref=object(),
            release_backend=release_backend,
        )
        first_device = object()
        second_device = object()
        first = store.commit_residency(
            handle,
            target_tier=ResidencyTier.DEVICE,
            result=ResidencyLoadResult(
                backend_ref=first_device,
                release_backend=release_backend,
            ),
        )
        with self.assertRaisesRegex(
            KeyError,
            "residency changed",
        ):
            store.commit_residency(
                handle,
                target_tier=ResidencyTier.DEVICE,
                result=ResidencyLoadResult(
                    backend_ref=second_device,
                    release_backend=release_backend,
                    release_on_stale=release_backend,
                ),
            )
        self.assertIs(first.backend_ref, first_device)
        self.assertIs(store.lookup(first.key).backend_ref, first_device)
        self.assertEqual(
            released,
            [
                (handle.backend_ref, ResidencyTier.HOST),
                (second_device, ResidencyTier.DEVICE),
            ],
        )

    def test_plugin_registry_rejects_duplicates(self):
        class Plugin:
            name = "copy"

            def build_plan(self, context, store):
                del context, store
                return KVReusePlan(target_token_ids=())

            def scheduler_metadata(self, context):
                del context
                return ()

        registry = RecoveryPluginRegistry()
        registry.register(Plugin())
        self.assertEqual(registry.names(), ("copy",))
        with self.assertRaisesRegex(ValueError, "already registered"):
            registry.register(Plugin())

    def test_reset_cancels_active_prefetch_and_releases_store(self):
        manager = ApproxKVManager(
            ApproxKVFeatureConfig(
                core_enabled=True,
                host_residency_enabled=True,
                async_prefetch_enabled=True,
            )
        )
        tokens = (20, 21)
        handle = manager.register_segment(
            key=make_key(tokens, "cancel"),
            token_ids=tokens,
            source_start=0,
            residency=ResidencyTier.HOST,
            backend_ref="host",
        )
        loader = RecordingAsyncLoader(
            ResidencyLoadResult(backend_ref="device"),
            done=False,
        )
        manager.bind_async_loader(loader)
        ticket = manager.begin_prefetch(handle)
        self.assertEqual(manager.active_ticket_count, 1)
        with self.assertRaises(TimeoutError):
            ticket.wait(0)
        self.assertEqual(ticket.state, AsyncTransferState.PENDING)
        manager.reset()
        self.assertTrue(loader.transfers)
        self.assertTrue(ticket.done)
        self.assertEqual(manager.active_ticket_count, 0)
        self.assertEqual(manager.store.record_count, 0)


if __name__ == "__main__":
    unittest.main()
