from __future__ import annotations

import contextlib
import unittest
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.approx_kv.hicache_backend import (
    HiCacheHostKVRef,
    HiCacheResidencyBackend,
)
from sglang.srt.mem_cache.approx_kv.radix_backend import DeviceKVRef
from sglang.srt.mem_cache.approx_kv.types import (
    KVSegmentHandle,
    KVSegmentKey,
    ResidencyTier,
    SegmentKind,
    token_ids_hash,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-c-test-cpu")


class FakeEvent:
    def __init__(self):
        self.recorded = False

    def record(self):
        self.recorded = True

    def wait(self, stream):
        del stream

    def query(self):
        return self.recorded

    def synchronize(self):
        return None

    def elapsed_time(self, other):
        del other
        return 1.25


class FakeDeviceModule:
    Event = FakeEvent

    @staticmethod
    def stream(stream):
        del stream
        return contextlib.nullcontext()


class FakeStream:
    def __init__(self):
        self.synchronized = False

    def synchronize(self):
        self.synchronized = True


class FakeHostPool:
    page_size = 1
    layer_num = 2

    def __init__(self):
        self.free_slots = list(range(64))
        self.backups = []
        self.loads = []
        self.freed = []

    def alloc(self, size):
        if size > len(self.free_slots):
            return None
        result = torch.tensor(self.free_slots[:size], dtype=torch.int64)
        self.free_slots = self.free_slots[size:]
        return result

    def backup_from_device_all_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        io_backend,
    ):
        self.backups.append(
            (device_pool, host_indices.clone(), device_indices.clone(), io_backend)
        )

    def load_to_device_per_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
    ):
        self.loads.append(
            (
                device_pool,
                host_indices.clone(),
                device_indices.clone(),
                layer_id,
                io_backend,
            )
        )

    def free(self, indices):
        self.freed.extend(int(index) for index in indices)
        return len(indices)


class FakeDevicePool:
    layer_num = 2

    @staticmethod
    def get_kv_size_bytes():
        return 6400, 6400


class FakeAllocator:
    size_full = 64

    def __init__(self):
        self.next_index = 100
        self.freed = []

    def alloc(self, size):
        result = torch.arange(
            self.next_index,
            self.next_index + size,
            dtype=torch.int64,
        )
        self.next_index += size
        return result

    def free(self, indices):
        self.freed.extend(int(index) for index in indices)


class FakeController:
    def __init__(self):
        self.mem_pool_host = FakeHostPool()
        self.mem_pool_device = FakeDevicePool()
        self.mem_pool_device_allocator = FakeAllocator()
        self.io_backend = "direct"
        self.write_stream = FakeStream()
        self.load_stream = FakeStream()
        self.layer_num = 2
        self.has_draft = False

    @staticmethod
    def move_indices(host_indices, device_indices):
        return host_indices, device_indices

    def evict_host(self, indices):
        return self.mem_pool_host.free(indices)

    def evict_device(self, indices):
        self.mem_pool_device_allocator.free(indices)
        return len(indices)


def make_handle(host_ref):
    tokens = (1, 2, 3)
    key = KVSegmentKey(
        content_hash="segment",
        token_hash=token_ids_hash(tokens),
        token_count=len(tokens),
        model_fingerprint="model",
        cache_dtype="fp16",
        kind=SegmentKind.ARTIFACT,
    )
    return KVSegmentHandle(
        key=key,
        generation=1,
        residency=ResidencyTier.HOST,
        source_start=0,
        token_ids=tokens,
        backend_ref=host_ref,
    )


class TestApproxKVHiCacheBackend(unittest.TestCase):
    def setUp(self):
        self.controller = FakeController()
        self.backend = HiCacheResidencyBackend(self.controller)
        self.device_patch = patch(
            "sglang.srt.mem_cache.approx_kv.hicache_backend.device_module",
            FakeDeviceModule,
        )
        self.timing_patch = patch(
            "sglang.srt.mem_cache.approx_kv.hicache_backend.make_timing_event_pair",
            lambda: (FakeEvent(), FakeEvent(), True),
        )
        self.device_patch.start()
        self.timing_patch.start()

    def tearDown(self):
        self.timing_patch.stop()
        self.device_patch.stop()

    def test_real_pool_export_and_async_load(self):
        export = self.backend.begin_export(
            DeviceKVRef(torch.tensor([10, 11, 12], dtype=torch.int64))
        )
        self.assertTrue(export.done)
        host_result = export.wait()
        self.assertIsInstance(host_result.backend_ref, HiCacheHostKVRef)
        self.assertEqual(host_result.num_tokens, 3)
        self.assertEqual(host_result.bytes_transferred, 600)
        self.assertEqual(host_result.duration_ms, 1.25)
        self.assertEqual(len(self.controller.mem_pool_host.backups), 1)

        load = self.backend.begin_load(
            make_handle(host_result.backend_ref),
            ResidencyTier.DEVICE,
        )
        self.assertTrue(load.done)
        device_result = load.wait()
        self.assertIsInstance(device_result.backend_ref, DeviceKVRef)
        self.assertEqual(len(self.controller.mem_pool_host.loads), 2)
        self.assertEqual(device_result.num_tokens, 3)
        device_result.release_backend(
            device_result.backend_ref,
            ResidencyTier.DEVICE,
        )
        self.assertEqual(
            self.controller.mem_pool_device_allocator.freed,
            [100, 101, 102],
        )
        host_result.release_backend(
            host_result.backend_ref,
            ResidencyTier.HOST,
        )
        self.assertEqual(self.controller.mem_pool_host.freed, [0, 1, 2])

    def test_cancel_releases_pending_device_allocation(self):
        host_ref = HiCacheHostKVRef(torch.tensor([0, 1, 2], dtype=torch.int64))
        transfer = self.backend.begin_load(
            make_handle(host_ref),
            ResidencyTier.DEVICE,
        )
        transfer.cancel()
        self.assertEqual(
            self.controller.mem_pool_device_allocator.freed,
            [100, 101, 102],
        )

    def test_export_exception_synchronizes_before_host_release(self):
        with patch.object(
            self.backend,
            "_record_stream",
            side_effect=RuntimeError("injected export failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "injected export failure"):
                self.backend.begin_export(
                    DeviceKVRef(torch.tensor([10, 11, 12], dtype=torch.int64))
                )
        self.assertTrue(self.controller.write_stream.synchronized)
        self.assertEqual(self.controller.mem_pool_host.freed, [0, 1, 2])

    def test_load_exception_synchronizes_before_device_release(self):
        host_ref = HiCacheHostKVRef(torch.tensor([0, 1, 2], dtype=torch.int64))
        with patch.object(
            self.backend,
            "_record_stream",
            side_effect=RuntimeError("injected load failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "injected load failure"):
                self.backend.begin_load(
                    make_handle(host_ref),
                    ResidencyTier.DEVICE,
                )
        self.assertTrue(self.controller.load_stream.synchronized)
        self.assertEqual(
            self.controller.mem_pool_device_allocator.freed,
            [100, 101, 102],
        )


if __name__ == "__main__":
    unittest.main()
