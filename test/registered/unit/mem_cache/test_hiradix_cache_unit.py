"""Unit tests for srt/mem_cache/hiradix_cache.py KV cache events."""

import os
import unittest
from array import array

import torch

from sglang.srt.disaggregation.kv_events import BlockStored, StorageMedium
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.cache_policy import (
    CacheProtectionMetadata,
    PrefetchMode,
)
from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=15, stage="stage-b", runner_config="1-gpu-small-amd")

PAGE_SIZE = 2


class TestHiRadixCacheKVEvents(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is required for HiRadixCache tests.")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29601")
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="gloo", rank=0, world_size=1)

    def _build_cache(self):
        server_args = ServerArgs(
            model_path="dummy",
            page_size=PAGE_SIZE,
            hicache_io_backend="direct",
            hicache_mem_layout="layer_first",
            hicache_write_policy="write_through",
        )
        set_global_server_args_for_scheduler(server_args)
        req_to_token_pool = ReqToTokenPool(
            size=10,
            max_context_len=512,
            device="cuda",
            enable_memory_saver=False,
        )
        kv_pool = MHATokenToKVPool(
            size=256,
            page_size=PAGE_SIZE,
            dtype=torch.bfloat16,
            head_num=2,
            head_dim=64,
            layer_num=4,
            device="cuda",
            enable_memory_saver=False,
        )
        allocator = TokenToKVPoolAllocator(
            size=256,
            dtype=torch.bfloat16,
            device="cuda",
            kvcache=kv_pool,
            need_sort=False,
        )
        params = CacheInitParams(
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=allocator,
            page_size=PAGE_SIZE,
            disable=False,
            enable_kv_cache_events=True,
            tp_cache_group=torch.distributed.group.WORLD,
        )
        cache = HiRadixCache(params, server_args)
        # Disable hit-count-driven write-through; tests back up explicitly.
        cache.write_through_threshold = 1 << 30
        return cache, allocator

    def _insert(self, cache, allocator, tokens):
        key = RadixKey(array("q", tokens))
        value = allocator.alloc(len(tokens))
        self.assertIsNotNone(value)
        return cache.insert(InsertParams(key=key, value=value[: len(tokens)]))

    def _leaf_for(self, cache, tokens):
        match = cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", tokens))))
        self.assertIsNot(match.last_device_node, cache.root_node)
        return match.last_device_node

    def _stored_cpu_events(self, cache):
        return [
            e
            for e in cache.take_events()
            if isinstance(e, BlockStored) and e.medium == StorageMedium.CPU
        ]

    def test_cache_protection_metadata_survives_split(self):
        cache, allocator = self._build_cache()
        first = CacheProtectionMetadata("first", next_use_step=5)
        second = CacheProtectionMetadata("second", next_use_step=2)

        first_value = allocator.alloc(4)
        second_value = allocator.alloc(4)
        self.assertIsNotNone(first_value)
        self.assertIsNotNone(second_value)
        cache.insert(
            InsertParams(
                key=RadixKey(array("q", [1, 2, 3, 4])),
                value=first_value,
                cache_protection=(first,),
            )
        )
        cache.insert(
            InsertParams(
                key=RadixKey(array("q", [1, 2, 5, 6])),
                value=second_value,
                cache_protection=(second,),
            )
        )

        shared = self._leaf_for(cache, [1, 2])
        first_leaf = self._leaf_for(cache, [1, 2, 3, 4])
        second_leaf = self._leaf_for(cache, [1, 2, 5, 6])
        self.assertEqual(
            set(shared.cache_protection.objects),
            {"first", "second"},
        )
        self.assertEqual(
            set(first_leaf.cache_protection.objects),
            {"first"},
        )
        self.assertEqual(
            set(second_leaf.cache_protection.objects),
            {"second"},
        )

    def test_free_space_prefetch_loads_back_and_releases_lock(self):
        cache, allocator = self._build_cache()
        cache.load_back_threshold = 1
        item = CacheProtectionMetadata(
            "target",
            protected_tokens=4,
            current_step=0,
            next_use_step=1,
            next_use_request_step=3,
            recoverable_from_lower_tier=True,
        )
        value = allocator.alloc(4)
        self.assertIsNotNone(value)
        cache.insert(
            InsertParams(
                key=RadixKey(array("q", [1, 2, 3, 4])),
                value=value,
                cache_protection=(item,),
            )
        )
        node = cache.find_cache_object_node("target")
        self.assertIsNotNone(node)
        self.assertEqual(cache.write_backup(node, write_back=True), 4)
        cache.writing_check(write_back=True)
        self.assertTrue(node.backuped)

        cache.evict(EvictParams(num_tokens=4))
        self.assertTrue(node.evicted)
        decision = cache.prefetch_cache_object(
            object_id="target",
            mode=PrefetchMode.FREE_SPACE_ONLY,
            target_next_use_step=3,
        )

        self.assertTrue(decision.admitted)
        self.assertEqual(decision.loaded_tokens, 4)
        self.assertFalse(node.evicted)
        self.assertEqual(node.lock_ref, 0)
        self.assertNotIn(node.id, cache.ongoing_load_back)

    def test_dead_object_prefetch_evicts_dynamic_suffix_subtree(self):
        cache, allocator = self._build_cache()
        cache.load_back_threshold = 1
        target = CacheProtectionMetadata(
            "target",
            protected_tokens=4,
            current_step=0,
            next_use_request_step=8,
            recoverable_from_lower_tier=True,
        )
        target_value = allocator.alloc(4)
        self.assertIsNotNone(target_value)
        cache.insert(
            InsertParams(
                key=RadixKey(array("q", [1, 2, 3, 4])),
                value=target_value,
                cache_protection=(target,),
            )
        )
        target_node = cache.find_cache_object_node("target")
        self.assertEqual(cache.write_backup(target_node, write_back=True), 4)
        cache.writing_check(write_back=True)
        cache.evict(EvictParams(num_tokens=4))
        self.assertTrue(target_node.evicted)

        victim = CacheProtectionMetadata(
            "dead",
            protected_tokens=4,
            current_step=1,
            retired=True,
        )
        victim_value = allocator.alloc(5)
        self.assertIsNotNone(victim_value)
        cache.insert(
            InsertParams(
                key=RadixKey(array("q", [10, 11, 12, 13, 99])),
                value=victim_value,
                cache_protection=(victim,),
            )
        )
        pressure = allocator.alloc(allocator.available_size() - 3)
        self.assertIsNotNone(pressure)
        try:
            decision = cache.prefetch_cache_object(
                object_id="target",
                mode=PrefetchMode.DEAD_OBJECT_ONLY,
                target_next_use_step=8,
            )
        finally:
            allocator.free(pressure)

        self.assertTrue(decision.admitted)
        self.assertGreaterEqual(decision.victim_tokens, 4)
        self.assertGreaterEqual(decision.loaded_tokens, 4)
        self.assertIsNone(cache.find_cache_object_node("dead"))
        self.assertFalse(target_node.evicted)

    def test_split_pending_write_through_publishes_fragments(self):
        cache, allocator = self._build_cache()
        cache.take_events()

        self._insert(cache, allocator, [1, 2, 3, 4])
        node = self._leaf_for(cache, [1, 2, 3, 4])
        backed_up = cache.write_backup(node, write_back=True)
        self.assertGreater(backed_up, 0)

        # Split the node while its write-through DMA is still pending.
        self._insert(cache, allocator, [1, 2, 5, 6])
        self.assertEqual(self._stored_cpu_events(cache), [])

        cache.writing_check(write_back=True)

        # Both split fragments must be published, with intact parentage.
        stored_cpu = self._stored_cpu_events(cache)
        self.assertEqual(
            [list(e.token_ids) for e in stored_cpu],
            [[1, 2], [3, 4]],
        )
        self.assertIsNone(stored_cpu[0].parent_block_hash)
        self.assertEqual(stored_cpu[1].parent_block_hash, stored_cpu[0].block_hashes[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
