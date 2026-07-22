from __future__ import annotations

import unittest

import torch

from sglang.srt.mem_cache.approx_kv.radix_backend import (
    DeviceKVRef,
    RadixKVTransferBackend,
    RoPEConfig,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")


class FakeKVCache:
    def __init__(self):
        self.layer_num = 2
        self.k_buffer = [
            torch.randn(16, 2, 8, device="cuda", dtype=torch.float16)
            for _ in range(self.layer_num)
        ]
        self.v_buffer = [
            torch.randn(16, 2, 8, device="cuda", dtype=torch.float16)
            for _ in range(self.layer_num)
        ]

    def move_kv_cache(self, target, source):
        for layer in range(self.layer_num):
            self.k_buffer[layer][target] = self.k_buffer[layer][source]
            self.v_buffer[layer][target] = self.v_buffer[layer][source]

    def get_key_buffer(self, layer_id):
        return self.k_buffer[layer_id]

    def get_value_buffer(self, layer_id):
        return self.v_buffer[layer_id]


class FakeAllocator:
    def __init__(self, kvcache):
        self.kvcache = kvcache

    def get_kvcache(self):
        return self.kvcache


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestApproxKVCuda(unittest.TestCase):
    def test_full_layer_copy_and_nonzero_rope(self):
        kvcache = FakeKVCache()
        allocator = FakeAllocator(kvcache)
        source = torch.tensor([0, 1, 2, 3], device="cuda")
        target = torch.tensor([4, 5, 6, 7], device="cuda")
        source_keys = [
            buffer[source].clone() for buffer in kvcache.k_buffer
        ]
        source_values = [
            buffer[source].clone() for buffer in kvcache.v_buffer
        ]
        backend = RadixKVTransferBackend(
            allocator=allocator,
            target_indices=lambda start, length: target[start : start + length],
            dense_prefill=lambda start, length, reason: None,
            rope=RoPEConfig(
                rotary_dim=8,
                base=10000.0,
                is_neox_style=True,
            ),
        )
        result = backend.copy_and_rotate(
            source_ref=DeviceKVRef(source),
            source_offset=0,
            target_start=0,
            length=4,
            rope_delta=3,
        )
        torch.cuda.synchronize()
        self.assertEqual(result.copied_k_tokens, 4)
        self.assertEqual(result.rotated_k_tokens, 4)
        self.assertEqual(result.copied_v_tokens, 4)
        for layer in range(kvcache.layer_num):
            torch.testing.assert_close(
                kvcache.v_buffer[layer][target],
                source_values[layer],
            )
            self.assertFalse(
                torch.equal(
                    kvcache.k_buffer[layer][target],
                    source_keys[layer],
                )
            )


if __name__ == "__main__":
    unittest.main()
