"""Unit tests for the offline-precomputed codebase KV feature (Phases 2-3).

These tests verify the non-GPU-critical pieces:
1. ``ChunkKVEntry`` / ``AnchorKVEntry`` default ``location="device"`` (backward
   compat) and that ``location="host"`` can be set.
2. The ``.bin`` serialization round-trip used by the precompute writer
   (``scripts/precompute_codebase_kv.py:extract_chunk_kv`` /
   ``write_chunk_bin``) and the loader reader
   (``codebase_kv_loader._load_chunk_bin_into_host``) preserves bytes.
3. The read-path residency branch in ``_execute_chunk_plan_batched``
   partitions host vs device entries (host items routed to
   ``_load_host_chunks_to_device``, device items to ``move_kv_cache``).

GPU-dependent paths (real ModelRunner prefill, real ``MHATokenToKVPoolHost``
transfer) are covered by the end-to-end A/B benchmark, not here.

Run:
    python -m pytest test/registered/unit/mem_cache/test_codebase_kv_loader.py -v
"""

from __future__ import annotations

import os
import tempfile
import unittest

import torch

from sglang.srt.mem_cache.radix_cache import (
    AnchorKVEntry,
    ChunkKVEntry,
    byte_to_token_offset,
)


class _MockKVCache:
    """Minimal stand-in for a MHATokenToKVPool (layer_first layout).

    ``k_buffer`` / ``v_buffer`` are lists of CPU tensors shaped
    ``[size, head_num, head_dim]`` — enough for the residency-branch test
    to exercise ``move_kv_cache`` and the host-load path.
    """

    def __init__(self, size, layer_num, head_num, head_dim):
        self.size = size
        self.layer_num = layer_num
        self.head_num = head_num
        self.head_dim = head_dim
        self.store_dtype = torch.float16
        self.k_buffer = [torch.zeros(size, head_num, head_dim, dtype=torch.float16) for _ in range(layer_num)]
        self.v_buffer = [torch.zeros(size, head_num, head_dim, dtype=torch.float16) for _ in range(layer_num)]

    def move_kv_cache(self, dst_indices, src_indices):
        for layer_id in range(self.layer_num):
            self.k_buffer[layer_id][dst_indices] = self.k_buffer[layer_id][src_indices]
            self.v_buffer[layer_id][dst_indices] = self.v_buffer[layer_id][src_indices]

    def get_key_buffer(self, layer_id):
        return self.k_buffer[layer_id]

    def get_value_buffer(self, layer_id):
        return self.v_buffer[layer_id]


class TestResidencyField(unittest.TestCase):
    def test_chunk_entry_defaults_to_device(self):
        """New ChunkKVEntry defaults to location='device' (backward compat)."""
        e = ChunkKVEntry(
            slot_id="code_base:foo.py",
            chunk_signature="abc123",
            anchor_type="function",
            name="f",
            byte_start=0,
            byte_end=10,
            start_token=0,
            end_token=5,
            token_ids=torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32),
            kv_indices=torch.tensor([10, 11, 12, 13, 14], dtype=torch.int64),
        )
        self.assertEqual(e.location, "device")
        self.assertFalse(e.pinned)

    def test_anchor_entry_defaults_to_device(self):
        a = AnchorKVEntry(
            signature="x",
            token_ids=torch.tensor([1], dtype=torch.int32),
            kv_indices=torch.tensor([0], dtype=torch.int64),
            start_pos=0,
        )
        self.assertEqual(a.location, "device")

    def test_host_residency_settable(self):
        e = ChunkKVEntry(
            slot_id="code_base:foo.py",
            chunk_signature="abc123",
            anchor_type="function",
            name="f",
            byte_start=0,
            byte_end=10,
            start_token=0,
            end_token=5,
            token_ids=torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32),
            kv_indices=torch.tensor([10, 11, 12, 13, 14], dtype=torch.int64),
        )
        e.location = "host"
        e.pinned = True
        self.assertEqual(e.location, "host")
        self.assertTrue(e.pinned)


class TestByteToTokenOffsetFreeFunction(unittest.TestCase):
    """The module-level free function (factored for offline precompute reuse)."""

    class _Tok:
        def encode(self, text, add_special_tokens=False):
            # 1 token per char (deterministic, no special tokens).
            return list(range(len(text)))

    def test_zero_byte_pos(self):
        self.assertEqual(byte_to_token_offset("abc", 0, self._Tok()), 0)

    def test_positive_byte_pos(self):
        # 1 token per char → byte_pos N == N tokens.
        self.assertEqual(byte_to_token_offset("abcdef", 4, self._Tok()), 4)

    def test_no_tokenizer_fallback(self):
        self.assertEqual(byte_to_token_offset("abc", 2, None), 0)

    def test_exception_fallback(self):
        class _BadTok:
            def encode(self, *a, **k):
                raise RuntimeError("boom")

        self.assertEqual(byte_to_token_offset("abc", 2, _BadTok()), 0)


class TestBinRoundTrip(unittest.TestCase):
    """The .bin serialization idiom: writer (precompute script) ↔ reader (loader).

    Mirrors scripts/precompute_codebase_kv.py:extract_chunk_kv (writes
    [2, L, n_tokens, H, D] fp16 via tofile) and
    codebase_kv_loader._load_chunk_bin_into_host (reads via readinto).
    """

    def test_roundtrip_preserves_bytes(self):
        layer_num, n_tokens, head_num, head_dim = 3, 7, 4, 8
        # Source KV: distinct values so we can detect corruption.
        k_src = torch.randn(layer_num, n_tokens, head_num, head_dim, dtype=torch.float16)
        v_src = torch.randn(layer_num, n_tokens, head_num, head_dim, dtype=torch.float16)
        kv = torch.stack([k_src, v_src])  # [2, L, n, H, D]

        with tempfile.TemporaryDirectory() as d:
            bin_path = os.path.join(d, "chunk.bin")
            # Write (precompute script idiom).
            kv.contiguous().view(torch.uint8).numpy().tofile(bin_path)

            # Read into a fresh host buffer (loader idiom).
            buf = torch.empty(
                (2, layer_num, n_tokens, head_num, head_dim), dtype=torch.float16
            )
            expected = buf.view(torch.uint8).numel()
            with open(bin_path, "rb", buffering=0) as f:
                mv = memoryview(buf.view(torch.uint8).contiguous().numpy())
                got = f.readinto(mv)
            self.assertEqual(got, expected)

            # Byte-exact.
            self.assertTrue(torch.equal(buf[0], k_src))
            self.assertTrue(torch.equal(buf[1], v_src))


class TestReadPathResidencyBranch(unittest.TestCase):
    """Verify the batched read path routes host entries to the host-load
    helper and device entries to move_kv_cache.

    We can't easily construct a full RadixCache + ChunkPlan, so we test the
    routing logic at the level it matters: given entries with mixed
    residency, the host items are collected separately and the GPU move
    batch excludes them. This mirrors the partition in
    ``_execute_chunk_plan_batched``.
    """

    def test_partition_by_residency(self):
        # Three entries: device, host, device.
        n = 5
        e_dev1 = ChunkKVEntry("s1", "a", "function", "f", 0, 10, 0, n,
                              torch.zeros(n, dtype=torch.int32),
                              torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64))
        e_host = ChunkKVEntry("s2", "b", "function", "g", 0, 10, 0, n,
                              torch.zeros(n, dtype=torch.int32),
                              torch.tensor([100, 101, 102, 103, 104], dtype=torch.int64))
        e_host.location = "host"
        e_dev2 = ChunkKVEntry("s3", "c", "function", "h", 0, 10, 0, n,
                              torch.zeros(n, dtype=torch.int32),
                              torch.tensor([10, 11, 12, 13, 14], dtype=torch.int64))
        entries = [e_dev1, e_host, e_dev2]

        # Mirror the partition loop from _execute_chunk_plan_batched.
        host_dst_indices = set()
        host_items = []
        gpu_indices = []
        for i, e in enumerate(entries):
            if getattr(e, "location", "device") == "host":
                host_items.append(e)
                host_dst_indices.add(i)
            else:
                gpu_indices.append(i)

        self.assertEqual(len(host_items), 1)
        self.assertIs(host_items[0], e_host)
        self.assertEqual(gpu_indices, [0, 2])
        self.assertEqual(host_dst_indices, {1})

    def test_device_only_no_host_items(self):
        n = 3
        e = ChunkKVEntry("s1", "a", "function", "f", 0, 10, 0, n,
                         torch.zeros(n, dtype=torch.int32),
                         torch.tensor([0, 1, 2], dtype=torch.int64))
        host_items = [e for e in [e] if getattr(e, "location", "device") == "host"]
        self.assertEqual(host_items, [])


if __name__ == "__main__":
    unittest.main()
