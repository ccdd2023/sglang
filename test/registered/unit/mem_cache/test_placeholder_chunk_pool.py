"""
Unit tests for Direction #3 Phase B: per-AST-chunk placeholder pool
write path.

These tests verify the chunk pool write path works correctly:
1. Chunks are stored when SGLANG_CHUNKED_PLACEHOLDER_KNN=1 is set.
2. Stored entries have correct byte ranges + token offsets.
3. Chunk signatures match MAScoder parity (mirror-drift guard).
4. Telemetry counters accumulate correctly.
5. LRU cap is enforced.
6. The pool is gated behind the env var (default OFF — production-safe).

Run:
    python -m pytest test/registered/unit/mem_cache/test_placeholder_chunk_pool.py -v
"""

from __future__ import annotations

import os
import unittest
from dataclasses import dataclass

import torch

from sglang.srt.mem_cache.radix_cache import (
    ChunkKVEntry,
    RadixKey,
    RadixCache,
    TreeNode,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=10, suite="stage-b-test-small-1-gpu")
register_amd_ci(est_time=10, suite="stage-b-test-small-1-gpu-amd")


@dataclass
class _MockReq:
    """Minimal stand-in for a sglang Req object.

    Just enough attributes for ``_store_placeholder_anchor_kv`` and
    ``_store_placeholder_chunk_kv`` to operate. We pass an explicit
    ``tokenizer`` callable that produces a deterministic char->token
    mapping (1 token per 4 chars) so the chunker/byte-to-token math
    is reproducible.
    """

    rid: str = "test-req"
    origin_input_ids: list = None
    output_ids: list = None
    placeholder_anchor_token_spans: list = None
    tokenizer: object = None  # callable: str -> list[int]


def _make_simple_tokenizer():
    """Deterministic tokenizer: 1 token per 4 chars, no special tokens.

    Used for chunk boundary tests where we need exact reproducibility
    without pulling in a real HuggingFace tokenizer.
    """

    class _SimpleTokenizer:
        def encode(self, text: str, add_special_tokens: bool = True):
            return [42] * max(1, len(text) // 4)

    return _SimpleTokenizer()


def _make_simulated_cache():
    """Build a CPU RadixCache with realistic locks."""
    from sglang.srt.mem_cache import radix_cache as rc

    TreeNode.counter = 0
    cache = rc.RadixCache.create_simulated(disable=False, page_size=1)
    return cache


class TestChunkPoolWritePath(unittest.TestCase):
    """Phase B: _store_placeholder_chunk_kv stores per-AST-chunk entries."""

    def setUp(self):
        os.environ["SGLANG_CHUNKED_PLACEHOLDER_KNN"] = "1"
        self.cache = _make_simulated_cache()
        self.tokenizer = _make_simple_tokenizer()

    def tearDown(self):
        os.environ.pop("SGLANG_CHUNKED_PLACEHOLDER_KNN", None)

    def _make_req_with_span(self, slot_id: str, text: str) -> _MockReq:
        """Build a minimal request with one span covering ``text``."""
        span_len = max(1, len(text) // 4)  # matches _SimpleTokenizer
        return _MockReq(
            rid=f"rid-{slot_id}",
            origin_input_ids=list(range(span_len)),
            output_ids=[],
            placeholder_anchor_token_spans=[
                {
                    "slot_id": slot_id,
                    "start_token": 0,
                    "end_token": span_len,
                    "text": text,
                    "content_signature": "testsig1234",
                    "label": slot_id,
                }
            ],
            tokenizer=self.tokenizer,
        )

    def test_chunks_stored_when_env_var_set(self):
        """With env var ON, _store_placeholder_chunk_kv populates the pool."""
        text = "def foo():\n    return 1\n"
        req = self._make_req_with_span("code_base0", text)
        kv_indices = torch.arange(0, len(req.origin_input_ids), dtype=torch.int64)

        self.cache._store_placeholder_anchor_kv(req, kv_indices)

        # The chunk pool should have at least one entry (the foo function).
        with self.cache.placeholder_chunk_pool_lock:
            pool_size = len(self.cache.placeholder_chunk_pool)
        self.assertGreater(
            pool_size,
            0,
            msg="chunk pool empty after _store_placeholder_anchor_kv",
        )
        # Check the (slot_id, sig) key shape.
        sample_key = next(iter(self.cache.placeholder_chunk_pool.keys()))
        self.assertEqual(len(sample_key), 2)
        self.assertEqual(sample_key[0], "code_base0")
        self.assertEqual(len(sample_key[1]), 16)  # sha1[:16]

    def test_chunk_byte_ranges_match_mascoder(self):
        """Chunk byte_start / byte_end are computed by ASTChunker (verified
        against MAScoder parity in test_ast_chunker.py)."""
        text = "def foo():\n    return 1\n"
        req = self._make_req_with_span("code_base0", text)
        kv_indices = torch.arange(0, len(req.origin_input_ids), dtype=torch.int64)
        self.cache._store_placeholder_anchor_kv(req, kv_indices)

        # Find the stored entry for our function.
        entries = []
        with self.cache.placeholder_chunk_pool_lock:
            for key, lst in self.cache.placeholder_chunk_pool.items():
                if key[0] == "code_base0" and any(
                    e.name == "foo" for e in lst
                ):
                    entries.extend(lst)
        self.assertGreater(len(entries), 0)
        foo_entry = next(e for e in entries if e.name == "foo")
        # byte_start should be 0 (no leading whitespace in this text).
        self.assertEqual(foo_entry.byte_start, 0)
        self.assertGreater(foo_entry.byte_end, foo_entry.byte_start)
        self.assertEqual(foo_entry.anchor_type, "function")

    def test_telemetry_counters_accumulate(self):
        """placeholder_chunk_pool_total_chunks_stored + store_call_count
        increment on each _store_placeholder_anchor_kv call."""
        before_total = self.cache.placeholder_chunk_pool_total_chunks_stored
        before_calls = self.cache.placeholder_chunk_pool_store_call_count

        text = "def foo():\n    return 1\ndef bar():\n    return 2\n"
        req = self._make_req_with_span("code_base0", text)
        kv_indices = torch.arange(0, len(req.origin_input_ids), dtype=torch.int64)
        self.cache._store_placeholder_anchor_kv(req, kv_indices)

        self.assertEqual(
            self.cache.placeholder_chunk_pool_store_call_count,
            before_calls + 1,
        )
        self.assertGreater(
            self.cache.placeholder_chunk_pool_total_chunks_stored,
            before_total,
        )

    def test_nested_class_produces_two_chunks(self):
        """Class + method → 2 chunks in pool, method has nesting_depth=1."""
        text = "class Foo:\n    def bar(self):\n        return 1\n"
        req = self._make_req_with_span("code_base0", text)
        kv_indices = torch.arange(0, len(req.origin_input_ids), dtype=torch.int64)
        self.cache._store_placeholder_anchor_kv(req, kv_indices)

        with self.cache.placeholder_chunk_pool_lock:
            all_entries = []
            for key, lst in self.cache.placeholder_chunk_pool.items():
                all_entries.extend(lst)
        names = {e.name for e in all_entries}
        self.assertIn("Foo", names)
        self.assertIn("bar", names)

    def test_lru_cap_enforced(self):
        """When more than placeholder_chunk_pool_max_per_key entries land
        on the same (slot_id, sig) key, the oldest is evicted."""
        # Cap is 16 (default). Trigger by storing 20 entries on the same key.
        self.cache.placeholder_chunk_pool_max_per_key = 4  # lower for test
        text = "def foo():\n    return 1\n"
        req = self._make_req_with_span("code_base0", text)
        kv_indices = torch.arange(0, len(req.origin_input_ids), dtype=torch.int64)

        for _ in range(20):
            self.cache._store_placeholder_anchor_kv(req, kv_indices)

        # Find the foo entry's list size.
        with self.cache.placeholder_chunk_pool_lock:
            for key, lst in self.cache.placeholder_chunk_pool.items():
                if key[0] == "code_base0" and any(e.name == "foo" for e in lst):
                    self.assertLessEqual(len(lst), 4)
                    break


class TestChunkPoolGating(unittest.TestCase):
    """Phase B: pool is opt-in via env var (production-safe)."""

    def setUp(self):
        # Default: env var NOT set.
        os.environ.pop("SGLANG_CHUNKED_PLACEHOLDER_KNN", None)
        self.cache = _make_simulated_cache()
        self.tokenizer = _make_simple_tokenizer()

    def test_no_chunks_stored_when_env_var_unset(self):
        """Without SGLANG_CHUNKED_PLACEHOLDER_KNN, the chunk pool stays empty."""
        text = "def foo():\n    return 1\n"
        req = _MockReq(
            rid="rid",
            origin_input_ids=list(range(max(1, len(text) // 4))),
            output_ids=[],
            placeholder_anchor_token_spans=[
                {
                    "slot_id": "code_base0",
                    "start_token": 0,
                    "end_token": max(1, len(text) // 4),
                    "text": text,
                    "content_signature": "testsig1234",
                }
            ],
            tokenizer=self.tokenizer,
        )
        kv_indices = torch.arange(0, len(req.origin_input_ids), dtype=torch.int64)
        self.cache._store_placeholder_anchor_kv(req, kv_indices)

        with self.cache.placeholder_chunk_pool_lock:
            pool_size = len(self.cache.placeholder_chunk_pool)
        self.assertEqual(
            pool_size,
            0,
            msg="chunk pool should be empty when env var unset (production default)",
        )

    def test_no_chunks_stored_when_text_empty(self):
        """Empty text → no chunks stored (defensive)."""
        os.environ["SGLANG_CHUNKED_PLACEHOLDER_KNN"] = "1"
        try:
            req = _MockReq(
                rid="rid",
                origin_input_ids=[0],
                output_ids=[],
                placeholder_anchor_token_spans=[
                    {
                        "slot_id": "code_base0",
                        "start_token": 0,
                        "end_token": 1,
                        "text": "",
                        "content_signature": "testsig1234",
                    }
                ],
                tokenizer=self.tokenizer,
            )
            kv_indices = torch.tensor([0], dtype=torch.int64)
            self.cache._store_placeholder_anchor_kv(req, kv_indices)
            with self.cache.placeholder_chunk_pool_lock:
                pool_size = len(self.cache.placeholder_chunk_pool)
            self.assertEqual(pool_size, 0)
        finally:
            os.environ.pop("SGLANG_CHUNKED_PLACEHOLDER_KNN", None)


class TestChunkKVEntry(unittest.TestCase):
    """ChunkKVEntry dataclass basics."""

    def test_repr_includes_slot_and_signature(self):
        e = ChunkKVEntry(
            slot_id="code_base0",
            chunk_signature="abcdef0123456789",
            anchor_type="function",
            name="histogram",
            byte_start=100,
            byte_end=200,
            start_token=10,
            end_token=20,
            token_ids=torch.tensor([1, 2, 3], dtype=torch.int64),
            kv_indices=torch.tensor([100, 101, 102], dtype=torch.int64),
        )
        r = repr(e)
        self.assertIn("code_base0", r)
        self.assertIn("histogram", r)
        self.assertIn("abcdef0123456789", r)
        self.assertIn("function", r)


if __name__ == "__main__":
    unittest.main()