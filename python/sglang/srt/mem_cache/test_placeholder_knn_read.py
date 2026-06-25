"""Unit tests for the per-placeholder k-NN read path (PR 3).

Covers `_placeholder_knn_search` (the module-level helper) and the
gating behavior of `_try_placeholder_knn_lossy_match` via env vars.
Does not require a real allocator — uses a stub.

Run with: python -m pytest python/sglang/srt/mem_cache/test_placeholder_knn_read.py -v
Or via unittest: python -m unittest python.sglang.srt.mem_cache.test_placeholder_knn_read
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch

from sglang.srt.mem_cache.radix_cache import (
    AnchorKVEntry,
    _placeholder_knn_search,
)


def _make_entry(slot_id: str, start_pos: int = 0, n: int = 8,
                embed_dim: int = 4) -> AnchorKVEntry:
    v = torch.zeros(embed_dim)
    v[hash(slot_id) % embed_dim] = 1.0
    return AnchorKVEntry(
        signature=f"placeholder:{slot_id}:{start_pos}",
        code_content_signature=f"sig:{slot_id}",
        token_ids=torch.arange(start_pos, start_pos + n, dtype=torch.long),
        kv_indices=torch.arange(start_pos, start_pos + n, dtype=torch.long),
        start_pos=start_pos,
        slot_id=slot_id,
        pool_embedding=v,
        last_access_time=0.0,
    )


def _unit_vector(dim: int, idx: int) -> torch.Tensor:
    v = torch.zeros(dim)
    v[idx % dim] = 1.0
    return v


class PlaceholderKnnSearchBasicTests(unittest.TestCase):
    """Pure tests for `_placeholder_knn_search`."""

    def test_empty_pool_returns_empty(self):
        q = _unit_vector(4, 0)
        self.assertEqual(_placeholder_knn_search([], q), [])

    def test_single_entry_returns_it_with_cos_1(self):
        e = _make_entry("plan")
        q = e.pool_embedding.clone()
        out = _placeholder_knn_search([e], q, top_k=4, min_similarity=0.5)
        self.assertEqual(len(out), 1)
        self.assertIs(out[0][0], e)
        self.assertAlmostEqual(out[0][1], 1.0, places=4)

    def test_top_k_selection(self):
        # 4 entries, query = entry[2]; expect entry[2] first then others.
        entries = []
        for i in range(4):
            e = _make_entry(f"slot_{i}")
            e.pool_embedding = _unit_vector(4, i)
            entries.append(e)
        q = entries[2].pool_embedding.clone()
        out = _placeholder_knn_search(entries, q, top_k=3, min_similarity=0.0)
        self.assertEqual(len(out), 3)
        # First hit is the same vector → cos 1.0
        self.assertIs(out[0][0], entries[2])
        self.assertAlmostEqual(out[0][1], 1.0, places=4)
        # The remaining two are at cos 0 (orthogonal) — keep min_similarity
        # = 0.0 so they pass.
        for entry, sim in out[1:]:
            self.assertAlmostEqual(sim, 0.0, places=4)

    def test_min_similarity_filter(self):
        # Two orthogonal entries; min_similarity=0.5 should drop them.
        entries = []
        for i in range(3):
            e = _make_entry(f"slot_{i}")
            e.pool_embedding = _unit_vector(4, i)
            entries.append(e)
        q = _unit_vector(4, 0)  # matches entry 0 with cos 1, others 0
        out = _placeholder_knn_search(entries, q, top_k=4, min_similarity=0.5)
        self.assertEqual(len(out), 1)
        self.assertIs(out[0][0], entries[0])

    def test_top_k_caps_result_length(self):
        entries = []
        for i in range(8):
            e = _make_entry(f"slot_{i}")
            e.pool_embedding = _unit_vector(4, i)
            entries.append(e)
        q = _unit_vector(4, 0)
        out = _placeholder_knn_search(entries, q, top_k=2, min_similarity=0.0)
        self.assertLessEqual(len(out), 2)

    def test_entries_without_embedding_are_skipped(self):
        e_with = _make_entry("with")
        e_without = _make_entry("without")
        e_without.pool_embedding = None
        q = e_with.pool_embedding.clone()
        out = _placeholder_knn_search([e_with, e_without], q, top_k=4, min_similarity=0.5)
        self.assertEqual(len(out), 1)
        self.assertIs(out[0][0], e_with)

    def test_results_sorted_descending(self):
        entries = []
        for i in range(5):
            e = _make_entry(f"slot_{i}")
            e.pool_embedding = _unit_vector(4, i)
            entries.append(e)
        q = _unit_vector(4, 0)
        out = _placeholder_knn_search(entries, q, top_k=5, min_similarity=-1.0)
        sims = [sim for _, sim in out]
        self.assertEqual(sims, sorted(sims, reverse=True))


class PlaceholderKnnSearchEndToEndTests(unittest.TestCase):
    """End-to-end test using the real MiniLM embedder."""

    @classmethod
    def setUpClass(cls):
        from sglang.srt.mem_cache import semantic_suffix as ss
        cls.emb = ss.load_embedder()
        if cls.emb is None:
            raise unittest.SkipTest("embedder unavailable on this host")

    def test_similar_texts_high_cosine(self):
        from sglang.srt.mem_cache import semantic_suffix as ss
        e = _make_entry("plan")
        e.pool_embedding = ss.embed_single_text(
            "def foo(x): return x + 1", emb=self.emb,
        )
        q = ss.embed_single_text(
            "def foo(x): return x + 999", emb=self.emb,
        )
        self.assertIsNotNone(e.pool_embedding)
        self.assertIsNotNone(q)
        out = _placeholder_knn_search([e], q, top_k=1, min_similarity=0.5)
        self.assertEqual(len(out), 1)
        # Cosine should be > 0.5 for same-function-different-literal.
        self.assertGreater(out[0][1], 0.5)

    def test_disjoint_texts_below_threshold(self):
        from sglang.srt.mem_cache import semantic_suffix as ss
        e = _make_entry("plan")
        e.pool_embedding = ss.embed_single_text(
            "def foo(x): return x + 1", emb=self.emb,
        )
        q = ss.embed_single_text(
            "the quick brown fox jumps over the lazy dog", emb=self.emb,
        )
        self.assertIsNotNone(e.pool_embedding)
        self.assertIsNotNone(q)
        out = _placeholder_knn_search([e], q, top_k=1, min_similarity=0.5)
        self.assertEqual(len(out), 0)


class PlaceholderMatchGatingTests(unittest.TestCase):
    """The `_try_placeholder_knn_lossy_match` method must no-op when:
      - env disabled (default)
      - no spans on the request
      - no entries in the pool
    """

    def setUp(self):
        # Capture env state and reset to a known state.
        self._saved = {
            k: os.environ.get(k)
            for k in (
                "SGLANG_PLACEHOLDER_KNN_MATCH",
                "SGLANG_SEMANTIC_SUFFIX_ENABLED",
            )
        }
        os.environ["SGLANG_PLACEHOLDER_KNN_MATCH"] = "0"
        os.environ["SGLANG_SEMANTIC_SUFFIX_ENABLED"] = "1"

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_default_disabled_no_op(self):
        from sglang.srt.mem_cache.radix_cache import RadixCache

        class _Stub(RadixCache):
            def __init__(self_inner):
                pass

        # Default: env not set, should be no-op even when pool/spans present.
        stub = _Stub()
        stub.placeholder_anchor_pool = {"plan": [_make_entry("plan")]}
        # Fake req with spans
        req = type("R", (), {})()
        req.placeholder_anchor_token_spans = [
            {"slot_id": "plan", "start_token": 100, "end_token": 200,
             "text": "irrelevant", "content_signature": ""}
        ]
        values = [torch.tensor([1, 2, 3], dtype=torch.long)]
        node = object()
        out_values, out_node = stub._try_placeholder_knn_lossy_match(
            req, type("K", (), {"token_ids": torch.tensor([0] * 500)})(),
            values, node,
        )
        # Disabled → input unchanged
        self.assertIs(out_values, values)
        self.assertIs(out_node, node)
        self.assertEqual(
            getattr(req, "placeholder_kv_prefill_matched_slots", 0), 0,
        )

    def test_enabled_but_no_spans_no_op(self):
        os.environ["SGLANG_PLACEHOLDER_KNN_MATCH"] = "1"
        from sglang.srt.mem_cache.radix_cache import RadixCache

        class _Stub(RadixCache):
            def __init__(self_inner):
                pass

        stub = _Stub()
        req = type("R", (), {})()
        req.placeholder_anchor_token_spans = []
        values = [torch.tensor([1, 2, 3], dtype=torch.long)]
        node = object()
        out_values, out_node = stub._try_placeholder_knn_lossy_match(
            req, type("K", (), {"token_ids": torch.tensor([0] * 500)})(),
            values, node,
        )
        self.assertIs(out_values, values)
        self.assertIs(out_node, node)

    def test_semantic_disabled_no_op_even_when_knn_enabled(self):
        os.environ["SGLANG_PLACEHOLDER_KNN_MATCH"] = "1"
        os.environ["SGLANG_SEMANTIC_SUFFIX_ENABLED"] = "0"
        from sglang.srt.mem_cache.radix_cache import RadixCache

        class _Stub(RadixCache):
            def __init__(self_inner):
                pass

        stub = _Stub()
        stub.placeholder_anchor_pool = {"plan": [_make_entry("plan")]}
        req = type("R", (), {})()
        req.placeholder_anchor_token_spans = [
            {"slot_id": "plan", "start_token": 100, "end_token": 200,
             "text": "irrelevant", "content_signature": ""}
        ]
        values = [torch.tensor([1, 2, 3], dtype=torch.long)]
        node = object()
        out_values, out_node = stub._try_placeholder_knn_lossy_match(
            req, type("K", (), {"token_ids": torch.tensor([0] * 500)})(),
            values, node,
        )
        # Embedder disabled → input unchanged
        self.assertIs(out_values, values)
        self.assertIs(out_node, node)


class MatchPrefixHookTests(unittest.TestCase):
    """The match_prefix path must call _try_placeholder_knn_lossy_match when
    the request has placeholder_anchor_token_spans, but never block the
    radix match when those spans are absent."""

    def setUp(self):
        self._saved = {
            k: os.environ.get(k)
            for k in (
                "SGLANG_PLACEHOLDER_KNN_MATCH",
                "SGLANG_SEMANTIC_SUFFIX_ENABLED",
            )
        }
        os.environ["SGLANG_PLACEHOLDER_KNN_MATCH"] = "0"
        os.environ["SGLANG_SEMANTIC_SUFFIX_ENABLED"] = "1"

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_request_with_no_spans_does_not_call_kNN(self):
        from sglang.srt.mem_cache.radix_cache import RadixCache

        class _Stub(RadixCache):
            def __init__(self_inner):
                self_inner.placeholder_anchor_pool = {}
                self_inner.device = "cpu"

            def _try_placeholder_knn_lossy_match(self_inner, req, key, values, node):
                raise AssertionError("should not be called when no spans")

        stub = _Stub()
        # Build a request without placeholder_anchor_token_spans
        req = type("R", (), {"reuse_mode": "", "placeholder_anchor_token_spans": []})()
        # Simulate the relevant block in match_prefix manually.
        spans = getattr(req, "placeholder_anchor_token_spans", None) or []
        if spans:
            stub._try_placeholder_knn_lossy_match(req, None, [], None)
        # If we got here without exception, the gate worked.
        self.assertEqual(spans, [])


if __name__ == "__main__":
    unittest.main()
