"""Unit tests for semantic_suffix.py.

Run with: python -m pytest python/sglang/srt/mem_cache/test_semantic_suffix.py -v

Or via unittest: python -m unittest python.sglang.srt.mem_cache.test_semantic_suffix
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Ensure the sglang package is importable when this file is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch

from sglang.srt.mem_cache import semantic_suffix as ss


def _unit_chunk(dim: int, dim_size: int = 384) -> torch.Tensor:
    """Build a unit vector pointing at dim."""
    v = torch.zeros(dim_size)
    v[dim] = 1.0
    return v


def _make_chunks(layout: list[int], dim_size: int = 384) -> torch.Tensor:
    """layout[i] = the dim that chunk i points at."""
    out = torch.zeros(len(layout), dim_size)
    for i, dim in enumerate(layout):
        out[i, dim] = 1.0
    return out


def _chunk_with_cosine_to_unit(
    dim_unit: int, target_cos: float, dim_size: int = 384
) -> torch.Tensor:
    """Construct a chunk vector that has exactly `target_cos` cosine to a unit
    vector at `dim_unit`, with ||v||=1."""
    v = torch.zeros(dim_size)
    v[dim_unit] = target_cos
    v[(dim_unit + 1) % dim_size] = (1.0 - target_cos * target_cos) ** 0.5
    return v


class CosineProfileTests(unittest.TestCase):
    """Pure-Python tests for `cosine_profile` (no embedder required)."""

    def test_identity_full_length(self):
        chunks = _make_chunks([0, 1, 2, 3])
        out = ss.cosine_profile(
            chunks, chunks,
            min_cosine_threshold=0.7, chunk_token_size=64, min_chunk_count=1,
        )
        self.assertEqual(out, 4 * 64)

    def test_first_two_of_three_match(self):
        req = _make_chunks([0, 1, 0])
        entry = _make_chunks([0, 1, 2])
        out = ss.cosine_profile(
            req, entry,
            min_cosine_threshold=0.7, chunk_token_size=64, min_chunk_count=1,
        )
        self.assertEqual(out, 2 * 64)

    def test_first_chunk_differs(self):
        req = _make_chunks([2, 1, 0])
        entry = _make_chunks([0, 1, 2])
        out = ss.cosine_profile(
            req, entry,
            min_cosine_threshold=0.7, chunk_token_size=64, min_chunk_count=1,
        )
        self.assertEqual(out, 0)

    def test_req_shorter_than_entry(self):
        req = _make_chunks([0, 1])
        entry = _make_chunks([0, 1, 2, 3])
        out = ss.cosine_profile(
            req, entry,
            min_cosine_threshold=0.7, chunk_token_size=64, min_chunk_count=1,
        )
        self.assertEqual(out, 2 * 64)

    def test_entry_shorter_than_req(self):
        req = _make_chunks([0, 1, 2, 3])
        entry = _make_chunks([0, 1])
        out = ss.cosine_profile(
            req, entry,
            min_cosine_threshold=0.7, chunk_token_size=64, min_chunk_count=1,
        )
        self.assertEqual(out, 2 * 64)

    def test_min_chunks_floor(self):
        chunks = _make_chunks([0, 1])
        out = ss.cosine_profile(
            chunks, chunks,
            min_cosine_threshold=0.7, chunk_token_size=64, min_chunk_count=3,
        )
        self.assertEqual(out, 0)

    def test_cosine_above_threshold(self):
        req = _make_chunks([0, 1])
        entry = torch.zeros(2, 384)
        entry[0, 0] = 1.0
        entry[1] = _chunk_with_cosine_to_unit(1, 0.75)  # above 0.7
        out = ss.cosine_profile(
            req, entry,
            min_cosine_threshold=0.7, chunk_token_size=64, min_chunk_count=1,
        )
        self.assertEqual(out, 2 * 64)

    def test_cosine_below_threshold(self):
        req = _make_chunks([0, 1])
        entry = torch.zeros(2, 384)
        entry[0, 0] = 1.0
        entry[1] = _chunk_with_cosine_to_unit(1, 0.65)  # below 0.7
        out = ss.cosine_profile(
            req, entry,
            min_cosine_threshold=0.7, chunk_token_size=64, min_chunk_count=1,
        )
        self.assertEqual(out, 1 * 64)

    def test_empty_inputs(self):
        empty = torch.zeros(0, 384)
        self.assertEqual(
            ss.cosine_profile(empty, empty,
                             min_cosine_threshold=0.7, chunk_token_size=64,
                             min_chunk_count=1),
            0,
        )
        self.assertEqual(
            ss.cosine_profile(None, None,
                             min_cosine_threshold=0.7, chunk_token_size=64,
                             min_chunk_count=1),
            0,
        )

    def test_chunk_size_scales_length(self):
        chunks = _make_chunks([0, 1, 2])
        out_64 = ss.cosine_profile(
            chunks, chunks,
            min_cosine_threshold=0.7, chunk_token_size=64, min_chunk_count=1,
        )
        out_128 = ss.cosine_profile(
            chunks, chunks,
            min_cosine_threshold=0.7, chunk_token_size=128, min_chunk_count=1,
        )
        self.assertEqual(out_64, 3 * 64)
        self.assertEqual(out_128, 3 * 128)


class EnvKnobTests(unittest.TestCase):
    """Env-var knobs read at call time, not import time."""

    def setUp(self):
        self._saved = {
            k: ss.os.environ.get(k)
            for k in (
                "SGLANG_SEMANTIC_SUFFIX_ENABLED",
                "SGLANG_SEMANTIC_SUFFIX_CHUNK_TOKENS",
                "SGLANG_SEMANTIC_SUFFIX_MIN_COSINE",
                "SGLANG_SEMANTIC_SUFFIX_MIN_CHUNKS",
            )
        }

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                ss.os.environ.pop(k, None)
            else:
                ss.os.environ[k] = v

    def test_default_enabled(self):
        ss.os.environ.pop("SGLANG_SEMANTIC_SUFFIX_ENABLED", None)
        self.assertTrue(ss.is_enabled())

    def test_disable_via_env(self):
        ss.os.environ["SGLANG_SEMANTIC_SUFFIX_ENABLED"] = "0"
        self.assertFalse(ss.is_enabled())

    def test_chunk_tokens_default(self):
        ss.os.environ.pop("SGLANG_SEMANTIC_SUFFIX_CHUNK_TOKENS", None)
        self.assertEqual(ss.chunk_tokens(), 64)

    def test_chunk_tokens_override(self):
        ss.os.environ["SGLANG_SEMANTIC_SUFFIX_CHUNK_TOKENS"] = "32"
        self.assertEqual(ss.chunk_tokens(), 32)

    def test_min_cosine_default(self):
        ss.os.environ.pop("SGLANG_SEMANTIC_SUFFIX_MIN_COSINE", None)
        self.assertAlmostEqual(ss.min_cosine(), 0.70)

    def test_min_cosine_override(self):
        ss.os.environ["SGLANG_SEMANTIC_SUFFIX_MIN_COSINE"] = "0.85"
        self.assertAlmostEqual(ss.min_cosine(), 0.85)

    def test_min_chunks_default(self):
        ss.os.environ.pop("SGLANG_SEMANTIC_SUFFIX_MIN_CHUNKS", None)
        self.assertEqual(ss.min_chunks(), 1)

    def test_min_chunks_override(self):
        ss.os.environ["SGLANG_SEMANTIC_SUFFIX_MIN_CHUNKS"] = "3"
        self.assertEqual(ss.min_chunks(), 3)


class EmbedderLazyLoadTests(unittest.TestCase):
    """The embedder singleton survives tests but resets cleanly via
    reset_for_tests()."""

    def setUp(self):
        ss.reset_for_tests()

    def tearDown(self):
        ss.reset_for_tests()

    def test_reset_clears_state(self):
        # Pre-poison the global
        ss._EMBEDDER = object()
        ss._EMBEDDER_LOAD_FAILED = True
        ss.reset_for_tests()
        self.assertIsNone(ss._EMBEDDER)
        self.assertFalse(ss._EMBEDDER_LOAD_FAILED)

    def test_load_embedder_returns_object_or_none(self):
        # This may take ~6s on first call due to model load. Use a tight
        # skip if the model is not available.
        emb = ss.load_embedder()
        # Either we got a valid embedder or None (offline / load failure).
        if emb is not None:
            self.assertTrue(hasattr(emb, "tokenizer"))
            self.assertTrue(hasattr(emb, "model"))
            self.assertEqual(emb.dim, 384)
        else:
            self.assertTrue(ss._EMBEDDER_LOAD_FAILED)


class EndToEndEmbedderTests(unittest.TestCase):
    """End-to-end test using the real MiniLM model. Skipped if load fails."""

    @classmethod
    def setUpClass(cls):
        cls.emb = ss.load_embedder()
        if cls.emb is None:
            raise unittest.SkipTest("embedder unavailable on this host")

    def test_real_code_similarity(self):
        # Same function, different return literal -> high cosine
        a = ss._embed_texts(
            ["def foo(x):\n    return x + 1", "def foo(x):\n    return x + 1"],
            self.emb,
        )
        b = ss._embed_texts(
            ["def foo(x):\n    return x + 1", "def foo(x):\n    return x + 999"],
            self.emb,
        )
        # First chunk cosine = 1.0 (identical), second = ~0.78 (same structure)
        out = ss.cosine_profile(
            a, b, min_cosine_threshold=0.7, chunk_token_size=64, min_chunk_count=1,
        )
        self.assertEqual(out, 2 * 64)

    def test_real_code_divergence(self):
        # Completely different functions -> low cosine
        a = ss._embed_texts(
            ["def foo(x):\n    return x + 1", "def foo(x):\n    return x + 1"],
            self.emb,
        )
        b = ss._embed_texts(
            ["the quick brown fox", "jumps over the lazy dog"],
            self.emb,
        )
        out = ss.cosine_profile(
            a, b, min_cosine_threshold=0.7, chunk_token_size=64, min_chunk_count=1,
        )
        self.assertEqual(out, 0)


class EmbedSingleTextTests(unittest.TestCase):
    """Unit tests for the `embed_single_text` helper added for the
    per-placeholder k-NN pool (Duke 2026 KVCOMM-style)."""

    def setUp(self):
        ss.reset_for_tests()
        self._saved_enabled = ss.os.environ.get(
            "SGLANG_SEMANTIC_SUFFIX_ENABLED",
        )

    def tearDown(self):
        ss.reset_for_tests()
        if self._saved_enabled is None:
            ss.os.environ.pop("SGLANG_SEMANTIC_SUFFIX_ENABLED", None)
        else:
            ss.os.environ["SGLANG_SEMANTIC_SUFFIX_ENABLED"] = self._saved_enabled

    def test_returns_none_when_disabled(self):
        ss.os.environ["SGLANG_SEMANTIC_SUFFIX_ENABLED"] = "0"
        self.assertIsNone(ss.embed_single_text("hello world"))

    def test_returns_none_for_empty_string_when_disabled(self):
        ss.os.environ["SGLANG_SEMANTIC_SUFFIX_ENABLED"] = "0"
        self.assertIsNone(ss.embed_single_text(""))

    def test_shape_when_enabled_and_load_succeeds(self):
        # May skip if embedder unavailable on this host.
        emb = ss.load_embedder()
        if emb is None:
            self.skipTest("embedder unavailable on this host")
        # Ensure is_enabled() is true during the test
        old = ss.os.environ.get("SGLANG_SEMANTIC_SUFFIX_ENABLED")
        ss.os.environ["SGLANG_SEMANTIC_SUFFIX_ENABLED"] = "1"
        try:
            out = ss.embed_single_text("hello world", emb=emb)
        finally:
            if old is None:
                ss.os.environ.pop("SGLANG_SEMANTIC_SUFFIX_ENABLED", None)
            else:
                ss.os.environ["SGLANG_SEMANTIC_SUFFIX_ENABLED"] = old
        self.assertIsNotNone(out)
        self.assertEqual(out.dim(), 1)
        self.assertEqual(out.shape[0], emb.dim)
        # L2-normalized: ||v|| ~ 1.0
        self.assertAlmostEqual(float(out.norm().item()), 1.0, places=4)

    def test_same_text_same_embedding(self):
        emb = ss.load_embedder()
        if emb is None:
            self.skipTest("embedder unavailable on this host")
        old = ss.os.environ.get("SGLANG_SEMANTIC_SUFFIX_ENABLED")
        ss.os.environ["SGLANG_SEMANTIC_SUFFIX_ENABLED"] = "1"
        try:
            a = ss.embed_single_text("def foo(x): return x + 1", emb=emb)
            b = ss.embed_single_text("def foo(x): return x + 1", emb=emb)
        finally:
            if old is None:
                ss.os.environ.pop("SGLANG_SEMANTIC_SUFFIX_ENABLED", None)
            else:
                ss.os.environ["SGLANG_SEMANTIC_SUFFIX_ENABLED"] = old
        self.assertIsNotNone(a)
        self.assertIsNotNone(b)
        # Identical text → cosine 1.0
        cos = float((a * b).sum().item())
        self.assertAlmostEqual(cos, 1.0, places=4)

    def test_different_text_low_similarity(self):
        emb = ss.load_embedder()
        if emb is None:
            self.skipTest("embedder unavailable on this host")
        old = ss.os.environ.get("SGLANG_SEMANTIC_SUFFIX_ENABLED")
        ss.os.environ["SGLANG_SEMANTIC_SUFFIX_ENABLED"] = "1"
        try:
            a = ss.embed_single_text("def foo(x): return x + 1", emb=emb)
            b = ss.embed_single_text("the quick brown fox jumps over the lazy dog", emb=emb)
        finally:
            if old is None:
                ss.os.environ.pop("SGLANG_SEMANTIC_SUFFIX_ENABLED", None)
            else:
                ss.os.environ["SGLANG_SEMANTIC_SUFFIX_ENABLED"] = old
        self.assertIsNotNone(a)
        self.assertIsNotNone(b)
        cos = float((a * b).sum().item())
        self.assertLess(cos, 0.5)


class EmbeddingCacheTests(unittest.TestCase):
    """O7: LRU cache for query embeddings keyed by text.  Saves ~24ms
    (MiniLM forward) per repeat lookup in the k-NN body.
    """

    def setUp(self):
        ss.reset_for_tests()
        ss.reset_embed_cache_for_tests()
        self._emb = ss.load_embedder()
        if self._emb is None:
            self.skipTest("embedder unavailable on this host")

    def tearDown(self):
        ss.reset_embed_cache_for_tests()

    def test_cache_hit_returns_same_embedding(self):
        """Calling embed_single_text_cached twice with the same text
        returns the same embedding object (cached)."""
        text = "test text for caching"
        e1 = ss.embed_single_text_cached(text, emb=self._emb)
        e2 = ss.embed_single_text_cached(text, emb=self._emb)
        self.assertIsNotNone(e1)
        self.assertIsNotNone(e2)
        # Cached: should be the same object (LRU returns same tensor).
        self.assertIs(e1, e2)

    def test_cache_miss_different_text(self):
        """Different text produces different cached entries."""
        e1 = ss.embed_single_text_cached("text one", emb=self._emb)
        e2 = ss.embed_single_text_cached("text two", emb=self._emb)
        self.assertIsNotNone(e1)
        self.assertIsNotNone(e2)
        self.assertIsNot(e1, e2)
        self.assertEqual(len(ss._EMBED_CACHE), 2)

    def test_cache_lru_eviction(self):
        """When cache fills, oldest entries are evicted (LRU)."""
        ss._EMBED_CACHE_MAX = 3  # tighten for test
        ss._EMBED_CACHE.clear()
        ss.embed_single_text_cached("t1", emb=self._emb)
        ss.embed_single_text_cached("t2", emb=self._emb)
        ss.embed_single_text_cached("t3", emb=self._emb)
        self.assertEqual(len(ss._EMBED_CACHE), 3)
        # Adding a 4th evicts t1.
        ss.embed_single_text_cached("t4", emb=self._emb)
        self.assertEqual(len(ss._EMBED_CACHE), 3)
        self.assertNotIn("t1", ss._EMBED_CACHE)
        self.assertIn("t4", ss._EMBED_CACHE)


if __name__ == "__main__":
    unittest.main()
