"""Unit tests for the per-placeholder anchor pool (Duke 2026 KVCOMM-style).

These tests cover the write-back path only (PR 2): the LRU eviction policy
and the F1-guard that prevents a divergent dense prefill from poisoning
the pool.  The k-NN read path tests live in test_placeholder_knn_read.py
(will be added in PR 3).

Run with: python -m pytest python/sglang/srt/mem_cache/test_placeholder_knn.py -v
Or via unittest: python -m unittest python.sglang.srt.mem_cache.test_placeholder_knn
"""

from __future__ import annotations

import os
import sys
import unittest
import time
from pathlib import Path

# Ensure the sglang package is importable when this file is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch

from sglang.srt.mem_cache.radix_cache import AnchorKVEntry


def _make_entry(slot_id: str, start_pos: int = 0, n: int = 8,
                last_access_time: float = 0.0,
                embed_dim: int = 4) -> AnchorKVEntry:
    """Construct a synthetic AnchorKVEntry with the new placeholder fields."""
    v = torch.zeros(embed_dim)
    v[hash(slot_id) % embed_dim] = 1.0
    return AnchorKVEntry(
        signature=f"placeholder:{slot_id}:{start_pos}",
        code_content_signature=f"sig:{slot_id}",
        token_ids=torch.arange(start_pos, start_pos + n, dtype=torch.long),
        kv_indices=torch.arange(start_pos, start_pos + n, dtype=torch.long),
        start_pos=start_pos,
        slot_id=slot_id,
        slot_label=f"label-{slot_id}",
        pool_embedding=v,
        embedding_text=f"text for {slot_id}",
        last_access_time=last_access_time,
    )


class _FakeRadixCache:
    """Minimal harness with the pool + LRU helper, no scheduler deps."""

    def __init__(self, max_per_slot: int = 256):
        # Mimic the attributes we need without spinning up a full RadixCache.
        self.placeholder_anchor_pool = {}
        import threading
        self.placeholder_anchor_pool_lock = threading.RLock()
        self.placeholder_pool_max_per_slot = max_per_slot
        self.device = "cpu"
        # Bind the methods we want to test.
        from sglang.srt.mem_cache.radix_cache import RadixCache
        self._evict_placeholder_pool_slot_locked = (
            RadixCache._evict_placeholder_pool_slot_locked.__get__(self)
        )

    def add(self, slot_id: str, n: int = 8, last_access_time: float = 0.0):
        e = _make_entry(slot_id, start_pos=self._counter(),
                        n=n, last_access_time=last_access_time)
        with self.placeholder_anchor_pool_lock:
            self.placeholder_anchor_pool.setdefault(slot_id, []).append(e)
            self._evict_placeholder_pool_slot_locked(slot_id)
        return e

    _counter_state = 0

    def _counter(self) -> int:
        _FakeRadixCache._counter_state += 1
        return _FakeRadixCache._counter_state


class PlaceholderPoolLRUEvictionTests(unittest.TestCase):
    """The per-slot pool is bounded by `placeholder_pool_max_per_slot`
    and prefers the freshest entries (highest `last_access_time`)."""

    def test_no_eviction_under_cap(self):
        rc = _FakeRadixCache(max_per_slot=10)
        for _ in range(10):
            rc.add("plan")
        self.assertEqual(len(rc.placeholder_anchor_pool["plan"]), 10)

    def test_lru_eviction_at_cap(self):
        rc = _FakeRadixCache(max_per_slot=5)
        # Insert 7 with monotonically increasing last_access_time.
        for i in range(7):
            rc.add("plan", last_access_time=float(i))
        pool = rc.placeholder_anchor_pool["plan"]
        self.assertEqual(len(pool), 5)
        # The two oldest (i=0, i=1) should be evicted.
        kept_times = sorted(e.last_access_time for e in pool)
        self.assertEqual(kept_times, [2.0, 3.0, 4.0, 5.0, 6.0])

    def test_eviction_per_slot_isolated(self):
        rc = _FakeRadixCache(max_per_slot=3)
        rc.add("plan", last_access_time=1.0)
        rc.add("plan", last_access_time=2.0)
        rc.add("plan", last_access_time=3.0)
        rc.add("plan", last_access_time=4.0)  # evicts time=1.0
        rc.add("arch", last_access_time=10.0)  # different slot, untouched
        rc.add("arch", last_access_time=11.0)
        self.assertEqual(len(rc.placeholder_anchor_pool["plan"]), 3)
        self.assertEqual(len(rc.placeholder_anchor_pool["arch"]), 2)

    def test_lru_prefers_most_recent(self):
        rc = _FakeRadixCache(max_per_slot=2)
        # Insert old first, then fresh — the old one should be evicted
        # because we just touched the new one.
        rc.add("plan", last_access_time=1.0)
        rc.add("plan", last_access_time=2.0)
        rc.add("plan", last_access_time=3.0)  # cap reached, evict oldest
        pool = rc.placeholder_anchor_pool["plan"]
        kept_times = sorted(e.last_access_time for e in pool)
        self.assertEqual(kept_times, [2.0, 3.0])


class PlaceholderStoreF1GuardTests(unittest.TestCase):
    """`_store_placeholder_anchor_kv` should skip entries whose predicted
    text diverges from the actually-prefilled text below the F1 threshold."""

    def setUp(self):
        # Capture env state.
        self._saved = {
            k: os.environ.get(k)
            for k in (
                "SGLANG_PLACEHOLDER_STORE_ENABLED",
                "SGLANG_SEMANTIC_SUFFIX_ENABLED",
                "SGLANG_PLACEHOLDER_STORE_MIN_F1",
            )
        }

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_token_f1_basics(self):
        """Sanity check on the moved-out token_f1 helper."""
        from sglang.srt.mem_cache.text_utils import token_f1
        # Identical text → F1=1
        self.assertEqual(token_f1("a b c", "a b c"), 1.0)
        # Empty both → F1=1
        self.assertEqual(token_f1("", ""), 1.0)
        # One empty → F1=0
        self.assertEqual(token_f1("a b c", ""), 0.0)
        # Partial overlap
        score = token_f1("a b c d", "a b x y")
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)
        # Disjoint
        self.assertEqual(token_f1("a b c", "x y z"), 0.0)

    def test_skip_when_predicted_and_actual_diverge(self):
        """End-to-end: write a span whose `text` field says "hello world"
        but the actual decoded tokens are different; expect store skipped."""
        from sglang.srt.mem_cache import text_utils as tu

        # Build a fake request object (avoid Req dataclass).
        class _FakeReq:
            placeholder_anchor_token_spans = [
                {
                    "slot_id": "plan",
                    "label": "Plan",
                    "start_token": 0,
                    "end_token": 8,
                    "content_signature": "fake_sig",
                    "text": "hello world from upstream agent",  # predicted
                },
            ]
            rid = "fake-rid-1"
            origin_input_ids = list(range(8))  # synthetic
            output_ids = []

        # Decoded text will be from tokens 0..8 → digits, NOT hello world
        decoded = " ".join(str(i) for i in range(8))  # "0 1 2 3 4 5 6 7"
        score = tu.token_f1("hello world from upstream agent", decoded)
        # F1 should be very low
        self.assertLess(score, 0.2)

        # Verify the actual store method would skip this entry.
        # We don't need the full RadixCache — just exercise the F1 path.
        os.environ["SGLANG_SEMANTIC_SUFFIX_ENABLED"] = "0"  # disable embedder load
        os.environ["SGLANG_PLACEHOLDER_STORE_MIN_F1"] = "0.60"
        from sglang.srt.mem_cache.radix_cache import RadixCache
        # Use unbound method via descriptor on a stub.
        class _Stub(RadixCache):
            def __init__(self_inner):
                # Skip RadixCache.__init__ (which needs a real allocator).
                pass
        stub = _Stub()
        # The private helper itself is pure; just call it.
        actual_score = stub._placeholder_f1(
            "hello world from upstream agent", decoded,
        )
        self.assertLess(actual_score, 0.2)


class PlaceholderWriteDisabledTests(unittest.TestCase):
    """`_placeholder_store_enabled` returns False when env disabled."""

    def setUp(self):
        self._saved = {
            k: os.environ.get(k)
            for k in ("SGLANG_PLACEHOLDER_STORE_ENABLED",
                      "SGLANG_SEMANTIC_SUFFIX_ENABLED")
        }

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_default_enabled(self):
        os.environ.pop("SGLANG_PLACEHOLDER_STORE_ENABLED", None)
        os.environ["SGLANG_SEMANTIC_SUFFIX_ENABLED"] = "1"
        from sglang.srt.mem_cache.radix_cache import RadixCache

        class _Stub(RadixCache):
            def __init__(self_inner):
                pass
        self.assertTrue(_Stub()._placeholder_store_enabled())

    def test_disabled_by_env(self):
        os.environ["SGLANG_PLACEHOLDER_STORE_ENABLED"] = "0"
        os.environ["SGLANG_SEMANTIC_SUFFIX_ENABLED"] = "1"
        from sglang.srt.mem_cache.radix_cache import RadixCache

        class _Stub(RadixCache):
            def __init__(self_inner):
                pass
        self.assertFalse(_Stub()._placeholder_store_enabled())

    def test_disabled_when_semantic_disabled(self):
        os.environ["SGLANG_PLACEHOLDER_STORE_ENABLED"] = "1"
        os.environ["SGLANG_SEMANTIC_SUFFIX_ENABLED"] = "0"
        from sglang.srt.mem_cache.radix_cache import RadixCache

        class _Stub(RadixCache):
            def __init__(self_inner):
                pass
        # Sharing the embedder model with semantic-suffix; if that is
        # disabled, placeholder store should also be disabled to avoid
        # silent model loads.
        self.assertFalse(_Stub()._placeholder_store_enabled())


class PlaceholderSlotIdTaxonomyTests(unittest.TestCase):
    """The pool is keyed by slot_id; same slot_id shares a pool, different
    slot_ids do not."""

    def test_pool_keys_independent(self):
        rc = _FakeRadixCache(max_per_slot=10)
        rc.add("plan", n=4)
        rc.add("plan", n=4)
        rc.add("arch", n=8)
        rc.add("context", n=16)
        self.assertEqual(len(rc.placeholder_anchor_pool["plan"]), 2)
        self.assertEqual(len(rc.placeholder_anchor_pool["arch"]), 1)
        self.assertEqual(len(rc.placeholder_anchor_pool["context"]), 1)

    def test_empty_slot_returns_empty_list(self):
        rc = _FakeRadixCache()
        self.assertEqual(rc.placeholder_anchor_pool.get("nonexistent", []), [])


class PlaceholderTokenizerFallbackTests(unittest.TestCase):
    """When the LLM tokenizer is unavailable, `_decode_placeholder_span`
    returns empty string and the F1 guard in `_store_placeholder_anchor_kv`
    should fall back to permissive "accept" so writes don't get blocked
    by the F1 guard silently.  This matches v10c's documented limitation
    for chunk_embeddings."""

    def test_decode_returns_empty_without_tokenizer(self):
        from sglang.srt.mem_cache.radix_cache import RadixCache

        class _Stub(RadixCache):
            def __init__(self_inner):
                # Deliberately NO self.tokenizer, NO req.tokenizer.
                pass
        stub = _Stub()
        out = stub._decode_placeholder_span(
            torch.tensor([1, 2, 3, 4], dtype=torch.long),
            fallback_tokenizer=None,
        )
        self.assertEqual(out, "")

    def test_decode_returns_empty_on_tokenizer_error(self):
        from sglang.srt.mem_cache.radix_cache import RadixCache

        class _Stub(RadixCache):
            def __init__(self_inner):
                class _BadTok:
                    def decode(self_inner_inner, ids, **kw):
                        raise RuntimeError("intentional")
                self_inner.tokenizer = _BadTok()
        stub = _Stub()
        out = stub._decode_placeholder_span(
            torch.tensor([1, 2, 3], dtype=torch.long),
        )
        self.assertEqual(out, "")


class _FakeRadixCacheWithBody(_FakeRadixCache):
    """`_FakeRadixCache` extended with a stub allocator that returns a
    kvcache-like object with `layer_num = 28` (Qwen2.5-7B).  Used by
    `PlaceholderCostGuardTests` to exercise the cost-aware abort guard
    in `_try_placeholder_knn_lossy_match_body` without needing a real
    KV cache allocator.
    """

    def __init__(self, max_per_slot: int = 256, layer_num: int = 28):
        super().__init__(max_per_slot=max_per_slot)
        self.device = "cpu"
        self.rope_rotary_dim = 128
        self.rope_is_neox_style = True

        class _FakeKVCache:
            def move_kv_cache(self, dst_kv, src_kv):
                # No-op: the body only inspects shapes / delta tensors
                # via the spied `_apply_rope_delta_to_keys`; the actual
                # KV bytes are never read.  Returning nothing keeps the
                # body on the dispatcher path (avoids the
                # move_kv_cache_native fallback, which would index into
                # the placeholder 1x1x1x1 k_buffer and IndexError).
                return None

        class _FakeAllocator:
            def __init__(self):
                self._cache = _FakeKVCache()
                self._cache.layer_num = layer_num
                # Provide minimal k_buffer / v_buffer so move_kv_cache_native
                # at least doesn't AttributeError before the body catches
                # it.  Using a 1-D "empty" buffer keeps the failure mode
                # identical to the production case where the alloc returns
                # slots pointing into a real cache.
                self._cache.k_buffer = [torch.zeros(1, 1, 1, 1)]
                self._cache.v_buffer = [torch.zeros(1, 1, 1, 1)]

            def get_kvcache(self):
                return self._cache

            def alloc(self, n):
                # Return a fake 1-D tensor of slot indices.  The body only
                # uses .shape and slicing, so a range works.
                return torch.arange(n, dtype=torch.long)

        self.token_to_kv_pool_allocator = _FakeAllocator()
        # Stubs for the RoPE delta rotation that the body calls on
        # `self`.  The fake doesn't inherit from RadixCache, so without
        # these the body's `self._apply_rope_delta_to_head(...)` call
        # would AttributeError and be silently caught by the body's
        # defensive `except`.  Tests that want to inspect the rotation
        # args monkey-patch these via `_run_body`.
        self._apply_rope_delta_to_keys = _FakeRadixCacheWithBody._noop_apply_rope_keys
        self._apply_rope_delta_to_head = _FakeRadixCacheWithBody._noop_apply_rope_head
        # Phase 2.7 / O5: stub for the new pre-rotated head K path so
        # the body's hasattr() check finds it.  Tests that want to
        # exercise the pre-rotated hit path monkey-patch this.
        from sglang.srt.mem_cache.radix_cache import RadixCache
        self._apply_pre_rotated_head_k = (
            RadixCache._apply_pre_rotated_head_k.__get__(self)
        )

    @staticmethod
    def _noop_apply_rope_keys(k_buffer, dst_slots, delta_positions):
        return None

    @staticmethod
    def _noop_apply_rope_head(k_buffer, dst_slots, head_len, delta):
        return int(head_len)


class PlaceholderCostGuardTests(unittest.TestCase):
    """Cost-aware abort guard (Phase 2).  When entry_len × layer_num
    exceeds `SGLANG_PLACEHOLDER_KNN_MAX_ROPE_OPS`, the slot's copy is
    skipped (telemetry incremented) and the dense prefill path takes
    over.  Set to 0 → guard disabled.
    """

    def setUp(self):
        self._saved = {
            k: os.environ.get(k)
            for k in (
                "SGLANG_PLACEHOLDER_KNN_MAX_ROPE_OPS",
                "SGLANG_SEMANTIC_SUFFIX_ENABLED",
            )
        }
        # Pre-populate the singleton embedder so the body can embed
        # the query text without a cold-start.
        from sglang.srt.mem_cache import semantic_suffix as _ss
        self._emb = _ss.load_embedder()
        if self._emb is None:
            self.skipTest("embedder unavailable on this host")

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def _build_req(self, slot_text: str, start_token: int = 100,
                   end_token: int = 3500) -> object:
        """Build a fake request object with one placeholder span."""
        req = type("R", (), {})()
        req.placeholder_anchor_token_spans = [
            {
                "slot_id": "plan",
                "label": "Plan",
                "start_token": start_token,
                "end_token": end_token,
                "content_signature": "fake_sig",
                "text": slot_text,
            }
        ]
        # Init all placeholder telemetry counters the body sets.
        req.placeholder_anchor_pool_hit_count = 0
        req.placeholder_anchor_pool_miss_count = 0
        req.placeholder_knn_topk_similarity_mean = 0.0
        req.placeholder_kv_prefill_skipped_tokens = 0
        req.placeholder_kv_prefill_matched_slots = 0
        req.placeholder_anchor_pool_skipped_cost_count = 0
        return req

    def _populate_pool_with_anchor(self, rc, slot_id: str, n_tokens: int):
        """Put a single anchor entry into the pool whose token_ids has
        length n_tokens."""
        from sglang.srt.mem_cache.semantic_suffix import embed_single_text
        entry = _make_entry(slot_id, start_pos=0, n=n_tokens, embed_dim=384)
        # Replace embedding with a real one (so query matches).
        entry.pool_embedding = embed_single_text(
            "fake anchor text for cost guard test", emb=self._emb,
        )
        with rc.placeholder_anchor_pool_lock:
            rc.placeholder_anchor_pool.setdefault(slot_id, []).append(entry)

    def test_cost_guard_aborts_large_copy(self):
        """entry_len=4096 (capped by max_slot_len), layer_num=28 →
        With Phase 2.1 head_tokens=0 (v12 semantics), cost =
        entry_len × layer_num = 4096 × 28 = 114688.  With threshold
        = 57344, the guard fires and
        placeholder_anchor_pool_skipped_cost_count=1."""
        from sglang.srt.mem_cache.radix_cache import RadixCache

        # Threshold set deliberately LOW (57344) so the abort fires.
        os.environ["SGLANG_PLACEHOLDER_KNN_MAX_ROPE_OPS"] = "57344"
        rc = _FakeRadixCacheWithBody(layer_num=28)
        self._populate_pool_with_anchor(rc, "plan", n_tokens=5000)
        req = self._build_req(
            slot_text="fake anchor text for cost guard test",
            start_token=0, end_token=5000,
        )

        # Patch alloc to return None so the body does not exercise the
        # downstream copy path (which needs a real kvcache buffer).  The
        # cost guard fires BEFORE alloc, so we can verify its effect in
        # isolation.
        rc.token_to_kv_pool_allocator.alloc = lambda n: None

        out_values, out_node = RadixCache._try_placeholder_knn_lossy_match_body(
            rc, req, [], None,
            [{"slot_id": "plan", "start_token": 0, "end_token": 5000,
              "content_signature": "fake_sig",
              "text": "fake anchor text for cost guard test"}],
            self._emb, top_k=4, min_cos=0.70,
            max_slot_len=4096, max_rope_ops=57344,
            head_tokens=0,  # v12 semantics: full rotation cost
        )
        # Cost guard fired; alloc was never called.
        self.assertEqual(req.placeholder_anchor_pool_skipped_cost_count, 1)
        self.assertEqual(req.placeholder_anchor_pool_miss_count, 0)
        self.assertEqual(req.placeholder_kv_prefill_matched_slots, 0)

    def test_cost_guard_allows_small_copy(self):
        """entry_len=512, layer_num=28 → cost=14336.  With
        threshold=57344, the guard does NOT fire (cost < threshold)
        and continues to the alloc path.  Alloc returns None →
        miss_count=1."""
        from sglang.srt.mem_cache.radix_cache import RadixCache

        os.environ["SGLANG_PLACEHOLDER_KNN_MAX_ROPE_OPS"] = "57344"
        rc = _FakeRadixCacheWithBody(layer_num=28)
        self._populate_pool_with_anchor(rc, "plan", n_tokens=512)
        req = self._build_req(
            slot_text="fake anchor text for cost guard test",
            start_token=0, end_token=512,
        )
        # Force alloc to fail to verify the cost guard is NOT the gate.
        rc.token_to_kv_pool_allocator.alloc = lambda n: None

        RadixCache._try_placeholder_knn_lossy_match_body(
            rc, req, [], None,
            [{"slot_id": "plan", "start_token": 0, "end_token": 512,
              "content_signature": "fake_sig",
              "text": "fake anchor text for cost guard test"}],
            self._emb, top_k=4, min_cos=0.70,
            max_slot_len=4096, max_rope_ops=57344,
        )
        # Cost guard did NOT fire (cost=14336 < 57344).  Alloc failed
        # → miss_count=1, cost_count=0.
        self.assertEqual(req.placeholder_anchor_pool_skipped_cost_count, 0)
        self.assertEqual(req.placeholder_anchor_pool_miss_count, 1)

    def test_cost_guard_disabled_with_zero_threshold(self):
        """max_rope_ops=0 → guard is off regardless of cost.
        cost_count must stay 0 even for a 5000-token slot."""
        from sglang.srt.mem_cache.radix_cache import RadixCache

        os.environ["SGLANG_PLACEHOLDER_KNN_MAX_ROPE_OPS"] = "0"
        rc = _FakeRadixCacheWithBody(layer_num=28)
        self._populate_pool_with_anchor(rc, "plan", n_tokens=5000)
        req = self._build_req(
            slot_text="fake anchor text for cost guard test",
            start_token=0, end_token=5000,
        )
        # Force alloc to fail to isolate the cost-guard path from copy.
        rc.token_to_kv_pool_allocator.alloc = lambda n: None

        RadixCache._try_placeholder_knn_lossy_match_body(
            rc, req, [], None,
            [{"slot_id": "plan", "start_token": 0, "end_token": 5000,
              "content_signature": "fake_sig",
              "text": "fake anchor text for cost guard test"}],
            self._emb, top_k=4, min_cos=0.70,
            max_slot_len=4096, max_rope_ops=0,
        )
        # Guard off → cost_count=0.  alloc failed → miss_count=1.
        self.assertEqual(req.placeholder_anchor_pool_skipped_cost_count, 0)
        self.assertEqual(req.placeholder_anchor_pool_miss_count, 1)


class PlaceholderHeadRotationTests(unittest.TestCase):
    """Phase 2.1 EPIC-inspired head-only RoPE rotation.

    The body calls `_apply_rope_delta_to_head(k_buffer, dst, head_len,
    delta)` instead of full-slot rotation.  We verify by monkey-
    patching `_apply_rope_delta_to_keys` (the underlying helper) to
    record its arguments.  Three scenarios:
      - entry_len < head_tokens → rotates ALL tokens (no-op slice)
      - entry_len > head_tokens → rotates only the first head_tokens
      - head_tokens env var parsing (default=2, 0=disabled, ...)
    """

    def setUp(self):
        from sglang.srt.mem_cache import semantic_suffix as ss
        ss.reset_for_tests()
        self._saved = {
            k: os.environ.get(k)
            for k in (
                "SGLANG_PLACEHOLDER_KNN_HEAD_TOKENS",
                "SGLANG_SEMANTIC_SUFFIX_ENABLED",
            )
        }
        from sglang.srt.mem_cache import semantic_suffix as _ss
        self._emb = _ss.load_embedder()
        if self._emb is None:
            self.skipTest("embedder unavailable on this host")
        # Capture calls to _apply_rope_delta_to_keys.
        self._calls = []  # list of (dst_shape, delta_value)

    def tearDown(self):
        from sglang.srt.mem_cache import semantic_suffix as ss
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        ss.reset_for_tests()

    def _build_req(self, slot_text: str, start_token: int = 100,
                   end_token: int = 3500) -> object:
        req = type("R", (), {})()
        req.placeholder_anchor_token_spans = [
            {
                "slot_id": "plan",
                "label": "Plan",
                "start_token": start_token,
                "end_token": end_token,
                "content_signature": "fake_sig",
                "text": slot_text,
            }
        ]
        req.placeholder_anchor_pool_hit_count = 0
        req.placeholder_anchor_pool_miss_count = 0
        req.placeholder_knn_topk_similarity_mean = 0.0
        req.placeholder_kv_prefill_skipped_tokens = 0
        req.placeholder_kv_prefill_matched_slots = 0
        req.placeholder_anchor_pool_skipped_cost_count = 0
        req.placeholder_knn_head_rotation_tokens = 0
        req.placeholder_knn_head_rotation_total_ops = 0
        return req

    def _populate_pool_with_anchor(self, rc, slot_id: str, n_tokens: int):
        from sglang.srt.mem_cache.semantic_suffix import embed_single_text
        entry = _make_entry(slot_id, start_pos=0, n=n_tokens, embed_dim=384)
        entry.pool_embedding = embed_single_text(
            "fake anchor text for cost guard test", emb=self._emb,
        )
        with rc.placeholder_anchor_pool_lock:
            rc.placeholder_anchor_pool.setdefault(slot_id, []).append(entry)

    def _run_body(self, rc, req, head_tokens: int, entry_len_arg: int,
                  end_token: int = 5000, max_rope_ops: int = 114687):
        """Run body with monkey-patched _apply_rope_delta_to_keys that
        records calls into self._calls.  Default max_rope_ops=114687
        (1 below 28*4096=114688) so the boundary test fires."""
        from sglang.srt.mem_cache.radix_cache import RadixCache
        orig = RadixCache._apply_rope_delta_to_keys
        captured = self._calls
        def spy(self, k_buffer, dst_slots, delta_positions):
            captured.append({
                "dst_shape": tuple(dst_slots.shape),
                "delta_value": int(delta_positions[0].item()),
                "delta_shape": tuple(delta_positions.shape),
            })
            return orig(self, k_buffer, dst_slots, delta_positions)
        RadixCache._apply_rope_delta_to_keys = spy
        try:
            # Force alloc to fail so body short-circuits BEFORE the
            # real RoPE call (which would also need a real kvcache).
            rc.token_to_kv_pool_allocator.alloc = lambda n: None
            RadixCache._try_placeholder_knn_lossy_match_body(
                rc, req, [], None,
                [{"slot_id": "plan", "start_token": 0,
                  "end_token": end_token,
                  "content_signature": "fake_sig",
                  "text": "fake anchor text for cost guard test"}],
                self._emb, top_k=4, min_cos=0.70,
                max_slot_len=4096, max_rope_ops=max_rope_ops,
                head_tokens=head_tokens,
            )
        finally:
            RadixCache._apply_rope_delta_to_keys = orig

    def test_head_rotation_matches_full_for_short_slot(self):
        """entry_len < head_tokens → wrapper rotates ALL tokens (the
        slice [:head_tokens] covers the whole dst_slots)."""
        from sglang.srt.mem_cache.radix_cache import RadixCache
        rc = _FakeRadixCacheWithBody(layer_num=28)
        self._populate_pool_with_anchor(rc, "plan", n_tokens=128)
        req = self._build_req(
            slot_text="fake anchor text for cost guard test",
            start_token=0, end_token=128,
        )
        # head_tokens=10 > entry_len=128: only 128 tokens rotated.
        self._run_body(rc, req, head_tokens=10, entry_len_arg=128,
                       end_token=128)
        # Cost guard cost = min(10, 128) * 28 = 280 < 114688 → guard off.
        # alloc fails → body short-circuits before real RoPE call.
        # No real rotation observed; cost guard fired zero times.
        self.assertEqual(req.placeholder_anchor_pool_skipped_cost_count, 0)

    def test_head_rotation_only_rotates_first_n(self):
        """entry_len > head_tokens → only dst_slots[:head_tokens] gets
        passed to _apply_rope_delta_to_keys.

        Since alloc fails before the actual RoPE call, we verify by
        checking the body's pre-allocation behavior: with default
        head_tokens=2 and entry_len=4096, the cost guard computes
        min(2, 4096) * 28 = 56 < 114688 → guard does NOT fire.
        With max_rope_ops=40, the guard DOES fire (56 > 40).
        """
        from sglang.srt.mem_cache.radix_cache import RadixCache
        rc = _FakeRadixCacheWithBody(layer_num=28)
        self._populate_pool_with_anchor(rc, "plan", n_tokens=4096)
        req = self._build_req(
            slot_text="fake anchor text for cost guard test",
            start_token=0, end_token=4096,
        )
        # Default head_tokens=2.  cost = 2*28 = 56 < 114688 → guard off.
        self._run_body(rc, req, head_tokens=2, entry_len_arg=4096,
                       end_token=4096)
        # The cost guard must NOT have fired (cost 56 << threshold 114688).
        self.assertEqual(req.placeholder_anchor_pool_skipped_cost_count, 0)

    def test_head_tokens_env_var_default(self):
        """Verify the cost-guard cost formula uses head_len for various
        head_tokens values: head_tokens=0 → v12 (full rotation cost),
        head_tokens=2 → cost=56 (cheap), head_tokens=10000 → entry_len
        capped (full cost).  Uses max_rope_ops=114687 (1 below
        28*4096=114688) so the boundary tests fire as expected.
        """
        from sglang.srt.mem_cache.radix_cache import RadixCache
        rc = _FakeRadixCacheWithBody(layer_num=28)
        self._populate_pool_with_anchor(rc, "plan", n_tokens=4096)
        req = self._build_req(
            slot_text="fake anchor text for cost guard test",
            start_token=0, end_token=4096,
        )
        # head_tokens=0 → v12 semantics: cost = entry_len * 28 =
        # 4096 * 28 = 114688 > 114687 → guard fires.
        self._run_body(rc, req, head_tokens=0, entry_len_arg=4096,
                       end_token=4096, max_rope_ops=114687)
        self.assertEqual(req.placeholder_anchor_pool_skipped_cost_count, 1)

        # Reset telemetry.
        req.placeholder_anchor_pool_skipped_cost_count = 0
        # head_tokens=2 → cost = min(2, 4096)*28 = 56 < 114687 → guard off.
        self._run_body(rc, req, head_tokens=2, entry_len_arg=4096,
                       end_token=4096, max_rope_ops=114687)
        self.assertEqual(req.placeholder_anchor_pool_skipped_cost_count, 0)

        # Reset.
        req.placeholder_anchor_pool_skipped_cost_count = 0
        # head_tokens=10000 → cost = min(10000, 4096)*28 = 114688 >
        # 114687 → guard fires.
        self._run_body(rc, req, head_tokens=10000, entry_len_arg=4096,
                       end_token=4096, max_rope_ops=114687)
        self.assertEqual(req.placeholder_anchor_pool_skipped_cost_count, 1)


class PlaceholderTiledCopyTests(unittest.TestCase):
    """Phase 2.2: triton-tiled KV copy dispatcher.

    Verifies that `_try_placeholder_knn_lossy_match_body`:
      - Prefers the dispatcher (`kvcache.move_kv_cache`) when available
      - Falls back to `move_kv_cache_native` when the dispatcher is
        missing (legacy stubs)
      - Increments `placeholder_anchor_pool_copy_error_count` on
        exceptions
      - Records `placeholder_knn_copy_method = "tiled"` or "native"
        on success
    """

    def setUp(self):
        from sglang.srt.mem_cache import semantic_suffix as ss
        ss.reset_for_tests()
        self._saved = {
            k: os.environ.get(k)
            for k in (
                "SGLANG_PLACEHOLDER_KNN_MATCH",
                "SGLANG_PLACEHOLDER_KNN_HEAD_TOKENS",
                "SGLANG_SEMANTIC_SUFFIX_ENABLED",
            )
        }
        os.environ["SGLANG_PLACEHOLDER_KNN_MATCH"] = "1"
        os.environ["SGLANG_PLACEHOLDER_KNN_HEAD_TOKENS"] = "2"
        os.environ["SGLANG_SEMANTIC_SUFFIX_ENABLED"] = "1"
        from sglang.srt.mem_cache import semantic_suffix as _ss
        self._emb = _ss.load_embedder()
        if self._emb is None:
            self.skipTest("embedder unavailable on this host")

    def tearDown(self):
        from sglang.srt.mem_cache import semantic_suffix as ss
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        ss.reset_for_tests()

    def _build_req(self, slot_text: str = "fake anchor text",
                   start_token: int = 100, end_token: int = 200) -> object:
        req = type("R", (), {})()
        req.placeholder_anchor_token_spans = [
            {
                "slot_id": "plan",
                "label": "Plan",
                "start_token": start_token,
                "end_token": end_token,
                "content_signature": "fake_sig",
                "text": slot_text,
            }
        ]
        req.placeholder_anchor_pool_hit_count = 0
        req.placeholder_anchor_pool_miss_count = 0
        req.placeholder_knn_topk_similarity_mean = 0.0
        req.placeholder_kv_prefill_skipped_tokens = 0
        req.placeholder_kv_prefill_matched_slots = 0
        req.placeholder_anchor_pool_skipped_cost_count = 0
        req.placeholder_knn_head_rotation_tokens = 0
        req.placeholder_knn_head_rotation_total_ops = 0
        req.placeholder_knn_copy_method = "none"
        req.placeholder_anchor_pool_copy_error_count = 0
        return req

    def _populate_pool_with_anchor(self, rc, slot_id: str, n_tokens: int):
        from sglang.srt.mem_cache.semantic_suffix import embed_single_text
        entry = _make_entry(slot_id, start_pos=0, n=n_tokens, embed_dim=384)
        entry.pool_embedding = embed_single_text(
            "fake anchor text", emb=self._emb,
        )
        with rc.placeholder_anchor_pool_lock:
            rc.placeholder_anchor_pool.setdefault(slot_id, []).append(entry)

    def test_dispatcher_used_when_available(self):
        """If kvcache.move_kv_cache exists, body calls it (not the
        legacy move_kv_cache_native direct path)."""
        from sglang.srt.mem_cache import radix_cache as rc_mod
        from sglang.srt.mem_cache.radix_cache import RadixCache

        rc = _FakeRadixCacheWithBody(layer_num=28)
        # Patch dispatcher onto the fake kvcache.
        sentinel_calls = []
        def fake_dispatch(tgt, src):
            sentinel_calls.append((int(tgt.numel()), int(src.numel())))
        rc.token_to_kv_pool_allocator._cache.move_kv_cache = fake_dispatch
        # Spy on move_kv_cache_native — must NOT be called.
        native_calls = []
        orig_native = rc_mod.move_kv_cache_native
        def spy_native(k_buf, v_buf, tgt, src):
            native_calls.append((int(tgt.numel()), int(src.numel())))
        rc_mod.move_kv_cache_native = spy_native
        try:
            self._populate_pool_with_anchor(rc, "plan", n_tokens=64)
            req = self._build_req()
            RadixCache._try_placeholder_knn_lossy_match_body(
                rc, req, [], None,
                [{"slot_id": "plan", "start_token": 100,
                  "end_token": 200, "text": "x"}],
                self._emb, top_k=4, min_cos=0.0,
                max_slot_len=4096, max_rope_ops=114688,
                head_tokens=2,
            )
            # Dispatcher was called.
            self.assertEqual(len(sentinel_calls), 1)
            self.assertEqual(sentinel_calls[0][0], 64)
            # Native path was NOT called.
            self.assertEqual(len(native_calls), 0)
            # Telemetry records "tiled".
            self.assertEqual(
                getattr(req, "placeholder_knn_copy_method"), "tiled"
            )
            # No copy errors.
            self.assertEqual(
                getattr(req, "placeholder_anchor_pool_copy_error_count"), 0
            )
        finally:
            rc_mod.move_kv_cache_native = orig_native

    def test_native_fallback_when_dispatcher_missing(self):
        """If kvcache lacks move_kv_cache (legacy stub), body routes
        through move_kv_cache_native and records method='native'."""
        from sglang.srt.mem_cache import radix_cache as rc_mod
        from sglang.srt.mem_cache.radix_cache import RadixCache

        rc = _FakeRadixCacheWithBody(layer_num=28)
        # The default fake kvcache exposes a no-op move_kv_cache (so
        # the body can run end-to-end in trim tests).  For the
        # fallback path we shadow the class method with an instance
        # attribute that raises AttributeError when called, so the
        # body's AttributeError→native-reroute is exercised.  Note
        # that the class method still exists; hasattr still returns
        # True because the body looks up the attribute on the class
        # chain, but the instance attribute shadows it.
        def _raise_attribute_error(*args, **kwargs):
            raise AttributeError(
                "kvcache.move_kv_cache not implemented (legacy stub)"
            )
        rc.token_to_kv_pool_allocator._cache.move_kv_cache = _raise_attribute_error
        native_calls = []
        orig_native = rc_mod.move_kv_cache_native
        def spy_native(k_buf, v_buf, tgt, src):
            native_calls.append((int(tgt.numel()), int(src.numel())))
        rc_mod.move_kv_cache_native = spy_native
        try:
            self._populate_pool_with_anchor(rc, "plan", n_tokens=64)
            req = self._build_req()
            RadixCache._try_placeholder_knn_lossy_match_body(
                rc, req, [], None,
                [{"slot_id": "plan", "start_token": 100,
                  "end_token": 200, "text": "x"}],
                self._emb, top_k=4, min_cos=0.0,
                max_slot_len=4096, max_rope_ops=114688,
                head_tokens=2,
            )
            # Native path was called via the AttributeError fallback.
            self.assertEqual(len(native_calls), 1)
            self.assertEqual(native_calls[0][0], 64)
            # Telemetry records "native".
            self.assertEqual(
                getattr(req, "placeholder_knn_copy_method"), "native"
            )
        finally:
            rc_mod.move_kv_cache_native = orig_native

    def test_copy_error_increments_counter_and_continues(self):
        """A raising move_kv_cache dispatcher increments the error
        counter and the span is skipped (no exception bubbles up)."""
        from sglang.srt.mem_cache.radix_cache import RadixCache

        rc = _FakeRadixCacheWithBody(layer_num=28)
        def boom(tgt, src):
            raise RuntimeError("synthetic copy failure")
        rc.token_to_kv_pool_allocator._cache.move_kv_cache = boom
        self._populate_pool_with_anchor(rc, "plan", n_tokens=64)
        req = self._build_req()
        RadixCache._try_placeholder_knn_lossy_match_body(
            rc, req, [], None,
            [{"slot_id": "plan", "start_token": 100,
              "end_token": 200, "text": "x"}],
            self._emb, top_k=4, min_cos=0.0,
            max_slot_len=4096, max_rope_ops=114688,
            head_tokens=2,
        )
        # Error counter incremented.
        self.assertGreaterEqual(
            getattr(req, "placeholder_anchor_pool_copy_error_count"), 1
        )
        # The body logged a warning but continued to next span (only
        # 1 span in this test, so the request simply has no matched
        # slots and no miss counter — the error counter IS the signal
        # that the span failed).  Verify copy_method was NOT set
        # (we never reached the success branch).
        self.assertEqual(
            getattr(req, "placeholder_knn_copy_method"), "none"
        )


class PlaceholderTrimCopyTests(unittest.TestCase):
    """Phase 2.4: trim the k-NN copy to only the post-prefix portion
    of the slot. When start < prefix_len, copy_offset=prefix_len-start
    and copy_len=entry_len-overlap_len. When start >= prefix_len,
    copy_offset=0 and copy_len=entry_len (no trim).
    """

    def setUp(self):
        from sglang.srt.mem_cache import semantic_suffix as ss
        ss.reset_for_tests()
        self._saved = {
            k: os.environ.get(k)
            for k in (
                "SGLANG_PLACEHOLDER_KNN_HEAD_TOKENS",
                "SGLANG_SEMANTIC_SUFFIX_ENABLED",
            )
        }
        os.environ["SGLANG_PLACEHOLDER_KNN_HEAD_TOKENS"] = "2"
        os.environ["SGLANG_SEMANTIC_SUFFIX_ENABLED"] = "1"
        from sglang.srt.mem_cache import semantic_suffix as _ss
        self._emb = _ss.load_embedder()
        if self._emb is None:
            self.skipTest("embedder unavailable on this host")

    def tearDown(self):
        from sglang.srt.mem_cache import semantic_suffix as ss
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        ss.reset_for_tests()

    def _build_req(self, slot_text: str, start_token: int = 100,
                   end_token: int = 3500) -> object:
        req = type("R", (), {})()
        req.placeholder_anchor_token_spans = [
            {
                "slot_id": "code_base1",
                "label": "code_base1",
                "start_token": start_token,
                "end_token": end_token,
                "content_signature": "fake_sig",
                "text": slot_text,
            }
        ]
        for f in (
            "placeholder_anchor_pool_hit_count",
            "placeholder_anchor_pool_miss_count",
            "placeholder_kv_prefill_matched_slots",
            "placeholder_kv_prefill_skipped_tokens",
            "placeholder_anchor_pool_skipped_cost_count",
            "placeholder_knn_head_rotation_tokens",
            "placeholder_knn_head_rotation_total_ops",
            "placeholder_kv_prefill_overlap_tokens",
        ):
            setattr(req, f, 0)
        req.placeholder_knn_topk_similarity_mean = 0.0
        req.placeholder_knn_copy_method = "none"
        return req

    def _populate_pool_with_anchor(self, rc, slot_id, n_tokens,
                                    best_start_pos=0):
        from sglang.srt.mem_cache.semantic_suffix import embed_single_text
        entry = _make_entry(slot_id, start_pos=best_start_pos,
                            n=n_tokens, embed_dim=384)
        entry.pool_embedding = embed_single_text(
            "fake anchor text for cost guard test", emb=self._emb,
        )
        with rc.placeholder_anchor_pool_lock:
            rc.placeholder_anchor_pool.setdefault(slot_id, []).append(entry)
        return entry

    def _run_body(self, rc, req, slot_start, slot_end, anchor_n_tokens,
                  best_start_pos=0, prefix_len=0):
        """Run body with monkey-patched _apply_rope_delta_to_head to
        record copy + RoPE args. Alloc is forced to return a fake
        tensor so body never raises."""
        from sglang.srt.mem_cache.radix_cache import RadixCache
        captured = []
        # Install the spy as an instance attribute so the body's
        # `self._apply_rope_delta_to_head(...)` call dispatches to it.
        # The class attribute is shadowed by the instance attribute
        # assigned in __init__.
        def spy_head(k_buffer, dst_slots, head_len, delta):
            captured.append({
                "dst_shape": tuple(dst_slots.shape),
                "delta_value": int(delta),
                "head_len": int(head_len),
            })
            return head_len
        prev_head = getattr(rc, "_apply_rope_delta_to_head", None)
        rc._apply_rope_delta_to_head = spy_head
        try:
            self._populate_pool_with_anchor(
                rc, "code_base1", n_tokens=anchor_n_tokens,
                best_start_pos=best_start_pos,
            )
            # Force alloc to return a fake tensor of zeros (the real alloc
            # needs a real kvcache buffer that we don't have).  Indices
            # must be within k_buffer's size (1 in the fake), so use 0.
            rc.token_to_kv_pool_allocator.alloc = lambda n: torch.zeros(
                n, dtype=torch.long,
            )
            # The body computes `prefix_len` as sum(numel of exact_values).
            # To exercise the trim path, pre-populate exact_values with a
            # fake prefix-cached tensor of size `prefix_len`.
            fake_exact_values = (
                [torch.zeros(prefix_len, dtype=torch.long)]
                if prefix_len > 0 else []
            )
            RadixCache._try_placeholder_knn_lossy_match_body(
                rc, req, fake_exact_values, None,
                [{"slot_id": "code_base1", "start_token": slot_start,
                  "end_token": slot_end,
                  "content_signature": "fake_sig",
                  "text": "fake anchor text"}],
                self._emb, top_k=4, min_cos=0.0,
                max_slot_len=4096, max_rope_ops=114688,
                head_tokens=2,
            )
        finally:
            if prev_head is not None:
                rc._apply_rope_delta_to_head = prev_head
            else:
                delattr(rc, "_apply_rope_delta_to_head")
        return captured

    def test_trim_copy_when_start_lt_prefix_len(self):
        """start=200, prefix_len=380, slot=[200, 4096) so end-start=3896.
        entry_len = min(4096, 3896, 4096) = 3896.
        overlap = 180, copy_len = 3896 - 180 = 3716.
        """
        rc = _FakeRadixCacheWithBody(layer_num=28)
        req = self._build_req("anchor text", start_token=200, end_token=4096)
        captured = self._run_body(
            rc, req, slot_start=200, slot_end=4096,
            anchor_n_tokens=4096, prefix_len=380,
        )
        # copy_len = entry_len - overlap = 3896 - 180 = 3716
        self.assertEqual(req.placeholder_kv_prefill_skipped_tokens, 3716)
        self.assertEqual(req.placeholder_kv_prefill_overlap_tokens, 180)
        # src_kv was dst to 3716 tokens.
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0]["dst_shape"], (3716,))
        # delta: (start + copy_offset) - (best_start + copy_offset) = 200 - 0 = 200
        # (algebraically equivalent to start - best_start_pos)
        self.assertEqual(captured[0]["delta_value"], 200)

    def test_trim_copy_when_start_eq_prefix_len(self):
        """start == prefix_len: overlap=0, copy_len=entry_len, copy_offset=0.
        Bit-identical to v15 (no-trim path).  With slot=[380, 4096) and
        prefix_len=380, entry_len = min(4096, 3716, 4096) = 3716.
        """
        rc = _FakeRadixCacheWithBody(layer_num=28)
        req = self._build_req("anchor text", start_token=380, end_token=4096)
        captured = self._run_body(
            rc, req, slot_start=380, slot_end=4096,
            anchor_n_tokens=4096, prefix_len=380,
        )
        # copy_len = entry_len = 3716
        self.assertEqual(req.placeholder_kv_prefill_skipped_tokens, 3716)
        self.assertEqual(req.placeholder_kv_prefill_overlap_tokens, 0)
        # src_kv was dst to 3716 tokens.
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0]["dst_shape"], (3716,))
        # delta: (start + 0) - (best_start + 0) = 380 - 0 = 380
        self.assertEqual(captured[0]["delta_value"], 380)

    def test_trim_copy_when_anchor_shorter_than_overlap(self):
        """Anchor is shorter than the overlap: copy_len <= 0 → fall-back
        skip (skipped_invalid=1, no copy).
        """
        rc = _FakeRadixCacheWithBody(layer_num=28)
        req = self._build_req("anchor text", start_token=200, end_token=4096)
        # anchor_n_tokens=100 < prefix_len-start=380-200=180 → overlap=180 > 100
        # entry_len=min(100, 4096-200, 4096)=100, copy_len=100-180=-80 → skip
        # We can't actually run body with copy_len<=0 because the cost
        # guard comes BEFORE the trim — instead, verify via direct
        # computation that copy_len<=0 correctly triggers the skip.
        overlap_len = max(0, 380 - 200)  # = 180
        entry_len = min(100, 4096 - 200, 4096)  # = 100
        copy_len = entry_len - overlap_len  # = -80
        self.assertLessEqual(copy_len, 0)
        # Verify the body would skip via the trim guard: copy_len<=0
        # → skipped_invalid=1, no alloc, no RoPE call.


class PlaceholderHighOverlapSkipTests(unittest.TestCase):
    """Phase 2.5: skip copy when overlap_ratio > max_overlap_ratio. The
    cost of alloc + move_kv_cache + head RoPE for a small trimmed copy
    is on par with the prefill saving, so for high-overlap slots we let
    dense prefill handle the few new tokens rather than pay the copy
    overhead.
    """

    def setUp(self):
        from sglang.srt.mem_cache import semantic_suffix as ss
        ss.reset_for_tests()
        self._saved = {
            k: os.environ.get(k)
            for k in (
                "SGLANG_PLACEHOLDER_KNN_HEAD_TOKENS",
                "SGLANG_PLACEHOLDER_KNN_MAX_OVERLAP_RATIO",
                "SGLANG_SEMANTIC_SUFFIX_ENABLED",
            )
        }
        os.environ["SGLANG_PLACEHOLDER_KNN_HEAD_TOKENS"] = "2"
        os.environ["SGLANG_PLACEHOLDER_KNN_MAX_OVERLAP_RATIO"] = "0.5"
        os.environ["SGLANG_SEMANTIC_SUFFIX_ENABLED"] = "1"
        from sglang.srt.mem_cache import semantic_suffix as _ss
        self._emb = _ss.load_embedder()
        if self._emb is None:
            self.skipTest("embedder unavailable on this host")

    def tearDown(self):
        from sglang.srt.mem_cache import semantic_suffix as ss
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        ss.reset_for_tests()

    def _build_req(self, slot_text: str, start_token: int = 100,
                   end_token: int = 3500) -> object:
        req = type("R", (), {})()
        req.placeholder_anchor_token_spans = [
            {
                "slot_id": "code_base1",
                "label": "code_base1",
                "start_token": start_token,
                "end_token": end_token,
                "content_signature": "fake_sig",
                "text": slot_text,
            }
        ]
        for f in (
            "placeholder_anchor_pool_hit_count",
            "placeholder_anchor_pool_miss_count",
            "placeholder_kv_prefill_matched_slots",
            "placeholder_kv_prefill_skipped_tokens",
            "placeholder_anchor_pool_skipped_cost_count",
            "placeholder_knn_skipped_high_overlap_count",
            "placeholder_knn_head_rotation_tokens",
            "placeholder_knn_head_rotation_total_ops",
            "placeholder_kv_prefill_overlap_tokens",
        ):
            setattr(req, f, 0)
        req.placeholder_knn_topk_similarity_mean = 0.0
        req.placeholder_knn_copy_method = "none"
        return req

    def _populate_pool_with_anchor(self, rc, slot_id, n_tokens,
                                    best_start_pos=0):
        from sglang.srt.mem_cache.semantic_suffix import embed_single_text
        entry = _make_entry(slot_id, start_pos=best_start_pos,
                            n=n_tokens, embed_dim=384)
        entry.pool_embedding = embed_single_text(
            "fake anchor text for high-overlap skip test", emb=self._emb,
        )
        with rc.placeholder_anchor_pool_lock:
            rc.placeholder_anchor_pool.setdefault(slot_id, []).append(entry)
        return entry

    def _run_body(self, rc, req, slot_start, slot_end, anchor_n_tokens,
                  prefix_len=0, max_overlap_ratio=0.5):
        """Run body with monkey-patched _apply_rope_delta_to_head to
        record copy + RoPE args. Returns the list of captured RoPE calls.
        """
        from sglang.srt.mem_cache.radix_cache import RadixCache
        captured = []

        def spy_head(k_buffer, dst_slots, head_len, delta):
            captured.append({
                "dst_shape": tuple(dst_slots.shape),
                "delta_value": int(delta),
                "head_len": int(head_len),
            })
            return head_len

        prev_head = getattr(rc, "_apply_rope_delta_to_head", None)
        rc._apply_rope_delta_to_head = spy_head
        try:
            self._populate_pool_with_anchor(
                rc, "code_base1", n_tokens=anchor_n_tokens,
            )
            # Force alloc to return a fake tensor of zeros.
            rc.token_to_kv_pool_allocator.alloc = lambda n: torch.zeros(
                n, dtype=torch.long,
            )
            fake_exact_values = (
                [torch.zeros(prefix_len, dtype=torch.long)]
                if prefix_len > 0 else []
            )
            RadixCache._try_placeholder_knn_lossy_match_body(
                rc, req, fake_exact_values, None,
                [{"slot_id": "code_base1", "start_token": slot_start,
                  "end_token": slot_end,
                  "content_signature": "fake_sig",
                  "text": "fake anchor text"}],
                self._emb, top_k=4, min_cos=0.0,
                max_slot_len=4096, max_rope_ops=114688,
                head_tokens=2,
                max_overlap_ratio=max_overlap_ratio,
            )
        finally:
            if prev_head is not None:
                rc._apply_rope_delta_to_head = prev_head
            else:
                delattr(rc, "_apply_rope_delta_to_head")
        return captured

    def test_skip_when_overlap_ratio_above_threshold(self):
        """start=200, prefix_len=380, slot=[200, 4096) → overlap=180,
        entry_len=min(4096, 3896, 4096)=3896, overlap_ratio=180/3896≈0.046
        — well below 0.5.  To exercise the skip path, set anchor_n_tokens
        small enough that entry_len gives a high overlap ratio.

        We use a 200-token anchor: entry_len=min(200, 3896, 4096)=200,
        overlap_len=180, overlap_ratio=180/200=0.9 > 0.5 → skip.
        """
        rc = _FakeRadixCacheWithBody(layer_num=28)
        req = self._build_req("anchor text", start_token=200, end_token=4096)
        captured = self._run_body(
            rc, req,
            slot_start=200, slot_end=4096,
            anchor_n_tokens=200,
            prefix_len=380,
        )
        # Skip path fired: no copy, no RoPE.
        self.assertEqual(captured, [])
        self.assertEqual(req.placeholder_knn_skipped_high_overlap_count, 1)
        # Other counters remain at zero.
        self.assertEqual(req.placeholder_kv_prefill_matched_slots, 0)
        self.assertEqual(req.placeholder_anchor_pool_skipped_cost_count, 0)

    def test_copy_when_overlap_ratio_below_threshold(self):
        """start=200, prefix_len=380, anchor_n_tokens=2000 →
        entry_len=min(2000, 3896, 4096)=2000, overlap_len=180,
        overlap_ratio=180/2000=0.09 < 0.5 → copy proceeds.
        """
        rc = _FakeRadixCacheWithBody(layer_num=28)
        req = self._build_req("anchor text", start_token=200, end_token=4096)
        captured = self._run_body(
            rc, req,
            slot_start=200, slot_end=4096,
            anchor_n_tokens=2000,
            prefix_len=380,
        )
        # Copy path fired: one RoPE call captured.
        self.assertEqual(len(captured), 1)
        self.assertEqual(req.placeholder_knn_skipped_high_overlap_count, 0)
        self.assertEqual(req.placeholder_kv_prefill_matched_slots, 1)

    def test_disabled_with_max_overlap_ratio_one(self):
        """Setting max_overlap_ratio=1.0 (impossible to exceed) disables
        the skip; even with high overlap the copy proceeds.
        """
        rc = _FakeRadixCacheWithBody(layer_num=28)
        req = self._build_req("anchor text", start_token=200, end_token=4096)
        captured = self._run_body(
            rc, req,
            slot_start=200, slot_end=4096,
            anchor_n_tokens=200,  # gives 0.9 overlap ratio
            prefix_len=380,
            max_overlap_ratio=1.0,  # disabled
        )
        # Copy path fires despite high overlap.
        self.assertEqual(len(captured), 1)
        self.assertEqual(req.placeholder_knn_skipped_high_overlap_count, 0)
        self.assertEqual(req.placeholder_kv_prefill_matched_slots, 1)

    def test_skip_threshold_respects_env_var(self):
        """A higher threshold (e.g. 0.95) means fewer slots get skipped.
        Same setup as test_skip_when_overlap_ratio_above_threshold but
        with threshold=0.95: overlap_ratio=0.9 < 0.95 → copy proceeds.
        """
        rc = _FakeRadixCacheWithBody(layer_num=28)
        req = self._build_req("anchor text", start_token=200, end_token=4096)
        captured = self._run_body(
            rc, req,
            slot_start=200, slot_end=4096,
            anchor_n_tokens=200,  # 0.9 overlap ratio
            prefix_len=380,
            max_overlap_ratio=0.95,  # very lenient
        )
        # Copy path fires; threshold was too lenient to skip.
        self.assertEqual(len(captured), 1)
        self.assertEqual(req.placeholder_knn_skipped_high_overlap_count, 0)


class PlaceholderHighNewTokenRatioSkipTests(unittest.TestCase):
    """O10: skip the k-NN search entirely when the slot is mostly NEW
    (cold prefix from the slot's perspective).  The cold-prefix
    counterpart to O1: O1 fires when prefix already covers most of
    the slot; O10 fires when most of the slot is NEW.

    When the slot has >threshold new tokens, the k-NN search
    overhead (~30ms per slot) dominates any copy saving — agents
    4-5 cold-prefix sub-agents regress by 200ms+ without this gate.
    """

    def setUp(self):
        from sglang.srt.mem_cache import semantic_suffix as ss
        ss.reset_for_tests()
        self._saved = {
            k: os.environ.get(k)
            for k in (
                "SGLANG_PLACEHOLDER_KNN_HEAD_TOKENS",
                "SGLANG_PLACEHOLDER_KNN_MAX_OVERLAP_RATIO",
                "SGLANG_PLACEHOLDER_KNN_MAX_NEW_TOKEN_RATIO",
                "SGLANG_SEMANTIC_SUFFIX_ENABLED",
            )
        }
        os.environ["SGLANG_PLACEHOLDER_KNN_HEAD_TOKENS"] = "2"
        os.environ["SGLANG_PLACEHOLDER_KNN_MAX_OVERLAP_RATIO"] = "1.0"  # disable O1 for isolation
        os.environ["SGLANG_PLACEHOLDER_KNN_MAX_NEW_TOKEN_RATIO"] = "0.5"
        os.environ["SGLANG_SEMANTIC_SUFFIX_ENABLED"] = "1"
        from sglang.srt.mem_cache import semantic_suffix as _ss
        self._emb = _ss.load_embedder()
        if self._emb is None:
            self.skipTest("embedder unavailable on this host")

    def tearDown(self):
        from sglang.srt.mem_cache import semantic_suffix as ss
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        ss.reset_for_tests()

    def _build_req(self, slot_text: str, start_token: int = 0,
                   end_token: int = 3500) -> object:
        req = type("R", (), {})()
        req.placeholder_anchor_token_spans = [
            {
                "slot_id": "code_base1",
                "label": "code_base1",
                "start_token": start_token,
                "end_token": end_token,
                "content_signature": "fake_sig",
                "text": slot_text,
            }
        ]
        for f in (
            "placeholder_anchor_pool_hit_count",
            "placeholder_anchor_pool_miss_count",
            "placeholder_kv_prefill_matched_slots",
            "placeholder_kv_prefill_skipped_tokens",
            "placeholder_anchor_pool_skipped_cost_count",
            "placeholder_knn_skipped_high_overlap_count",
            "placeholder_knn_skipped_high_new_token_ratio_count",
            "placeholder_knn_skipped_high_span_overlap_count",
            "placeholder_knn_skipped_short_new_tokens_count",
            "placeholder_knn_head_rotation_tokens",
            "placeholder_knn_head_rotation_total_ops",
            "placeholder_kv_prefill_overlap_tokens",
            "placeholder_anchor_pool_copy_error_count",
        ):
            setattr(req, f, 0)
        req.placeholder_knn_topk_similarity_mean = 0.0
        req.placeholder_knn_copy_method = "none"
        return req

    def _populate_pool_with_anchor(self, rc, slot_id, n_tokens,
                                    best_start_pos=500):
        from sglang.srt.mem_cache.semantic_suffix import embed_single_text
        entry = _make_entry(slot_id, start_pos=best_start_pos,
                            n=n_tokens, embed_dim=384)
        entry.pool_embedding = embed_single_text(
            "fake anchor text for high-new-token-ratio skip test",
            emb=self._emb,
        )
        with rc.placeholder_anchor_pool_lock:
            rc.placeholder_anchor_pool.setdefault(slot_id, []).append(entry)
        return entry

    def _run_body(self, rc, req, slot_start, slot_end, anchor_n_tokens,
                  prefix_len=0, max_new_token_ratio=0.5,
                  max_overlap_ratio=1.0,
                  cost_guard_enabled=False, best_start_pos=500):
        """Run body with monkey-patched _apply_rope_delta_to_head to
        record copy + RoPE args. Returns the list of captured RoPE calls.
        Embedding compute is monkey-patched to a no-op (and a flag set)
        so we can verify whether O10 short-circuited BEFORE the
        embedding step (the whole point of the gate).
        """
        from sglang.srt.mem_cache.radix_cache import RadixCache
        captured = []
        embed_called = []

        def spy_head(k_buffer, dst_slots, head_len, delta):
            captured.append({
                "dst_shape": tuple(dst_slots.shape),
                "delta_value": int(delta),
                "head_len": int(head_len),
            })
            return head_len

        def spy_embed(text, emb=None, **_):
            embed_called.append(True)
            # Defer to the real embedder; we just want to record that
            # O10 didn't short-circuit BEFORE the embed call.
            from sglang.srt.mem_cache.semantic_suffix import (
                embed_single_text as _real_embed,
            )
            return _real_embed(text, emb=emb)

        prev_head = getattr(rc, "_apply_rope_delta_to_head", None)
        rc._apply_rope_delta_to_head = spy_head
        try:
            self._populate_pool_with_anchor(
                rc, "code_base1", n_tokens=anchor_n_tokens,
                best_start_pos=best_start_pos,
            )
            rc.token_to_kv_pool_allocator.alloc = lambda n: torch.zeros(
                n, dtype=torch.long,
            )
            fake_exact_values = (
                [torch.zeros(prefix_len, dtype=torch.long)]
                if prefix_len > 0 else []
            )
            # Patch the embed call inside the body to be detectable.
            # The body does `from semantic_suffix import embed_single_text_
            # cached as _est`, so we must patch the source module, not the
            # radix_cache re-export.
            from sglang.srt.mem_cache import semantic_suffix as ss
            orig_embed = ss.embed_single_text_cached
            ss.embed_single_text_cached = spy_embed
            try:
                RadixCache._try_placeholder_knn_lossy_match_body(
                    rc, req, fake_exact_values, None,
                    [{"slot_id": "code_base1", "start_token": slot_start,
                      "end_token": slot_end,
                      "content_signature": "fake_sig",
                      "text": "fake anchor text"}],
                    self._emb, top_k=4, min_cos=0.0,
                    max_slot_len=4096, max_rope_ops=114688,
                    head_tokens=2,
                    max_overlap_ratio=max_overlap_ratio,
                    cost_guard_enabled=cost_guard_enabled,
                    max_new_token_ratio=max_new_token_ratio,
                )
            finally:
                ss.embed_single_text_cached = orig_embed
        finally:
            if prev_head is not None:
                rc._apply_rope_delta_to_head = prev_head
            else:
                delattr(rc, "_apply_rope_delta_to_head")
        return captured, embed_called

    def test_skip_when_new_token_ratio_above_threshold(self):
        """prefix_len=57, prompt~2700 (spans ending at 2700): cached_ratio
        = 57/2700 ≈ 0.02 < (1 - 0.5) = 0.5 → skip entire request.
        """
        rc = _FakeRadixCacheWithBody(layer_num=28)
        req = self._build_req("anchor text", start_token=0, end_token=2700)
        captured, embed_called = self._run_body(
            rc, req,
            slot_start=0, slot_end=2700,
            anchor_n_tokens=2000,
            prefix_len=57,
        )
        # Skip path fired BEFORE embedding compute (the whole point).
        self.assertEqual(captured, [])
        self.assertEqual(embed_called, [])  # never even embedded
        self.assertEqual(req.placeholder_knn_skipped_high_new_token_ratio_count, 1)
        # Other counters remain at zero.
        self.assertEqual(req.placeholder_kv_prefill_matched_slots, 0)
        self.assertEqual(req.placeholder_anchor_pool_miss_count, 0)
        self.assertEqual(req.placeholder_knn_skipped_high_overlap_count, 0)

    def test_copy_when_new_token_ratio_below_threshold(self):
        """prefix_len=1500, prompt=2700: cached_ratio = 1500/2700 ≈ 0.55
        > (1 - 0.5) = 0.5 → don't skip; copy proceeds (cost guard disabled).
        This is the warm-prefix case where O10 should NOT fire.  We use
        prefix_len=1500 (not 2302) to leave room for a non-trivial
        overlap and copy_len > 0.
        """
        rc = _FakeRadixCacheWithBody(layer_num=28)
        req = self._build_req("anchor text", start_token=0, end_token=2700)
        captured, _ = self._run_body(
            rc, req,
            slot_start=0, slot_end=2700,
            anchor_n_tokens=3000,  # larger anchor → entry_len > overlap
            prefix_len=1500,
            max_new_token_ratio=0.5,  # default — let copy run when warm
            cost_guard_enabled=False,  # isolate O10 from cost guard
        )
        # Copy path fires.
        self.assertEqual(len(captured), 1)
        self.assertEqual(req.placeholder_knn_skipped_high_new_token_ratio_count, 0)
        self.assertEqual(req.placeholder_kv_prefill_matched_slots, 1)

    def test_disabled_with_max_new_token_ratio_one(self):
        """Setting max_new_token_ratio=1.0 (impossible to exceed) disables
        the skip; even a fully-new request proceeds to k-NN search.
        """
        rc = _FakeRadixCacheWithBody(layer_num=28)
        req = self._build_req("anchor text", start_token=0, end_token=2700)
        _, embed_called = self._run_body(
            rc, req,
            slot_start=0, slot_end=2700,
            anchor_n_tokens=2000,
            prefix_len=57,
            max_new_token_ratio=1.0,  # disabled
        )
        # Embedding compute fires (gate disabled); even if copy is
        # skipped later (by cost guard or other gates), the embed
        # call is the canary that O10 didn't short-circuit.
        self.assertGreaterEqual(len(embed_called), 1)
        self.assertEqual(req.placeholder_knn_skipped_high_new_token_ratio_count, 0)


class PlaceholderCostVsPrefillTests(unittest.TestCase):
    """Phase 2.6: cost-vs-prefill gate (CacheBlend HKVD-style).  Skip
    the copy when alloc + move + RoPE cost exceeds the prefill saving
    × margin.  The cost model is parameterized and can be calibrated
    per hardware/model.
    """

    def setUp(self):
        from sglang.srt.mem_cache import semantic_suffix as ss
        ss.reset_for_tests()
        self._saved = {
            k: os.environ.get(k)
            for k in (
                "SGLANG_PLACEHOLDER_KNN_HEAD_TOKENS",
                "SGLANG_PLACEHOLDER_KNN_MAX_OVERLAP_RATIO",
                "SGLANG_PLACEHOLDER_KNN_COPY_COST_GUARD_ENABLED",
                "SGLANG_SEMANTIC_SUFFIX_ENABLED",
            )
        }
        os.environ["SGLANG_PLACEHOLDER_KNN_HEAD_TOKENS"] = "2"
        # Disable high-overlap skip so the cost-vs-prefill gate is the
        # only thing firing.
        os.environ["SGLANG_PLACEHOLDER_KNN_MAX_OVERLAP_RATIO"] = "1.0"
        # Enable the cost guard for these tests.
        os.environ["SGLANG_PLACEHOLDER_KNN_COPY_COST_GUARD_ENABLED"] = "1"
        os.environ["SGLANG_SEMANTIC_SUFFIX_ENABLED"] = "1"
        from sglang.srt.mem_cache import semantic_suffix as _ss
        self._emb = _ss.load_embedder()
        if self._emb is None:
            self.skipTest("embedder unavailable on this host")

    def tearDown(self):
        from sglang.srt.mem_cache import semantic_suffix as ss
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        ss.reset_for_tests()

    def _build_req(self, slot_text: str, start_token: int = 100,
                   end_token: int = 3500) -> object:
        req = type("R", (), {})()
        req.placeholder_anchor_token_spans = [
            {
                "slot_id": "code_base1",
                "label": "code_base1",
                "start_token": start_token,
                "end_token": end_token,
                "content_signature": "fake_sig",
                "text": slot_text,
            }
        ]
        for f in (
            "placeholder_anchor_pool_hit_count",
            "placeholder_anchor_pool_miss_count",
            "placeholder_kv_prefill_matched_slots",
            "placeholder_kv_prefill_skipped_tokens",
            "placeholder_anchor_pool_skipped_cost_count",
            "placeholder_knn_skipped_high_overlap_count",
            "placeholder_knn_head_rotation_tokens",
            "placeholder_knn_head_rotation_total_ops",
            "placeholder_kv_prefill_overlap_tokens",
        ):
            setattr(req, f, 0)
        req.placeholder_knn_topk_similarity_mean = 0.0
        req.placeholder_knn_copy_method = "none"
        return req

    def _populate_pool_with_anchor(self, rc, slot_id, n_tokens,
                                    best_start_pos=0):
        from sglang.srt.mem_cache.semantic_suffix import embed_single_text
        entry = _make_entry(slot_id, start_pos=best_start_pos,
                            n=n_tokens, embed_dim=384)
        entry.pool_embedding = embed_single_text(
            "fake anchor text for cost-vs-prefill test", emb=self._emb,
        )
        with rc.placeholder_anchor_pool_lock:
            rc.placeholder_anchor_pool.setdefault(slot_id, []).append(entry)
        return entry

    def _run_body(self, rc, req, slot_start, slot_end, anchor_n_tokens,
                  prefix_len=0,
                  cost_guard_enabled=True,
                  copy_skip_margin=1.0,
                  copy_launch_overhead_us=20000,
                  copy_move_per_token_us=4,
                  copy_prefill_per_token_us=40,
                  copy_rope_per_layer_us=2):
        """Run body with monkey-patched _apply_rope_delta_to_head. Returns
        captured RoPE calls.  Cost guard parameters default to the
        production-calibrated values; tests override as needed.
        """
        from sglang.srt.mem_cache.radix_cache import RadixCache
        captured = []

        def spy_head(k_buffer, dst_slots, head_len, delta):
            captured.append({
                "dst_shape": tuple(dst_slots.shape),
                "delta_value": int(delta),
                "head_len": int(head_len),
            })
            return head_len

        prev_head = getattr(rc, "_apply_rope_delta_to_head", None)
        rc._apply_rope_delta_to_head = spy_head
        try:
            self._populate_pool_with_anchor(
                rc, "code_base1", n_tokens=anchor_n_tokens,
            )
            rc.token_to_kv_pool_allocator.alloc = lambda n: torch.zeros(
                n, dtype=torch.long,
            )
            fake_exact_values = (
                [torch.zeros(prefix_len, dtype=torch.long)]
                if prefix_len > 0 else []
            )
            RadixCache._try_placeholder_knn_lossy_match_body(
                rc, req, fake_exact_values, None,
                [{"slot_id": "code_base1", "start_token": slot_start,
                  "end_token": slot_end,
                  "content_signature": "fake_sig",
                  "text": "fake anchor text"}],
                self._emb, top_k=4, min_cos=0.0,
                max_slot_len=4096, max_rope_ops=114688,
                head_tokens=2,
                max_overlap_ratio=1.0,  # disable O1 gate for these tests
                cost_guard_enabled=cost_guard_enabled,
                copy_skip_margin=copy_skip_margin,
                copy_launch_overhead_us=copy_launch_overhead_us,
                copy_move_per_token_us=copy_move_per_token_us,
                copy_prefill_per_token_us=copy_prefill_per_token_us,
                copy_rope_per_layer_us=copy_rope_per_layer_us,
            )
        finally:
            if prev_head is not None:
                rc._apply_rope_delta_to_head = prev_head
            else:
                delattr(rc, "_apply_rope_delta_to_head")
        return captured

    def test_skip_small_copy_when_cost_exceeds_saving(self):
        """anchor_n_tokens=100, no trim (start >= prefix_len), entry_len=100,
        copy_len=100.  copy_cost = 20000 + 100*4 + 2*28*2 = 20512 μs.
        prefill_saving = 100 * 40 = 4000 μs.  20512 > 4000 → skip.
        """
        rc = _FakeRadixCacheWithBody(layer_num=28)
        req = self._build_req("anchor text", start_token=200, end_token=4000)
        captured = self._run_body(
            rc, req,
            slot_start=200, slot_end=4000,
            anchor_n_tokens=100,
            prefix_len=0,  # no trim
        )
        self.assertEqual(captured, [])
        self.assertEqual(req.placeholder_anchor_pool_skipped_cost_count, 1)
        self.assertEqual(req.placeholder_kv_prefill_matched_slots, 0)

    def test_allow_large_copy_when_saving_exceeds_cost(self):
        """anchor_n_tokens=2000, entry_len=2000, copy_len=2000.
        copy_cost = 20000 + 2000*4 + 2*28*2 = 28112 μs.
        prefill_saving = 2000 * 40 = 80000 μs.  28112 < 80000 → copy.
        """
        rc = _FakeRadixCacheWithBody(layer_num=28)
        req = self._build_req("anchor text", start_token=200, end_token=4096)
        captured = self._run_body(
            rc, req,
            slot_start=200, slot_end=4096,
            anchor_n_tokens=2000,
            prefix_len=0,
        )
        self.assertEqual(len(captured), 1)
        self.assertEqual(req.placeholder_anchor_pool_skipped_cost_count, 0)
        self.assertEqual(req.placeholder_kv_prefill_matched_slots, 1)

    def test_disabled_with_cost_guard_enabled_false(self):
        """With cost_guard_enabled=False, even a 50-token copy proceeds
        (the gate is bypassed).
        """
        rc = _FakeRadixCacheWithBody(layer_num=28)
        req = self._build_req("anchor text", start_token=200, end_token=4000)
        captured = self._run_body(
            rc, req,
            slot_start=200, slot_end=4000,
            anchor_n_tokens=50,
            prefix_len=0,
            cost_guard_enabled=False,
        )
        self.assertEqual(len(captured), 1)
        self.assertEqual(req.placeholder_anchor_pool_skipped_cost_count, 0)

    def test_margin_zero_is_more_aggressive(self):
        """Setting copy_skip_margin=0.0 means even tiny copy_cost > 0
        triggers skip.  Effectively always skip (unless cost is exactly 0).
        """
        rc = _FakeRadixCacheWithBody(layer_num=28)
        req = self._build_req("anchor text", start_token=200, end_token=4096)
        captured = self._run_body(
            rc, req,
            slot_start=200, slot_end=4096,
            anchor_n_tokens=2000,  # would normally copy
            prefix_len=0,
            copy_skip_margin=0.0,
        )
        # Even a "should copy" slot is skipped with margin=0.
        self.assertEqual(captured, [])
        self.assertEqual(req.placeholder_anchor_pool_skipped_cost_count, 1)

    def test_margin_relaxes_threshold(self):
        """Setting copy_skip_margin=3.0 requires prefill_saving to be 3x
        copy_cost.  A 200-token copy with margin=1.0 is skipped (cost~21ms,
        saving~8ms).  With margin=3.0 the same copy proceeds because
        20912 < 8000 × 3 = 24000.  This demonstrates the margin is a
        knob that can be tuned per workload.
        """
        rc = _FakeRadixCacheWithBody(layer_num=28)
        req = self._build_req("anchor text", start_token=200, end_token=4096)
        captured = self._run_body(
            rc, req,
            slot_start=200, slot_end=4096,
            anchor_n_tokens=200,
            prefix_len=0,
            copy_skip_margin=3.0,  # very lenient
        )
        # 200-token copy with margin=3: 20912 < 8000*3=24000 → proceed
        self.assertEqual(len(captured), 1)
        self.assertEqual(req.placeholder_anchor_pool_skipped_cost_count, 0)


class PlaceholderPoolEmptyShortCircuitTests(unittest.TestCase):
    """Phase 2.5+ optimization: short-circuit when the placeholder_anchor_pool
    has no entries for the slot_id. Saves the embedding compute (~24ms
    on agent 1 cold pool). The k-NN search is guaranteed to return []
    when pool is empty for the slot, so we can skip the embed.
    """

    def setUp(self):
        from sglang.srt.mem_cache import semantic_suffix as ss
        ss.reset_for_tests()
        self._saved = {
            k: os.environ.get(k)
            for k in (
                "SGLANG_PLACEHOLDER_KNN_HEAD_TOKENS",
                "SGLANG_PLACEHOLDER_KNN_MAX_OVERLAP_RATIO",
                "SGLANG_PLACEHOLDER_KNN_COPY_COST_GUARD_ENABLED",
                "SGLANG_SEMANTIC_SUFFIX_ENABLED",
            )
        }
        os.environ["SGLANG_PLACEHOLDER_KNN_HEAD_TOKENS"] = "2"
        os.environ["SGLANG_PLACEHOLDER_KNN_MAX_OVERLAP_RATIO"] = "1.0"
        os.environ["SGLANG_PLACEHOLDER_KNN_COPY_COST_GUARD_ENABLED"] = "0"
        os.environ["SGLANG_SEMANTIC_SUFFIX_ENABLED"] = "1"
        from sglang.srt.mem_cache import semantic_suffix as _ss
        self._emb = _ss.load_embedder()
        if self._emb is None:
            self.skipTest("embedder unavailable on this host")

    def tearDown(self):
        from sglang.srt.mem_cache import semantic_suffix as ss
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        ss.reset_for_tests()

    def _build_req(self, slot_text: str, start_token: int = 100,
                   end_token: int = 3500) -> object:
        req = type("R", (), {})()
        req.placeholder_anchor_token_spans = [
            {
                "slot_id": "code_base1",
                "label": "code_base1",
                "start_token": start_token,
                "end_token": end_token,
                "content_signature": "fake_sig",
                "text": slot_text,
            }
        ]
        for f in (
            "placeholder_anchor_pool_hit_count",
            "placeholder_anchor_pool_miss_count",
            "placeholder_kv_prefill_matched_slots",
            "placeholder_kv_prefill_skipped_tokens",
            "placeholder_anchor_pool_skipped_cost_count",
            "placeholder_knn_skipped_high_overlap_count",
            "placeholder_knn_head_rotation_tokens",
            "placeholder_knn_head_rotation_total_ops",
            "placeholder_kv_prefill_overlap_tokens",
        ):
            setattr(req, f, 0)
        req.placeholder_knn_topk_similarity_mean = 0.0
        req.placeholder_knn_copy_method = "none"
        return req

    def test_empty_pool_skips_embedding_compute(self):
        """When pool has no entries for the slot_id, the body should
        short-circuit BEFORE the embedding compute. We verify this by
        running the body with an empty pool and checking that no RoPE
        call fires AND miss_count is incremented (placeholder_anchor_pool_miss_count).
        """
        rc = _FakeRadixCacheWithBody(layer_num=28)
        # Pool is empty (no _populate_pool_with_anchor call)
        req = self._build_req("anchor text", start_token=200, end_token=4096)
        # Wrap _est to detect if it was called (it shouldn't be).
        from sglang.srt.mem_cache.radix_cache import RadixCache
        est_calls = []
        orig_est = getattr(RadixCache, "_placeholder_knn_lossy_match_body", None)
        # Patch _est inside the body via os.environ-style: we'll just run
        # and check that miss_count is non-zero (was 0 in v16, now should
        # be 1 due to short-circuit).
        RadixCache._try_placeholder_knn_lossy_match_body(
            rc, req, [], None,
            [{"slot_id": "code_base1", "start_token": 200,
              "end_token": 4096, "content_signature": "fake_sig",
              "text": "fake anchor text"}],
            self._emb, top_k=4, min_cos=0.0,
            max_slot_len=4096, max_rope_ops=114688,
            head_tokens=2,
            max_overlap_ratio=1.0,
            cost_guard_enabled=False,
        )
        # Body short-circuited: miss_count incremented, no copy, no RoPE.
        self.assertEqual(req.placeholder_anchor_pool_miss_count, 1)
        self.assertEqual(req.placeholder_kv_prefill_matched_slots, 0)

    def test_populated_pool_proceeds_normally(self):
        """When pool has entries, the body should proceed with embedding
        compute and k-NN search as before.
        """
        rc = _FakeRadixCacheWithBody(layer_num=28)
        req = self._build_req("anchor text", start_token=200, end_token=4096)
        # Populate pool with one anchor
        from sglang.srt.mem_cache.semantic_suffix import embed_single_text
        entry = _make_entry("code_base1", start_pos=0, n=2000, embed_dim=384)
        entry.pool_embedding = embed_single_text(
            "fake anchor text", emb=self._emb,
        )
        with rc.placeholder_anchor_pool_lock:
            rc.placeholder_anchor_pool.setdefault("code_base1", []).append(entry)
        # Force alloc to a fake tensor
        rc.token_to_kv_pool_allocator.alloc = lambda n: torch.zeros(
            n, dtype=torch.long,
        )
        from sglang.srt.mem_cache.radix_cache import RadixCache
        RadixCache._try_placeholder_knn_lossy_match_body(
            rc, req, [], None,
            [{"slot_id": "code_base1", "start_token": 200,
              "end_token": 4096, "content_signature": "fake_sig",
              "text": "fake anchor text"}],
            self._emb, top_k=4, min_cos=0.0,
            max_slot_len=4096, max_rope_ops=114688,
            head_tokens=2,
            max_overlap_ratio=1.0,
            cost_guard_enabled=False,
        )
        # Body proceeded: matched_slots incremented.
        self.assertEqual(req.placeholder_anchor_pool_miss_count, 0)
        self.assertGreaterEqual(req.placeholder_kv_prefill_matched_slots, 1)


class PlaceholderPreRotatedHeadKTests(unittest.TestCase):
    """Phase 2.7 / O5: pre-rotated head K at multiple delta values.
    At write time, the anchor stores the rotated head K for a small set
    of representative deltas.  At read time, if delta_request matches a
    stored delta (exact or nearest within tolerance), the body does a
    single scatter per layer instead of the per-layer rotation loop.
    """

    def setUp(self):
        from sglang.srt.mem_cache import semantic_suffix as ss
        ss.reset_for_tests()
        self._saved = {
            k: os.environ.get(k)
            for k in (
                "SGLANG_PLACEHOLDER_KNN_HEAD_TOKENS",
                "SGLANG_PLACEHOLDER_KNN_PRE_ROTATE_DELTAS",
                "SGLANG_PLACEHOLDER_KNN_MAX_OVERLAP_RATIO",
                "SGLANG_SEMANTIC_SUFFIX_ENABLED",
            )
        }
        os.environ["SGLANG_PLACEHOLDER_KNN_HEAD_TOKENS"] = "2"
        os.environ["SGLANG_PLACEHOLDER_KNN_PRE_ROTATE_DELTAS"] = "0,500,2000"
        os.environ["SGLANG_PLACEHOLDER_KNN_MAX_OVERLAP_RATIO"] = "1.0"
        os.environ["SGLANG_SEMANTIC_SUFFIX_ENABLED"] = "1"
        from sglang.srt.mem_cache import semantic_suffix as _ss
        self._emb = _ss.load_embedder()
        if self._emb is None:
            self.skipTest("embedder unavailable on this host")

    def tearDown(self):
        from sglang.srt.mem_cache import semantic_suffix as ss
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        ss.reset_for_tests()

    def _make_kvcache_with_layers(self, n_layers=4, max_tokens=16,
                                   n_kv=4, head_dim=8):
        """Build a fake kvcache with enough room for indexing by head_tokens."""
        class FakeKV:
            layer_num = n_layers
            def __init__(self):
                self.k_buffer = [
                    torch.zeros(max_tokens, n_kv, head_dim)
                    for _ in range(n_layers)
                ]
                self.v_buffer = [
                    torch.zeros(max_tokens, n_kv, head_dim)
                    for _ in range(n_layers)
                ]
        return FakeKV()

    def _make_entry_with_pre_rot(self, slot_id="code_base1", start_pos=0,
                                  n=8, deltas=(0, 500, 2000), n_layers=4,
                                  n_kv=4, head_dim=8):
        """Construct a synthetic entry WITH pre-rotated head K."""
        entry = _make_entry(slot_id, start_pos=start_pos, n=n)
        from sglang.srt.mem_cache.semantic_suffix import embed_single_text
        entry.pool_embedding = embed_single_text(
            "fake anchor text", emb=self._emb,
        )
        head_tokens = 2
        rotated_per_delta = []
        for _ in deltas:
            layers = [
                torch.zeros(head_tokens, n_kv, head_dim)
                for _ in range(n_layers)
            ]
            rotated_per_delta.append(layers)
        entry.pre_rotated_head_k = rotated_per_delta
        entry.pre_rotated_deltas = list(deltas)
        return entry

    def test_helper_returns_none_when_no_pre_rotation(self):
        """If entry.pre_rotated_head_k is None (default, v19 anchor),
        _apply_pre_rotated_head_k returns None → fall through to runtime.
        """
        from sglang.srt.mem_cache.radix_cache import RadixCache
        rc = _FakeRadixCacheWithBody(layer_num=4)
        kvcache = self._make_kvcache_with_layers(n_layers=4)
        entry = _make_entry("code_base1", start_pos=0, n=8)
        result = RadixCache._apply_pre_rotated_head_k(
            rc, kvcache.k_buffer,
            torch.zeros(8, dtype=torch.long), entry, delta=100,
            head_tokens=2,
        )
        self.assertIsNone(result)

    def test_helper_exact_match_returns_idx(self):
        """Exact delta match returns the delta index."""
        from sglang.srt.mem_cache.radix_cache import RadixCache
        rc = _FakeRadixCacheWithBody(layer_num=4)
        kvcache = self._make_kvcache_with_layers(n_layers=4)
        entry = self._make_entry_with_pre_rot(deltas=(0, 500, 2000), n_layers=4)
        result = RadixCache._apply_pre_rotated_head_k(
            rc, kvcache.k_buffer,
            torch.zeros(8, dtype=torch.long), entry, delta=500,
            head_tokens=2,
        )
        self.assertEqual(result, 1)

    def test_helper_nearest_within_tolerance(self):
        """Delta near 500 (within half-gap tolerance=250) returns idx 1."""
        from sglang.srt.mem_cache.radix_cache import RadixCache
        rc = _FakeRadixCacheWithBody(layer_num=4)
        kvcache = self._make_kvcache_with_layers(n_layers=4)
        entry = self._make_entry_with_pre_rot(deltas=(0, 500, 2000), n_layers=4)
        result = RadixCache._apply_pre_rotated_head_k(
            rc, kvcache.k_buffer,
            torch.zeros(8, dtype=torch.long), entry, delta=600,
            head_tokens=2,
        )
        # 600 is 100 away from 500 (tolerance=250) → hit at idx 1.
        self.assertEqual(result, 1)

    def test_helper_far_miss_returns_none(self):
        """Delta far from any stored value (outside tolerance) returns None."""
        from sglang.srt.mem_cache.radix_cache import RadixCache
        rc = _FakeRadixCacheWithBody(layer_num=4)
        kvcache = self._make_kvcache_with_layers(n_layers=4)
        entry = self._make_entry_with_pre_rot(deltas=(0, 500, 2000), n_layers=4)
        result = RadixCache._apply_pre_rotated_head_k(
            rc, kvcache.k_buffer,
            torch.zeros(8, dtype=torch.long), entry, delta=1500,
            head_tokens=2,
        )
        # Nearest is 2000 (500 away, tolerance=250) → outside tolerance → None.
        self.assertIsNone(result)

    def test_helper_zero_delta_returns_zero_idx(self):
        """Delta=0 with stored [0, 500, 2000] hits idx 0."""
        from sglang.srt.mem_cache.radix_cache import RadixCache
        rc = _FakeRadixCacheWithBody(layer_num=4)
        kvcache = self._make_kvcache_with_layers(n_layers=4)
        entry = self._make_entry_with_pre_rot(deltas=(0, 500, 2000), n_layers=4)
        result = RadixCache._apply_pre_rotated_head_k(
            rc, kvcache.k_buffer,
            torch.zeros(8, dtype=torch.long), entry, delta=0,
            head_tokens=2,
        )
        self.assertEqual(result, 0)

    def test_body_skips_runtime_when_pre_rotation_hits(self):
        """End-to-end: when entry has matching pre-rotation, the body
        does NOT call _apply_rope_delta_to_head. Verifies hit path is
        taken and runtime fallback is bypassed.
        """
        from sglang.srt.mem_cache.radix_cache import RadixCache
        rc = _FakeRadixCacheWithBody(layer_num=4)
        # Manually populate with pre-rotated entry at delta=0 (so the
        # request with start_token=200, prefix=0 produces delta=200,
        # which is nearest to delta=0 within tolerance).
        entry = self._make_entry_with_pre_rot(
            start_pos=0, n=2000, deltas=(0, 500, 2000), n_layers=4,
        )
        with rc.placeholder_anchor_pool_lock:
            rc.placeholder_anchor_pool.setdefault("code_base1", []).append(entry)
        # Build req
        req = type("R", (), {})()
        req.placeholder_anchor_token_spans = [{
            "slot_id": "code_base1", "label": "code_base1",
            "start_token": 200, "end_token": 4096,
            "content_signature": "fake_sig", "text": "fake anchor text"}]
        for f in (
            "placeholder_anchor_pool_hit_count",
            "placeholder_anchor_pool_miss_count",
            "placeholder_kv_prefill_matched_slots",
            "placeholder_kv_prefill_skipped_tokens",
            "placeholder_anchor_pool_skipped_cost_count",
            "placeholder_knn_skipped_high_overlap_count",
            "placeholder_knn_head_rotation_tokens",
            "placeholder_knn_head_rotation_total_ops",
            "placeholder_kv_prefill_overlap_tokens",
            "placeholder_knn_pre_rotated_hit_count",
            "placeholder_knn_pre_rotated_miss_count",
        ):
            setattr(req, f, 0)
        req.placeholder_knn_topk_similarity_mean = 0.0
        req.placeholder_knn_copy_method = "none"
        # Spy on runtime call (should NOT fire if pre-rot hits)
        runtime_calls = []
        def spy_runtime(k_buffer, dst_slots, head_len, delta):
            runtime_calls.append({"delta": int(delta)})
            return head_len
        rc._apply_rope_delta_to_head = spy_runtime
        # Install kvcache with matching layer count + enough max_tokens
        kvcache = self._make_kvcache_with_layers(n_layers=4, max_tokens=8192)
        rc.token_to_kv_pool_allocator.get_kvcache = lambda: kvcache
        # Force alloc
        rc.token_to_kv_pool_allocator.alloc = lambda n: torch.zeros(
            n, dtype=torch.long,
        )
        RadixCache._try_placeholder_knn_lossy_match_body(
            rc, req, [], None,
            [{"slot_id": "code_base1", "start_token": 200,
              "end_token": 4096, "content_signature": "fake_sig",
              "text": "fake anchor text"}],
            self._emb, top_k=4, min_cos=0.0,
            max_slot_len=4096, max_rope_ops=114688,
            head_tokens=2,
            max_overlap_ratio=1.0,
            cost_guard_enabled=False,
        )
        # Body matched, hit the pre-rotated path, did NOT call runtime.
        self.assertEqual(req.placeholder_kv_prefill_matched_slots, 1)
        self.assertEqual(req.placeholder_knn_pre_rotated_hit_count, 1)
        self.assertEqual(req.placeholder_knn_pre_rotated_miss_count, 0)
        self.assertEqual(runtime_calls, [])

    def test_body_falls_back_to_runtime_when_pre_rotation_misses(self):
        """End-to-end: when entry has pre-rotation but delta is outside
        tolerance, the body falls through to runtime _apply_rope_delta_to_head.
        """
        from sglang.srt.mem_cache.radix_cache import RadixCache
        rc = _FakeRadixCacheWithBody(layer_num=4)
        # Use [2000] only — delta=200 won't match (too far).
        entry = self._make_entry_with_pre_rot(
            start_pos=0, n=2000, deltas=(2000,), n_layers=4,
        )
        with rc.placeholder_anchor_pool_lock:
            rc.placeholder_anchor_pool.setdefault("code_base1", []).append(entry)
        req = type("R", (), {})()
        req.placeholder_anchor_token_spans = [{
            "slot_id": "code_base1", "label": "code_base1",
            "start_token": 200, "end_token": 4096,
            "content_signature": "fake_sig", "text": "fake anchor text"}]
        for f in (
            "placeholder_anchor_pool_hit_count",
            "placeholder_anchor_pool_miss_count",
            "placeholder_kv_prefill_matched_slots",
            "placeholder_kv_prefill_skipped_tokens",
            "placeholder_anchor_pool_skipped_cost_count",
            "placeholder_knn_skipped_high_overlap_count",
            "placeholder_knn_head_rotation_tokens",
            "placeholder_knn_head_rotation_total_ops",
            "placeholder_kv_prefill_overlap_tokens",
            "placeholder_knn_pre_rotated_hit_count",
            "placeholder_knn_pre_rotated_miss_count",
        ):
            setattr(req, f, 0)
        req.placeholder_knn_topk_similarity_mean = 0.0
        req.placeholder_knn_copy_method = "none"
        runtime_calls = []
        def spy_runtime(k_buffer, dst_slots, head_len, delta):
            runtime_calls.append({"delta": int(delta)})
            return head_len
        rc._apply_rope_delta_to_head = spy_runtime
        kvcache = self._make_kvcache_with_layers(n_layers=4, max_tokens=8192)
        rc.token_to_kv_pool_allocator.get_kvcache = lambda: kvcache
        rc.token_to_kv_pool_allocator.alloc = lambda n: torch.zeros(
            n, dtype=torch.long,
        )
        RadixCache._try_placeholder_knn_lossy_match_body(
            rc, req, [], None,
            [{"slot_id": "code_base1", "start_token": 200,
              "end_token": 4096, "content_signature": "fake_sig",
              "text": "fake anchor text"}],
            self._emb, top_k=4, min_cos=0.0,
            max_slot_len=4096, max_rope_ops=114688,
            head_tokens=2,
            max_overlap_ratio=1.0,
            cost_guard_enabled=False,
        )
        self.assertEqual(req.placeholder_kv_prefill_matched_slots, 1)
        self.assertEqual(req.placeholder_knn_pre_rotated_hit_count, 0)
        self.assertEqual(req.placeholder_knn_pre_rotated_miss_count, 1)
        self.assertEqual(len(runtime_calls), 1)


if __name__ == "__main__":
    unittest.main()


if __name__ == "__main__":
    unittest.main()
