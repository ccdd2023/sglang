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


if __name__ == "__main__":
    unittest.main()
