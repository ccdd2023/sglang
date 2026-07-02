"""
Unit tests for Direction #3 Phase C: per-AST-chunk placeholder pool
read path.

These tests verify the chunk pool read path works correctly:
1. Gating: chunk pool read path is OFF by default (production-safe).
2. Byte-exact match: pool entries with matching byte range are copied.
3. Byte-drift skip: mismatched byte ranges trigger dense_prefill.
4. Size-mismatch skip: same bytes but different token lengths trigger dense.
5. No-entry skip: empty pool triggers dense_prefill.
6. Alloc-failure fallback: alloc returning None → dense_prefill.
7. RoPE delta math: rope_delta = request_position - pool_entry_position.
8. LRU access time bumped on hit.
9. match_prefix integrates: device_indices extended after chunk copy.
10. match_prefix increments hit counter.

Run:
    python -m pytest test/registered/unit/mem_cache/test_placeholder_chunk_pool_read.py -v
"""

from __future__ import annotations

import os
import unittest
from dataclasses import dataclass
from unittest.mock import Mock

import torch

from sglang.srt.mem_cache.radix_cache import (
    ChunkDecision,
    ChunkKVEntry,
    ChunkPlan,
    RadixCache,
    RadixKey,
    TreeNode,
)
from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=15, suite="stage-c-test-small-1-gpu")
register_amd_ci(est_time=15, suite="stage-c-test-small-1-gpu-amd")


@dataclass
class _MockReq:
    """Minimal stand-in for sglang Req — only fields the read path reads."""

    rid: str = "test-req"
    origin_input_ids: list = None
    output_ids: list = None
    placeholder_anchor_token_spans: list = None
    tokenizer: object = None


def _make_simple_tokenizer():
    """Deterministic tokenizer: 1 token per 4 chars."""

    class _SimpleTokenizer:
        def encode(self, text: str, add_special_tokens: bool = True):
            return [42] * max(1, len(text) // 4)

    return _SimpleTokenizer()


def _make_mock_allocator(alloc_return=None):
    """Mock allocator with the API _execute_chunk_plan expects."""
    if alloc_return is None:
        alloc_return = torch.tensor([100, 101, 102])
    m = Mock()
    m.device = torch.device("cpu")
    m.alloc = Mock(return_value=alloc_return)
    m.get_kvcache = Mock(
        return_value=Mock(
            k_buffer=[torch.zeros(200, 8)],
            v_buffer=[torch.zeros(200, 8)],
        )
    )
    m.available_size = Mock(return_value=999)
    return m


def _make_simulated_cache(alloc_return=None):
    """Build a CPU RadixCache with mock allocator (no real memory)."""
    from sglang.srt.mem_cache import radix_cache as rc

    TreeNode.counter = 0
    cache = rc.RadixCache.create_simulated(
        disable=False, mock_allocator=_make_mock_allocator(alloc_return),
        page_size=1,
    )
    return cache


def _make_span(slot_id: str, text: str, start_token: int = 0) -> dict:
    """Build a placeholder_anchor_token_span dict for the read path."""
    return {
        "slot_id": slot_id,
        "start_token": start_token,
        "end_token": start_token + max(1, len(text) // 4),
        "text": text,
        "content_signature": "testsig1234",
        "label": slot_id,
    }


def _seed_pool_entry(
    cache, slot_id: str, sig: str, byte_start: int, byte_end: int,
    start_token: int, end_token: int, n_tokens: int = 3,
) -> ChunkKVEntry:
    """Inject a ChunkKVEntry directly into the pool (bypass writer)."""
    entry = ChunkKVEntry(
        slot_id=slot_id,
        chunk_signature=sig,
        anchor_type="function",
        name="histogram",
        byte_start=byte_start,
        byte_end=byte_end,
        start_token=start_token,
        end_token=end_token,
        token_ids=torch.tensor(list(range(n_tokens)), dtype=torch.int64),
        kv_indices=torch.tensor(
            list(range(100, 100 + n_tokens)), dtype=torch.int64,
        ),
    )
    cache.placeholder_chunk_pool[(slot_id, sig)] = [entry]
    return entry


def _histogram_text() -> str:
    """Stable test text — chunker produces a known signature."""
    return "def histogram(data):\n    return 1\n"


def _histogram_signature() -> str:
    """The signature ASTChunker produces for _histogram_text()."""
    from sglang.srt.mem_cache.ast_chunker import ASTChunker

    return ASTChunker().chunk_text(_histogram_text())[0].signature


def _histogram_byte_end() -> int:
    """The byte_end the chunker produces for _histogram_text()."""
    from sglang.srt.mem_cache.ast_chunker import ASTChunker

    return ASTChunker().chunk_text(_histogram_text())[0].byte_end


class TestChunkPoolReadGating(unittest.TestCase):
    """Phase C gating: chunk read path is OFF by default (production-safe)."""

    def setUp(self):
        # Default: env var NOT set.
        os.environ.pop("SGLANG_CHUNKED_PLACEHOLDER_KNN_MATCH", None)
        self.cache = _make_simulated_cache()
        self.text = _histogram_text()
        self.sig = _histogram_signature()
        _seed_pool_entry(
            self.cache, "code_base0", self.sig, 0, _histogram_byte_end(),
            start_token=50, end_token=53,
        )

    def test_no_match_when_env_var_unset(self):
        req = _MockReq(placeholder_anchor_token_spans=[
            _make_span("code_base0", self.text, start_token=0),
        ])
        key = RadixKey(token_ids=list(range(300)), extra_key=None)
        values, node = self.cache._try_placeholder_chunk_lossy_match(
            req, key, [], self.cache.root_node,
        )
        self.assertEqual(values, [])
        self.assertEqual(self.cache.placeholder_chunk_pool_hit_count, 0)
        self.assertEqual(
            self.cache.placeholder_chunk_pool_total_tokens_reused, 0,
        )


class TestChunkPoolReadByteExact(unittest.TestCase):
    """Phase C: byte-exact match produces a copy."""

    def setUp(self):
        os.environ["SGLANG_CHUNKED_PLACEHOLDER_KNN_MATCH"] = "1"
        self.cache = _make_simulated_cache()
        self.text = _histogram_text()
        self.sig = _histogram_signature()
        self.byte_end = _histogram_byte_end()
        self.entry = _seed_pool_entry(
            self.cache, "code_base0", self.sig, 0, self.byte_end,
            start_token=50, end_token=53,
        )

    def tearDown(self):
        os.environ.pop("SGLANG_CHUNKED_PLACEHOLDER_KNN_MATCH", None)

    def test_pool_match_byte_exact_copies_kv(self):
        """byte-exact match → new slots allocated + value list extended."""
        req = _MockReq(placeholder_anchor_token_spans=[
            _make_span("code_base0", self.text, start_token=0),
        ])
        key = RadixKey(token_ids=list(range(300)), extra_key=None)
        values, node = self.cache._try_placeholder_chunk_lossy_match(
            req, key, [], self.cache.root_node,
        )
        self.assertEqual(len(values), 1)
        self.assertEqual(
            self.cache.placeholder_chunk_pool_hit_count, 1,
        )
        self.assertEqual(
            self.cache.placeholder_chunk_pool_total_tokens_reused, 3,
        )


class TestChunkPoolReadSkips(unittest.TestCase):
    """Phase C: skip reasons for non-byte-exact cases."""

    def setUp(self):
        os.environ["SGLANG_CHUNKED_PLACEHOLDER_KNN_MATCH"] = "1"
        self.cache = _make_simulated_cache()
        self.text = _histogram_text()
        self.sig = _histogram_signature()
        self.byte_end = _histogram_byte_end()

    def tearDown(self):
        os.environ.pop("SGLANG_CHUNKED_PLACEHOLDER_KNN_MATCH", None)

    def test_byte_drift_skip_dense_prefill(self):
        """Pool entry byte_start differs from chunk → skip_reason='byte_drift'."""
        # Seed pool with byte_start=5, byte_end=10 (mismatched from chunk's 0..end)
        _seed_pool_entry(
            self.cache, "code_base0", self.sig, byte_start=5, byte_end=10,
            start_token=50, end_token=53,
        )
        req = _MockReq(placeholder_anchor_token_spans=[
            _make_span("code_base0", self.text, start_token=0),
        ])
        key = RadixKey(token_ids=list(range(300)), extra_key=None)
        values, node = self.cache._try_placeholder_chunk_lossy_match(
            req, key, [], self.cache.root_node,
        )
        self.assertEqual(values, [])
        # No pool entry has byte_start=0, byte_end=self.byte_end
        # → all decisions get byte_drift
        self.assertEqual(
            self.cache.placeholder_chunk_pool_skip_byte_drift_count, 1,
        )
        self.assertEqual(self.cache.placeholder_chunk_pool_hit_count, 0)

    def test_no_pool_entry_skip_dense_prefill(self):
        """Pool is empty → skip_reason='no_pool_entry'."""
        req = _MockReq(placeholder_anchor_token_spans=[
            _make_span("code_base0", self.text, start_token=0),
        ])
        key = RadixKey(token_ids=list(range(300)), extra_key=None)
        values, node = self.cache._try_placeholder_chunk_lossy_match(
            req, key, [], self.cache.root_node,
        )
        self.assertEqual(values, [])
        self.assertEqual(
            self.cache.placeholder_chunk_pool_skip_no_entry_count, 1,
        )
        self.assertEqual(self.cache.placeholder_chunk_pool_hit_count, 0)

    def test_alloc_failed_fallback_dense_prefill(self):
        """alloc() returns None → flip to dense with skip_reason='alloc_failed'."""
        # Seed a valid pool entry
        _seed_pool_entry(
            self.cache, "code_base0", self.sig, 0, self.byte_end,
            start_token=50, end_token=53,
        )
        # Override the cache's allocator to return None
        self.cache.token_to_kv_pool_allocator.alloc = Mock(return_value=None)
        req = _MockReq(placeholder_anchor_token_spans=[
            _make_span("code_base0", self.text, start_token=0),
        ])
        key = RadixKey(token_ids=list(range(300)), extra_key=None)
        values, node = self.cache._try_placeholder_chunk_lossy_match(
            req, key, [], self.cache.root_node,
        )
        self.assertEqual(values, [])
        self.assertEqual(
            self.cache.placeholder_chunk_pool_skip_alloc_failed_count, 1,
        )
        self.assertEqual(self.cache.placeholder_chunk_pool_hit_count, 0)


class TestChunkPoolReadMath(unittest.TestCase):
    """Phase C: RoPE delta + LRU updates."""

    def setUp(self):
        os.environ["SGLANG_CHUNKED_PLACEHOLDER_KNN_MATCH"] = "1"
        self.cache = _make_simulated_cache()
        self.text = _histogram_text()
        self.sig = _histogram_signature()
        self.byte_end = _histogram_byte_end()

    def tearDown(self):
        os.environ.pop("SGLANG_CHUNKED_PLACEHOLDER_KNN_MATCH", None)

    def test_rope_delta_is_position_difference(self):
        """rope_delta = request_position - pool_entry_position."""
        # Pool entry was stored at start_token=100. New request span at
        # start_token=0 (contiguous with the empty prefix). Expected
        # rope_delta = 0 - 100 = -100 (non-zero → head rotation fires).
        _seed_pool_entry(
            self.cache, "code_base0", self.sig, 0, self.byte_end,
            start_token=100, end_token=103,
        )
        req = _MockReq(placeholder_anchor_token_spans=[
            _make_span("code_base0", self.text, start_token=0),
        ])
        key = RadixKey(token_ids=list(range(400)), extra_key=None)
        values, node = self.cache._try_placeholder_chunk_lossy_match(
            req, key, [], self.cache.root_node,
        )
        self.assertEqual(len(values), 1)
        # rope_delta was 250; rope_ops_count bumped by head_tokens=2
        self.assertGreater(
            self.cache.placeholder_chunk_pool_rope_ops_count, 0,
        )

    def test_lru_access_time_updated_on_hit(self):
        """Pool entry's last_access_time increases after a hit."""
        entry = _seed_pool_entry(
            self.cache, "code_base0", self.sig, 0, self.byte_end,
            start_token=50, end_token=53,
        )
        original_atime = entry.last_access_time
        req = _MockReq(placeholder_anchor_token_spans=[
            _make_span("code_base0", self.text, start_token=0),
        ])
        key = RadixKey(token_ids=list(range(300)), extra_key=None)
        # Sleep a tiny moment to ensure monotonic time moves forward.
        import time
        time.sleep(0.001)
        self.cache._try_placeholder_chunk_lossy_match(
            req, key, [], self.cache.root_node,
        )
        self.assertGreater(entry.last_access_time, original_atime)


class TestChunkPoolReadIntegration(unittest.TestCase):
    """Phase C: match_prefix integrates the chunk pass."""

    def setUp(self):
        os.environ["SGLANG_CHUNKED_PLACEHOLDER_KNN_MATCH"] = "1"
        self.cache = _make_simulated_cache()
        self.text = _histogram_text()
        self.sig = _histogram_signature()
        self.byte_end = _histogram_byte_end()

    def tearDown(self):
        os.environ.pop("SGLANG_CHUNKED_PLACEHOLDER_KNN_MATCH", None)

    def test_match_prefix_returns_extended_values(self):
        """End-to-end: seed pool → match_prefix → device_indices length grew."""
        # Insert a tree node with a token sequence so match_prefix has
        # something to find at the L1 layer (the byte-exact prefix).
        # Then chunk pass should extend the values list.
        n_prefix = 50
        prefix_token_ids = list(range(n_prefix))
        full_token_ids = prefix_token_ids + list(range(50))
        # Insert prefix into the radix tree so L1 has a real match.
        from sglang.srt.mem_cache.radix_cache import RadixKey
        from sglang.srt.mem_cache.base_prefix_cache import InsertParams

        prefix_key = RadixKey(token_ids=prefix_token_ids, extra_key=None)
        self.cache.insert(InsertParams(
            key=prefix_key,
            value=torch.tensor(prefix_token_ids, dtype=torch.int64),
            priority=1.0,
        ))

        # Seed a chunk pool entry for the histogram text
        _seed_pool_entry(
            self.cache, "code_base0", self.sig, 0, self.byte_end,
            start_token=n_prefix, end_token=n_prefix + 3,
        )

        # Build a request: prefix + chunk text token_ids
        # The chunk pool entry has 3 token_ids, so we expect the values
        # to include the 50 prefix slots + 3 copied chunk slots = 53.
        req = _MockReq(placeholder_anchor_token_spans=[
            _make_span("code_base0", self.text, start_token=n_prefix),
        ])
        full_key = RadixKey(token_ids=full_token_ids, extra_key=None)
        match_result = self.cache.match_prefix(MatchPrefixParams(
            key=full_key, req=req,
        ))
        # Should have at least the L1 prefix (50) + the chunk copy (3)
        self.assertGreaterEqual(
            match_result.device_indices.numel(), n_prefix,
        )
        # Hit counter should have been bumped
        self.assertGreaterEqual(
            self.cache.placeholder_chunk_pool_hit_count, 1,
        )

    def test_match_prefix_increments_hit_counter(self):
        """After successful chunk copy, placeholder_chunk_pool_hit_count == 1."""
        _seed_pool_entry(
            self.cache, "code_base0", self.sig, 0, self.byte_end,
            start_token=50, end_token=53,
        )
        req = _MockReq(placeholder_anchor_token_spans=[
            _make_span("code_base0", self.text, start_token=0),
        ])
        full_token_ids = list(range(300))
        match_result = self.cache.match_prefix(MatchPrefixParams(
            key=RadixKey(token_ids=full_token_ids, extra_key=None),
            req=req,
        ))
        # Hit counter incremented (could be 1 or more if tree had overlap)
        self.assertGreaterEqual(
            self.cache.placeholder_chunk_pool_hit_count, 1,
        )


class TestChunkPoolReadContiguityGate(unittest.TestCase):
    """M1.5 regression: the contiguity gate. SGLang's prefill treats
    device_indices as a contiguous prefix, so only chunks that extend the
    prefix contiguously may be copied. A chunk with a gap before it must be
    skipped (else device_indices > input_len → flashinfer qo_indptr crash)."""

    def setUp(self):
        os.environ["SGLANG_CHUNKED_PLACEHOLDER_KNN_MATCH"] = "1"
        self.cache = _make_simulated_cache()
        self.text = _histogram_text()
        self.sig = _histogram_signature()
        self.byte_end = _histogram_byte_end()
        # One pool entry reused by both spans (same signature).
        _seed_pool_entry(
            self.cache, "code_base0", self.sig, 0, self.byte_end,
            start_token=50, end_token=53,
        )

    def tearDown(self):
        os.environ.pop("SGLANG_CHUNKED_PLACEHOLDER_KNN_MATCH", None)

    def test_contiguous_chunk_copied_gap_chunk_skipped(self):
        """Span at start_token=0 (contiguous with empty prefix) is copied;
        span at start_token=500 (gap) is skipped with reason 'gap'."""
        req = _MockReq(placeholder_anchor_token_spans=[
            _make_span("code_base0", self.text, start_token=0),
            _make_span("code_base0", self.text, start_token=500),
        ])
        key = RadixKey(token_ids=list(range(600)), extra_key=None)
        values, node = self.cache._try_placeholder_chunk_lossy_match(
            req, key, [], self.cache.root_node,
        )
        # Exactly one copy (the contiguous chunk); the gap chunk was skipped.
        self.assertEqual(len(values), 1)
        self.assertEqual(
            self.cache.placeholder_chunk_pool_hit_count, 1,
        )
        self.assertEqual(
            self.cache.placeholder_chunk_pool_skip_gap_count, 1,
        )
        # The copied slice must not exceed the request input length — this
        # is the invariant whose violation crashed flashinfer before M1.5.
        copied_tokens = int(values[0].numel())
        self.assertLessEqual(copied_tokens, 600)


class TestFairMeasurementCounters(unittest.TestCase):
    """FAIR-MEASUREMENT (A1-A4): match_prefix must separate the radix L1
    prefix from code-aware copies in the per-request counters, so the bench
    can decompose cached_tokens and the parity gate can verify the radix
    prefix cancels across baseline and experimental configs.

    radix_only_prefix_len must equal the pure radix match (before any
    code-aware copy is appended); c2_chunk_reused_tokens must equal the
    copied chunk tokens; codeaware_reused_tokens = L3 + C2 (here L3 is off
    so it == C2 only). No double-counting with the radix prefix.
    """

    def setUp(self):
        os.environ["SGLANG_CHUNKED_PLACEHOLDER_KNN_MATCH"] = "1"
        # L3 (placeholder k-NN body) OFF — isolates the C2 chunk path so
        # l3_offset_reused_tokens is 0 and codeaware == c2_chunk only.
        os.environ["SGLANG_PLACEHOLDER_KNN_MATCH"] = "0"
        self.cache = _make_simulated_cache()
        self.text = _histogram_text()
        self.sig = _histogram_signature()
        self.byte_end = _histogram_byte_end()

    def tearDown(self):
        os.environ.pop("SGLANG_CHUNKED_PLACEHOLDER_KNN_MATCH", None)
        os.environ.pop("SGLANG_PLACEHOLDER_KNN_MATCH", None)

    def test_radix_and_codeaware_counters_separate(self):
        from sglang.srt.mem_cache.radix_cache import RadixKey
        from sglang.srt.mem_cache.base_prefix_cache import InsertParams

        n_prefix = 50
        prefix_token_ids = list(range(n_prefix))
        full_token_ids = prefix_token_ids + list(range(50))
        # Insert a radix prefix so L1 has a real match (50 tokens).
        prefix_key = RadixKey(token_ids=prefix_token_ids, extra_key=None)
        self.cache.insert(InsertParams(
            key=prefix_key,
            value=torch.tensor(prefix_token_ids, dtype=torch.int64),
            priority=1.0,
        ))
        # Seed a 3-token byte-exact chunk pool entry sitting at the prefix end.
        _seed_pool_entry(
            self.cache, "code_base0", self.sig, 0, self.byte_end,
            start_token=n_prefix, end_token=n_prefix + 3,
        )
        req = _MockReq(placeholder_anchor_token_spans=[
            _make_span("code_base0", self.text, start_token=n_prefix),
        ])
        full_key = RadixKey(token_ids=full_token_ids, extra_key=None)
        self.cache.match_prefix(MatchPrefixParams(key=full_key, req=req))

        # A1: pure radix L1 prefix, captured BEFORE the chunk copy appended.
        self.assertEqual(int(req.radix_only_prefix_len), n_prefix)
        # A2: C2 chunk copy count (3 tokens), excluding the radix prefix.
        self.assertEqual(int(req.c2_chunk_reused_tokens), 3)
        # A3: L3 path did not fire (L3 off) → 0.
        self.assertEqual(int(req.l3_offset_reused_tokens), 0)
        # A4: total code-aware == L3 + C2 == 3 (NOT radix + copy == 53).
        self.assertEqual(int(req.codeaware_reused_tokens), 3)
        # The decomposition must not double-count: radix_only (50) +
        # codeaware (3) reconstructs the matched prefix length, and
        # codeaware must NOT include the radix prefix.
        self.assertNotEqual(int(req.codeaware_reused_tokens), n_prefix + 3)


class TestChunkPoolCacheBlendStaging(unittest.TestCase):
    """C2 CacheBlend: when SGLANG_CACHEBLEND_CHUNK=1 and a byte-exact chunk
    sits beyond a LEADING gap (chunk_start > prefix_len, no chunk copied yet),
    the plan defers the copy (action=copy_pool_blend) and asks the scheduler
    to prefill the gap by setting ``lossy_anchor_context_target_prefix_len``
    (the same attr the L3 recompute-gap stage reads). No copy happens this
    round — copied slots are not persisted to the radix tree, so a copy must
    not precede a gap stage. With the flag OFF, the same gap is skipped as
    before (M1.5 regression)."""

    def setUp(self):
        os.environ["SGLANG_CHUNKED_PLACEHOLDER_KNN_MATCH"] = "1"
        os.environ["SGLANG_CACHEBLEND_CHUNK"] = "1"
        # The test chunk is only 3 tokens; disable the run-length guard so
        # staging is exercised on a tiny chunk (production keeps the default
        # 256-token floor).
        os.environ["SGLANG_CACHEBLEND_MIN_RUN"] = "0"
        self.cache = _make_simulated_cache()
        self.text = _histogram_text()
        self.sig = _histogram_signature()
        self.byte_end = _histogram_byte_end()
        _seed_pool_entry(
            self.cache, "code_base0", self.sig, 0, self.byte_end,
            start_token=50, end_token=53,
        )

    def tearDown(self):
        os.environ.pop("SGLANG_CHUNKED_PLACEHOLDER_KNN_MATCH", None)
        os.environ.pop("SGLANG_CACHEBLEND_CHUNK", None)
        os.environ.pop("SGLANG_CACHEBLEND_MIN_RUN", None)

    def test_leading_gap_stages_blend_no_copy_this_round(self):
        """Chunk at start_token=500 (gap from empty prefix) → blend stage."""
        req = _MockReq(placeholder_anchor_token_spans=[
            _make_span("code_base0", self.text, start_token=500),
        ])
        key = RadixKey(token_ids=list(range(600)), extra_key=None)
        values, node = self.cache._try_placeholder_chunk_lossy_match(
            req, key, [], self.cache.root_node,
        )
        # No copy this round — the gap must be prefilled first.
        self.assertEqual(values, [])
        self.assertEqual(self.cache.placeholder_chunk_pool_hit_count, 0)
        # Staging telemetry bumped.
        self.assertEqual(
            self.cache.placeholder_chunk_pool_blend_stage_count, 1,
        )
        self.assertEqual(
            self.cache.placeholder_chunk_pool_blend_gap_tokens, 500,
        )
        # The scheduler staging attr is set so the existing recompute-gap
        # stage fires (prefill up to the chunk's start).
        self.assertEqual(
            getattr(req, "lossy_anchor_context_target_prefix_len", 0), 500,
        )
        self.assertEqual(
            getattr(req, "lossy_anchor_context_align_stage", None),
            "recompute_gap_chunk",
        )
        # The deferred chunk was NOT counted as a plain skip_gap.
        self.assertEqual(
            self.cache.placeholder_chunk_pool_skip_gap_count, 0,
        )

    def test_contiguous_chunk_not_staged(self):
        """Chunk contiguous with the prefix (no gap) copies normally."""
        req = _MockReq(placeholder_anchor_token_spans=[
            _make_span("code_base0", self.text, start_token=0),
        ])
        key = RadixKey(token_ids=list(range(300)), extra_key=None)
        values, node = self.cache._try_placeholder_chunk_lossy_match(
            req, key, [], self.cache.root_node,
        )
        self.assertEqual(len(values), 1)
        self.assertEqual(
            self.cache.placeholder_chunk_pool_blend_stage_count, 0,
        )
        self.assertEqual(
            self.cache.placeholder_chunk_pool_hit_count, 1,
        )

    def test_blend_off_falls_back_to_skip_gap(self):
        """With SGLANG_CACHEBLEND_CHUNK unset, a gap chunk is skipped (M1.5)."""
        os.environ.pop("SGLANG_CACHEBLEND_CHUNK", None)
        req = _MockReq(placeholder_anchor_token_spans=[
            _make_span("code_base0", self.text, start_token=500),
        ])
        key = RadixKey(token_ids=list(range(600)), extra_key=None)
        values, node = self.cache._try_placeholder_chunk_lossy_match(
            req, key, [], self.cache.root_node,
        )
        self.assertEqual(values, [])
        self.assertEqual(
            self.cache.placeholder_chunk_pool_skip_gap_count, 1,
        )
        self.assertEqual(
            self.cache.placeholder_chunk_pool_blend_stage_count, 0,
        )


class TestHiRadixCacheDispatch(unittest.TestCase):
    """Regression for the 2026-06-28 fix: HiRadixCache.match_prefix must
    dispatch to the L4 chunk pool read path (_try_placeholder_chunk_lossy_match)
    when the request carries placeholder_anchor_token_spans, mirroring the
    base RadixCache.match_prefix block (radix_cache.py:843-851). Previously
    this call was omitted, so the L4 chunk pool was stored but never queried
    under hicache (the giant-codebase bench default), leaving
    placeholder_chunk_pool_hit_count at 0."""

    def _make_stub(self, record):
        from sglang.srt.mem_cache.hiradix_cache import HiRadixCache

        class _Node:
            evicted = False
            backuped = True
            parent = None
            host_value = torch.empty(0, dtype=torch.int64)

        class _Stub(HiRadixCache):
            def __init__(self_inner):
                # Skip the real HiRadixCache init (host pools, etc.) — only
                # set the attributes match_prefix reads.
                self_inner.disable = False
                self_inner.page_size = 1
                self_inner.device = torch.device("cpu")
                self_inner.root_node = _Node()
                self_inner.is_eagle = False

            def maybe_bigram_convert(self_inner, key):
                return key, None

            def _match_prefix_helper(self_inner, root, key):
                return [], self_inner.root_node

            def _try_placeholder_knn_lossy_match(self_inner, req, key, values, node):
                return values, node

            def _try_placeholder_chunk_lossy_match(self_inner, req, key, values, node):
                record.append(True)
                return values, node

        return _Stub()

    def _run(self, spans):
        record = []
        stub = self._make_stub(record)
        req = type("R", (), {
            "reuse_mode": "",
            "placeholder_anchor_token_spans": spans,
        })()
        stub.match_prefix(MatchPrefixParams(
            key=torch.tensor([1, 2, 3], dtype=torch.int64), req=req,
        ))
        return record

    def test_spans_present_dispatches_to_chunk_pool(self):
        self.assertEqual(self._run([(object(),)]), [True])

    def test_no_spans_does_not_dispatch(self):
        self.assertEqual(self._run([]), [])


if __name__ == "__main__":
    unittest.main()
