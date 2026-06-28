"""
Unit tests for Direction #3 Phase D: simplified confidence policy.

Phase D adds per-decision telemetry counters to the chunk pool:
- placeholder_chunk_pool_skip_byte_drift_count
- placeholder_chunk_pool_skip_size_mismatch_count
- placeholder_chunk_pool_skip_no_entry_count
- placeholder_chunk_pool_skip_alloc_failed_count
- placeholder_chunk_pool_rope_ops_count
- placeholder_chunk_pool_total_tokens_reused
- placeholder_chunk_pool_total_tokens_dense

It also formalizes the "byte-exact = binary confidence" policy: every
ChunkDecision.confidence is either 1.0 (byte-exact hit) or 0.0 (any
skip). No fractional confidence in production.

Run:
    python -m pytest test/registered/unit/mem_cache/test_placeholder_chunk_pool_policy.py -v
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
    RadixCache,
    RadixKey,
    TreeNode,
)
from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=10, suite="stage-d-test-small-1-gpu")
register_amd_ci(est_time=10, suite="stage-d-test-small-1-gpu-amd")


@dataclass
class _MockReq:
    rid: str = "rid"
    origin_input_ids: list = None
    output_ids: list = None
    placeholder_anchor_token_spans: list = None


def _make_mock_allocator(alloc_return=None):
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
    from sglang.srt.mem_cache import radix_cache as rc

    TreeNode.counter = 0
    return rc.RadixCache.create_simulated(
        disable=False,
        mock_allocator=_make_mock_allocator(alloc_return),
        page_size=1,
    )


def _make_span(slot_id: str, text: str, start_token: int = 0) -> dict:
    return {
        "slot_id": slot_id,
        "start_token": start_token,
        "end_token": start_token + max(1, len(text) // 4),
        "text": text,
        "content_signature": "testsig1234",
        "label": slot_id,
    }


def _histogram_text() -> str:
    return "def histogram(data):\n    return 1\n"


def _histogram_signature() -> str:
    from sglang.srt.mem_cache.ast_chunker import ASTChunker

    return ASTChunker().chunk_text(_histogram_text())[0].signature


def _histogram_byte_end() -> int:
    from sglang.srt.mem_cache.ast_chunker import ASTChunker

    return ASTChunker().chunk_text(_histogram_text())[0].byte_end


def _seed_pool_entry(
    cache, slot_id: str, sig: str, byte_start: int, byte_end: int,
    start_token: int, end_token: int, n_tokens: int = 3,
) -> ChunkKVEntry:
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


class TestPhaseDTelemetry(unittest.TestCase):
    """Phase D telemetry: per-skip-reason counters + token totals."""

    def setUp(self):
        os.environ["SGLANG_CHUNKED_PLACEHOLDER_KNN_MATCH"] = "1"
        self.cache = _make_simulated_cache()
        self.text = _histogram_text()
        self.sig = _histogram_signature()
        self.byte_end = _histogram_byte_end()

    def tearDown(self):
        os.environ.pop("SGLANG_CHUNKED_PLACEHOLDER_KNN_MATCH", None)

    def test_byte_drift_increments_skip_counter(self):
        """Byte-drift skip bumps placeholder_chunk_pool_skip_byte_drift_count."""
        # Pool entry with mismatched byte_start (5) vs chunk's byte_start (0)
        _seed_pool_entry(
            self.cache, "code_base0", self.sig,
            byte_start=5, byte_end=10,
            start_token=50, end_token=53,
        )
        req = _MockReq(placeholder_anchor_token_spans=[
            _make_span("code_base0", self.text, start_token=0),
        ])
        key = RadixKey(token_ids=list(range(300)), extra_key=None)
        self.cache._try_placeholder_chunk_lossy_match(
            req, key, [], self.cache.root_node,
        )
        self.assertEqual(
            self.cache.placeholder_chunk_pool_skip_byte_drift_count, 1,
        )

    def test_total_tokens_reused_tracked(self):
        """Successful copy bumps placeholder_chunk_pool_total_tokens_reused."""
        _seed_pool_entry(
            self.cache, "code_base0", self.sig, 0, self.byte_end,
            start_token=50, end_token=53, n_tokens=3,
        )
        req = _MockReq(placeholder_anchor_token_spans=[
            _make_span("code_base0", self.text, start_token=0),
        ])
        key = RadixKey(token_ids=list(range(300)), extra_key=None)
        self.cache._try_placeholder_chunk_lossy_match(
            req, key, [], self.cache.root_node,
        )
        # Chunk has 3 tokens; tokens_reused should be 3
        self.assertEqual(
            self.cache.placeholder_chunk_pool_total_tokens_reused, 3,
        )

    def test_confidence_always_binary(self):
        """All ChunkDecision.confidence values are 0.0 or 1.0 — binary."""
        # Set up a span with two chunks: one matches, one doesn't.
        text_two_chunks = (
            "def alpha():\n    return 1\n"
            "def beta():\n    return 2\n"
        )
        from sglang.srt.mem_cache.ast_chunker import ASTChunker

        chunker = ASTChunker()
        chunks = chunker.chunk_text(text_two_chunks)
        # Seed pool only for the first chunk (alpha); leave beta un-seeded.
        if len(chunks) >= 2:
            alpha_sig = chunks[0].signature
            alpha_byte_end = chunks[0].byte_end
            _seed_pool_entry(
                self.cache, "code_base0", alpha_sig, 0, alpha_byte_end,
                start_token=50, end_token=53, n_tokens=3,
            )
        req = _MockReq(placeholder_anchor_token_spans=[
            _make_span("code_base0", text_two_chunks, start_token=0),
        ])
        key = RadixKey(token_ids=list(range(300)), extra_key=None)
        # Build plan directly so we can inspect all decisions.
        plan = self.cache._build_chunk_plan(req, req.placeholder_anchor_token_spans)
        self.assertGreater(len(plan.decisions), 0)
        for d in plan.decisions:
            self.assertIn(
                d.confidence, (0.0, 1.0),
                msg=f"decision {d.name} confidence={d.confidence} not binary",
            )


if __name__ == "__main__":
    unittest.main()
