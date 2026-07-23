from __future__ import annotations

import unittest
from pathlib import Path

from benchmark.approx_kv.run_phase4_cacheblend_pressure import (
    build_target_segments,
    compute_filler_count,
    dense_persistent_token_estimate,
    expected_selected_tokens,
    persistent_token_estimate,
    segment_chunks,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-c-test-cpu")

RUNNER_PATH = (
    Path(__file__).resolve().parents[4]
    / "benchmark/approx_kv/run_phase4_cacheblend_pressure.py"
)


class TestSegmentChunks(unittest.TestCase):
    def test_splits_into_at_most_segment_tokens_each(self):
        tokens = list(range(2048))
        chunks = segment_chunks(tokens, 512)
        self.assertEqual(len(chunks), 4)
        self.assertTrue(all(len(chunk) == 512 for chunk in chunks))
        self.assertEqual([token for chunk in chunks for token in chunk], tokens)

    def test_short_body_is_a_single_chunk(self):
        tokens = list(range(300))
        chunks = segment_chunks(tokens, 512)
        self.assertEqual(chunks, [tokens])

    def test_uneven_length_leaves_a_short_final_chunk(self):
        tokens = list(range(1000))
        chunks = segment_chunks(tokens, 512)
        self.assertEqual([len(c) for c in chunks], [512, 488])

    def test_rejects_non_positive_segment_tokens(self):
        with self.assertRaises(ValueError):
            segment_chunks([1, 2, 3], 0)


class TestBuildTargetSegments(unittest.TestCase):
    def test_segments_are_contiguous_starting_at_header_tokens(self):
        segments = build_target_segments(
            [512, 488],
            header_tokens=64,
            hash_prefix="cacheblend-raw:",
            content_hash_base="base",
        )
        self.assertEqual(
            segments,
            [
                {
                    "content_hash": "cacheblend-raw:base-chunk0",
                    "target_start": 64,
                    "length": 512,
                },
                {
                    "content_hash": "cacheblend-raw:base-chunk1",
                    "target_start": 576,
                    "length": 488,
                },
            ],
        )

    def test_zero_header_starts_at_position_zero(self):
        segments = build_target_segments(
            [256],
            header_tokens=0,
            hash_prefix="cacheblend-fresh:",
            content_hash_base="base",
        )
        self.assertEqual(segments[0]["target_start"], 0)


class TestExpectedSelectedTokens(unittest.TestCase):
    def test_matches_hkvd_rounding_rule(self):
        # Mirrors cacheblend.hkvd.select_hkvd_tokens's
        # `max(1, round(total * final_ratio))`.
        self.assertEqual(expected_selected_tokens(512, 0.05), round(512 * 0.05))
        self.assertEqual(expected_selected_tokens(512, 0.01), round(512 * 0.01))
        self.assertEqual(expected_selected_tokens(512, 0.15), round(512 * 0.15))
        self.assertEqual(expected_selected_tokens(512, 0.30), round(512 * 0.30))

    def test_floors_at_one_token_for_tiny_bodies(self):
        self.assertEqual(expected_selected_tokens(1, 0.01), 1)


class TestPersistentTokenEstimate(unittest.TestCase):
    def test_accounts_for_raw_and_fresh_plus_target(self):
        # raw(body) + fresh(body) + target(header+body+1)
        estimate = persistent_token_estimate(header_tokens=64, body_tokens=512)
        self.assertEqual(estimate, 512 + 512 + 64 + 512 + 1)

    def test_scales_linearly_with_body_tokens(self):
        small = persistent_token_estimate(header_tokens=0, body_tokens=512)
        large = persistent_token_estimate(header_tokens=0, body_tokens=1024)
        # Three body-sized contributions (raw + fresh + target).
        self.assertEqual(large - small, 3 * (1024 - 512))


class TestDensePersistentTokenEstimate(unittest.TestCase):
    def test_accounts_for_only_the_targets_own_commit(self):
        # Dense mode has exactly one body-sized contribution: the
        # target's own eventual commit (header + body + 1). No priming
        # term is added, because dense mode issues no body-chunk priming
        # request of its own (see the module and function docstrings).
        estimate = dense_persistent_token_estimate(header_tokens=64, body_tokens=512)
        self.assertEqual(estimate, 64 + 512 + 1)

    def test_scales_linearly_with_body_tokens(self):
        small = dense_persistent_token_estimate(header_tokens=0, body_tokens=512)
        large = dense_persistent_token_estimate(header_tokens=0, body_tokens=1024)
        # A single body-sized contribution (the target's own commit).
        self.assertEqual(large - small, 1024 - 512)

    def test_is_exactly_two_body_tokens_below_cacheblend_estimate(self):
        # CacheBlend adds two *extra* body-sized footprints beyond the
        # target's own commit (raw + fresh source registrations) that
        # dense mode has no equivalent of -- these two functions must
        # never converge to the same value for body_tokens > 0, or one
        # of them is silently missing/double-counting a real footprint.
        header_tokens, body_tokens = 64, 512
        dense = dense_persistent_token_estimate(header_tokens, body_tokens)
        cacheblend = persistent_token_estimate(header_tokens, body_tokens)
        self.assertEqual(cacheblend - dense, 2 * body_tokens)


class TestDenseModeIssuesNoBodyChunkPriming(unittest.TestCase):
    """Regression guard for the fairness bug where dense mode primed the
    body chunks as bare (headerless) ordinary dense requests before the
    filler loop -- a real ~body_tokens exact-cache entry that shared no
    prefix with (and was never reused by) the actual target request,
    silently inflating dense's true resident footprint above what
    `dense_persistent_token_estimate` declared and biasing `target_rho`
    unfairly tighter for dense than for cacheblend at the same nominal
    setting."""

    def test_runner_source_no_longer_primes_bare_body_chunks_for_dense(self):
        source = RUNNER_PATH.read_text()
        # The exact buggy call this bug used to send: a bare per-chunk
        # dense request with no header and no approx_kv metadata.
        self.assertNotIn("request(args.base_url, chunk + [900 + index])", source)

    def test_runner_uses_dense_specific_estimate_in_run_round(self):
        source = RUNNER_PATH.read_text()
        self.assertIn("dense_persistent_token_estimate(", source)


class TestComputeFillerCount(unittest.TestCase):
    def test_zero_when_persistent_tokens_already_exceed_target(self):
        count = compute_filler_count(
            capacity=13_130,
            target_rho=0.5,
            persistent_tokens=20_000,
            filler_tokens=736,
        )
        self.assertEqual(count, 0)

    def test_increases_with_target_rho(self):
        low = compute_filler_count(
            capacity=13_130,
            target_rho=1.0,
            persistent_tokens=2_000,
            filler_tokens=736,
        )
        high = compute_filler_count(
            capacity=13_130,
            target_rho=3.0,
            persistent_tokens=2_000,
            filler_tokens=736,
        )
        self.assertGreater(high, low)

    def test_rejects_non_positive_filler_tokens(self):
        with self.assertRaises(ValueError):
            compute_filler_count(
                capacity=13_130,
                target_rho=1.0,
                persistent_tokens=100,
                filler_tokens=0,
            )


if __name__ == "__main__":
    unittest.main()
