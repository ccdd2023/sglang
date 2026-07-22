from __future__ import annotations

import unittest

from benchmark.approx_kv.run_phase4_cacheblend_pressure import (
    build_target_segments,
    compute_filler_count,
    expected_selected_tokens,
    persistent_token_estimate,
    segment_chunks,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-c-test-cpu")


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
