from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from benchmark.approx_kv.run_phase4_cacheblend_pressure import (
    build_causal_registration_requests,
    build_target_segments,
    compute_filler_count,
    dense_persistent_token_estimate,
    expected_selected_tokens,
    persistent_token_estimate,
    register_source_segments,
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


class TestCausalRegistrationRequests(unittest.TestCase):
    def test_later_chunks_include_all_prior_body_tokens(self):
        requests = build_causal_registration_requests(
            header=[10, 11],
            chunks=[[20, 21], [22, 23], [24]],
            hash_prefix="fresh:",
            content_hash_base="artifact",
            sentinel_base=900,
        )

        self.assertEqual(requests[0]["input_ids"], [10, 11, 20, 21, 900])
        self.assertEqual(requests[1]["input_ids"], [10, 11, 20, 21, 22, 23, 901])
        self.assertEqual(requests[2]["input_ids"], [10, 11, 20, 21, 22, 23, 24, 902])

    def test_segments_use_true_absolute_body_offsets(self):
        requests = build_causal_registration_requests(
            header=[10, 11],
            chunks=[[20, 21], [22, 23], [24]],
            hash_prefix="raw:",
            content_hash_base="artifact",
            sentinel_base=900,
        )

        self.assertEqual(
            [request["segment"]["target_start"] for request in requests],
            [2, 4, 6],
        )
        self.assertEqual(
            [request["segment"]["length"] for request in requests],
            [2, 2, 1],
        )
        self.assertEqual(
            [request["segment"]["content_hash"] for request in requests],
            ["raw:artifact-chunk0", "raw:artifact-chunk1", "raw:artifact-chunk2"],
        )

    def test_registration_incrementally_materializes_then_copies_each_chunk(self):
        responses = [
            {"ttft_ms": 1.0, "cached_tokens": 0, "output_ids": [1]},
            {"ttft_ms": 2.0, "cached_tokens": 4, "output_ids": [1]},
            {"ttft_ms": 3.0, "cached_tokens": 4, "output_ids": [1]},
            {"ttft_ms": 4.0, "cached_tokens": 6, "output_ids": [1]},
        ]
        with mock.patch(
            "benchmark.approx_kv.run_phase4_cacheblend_pressure.request",
            side_effect=responses,
        ) as request_mock:
            result = register_source_segments(
                "http://127.0.0.1:30011",
                header=[10, 11],
                chunks=[[20, 21], [22, 23]],
                hash_prefix="raw:",
                content_hash_base="artifact",
                sentinel_base=900,
            )

        self.assertEqual(request_mock.call_count, 4)
        self.assertEqual(len(request_mock.call_args_list[0].args), 2)
        self.assertTrue(
            request_mock.call_args_list[0]
            .kwargs["extra_key"]
            .startswith("approx-kv-causal:")
        )
        self.assertEqual(
            request_mock.call_args_list[1].args[2]["operation"], "register"
        )
        self.assertEqual(result["total_ms"], 10.0)
        self.assertTrue(result["incremental_exact_materialization"])
        self.assertEqual(
            [row["materialize_cached_tokens"] for row in result["rows"]], [0, 4]
        )
        self.assertEqual(
            [row["register_cached_tokens"] for row in result["rows"]], [4, 6]
        )


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


# The contract's lossy body sizes: 512 is exactly one segment at the
# runner's default --segment-tokens=512, the other three each need
# multiple raw/fresh segments registered and recovered contiguously.
CONTRACT_BODY_TOKENS = (512, 768, 1024, 2048)
DEFAULT_SEGMENT_TOKENS = 512
CONTRACT_HEADER_TOKENS = (0, 32, 64, 128, 256)


class TestSegmentedRegistrationCoversFullBodyPerSetting(unittest.TestCase):
    """`segment_chunks` + `build_target_segments` are the two pure
    functions that decide, for every (header, body) contract setting,
    how a body > 512 tokens is split into <=512-token raw/fresh source
    segments and how those segments are expected to recover *contiguously*
    at the target. This is real per-setting coverage of every lossy body
    size in the contract (512/768/1024/2048), not just the single 512/488
    example the earlier tests exercised."""

    def test_chunk_count_and_lengths_match_expected_512_token_segments(self):
        expected = {
            512: [512],
            768: [512, 256],
            1024: [512, 512],
            2048: [512, 512, 512, 512],
        }
        for body_tokens in CONTRACT_BODY_TOKENS:
            with self.subTest(body_tokens=body_tokens):
                body = list(range(1_000, 1_000 + body_tokens))
                chunks = segment_chunks(body, DEFAULT_SEGMENT_TOKENS)
                self.assertEqual(
                    [len(chunk) for chunk in chunks], expected[body_tokens]
                )
                # Splitting must be lossless: concatenating every chunk
                # back together reproduces the original body exactly, in
                # order, with nothing dropped or duplicated at a chunk
                # boundary.
                self.assertEqual([token for chunk in chunks for token in chunk], body)

    def test_target_segments_are_gapless_and_cover_the_whole_body(self):
        for body_tokens in CONTRACT_BODY_TOKENS:
            for header_tokens in CONTRACT_HEADER_TOKENS:
                with self.subTest(body_tokens=body_tokens, header_tokens=header_tokens):
                    body = list(range(1_000, 1_000 + body_tokens))
                    chunks = segment_chunks(body, DEFAULT_SEGMENT_TOKENS)
                    chunk_lengths = [len(chunk) for chunk in chunks]
                    segments = build_target_segments(
                        chunk_lengths,
                        header_tokens=header_tokens,
                        hash_prefix="cacheblend-raw:",
                        content_hash_base="base",
                    )
                    # First segment starts exactly at the end of the
                    # (already exact-cache-seeded) header -- restoring
                    # the body picks up immediately where the target's
                    # own exact prefix leaves off.
                    self.assertEqual(segments[0]["target_start"], header_tokens)
                    # Every following segment starts exactly where the
                    # previous one ends: no gap (which `runtime.py`
                    # would refuse to bridge) and no overlap.
                    for previous, current in zip(segments, segments[1:]):
                        previous_end = previous["target_start"] + previous["length"]
                        self.assertEqual(current["target_start"], previous_end)
                    # The final segment ends exactly at header + body:
                    # the whole body is covered, continuously, by the
                    # segmented registration -- for every body size in
                    # the contract, not just 512.
                    last = segments[-1]
                    self.assertEqual(
                        last["target_start"] + last["length"],
                        header_tokens + body_tokens,
                    )
                    # Segment indices are 0..n-1 in order, so raw and
                    # fresh registration (which reuse this same chunk
                    # list, see TestRawAndFreshSegmentsShareChunkBoundaries
                    # below) address identical content-hash keys.
                    self.assertEqual(
                        [segment["content_hash"] for segment in segments],
                        [f"cacheblend-raw:base-chunk{i}" for i in range(len(segments))],
                    )


class TestRawAndFreshSegmentsShareChunkBoundaries(unittest.TestCase):
    """`run_round` must compute the segment boundaries exactly once
    (a single `chunks = segment_chunks(...)` call) and reuse that same
    `chunks`/`chunk_lengths` value for the raw registration, the fresh
    registration, and the target's restore segments. If raw and fresh
    were ever segmented independently (e.g. two separate
    `segment_chunks` calls), a future edit could silently desync their
    chunk boundaries -- registering under one set of content-hash keys
    while restore looks up another -- and every setting with body > 512
    would dense-fallback via `store_miss` instead of truly restoring."""

    def test_run_round_computes_chunk_boundaries_exactly_once(self):
        source = RUNNER_PATH.read_text()
        self.assertEqual(source.count("segment_chunks(body, args.segment_tokens)"), 1)

    def test_both_raw_and_fresh_registration_calls_reuse_the_same_chunks(self):
        source = RUNNER_PATH.read_text()
        run_round_source = source[
            source.index("def run_round") : source.index("\ndef main")
        ]
        # Both register_source_segments calls in run_round must pass the
        # single `chunks` variable computed above -- not a fresh,
        # independently-segmented list of their own.
        self.assertEqual(run_round_source.count("chunks=chunks,"), 2)

    def test_target_segments_are_built_from_the_same_chunk_lengths(self):
        source = RUNNER_PATH.read_text()
        self.assertIn("chunk_lengths = [len(chunk) for chunk in chunks]", source)
        self.assertIn("build_target_segments(\n                chunk_lengths,", source)


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
