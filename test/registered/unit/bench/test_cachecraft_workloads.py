from __future__ import annotations

import unittest

from benchmark.approx_kv.cachecraft_workloads import (
    UNIFIED_CANONICAL_SEGMENT_TOKENS,
    UNIFIED_DEFAULT_FORMAL_REPEATS,
    UNIFIED_EXACT_HEADER_TOKENS,
    UNIFIED_LOSSY_BODY_TOKENS,
    UNIFIED_MEM_FRACTION_STATIC,
    UNIFIED_MIN_FORMAL_REPEATS,
    UNIFIED_TARGET_RHO,
    UNIFIED_WARMUP_PASSES,
    build_canonical_chunk,
    build_non_prefix_segmented_workload,
    segment_into_canonical_chunks,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-c-test-cpu")


class TestUnifiedContractConstants(unittest.TestCase):
    def test_matches_documented_phase4_unified_contract(self):
        self.assertEqual(UNIFIED_EXACT_HEADER_TOKENS, (0, 32, 64, 128, 256))
        self.assertEqual(UNIFIED_LOSSY_BODY_TOKENS, (512, 768, 1024, 2048))
        self.assertEqual(UNIFIED_TARGET_RHO, (0.9, 1.1, 1.5, 2.0, 3.0))
        self.assertAlmostEqual(UNIFIED_MEM_FRACTION_STATIC, 0.35)
        self.assertEqual(UNIFIED_CANONICAL_SEGMENT_TOKENS, 512)
        self.assertEqual(UNIFIED_WARMUP_PASSES, 1)
        self.assertEqual(UNIFIED_DEFAULT_FORMAL_REPEATS, 4)
        self.assertEqual(UNIFIED_MIN_FORMAL_REPEATS, 2)


class TestSegmentIntoCanonicalChunks(unittest.TestCase):
    def test_exactly_512_tokens_stays_a_single_segment(self):
        tokens = list(range(512))
        segments = segment_into_canonical_chunks(tokens)
        self.assertEqual(len(segments), 1)
        self.assertEqual(len(segments[0]), 512)

    def test_over_512_tokens_splits_into_bounded_segments(self):
        tokens = list(range(1024))
        segments = segment_into_canonical_chunks(tokens)
        self.assertEqual(len(segments), 2)
        self.assertTrue(all(len(segment) <= 512 for segment in segments))
        self.assertEqual(sum(len(segment) for segment in segments), 1024)

    def test_2048_tokens_splits_into_four_bounded_segments(self):
        tokens = list(range(2048))
        segments = segment_into_canonical_chunks(tokens)
        self.assertEqual(len(segments), 4)
        self.assertTrue(all(len(segment) <= 512 for segment in segments))

    def test_odd_length_last_segment_is_shorter(self):
        tokens = list(range(768))
        segments = segment_into_canonical_chunks(tokens)
        self.assertEqual([len(segment) for segment in segments], [512, 256])

    def test_rejects_non_positive_max_segment_tokens(self):
        with self.assertRaises(ValueError):
            segment_into_canonical_chunks([1, 2, 3], max_segment_tokens=0)


class TestBuildCanonicalChunk(unittest.TestCase):
    def test_segments_cover_full_chunk_contiguously_and_distinctly(self):
        chunk = build_canonical_chunk("chunk-A", list(range(2000, 2000 + 1200)))
        self.assertEqual(chunk.length, 1200)
        self.assertEqual(len(chunk.segments), 3)
        # segments are contiguous and reconstruct the original token ids
        reconstructed = []
        for segment in chunk.segments:
            reconstructed.extend(segment.token_ids)
        self.assertEqual(tuple(reconstructed), chunk.token_ids)
        # every segment has a distinct content hash
        hashes = {segment.content_hash for segment in chunk.segments}
        self.assertEqual(len(hashes), len(chunk.segments))
        for segment in chunk.segments:
            self.assertLessEqual(segment.length, UNIFIED_CANONICAL_SEGMENT_TOKENS)


class TestBuildNonPrefixSegmentedWorkload(unittest.TestCase):
    def test_builds_reordered_target_distinct_from_canonical_order(self):
        workload = build_non_prefix_segmented_workload(
            body_tokens=1024,
            header_tokens=64,
            num_chunks=3,
        )
        self.assertEqual(len(workload.chunks), 3)
        self.assertTrue(workload.is_reordered)
        self.assertEqual(
            set(workload.target_chunk_order), set(workload.canonical_chunk_order)
        )
        self.assertNotEqual(workload.target_chunk_order, workload.canonical_chunk_order)

    def test_target_token_ids_include_header_body_and_final_token(self):
        workload = build_non_prefix_segmented_workload(
            body_tokens=512,
            header_tokens=32,
            num_chunks=2,
            final_token_id=9999,
        )
        target = workload.target_token_ids
        self.assertEqual(target[:32], workload.header_token_ids)
        self.assertEqual(target[-1], 9999)
        # total length is header + full reordered body + 1 final token
        total_body = sum(chunk.length for chunk in workload.chunks)
        self.assertEqual(len(target), 32 + total_body + 1)

    def test_every_long_body_setting_in_unified_contract_builds_cleanly(self):
        for body_tokens in UNIFIED_LOSSY_BODY_TOKENS:
            for header_tokens in UNIFIED_EXACT_HEADER_TOKENS:
                workload = build_non_prefix_segmented_workload(
                    body_tokens=body_tokens,
                    header_tokens=header_tokens,
                )
                for chunk in workload.chunks:
                    self.assertTrue(
                        all(
                            segment.length <= UNIFIED_CANONICAL_SEGMENT_TOKENS
                            for segment in chunk.segments
                        )
                    )
                self.assertEqual(
                    sum(chunk.length for chunk in workload.chunks), body_tokens
                )
                self.assertEqual(len(workload.header_token_ids), header_tokens)

    def test_single_chunk_workload_still_reorders_via_swap_fallback(self):
        # num_chunks=1 has nothing to rotate; still must not silently claim
        # to be a non-prefix workload when it structurally cannot be one.
        workload = build_non_prefix_segmented_workload(
            body_tokens=512,
            num_chunks=1,
        )
        self.assertFalse(workload.is_reordered)
        self.assertEqual(workload.target_chunk_order, workload.canonical_chunk_order)

    def test_rejects_invalid_arguments(self):
        with self.assertRaises(ValueError):
            build_non_prefix_segmented_workload(body_tokens=0)
        with self.assertRaises(ValueError):
            build_non_prefix_segmented_workload(body_tokens=10, header_tokens=-1)
        with self.assertRaises(ValueError):
            build_non_prefix_segmented_workload(body_tokens=10, num_chunks=0)
        with self.assertRaises(ValueError):
            build_non_prefix_segmented_workload(body_tokens=2, num_chunks=5)

    def test_chunk_lookup_by_id_and_unknown_id_raises(self):
        workload = build_non_prefix_segmented_workload(body_tokens=512, num_chunks=2)
        first_id = workload.canonical_chunk_order[0]
        self.assertEqual(workload.chunk(first_id).chunk_id, first_id)
        with self.assertRaises(KeyError):
            workload.chunk("does-not-exist")


if __name__ == "__main__":
    unittest.main()
