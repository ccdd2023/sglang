from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import json
import math
import statistics
import tempfile
import time
import unittest
import unittest.mock
from pathlib import Path

from benchmark.approx_kv.metrics import idle_pool_invariant
from benchmark.approx_kv.run_phase4_cachetune_canary import (
    _PRESSURE_FILLER_MARKER_CODEPOINT_BLOCKS,
    _PRESSURE_FILLER_MARKER_CODEPOINT_POOL_SIZE,
    _PRESSURE_FILLER_MARKER_CODEPOINT_STRIDE,
    MAX_REASONABLE_EVICTION_PRESSURE_FILLER_COUNT,
    NON_PREFIX_HEAD_TOKENS,
    NON_PREFIX_SEED_SENTINEL_TOKENS,
    NON_PREFIX_TAIL_TOKENS,
    WARMUP_PASSES_PER_SETTING,
    NonPrefixSegmentWorkload,
    _build_seed_sentinel_ids_avoiding_body_first_token_collision,
    _deterministic_token_ids,
    _first_common_prefix_length,
    _non_negative_int_choice_list,
    _positive_float,
    _positive_float_choice_list,
    _positive_int,
    _positive_int_choice_list,
    _pressure_filler_head_literal_prefix,
    _pressure_filler_marker_codepoint_for_combined_index,
    _repeat_count,
    append_run_log,
    body_segments_for_hash,
    build_eviction_pressure_workloads,
    build_non_prefix_segment_workload,
    build_settings,
    build_sweep_point_result,
    capture_final_pool_reset_and_invariant,
    chunk_offsets,
    dense_generate_payload,
    ensure_target_head_resident,
    eviction_pressure_filler_count_for_rho,
    eviction_pressure_total_tokens,
    expected_repair_totals,
    flush_exact_radix_cache,
    main,
    observed_rho,
    register_body_chunks,
    register_eviction_pressure_objects,
    register_generate_payload,
    register_round_setup,
    require_cached_tokens,
    require_finished_by_length,
    reuse_generate_payload,
    run_exact_context_control_point,
    run_non_prefix_setting,
    run_target_reuse,
    timed_post,
    validate_pairwise_head_isolation,
)
from sglang.srt.mem_cache.approx_kv.request import (
    ApproxKVRequestMetadata,
    ApproxKVRequestOperation,
    ApproxKVRequestSegment,
)
from sglang.srt.mem_cache.cachetune.hardware_profile import (
    CacheTuneMode,
    RatioBounds,
    quantize_ratio,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-c-test-cpu")


class FakeTokenizer:
    """Content-sensitive fake: different text deterministically produces
    different token ids (unlike a word-count-only fake), so tests can
    assert on distinctness between differently-seeded pieces the same
    way a real BPE tokenizer would. Uses a stable hash (never Python's
    salted ``hash()``) so results are reproducible across processes."""

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [
            int(hashlib.sha256(word.encode("utf-8")).hexdigest()[:8], 16) % 100000
            for word in text.split()
        ]


class SparseFakeTokenizer:
    """Encodes very inefficiently (about one token per 200 characters),
    forcing ``_deterministic_token_ids``'s retry-doubling loop to
    iterate more than once to reach the requested count -- proving it
    converges rather than only working by chance on the first attempt."""

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return list(range(max(1, len(text) // 200)))


class NeverConvergingFakeTokenizer:
    """Always encodes to a single token regardless of input size -- used
    to prove ``_deterministic_token_ids``'s retry loop is bounded and
    raises cleanly instead of looping forever when convergence is
    genuinely impossible."""

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens, text
        return [0]


class CharLevelFakeTokenizer:
    """Content-sensitive, *character*-granularity fake (unlike
    ``FakeTokenizer``, which hashes whole whitespace-split words).

    A real BPE tokenizer splits ``workloads.deterministic_code``'s output
    into subword pieces, so two differently-seeded calls share several
    *identical leading tokens* for as long as the generated text's
    literal, seed-independent boilerplate (``"def synthetic_0_"``) lasts,
    then diverge only once the seed-dependent digest characters begin --
    this was exactly the bug ``_SOURCE_HEAD_LITERAL_PREFIX``/
    ``_TARGET_HEAD_LITERAL_PREFIX`` exist to fix (see the module
    docstring). ``FakeTokenizer``'s word-granularity hashing cannot
    reproduce this failure mode at all, because the diverging digest is
    embedded *inside* the same whitespace-delimited "word" as the literal
    prefix, so it never tests the fix this class is built to exercise.
    This fake instead hashes one token id per *character*, so shared
    leading characters deterministically produce shared leading token
    ids, and the first differing character deterministically produces
    the first differing token -- the same subword-boundary behavior that
    matters here, without depending on a real tokenizer.
    """

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [
            int(hashlib.sha256(char.encode("utf-8")).hexdigest()[:8], 16) % 100000
            for char in text
        ]


class AlwaysCollidingFillerMarkerFakeTokenizer:
    """Pathological fake used ONLY to prove
    ``_build_pressure_filler_workload_avoiding_first_token_collisions``'s
    bounded retry budget actually exhausts and raises ``RuntimeError``
    cleanly -- never hangs, never silently returns a colliding workload
    -- when a real vocabulary is genuinely too impoverished to keep every
    filler's target head pairwise first-token-distinct.

    Every ``_pressure_filler_head_literal_prefix`` candidate marker
    starts with a non-ASCII Unicode code point (see
    ``_PRESSURE_FILLER_MARKER_CODEPOINT_BLOCKS``), while every OTHER
    piece this module tokenizes (source/target head literal prefixes,
    ``workloads.deterministic_code``'s body/tail text) is plain ASCII.
    This fake exploits exactly that split: any text starting with a
    non-ASCII character always encodes to first token id 0, regardless
    of which code point it actually is, simulating a vocabulary that
    cannot distinguish ANY of these markers; everything else hashes
    normally (per whitespace-split word, like ``FakeTokenizer``) so
    non-filler content (and a filler's own body/tail) stays properly
    self-consistent and never trips
    ``NonPrefixSegmentWorkload.__post_init__``'s unrelated source/target
    common-prefix check.
    """

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        if text and ord(text[0]) > 127:
            rest = [
                int(hashlib.sha256(word.encode("utf-8")).hexdigest()[:8], 16) % 100000
                + 1
                for word in text.split()
            ] or [1]
            return [0] + rest
        return [
            int(hashlib.sha256(word.encode("utf-8")).hexdigest()[:8], 16) % 100000
            for word in text.split()
        ] or [0]


class TestRepeatCount(unittest.TestCase):
    """``--repeats`` must be rejected below 2: a single formal repeat can't
    be distinguished from measurement noise (this is the argparse-level
    enforcement of the mandated warmup+repeats measurement discipline)."""

    def test_accepts_minimum_valid_value(self):
        self.assertEqual(_repeat_count("2"), 2)

    def test_accepts_larger_values(self):
        self.assertEqual(_repeat_count("4"), 4)
        self.assertEqual(_repeat_count("100"), 100)

    def test_rejects_one(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            _repeat_count("1")

    def test_rejects_zero(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            _repeat_count("0")

    def test_rejects_negative(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            _repeat_count("-3")

    def test_rejects_non_integer_with_value_error(self):
        # int("abc") itself raises ValueError; this is intentionally not
        # caught/rewrapped, so argparse reports the underlying cause.
        with self.assertRaises(ValueError):
            _repeat_count("abc")


class TestNewChoiceListArgparseValidators(unittest.TestCase):
    """``--header-tokens-choices`` / ``--body-tokens-choices`` /
    ``--target-rho-choices`` must each reject invalid CLI input up
    front, at parse time -- never silently clamped, never deferred to a
    confusing failure deep inside the canary run."""

    def test_non_negative_int_choice_list_accepts_zero(self):
        self.assertEqual(_non_negative_int_choice_list("0"), (0,))

    def test_non_negative_int_choice_list_accepts_multiple_values(self):
        self.assertEqual(
            _non_negative_int_choice_list("0,32,64,128,256"),
            (0, 32, 64, 128, 256),
        )

    def test_non_negative_int_choice_list_rejects_negative(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            _non_negative_int_choice_list("0,-1,64")

    def test_non_negative_int_choice_list_rejects_empty(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            _non_negative_int_choice_list("")

    def test_non_negative_int_choice_list_skips_blank_entries(self):
        # A trailing comma or accidental double comma should not produce
        # a spurious empty-string int() conversion.
        self.assertEqual(_non_negative_int_choice_list("0,32,"), (0, 32))

    def test_positive_int_choice_list_accepts_multiple_values(self):
        self.assertEqual(
            _positive_int_choice_list("512,768,1024,2048"),
            (512, 768, 1024, 2048),
        )

    def test_positive_int_choice_list_rejects_zero(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            _positive_int_choice_list("512,0")

    def test_positive_int_choice_list_rejects_negative(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            _positive_int_choice_list("-512")

    def test_positive_int_choice_list_rejects_empty(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            _positive_int_choice_list("")

    def test_positive_float_choice_list_accepts_multiple_values(self):
        self.assertEqual(
            _positive_float_choice_list("0.9,1.1,1.5,2,3"),
            (0.9, 1.1, 1.5, 2.0, 3.0),
        )

    def test_positive_float_choice_list_rejects_zero(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            _positive_float_choice_list("0.9,0")

    def test_positive_float_choice_list_rejects_negative(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            _positive_float_choice_list("-0.5")

    def test_positive_float_choice_list_rejects_empty(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            _positive_float_choice_list("")


class TestPositiveIntAndFloatArgparseValidators(unittest.TestCase):
    """``_positive_int``/``_positive_float`` back
    ``--main-header-tokens``, ``--main-body-tokens``,
    ``--max-segment-chunk-tokens``, ``--pressure-filler-head-tokens``,
    ``--pressure-filler-body-tokens``, ``--main-target-rho``, and
    ``--length-sweep-rho``: all of these are structurally required to be
    positive (a zero or negative header/body/chunk/rho value is either
    meaningless or -- for header specifically -- must go through the
    dedicated header=0 control point in ``--header-tokens-choices``
    instead of ever being accepted as a single scalar main/chunk value)."""

    def test_positive_int_accepts_positive_value(self):
        self.assertEqual(_positive_int("64"), 64)

    def test_positive_int_rejects_zero(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            _positive_int("0")

    def test_positive_int_rejects_negative(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            _positive_int("-1")

    def test_positive_float_accepts_positive_value(self):
        self.assertEqual(_positive_float("1.5"), 1.5)

    def test_positive_float_rejects_zero(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            _positive_float("0")

    def test_positive_float_rejects_negative(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            _positive_float("-0.1")


class TestExpectedRepairTotals(unittest.TestCase):
    """Pure arithmetic that must scale strictly with the *formal* repeat
    count only -- this is the exact computation whose omission (using the
    per-call expectation without multiplying by repeats, or accidentally
    including the discarded warmup pass) was the telemetry half of the
    bug this fix addresses."""

    def test_compute_bound_like_case_scales_by_formal_repeats(self):
        totals = expected_repair_totals(
            repair_tokens_per_call=37,
            recomputed_layers_per_call=23,
            repeats=4,
        )
        self.assertEqual(
            totals,
            {
                "expect_precomputed_adapter": True,
                "expected_selected_tokens_total": 148,
                "expected_recomputed_layers_total": 92,
                "expected_precomputed_total": 4,
            },
        )

    def test_io_bound_like_case_with_zero_repair_tokens(self):
        # A fully transfer/IO-bound quantized ratio selects zero repair
        # tokens per call; the precomputed adapter is then never invoked
        # and every "total" must stay at zero regardless of repeat count.
        totals = expected_repair_totals(
            repair_tokens_per_call=0,
            recomputed_layers_per_call=23,
            repeats=4,
        )
        self.assertEqual(
            totals,
            {
                "expect_precomputed_adapter": False,
                "expected_selected_tokens_total": 0,
                "expected_recomputed_layers_total": 0,
                "expected_precomputed_total": 0,
            },
        )

    def test_matches_manual_multiplication_across_repeat_counts(self):
        for repeats in (1, 2, 4, 8):
            totals = expected_repair_totals(
                repair_tokens_per_call=13,
                recomputed_layers_per_call=5,
                repeats=repeats,
            )
            self.assertEqual(totals["expected_selected_tokens_total"], 13 * repeats)
            self.assertEqual(totals["expected_recomputed_layers_total"], 5 * repeats)
            self.assertEqual(totals["expected_precomputed_total"], repeats)

    def test_single_repeat_is_valid_pure_arithmetic(self):
        # The >=2 measurement-discipline floor is enforced at the CLI
        # layer (_repeat_count); this pure helper stays usable at
        # repeats=1 since the arithmetic itself is well-defined there.
        totals = expected_repair_totals(
            repair_tokens_per_call=10,
            recomputed_layers_per_call=2,
            repeats=1,
        )
        self.assertEqual(totals["expected_selected_tokens_total"], 10)
        self.assertEqual(totals["expected_precomputed_total"], 1)

    def test_rejects_zero_repeats(self):
        with self.assertRaises(ValueError):
            expected_repair_totals(
                repair_tokens_per_call=10,
                recomputed_layers_per_call=2,
                repeats=0,
            )

    def test_rejects_negative_repair_tokens(self):
        with self.assertRaises(ValueError):
            expected_repair_totals(
                repair_tokens_per_call=-1,
                recomputed_layers_per_call=2,
                repeats=4,
            )

    def test_rejects_negative_recomputed_layers(self):
        with self.assertRaises(ValueError):
            expected_repair_totals(
                repair_tokens_per_call=10,
                recomputed_layers_per_call=-1,
                repeats=4,
            )


class TestAppendRunLog(unittest.TestCase):
    """The shared ``--central-log`` JSONL writer: must append (never
    truncate), auto-create parent directories, and preserve every field
    of each lifecycle record exactly."""

    def test_appends_jsonl_lines_without_truncating(self):
        with tempfile.TemporaryDirectory(prefix="cachetune_canary_test_") as tmp:
            log_path = Path(tmp) / "central.jsonl"
            append_run_log(log_path, {"status": "running", "run_id": "r1"})
            append_run_log(log_path, {"status": "completed", "run_id": "r1"})

            lines = log_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(
                json.loads(lines[0]), {"status": "running", "run_id": "r1"}
            )
            self.assertEqual(
                json.loads(lines[1]), {"status": "completed", "run_id": "r1"}
            )

    def test_creates_parent_directories(self):
        with tempfile.TemporaryDirectory(prefix="cachetune_canary_test_") as tmp:
            log_path = Path(tmp) / "nested" / "sub" / "central.jsonl"
            self.assertFalse(log_path.parent.exists())
            append_run_log(log_path, {"status": "running"})
            self.assertTrue(log_path.exists())
            self.assertEqual(
                json.loads(log_path.read_text(encoding="utf-8").strip()),
                {"status": "running"},
            )

    def test_each_line_is_valid_standalone_json(self):
        with tempfile.TemporaryDirectory(prefix="cachetune_canary_test_") as tmp:
            log_path = Path(tmp) / "central.jsonl"
            for index in range(5):
                append_run_log(log_path, {"i": index})
            lines = log_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 5)
            for index, line in enumerate(lines):
                self.assertEqual(json.loads(line), {"i": index})


class TestBuildSettings(unittest.TestCase):
    """Every field the central log's ``settings`` block depends on must be
    carried through from parsed args, including the fixed measurement
    constants (warmup passes, scheduler/tier/prefetch)."""

    def _fake_args(self, **overrides) -> argparse.Namespace:
        defaults = dict(
            base_url="http://127.0.0.1:30000",
            model="Qwen/Qwen3-0.6B",
            model_revision="deadbeef",
            model_fingerprint="qwen3-0.6b-sm75",
            cache_dtype="fp16",
            mode="paper_mechanism",
            t_c_ms=1.0,
            t_i_ms=2.0,
            t_o_ms=0.5,
            first_recompute_layer=1,
            main_header_tokens=64,
            main_body_tokens=1024,
            main_target_rho=1.5,
            header_tokens_choices=(0, 32, 64, 128, 256),
            body_tokens_choices=(512, 768, 1024, 2048),
            target_rho_choices=(0.9, 1.1, 1.5, 2.0, 3.0),
            length_sweep_rho=1.5,
            max_segment_chunk_tokens=512,
            pressure_filler_head_tokens=NON_PREFIX_HEAD_TOKENS,
            pressure_filler_body_tokens=2048,
            repeats=4,
            runner_git_sha="abc123",
            image_digest="sha256:deadbeef",
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_carries_through_parsed_values(self):
        args = self._fake_args(repeats=6, mode="speed_only")
        settings = build_settings(args)
        self.assertEqual(settings["mode"], "speed_only")
        self.assertEqual(settings["repeats_per_setting"], 6)
        self.assertEqual(settings["t_c_ms"], 1.0)
        self.assertEqual(settings["t_i_ms"], 2.0)
        self.assertEqual(settings["t_o_ms"], 0.5)
        self.assertEqual(settings["runner_git_sha"], "abc123")
        self.assertEqual(settings["image_digest"], "sha256:deadbeef")

    def test_carries_through_unified_pressure_matrix_settings(self):
        args = self._fake_args(
            main_target_rho=2.0,
            header_tokens_choices=(0, 64),
            body_tokens_choices=(512,),
            target_rho_choices=(1.0, 2.0),
            length_sweep_rho=2.0,
            pressure_filler_head_tokens=40,
            pressure_filler_body_tokens=4096,
        )
        settings = build_settings(args)
        self.assertEqual(settings["main_target_rho"], 2.0)
        self.assertEqual(settings["header_tokens_choices"], (0, 64))
        self.assertEqual(settings["body_tokens_choices"], (512,))
        self.assertEqual(settings["target_rho_choices"], (1.0, 2.0))
        self.assertEqual(settings["length_sweep_rho"], 2.0)
        self.assertEqual(settings["pressure_filler_head_tokens"], 40)
        self.assertEqual(settings["pressure_filler_body_tokens"], 4096)

    def test_carries_through_main_shape_and_fixed_tail(self):
        # main_header_tokens/main_body_tokens come from parsed args; tail
        # is a fixed measurement-protocol constant, not CLI-controlled.
        args = self._fake_args(main_header_tokens=128, main_body_tokens=384)
        settings = build_settings(args)
        self.assertEqual(settings["main_header_tokens"], 128)
        self.assertEqual(settings["main_body_tokens"], 384)
        self.assertEqual(settings["tail_tokens"], NON_PREFIX_TAIL_TOKENS)

    def test_carries_through_max_segment_chunk_tokens(self):
        args = self._fake_args(max_segment_chunk_tokens=256)
        settings = build_settings(args)
        self.assertEqual(settings["max_segment_chunk_tokens"], 256)

    def test_fixed_measurement_protocol_fields(self):
        settings = build_settings(self._fake_args())
        self.assertEqual(
            settings["warmup_passes_per_setting"], WARMUP_PASSES_PER_SETTING
        )
        self.assertEqual(settings["scheduler"], "S0 LRU")
        self.assertEqual(settings["tier"], "GPU-only")
        self.assertFalse(settings["prefetch"])
        self.assertFalse(settings["accuracy_metric"])

    def test_isolated_across_two_different_args(self):
        # Two settings dicts built from independently-constructed args
        # must not alias or leak values -- guards against accidental
        # shared mutable state in build_settings.
        settings_a = build_settings(self._fake_args(mode="paper_mechanism"))
        settings_b = build_settings(self._fake_args(mode="speed_only"))
        self.assertEqual(settings_a["mode"], "paper_mechanism")
        self.assertEqual(settings_b["mode"], "speed_only")
        self.assertNotEqual(settings_a["mode"], settings_b["mode"])


class TestFlushExactRadixCache(unittest.TestCase):
    """The flush helper must hit exactly ``/flush_cache?timeout=30`` via
    an empty-body POST -- the same idiom already established by this
    directory's ``run_phase3_canary.py``/``run_phase2_matrix.py``."""

    def test_builds_expected_request(self):
        captured: dict[str, object] = {}

        class FakeResponse:
            def read(self):
                return b"Cache flushed.\n"

            def __enter__(self):
                return self

            def __exit__(self, *exc_info):
                return False

        def fake_urlopen(request, timeout=None):
            captured["url"] = request.full_url
            captured["method"] = request.get_method()
            captured["data"] = request.data
            captured["timeout"] = timeout
            return FakeResponse()

        with unittest.mock.patch("urllib.request.urlopen", fake_urlopen):
            result = flush_exact_radix_cache("http://127.0.0.1:30000")

        self.assertEqual(
            captured["url"], "http://127.0.0.1:30000/flush_cache?timeout=30"
        )
        self.assertEqual(captured["method"], "POST")
        self.assertEqual(captured["data"], b"")
        self.assertEqual(result, "Cache flushed.\n")


class TestDeterministicTokenIds(unittest.TestCase):
    """``_deterministic_token_ids`` is the offset-safety primitive the
    whole non-prefix workload redesign depends on: it must produce
    *exactly* the requested count, be fully reproducible, and never
    silently return a short/partial result or loop forever."""

    def test_produces_exact_requested_count(self):
        tokenizer = FakeTokenizer()
        for count in (1, 5, 34, 100):
            result = _deterministic_token_ids(tokenizer, "seed-a", count)
            self.assertEqual(len(result), count)

    def test_deterministic_same_seed_same_count(self):
        tokenizer = FakeTokenizer()
        first = _deterministic_token_ids(tokenizer, "cachetune-source-head", 34)
        second = _deterministic_token_ids(tokenizer, "cachetune-source-head", 34)
        self.assertEqual(first, second)

    def test_different_seeds_produce_different_ids(self):
        tokenizer = FakeTokenizer()
        source_head = _deterministic_token_ids(tokenizer, "seed-source-head", 34)
        target_head = _deterministic_token_ids(tokenizer, "seed-target-head", 34)
        self.assertNotEqual(source_head, target_head)

    def test_returns_tuple_of_int(self):
        tokenizer = FakeTokenizer()
        result = _deterministic_token_ids(tokenizer, "seed-b", 8)
        self.assertIsInstance(result, tuple)
        self.assertTrue(all(isinstance(token_id, int) for token_id in result))

    def test_rejects_zero_count(self):
        with self.assertRaises(ValueError):
            _deterministic_token_ids(FakeTokenizer(), "seed-c", 0)

    def test_rejects_negative_count(self):
        with self.assertRaises(ValueError):
            _deterministic_token_ids(FakeTokenizer(), "seed-d", -5)

    def test_converges_with_sparse_tokenizer(self):
        # SparseFakeTokenizer only yields ~1 token per 200 characters, so
        # the first attempt (a handful of blocks) cannot possibly reach
        # count=64; the retry-doubling loop must grow blocks until it
        # does, and the result must still be exactly 64 long.
        result = _deterministic_token_ids(SparseFakeTokenizer(), "seed-e", 64)
        self.assertEqual(len(result), 64)

    def test_raises_cleanly_when_convergence_is_impossible(self):
        # NeverConvergingFakeTokenizer always yields exactly one token,
        # so no amount of retry-doubling can ever reach count=10 -- this
        # must raise a clear, bounded RuntimeError, never hang.
        with self.assertRaises(RuntimeError):
            _deterministic_token_ids(NeverConvergingFakeTokenizer(), "seed-f", 10)

    def test_literal_prefix_defaults_to_no_marker(self):
        # Omitting literal_prefix must be identical to passing "" -- no
        # accidental behavior change for the existing shared-body/tail
        # call sites, which never pass this parameter.
        tokenizer = FakeTokenizer()
        without_kwarg = _deterministic_token_ids(tokenizer, "seed-g", 12)
        with_empty = _deterministic_token_ids(
            tokenizer, "seed-g", 12, literal_prefix=""
        )
        self.assertEqual(without_kwarg, with_empty)

    def test_literal_prefix_changes_result(self):
        tokenizer = FakeTokenizer()
        plain = _deterministic_token_ids(tokenizer, "seed-h", 12)
        marked = _deterministic_token_ids(
            tokenizer, "seed-h", 12, literal_prefix="MARKER\n"
        )
        self.assertNotEqual(plain, marked)
        self.assertEqual(len(marked), 12)

    def test_literal_prefix_forces_divergence_despite_shared_literal_body(self):
        # This is the exact failure mode _SOURCE_HEAD_LITERAL_PREFIX /
        # _TARGET_HEAD_LITERAL_PREFIX exist to fix: workloads.deterministic_code
        # always begins with the same seed-independent literal text
        # ("def synthetic_0_...") before any seed-dependent digest
        # character, so two different seeds alone are not sufficient --
        # CharLevelFakeTokenizer reproduces that subword-level sharing
        # (unlike FakeTokenizer's word-granularity hashing).
        tokenizer = CharLevelFakeTokenizer()
        unmarked_a = _deterministic_token_ids(tokenizer, "role-a-head", 20)
        unmarked_b = _deterministic_token_ids(tokenizer, "role-b-head", 20)
        # Sanity-check the fake is actually faithful to the real bug: two
        # different seeds without any literal_prefix must still share a
        # nonzero common prefix (proving this fake exercises the same
        # failure mode the real Qwen3-0.6B tokenizer exhibited).
        shared = 0
        for a, b in zip(unmarked_a, unmarked_b):
            if a != b:
                break
            shared += 1
        self.assertGreater(
            shared,
            0,
            "CharLevelFakeTokenizer did not reproduce the shared-literal-"
            "prefix failure mode -- this test fixture would not have "
            "caught the original bug",
        )

        marked_a = _deterministic_token_ids(
            tokenizer, "role-a-head", 20, literal_prefix="A_MARKER\n"
        )
        marked_b = _deterministic_token_ids(
            tokenizer, "role-b-head", 20, literal_prefix="B_MARKER\n"
        )
        self.assertNotEqual(marked_a[0], marked_b[0])


class TestNonPrefixSegmentWorkload(unittest.TestCase):
    """The dataclass whose self-validation is the last line of defense
    against ever again registering a "lossy" segment that is actually
    an exact-content transplant (the bug this whole fix addresses)."""

    def _workload(self, **overrides):
        defaults = dict(
            source_head_ids=(1, 2, 3),
            target_head_ids=(9, 8, 7),
            shared_body_ids=(4, 5, 6, 7),
            tail_ids=(99,),
            seed_sentinel_ids=(50,),
        )
        defaults.update(overrides)
        return NonPrefixSegmentWorkload(**defaults)

    def test_valid_construction_computes_properties(self):
        workload = self._workload()
        self.assertEqual(workload.body_tokens, 4)
        self.assertEqual(workload.body_start_in_source, 3)
        self.assertEqual(workload.body_start_in_target, 3)
        # Every prompt appends the same trailing tail_ids=(99,) -- see
        # NonPrefixSegmentWorkload's own docstring for why source/fresh
        # need it too, not just target (the real scheduler-crash bug
        # this dataclass's tail handling now fixes).
        self.assertEqual(workload.source_prompt_ids, (1, 2, 3, 4, 5, 6, 7, 99))
        self.assertEqual(workload.target_prompt_ids, (9, 8, 7, 4, 5, 6, 7, 99))
        self.assertEqual(workload.fresh_prompt_ids, (9, 8, 7, 4, 5, 6, 7, 99))
        self.assertEqual(workload.fresh_prompt_ids, workload.target_prompt_ids)
        # seed_prompt_ids is target_head_ids + seed_sentinel_ids -- never
        # target_head_ids alone (the real SM75 header-sweep bug this
        # property exists to fix; see its own docstring).
        self.assertEqual(workload.seed_prompt_ids, (9, 8, 7, 50))
        self.assertTrue(workload.body_source_context_differs_from_target)

    def test_shared_body_appears_identically_in_both_prompts(self):
        workload = self._workload()
        # Sliced to an explicit end (not left open-ended): all three
        # prompts now have a trailing tail_ids beyond the body, so an
        # open-ended slice from body_start would incorrectly include it.
        source_slice = workload.source_prompt_ids[
            workload.body_start_in_source : workload.body_start_in_source
            + workload.body_tokens
        ]
        target_slice = workload.target_prompt_ids[
            workload.body_start_in_target : workload.body_start_in_target
            + workload.body_tokens
        ]
        self.assertEqual(source_slice, workload.shared_body_ids)
        self.assertEqual(target_slice, workload.shared_body_ids)

    def test_rejects_identical_heads(self):
        # This is the core invariant this fix exists to enforce: an
        # identical head makes source/target KV indistinguishable.
        with self.assertRaises(ValueError):
            self._workload(target_head_ids=(1, 2, 3))

    def test_rejects_heads_sharing_a_common_exact_match_prefix(self):
        # Merely *unequal* heads are not enough: a live server's exact
        # radix tree does prefix matching, so a shared *leading* run of
        # tokens (even if the tails differ) would still report a nonzero
        # cached_tokens for the raw-segment register request immediately
        # after target_head_ids is seeded -- this is the actual bug this
        # check exists to catch (see the module docstring).
        with self.assertRaises(ValueError):
            self._workload(
                source_head_ids=(1, 2, 3, 100),
                target_head_ids=(1, 2, 3, 200),
            )

    def test_accepts_heads_sharing_no_common_prefix(self):
        # The positive case: heads that differ from the very first token
        # must construct without error.
        workload = self._workload(
            source_head_ids=(1, 2, 3),
            target_head_ids=(9, 2, 3),
        )
        self.assertTrue(workload.body_source_context_differs_from_target)

    def test_rejects_empty_source_head(self):
        with self.assertRaises(ValueError):
            self._workload(source_head_ids=())

    def test_rejects_empty_target_head(self):
        with self.assertRaises(ValueError):
            self._workload(target_head_ids=())

    def test_rejects_empty_shared_body(self):
        with self.assertRaises(ValueError):
            self._workload(shared_body_ids=())

    def test_rejects_empty_tail(self):
        with self.assertRaises(ValueError):
            self._workload(tail_ids=())

    def test_rejects_empty_seed_sentinel(self):
        with self.assertRaises(ValueError):
            self._workload(seed_sentinel_ids=())

    def test_rejects_seed_sentinel_colliding_with_body_first_token(self):
        # This is the exact real SM75 header-sweep bug this validation
        # exists to catch at construction time: a seed_sentinel_ids[0]
        # equal to shared_body_ids[0] would let the target-head seed
        # request's own exact-match tree entry spuriously extend into
        # the body on any later request matching target_head_ids +
        # shared_body_ids + ... (see seed_prompt_ids's own docstring).
        with self.assertRaises(ValueError):
            self._workload(shared_body_ids=(4, 5, 6, 7), seed_sentinel_ids=(4,))

    def test_accepts_seed_sentinel_distinct_from_body_first_token(self):
        workload = self._workload(shared_body_ids=(4, 5, 6, 7), seed_sentinel_ids=(50,))
        self.assertEqual(workload.seed_sentinel_ids, (50,))

    def test_is_frozen(self):
        workload = self._workload()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            workload.source_head_ids = (1,)  # type: ignore[misc]


class TestPressureFillerMarkerCodepointForCombinedIndex(unittest.TestCase):
    """The low-level combined-index -> Unicode code point mapper behind
    ``_pressure_filler_head_literal_prefix`` (see
    ``_PRESSURE_FILLER_MARKER_CODEPOINT_BLOCKS``).

    This replaces an original fixed 24-object single-letter-alphabet
    scheme (which raised ``ValueError`` once a setting's reverse-
    computed filler count exceeded 24 on a real GPU run) AND a since-
    abandoned width-based multi-letter-code redesign that was
    empirically found to be insufficient: two distinct, equal-length
    letter codes could still tokenize to the SAME first token under the
    real Qwen3-0.6B tokenizer's BPE merge behavior, and short synthetic
    ASCII/hex text was separately found to plateau at only ~183-400
    distinct achievable first tokens regardless of alphabet or length --
    far short of what hundreds/thousands of filler objects need. This
    code-point-pool-based scheme was verified, against the real
    Qwen3-0.6B tokenizer outside this test suite, to sustain >=8,000
    zero-collision fillers.
    """

    def test_is_deterministic(self):
        self.assertEqual(
            _pressure_filler_marker_codepoint_for_combined_index(5),
            _pressure_filler_marker_codepoint_for_combined_index(5),
        )

    def test_rejects_negative_combined_index(self):
        with self.assertRaises(ValueError):
            _pressure_filler_marker_codepoint_for_combined_index(-1)

    def test_every_codepoint_falls_within_a_declared_block(self):
        # Sampled, not exhaustive, across several full pool cycles.
        for combined_index in range(
            0, _PRESSURE_FILLER_MARKER_CODEPOINT_POOL_SIZE * 3, 37
        ):
            codepoint = _pressure_filler_marker_codepoint_for_combined_index(
                combined_index
            )
            in_some_block = any(
                low <= codepoint < high
                for low, high in _PRESSURE_FILLER_MARKER_CODEPOINT_BLOCKS
            )
            self.assertTrue(
                in_some_block,
                f"codepoint {codepoint:#x} for combined_index="
                f"{combined_index} must fall within one of "
                "_PRESSURE_FILLER_MARKER_CODEPOINT_BLOCKS",
            )

    def test_combined_indices_within_one_pool_cycle_are_all_distinct(self):
        # The stride is coprime with the pool size (see
        # test_stride_is_coprime_with_pool_size below), so this must be
        # a true bijection over one full pool-sized cycle -- this is the
        # exact property build_eviction_pressure_workloads's retry
        # search depends on to reach thousands of fillers.
        codepoints = [
            _pressure_filler_marker_codepoint_for_combined_index(i)
            for i in range(_PRESSURE_FILLER_MARKER_CODEPOINT_POOL_SIZE)
        ]
        self.assertEqual(len(codepoints), len(set(codepoints)))

    def test_wraps_around_by_modulo_beyond_one_pool_cycle(self):
        # Total (defined for any non-negative integer) rather than
        # raising once the immediate pool is exhausted -- the bounded
        # retry budget in _build_pressure_filler_workload_avoiding_
        # first_token_collisions, not this function, is what actually
        # caps how many attempts get made in practice.
        pool_size = _PRESSURE_FILLER_MARKER_CODEPOINT_POOL_SIZE
        self.assertEqual(
            _pressure_filler_marker_codepoint_for_combined_index(0),
            _pressure_filler_marker_codepoint_for_combined_index(pool_size),
        )
        self.assertEqual(
            _pressure_filler_marker_codepoint_for_combined_index(41),
            _pressure_filler_marker_codepoint_for_combined_index(pool_size + 41),
        )

    def test_stride_is_coprime_with_pool_size(self):
        # This is the mathematical property the whole scheme depends on:
        # multiplying by the stride and reducing modulo the pool size is
        # only a true bijection (every combined_index in one cycle maps
        # to a DIFFERENT code point) if the two are coprime. If a future
        # edit to _PRESSURE_FILLER_MARKER_CODEPOINT_BLOCKS ever changes
        # the pool size to something sharing a common factor with the
        # stride, this test must catch it up front, before
        # test_combined_indices_within_one_pool_cycle_are_all_distinct
        # would otherwise start silently failing in a confusing way.
        self.assertEqual(
            math.gcd(
                _PRESSURE_FILLER_MARKER_CODEPOINT_STRIDE,
                _PRESSURE_FILLER_MARKER_CODEPOINT_POOL_SIZE,
            ),
            1,
        )


class TestPressureFillerHeadLiteralPrefix(unittest.TestCase):
    """Every eviction-pressure filler object needs its own CANDIDATE
    target-head literal-prefix marker (see
    ``_PRESSURE_FILLER_MARKER_CODEPOINT_BLOCKS``); this generator must be
    deterministic and its text's leading character must be exactly the
    code point ``_pressure_filler_marker_codepoint_for_combined_index``
    picked for that same combined index. A distinct leading code point
    is only an empirically-good STARTING candidate, never proof, of
    first-TOKEN distinctness on its own -- ``build_eviction_pressure_
    workloads`` (via ``_build_pressure_filler_workload_avoiding_first_
    token_collisions``) is what actually guarantees that, by validating
    each candidate against a real tokenizer and retrying on collision
    (see ``TestBuildEvictionPressureWorkloads`` for that guarantee's own
    tests)."""

    def test_is_deterministic(self):
        self.assertEqual(
            _pressure_filler_head_literal_prefix(3),
            _pressure_filler_head_literal_prefix(3),
        )

    def test_leading_character_matches_the_codepoint_mapper(self):
        for combined_index in (0, 1, 2, 1000, 5000):
            marker = _pressure_filler_head_literal_prefix(combined_index)
            expected_codepoint = _pressure_filler_marker_codepoint_for_combined_index(
                combined_index
            )
            self.assertEqual(ord(marker[0]), expected_codepoint)

    def test_two_different_combined_indices_produce_different_markers(self):
        self.assertNotEqual(
            _pressure_filler_head_literal_prefix(0),
            _pressure_filler_head_literal_prefix(1),
        )

    def test_rejects_negative_combined_index(self):
        with self.assertRaises(ValueError):
            _pressure_filler_head_literal_prefix(-1)

    def test_64_candidate_markers_are_all_distinct(self):
        # The exact real-GPU-reported scenario: 64 filler objects, well
        # past the original 24-marker cap.
        markers = [_pressure_filler_head_literal_prefix(i) for i in range(64)]
        self.assertEqual(len(markers), len(set(markers)))


class TestBuildNonPrefixSegmentWorkload(unittest.TestCase):
    """The builder that assembles a validated ``NonPrefixSegmentWorkload``
    from four independently tokenized pieces -- exercised with a real
    (fake) tokenizer end to end, not just the dataclass's own checks."""

    def test_produces_correct_lengths(self):
        workload = build_non_prefix_segment_workload(
            FakeTokenizer(),
            body_tokens=128,
            head_tokens=34,
            tail_tokens=1,
            salt="unit-test-main",
        )
        self.assertEqual(workload.body_tokens, 128)
        self.assertEqual(len(workload.source_head_ids), 34)
        self.assertEqual(len(workload.target_head_ids), 34)
        self.assertEqual(len(workload.tail_ids), 1)

    def test_source_head_differs_from_target_head(self):
        workload = build_non_prefix_segment_workload(
            FakeTokenizer(),
            body_tokens=16,
            head_tokens=34,
            tail_tokens=1,
            salt="unit-test-heads",
        )
        self.assertNotEqual(workload.source_head_ids, workload.target_head_ids)
        self.assertTrue(workload.body_source_context_differs_from_target)

    def test_source_and_target_heads_share_no_common_exact_match_prefix(self):
        # Regression test for the bug _SOURCE_HEAD_LITERAL_PREFIX /
        # _TARGET_HEAD_LITERAL_PREFIX exist to fix: under a tokenizer
        # that reproduces real BPE's subword-sharing behavior
        # (CharLevelFakeTokenizer -- FakeTokenizer's word-granularity
        # hashing cannot expose this), source_head_ids and
        # target_head_ids must not merely differ overall but must
        # diverge starting at token 0. This was independently confirmed
        # against Qwen3-0.6B's real tokenizer (4-6 shared leading tokens
        # before this fix, 0 after).
        tokenizer = CharLevelFakeTokenizer()
        for salt in ("phase4-r5-cachetune-main", "phase4-r5-cachetune-sweep-128"):
            workload = build_non_prefix_segment_workload(
                tokenizer,
                body_tokens=64,
                head_tokens=NON_PREFIX_HEAD_TOKENS,
                tail_tokens=NON_PREFIX_TAIL_TOKENS,
                salt=salt,
            )
            self.assertNotEqual(
                workload.source_head_ids[0],
                workload.target_head_ids[0],
                f"salt={salt!r}: source_head_ids and target_head_ids share "
                "a first token -- would make register_raw's cached_tokens "
                "nonzero immediately after seeding target_head_ids",
            )
            # NonPrefixSegmentWorkload.__post_init__ already enforces the
            # full zero-common-prefix invariant at construction time; a
            # successful return here is itself proof it held.
            self.assertTrue(workload.body_source_context_differs_from_target)

    def test_same_salt_is_deterministic(self):
        first = build_non_prefix_segment_workload(
            FakeTokenizer(),
            body_tokens=32,
            head_tokens=34,
            tail_tokens=1,
            salt="unit-test-repro",
        )
        second = build_non_prefix_segment_workload(
            FakeTokenizer(),
            body_tokens=32,
            head_tokens=34,
            tail_tokens=1,
            salt="unit-test-repro",
        )
        self.assertEqual(first, second)

    def test_different_salts_isolate_heads_and_bodies(self):
        # Distinct settings (main vs each length-sweep point) must never
        # accidentally share head/body content -- this is what prevents
        # one setting's seeded exact-cache head from spuriously
        # exact-matching another setting's requests.
        main = build_non_prefix_segment_workload(
            FakeTokenizer(),
            body_tokens=256,
            head_tokens=34,
            tail_tokens=1,
            salt="phase4-r5-cachetune-main",
        )
        sweep = build_non_prefix_segment_workload(
            FakeTokenizer(),
            body_tokens=128,
            head_tokens=34,
            tail_tokens=1,
            salt="phase4-r5-cachetune-sweep-128",
        )
        self.assertNotEqual(main.target_head_ids, sweep.target_head_ids)
        self.assertNotEqual(main.source_head_ids, sweep.source_head_ids)
        self.assertNotEqual(main.shared_body_ids, sweep.shared_body_ids)

    def test_body_tokens_sweep_values_each_produce_isolated_workloads(self):
        # Mirrors the actual body-token sweep values this canary uses.
        workloads = [
            build_non_prefix_segment_workload(
                FakeTokenizer(),
                body_tokens=body_tokens,
                head_tokens=34,
                tail_tokens=1,
                salt=f"phase4-r5-cachetune-sweep-{body_tokens}",
            )
            for body_tokens in (128, 256, 512)
        ]
        target_heads = [workload.target_head_ids for workload in workloads]
        self.assertEqual(len(target_heads), len(set(target_heads)))

    def test_default_head_literal_prefixes_match_omitting_the_kwargs(self):
        # Every existing caller (main setting, every length-sweep point)
        # must see byte-for-byte identical behavior after this function
        # grew its two new optional keyword parameters.
        with_defaults_explicit = build_non_prefix_segment_workload(
            FakeTokenizer(),
            body_tokens=32,
            head_tokens=34,
            tail_tokens=1,
            salt="unit-test-default-equivalence",
            source_head_literal_prefix="SOURCE_HEAD_MARKER_TEXT\n",
            target_head_literal_prefix="TARGET_HEAD_MARKER_TEXT\n",
        )
        with_defaults_omitted = build_non_prefix_segment_workload(
            FakeTokenizer(),
            body_tokens=32,
            head_tokens=34,
            tail_tokens=1,
            salt="unit-test-default-equivalence",
        )
        self.assertEqual(with_defaults_explicit, with_defaults_omitted)

    def test_custom_target_head_literal_prefix_changes_only_target_head(self):
        salt = "unit-test-custom-prefix"
        common_kwargs = dict(
            body_tokens=32,
            head_tokens=34,
            tail_tokens=1,
            salt=salt,
        )
        default_prefix_workload = build_non_prefix_segment_workload(
            FakeTokenizer(), **common_kwargs
        )
        custom_prefix_workload = build_non_prefix_segment_workload(
            FakeTokenizer(),
            target_head_literal_prefix=_pressure_filler_head_literal_prefix(0),
            **common_kwargs,
        )
        self.assertNotEqual(
            default_prefix_workload.target_head_ids,
            custom_prefix_workload.target_head_ids,
        )
        # Everything else (built from the same salt, unaffected pieces)
        # must be untouched by only overriding the target-head marker.
        self.assertEqual(
            default_prefix_workload.source_head_ids,
            custom_prefix_workload.source_head_ids,
        )
        self.assertEqual(
            default_prefix_workload.shared_body_ids,
            custom_prefix_workload.shared_body_ids,
        )
        self.assertEqual(
            default_prefix_workload.tail_ids, custom_prefix_workload.tail_ids
        )

    def test_two_pressure_filler_target_head_prefixes_diverge_at_token_zero(self):
        # Regression test for the exact collision risk
        # _pressure_filler_head_literal_prefix/
        # validate_pairwise_head_isolation exist to guard against: under
        # a tokenizer that reproduces real BPE's subword-sharing
        # behavior, two fillers built from the *same* salt template but
        # different indices must still diverge starting at token 0 (see
        # TestBuildNonPrefixSegmentWorkload
        # .test_source_and_target_heads_share_no_common_exact_match_prefix
        # for the analogous source/target-head regression test).
        tokenizer = CharLevelFakeTokenizer()
        filler_0 = build_non_prefix_segment_workload(
            tokenizer,
            body_tokens=16,
            head_tokens=NON_PREFIX_HEAD_TOKENS,
            tail_tokens=NON_PREFIX_TAIL_TOKENS,
            salt="phase4-r5-pressure-filler-0",
            target_head_literal_prefix=_pressure_filler_head_literal_prefix(0),
        )
        filler_1 = build_non_prefix_segment_workload(
            tokenizer,
            body_tokens=16,
            head_tokens=NON_PREFIX_HEAD_TOKENS,
            tail_tokens=NON_PREFIX_TAIL_TOKENS,
            salt="phase4-r5-pressure-filler-1",
            target_head_literal_prefix=_pressure_filler_head_literal_prefix(1),
        )
        self.assertNotEqual(
            filler_0.target_head_ids[0],
            filler_1.target_head_ids[0],
            "two pressure-filler target heads share a first token -- "
            "would let a later filler's seed request silently observe a "
            "nonzero cached_tokens match against an earlier filler's "
            "head already sitting in the exact radix tree",
        )

    def test_seed_sentinel_first_token_differs_from_body_first_token(self):
        # The core regression test for the real SM75 header=32 bug: the
        # built workload's own seed_sentinel_ids[0] must always differ
        # from that SAME workload's own shared_body_ids[0] (see
        # NonPrefixSegmentWorkload.seed_prompt_ids for the full
        # mechanism this protects).
        workload = build_non_prefix_segment_workload(
            FakeTokenizer(),
            body_tokens=64,
            head_tokens=32,
            tail_tokens=1,
            salt="unit-test-seed-sentinel-header-32",
        )
        self.assertNotEqual(workload.seed_sentinel_ids[0], workload.shared_body_ids[0])
        self.assertEqual(
            workload.seed_prompt_ids,
            workload.target_head_ids + workload.seed_sentinel_ids,
        )
        self.assertEqual(
            len(workload.seed_prompt_ids), 32 + NON_PREFIX_SEED_SENTINEL_TOKENS
        )

    def test_seed_sentinel_distinct_from_body_first_token_across_header_and_body_lengths(
        self,
    ):
        # Multi-header/multi-body real-token coverage (using
        # CharLevelFakeTokenizer, the closest fake to real BPE subword
        # behavior this suite has -- see that class's own docstring):
        # every (header, body) combination this canary's own sweeps
        # actually exercise must independently satisfy the same
        # invariant, not just one hand-picked shape.
        tokenizer = CharLevelFakeTokenizer()
        for header_tokens, body_tokens in (
            (0 + 1, 16),  # smallest realistic header (>=1 token)
            (32, 512),  # the exact real-bug header/body shape
            (34, 128),  # this script's own default head length
            (64, 1024),
            (128, 2048),
            (256, 768),
        ):
            workload = build_non_prefix_segment_workload(
                tokenizer,
                body_tokens=body_tokens,
                head_tokens=header_tokens,
                tail_tokens=NON_PREFIX_TAIL_TOKENS,
                salt=f"unit-test-seed-sentinel-h{header_tokens}-b{body_tokens}",
            )
            self.assertNotEqual(
                workload.seed_sentinel_ids[0],
                workload.shared_body_ids[0],
                f"header_tokens={header_tokens}, body_tokens={body_tokens}: "
                "seed_sentinel_ids[0] collides with shared_body_ids[0]",
            )
            self.assertEqual(len(workload.target_head_ids), header_tokens)
            self.assertEqual(len(workload.shared_body_ids), body_tokens)

    def test_seed_sentinel_is_deterministic_for_the_same_salt(self):
        first = build_non_prefix_segment_workload(
            FakeTokenizer(),
            body_tokens=32,
            head_tokens=34,
            tail_tokens=1,
            salt="unit-test-seed-sentinel-repro",
        )
        second = build_non_prefix_segment_workload(
            FakeTokenizer(),
            body_tokens=32,
            head_tokens=34,
            tail_tokens=1,
            salt="unit-test-seed-sentinel-repro",
        )
        self.assertEqual(first.seed_sentinel_ids, second.seed_sentinel_ids)


class TestBuildSeedSentinelIdsAvoidingBodyFirstTokenCollision(unittest.TestCase):
    """The retry helper behind ``NonPrefixSegmentWorkload.seed_sentinel_ids``
    -- the real SM75 header=32 bug fix (see module docstring / that
    property's own docstring): a bare ``target_head_ids``-only seed
    request's own single generated token could coincidentally equal
    ``shared_body_ids[0]``, silently extending the exact radix tree's
    matched boundary for that head by one token."""

    def test_returns_a_sentinel_distinct_from_body_first_token(self):
        sentinel = _build_seed_sentinel_ids_avoiding_body_first_token_collision(
            FakeTokenizer(),
            salt="unit-test-sentinel-helper",
            shared_body_ids=(4, 5, 6, 7),
        )
        self.assertNotEqual(sentinel[0], 4)

    def test_produces_exactly_non_prefix_seed_sentinel_tokens_count(self):
        sentinel = _build_seed_sentinel_ids_avoiding_body_first_token_collision(
            FakeTokenizer(),
            salt="unit-test-sentinel-length",
            shared_body_ids=(1, 2, 3),
        )
        self.assertEqual(len(sentinel), NON_PREFIX_SEED_SENTINEL_TOKENS)

    def test_is_deterministic_for_the_same_salt_and_body(self):
        first = _build_seed_sentinel_ids_avoiding_body_first_token_collision(
            FakeTokenizer(),
            salt="unit-test-sentinel-repro",
            shared_body_ids=(10, 20, 30),
        )
        second = _build_seed_sentinel_ids_avoiding_body_first_token_collision(
            FakeTokenizer(),
            salt="unit-test-sentinel-repro",
            shared_body_ids=(10, 20, 30),
        )
        self.assertEqual(first, second)

    def test_retries_actually_vary_the_candidate_first_token(self):
        # Direct regression test for a real design flaw caught and fixed
        # before this ever reached production: varying only the `seed`
        # string passed to _deterministic_token_ids, while reusing ONE
        # fixed literal_prefix across every attempt, would NOT let
        # retries reach a different first token at all (that function
        # always prepends literal_prefix verbatim, ahead of the
        # seed-dependent digest text, so a word-granularity tokenizer's
        # first token is fixed by the literal text alone). This proves
        # at least two distinct first-token candidates are reachable
        # across the first several attempts, confirming the per-attempt
        # marker actually varies (see _pressure_filler_marker_codepoint_
        # for_combined_index reuse in the production helper).
        tokenizer = FakeTokenizer()
        candidates = set()
        for attempt in range(8):
            codepoint = _pressure_filler_marker_codepoint_for_combined_index(attempt)
            marker = chr(codepoint) + "SEED_SENTINEL_MARKER_TEXT\n"
            candidate = _deterministic_token_ids(
                tokenizer,
                f"unit-test-vary-{attempt}",
                NON_PREFIX_SEED_SENTINEL_TOKENS,
                literal_prefix=marker,
            )
            candidates.add(candidate[0])
        self.assertGreater(
            len(candidates),
            1,
            "every retry attempt produced the identical first token -- "
            "the per-attempt marker is not actually varying",
        )

    def test_raises_when_every_attempt_collides_against_a_pathological_tokenizer(self):
        # AlwaysCollidingFillerMarkerFakeTokenizer maps ANY text whose
        # first character is non-ASCII (exactly what every one of this
        # helper's own candidate markers is, by construction -- see
        # _PRESSURE_FILLER_MARKER_CODEPOINT_BLOCKS) to first token id 0,
        # regardless of which specific code point it is -- simulating a
        # vocabulary that cannot distinguish any candidate marker at
        # all. With shared_body_ids[0] == 0, every attempt collides, and
        # this must raise RuntimeError -- never loop forever, never
        # silently return a colliding sentinel.
        with self.assertRaises(RuntimeError) as ctx:
            _build_seed_sentinel_ids_avoiding_body_first_token_collision(
                AlwaysCollidingFillerMarkerFakeTokenizer(),
                salt="unit-test-sentinel-exhaustion",
                shared_body_ids=(0, 1, 2),
            )
        self.assertIn("unit-test-sentinel-exhaustion", str(ctx.exception))


class TestBuildEvictionPressureWorkloads(unittest.TestCase):
    """Builds the N distinct filler workloads a setting's eviction-
    pressure phase materializes before its own measurement -- every
    filler must be mutually content-isolated AND mutually head-isolated,
    never just individually valid."""

    def test_produces_requested_object_count(self):
        workloads = build_eviction_pressure_workloads(
            FakeTokenizer(),
            object_count=4,
            body_tokens=64,
            head_tokens=34,
            tail_tokens=1,
            salt_prefix="phase4-r5-pressure",
        )
        self.assertEqual(len(workloads), 4)

    def test_every_workload_has_the_requested_body_length(self):
        workloads = build_eviction_pressure_workloads(
            FakeTokenizer(),
            object_count=3,
            body_tokens=128,
            head_tokens=34,
            tail_tokens=1,
            salt_prefix="phase4-r5-pressure",
        )
        for workload in workloads:
            self.assertEqual(workload.body_tokens, 128)

    def test_rejects_zero_object_count(self):
        with self.assertRaises(ValueError):
            build_eviction_pressure_workloads(
                FakeTokenizer(),
                object_count=0,
                body_tokens=64,
                head_tokens=34,
                tail_tokens=1,
                salt_prefix="phase4-r5-pressure",
            )

    def test_rejects_negative_object_count(self):
        with self.assertRaises(ValueError):
            build_eviction_pressure_workloads(
                FakeTokenizer(),
                object_count=-1,
                body_tokens=64,
                head_tokens=34,
                tail_tokens=1,
                salt_prefix="phase4-r5-pressure",
            )

    def test_fillers_have_mutually_distinct_bodies_and_tails(self):
        workloads = build_eviction_pressure_workloads(
            FakeTokenizer(),
            object_count=4,
            body_tokens=64,
            head_tokens=34,
            tail_tokens=1,
            salt_prefix="phase4-r5-pressure",
        )
        bodies = [workload.shared_body_ids for workload in workloads]
        self.assertEqual(len(bodies), len(set(bodies)))

    def test_fillers_have_mutually_distinct_target_heads(self):
        workloads = build_eviction_pressure_workloads(
            FakeTokenizer(),
            object_count=4,
            body_tokens=64,
            head_tokens=34,
            tail_tokens=1,
            salt_prefix="phase4-r5-pressure",
        )
        target_heads = [workload.target_head_ids for workload in workloads]
        self.assertEqual(len(target_heads), len(set(target_heads)))

    def test_fillers_pass_pairwise_head_isolation_against_a_bpe_like_tokenizer(self):
        # End-to-end proof (with the finer-granularity fake, unlike the
        # word-level FakeTokenizer used by the other tests in this
        # class) that build_eviction_pressure_workloads's own output is
        # self-consistent from the moment it is constructed: every
        # filler's target head must already satisfy
        # validate_pairwise_head_isolation with zero collisions, before
        # this is ever combined with a setting's own head.
        workloads = build_eviction_pressure_workloads(
            CharLevelFakeTokenizer(),
            object_count=6,
            body_tokens=32,
            head_tokens=NON_PREFIX_HEAD_TOKENS,
            tail_tokens=NON_PREFIX_TAIL_TOKENS,
            salt_prefix="phase4-r5-pressure",
        )
        labeled_heads = [
            (f"pressure-filler[{index}]", workload.target_head_ids)
            for index, workload in enumerate(workloads)
        ]
        validate_pairwise_head_isolation(labeled_heads)  # must not raise

    def test_64_fillers_past_the_original_24_marker_cap_are_all_distinct(self):
        # Regression test for a real report from a live GPU run: a
        # larger live capacity/rho combination reverse-computed a
        # filler_count of 64, past the OLD single-letter-alphabet cap
        # of 24, which raised ValueError deep into a long-running
        # canary. The code-point-pool-based marker scheme (see
        # _PRESSURE_FILLER_MARKER_CODEPOINT_BLOCKS /
        # _pressure_filler_head_literal_prefix) must keep working --
        # with all 64 fillers mutually distinct in both body and target
        # head -- well past that old cap.
        workloads = build_eviction_pressure_workloads(
            FakeTokenizer(),
            object_count=64,
            body_tokens=48,
            head_tokens=34,
            tail_tokens=1,
            salt_prefix="phase4-r5-pressure-past-cap",
        )
        self.assertEqual(len(workloads), 64)
        target_heads = [workload.target_head_ids for workload in workloads]
        self.assertEqual(
            len(target_heads),
            len(set(target_heads)),
            "all 64 filler target heads must be mutually distinct",
        )
        bodies = [workload.shared_body_ids for workload in workloads]
        self.assertEqual(
            len(bodies),
            len(set(bodies)),
            "all 64 filler bodies must be mutually distinct",
        )

    def test_64_fillers_past_the_original_24_marker_cap_pass_pairwise_head_isolation(
        self,
    ):
        # Same >24-filler scenario as above, but proven against the
        # real safety net (validate_pairwise_head_isolation, a genuine
        # token-ID-level pairwise common-prefix check) using the
        # finer-granularity char-level fake tokenizer, exactly like
        # test_fillers_pass_pairwise_head_isolation_against_a_bpe_like_tokenizer
        # above does for the <=24 case.
        workloads = build_eviction_pressure_workloads(
            CharLevelFakeTokenizer(),
            object_count=64,
            body_tokens=32,
            head_tokens=NON_PREFIX_HEAD_TOKENS,
            tail_tokens=NON_PREFIX_TAIL_TOKENS,
            salt_prefix="phase4-r5-pressure-past-cap",
        )
        labeled_heads = [
            (f"pressure-filler[{index}]", workload.target_head_ids)
            for index, workload in enumerate(workloads)
        ]
        validate_pairwise_head_isolation(labeled_heads)  # must not raise

    def test_500_fillers_well_past_any_short_ascii_text_ceiling_pass_pairwise_head_isolation(
        self,
    ):
        # A short synthetic-ASCII/hex-text marker scheme (this module's
        # own earlier, abandoned design) was empirically measured to
        # plateau at only ~183-400 distinct achievable first tokens
        # under the real Qwen3-0.6B tokenizer, regardless of alphabet or
        # length -- far short of a genuinely "arbitrary reasonable"
        # filler count. This proves the CURRENT code-point-pool-based
        # scheme clears that ceiling by a wide margin (500 is also
        # comfortably past this project's own ~540-filler realistic
        # worst-case estimate for --pressure-filler-body-tokens's
        # documented default against a large RTX 6000-class pool -- see
        # MAX_REASONABLE_EVICTION_PRESSURE_FILLER_COUNT's own comment).
        workloads = build_eviction_pressure_workloads(
            CharLevelFakeTokenizer(),
            object_count=500,
            body_tokens=8,
            head_tokens=NON_PREFIX_HEAD_TOKENS,
            tail_tokens=NON_PREFIX_TAIL_TOKENS,
            salt_prefix="phase4-r5-pressure-past-ascii-ceiling",
        )
        self.assertEqual(len(workloads), 500)
        labeled_heads = [
            (f"pressure-filler[{index}]", workload.target_head_ids)
            for index, workload in enumerate(workloads)
        ]
        validate_pairwise_head_isolation(labeled_heads)  # must not raise

    def test_reserved_first_token_ids_are_also_avoided(self):
        # Mirrors run_non_prefix_setting's real call site: the setting's
        # own head's first token is passed in as reserved so a filler
        # can never collide with it either, not just with other
        # fillers.
        tokenizer = CharLevelFakeTokenizer()
        reserved_workload = build_non_prefix_segment_workload(
            tokenizer,
            body_tokens=8,
            head_tokens=NON_PREFIX_HEAD_TOKENS,
            tail_tokens=NON_PREFIX_TAIL_TOKENS,
            salt="phase4-r5-pressure-reserved-setting-head",
        )
        reserved_first_token = reserved_workload.target_head_ids[0]
        workloads = build_eviction_pressure_workloads(
            tokenizer,
            object_count=16,
            body_tokens=8,
            head_tokens=NON_PREFIX_HEAD_TOKENS,
            tail_tokens=NON_PREFIX_TAIL_TOKENS,
            salt_prefix="phase4-r5-pressure-reserved",
            reserved_first_token_ids=frozenset({reserved_first_token}),
        )
        for workload in workloads:
            self.assertNotEqual(workload.target_head_ids[0], reserved_first_token)
        labeled_heads = [("setting-own-head", reserved_workload.target_head_ids)] + [
            (f"pressure-filler[{index}]", workload.target_head_ids)
            for index, workload in enumerate(workloads)
        ]
        validate_pairwise_head_isolation(labeled_heads)  # must not raise

    def test_retry_search_raises_cleanly_instead_of_hanging_when_exhausted(self):
        # Proves _build_pressure_filler_workload_avoiding_first_token_
        # collisions's bounded retry budget actually fires and raises
        # RuntimeError -- never hangs, never silently returns a
        # colliding workload -- when a vocabulary is genuinely too
        # impoverished to keep filler heads pairwise first-token-
        # distinct (see AlwaysCollidingFillerMarkerFakeTokenizer). The
        # first filler must still succeed (nothing reserved yet); only
        # the second, which collides on every one of its attempts,
        # must raise.
        with self.assertRaises(RuntimeError) as ctx:
            build_eviction_pressure_workloads(
                AlwaysCollidingFillerMarkerFakeTokenizer(),
                object_count=2,
                body_tokens=4,
                head_tokens=2,
                tail_tokens=1,
                salt_prefix="phase4-r5-pressure-exhaustion",
            )
        self.assertIn("pressure filler 1", str(ctx.exception))

    def test_retry_search_single_filler_succeeds_even_against_a_hostile_tokenizer(
        self,
    ):
        # With nothing yet reserved, even the hostile
        # AlwaysCollidingFillerMarkerFakeTokenizer accepts the very
        # first candidate for filler 0 -- the exhaustion failure mode
        # above is about the SECOND-and-later fillers colliding with an
        # already-accepted first token, not about the first attempt
        # ever being rejected outright.
        workloads = build_eviction_pressure_workloads(
            AlwaysCollidingFillerMarkerFakeTokenizer(),
            object_count=1,
            body_tokens=4,
            head_tokens=2,
            tail_tokens=1,
            salt_prefix="phase4-r5-pressure-single-hostile",
        )
        self.assertEqual(len(workloads), 1)

    def test_different_salt_prefixes_isolate_two_settings_filler_sets(self):
        # Mirrors two different settings (e.g. the main setting and a
        # length-sweep point) each building their own filler set: they
        # must never accidentally share content, exactly like
        # TestBuildNonPrefixSegmentWorkload
        # .test_different_salts_isolate_heads_and_bodies for the
        # non-filler case.
        main_fillers = build_eviction_pressure_workloads(
            FakeTokenizer(),
            object_count=2,
            body_tokens=64,
            head_tokens=34,
            tail_tokens=1,
            salt_prefix="phase4-r5-cachetune-main",
        )
        sweep_fillers = build_eviction_pressure_workloads(
            FakeTokenizer(),
            object_count=2,
            body_tokens=64,
            head_tokens=34,
            tail_tokens=1,
            salt_prefix="phase4-r5-cachetune-sweep-128",
        )
        for main_filler, sweep_filler in zip(main_fillers, sweep_fillers):
            self.assertNotEqual(
                main_filler.shared_body_ids, sweep_filler.shared_body_ids
            )


class TestEvictionPressureTotalTokens(unittest.TestCase):
    """The floor-estimate token sum reported alongside
    ``observed_rho_after_pressure`` in ``register_eviction_pressure_objects``'s
    own returned telemetry dict."""

    def _workload(self, body_tokens: int) -> NonPrefixSegmentWorkload:
        return NonPrefixSegmentWorkload(
            source_head_ids=(1, 2, 3),
            target_head_ids=(9, 8, 7),
            shared_body_ids=tuple(range(body_tokens)),
            tail_ids=(99,),
            seed_sentinel_ids=(999,),
        )

    def test_sums_body_tokens_across_workloads(self):
        workloads = [self._workload(10), self._workload(20), self._workload(30)]
        self.assertEqual(eviction_pressure_total_tokens(workloads), 60)

    def test_single_workload(self):
        self.assertEqual(eviction_pressure_total_tokens([self._workload(128)]), 128)

    def test_empty_sequence_sums_to_zero(self):
        self.assertEqual(eviction_pressure_total_tokens([]), 0)


class TestEvictionPressureFillerCountForRho(unittest.TestCase):
    """Reverse-computes the filler object count from a target nominal
    rho and a real, measured capacity -- the "actual capacity自动反算
    ...所需filler数" requirement."""

    def test_exact_division_needs_no_rounding_up(self):
        # target_total = 1.5 * 1000 = 1500; 1500 / 500 = 3.0 exactly.
        count = eviction_pressure_filler_count_for_rho(
            target_rho=1.5,
            usable_capacity_tokens=1000,
            tokens_per_filler=500,
        )
        self.assertEqual(count, 3)

    def test_rounds_up_on_fractional_result(self):
        # target_total = 0.9 * 1000 = 900; 900 / 500 = 1.8 -> ceil to 2.
        count = eviction_pressure_filler_count_for_rho(
            target_rho=0.9,
            usable_capacity_tokens=1000,
            tokens_per_filler=500,
        )
        self.assertEqual(count, 2)

    def test_large_target_rho_needs_many_fillers(self):
        count = eviction_pressure_filler_count_for_rho(
            target_rho=3.0,
            usable_capacity_tokens=2048,
            tokens_per_filler=2082,  # 34 head + 2048 body
        )
        self.assertEqual(count, math.ceil(3.0 * 2048 / 2082))

    def test_achieved_nominal_ratio_is_never_below_target(self):
        for target_rho in (0.9, 1.1, 1.5, 2.0, 3.0):
            count = eviction_pressure_filler_count_for_rho(
                target_rho=target_rho,
                usable_capacity_tokens=12345,
                tokens_per_filler=2082,
            )
            achieved = (count * 2082) / 12345
            self.assertGreaterEqual(achieved, target_rho)

    def test_rejects_non_positive_target_rho(self):
        with self.assertRaises(ValueError):
            eviction_pressure_filler_count_for_rho(
                target_rho=0.0,
                usable_capacity_tokens=1000,
                tokens_per_filler=500,
            )

    def test_rejects_negative_target_rho(self):
        with self.assertRaises(ValueError):
            eviction_pressure_filler_count_for_rho(
                target_rho=-1.5,
                usable_capacity_tokens=1000,
                tokens_per_filler=500,
            )

    def test_rejects_non_positive_capacity(self):
        with self.assertRaises(ValueError):
            eviction_pressure_filler_count_for_rho(
                target_rho=1.5,
                usable_capacity_tokens=0,
                tokens_per_filler=500,
            )

    def test_rejects_non_positive_tokens_per_filler(self):
        with self.assertRaises(ValueError):
            eviction_pressure_filler_count_for_rho(
                target_rho=1.5,
                usable_capacity_tokens=1000,
                tokens_per_filler=0,
            )

    def test_rejects_reverse_computed_count_beyond_the_sanity_bound(self):
        # Defense-in-depth fail-fast: the exact filler count needed
        # cannot be known before the run's own live capacity probe (see
        # module docstring / this function's docstring), so this cannot
        # catch every pathological input "before ANY request" -- but it
        # DOES catch it before the actual per-filler HTTP registration
        # loop starts for that setting, which is the earliest point the
        # real count is known. A tiny tokens_per_filler against a huge
        # capacity is exactly the kind of misconfiguration (e.g. an
        # accidentally-small --pressure-filler-body-tokens) that would
        # otherwise silently launch thousands of real, blocking
        # per-filler HTTP registrations.
        with self.assertRaises(ValueError):
            eviction_pressure_filler_count_for_rho(
                target_rho=100.0,
                usable_capacity_tokens=10_000_000,
                tokens_per_filler=1,
            )

    def test_accepts_reverse_computed_count_at_the_sanity_bound(self):
        # Exactly MAX_REASONABLE_EVICTION_PRESSURE_FILLER_COUNT must
        # still be accepted -- only counts strictly beyond it fail fast.
        count = eviction_pressure_filler_count_for_rho(
            target_rho=1.0,
            usable_capacity_tokens=MAX_REASONABLE_EVICTION_PRESSURE_FILLER_COUNT * 500,
            tokens_per_filler=500,
        )
        self.assertEqual(count, MAX_REASONABLE_EVICTION_PRESSURE_FILLER_COUNT)

    def test_already_pinned_tokens_defaults_to_zero_matching_original_formula(self):
        # already_pinned_tokens=0 (the default) must reproduce the
        # original, pre-fix formula exactly -- any caller that has no
        # already-resident setup footprint to account for (e.g. a
        # standalone reverse-computation, or a future call site that
        # never registers sources first) is unaffected by this fix.
        without_kwarg = eviction_pressure_filler_count_for_rho(
            target_rho=1.5,
            usable_capacity_tokens=1000,
            tokens_per_filler=500,
        )
        with_explicit_zero = eviction_pressure_filler_count_for_rho(
            target_rho=1.5,
            usable_capacity_tokens=1000,
            tokens_per_filler=500,
            already_pinned_tokens=0,
        )
        self.assertEqual(without_kwarg, 3)
        self.assertEqual(with_explicit_zero, 3)

    def test_already_pinned_tokens_reduces_the_filler_count(self):
        # target_total = 1.5 * 1000 = 1500. With already_pinned_tokens=0
        # this needs ceil(1500/500)=3 fillers (see the default-matching
        # test above); the setting's own raw+fresh source-registration
        # footprint (500 tokens, already counted toward target_rho by
        # construction -- see run_non_prefix_setting's own docstring)
        # must reduce that to ceil((1500-500)/500)=2, never still 3.
        count = eviction_pressure_filler_count_for_rho(
            target_rho=1.5,
            usable_capacity_tokens=1000,
            tokens_per_filler=500,
            already_pinned_tokens=500,
        )
        self.assertEqual(count, 2)

    def test_already_pinned_tokens_exactly_meeting_target_returns_zero_fillers(self):
        # already_pinned_tokens == target_total exactly: the setup's own
        # footprint alone already satisfies target_rho, a legitimate
        # "zero fillers needed" outcome, never an error.
        count = eviction_pressure_filler_count_for_rho(
            target_rho=1.5,
            usable_capacity_tokens=1000,
            tokens_per_filler=500,
            already_pinned_tokens=1500,
        )
        self.assertEqual(count, 0)

    def test_already_pinned_tokens_exceeding_target_returns_zero_not_negative(self):
        # already_pinned_tokens > target_total: remaining_tokens goes
        # negative internally, but the returned count must still be the
        # non-negative 0, never a negative filler count.
        count = eviction_pressure_filler_count_for_rho(
            target_rho=1.5,
            usable_capacity_tokens=1000,
            tokens_per_filler=500,
            already_pinned_tokens=2000,
        )
        self.assertEqual(count, 0)

    def test_rejects_negative_already_pinned_tokens(self):
        with self.assertRaises(ValueError):
            eviction_pressure_filler_count_for_rho(
                target_rho=1.5,
                usable_capacity_tokens=1000,
                tokens_per_filler=500,
                already_pinned_tokens=-1,
            )


class TestObservedRho(unittest.TestCase):
    """The genuine, sampled RESIDENT-occupancy ratio -- live
    ``sglang:kv_used_tokens`` PLUS ``sglang:kv_evictable_tokens`` gauges,
    summed, against a fixed capacity reference -- distinct from the
    nominal (requested-tokens) nature of
    ``eviction_pressure_filler_count_for_rho``, and distinct from
    ``kv_used_tokens`` alone: a real SM75 bug this fixes, since
    ``kv_used_tokens`` alone undercounts genuine pressure whenever
    LRU-evictable filler objects remain resident without yet being
    reclaimed."""

    def test_computes_simple_ratio_from_used_alone_when_evictable_is_zero(self):
        snapshot = {
            "sglang:kv_used_tokens": 500.0,
            "sglang:kv_evictable_tokens": 0.0,
        }
        self.assertAlmostEqual(observed_rho(snapshot, capacity_tokens=1000), 0.5)

    def test_sums_used_and_evictable_tokens_before_dividing_by_capacity(self):
        # The core fix under test: evictable (LRU-evictable, still
        # resident, not-yet-reclaimed) tokens count toward genuine
        # pressure exactly like used (pinned) tokens do -- never
        # dropped from the numerator the way an earlier, buggy version
        # of this function dropped them.
        snapshot = {
            "sglang:kv_used_tokens": 500.0,
            "sglang:kv_evictable_tokens": 300.0,
        }
        self.assertAlmostEqual(observed_rho(snapshot, capacity_tokens=1000), 0.8)

    def test_real_sm75_rho2_canary_shape_reports_genuine_high_resident_occupancy(
        self,
    ):
        # The exact real-world regression this fix addresses: a real
        # SM75 target_rho=2 canary's post-pressure snapshot, under the
        # OLD (kv_used_tokens-alone) formula, reported
        # peak_rho_observed=0.156 (2048 / 13130) -- appearing as LOW
        # pressure despite target_rho=2 -- because the vast majority of
        # the pool's genuine occupancy was sitting in
        # kv_evictable_tokens (surviving, not-yet-evicted dense
        # eviction-pressure fillers), never counted. The corrected
        # formula reports the pool as genuinely ~99% resident, matching
        # the real high-pressure condition target_rho=2 was configured
        # to produce.
        snapshot = {
            "sglang:kv_used_tokens": 2048.0,
            "sglang:kv_evictable_tokens": 10960.0,
        }
        self.assertAlmostEqual(
            observed_rho(snapshot, capacity_tokens=13130), 0.99071, places=4
        )

    def test_ratio_can_exceed_one_under_real_pressure(self):
        # The pool is a fixed physical size, but the SUM of nominal
        # filler requests can (by design) exceed it; the actual gauge
        # readings are capped at whatever physically fits, but a
        # snapshot taken transiently mid-registration could still
        # legitimately read higher than the fixed idle-capacity
        # reference if that reference itself under-counts a
        # since-grown pool -- this function must not silently clamp
        # such a reading.
        snapshot = {
            "sglang:kv_used_tokens": 900.0,
            "sglang:kv_evictable_tokens": 300.0,
        }
        self.assertAlmostEqual(observed_rho(snapshot, capacity_tokens=1000), 1.2)

    def test_zero_used_and_evictable_tokens_is_zero_ratio(self):
        snapshot = {
            "sglang:kv_used_tokens": 0.0,
            "sglang:kv_evictable_tokens": 0.0,
        }
        self.assertAlmostEqual(observed_rho(snapshot, capacity_tokens=1000), 0.0)

    def test_rejects_non_positive_capacity_tokens(self):
        with self.assertRaises(ValueError):
            observed_rho(
                {
                    "sglang:kv_used_tokens": 500.0,
                    "sglang:kv_evictable_tokens": 0.0,
                },
                capacity_tokens=0,
            )

    def test_raises_when_both_gauges_missing_from_snapshot(self):
        with self.assertRaises(ValueError) as ctx:
            observed_rho({}, capacity_tokens=1000)
        message = str(ctx.exception)
        self.assertIn("sglang:kv_used_tokens", message)
        self.assertIn("sglang:kv_evictable_tokens", message)

    def test_raises_when_only_evictable_tokens_is_missing(self):
        # A fail-fast requirement: a partially-available snapshot must
        # never silently fall back to used-alone (the exact bug this
        # fix addresses) -- it must raise, naming precisely the metric
        # that is actually missing.
        with self.assertRaises(ValueError) as ctx:
            observed_rho({"sglang:kv_used_tokens": 500.0}, capacity_tokens=1000)
        message = str(ctx.exception)
        self.assertIn("sglang:kv_evictable_tokens", message)
        self.assertNotIn("sglang:kv_used_tokens", message)

    def test_raises_when_only_used_tokens_is_missing(self):
        with self.assertRaises(ValueError) as ctx:
            observed_rho({"sglang:kv_evictable_tokens": 300.0}, capacity_tokens=1000)
        message = str(ctx.exception)
        self.assertIn("sglang:kv_used_tokens", message)
        self.assertNotIn("sglang:kv_evictable_tokens", message)


class TestChunkOffsets(unittest.TestCase):
    """Splits a body span into contiguous, <=max_chunk_tokens-long
    ``(offset, length)`` chunks -- the client-side mechanism backing
    ``body_segments_for_hash``."""

    def test_single_chunk_when_body_fits(self):
        self.assertEqual(chunk_offsets(300, 512), ((0, 300),))

    def test_exact_multiple_splits_evenly(self):
        self.assertEqual(
            chunk_offsets(1024, 512),
            ((0, 512), (512, 512)),
        )

    def test_remainder_chunk_is_shorter(self):
        self.assertEqual(
            chunk_offsets(1000, 512),
            ((0, 512), (512, 488)),
        )

    def test_body_exactly_equal_to_max_chunk(self):
        self.assertEqual(chunk_offsets(512, 512), ((0, 512),))

    def test_chunks_cover_total_tokens_with_no_gap_or_overlap(self):
        chunks = chunk_offsets(2048, 512)
        covered = 0
        for offset, length in chunks:
            self.assertEqual(offset, covered)
            covered += length
        self.assertEqual(covered, 2048)

    def test_rejects_non_positive_total_tokens(self):
        with self.assertRaises(ValueError):
            chunk_offsets(0, 512)

    def test_rejects_non_positive_max_chunk_tokens(self):
        with self.assertRaises(ValueError):
            chunk_offsets(1024, 0)


class TestBodySegmentsForHash(unittest.TestCase):
    """Builds the ``"segments"`` payload list for one body span, with
    distinct per-chunk ``content_hash`` values and offsets anchored at
    ``body_start``."""

    def test_single_chunk_body_produces_one_segment(self):
        segments = body_segments_for_hash(
            hash_prefix="cachetune-raw:test",
            body_start=34,
            body_tokens=300,
            max_chunk_tokens=512,
        )
        self.assertEqual(
            segments,
            [
                {
                    "content_hash": "cachetune-raw:test:chunk0",
                    "target_start": 34,
                    "length": 300,
                }
            ],
        )

    def test_multi_chunk_body_produces_multiple_segments_with_distinct_hashes(self):
        segments = body_segments_for_hash(
            hash_prefix="cachetune-raw:test",
            body_start=64,
            body_tokens=1024,
            max_chunk_tokens=512,
        )
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0]["content_hash"], "cachetune-raw:test:chunk0")
        self.assertEqual(segments[0]["target_start"], 64)
        self.assertEqual(segments[0]["length"], 512)
        self.assertEqual(segments[1]["content_hash"], "cachetune-raw:test:chunk1")
        self.assertEqual(segments[1]["target_start"], 64 + 512)
        self.assertEqual(segments[1]["length"], 512)

    def test_raw_and_fresh_hash_pairs_stay_paired_per_chunk(self):
        # restore_request_prefix_cachetune discovers a segment's "fresh"
        # companion via content_hash.replace(_RAW_PREFIX, _FRESH_PREFIX,
        # 1) -- so for every chunk index, the raw and fresh segment
        # lists built from "cachetune-raw:X"/"cachetune-fresh:X" must
        # differ ONLY by that prefix.
        raw_segments = body_segments_for_hash(
            hash_prefix="cachetune-raw:phase4-r5-main",
            body_start=64,
            body_tokens=1024,
            max_chunk_tokens=512,
        )
        fresh_segments = body_segments_for_hash(
            hash_prefix="cachetune-fresh:phase4-r5-main",
            body_start=64,
            body_tokens=1024,
            max_chunk_tokens=512,
        )
        for raw_segment, fresh_segment in zip(raw_segments, fresh_segments):
            self.assertEqual(
                raw_segment["content_hash"].replace(
                    "cachetune-raw:", "cachetune-fresh:", 1
                ),
                fresh_segment["content_hash"],
            )

    def test_target_start_offsets_are_relative_to_body_start(self):
        segments = body_segments_for_hash(
            hash_prefix="cachetune-raw:test",
            body_start=100,
            body_tokens=1500,
            max_chunk_tokens=512,
        )
        expected_starts = [100, 100 + 512, 100 + 1024]
        self.assertEqual([s["target_start"] for s in segments], expected_starts)

    def test_segment_lengths_sum_to_body_tokens(self):
        segments = body_segments_for_hash(
            hash_prefix="cachetune-raw:test",
            body_start=0,
            body_tokens=2048,
            max_chunk_tokens=512,
        )
        self.assertEqual(sum(s["length"] for s in segments), 2048)


class TestNonPrefixSegmentWorkloadPromptsLeaveFinalTokenForRealForwardPass(
    unittest.TestCase
):
    """Regression coverage for a real SM75 scheduler crash: an earlier
    version of ``NonPrefixSegmentWorkload.source_prompt_ids``/
    ``fresh_prompt_ids`` omitted ``tail_ids`` (only ``target_prompt_ids``
    had it), so the raw/fresh register call's own body segment(s) ended
    EXACTLY at ``len(prompt)``. That tripped ``ApproxKVRequestMetadata.
    validate_prompt_length``'s "approximate KV segments must leave the
    final prompt token for a real forward pass" check inside
    ``Req.__init__`` (``schedule_batch.py``) -- synchronously, on the
    scheduler's own request-admission path -- and killed the scheduler
    on a real GPU run.

    These tests feed this canary's own ``body_segments_for_hash`` output
    through the REAL, production ``ApproxKVRequestMetadata.
    validate_prompt_length`` (never a reimplementation of its own
    arithmetic) for every raw and fresh segment this canary would
    actually register, across the main setting's own shape, the full
    header x body shape sweep, an eviction-pressure filler's shape, and
    a body=2048 multi-chunk shape (4 x 512-token segments under the
    default ``--max-segment-chunk-tokens``) -- never just the trivial
    single-segment case, since a multi-chunk body's LAST chunk is the
    one that actually abuts the tail and is at risk.
    """

    @staticmethod
    def _metadata_from_segments(segments):
        return ApproxKVRequestMetadata(
            operation=ApproxKVRequestOperation.REGISTER,
            segments=tuple(
                ApproxKVRequestSegment(
                    content_hash=str(segment["content_hash"]),
                    target_start=int(segment["target_start"]),
                    length=int(segment["length"]),
                )
                for segment in segments
            ),
            model_fingerprint="test",
            cache_dtype="auto",
        )

    def _assert_segments_pass_real_validation(
        self, *, segments, prompt_length, context
    ):
        metadata = self._metadata_from_segments(segments)
        try:
            metadata.validate_prompt_length(prompt_length)
        except ValueError as exc:
            self.fail(
                f"{context}: real ApproxKVRequestMetadata.validate_prompt_length "
                f"rejected prompt_length={prompt_length} for segments={segments}: "
                f"{exc}"
            )

    def _assert_raw_and_fresh_segments_pass_real_validation(
        self, workload, *, max_chunk_tokens, context
    ):
        raw_segments = body_segments_for_hash(
            hash_prefix="cachetune-raw:test",
            body_start=workload.body_start_in_source,
            body_tokens=workload.body_tokens,
            max_chunk_tokens=max_chunk_tokens,
        )
        fresh_segments = body_segments_for_hash(
            hash_prefix="cachetune-fresh:test",
            body_start=workload.body_start_in_target,
            body_tokens=workload.body_tokens,
            max_chunk_tokens=max_chunk_tokens,
        )
        self.assertGreaterEqual(len(raw_segments), 1)
        self.assertGreaterEqual(len(fresh_segments), 1)
        self._assert_segments_pass_real_validation(
            segments=raw_segments,
            prompt_length=len(workload.source_prompt_ids),
            context=f"{context} raw",
        )
        self._assert_segments_pass_real_validation(
            segments=fresh_segments,
            prompt_length=len(workload.fresh_prompt_ids),
            context=f"{context} fresh",
        )
        # The exact, tightest-possible boundary (not just "doesn't
        # raise"): the LAST segment's target_end must land exactly one
        # token short of that prompt's own length -- the margin
        # NON_PREFIX_TAIL_TOKENS=1 provides, never more loosely
        # "somewhere less than".
        last_raw_end = max(s["target_start"] + s["length"] for s in raw_segments)
        last_fresh_end = max(s["target_start"] + s["length"] for s in fresh_segments)
        self.assertEqual(
            last_raw_end, workload.body_start_in_source + workload.body_tokens
        )
        self.assertEqual(
            last_fresh_end, workload.body_start_in_target + workload.body_tokens
        )
        self.assertLessEqual(last_raw_end, len(workload.source_prompt_ids) - 1)
        self.assertLessEqual(last_fresh_end, len(workload.fresh_prompt_ids) - 1)

    def test_main_setting_shape(self):
        # --main-header-tokens/--main-body-tokens defaults.
        workload = build_non_prefix_segment_workload(
            FakeTokenizer(),
            body_tokens=1024,
            head_tokens=64,
            tail_tokens=NON_PREFIX_TAIL_TOKENS,
            salt="phase4-r5-tail-regression-main",
        )
        self._assert_raw_and_fresh_segments_pass_real_validation(
            workload, max_chunk_tokens=512, context="main"
        )

    def test_every_shape_sweep_header_x_body_combination(self):
        # --header-tokens-choices x --body-tokens-choices defaults.
        # header=0 is a dedicated exact-context control point that never
        # builds a NonPrefixSegmentWorkload at all (see
        # run_exact_context_control_point) so it is correctly excluded
        # here -- it sends no approx_kv segments whatsoever.
        for header_tokens in (32, 64, 128, 256):
            for body_tokens in (512, 768, 1024, 2048):
                with self.subTest(header_tokens=header_tokens, body_tokens=body_tokens):
                    workload = build_non_prefix_segment_workload(
                        FakeTokenizer(),
                        body_tokens=body_tokens,
                        head_tokens=header_tokens,
                        tail_tokens=NON_PREFIX_TAIL_TOKENS,
                        salt=(
                            "phase4-r5-tail-regression-shape-"
                            f"h{header_tokens}-b{body_tokens}"
                        ),
                    )
                    self._assert_raw_and_fresh_segments_pass_real_validation(
                        workload,
                        max_chunk_tokens=512,
                        context=f"shape[header={header_tokens},body={body_tokens}]",
                    )

    def test_eviction_pressure_filler_shape(self):
        # --pressure-filler-head-tokens/--pressure-filler-body-tokens
        # defaults: every filler is its own NonPrefixSegmentWorkload.
        # register_eviction_pressure_objects only ever sends each
        # filler's own target_prompt_ids as a single plain dense request
        # (see that function's own docstring) -- it never registers or
        # reuses a filler's source_prompt_ids/fresh_prompt_ids. This
        # test nonetheless still verifies those NEVER-SENT raw/fresh
        # segments would themselves pass real ApproxKVRequestMetadata
        # validation: NonPrefixSegmentWorkload is shared, well-tested
        # infrastructure with the main setting's own genuine repair
        # workload, and this is a regression check on that shared
        # dataclass's own derivation logic, not a claim that fillers are
        # materialized this way.
        workloads = build_eviction_pressure_workloads(
            FakeTokenizer(),
            object_count=3,
            body_tokens=2048,
            head_tokens=NON_PREFIX_HEAD_TOKENS,
            tail_tokens=NON_PREFIX_TAIL_TOKENS,
            salt_prefix="phase4-r5-tail-regression-pressure",
        )
        for index, workload in enumerate(workloads):
            with self.subTest(filler_index=index):
                self._assert_raw_and_fresh_segments_pass_real_validation(
                    workload,
                    max_chunk_tokens=512,
                    context=f"pressure-filler[{index}]",
                )

    def test_body_2048_registers_as_four_chunks_and_every_chunk_passes(self):
        # The multi-segment case this regression explicitly calls out:
        # with the default --max-segment-chunk-tokens=512, a body=2048
        # span is registered as 4 distinct <=512-token segments per
        # register/reuse call (see chunk_offsets) -- assert each of the
        # 4, not just the last, individually satisfies the real
        # validator (chunk_offsets tiles the body contiguously with no
        # gaps, so only the LAST chunk can ever be the binding
        # constraint -- but this checks all of them directly rather
        # than assuming that).
        workload = build_non_prefix_segment_workload(
            FakeTokenizer(),
            body_tokens=2048,
            head_tokens=64,
            tail_tokens=NON_PREFIX_TAIL_TOKENS,
            salt="phase4-r5-tail-regression-body2048",
        )
        raw_segments = body_segments_for_hash(
            hash_prefix="cachetune-raw:test",
            body_start=workload.body_start_in_source,
            body_tokens=workload.body_tokens,
            max_chunk_tokens=512,
        )
        fresh_segments = body_segments_for_hash(
            hash_prefix="cachetune-fresh:test",
            body_start=workload.body_start_in_target,
            body_tokens=workload.body_tokens,
            max_chunk_tokens=512,
        )
        self.assertEqual(len(raw_segments), 4)
        self.assertEqual(len(fresh_segments), 4)
        for index, segment in enumerate(raw_segments):
            with self.subTest(chunk=index, prompt="source"):
                self._assert_segments_pass_real_validation(
                    segments=[segment],
                    prompt_length=len(workload.source_prompt_ids),
                    context=f"body2048 raw chunk {index}",
                )
        for index, segment in enumerate(fresh_segments):
            with self.subTest(chunk=index, prompt="fresh"):
                self._assert_segments_pass_real_validation(
                    segments=[segment],
                    prompt_length=len(workload.fresh_prompt_ids),
                    context=f"body2048 fresh chunk {index}",
                )

    def test_omitting_tail_from_source_would_have_raised(self):
        # Direct proof this test class is not vacuous: reconstructing
        # the OLD, buggy formula (head + body, no tail) that
        # source_prompt_ids used to return, and feeding its own segment
        # through the same real validator, DOES raise -- exactly the
        # ValueError that killed the scheduler on a real SM75 run
        # before this fix.
        workload = build_non_prefix_segment_workload(
            FakeTokenizer(),
            body_tokens=64,
            head_tokens=8,
            tail_tokens=NON_PREFIX_TAIL_TOKENS,
            salt="phase4-r5-tail-regression-negative-control",
        )
        raw_segments = body_segments_for_hash(
            hash_prefix="cachetune-raw:test",
            body_start=workload.body_start_in_source,
            body_tokens=workload.body_tokens,
            max_chunk_tokens=512,
        )
        metadata = self._metadata_from_segments(raw_segments)
        old_buggy_prompt_length = len(workload.source_head_ids) + len(
            workload.shared_body_ids
        )
        with self.assertRaises(ValueError) as ctx:
            metadata.validate_prompt_length(old_buggy_prompt_length)
        self.assertIn("must leave the final prompt token", str(ctx.exception))
        # Today's actual (fixed) source_prompt_ids must NOT raise for
        # the exact same segments.
        metadata.validate_prompt_length(len(workload.source_prompt_ids))


class TestFirstCommonPrefixLength(unittest.TestCase):
    """The token-id-level primitive ``validate_pairwise_head_isolation``
    is built on."""

    def test_identical_sequences_share_full_length(self):
        self.assertEqual(_first_common_prefix_length((1, 2, 3), (1, 2, 3)), 3)

    def test_diverging_at_first_token_shares_zero(self):
        self.assertEqual(_first_common_prefix_length((1, 2, 3), (9, 2, 3)), 0)

    def test_diverging_partway_through_shares_the_common_run(self):
        self.assertEqual(_first_common_prefix_length((1, 2, 3, 4), (1, 2, 9, 4)), 2)

    def test_empty_sequence_shares_zero(self):
        self.assertEqual(_first_common_prefix_length((), (1, 2, 3)), 0)
        self.assertEqual(_first_common_prefix_length((1, 2, 3), ()), 0)

    def test_shorter_sequence_that_is_a_prefix_of_the_longer_one(self):
        self.assertEqual(_first_common_prefix_length((1, 2), (1, 2, 3, 4)), 2)


class TestValidatePairwiseHeadIsolation(unittest.TestCase):
    """The runtime safety net for the exact collision risk this whole
    eviction-pressure feature had to design around: every simultaneously
    dense-seeded target head (the setting's own head plus every filler
    object's head) must be pairwise zero-common-prefix, checked against
    real token-id sequences, never a textual heuristic alone."""

    def test_no_collision_across_several_heads_does_not_raise(self):
        validate_pairwise_head_isolation(
            [
                ("setting", (1, 2, 3)),
                ("pressure-filler[0]", (4, 5, 6)),
                ("pressure-filler[1]", (7, 8, 9)),
            ]
        )  # must not raise

    def test_empty_list_does_not_raise(self):
        validate_pairwise_head_isolation([])  # must not raise

    def test_single_head_does_not_raise(self):
        validate_pairwise_head_isolation([("setting", (1, 2, 3))])  # must not raise

    def test_two_colliding_heads_raise_with_both_labels_named(self):
        with self.assertRaises(RuntimeError) as ctx:
            validate_pairwise_head_isolation(
                [
                    ("setting", (1, 2, 3)),
                    ("pressure-filler[0]", (1, 2, 9)),
                ]
            )
        message = str(ctx.exception)
        self.assertIn("setting", message)
        self.assertIn("pressure-filler[0]", message)

    def test_collision_among_a_non_adjacent_pair_is_still_caught(self):
        # The collision is between indices 0 and 2, not any adjacent
        # pair -- proves this checks every pair, not just neighbors.
        with self.assertRaises(RuntimeError) as ctx:
            validate_pairwise_head_isolation(
                [
                    ("pressure-filler[0]", (1, 2, 3)),
                    ("pressure-filler[1]", (4, 5, 6)),
                    ("pressure-filler[2]", (1, 2, 9)),
                ]
            )
        message = str(ctx.exception)
        self.assertIn("pressure-filler[0]", message)
        self.assertIn("pressure-filler[2]", message)

    def test_fully_identical_heads_are_a_collision_too(self):
        with self.assertRaises(RuntimeError):
            validate_pairwise_head_isolation(
                [
                    ("a", (1, 2, 3)),
                    ("b", (1, 2, 3)),
                ]
            )


class TestGeneratePayloadBuilders(unittest.TestCase):
    """The native ``/generate`` payload shape: ``input_ids`` at top
    level, ``custom_params`` nested inside ``sampling_params`` (unlike
    the OpenAI-compatible endpoint's top-level ``custom_params``)."""

    def test_dense_generate_payload_has_no_custom_params(self):
        payload = dense_generate_payload((1, 2, 3))
        self.assertEqual(payload["input_ids"], [1, 2, 3])
        self.assertIsInstance(payload["input_ids"], list)
        self.assertEqual(payload["sampling_params"]["max_new_tokens"], 1)
        self.assertNotIn("custom_params", payload["sampling_params"])

    def test_register_generate_payload_shape(self):
        payload = register_generate_payload(
            input_ids=(1, 2, 3, 4),
            segments=[
                {"content_hash": "cachetune-raw:test", "target_start": 2, "length": 2}
            ],
            model_fingerprint="fp",
            cache_dtype="fp16",
        )
        approx_kv = payload["sampling_params"]["custom_params"]["approx_kv"]
        self.assertEqual(approx_kv["operation"], "register")
        self.assertNotIn("plugin", approx_kv)
        self.assertEqual(
            approx_kv["segments"],
            [{"content_hash": "cachetune-raw:test", "target_start": 2, "length": 2}],
        )
        self.assertEqual(approx_kv["model_fingerprint"], "fp")
        self.assertEqual(approx_kv["cache_dtype"], "fp16")

    def test_register_generate_payload_supports_multiple_chunked_segments(self):
        segments = body_segments_for_hash(
            hash_prefix="cachetune-raw:test",
            body_start=64,
            body_tokens=1024,
            max_chunk_tokens=512,
        )
        payload = register_generate_payload(
            input_ids=tuple(range(1024 + 64)),
            segments=segments,
            model_fingerprint="fp",
            cache_dtype="fp16",
        )
        approx_kv = payload["sampling_params"]["custom_params"]["approx_kv"]
        self.assertEqual(len(approx_kv["segments"]), 2)
        self.assertEqual(approx_kv["segments"], segments)

    def test_reuse_generate_payload_shape(self):
        payload = reuse_generate_payload(
            input_ids=(1, 2, 3, 4, 5),
            segments=[
                {"content_hash": "cachetune-raw:test", "target_start": 2, "length": 2}
            ],
            model_fingerprint="fp",
            cache_dtype="fp16",
        )
        approx_kv = payload["sampling_params"]["custom_params"]["approx_kv"]
        self.assertEqual(approx_kv["operation"], "reuse")
        self.assertEqual(approx_kv["plugin"], "cachetune")
        self.assertEqual(
            approx_kv["segments"],
            [{"content_hash": "cachetune-raw:test", "target_start": 2, "length": 2}],
        )

    def test_input_ids_materialized_as_plain_list(self):
        # NonPrefixSegmentWorkload's properties return tuples; every
        # payload builder must convert to a plain list for the JSON body.
        payload = register_generate_payload(
            input_ids=(7, 8, 9),
            segments=[{"content_hash": "h", "target_start": 0, "length": 3}],
            model_fingerprint="fp",
            cache_dtype="fp16",
        )
        self.assertIsInstance(payload["input_ids"], list)
        self.assertEqual(payload["input_ids"], [7, 8, 9])

    def test_segments_materialized_as_plain_dicts_not_aliased(self):
        # register_generate_payload/reuse_generate_payload must copy each
        # segment mapping rather than embedding the caller's own mutable
        # dict by reference.
        original_segment = {"content_hash": "h", "target_start": 0, "length": 3}
        payload = register_generate_payload(
            input_ids=(1, 2, 3),
            segments=[original_segment],
            model_fingerprint="fp",
            cache_dtype="fp16",
        )
        approx_kv = payload["sampling_params"]["custom_params"]["approx_kv"]
        original_segment["length"] = 999
        self.assertEqual(approx_kv["segments"][0]["length"], 3)


class TestRequireFinishedByLength(unittest.TestCase):
    """Native ``/generate`` shapes ``finish_reason`` as a dict (see
    ``schedule_batch.FINISH_LENGTH.to_json``); unlike the OpenAI-
    compatible endpoint's plain string, so this must read ``.type``."""

    def test_passes_when_type_is_length(self):
        response = {"meta_info": {"finish_reason": {"type": "length", "length": 1}}}
        require_finished_by_length(response, "test")  # must not raise

    def test_raises_when_type_is_stop(self):
        response = {"meta_info": {"finish_reason": {"type": "stop", "matched": 5}}}
        with self.assertRaises(RuntimeError):
            require_finished_by_length(response, "test")

    def test_raises_when_type_is_abort(self):
        response = {"meta_info": {"finish_reason": {"type": "abort"}}}
        with self.assertRaises(RuntimeError):
            require_finished_by_length(response, "test")

    def test_error_message_includes_label(self):
        response = {"meta_info": {"finish_reason": {"type": "stop"}}}
        with self.assertRaises(RuntimeError) as ctx:
            require_finished_by_length(response, "sweep[128] reuse")
        self.assertIn("sweep[128] reuse", str(ctx.exception))


class TestRequireCachedTokens(unittest.TestCase):
    """``meta_info.cached_tokens`` is generic SGLang accounting
    (unrelated to CacheTune's own Prometheus counters) for the prefix
    already resolved without a fresh forward pass -- for a REGISTER
    request this is exact-match-only, but for a successful CacheTune
    reuse it also includes the entire restored body (see
    ``require_cached_tokens``'s own docstring). These tests exercise the
    generic assert-equal-else-raise behavior itself with arbitrary
    values, independent of any specific real call site's expected
    value."""

    def test_passes_and_returns_observed_when_matches(self):
        response = {"meta_info": {"cached_tokens": 34}}
        self.assertEqual(require_cached_tokens(response, 34, "test"), 34)

    def test_raises_on_mismatch(self):
        response = {"meta_info": {"cached_tokens": 0}}
        with self.assertRaises(RuntimeError):
            require_cached_tokens(response, 34, "test")

    def test_error_message_includes_both_values_and_label(self):
        response = {"meta_info": {"cached_tokens": 0}}
        with self.assertRaises(RuntimeError) as ctx:
            require_cached_tokens(response, 34, "main reuse")
        message = str(ctx.exception)
        self.assertIn("main reuse", message)
        self.assertIn("0", message)
        self.assertIn("34", message)


class _FakeAsyncLineIterator:
    """Fakes aiohttp's ``StreamReader`` async-for-line iteration
    (``async for raw_line in response.content``): each ``__anext__`` pops
    the next queued raw line, optionally sleeping first so tests can
    prove a client measures TTFT at the *first* chunk, never after the
    stream fully drains."""

    def __init__(self, lines, delay_before_index=None, delay_seconds=0.0):
        self._lines = list(lines)
        self._index = 0
        self._delay_before_index = delay_before_index
        self._delay_seconds = delay_seconds

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._lines):
            raise StopAsyncIteration
        if self._index == self._delay_before_index:
            await asyncio.sleep(self._delay_seconds)
        line = self._lines[self._index]
        self._index += 1
        return line


class _FakeStreamResponse:
    """Fakes an aiohttp streamed response: ``.status``, an async-iterable
    ``.content``, and an async ``.text()`` for non-2xx error bodies."""

    def __init__(self, lines, status=200, error_text="", **iterator_kwargs):
        self._lines = lines
        self.status = status
        self._error_text = error_text
        self._iterator_kwargs = iterator_kwargs

    @property
    def content(self):
        return _FakeAsyncLineIterator(self._lines, **self._iterator_kwargs)

    async def text(self):
        return self._error_text


class _FakePostContextManager:
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *exc_info):
        return False


class _FakeClientSession:
    """Fakes ``aiohttp.ClientSession``: records every ``.post(url,
    json=payload)`` call and always returns a preconfigured streamed
    response."""

    def __init__(self, response):
        self._response = response
        self.post_calls: list[tuple[str, dict]] = []

    def post(self, url, json):
        self.post_calls.append((url, json))
        return _FakePostContextManager(self._response)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False


class _SequencedFakeClientSession:
    """Fakes ``aiohttp.ClientSession`` for a call sequence where each
    ``.post()`` must return a DIFFERENT preconfigured response, in
    order -- unlike ``_FakeClientSession``, which always replays the
    same single response.

    Needed to test ``register_round_setup`` / ``run_target_reuse`` /
    ``register_eviction_pressure_objects``: a single round's own setup
    plus one reuse call alone already issues four sequential requests
    (seed head, register raw, register fresh, reuse), each expected to
    report different ``cached_tokens``, and multiple filler objects
    chain many such sequences back to back. Raises ``AssertionError``
    (never silently replaying a stale response) if more ``.post()``
    calls happen than responses were provided -- an unexpected extra
    call is itself a sign the
    production code under test regressed.
    """

    def __init__(self, responses: list):
        self._responses = list(responses)
        self._next_index = 0
        self.post_calls: list[tuple[str, dict]] = []

    def post(self, url, json):
        self.post_calls.append((url, json))
        if self._next_index >= len(self._responses):
            raise AssertionError(
                f"unexpected extra POST call #{self._next_index + 1} to "
                f"{url!r}; only {len(self._responses)} responses were "
                "configured"
            )
        response = self._responses[self._next_index]
        self._next_index += 1
        return _FakePostContextManager(response)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False


class _LabeledSequencedFakeClientSession(_SequencedFakeClientSession):
    """Like ``_SequencedFakeClientSession``, but appends a PER-CALL
    label (looked up positionally from ``labels``, parallel to
    ``responses``) to a shared ``call_order`` list on every
    ``.post()``.

    Needed to prove the RELATIVE order of several DIFFERENT kinds of
    HTTP request a single ``run_non_prefix_setting`` call makes (round
    setup, pressure filler, head re-seed, warmup round, formal repeat)
    -- unlike ``_OrderTrackingSequencedFakeClientSession`` (below),
    which only supports one fixed label shared by every call in the
    sequence, sufficient for a flow with just one kind of HTTP request
    but not for this multi-phase one.
    """

    def __init__(self, responses: list, labels: list[str], call_order: list[str]):
        super().__init__(responses)
        assert len(labels) == len(responses), (
            f"labels ({len(labels)}) must be parallel to responses "
            f"({len(responses)})"
        )
        self._labels = labels
        self._call_order = call_order

    def post(self, url, json):
        self._call_order.append(self._labels[self._next_index])
        return super().post(url, json)


def _sse_data_line(payload: dict) -> bytes:
    return ("data: " + json.dumps(payload) + "\n").encode("utf-8")


_SSE_DONE_LINE = b"data: [DONE]\n"


class TestTimedPost(unittest.TestCase):
    """``timed_post`` must target native ``/generate`` with a genuine
    streaming (``stream: true``) request and return the parsed final
    JSON chunk alongside a genuine client time-to-first-token
    measurement -- never a blocking whole-request elapsed-time
    approximation (see module docstring's "TTFT measurement
    methodology" section)."""

    def _run_with_fake_session(self, session):
        with unittest.mock.patch("aiohttp.ClientSession", return_value=session):
            return timed_post("http://127.0.0.1:30000", {"input_ids": [1, 2, 3]})

    def test_posts_to_generate_endpoint_with_stream_true(self):
        response_chunk = {
            "meta_info": {"finish_reason": {"type": "length"}, "cached_tokens": 0}
        }
        fake_response = _FakeStreamResponse(
            [_sse_data_line(response_chunk), _SSE_DONE_LINE]
        )
        session = _FakeClientSession(fake_response)

        response, ttft_ms = self._run_with_fake_session(session)

        self.assertEqual(len(session.post_calls), 1)
        url, payload = session.post_calls[0]
        self.assertEqual(url, "http://127.0.0.1:30000/generate")
        self.assertEqual(payload, {"input_ids": [1, 2, 3], "stream": True})
        self.assertEqual(response, response_chunk)
        self.assertGreaterEqual(ttft_ms, 0.0)

    def test_preserves_original_payload_keys_alongside_stream_true(self):
        response_chunk = {"meta_info": {"finish_reason": {"type": "length"}}}
        fake_response = _FakeStreamResponse(
            [_sse_data_line(response_chunk), _SSE_DONE_LINE]
        )
        session = _FakeClientSession(fake_response)
        original_payload = {
            "input_ids": [1, 2, 3],
            "sampling_params": {"max_new_tokens": 1, "temperature": 0},
        }

        with unittest.mock.patch("aiohttp.ClientSession", return_value=session):
            timed_post("http://127.0.0.1:30000", original_payload)

        _, sent_payload = session.post_calls[0]
        self.assertEqual(sent_payload["input_ids"], [1, 2, 3])
        self.assertEqual(
            sent_payload["sampling_params"], {"max_new_tokens": 1, "temperature": 0}
        )
        self.assertIs(sent_payload["stream"], True)
        # The caller's own dict must never be mutated by the merge.
        self.assertNotIn("stream", original_payload)

    def test_ttft_measures_time_to_first_chunk_not_full_stream_duration(self):
        """The defining behavioral test: TTFT must be timestamped at the
        first data chunk, never after the connection fully drains. A
        real server would never delay [DONE] after the (only) token for
        max_new_tokens=1, but this directly proves the client-side
        timing logic itself does not wait for stream completion."""
        response_chunk = {"meta_info": {"finish_reason": {"type": "length"}}}
        fake_response = _FakeStreamResponse(
            [_sse_data_line(response_chunk), _SSE_DONE_LINE],
            delay_before_index=1,
            delay_seconds=0.25,
        )
        session = _FakeClientSession(fake_response)

        wall_clock_start = time.perf_counter()
        _, ttft_ms = self._run_with_fake_session(session)
        wall_clock_elapsed_ms = (time.perf_counter() - wall_clock_start) * 1000.0

        self.assertGreaterEqual(wall_clock_elapsed_ms, 250.0)
        self.assertLess(ttft_ms, 100.0)

    def test_ttft_still_includes_delay_before_the_real_chunk_itself(self):
        """Ignorable frames (blank/keepalive) preceding the real data
        chunk must NOT be mistaken for the first token: if the server
        delays *before* emitting the real chunk, that delay must be
        counted in ttft_ms, not skipped over."""
        response_chunk = {"meta_info": {"finish_reason": {"type": "length"}}}
        fake_response = _FakeStreamResponse(
            [b"\n", b": keepalive\n", _sse_data_line(response_chunk), _SSE_DONE_LINE],
            delay_before_index=2,
            delay_seconds=0.25,
        )
        session = _FakeClientSession(fake_response)

        _, ttft_ms = self._run_with_fake_session(session)

        self.assertGreaterEqual(ttft_ms, 250.0)

    def test_returns_last_data_chunk_when_stream_has_multiple_frames(self):
        first_chunk = {"meta_info": {"finish_reason": {"type": "length"}, "n": 1}}
        last_chunk = {"meta_info": {"finish_reason": {"type": "length"}, "n": 2}}
        fake_response = _FakeStreamResponse(
            [
                _sse_data_line(first_chunk),
                _sse_data_line(last_chunk),
                _SSE_DONE_LINE,
            ]
        )
        session = _FakeClientSession(fake_response)

        response, _ = self._run_with_fake_session(session)

        self.assertEqual(response, last_chunk)

    def test_ignores_blank_and_non_data_prefixed_lines(self):
        response_chunk = {"meta_info": {"finish_reason": {"type": "length"}}}
        fake_response = _FakeStreamResponse(
            [
                b"\n",
                b": keepalive\n",
                _sse_data_line(response_chunk),
                b"\n",
                _SSE_DONE_LINE,
            ]
        )
        session = _FakeClientSession(fake_response)

        response, ttft_ms = self._run_with_fake_session(session)

        self.assertEqual(response, response_chunk)
        self.assertGreaterEqual(ttft_ms, 0.0)

    def test_raises_if_stream_never_terminates_with_done(self):
        response_chunk = {"meta_info": {"finish_reason": {"type": "length"}}}
        fake_response = _FakeStreamResponse([_sse_data_line(response_chunk)])
        session = _FakeClientSession(fake_response)

        with self.assertRaises(RuntimeError) as ctx:
            self._run_with_fake_session(session)
        self.assertIn("[DONE]", str(ctx.exception))

    def test_raises_if_stream_ends_with_done_but_no_data_chunk(self):
        fake_response = _FakeStreamResponse([_SSE_DONE_LINE])
        session = _FakeClientSession(fake_response)

        with self.assertRaises(RuntimeError) as ctx:
            self._run_with_fake_session(session)
        self.assertIn("no token", str(ctx.exception).lower())

    def test_raises_on_mid_stream_error_chunk(self):
        error_chunk = {
            "error": {
                "message": "boom",
                "type": "invalid_request_error",
                "code": 400,
            }
        }
        fake_response = _FakeStreamResponse(
            [_sse_data_line(error_chunk), _SSE_DONE_LINE]
        )
        session = _FakeClientSession(fake_response)

        with self.assertRaises(RuntimeError) as ctx:
            self._run_with_fake_session(session)
        self.assertIn("boom", str(ctx.exception))

    def test_raises_on_non_200_status_with_body_in_message(self):
        fake_response = _FakeStreamResponse(
            [], status=500, error_text="internal server error detail"
        )
        session = _FakeClientSession(fake_response)

        with self.assertRaises(RuntimeError) as ctx:
            self._run_with_fake_session(session)
        message = str(ctx.exception)
        self.assertIn("500", message)
        self.assertIn("internal server error detail", message)

    def test_response_meta_info_cached_tokens_usable_by_require_cached_tokens(self):
        response_chunk = {
            "meta_info": {"finish_reason": {"type": "length"}, "cached_tokens": 34}
        }
        fake_response = _FakeStreamResponse(
            [_sse_data_line(response_chunk), _SSE_DONE_LINE]
        )
        session = _FakeClientSession(fake_response)

        response, _ = self._run_with_fake_session(session)

        self.assertEqual(require_cached_tokens(response, 34, "test"), 34)


class TestRegisterBodyChunks(unittest.TestCase):
    """``register_body_chunks`` must split a body strictly longer than
    ``max_chunk_tokens`` into that many INDEPENDENT ``/generate``
    register calls -- one per chunk, each bounded at ``len(head_ids) +
    max_chunk_tokens + len(tail_ids)`` -- never one oversized call
    whose own transient per-request KV footprint scales with the full,
    un-chunked body (the real SM75 register-time OOM this function
    exists to avoid; see its own docstring), while staying identical to
    the previous single-call behavior whenever the body already fits in
    one chunk."""

    def _success_response(self, cached_tokens: int) -> _FakeStreamResponse:
        chunk = {
            "meta_info": {
                "finish_reason": {"type": "length"},
                "cached_tokens": cached_tokens,
            }
        }
        return _FakeStreamResponse([_sse_data_line(chunk), _SSE_DONE_LINE])

    def test_single_chunk_body_issues_exactly_one_call(self):
        head_ids = (1, 2, 3)
        body_ids = tuple(range(100, 108))
        tail_ids = (999,)
        session = _SequencedFakeClientSession([self._success_response(0)])

        with unittest.mock.patch("aiohttp.ClientSession", return_value=session):
            result = register_body_chunks(
                "http://127.0.0.1:30000",
                head_ids=head_ids,
                shared_body_ids=body_ids,
                tail_ids=tail_ids,
                hash_prefix="raw-hash",
                model_fingerprint="qwen3-0.6b-sm75",
                cache_dtype="fp16",
                max_chunk_tokens=512,
                expected_cached_tokens=0,
                label="unit-test",
            )

        self.assertEqual(len(session.post_calls), 1)
        url, payload = session.post_calls[0]
        self.assertEqual(url, "http://127.0.0.1:30000/generate")
        self.assertEqual(payload["input_ids"], list(head_ids + body_ids + tail_ids))
        segments = payload["sampling_params"]["custom_params"]["approx_kv"]["segments"]
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["content_hash"], "raw-hash:chunk0")
        self.assertEqual(segments[0]["target_start"], len(head_ids))
        self.assertEqual(segments[0]["length"], len(body_ids))
        self.assertEqual(result["chunk_count"], 1)
        self.assertEqual(result["cached_tokens"], 0)
        self.assertGreaterEqual(result["total_ms"], 0.0)

    def test_body_1024_issues_exactly_two_independent_calls(self):
        head_ids = tuple(range(64))
        body_ids = tuple(range(1000, 1000 + 1024))
        tail_ids = (999999,)
        session = _SequencedFakeClientSession(
            [self._success_response(0), self._success_response(0)]
        )

        with unittest.mock.patch("aiohttp.ClientSession", return_value=session):
            result = register_body_chunks(
                "http://127.0.0.1:30000",
                head_ids=head_ids,
                shared_body_ids=body_ids,
                tail_ids=tail_ids,
                hash_prefix="raw-hash",
                model_fingerprint="qwen3-0.6b-sm75",
                cache_dtype="fp16",
                max_chunk_tokens=512,
                expected_cached_tokens=0,
                label="unit-test",
            )

        self.assertEqual(len(session.post_calls), 2)
        self.assertEqual(result["chunk_count"], 2)
        bound = len(head_ids) + 512 + len(tail_ids)
        expected_bodies = [body_ids[0:512], body_ids[512:1024]]
        for index, (url, payload) in enumerate(session.post_calls):
            self.assertEqual(url, "http://127.0.0.1:30000/generate")
            self.assertLessEqual(len(payload["input_ids"]), bound)
            segments = payload["sampling_params"]["custom_params"]["approx_kv"][
                "segments"
            ]
            self.assertEqual(len(segments), 1)
            self.assertEqual(segments[0]["content_hash"], f"raw-hash:chunk{index}")
            # Every chunk's own local target_start is len(head_ids) --
            # identical for every chunk index, never body_start + offset.
            self.assertEqual(segments[0]["target_start"], len(head_ids))
            body_slice = payload["input_ids"][len(head_ids) : -len(tail_ids)]
            self.assertEqual(body_slice, list(expected_bodies[index]))

    def test_body_2048_issues_exactly_four_independent_calls(self):
        head_ids = tuple(range(64))
        body_ids = tuple(range(2000, 2000 + 2048))
        tail_ids = (999999,)
        session = _SequencedFakeClientSession([self._success_response(0)] * 4)

        with unittest.mock.patch("aiohttp.ClientSession", return_value=session):
            result = register_body_chunks(
                "http://127.0.0.1:30000",
                head_ids=head_ids,
                shared_body_ids=body_ids,
                tail_ids=tail_ids,
                hash_prefix="raw-hash",
                model_fingerprint="qwen3-0.6b-sm75",
                cache_dtype="fp16",
                max_chunk_tokens=512,
                expected_cached_tokens=0,
                label="unit-test",
            )

        self.assertEqual(len(session.post_calls), 4)
        self.assertEqual(result["chunk_count"], 4)
        for index, (_, payload) in enumerate(session.post_calls):
            segments = payload["sampling_params"]["custom_params"]["approx_kv"][
                "segments"
            ]
            self.assertEqual(segments[0]["content_hash"], f"raw-hash:chunk{index}")
            self.assertEqual(segments[0]["target_start"], len(head_ids))
            self.assertEqual(segments[0]["length"], 512)

    def test_uneven_final_chunk_still_respects_the_max_chunk_tokens_bound(self):
        head_ids = tuple(range(10))
        body_ids = tuple(range(1000))  # 1000 tokens -> chunks of 512 + 488
        tail_ids = (999999,)
        session = _SequencedFakeClientSession(
            [self._success_response(0), self._success_response(0)]
        )

        with unittest.mock.patch("aiohttp.ClientSession", return_value=session):
            register_body_chunks(
                "http://127.0.0.1:30000",
                head_ids=head_ids,
                shared_body_ids=body_ids,
                tail_ids=tail_ids,
                hash_prefix="raw-hash",
                model_fingerprint="qwen3-0.6b-sm75",
                cache_dtype="fp16",
                max_chunk_tokens=512,
                expected_cached_tokens=0,
                label="unit-test",
            )

        first_segments = session.post_calls[0][1]["sampling_params"]["custom_params"][
            "approx_kv"
        ]["segments"]
        second_segments = session.post_calls[1][1]["sampling_params"]["custom_params"][
            "approx_kv"
        ]["segments"]
        self.assertEqual(first_segments[0]["length"], 512)
        self.assertEqual(second_segments[0]["length"], 488)
        bound = len(head_ids) + 512 + len(tail_ids)
        self.assertLessEqual(len(session.post_calls[0][1]["input_ids"]), bound)
        self.assertLessEqual(len(session.post_calls[1][1]["input_ids"]), bound)

    def test_total_ms_is_the_exact_sum_of_every_chunk_ttft(self):
        # Patch timed_post directly (bypassing the aiohttp session
        # entirely) so this assertion can use exact arithmetic instead
        # of relying on real wall-clock timing noise between chunks.
        response = {
            "meta_info": {"finish_reason": {"type": "length"}, "cached_tokens": 0}
        }
        with unittest.mock.patch(
            "benchmark.approx_kv.run_phase4_cachetune_canary.timed_post",
            side_effect=[(response, 12.5), (response, 7.25)],
        ) as mock_timed_post:
            result = register_body_chunks(
                "http://127.0.0.1:30000",
                head_ids=tuple(range(64)),
                shared_body_ids=tuple(range(1024)),
                tail_ids=(999999,),
                hash_prefix="raw-hash",
                model_fingerprint="qwen3-0.6b-sm75",
                cache_dtype="fp16",
                max_chunk_tokens=512,
                expected_cached_tokens=0,
                label="unit-test",
            )

        self.assertEqual(mock_timed_post.call_count, 2)
        self.assertEqual(result["total_ms"], 19.75)

    def test_content_hash_sequence_matches_reuse_side_body_segments_for_hash(self):
        # register_body_chunks's own per-chunk content_hash sequence
        # must be exactly what body_segments_for_hash (the REUSE-side
        # builder) produces for the same hash_prefix/max_chunk_tokens --
        # this is the only thing that makes a chunk registered here
        # resolvable by a later reuse call's lookup (see
        # approx_kv/runtime.py's _segment_key: content_hash + token
        # content, never target_start).
        head_ids = tuple(range(64))
        body_ids = tuple(range(2048))
        tail_ids = (999999,)
        session = _SequencedFakeClientSession([self._success_response(0)] * 4)

        with unittest.mock.patch("aiohttp.ClientSession", return_value=session):
            register_body_chunks(
                "http://127.0.0.1:30000",
                head_ids=head_ids,
                shared_body_ids=body_ids,
                tail_ids=tail_ids,
                hash_prefix="cachetune-raw:unit-test",
                model_fingerprint="qwen3-0.6b-sm75",
                cache_dtype="fp16",
                max_chunk_tokens=512,
                expected_cached_tokens=0,
                label="unit-test",
            )

        observed_hashes = [
            call[1]["sampling_params"]["custom_params"]["approx_kv"]["segments"][0][
                "content_hash"
            ]
            for call in session.post_calls
        ]
        expected_hashes = [
            segment["content_hash"]
            for segment in body_segments_for_hash(
                hash_prefix="cachetune-raw:unit-test",
                body_start=0,  # reuse's own body_start is irrelevant to the hash
                body_tokens=len(body_ids),
                max_chunk_tokens=512,
            )
        ]
        self.assertEqual(observed_hashes, expected_hashes)

    def test_raises_when_a_later_chunk_reports_unexpected_cached_tokens(self):
        session = _SequencedFakeClientSession(
            [self._success_response(0), self._success_response(999)]
        )

        with unittest.mock.patch("aiohttp.ClientSession", return_value=session):
            with self.assertRaises(RuntimeError) as ctx:
                register_body_chunks(
                    "http://127.0.0.1:30000",
                    head_ids=tuple(range(64)),
                    shared_body_ids=tuple(range(1024)),
                    tail_ids=(999999,),
                    hash_prefix="raw-hash",
                    model_fingerprint="qwen3-0.6b-sm75",
                    cache_dtype="fp16",
                    max_chunk_tokens=512,
                    expected_cached_tokens=0,
                    label="unit-test",
                )
        self.assertIn("unit-test chunk1", str(ctx.exception))


class TestRegisterRoundSetup(unittest.TestCase):
    """``register_round_setup`` is one ROUND's own complete, indivisible
    setup: seed the exact-match target head, then register BOTH the raw
    (source-context) AND fresh (target-context) body segments, in that
    order -- always finished in full while THIS round's own eviction
    pressure is still low/absent. Both body registrations must route
    through ``register_body_chunks`` -- one independent ``/generate``
    call per ``<= max_chunk_tokens`` chunk, never one oversized call
    spanning the entire body.

    Registering raw AND fresh together (rather than raw once per
    *setting* and fresh again on every repeat, an earlier design's
    split between a since-removed ``register_non_prefix_sources``
    function and a since-removed ``run_reuse_once`` function) is a
    deliberate fix for a real SM75 ``target_rho=2`` ``MemoryError`` --
    see ``run_independent_round``'s own docstring for the full root
    cause."""

    def _workload(self, body_tokens):
        return build_non_prefix_segment_workload(
            FakeTokenizer(),
            body_tokens=body_tokens,
            head_tokens=64,
            tail_tokens=1,
            salt=f"unit-test-register-round-setup-{body_tokens}",
        )

    def _response(self, cached_tokens):
        chunk = {
            "meta_info": {
                "finish_reason": {"type": "length"},
                "cached_tokens": cached_tokens,
            }
        }
        return _FakeStreamResponse([_sse_data_line(chunk), _SSE_DONE_LINE])

    def _success_responses(self, workload, chunk_count):
        return (
            [self._response(0)]  # seed
            + [self._response(0)] * chunk_count  # raw chunks
            + [self._response(workload.body_start_in_target)] * chunk_count  # fresh
        )

    def test_body_1024_issues_one_seed_two_raw_and_two_fresh_register_calls(self):
        workload = self._workload(body_tokens=1024)
        chunk_count = len(chunk_offsets(workload.body_tokens, 512))
        self.assertEqual(chunk_count, 2)
        session = _SequencedFakeClientSession(
            self._success_responses(workload, chunk_count)
        )

        with unittest.mock.patch("aiohttp.ClientSession", return_value=session):
            result = register_round_setup(
                "http://127.0.0.1:30000",
                workload,
                raw_hash="cachetune-raw:unit-test-1024",
                fresh_hash="cachetune-fresh:unit-test-1024",
                model_fingerprint="qwen3-0.6b-sm75",
                cache_dtype="fp16",
                label="unit-test",
                max_chunk_tokens=512,
            )

        # 1 seed + 2 raw chunks + 2 fresh chunks == 5 total calls -- both
        # raw AND fresh registered together as one setup, never just raw
        # (that would be an earlier design's split), and never a reuse
        # call at all (that is run_target_reuse's own job).
        self.assertEqual(len(session.post_calls), 5)
        self.assertGreaterEqual(result["seed_head_ms"], 0.0)
        self.assertGreaterEqual(result["register_raw_ms"], 0.0)
        self.assertGreaterEqual(result["register_fresh_ms"], 0.0)
        self.assertEqual(result["fresh_cached_tokens"], workload.body_start_in_target)

        seed_url, seed_payload = session.post_calls[0]
        self.assertEqual(seed_url, "http://127.0.0.1:30000/generate")
        # The seed prompt is target_head_ids + seed_sentinel_ids -- never
        # target_head_ids alone (the real SM75 header-sweep bug this
        # fixes; see NonPrefixSegmentWorkload.seed_prompt_ids).
        self.assertEqual(seed_payload["input_ids"], list(workload.seed_prompt_ids))
        self.assertGreater(len(workload.seed_prompt_ids), len(workload.target_head_ids))
        self.assertNotIn("custom_params", seed_payload["sampling_params"])

    def test_body_2048_issues_one_seed_four_raw_and_four_fresh_register_calls(self):
        workload = self._workload(body_tokens=2048)
        chunk_count = len(chunk_offsets(workload.body_tokens, 512))
        self.assertEqual(chunk_count, 4)
        session = _SequencedFakeClientSession(
            self._success_responses(workload, chunk_count)
        )

        with unittest.mock.patch("aiohttp.ClientSession", return_value=session):
            register_round_setup(
                "http://127.0.0.1:30000",
                workload,
                raw_hash="cachetune-raw:unit-test-2048",
                fresh_hash="cachetune-fresh:unit-test-2048",
                model_fingerprint="qwen3-0.6b-sm75",
                cache_dtype="fp16",
                label="unit-test",
                max_chunk_tokens=512,
            )

        # 1 seed + 4 raw chunks + 4 fresh chunks == 9 total calls.
        self.assertEqual(len(session.post_calls), 9)

    def test_raw_runs_entirely_before_fresh(self):
        # Raw must be registered BEFORE fresh, in that order, within
        # this one setup step -- see this class's own docstring for why
        # the exact ordering inside setup itself does not matter for
        # correctness (both complete before any pressure filler either
        # way), but a stable, deterministic order still makes the
        # content-hash-pairing assertions below unambiguous.
        workload = self._workload(body_tokens=1024)
        chunk_count = len(chunk_offsets(workload.body_tokens, 512))
        session = _SequencedFakeClientSession(
            self._success_responses(workload, chunk_count)
        )

        with unittest.mock.patch("aiohttp.ClientSession", return_value=session):
            register_round_setup(
                "http://127.0.0.1:30000",
                workload,
                raw_hash="cachetune-raw:unit-test-order",
                fresh_hash="cachetune-fresh:unit-test-order",
                model_fingerprint="qwen3-0.6b-sm75",
                cache_dtype="fp16",
                label="unit-test",
                max_chunk_tokens=512,
            )

        raw_calls = session.post_calls[1 : 1 + chunk_count]
        fresh_calls = session.post_calls[1 + chunk_count :]
        for _, payload in raw_calls:
            segment = payload["sampling_params"]["custom_params"]["approx_kv"][
                "segments"
            ][0]
            self.assertTrue(segment["content_hash"].startswith("cachetune-raw:"))
        for _, payload in fresh_calls:
            segment = payload["sampling_params"]["custom_params"]["approx_kv"][
                "segments"
            ][0]
            self.assertTrue(segment["content_hash"].startswith("cachetune-fresh:"))

    def test_every_raw_and_fresh_chunk_prompt_is_bounded_by_head_plus_chunk_plus_tail(
        self,
    ):
        # Direct proof the old oversized-single-call register path (which
        # would have sent a ~1089-token prompt in one call for this exact
        # body/head/tail combination, OOM'ing a real SM75 server at
        # register time) is gone from BOTH raw and fresh registration.
        workload = self._workload(body_tokens=1024)
        chunk_count = len(chunk_offsets(workload.body_tokens, 512))
        session = _SequencedFakeClientSession(
            self._success_responses(workload, chunk_count)
        )

        with unittest.mock.patch("aiohttp.ClientSession", return_value=session):
            register_round_setup(
                "http://127.0.0.1:30000",
                workload,
                raw_hash="cachetune-raw:unit-test-1024b",
                fresh_hash="cachetune-fresh:unit-test-1024b",
                model_fingerprint="qwen3-0.6b-sm75",
                cache_dtype="fp16",
                label="unit-test",
                max_chunk_tokens=512,
            )

        bound = 64 + 512 + 1  # head_tokens + max_chunk_tokens + tail_tokens
        register_calls = session.post_calls[1:]  # everything after the seed call
        self.assertEqual(len(register_calls), 2 * chunk_count)
        for _, payload in register_calls:
            self.assertLessEqual(len(payload["input_ids"]), bound)

    def test_raw_and_fresh_content_hashes_match_reuse_side_body_segments_for_hash(self):
        # Both raw's and fresh's own per-chunk content_hash sequence
        # must be exactly what body_segments_for_hash (the REUSE-side
        # builder) produces for the respective raw_hash/fresh_hash --
        # this is the only thing that makes a chunk registered here
        # resolvable by a later reuse call's lookup.
        workload = self._workload(body_tokens=2048)
        chunk_count = len(chunk_offsets(workload.body_tokens, 512))
        raw_hash = "cachetune-raw:unit-test-1024c"
        fresh_hash = "cachetune-fresh:unit-test-1024c"
        session = _SequencedFakeClientSession(
            self._success_responses(workload, chunk_count)
        )

        with unittest.mock.patch("aiohttp.ClientSession", return_value=session):
            register_round_setup(
                "http://127.0.0.1:30000",
                workload,
                raw_hash=raw_hash,
                fresh_hash=fresh_hash,
                model_fingerprint="qwen3-0.6b-sm75",
                cache_dtype="fp16",
                label="unit-test",
                max_chunk_tokens=512,
            )

        raw_calls = session.post_calls[1 : 1 + chunk_count]
        fresh_calls = session.post_calls[1 + chunk_count :]
        observed_raw_hashes = [
            payload["sampling_params"]["custom_params"]["approx_kv"]["segments"][0][
                "content_hash"
            ]
            for _, payload in raw_calls
        ]
        observed_fresh_hashes = [
            payload["sampling_params"]["custom_params"]["approx_kv"]["segments"][0][
                "content_hash"
            ]
            for _, payload in fresh_calls
        ]
        expected_raw_hashes = [
            segment["content_hash"]
            for segment in body_segments_for_hash(
                hash_prefix=raw_hash,
                body_start=0,
                body_tokens=workload.body_tokens,
                max_chunk_tokens=512,
            )
        ]
        expected_fresh_hashes = [
            segment["content_hash"]
            for segment in body_segments_for_hash(
                hash_prefix=fresh_hash,
                body_start=0,
                body_tokens=workload.body_tokens,
                max_chunk_tokens=512,
            )
        ]
        self.assertEqual(observed_raw_hashes, expected_raw_hashes)
        self.assertEqual(observed_fresh_hashes, expected_fresh_hashes)

    def test_raises_when_seed_reports_nonzero_cached_tokens(self):
        # The seed request must be the FIRST appearance of this exact
        # target_head_ids in the tree (this round's own just-completed
        # flush guarantees this) -- a nonzero cached_tokens here means
        # something upstream did not actually flush.
        workload = self._workload(body_tokens=64)
        chunk = {"meta_info": {"finish_reason": {"type": "length"}, "cached_tokens": 5}}
        session = _SequencedFakeClientSession(
            [_FakeStreamResponse([_sse_data_line(chunk), _SSE_DONE_LINE])]
        )

        with unittest.mock.patch("aiohttp.ClientSession", return_value=session):
            with self.assertRaises(RuntimeError) as ctx:
                register_round_setup(
                    "http://127.0.0.1:30000",
                    workload,
                    raw_hash="cachetune-raw:unit-test-seed-nonzero",
                    fresh_hash="cachetune-fresh:unit-test-seed-nonzero",
                    model_fingerprint="qwen3-0.6b-sm75",
                    cache_dtype="fp16",
                    label="unit-test",
                    max_chunk_tokens=512,
                )
        self.assertIn("seed target_head", str(ctx.exception))

    def test_seed_response_arbitrary_generated_content_does_not_corrupt_fresh_cached_tokens(
        self,
    ):
        # Regression test for the real SM75 header=32 bug's root cause:
        # the seed request's own GENERATED token content must never be
        # able to influence a later request's own reported
        # cached_tokens. These fakes have no persistent, stateful radix
        # tree, so this cannot literally reproduce the live server-side
        # tree-extension bug end to end -- what it DOES prove, directly
        # against the real production code path, is that (a)
        # require_finished_by_length/require_cached_tokens (the only
        # two places this script ever inspects a /generate response)
        # never read anything from the response except meta_info.
        # finish_reason/cached_tokens -- so an arbitrary extra field
        # simulating "the model happened to generate shared_body_ids[0]"
        # is silently ignored, never consulted -- and (b) the seed
        # request's own PROMPT is always seed_prompt_ids (target_head_ids
        # + seed_sentinel_ids), independent of whatever the mocked
        # response claims to have generated. Together, on a real server,
        # these two facts are exactly why the fix works: the seed
        # request's own generated-token VALUE can only ever extend the
        # tree past the sentinel node, never collide with
        # shared_body_ids[0] at the position that matters (see
        # NonPrefixSegmentWorkload.seed_prompt_ids's own docstring).
        workload = self._workload(body_tokens=64)
        chunk_count = len(chunk_offsets(workload.body_tokens, 512))
        seed_chunk_simulating_collision = {
            "meta_info": {
                "finish_reason": {"type": "length"},
                "cached_tokens": 0,
            },
            # Arbitrary extra content a real server's response might
            # carry -- deliberately chosen to equal shared_body_ids[0],
            # simulating exactly the collision this fix guards against.
            # Neither require_finished_by_length nor require_cached_
            # tokens ever reads this key.
            "output_ids": [workload.shared_body_ids[0]],
            "text": "arbitrary generated content unrelated to correctness",
        }
        responses = (
            [
                _FakeStreamResponse(
                    [_sse_data_line(seed_chunk_simulating_collision), _SSE_DONE_LINE]
                )
            ]
            + [self._response(0)] * chunk_count
            + [self._response(workload.body_start_in_target)] * chunk_count
        )
        session = _SequencedFakeClientSession(responses)

        with unittest.mock.patch("aiohttp.ClientSession", return_value=session):
            result = register_round_setup(
                "http://127.0.0.1:30000",
                workload,
                raw_hash="cachetune-raw:unit-test-seed-collision",
                fresh_hash="cachetune-fresh:unit-test-seed-collision",
                model_fingerprint="qwen3-0.6b-sm75",
                cache_dtype="fp16",
                label="unit-test",
                max_chunk_tokens=512,
            )

        seed_url, seed_payload = session.post_calls[0]
        del seed_url
        # The seed prompt itself never changes based on what the
        # response claims to have generated.
        self.assertEqual(seed_payload["input_ids"], list(workload.seed_prompt_ids))
        # And the fresh registration's own reported cached_tokens is
        # exactly the header length -- never header+1 -- regardless of
        # the seed response's own "colliding" extra content.
        self.assertEqual(result["fresh_cached_tokens"], workload.body_start_in_target)

    def test_fresh_register_reports_cached_tokens_exactly_equal_to_header_length(self):
        # Explicit regression test using the exact real SM75 bug's own
        # numbers: a header=32 sweep point observed fresh-register
        # cached_tokens=33 (one MORE than the header's true length)
        # before this fix, because the bare target_head_ids-only seed
        # request's own generated token happened to equal
        # shared_body_ids[0]. After the fix, the seed prompt is
        # seed_prompt_ids (never target_head_ids alone), so this must
        # always equal EXACTLY body_start_in_target == 32, never 33 or
        # any other off-by-one value.
        workload = build_non_prefix_segment_workload(
            FakeTokenizer(),
            body_tokens=512,
            head_tokens=32,
            tail_tokens=1,
            salt="unit-test-header-32-real-bug-regression",
        )
        self.assertEqual(workload.body_start_in_target, 32)
        chunk_count = len(chunk_offsets(workload.body_tokens, 512))
        self.assertEqual(chunk_count, 1)
        session = _SequencedFakeClientSession(
            self._success_responses(workload, chunk_count)
        )

        with unittest.mock.patch("aiohttp.ClientSession", return_value=session):
            result = register_round_setup(
                "http://127.0.0.1:30000",
                workload,
                raw_hash="cachetune-raw:unit-test-header-32",
                fresh_hash="cachetune-fresh:unit-test-header-32",
                model_fingerprint="qwen3-0.6b-sm75",
                cache_dtype="fp16",
                label="unit-test",
                max_chunk_tokens=512,
            )

        self.assertEqual(result["fresh_cached_tokens"], 32)
        self.assertNotEqual(result["fresh_cached_tokens"], 33)


class TestRunTargetReuse(unittest.TestCase):
    """``run_target_reuse`` is the reuse-ONLY half of one round's own
    measurement -- issue exactly one reuse request against segments a
    prior ``register_round_setup`` call already registered for the SAME
    round. Used identically for the discarded warmup round and every
    formal repeat, via ``run_independent_round``.

    Performs NO registration of its own: an earlier design (a since-
    removed ``run_reuse_once`` function) registered fresh again on every
    call and shared ONE raw registration (from a since-removed
    ``register_non_prefix_sources``, called once per *setting*) across
    every repeat -- see ``register_round_setup``'s own docstring for the
    real SM75 ``target_rho=2`` ``MemoryError`` that split caused."""

    def _workload(self, body_tokens):
        return build_non_prefix_segment_workload(
            FakeTokenizer(),
            body_tokens=body_tokens,
            head_tokens=64,
            tail_tokens=1,
            salt=f"unit-test-run-target-reuse-{body_tokens}",
        )

    def _response(self, cached_tokens):
        chunk = {
            "meta_info": {
                "finish_reason": {"type": "length"},
                "cached_tokens": cached_tokens,
            }
        }
        return _FakeStreamResponse([_sse_data_line(chunk), _SSE_DONE_LINE])

    def test_body_1024_issues_exactly_one_reuse_call(self):
        workload = self._workload(body_tokens=1024)
        chunk_count = len(chunk_offsets(workload.body_tokens, 512))
        self.assertEqual(chunk_count, 2)
        reuse_cached = workload.body_start_in_target + workload.body_tokens
        session = _SequencedFakeClientSession([self._response(reuse_cached)])

        with unittest.mock.patch("aiohttp.ClientSession", return_value=session):
            result = run_target_reuse(
                "http://127.0.0.1:30000",
                workload,
                raw_hash="cachetune-raw:unit-test-reuse-1024",
                model_fingerprint="qwen3-0.6b-sm75",
                cache_dtype="fp16",
                label="unit-test",
                max_chunk_tokens=512,
            )

        # Exactly ONE call -- never a seed, raw-register, or
        # fresh-register call (register_round_setup's own job).
        self.assertEqual(len(session.post_calls), 1)
        self.assertGreaterEqual(result["reuse_ms"], 0.0)
        self.assertEqual(result["reuse_cached_tokens"], reuse_cached)
        self.assertIsInstance(result["reuse_response"], dict)

        # The reuse call must still be a single call over the COMPLETE,
        # un-chunked target_prompt_ids -- never split into per-chunk
        # calls the way register is.
        _, reuse_payload = session.post_calls[0]
        self.assertEqual(reuse_payload["input_ids"], list(workload.target_prompt_ids))
        reuse_segments = reuse_payload["sampling_params"]["custom_params"]["approx_kv"][
            "segments"
        ]
        self.assertEqual(len(reuse_segments), chunk_count)

    def test_body_2048_still_issues_exactly_one_reuse_call(self):
        workload = self._workload(body_tokens=2048)
        chunk_count = len(chunk_offsets(workload.body_tokens, 512))
        self.assertEqual(chunk_count, 4)
        reuse_cached = workload.body_start_in_target + workload.body_tokens
        session = _SequencedFakeClientSession([self._response(reuse_cached)])

        with unittest.mock.patch("aiohttp.ClientSession", return_value=session):
            run_target_reuse(
                "http://127.0.0.1:30000",
                workload,
                raw_hash="cachetune-raw:unit-test-reuse-2048",
                model_fingerprint="qwen3-0.6b-sm75",
                cache_dtype="fp16",
                label="unit-test",
                max_chunk_tokens=512,
            )

        self.assertEqual(len(session.post_calls), 1)

    def test_reuse_content_hashes_are_built_from_raw_hash_not_a_fresh_hash(self):
        workload = self._workload(body_tokens=1024)
        chunk_count = len(chunk_offsets(workload.body_tokens, 512))
        raw_hash = "cachetune-raw:unit-test-reuse-1024c"
        reuse_cached = workload.body_start_in_target + workload.body_tokens
        session = _SequencedFakeClientSession([self._response(reuse_cached)])

        with unittest.mock.patch("aiohttp.ClientSession", return_value=session):
            run_target_reuse(
                "http://127.0.0.1:30000",
                workload,
                raw_hash=raw_hash,
                model_fingerprint="qwen3-0.6b-sm75",
                cache_dtype="fp16",
                label="unit-test",
                max_chunk_tokens=512,
            )

        reuse_segments = session.post_calls[0][1]["sampling_params"]["custom_params"][
            "approx_kv"
        ]["segments"]
        expected_hashes = [
            segment["content_hash"]
            for segment in body_segments_for_hash(
                hash_prefix=raw_hash,
                body_start=workload.body_start_in_target,
                body_tokens=workload.body_tokens,
                max_chunk_tokens=512,
            )
        ]
        self.assertEqual(len(reuse_segments), chunk_count)
        self.assertEqual(
            [segment["content_hash"] for segment in reuse_segments], expected_hashes
        )

    def test_can_be_called_repeatedly_against_independently_registered_segments(self):
        # The exact reuse this fix depends on: run_independent_round
        # calls this SAME function identically for the discarded warmup
        # round and every formal repeat -- proving it is safely callable
        # more than once in a row for the same workload, with no
        # internal state of its own carried between calls.
        workload = self._workload(body_tokens=64)
        reuse_cached = workload.body_start_in_target + workload.body_tokens
        session = _SequencedFakeClientSession([self._response(reuse_cached)] * 3)

        with unittest.mock.patch("aiohttp.ClientSession", return_value=session):
            for _ in range(3):
                result = run_target_reuse(
                    "http://127.0.0.1:30000",
                    workload,
                    raw_hash="cachetune-raw:unit-test-repeatable",
                    model_fingerprint="qwen3-0.6b-sm75",
                    cache_dtype="fp16",
                    label="unit-test",
                    max_chunk_tokens=512,
                )
                self.assertEqual(result["reuse_cached_tokens"], reuse_cached)

        self.assertEqual(len(session.post_calls), 3)

    def test_raises_when_reuse_reports_head_only_cached_tokens(self):
        # A successful CacheTune reuse always extends prefix_indices by
        # the FULL restored body (body_start_in_target + body_tokens),
        # never just the exact-match head alone -- a head-only value
        # here means the recovery-slot allocation silently failed to
        # restore the body.
        workload = self._workload(body_tokens=64)
        session = _SequencedFakeClientSession(
            [self._response(workload.body_start_in_target)]
        )

        with unittest.mock.patch("aiohttp.ClientSession", return_value=session):
            with self.assertRaises(RuntimeError) as ctx:
                run_target_reuse(
                    "http://127.0.0.1:30000",
                    workload,
                    raw_hash="cachetune-raw:unit-test-reuse-head-only",
                    model_fingerprint="qwen3-0.6b-sm75",
                    cache_dtype="fp16",
                    label="unit-test",
                    max_chunk_tokens=512,
                )
        self.assertIn("reuse", str(ctx.exception))


class TestEnsureTargetHeadResident(unittest.TestCase):
    """``ensure_target_head_resident`` re-seeds a round's own target
    head with one plain dense request (over ``target_head_ids +
    seed_sentinel_ids`` -- never ``target_head_ids`` alone; see
    ``NonPrefixSegmentWorkload.seed_prompt_ids`` for the real SM75
    header-sweep bug this fixes) after that round's own
    eviction-pressure phase, tolerant of any of THREE outcomes: a full
    hit (``cached_tokens == len(seed_prompt_ids)``, head AND sentinel
    both survived pressure), a head-only hit (``cached_tokens ==
    len(target_head_ids)``, the head survived but the deeper sentinel
    node was independently evicted), or a clean miss (``cached_tokens
    == 0``, the head was evicted by pressure and just got recomputed) --
    but never a corrupted partial value that is none of these. This is
    an additional defensive measure this script adds (not part of
    CacheTune's own design) because sending genuine LRU eviction
    pressure immediately after seeding the head -- this script's own
    "register setup before pressure" ordering, per round -- makes the
    head itself a plausible LRU-eviction candidate; without this guard,
    an evicted head could never be restored by any later register/reuse
    call (both always skip radix insertion)."""

    def _workload(self):
        return build_non_prefix_segment_workload(
            FakeTokenizer(),
            body_tokens=64,
            head_tokens=34,
            tail_tokens=1,
            salt="unit-test-ensure-head-resident",
        )

    def _response(self, cached_tokens):
        chunk = {
            "meta_info": {
                "finish_reason": {"type": "length"},
                "cached_tokens": cached_tokens,
            }
        }
        return _FakeStreamResponse([_sse_data_line(chunk), _SSE_DONE_LINE])

    def test_reports_full_hit_when_head_and_sentinel_survived_pressure(self):
        workload = self._workload()
        session = _FakeClientSession(self._response(len(workload.seed_prompt_ids)))

        with unittest.mock.patch("aiohttp.ClientSession", return_value=session):
            result = ensure_target_head_resident(
                "http://127.0.0.1:30000", workload, label="unit-test"
            )

        self.assertEqual(len(session.post_calls), 1)
        url, payload = session.post_calls[0]
        self.assertEqual(url, "http://127.0.0.1:30000/generate")
        # The re-seed prompt is target_head_ids + seed_sentinel_ids --
        # never target_head_ids alone (the real SM75 header-sweep bug
        # this fixes; see NonPrefixSegmentWorkload.seed_prompt_ids).
        self.assertEqual(payload["input_ids"], list(workload.seed_prompt_ids))
        self.assertNotIn("custom_params", payload["sampling_params"])
        self.assertEqual(result["cached_tokens"], len(workload.seed_prompt_ids))
        self.assertFalse(result["was_evicted_by_pressure"])
        self.assertGreaterEqual(result["ttft_ms"], 0.0)

    def test_reports_head_only_hit_when_sentinel_extension_was_evicted(self):
        # The head itself survived pressure, but the deeper sentinel-
        # extension node (strictly newer/deeper in the tree than the
        # head node) was independently reclaimed by LRU -- still a
        # legitimate "head survived" outcome, not a corrupted value.
        workload = self._workload()
        session = _FakeClientSession(self._response(len(workload.target_head_ids)))

        with unittest.mock.patch("aiohttp.ClientSession", return_value=session):
            result = ensure_target_head_resident(
                "http://127.0.0.1:30000", workload, label="unit-test"
            )

        self.assertEqual(result["cached_tokens"], len(workload.target_head_ids))
        self.assertFalse(result["was_evicted_by_pressure"])

    def test_reports_miss_when_the_head_was_evicted_by_pressure(self):
        workload = self._workload()
        session = _FakeClientSession(self._response(0))

        with unittest.mock.patch("aiohttp.ClientSession", return_value=session):
            result = ensure_target_head_resident(
                "http://127.0.0.1:30000", workload, label="unit-test"
            )

        self.assertEqual(result["cached_tokens"], 0)
        self.assertTrue(result["was_evicted_by_pressure"])

    def test_raises_on_a_corrupted_partial_cached_tokens_value(self):
        # Neither 0 (clean miss), len(target_head_ids) (head-only hit),
        # nor len(seed_prompt_ids) (full hit) -- a corrupted or
        # unexpected partial radix match, which must never be silently
        # accepted as one of the three tolerated outcomes.
        workload = self._workload()
        session = _FakeClientSession(self._response(17))

        with unittest.mock.patch("aiohttp.ClientSession", return_value=session):
            with self.assertRaises(RuntimeError) as ctx:
                ensure_target_head_resident(
                    "http://127.0.0.1:30000", workload, label="unit-test"
                )
        message = str(ctx.exception)
        self.assertIn("unit-test", message)
        self.assertIn("17", message)
        self.assertIn(str(len(workload.target_head_ids)), message)
        self.assertIn(str(len(workload.seed_prompt_ids)), message)


class TestRegisterEvictionPressureObjects(unittest.TestCase):
    """Every eviction-pressure filler object must be sent as exactly ONE
    plain, ordinary dense ``/generate`` request -- carrying NO
    ``approx_kv`` custom_params metadata at all -- so its KV lands in
    the server's ordinary, LRU-evictable exact radix tree rather than
    CacheTune's own un-evictable segment store.

    This replaces an earlier design that ran every filler through a
    full seed+raw-register+fresh-register+reuse CacheTune cycle (what
    is now ``register_round_setup`` + ``run_target_reuse``): a
    real SM75 run at ``target_rho=2`` showed that design's raw/fresh
    segments accumulate as permanently un-evictable residency
    (``ApproxKVManager``'s own segment store is invisible to Radix LRU
    eviction), eventually starving the setting's own target recovery-
    slot allocation. See ``register_eviction_pressure_objects``'s own
    docstring for the full root-cause account.

    A nonzero dense-fallback delta observed during this phase must
    still raise loudly (a plain dense request should never be able to
    move that CacheTune-reuse-specific counter at all), and a pressure
    configuration that nominally exceeds this setting's own idle
    capacity must show genuine ``sglang:evicted_tokens_total`` movement,
    or this function must raise rather than silently trust an unverified
    "genuine pressure" claim.

    A SECOND, separate real SM75 bug at ``target_rho=2`` -- fillers
    being sent BEFORE the setting's own raw+fresh source registration,
    which is NOT wired to evict exact-radix victims for itself -- is
    fixed by this function's own ``already_pinned_tokens`` parameter:
    the setting's own already-resident setup footprint (measured, never
    estimated, by the caller via ``metric_delta`` on
    ``sglang:kv_used_tokens`` immediately after source setup completes)
    is set aside from ``capacity_tokens`` before gating the "genuine
    eviction occurred" assertion, and is rejected immediately if it
    alone already meets or exceeds ``capacity_tokens``."""

    def _filler_workloads(self, object_count=2, body_tokens=8):
        return build_eviction_pressure_workloads(
            FakeTokenizer(),
            object_count=object_count,
            body_tokens=body_tokens,
            head_tokens=6,
            tail_tokens=1,
            salt_prefix="unit-test-pressure",
        )

    def _dense_zero_cached_response(self):
        """The single response every plain-dense filler request expects:
        ``finish_reason=length`` and ``cached_tokens=0`` (this exact
        content genuinely never seen before -- pairwise first-token
        isolation plus the setting's own just-completed flush together
        guarantee this)."""
        chunk = {"meta_info": {"finish_reason": {"type": "length"}, "cached_tokens": 0}}
        return _FakeStreamResponse([_sse_data_line(chunk), _SSE_DONE_LINE])

    def test_sends_exactly_one_plain_dense_request_per_filler(self):
        workloads = self._filler_workloads(object_count=3, body_tokens=8)
        session = _FakeClientSession(self._dense_zero_cached_response())
        metrics_before = {
            "sglang:evicted_tokens_total": 0.0,
            "sglang:kv_used_tokens": 0.0,
        }
        metrics_after = {
            "sglang:evicted_tokens_total": 0.0,
            "sglang:kv_used_tokens": 24.0,
            "sglang:kv_evictable_tokens": 0.0,
        }

        with unittest.mock.patch(
            "aiohttp.ClientSession", return_value=session
        ), unittest.mock.patch(
            "benchmark.approx_kv.run_phase4_cachetune_canary.metric_snapshot",
            side_effect=[metrics_before, metrics_after],
        ):
            register_eviction_pressure_objects(
                "http://127.0.0.1:30000",
                workloads,
                label="unit-test",
                capacity_tokens=1000,
                target_rho=0.5,
            )

        # Exactly one request per filler -- never the old four-request
        # (seed/raw-register/fresh-register/reuse) cycle this replaces.
        self.assertEqual(len(session.post_calls), 3)
        for index, (_, payload) in enumerate(session.post_calls):
            # A plain dense_generate_payload, byte-for-byte (plus the
            # ``stream: True`` that ``timed_post`` merges in for every
            # request regardless of payload shape): no approx_kv
            # custom_params key anywhere in sampling_params.
            self.assertEqual(
                payload,
                {
                    **dense_generate_payload(workloads[index].target_prompt_ids),
                    "stream": True,
                },
            )
            self.assertNotIn("custom_params", payload["sampling_params"])

    def test_returns_expected_telemetry(self):
        workloads = self._filler_workloads(object_count=2, body_tokens=8)
        session = _FakeClientSession(self._dense_zero_cached_response())
        metrics_before = {
            "sglang:approx_kv_dense_fallback_total": 3.0,
            "sglang:evicted_tokens_total": 100.0,
            "sglang:kv_used_tokens": 400.0,
        }
        metrics_after = {
            "sglang:approx_kv_dense_fallback_total": 3.0,
            "sglang:evicted_tokens_total": 100.0,
            # used + evictable = 900 -- proves observed_rho_after_pressure
            # sums both gauges, not just kv_used_tokens, within the full
            # register_eviction_pressure_objects call path (not merely
            # observed_rho's own isolated unit tests).
            "sglang:kv_used_tokens": 600.0,
            "sglang:kv_evictable_tokens": 300.0,
        }

        with unittest.mock.patch(
            "aiohttp.ClientSession", return_value=session
        ), unittest.mock.patch(
            "benchmark.approx_kv.run_phase4_cachetune_canary.metric_snapshot",
            side_effect=[metrics_before, metrics_after],
        ):
            result = register_eviction_pressure_objects(
                "http://127.0.0.1:30000",
                workloads,
                label="unit-test",
                capacity_tokens=1000,
                target_rho=1.5,
            )

        self.assertEqual(len(session.post_calls), 2)
        self.assertEqual(result["object_count"], 2)
        self.assertEqual(result["total_pressure_tokens"], 16)
        self.assertEqual(result["target_rho"], 1.5)
        self.assertEqual(result["capacity_tokens"], 1000)
        self.assertAlmostEqual(result["observed_rho_after_pressure"], 0.9)
        # 16 nominal tokens is well within capacity_tokens=1000, so the
        # new evicted_tokens_total_delta assertion never engages here --
        # a zero delta is expected and fine.
        self.assertEqual(result["evicted_tokens_total_delta"], 0.0)
        self.assertEqual(result["dense_fallback_total_delta"], 0.0)
        self.assertEqual(result["metrics_before"], metrics_before)
        self.assertEqual(result["metrics_after"], metrics_after)

    def test_raises_when_dense_fallback_delta_is_nonzero(self):
        # A plain dense request carries no approx_kv metadata and
        # should never be able to move this CacheTune-reuse-specific
        # counter at all; a nonzero delta here is a real defect that
        # must never be treated as a harmless, ignorable detail.
        workloads = self._filler_workloads(object_count=1, body_tokens=8)
        session = _FakeClientSession(self._dense_zero_cached_response())
        metrics_before = {"sglang:approx_kv_dense_fallback_total": 3.0}
        metrics_after = {"sglang:approx_kv_dense_fallback_total": 4.0}

        with unittest.mock.patch(
            "aiohttp.ClientSession", return_value=session
        ), unittest.mock.patch(
            "benchmark.approx_kv.run_phase4_cachetune_canary.metric_snapshot",
            side_effect=[metrics_before, metrics_after],
        ):
            with self.assertRaises(RuntimeError) as ctx:
                register_eviction_pressure_objects(
                    "http://127.0.0.1:30000",
                    workloads,
                    label="unit-test",
                    capacity_tokens=1000,
                    target_rho=1.5,
                )
        message = str(ctx.exception)
        self.assertIn("unit-test", message)
        self.assertIn("1 plain-dense eviction-pressure", message)

    def test_does_not_raise_when_dense_fallback_delta_is_zero_despite_other_deltas(
        self,
    ):
        # A positive control alongside the raise-test above: OTHER
        # counters (e.g. evicted_tokens_total) moving is expected and
        # fine; only a nonzero dense_fallback delta must raise. Nominal
        # pressure tokens (8) stay within capacity_tokens=1000 here, so
        # the eviction assertion never engages either.
        workloads = self._filler_workloads(object_count=1, body_tokens=8)
        session = _FakeClientSession(self._dense_zero_cached_response())
        metrics_before = {
            "sglang:approx_kv_dense_fallback_total": 0.0,
            "sglang:evicted_tokens_total": 0.0,
            "sglang:kv_used_tokens": 0.0,
        }
        metrics_after = {
            "sglang:approx_kv_dense_fallback_total": 0.0,
            "sglang:evicted_tokens_total": 8.0,
            "sglang:kv_used_tokens": 8.0,
            "sglang:kv_evictable_tokens": 0.0,
        }

        with unittest.mock.patch(
            "aiohttp.ClientSession", return_value=session
        ), unittest.mock.patch(
            "benchmark.approx_kv.run_phase4_cachetune_canary.metric_snapshot",
            side_effect=[metrics_before, metrics_after],
        ):
            result = register_eviction_pressure_objects(
                "http://127.0.0.1:30000",
                workloads,
                label="unit-test",
                capacity_tokens=1000,
                target_rho=1.5,
            )
        self.assertEqual(result["dense_fallback_total_delta"], 0.0)
        self.assertEqual(result["evicted_tokens_total_delta"], 8.0)
        self.assertAlmostEqual(result["observed_rho_after_pressure"], 0.008)

    def test_raises_when_pressure_exceeds_capacity_but_no_eviction_observed(self):
        # Core NEW safety invariant this fix adds: when the fillers'
        # own nominal footprint ALONE already exceeds this setting's
        # idle capacity, later fillers necessarily had to evict earlier
        # ones just to fit -- a zero evicted_tokens_total delta despite
        # that must raise, never be silently trusted as "genuine
        # pressure occurred".
        workloads = self._filler_workloads(object_count=3, body_tokens=50)
        # total_pressure_tokens = 150 > capacity_tokens = 100.
        session = _FakeClientSession(self._dense_zero_cached_response())
        metrics_before = {
            "sglang:approx_kv_dense_fallback_total": 0.0,
            "sglang:evicted_tokens_total": 20.0,
        }
        metrics_after = {
            "sglang:approx_kv_dense_fallback_total": 0.0,
            "sglang:evicted_tokens_total": 20.0,  # unchanged: no eviction.
        }

        with unittest.mock.patch(
            "aiohttp.ClientSession", return_value=session
        ), unittest.mock.patch(
            "benchmark.approx_kv.run_phase4_cachetune_canary.metric_snapshot",
            side_effect=[metrics_before, metrics_after],
        ):
            with self.assertRaises(RuntimeError) as ctx:
                register_eviction_pressure_objects(
                    "http://127.0.0.1:30000",
                    workloads,
                    label="unit-test",
                    capacity_tokens=100,
                    target_rho=1.5,
                )
        message = str(ctx.exception)
        self.assertIn("unit-test", message)
        self.assertIn("150", message)
        self.assertIn("100", message)

    def test_does_not_raise_when_pressure_exceeds_capacity_and_eviction_observed(
        self,
    ):
        # Positive control for the above: the exact same
        # exceeds-capacity configuration, but a genuine nonzero eviction
        # delta this time -- must not raise.
        workloads = self._filler_workloads(object_count=3, body_tokens=50)
        session = _FakeClientSession(self._dense_zero_cached_response())
        metrics_before = {
            "sglang:approx_kv_dense_fallback_total": 0.0,
            "sglang:evicted_tokens_total": 20.0,
            "sglang:kv_used_tokens": 100.0,
        }
        metrics_after = {
            "sglang:approx_kv_dense_fallback_total": 0.0,
            "sglang:evicted_tokens_total": 90.0,
            "sglang:kv_used_tokens": 100.0,
            "sglang:kv_evictable_tokens": 0.0,
        }

        with unittest.mock.patch(
            "aiohttp.ClientSession", return_value=session
        ), unittest.mock.patch(
            "benchmark.approx_kv.run_phase4_cachetune_canary.metric_snapshot",
            side_effect=[metrics_before, metrics_after],
        ):
            result = register_eviction_pressure_objects(
                "http://127.0.0.1:30000",
                workloads,
                label="unit-test",
                capacity_tokens=100,
                target_rho=1.5,
            )
        self.assertEqual(result["evicted_tokens_total_delta"], 70.0)

    def test_does_not_raise_when_pressure_within_capacity_despite_zero_eviction(self):
        # target_rho <= 1 (nominal pressure stays within capacity):
        # fillers may legitimately all coexist without any eviction at
        # all -- the eviction assertion must never engage here.
        workloads = self._filler_workloads(object_count=1, body_tokens=8)
        session = _FakeClientSession(self._dense_zero_cached_response())
        metrics_before = {
            "sglang:approx_kv_dense_fallback_total": 0.0,
            "sglang:evicted_tokens_total": 0.0,
            "sglang:kv_used_tokens": 0.0,
        }
        metrics_after = {
            "sglang:approx_kv_dense_fallback_total": 0.0,
            "sglang:evicted_tokens_total": 0.0,
            "sglang:kv_used_tokens": 8.0,
            "sglang:kv_evictable_tokens": 0.0,
        }

        with unittest.mock.patch(
            "aiohttp.ClientSession", return_value=session
        ), unittest.mock.patch(
            "benchmark.approx_kv.run_phase4_cachetune_canary.metric_snapshot",
            side_effect=[metrics_before, metrics_after],
        ):
            result = register_eviction_pressure_objects(
                "http://127.0.0.1:30000",
                workloads,
                label="unit-test",
                capacity_tokens=1000,
                target_rho=0.5,
            )
        self.assertEqual(result["evicted_tokens_total_delta"], 0.0)

    def test_raises_when_a_filler_reports_nonzero_cached_tokens(self):
        # Pairwise isolation failing to hold (or this setting's own
        # pre-pressure flush not actually having happened) would
        # surface as a filler's first appearance reporting a nonzero
        # cached_tokens -- require_cached_tokens must catch this
        # immediately, never silently accept a stray exact-prefix hit.
        workloads = self._filler_workloads(object_count=1, body_tokens=8)
        chunk = {"meta_info": {"finish_reason": {"type": "length"}, "cached_tokens": 6}}
        response = _FakeStreamResponse([_sse_data_line(chunk), _SSE_DONE_LINE])
        session = _FakeClientSession(response)

        with unittest.mock.patch(
            "aiohttp.ClientSession", return_value=session
        ), unittest.mock.patch(
            "benchmark.approx_kv.run_phase4_cachetune_canary.metric_snapshot",
            side_effect=[{}, {}],
        ):
            with self.assertRaises(RuntimeError) as ctx:
                register_eviction_pressure_objects(
                    "http://127.0.0.1:30000",
                    workloads,
                    label="unit-test",
                    capacity_tokens=1000,
                    target_rho=1.5,
                )
        self.assertIn("pressure-filler[0]", str(ctx.exception))
        self.assertIn("cached_tokens=6", str(ctx.exception))

    def test_already_pinned_tokens_defaults_to_zero_in_telemetry(self):
        workloads = self._filler_workloads(object_count=1, body_tokens=8)
        session = _FakeClientSession(self._dense_zero_cached_response())
        metrics_before = {
            "sglang:evicted_tokens_total": 0.0,
            "sglang:kv_used_tokens": 0.0,
        }
        metrics_after = {
            "sglang:evicted_tokens_total": 0.0,
            "sglang:kv_used_tokens": 8.0,
            "sglang:kv_evictable_tokens": 0.0,
        }

        with unittest.mock.patch(
            "aiohttp.ClientSession", return_value=session
        ), unittest.mock.patch(
            "benchmark.approx_kv.run_phase4_cachetune_canary.metric_snapshot",
            side_effect=[metrics_before, metrics_after],
        ):
            result = register_eviction_pressure_objects(
                "http://127.0.0.1:30000",
                workloads,
                label="unit-test",
                capacity_tokens=1000,
                target_rho=0.5,
            )
        self.assertEqual(result["already_pinned_tokens"], 0)

    def test_returns_already_pinned_tokens_verbatim_in_telemetry(self):
        workloads = self._filler_workloads(object_count=1, body_tokens=8)
        session = _FakeClientSession(self._dense_zero_cached_response())
        metrics_before = {
            "sglang:evicted_tokens_total": 0.0,
            "sglang:kv_used_tokens": 123.0,
        }
        metrics_after = {
            "sglang:evicted_tokens_total": 0.0,
            "sglang:kv_used_tokens": 131.0,
            "sglang:kv_evictable_tokens": 0.0,
        }

        with unittest.mock.patch(
            "aiohttp.ClientSession", return_value=session
        ), unittest.mock.patch(
            "benchmark.approx_kv.run_phase4_cachetune_canary.metric_snapshot",
            side_effect=[metrics_before, metrics_after],
        ):
            result = register_eviction_pressure_objects(
                "http://127.0.0.1:30000",
                workloads,
                label="unit-test",
                capacity_tokens=1000,
                target_rho=0.5,
                already_pinned_tokens=123,
            )
        self.assertEqual(result["already_pinned_tokens"], 123)

    def test_rejects_negative_already_pinned_tokens(self):
        workloads = self._filler_workloads(object_count=1, body_tokens=8)
        with self.assertRaises(ValueError):
            register_eviction_pressure_objects(
                "http://127.0.0.1:30000",
                workloads,
                label="unit-test",
                capacity_tokens=1000,
                target_rho=1.5,
                already_pinned_tokens=-1,
            )

    def test_raises_immediately_when_already_pinned_tokens_meets_or_exceeds_capacity(
        self,
    ):
        # Raised BEFORE any HTTP request is ever sent (the setting's own
        # raw+fresh source-registration footprint alone already consumes
        # the entire measured pool -- an unrecoverable misconfiguration,
        # never something a filler HTTP loop should even attempt). No
        # aiohttp/metric_snapshot mock is installed on purpose: if this
        # ever regressed into performing I/O first, this test would fail
        # with an unmocked-network error rather than the expected
        # ValueError.
        workloads = self._filler_workloads(object_count=1, body_tokens=8)
        with self.assertRaises(ValueError) as ctx:
            register_eviction_pressure_objects(
                "http://127.0.0.1:30000",
                workloads,
                label="unit-test",
                capacity_tokens=100,
                target_rho=1.5,
                already_pinned_tokens=100,
            )
        self.assertIn("unit-test", str(ctx.exception))
        self.assertIn("100", str(ctx.exception))

    def test_already_pinned_tokens_reduces_effective_headroom_for_the_eviction_assertion(
        self,
    ):
        # Same nominal capacity_tokens and filler workloads as the
        # sibling "does not raise" test directly below -- only
        # already_pinned_tokens differs, shrinking the TRUE evictable
        # headroom (capacity_tokens - already_pinned_tokens) below the
        # fillers' own nominal footprint. A zero eviction delta despite
        # that must still raise -- this is the fix for a real SM75
        # ordering bug at target_rho=2 where the setting's own raw/fresh
        # source-registration footprint was never accounted for here.
        workloads = self._filler_workloads(object_count=3, body_tokens=50)
        # eviction_pressure_total_tokens sums body_tokens only (a
        # deliberate lower-bound floor -- see that function's own
        # docstring): total_pressure_tokens = 3 * 50 = 150.
        session = _FakeClientSession(self._dense_zero_cached_response())
        metrics_before = {
            "sglang:approx_kv_dense_fallback_total": 0.0,
            "sglang:evicted_tokens_total": 20.0,
        }
        metrics_after = {
            "sglang:approx_kv_dense_fallback_total": 0.0,
            "sglang:evicted_tokens_total": 20.0,  # unchanged: no eviction.
        }

        with unittest.mock.patch(
            "aiohttp.ClientSession", return_value=session
        ), unittest.mock.patch(
            "benchmark.approx_kv.run_phase4_cachetune_canary.metric_snapshot",
            side_effect=[metrics_before, metrics_after],
        ):
            with self.assertRaises(RuntimeError) as ctx:
                register_eviction_pressure_objects(
                    "http://127.0.0.1:30000",
                    workloads,
                    label="unit-test",
                    capacity_tokens=1000,
                    target_rho=1.5,
                    already_pinned_tokens=900,  # effective_available=100 < 150
                )
        message = str(ctx.exception)
        self.assertIn("150", message)
        self.assertIn("100", message)

    def test_zero_already_pinned_tokens_does_not_raise_for_the_same_configuration(self):
        # The exact same capacity_tokens/workloads as the test directly
        # above -- but already_pinned_tokens=0 (the default) means the
        # TRUE evictable headroom is the full capacity_tokens=1000,
        # comfortably above the fillers' 150 nominal (body-only) tokens,
        # so the eviction assertion never engages.
        workloads = self._filler_workloads(object_count=3, body_tokens=50)
        session = _FakeClientSession(self._dense_zero_cached_response())
        metrics_before = {
            "sglang:approx_kv_dense_fallback_total": 0.0,
            "sglang:evicted_tokens_total": 20.0,
            "sglang:kv_used_tokens": 0.0,
        }
        metrics_after = {
            "sglang:approx_kv_dense_fallback_total": 0.0,
            "sglang:evicted_tokens_total": 20.0,
            "sglang:kv_used_tokens": 150.0,
            "sglang:kv_evictable_tokens": 0.0,
        }

        with unittest.mock.patch(
            "aiohttp.ClientSession", return_value=session
        ), unittest.mock.patch(
            "benchmark.approx_kv.run_phase4_cachetune_canary.metric_snapshot",
            side_effect=[metrics_before, metrics_after],
        ):
            result = register_eviction_pressure_objects(
                "http://127.0.0.1:30000",
                workloads,
                label="unit-test",
                capacity_tokens=1000,
                target_rho=1.5,
            )
        self.assertEqual(result["already_pinned_tokens"], 0)
        self.assertEqual(result["evicted_tokens_total_delta"], 0.0)


class TestRunNonPrefixSettingWithEvictionPressure(unittest.TestCase):
    """``run_non_prefix_setting``'s own pressure-phase wiring, exercised
    end to end (not just its constituent functions in isolation): every
    ROUND -- the one discarded warmup round AND every formal repeat,
    each via ``run_independent_round`` -- must flush, complete its OWN
    setup (seed-head + raw-register + fresh-register, via
    ``register_round_setup``) BEFORE reverse-computing or sending any
    eviction-pressure filler sized from THAT round's own post-setup
    measurement, then re-seed the target head (via
    ``ensure_target_head_resident``) before that SAME round's own
    ``run_target_reuse`` call.

    This is the current (fully-independent-round) architecture, fixing
    a real SM75 ``target_rho=2`` ``MemoryError``: an earlier design
    shared ONE raw registration across the discarded warmup and every
    formal repeat, re-registering only fresh (and re-sizing pressure)
    on each repeat -- see ``run_independent_round``'s own docstring for
    the full root-cause account. An even earlier design before that
    proved and required sending pressure fillers BEFORE any source
    setup at all, the first of the two real SM75 bugs this module's
    test suite has now fixed.
    """

    def _workload(self):
        return build_non_prefix_segment_workload(
            FakeTokenizer(),
            body_tokens=4,
            head_tokens=3,
            tail_tokens=NON_PREFIX_TAIL_TOKENS,
            salt="unit-test-pressure-wiring",
        )

    @staticmethod
    def _response(cached_tokens):
        chunk = {
            "meta_info": {
                "finish_reason": {"type": "length"},
                "cached_tokens": cached_tokens,
            }
        }
        return _FakeStreamResponse([_sse_data_line(chunk), _SSE_DONE_LINE])

    @staticmethod
    def _labeled_metric_snapshot(call_order: list[str], snapshots: list[dict]):
        state = {"count": 0}

        def fake_metric_snapshot(base_url):
            index = state["count"]
            state["count"] += 1
            call_order.append(f"metric_snapshot[{index}]")
            return snapshots[index]

        return fake_metric_snapshot

    def _one_round_responses_and_labels(
        self, workload, round_name: str, filler_count: int, *, head_reseed_hit: bool
    ):
        """One ROUND's own complete HTTP call sequence: setup (seed +
        raw + fresh, always 3 calls for this single-chunk workload) ->
        ``filler_count`` plain-dense pressure fillers -> head re-seed ->
        reuse. Used identically to build both the discarded warmup
        round's own sequence and every formal round's own sequence --
        never shared or reused between rounds."""
        reuse_cached = workload.body_start_in_target + workload.body_tokens
        responses = (
            [self._response(0)]  # setup: seed target_head
            + [self._response(0)]  # setup: register raw (1 chunk)
            + [self._response(workload.body_start_in_target)]  # setup: register fresh
            + [self._response(0)] * filler_count  # pressure fillers
            + [
                self._response(len(workload.target_head_ids) if head_reseed_hit else 0)
            ]  # head re-seed
            + [self._response(reuse_cached)]  # reuse
        )
        labels = (
            [
                f"{round_name}_setup_seed",
                f"{round_name}_setup_raw_chunk0",
                f"{round_name}_setup_fresh_chunk0",
            ]
            + [f"{round_name}_pressure_filler{i}" for i in range(filler_count)]
            + [f"{round_name}_head_reseed", f"{round_name}_reuse"]
        )
        return responses, labels

    def _run_low_pressure_setting(self, call_order: list[str]):
        """Shared fixture for the ordering/telemetry tests below: ONE
        discarded warmup round plus ONE formal repeat, each
        independently sized to a single filler (target_rho=0.1, low
        pressure, no genuine eviction expected), each with its own
        already_pinned_tokens=4 threaded from THAT round's own measured
        post-setup delta, and a head that survives each round's own
        pressure phase (a hit, cached_tokens == full head length)."""
        workload = self._workload()
        warmup_responses, warmup_labels = self._one_round_responses_and_labels(
            workload, "warmup", filler_count=1, head_reseed_hit=True
        )
        formal_responses, formal_labels = self._one_round_responses_and_labels(
            workload, "formal", filler_count=1, head_reseed_hit=True
        )
        responses = warmup_responses + formal_responses
        labels = warmup_labels + formal_labels
        session = _LabeledSequencedFakeClientSession(responses, labels, call_order)

        # capacity_tokens=100 (max_total_num_tokens fallback), re-read
        # fresh from EVERY round's own round_start snapshot;
        # already_pinned_tokens = 4.0 - 0.0 = 4 for EACH round (that
        # round's own measured post-setup delta); tokens_per_filler =
        # pressure_filler_head_tokens(2) + pressure_filler_body_tokens(4)
        # = 6; target_rho=0.1 -> target_total=10, remaining=10-4=6 ->
        # ceil(6/6)=1 filler each round (never still 2, the
        # already_pinned_tokens=0 count -- proof the already-pinned
        # footprint genuinely reduces the filler count every round).
        def _one_round_snapshots():
            return [
                {"sglang:max_total_num_tokens": 100.0, "sglang:kv_used_tokens": 0.0},
                {"sglang:kv_used_tokens": 4.0},
                {
                    "sglang:evicted_tokens_total": 0.0,
                    "sglang:kv_used_tokens": 4.0,
                    "sglang:approx_kv_dense_fallback_total": 0.0,
                },
                {
                    "sglang:evicted_tokens_total": 0.0,
                    "sglang:kv_used_tokens": 10.0,
                    "sglang:kv_evictable_tokens": 0.0,
                    "sglang:approx_kv_dense_fallback_total": 0.0,
                },
                {
                    "sglang:kv_used_tokens": 20.0,
                    "sglang:kv_evictable_tokens": 0.0,
                    "sglang:evicted_tokens_total": 0.0,
                },
            ]

        snapshots = _one_round_snapshots() + _one_round_snapshots()

        with unittest.mock.patch(
            "aiohttp.ClientSession", return_value=session
        ), unittest.mock.patch(
            "urllib.request.urlopen", _labeled_flush_urlopen(call_order)
        ), unittest.mock.patch(
            "benchmark.approx_kv.run_phase4_cachetune_canary.metric_snapshot",
            side_effect=self._labeled_metric_snapshot(call_order, snapshots),
        ) as metric_snapshot_mock:
            result = run_non_prefix_setting(
                base_url="http://127.0.0.1:30000",
                tokenizer=FakeTokenizer(),
                workload=workload,
                raw_hash="cachetune-raw:unit-test-pressure-wiring",
                fresh_hash="cachetune-fresh:unit-test-pressure-wiring",
                model_fingerprint="qwen3-0.6b-sm75",
                cache_dtype="fp16",
                repeats=1,
                label="unit-test",
                max_chunk_tokens=512,
                target_rho=0.1,
                pressure_filler_head_tokens=2,
                pressure_filler_body_tokens=4,
            )

        return workload, session, result, metric_snapshot_mock

    def test_every_round_flushes_before_its_own_setup(self):
        call_order: list[str] = []
        _workload, _session, _result, _mock = self._run_low_pressure_setting(call_order)

        flush0 = call_order.index("flush[0]")
        flush1 = call_order.index("flush[1]")
        warmup_setup = call_order.index("warmup_setup_seed")
        formal_setup = call_order.index("formal_setup_seed")
        warmup_reuse = call_order.index("warmup_reuse")

        # Two independent flushes -- one per round -- never just once
        # for the whole setting.
        self.assertLess(flush0, warmup_setup)
        # The second flush happens strictly AFTER the warmup round's own
        # reuse call completes, and strictly BEFORE the formal round's
        # own setup begins: the formal round never depends on anything
        # left resident by the warmup round.
        self.assertLess(warmup_reuse, flush1)
        self.assertLess(flush1, formal_setup)

    def test_setup_seed_raw_fresh_all_precede_that_same_rounds_pressure_phase(self):
        call_order: list[str] = []
        _workload, session, result, _mock = self._run_low_pressure_setting(call_order)

        self.assertEqual(len(session.post_calls), 12)
        for round_name in ("warmup", "formal"):
            setup_calls = [
                index
                for index, label in enumerate(call_order)
                if label.startswith(f"{round_name}_setup")
            ]
            pressure_call = call_order.index(f"{round_name}_pressure_filler0")
            self.assertEqual(len(setup_calls), 3)
            self.assertTrue(all(index < pressure_call for index in setup_calls))

        self.assertIsNotNone(result["pressure_phase"])
        self.assertEqual(result["pressure_phase"]["object_count"], 1)

    def test_fresh_is_registered_exactly_once_per_round_never_again_after_pressure(
        self,
    ):
        call_order: list[str] = []
        _workload, _session, _result, _mock = self._run_low_pressure_setting(call_order)

        fresh_labels = [
            label for label in call_order if label.endswith("_setup_fresh_chunk0")
        ]
        # Exactly one fresh-register call per round (warmup + 1 formal
        # repeat == 2 total) -- never re-registered separately after
        # that SAME round's own pressure phase, and never shared across
        # rounds.
        self.assertEqual(
            fresh_labels, ["warmup_setup_fresh_chunk0", "formal_setup_fresh_chunk0"]
        )
        for round_name in ("warmup", "formal"):
            fresh_index = call_order.index(f"{round_name}_setup_fresh_chunk0")
            pressure_index = call_order.index(f"{round_name}_pressure_filler0")
            reuse_index = call_order.index(f"{round_name}_reuse")
            self.assertLess(fresh_index, pressure_index)
            self.assertLess(pressure_index, reuse_index)

    def test_head_reseed_runs_after_pressure_and_before_reuse_each_round(self):
        call_order: list[str] = []
        self._run_low_pressure_setting(call_order)

        for round_name in ("warmup", "formal"):
            pressure_call = call_order.index(f"{round_name}_pressure_filler0")
            reseed_call = call_order.index(f"{round_name}_head_reseed")
            reuse_call = call_order.index(f"{round_name}_reuse")
            self.assertLess(pressure_call, reseed_call)
            self.assertLess(reseed_call, reuse_call)

    def test_warmup_round_precedes_every_formal_repeat(self):
        call_order: list[str] = []
        self._run_low_pressure_setting(call_order)

        warmup_call = call_order.index("warmup_reuse")
        formal_call = call_order.index("formal_reuse")
        self.assertLess(warmup_call, formal_call)

    def test_metric_snapshot_is_called_five_times_per_round_with_pressure(self):
        call_order: list[str] = []
        _workload, _session, _result, metric_snapshot_mock = (
            self._run_low_pressure_setting(call_order)
        )
        # 5 snapshots/round (round_start, after_setup, pressure_before,
        # pressure_after, round_end) x 2 rounds (warmup + 1 formal
        # repeat) == 10 -- never just 5 total (that would mean the
        # pressure phase and its own sizing snapshot only ran once for
        # the whole setting, the exact bug this architecture fixes).
        self.assertEqual(metric_snapshot_mock.call_count, 10)

    def test_already_pinned_tokens_is_threaded_from_the_measured_post_setup_delta(
        self,
    ):
        call_order: list[str] = []
        _workload, _session, result, _mock = self._run_low_pressure_setting(call_order)

        self.assertEqual(result["already_pinned_tokens"], 4)
        self.assertEqual(result["pressure_phase"]["already_pinned_tokens"], 4)
        # already_pinned_tokens=4 -> remaining=10-4=6 -> ceil(6/6)=1
        # filler, never the already_pinned_tokens=0 count of 2 (see
        # eviction_pressure_filler_count_for_rho's own dedicated tests
        # for that formula in isolation).
        self.assertEqual(result["pressure_phase"]["object_count"], 1)
        # The formal repeat's own round is also directly visible (in
        # full) via the new "rounds" key.
        self.assertEqual(len(result["rounds"]), 1)
        self.assertEqual(result["rounds"][0]["already_pinned_tokens"], 4)

    def test_returns_head_reseed_telemetry(self):
        call_order: list[str] = []
        _workload, _session, result, _mock = self._run_low_pressure_setting(call_order)

        self.assertIsNotNone(result["head_reseed_after_pressure"])
        self.assertFalse(
            result["head_reseed_after_pressure"]["was_evicted_by_pressure"]
        )
        self.assertEqual(result["head_reseed_after_pressure"]["cached_tokens"], 3)

    def test_pressure_fillers_are_plain_dense_requests_distinct_from_the_main_head(
        self,
    ):
        call_order: list[str] = []
        workload, session, _result, _mock = self._run_low_pressure_setting(call_order)

        # session.post_calls is HTTP-only (no metric_snapshot/flush
        # entries), unlike the shared call_order list -- look the
        # offset up via the session's own parallel _labels list, never
        # via call_order.index (which would be off, since call_order
        # interleaves metric_snapshot and flush labels too).
        for round_name in ("warmup", "formal"):
            pressure_index = session._labels.index(f"{round_name}_pressure_filler0")
            _, pressure_payload = session.post_calls[pressure_index]
            self.assertNotIn("custom_params", pressure_payload["sampling_params"])
            self.assertNotEqual(
                pressure_payload["input_ids"], list(workload.target_head_ids)
            )

    def test_each_round_computes_its_own_independent_already_pinned_tokens(self):
        # The direct "no cross-round store/state reuse" proof: three
        # rounds (warmup + 2 formal repeats), each with a DIFFERENT
        # already_pinned_tokens (4, 6, 8) from THAT round's own
        # post-setup measurement alone. If a bug ever let a later
        # round's computation inherit or accumulate an earlier round's
        # own pinned footprint (e.g. reusing round 0's snapshot pair, or
        # summing deltas across rounds), the values asserted below would
        # not match -- they can ONLY match if every round freshly
        # re-measures and re-computes from its OWN flush-reset baseline.
        workload = self._workload()
        call_order: list[str] = []
        already_pinned_by_round = [4.0, 6.0, 8.0]
        round_names = ["warmup", "formal0", "formal1"]
        responses: list = []
        labels: list[str] = []
        snapshots: list[dict] = []
        for round_name, pinned in zip(round_names, already_pinned_by_round):
            # target_rho=0.1 -> target_total=10; remaining = 10 - pinned
            # is always in (0, 6] for pinned in {4, 6, 8}, so
            # eviction_pressure_filler_count_for_rho always returns 1 --
            # keeping the HTTP call shape uniform across rounds while
            # still varying the underlying already_pinned computation.
            round_responses, round_labels = self._one_round_responses_and_labels(
                workload, round_name, filler_count=1, head_reseed_hit=True
            )
            responses += round_responses
            labels += round_labels
            snapshots += [
                {"sglang:max_total_num_tokens": 100.0, "sglang:kv_used_tokens": 0.0},
                {"sglang:kv_used_tokens": pinned},
                {
                    "sglang:evicted_tokens_total": 0.0,
                    "sglang:kv_used_tokens": pinned,
                    "sglang:approx_kv_dense_fallback_total": 0.0,
                },
                {
                    "sglang:evicted_tokens_total": 0.0,
                    "sglang:kv_used_tokens": pinned + 6.0,
                    "sglang:kv_evictable_tokens": 0.0,
                    "sglang:approx_kv_dense_fallback_total": 0.0,
                },
                {
                    "sglang:kv_used_tokens": pinned + 10.0,
                    "sglang:kv_evictable_tokens": 0.0,
                    "sglang:evicted_tokens_total": 0.0,
                },
            ]
        session = _LabeledSequencedFakeClientSession(responses, labels, call_order)

        with unittest.mock.patch(
            "aiohttp.ClientSession", return_value=session
        ), unittest.mock.patch(
            "urllib.request.urlopen", _labeled_flush_urlopen(call_order)
        ), unittest.mock.patch(
            "benchmark.approx_kv.run_phase4_cachetune_canary.metric_snapshot",
            side_effect=self._labeled_metric_snapshot(call_order, snapshots),
        ):
            result = run_non_prefix_setting(
                base_url="http://127.0.0.1:30000",
                tokenizer=FakeTokenizer(),
                workload=workload,
                raw_hash="cachetune-raw:unit-test-round-independence",
                fresh_hash="cachetune-fresh:unit-test-round-independence",
                model_fingerprint="qwen3-0.6b-sm75",
                cache_dtype="fp16",
                repeats=2,
                label="unit-test",
                max_chunk_tokens=512,
                target_rho=0.1,
                pressure_filler_head_tokens=2,
                pressure_filler_body_tokens=4,
            )

        # "rounds" holds only the 2 FORMAL repeats, excluding the
        # discarded warmup round -- each with its OWN already_pinned_
        # tokens, exactly matching that round's own snapshot pair
        # (6 for the first formal repeat, 8 for the second), never the
        # warmup's own 4, and never an accumulated/inherited value.
        self.assertEqual(len(result["rounds"]), 2)
        self.assertEqual(result["rounds"][0]["already_pinned_tokens"], 6)
        self.assertEqual(result["rounds"][1]["already_pinned_tokens"], 8)
        # The setting-level aggregate reports the LAST formal round's
        # own value.
        self.assertEqual(result["already_pinned_tokens"], 8)
        # Every round -- including both formal repeats -- independently
        # computed the SAME filler_count=1 from its own (different)
        # pinned footprint, never silently drifting or accumulating.
        self.assertEqual(result["rounds"][0]["pressure_phase"]["object_count"], 1)
        self.assertEqual(result["rounds"][1]["pressure_phase"]["object_count"], 1)

    def test_high_pressure_capacity_fake_full_body_restored_genuine_eviction_zero_fallback(
        self,
    ):
        # The explicitly required "high-pressure capacity fake": a
        # target_rho=2-equivalent configuration (net of
        # already_pinned_tokens=4, remaining=196 -> ceil(196/6)=33
        # fillers, deliberately exceeding the pool's TRUE evictable
        # headroom of 96) must still: (1) fully restore the target's
        # head+body on EVERY round's own reuse call (never merely the
        # head), (2) show a genuine evicted_tokens_total_delta > 0
        # during EVERY round's own pressure phase, and (3) never move
        # the dense_fallback counter at all -- proving the per-round
        # setup-before-pressure design remains correct even under
        # genuine, heavy, REPEATED real eviction pressure (the
        # discarded warmup round AND the formal repeat each
        # independently re-derive and re-send their own 33-filler
        # batch from their own fresh post-setup measurement), not just
        # a single round or the low-pressure/no-eviction case the other
        # tests above exercise.
        workload = self._workload()
        reuse_cached = workload.body_start_in_target + workload.body_tokens
        filler_count = 33  # ceil((2.0 * 100 - 4) / 6)

        warmup_responses, warmup_labels = self._one_round_responses_and_labels(
            workload, "warmup", filler_count=filler_count, head_reseed_hit=False
        )
        formal_responses, formal_labels = self._one_round_responses_and_labels(
            workload, "formal", filler_count=filler_count, head_reseed_hit=False
        )
        responses = warmup_responses + formal_responses
        session = _SequencedFakeClientSession(responses)

        # sglang:kv_used_tokens is a Gauge -- it resets to 0 at the
        # start of EVERY round (right after that round's own flush).
        # sglang:evicted_tokens_total is a Counter -- it is monotonic
        # and carries on accumulating ACROSS rounds even though the
        # Gauge resets; the formal round's own round_start value
        # (150.0) below is exactly the warmup round's own round_end
        # value, never reset back to 0.0 by that intervening flush.
        #
        # kv_used_tokens stays pinned at 4.0 throughout pressure/reuse
        # (this round's own setup raw+fresh footprint, invisible to
        # Radix LRU eviction, never freed mid-round): the 33 completed
        # dense pressure fillers instead land in kv_evictable_tokens
        # (ordinary LRU-evictable exact-radix entries) -- exactly the
        # real SM75 distinction observed_rho's own fix accounts for.
        # pressure_after reaches the full capacity_tokens=100 (4 used +
        # 96 evictable): the fillers' own 198 nominal tokens (33 * 6)
        # exceed the true evictable headroom of 96, so genuine eviction
        # (130.0 counter delta) already reclaimed some of them to fit.
        # round_end settles at 95 (4 used + 91 evictable): the target's
        # own recovery-slot allocation evicted a further few filler
        # tokens to make room for its own restored body.
        warmup_snapshots = [
            {
                "sglang:max_total_num_tokens": 100.0,
                "sglang:kv_used_tokens": 0.0,
                "sglang:evicted_tokens_total": 0.0,
            },
            {"sglang:kv_used_tokens": 4.0},
            {
                "sglang:evicted_tokens_total": 0.0,
                "sglang:kv_used_tokens": 4.0,
                "sglang:kv_evictable_tokens": 0.0,
                "sglang:approx_kv_dense_fallback_total": 0.0,
            },
            {
                "sglang:evicted_tokens_total": 130.0,
                "sglang:kv_used_tokens": 4.0,
                "sglang:kv_evictable_tokens": 96.0,
                "sglang:approx_kv_dense_fallback_total": 0.0,
            },
            {
                "sglang:kv_used_tokens": 4.0,
                "sglang:kv_evictable_tokens": 91.0,
                "sglang:evicted_tokens_total": 150.0,
            },
        ]
        formal_snapshots = [
            {
                "sglang:max_total_num_tokens": 100.0,
                "sglang:kv_used_tokens": 0.0,
                "sglang:evicted_tokens_total": 150.0,
            },
            {"sglang:kv_used_tokens": 4.0},
            {
                "sglang:evicted_tokens_total": 150.0,
                "sglang:kv_used_tokens": 4.0,
                "sglang:kv_evictable_tokens": 0.0,
                "sglang:approx_kv_dense_fallback_total": 0.0,
            },
            {
                "sglang:evicted_tokens_total": 280.0,
                "sglang:kv_used_tokens": 4.0,
                "sglang:kv_evictable_tokens": 96.0,
                "sglang:approx_kv_dense_fallback_total": 0.0,
            },
            {
                "sglang:kv_used_tokens": 4.0,
                "sglang:kv_evictable_tokens": 91.0,
                "sglang:evicted_tokens_total": 300.0,
            },
        ]

        with unittest.mock.patch(
            "aiohttp.ClientSession", return_value=session
        ), unittest.mock.patch(
            "urllib.request.urlopen", _fake_flush_urlopen([])
        ), unittest.mock.patch(
            "benchmark.approx_kv.run_phase4_cachetune_canary.metric_snapshot",
            side_effect=warmup_snapshots + formal_snapshots,
        ):
            result = run_non_prefix_setting(
                base_url="http://127.0.0.1:30000",
                tokenizer=FakeTokenizer(),
                workload=workload,
                raw_hash="cachetune-raw:unit-test-high-pressure",
                fresh_hash="cachetune-fresh:unit-test-high-pressure",
                model_fingerprint="qwen3-0.6b-sm75",
                cache_dtype="fp16",
                repeats=1,
                label="unit-test",
                max_chunk_tokens=512,
                target_rho=2.0,
                pressure_filler_head_tokens=2,
                pressure_filler_body_tokens=4,
            )

        self.assertEqual(len(session.post_calls), len(responses))
        self.assertEqual(result["pressure_phase"]["object_count"], filler_count)
        # (1) Full head+body restored on the formal reuse call, despite
        # heavy pressure -- never merely the exact-match head.
        self.assertEqual(result["reuse_raw_samples"][-1]["cached_tokens"], reuse_cached)
        self.assertEqual(result["observed_cached_tokens_per_call"], [reuse_cached])
        # (2) Genuine device-pool eviction actually occurred during the
        # LAST formal round's own pressure phase -- not merely reported,
        # but the real Prometheus delta that pressure phase itself
        # asserts on.
        self.assertGreater(result["pressure_phase"]["evicted_tokens_total_delta"], 0)
        # ... and the setting-level aggregate delta spans the ENTIRE
        # last (only) formal round, start to end (150.0 -> 300.0 ==
        # 150.0) -- the pressure phase's own 130.0 contribution PLUS a
        # further 20.0 evicted by the target's own recovery-slot
        # allocation during that same round's reuse call, EXCLUDING the
        # discarded warmup round's own independent 150.0 contribution
        # entirely (metric_delta always spans the FIRST formal round's
        # own start through the LAST formal round's own end).
        self.assertEqual(
            result["pressure_and_target_evicted_tokens_total_delta"], 150.0
        )
        # (3) Zero dense-fallback throughout every round: a plain dense
        # filler request, a plain dense head re-seed, and a genuine
        # CacheTune repair all completed successfully with no fallback
        # anywhere, in EVERY round.
        self.assertEqual(result["pressure_phase"]["dense_fallback_total_delta"], 0.0)
        # The head re-seed guard tolerated the eviction (a miss) without
        # raising, in every round, and every subsequent reuse call still
        # succeeded against the freshly-reseeded head.
        self.assertTrue(result["head_reseed_after_pressure"]["was_evicted_by_pressure"])
        # Both rounds independently rebuilt the SAME 33-filler batch
        # from their own fresh post-setup measurement -- proof the
        # per-round pressure phase is not a one-time setup this
        # architecture accidentally still shares.
        self.assertEqual(len(result["rounds"]), 1)
        self.assertEqual(result["rounds"][0]["pressure_phase"]["object_count"], 33)
        # (4) The real SM75 regression this fix addresses: under this
        # genuine high-pressure configuration, kv_used_tokens ALONE
        # stays pinned at a small, constant 4 throughout (this round's
        # own setup footprint) while the 33 completed dense fillers
        # accumulate as kv_evictable_tokens instead -- an earlier,
        # buggy observed_rho read kv_used_tokens alone and would have
        # reported this as near-ZERO pressure (4 / 100 = 0.04) despite
        # target_rho=2. The corrected formula (kv_used_tokens PLUS
        # kv_evictable_tokens) reports the pool as genuinely fully
        # resident during the pressure phase (100 / 100 == 1.0) and
        # still highly resident after the target's own recovery-slot
        # allocation reclaims a few filler tokens (95 / 100 == 0.95) --
        # peak_rho_observed takes the greater of the two, 1.0.
        self.assertAlmostEqual(
            result["pressure_phase"]["observed_rho_after_pressure"], 1.0
        )
        self.assertAlmostEqual(result["observed_rho_after_target"], 0.95)
        self.assertAlmostEqual(result["peak_rho_observed"], 1.0)


def _fake_flush_urlopen(flush_urls: list) -> callable:
    """Build a ``urllib.request.urlopen`` fake for ``flush_exact_radix_
    cache``'s ``post_empty`` call, recording each flushed URL into
    ``flush_urls`` -- reused by every test below that drives
    ``run_exact_context_control_point`` end to end, so each one can
    assert exactly how many flushes happened without duplicating the
    same small fake-response boilerplate as ``TestFlushExactRadixCache``.
    """

    class _FakeUrlResponse:
        def read(self):
            return b"Cache flushed.\n"

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

    def fake_urlopen(request, timeout=None):
        flush_urls.append(request.full_url)
        return _FakeUrlResponse()

    return fake_urlopen


def _labeled_flush_urlopen(call_order: list[str]) -> callable:
    """Like ``_fake_flush_urlopen``, but appends a positional
    ``flush[N]`` label to a SHARED ``call_order`` list on every flush --
    the same list a ``_LabeledSequencedFakeClientSession`` (HTTP POST
    calls) and a labeled ``metric_snapshot`` fake (see
    ``TestRunNonPrefixSettingWithEvictionPressure``'s own
    ``_labeled_metric_snapshot``) also append to, so a single test can
    verify the RELATIVE order of flush, metric-snapshot, and POST calls
    all together -- proving every independent round's own flush truly
    is that round's very first action, strictly before that SAME
    round's own setup/pressure/reuse calls and strictly after the
    PREVIOUS round's own reuse call.
    """

    class _FakeUrlResponse:
        def read(self):
            return b"Cache flushed.\n"

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

    state = {"count": 0}

    def fake_urlopen(request, timeout=None):
        index = state["count"]
        state["count"] += 1
        call_order.append(f"flush[{index}]")
        return _FakeUrlResponse()

    return fake_urlopen


class TestRunExactContextControlPoint(unittest.TestCase):
    """The header=0 shape-sweep control point: an honest dense-only
    reference measurement, never a genuine CacheTune repair -- see the
    function's own docstring for why header=0 cannot build a
    ``NonPrefixSegmentWorkload`` at all (its ``__post_init__`` requires
    ``source_head_ids != target_head_ids``, impossible for empty
    heads)."""

    @staticmethod
    def _dense_response(cached_tokens=0, finish_type="length"):
        chunk = {
            "meta_info": {
                "finish_reason": {"type": finish_type},
                "cached_tokens": cached_tokens,
            }
        }
        return _FakeStreamResponse([_sse_data_line(chunk), _SSE_DONE_LINE])

    def test_rejects_non_positive_body_tokens(self):
        with self.assertRaises(ValueError):
            run_exact_context_control_point(
                base_url="http://127.0.0.1:30000",
                tokenizer=FakeTokenizer(),
                body_tokens=0,
                tail_tokens=1,
                salt="unit-test",
                repeats=2,
            )

    def test_rejects_repeats_below_one(self):
        with self.assertRaises(ValueError):
            run_exact_context_control_point(
                base_url="http://127.0.0.1:30000",
                tokenizer=FakeTokenizer(),
                body_tokens=8,
                tail_tokens=1,
                salt="unit-test",
                repeats=0,
            )

    def test_runs_one_flush_and_warmup_plus_repeats_flush_and_dense_pairs(self):
        repeats = 2
        responses = [self._dense_response() for _ in range(1 + repeats)]
        session = _SequencedFakeClientSession(responses)
        flush_urls: list[str] = []

        with unittest.mock.patch(
            "aiohttp.ClientSession", return_value=session
        ), unittest.mock.patch(
            "urllib.request.urlopen", _fake_flush_urlopen(flush_urls)
        ):
            result = run_exact_context_control_point(
                base_url="http://127.0.0.1:30000",
                tokenizer=FakeTokenizer(),
                body_tokens=8,
                tail_tokens=1,
                salt="unit-test",
                repeats=repeats,
            )

        # One flush ahead of the discarded warmup, one flush ahead of
        # each formal repeat -- 1 + repeats total, and never a flush
        # anywhere else (there is no register/reuse phase at all for
        # this control point).
        self.assertEqual(len(flush_urls), 1 + repeats)
        self.assertTrue(
            all(url.endswith("/flush_cache?timeout=30") for url in flush_urls)
        )
        # One discarded warmup generate call + `repeats` formal ones.
        self.assertEqual(len(session.post_calls), 1 + repeats)

        self.assertEqual(result["body_tokens"], 8)
        self.assertTrue(result["is_exact_context_control"])
        self.assertFalse(result["body_source_context_differs_from_target"])
        self.assertEqual(len(result["dense_raw_samples"]), repeats)
        self.assertEqual(len(result["dense_ms_samples"]), repeats)

    def test_dense_ms_samples_mirror_raw_sample_ttft(self):
        repeats = 3
        responses = [self._dense_response() for _ in range(1 + repeats)]
        session = _SequencedFakeClientSession(responses)

        with unittest.mock.patch(
            "aiohttp.ClientSession", return_value=session
        ), unittest.mock.patch("urllib.request.urlopen", _fake_flush_urlopen([])):
            result = run_exact_context_control_point(
                base_url="http://127.0.0.1:30000",
                tokenizer=FakeTokenizer(),
                body_tokens=8,
                tail_tokens=1,
                salt="unit-test",
                repeats=repeats,
            )

        self.assertEqual(
            result["dense_ms_samples"],
            [sample["ttft_ms"] for sample in result["dense_raw_samples"]],
        )
        self.assertAlmostEqual(
            result["dense_p50_ms"], statistics.median(result["dense_ms_samples"])
        )

    def test_cached_tokens_recorded_from_server_meta_info(self):
        repeats = 1
        responses = [
            self._dense_response(cached_tokens=0),
            self._dense_response(cached_tokens=0),
        ]
        session = _SequencedFakeClientSession(responses)

        with unittest.mock.patch(
            "aiohttp.ClientSession", return_value=session
        ), unittest.mock.patch("urllib.request.urlopen", _fake_flush_urlopen([])):
            result = run_exact_context_control_point(
                base_url="http://127.0.0.1:30000",
                tokenizer=FakeTokenizer(),
                body_tokens=8,
                tail_tokens=1,
                salt="unit-test",
                repeats=repeats,
            )

        self.assertEqual(result["dense_raw_samples"][0]["cached_tokens"], 0)

    def test_warmup_failure_to_finish_by_length_raises_and_is_not_swallowed(self):
        # The warmup response never finishes by length -- production
        # code must let this propagate (never silently discard it and
        # proceed as if the warmup succeeded).
        responses = [self._dense_response(finish_type="abort")]
        session = _SequencedFakeClientSession(responses)

        with unittest.mock.patch(
            "aiohttp.ClientSession", return_value=session
        ), unittest.mock.patch("urllib.request.urlopen", _fake_flush_urlopen([])):
            with self.assertRaises(RuntimeError) as ctx:
                run_exact_context_control_point(
                    base_url="http://127.0.0.1:30000",
                    tokenizer=FakeTokenizer(),
                    body_tokens=8,
                    tail_tokens=1,
                    salt="unit-test",
                    repeats=2,
                )
        self.assertIn("exact-context-control warmup", str(ctx.exception))

    def test_formal_repeat_failure_to_finish_by_length_raises(self):
        responses = [
            self._dense_response(),  # warmup succeeds
            self._dense_response(finish_type="abort"),  # formal repeat 1 fails
        ]
        session = _SequencedFakeClientSession(responses)

        with unittest.mock.patch(
            "aiohttp.ClientSession", return_value=session
        ), unittest.mock.patch("urllib.request.urlopen", _fake_flush_urlopen([])):
            with self.assertRaises(RuntimeError) as ctx:
                run_exact_context_control_point(
                    base_url="http://127.0.0.1:30000",
                    tokenizer=FakeTokenizer(),
                    body_tokens=8,
                    tail_tokens=1,
                    salt="unit-test",
                    repeats=2,
                )
        self.assertIn(
            "exact-context-control request did not finish", str(ctx.exception)
        )


class TestRunNonPrefixSettingChunkedFreshRegister(unittest.TestCase):
    """``run_non_prefix_setting``'s EVERY round -- the one discarded
    warmup round AND each of ``repeats`` formal repeats alike, all via
    ``run_independent_round`` -- must INDEPENDENTLY route its OWN seed +
    raw-register + fresh-register through ``register_body_chunks``: one
    independent ``/generate`` call per ``<= max_chunk_tokens`` chunk,
    never one oversized call spanning the entire body, and never reusing
    an earlier round's own raw/fresh registration.

    This is the current (fully-independent-round) architecture -- an
    earlier design had ``register_non_prefix_sources`` register raw ONCE
    for the whole setting and a since-removed ``run_reuse_once`` merely
    re-register fresh (chunked) on every repeat, reusing that one raw
    registration across warmup and every formal repeat; see
    ``run_independent_round``'s own docstring for the real SM75
    ``target_rho=2`` ``MemoryError`` this fixed. Uses ``target_rho=
    None`` (no eviction-pressure phase) to isolate this proof to the
    chunked-registration call structure across multiple independent
    rounds alone."""

    def _workload(self, body_tokens):
        return build_non_prefix_segment_workload(
            FakeTokenizer(),
            body_tokens=body_tokens,
            head_tokens=64,
            tail_tokens=NON_PREFIX_TAIL_TOKENS,
            salt=f"unit-test-run-non-prefix-chunked-{body_tokens}",
        )

    def _response(self, cached_tokens):
        chunk = {
            "meta_info": {
                "finish_reason": {"type": "length"},
                "cached_tokens": cached_tokens,
            }
        }
        return _FakeStreamResponse([_sse_data_line(chunk), _SSE_DONE_LINE])

    def _one_round_responses(self, workload, chunk_count, reuse_cached):
        """One ROUND's own complete response sequence: seed head ->
        ``chunk_count`` raw-register chunks -> ``chunk_count``
        fresh-register chunks -> one (unchunked) reuse call --
        IDENTICAL shape for the discarded warmup round and every formal
        repeat alike, since every round now independently performs its
        own full setup (never just the warmup)."""
        return (
            [self._response(0)]  # seed target_head
            + [self._response(0)] * chunk_count  # raw register chunks
            + [self._response(workload.body_start_in_target)]
            * chunk_count  # fresh register chunks
            + [self._response(reuse_cached)]  # reuse
        )

    def _round_names(self, repeats):
        return ["warmup"] + [f"formal{i}" for i in range(repeats)]

    def _round_snapshots(self):
        """One round's own 2 ``metric_snapshot`` calls (round_start,
        round_end) -- ``target_rho=None`` means no post-setup/pressure
        snapshots this round. round_end must carry both
        ``kv_used_tokens`` and ``kv_evictable_tokens`` -- it feeds
        ``observed_rho_after_target``, which sums the two (see
        ``observed_rho``'s own real-SM75-bug docstring)."""
        return [
            {"sglang:max_total_num_tokens": 10000.0, "sglang:kv_used_tokens": 100.0},
            {"sglang:kv_used_tokens": 200.0, "sglang:kv_evictable_tokens": 0.0},
        ]

    def _run_with_body(self, body_tokens, repeats):
        workload = self._workload(body_tokens)
        chunk_count = len(chunk_offsets(workload.body_tokens, 512))
        reuse_cached = workload.body_start_in_target + workload.body_tokens
        rounds = 1 + repeats  # discarded warmup + `repeats` formal

        responses = []
        for _ in range(rounds):
            responses += self._one_round_responses(workload, chunk_count, reuse_cached)
        session = _SequencedFakeClientSession(responses)

        snapshots = []
        for _ in range(rounds):
            snapshots += self._round_snapshots()

        with unittest.mock.patch(
            "aiohttp.ClientSession", return_value=session
        ), unittest.mock.patch(
            "urllib.request.urlopen", _fake_flush_urlopen([])
        ), unittest.mock.patch(
            "benchmark.approx_kv.run_phase4_cachetune_canary.metric_snapshot",
            side_effect=snapshots,
        ):
            result = run_non_prefix_setting(
                base_url="http://127.0.0.1:30000",
                tokenizer=FakeTokenizer(),
                workload=workload,
                raw_hash="cachetune-raw:unit-test-run",
                fresh_hash="cachetune-fresh:unit-test-run",
                model_fingerprint="qwen3-0.6b-sm75",
                cache_dtype="fp16",
                repeats=repeats,
                label="unit-test",
                max_chunk_tokens=512,
                target_rho=None,
                pressure_filler_head_tokens=6,
                pressure_filler_body_tokens=8,
            )
        return workload, chunk_count, reuse_cached, session, result

    def test_body_1024_two_repeats_every_round_issues_two_raw_and_two_fresh_chunk_calls(
        self,
    ):
        repeats = 2
        workload, chunk_count, reuse_cached, session, result = self._run_with_body(
            body_tokens=1024, repeats=repeats
        )
        self.assertEqual(chunk_count, 2)

        # (1 seed + 2 raw + 2 fresh + 1 reuse) = 6 calls PER round, times
        # (1 discarded warmup + 2 formal repeats) = 3 rounds -- never
        # just 6 total (that would mean only the warmup round ever
        # registered raw/fresh, and every formal repeat merely replayed
        # it or re-registered fresh alone, the exact bug this
        # architecture fixes).
        calls_per_round = 2 + 2 * chunk_count
        rounds = 1 + repeats
        self.assertEqual(len(session.post_calls), calls_per_round * rounds)
        self.assertEqual(len(result["fresh_raw_samples"]), repeats)
        self.assertEqual(len(result["reuse_raw_samples"]), repeats)
        for sample in result["fresh_raw_samples"]:
            self.assertEqual(sample["cached_tokens"], workload.body_start_in_target)
        for sample in result["reuse_raw_samples"]:
            self.assertEqual(sample["cached_tokens"], reuse_cached)

        # Every round's own raw+fresh register chunk call (seed and
        # reuse excluded) must itself stay within one chunk's own small
        # bound -- proof the old single-oversized-call code path is gone
        # from EVERY round, not just the warmup -- and every round's own
        # seed/reuse call still carries that round's own full,
        # un-chunked payload.
        bound = 64 + 512 + len(workload.tail_ids)
        for round_index in range(rounds):
            base = round_index * calls_per_round
            round_calls = session.post_calls[base : base + calls_per_round]
            seed_call = round_calls[0]
            raw_and_fresh_calls = round_calls[1:-1]
            reuse_call = round_calls[-1]
            self.assertEqual(seed_call[1]["input_ids"], list(workload.seed_prompt_ids))
            self.assertEqual(len(raw_and_fresh_calls), 2 * chunk_count)
            for _, payload in raw_and_fresh_calls:
                self.assertLessEqual(len(payload["input_ids"]), bound)
            self.assertEqual(
                reuse_call[1]["input_ids"], list(workload.target_prompt_ids)
            )

    def test_body_2048_two_repeats_every_round_issues_four_raw_and_four_fresh_chunk_calls(
        self,
    ):
        repeats = 2
        workload, chunk_count, reuse_cached, session, result = self._run_with_body(
            body_tokens=2048, repeats=repeats
        )
        self.assertEqual(chunk_count, 4)

        calls_per_round = 2 + 2 * chunk_count
        rounds = 1 + repeats
        self.assertEqual(len(session.post_calls), calls_per_round * rounds)
        self.assertEqual(len(result["fresh_raw_samples"]), repeats)
        self.assertEqual(len(result["reuse_raw_samples"]), repeats)

    def test_every_round_including_warmup_independently_registers_its_own_raw_and_fresh(
        self,
    ):
        # The direct "no cross-round store reuse" proof for THIS class:
        # every one of (1 discarded warmup + repeats formal) rounds
        # issues its OWN chunk_count raw-register calls AND its OWN
        # chunk_count fresh-register calls -- never just once for the
        # warmup, with formal repeats only re-registering fresh (or
        # worse, nothing at all).
        repeats = 3
        workload, chunk_count, reuse_cached, session, result = self._run_with_body(
            body_tokens=1024, repeats=repeats
        )
        del result
        rounds = 1 + repeats
        calls_per_round = 2 + 2 * chunk_count
        self.assertEqual(len(session.post_calls), calls_per_round * rounds)
        # Every round's raw chunk payloads carry a distinct
        # ``content_hash`` per chunk, and every round's fresh chunk
        # payloads (the next ``chunk_count`` calls) carry the parallel
        # per-chunk ``content_hash`` pattern -- proof EVERY round
        # genuinely re-registers both raw and fresh from scratch, never
        # merely a resend of an already-registered segment left behind
        # by an earlier round.
        for round_index in range(rounds):
            base = round_index * calls_per_round
            raw_calls = session.post_calls[base + 1 : base + 1 + chunk_count]
            fresh_calls = session.post_calls[
                base + 1 + chunk_count : base + 1 + 2 * chunk_count
            ]
            for chunk_index, (_, payload) in enumerate(raw_calls):
                segment = payload["sampling_params"]["custom_params"]["approx_kv"][
                    "segments"
                ][0]
                self.assertEqual(
                    segment["content_hash"],
                    f"cachetune-raw:unit-test-run:chunk{chunk_index}",
                )
            for chunk_index, (_, payload) in enumerate(fresh_calls):
                segment = payload["sampling_params"]["custom_params"]["approx_kv"][
                    "segments"
                ][0]
                self.assertEqual(
                    segment["content_hash"],
                    f"cachetune-fresh:unit-test-run:chunk{chunk_index}",
                )

    def test_formal_repeat_fresh_chunk_content_hashes_match_reuse_segments(self):
        repeats = 2
        workload, chunk_count, reuse_cached, session, result = self._run_with_body(
            body_tokens=1024, repeats=repeats
        )
        del result  # only the raw HTTP call sequence matters here

        # The first FORMAL round is round index 1 (index 0 is the
        # discarded warmup, which independently repeats the same
        # content-hash pattern -- checked separately above).
        calls_per_round = 2 + 2 * chunk_count
        first_formal_base = 1 * calls_per_round
        first_formal_fresh_calls = session.post_calls[
            first_formal_base
            + 1
            + chunk_count : first_formal_base
            + 1
            + 2 * chunk_count
        ]
        first_formal_reuse_segments = session.post_calls[
            first_formal_base + calls_per_round - 1
        ][1]["sampling_params"]["custom_params"]["approx_kv"]["segments"]

        for index, (_, payload) in enumerate(first_formal_fresh_calls):
            segment = payload["sampling_params"]["custom_params"]["approx_kv"][
                "segments"
            ][0]
            self.assertEqual(
                segment["content_hash"], f"cachetune-fresh:unit-test-run:chunk{index}"
            )
            self.assertEqual(
                first_formal_reuse_segments[index]["content_hash"],
                f"cachetune-raw:unit-test-run:chunk{index}",
            )

    def test_every_round_flushes_before_its_own_setup_and_after_the_previous_reuse(
        self,
    ):
        repeats = 2
        workload = self._workload(1024)
        chunk_count = len(chunk_offsets(workload.body_tokens, 512))
        reuse_cached = workload.body_start_in_target + workload.body_tokens
        round_names = self._round_names(repeats)
        rounds = len(round_names)

        responses = []
        labels = []
        for round_name in round_names:
            responses += self._one_round_responses(workload, chunk_count, reuse_cached)
            labels += (
                [f"{round_name}_seed"]
                + [f"{round_name}_raw_chunk{i}" for i in range(chunk_count)]
                + [f"{round_name}_fresh_chunk{i}" for i in range(chunk_count)]
                + [f"{round_name}_reuse"]
            )
        call_order: list = []
        session = _LabeledSequencedFakeClientSession(responses, labels, call_order)

        snapshots = []
        for _ in range(rounds):
            snapshots += self._round_snapshots()

        with unittest.mock.patch(
            "aiohttp.ClientSession", return_value=session
        ), unittest.mock.patch(
            "urllib.request.urlopen", _labeled_flush_urlopen(call_order)
        ), unittest.mock.patch(
            "benchmark.approx_kv.run_phase4_cachetune_canary.metric_snapshot",
            side_effect=snapshots,
        ):
            run_non_prefix_setting(
                base_url="http://127.0.0.1:30000",
                tokenizer=FakeTokenizer(),
                workload=workload,
                raw_hash="cachetune-raw:unit-test-run",
                fresh_hash="cachetune-fresh:unit-test-run",
                model_fingerprint="qwen3-0.6b-sm75",
                cache_dtype="fp16",
                repeats=repeats,
                label="unit-test",
                max_chunk_tokens=512,
                target_rho=None,
                pressure_filler_head_tokens=6,
                pressure_filler_body_tokens=8,
            )

        # Exactly `rounds` independent flushes -- one strictly before
        # EACH round's own seed call and strictly after the PREVIOUS
        # round's own reuse call -- never a single flush shared by the
        # whole setting, and never a flush that lands inside another
        # round's own call sequence.
        for round_index, round_name in enumerate(round_names):
            flush_call = call_order.index(f"flush[{round_index}]")
            seed_call = call_order.index(f"{round_name}_seed")
            reuse_call = call_order.index(f"{round_name}_reuse")
            self.assertLess(flush_call, seed_call)
            self.assertLess(seed_call, reuse_call)
            if round_index > 0:
                previous_reuse = call_order.index(
                    f"{round_names[round_index - 1]}_reuse"
                )
                self.assertLess(previous_reuse, flush_call)


class TestBuildSweepPointResult(unittest.TestCase):
    """Shared point-result assembly reused by both the shape sweep and
    the rho sweep in ``run_canary`` -- exercised here against a
    hand-built ``setting_result`` fixture matching ``run_non_prefix_
    setting``'s exact return shape, so this doesn't need a live server
    to validate the cross-check/``passed`` logic."""

    @staticmethod
    def _workload():
        return build_non_prefix_segment_workload(
            FakeTokenizer(),
            body_tokens=128,
            head_tokens=34,
            tail_tokens=1,
            salt="unit-test-sweep-point",
        )

    @staticmethod
    def _quantized(ratio=0.3, context_length=163):
        return quantize_ratio(
            ratio,
            context_length=context_length,
            bounds=RatioBounds.for_mode(CacheTuneMode.SPEED_ONLY),
        )

    @staticmethod
    def _setting_result(**overrides):
        # workload.body_start_in_target (34) + workload.body_tokens (128)
        # == 162: the full restored prefix (exact-match head plus the
        # entire restored body), never head-only -- see
        # require_cached_tokens's own docstring for why a successful
        # CacheTune reuse always extends prefix_indices by the complete
        # restore_length regardless of the controller's selected ratio.
        base = {
            "metrics_before": {
                "sglang:approx_kv_cachetune_selected_tokens_total": 0.0,
                "sglang:approx_kv_dense_fallback_total": 0.0,
            },
            "metrics_after": {
                "sglang:approx_kv_cachetune_selected_tokens_total": 0.0,
                "sglang:approx_kv_dense_fallback_total": 0.0,
            },
            "observed_cached_tokens_per_call": [162, 162],
            "seed_head_ms": 1.0,
            "register_raw_ms": 2.0,
            "fresh_raw_samples": [{"ttft_ms": 10.0, "cached_tokens": 34}],
            "reuse_raw_samples": [{"ttft_ms": 5.0, "cached_tokens": 162}],
            "fresh_ms_samples": [10.0],
            "reuse_ms_samples": [5.0],
            "combined_ms_samples": [15.0],
            "capacity_tokens": 4096,
            "already_pinned_tokens": 512,
            "head_reseed_after_pressure": {
                "ttft_ms": 3.0,
                "cached_tokens": 34,
                "was_evicted_by_pressure": False,
            },
            "observed_rho_after_target": 1.5,
            "peak_rho_observed": 1.6,
            "pressure_and_target_evicted_tokens_total_delta": 200.0,
            "pressure_phase": {"target_rho": 1.5},
            # The 2 formal rounds' own raw run_independent_round result
            # dicts (repeats=2 in every test below) -- deliberately
            # DIFFERENT per-round already_pinned_tokens values so a
            # naive "just copy round 0" passthrough bug would be
            # visible in test_rounds_passed_through_verbatim below.
            "rounds": [
                {"already_pinned_tokens": 500, "pressure_phase": {"target_rho": 1.5}},
                {"already_pinned_tokens": 512, "pressure_phase": {"target_rho": 1.5}},
            ],
        }
        base.update(overrides)
        return base

    def _metrics_after_for(self, quantized, *, repeats, dense_fallback=0.0):
        return {
            "sglang:approx_kv_cachetune_selected_tokens_total": float(
                quantized.repair_tokens * repeats
            ),
            "sglang:approx_kv_dense_fallback_total": dense_fallback,
        }

    def test_passed_true_on_exact_match(self):
        workload = self._workload()
        quantized = self._quantized()
        setting_result = self._setting_result(
            metrics_after=self._metrics_after_for(quantized, repeats=2)
        )

        result = build_sweep_point_result(
            workload=workload,
            quantized=quantized,
            repeats=2,
            setting_result=setting_result,
        )

        self.assertTrue(result["passed"])
        self.assertEqual(result["header_tokens"], 34)
        self.assertEqual(result["body_tokens"], 128)
        self.assertEqual(result["tail_tokens"], 1)
        self.assertFalse(result["is_exact_context_control"])
        self.assertTrue(result["body_source_context_differs_from_target"])
        self.assertEqual(
            result["expected_selected_tokens_per_call"], quantized.repair_tokens
        )
        self.assertEqual(
            result["expected_selected_tokens_total"], quantized.repair_tokens * 2
        )
        self.assertEqual(
            result["expected_executable_ratio"], quantized.executable_ratio
        )
        self.assertEqual(
            result["observed_selected_tokens_total"], quantized.repair_tokens * 2
        )
        self.assertEqual(result["observed_dense_fallback"], 0.0)
        self.assertEqual(result["target_rho"], 1.5)
        self.assertEqual(result["capacity_tokens"], 4096)
        self.assertEqual(result["already_pinned_tokens"], 512)
        self.assertEqual(result["head_reseed_after_pressure"]["cached_tokens"], 34)
        self.assertEqual(result["observed_rho_after_target"], 1.5)
        self.assertEqual(result["peak_rho_observed"], 1.6)
        self.assertEqual(
            result["pressure_and_target_evicted_tokens_total_delta"], 200.0
        )
        self.assertEqual(result["fresh_p50_ms"], 10.0)
        self.assertEqual(result["reuse_p50_ms"], 5.0)
        self.assertEqual(result["combined_p50_ms"], 15.0)
        # Full restored prefix (exact-match head + entire restored
        # body), never head-only: 34 (body_start_in_target) + 128
        # (body_tokens) == 162 -- confirmed on a real SM75 run that an
        # earlier version of this expectation (head-only, 34) was wrong.
        self.assertEqual(result["expected_cached_tokens_per_call"], 162)

    def test_passed_false_when_observed_selected_tokens_falls_short(self):
        workload = self._workload()
        quantized = self._quantized()
        # Only one repeat's worth of selected tokens actually landed --
        # e.g. a silent dense-fallback that still incremented the
        # selected-tokens counter on one call only.
        setting_result = self._setting_result(
            metrics_after=self._metrics_after_for(quantized, repeats=1)
        )

        result = build_sweep_point_result(
            workload=workload,
            quantized=quantized,
            repeats=2,
            setting_result=setting_result,
        )

        self.assertFalse(result["passed"])

    def test_passed_false_on_nonzero_dense_fallback(self):
        workload = self._workload()
        quantized = self._quantized()
        setting_result = self._setting_result(
            metrics_after=self._metrics_after_for(
                quantized, repeats=2, dense_fallback=1.0
            )
        )

        result = build_sweep_point_result(
            workload=workload,
            quantized=quantized,
            repeats=2,
            setting_result=setting_result,
        )

        self.assertFalse(result["passed"])
        self.assertEqual(result["observed_dense_fallback"], 1.0)

    def test_passed_false_on_cached_tokens_mismatch(self):
        workload = self._workload()
        quantized = self._quantized()
        # Correct expected value is 162 (body_start_in_target=34 +
        # body_tokens=128, the full restored prefix); 34 alone (the old,
        # buggy head-only expectation) must still be treated as a
        # mismatch here.
        setting_result = self._setting_result(
            metrics_after=self._metrics_after_for(quantized, repeats=2),
            observed_cached_tokens_per_call=[162, 34],
        )

        result = build_sweep_point_result(
            workload=workload,
            quantized=quantized,
            repeats=2,
            setting_result=setting_result,
        )

        self.assertFalse(result["passed"])

    def test_pressure_phase_none_yields_target_rho_none(self):
        workload = self._workload()
        quantized = self._quantized()
        setting_result = self._setting_result(
            metrics_after=self._metrics_after_for(quantized, repeats=2),
            pressure_phase=None,
        )

        result = build_sweep_point_result(
            workload=workload,
            quantized=quantized,
            repeats=2,
            setting_result=setting_result,
        )

        self.assertIsNone(result["target_rho"])
        self.assertIsNone(result["pressure_phase"])
        self.assertTrue(result["passed"])

    def test_already_pinned_tokens_and_head_reseed_are_none_when_no_pressure_phase_ran(
        self,
    ):
        # The genuine production shape for a header=0 or target_rho=None
        # setting: run_non_prefix_setting never even samples a
        # post-setup snapshot or runs a pressure phase in that case, so
        # both keys are None -- must pass through as None, never KeyError
        # or a stale nonzero placeholder from an unrelated setting.
        workload = self._workload()
        quantized = self._quantized()
        setting_result = self._setting_result(
            metrics_after=self._metrics_after_for(quantized, repeats=2),
            pressure_phase=None,
            already_pinned_tokens=None,
            head_reseed_after_pressure=None,
        )

        result = build_sweep_point_result(
            workload=workload,
            quantized=quantized,
            repeats=2,
            setting_result=setting_result,
        )

        self.assertIsNone(result["already_pinned_tokens"])
        self.assertIsNone(result["head_reseed_after_pressure"])
        self.assertTrue(result["passed"])

    def test_pressure_telemetry_passed_through_unchanged(self):
        workload = self._workload()
        quantized = self._quantized()
        head_reseed = {
            "ttft_ms": 4.0,
            "cached_tokens": 0,
            "was_evicted_by_pressure": True,
        }
        setting_result = self._setting_result(
            metrics_after=self._metrics_after_for(quantized, repeats=2),
            capacity_tokens=8192,
            already_pinned_tokens=2048,
            head_reseed_after_pressure=head_reseed,
            observed_rho_after_target=2.0,
            peak_rho_observed=2.4,
            pressure_and_target_evicted_tokens_total_delta=999.0,
            pressure_phase={"target_rho": 2.0},
        )

        result = build_sweep_point_result(
            workload=workload,
            quantized=quantized,
            repeats=2,
            setting_result=setting_result,
        )

        self.assertEqual(result["capacity_tokens"], 8192)
        self.assertEqual(result["already_pinned_tokens"], 2048)
        self.assertEqual(result["head_reseed_after_pressure"], head_reseed)
        self.assertEqual(result["observed_rho_after_target"], 2.0)
        self.assertEqual(result["peak_rho_observed"], 2.4)
        self.assertEqual(
            result["pressure_and_target_evicted_tokens_total_delta"], 999.0
        )
        self.assertEqual(result["target_rho"], 2.0)

    def test_rounds_passed_through_verbatim(self):
        # "rounds" is a pure passthrough of every formal round's own
        # raw run_independent_round result -- never re-derived,
        # truncated, or collapsed to just the last round's own entry.
        workload = self._workload()
        quantized = self._quantized()
        rounds = [
            {"already_pinned_tokens": 100, "pressure_phase": {"target_rho": 0.5}},
            {"already_pinned_tokens": 200, "pressure_phase": {"target_rho": 0.5}},
        ]
        setting_result = self._setting_result(
            metrics_after=self._metrics_after_for(quantized, repeats=2),
            rounds=rounds,
        )

        result = build_sweep_point_result(
            workload=workload,
            quantized=quantized,
            repeats=2,
            setting_result=setting_result,
        )

        self.assertEqual(result["rounds"], rounds)
        # A genuine passthrough preserves object identity (the
        # production code reads "rounds" straight from setting_result,
        # never a re-derived or copied list).
        self.assertIs(result["rounds"], rounds)


class TestValidatePairwiseHeadIsolationAgainstProductionCallShape(unittest.TestCase):
    """Combines ``build_eviction_pressure_workloads`` output with a
    default-prefix setting head through ``validate_pairwise_head_isolation``,
    reproducing the exact call shape ``run_non_prefix_setting`` uses in
    production (every filler's head plus the setting's own head, never
    two different settings' heads at once -- each setting gets its own
    flush-isolated call)."""

    def test_fillers_and_a_default_prefix_setting_head_are_mutually_isolated(self):
        tokenizer = CharLevelFakeTokenizer()
        pressure_workloads = build_eviction_pressure_workloads(
            tokenizer,
            object_count=4,
            body_tokens=32,
            head_tokens=NON_PREFIX_HEAD_TOKENS,
            tail_tokens=NON_PREFIX_TAIL_TOKENS,
            salt_prefix="phase4-r5-pressure",
        )
        setting_workload = build_non_prefix_segment_workload(
            tokenizer,
            body_tokens=256,
            head_tokens=NON_PREFIX_HEAD_TOKENS,
            tail_tokens=NON_PREFIX_TAIL_TOKENS,
            salt="phase4-r5-cachetune-main",
        )
        labeled_heads = [
            (f"pressure-filler[{index}]", filler.target_head_ids)
            for index, filler in enumerate(pressure_workloads)
        ] + [("main", setting_workload.target_head_ids)]

        validate_pairwise_head_isolation(labeled_heads)  # must not raise


class _OrderTrackingSequencedFakeClientSession(_SequencedFakeClientSession):
    """Like ``_SequencedFakeClientSession``, but appends ``label`` to a
    shared ``call_order`` list on every ``.post()`` -- lets a test
    interleave-verify HTTP-layer call order against ``urllib.request.
    urlopen``-based calls (e.g. ``flush_exact_radix_cache``) and
    ``metric_snapshot`` calls, none of which share a single mockable
    entry point on their own."""

    def __init__(self, responses: list, call_order: list[str], label: str):
        super().__init__(responses)
        self._call_order = call_order
        self._label = label

    def post(self, url, json):
        self._call_order.append(self._label)
        return super().post(url, json)


class TestCaptureFinalPoolResetAndInvariant(unittest.TestCase):
    """``capture_final_pool_reset_and_invariant`` must snapshot ``/metrics``
    BEFORE flushing (informational only, never gating), flush + force one
    real sentinel request, and ONLY THEN snapshot again and compute
    ``idle_pool_invariant`` -- never against the pre-flush snapshot,
    whose nonzero ``kv_used_tokens`` is expected (every setting's own
    raw/fresh CacheTune segments, plus every eviction-pressure filler's
    plain dense KV cache entry, are still resident by design at that
    point -- a real SM75 run observed exactly this: 4096 used tokens
    with ``accounted_tokens`` already matching
    ``max_total_num_tokens``) and must never be misread as a pool leak.
    """

    @staticmethod
    def _fake_flush_urlopen(call_order: list[str]) -> callable:
        class _FakeUrlResponse:
            def read(self):
                return b"Cache flushed.\n"

            def __enter__(self):
                return self

            def __exit__(self, *exc_info):
                return False

        def fake_urlopen(request, timeout=None):
            call_order.append("flush")
            return _FakeUrlResponse()

        return fake_urlopen

    @staticmethod
    def _sentinel_response(finish_type: str = "length"):
        chunk = {
            "meta_info": {
                "finish_reason": {"type": finish_type},
                "cached_tokens": 0,
            }
        }
        return _FakeStreamResponse([_sse_data_line(chunk), _SSE_DONE_LINE])

    @staticmethod
    def _fake_metric_snapshot(
        call_order: list[str],
        metrics_pre: dict,
        metrics_post: dict,
    ) -> callable:
        state = {"count": 0}

        def fake_metric_snapshot(base_url):
            state["count"] += 1
            call_order.append(f"metric_snapshot_{state['count']}")
            return metrics_pre if state["count"] == 1 else metrics_post

        return fake_metric_snapshot

    _METRICS_PRE_RESET_STILL_RESIDENT = {
        "sglang:max_total_num_tokens": 13130.0,
        "sglang:kv_available_tokens": 9034.0,
        "sglang:kv_evictable_tokens": 0.0,
        "sglang:kv_used_tokens": 4096.0,
    }
    _METRICS_POST_RESET_IDLE = {
        "sglang:max_total_num_tokens": 13130.0,
        "sglang:kv_available_tokens": 13130.0,
        "sglang:kv_evictable_tokens": 0.0,
        "sglang:kv_used_tokens": 0.0,
    }

    def test_flush_and_sentinel_happen_between_pre_and_post_snapshots(self):
        call_order: list[str] = []
        session = _OrderTrackingSequencedFakeClientSession(
            [self._sentinel_response()], call_order, "sentinel_generate"
        )

        with unittest.mock.patch(
            "urllib.request.urlopen", self._fake_flush_urlopen(call_order)
        ), unittest.mock.patch(
            "aiohttp.ClientSession", return_value=session
        ), unittest.mock.patch(
            "benchmark.approx_kv.run_phase4_cachetune_canary.metric_snapshot",
            side_effect=self._fake_metric_snapshot(
                call_order,
                self._METRICS_PRE_RESET_STILL_RESIDENT,
                self._METRICS_POST_RESET_IDLE,
            ),
        ):
            result = capture_final_pool_reset_and_invariant(
                "http://127.0.0.1:30000", FakeTokenizer()
            )

        # The exact ordering this fix exists to guarantee: pre-reset
        # snapshot, THEN flush, THEN the sentinel request, THEN the
        # post-reset snapshot idle_pool_invariant is computed from.
        self.assertEqual(
            call_order,
            [
                "metric_snapshot_1",
                "flush",
                "sentinel_generate",
                "metric_snapshot_2",
            ],
        )
        self.assertEqual(len(session.post_calls), 1)
        self.assertEqual(
            result["metrics_pre_reset"], self._METRICS_PRE_RESET_STILL_RESIDENT
        )
        self.assertEqual(result["metrics_post_reset"], self._METRICS_POST_RESET_IDLE)

    def test_used_tokens_reach_idle_only_after_reset_never_treated_as_leak(self):
        call_order: list[str] = []
        session = _OrderTrackingSequencedFakeClientSession(
            [self._sentinel_response()], call_order, "sentinel_generate"
        )

        with unittest.mock.patch(
            "urllib.request.urlopen", self._fake_flush_urlopen(call_order)
        ), unittest.mock.patch(
            "aiohttp.ClientSession", return_value=session
        ), unittest.mock.patch(
            "benchmark.approx_kv.run_phase4_cachetune_canary.metric_snapshot",
            side_effect=self._fake_metric_snapshot(
                call_order,
                self._METRICS_PRE_RESET_STILL_RESIDENT,
                self._METRICS_POST_RESET_IDLE,
            ),
        ):
            result = capture_final_pool_reset_and_invariant(
                "http://127.0.0.1:30000", FakeTokenizer()
            )

        # The still-resident pre-reset snapshot would FAIL idle_pool_
        # invariant on its own (kv_used_tokens=4096, exactly the real
        # SM75 observation) -- proving that residency really is nonzero
        # and would misreport as a leak if it were ever (wrongly) used
        # to gate pass/fail directly.
        pre_reset_checked_directly = idle_pool_invariant(result["metrics_pre_reset"])
        self.assertFalse(pre_reset_checked_directly["passed"])
        self.assertEqual(pre_reset_checked_directly["kv_used_tokens"], 4096.0)
        # But the function's OWN returned invariant is computed from the
        # post-reset snapshot alone, where used tokens genuinely reached
        # (near-)zero after the flush + sentinel -- this is what
        # actually gates pass/fail, and it must pass.
        self.assertTrue(result["pool_invariant"]["passed"])
        self.assertEqual(result["pool_invariant"]["kv_used_tokens"], 0.0)

    def test_sentinel_payload_is_fixed_deterministic_dense_request(self):
        call_order: list[str] = []
        session = _OrderTrackingSequencedFakeClientSession(
            [self._sentinel_response()], call_order, "sentinel_generate"
        )

        with unittest.mock.patch(
            "urllib.request.urlopen", self._fake_flush_urlopen(call_order)
        ), unittest.mock.patch(
            "aiohttp.ClientSession", return_value=session
        ), unittest.mock.patch(
            "benchmark.approx_kv.run_phase4_cachetune_canary.metric_snapshot",
            side_effect=self._fake_metric_snapshot(
                call_order,
                self._METRICS_PRE_RESET_STILL_RESIDENT,
                self._METRICS_POST_RESET_IDLE,
            ),
        ):
            capture_final_pool_reset_and_invariant(
                "http://127.0.0.1:30000", FakeTokenizer()
            )

        self.assertEqual(len(session.post_calls), 1)
        _, posted_payload = session.post_calls[0]
        # A plain dense request (no approx_kv metadata at all): its only
        # job is forcing one real scheduler iteration, never exercising
        # any CacheTune-specific path.
        self.assertNotIn("approx_kv_metadata", posted_payload)
        self.assertEqual(posted_payload["sampling_params"]["max_new_tokens"], 1)
        self.assertGreater(len(posted_payload["input_ids"]), 0)

    def test_flush_failure_propagates_uncaught(self):
        # No step in this function may catch its own exceptions (see its
        # own docstring): a real flush failure must reach main()'s
        # existing central-log "failed" entry, never be silently
        # swallowed into a misleadingly "passed" result.
        def raising_urlopen(request, timeout=None):
            raise OSError("flush endpoint unreachable")

        with unittest.mock.patch(
            "urllib.request.urlopen", raising_urlopen
        ), unittest.mock.patch(
            "benchmark.approx_kv.run_phase4_cachetune_canary.metric_snapshot",
            return_value=self._METRICS_PRE_RESET_STILL_RESIDENT,
        ):
            with self.assertRaises(OSError):
                capture_final_pool_reset_and_invariant(
                    "http://127.0.0.1:30000", FakeTokenizer()
                )

    def test_sentinel_not_finished_by_length_propagates_uncaught(self):
        call_order: list[str] = []
        # The sentinel request itself getting cut short (e.g. an abort
        # or length-limit mismatch) must also propagate, not be
        # swallowed -- a stuck/failed sentinel means the post-reset
        # snapshot can never be trusted at all.
        session = _OrderTrackingSequencedFakeClientSession(
            [self._sentinel_response(finish_type="abort")],
            call_order,
            "sentinel_generate",
        )

        with unittest.mock.patch(
            "urllib.request.urlopen", self._fake_flush_urlopen(call_order)
        ), unittest.mock.patch(
            "aiohttp.ClientSession", return_value=session
        ), unittest.mock.patch(
            "benchmark.approx_kv.run_phase4_cachetune_canary.metric_snapshot",
            return_value=self._METRICS_PRE_RESET_STILL_RESIDENT,
        ):
            with self.assertRaises(RuntimeError):
                capture_final_pool_reset_and_invariant(
                    "http://127.0.0.1:30000", FakeTokenizer()
                )


class TestMainRunCanaryExceptionCentralLog(unittest.TestCase):
    """``main()``'s pre-existing ``try``/``except`` around
    ``run_canary(args)`` must append a ``status="failed"`` central-log
    entry (carrying the exception) and then RE-RAISE -- never swallow --
    when ``run_canary`` fails for any reason, including a failure inside
    the new ``capture_final_pool_reset_and_invariant`` step this fix
    adds (e.g. the final flush or sentinel request itself failing). This
    proves that step's exceptions are never accidentally caught before
    reaching this pre-existing central-log-on-failure path -- ``main``
    itself is intentionally left unmodified by this fix, so this test
    also guards against a future regression silently breaking that
    composition."""

    def test_run_canary_failure_appends_failed_entry_and_reraises(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            central_log = Path(tmp_dir) / "central.jsonl"
            output_path = Path(tmp_dir) / "result.json"
            fake_args = argparse.Namespace(
                central_log=central_log,
                output=output_path,
                mode="speed_only",
            )
            failure = RuntimeError(
                "final pool-reset sentinel request did not finish by length"
            )

            with unittest.mock.patch(
                "benchmark.approx_kv.run_phase4_cachetune_canary.parse_args",
                return_value=fake_args,
            ), unittest.mock.patch(
                "benchmark.approx_kv.run_phase4_cachetune_canary.build_settings",
                return_value={"mode": "speed_only"},
            ), unittest.mock.patch(
                "benchmark.approx_kv.run_phase4_cachetune_canary.run_canary",
                side_effect=failure,
            ):
                with self.assertRaises(RuntimeError):
                    main()

            lines = central_log.read_text(encoding="utf-8").strip().splitlines()

        self.assertEqual(len(lines), 2)
        running_entry = json.loads(lines[0])
        failed_entry = json.loads(lines[1])
        self.assertEqual(running_entry["status"], "running")
        self.assertEqual(failed_entry["status"], "failed")
        self.assertIn(
            "final pool-reset sentinel request did not finish by length",
            failed_entry["error"],
        )
        self.assertFalse(output_path.exists())


if __name__ == "__main__":
    unittest.main()
