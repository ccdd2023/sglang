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

from benchmark.approx_kv.run_phase4_cachetune_canary import (
    _PRESSURE_FILLER_HEAD_ALPHABET,
    NON_PREFIX_HEAD_TOKENS,
    NON_PREFIX_TAIL_TOKENS,
    WARMUP_PASSES_PER_SETTING,
    NonPrefixSegmentWorkload,
    _deterministic_token_ids,
    _first_common_prefix_length,
    _non_negative_int_choice_list,
    _positive_float,
    _positive_float_choice_list,
    _positive_int,
    _positive_int_choice_list,
    _pressure_filler_head_literal_prefix,
    _repeat_count,
    append_run_log,
    body_segments_for_hash,
    build_eviction_pressure_workloads,
    build_non_prefix_segment_workload,
    build_settings,
    build_sweep_point_result,
    chunk_offsets,
    dense_generate_payload,
    eviction_pressure_filler_count_for_rho,
    eviction_pressure_total_tokens,
    expected_repair_totals,
    flush_exact_radix_cache,
    observed_rho,
    register_eviction_pressure_objects,
    register_generate_payload,
    require_cached_tokens,
    require_finished_by_length,
    reuse_generate_payload,
    run_exact_context_control_point,
    timed_post,
    validate_pairwise_head_isolation,
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
        )
        defaults.update(overrides)
        return NonPrefixSegmentWorkload(**defaults)

    def test_valid_construction_computes_properties(self):
        workload = self._workload()
        self.assertEqual(workload.body_tokens, 4)
        self.assertEqual(workload.body_start_in_source, 3)
        self.assertEqual(workload.body_start_in_target, 3)
        self.assertEqual(workload.source_prompt_ids, (1, 2, 3, 4, 5, 6, 7))
        self.assertEqual(workload.target_prompt_ids, (9, 8, 7, 4, 5, 6, 7, 99))
        self.assertEqual(workload.fresh_prompt_ids, (9, 8, 7, 4, 5, 6, 7))
        self.assertTrue(workload.body_source_context_differs_from_target)

    def test_shared_body_appears_identically_in_both_prompts(self):
        workload = self._workload()
        source_slice = workload.source_prompt_ids[workload.body_start_in_source :]
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

    def test_is_frozen(self):
        workload = self._workload()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            workload.source_head_ids = (1,)  # type: ignore[misc]


class TestPressureFillerHeadLiteralPrefix(unittest.TestCase):
    """Every eviction-pressure filler object needs its own mutually
    distinct target-head literal-prefix marker (see
    ``_PRESSURE_FILLER_HEAD_ALPHABET``); this generator must be
    deterministic, in-range-valid, and loudly reject anything it cannot
    keep distinct."""

    def test_index_zero_uses_first_alphabet_letter(self):
        self.assertEqual(
            _pressure_filler_head_literal_prefix(0), "AFILLERHEAD_MARKER_TEXT\n"
        )

    def test_index_one_uses_second_alphabet_letter(self):
        self.assertEqual(
            _pressure_filler_head_literal_prefix(1), "BFILLERHEAD_MARKER_TEXT\n"
        )

    def test_last_valid_index_uses_last_alphabet_letter(self):
        last_index = len(_PRESSURE_FILLER_HEAD_ALPHABET) - 1
        prefix = _pressure_filler_head_literal_prefix(last_index)
        self.assertTrue(prefix.startswith(_PRESSURE_FILLER_HEAD_ALPHABET[-1]))

    def test_is_deterministic(self):
        self.assertEqual(
            _pressure_filler_head_literal_prefix(3),
            _pressure_filler_head_literal_prefix(3),
        )

    def test_every_valid_index_produces_a_distinct_prefix(self):
        prefixes = [
            _pressure_filler_head_literal_prefix(index)
            for index in range(len(_PRESSURE_FILLER_HEAD_ALPHABET))
        ]
        self.assertEqual(len(prefixes), len(set(prefixes)))

    def test_rejects_negative_index(self):
        with self.assertRaises(ValueError):
            _pressure_filler_head_literal_prefix(-1)

    def test_rejects_index_at_alphabet_length(self):
        with self.assertRaises(ValueError):
            _pressure_filler_head_literal_prefix(len(_PRESSURE_FILLER_HEAD_ALPHABET))

    def test_rejects_index_far_beyond_alphabet_length(self):
        with self.assertRaises(ValueError):
            _pressure_filler_head_literal_prefix(999)

    def test_excludes_source_and_target_head_markers_own_letters(self):
        # S and T are already used by _SOURCE_HEAD_LITERAL_PREFIX /
        # _TARGET_HEAD_LITERAL_PREFIX; the pressure-filler alphabet must
        # never reuse either, or a filler could collide with the
        # setting's own head/source markers.
        self.assertNotIn("S", _PRESSURE_FILLER_HEAD_ALPHABET)
        self.assertNotIn("T", _PRESSURE_FILLER_HEAD_ALPHABET)


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


class TestObservedRho(unittest.TestCase):
    """The genuine, sampled occupancy ratio from a live
    ``sglang:kv_used_tokens`` gauge snapshot against a fixed capacity
    reference -- distinct from the nominal (requested-tokens) nature of
    ``eviction_pressure_filler_count_for_rho``."""

    def test_computes_simple_ratio(self):
        snapshot = {"sglang:kv_used_tokens": 500.0}
        self.assertAlmostEqual(observed_rho(snapshot, capacity_tokens=1000), 0.5)

    def test_ratio_can_exceed_one_under_real_pressure(self):
        # The pool is a fixed physical size, but the SUM of nominal
        # filler requests can (by design) exceed it; the actual gauge
        # reading is capped at whatever physically fits, but a snapshot
        # taken transiently mid-registration could still legitimately
        # read higher than the fixed idle-capacity reference if that
        # reference itself under-counts a since-grown pool -- this
        # function must not silently clamp such a reading.
        snapshot = {"sglang:kv_used_tokens": 1200.0}
        self.assertAlmostEqual(observed_rho(snapshot, capacity_tokens=1000), 1.2)

    def test_zero_used_tokens_is_zero_ratio(self):
        snapshot = {"sglang:kv_used_tokens": 0.0}
        self.assertAlmostEqual(observed_rho(snapshot, capacity_tokens=1000), 0.0)

    def test_rejects_non_positive_capacity_tokens(self):
        with self.assertRaises(ValueError):
            observed_rho({"sglang:kv_used_tokens": 500.0}, capacity_tokens=0)

    def test_raises_when_gauge_missing_from_snapshot(self):
        with self.assertRaises(ValueError):
            observed_rho({}, capacity_tokens=1000)


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
    """``meta_info.cached_tokens`` is generic SGLang exact-prefix
    accounting (unrelated to CacheTune's own Prometheus counters) --
    the independent per-request signal that a reuse request's own
    exact-match boundary landed exactly where the registered segment
    expects."""

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

    Needed to test ``materialize_workload_via_reuse`` /
    ``register_eviction_pressure_objects``: a single filler object's
    materialization alone already issues four sequential requests (seed
    head, register raw, register fresh, reuse), each expected to report
    different ``cached_tokens``, and multiple filler objects chain many
    such sequences back to back. Raises ``AssertionError`` (never
    silently replaying a stale response) if more ``.post()`` calls
    happen than responses were provided -- an unexpected extra call is
    itself a sign the production code under test regressed.
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


class TestRegisterEvictionPressureObjects(unittest.TestCase):
    """Every filler object's mandatory register+reuse materialization
    must run in order against the live server, and a nonzero
    dense-fallback delta observed during this phase must raise loudly:
    a filler silently falling back to dense would mean it never
    actually became a genuine device-resident CacheTune-repaired
    occupant, defeating the whole point of the eviction-pressure
    phase."""

    def _filler_workloads(self, object_count=2, body_tokens=8):
        return build_eviction_pressure_workloads(
            FakeTokenizer(),
            object_count=object_count,
            body_tokens=body_tokens,
            head_tokens=6,
            tail_tokens=1,
            salt_prefix="unit-test-pressure",
        )

    def _materialize_success_responses(self, workload):
        """The four responses one filler's own
        ``materialize_workload_via_reuse`` call needs, in call order, to
        pass every check that function performs: seed target_head
        (``cached_tokens=0``), register raw (``cached_tokens=0``),
        register fresh (``cached_tokens`` unchecked by that step), reuse
        (``cached_tokens=workload.body_start_in_target``)."""
        zero_cached_chunk = {
            "meta_info": {"finish_reason": {"type": "length"}, "cached_tokens": 0}
        }
        reuse_chunk = {
            "meta_info": {
                "finish_reason": {"type": "length"},
                "cached_tokens": workload.body_start_in_target,
            }
        }
        chunks = [
            zero_cached_chunk,  # seed target_head
            zero_cached_chunk,  # register raw
            zero_cached_chunk,  # register fresh
            reuse_chunk,  # reuse
        ]
        return [
            _FakeStreamResponse([_sse_data_line(chunk), _SSE_DONE_LINE])
            for chunk in chunks
        ]

    def test_materializes_every_filler_in_order_and_returns_telemetry(self):
        workloads = self._filler_workloads(object_count=2, body_tokens=8)
        responses = [
            response
            for workload in workloads
            for response in self._materialize_success_responses(workload)
        ]
        session = _SequencedFakeClientSession(responses)
        metrics_before = {
            "sglang:approx_kv_dense_fallback_total": 3.0,
            "sglang:evicted_tokens_total": 100.0,
            "sglang:kv_used_tokens": 400.0,
        }
        metrics_after = {
            "sglang:approx_kv_dense_fallback_total": 3.0,
            "sglang:evicted_tokens_total": 116.0,
            "sglang:kv_used_tokens": 900.0,
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
                model_fingerprint="qwen3-0.6b-sm75",
                cache_dtype="fp16",
                label="unit-test",
                max_chunk_tokens=512,
                capacity_tokens=1000,
                target_rho=1.5,
            )

        # Every filler's own four-request sequence must have actually
        # run, in order (a wrong call count would mean either a filler
        # was skipped or the four-step materialize sequence itself
        # regressed).
        self.assertEqual(len(session.post_calls), 8)
        self.assertEqual(result["object_count"], 2)
        self.assertEqual(result["total_pressure_tokens"], 16)
        self.assertEqual(result["target_rho"], 1.5)
        self.assertEqual(result["capacity_tokens"], 1000)
        self.assertAlmostEqual(result["observed_rho_after_pressure"], 0.9)
        self.assertEqual(result["evicted_tokens_total_delta"], 16.0)
        self.assertEqual(result["dense_fallback_total_delta"], 0.0)
        self.assertEqual(result["metrics_before"], metrics_before)
        self.assertEqual(result["metrics_after"], metrics_after)

    def test_raises_when_dense_fallback_delta_is_nonzero(self):
        # The core safety invariant this function exists to enforce: a
        # filler silently falling back to dense during materialization
        # must never be treated as a harmless, ignorable detail.
        workloads = self._filler_workloads(object_count=1, body_tokens=8)
        responses = self._materialize_success_responses(workloads[0])
        session = _SequencedFakeClientSession(responses)
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
                    model_fingerprint="qwen3-0.6b-sm75",
                    cache_dtype="fp16",
                    label="unit-test",
                    max_chunk_tokens=512,
                    capacity_tokens=1000,
                    target_rho=1.5,
                )
        message = str(ctx.exception)
        self.assertIn("unit-test", message)
        self.assertIn("1 eviction-pressure filler", message)

    def test_does_not_raise_when_dense_fallback_delta_is_zero_despite_other_deltas(
        self,
    ):
        # A positive control alongside the raise-test above: OTHER
        # counters (e.g. evicted_tokens_total) moving is expected and
        # fine; only a nonzero dense_fallback delta must raise.
        workloads = self._filler_workloads(object_count=1, body_tokens=8)
        responses = self._materialize_success_responses(workloads[0])
        session = _SequencedFakeClientSession(responses)
        metrics_before = {
            "sglang:approx_kv_dense_fallback_total": 0.0,
            "sglang:evicted_tokens_total": 0.0,
            "sglang:kv_used_tokens": 0.0,
        }
        metrics_after = {
            "sglang:approx_kv_dense_fallback_total": 0.0,
            "sglang:evicted_tokens_total": 8.0,
            "sglang:kv_used_tokens": 8.0,
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
                model_fingerprint="qwen3-0.6b-sm75",
                cache_dtype="fp16",
                label="unit-test",
                max_chunk_tokens=512,
                capacity_tokens=1000,
                target_rho=1.5,
            )
        self.assertEqual(result["dense_fallback_total_delta"], 0.0)
        self.assertEqual(result["evicted_tokens_total_delta"], 8.0)
        self.assertAlmostEqual(result["observed_rho_after_pressure"], 0.008)


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
        base = {
            "metrics_before": {
                "sglang:approx_kv_cachetune_selected_tokens_total": 0.0,
                "sglang:approx_kv_dense_fallback_total": 0.0,
            },
            "metrics_after": {
                "sglang:approx_kv_cachetune_selected_tokens_total": 0.0,
                "sglang:approx_kv_dense_fallback_total": 0.0,
            },
            "observed_cached_tokens_per_call": [34, 34],
            "seed_head_ms": 1.0,
            "register_raw_ms": 2.0,
            "fresh_raw_samples": [{"ttft_ms": 10.0, "cached_tokens": 34}],
            "reuse_raw_samples": [{"ttft_ms": 5.0, "cached_tokens": 34}],
            "fresh_ms_samples": [10.0],
            "reuse_ms_samples": [5.0],
            "combined_ms_samples": [15.0],
            "capacity_tokens": 4096,
            "observed_rho_after_target": 1.5,
            "peak_rho_observed": 1.6,
            "pressure_and_target_evicted_tokens_total_delta": 200.0,
            "pressure_phase": {"target_rho": 1.5},
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
        self.assertEqual(result["observed_rho_after_target"], 1.5)
        self.assertEqual(result["peak_rho_observed"], 1.6)
        self.assertEqual(
            result["pressure_and_target_evicted_tokens_total_delta"], 200.0
        )
        self.assertEqual(result["fresh_p50_ms"], 10.0)
        self.assertEqual(result["reuse_p50_ms"], 5.0)
        self.assertEqual(result["combined_p50_ms"], 15.0)

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
        setting_result = self._setting_result(
            metrics_after=self._metrics_after_for(quantized, repeats=2),
            observed_cached_tokens_per_call=[34, 33],
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

    def test_pressure_telemetry_passed_through_unchanged(self):
        workload = self._workload()
        quantized = self._quantized()
        setting_result = self._setting_result(
            metrics_after=self._metrics_after_for(quantized, repeats=2),
            capacity_tokens=8192,
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
        self.assertEqual(result["observed_rho_after_target"], 2.0)
        self.assertEqual(result["peak_rho_observed"], 2.4)
        self.assertEqual(
            result["pressure_and_target_evicted_tokens_total_delta"], 999.0
        )
        self.assertEqual(result["target_rho"], 2.0)


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


if __name__ == "__main__":
    unittest.main()
