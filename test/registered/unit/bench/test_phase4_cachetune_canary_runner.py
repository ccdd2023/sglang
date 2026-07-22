from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from benchmark.approx_kv.run_phase4_cachetune_canary import (
    NON_PREFIX_HEAD_TOKENS,
    NON_PREFIX_TAIL_TOKENS,
    WARMUP_PASSES_PER_SETTING,
    NonPrefixSegmentWorkload,
    _deterministic_token_ids,
    _repeat_count,
    append_run_log,
    build_non_prefix_segment_workload,
    build_settings,
    dense_generate_payload,
    expected_repair_totals,
    flush_exact_radix_cache,
    register_generate_payload,
    require_cached_tokens,
    require_finished_by_length,
    reuse_generate_payload,
    timed_post,
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
            body_tokens=256,
            length_sweep="128,512",
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
        self.assertEqual(settings["length_sweep"], "128,512")
        self.assertEqual(settings["runner_git_sha"], "abc123")
        self.assertEqual(settings["image_digest"], "sha256:deadbeef")

    def test_carries_through_body_tokens_and_fixed_head_tail(self):
        # body_tokens comes from parsed args; head/tail are fixed
        # measurement-protocol constants, not CLI-controlled, matching
        # the "target head固定34即可" requirement.
        args = self._fake_args(body_tokens=384)
        settings = build_settings(args)
        self.assertEqual(settings["body_tokens"], 384)
        self.assertEqual(settings["head_tokens"], NON_PREFIX_HEAD_TOKENS)
        self.assertEqual(settings["tail_tokens"], NON_PREFIX_TAIL_TOKENS)

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
            content_hash="cachetune-raw:test",
            target_start=2,
            length=2,
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

    def test_reuse_generate_payload_shape(self):
        payload = reuse_generate_payload(
            input_ids=(1, 2, 3, 4, 5),
            raw_content_hash="cachetune-raw:test",
            target_start=2,
            length=2,
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
            content_hash="h",
            target_start=0,
            length=3,
            model_fingerprint="fp",
            cache_dtype="fp16",
        )
        self.assertIsInstance(payload["input_ids"], list)
        self.assertEqual(payload["input_ids"], [7, 8, 9])


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


class TestTimedPost(unittest.TestCase):
    """``timed_post`` must target native ``/generate`` (never
    ``/v1/chat/completions``) and return the parsed JSON response
    alongside a genuine non-negative elapsed-time measurement."""

    def test_posts_to_generate_endpoint(self):
        captured: dict[str, object] = {}

        class FakeResponse:
            def read(self):
                return json.dumps(
                    {"meta_info": {"finish_reason": {"type": "length"}}}
                ).encode("utf-8")

            def __enter__(self):
                return self

            def __exit__(self, *exc_info):
                return False

        def fake_urlopen(request, timeout=None):
            captured["url"] = request.full_url
            captured["data"] = json.loads(request.data)
            captured["timeout"] = timeout
            return FakeResponse()

        with unittest.mock.patch("urllib.request.urlopen", fake_urlopen):
            response, elapsed_ms = timed_post(
                "http://127.0.0.1:30000", {"input_ids": [1, 2, 3]}
            )

        self.assertEqual(captured["url"], "http://127.0.0.1:30000/generate")
        self.assertEqual(captured["data"], {"input_ids": [1, 2, 3]})
        self.assertEqual(response, {"meta_info": {"finish_reason": {"type": "length"}}})
        self.assertGreaterEqual(elapsed_ms, 0.0)


if __name__ == "__main__":
    unittest.main()
