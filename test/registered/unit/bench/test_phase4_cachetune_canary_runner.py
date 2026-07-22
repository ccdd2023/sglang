from __future__ import annotations

import argparse
import json
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from benchmark.approx_kv.run_phase4_cachetune_canary import (
    WARMUP_PASSES_PER_SETTING,
    _repeat_count,
    append_run_log,
    build_settings,
    expected_repair_totals,
    flush_exact_radix_cache,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-c-test-cpu")


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
            target_prefix_tokens=256,
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


if __name__ == "__main__":
    unittest.main()
