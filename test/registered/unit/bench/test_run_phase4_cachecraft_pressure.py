from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmark.approx_kv import run_phase4_cachecraft_pressure as runner
from sglang.srt.mem_cache.approx_kv.cachecraft_capability import (
    CacheCraftServerCapability,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-c-test-cpu")


def make_args(**overrides):
    defaults = dict(
        base_url="http://fake",
        header_tokens=64,
        body_tokens=1024,
        target_rho=2.0,
        filler_tokens=736,
        num_chunks=3,
        segment_tokens=512,
        repeats=4,
        runner_git_sha="deadbeef",
        image_digest="sha256:fake",
        output="unused-output.json",
        central_log="unused-log.jsonl",
        allow_real_run=False,
    )
    defaults.update(overrides)
    return runner.argparse.Namespace(**defaults)


class TestBuildSettings(unittest.TestCase):
    def test_matches_unified_phase4_contract(self):
        args = make_args()
        settings = runner.build_settings(args)
        self.assertEqual(settings["mode"], "cachecraft")
        self.assertEqual(settings["plugin"], "cachecraft")
        self.assertEqual(settings["mem_fraction_static"], 0.35)
        self.assertEqual(settings["scheduler"], "S0 LRU")
        self.assertEqual(settings["tier"], "GPU-only")
        self.assertFalse(settings["prefetch"])
        self.assertEqual(settings["global_warmup_passes"], 1)
        self.assertEqual(settings["per_setting_warmup_passes"], 1)
        self.assertEqual(settings["formal_repeats"], 4)
        self.assertEqual(settings["header_tokens"], 64)
        self.assertEqual(settings["body_tokens"], 1024)
        self.assertEqual(settings["target_prompt_tokens"], 64 + 1024 + 1)
        self.assertTrue(settings["crosses_1024_token_chunk_boundary"])

    def test_flags_cross_chunk_boundary_for_large_combinations(self):
        args = make_args(header_tokens=256, body_tokens=2048)
        settings = runner.build_settings(args)
        self.assertTrue(settings["crosses_1024_token_chunk_boundary"])


class TestRepeatCountValidation(unittest.TestCase):
    def test_rejects_fewer_than_minimum_repeats(self):
        with self.assertRaises(runner.argparse.ArgumentTypeError):
            runner._repeat_count("1")

    def test_accepts_minimum_repeats(self):
        self.assertEqual(runner._repeat_count("2"), 2)


class TestBlockedPathMakesNoNetworkCalls(unittest.TestCase):
    def test_capability_reports_unsupported_in_this_worktree_today(self):
        # This is the honest, currently-true state: real dispatch is not
        # wired, so the runner must refuse to fabricate a result.
        capability = runner.inspect_scheduler_dispatch_capability()
        self.assertFalse(bool(capability))

    def test_main_blocked_path_writes_log_entry_and_no_output_file(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "result.json"
            log_path = Path(tmpdir) / "central.jsonl"
            args = make_args(output=str(output_path), central_log=str(log_path))

            with patch.object(
                runner,
                "parse_args",
                return_value=args,
            ), patch.object(
                runner,
                "inspect_scheduler_dispatch_capability",
                return_value=CacheCraftServerCapability(
                    supported=False, reason="no dispatch wired"
                ),
            ), patch.object(
                runner,
                "run_real",
                side_effect=AssertionError(
                    "run_real must never be called when capability is unsupported"
                ),
            ):
                exit_code = runner.main()

            self.assertEqual(exit_code, runner.BLOCKED_EXIT_CODE)
            self.assertFalse(output_path.exists())
            self.assertTrue(log_path.exists())
            lines = log_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)
            entry = json.loads(lines[0])
            self.assertEqual(entry["status"], "blocked")
            self.assertEqual(entry["reason"], "no dispatch wired")
            self.assertEqual(entry["settings"]["mode"], "cachecraft")
            self.assertEqual(entry["settings"]["formal_repeats"], 4)

    def test_main_dispatches_to_run_real_when_capability_supported(self):
        args = make_args()
        with patch.object(runner, "parse_args", return_value=args), patch.object(
            runner,
            "inspect_scheduler_dispatch_capability",
            return_value=CacheCraftServerCapability(supported=True, reason="wired"),
        ), patch.object(runner, "run_real", return_value=0) as fake_run_real:
            exit_code = runner.main()
        self.assertEqual(exit_code, 0)
        fake_run_real.assert_called_once_with(args)


# ---------------------------------------------------------------------------
# The real-run client logic (request/build_metadata/run_round) is tested
# directly against a fake HTTP transport -- it is unreachable via main() in
# this worktree today (capability is unsupported), but must still be
# structurally correct so it needs no redesign once real dispatch exists.
# ---------------------------------------------------------------------------


class FakeStreamResponse:
    def __init__(self, lines: list[bytes]):
        self._lines = lines

    def raise_for_status(self):
        return None

    def iter_lines(self):
        return iter(self._lines)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class FakePlainResponse:
    def __init__(self, text: str = ""):
        self.text = text

    def raise_for_status(self):
        return None


FAKE_PROMETHEUS_TEXT = (
    "sglang:max_total_num_tokens 100000\n"
    "sglang:kv_available_tokens 100000\n"
    "sglang:kv_evictable_tokens 0\n"
    "sglang:kv_used_tokens 0\n"
)


class FakeSession:
    """A minimal fake HTTP transport recording every call it serves."""

    def __init__(self):
        self.calls: list[tuple[str, dict | None]] = []

    def get(self, url, timeout):
        del timeout
        self.calls.append(("GET", url, None))
        return FakePlainResponse(FAKE_PROMETHEUS_TEXT)

    def post(self, url, json=None, stream=False, timeout=None):
        del timeout
        self.calls.append(("POST", url, json))
        if url.endswith("/flush_cache"):
            return FakePlainResponse("")
        if url.endswith("/generate"):
            assert stream is True
            input_ids = json["input_ids"]
            cached_tokens = max(0, len(input_ids) - 1)
            payload = {
                "meta_info": {"cached_tokens": cached_tokens},
                "output_ids": [1],
            }
            return FakeStreamResponse(
                [
                    ("data: " + __import__("json").dumps(payload)).encode("utf-8"),
                    b"data: [DONE]",
                ]
            )
        raise AssertionError(f"unexpected URL: {url}")


class TestRequestClientLogic(unittest.TestCase):
    def test_request_parses_streaming_ttft_and_payload(self):
        session = FakeSession()
        result = runner.request(session, "http://fake", [1, 2, 3], None)
        self.assertGreaterEqual(result["ttft_ms"], 0.0)
        self.assertEqual(result["cached_tokens"], 2)
        self.assertEqual(result["output_ids"], [1])

    def test_build_metadata_shape(self):
        metadata = runner.build_metadata(
            operation="reuse",
            segments=[{"content_hash": "a", "target_start": 0, "length": 3}],
            plugin="cachecraft",
        )
        self.assertEqual(metadata["operation"], "reuse")
        self.assertEqual(metadata["plugin"], "cachecraft")
        self.assertEqual(len(metadata["segments"]), 1)

    def test_build_metadata_omits_plugin_when_not_given(self):
        metadata = runner.build_metadata(operation="register", segments=[])
        self.assertNotIn("plugin", metadata)

    def test_filler_prompt_is_deterministic(self):
        first = runner.filler_prompt(3, 16)
        second = runner.filler_prompt(3, 16)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 16)


class TestRunRoundContractShape(unittest.TestCase):
    def test_run_round_produces_expected_keys_and_reordered_workload(self):
        session = FakeSession()
        args = make_args(
            header_tokens=64,
            body_tokens=1024,
            target_rho=2.0,
            filler_tokens=736,
            num_chunks=3,
            segment_tokens=512,
        )
        row = runner.run_round(session, args, round_index=0)

        for key in (
            "round_index",
            "capacity_tokens",
            "target_rho",
            "actual_declared_rho",
            "peak_rho_with_target",
            "filler_count",
            "declared_working_tokens",
            "is_reordered_workload",
            "target",
            "baseline_metrics",
            "before_target_metrics",
            "after_target_metrics",
            "pressure_delta",
            "target_delta",
        ):
            self.assertIn(key, row)

        self.assertEqual(row["capacity_tokens"], 100000)
        self.assertTrue(row["is_reordered_workload"])
        self.assertGreater(row["filler_count"], 0)
        # peak includes the target body on top of the declared working set
        self.assertEqual(
            row["peak_rho_with_target"],
            (row["declared_working_tokens"] + args.body_tokens)
            / row["capacity_tokens"],
        )

        # A flush + baseline + (register per segment) + filler + header
        # probe + before-target snapshot + target request all happened
        # in a well-defined order.
        methods = [call[0] for call in session.calls]
        self.assertIn("POST", methods)
        self.assertIn("GET", methods)
        generate_calls = [
            call for call in session.calls if call[1].endswith("/generate")
        ]
        self.assertGreaterEqual(len(generate_calls), 1)

    def test_single_chunk_body_is_not_reordered(self):
        session = FakeSession()
        args = make_args(
            header_tokens=0,
            body_tokens=512,
            target_rho=1.0,
            num_chunks=1,
        )
        row = runner.run_round(session, args, round_index=0)
        self.assertFalse(row["is_reordered_workload"])


if __name__ == "__main__":
    unittest.main()
