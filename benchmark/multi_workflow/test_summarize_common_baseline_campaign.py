from __future__ import annotations

import json

from benchmark.multi_workflow import summarize_common_baseline_campaign as summary


def dump(path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_sglang_accuracy_and_exact_rows_use_common_schema(tmp_path) -> None:
    run = tmp_path / "runs/sglang_formal/dense/full_24"
    dump(
        run / "RUNTIME_SUMMARY.json",
        {
            "requests": 100,
            "median_ttft_ms": 50.0,
            "target_copy_events": 0,
            "copied_tokens": 0,
            "target_fallback_events": 0,
        },
    )
    dump(
        run / "reports/enroot/report.json",
        {"submitted_instances": 24, "resolved_instances": 3},
    )
    row = summary.sglang_accuracy_row(tmp_path, "formal", "dense", "dense")
    assert row is not None
    assert row["resolved"] == 3
    assert row["physical_reuse_requests"] == 0

    dump(
        tmp_path / "exact_prompt_replay/fresh24/sglang_coding/RESULT.json",
        {
            "status": "PASS",
            "summary": {"physical_copy_events": 28},
            "targets": [
                {
                    "rounds_per_arm": 10,
                    "cache_ready_speedup": 1.2,
                    "n1_including_build_speedup": 0.8,
                    "n4_including_build_speedup": 1.05,
                    "n16_including_build_speedup": 1.15,
                }
            ],
        },
    )
    exact = summary.exact_row(tmp_path, "fresh24", "sglang_coding")
    assert exact is not None
    assert exact["physical_reuse_rounds"] == 28
    assert exact["median_n4_including_build_speedup"] == 1.05
