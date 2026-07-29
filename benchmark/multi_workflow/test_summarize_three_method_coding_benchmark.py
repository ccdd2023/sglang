import json

from benchmark.multi_workflow.summarize_three_method_coding_benchmark import (
    build_audit,
    build_stability_diagnostic,
    render_markdown,
)


def _static(method, dense, reuse, speedup, n4):
    latency = {
        "cache_ready_speedup_vs_native_dense": speedup,
        "build_amortized": {"4": {"speedup_vs_native_dense": n4}},
    }
    if method.startswith("coding_"):
        latency = {
            "cache_ready_speedup": speedup,
            "n4_including_build_speedup": n4,
        }
    return {
        "method": method,
        "engine": f"{method}-engine",
        "samples": 50,
        "quality": {
            "dense_exact_line": dense,
            "reuse_exact_line": reuse,
            "dense_code_sim_percent": 50.0,
            "reuse_code_sim_percent": 51.0,
        },
        "latency": latency,
        "physical_reuse": {"mean_reused_k_tokens": 100},
    }


def _swe(resolved, ids, median, p95, copies=0):
    submitted = ["a", "b", "c"]
    return {
        "official": {
            "total_instances": 3,
            "resolved_instances": resolved,
            "resolved_ids": ids,
            "submitted_ids": submitted,
            "empty_patch_instances": 3 - resolved,
        },
        "runtime": {
            "requests": 10,
            "median_ttft_ms": median,
            "p95_ttft_ms": p95,
            "target_copy_events": copies,
            "target_fallback_events": 0,
            "copied_tokens": copies * 100,
        },
    }


def test_build_audit_keeps_claim_scopes_and_paired_outcomes():
    audit = build_audit(
        _static("coding_grounded_observation_island_v40", 5, 4, 1.1, 0.9),
        _static("CacheBlend", 5, 4, 1.5, 0.8),
        _static("KVCOMM", 4, 5, 13.0, 8.0),
        _swe(2, ["a", "b"], 300, 700),
        _swe(1, ["a"], 250, 500, copies=7),
    )

    assert audit["swebench_verified_agent"]["paired_outcomes"][
        "dense_pass_v40_fail"
    ] == ["b"]
    assert audit["swebench_verified_agent"]["v40"][
        "median_ttft_speedup_vs_dense"
    ] == 1.2
    assert audit["decision"]["v40_preserves_dense_swe_accuracy"] is False
    assert audit["decision"]["v40_beats_both_static_exact_line"] is False

    text = render_markdown(audit, {})
    assert "Absolute TTFT" in text
    assert "Dense-pass → V40-fail: b" in text
    assert "Native KVCOMM and CacheBlend SWE-bench results were not run" in text


def test_stability_diagnostic_does_not_promote_post_hoc_repeat(tmp_path):
    headline = {
        "paired_outcomes": {"dense_pass_v40_fail": ["task-a"]}
    }
    for arm, resolved, empty, copies in [
        ("dense", 0, 1, 0),
        ("coding_grounded_observation_island_v40", 0, 0, 4),
    ]:
        path = tmp_path / arm / "canary_task-a"
        path.mkdir(parents=True)
        (path / "PIPELINE_STATUS.json").write_text(
            json.dumps(
                {
                    "official": {
                        "resolved_instances": resolved,
                        "empty_patch_instances": empty,
                    },
                    "runtime": {"target_copy_events": copies},
                }
            )
        )

    value = build_stability_diagnostic(tmp_path, headline)
    assert value["status"] == "post_hoc_diagnostic_not_headline"
    assert value["stable_regressions_reproduced"] == 0
    assert value["tasks"][0]["repeat_v40_copy_events"] == 4
