from __future__ import annotations

from benchmark.multi_workflow.analyze_single_island_probe_transfer_failure import (
    analyze,
    drift_quartiles,
)


def _registered_result() -> dict:
    arms = {"current_recency": {"median_final_logit_js": 1.0}}
    for name, win, ratio in (
        ("fixed_probe_min", 0.5, 0.8),
        ("module_attention_oracle", 0.6, 0.7),
        ("seeded_random", 0.4, 1.1),
    ):
        arms[name] = {
            "vs_recency": {
                "win_fraction": win,
                "median_js_ratio_all_cases": ratio,
            }
        }
    return {"arms": arms, "decision": "NOT_SUPPORTED_FOR_RUNTIME_CANARY"}


def test_analysis_separates_probe_approximation_from_final_target() -> None:
    rows = []
    # The probe perfectly follows drift, while the final metric alternates.
    for index, final in enumerate((1.0, 4.0, 2.0, 3.0, 1.5, 3.5, 2.5, 0.5), 1):
        rows.append(
            {
                "case_id": f"case-{index}",
                "instance_id": f"task-{index % 4}",
                "probe_score": float(index),
                "full_128_token_kv_drift": float(index),
                "max_qualifying_module_attention": 0.1,
                "max_module_attention_x_full_drift": float(index) * 0.1,
                "module_oracle_risk": float(9 - index),
                "final_logit_js": final,
                "top1_changed": index == 8,
            }
        )
    result = analyze(rows, _registered_result())
    assert result["correlations"]["probe_to_full_kv_drift"] == 1.0
    assert result["correlations"]["probe_to_final_js"] < 0.5
    assert result["immediate_behavior_resolution"]["top1_changes"] == 1
    assert result["development_decision"].startswith("STOP_PROBE_TUNING")


def test_drift_quartiles_preserve_all_candidates() -> None:
    rows = [
        {"full_128_token_kv_drift": float(index), "final_logit_js": 10.0 - index}
        for index in range(1, 10)
    ]
    groups = drift_quartiles(rows)
    assert len(groups) == 4
    assert sum(group["candidates"] for group in groups) == len(rows)
    assert groups[0]["median_full_kv_drift"] < groups[-1]["median_full_kv_drift"]
