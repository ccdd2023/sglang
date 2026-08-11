from __future__ import annotations

import math

from benchmark.multi_workflow.analyze_attention_kv_factorial import (
    _cluster_bootstrap_cell_medians,
    _interaction_regression,
    assign_cell,
)


def test_assign_cell_uses_inclusive_frozen_medians() -> None:
    assert (
        assign_cell(attention=0.5, drift=0.5, attention_median=0.5, drift_median=0.5)
        == "high_attention__high_drift"
    )
    assert (
        assign_cell(attention=0.4, drift=0.6, attention_median=0.5, drift_median=0.5)
        == "low_attention__high_drift"
    )


def test_cluster_bootstrap_is_deterministic() -> None:
    rows = [
        {"case_id": f"c{index}", "cell": cell, "causal_splice_logit_js": value}
        for index, (cell, value) in enumerate(
            [
                ("high_attention__high_drift", 4.0),
                ("high_attention__high_drift", 3.0),
                ("low_attention__high_drift", 1.0),
                ("low_attention__high_drift", 2.0),
            ]
        )
    ]
    left = _cluster_bootstrap_cell_medians(rows, draws=100, seed=17)
    right = _cluster_bootstrap_cell_medians(rows, draws=100, seed=17)
    assert left == right
    assert left["high_attention__high_drift"]["median"] > left[
        "low_attention__high_drift"
    ]["median"]


def test_interaction_regression_is_finite() -> None:
    rows = []
    for attention in (0.1, 0.2, 0.4, 0.8):
        for drift in (0.01, 0.02, 0.04, 0.08):
            rows.append(
                {
                    "attention_mean": attention,
                    "kv_cosine_drift_mean": drift,
                    "causal_splice_logit_js": attention * drift,
                }
            )
    result = _interaction_regression(rows)
    assert all(math.isfinite(value) for value in result.values())
    assert result["prediction_spearman"] > 0.9
