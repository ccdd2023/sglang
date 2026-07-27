from __future__ import annotations

import math

from benchmark.multi_workflow.audit_bridge_three_axis_v18 import (
    exact_mcnemar_p,
    paired_bootstrap_ci,
    transition_audit,
    wilson_ci,
)


def test_transition_audit_separates_damage_and_rescue() -> None:
    universe = ["a", "b", "c", "d", "e"]
    result = transition_audit({"a", "b"}, {"a", "c"}, universe)

    assert result["both_pass"] == 1
    assert result["damage_dense_pass_to_reuse_fail"] == 1
    assert result["damage_ids"] == ["b"]
    assert result["rescue_dense_fail_to_reuse_pass"] == 1
    assert result["rescue_ids"] == ["c"]
    assert result["both_fail"] == 2
    assert result["reuse_minus_dense_pp"] == 0.0


def test_exact_mcnemar_is_two_sided_and_capped() -> None:
    assert exact_mcnemar_p(0, 0) == 1.0
    assert exact_mcnemar_p(3, 3) == 1.0
    assert math.isclose(exact_mcnemar_p(0, 4), 0.125)


def test_wilson_interval_contains_observed_rate() -> None:
    low, high = wilson_ci(2, 6)
    assert low < 2 / 6 < high
    assert 0.0 <= low <= high <= 1.0


def test_paired_bootstrap_is_deterministic() -> None:
    deltas = [100.0, -100.0, 0.0, 0.0]
    left = paired_bootstrap_ci(deltas, seed=7, iterations=1_000)
    right = paired_bootstrap_ci(deltas, seed=7, iterations=1_000)
    assert left == right
    assert left[0] <= 0.0 <= left[1]
