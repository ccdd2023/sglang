from __future__ import annotations

from benchmark.multi_workflow.validate_single_island_probe_transfer import (
    mean_within_case_spearman,
    select_arms,
)


def test_select_arms_keeps_equal_budget_and_uses_frozen_orders() -> None:
    case = {
        "case_id": "task-q8",
        "candidates": [
            {"candidate_id": "recent"},
            {"candidate_id": "middle"},
            {"candidate_id": "old"},
        ],
    }
    arms = select_arms(
        case,
        probe_by_id={"recent": 0.3, "middle": 0.1, "old": 0.2},
        oracle_by_id={"recent": 0.2, "middle": 0.3, "old": 0.1},
    )
    assert arms["current_recency"] == "recent"
    assert arms["fixed_probe_min"] == "middle"
    assert arms["module_attention_oracle"] == "old"
    assert set(arms.values()) <= {"recent", "middle", "old"}


def test_mean_within_case_spearman_ignores_singletons() -> None:
    scores = {
        "a": {"x": 1.0, "y": 2.0, "z": 3.0},
        "b": {"one": 7.0},
        "c": {"left": 1.0, "right": 2.0},
    }
    outcomes = {
        "a": {"x": 2.0, "y": 4.0, "z": 8.0},
        "b": {"one": 1.0},
        "c": {"left": 9.0, "right": 3.0},
    }
    # Case a has rho=1 and case c has rho=-1; case b is not rankable.
    assert mean_within_case_spearman(scores, outcomes) == 0.0
