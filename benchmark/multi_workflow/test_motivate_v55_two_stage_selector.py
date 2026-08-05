from __future__ import annotations

from benchmark.multi_workflow import motivate_v55_two_stage_selector as m55


def _row(relevant_risk: float, disjoint_risk: float) -> dict:
    return {
        "case_id": "case",
        "instance_id": "task",
        "candidates": [
            {
                "candidate_id": "path_relevant",
                "group_index": 3,
                "target_start": 100,
                "length": 128,
                "probe_score": relevant_risk,
                "attention_mean": 0.04,
                "causal_splice_logit_js": 0.003,
            },
            {
                "candidate_id": "path_disjoint",
                "group_index": 5,
                "target_start": 300,
                "length": 128,
                "probe_score": disjoint_risk,
                "attention_mean": 0.01,
                "causal_splice_logit_js": 0.001,
            },
        ],
    }


def test_two_stage_filters_risk_before_path_priority() -> None:
    threshold = m55.PROBE_THRESHOLD
    safe_relevant = _row(threshold / 2, threshold / 3)
    unsafe_relevant = _row(threshold * 2, threshold / 3)
    all_unsafe = _row(threshold * 2, threshold * 3)
    assert m55.select(safe_relevant, "two_stage")["candidate_id"] == "path_relevant"
    assert m55.select(unsafe_relevant, "two_stage")["candidate_id"] == "path_disjoint"
    assert m55.select(all_unsafe, "two_stage") is None


def test_control_selectors_are_deterministic() -> None:
    row = _row(0.01, 0.005)
    assert m55.select(row, "path_only")["candidate_id"] == "path_relevant"
    assert m55.select(row, "probe_only")["candidate_id"] == "path_disjoint"
    assert m55.select(row, "v40_fixed_budget_recency")["candidate_id"] == "path_disjoint"
    assert m55.select(row, "seeded_random") == m55.select(row, "seeded_random")


def test_pair_metrics_treat_ties_as_half() -> None:
    assert m55._direction_fraction([1.0, 2.0], [1.0, 1.0], higher=True) == 0.75
    assert m55._direction_fraction([1.0, 0.0], [1.0, 1.0], higher=False) == 0.75
    assert m55._geometric_ratio([2.0, 8.0], [1.0, 2.0]) > 2.8


def _command_group(command: str, output: str = "") -> list[dict]:
    return [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "bash",
                        "arguments": {"command": command},
                    }
                }
            ],
        },
        {
            "role": "tool",
            "content": f"<returncode>0</returncode><output>{output}</output>",
        },
    ]


def test_version_validity_is_fail_closed_at_target() -> None:
    candidate = {"group_index": 0, "repository_paths": ["pkg/module.py"]}
    read = _command_group(
        "sed -n '1,200p' /testbed/pkg/module.py", "source " * 80
    )
    disjoint_write = _command_group("sed -i 's/x/y/' /testbed/pkg/other.py")
    same_path_write = _command_group("sed -i 's/x/y/' /testbed/pkg/module.py")
    assert m55._version_valid_at_target(candidate, [read, disjoint_write], 2)
    assert not m55._version_valid_at_target(candidate, [read, same_path_write], 2)
    assert not m55._version_valid_at_target(
        {"group_index": 0, "repository_paths": []}, [read], 1
    )


def test_gpu_labels_stay_sealed_when_task_capacity_fails() -> None:
    assert m55._gpu_eligible(m55.MIN_CASES, m55.MIN_TASKS)
    assert not m55._gpu_eligible(m55.MIN_CASES + 100, m55.MIN_TASKS - 1)
    assert not m55._gpu_eligible(m55.MIN_CASES - 1, m55.MIN_TASKS + 100)
