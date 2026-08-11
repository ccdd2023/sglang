from __future__ import annotations

from benchmark.multi_workflow.motivate_module_conditioned_attention_kv import (
    _balanced_requests,
    _crossfit_candidate_risk,
    _select_physical_candidates,
    module_for_block,
)


def test_module_for_block_is_candidate_path_conditional() -> None:
    block = {
        "category": "read_observation",
        "paths": ["src/parser.py"],
    }
    assert (
        module_for_block(block, {"src/parser.py"})
        == "read_observation_path_relevant"
    )
    assert (
        module_for_block(block, {"tests/test_other.py"})
        == "read_observation_path_disjoint"
    )
    assert module_for_block({"category": "generation_marker"}, set()) == "generation_marker"


def test_balanced_requests_takes_one_per_task_before_second() -> None:
    rows = [
        {"instance_id": task, "request_index": request, "case_id": f"{task}-{request}"}
        for task in ("a", "b", "c")
        for request in range(6)
    ]
    selected = _balanced_requests(rows, 6)
    assert len(selected) == 6
    assert {row["instance_id"] for row in selected[:3]} == {"a", "b", "c"}
    assert max(sum(row["instance_id"] == task for row in selected) for task in ("a", "b", "c")) == 2


def test_physical_subset_greedily_covers_module_cells() -> None:
    modules = ("generation_marker", "assistant_action", "other_tool_result")
    cells = (
        "low_attention__low_drift",
        "high_attention__low_drift",
        "low_attention__high_drift",
        "high_attention__high_drift",
    )
    points = []
    for cell_index, cell in enumerate(cells):
        for candidate_index in range(12):
            for module in modules:
                points.append(
                    {
                        "case_id": f"case-{cell_index}-{candidate_index}",
                        "candidate_id": "candidate",
                        "instance_id": f"task-{candidate_index % 6}",
                        "module": module,
                        "cell": cell,
                    }
                )
    selected = _select_physical_candidates(points, modules)
    assert len(selected) == 48


def test_physical_subset_keeps_sampling_until_task_coverage_is_met() -> None:
    module = "generation_marker"
    cells = (
        "low_attention__low_drift",
        "high_attention__low_drift",
        "low_attention__high_drift",
        "high_attention__high_drift",
    )
    points = []
    # Give one task enough candidates to fill the twelve-point quota by itself.
    # The selector must nevertheless retain candidates from five more tasks.
    for cell_index, cell in enumerate(cells):
        for candidate_index in range(12):
            points.append(
                {
                    "case_id": f"bulk-{cell_index}-{candidate_index}",
                    "candidate_id": "candidate",
                    "instance_id": "task-0",
                    "module": module,
                    "cell": cell,
                }
            )
        for task_index in range(1, 6):
            points.append(
                {
                    "case_id": f"coverage-{cell_index}-{task_index}",
                    "candidate_id": "candidate",
                    "instance_id": f"task-{task_index}",
                    "module": module,
                    "cell": cell,
                }
            )
    selected = set(_select_physical_candidates(points, (module,)))
    for cell_index, cell in enumerate(cells):
        selected_points = [
            point
            for point in points
            if point["cell"] == cell
            and f"{point['case_id']}::{point['candidate_id']}" in selected
        ]
        assert len(selected_points) >= 12
        assert len({point["instance_id"] for point in selected_points}) >= 6


def test_crossfit_risk_never_trains_on_held_out_task() -> None:
    modules = ("generation_marker", "assistant_action")
    points = []
    for task_index in range(4):
        for candidate_index in range(4):
            for module_index, module in enumerate(modules):
                attention = 0.01 * (candidate_index + 1 + module_index)
                drift = 0.02 * (candidate_index + 1)
                points.append(
                    {
                        "instance_id": f"task-{task_index}",
                        "case_id": f"task-{task_index}-case",
                        "candidate_id": f"candidate-{candidate_index}",
                        "module": module,
                        "attention_mass": attention,
                        "raw_kv_drift": drift,
                        "actual_kv_output_relative_mean": attention * drift,
                    }
                )
    risks, thresholds = _crossfit_candidate_risk(
        training_points=points, all_points=points, modules=modules
    )
    assert len(risks) == 16
    assert len(thresholds) == 4
    assert all(value == value for value in risks.values())
