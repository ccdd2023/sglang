from __future__ import annotations

import math

from benchmark.multi_workflow.motivate_natural_module_attention import (
    _balanced_cases,
    crossfit_predictions,
    spearman,
)


def test_balanced_cases_round_robins_tasks_and_prefers_type_coverage() -> None:
    rows = []
    for task in ("a", "b", "c"):
        for request in range(6):
            candidates = [{"module_type": "repository_code"}]
            if request == 5:
                candidates.append(
                    {
                        "module_type": "assistant_interpretation",
                        "relation_control": {"matched": True},
                    }
                )
            for candidate in candidates:
                candidate.setdefault("relation_control", None)
            rows.append(
                {
                    "instance_id": task,
                    "case_id": f"{task}-{request}",
                    "candidates": candidates,
                }
            )
    selected = _balanced_cases(rows)
    assert len(selected) == 12
    assert {row["instance_id"] for row in selected[:3]} == {"a", "b", "c"}
    assert all(len(row["candidates"]) == 2 for row in selected[:3])


def test_spearman_and_crossfit_prediction_are_finite() -> None:
    rows = []
    for task_index in range(4):
        for point_index in range(8):
            density = 0.01 + 0.002 * point_index + 0.003 * task_index
            rows.append(
                {
                    "instance_id": f"task-{task_index}",
                    "key_tokens": 32 + point_index,
                    "query_tokens": 16 + point_index,
                    "token_distance": 100 * point_index,
                    "key_position": point_index / 10,
                    "query_position": (point_index + 1) / 10,
                    "interaction_distance": point_index % 3,
                    "layer": (0, 8, 17, 26, 35)[point_index % 5],
                    "module_type": "repository_code" if point_index % 2 else "assistant_interpretation",
                    "query_module_type": "assistant_interpretation",
                    "kind": "intra_natural" if point_index % 3 else "intra_boundary",
                    "exact_path": point_index % 2 == 0,
                    "same_directory": True,
                    "shared_symbol": point_index % 3 == 0,
                    "interpretation_grounding": False,
                    "attention_density": density,
                }
            )
    prediction = crossfit_predictions(rows, True)
    assert all(math.isfinite(value) for value in prediction)
    assert -1 <= spearman(prediction, [row["attention_density"] for row in rows]) <= 1
