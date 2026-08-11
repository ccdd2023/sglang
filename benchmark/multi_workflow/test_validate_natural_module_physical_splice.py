from __future__ import annotations

from benchmark.multi_workflow.validate_natural_module_physical_splice import (
    MINIMAL_RELIABLE_GATE_POLICY,
    STRONG_GATE_POLICY,
    attention_admission_gates,
    select_balanced,
)


def test_physical_selection_round_robins_tasks() -> None:
    rows = [
        {
            "instance_id": f"task-{task}",
            "candidate_key": f"task-{task}-candidate-{candidate}",
        }
        for task in range(10)
        for candidate in range(6)
    ]
    selected = select_balanced(rows, 32)
    assert len(selected) == 32
    assert len({value.split("-candidate")[0] for value in selected[:10]}) == 10


def test_minimal_reliable_gate_accepts_small_but_reliable_advantage() -> None:
    result = {
        "status": "STOP_BEFORE_PHYSICAL_SPLICE",
        "gates": {"old_effect_size_gate": False},
        "type_results": {
            module_type: {
                "raw_natural_to_boundary_paired_direction": 0.51,
                "task_bootstrap_adjusted_ratio_q025_q50_q975": [1.001, 1.02, 1.04],
            }
            for module_type in ("repository_code", "assistant_interpretation")
        },
        "prediction": {
            "task_bootstrap_improvement_q025_q50_q975": [0.001, 0.02, 0.04]
        },
    }
    assert all(
        attention_admission_gates(result, MINIMAL_RELIABLE_GATE_POLICY).values()
    )
    assert not all(attention_admission_gates(result, STRONG_GATE_POLICY).values())


def test_minimal_reliable_gate_rejects_uncertain_advantage() -> None:
    result = {
        "status": "STOP_BEFORE_PHYSICAL_SPLICE",
        "gates": {},
        "type_results": {
            module_type: {
                "raw_natural_to_boundary_paired_direction": 0.60,
                "task_bootstrap_adjusted_ratio_q025_q50_q975": [0.99, 1.02, 1.05],
            }
            for module_type in ("repository_code", "assistant_interpretation")
        },
        "prediction": {
            "task_bootstrap_improvement_q025_q50_q975": [0.001, 0.02, 0.04]
        },
    }
    assert not all(
        attention_admission_gates(result, MINIMAL_RELIABLE_GATE_POLICY).values()
    )
