from __future__ import annotations

import pytest

from benchmark.multi_workflow.run_natural_code_module_stage_overhead import (
    select_code_cases,
)


def _case(candidates: list[dict]) -> dict:
    source = list(range(30))
    target = list(range(30))
    return {
        "case_id": "case",
        "instance_id": "task",
        "source_input_ids": source,
        "target_input_ids": target,
        "candidates": candidates,
    }


def test_select_code_cases_keeps_variable_lengths_and_ignores_interpretation() -> None:
    design = {
        "cases": [
            _case(
                [
                    {"candidate_id": "code", "module_type": "repository_code", "natural_length": 3, "source_start": 4, "target_start": 4},
                    {"candidate_id": "thought", "module_type": "assistant_interpretation", "natural_length": 4, "source_start": 12, "target_start": 12},
                ]
            )
        ]
    }
    result = select_code_cases(design, ["case::code", "case::thought"])
    assert [row["length"] for row in result[0]["spans"]] == [3]


def test_select_code_cases_rejects_overlapping_modules() -> None:
    design = {
        "cases": [
            _case(
                [
                    {"candidate_id": "a", "module_type": "repository_code", "natural_length": 5, "source_start": 2, "target_start": 2},
                    {"candidate_id": "b", "module_type": "repository_code", "natural_length": 4, "source_start": 5, "target_start": 5},
                ]
            )
        ]
    }
    with pytest.raises(ValueError, match="overlap"):
        select_code_cases(design, ["case::a", "case::b"])
