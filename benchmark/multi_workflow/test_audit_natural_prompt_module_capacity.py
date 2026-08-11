from __future__ import annotations

from benchmark.multi_workflow.audit_natural_prompt_module_capacity import (
    boundary_control,
    recency_control,
)


def _module(identifier: str, kind: str, start: int, end: int, parent: str, request: int) -> dict:
    return {
        "module_id": identifier,
        "module_type": kind,
        "token_start": start,
        "token_end": end,
        "natural_length": end - start,
        "parent_interaction_id": parent,
        "source_request_index": request,
    }


def test_boundary_control_crosses_a_real_same_parent_edge() -> None:
    modules = [
        _module("interpret", "assistant_interpretation", 100, 140, "p", 2),
        _module("command", "tool_command", 140, 170, "p", 2),
        _module("code", "repository_code", 170, 270, "p", 2),
    ]
    control = boundary_control(modules[2], modules)
    assert control is not None
    start, end = control
    assert end - start == 100
    assert (start < 140 < end) or (start < 170 < end)
    assert (start, end) != (170, 270)


def test_recency_control_preserves_type_and_equal_length_capacity() -> None:
    modules = [
        _module("old", "repository_code", 100, 180, "old-parent", 1),
        _module("candidate", "repository_code", 200, 250, "new-parent", 2),
        _module("wrong", "assistant_interpretation", 260, 400, "new-parent", 2),
    ]
    selected = recency_control(modules[1], modules, 500)
    assert selected is modules[0]
    assert selected["natural_length"] >= modules[1]["natural_length"]

