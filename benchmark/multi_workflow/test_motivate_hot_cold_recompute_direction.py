from benchmark.multi_workflow.motivate_hot_cold_recompute_direction import (
    COPY_TOKENS,
    _fixed_tail_candidate,
    pair_same_task_candidates,
)


def _candidate(candidate_id: str, hot: bool, length: int, start: int):
    return {
        "candidate_id": candidate_id,
        "module_type": "repository_code",
        "natural_length": length,
        "source_start": start + 10,
        "target_start": start,
        "target_end": start + length,
        "relation_control": {"relation": {"exact_path": True}} if hot else None,
    }


def _case(task: str, case_id: str, candidates):
    return {
        "instance_id": task,
        "case_id": case_id,
        "target_input_ids": list(range(1000)),
        "source_input_ids": list(range(1000)),
        "candidates": candidates,
    }


def test_pair_same_task_candidates_never_crosses_tasks():
    design = {
        "cases": [
            _case("task-a", "task-a-q1", [_candidate("cold-a", False, 256, 200)]),
            _case("task-a", "task-a-q2", [_candidate("hot-a", True, 260, 220)]),
            _case("task-b", "task-b-q1", [_candidate("cold-b", False, 300, 300)]),
            _case("task-b", "task-b-q2", [_candidate("hot-b", True, 320, 320)]),
        ]
    }
    pairs = pair_same_task_candidates(design)
    assert len(pairs) == 2
    assert all(pair["cold"]["instance_id"] == pair["hot"]["instance_id"] for pair in pairs)
    assert len({pair["cold"]["key"] for pair in pairs}) == 2
    assert len({pair["hot"]["key"] for pair in pairs}) == 2


def test_fixed_tail_candidate_has_exact_equal_budget():
    case = _case("task", "task-q1", [_candidate("cold", False, 300, 400)])
    row = {
        "candidate": case["candidates"][0],
        "candidate_id": "cold",
    }
    candidate = _fixed_tail_candidate(row, "pair", "cold")
    assert candidate["length"] == COPY_TOKENS
    assert candidate["target_start"] == 400 + 300 - COPY_TOKENS
    assert candidate["source_start"] == 410 + 300 - COPY_TOKENS
    assert candidate["dependency_hot"] is False
