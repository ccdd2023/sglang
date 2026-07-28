from benchmark.multi_workflow.run_v44_dense_sensitive_v40_campaign import (
    ARMS,
    DENSE,
    DENSE_PASS_SENSITIVITY_MIN,
    GENERAL,
    SELECTION_SHA256,
    STEP_LIMIT,
    TASKS,
    V40,
    _selected_tasks,
    _selection_hash,
)


def test_v44_selection_and_protocol_are_frozen() -> None:
    assert V40 == "coding_grounded_observation_island_v40"
    assert GENERAL == "general"
    assert DENSE == "dense"
    assert ARMS == (V40, GENERAL, DENSE)
    assert STEP_LIMIT == 32
    assert DENSE_PASS_SENSITIVITY_MIN == 2
    assert _selection_hash() == SELECTION_SHA256
    assert _selected_tasks() == TASKS
    assert len(TASKS) == 12
    assert len(set(TASKS)) == len(TASKS)
