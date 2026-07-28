from benchmark.multi_workflow.run_v43_new_verified_v40_campaign import (
    ARMS,
    SELECTION_SHA256,
    TASKS,
    _selection_hash,
    _selected_tasks,
)


def test_v43_selection_is_frozen_and_repository_diverse() -> None:
    assert len(TASKS) == 6
    assert len({task.split("__", 1)[0] for task in TASKS}) == 6
    assert _selected_tasks() == TASKS
    assert _selection_hash() == SELECTION_SHA256
    assert ARMS == (
        "coding_grounded_observation_island_v40",
        "general",
        "dense",
    )
