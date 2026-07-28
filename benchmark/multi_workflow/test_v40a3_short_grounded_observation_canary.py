from benchmark.multi_workflow import (
    run_v40a3_short_grounded_observation_canary as v40a3,
)


def test_v40a3_selection_is_short_and_outcome_independent() -> None:
    assert v40a3._selected_task() == ("pytest-dev__pytest-7982", 13, 6)
