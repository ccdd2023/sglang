from benchmark.multi_workflow import (
    run_v40a2_grounded_observation_canary as v40a2,
)


def test_v40a2_selection_is_outcome_independent_and_source_rich() -> None:
    assert v40a2._selected_task() == ("sphinx-doc__sphinx-9230", 14)
