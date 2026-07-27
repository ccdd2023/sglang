from benchmark.multi_workflow.run_v27_dense_pass_triple_campaign import (
    DENSE_SCREEN,
    selection,
)
from benchmark.multi_workflow.run_frozen_trajectory_replay_v18 import read_json


def test_v27_selection_is_exact_frozen_dense_pass_set() -> None:
    expected = sorted(read_json(DENSE_SCREEN)["report"]["resolved_ids"])
    actual = [row["instance_id"] for row in selection()]
    assert actual == expected
    assert len(actual) == 6
    assert len(set(actual)) == len(actual)
