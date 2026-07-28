from benchmark.multi_workflow import run_v41_v40_independent_campaign as v41


def test_v41_selection_is_frozen() -> None:
    assert v41._selection_hash() == v41.SELECTION_SHA256
    assert len(v41.TASKS) == 6
