from benchmark.multi_workflow import run_v36_v35b_task_level_campaign as v36


def test_v36_sha_selection_is_frozen_and_outcome_independent():
    selected = v36._selection()

    assert tuple(row["instance_id"] for row in selected) == v36.EXPECTED
    assert len(selected) == v36.SAMPLE_SIZE
    assert all(row["reached"] for row in selected)
    assert all(row["instance_id"] not in v36.EXCLUDED for row in selected)
    assert all("resolved" not in row for row in selected)


def test_v36_bootstrap_and_wilson_are_deterministic():
    assert v36._bootstrap([1, 0, 1, -1, 0, 1]) == v36._bootstrap(
        [1, 0, 1, -1, 0, 1]
    )
    interval = v36._wilson(3, 6)
    assert interval is not None
    assert interval[0] < 0.5 < interval[1]
