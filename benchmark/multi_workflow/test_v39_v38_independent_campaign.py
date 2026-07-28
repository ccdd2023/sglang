from benchmark.multi_workflow import (
    run_v39_v38_independent_campaign as v39,
)


def test_v39_selection_is_frozen_and_all_tasks_latch() -> None:
    assert v39._selection_hash() == v39.SELECTION_SHA256
    rows = v39._motivation_rows()
    selected = [rows[instance_id] for instance_id in v39.TASKS]
    assert len(selected) == 6
    assert all(row["latched"] for row in selected)
    assert {
        "django__django-14855",
        "pydata__xarray-6461",
        "pylint-dev__pylint-4970",
        "pytest-dev__pytest-7432",
        "pytest-dev__pytest-7982",
        "sympy__sympy-24539",
    } == set(v39.TASKS)
