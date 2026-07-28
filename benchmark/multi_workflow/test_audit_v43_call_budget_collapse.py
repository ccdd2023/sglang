from benchmark.multi_workflow.audit_v43_call_budget_collapse import (
    ARMS,
    BRANCH_REQUEST_INDEX,
    DENSE,
    GENERAL,
    SHARED_CALLS,
    STEP_LIMIT,
    TASKS,
    V40,
)


def test_v43_call_budget_audit_constants_are_frozen() -> None:
    assert V40 == "coding_grounded_observation_island_v40"
    assert GENERAL == "general"
    assert DENSE == "dense"
    assert ARMS == (V40, GENERAL, DENSE)
    assert STEP_LIMIT == 20
    assert SHARED_CALLS == 7
    assert BRANCH_REQUEST_INDEX == 8
    assert TASKS == (
        "sphinx-doc__sphinx-9461",
        "pydata__xarray-2905",
        "sympy__sympy-21930",
        "django__django-16263",
        "mwaskom__seaborn-3187",
        "pytest-dev__pytest-5840",
    )
