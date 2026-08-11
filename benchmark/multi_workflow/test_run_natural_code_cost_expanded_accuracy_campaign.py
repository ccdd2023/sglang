from collections import Counter

from benchmark.multi_workflow.run_natural_code_cost_expanded_accuracy_campaign import (
    DIFFICULTY_QUOTAS,
    REPO_CAP,
    TASKS,
    _walk_instance_ids,
    mcnemar_exact_two_sided,
    select_cohort,
)


def test_select_cohort_satisfies_frozen_quotas_and_repository_cap() -> None:
    rows = []
    difficulties = tuple(DIFFICULTY_QUOTAS)
    for repo_index in range(8):
        for difficulty_index, difficulty in enumerate(difficulties):
            for item_index in range(5):
                rows.append(
                    {
                        "instance_id": (
                            f"repo{repo_index}__pkg-"
                            f"{100 * difficulty_index + item_index}"
                        ),
                        "repo": f"owner/repo{repo_index}",
                        "difficulty": difficulty,
                    }
                )
    selected = select_cohort(rows, set())
    assert len(selected) == TASKS
    assert Counter(row["difficulty"] for row in selected) == Counter(
        DIFFICULTY_QUOTAS
    )
    assert max(Counter(row["repo"] for row in selected).values()) <= REPO_CAP


def test_selection_exclusion_is_hard() -> None:
    rows = []
    for repo_index in range(8):
        for difficulty_index, difficulty in enumerate(DIFFICULTY_QUOTAS):
            for item_index in range(6):
                rows.append(
                    {
                        "instance_id": (
                            f"repo{repo_index}__pkg-"
                            f"{100 * difficulty_index + item_index}"
                        ),
                        "repo": f"owner/repo{repo_index}",
                        "difficulty": difficulty,
                    }
                )
    excluded = {row["instance_id"] for row in rows[::7]}
    selected = select_cohort(rows, excluded)
    assert not excluded.intersection(row["instance_id"] for row in selected)


def test_walk_instance_ids_and_mcnemar() -> None:
    value = {
        "instances": [
            {"instance_id": "django__django-123"},
            {"instance_id": "not-an-instance"},
        ],
        "nested": {"instance_id": "sympy__sympy-456"},
    }
    assert set(_walk_instance_ids(value)) == {
        "django__django-123",
        "sympy__sympy-456",
    }
    assert mcnemar_exact_two_sided(0, 0) == 1.0
    assert mcnemar_exact_two_sided(2, 1) == 1.0
    assert mcnemar_exact_two_sided(5, 0) == 0.0625
