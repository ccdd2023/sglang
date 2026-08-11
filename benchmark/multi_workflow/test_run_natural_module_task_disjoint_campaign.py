from __future__ import annotations

from collections import Counter

from benchmark.multi_workflow.run_natural_module_task_disjoint_campaign import (
    DIFFICULTY_QUOTAS,
    INITIAL_REPO_CAP,
    MAX_REPO_CAP,
    expand_capacity_ceiling,
    select_initial,
)


def test_selection_freezes_initial_and_expansion_before_outcomes() -> None:
    difficulties = tuple(DIFFICULTY_QUOTAS)
    rows = [
        {
            "instance_id": f"task-{index}",
            "repo": f"owner/repo-{index % 12}",
            "difficulty": difficulties[index % len(difficulties)],
        }
        for index in range(120)
    ]
    excluded = {"task-0", "task-1"}
    initial = select_initial(rows, excluded)
    maximum = expand_capacity_ceiling(rows, excluded, initial)
    assert len(initial) == 20
    assert len(maximum) == 29
    assert maximum[:20] == initial
    assert not ({row["instance_id"] for row in maximum} & excluded)
    assert max(Counter(row["repo"] for row in initial).values()) <= INITIAL_REPO_CAP
    assert max(Counter(row["repo"] for row in maximum).values()) <= MAX_REPO_CAP
    assert Counter(row["difficulty"] for row in initial) == Counter(DIFFICULTY_QUOTAS)

