from __future__ import annotations

from collections import Counter

from benchmark.multi_workflow.run_attention_kv_task_disjoint_campaign import (
    MAX_PER_REPOSITORY,
    select_tasks,
)


def test_select_tasks_is_deterministic_disjoint_and_repo_capped() -> None:
    difficulties = ("15 min - 1 hour", "1-4 hours", ">4 hours")
    rows = [
        {
            "instance_id": f"repo{index % 12}__task-{index}",
            "repo": f"owner/repo{index % 12}",
            "difficulty": difficulties[index % 3],
        }
        for index in range(80)
    ]
    excluded = {"repo0__task-0", "repo1__task-1"}
    left = select_tasks(rows, excluded, 20)
    right = select_tasks(rows, excluded, 20)
    assert left == right
    assert len(left) == 20
    assert not ({row["instance_id"] for row in left} & excluded)
    counts = Counter(row["repo"] for row in left)
    assert max(counts.values()) <= MAX_PER_REPOSITORY
