from __future__ import annotations

import json
from pathlib import Path

from benchmark.multi_workflow import run_m55_v40_task_disjoint_campaign as m55


def _used_tasks(path: Path) -> set[str]:
    return {
        str(row["instance_id"])
        for row in json.loads(path.read_text(encoding="utf-8"))["cases"]
    }


def test_fresh13_selection_is_locked_and_task_disjoint() -> None:
    assert m55._selection_hash() == m55.SELECTION_SHA256
    used = set()
    for path in (
        Path(
            "/home/gfy/CodeMAS_Project/kvflow-artifacts/"
            "impactkv_m52_path_dependency_20260805/matched20/DESIGN.json"
        ),
        Path(
            "/home/gfy/CodeMAS_Project/kvflow-artifacts/"
            "impactkv_m53_path_dependency_holdout_20260805/"
            "request_disjoint19/DESIGN.json"
        ),
        Path(
            "/home/gfy/CodeMAS_Project/kvflow-artifacts/"
            "impactkv_m54_dependency_drift_hybrid_20260805/"
            "untouched14/DESIGN.json"
        ),
    ):
        used.update(_used_tasks(path))
    assert len(m55.TASKS) == 13
    assert not (set(m55.TASKS) & used)


def test_fresh13_tasks_exist_in_local_population() -> None:
    population = {
        str(row["instance_id"]) for row in m55.prior._population_rows()
    }
    assert set(m55.TASKS) <= population


def test_pooled_gate_uses_v44_without_changing_it() -> None:
    fresh = {m55.V40: 2, m55.GENERAL: 1, m55.DENSE: 2}
    pooled = m55._pooled_resolved(fresh)
    assert pooled is not None
    old = json.loads(m55.V44_RESULT.read_text(encoding="utf-8"))["aggregate"][
        "resolved"
    ]
    assert pooled == {arm: int(old[arm]) + fresh[arm] for arm in m55.ARMS}
