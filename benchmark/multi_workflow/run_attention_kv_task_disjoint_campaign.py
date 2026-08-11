#!/usr/bin/env python3
"""Build a task-disjoint Dense coding-agent cohort for Attention/KV studies.

The cohort is selected without model outcomes from the local SWE-bench
Verified population.  It excludes every task used by the trajectory-backed
M50--M56 mechanism studies, freezes a 20-task dataset, and runs only the Dense
arm of the existing fixed coding-agent backend.  Official evaluation is not
part of this mechanism campaign.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark.multi_workflow import run_bridge_reuse_agent_experiment as bridge


ROOT = Path("/home/gfy/CodeMAS_Project")
ARTIFACTS = ROOT / "kvflow-artifacts"
POPULATION = (
    ROOT
    / "sglang-kvflow/results/repo_level_datasets/"
    "swe_verified_500_instances.json"
)
DEFAULT_OUTPUT = ARTIFACTS / "impactkv_attention_kv_task_disjoint_20260807_r1"
SELECTION_SALT = "attention-kv-module-confirm-20260807-v1"
INITIAL_TASKS = 20
MAX_TASKS = 21
MAX_PER_REPOSITORY = 2
STEP_LIMIT = 32
DIFFICULTY_QUOTAS = {
    "15 min - 1 hour": 7,
    "1-4 hours": 7,
    "<15 min fix": 6,
}
PRIOR_TRAJECTORY_ROOTS = (
    ARTIFACTS / "impactkv_v44_dense_sensitive_v40_20260728/tasks",
    ARTIFACTS / "impactkv_bridge_agent_accuracy_speed_20260726/dense/full_18",
    ARTIFACTS / "impactkv_v43_new_verified_v40_20260728/tasks",
    ARTIFACTS / "impactkv_m55_v40_task_disjoint_20260805/tasks",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _population_rows() -> list[dict[str, Any]]:
    value = read_json(POPULATION)
    if not isinstance(value, list) or len(value) != 500:
        raise AssertionError("SWE-bench Verified population changed")
    return value


def _repo_key(row: dict[str, Any]) -> str:
    return str(row.get("repo") or "").strip()


def _rank(instance_id: str) -> tuple[str, str]:
    return (
        hashlib.sha256(f"{SELECTION_SALT}:{instance_id}".encode()).hexdigest(),
        instance_id,
    )


def _trajectory_task_ids() -> tuple[set[str], dict[str, str]]:
    task_ids: set[str] = set()
    hashes: dict[str, str] = {}
    for root in PRIOR_TRAJECTORY_ROOTS:
        paths = sorted(root.glob("**/*.traj.json")) if root.exists() else []
        digest = hashlib.sha256()
        for path in paths:
            value = read_json(path)
            instance_id = str(value.get("instance_id") or "")
            if instance_id:
                task_ids.add(instance_id)
            digest.update(str(path).encode())
            digest.update(bytes.fromhex(sha256(path)))
        hashes[str(root)] = digest.hexdigest()
    return task_ids, hashes


def select_tasks(
    rows: list[dict[str, Any]], excluded: set[str], limit: int
) -> list[dict[str, Any]]:
    """Outcome-blind, difficulty-stratified selection with a repository cap."""

    if limit <= 0:
        raise ValueError("limit must be positive")
    eligible = [
        row
        for row in rows
        if str(row["instance_id"]) not in excluded and _repo_key(row)
    ]
    buckets: dict[str, list[dict[str, Any]]] = {}
    for difficulty in DIFFICULTY_QUOTAS:
        buckets[difficulty] = sorted(
            [row for row in eligible if row.get("difficulty") == difficulty],
            key=lambda row: _rank(str(row["instance_id"])),
        )
    other = sorted(
        [row for row in eligible if row.get("difficulty") not in DIFFICULTY_QUOTAS],
        key=lambda row: _rank(str(row["instance_id"])),
    )
    selected: list[dict[str, Any]] = []
    repo_counts: Counter[str] = Counter()

    def consume(candidates: list[dict[str, Any]], wanted: int) -> None:
        for row in candidates:
            if wanted <= 0 or len(selected) >= limit:
                break
            repo = _repo_key(row)
            if repo_counts[repo] >= MAX_PER_REPOSITORY:
                continue
            selected.append(row)
            repo_counts[repo] += 1
            wanted -= 1

    scaled = {
        difficulty: round(limit * quota / INITIAL_TASKS)
        for difficulty, quota in DIFFICULTY_QUOTAS.items()
    }
    while sum(scaled.values()) > limit:
        key = max(scaled, key=lambda value: (scaled[value], value))
        scaled[key] -= 1
    while sum(scaled.values()) < limit:
        key = min(scaled, key=lambda value: (scaled[value], value))
        scaled[key] += 1
    for difficulty in DIFFICULTY_QUOTAS:
        consume(buckets[difficulty], scaled[difficulty])
    remainder = sorted(
        [row for row in eligible if row not in selected],
        key=lambda row: _rank(str(row["instance_id"])),
    )
    consume(other + remainder, limit - len(selected))
    if len(selected) != limit:
        raise ValueError(f"only {len(selected)} eligible tasks for requested {limit}")
    return selected


def _selection_hash(rows: list[dict[str, Any]]) -> str:
    value = {
        "salt": SELECTION_SALT,
        "tasks": [str(row["instance_id"]) for row in rows],
    }
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def register(output: Path, task_limit: int) -> dict[str, Any]:
    registration_path = output / "COHORT_REGISTRATION.json"
    if registration_path.exists():
        value = read_json(registration_path)
        if int(value["selection"]["task_limit"]) != task_limit:
            raise ValueError("registered cohort task limit differs from request")
        return value
    if output.exists():
        raise FileExistsError(output)
    if task_limit not in (INITIAL_TASKS, MAX_TASKS):
        raise ValueError("task_limit must be the frozen initial or expansion size")
    excluded, root_hashes = _trajectory_task_ids()
    population_rows = _population_rows()
    selected = select_tasks(population_rows, excluded, task_limit)
    output.mkdir(parents=True)
    snapshot = output / "FROZEN_SUBSET.json"
    dataset_root = output / "minisweagent_dataset"
    dataset = dataset_root / "test.jsonl"
    evaluation = output / "EVAL_REGISTRATION.json"
    write_json(snapshot, selected)
    dataset_root.mkdir(parents=True)
    dataset.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected),
        encoding="utf-8",
    )
    write_json(
        evaluation,
        {
            "schema_version": 1,
            "registration_id": output.name,
            "registered_at_utc": utc_now(),
            "dataset": {
                "name": "princeton-nlp/SWE-bench_Verified",
                "split": "test",
                "population_size": 500,
                "local_snapshot": str(snapshot),
                "local_snapshot_sha256": sha256(snapshot),
            },
            "instances": [
                {"instance_id": str(row["instance_id"])} for row in selected
            ],
        },
    )
    value = {
        "status": "REGISTERED_BEFORE_NEW_DENSE_TRAJECTORIES",
        "registered_at_utc": utc_now(),
        "purpose": "task-disjoint module-conditioned Attention/KV mechanism cohort",
        "selection": {
            "salt": SELECTION_SALT,
            "task_limit": task_limit,
            "maximum_tasks_after_capacity_expansion": MAX_TASKS,
            "selection_sha256": _selection_hash(selected),
            "outcome_used_for_selection": False,
            "excluded_prior_trajectory_tasks": sorted(excluded),
            "excluded_task_count": len(excluded),
            "difficulty_quotas_initial20": DIFFICULTY_QUOTAS,
            "difficulty_note": (
                "After task-disjoint exclusions the local Verified-500 snapshot "
                "contains only one >4-hours task, so the frozen achievable strata "
                "are 7 medium, 7 long, and 6 short tasks."
            ),
            "maximum_tasks_per_repository": MAX_PER_REPOSITORY,
            "tasks": [
                {
                    "instance_id": str(row["instance_id"]),
                    "repo": _repo_key(row),
                    "difficulty": row.get("difficulty"),
                }
                for row in selected
            ],
        },
        "protocol": {
            "arm": "dense",
            "model": bridge.MODEL,
            "step_limit": STEP_LIMIT,
            "temperature": 0,
            "rolling_history_groups": 6,
            "official_evaluation": False,
            "prefetch": False,
        },
        "inputs": {
            "population": str(POPULATION),
            "population_sha256": sha256(POPULATION),
            "prior_trajectory_root_digests": root_hashes,
            "snapshot": str(snapshot),
            "snapshot_sha256": sha256(snapshot),
            "dataset": str(dataset),
            "dataset_sha256": sha256(dataset),
            "evaluation_registration": str(evaluation),
            "evaluation_registration_sha256": sha256(evaluation),
        },
        "protected": {
            "old_dirty_checkout_modified": False,
            "paper_modified": False,
            "old_preregistration_thresholds_modified": False,
            "prefetch": False,
        },
    }
    write_json(registration_path, value)
    return value


def run_dense(output: Path, port: int) -> dict[str, Any]:
    registration = register(output, INITIAL_TASKS)
    bridge.DATASET = Path(registration["inputs"]["dataset"]).parent
    bridge.SNAPSHOT = Path(registration["inputs"]["snapshot"])
    bridge.REGISTRATION = Path(registration["inputs"]["evaluation_registration"])
    bridge.AGENT_STEP_LIMIT = STEP_LIMIT
    return bridge.run_arm(
        output=output,
        arm="dense",
        scope="full",
        port=port,
        instance_filter=None,
        official=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("register", "run-dense"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--task-limit", type=int, default=INITIAL_TASKS)
    parser.add_argument("--port", type=int, default=30170)
    args = parser.parse_args()
    value = (
        register(args.output, args.task_limit)
        if args.command == "register"
        else run_dense(args.output, args.port)
    )
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
