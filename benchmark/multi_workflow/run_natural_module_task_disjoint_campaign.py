#!/usr/bin/env python3
"""Freeze and collect the fresh Dense cohort for natural-module studies.

Both the 20-task primary cohort and the capacity-only 29-task ceiling are
selected before any new trajectory is generated.  The expansion list cannot
be changed after seeing Attention or splice outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from benchmark.multi_workflow import run_bridge_reuse_agent_experiment as bridge


ROOT = Path("/home/gfy/CodeMAS_Project")
ARTIFACTS = ROOT / "kvflow-artifacts"
POPULATION = (
    ROOT
    / "sglang-kvflow/results/repo_level_datasets/"
    "swe_verified_500_instances.json"
)
DEFAULT_OUTPUT = ARTIFACTS / "impactkv_natural_module_attention_20260808"
SELECTION_SALT = "natural-module-attention-confirm-20260808-v1"
INITIAL_TASKS = 20
MAX_TASKS = 29
INITIAL_REPO_CAP = 2
MAX_REPO_CAP = 3
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
PRIOR_COHORT_SNAPSHOTS = (
    ARTIFACTS
    / "impactkv_attention_kv_task_disjoint_20260807_r1/FROZEN_SUBSET.json",
)
CAPACITY_AUDIT = (
    ARTIFACTS
    / "impactkv_natural_prompt_modules_20260808/development64/CAPACITY.json"
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


def _rank(instance_id: str) -> tuple[str, str]:
    return (
        hashlib.sha256(f"{SELECTION_SALT}:{instance_id}".encode()).hexdigest(),
        instance_id,
    )


def _repo(row: dict[str, Any]) -> str:
    return str(row.get("repo") or "").strip()


def _prior_task_ids() -> tuple[set[str], dict[str, str]]:
    task_ids: set[str] = set()
    digests: dict[str, str] = {}
    for root in PRIOR_TRAJECTORY_ROOTS:
        digest = hashlib.sha256()
        for path in sorted(root.glob("**/*.traj.json")) if root.exists() else ():
            value = read_json(path)
            instance_id = str(value.get("instance_id") or "")
            if instance_id:
                task_ids.add(instance_id)
            digest.update(str(path).encode())
            digest.update(bytes.fromhex(sha256(path)))
        digests[str(root)] = digest.hexdigest()
    for snapshot in PRIOR_COHORT_SNAPSHOTS:
        rows = read_json(snapshot)
        task_ids.update(str(row["instance_id"]) for row in rows)
        digests[str(snapshot)] = sha256(snapshot)
    return task_ids, digests


def select_initial(
    rows: Sequence[dict[str, Any]], excluded: set[str]
) -> list[dict[str, Any]]:
    eligible = [
        row for row in rows if str(row["instance_id"]) not in excluded and _repo(row)
    ]
    difficulties = tuple(DIFFICULTY_QUOTAS)
    quotas = tuple(DIFFICULTY_QUOTAS[value] for value in difficulties)
    by_repo: dict[str, list[dict[str, Any]]] = {}
    for row in eligible:
        if row.get("difficulty") in DIFFICULTY_QUOTAS:
            by_repo.setdefault(_repo(row), []).append(row)

    # A greedy pass can consume both slots of a repository for the first
    # difficulty and make a feasible 7/7/6 allocation appear impossible.
    # Dynamic programming freezes the minimum salted-rank feasible allocation.
    states: dict[tuple[int, ...], tuple[int, tuple[str, ...], list[dict[str, Any]]]] = {
        (0,) * len(difficulties): (0, (), [])
    }
    for repo in sorted(by_repo):
        ranked: dict[str, list[dict[str, Any]]] = {}
        for difficulty in difficulties:
            ranked[difficulty] = sorted(
                [row for row in by_repo[repo] if row.get("difficulty") == difficulty],
                key=lambda row: _rank(str(row["instance_id"])),
            )[:INITIAL_REPO_CAP]
        options: list[list[dict[str, Any]]] = [[]]
        for difficulty_index, difficulty in enumerate(difficulties):
            if ranked[difficulty]:
                options.append([ranked[difficulty][0]])
            if len(ranked[difficulty]) >= 2:
                options.append(ranked[difficulty][:2])
            for other in difficulties[difficulty_index + 1 :]:
                if ranked[difficulty] and ranked[other]:
                    options.append([ranked[difficulty][0], ranked[other][0]])
        updated = dict(states)
        for counts, (cost, identifiers, chosen) in states.items():
            for option in options[1:]:
                increments = tuple(
                    sum(row.get("difficulty") == difficulty for row in option)
                    for difficulty in difficulties
                )
                next_counts = tuple(left + right for left, right in zip(counts, increments))
                if any(value > quota for value, quota in zip(next_counts, quotas)):
                    continue
                option_ids = tuple(sorted(str(row["instance_id"]) for row in option))
                option_cost = sum(int(_rank(identifier)[0], 16) for identifier in option_ids)
                proposal = (cost + option_cost, tuple(sorted(identifiers + option_ids)), chosen + option)
                if next_counts not in updated or proposal[:2] < updated[next_counts][:2]:
                    updated[next_counts] = proposal
        states = updated
    if quotas not in states:
        raise ValueError("no repository-capped allocation satisfies the frozen difficulty quotas")
    selected = sorted(states[quotas][2], key=lambda row: _rank(str(row["instance_id"])))
    if len(selected) != INITIAL_TASKS:
        raise AssertionError("initial cohort does not contain twenty tasks")
    return selected


def expand_capacity_ceiling(
    rows: Sequence[dict[str, Any]], excluded: set[str], initial: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    selected = list(initial)
    selected_ids = {str(row["instance_id"]) for row in selected}
    counts = Counter(_repo(row) for row in selected)
    candidates = sorted(
        [
            row
            for row in rows
            if str(row["instance_id"]) not in excluded | selected_ids and _repo(row)
        ],
        key=lambda row: _rank(str(row["instance_id"])),
    )
    for row in candidates:
        if len(selected) >= MAX_TASKS:
            break
        if counts[_repo(row)] >= MAX_REPO_CAP:
            continue
        selected.append(row)
        counts[_repo(row)] += 1
    if len(selected) != MAX_TASKS:
        raise ValueError("could not freeze the 29-task capacity ceiling")
    return selected


def _write_dataset(root: Path, rows: Sequence[dict[str, Any]]) -> Path:
    path = root / "test.jsonl"
    root.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def register(output: Path) -> dict[str, Any]:
    registration_path = output / "COHORT_REGISTRATION.json"
    if registration_path.exists():
        return read_json(registration_path)
    if output.exists():
        raise FileExistsError(output)
    capacity = read_json(CAPACITY_AUDIT)
    if capacity.get("status") != "PASS" or not all(capacity["gates"].values()):
        raise RuntimeError("development capacity audit did not pass")
    population = read_json(POPULATION)
    if not isinstance(population, list) or len(population) != 500:
        raise AssertionError("SWE-bench Verified population changed")
    excluded, prior_digests = _prior_task_ids()
    initial = select_initial(population, excluded)
    maximum = expand_capacity_ceiling(population, excluded, initial)

    output.mkdir(parents=True)
    initial_snapshot = output / "FROZEN_INITIAL20.json"
    maximum_snapshot = output / "FROZEN_MAX29.json"
    write_json(initial_snapshot, initial)
    write_json(maximum_snapshot, maximum)
    initial_dataset = _write_dataset(output / "dataset_initial20", initial)
    maximum_dataset = _write_dataset(output / "dataset_max29", maximum)
    evaluation = output / "EVAL_REGISTRATION.json"
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
            },
            "instances_initial20": [str(row["instance_id"]) for row in initial],
            "instances_capacity_ceiling29": [str(row["instance_id"]) for row in maximum],
        },
    )
    value = {
        "status": "REGISTERED_BEFORE_FRESH_DENSE_TRAJECTORIES",
        "registered_at_utc": utc_now(),
        "purpose": "confirmatory natural-module Attention/KV mechanism cohort",
        "selection": {
            "salt": SELECTION_SALT,
            "outcome_used_for_selection": False,
            "initial_tasks": INITIAL_TASKS,
            "capacity_ceiling_tasks": MAX_TASKS,
            "initial_repo_cap": INITIAL_REPO_CAP,
            "capacity_ceiling_repo_cap": MAX_REPO_CAP,
            "difficulty_quotas_initial20": DIFFICULTY_QUOTAS,
            "excluded_task_count": len(excluded),
            "excluded_prior_tasks": sorted(excluded),
            "initial": [
                {
                    "instance_id": str(row["instance_id"]),
                    "repo": _repo(row),
                    "difficulty": row.get("difficulty"),
                }
                for row in initial
            ],
            "capacity_ceiling": [
                {
                    "instance_id": str(row["instance_id"]),
                    "repo": _repo(row),
                    "difficulty": row.get("difficulty"),
                }
                for row in maximum
            ],
            "expansion_rule": (
                "The additional nine tasks may be collected only for a "
                "capacity shortfall discovered before opening fresh Attention."
            ),
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
            "capacity_audit": str(CAPACITY_AUDIT),
            "capacity_audit_sha256": sha256(CAPACITY_AUDIT),
            "prior_input_digests": prior_digests,
            "initial_snapshot": str(initial_snapshot),
            "initial_snapshot_sha256": sha256(initial_snapshot),
            "maximum_snapshot": str(maximum_snapshot),
            "maximum_snapshot_sha256": sha256(maximum_snapshot),
            "initial_dataset": str(initial_dataset),
            "initial_dataset_sha256": sha256(initial_dataset),
            "maximum_dataset": str(maximum_dataset),
            "maximum_dataset_sha256": sha256(maximum_dataset),
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


def bridge_initial_registration(output: Path, registration: dict[str, Any]) -> Path:
    """Materialize the legacy runner schema without changing cohort membership."""

    path = output / "BRIDGE_INITIAL20_REGISTRATION.json"
    rows = read_json(Path(registration["inputs"]["initial_snapshot"]))
    value = {
        "schema_version": 1,
        "registration_id": f"{output.name}-initial20",
        "instances": [
            {"instance_id": str(row["instance_id"])} for row in rows
        ],
    }
    if path.exists() and read_json(path) != value:
        raise ValueError("legacy bridge registration changed")
    if not path.exists():
        write_json(path, value)
        write_json(
            output / "REGISTRATION_AMENDMENT_01.json",
            {
                "status": "SCHEMA_ONLY_BEFORE_MODEL_OUTCOMES",
                "reason": (
                    "The legacy bridge runner requires registration_id and "
                    "instances fields. This adapter preserves the exact frozen "
                    "initial20 membership and order."
                ),
                "source_snapshot": registration["inputs"]["initial_snapshot"],
                "source_snapshot_sha256": registration["inputs"][
                    "initial_snapshot_sha256"
                ],
                "adapter": str(path),
                "adapter_sha256": sha256(path),
                "selection_changed": False,
                "model_outcome_opened_before_amendment": False,
            },
        )
    return path


def record_parser_amendment(output: Path) -> dict[str, Any]:
    """Record a pre-Attention semantic fix without changing frozen tasks."""

    registration = register(output)
    capacity = read_json(CAPACITY_AUDIT)
    if capacity.get("status") != "PASS":
        raise RuntimeError("revised parser capacity audit did not pass")
    value = {
        "status": "SEMANTIC_CLASSIFICATION_FIX_BEFORE_ATTENTION",
        "reason": (
            "Natural module identity is based on successful read/search command "
            "semantics. The legacy V40 400-character eligibility threshold is "
            "applied only by reuse experiments, not by module classification."
        ),
        "original_capacity_sha256": registration["inputs"][
            "capacity_audit_sha256"
        ],
        "revised_capacity": str(CAPACITY_AUDIT),
        "revised_capacity_sha256": sha256(CAPACITY_AUDIT),
        "cohort_selection_changed": False,
        "confirmatory_thresholds_changed": False,
        "attention_outcome_opened_before_amendment": False,
    }
    path = output / "PARSER_AMENDMENT_BEFORE_ATTENTION.json"
    if path.exists() and read_json(path) != value:
        raise ValueError("parser amendment changed")
    if not path.exists():
        write_json(path, value)
    return value


def run_dense(output: Path, port: int) -> dict[str, Any]:
    registration = register(output)
    bridge.DATASET = Path(registration["inputs"]["initial_dataset"]).parent
    bridge.SNAPSHOT = Path(registration["inputs"]["initial_snapshot"])
    bridge.REGISTRATION = bridge_initial_registration(output, registration)
    bridge.AGENT_STEP_LIMIT = STEP_LIMIT
    return bridge.run_arm(
        output=output / "initial20",
        arm="dense",
        scope="full",
        port=port,
        instance_filter=None,
        official=False,
    )


def run_capacity_expansion(output: Path, port: int) -> dict[str, Any]:
    """Collect only the pre-frozen extra nine after an offline shortfall."""

    registration = register(output)
    initial_capacity = output / "attention_initial20/CAPACITY.json"
    if not initial_capacity.exists():
        raise FileNotFoundError(
            "initial natural-module capacity must be audited before expansion"
        )
    capacity = read_json(initial_capacity)
    if all(capacity.get("gates", {}).values()):
        raise RuntimeError("initial20 capacity passed; expansion is forbidden")
    initial = read_json(Path(registration["inputs"]["initial_snapshot"]))
    maximum = read_json(Path(registration["inputs"]["maximum_snapshot"]))
    initial_ids = [str(row["instance_id"]) for row in initial]
    maximum_ids = [str(row["instance_id"]) for row in maximum]
    if maximum_ids[: len(initial_ids)] != initial_ids:
        raise ValueError("capacity ceiling no longer extends initial20")
    extra = maximum[len(initial) :]
    dataset = _write_dataset(output / "dataset_expansion9", extra)
    bridge_registration = output / "BRIDGE_EXPANSION9_REGISTRATION.json"
    write_json(
        bridge_registration,
        {
            "schema_version": 1,
            "registration_id": f"{output.name}-expansion9",
            "instances": [
                {"instance_id": str(row["instance_id"])} for row in extra
            ],
        },
    )
    activation = {
        "status": "CAPACITY_ONLY_EXPANSION_BEFORE_ATTENTION",
        "initial_capacity": str(initial_capacity),
        "initial_capacity_sha256": sha256(initial_capacity),
        "extra_task_ids": [str(row["instance_id"]) for row in extra],
        "extra_tasks": len(extra),
        "attention_outcome_opened_before_expansion": False,
        "selection_changed": False,
    }
    write_json(output / "EXPANSION_ACTIVATION.json", activation)
    bridge.DATASET = dataset.parent
    bridge.SNAPSHOT = Path(registration["inputs"]["maximum_snapshot"])
    bridge.REGISTRATION = bridge_registration
    bridge.AGENT_STEP_LIMIT = STEP_LIMIT
    return bridge.run_arm(
        output=output / "expansion9",
        arm="dense",
        scope="full",
        port=port,
        instance_filter=None,
        official=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "register",
            "run-dense",
            "run-expansion",
            "record-parser-amendment",
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--port", type=int, default=30180)
    args = parser.parse_args()
    if args.command == "register":
        value = register(args.output)
    elif args.command == "record-parser-amendment":
        value = record_parser_amendment(args.output)
    elif args.command == "run-expansion":
        value = run_capacity_expansion(args.output, args.port)
    else:
        value = run_dense(args.output, args.port)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
