#!/usr/bin/env python3
"""Freeze and run a task-disjoint 24-case accuracy expansion.

The campaign preserves the fresh9 rolling MAS/prompt protocol and the already
implemented natural repository-code cost policy.  Selection is based only on
instance identity, repository, difficulty, deterministic salted rank, and a
predeclared local-infrastructure exclusion.  Every instance found in prior
artifact datasets, trajectories, frozen snapshots, or registrations is
excluded before any model outcome from this campaign exists.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import re
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from benchmark.multi_workflow.runtime_paths import RuntimePaths


PROJECT = Path(__file__).resolve().parents[2]
PATHS = RuntimePaths.from_project(PROJECT)
ARTIFACTS = PATHS.artifacts
POPULATION = PATHS.population
DEFAULT_OUTPUT = (
    ARTIFACTS / "impactkv_natural_code_cost_agent_expanded24_20260808"
)
FRESH9 = ARTIFACTS / "impactkv_natural_code_cost_agent_20260808"
BRIDGE_RUNNER = (
    PROJECT / "benchmark/multi_workflow/run_bridge_reuse_agent_experiment.py"
)
MINI_PYTHON = PATHS.mini_python
SELECTION_SALT = "natural-code-cost-expanded-accuracy-20260808-v1"
TASKS = 24
REPO_CAP = 4
DIFFICULTY_QUOTAS = {
    "<15 min fix": 9,
    "15 min - 1 hour": 9,
    "1-4 hours": 6,
}
INFRA_EXCLUDED_REPOS = {
    "matplotlib/matplotlib": (
        "known rootless-Docker subuid failure before agent inference"
    )
}
ARMS = ("dense", "coding_natural_code_cost")
INSTANCE_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+__[A-Za-z0-9_.-]+-\d+$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rank(instance_id: str) -> tuple[str, str]:
    digest = hashlib.sha256(
        f"{SELECTION_SALT}:{instance_id}".encode()
    ).hexdigest()
    return digest, instance_id


def _walk_instance_ids(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "instance_id" and isinstance(item, str):
                if INSTANCE_PATTERN.fullmatch(item):
                    yield item
            else:
                yield from _walk_instance_ids(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_instance_ids(item)


def historical_exclusions() -> tuple[set[str], dict[str, Any]]:
    """Collect all task IDs exposed by earlier local experiment inputs."""

    paths: set[Path] = set()
    for pattern in (
        "**/test.jsonl",
        "**/*.traj.json",
        "**/FROZEN*.json",
        "**/*REGISTRATION*.json",
    ):
        paths.update(path for path in ARTIFACTS.glob(pattern) if path.is_file())

    identifiers: set[str] = set()
    digest = hashlib.sha256()
    parsed = 0
    failures: list[str] = []
    for path in sorted(paths):
        try:
            if path.suffix == ".jsonl":
                values = [
                    json.loads(line)
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            else:
                values = [read_json(path)]
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            failures.append(str(path))
            continue
        for value in values:
            identifiers.update(_walk_instance_ids(value))
        relative = str(path.relative_to(ARTIFACTS))
        digest.update(relative.encode())
        digest.update(bytes.fromhex(sha256(path)))
        parsed += 1

    # Some early campaigns exposed tasks only as directory names.
    for path in ARTIFACTS.glob("**/*__*-*"):
        if path.is_dir() and INSTANCE_PATTERN.fullmatch(path.name):
            identifiers.add(path.name)
    return identifiers, {
        "artifact_files_parsed": parsed,
        "artifact_input_digest": digest.hexdigest(),
        "parse_failures": failures,
    }


def select_cohort(
    population: list[dict[str, Any]], excluded: set[str]
) -> list[dict[str, Any]]:
    """Find the minimum salted-rank allocation under frozen quotas/cap."""

    difficulties = tuple(DIFFICULTY_QUOTAS)
    quotas = tuple(DIFFICULTY_QUOTAS[value] for value in difficulties)
    by_repo: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in population:
        instance_id = str(row["instance_id"])
        repo = str(row.get("repo") or "")
        difficulty = str(row.get("difficulty") or "")
        if (
            instance_id in excluded
            or repo in INFRA_EXCLUDED_REPOS
            or difficulty not in DIFFICULTY_QUOTAS
        ):
            continue
        by_repo.setdefault(repo, {}).setdefault(difficulty, []).append(row)

    # Only the best REPO_CAP candidates per difficulty/repository can be part
    # of a solution with that same repository cap.
    states: dict[
        tuple[int, ...], tuple[int, tuple[str, ...], list[dict[str, Any]]]
    ] = {(0,) * len(difficulties): (0, (), [])}
    for repo in sorted(by_repo):
        candidates: list[dict[str, Any]] = []
        for difficulty in difficulties:
            candidates.extend(
                sorted(
                    by_repo[repo].get(difficulty, []),
                    key=lambda row: _rank(str(row["instance_id"])),
                )[:REPO_CAP]
            )
        options = [()]
        for size in range(1, min(REPO_CAP, len(candidates)) + 1):
            options.extend(itertools.combinations(candidates, size))

        updated = dict(states)
        for counts, (cost, identifiers, selected) in states.items():
            for option in options[1:]:
                increments = tuple(
                    sum(row.get("difficulty") == difficulty for row in option)
                    for difficulty in difficulties
                )
                next_counts = tuple(
                    left + right for left, right in zip(counts, increments)
                )
                if any(
                    value > quota
                    for value, quota in zip(next_counts, quotas)
                ):
                    continue
                option_ids = tuple(
                    sorted(str(row["instance_id"]) for row in option)
                )
                option_cost = sum(
                    int(_rank(instance_id)[0], 16)
                    for instance_id in option_ids
                )
                proposal = (
                    cost + option_cost,
                    tuple(sorted(identifiers + option_ids)),
                    selected + list(option),
                )
                if (
                    next_counts not in updated
                    or proposal[:2] < updated[next_counts][:2]
                ):
                    updated[next_counts] = proposal
        states = updated

    if quotas not in states:
        raise ValueError("no fresh repository-capped quota allocation exists")
    selected = sorted(
        states[quotas][2], key=lambda row: _rank(str(row["instance_id"]))
    )
    if len(selected) != TASKS:
        raise AssertionError("expanded cohort size changed")
    if max(Counter(str(row["repo"]) for row in selected).values()) > REPO_CAP:
        raise AssertionError("repository cap was violated")
    return selected


def prepare(output: Path) -> dict[str, Any]:
    registration_path = output / "CAMPAIGN_REGISTRATION.json"
    if registration_path.exists():
        return read_json(registration_path)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)

    population = read_json(POPULATION)
    if not isinstance(population, list) or len(population) != 500:
        raise AssertionError("SWE-bench Verified population changed")
    excluded, exclusion_audit = historical_exclusions()
    selected = select_cohort(population, excluded)
    identifiers = [str(row["instance_id"]) for row in selected]
    if excluded.intersection(identifiers):
        raise AssertionError("historical task leaked into the fresh cohort")

    output.mkdir(parents=True)
    snapshot = output / "FROZEN_EXPANDED24.json"
    dataset = output / "dataset/test.jsonl"
    bridge_registration = output / "BRIDGE_EXPANDED24_REGISTRATION.json"
    write_json(snapshot, selected)
    write_jsonl(dataset, selected)
    write_json(
        bridge_registration,
        {
            "schema_version": 1,
            "registration_id": "impactkv-natural-code-cost-expanded24",
            "registered_at_utc": utc_now(),
            "dataset": {
                "name": "princeton-nlp/SWE-bench_Verified",
                "split": "test",
            },
            "instances": [{"instance_id": value} for value in identifiers],
        },
    )

    sources = (
        PROJECT / "benchmark/multi_workflow/coding_reuse_policy.py",
        PROJECT / "benchmark/multi_workflow/bridge_reuse_litellm_model.py",
        PROJECT / "benchmark/multi_workflow/run_bridge_reuse_agent_experiment.py",
        Path(__file__).resolve(),
    )
    value = {
        "status": "REGISTERED_BEFORE_EXPANDED24_MODEL_OUTCOMES",
        "registered_at_utc": utc_now(),
        "purpose": (
            "independent task-disjoint expansion of official accuracy for "
            "Dense versus natural repository-code cost-gated lossy KV reuse"
        ),
        "selection": {
            "salt": SELECTION_SALT,
            "outcome_used_for_selection": False,
            "tasks": TASKS,
            "repository_cap": REPO_CAP,
            "difficulty_quotas": DIFFICULTY_QUOTAS,
            "capacity_amendment_before_registration": (
                "The proposed cap=3 and 8/8/8 quotas were infeasible before "
                "registration or model execution: remaining 1-4 hour tasks "
                "occur in only Django and SymPy, allowing at most six under "
                "cap=3, while Pylint has only one unused task. The frozen "
                "feasible design uses cap=4 and quotas 9/9/6."
            ),
            "historical_excluded_tasks": len(excluded),
            "infrastructure_excluded_repositories": INFRA_EXCLUDED_REPOS,
            "instances": [
                {
                    "instance_id": str(row["instance_id"]),
                    "repo": str(row["repo"]),
                    "difficulty": row.get("difficulty"),
                }
                for row in selected
            ],
        },
        "protocol": {
            "arms": list(ARMS),
            "arm_execution_order": list(ARMS),
            "same_as_fresh9": True,
            "backend": "mini-SWE-agent rolling6 + SGLang",
            "model": "Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit",
            "temperature": 0,
            "step_limit": 32,
            "same_system_and_agent_templates": True,
            "official_metric": "SWE-bench resolved",
            "run_both_arms_regardless_of_first_arm_outcome": True,
            "prefetch": False,
        },
        "policy": {
            "eligible": (
                "successful version-valid single-file direct repository-code "
                "tool results with positive frozen predicted saving"
            ),
            "cost_formula": (
                "0.13169242 * (island_tokens * target_prompt_tokens / 10000) "
                "- 14.66811245 ms"
            ),
            "always_dense": (
                "search, assistant reasoning, tool calls, tests, mutations, "
                "ambiguous multi-file output, stale versions, negative cost"
            ),
            "prefetch": False,
        },
        "analysis": {
            "primary": "paired expanded24 official resolved difference",
            "paired_counts": "rescue, damage, both-resolved, both-unresolved",
            "uncertainty": "exact two-sided McNemar test on discordant pairs",
            "secondary": (
                "transparent fresh33 aggregate with the prior disjoint fresh9"
            ),
            "speed": (
                "free-running telemetry is descriptive; retain prior exact-"
                "prompt replay as the causal latency result"
            ),
        },
        "external_baseline_boundary": {
            "CacheBlend_and_KVCOMM": (
                "not ranked until their rolling-history token-preserving "
                "adapters exist"
            )
        },
        "inputs": {
            "population": str(POPULATION),
            "population_sha256": sha256(POPULATION),
            "exclusion_audit": exclusion_audit,
            "snapshot": str(snapshot),
            "snapshot_sha256": sha256(snapshot),
            "dataset": str(dataset),
            "dataset_sha256": sha256(dataset),
            "bridge_registration": str(bridge_registration),
            "bridge_registration_sha256": sha256(bridge_registration),
            "fresh9_result": str(FRESH9 / "RESULT.json"),
            "fresh9_result_sha256": sha256(FRESH9 / "RESULT.json"),
            "source_sha256": {
                str(path.relative_to(PROJECT)): sha256(path) for path in sources
            },
        },
        "protected": {
            "old_dirty_checkout_modified": False,
            "old_preregistration_thresholds_modified": False,
            "paper_modified": False,
            "prefetch": False,
        },
    }
    write_json(registration_path, value)
    return value


def run_arm(output: Path, arm: str, port: int) -> None:
    if arm not in ARMS:
        raise ValueError(arm)
    prepare(output)
    env = os.environ.copy()
    env.update(
        {
            "IMPACTKV_DATASET_ROOT": str(output / "dataset"),
            "IMPACTKV_EVAL_SNAPSHOT": str(output / "FROZEN_EXPANDED24.json"),
            "IMPACTKV_EVAL_REGISTRATION": str(
                output / "BRIDGE_EXPANDED24_REGISTRATION.json"
            ),
            "IMPACTKV_AGENT_STEP_LIMIT": "32",
            "PYTHONPATH": (
                str(PROJECT)
                + (
                    os.pathsep + env["PYTHONPATH"]
                    if env.get("PYTHONPATH")
                    else ""
                )
            ),
        }
    )
    command = [
        str(MINI_PYTHON),
        str(BRIDGE_RUNNER),
        "--output",
        str(output / "online"),
        "run-arm",
        "--arm",
        arm,
        "--scope",
        "full",
        "--port",
        str(port),
        "--official",
    ]
    subprocess.run(command, cwd=PROJECT, env=env, check=True)


def _official(output: Path, arm: str) -> dict[str, Any]:
    path = output / f"online/{arm}/full_{TASKS}/OFFICIAL_RESULT.json"
    value = read_json(path)
    if value.get("report") is None:
        raise ValueError(f"official evaluator report absent: {path}")
    return dict(value["report"])


def _runtime(output: Path, arm: str) -> dict[str, Any]:
    return read_json(output / f"online/{arm}/full_{TASKS}/RUNTIME_SUMMARY.json")


def mcnemar_exact_two_sided(rescues: int, damages: int) -> float:
    discordant = rescues + damages
    if not discordant:
        return 1.0
    tail = sum(
        math.comb(discordant, index) for index in range(min(rescues, damages) + 1)
    ) / (2**discordant)
    return min(1.0, 2 * tail)


def _paired_summary(
    instance_ids: list[str], dense_ids: set[str], policy_ids: set[str]
) -> dict[str, Any]:
    rescues = policy_ids - dense_ids
    damages = dense_ids - policy_ids
    both_resolved = dense_ids & policy_ids
    neither = set(instance_ids) - (dense_ids | policy_ids)
    return {
        "denominator": len(instance_ids),
        "dense_resolved": len(dense_ids),
        "policy_resolved": len(policy_ids),
        "absolute_accuracy_point_change": (
            (len(policy_ids) - len(dense_ids)) / len(instance_ids)
        ),
        "rescues": sorted(rescues),
        "damages": sorted(damages),
        "both_resolved": sorted(both_resolved),
        "both_unresolved": sorted(neither),
        "discordant_pairs": len(rescues) + len(damages),
        "mcnemar_exact_two_sided_p": mcnemar_exact_two_sided(
            len(rescues), len(damages)
        ),
    }


def summarize(output: Path) -> dict[str, Any]:
    registration = prepare(output)
    instance_ids = [
        row["instance_id"] for row in registration["selection"]["instances"]
    ]
    dense = _official(output, "dense")
    policy = _official(output, "coding_natural_code_cost")
    dense_ids = set(dense["resolved_ids"])
    policy_ids = set(policy["resolved_ids"])
    expanded = _paired_summary(instance_ids, dense_ids, policy_ids)

    prior = read_json(FRESH9 / "RESULT.json")["accuracy"]
    prior_ids = [row["instance_id"] for row in prior["per_task"]]
    if set(prior_ids) & set(instance_ids):
        raise AssertionError("fresh9 and expanded24 are not task-disjoint")
    prior_dense = {
        row["instance_id"] for row in prior["per_task"] if row["dense_resolved"]
    }
    prior_policy = {
        row["instance_id"] for row in prior["per_task"] if row["policy_resolved"]
    }
    aggregate = _paired_summary(
        prior_ids + instance_ids,
        prior_dense | dense_ids,
        prior_policy | policy_ids,
    )
    dense_runtime = _runtime(output, "dense")
    policy_runtime = _runtime(output, "coding_natural_code_cost")
    value = {
        "status": "COMPLETE",
        "classification": "independent expanded24 official accuracy validation",
        "expanded24": expanded,
        "fresh33_transparent_aggregate": aggregate,
        "official_evaluator": {
            "dense": dense,
            "coding_natural_code_cost": policy,
        },
        "physical_reuse": {
            "source_materialized_events": policy_runtime[
                "source_materialized_events"
            ],
            "target_copy_events": policy_runtime["target_copy_events"],
            "copied_tokens": policy_runtime["copied_tokens"],
            "target_fallback_events": policy_runtime[
                "target_fallback_events"
            ],
            "prefetch": False,
        },
        "free_running_latency_descriptive_only": {
            "dense": {
                "requests": dense_runtime["requests"],
                "median_ttft_ms": dense_runtime["median_ttft_ms"],
                "p95_ttft_ms": dense_runtime["p95_ttft_ms"],
            },
            "policy": {
                "requests": policy_runtime["requests"],
                "median_ttft_ms": policy_runtime["median_ttft_ms"],
                "p95_ttft_ms": policy_runtime["p95_ttft_ms"],
            },
            "causal_speed_claim_allowed": False,
            "reason": "independently running agent trajectories differ",
        },
        "prior_exact_prompt_speed_reference": read_json(
            FRESH9 / "exact_prompt_speed/RESULT.json"
        ),
        "external_baseline_ranking_allowed": False,
    }
    write_json(output / "RESULT.json", value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    run = sub.add_parser("run-arm")
    run.add_argument("--arm", choices=ARMS, required=True)
    run.add_argument("--port", type=int, default=30000)
    sub.add_parser("summarize")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.command == "prepare":
        value = prepare(output)
    elif args.command == "run-arm":
        run_arm(output, args.arm, args.port)
        value = {"arm": args.arm, "status": "COMPLETE"}
    else:
        value = summarize(output)
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
