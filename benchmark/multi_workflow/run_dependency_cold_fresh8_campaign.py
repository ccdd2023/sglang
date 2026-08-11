#!/usr/bin/env python3
"""Run a task-disjoint fresh8 accuracy campaign for dependency-cold reuse.

The cohort is selected before model execution from SWE-bench Verified after
excluding every task identifier exposed by earlier local artifacts.  Selection
uses only a fixed salted rank, difficulty quotas, and a repository cap.  Dense
and dependency-cold reuse then run under the same rolling mini-SWE-agent
protocol and are judged by the official SWE-bench evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark.multi_workflow import (
    run_natural_code_cost_expanded_accuracy_campaign as prior,
)


PROJECT = Path(__file__).resolve().parents[2]
ROOT = Path("/home/gfy/CodeMAS_Project")
ARTIFACTS = ROOT / "kvflow-artifacts"
POPULATION = (
    ROOT
    / "sglang-kvflow/results/repo_level_datasets/"
    "swe_verified_500_instances.json"
)
DEFAULT_OUTPUT = (
    ARTIFACTS / "impactkv_dependency_cold_fresh8_20260810"
)
BRIDGE_RUNNER = (
    PROJECT / "benchmark/multi_workflow/run_bridge_reuse_agent_experiment.py"
)
MINI_PYTHON = Path("/home/gfy/.venvs/mini-swe-agent-v2.3.0/bin/python")
SELECTION_SALT = "dependency-hot-recompute-cold-lossy-fresh8-20260810-v1"
TASKS = 8
REPO_CAP = 2
DIFFICULTY_QUOTAS = {
    "<15 min fix": 3,
    "15 min - 1 hour": 3,
    "1-4 hours": 2,
}
INFRA_EXCLUDED_REPOS = dict(prior.INFRA_EXCLUDED_REPOS)
POLICY_ARM = "coding_dependency_cold_cost"
ARMS = ("dense", POLICY_ARM)


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
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


def select_cohort(
    population: list[dict[str, Any]], excluded: set[str]
) -> list[dict[str, Any]]:
    """Use the prior exact optimizer with this campaign's frozen constants."""

    saved = (
        prior.SELECTION_SALT,
        prior.TASKS,
        prior.REPO_CAP,
        prior.DIFFICULTY_QUOTAS,
        prior.INFRA_EXCLUDED_REPOS,
    )
    try:
        prior.SELECTION_SALT = SELECTION_SALT
        prior.TASKS = TASKS
        prior.REPO_CAP = REPO_CAP
        prior.DIFFICULTY_QUOTAS = DIFFICULTY_QUOTAS
        prior.INFRA_EXCLUDED_REPOS = INFRA_EXCLUDED_REPOS
        return prior.select_cohort(population, excluded)
    finally:
        (
            prior.SELECTION_SALT,
            prior.TASKS,
            prior.REPO_CAP,
            prior.DIFFICULTY_QUOTAS,
            prior.INFRA_EXCLUDED_REPOS,
        ) = saved


def prepare(output: Path) -> dict[str, Any]:
    registration_path = output / "CAMPAIGN_REGISTRATION.json"
    if registration_path.exists():
        return read_json(registration_path)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)

    population = read_json(POPULATION)
    if not isinstance(population, list) or len(population) != 500:
        raise AssertionError("SWE-bench Verified population changed")
    excluded, exclusion_audit = prior.historical_exclusions()
    selected = select_cohort(population, excluded)
    identifiers = [str(row["instance_id"]) for row in selected]
    if excluded.intersection(identifiers):
        raise AssertionError("historical task leaked into fresh8")
    if max(Counter(str(row["repo"]) for row in selected).values()) > REPO_CAP:
        raise AssertionError("repository cap changed")
    observed_quotas = Counter(str(row.get("difficulty")) for row in selected)
    if dict(observed_quotas) != DIFFICULTY_QUOTAS:
        raise AssertionError(f"difficulty quotas changed: {observed_quotas}")

    output.mkdir(parents=True)
    snapshot = output / "FROZEN_FRESH8.json"
    dataset = output / "dataset/test.jsonl"
    bridge_registration = output / "BRIDGE_FRESH8_REGISTRATION.json"
    write_json(snapshot, selected)
    write_jsonl(dataset, selected)
    write_json(
        bridge_registration,
        {
            "schema_version": 1,
            "registration_id": "impactkv-dependency-cold-fresh8",
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
    registration = {
        "status": "REGISTERED_BEFORE_FRESH8_MODEL_OUTCOMES",
        "registered_at_utc": utc_now(),
        "purpose": (
            "test whether online dependency-hot recomputation plus "
            "dependency-cold lossy KV reuse has a favorable official "
            "accuracy/TTFT direction on task-disjoint coding tasks"
        ),
        "selection": {
            "salt": SELECTION_SALT,
            "outcome_used_for_selection": False,
            "tasks": TASKS,
            "repository_cap": REPO_CAP,
            "difficulty_quotas": DIFFICULTY_QUOTAS,
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
            "run_both_arms_regardless_of_first_arm_outcome": True,
            "backend": "mini-SWE-agent rolling6 + SGLang",
            "model": "Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit",
            "temperature": 0,
            "step_limit": 32,
            "same_system_agent_and_prompt_templates": True,
            "official_metric": "SWE-bench resolved",
            "prefetch": False,
        },
        "policy": {
            "direction": "dependency-hot recompute; dependency-cold lossy copy",
            "online_evidence_only": True,
            "hot_relation": (
                "a later visible group names the same source path or an "
                "explicit symbol from that source"
            ),
            "eligible": (
                "successful, version-valid, single-file direct repository "
                "code observation with positive frozen cost estimate"
            ),
            "copy_semantics": "physical K/V copy with K RoPE rotation",
            "exact_only": False,
            "prefetch": False,
        },
        "gates": {
            "accuracy_direction": "policy_resolved >= dense_resolved",
            "physical_copy_required": True,
            "target_fallback_events": 0,
            "latency_scope": (
                "free-running TTFT is descriptive; causal speed remains "
                "same-target cache-ready fork TTFT"
            ),
            "claim_limit": (
                "fresh8 establishes direction only, not statistical "
                "significance or superiority to external baselines"
            ),
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
    write_json(registration_path, registration)
    return registration


def run_arm(output: Path, arm: str, port: int) -> None:
    if arm not in ARMS:
        raise ValueError(arm)
    prepare(output)
    env = os.environ.copy()
    for key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        env.pop(key, None)
    env.update(
        {
            "IMPACTKV_DATASET_ROOT": str(output / "dataset"),
            "IMPACTKV_EVAL_SNAPSHOT": str(output / "FROZEN_FRESH8.json"),
            "IMPACTKV_EVAL_REGISTRATION": str(
                output / "BRIDGE_FRESH8_REGISTRATION.json"
            ),
            "IMPACTKV_AGENT_STEP_LIMIT": "32",
            "IMPACTKV_CAPTURE_LIMIT_PATCH": "1",
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
            "PYTHONPATH": str(PROJECT),
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
    value = read_json(
        output / f"online/{arm}/full_{TASKS}/OFFICIAL_RESULT.json"
    )
    report = value.get("report")
    if report is None:
        raise ValueError(f"official report absent for {arm}")
    return dict(report)


def _runtime(output: Path, arm: str) -> dict[str, Any]:
    return read_json(
        output / f"online/{arm}/full_{TASKS}/RUNTIME_SUMMARY.json"
    )


def summarize(output: Path) -> dict[str, Any]:
    registration = prepare(output)
    instance_ids = [
        row["instance_id"] for row in registration["selection"]["instances"]
    ]
    dense = _official(output, "dense")
    policy = _official(output, POLICY_ARM)
    paired = prior._paired_summary(
        instance_ids,
        set(dense["resolved_ids"]),
        set(policy["resolved_ids"]),
    )
    dense_runtime = _runtime(output, "dense")
    policy_runtime = _runtime(output, POLICY_ARM)
    gates = {
        "accuracy_noninferior_point_estimate": (
            paired["policy_resolved"] >= paired["dense_resolved"]
        ),
        "physical_lossy_copy": policy_runtime["copied_tokens"] > 0,
        "no_target_fallback": policy_runtime["target_fallback_events"] == 0,
    }
    result = {
        "status": "COMPLETE",
        "classification": "task-disjoint fresh8 directional validation",
        "paired_official_accuracy": paired,
        "official_evaluator": {"dense": dense, POLICY_ARM: policy},
        "physical_reuse": {
            "source_materialized_events": policy_runtime[
                "source_materialized_events"
            ],
            "target_copy_events": policy_runtime["target_copy_events"],
            "copied_tokens": policy_runtime["copied_tokens"],
            "rotated_k_tokens": policy_runtime["rotated_k_tokens"],
            "target_fallback_events": policy_runtime[
                "target_fallback_events"
            ],
            "exact_only": False,
            "prefetch": False,
        },
        "free_running_latency_descriptive_only": {
            arm: {
                "requests": runtime["requests"],
                "median_ttft_ms": runtime["median_ttft_ms"],
                "p95_ttft_ms": runtime["p95_ttft_ms"],
            }
            for arm, runtime in (
                ("dense", dense_runtime),
                (POLICY_ARM, policy_runtime),
            )
        },
        "gates": gates,
        "decision": (
            "DIRECTION_PASS_EXPAND"
            if all(gates.values())
            else "DIRECTION_FAIL_DIAGNOSE"
        ),
        "claim_limit": (
            "N=8 is a directional check; it does not establish statistical "
            "significance or external-baseline superiority"
        ),
    }
    write_json(output / "RESULT.json", result)
    return result


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
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
