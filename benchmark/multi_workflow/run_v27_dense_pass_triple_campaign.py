#!/usr/bin/env python3
"""Pre-registered V23/General/Dense campaign on the frozen Dense-pass set."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

from benchmark.multi_workflow.run_bridge_reuse_agent_experiment import DATASET
from benchmark.multi_workflow.run_frozen_trajectory_replay_v18 import (
    read_json,
    sha256,
    utc_now,
    write_json,
)


ARTIFACTS = Path("/home/gfy/CodeMAS_Project/kvflow-artifacts")
DEFAULT_OUTPUT = ARTIFACTS / "impactkv_v27c_dense_pass_triple_20260727"
PRIOR_V27 = ARTIFACTS / "impactkv_v27_dense_pass_triple_20260727"
PRIOR_V27B = ARTIFACTS / "impactkv_v27b_dense_pass_triple_20260727"
PROJECT = Path(__file__).resolve().parents[2]
RUNNER = PROJECT / "benchmark/multi_workflow/run_v25_paired_agent_canary.py"
PYTHON = Path("/home/gfy/.venvs/mini-swe-agent-v2.3.0/bin/python")
DENSE_SCREEN = (
    ARTIFACTS
    / "impactkv_v18c_full18_accuracy_20260727"
    / "dense/full_18/OFFICIAL_RESULT.json"
)
CACHEBLEND_AUDIT = (
    ARTIFACTS
    / "impactkv_v15_cacheblend_flip_audit_20260727"
    / "V15_BASELINE_AUDIT.json"
)
V23 = "coding_post_mutation_target_prefix_v23"
GENERAL = "general"
DENSE = "dense"
ARMS = (V23, GENERAL, DENSE)
REUSE_ARMS = (V23, GENERAL)
EXPECTED_TASKS = 6
BOOTSTRAP_SEED = 20260727
BOOTSTRAPS = 100_000


def _dataset_rows() -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (DATASET / "test.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]


def selection() -> list[dict[str, str]]:
    """Return every frozen Dense pass without reading either reuse outcome."""

    dense = read_json(DENSE_SCREEN)
    resolved = sorted(dense["report"]["resolved_ids"])
    dataset = {str(row["instance_id"]): row for row in _dataset_rows()}
    if len(resolved) != EXPECTED_TASKS:
        raise ValueError(
            f"expected {EXPECTED_TASKS} frozen Dense passes, got {len(resolved)}"
        )
    missing = sorted(set(resolved) - set(dataset))
    if missing:
        raise ValueError(f"Dense-pass IDs absent from dataset: {missing}")
    return [
        {
            "instance_id": instance_id,
            "problem_statement_sha256": hashlib.sha256(
                dataset[instance_id]["problem_statement"].encode()
            ).hexdigest(),
        }
        for instance_id in resolved
    ]


def _cacheblend_reference() -> dict[str, Any]:
    audit = read_json(CACHEBLEND_AUDIT)
    preservation = audit["dense_preservation"]
    return {
        "damage_count": int(preservation["damage_count"]),
        "dense_passed": int(audit["task_correctness"]["dense_passed"]),
        "damage_rate_given_dense_pass": float(
            preservation["damage_rate_given_dense_pass"]
        ),
        "evidence_status": "retrospective_native_same_engine_full225",
        "artifact": str(CACHEBLEND_AUDIT),
        "artifact_sha256": sha256(CACHEBLEND_AUDIT),
    }


def task_dir(output: Path, instance_id: str) -> Path:
    return output / "tasks" / instance_id


def _command(output: Path, instance_id: str, stage: str) -> list[str]:
    return [
        str(PYTHON),
        str(RUNNER),
        stage,
        "--output",
        str(task_dir(output, instance_id)),
    ]


def _environment(instance_id: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "ALL_PROXY": "",
            "HTTPS_PROXY": "",
            "HTTP_PROXY": "",
            "all_proxy": "",
            "https_proxy": "",
            "http_proxy": "",
            "NO_PROXY": "localhost,127.0.0.1",
            "no_proxy": "localhost,127.0.0.1",
            "HF_HUB_OFFLINE": "1",
            "PYTHONPATH": f"{PROJECT / 'python'}:{PROJECT}",
            "IMPACTKV_PAIRED_INSTANCE_ID": instance_id,
            "IMPACTKV_REQUEST_TIMEOUT_SECONDS": "180",
            "IMPACTKV_PAIRED_DENSE_CONTROL": "1",
        }
    )
    return env


def register(output: Path) -> dict[str, Any]:
    path = output / "V27C_REGISTRATION.json"
    if path.exists():
        return read_json(path)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    selected = selection()
    cacheblend = _cacheblend_reference()
    value = {
        "registered_at_utc": utc_now(),
        "status": "REGISTERED_BEFORE_ANY_V27C_GPU_RUN",
        "experiment": (
            "V27C frozen-Dense-pass shared-history triple-control campaign"
        ),
        "motivation": (
            "V26 sampled five tasks without using outcomes, but all five "
            "failed under both reuse arms and therefore had no power to "
            "measure accuracy preservation. Evaluate every task passed by "
            "the frozen V18C Dense arm and add a same-run Dense branch at the "
            "identical repository snapshot. This directly measures whether "
            "reuse damages tasks that the concurrent Dense control solves."
        ),
        "prior_infrastructure_attempts": [
            {
                "artifact": str(PRIOR_V27),
                "registration_sha256": sha256(
                    PRIOR_V27 / "V27_REGISTRATION.json"
                ),
                "failed_task": "astropy__astropy-7336",
                "classification": (
                    "The third identical Dense request reused the final "
                    "General case cursor and emitted missing_source fallbacks. "
                    "No runtime or official result was accepted."
                ),
                "server_ledger_sha256": sha256(
                    PRIOR_V27
                    / "tasks/astropy__astropy-7336/run/SERVER_LEDGER.jsonl"
                ),
            },
            {
                "artifact": str(PRIOR_V27B),
                "registration_sha256": sha256(
                    PRIOR_V27B / "V27B_REGISTRATION.json"
                ),
                "failed_task": "astropy__astropy-7336",
                "classification": (
                    "An orphaned V27 HTTP process still owned port 32950. "
                    "The V27B server loaded its model but could not bind; the "
                    "readiness probe accepted the stale endpoint. No runtime "
                    "or official result was accepted. V27C refuses occupied "
                    "ports before launching and rechecks process liveness "
                    "after readiness."
                ),
                "server_log_sha256": sha256(
                    PRIOR_V27B
                    / "tasks/astropy__astropy-7336/run/sglang_server.log"
                ),
            },
        ],
        "selection": {
            "rule": (
                "Take all resolved_ids from the immutable V18C Dense official "
                "result; do not inspect General or V23 outcomes."
            ),
            "selected": selected,
            "sample_size": len(selected),
            "reuse_outcomes_used": False,
            "dense_screen": str(DENSE_SCREEN),
            "dense_screen_sha256": sha256(DENSE_SCREEN),
        },
        "protocol": {
            "runner": str(RUNNER),
            "arms": list(ARMS),
            "same_model_engine_tokenization": True,
            "shared_dense_history_until_first_unequal_online_span": True,
            "same_repository_snapshot_for_all_branched_arms": True,
            "first_branched_prompt_identical_for_all_arms": True,
            "target_execution_order": list(ARMS),
            "no_unequal_span_or_early_exit_is_shared_dense_itt_tie": True,
            "step_limit": 20,
            "request_timeout_seconds": 180,
            "official_swebench_container_each_arm": True,
            "prefetch": False,
            "reference_patch_or_future_output_used": False,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_iterations": BOOTSTRAPS,
        },
        "cacheblend_reference": cacheblend,
        "frozen_development_gates": {
            "official_tasks_completed": EXPECTED_TASKS,
            "runner_infrastructure_failures": 0,
            "target_fallbacks": 0,
            "explicit_dense_control_requests_if_treated_min": 1,
            "concurrent_dense_control_present": True,
            "treated_tasks_min": 3,
            "v23_resolved_not_below_general": True,
            "v23_resolved_not_below_concurrent_dense": True,
            "v23_damage_not_above_general": True,
            "v23_damage_rate_strictly_below_cacheblend": (
                cacheblend["damage_rate_given_dense_pass"]
            ),
            "do_not_promote_if_any_gate_fails": True,
        },
        "commands": {
            row["instance_id"]: {
                stage: _command(output, row["instance_id"], stage)
                for stage in ("register", "run", "evaluate")
            }
            for row in selected
        },
        "inputs": {
            "dataset_path": str(DATASET / "test.jsonl"),
            "dataset_sha256": sha256(DATASET / "test.jsonl"),
            "runner_sha256": sha256(RUNNER),
            "campaign_sha256": sha256(Path(__file__)),
        },
        "protected": {
            "prefetch": False,
            "paper_modified": False,
            "old_preregistration_thresholds_modified": False,
            "old_dirty_checkout_modified": False,
        },
    }
    write_json(path, value)
    return value


def _run_stage(
    output: Path, instance_id: str, stage: str
) -> dict[str, Any]:
    command = _command(output, instance_id, stage)
    log_path = (
        output / "orchestration_logs" / instance_id / f"V27C_{stage}.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("a", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=PROJECT,
            env=_environment(instance_id),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    value = {
        "instance_id": instance_id,
        "stage": stage,
        "returncode": result.returncode,
        "elapsed_seconds": time.perf_counter() - started,
        "log_path": str(log_path),
    }
    write_json(
        output
        / "orchestration_status"
        / instance_id
        / f"V27C_{stage}_STATUS.json",
        value,
    )
    return value


def preregister_tasks(output: Path) -> list[dict[str, Any]]:
    registration = register(output)
    rows = []
    for selected in registration["selection"]["selected"]:
        instance_id = selected["instance_id"]
        child = task_dir(output, instance_id) / "V25_REGISTRATION.json"
        rows.append(
            {
                "instance_id": instance_id,
                "stage": "register",
                "returncode": 0,
                "resumed": True,
            }
            if child.exists()
            else _run_stage(output, instance_id, "register")
        )
    write_json(output / "V27C_CHILD_REGISTRATIONS.json", rows)
    if any(row["returncode"] != 0 for row in rows):
        raise RuntimeError("one or more V27C child preregistrations failed")
    return rows


def run_campaign(output: Path) -> dict[str, Any]:
    registration = register(output)
    preregister_tasks(output)
    stages = []
    for selected in registration["selection"]["selected"]:
        instance_id = selected["instance_id"]
        child = task_dir(output, instance_id)
        if not (child / "V25_RESULT.json").exists():
            stages.append(_run_stage(output, instance_id, "run"))
        else:
            stages.append(
                {
                    "instance_id": instance_id,
                    "stage": "run",
                    "returncode": 0,
                    "resumed": True,
                }
            )
        if not (child / "V25_RESULT.json").exists():
            continue
        if not (child / "V25_OFFICIAL_RESULT.json").exists():
            stages.append(_run_stage(output, instance_id, "evaluate"))
        else:
            stages.append(
                {
                    "instance_id": instance_id,
                    "stage": "evaluate",
                    "returncode": 0,
                    "resumed": True,
                }
            )
    write_json(output / "V27C_STAGE_STATUS.json", stages)
    return summarize(output)


def _wilson(successes: int, total: int) -> list[float] | None:
    if total == 0:
        return None
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            p * (1 - p) / total + z * z / (4 * total * total)
        )
        / denominator
    )
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _bootstrap_difference(values: list[int]) -> list[float] | None:
    if not values:
        return None
    rng = random.Random(BOOTSTRAP_SEED)
    samples = sorted(
        statistics.fmean(rng.choice(values) for _ in values)
        for _ in range(BOOTSTRAPS)
    )
    return [
        samples[int(0.025 * BOOTSTRAPS)],
        samples[min(BOOTSTRAPS - 1, int(0.975 * BOOTSTRAPS))],
    ]


def summarize(output: Path) -> dict[str, Any]:
    registration = register(output)
    rows = []
    infrastructure_failures = 0
    for selected in registration["selection"]["selected"]:
        instance_id = selected["instance_id"]
        child = task_dir(output, instance_id)
        runtime_path = child / "V25_RESULT.json"
        official_path = child / "V25_OFFICIAL_RESULT.json"
        if not runtime_path.exists() or not official_path.exists():
            infrastructure_failures += 1
            rows.append({"instance_id": instance_id, "status": "INCOMPLETE"})
            continue
        runtime = read_json(runtime_path)
        official = read_json(official_path)
        if any(arm not in official["arms"] for arm in ARMS):
            infrastructure_failures += 1
            rows.append(
                {
                    "instance_id": instance_id,
                    "status": "INCOMPLETE_MISSING_DENSE_CONTROL",
                }
            )
            continue
        resolved = {
            arm: int(official["arms"][arm]["resolved"]) for arm in ARMS
        }
        rows.append(
            {
                "instance_id": instance_id,
                "status": "COMPLETE",
                "historical_dense_screen_pass": True,
                "treated": runtime["branch"] is not None,
                "resolved": resolved,
                "v23_minus_general": resolved[V23] - resolved[GENERAL],
                "v23_minus_dense": resolved[V23] - resolved[DENSE],
                "general_minus_dense": resolved[GENERAL] - resolved[DENSE],
                "v23_damage": resolved[DENSE] == 1
                and resolved[V23] == 0,
                "general_damage": resolved[DENSE] == 1
                and resolved[GENERAL] == 0,
                "v23_rescue": resolved[DENSE] == 0
                and resolved[V23] == 1,
                "general_rescue": resolved[DENSE] == 0
                and resolved[GENERAL] == 1,
                "branch": runtime["branch"],
                "copy_counts": runtime["server"]["copy_counts"],
                "dense_control_requests": runtime["server"].get(
                    "dense_control_requests", 0
                ),
                "target_fallbacks": runtime["server"]["target_fallbacks"],
                "branched_agent_elapsed_seconds": runtime.get(
                    "branched_agent_elapsed_seconds"
                ),
                "official_metrics": official["arms"],
            }
        )

    complete = [row for row in rows if row["status"] == "COMPLETE"]
    resolved_counts = {
        arm: sum(row["resolved"][arm] for row in complete) for arm in ARMS
    }
    dense_passes = resolved_counts[DENSE]
    damages = {
        V23: sum(row["v23_damage"] for row in complete),
        GENERAL: sum(row["general_damage"] for row in complete),
    }
    rescues = {
        V23: sum(row["v23_rescue"] for row in complete),
        GENERAL: sum(row["general_rescue"] for row in complete),
    }
    damage_rates = {
        arm: damages[arm] / dense_passes if dense_passes else None
        for arm in REUSE_ARMS
    }
    treated = sum(row["treated"] for row in complete)
    fallbacks = sum(row["target_fallbacks"] for row in complete)
    dense_control_requests = sum(
        row["dense_control_requests"] for row in complete
    )
    cacheblend_rate = registration["cacheblend_reference"][
        "damage_rate_given_dense_pass"
    ]
    gates = {
        "official_tasks_completed": len(complete) == EXPECTED_TASKS,
        "runner_infrastructure_failures": infrastructure_failures == 0,
        "target_fallbacks": fallbacks == 0,
        "explicit_dense_control_requests_if_treated_min": (
            dense_control_requests >= treated
        ),
        "concurrent_dense_control_present": len(complete) == EXPECTED_TASKS,
        "treated_tasks_min": treated >= 3,
        "v23_resolved_not_below_general": (
            resolved_counts[V23] >= resolved_counts[GENERAL]
        ),
        "v23_resolved_not_below_concurrent_dense": (
            resolved_counts[V23] >= resolved_counts[DENSE]
        ),
        "v23_damage_not_above_general": damages[V23] <= damages[GENERAL],
        "v23_damage_rate_strictly_below_cacheblend": (
            damage_rates[V23] is not None
            and damage_rates[V23] < cacheblend_rate
        ),
        "do_not_promote_if_any_gate_fails": True,
    }
    differences = {
        "v23_minus_general": [
            row["v23_minus_general"] for row in complete
        ],
        "v23_minus_dense": [row["v23_minus_dense"] for row in complete],
        "general_minus_dense": [
            row["general_minus_dense"] for row in complete
        ],
    }
    value = {
        "summarized_at_utc": utc_now(),
        "status": (
            "PASS_DENSE_PRESERVATION_SCREEN"
            if all(gates.values())
            else "INCOMPLETE"
            if len(complete) < EXPECTED_TASKS
            else "FAIL_DENSE_PRESERVATION_SCREEN"
        ),
        "tasks": rows,
        "aggregate": {
            "tasks_complete": len(complete),
            "treated_tasks": treated,
            "abstained_or_shared_completion_tasks": len(complete) - treated,
            "infrastructure_failures": infrastructure_failures,
            "resolved": resolved_counts,
            "accuracy": {
                arm: resolved_counts[arm] / len(complete) if complete else None
                for arm in ARMS
            },
            "accuracy_wilson95": {
                arm: _wilson(resolved_counts[arm], len(complete))
                for arm in ARMS
            },
            "concurrent_dense_passes": dense_passes,
            "damage_count_given_concurrent_dense_pass": damages,
            "damage_rate_given_concurrent_dense_pass": damage_rates,
            "damage_rate_wilson95": {
                arm: _wilson(damages[arm], dense_passes)
                for arm in REUSE_ARMS
            },
            "rescue_count_given_concurrent_dense_fail": rescues,
            "paired_mean_differences": {
                name: statistics.fmean(values) if values else None
                for name, values in differences.items()
            },
            "paired_difference_bootstrap95": {
                name: _bootstrap_difference(values)
                for name, values in differences.items()
            },
            "target_fallbacks": fallbacks,
            "dense_control_requests": dense_control_requests,
            "cacheblend_damage_rate_reference": cacheblend_rate,
        },
        "gate_outcomes": gates,
        "decision": (
            "Eligible for a separately preregistered wider paired screen; "
            "this is not yet a full225 or SWE-bench Verified claim."
            if all(gates.values())
            else "Do not promote; use task-level treated damage and branch "
            "traces to revise or reject the V23 selector."
        ),
    }
    write_json(output / "V27C_RESULT.json", value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("register", "preregister", "run", "summarize"),
        nargs="?",
        default="run",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.command == "register":
        value = register(args.output)
    elif args.command == "preregister":
        value = {"children": preregister_tasks(args.output)}
    elif args.command == "run":
        value = run_campaign(args.output)
    else:
        value = summarize(args.output)
    print(
        {
            "status": value.get("status"),
            "output": str(args.output),
            "gate_outcomes": value.get("gate_outcomes"),
        }
    )


if __name__ == "__main__":
    main()
