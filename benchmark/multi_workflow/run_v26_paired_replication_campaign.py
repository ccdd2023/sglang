#!/usr/bin/env python3
"""Pre-registered five-task causal paired V23 replication campaign."""

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
DEFAULT_OUTPUT = ARTIFACTS / "impactkv_v26_paired_replication_20260727"
PROJECT = Path(__file__).resolve().parents[2]
RUNNER = PROJECT / "benchmark/multi_workflow/run_v25_paired_agent_canary.py"
PYTHON = Path("/home/gfy/.venvs/mini-swe-agent-v2.3.0/bin/python")
SELECTION_SALT = "v26-paired-replication-v1\n"
SAMPLE_SIZE = 5
BOOTSTRAP_SEED = 20260727
BOOTSTRAPS = 100_000
V23 = "coding_post_mutation_target_prefix_v23"
GENERAL = "general"
ARMS = (V23, GENERAL)
EXCLUDED_TRANSITION_AUDIT = {
    "scikit-learn__scikit-learn-12585",
    "scikit-learn__scikit-learn-13779",
    "pylint-dev__pylint-7277",
}


def _dataset_rows() -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (DATASET / "test.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]


def selection() -> list[dict[str, str]]:
    candidates = []
    for row in _dataset_rows():
        instance_id = str(row["instance_id"])
        if instance_id in EXCLUDED_TRANSITION_AUDIT:
            continue
        digest = hashlib.sha256(
            (SELECTION_SALT + instance_id).encode()
        ).hexdigest()
        candidates.append(
            {"instance_id": instance_id, "selection_sha256": digest}
        )
    return sorted(candidates, key=lambda row: row["selection_sha256"])[
        :SAMPLE_SIZE
    ]


def task_dir(output: Path, instance_id: str) -> Path:
    return output / "tasks" / instance_id


def _command(
    output: Path,
    instance_id: str,
    command: str,
) -> list[str]:
    return [
        str(PYTHON),
        str(RUNNER),
        command,
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
        }
    )
    return env


def register(output: Path) -> dict[str, Any]:
    path = output / "V26_REGISTRATION.json"
    if path.exists():
        return read_json(path)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    selected = selection()
    value = {
        "registered_at_utc": utc_now(),
        "status": "REGISTERED_BEFORE_ANY_V26_GPU_RUN",
        "experiment": (
            "V26 outcome-independent five-task shared-history paired "
            "V23/General replication"
        ),
        "motivation": (
            "The transition-selected V25 audit had one causal V23-only "
            "resolution, zero General-only resolutions, and two joint "
            "failures.  Replicate on tasks selected without reading any arm "
            "outcome before changing the selector or expanding to full225."
        ),
        "selection": {
            "salt": SELECTION_SALT,
            "rule": (
                "Exclude the three V25 transition-audit tasks, SHA-256 "
                "(salt || instance_id), sort ascending, take first five."
            ),
            "sample_size": SAMPLE_SIZE,
            "excluded_transition_audit": sorted(
                EXCLUDED_TRANSITION_AUDIT
            ),
            "selected": selected,
            "outcomes_used": False,
        },
        "protocol": {
            "runner": str(RUNNER),
            "same_model_engine_tokenization": True,
            "shared_dense_history_until_first_unequal_online_span": True,
            "no_unequal_span_or_early_exit_is_shared_itt_tie": True,
            "repository_container_snapshot_before_branch": True,
            "target_order": list(ARMS),
            "step_limit": 20,
            "official_swebench_container_each_arm": True,
            "prefetch": False,
            "reference_patch_or_future_output_used": False,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_iterations": BOOTSTRAPS,
        },
        "frozen_development_gates": {
            "official_tasks_completed": SAMPLE_SIZE,
            "runner_infrastructure_failures": 0,
            "target_fallbacks": 0,
            "v23_resolved_not_below_general": True,
            "v23_only_not_below_general_only": True,
            "v23_only_min": 1,
            "report_paired_accuracy_difference_bootstrap95": True,
            "do_not_promote_to_full225_if_gate_fails": True,
        },
        "commands": {
            row["instance_id"]: {
                stage: _command(
                    output,
                    row["instance_id"],
                    stage,
                )
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
    output: Path,
    instance_id: str,
    stage: str,
) -> dict[str, Any]:
    command = _command(output, instance_id, stage)
    log_path = (
        output / "orchestration_logs" / instance_id / f"V26_{stage}.log"
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
        / f"V26_{stage}_STATUS.json",
        value,
    )
    return value


def preregister_tasks(output: Path) -> list[dict[str, Any]]:
    registration = register(output)
    rows = []
    for selected in registration["selection"]["selected"]:
        instance_id = selected["instance_id"]
        child = task_dir(output, instance_id) / "V25_REGISTRATION.json"
        if child.exists():
            rows.append(
                {
                    "instance_id": instance_id,
                    "stage": "register",
                    "returncode": 0,
                    "resumed": True,
                }
            )
        else:
            rows.append(_run_stage(output, instance_id, "register"))
    write_json(output / "V26_CHILD_REGISTRATIONS.json", rows)
    if any(row["returncode"] != 0 for row in rows):
        raise RuntimeError("one or more child preregistrations failed")
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
    write_json(output / "V26_STAGE_STATUS.json", stages)
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
            rows.append(
                {
                    "instance_id": instance_id,
                    "status": "INCOMPLETE",
                }
            )
            continue
        runtime = read_json(runtime_path)
        official = read_json(official_path)
        resolved = {
            arm: int(official["arms"][arm]["resolved"]) for arm in ARMS
        }
        rows.append(
            {
                "instance_id": instance_id,
                "status": "COMPLETE",
                "resolved": resolved,
                "paired_difference": resolved[V23] - resolved[GENERAL],
                "v23_only": resolved[V23] == 1
                and resolved[GENERAL] == 0,
                "general_only": resolved[V23] == 0
                and resolved[GENERAL] == 1,
                "both_resolved": all(resolved.values()),
                "both_failed": not any(resolved.values()),
                "branch": runtime["branch"],
                "copy_counts": runtime["server"]["copy_counts"],
                "target_fallbacks": runtime["server"][
                    "target_fallbacks"
                ],
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
    differences = [row["paired_difference"] for row in complete]
    v23_only = sum(row["v23_only"] for row in complete)
    general_only = sum(row["general_only"] for row in complete)
    fallbacks = sum(row["target_fallbacks"] for row in complete)
    gate_outcomes = {
        "official_tasks_completed": len(complete) == SAMPLE_SIZE,
        "runner_infrastructure_failures": infrastructure_failures == 0,
        "target_fallbacks": fallbacks == 0,
        "v23_resolved_not_below_general": (
            resolved_counts[V23] >= resolved_counts[GENERAL]
        ),
        "v23_only_not_below_general_only": v23_only >= general_only,
        "v23_only_min": v23_only >= 1,
        "report_paired_accuracy_difference_bootstrap95": bool(complete),
        "do_not_promote_to_full225_if_gate_fails": True,
    }
    value = {
        "summarized_at_utc": utc_now(),
        "status": (
            "PASS_DEVELOPMENT_REPLICATION"
            if all(gate_outcomes.values())
            else "INCOMPLETE"
            if len(complete) < SAMPLE_SIZE
            else "FAIL_DEVELOPMENT_REPLICATION"
        ),
        "tasks": rows,
        "aggregate": {
            "tasks_complete": len(complete),
            "infrastructure_failures": infrastructure_failures,
            "resolved": resolved_counts,
            "accuracy": {
                arm: (
                    resolved_counts[arm] / len(complete)
                    if complete
                    else None
                )
                for arm in ARMS
            },
            "accuracy_wilson95": {
                arm: _wilson(resolved_counts[arm], len(complete))
                for arm in ARMS
            },
            "paired_accuracy_difference_v23_minus_general": (
                statistics.fmean(differences) if differences else None
            ),
            "paired_difference_bootstrap95": _bootstrap_difference(
                differences
            ),
            "v23_only": v23_only,
            "general_only": general_only,
            "both_resolved": sum(
                row["both_resolved"] for row in complete
            ),
            "both_failed": sum(row["both_failed"] for row in complete),
            "target_fallbacks": fallbacks,
        },
        "gate_outcomes": gate_outcomes,
        "decision": (
            "Eligible for a separately pre-registered full225 development "
            "screen; not yet a KVCOMM/CacheBlend or Verified promotion claim."
            if all(gate_outcomes.values())
            else "Do not expand to full225; audit failures and revise."
        ),
    }
    write_json(output / "V26_RESULT.json", value)
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
