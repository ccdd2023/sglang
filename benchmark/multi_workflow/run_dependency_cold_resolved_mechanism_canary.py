#!/usr/bin/env python3
"""Accuracy-resolved same-history mechanism canary for dependency-cold reuse.

This deliberately outcome-selects one historical Dense-resolved task whose
first eight requests are read-only and whose ninth request has a positive-cost
620-token dependency-cold target.  It can test causal preservation on this
mechanism case, but it must not be reported as a population accuracy estimate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmark.multi_workflow import run_dependency_cold_same_history_canary as trial
from benchmark.multi_workflow import run_same_history_fork_continuation as base


ARTIFACTS = Path("/home/gfy/CodeMAS_Project/kvflow-artifacts")
SOURCE_ROOT = (
    ARTIFACTS / "impactkv_natural_code_cost_same_history_fork_20260809/dense"
)
SOURCE_SNAPSHOT = (
    ARTIFACTS
    / "impactkv_natural_code_cost_discordant7_repeat_20260809/FROZEN_DISCORDANT7.json"
)
DEFAULT_OUTPUT = (
    ARTIFACTS
    / "impactkv_dependency_cold_resolved_mechanism_20260810/sympy22914_q9"
)
POLICY_ARM = trial.POLICY_ARM
ARMS = ("dense", POLICY_ARM)
ARM_PORTS = {"dense": 32540, POLICY_ARM: 32541}
FORKS = {
    "sympy__sympy-22914": {
        "request": 9,
        "planned_tokens": 620,
        "predicted_saving_ms": 37.955,
    }
}


def configure() -> None:
    trial.SOURCE_ROOT = SOURCE_ROOT
    trial.SOURCE_SNAPSHOT = SOURCE_SNAPSHOT
    trial.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    trial.POLICY_ARM = POLICY_ARM
    trial.ARMS = ARMS
    trial.ARM_PORTS = ARM_PORTS
    trial.FORKS = FORKS
    trial.ELIGIBLE_CAPACITY = {
        "historical_official_dense_resolved_trajectories_audited": 7,
        "resolved_tasks_with_positive_dependency_cold_target": 2,
        "selected_clean_read_only_prefix_tasks": 1,
    }
    trial.configure_base()
    base.SOURCE_TRAJECTORIES = SOURCE_ROOT
    base.SOURCE_SNAPSHOT = SOURCE_SNAPSHOT
    base.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    base.POLICY_ARM = POLICY_ARM
    base.ARMS = ARMS
    base.ARM_PORTS = ARM_PORTS
    base.FORK_TASKS = {
        instance_id: int(value["request"])
        for instance_id, value in FORKS.items()
    }
    base.CAUSAL_SCOPE = (
        "Outcome-selected one-task accuracy-resolved mechanism canary. It can "
        "test this causal fork but cannot estimate population accuracy."
    )
    base.prepare = prepare


def registration_payload() -> dict[str, Any]:
    value = trial.registration_payload()
    value.update(
        status="REGISTERED_BEFORE_RESOLVED_MECHANISM_FORK_OUTCOMES",
        purpose=(
            "test whether dependency-cold lossy reuse preserves the official "
            "resolved outcome of a late, read-only-prefix same-history fork "
            "while reducing cache-ready target TTFT"
        ),
        metric_scope=(
            "outcome-selected single-task mechanism canary; never report its "
            "resolved fraction as a population accuracy estimate"
        ),
    )
    value["selection"].update(
        used_official_or_treatment_outcomes=True,
        outcome_selected=True,
        rule=(
            "historical official Dense-resolved task with a positive-cost "
            "dependency-cold target, read-only workspace prefix, and no more "
            "than six historical requests remaining"
        ),
        reason=(
            "the preceding prompt-structure-selected cohort had no paired "
            "non-empty official patches and therefore no accuracy resolution"
        ),
        population_claim_allowed=False,
    )
    value["gates"].update(
        paired_official_completed_tasks_min=1,
        dense_resolved_preserved=True,
        target_response_equality_required=False,
    )
    return value


def prepare(output: Path) -> dict[str, Any]:
    path = output / "CAMPAIGN_REGISTRATION.json"
    if path.exists():
        return base.read_json(path)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True)
    registration = registration_payload()
    base.write_json(path, registration)
    base.write_json(
        output / "BRIDGE_REGISTRATION.json",
        {
            "schema_version": 1,
            "registration_id": output.name,
            "registered_at_utc": registration["registered_at_utc"],
            "dataset": {
                "name": "princeton-nlp/SWE-bench_Verified",
                "split": "test",
            },
            "instances": [
                {"instance_id": instance_id} for instance_id in FORKS
            ],
        },
    )
    return registration


def summarize(output: Path) -> dict[str, Any]:
    value = trial.summarize(output)
    preserved = (
        value["accuracy"]["dense_resolved"] == len(FORKS)
        and value["accuracy"]["policy_resolved"] == len(FORKS)
    )
    value["mechanism_gate"] = {
        "paired_official_completed_tasks_min_1": bool(
            value["posthoc_validity"]["paired_official_completed_tasks"]
        ),
        "historical_dense_resolved_preserved": preserved,
        "cache_ready_ttft_saving_positive": value["fork_target_latency"][
            "median_ttft_saving_percent"
        ]
        > 0,
        "physical_lossy_copy": value["physical_reuse"][
            "fork_target_copied_k_tokens"
        ]
        == sum(int(row["planned_tokens"]) for row in FORKS.values()),
    }
    value["decision"] = (
        "MECHANISM_PASS_EXPAND_TO_FRESH_ACCURACY_COHORT"
        if all(value["mechanism_gate"].values())
        else "MECHANISM_FAIL_DIAGNOSE_BEFORE_EXPANSION"
    )
    base.write_json(output / "RESULT.json", value)
    return value


def main() -> None:
    configure()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    run = sub.add_parser("run-arm")
    run.add_argument("--arm", choices=ARMS, required=True)
    run.add_argument("--port", type=int)
    evaluate = sub.add_parser("evaluate-arm")
    evaluate.add_argument("--arm", choices=ARMS, required=True)
    sub.add_parser("summarize")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.command == "prepare":
        value = prepare(output)
    elif args.command == "run-arm":
        value = base.run_arm(output, args.arm, args.port or ARM_PORTS[args.arm])
    elif args.command == "evaluate-arm":
        value = base.evaluate_arm(output, args.arm)
    else:
        value = summarize(output)
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
