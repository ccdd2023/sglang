#!/usr/bin/env python3
"""Run the preregistered three-island search-file counterfactual."""

from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path
from typing import Any

from benchmark.multi_workflow import monitor_dependency_graph_mean_counterfactual as base
from benchmark.multi_workflow.monitor_search_file_section_counterfactual import (
    effective_speed_result,
)
from benchmark.multi_workflow.prepare_search_file_section_multi_counterfactual import (
    BASELINE_N1_MAX,
    SINGLE_ACTION_SPEED,
)


ARM = "coding_search_file_section_multi_mean"
CAMPAIGN_NAME = "impactkv_common_agent_search_file_section_multi_20260813"
SOURCE_NAME = "impactkv_common_agent_search_file_section_20260812"
BASELINE_NAME = "impactkv_common_agent_baselines_fresh24_20260812"
STATUS_NAME = "AUTOMATED_SEARCH_FILE_MULTI_STATUS.json"


def multi_manifest_gate(campaign: Path, scope: str, tasks: int) -> dict[str, Any]:
    run = campaign / f"runs/sglang_{scope}/{ARM}/full_{tasks}"
    manifest = base.read_json(run / "DYNAMIC_MANIFEST.json")
    counts = Counter(str(row["target_group_id"]) for row in manifest.get("cases") or ())
    if not counts:
        raise RuntimeError("multi-island online run registered no target groups")
    if min(counts.values()) < 2 or max(counts.values()) > 3:
        raise RuntimeError(f"multi-island manifest gate failed: {sorted(counts.values())}")
    return {
        "target_groups": len(counts),
        "target_islands": sum(counts.values()),
        "minimum_islands_per_target": min(counts.values()),
        "maximum_islands_per_target": max(counts.values()),
    }


def dense_resolved_for_ids(source: Path, instance_ids: set[str]) -> int:
    report = base.official_report(
        base.read_json(
            source / "runs/sglang_formal/dense/full_24/OFFICIAL_RESULT.json"
        )
    )
    return len(instance_ids & set(report.get("resolved_ids") or ()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    base.ARM = ARM
    home = Path.home()
    project = home / "CodeMAS_Project/worktrees/sglang-common-agent"
    artifacts = home / "CodeMAS_Project/kvflow-artifacts"
    campaign = artifacts / CAMPAIGN_NAME
    source = artifacts / SOURCE_NAME
    baseline = artifacts / BASELINE_NAME
    logs = home / "impactkv-runtime/logs/common-baselines"
    status_path = campaign / STATUS_NAME
    state = base.read_json(status_path)
    canary_tasks = len(base.read_json(campaign / "CANARY4.json"))
    agent_script = project / "benchmark/multi_workflow/slurm/common_sglang_agent.sbatch"
    exact_script = project / "benchmark/multi_workflow/slurm/common_sglang_exact_prompt_replay.sbatch"
    common = {"IMPACTKV_COMMON_ARM": ARM, "IMPACTKV_COMMON_CAMPAIGN": str(campaign)}
    try:
        if state["state"] in {"registered", "waiting_single_island_terminal"}:
            state["state"] = "waiting_single_island_terminal"
            while True:
                upstream = base.read_json(
                    source / "AUTOMATED_SEARCH_FILE_SECTION_STATUS.json"
                )
                state["wait_reason"] = f"single-island state={upstream.get('state')}"
                state["updated_at_utc"] = base.utc_now()
                base.atomic_json(status_path, state)
                if upstream.get("state") in {"complete", "blocked"}:
                    break
                time.sleep(args.poll_seconds)
            if upstream.get("state") != "complete":
                raise RuntimeError(f"single-island campaign blocked: {upstream.get('error')}")
            name = "search_multi_canary"
            state["jobs"][name] = base.submit(
                agent_script, logs, {**common, "IMPACTKV_COMMON_SCOPE": "canary"}
            )
            state["state"] = "canary_submitted"
            state.pop("wait_reason", None)
            base.atomic_json(status_path, state)

        if state["state"] == "canary_submitted":
            base.wait_job(state, status_path, "search_multi_canary", args.poll_seconds)
            state["canary"] = base.validate_online(campaign, "canary", canary_tasks)
            state["canary_manifest"] = multi_manifest_gate(
                campaign, "canary", canary_tasks
            )
            state["canary_first_prompt_identity"] = base.validate_first_prompt_identity(
                campaign, baseline, "canary", canary_tasks
            )
            state["model_requests_issued"] = state["canary"]["requests"]
            name = "search_multi_canary_exact"
            state["jobs"][name] = base.submit(
                exact_script,
                logs,
                {**common, "IMPACTKV_COMMON_REPLAY_LABEL": "canary4"},
            )
            state["state"] = "canary_exact_submitted"
            base.atomic_json(status_path, state)

        if state["state"] == "canary_exact_submitted":
            base.wait_job(
                state, status_path, "search_multi_canary_exact", args.poll_seconds
            )
            state["canary_exact"] = base.validate_exact(campaign, "canary4")
            state["canary_effective_speed"] = effective_speed_result(
                project, campaign, "canary4", ARM
            )
            canary_ids = {
                str(row["instance_id"])
                for row in base.read_json(campaign / "CANARY4.json")
            }
            dense_resolved = dense_resolved_for_ids(baseline, canary_ids)
            treatment_resolved = int(state["canary"]["official_resolved"])
            action_speed = float(
                state["canary_effective_speed"]["saved_action_targets"][
                    "cache_ready_speedup_ratio_of_sums"
                ]
            )
            speed_pass = action_speed > SINGLE_ACTION_SPEED
            accuracy_pass = treatment_resolved >= dense_resolved
            state["canary_gate"] = {
                "dense_official_resolved": dense_resolved,
                "treatment_official_resolved": treatment_resolved,
                "accuracy_non_degradation_gate": accuracy_pass,
                "saved_action_ratio_of_sums_speedup": action_speed,
                "single_island_reference": SINGLE_ACTION_SPEED,
                "speed_improvement_gate": speed_pass,
            }
            if not (speed_pass and accuracy_pass):
                state["decision"] = "drop"
                state["state"] = "complete"
                state["finished_at_utc"] = base.utc_now()
                base.atomic_json(status_path, state)
                return
            name = "search_multi_fresh24"
            state["jobs"][name] = base.submit(
                agent_script, logs, {**common, "IMPACTKV_COMMON_SCOPE": "formal"}
            )
            state["state"] = "fresh24_submitted"
            base.atomic_json(status_path, state)

        if state["state"] == "fresh24_submitted":
            base.wait_job(state, status_path, "search_multi_fresh24", args.poll_seconds)
            state["fresh24"] = base.validate_online(campaign, "formal", 24)
            state["fresh24_manifest"] = multi_manifest_gate(campaign, "formal", 24)
            state["fresh24_first_prompt_identity"] = base.validate_first_prompt_identity(
                campaign, baseline, "formal", 24
            )
            state["model_requests_issued"] += state["fresh24"]["requests"]
            name = "search_multi_fresh24_exact"
            state["jobs"][name] = base.submit(
                exact_script,
                logs,
                {**common, "IMPACTKV_COMMON_REPLAY_LABEL": "fresh24"},
            )
            state["state"] = "fresh24_exact_submitted"
            base.atomic_json(status_path, state)

        if state["state"] == "fresh24_exact_submitted":
            base.wait_job(
                state, status_path, "search_multi_fresh24_exact", args.poll_seconds
            )
            state["fresh24_exact"] = base.validate_exact(campaign, "fresh24")
            state["fresh24_effective_speed"] = effective_speed_result(
                project, campaign, "fresh24", ARM
            )
            dense_report = base.official_report(
                base.read_json(
                    baseline / "runs/sglang_formal/dense/full_24/OFFICIAL_RESULT.json"
                )
            )
            dense_resolved = int(dense_report.get("resolved_instances") or 0)
            treatment_resolved = int(state["fresh24"]["official_resolved"])
            lifecycle = float(
                state["fresh24_effective_speed"]["actual_online_materialization"][
                    "observed_online_lifecycle_speedup"
                ]
            )
            accuracy_pass = treatment_resolved >= dense_resolved
            speed_pass = lifecycle > BASELINE_N1_MAX
            state["comparison_to_frozen_baselines"] = {
                "dense_official_resolved": dense_resolved,
                "treatment_official_resolved": treatment_resolved,
                "accuracy_non_degradation_gate": accuracy_pass,
                "observed_online_lifecycle_speedup": lifecycle,
                "native_baseline_best_n1": BASELINE_N1_MAX,
                "native_baseline_n1_speed_gate": speed_pass,
            }
            state["decision"] = "keep" if speed_pass and accuracy_pass else "drop"
            state["state"] = "complete"
            state["finished_at_utc"] = base.utc_now()
            base.atomic_json(status_path, state)
    except Exception as error:
        state["state"] = "blocked"
        state["error"] = f"{type(error).__name__}: {error}"
        state["updated_at_utc"] = base.utc_now()
        base.atomic_json(status_path, state)
        raise


if __name__ == "__main__":
    main()
