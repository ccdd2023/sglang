#!/usr/bin/env python3
"""Run the preregistered search-file-section canary and Fresh24 chain."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from benchmark.multi_workflow import monitor_dependency_graph_mean_counterfactual as base


ARM = "coding_search_file_section_mean"
CAMPAIGN_NAME = "impactkv_common_agent_search_file_section_20260812"
GRAPH_NAME = "impactkv_common_agent_graph_mean_20260812"
SOURCE_NAME = "impactkv_common_agent_baselines_fresh24_20260812"
STATUS_NAME = "AUTOMATED_SEARCH_FILE_SECTION_STATUS.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    base.ARM = ARM
    home = Path.home()
    project = home / "CodeMAS_Project/worktrees/sglang-common-agent"
    artifacts = home / "CodeMAS_Project/kvflow-artifacts"
    campaign = artifacts / CAMPAIGN_NAME
    graph = artifacts / GRAPH_NAME
    source = artifacts / SOURCE_NAME
    logs = home / "impactkv-runtime/logs/common-baselines"
    status_path = campaign / STATUS_NAME
    state = base.read_json(status_path)
    canary_tasks = len(base.read_json(campaign / "CANARY4.json"))
    agent_script = project / "benchmark/multi_workflow/slurm/common_sglang_agent.sbatch"
    exact_script = (
        project
        / "benchmark/multi_workflow/slurm/common_sglang_exact_prompt_replay.sbatch"
    )
    common = {
        "IMPACTKV_COMMON_ARM": ARM,
        "IMPACTKV_COMMON_CAMPAIGN": str(campaign),
    }
    try:
        if state["state"] in {"registered", "waiting_graph_mean_terminal"}:
            state["state"] = "waiting_graph_mean_terminal"
            while True:
                graph_state = base.read_json(
                    graph / "AUTOMATED_GRAPH_MEAN_STATUS.json"
                )
                state["wait_reason"] = f"graph-mean state={graph_state.get('state')}"
                state["updated_at_utc"] = base.utc_now()
                base.atomic_json(status_path, state)
                if graph_state.get("state") in {"complete", "blocked"}:
                    break
                time.sleep(args.poll_seconds)
            name = "search_section_canary4"
            state["jobs"][name] = base.submit(
                agent_script,
                logs,
                {**common, "IMPACTKV_COMMON_SCOPE": "canary"},
            )
            state["state"] = "canary4_submitted"
            state.pop("wait_reason", None)
            base.atomic_json(status_path, state)

        if state["state"] == "canary4_submitted":
            base.wait_job(state, status_path, "search_section_canary4", args.poll_seconds)
            state["canary4"] = base.validate_online(
                campaign, "canary", canary_tasks
            )
            state["model_requests_issued"] = state["canary4"]["requests"]
            state["canary4_first_prompt_identity"] = base.validate_first_prompt_identity(
                campaign, source, "canary", canary_tasks
            )
            name = "search_section_canary4_exact"
            state["jobs"][name] = base.submit(
                exact_script,
                logs,
                {**common, "IMPACTKV_COMMON_REPLAY_LABEL": "canary4"},
            )
            state["state"] = "canary4_exact_submitted"
            base.atomic_json(status_path, state)

        if state["state"] == "canary4_exact_submitted":
            base.wait_job(
                state, status_path, "search_section_canary4_exact", args.poll_seconds
            )
            state["canary4_exact"] = base.validate_exact(campaign, "canary4")
            name = "search_section_fresh24"
            state["jobs"][name] = base.submit(
                agent_script,
                logs,
                {**common, "IMPACTKV_COMMON_SCOPE": "formal"},
            )
            state["state"] = "fresh24_submitted"
            base.atomic_json(status_path, state)

        if state["state"] == "fresh24_submitted":
            base.wait_job(state, status_path, "search_section_fresh24", args.poll_seconds)
            state["fresh24"] = base.validate_online(campaign, "formal", 24)
            state["model_requests_issued"] += state["fresh24"]["requests"]
            state["fresh24_first_prompt_identity"] = base.validate_first_prompt_identity(
                campaign, source, "formal", 24
            )
            name = "search_section_fresh24_exact"
            state["jobs"][name] = base.submit(
                exact_script,
                logs,
                {**common, "IMPACTKV_COMMON_REPLAY_LABEL": "fresh24"},
            )
            state["state"] = "fresh24_exact_submitted"
            base.atomic_json(status_path, state)

        if state["state"] == "fresh24_exact_submitted":
            base.wait_job(
                state, status_path, "search_section_fresh24_exact", args.poll_seconds
            )
            state["fresh24_exact"] = base.validate_exact(campaign, "fresh24")
            dense = base.official_report(
                base.read_json(
                    source
                    / "runs/sglang_formal/dense/full_24/OFFICIAL_RESULT.json"
                )
            )
            dense_resolved = int(dense.get("resolved_instances") or 0)
            treatment_resolved = int(state["fresh24"]["official_resolved"])
            speed_pass = state["fresh24_exact"]["cache_ready_speedup"] > 1
            accuracy_pass = treatment_resolved >= dense_resolved
            state["comparison_to_frozen_dense"] = {
                "dense_official_resolved": dense_resolved,
                "treatment_official_resolved": treatment_resolved,
                "accuracy_non_degradation_gate": accuracy_pass,
                "cache_ready_speed_gate": speed_pass,
            }
            state["decision"] = "keep" if speed_pass and accuracy_pass else "drop"
            state["state"] = "complete"
            state["finished_at_utc"] = base.utc_now()
            state["updated_at_utc"] = base.utc_now()
            base.atomic_json(status_path, state)
    except Exception as error:
        state["state"] = "blocked"
        state["error"] = f"{type(error).__name__}: {error}"
        state["updated_at_utc"] = base.utc_now()
        base.atomic_json(status_path, state)
        raise


if __name__ == "__main__":
    main()
