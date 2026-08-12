#!/usr/bin/env python3
"""Run and decide the preregistered four-task format-loop counterfactual."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from benchmark.multi_workflow import monitor_common_baseline_campaign as base


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, value: dict[str, Any]) -> None:
    value["updated_at_utc"] = base.utc_now()
    base.atomic_json(path, value)


def result_row(path: Path) -> dict[str, Any]:
    value = read(path)
    info = value.get("info") or {}
    messages = value.get("messages") or []
    return {
        "instance_id": str(value.get("instance_id") or path.parent.name),
        "exit_status": info.get("exit_status"),
        "api_calls": int((info.get("model_stats") or {}).get("api_calls") or 0),
        "format_errors": sum(
            message.get("role") == "user"
            and (message.get("extra") or {}).get("interrupt_type") == "FormatError"
            for message in messages
        ),
        "executed_tool_calls": sum(message.get("role") == "tool" for message in messages),
        "submission_nonempty": bool(str(info.get("submission") or "").strip()),
        "notice_tool_calls": sum(
            message.get("role") == "tool"
            and "preceding assistant text contained no executable tool call"
            in str(message.get("content") or "")
            for message in messages
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    home = Path.home()
    project = home / "CodeMAS_Project/worktrees/sglang-common-agent"
    source = (
        home
        / "CodeMAS_Project/kvflow-artifacts/impactkv_common_agent_baselines_fresh24_20260812"
    )
    graph_mean = (
        home
        / "CodeMAS_Project/kvflow-artifacts/impactkv_common_agent_graph_mean_20260812"
    )
    campaign = (
        home
        / "CodeMAS_Project/kvflow-artifacts/impactkv_common_agent_format_guard_20260812"
    )
    logs = home / "impactkv-runtime/logs/common-baselines"
    registration = read(campaign / "FORMAT_GUARD_REGISTRATION.json")
    status_path = campaign / "AUTOMATED_FORMAT_GUARD_STATUS.json"
    if status_path.exists():
        state = read(status_path)
    else:
        state = {
            "schema_version": 1,
            "state": "waiting_sglang_complete",
            "started_at_utc": base.utc_now(),
            "status_path": str(status_path),
            "jobs": {},
            "hypothesis": registration["motivation"]["falsifiable_hypothesis"],
        }
        save(status_path, state)

    try:
        if state["state"] == "waiting_sglang_complete":
            while True:
                sglang = read(source / "AUTOMATED_SGLANG_STATUS.json")
                upstream_block_is_expected_copy_abstention = (
                    sglang.get("state") == "blocked"
                    and "no physical coding-aware K/V copy"
                    in str(sglang.get("error") or "")
                )
                if sglang.get("state") == "blocked" and not (
                    upstream_block_is_expected_copy_abstention
                ):
                    raise RuntimeError(
                        f"upstream SGLang campaign blocked: {sglang.get('error')}"
                    )
                state["wait_reason"] = (
                    "global GPU serialization; SGLang state="
                    f"{sglang.get('state')}"
                )
                save(status_path, state)
                if sglang.get("state") == "complete" or (
                    upstream_block_is_expected_copy_abstention
                ):
                    break
                time.sleep(args.poll_seconds)
            graph_status_path = graph_mean / "AUTOMATED_GRAPH_MEAN_STATUS.json"
            while graph_status_path.is_file():
                graph_state = read(graph_status_path)
                if graph_state.get("state") in {"complete", "blocked"}:
                    break
                state["wait_reason"] = (
                    "global GPU serialization; graph-mean state="
                    f"{graph_state.get('state')}"
                )
                save(status_path, state)
                time.sleep(args.poll_seconds)
            job_id = base.submit(
                script=project / "benchmark/multi_workflow/slurm/common_native_agent.sbatch",
                logs=logs,
                exports={
                    "IMPACTKV_COMMON_BACKEND": "cacheblend",
                    "IMPACTKV_COMMON_MODE": "dense",
                    "IMPACTKV_COMMON_SCOPE": "custom",
                    "IMPACTKV_COMMON_CAMPAIGN": str(campaign),
                    "IMPACTKV_COMMON_DATASET": str(campaign / "format_guard4_dataset"),
                    "IMPACTKV_COMMON_SNAPSHOT": str(campaign / "FROZEN_FORMAT_GUARD4.json"),
                    "IMPACTKV_COMMON_REGISTRATION": str(
                        campaign / "BRIDGE_FORMAT_GUARD4_REGISTRATION.json"
                    ),
                    "IMPACTKV_COMMON_RUN_NAMESPACE": "format_guard32",
                    "IMPACTKV_AGENT_STEP_LIMIT": "32",
                    "IMPACTKV_RECOVER_UNPARSED_OUTPUT_WITH_NOTICE": "1",
                },
            )
            state["jobs"]["dense_format_guard4"] = job_id
            state["active_jobs"] = ["dense_format_guard4"]
            state["state"] = "format_guard_submitted"
            state.pop("wait_reason", None)
            save(status_path, state)

        if state["state"] == "format_guard_submitted":
            base.wait_jobs(state, list(state["active_jobs"]), args.poll_seconds)
            run = campaign / "runs/format_guard32/cacheblend_dense/all"
            rows = [result_row(path) for path in sorted(run.glob("*/*.traj.json"))]
            if len(rows) != 4:
                raise RuntimeError(f"expected four treatment trajectories, found {len(rows)}")
            official = read(run / "OFFICIAL_RESULT.json")
            report = official.get("report") or {}
            if (
                official.get("returncode") != 0
                or int(report.get("total_instances") or 0) != 4
                or int(report.get("error_instances") or 0) != 0
            ):
                raise RuntimeError(f"format-guard official validity gate failed: {official}")
            controls = registration["selection"]["selected"]
            control_errors = sum(int(row["format_errors"]) for row in controls)
            control_tools = sum(int(row["executed_tool_calls"]) for row in controls)
            treatment_errors = sum(row["format_errors"] for row in rows)
            treatment_tools = sum(row["executed_tool_calls"] for row in rows)
            mechanism_passed = (
                treatment_errors <= 0.5 * control_errors
                and treatment_tools > control_tools
            )
            task_signal = (
                int(report.get("completed_instances") or 0) >= 1
                or int(report.get("resolved_instances") or 0) >= 1
            )
            state["result"] = {
                "control_format_errors": control_errors,
                "treatment_format_errors": treatment_errors,
                "control_executed_tool_calls": control_tools,
                "treatment_executed_tool_calls": treatment_tools,
                "notice_tool_calls": sum(row["notice_tool_calls"] for row in rows),
                "nonempty_submissions": sum(row["submission_nonempty"] for row in rows),
                "official_completed": int(report.get("completed_instances") or 0),
                "official_resolved": int(report.get("resolved_instances") or 0),
                "official_total": 4,
                "trajectories": rows,
            }
            state["mechanism_gate_passed"] = mechanism_passed
            state["task_signal_gate_passed"] = task_signal
            if mechanism_passed and task_signal:
                state["decision"] = (
                    "keep the transport guard and preregister a common-protocol accuracy rerun"
                )
            elif mechanism_passed:
                state["decision"] = (
                    "keep only as a transport fix; it is insufficient to restore accuracy"
                )
            else:
                state["decision"] = "drop the notice guard"
            state["state"] = "complete"
            state["finished_at_utc"] = base.utc_now()
            save(status_path, state)
    except Exception as error:
        state["state"] = "blocked"
        state["error"] = f"{type(error).__name__}: {error}"
        save(status_path, state)
        raise


if __name__ == "__main__":
    main()
