#!/usr/bin/env python3
"""Run the preregistered Dense 32-to-64-call calibration after SGLang.

The monitor preserves global GPU serialization by waiting for the complete
common SGLang campaign, submits one bounded Dense job, and records whether the
accuracy-floor hypothesis passed.  It never expands to reuse arms by itself.
"""

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
    campaign = (
        home
        / "CodeMAS_Project/kvflow-artifacts/impactkv_common_agent_capacity64_20260812"
    )
    logs = home / "impactkv-runtime/logs/common-baselines"
    registration = read(campaign / "CAPACITY64_REGISTRATION.json")
    status_path = campaign / "AUTOMATED_CAPACITY64_STATUS.json"
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
            "node_policy": "gpu17; gpu10-13 and gpu23-24 excluded",
        }
        save(status_path, state)

    try:
        if state["state"] == "waiting_sglang_complete":
            while True:
                sglang = read(source / "AUTOMATED_SGLANG_STATUS.json")
                if sglang.get("state") == "blocked":
                    raise RuntimeError(
                        f"upstream SGLang campaign blocked: {sglang.get('error')}"
                    )
                state["wait_reason"] = (
                    "global GPU serialization; SGLang state="
                    f"{sglang.get('state')}"
                )
                save(status_path, state)
                if sglang.get("state") == "complete":
                    break
                time.sleep(args.poll_seconds)
            job_id = base.submit(
                script=project / "benchmark/multi_workflow/slurm/common_native_agent.sbatch",
                logs=logs,
                exports={
                    "IMPACTKV_COMMON_BACKEND": "cacheblend",
                    "IMPACTKV_COMMON_MODE": "dense",
                    "IMPACTKV_COMMON_SCOPE": "custom",
                    "IMPACTKV_COMMON_CAMPAIGN": str(campaign),
                    "IMPACTKV_COMMON_DATASET": str(campaign / "capacity12_dataset"),
                    "IMPACTKV_COMMON_SNAPSHOT": str(campaign / "FROZEN_CAPACITY12.json"),
                    "IMPACTKV_COMMON_REGISTRATION": str(
                        campaign / "BRIDGE_CAPACITY12_REGISTRATION.json"
                    ),
                    "IMPACTKV_COMMON_RUN_NAMESPACE": "capacity64",
                    "IMPACTKV_AGENT_STEP_LIMIT": "64",
                },
            )
            state["jobs"]["dense64_capacity12"] = job_id
            state["active_jobs"] = ["dense64_capacity12"]
            state["state"] = "dense64_submitted"
            state.pop("wait_reason", None)
            save(status_path, state)

        if state["state"] == "dense64_submitted":
            base.wait_jobs(state, list(state["active_jobs"]), args.poll_seconds)
            run = campaign / "runs/capacity64/cacheblend_dense/all"
            official = read(run / "OFFICIAL_RESULT.json")
            report = official.get("report") or {}
            total = int(report.get("total_instances") or 0)
            errors = int(report.get("error_instances") or 0)
            resolved = int(report.get("resolved_instances") or 0)
            if official.get("returncode") != 0 or total != 12 or errors != 0:
                raise RuntimeError(f"Dense64 official validity gate failed: {official}")
            trajectories = []
            for path in sorted(run.glob("*/*.traj.json")):
                value = read(path)
                info = value.get("info") or {}
                trajectories.append(
                    {
                        "instance_id": value.get("instance_id") or path.parent.name,
                        "exit_status": info.get("exit_status"),
                        "api_calls": int(
                            (info.get("model_stats") or {}).get("api_calls") or 0
                        ),
                    }
                )
            state["result"] = {
                "official_resolved": resolved,
                "official_total": total,
                "official_errors": errors,
                "limits_exceeded_at_64": sum(
                    row["exit_status"] == "LimitsExceeded" and row["api_calls"] == 64
                    for row in trajectories
                ),
                "trajectories": trajectories,
            }
            state["hypothesis_passed"] = resolved >= 1
            state["decision"] = (
                "advance to a uniformly 64-call, original Fresh24 six-arm comparison"
                if resolved >= 1
                else "drop call-budget expansion as the sole accuracy-floor remedy"
            )
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
