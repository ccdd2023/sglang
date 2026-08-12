#!/usr/bin/env python3
"""Run the preregistered graph-mean canary, Fresh24, and exact TTFT replay."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ARM = "coding_dependency_graph_cold_mean"
CAMPAIGN_NAME = "impactkv_common_agent_graph_mean_20260812"
SOURCE_NAME = "impactkv_common_agent_baselines_fresh24_20260812"
STATUS_NAME = "AUTOMATED_GRAPH_MEAN_STATUS.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def slurm_state(job_id: str) -> str:
    queued = subprocess.run(
        ["squeue", "-h", "-j", job_id, "-o", "%T"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    ).stdout.strip()
    if queued:
        return queued.splitlines()[0]
    accounted = subprocess.run(
        ["sacct", "-n", "-X", "-j", job_id, "-o", "State"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    ).stdout.strip()
    return accounted.splitlines()[0].split()[0] if accounted else "UNKNOWN"


def submit(script: Path, logs: Path, exports: dict[str, str]) -> str:
    result = subprocess.run(
        [
            "sbatch",
            "--parsable",
            f"--chdir={logs}",
            "--export=ALL," + ",".join(f"{key}={value}" for key, value in exports.items()),
            str(script),
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return result.stdout.strip().split(";")[0]


def wait_job(state: dict[str, Any], status_path: Path, name: str, poll: int) -> None:
    while True:
        observed = slurm_state(str(state["jobs"][name]))
        state.setdefault("slurm_states", {})[name] = observed
        state["updated_at_utc"] = utc_now()
        atomic_json(status_path, state)
        if observed == "COMPLETED":
            return
        if observed.startswith(
            ("FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL")
        ):
            raise RuntimeError(f"{name} failed: {observed}")
        time.sleep(poll)


def official_report(value: dict[str, Any]) -> dict[str, Any]:
    report = value.get("report")
    if not isinstance(report, dict):
        nested = value.get("result") or {}
        report = nested.get("report")
    if not isinstance(report, dict):
        raise ValueError("official evaluator report absent")
    return report


def validate_online(campaign: Path, scope: str, tasks: int) -> dict[str, Any]:
    run = campaign / "runs" / f"sglang_{scope}" / ARM / f"full_{tasks}"
    runtime = read_json(run / "RUNTIME_SUMMARY.json")
    official = official_report(read_json(run / "OFFICIAL_RESULT.json"))
    telemetry = read_json(run / "TELEMETRY.json")
    infra = {
        key: row.get("exit_status")
        for key, row in (telemetry.get("instances") or {}).items()
        if row.get("exit_status") in {"HTTPError", "ConnectionError"}
    }
    if infra:
        raise RuntimeError(f"transport failures: {infra}")
    copies = int(runtime.get("target_copy_events") or 0)
    fallbacks = int(runtime.get("target_fallback_events") or 0)
    submitted = int(official.get("submitted_instances") or 0)
    if submitted != tasks:
        raise RuntimeError(f"official coverage {submitted}/{tasks}")
    if copies < 1 or fallbacks != 0:
        raise RuntimeError(
            f"physical copy gate failed: copies={copies}, fallbacks={fallbacks}"
        )
    return {
        "run_dir": str(run),
        "requests": int(runtime.get("requests") or 0),
        "target_copy_events": copies,
        "copied_tokens": int(runtime.get("copied_tokens") or 0),
        "target_fallback_events": fallbacks,
        "official_resolved": int(official.get("resolved_instances") or 0),
        "official_submitted": submitted,
    }


def validate_exact(campaign: Path, label: str) -> dict[str, Any]:
    path = campaign / "exact_prompt_replay" / label / "sglang_coding/RESULT.json"
    result = read_json(path)
    if result.get("status") != "PASS":
        raise RuntimeError(f"exact replay failed: {result.get('status')}")
    summary = result["summary"]
    return {
        "result": str(path),
        "targets": int(summary["targets"]),
        "cache_ready_speedup": float(summary["median_target_cache_ready_speedup"]),
        "n1_including_build_speedup": float(
            summary["median_target_n1_including_build_speedup"]
        ),
        "n4_including_build_speedup": float(
            summary["median_target_n4_including_build_speedup"]
        ),
        "n16_including_build_speedup": float(
            summary["median_target_n16_including_build_speedup"]
        ),
        "physical_copy_events": int(summary["physical_copy_events"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    home = Path.home()
    project = home / "CodeMAS_Project/worktrees/sglang-common-agent"
    artifacts = home / "CodeMAS_Project/kvflow-artifacts"
    campaign = artifacts / CAMPAIGN_NAME
    source = artifacts / SOURCE_NAME
    logs = home / "impactkv-runtime/logs/common-baselines"
    status_path = campaign / STATUS_NAME
    if not (campaign / "GRAPH_MEAN_PREREGISTRATION.json").is_file():
        raise FileNotFoundError("graph-mean preregistration absent")
    state = read_json(status_path)
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
        if state["state"] == "registered":
            source_status = read_json(source / "AUTOMATED_SGLANG_STATUS.json")
            if source_status.get("state") not in {"complete", "blocked"}:
                state["state"] = "waiting_source_campaign_terminal"
                atomic_json(status_path, state)

        if state["state"] == "waiting_source_campaign_terminal":
            while True:
                source_status = read_json(source / "AUTOMATED_SGLANG_STATUS.json")
                if source_status.get("state") in {"complete", "blocked"}:
                    break
                state["source_state"] = source_status.get("state")
                state["updated_at_utc"] = utc_now()
                atomic_json(status_path, state)
                time.sleep(args.poll_seconds)
            state["state"] = "registered"
            atomic_json(status_path, state)

        if state["state"] == "registered":
            name = "graph_mean_canary4"
            state["jobs"][name] = submit(
                agent_script,
                logs,
                {**common, "IMPACTKV_COMMON_SCOPE": "canary"},
            )
            state["model_requests_issued"] = 0
            state["state"] = "canary4_submitted"
            atomic_json(status_path, state)

        if state["state"] == "canary4_submitted":
            wait_job(state, status_path, "graph_mean_canary4", args.poll_seconds)
            state["canary4"] = validate_online(campaign, "canary", 4)
            state["model_requests_issued"] = state["canary4"]["requests"]
            name = "graph_mean_canary4_exact"
            state["jobs"][name] = submit(
                exact_script,
                logs,
                {**common, "IMPACTKV_COMMON_REPLAY_LABEL": "canary4"},
            )
            state["state"] = "canary4_exact_submitted"
            atomic_json(status_path, state)

        if state["state"] == "canary4_exact_submitted":
            wait_job(state, status_path, "graph_mean_canary4_exact", args.poll_seconds)
            state["canary4_exact"] = validate_exact(campaign, "canary4")
            name = "graph_mean_fresh24"
            state["jobs"][name] = submit(
                agent_script,
                logs,
                {**common, "IMPACTKV_COMMON_SCOPE": "formal"},
            )
            state["state"] = "fresh24_submitted"
            atomic_json(status_path, state)

        if state["state"] == "fresh24_submitted":
            wait_job(state, status_path, "graph_mean_fresh24", args.poll_seconds)
            state["fresh24"] = validate_online(campaign, "formal", 24)
            state["model_requests_issued"] += state["fresh24"]["requests"]
            name = "graph_mean_fresh24_exact"
            state["jobs"][name] = submit(
                exact_script,
                logs,
                {**common, "IMPACTKV_COMMON_REPLAY_LABEL": "fresh24"},
            )
            state["state"] = "fresh24_exact_submitted"
            atomic_json(status_path, state)

        if state["state"] == "fresh24_exact_submitted":
            wait_job(state, status_path, "graph_mean_fresh24_exact", args.poll_seconds)
            state["fresh24_exact"] = validate_exact(campaign, "fresh24")
            state["state"] = "complete"
            state["decision"] = (
                "keep" if state["fresh24_exact"]["cache_ready_speedup"] > 1 else "drop"
            )
            state["finished_at_utc"] = utc_now()
            state["updated_at_utc"] = utc_now()
            atomic_json(status_path, state)
    except Exception as error:
        state["state"] = "blocked"
        state["error"] = f"{type(error).__name__}: {error}"
        state["updated_at_utc"] = utc_now()
        atomic_json(status_path, state)
        raise


if __name__ == "__main__":
    main()
