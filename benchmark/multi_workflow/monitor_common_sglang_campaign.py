#!/usr/bin/env python3
"""Advance the SGLang arms of the common Qwen2.5 rolling-agent campaign.

This monitor complements ``monitor_common_baseline_campaign.py``.  It waits
for the native CacheBlend/KVCOMM Canary4 identity gate, then runs SGLang Dense
and the dependency-graph cold-reuse policy on the same Canary4.  Only a valid
physical-copy canary is allowed to expand to the already frozen Fresh24.  The
state file is resumable: restarting the monitor never resubmits recorded jobs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ARMS = ("dense", "coding_dependency_graph_cold_lcb")
NATIVE_STATUS = "AUTOMATED_CAMPAIGN_STATUS.json"
STATUS_NAME = "AUTOMATED_SGLANG_STATUS.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def submit(
    *, script: Path, logs: Path, exports: dict[str, str], dependency: str | None
) -> str:
    command = [
        "sbatch",
        "--parsable",
        f"--chdir={logs}",
        "--export=ALL," + ",".join(f"{key}={value}" for key, value in exports.items()),
    ]
    if dependency:
        command.append(f"--dependency=afterok:{dependency}")
    command.append(str(script))
    result = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return result.stdout.strip().split(";")[0]


def wait_jobs(
    state: dict[str, Any], status_path: Path, names: list[str], poll_seconds: int
) -> None:
    while True:
        observed = {name: slurm_state(state["jobs"][name]) for name in names}
        state["slurm_states"] = observed
        state["updated_at_utc"] = utc_now()
        atomic_json(status_path, state)
        failed = {
            name: value
            for name, value in observed.items()
            if value.startswith(
                ("FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL")
            )
        }
        if failed:
            unfinished = [
                state["jobs"][name]
                for name, value in observed.items()
                if value not in {"COMPLETED", "CANCELLED"}
            ]
            if unfinished:
                subprocess.run(
                    ["scancel", *unfinished],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            raise RuntimeError(f"SGLang Slurm stage failed: {failed}")
        if all(value == "COMPLETED" for value in observed.values()):
            return
        time.sleep(poll_seconds)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def official_report(result: dict[str, Any]) -> dict[str, Any] | None:
    report = result.get("report")
    if isinstance(report, dict):
        return report
    # The Enroot wrapper may expose the compact report one layer deeper.
    nested = result.get("result")
    if isinstance(nested, dict) and isinstance(nested.get("report"), dict):
        return nested["report"]
    return None


def validate_sglang_runs(
    campaign: Path, scope: str, tasks: int
) -> tuple[bool, str, dict[str, Any]]:
    run_root = campaign / "runs" / f"sglang_{scope}"
    rows: dict[str, Any] = {}
    first_hashes: list[str | None] = []
    for arm in ARMS:
        run_dir = run_root / arm / f"full_{tasks}"
        required = (
            run_dir / "RUNTIME_SUMMARY.json",
            run_dir / "CLIENT_LEDGER.jsonl",
            run_dir / "OFFICIAL_RESULT.json",
        )
        missing = [path.name for path in required if not path.is_file()]
        if missing:
            return False, f"missing completed artifacts {missing}: {run_dir}", rows
        runtime = read_json(required[0])
        client = load_jsonl(required[1])
        official = read_json(required[2])
        telemetry = read_json(run_dir / "TELEMETRY.json")
        requests = [row for row in client if row.get("event") == "request_complete"]
        if not requests or int(runtime.get("requests") or 0) <= 0:
            return False, f"no model requests: {run_dir}", rows
        first_hashes.append(requests[0].get("input_ids_sha256"))
        report = official_report(official)
        if report is None:
            return False, f"official report absent: {run_dir}", rows
        infrastructure_failures = {
            str(instance_id): str(value.get("exit_status"))
            for instance_id, value in (telemetry.get("instances") or {}).items()
            if str(value.get("exit_status")) in {"HTTPError", "ConnectionError"}
        }
        if infrastructure_failures:
            return False, (
                f"backend transport failures {infrastructure_failures}: {run_dir}"
            ), rows
        if arm != "dense" and int(runtime.get("target_copy_events") or 0) <= 0:
            return False, f"no physical coding-aware K/V copy: {run_dir}", rows
        rows[arm] = {
            "requests": int(runtime["requests"]),
            "target_copy_events": int(runtime.get("target_copy_events") or 0),
            "copied_tokens": int(runtime.get("copied_tokens") or 0),
            "target_fallback_events": int(runtime.get("target_fallback_events") or 0),
            "resolved": int(report.get("resolved_instances") or 0),
            "submitted": int(report.get("submitted_instances") or 0),
            "run_dir": str(run_dir),
        }
    if None in first_hashes or len(set(first_hashes)) != 1:
        return False, f"first-request prompt hashes differ: {first_hashes}", rows
    return (
        True,
        f"{scope} SGLang runs passed identity, official-evaluator, and physical-copy gates",
        rows,
    )


def submit_stage(
    state: dict[str, Any], *, project: Path, logs: Path, scope: str
) -> list[str]:
    dependency = None
    names = []
    for arm in ARMS:
        name = f"{scope}_sglang_{arm}"
        job_id = submit(
            script=project / "benchmark/multi_workflow/slurm/common_sglang_agent.sbatch",
            logs=logs,
            exports={"IMPACTKV_COMMON_ARM": arm, "IMPACTKV_COMMON_SCOPE": scope},
            dependency=dependency,
        )
        state["jobs"][name] = job_id
        names.append(name)
        dependency = job_id
    return names


def submit_exact(
    state: dict[str, Any], *, project: Path, logs: Path, label: str
) -> str:
    name = f"{label}_sglang_exact"
    job_id = submit(
        script=(
            project
            / "benchmark/multi_workflow/slurm/common_sglang_exact_prompt_replay.sbatch"
        ),
        logs=logs,
        exports={"IMPACTKV_COMMON_REPLAY_LABEL": label},
        dependency=None,
    )
    state["jobs"][name] = job_id
    state["active_jobs"] = [name]
    return name


def validate_exact(campaign: Path, label: str) -> dict[str, Any]:
    path = campaign / f"exact_prompt_replay/{label}/sglang_coding/RESULT.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    result = read_json(path)
    if result.get("status") != "PASS":
        raise RuntimeError(f"{label} SGLang exact replay failed: {result}")
    return {
        "targets": int((result.get("summary") or {}).get("targets") or 0),
        "median_cache_ready_speedup": float(
            result["summary"]["median_target_cache_ready_speedup"]
        ),
        "median_n1_including_build_speedup": float(
            result["summary"]["median_target_n1_including_build_speedup"]
        ),
        "median_n4_including_build_speedup": float(
            result["summary"]["median_target_n4_including_build_speedup"]
        ),
        "median_n16_including_build_speedup": float(
            result["summary"]["median_target_n16_including_build_speedup"]
        ),
        "physical_copy_events": int(result["summary"]["physical_copy_events"]),
        "result": str(path),
    }


def native_ready_for_canary(campaign: Path) -> tuple[bool, str]:
    path = campaign / NATIVE_STATUS
    if not path.is_file():
        return False, "native status absent"
    value = read_json(path)
    if value.get("state") == "blocked":
        raise RuntimeError(f"native campaign blocked: {value.get('error')}")
    gate = value.get("canary4_gate")
    return bool(gate), str(gate or f"native state={value.get('state')}")


def formal_images_ready(campaign: Path) -> tuple[bool, str]:
    value = read_json(campaign / NATIVE_STATUS)
    if value.get("state") == "blocked":
        raise RuntimeError(f"native campaign blocked: {value.get('error')}")
    job = (value.get("jobs") or {}).get("formal_images")
    if not job:
        return False, f"native state={value.get('state')}; formal image job absent"
    state = slurm_state(str(job))
    if state.startswith(("FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL")):
        raise RuntimeError(f"formal image job {job} failed: {state}")
    return state == "COMPLETED", f"formal image job {job}: {state}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    home = Path.home()
    project = home / "CodeMAS_Project/worktrees/sglang-common-agent"
    campaign = (
        home
        / "CodeMAS_Project/kvflow-artifacts/impactkv_common_agent_baselines_fresh24_20260812"
    )
    logs = home / "impactkv-runtime/logs/common-baselines"
    status_path = campaign / STATUS_NAME
    if status_path.is_file():
        state = read_json(status_path)
    else:
        state = {
            "schema_version": 1,
            "state": "waiting_native_canary4",
            "started_at_utc": utc_now(),
            "updated_at_utc": utc_now(),
            "status_path": str(status_path),
            "jobs": {},
            "node_policy": (
                "all measured GPU jobs pin gpu17 RTX4090; "
                "gpu10-13 and gpu23-24 excluded"
            ),
        }
        atomic_json(status_path, state)

    try:
        if state["state"] == "waiting_native_canary4":
            while True:
                ready, reason = native_ready_for_canary(campaign)
                state["wait_reason"] = reason
                state["updated_at_utc"] = utc_now()
                atomic_json(status_path, state)
                if ready:
                    break
                time.sleep(args.poll_seconds)
            names = submit_stage(state, project=project, logs=logs, scope="canary")
            state["state"] = "canary4_submitted"
            state["active_jobs"] = names
            atomic_json(status_path, state)

        if state["state"] == "canary4_submitted":
            names = list(state["active_jobs"])
            wait_jobs(state, status_path, names, args.poll_seconds)
            passed, reason, rows = validate_sglang_runs(campaign, "canary", 4)
            state["canary4"] = rows
            state["canary4_gate"] = reason
            if not passed:
                raise RuntimeError(reason)
            submit_exact(state, project=project, logs=logs, label="canary4")
            state["state"] = "canary4_exact_submitted"
            atomic_json(status_path, state)

        if state["state"] == "canary4_exact_submitted":
            names = list(state["active_jobs"])
            wait_jobs(state, status_path, names, args.poll_seconds)
            state["canary4_exact"] = validate_exact(campaign, "canary4")
            state["state"] = "waiting_formal_images"
            atomic_json(status_path, state)

        if state["state"] == "waiting_formal_images":
            while True:
                ready, reason = formal_images_ready(campaign)
                state["wait_reason"] = reason
                state["updated_at_utc"] = utc_now()
                atomic_json(status_path, state)
                if ready:
                    break
                time.sleep(args.poll_seconds)
            names = submit_stage(state, project=project, logs=logs, scope="formal")
            state["state"] = "fresh24_submitted"
            state["active_jobs"] = names
            atomic_json(status_path, state)

        if state["state"] == "fresh24_submitted":
            names = list(state["active_jobs"])
            wait_jobs(state, status_path, names, args.poll_seconds)
            passed, reason, rows = validate_sglang_runs(campaign, "formal", 24)
            state["fresh24"] = rows
            state["fresh24_gate"] = reason
            if not passed:
                raise RuntimeError(reason)
            submit_exact(state, project=project, logs=logs, label="fresh24")
            state["state"] = "fresh24_exact_submitted"
            atomic_json(status_path, state)

        if state["state"] == "fresh24_exact_submitted":
            names = list(state["active_jobs"])
            wait_jobs(state, status_path, names, args.poll_seconds)
            state["fresh24_exact"] = validate_exact(campaign, "fresh24")
            state["state"] = "complete"
            state["finished_at_utc"] = utc_now()
            atomic_json(status_path, state)
    except Exception as error:
        state["state"] = "blocked"
        state["error"] = f"{type(error).__name__}: {error}"
        state["updated_at_utc"] = utc_now()
        atomic_json(status_path, state)
        raise


if __name__ == "__main__":
    main()
