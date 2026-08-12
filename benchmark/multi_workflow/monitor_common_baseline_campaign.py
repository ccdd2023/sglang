#!/usr/bin/env python3
"""Monitor and advance the registered CacheBlend/KVCOMM comparison campaign.

This login-node process only submits bounded Slurm jobs.  It first runs one
historical task, checks prompt identity plus physical K/V reuse, expands to the
four-task canary, and only then submits the frozen Fresh24 cohort.  Any failed
job or failed measurement gate stops the campaign without deleting artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ARMS = (
    ("cacheblend", "dense"),
    ("cacheblend", "reuse"),
    ("kvcomm", "dense"),
    ("kvcomm", "reuse"),
)
CANARY_INSTANCE = "django__django-16631"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run(command: list[str], *, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )
    return result.stdout.strip()


def environment_ready(home: Path) -> tuple[bool, str]:
    model = home / "models/Qwen2.5-Coder-7B-Instruct"
    hashes = model / "MODEL_SHA256SUMS"
    expected = {
        "0b6f069918b07c064cbba8ae4f00f529aa9bbf84b7cdfcb7fc2694a40f6aa8ef",
        "c3d46733e7aa054ea7b063fbccd0a5a08446e7bd1814bef26936c5aa1331da62",
        "9fe45dacee087385b3d2d6dd27a7413a8a56d95f145772facc148fa86fc73446",
        "5aa6e5cbe642377fd441fb4e60e83cca96b2bcd9820e245b9ea06d94653f17f2",
    }
    if not hashes.is_file():
        return False, "model hashes are still being generated"
    observed = {
        line.split()[0]
        for line in hashes.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if not expected.issubset(observed):
        return False, "model shard hashes do not match the frozen snapshot"
    checks = [
        (
            home / ".venvs/cacheblend-native/bin/python",
            home / "CodeMAS_Project/worktrees/cacheblend-common-agent/vllm_blend",
            "import torch, transformers, vllm",
        ),
        (
            home / ".venvs/kvcomm-native/bin/python",
            home / "CodeMAS_Project/worktrees/kvcomm-common-agent",
            "import torch, transformers; import KVCOMM",
        ),
    ]
    for python, pythonpath, statement in checks:
        if not python.is_file():
            return False, f"environment incomplete: {python}"
        env = os.environ.copy()
        env.update(PYTHONNOUSERSITE="1", PYTHONPATH=str(pythonpath))
        result = subprocess.run(
            [str(python), "-c", statement],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
        )
        if result.returncode:
            return False, f"environment import pending: {result.stdout[-400:]}"
    return True, "native environments and model hashes ready"


def images_ready(home: Path, instances: list[str]) -> bool:
    index_path = home / "impactkv-runtime/enroot/images/IMAGE_INDEX.json"
    if not index_path.is_file():
        return False
    index = json.loads(index_path.read_text(encoding="utf-8"))
    available: set[str] = set()
    for record in index.get("images", {}).values():
        if Path(record.get("sqsh_path", "")).is_file():
            available.update(record.get("instance_ids") or [])
    return set(instances).issubset(available)


def slurm_state(job_id: str) -> str:
    queued = subprocess.run(
        ["squeue", "-h", "-j", job_id, "-o", "%T"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    ).stdout.strip()
    if queued:
        return queued.splitlines()[0]
    accounting = subprocess.run(
        ["sacct", "-n", "-X", "-j", job_id, "-o", "State"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    ).stdout.strip()
    return accounting.splitlines()[0].split()[0] if accounting else "UNKNOWN"


def submit(
    *,
    script: Path,
    logs: Path,
    exports: dict[str, str],
    dependency: str | None = None,
) -> str:
    export_arg = "ALL," + ",".join(f"{key}={value}" for key, value in exports.items())
    command = [
        "sbatch",
        "--parsable",
        f"--chdir={logs}",
        f"--export={export_arg}",
    ]
    if dependency:
        command.append(f"--dependency=afterok:{dependency}")
    command.append(str(script))
    return run(command).split(";")[0]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_native_runs(
    campaign: Path,
    scope: str,
    instance: str | None,
) -> tuple[bool, str]:
    key = instance or "all"
    ledgers: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for backend, mode in ARMS:
        run_dir = campaign / "runs" / scope / f"{backend}_{mode}" / key
        summary_path = run_dir / "RUNTIME_SUMMARY.json"
        ledger_path = run_dir / "CLIENT_LEDGER.jsonl"
        if not summary_path.is_file() or not ledger_path.is_file():
            return False, f"missing completed artifacts: {run_dir}"
        summary = read_json(summary_path)
        if summary.get("requests", 0) <= 0:
            return False, f"no model requests: {run_dir}"
        if summary.get("input_identity_rows") != summary.get("requests"):
            return False, f"input identity incomplete: {run_dir}"
        if mode == "reuse" and summary.get("physical_reuse_requests", 0) <= 0:
            return False, f"no physical K/V reuse: {run_dir}"
        ledgers[(backend, mode)] = [
            json.loads(line)
            for line in ledger_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    first_hashes = []
    for rows in ledgers.values():
        request = next(row for row in rows if row.get("event") == "request_complete")
        first_hashes.append(request.get("input_ids_sha256"))
    if len(set(first_hashes)) != 1:
        return False, f"first-request prompt hashes differ: {first_hashes}"
    return True, f"{scope} native runs passed identity and physical-reuse gates"


def wait_jobs(state: dict[str, Any], names: list[str], poll_seconds: int) -> None:
    while True:
        states = {name: slurm_state(state["jobs"][name]) for name in names}
        state["slurm_states"] = states
        state["updated_at_utc"] = utc_now()
        atomic_json(Path(state["status_path"]), state)
        failed = {
            name: value
            for name, value in states.items()
            if value.startswith(("FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"))
        }
        if failed:
            raise RuntimeError(f"Slurm stage failed: {failed}")
        if all(value == "COMPLETED" for value in states.values()):
            return
        time.sleep(poll_seconds)


def submit_native_stage(
    state: dict[str, Any],
    *,
    project: Path,
    logs: Path,
    scope: str,
    instance: str | None,
) -> list[str]:
    names = []
    dependency = None
    for backend, mode in ARMS:
        name = f"{scope}_{backend}_{mode}_{instance or 'all'}"
        exports = {
            "IMPACTKV_COMMON_BACKEND": backend,
            "IMPACTKV_COMMON_MODE": mode,
            "IMPACTKV_COMMON_SCOPE": scope,
        }
        if instance:
            exports["IMPACTKV_COMMON_INSTANCE"] = instance
        job_id = submit(
            script=project / "benchmark/multi_workflow/slurm/common_native_agent.sbatch",
            logs=logs,
            exports=exports,
            dependency=dependency,
        )
        state["jobs"][name] = job_id
        names.append(name)
        dependency = job_id
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--canary-image-job", default="73785")
    args = parser.parse_args()
    home = Path.home()
    project = home / "CodeMAS_Project/worktrees/sglang-common-agent"
    campaign = (
        home
        / "CodeMAS_Project/kvflow-artifacts/impactkv_common_agent_baselines_fresh24_20260812"
    )
    logs = home / "impactkv-runtime/logs/common-baselines"
    status_path = campaign / "AUTOMATED_CAMPAIGN_STATUS.json"
    state: dict[str, Any] = {
        "schema_version": 1,
        "state": "waiting_preconditions",
        "started_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "status_path": str(status_path),
        "jobs": {"canary_images": args.canary_image_job},
        "node_policy": "all GPU jobs pin gpu18 RTX4090; gpu10-13 and gpu23-24 excluded",
    }
    atomic_json(status_path, state)
    try:
        canary_ids = [
            row["instance_id"] for row in read_json(campaign / "CANARY4.json")
        ]
        while True:
            ready, reason = environment_ready(home)
            image_state = slurm_state(args.canary_image_job)
            state.update(
                state="waiting_preconditions",
                precondition_reason=reason,
                canary_image_state=image_state,
                updated_at_utc=utc_now(),
            )
            atomic_json(status_path, state)
            if image_state.startswith(("FAILED", "CANCELLED", "TIMEOUT")):
                raise RuntimeError(f"canary image job failed: {image_state}")
            if ready and image_state == "COMPLETED" and images_ready(home, canary_ids):
                break
            time.sleep(args.poll_seconds)

        state["state"] = "one_task_canary_submitted"
        names = submit_native_stage(
            state,
            project=project,
            logs=logs,
            scope="canary",
            instance=CANARY_INSTANCE,
        )
        atomic_json(status_path, state)
        wait_jobs(state, names, args.poll_seconds)
        passed, reason = validate_native_runs(campaign, "canary", CANARY_INSTANCE)
        if not passed:
            raise RuntimeError(reason)
        state["one_task_gate"] = reason

        source_ledger = (
            campaign
            / "runs/canary/cacheblend_dense"
            / CANARY_INSTANCE
            / "BACKEND_LEDGER.jsonl"
        )
        dependency = state["jobs"][names[-1]]
        replay_names = []
        for backend in ("cacheblend", "kvcomm"):
            name = f"one_task_exact_{backend}"
            job_id = submit(
                script=project
                / "benchmark/multi_workflow/slurm/common_exact_prompt_replay.sbatch",
                logs=logs,
                exports={
                    "IMPACTKV_COMMON_BACKEND": backend,
                    "IMPACTKV_COMMON_SOURCE_LEDGER": str(source_ledger),
                    "IMPACTKV_COMMON_REPLAY_LABEL": "one_task_canary",
                    "IMPACTKV_COMMON_REPLAY_LIMIT": "4",
                },
                dependency=dependency,
            )
            state["jobs"][name] = job_id
            replay_names.append(name)
            dependency = job_id
        state["state"] = "one_task_exact_submitted"
        atomic_json(status_path, state)
        wait_jobs(state, replay_names, args.poll_seconds)
        for backend in ("cacheblend", "kvcomm"):
            result = read_json(
                campaign / f"exact_prompt_replay/one_task_canary/{backend}/RESULT.json"
            )
            if result.get("status") != "PASS":
                raise RuntimeError(f"exact replay failed for {backend}: {result}")

        state["state"] = "canary4_submitted"
        names = submit_native_stage(
            state,
            project=project,
            logs=logs,
            scope="canary",
            instance=None,
        )
        atomic_json(status_path, state)
        wait_jobs(state, names, args.poll_seconds)
        passed, reason = validate_native_runs(campaign, "canary", None)
        if not passed:
            raise RuntimeError(reason)
        state["canary4_gate"] = reason

        formal_image_name = "formal_images"
        formal_image_job = submit(
            script=project / "benchmark/multi_workflow/slurm/common_agent_prepare_images.sbatch",
            logs=logs,
            exports={"IMPACTKV_COMMON_SCOPE": "formal"},
            dependency=state["jobs"][names[-1]],
        )
        state["jobs"][formal_image_name] = formal_image_job
        state["state"] = "formal_images_submitted"
        atomic_json(status_path, state)
        wait_jobs(state, [formal_image_name], args.poll_seconds)

        state["state"] = "fresh24_submitted"
        names = submit_native_stage(
            state,
            project=project,
            logs=logs,
            scope="formal",
            instance=None,
        )
        atomic_json(status_path, state)
        wait_jobs(state, names, args.poll_seconds)
        passed, reason = validate_native_runs(campaign, "formal", None)
        if not passed:
            raise RuntimeError(reason)
        state["fresh24_gate"] = reason
        source_ledger = (
            campaign
            / "runs/formal/cacheblend_dense/all/BACKEND_LEDGER.jsonl"
        )
        dependency = state["jobs"][names[-1]]
        replay_names = []
        for backend in ("cacheblend", "kvcomm"):
            name = f"fresh24_exact_{backend}"
            job_id = submit(
                script=project
                / "benchmark/multi_workflow/slurm/common_exact_prompt_replay.sbatch",
                logs=logs,
                exports={
                    "IMPACTKV_COMMON_BACKEND": backend,
                    "IMPACTKV_COMMON_SOURCE_LEDGER": str(source_ledger),
                    "IMPACTKV_COMMON_REPLAY_LABEL": "fresh24",
                    "IMPACTKV_COMMON_REPLAY_LIMIT": "16",
                },
                dependency=dependency,
            )
            state["jobs"][name] = job_id
            replay_names.append(name)
            dependency = job_id
        state["state"] = "fresh24_exact_submitted"
        atomic_json(status_path, state)
        wait_jobs(state, replay_names, args.poll_seconds)
        for backend in ("cacheblend", "kvcomm"):
            result = read_json(
                campaign / f"exact_prompt_replay/fresh24/{backend}/RESULT.json"
            )
            if result.get("status") != "PASS":
                raise RuntimeError(f"formal exact replay failed for {backend}: {result}")
        run(
            [
                sys.executable,
                str(project / "benchmark/multi_workflow/summarize_common_baseline_campaign.py"),
                "--campaign",
                str(campaign),
            ]
        )
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
