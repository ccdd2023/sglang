#!/usr/bin/env python3
"""Run a frozen SWE-bench Verified subset with the official Docker harness."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_REGISTRATION = HERE / "swebench_verified_complex_v1.json"
DEFAULT_OUTPUT = Path(
    "/home/gfy/CodeMAS_Project/kvflow-artifacts/"
    "swebench_verified_complex_v1_20260724"
)
DEFAULT_PYTHON = Path("/home/gfy/.conda/envs/sglang-kvflow/bin/python")


def run(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def load_registration(path: Path) -> dict[str, Any]:
    registration = json.loads(path.read_text(encoding="utf-8"))
    if not registration["selection"]["frozen_before_container_outcomes"]:
        raise ValueError("the instance list must be frozen before container outcomes")
    if registration["scope"]["prefetch_allowed"]:
        raise ValueError("this project forbids prefetch in the registered treatment")
    return registration


def docker_endpoint() -> str:
    explicit = os.environ.get("DOCKER_HOST")
    if explicit:
        return explicit
    proc = run(
        [
            "docker",
            "context",
            "inspect",
            "--format",
            "{{.Endpoints.docker.Host}}",
        ]
    )
    if proc.returncode == 0 and proc.stdout.strip():
        return proc.stdout.strip()
    return "unix:///var/run/docker.sock"


def require_docker() -> None:
    endpoint = docker_endpoint()
    env = dict(os.environ)
    env["DOCKER_HOST"] = endpoint
    proc = run(
        ["docker", "info", "--format", "{{.ServerVersion}}"],
        env=env,
    )
    if proc.returncode == 0:
        print(f"Docker server {proc.stdout.strip()} via {endpoint}")
        return
    detail = (proc.stderr or proc.stdout).strip()
    raise RuntimeError(
        "Docker daemon is unavailable to this user. "
        f"Endpoint: {docker_endpoint()}\n{detail}\n"
        "For this host, install uidmap once as an administrator, then run "
        "setup_rootless_docker_for_swebench.sh."
    )


def write_base_probe_predictions(path: Path, instance_ids: list[str]) -> None:
    marker_patch = """diff --git a/impactkv_base_probe.txt b/impactkv_base_probe.txt
new file mode 100644
index 0000000..41fe2a3
--- /dev/null
+++ b/impactkv_base_probe.txt
@@ -0,0 +1 @@
+ImpactKV base-oracle probe: intentionally no source-code change.
"""
    rows = [
        {
            "instance_id": instance_id,
            "model_name_or_path": "impactkv/base-nonsemantic-probe",
            "model_patch": marker_patch,
        }
        for instance_id in instance_ids
    ]
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def validate_prediction_ids(path: Path, expected: set[str]) -> None:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    actual = {row["instance_id"] for row in rows}
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ValueError(f"prediction IDs mismatch: missing={missing}, extra={extra}")
    for row in rows:
        for key in ("instance_id", "model_name_or_path", "model_patch"):
            if key not in row:
                raise ValueError(f"prediction row lacks {key}: {row}")


def validate_dataset_snapshot(
    path: Path,
    expected_sha256: str,
    expected_ids: set[str],
) -> None:
    payload = path.read_bytes()
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"dataset snapshot hash mismatch: {actual_sha256} != {expected_sha256}"
        )
    rows = json.loads(payload)
    actual_ids = {row["instance_id"] for row in rows}
    if actual_ids != expected_ids or len(rows) != len(expected_ids):
        raise ValueError("dataset snapshot IDs do not match the registration")


def harness_command(
    *,
    python: Path,
    dataset: str,
    split: str,
    predictions: str,
    instance_ids: list[str],
    run_id: str,
    workers: int,
    timeout: int,
    cache_level: str,
    output: Path,
) -> list[str]:
    return [
        str(python),
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        dataset,
        "--split",
        split,
        "--predictions_path",
        predictions,
        "--instance_ids",
        *instance_ids,
        "--max_workers",
        str(workers),
        "--timeout",
        str(timeout),
        "--cache_level",
        cache_level,
        "--clean",
        "true",
        "--run_id",
        run_id,
        "--namespace",
        "swebench",
        "--report_dir",
        str(output / "reports"),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("check", "pull", "base", "gold", "candidate"),
        help="check/pull images, or execute one official harness treatment",
    )
    parser.add_argument("--registration", type=Path, default=DEFAULT_REGISTRATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument(
        "--run-label",
        help="Optional candidate-specific suffix for isolated logs and reports.",
    )
    parser.add_argument(
        "--dataset-snapshot",
        type=Path,
        help="Override the registered frozen local dataset JSON.",
    )
    parser.add_argument(
        "--oracle-result",
        type=Path,
        default=DEFAULT_OUTPUT / "ORACLE_RESULT.json",
        help="Frozen Base-fail/Gold-pass denominator used in candidate mode.",
    )
    parser.add_argument(
        "--cache-level",
        choices=("none", "base", "env", "instance"),
        default="instance",
    )
    args = parser.parse_args()

    registration = load_registration(args.registration)
    instances = registration["instances"]
    instance_ids = [row["instance_id"] for row in instances]
    if len(instance_ids) != len(set(instance_ids)):
        raise ValueError("registration contains duplicate instance IDs")
    if len(instance_ids) != registration["selection"]["instance_count"]:
        raise ValueError("registered instance count does not match the instance list")
    dataset = registration["dataset"]
    args.output.mkdir(parents=True, exist_ok=True)
    snapshot = args.dataset_snapshot or Path(dataset["local_snapshot"])
    if not snapshot.exists():
        raise FileNotFoundError(
            f"frozen dataset snapshot is absent: {snapshot}; run "
            "freeze_swebench_verified_subset.py"
        )
    validate_dataset_snapshot(
        snapshot,
        dataset["local_snapshot_sha256"],
        set(instance_ids),
    )
    dataset_source = str(snapshot.resolve())

    try:
        require_docker()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.mode == "check":
        missing = []
        for row in instances:
            proc = run(["docker", "manifest", "inspect", row["image"]])
            state = "FOUND" if proc.returncode == 0 else "MISSING"
            print(f"{state}\t{row['instance_id']}\t{row['image']}")
            if proc.returncode != 0:
                missing.append(row["instance_id"])
        return 1 if missing else 0
    if args.mode == "pull":
        docker_env = dict(os.environ)
        docker_env["DOCKER_HOST"] = docker_endpoint()
        states = []
        failed = False
        for index, row in enumerate(instances, start=1):
            print(
                f"[{index}/{len(instances)}] pulling "
                f"{row['instance_id']}: {row['image']}",
                flush=True,
            )
            returncode = subprocess.call(
                ["docker", "pull", "--quiet", row["image"]],
                env=docker_env,
            )
            states.append(
                {
                    "instance_id": row["instance_id"],
                    "image": row["image"],
                    "returncode": returncode,
                }
            )
            (args.output / "image_pull_status.json").write_text(
                json.dumps(states, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            if returncode != 0:
                failed = True
        return 1 if failed else 0

    if args.mode == "candidate":
        if args.predictions is None:
            parser.error("--predictions is required in candidate mode")
        oracle = json.loads(args.oracle_result.read_text(encoding="utf-8"))
        instance_ids = list(oracle["oracle_valid_instance_ids"])
        validate_prediction_ids(args.predictions, set(instance_ids))
        predictions = str(args.predictions.resolve())
    elif args.mode == "base":
        probe_path = args.output / "base_probe_predictions.jsonl"
        write_base_probe_predictions(probe_path, instance_ids)
        predictions = str(probe_path)
    else:
        predictions = "gold"

    run_suffix = (
        args.run_label
        if args.mode == "candidate" and args.run_label
        else ("base-probe" if args.mode == "base" else args.mode)
    )
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_suffix):
        raise ValueError(f"unsafe run label: {run_suffix}")
    run_id = f"{registration['registration_id']}-{run_suffix}"
    command = harness_command(
        python=args.python,
        dataset=dataset_source,
        split=dataset["split"],
        predictions=predictions,
        instance_ids=instance_ids,
        run_id=run_id,
        workers=args.workers,
        timeout=args.timeout,
        cache_level=args.cache_level,
        output=args.output,
    )
    command_path = args.output / f"{args.mode}_command.json"
    command_path.write_text(
        json.dumps(command, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("Executing:", " ".join(command))
    harness_env = dict(os.environ)
    harness_env["DOCKER_HOST"] = docker_endpoint()
    return subprocess.call(command, cwd=args.output, env=harness_env)


if __name__ == "__main__":
    raise SystemExit(main())
