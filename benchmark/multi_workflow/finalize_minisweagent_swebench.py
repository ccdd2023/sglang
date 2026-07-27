#!/usr/bin/env python3
"""Wait for a mini-SWE-agent batch, normalize it, and run official evaluation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from prepare_minisweagent_swebench import normalize_predictions, read_json, write_json


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def process_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-pid", type=int, required=True)
    parser.add_argument("--batch-output", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--docker-host", required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--poll-seconds", type=int, default=10)
    args = parser.parse_args()

    status_path = args.batch_output / "PIPELINE_STATUS.json"
    status: dict[str, Any] = {
        "state": "waiting_for_agent",
        "started_at_utc": utc_now(),
        "batch_pid": args.batch_pid,
    }
    write_json(status_path, status)
    while process_is_alive(args.batch_pid):
        time.sleep(args.poll_seconds)

    status["agent_finished_at_utc"] = utc_now()
    preds_path = args.batch_output / "preds.json"
    if not preds_path.exists():
        status.update(state="agent_failed", reason="preds.json is absent")
        write_json(status_path, status)
        return 2

    predictions_jsonl = args.batch_output / "predictions.jsonl"
    telemetry_path = args.batch_output / "TELEMETRY.json"
    try:
        normalize_predictions(
            args.batch_output,
            args.registration,
            predictions_jsonl,
            telemetry_path,
            args.model_label,
            allow_partial=False,
        )
    except Exception as exc:
        status.update(
            state="normalization_failed",
            reason=f"{type(exc).__name__}: {exc}",
        )
        write_json(status_path, status)
        return 3

    registration = read_json(args.registration)
    instance_ids = [row["instance_id"] for row in registration["instances"]]
    report_dir = args.batch_output / "reports"
    command = [
        str(args.python),
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        str(args.snapshot.resolve()),
        "--split",
        registration["dataset"]["split"],
        "--predictions_path",
        str(predictions_jsonl.resolve()),
        "--instance_ids",
        *instance_ids,
        "--max_workers",
        str(args.workers),
        "--timeout",
        str(args.timeout),
        "--cache_level",
        "instance",
        "--clean",
        "true",
        "--run_id",
        args.run_id,
        "--namespace",
        "swebench",
        "--report_dir",
        str(report_dir.resolve()),
    ]
    write_json(args.batch_output / "official_evaluation_command.json", command)
    status.update(state="official_evaluation", evaluation_started_at_utc=utc_now())
    write_json(status_path, status)
    env = dict(os.environ)
    env["DOCKER_HOST"] = args.docker_host
    with (args.batch_output / "official_evaluation.stdout.log").open(
        "w", encoding="utf-8"
    ) as log:
        result = subprocess.run(
            command,
            cwd=args.batch_output,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    status.update(
        state="complete" if result.returncode == 0 else "evaluation_failed",
        evaluation_returncode=result.returncode,
        finished_at_utc=utc_now(),
    )
    write_json(status_path, status)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
