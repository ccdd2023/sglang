#!/usr/bin/env python3
"""Prepare frozen SWE-bench data and normalize mini-SWE-agent batch outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def prepare_dataset(snapshot: Path, registration: Path, output: Path) -> None:
    payload = snapshot.read_bytes()
    rows = json.loads(payload)
    spec = read_json(registration)
    expected_ids = [row["instance_id"] for row in spec["instances"]]
    actual_ids = [row["instance_id"] for row in rows]
    expected_sha = spec["dataset"]["local_snapshot_sha256"]
    actual_sha = hashlib.sha256(payload).hexdigest()
    if actual_sha != expected_sha:
        raise ValueError(f"snapshot SHA mismatch: {actual_sha} != {expected_sha}")
    if actual_ids != expected_ids:
        raise ValueError("snapshot order/IDs differ from the frozen registration")

    output.mkdir(parents=True, exist_ok=True)
    data_path = output / "test.jsonl"
    data_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    write_json(
        output / "DATASET_MANIFEST.json",
        {
            "registration_id": spec["registration_id"],
            "source_snapshot": str(snapshot.resolve()),
            "source_snapshot_sha256": actual_sha,
            "instance_count": len(rows),
            "instance_ids": actual_ids,
            "datasets_loader_split": "test",
            "data_file": str(data_path.resolve()),
        },
    )
    print(f"Prepared {len(rows)} frozen instances at {data_path}")


def request_telemetry(trajectory: dict[str, Any]) -> dict[str, Any]:
    calls = []
    for message in trajectory.get("messages", []):
        extra = message.get("extra") or {}
        response = extra.get("response") or {}
        if not response:
            continue
        usage = response.get("usage") or {}
        calls.append(
            {
                "timestamp": extra.get("timestamp"),
                "request_latency_seconds": extra.get("request_latency_seconds"),
                "context_compaction": extra.get("context_compaction"),
                "tool_call_limit": extra.get("tool_call_limit"),
                "model": response.get("model"),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "finish_reason": (response.get("choices") or [{}])[0].get(
                    "finish_reason"
                ),
            }
        )
    return {
        "api_calls": trajectory.get("info", {})
        .get("model_stats", {})
        .get("api_calls", len(calls)),
        "calls": calls,
    }


def normalize_predictions(
    batch_output: Path,
    registration: Path,
    output_jsonl: Path,
    telemetry_output: Path,
    model_label: str,
    allow_partial: bool,
) -> None:
    spec = read_json(registration)
    expected_ids = [row["instance_id"] for row in spec["instances"]]
    raw = read_json(batch_output / "preds.json")
    actual_ids = set(raw)
    missing = [instance_id for instance_id in expected_ids if instance_id not in actual_ids]
    extra = sorted(actual_ids - set(expected_ids))
    if extra or (missing and not allow_partial):
        raise ValueError(f"prediction IDs mismatch: missing={missing}, extra={extra}")

    rows = []
    telemetry: dict[str, Any] = {
        "registration_id": spec["registration_id"],
        "agent": "mini-swe-agent",
        "agent_version": "2.3.0",
        "model_label": model_label,
        "complete": not missing,
        "missing_instance_ids": missing,
        "instances": {},
    }
    for instance_id in expected_ids:
        if instance_id not in raw:
            continue
        model_patch = raw[instance_id].get("model_patch") or ""
        rows.append(
            {
                "instance_id": instance_id,
                "model_name_or_path": model_label,
                "model_patch": model_patch,
            }
        )
        trajectory_path = (
            batch_output / instance_id / f"{instance_id}.traj.json"
        )
        if trajectory_path.exists():
            trajectory = read_json(trajectory_path)
            telemetry["instances"][instance_id] = {
                "exit_status": trajectory.get("info", {}).get("exit_status"),
                "patch_characters": len(model_patch),
                **request_telemetry(trajectory),
            }
        else:
            telemetry["instances"][instance_id] = {
                "exit_status": "trajectory_missing",
                "patch_characters": len(model_patch),
                "api_calls": None,
                "calls": [],
            }

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    write_json(telemetry_output, telemetry)
    print(
        f"Normalized {len(rows)}/{len(expected_ids)} predictions; "
        f"missing={len(missing)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    dataset = subparsers.add_parser("dataset")
    dataset.add_argument("--snapshot", type=Path, required=True)
    dataset.add_argument("--registration", type=Path, required=True)
    dataset.add_argument("--output", type=Path, required=True)

    normalize = subparsers.add_parser("normalize")
    normalize.add_argument("--batch-output", type=Path, required=True)
    normalize.add_argument("--registration", type=Path, required=True)
    normalize.add_argument("--output-jsonl", type=Path, required=True)
    normalize.add_argument("--telemetry-output", type=Path, required=True)
    normalize.add_argument("--model-label", required=True)
    normalize.add_argument("--allow-partial", action="store_true")

    args = parser.parse_args()
    if args.command == "dataset":
        prepare_dataset(args.snapshot, args.registration, args.output)
    else:
        normalize_predictions(
            args.batch_output,
            args.registration,
            args.output_jsonl,
            args.telemetry_output,
            args.model_label,
            args.allow_partial,
        )


if __name__ == "__main__":
    main()
