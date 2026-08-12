#!/usr/bin/env python3
"""Register a four-task counterfactual for repeated tool-format failures."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark.multi_workflow.runtime_paths import RuntimePaths


PROJECT = Path(__file__).resolve().parents[2]
PATHS = RuntimePaths.from_project(PROJECT)
SOURCE = PATHS.artifacts / "impactkv_common_agent_baselines_fresh24_20260812"
DEFAULT_OUTPUT = PATHS.artifacts / "impactkv_common_agent_format_guard_20260812"
TASKS = 4
MIN_CONTROL_FORMAT_ERRORS = 23


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def trajectory_row(path: Path) -> dict[str, Any]:
    value = read_json(path)
    info = value.get("info") or {}
    messages = value.get("messages") or []
    return {
        "instance_id": str(value.get("instance_id") or path.parent.name),
        "repo": str(value.get("instance_id") or path.parent.name).split("__", 1)[0],
        "exit_status": info.get("exit_status"),
        "api_calls": int((info.get("model_stats") or {}).get("api_calls") or 0),
        "format_errors": sum(
            message.get("role") == "user"
            and (message.get("extra") or {}).get("interrupt_type") == "FormatError"
            for message in messages
        ),
        "executed_tool_calls": sum(message.get("role") == "tool" for message in messages),
        "submission_nonempty": bool(str(info.get("submission") or "").strip()),
        "trajectory_sha256": sha256(path),
    }


def select(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    eligible = [
        row
        for row in rows
        if row["exit_status"] == "LimitsExceeded"
        and row["api_calls"] == 32
        and row["format_errors"] >= MIN_CONTROL_FORMAT_ERRORS
    ]
    by_repo: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        by_repo[row["repo"]].append(row)
    representatives = [
        sorted(values, key=lambda row: (-row["format_errors"], row["instance_id"]))[0]
        for values in by_repo.values()
    ]
    selected = sorted(
        representatives,
        key=lambda row: (-row["format_errors"], row["repo"], row["instance_id"]),
    )[:TASKS]
    if len(selected) != TASKS:
        raise RuntimeError(f"only {len(selected)} repositories meet the format-loop gate")
    return selected


def prepare(output: Path, source: Path = SOURCE) -> dict[str, Any]:
    path = output / "FORMAT_GUARD_REGISTRATION.json"
    if path.exists():
        return read_json(path)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    dense = source / "runs/formal/cacheblend_dense/all"
    rows = [trajectory_row(value) for value in sorted(dense.glob("*/*.traj.json"))]
    if len(rows) != 24:
        raise RuntimeError(f"expected 24 control trajectories, found {len(rows)}")
    total_calls = sum(row["api_calls"] for row in rows)
    total_errors = sum(row["format_errors"] for row in rows)
    total_tools = sum(row["executed_tool_calls"] for row in rows)
    selected = select(rows)

    source_rows = [
        json.loads(line)
        for line in (source / "formal_dataset/test.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_id = {str(row["instance_id"]): row for row in source_rows}
    selected_ids = [row["instance_id"] for row in selected]
    frozen = [by_id[value] for value in selected_ids]
    output.mkdir(parents=True)
    write_json(output / "FROZEN_FORMAT_GUARD4.json", frozen)
    write_jsonl(output / "format_guard4_dataset/test.jsonl", frozen)
    write_json(
        output / "BRIDGE_FORMAT_GUARD4_REGISTRATION.json",
        {
            "schema_version": 1,
            "registration_id": "impactkv-common-agent-format-guard4-20260812",
            "dataset": {"name": "princeton-nlp/SWE-bench_Verified", "split": "test"},
            "instances": [{"instance_id": value} for value in selected_ids],
        },
    )
    value = {
        "schema_version": 1,
        "status": "REGISTERED_BEFORE_FORMAT_GUARD_MODEL_REQUESTS",
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "motivation": {
            "control_tasks": 24,
            "control_model_calls": total_calls,
            "control_format_errors": total_errors,
            "control_executed_tool_calls": total_tools,
            "format_error_fraction": total_errors / total_calls,
            "call_limited_tasks": sum(row["exit_status"] == "LimitsExceeded" for row in rows),
            "call_limited_tasks_with_format_errors": sum(
                row["exit_status"] == "LimitsExceeded" and row["format_errors"] > 0
                for row in rows
            ),
            "falsifiable_hypothesis": (
                "At the same 32-call budget, retaining an otherwise unparseable "
                "assistant turn as a non-mutating notice tool interaction will break "
                "the repeated FormatError loop and increase executed tool calls."
            ),
        },
        "intervention": {
            "changed_field": "model.recover_unparsed_output_with_notice",
            "control": False,
            "treatment": True,
            "step_limit": 32,
            "notice_is_task_specific": False,
            "notice_mutates_repository": False,
            "notice_invents_model_command": False,
            "unchanged": [
                "Qwen2.5-Coder-7B-Instruct and BF16 backend",
                "rolling-6 prompt and task messages",
                "temperature, token limits, and one-tool schema",
                "official SWE-bench evaluator",
            ],
        },
        "selection": {
            "classification": "post-run mechanism canary; not a method-ranking cohort",
            "rule": (
                "one highest-format-error 32-call limit task per repository; "
                "at least 23 control FormatErrors; top four repository representatives"
            ),
            "selected": selected,
            "reuse_outcomes_used": False,
        },
        "gates": {
            "official_total_instances": TASKS,
            "official_error_instances": 0,
            "format_errors_at_most_fraction_of_control": 0.5,
            "executed_tool_calls_strictly_greater_than_control": True,
            "task_signal": "at least one applied nonempty patch or one resolved task",
            "keep": (
                "Keep the transport guard only if the mechanism gate passes; task "
                "signal determines whether it is sufficient for an accuracy rerun."
            ),
        },
        "inputs": {
            "source_campaign": str(source),
            "source_dataset_sha256": sha256(source / "formal_dataset/test.jsonl"),
            "frozen_guard4_sha256": sha256(output / "FROZEN_FORMAT_GUARD4.json"),
        },
        "protected": {
            "prefetch": False,
            "old_dirty_checkout_modified": False,
            "paper_modified": False,
            "old_preregistration_thresholds_modified": False,
        },
    }
    write_json(path, value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output, args.source), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
