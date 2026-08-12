#!/usr/bin/env python3
"""Attribute the graph-mean canary's physical copy against Dense and LCB.

This is a post-run descriptive audit.  It does not select tasks or tune the
online policy.  For every request physically exposed to the graph-mean arm,
it verifies the pre-generation input token identity and compares the emitted
bash action plus the subsequent request/action trajectory with the already
frozen Dense and LCB runs of the same task.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark.multi_workflow.runtime_paths import RuntimePaths


PROJECT = Path(__file__).resolve().parents[2]
ARTIFACTS = RuntimePaths.from_project(PROJECT).artifacts
BASE = ARTIFACTS / "impactkv_common_agent_baselines_fresh24_20260812"
MEAN = ARTIFACTS / "impactkv_common_agent_graph_mean_20260812"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def command(message: dict[str, Any]) -> str | None:
    calls = message.get("tool_calls") or ()
    if not calls:
        return None
    function = calls[0].get("function") or {}
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return arguments
    return str(arguments.get("command")) if isinstance(arguments, dict) else None


def requests(path: Path) -> list[dict[str, Any]]:
    trajectory = read_json(path)
    rows = []
    for message in trajectory.get("messages") or ():
        treatment = (message.get("extra") or {}).get("reuse_treatment") or {}
        if treatment.get("request_index") is None:
            continue
        metrics = treatment.get("native_backend_metrics") or {}
        rows.append(
            {
                "request_index": int(treatment["request_index"]),
                "input_ids_sha256": str(treatment.get("input_ids_sha256") or ""),
                "command": command(message),
                "target_registered": bool(treatment.get("target_registered")),
                "copied_tokens_planned": int(
                    treatment.get("copied_tokens_planned") or 0
                ),
                "physical_reuse": metrics.get("physical_reuse"),
            }
        )
    return rows


def official_instance(run: Path, instance_id: str) -> dict[str, Any]:
    value = read_json(run / "OFFICIAL_RESULT.json")
    report = value.get("report") or (value.get("result") or {}).get("report") or {}
    row = next(
        (
            item
            for item in report.get("instances") or ()
            if str(item.get("instance_id")) == instance_id
        ),
        None,
    )
    if row is None:
        raise ValueError(f"official task row absent: {instance_id} in {run}")
    return {
        key: row.get(key)
        for key in ("completed", "resolved", "empty_patch", "patch_applied")
        if key in row
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", default=MEAN / "GRAPH_MEAN_CANARY_COUNTERFACTUAL.json", type=Path
    )
    args = parser.parse_args()
    roots = {
        "dense": BASE / "runs/sglang_formal/dense/full_24",
        "lcb": (
            BASE
            / "runs/sglang_formal/coding_dependency_graph_cold_lcb/full_24"
        ),
        "mean": (
            MEAN
            / "runs/sglang_canary/coding_dependency_graph_cold_mean/full_4"
        ),
    }
    exposed = []
    for trajectory in sorted(roots["mean"].glob("*/*.traj.json")):
        instance_id = trajectory.parent.name
        per_arm = {
            arm: requests(root / instance_id / f"{instance_id}.traj.json")
            for arm, root in roots.items()
        }
        by_arm = {
            arm: {row["request_index"]: row for row in rows}
            for arm, rows in per_arm.items()
        }
        for mean_row in per_arm["mean"]:
            if not mean_row["target_registered"]:
                continue
            request_index = mean_row["request_index"]
            matched = {
                arm: by_arm[arm].get(request_index) for arm in ("dense", "lcb")
            }
            if any(row is None for row in matched.values()):
                raise ValueError(
                    f"missing request {request_index} counterpart for {instance_id}"
                )
            suffixes = {
                arm: [
                    (row["request_index"], row["input_ids_sha256"], row["command"])
                    for row in rows
                    if row["request_index"] >= request_index
                ]
                for arm, rows in per_arm.items()
            }
            exposed.append(
                {
                    "instance_id": instance_id,
                    "request_index": request_index,
                    "copied_tokens": mean_row["copied_tokens_planned"],
                    "input_identity": {
                        arm: matched[arm]["input_ids_sha256"]
                        == mean_row["input_ids_sha256"]
                        for arm in ("dense", "lcb")
                    },
                    "copy_request_action_identity": {
                        arm: matched[arm]["command"] == mean_row["command"]
                        for arm in ("dense", "lcb")
                    },
                    "post_copy_suffix_identity": {
                        arm: suffixes[arm] == suffixes["mean"]
                        for arm in ("dense", "lcb")
                    },
                    "commands_from_copy": {
                        arm: [row[2] for row in suffixes[arm]]
                        for arm in ("dense", "lcb", "mean")
                    },
                    "official": {
                        arm: official_instance(root, instance_id)
                        for arm, root in roots.items()
                    },
                }
            )
    if not exposed:
        raise RuntimeError("mean canary has no copy-exposed requests")
    value = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "classification": "post-run descriptive attribution; not policy tuning",
        "copy_exposed_requests": len(exposed),
        "all_input_identity": all(
            all(row["input_identity"].values()) for row in exposed
        ),
        "all_copy_request_action_identity": all(
            all(row["copy_request_action_identity"].values()) for row in exposed
        ),
        "all_post_copy_suffix_identity": all(
            all(row["post_copy_suffix_identity"].values()) for row in exposed
        ),
        "exposed": exposed,
        "interpretation_limit": (
            "Identity on one exposed target supports no observed trajectory damage "
            "for this target only; it does not prove accuracy or safety globally."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
