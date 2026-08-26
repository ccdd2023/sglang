#!/usr/bin/env python3
"""7B dual-island prefix / lossy / prefetch replay. Does not replace job 96092."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path
from typing import Any

from benchmark.multi_workflow.prepare_swebench_prerotated_file_modules import (
    MODEL_DEFAULT,
)
from benchmark.multi_workflow.template_prefetch_modes import (
    MODES,
    arm_json,
    combined_vs_coding_speedup,
    ledger_counts,
    mode_env,
    mode_manifest,
    parse_modes,
)
from benchmark.multi_workflow.run_natural_code_cost_exact_prompt_speed import (
    read_json,
    write_json,
)
from benchmark.multi_workflow.run_swebench_prerotated_file_modules import (
    run_arm as run_prerotated_arm,
)
from benchmark.multi_workflow import run_swebench_prerotated_file_modules as prerot




def _paired_speedup(dense: dict[str, Any], other: dict[str, Any]) -> dict[str, float]:
    dense_rows = {
        (int(row["group_index"]), int(row["round_index"])): row
        for row in dense["targets"]
        if not row["warmup"]
    }
    other_rows = {
        (int(row["group_index"]), int(row["round_index"])): row
        for row in other["targets"]
        if not row["warmup"]
    }
    if set(dense_rows) != set(other_rows):
        raise ValueError("paired targets differ")
    dense_mean = statistics.fmean(float(row["ttft_ms"]) for row in dense_rows.values())
    other_mean = statistics.fmean(float(row["ttft_ms"]) for row in other_rows.values())
    savings = [
        1 - float(other_rows[key]["ttft_ms"]) / float(dense_rows[key]["ttft_ms"])
        for key in dense_rows
    ]
    return {
        "cache_ready_speedup_ratio_of_means": dense_mean / other_mean,
        "paired_ttft_saving_median": statistics.median(savings),
        "paired_ttft_win_rate": sum(value > 0 for value in savings) / len(savings),
        "measured_pairs": float(len(savings)),
    }


def run_mode(output: Path, mode: str, port: int, model: str) -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError(mode)
    for key, value in mode_env(mode).items():
        os.environ[key] = value
    server_arm = "dense" if mode == "dense" else "reuse"
    original_manifest = prerot._manifest
    prerot._manifest = lambda output_path, group, model_value: mode_manifest(
        output_path, group, model_value, mode
    )
    sidecar = output / "arms" / mode
    sidecar.mkdir(parents=True, exist_ok=True)
    plan_dst = sidecar / "PLAN.json"
    if not plan_dst.exists():
        plan_dst.symlink_to((output / "PLAN.json").resolve())
    try:
        summary = run_prerotated_arm(sidecar, server_arm, port, model)
    finally:
        prerot._manifest = original_manifest
    arm_json = sidecar / f"{server_arm}.json"
    dest = output / f"{mode}.json"
    if arm_json.exists():
        dest.write_bytes(arm_json.read_bytes())
    return {"mode": mode, **summary}


def summarize(output: Path) -> dict[str, Any]:
    plan_meta = read_json(output / "PLAN.json")
    plan = plan_meta["groups"]
    dense = read_json(output / "dense.json")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE",
        "classification": "7B dual-island prefix + lossy file-island + template prefetch",
        "official_96092_prefetch": False,
        "prefetch_campaign": True,
        "ordinary_prefix_reuse": True,
        "model": plan_meta.get("model") or "Qwen2.5-Coder-7B-Instruct",
        "not_30b_swebench_plan": True,
        "coverage": {
            "target_groups": len(plan),
            "islands": sum(row["islands"] for row in plan),
        },
        "latency": {},
        "mechanism": {},
        "one_token_output_agreement": {"not_accuracy": True},
    }
    for mode in ("prefix_only", "lossy_only", "dual", "combined"):
        path = output / f"{mode}.json"
        if not path.exists():
            payload["status"] = "PARTIAL"
            continue
        other = read_json(path)
        payload["latency"][mode] = _paired_speedup(dense, other)
        payload["mechanism"][mode] = ledger_counts(other.get("ledger_rows") or [])
    if payload["status"] == "COMPLETE":
        lat = payload["latency"]
        mech = payload["mechanism"]
        payload["algorithm_bars"] = {
            "prefix_vs_dense": lat["prefix_only"]["cache_ready_speedup_ratio_of_means"],
            "lossy_vs_dense": lat["lossy_only"]["cache_ready_speedup_ratio_of_means"],
            "dual_vs_dense": lat["dual"]["cache_ready_speedup_ratio_of_means"],
            "prefix_matched": mech["prefix_only"]["ordinary_prefix_matched"],
            "lossy_copies": mech["lossy_only"]["copy_events"],
            "significant_threshold": 1.05,
        }
        if "combined" in lat and "dual" in lat:
            payload["prefetch_increment"] = {
                "combined_vs_coding": combined_vs_coding_speedup(lat),
                "significant_threshold": 1.05,
            }
    write_json(output / "RESULT.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model", default=os.environ.get("IMPACTKV_MODEL", MODEL_DEFAULT))
    parser.add_argument("--modes", default=",".join(MODES))
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    summaries = []
    for mode in parse_modes(args.modes):
        dest = arm_json(artifact, mode)
        if dest.exists():
            summaries.append(
                {"mode": mode, "skipped_existing": True, "path": str(dest)}
            )
            continue
        summaries.append(run_mode(artifact, mode, args.port, args.model))
    if (artifact / "dense.json").exists():
        result = summarize(artifact)
    else:
        result = {"status": "PARTIAL"}
    print(json.dumps({"runs": summaries, "result": result}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
