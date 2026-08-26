#!/usr/bin/env python3
"""235-group 7B prefix-on increment. Same PLAN as job 137185. Not eval-summary.

Modes: dense, prefix_only, lossy_only, dual. Prefetch stays off.
Does not write into 137185 / 96092 / dual-island artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from benchmark.multi_workflow.prepare_swebench_prerotated_file_modules import (
    MODEL_DEFAULT,
)
from benchmark.multi_workflow.run_natural_code_cost_exact_prompt_speed import (
    read_json,
    write_json,
)
from benchmark.multi_workflow.run_swebench_template_prefetch import (
    _paired_speedup,
    run_mode,
)
from benchmark.multi_workflow.template_prefetch_modes import (
    arm_json,
    ledger_counts,
    parse_modes,
)

MODES = ("dense", "prefix_only", "lossy_only", "dual")
COMPARE = ("prefix_only", "lossy_only", "dual")


def summarize(output: Path, modes: tuple[str, ...] = MODES) -> dict[str, Any]:
    plan_meta = read_json(output / "PLAN.json")
    plan = plan_meta["groups"]
    dense = read_json(output / "dense.json")
    compare = tuple(mode for mode in modes if mode != "dense")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE",
        "classification": (
            "7B 235-group prefix-on increment vs own Dense; not job 137185 "
            "eval-summary and not the seven-group dual-island table"
        ),
        "prefetch": False,
        "ordinary_prefix_reuse": True,
        "not_eval_summary": True,
        "not_7b_dual_island": True,
        "not_30b_swebench_plan": True,
        "model": "Qwen2.5-Coder-7B-Instruct",
        "qwen25_rope_ok": True,
        "parent_plan": "137185",
        "coverage": {
            "target_groups": len(plan),
            "islands": sum(int(row["islands"]) for row in plan),
        },
        "latency": {},
        "mechanism": {},
        "one_token_output_agreement": {"not_accuracy": True},
    }
    missing = [mode for mode in compare if not (output / f"{mode}.json").exists()]
    if missing or not (output / "dense.json").exists():
        payload["status"] = "PARTIAL"
        payload["missing_modes"] = missing
    for mode in compare:
        path = output / f"{mode}.json"
        if not path.exists():
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
            "copy_on_prefix": (
                lat["dual"]["cache_ready_speedup_ratio_of_means"]
                / lat["prefix_only"]["cache_ready_speedup_ratio_of_means"]
            ),
            "prefix_matched": mech["prefix_only"]["ordinary_prefix_matched"],
            "lossy_copies": mech["lossy_only"]["copy_events"],
            "dual_copies": mech["dual"]["copy_events"],
        }
        payload["one_token_output_agreement"]["not_accuracy"] = True
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
    modes = tuple(parse_modes(args.modes))
    unknown = [mode for mode in modes if mode not in MODES]
    if unknown:
        raise ValueError(f"prefix-on runner refuses {unknown}; prefetch combined is off")
    summaries = []
    for mode in modes:
        dest = arm_json(artifact, mode)
        if dest.exists():
            summaries.append({"mode": mode, "skipped_existing": True, "path": str(dest)})
            continue
        summaries.append(run_mode(artifact, mode, args.port, args.model))
    if (artifact / "dense.json").exists():
        result = summarize(artifact, modes)
    else:
        result = {"status": "PARTIAL"}
    print(json.dumps({"runs": summaries, "result": result}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
