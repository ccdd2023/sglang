#!/usr/bin/env python3
"""7B Dense vs coding-aware file-module vs general LCS. Not job 96092."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

from benchmark.multi_workflow.prepare_swebench_prerotated_file_modules import MODEL_DEFAULT
from benchmark.multi_workflow.run_natural_code_cost_exact_prompt_speed import (
    read_json,
    write_json,
)
from benchmark.multi_workflow.run_swebench_prerotated_file_modules import run_arm, summarize


def _install_plan(artifact: Path, source: Path) -> None:
    shutil.copyfile(source, artifact / "PLAN.json")


def _summarize_arm(artifact: Path, reuse_name: str, result_name: str) -> dict[str, Any]:
    shutil.copyfile(artifact / reuse_name, artifact / "reuse.json")
    result = summarize(artifact)
    shutil.copyfile(artifact / "RESULT.json", artifact / result_name)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model", default=os.environ.get("IMPACTKV_MODEL", MODEL_DEFAULT))
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    coding_plan = artifact / "PLAN.coding.json"
    lcs_plan = artifact / "PLAN.lcs.json"
    if not coding_plan.is_file() or not lcs_plan.is_file():
        raise FileNotFoundError("need PLAN.coding.json and PLAN.lcs.json")

    _install_plan(artifact, coding_plan)
    dense = run_arm(artifact, "dense", args.port, args.model)
    coding_run = run_arm(artifact, "reuse", args.port, args.model)
    shutil.copyfile(artifact / "reuse.json", artifact / "coding.json")
    coding_result = _summarize_arm(artifact, "coding.json", "RESULT.coding.json")

    if (artifact / "reuse.json").exists():
        (artifact / "reuse.json").unlink()
    if (artifact / "reuse.partial.json").exists():
        (artifact / "reuse.partial.json").unlink()
    shutil.rmtree(artifact / "server" / "reuse", ignore_errors=True)

    _install_plan(artifact, lcs_plan)
    lcs_run = run_arm(artifact, "reuse", args.port, args.model)
    shutil.copyfile(artifact / "reuse.json", artifact / "lcs.json")
    lcs_result = _summarize_arm(artifact, "lcs.json", "RESULT.lcs.json")

    combined = {
        "schema_version": 1,
        "status": "COMPLETE",
        "classification": (
            "7B-native SWE-bench file-module and LCS copier vs 7B Dense; not job 96092"
        ),
        "model": "Qwen2.5-Coder-7B-Instruct",
        "not_30b_swebench_plan": True,
        "same_token_ids_as_96092": False,
        "official_96092_prefetch": False,
        "prefetch": False,
        "ordinary_prefix_reuse": False,
        "coding": coding_result,
        "lcs": lcs_result,
    }
    if (
        coding_result.get("status") != "COMPLETE"
        or lcs_result.get("status") != "COMPLETE"
    ):
        combined["status"] = "PARTIAL"
    write_json(artifact / "RESULT.json", combined)
    print(
        json.dumps(
            {
                "dense": dense,
                "coding": {
                    "arm": coding_run,
                    "cache_ready_speedup": coding_result["latency"][
                        "cache_ready_speedup_ratio_of_means"
                    ],
                    "copy_events": coding_result["mechanism"]["copy_events"],
                    "fallback_events": coding_result["mechanism"]["fallback_events"],
                },
                "lcs": {
                    "arm": lcs_run,
                    "cache_ready_speedup": lcs_result["latency"][
                        "cache_ready_speedup_ratio_of_means"
                    ],
                    "copy_events": lcs_result["mechanism"]["copy_events"],
                    "fallback_events": lcs_result["mechanism"]["fallback_events"],
                },
                "result_status": combined["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
