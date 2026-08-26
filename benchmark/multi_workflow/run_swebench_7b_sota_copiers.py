#!/usr/bin/env python3
"""7B Dense vs file-module vs KVCOMM-style LCS vs CacheBlend-style blend.

Same engine, same 7B SWE-bench prompts as job 137185. Not native
CacheBlend/KVCOMM stacks. Does not write into 137185 or 96092.
"""

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


def _clear_reuse(artifact: Path) -> None:
    for name in ("reuse.json", "reuse.partial.json"):
        path = artifact / name
        if path.exists():
            path.unlink()
    shutil.rmtree(artifact / "server" / "reuse", ignore_errors=True)


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
    kvcomm_plan = artifact / "PLAN.kvcomm.json"
    blend_plan = artifact / "PLAN.cacheblend.json"
    for path in (coding_plan, kvcomm_plan, blend_plan):
        if not path.is_file():
            raise FileNotFoundError(path)

    if not (artifact / "dense.json").exists():
        _install_plan(artifact, coding_plan)
        run_arm(artifact, "dense", args.port, args.model)

    if not (artifact / "RESULT.coding.json").exists():
        _install_plan(artifact, coding_plan)
        _clear_reuse(artifact)
        run_arm(artifact, "reuse", args.port, args.model)
        shutil.copyfile(artifact / "reuse.json", artifact / "coding.json")
        _summarize_arm(artifact, "coding.json", "RESULT.coding.json")

    _clear_reuse(artifact)
    _install_plan(artifact, kvcomm_plan)
    run_arm(artifact, "reuse", args.port, args.model)
    shutil.copyfile(artifact / "reuse.json", artifact / "kvcomm.json")
    kvcomm_result = _summarize_arm(artifact, "kvcomm.json", "RESULT.kvcomm.json")

    _clear_reuse(artifact)
    _install_plan(artifact, blend_plan)
    run_arm(artifact, "reuse", args.port, args.model)
    shutil.copyfile(artifact / "reuse.json", artifact / "cacheblend.json")
    blend_result = _summarize_arm(artifact, "cacheblend.json", "RESULT.cacheblend.json")

    coding_result = read_json(artifact / "RESULT.coding.json")
    combined = {
        "schema_version": 1,
        "status": "COMPLETE",
        "classification": (
            "7B same-engine file-module vs KVCOMM-style LCS vs "
            "CacheBlend-style 15% blend; not native stacks; not job 96092"
        ),
        "model": "Qwen2.5-Coder-7B-Instruct",
        "not_30b_swebench_plan": True,
        "same_token_ids_as_96092": False,
        "not_native_cacheblend_or_kvcomm_stack": True,
        "official_96092_prefetch": False,
        "prefetch": False,
        "ordinary_prefix_reuse": False,
        "qwen25_rope_ok": True,
        "rope_base": 1_000_000,
        "coding": coding_result,
        "kvcomm_style": kvcomm_result,
        "cacheblend_style": blend_result,
    }
    if any(
        arm.get("status") != "COMPLETE"
        for arm in (coding_result, kvcomm_result, blend_result)
    ):
        combined["status"] = "PARTIAL"
    write_json(artifact / "RESULT.json", combined)
    print(json.dumps({"status": combined["status"]}, indent=2))


if __name__ == "__main__":
    main()
