#!/usr/bin/env python3
"""Freeze SWE-bench true-lossy file-module copy with source-side K pre-rotation.

Uses the already-run expanded24 official SWE-bench Verified trajectories
(long rolling-6 multi-file prompts). Keeps only islands whose token IDs
match and whose logical shift is nonzero. Exact-position copies are
dropped so this cannot degenerate to prefix cache. Prefetch stays off.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

def token_ids_hash(ids: list[int]) -> str:
    packed = json.dumps(list(map(int, ids)), separators=(",", ":")).encode()
    return hashlib.sha256(packed).hexdigest()


ARTIFACT = "impactkv_swebench_prerotated_file_modules_20260818"
SOURCE_RUN = (
    "impactkv_natural_code_cost_agent_expanded24_20260808/"
    "online/coding_natural_code_cost/full_24"
)
ARM = "coding_natural_code_cost"
MODEL_DEFAULT = "/home/gfy/models/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit"
WARMUPS = 1
MEASURED_ROUNDS = 3
TOTAL_ROUNDS = WARMUPS + MEASURED_ROUNDS


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def renderer(model: Path, chat_template: Path):
    from jinja2 import StrictUndefined, Template
    from tokenizers import Tokenizer

    from benchmark.multi_workflow.bridge_reuse_litellm_model import BridgeReuseLitellmModel

    model_obj = object.__new__(BridgeReuseLitellmModel)
    model_obj.config = SimpleNamespace(
        reuse_arm=ARM,
        rolling_history_groups=6,
        prompt_token_limit=28_000,
        max_tool_observation_chars=6_000,
        max_assistant_reasoning_chars=3_000,
        emergency_message_chars=1_500,
    )
    model_obj._tokenizer = Tokenizer.from_file(str(model / "tokenizer.json"))
    model_obj._chat_template = Template(
        chat_template.read_text(encoding="utf-8"),
        undefined=StrictUndefined,
    )
    return model_obj


def reconstruct_prompt_index(policy_run: Path, model: Path, chat_template: Path) -> dict[str, list[int]]:
    from benchmark.multi_workflow.bridge_reuse_litellm_model import (
        token_ids_hash as official_token_ids_hash,
    )
    from benchmark.multi_workflow.run_natural_code_cost_exact_prompt_speed import (
        request_prompt_cutoffs,
    )

    model_obj = renderer(model, chat_template)
    prompts: dict[str, list[int]] = {}
    for trajectory_path in sorted(policy_run.rglob("*.traj.json")):
        messages = read_json(trajectory_path)["messages"]
        for index in request_prompt_cutoffs(messages):
            rolling, _, _ = model_obj._rolling_messages(messages[:index])
            compacted, _ = model_obj.compact_messages(rolling)
            ids = model_obj._render_prompt_ids(compacted)
            digest = official_token_ids_hash(ids)
            if digest in prompts and prompts[digest] != ids:
                raise ValueError("prompt hash collision")
            prompts[digest] = ids
    return prompts


def lossy_groups(
    manifest: dict[str, Any],
    prompt_index: dict[str, list[int]],
    *,
    target_uses: int = TOTAL_ROUNDS,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    sources = {str(row["source_id"]): row for row in manifest["sources"]}
    grouped: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    skipped_zero = 0
    skipped_missing = 0
    for case in manifest["cases"]:
        shift = int(case["target_start"]) - int(case["source_start"])
        if shift == 0:
            skipped_zero += 1
            continue
        group_id = str(case["target_group_id"])
        if group_id not in grouped:
            order.append(group_id)
            grouped[group_id] = []
        grouped[group_id].append(case)

    plan = []
    for index, group_id in enumerate(order):
        cases = sorted(grouped[group_id], key=lambda row: int(row["target_start"]))
        target_hashes = {str(row["target_prompt_hash"]) for row in cases}
        if len(target_hashes) != 1:
            skipped_missing += 1
            continue
        target_hash = next(iter(target_hashes))
        if target_hash not in prompt_index:
            skipped_missing += 1
            continue
        source_ids = list(dict.fromkeys(str(row["source_id"]) for row in cases))
        source_rows = []
        source_hashes = []
        source_input = []
        ok = True
        for source_id in source_ids:
            row = copy.deepcopy(sources[source_id])
            source_hash = str(row["source_prompt_hash"])
            if source_hash not in prompt_index:
                ok = False
                break
            matching = [case for case in cases if str(case["source_id"]) == source_id]
            if not matching:
                ok = False
                break
            shift = int(matching[0]["target_start"]) - int(row["source_start"])
            if shift == 0 or any(
                int(case["target_start"]) - int(row["source_start"]) != shift
                for case in matching
            ):
                ok = False
                break
            ids = prompt_index[source_hash]
            start = int(row["source_start"])
            length = int(row["length"])
            target_ids = prompt_index[target_hash]
            target_start = int(matching[0]["target_start"])
            if ids[start : start + length] != target_ids[target_start : target_start + length]:
                ok = False
                break
            row["pre_rotate_delta"] = shift
            row["persistent"] = True
            source_rows.append(row)
            if source_hash not in source_hashes:
                source_hashes.append(source_hash)
                source_input.append(ids)
        if not ok or not source_rows:
            skipped_missing += 1
            continue
        replay_cases = []
        for island_index, row in enumerate(cases):
            replay_cases.append(
                {
                    **copy.deepcopy(row),
                    "case_id": f"swe-prerotate-g{index:03d}-i{island_index}",
                    "target_group_id": f"swe-prerotate-g{index:03d}",
                    "target_uses": target_uses,
                }
            )
        plan.append(
            {
                "group_index": index,
                "original_target_group_id": group_id,
                "target_prompt_hash": target_hash,
                "target_input_ids": prompt_index[target_hash],
                "source_prompt_hashes": source_hashes,
                "source_input_ids": source_input,
                "sources": source_rows,
                "cases": replay_cases,
                "islands": len(replay_cases),
                "copied_tokens": sum(int(row["length"]) for row in replay_cases),
                "pre_rotate_delta": int(source_rows[0]["pre_rotate_delta"]),
            }
        )
    if not plan:
        raise RuntimeError("no true-lossy SWE-bench file-module groups")
    return plan, {
        "zero_shift_islands": skipped_zero,
        "unreconstructed_groups": skipped_missing,
    }


def prepare(
    artifacts: Path,
    output: Path,
    *,
    model: Path,
    chat_template: Path,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    policy_run = artifacts / SOURCE_RUN
    manifest = read_json(policy_run / "DYNAMIC_MANIFEST.json")
    prompt_index = reconstruct_prompt_index(policy_run, model, chat_template)
    groups, skipped = lossy_groups(manifest, prompt_index)
    output.mkdir(parents=True)
    plan_path = output / "PLAN.json"
    write_json(plan_path, {"groups": groups})
    campaign = {
        "schema_version": 1,
        "status": "FROZEN_BEFORE_ANY_SWEBENCH_PREROTATE_REQUEST",
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "classification": (
            "SWE-bench Verified expanded24 exact-prompt true-lossy "
            "repository-code copy with source-side K pre-rotation; "
            "long multi-file rolling-6 prompts; not prefetch; not 7B DS-1000"
        ),
        "dataset": {
            "name": "princeton-nlp/SWE-bench_Verified",
            "source_run": SOURCE_RUN,
            "tasks_in_source_run": 24,
            "target_groups": len(groups),
            "islands": sum(row["islands"] for row in groups),
            "copied_tokens_per_round": sum(row["copied_tokens"] for row in groups),
            "skipped": skipped,
        },
        "candidate": {
            "arm": "swebench_file_module_prerotated",
            "eligible": "version-valid single-file repository_code with matching token IDs and nonzero shift",
            "prefetch": False,
            "ordinary_prefix_reuse": False,
            "source_side_k_prerotation": True,
        },
        "decision_rules": {
            "mechanical": "every planned island copies; residual RoPE delta 0; zero fallback",
            "speed": "cache-ready target TTFT vs same-engine Dense; source/build separate",
            "accuracy": (
                "one-token replay is not official resolved; official Accuracy "
                "remains the expanded24 agent result (Dense 3/24 vs policy 5/24)"
            ),
            "on_fail": "keep DS-1000 four-role one-island as 7B official method; do not mix 30B numbers into 7B tables",
        },
        "protocol": {
            "model": str(model),
            "arms": ["dense", "prerotated_file_module"],
            "decode_tokens": 1,
            "warmups": WARMUPS,
            "measured_rounds": MEASURED_ROUNDS,
            "exact_target_prompt_tokens": True,
            "ordinary_radix_prefix_reuse": False,
            "source_build_reported_separately": True,
            "prefetch": False,
        },
        "protected": {
            "prefetch": False,
            "ordinary_prefix_reuse": False,
            "historical_thresholds_modified": False,
            "called_7b_official": False,
            "called_sota": False,
        },
        "inputs": {
            "plan_sha256": sha256(plan_path),
            "online_manifest_sha256": sha256(policy_run / "DYNAMIC_MANIFEST.json"),
            "source_sha256": sha256(Path(__file__).resolve()),
        },
    }
    write_json(output / "REGISTRATION.json", campaign)
    write_json(output / "STATUS.json", {"schema_version": 1, "state": "registered", "jobs": {}})
    return campaign


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    artifacts = Path.home() / "CodeMAS_Project/kvflow-artifacts"
    project = Path(__file__).resolve().parents[2]
    parser.add_argument("--artifacts", type=Path, default=artifacts)
    parser.add_argument("--output", type=Path, default=artifacts / ARTIFACT)
    parser.add_argument("--model", type=Path, default=Path(MODEL_DEFAULT))
    parser.add_argument(
        "--chat-template",
        type=Path,
        default=project / "benchmark/multi_workflow/qwen3_coder_tool_chat_template.jinja",
    )
    args = parser.parse_args()
    campaign = prepare(
        args.artifacts.resolve(),
        args.output.resolve(),
        model=args.model.resolve(),
        chat_template=args.chat_template.resolve(),
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "target_groups": campaign["dataset"]["target_groups"],
        "islands": campaign["dataset"]["islands"],
        "copied_tokens_per_round": campaign["dataset"]["copied_tokens_per_round"],
        "skipped": campaign["dataset"]["skipped"],
        "prefetch": campaign["protected"]["prefetch"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
