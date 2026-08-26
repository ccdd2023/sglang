#!/usr/bin/env python3
"""Same-token 7B copier PLANs: KVCOMM-style unconstrained LCS and CacheBlend-style 15% blend.

Not native CacheBlend/KVCOMM engines. Same 7B SWE-bench prompts as job 137185.
Does not overwrite 137185 or 96092.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from sglang.srt.mem_cache.kvcomm.types import token_ids_hash

from benchmark.multi_workflow.prepare_swebench_general_lcs_plan import build_groups

BLEND_RATIO = 0.15
CODING_PLAN = Path(
    "/home/gfy/CodeMAS_Project/kvflow-artifacts/"
    "impactkv_swebench_7b_file_modules_prefixkey_20260824/PLAN.json"
)


def shrink_island_for_blend(
    source_start: int,
    target_start: int,
    length: int,
    *,
    ratio: float = BLEND_RATIO,
) -> tuple[int, int, int, int] | None:
    """Keep a Dense blend prefix; copy the remainder. Δ is unchanged."""
    if length <= 1:
        return None
    recompute = max(1, int(round(length * ratio)))
    if recompute >= length:
        return None
    copied = length - recompute
    return source_start + recompute, target_start + recompute, copied, recompute


def cacheblend_groups(lcs_groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in lcs_groups:
        source_rows = []
        case_rows = []
        source_ids_out = []
        copied = 0
        recomputed = 0
        for island_index, (source, case) in enumerate(
            zip(row["sources"], row["cases"], strict=True)
        ):
            shrunk = shrink_island_for_blend(
                int(source["source_start"]),
                int(case["target_start"]),
                int(case["length"]),
            )
            if shrunk is None:
                continue
            source_start, target_start, length, recompute = shrunk
            source_ids = [int(v) for v in row["source_input_ids"][island_index]]
            target_ids = [int(v) for v in row["target_input_ids"]]
            if source_ids[source_start : source_start + length] != (
                target_ids[target_start : target_start + length]
            ):
                raise ValueError("CacheBlend shrink broke token equality")
            if source_start == target_start:
                raise ValueError("CacheBlend shrink leaked Δ=0")
            segment = source_ids[source_start : source_start + length]
            segment_hash = token_ids_hash(segment)
            source_id = f"blend-g{int(row['group_index']):03d}-s{island_index}"
            shift = target_start - source_start
            source_rows.append(
                {
                    "source_id": source_id,
                    "source_prompt_hash": token_ids_hash(source_ids),
                    "segment_token_hash": segment_hash,
                    "source_prefix_token_hash": token_ids_hash(
                        source_ids[:source_start]
                    ),
                    "source_start": source_start,
                    "length": length,
                    "content_hash": segment_hash,
                    "policy_label": "cacheblend_style_blend15",
                    "persistent": True,
                    "pre_rotate_delta": shift,
                }
            )
            case_rows.append(
                {
                    "case_id": f"blend-g{int(row['group_index']):03d}-i{island_index}",
                    "source_id": source_id,
                    "source_prompt_hash": token_ids_hash(source_ids),
                    "target_prompt_hash": token_ids_hash(target_ids),
                    "segment_token_hash": segment_hash,
                    "source_prefix_token_hash": token_ids_hash(
                        source_ids[:source_start]
                    ),
                    "target_prefix_token_hash": token_ids_hash(
                        target_ids[:target_start]
                    ),
                    "source_start": source_start,
                    "target_start": target_start,
                    "length": length,
                    "content_hash": segment_hash,
                    "policy_label": "cacheblend_style_blend15",
                    "target_group_id": f"blend-g{int(row['group_index']):03d}",
                    "target_uses": 4,
                    "allow_shifted_copy": True,
                    "blend_recompute_tokens": recompute,
                }
            )
            source_ids_out.append(source_ids)
            copied += length
            recomputed += recompute
        if not case_rows:
            continue
        group = copy.deepcopy(row)
        group["source_input_ids"] = source_ids_out
        group["sources"] = source_rows
        group["cases"] = case_rows
        group["source_prompt_hashes"] = [
            item["source_prompt_hash"] for item in source_rows
        ]
        group["islands"] = len(case_rows)
        group["copied_tokens"] = copied
        group["blend_recompute_tokens"] = recomputed
        group["pre_rotate_delta"] = (
            case_rows[0]["target_start"] - case_rows[0]["source_start"]
        )
        group["policy_label"] = "cacheblend_style_blend15"
        output.append(group)
    return output


def write_plan(path: Path, groups: list[dict[str, Any]], *, policy: str) -> None:
    payload = {
        "model": "Qwen2.5-Coder-7B-Instruct",
        "tokenizer": "Qwen2.5-Coder-7B-Instruct",
        "not_30b_swebench_plan": True,
        "not_96092_coding_plan": True,
        "same_token_ids_as_96092": False,
        "retokenized_from_30b": True,
        "same_engine_policy_clone": True,
        "not_native_cacheblend_or_kvcomm_stack": True,
        "policy_label": policy,
        "prefetch": False,
        "ordinary_prefix_reuse": False,
        "groups": groups,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coding-plan", type=Path, default=CODING_PLAN)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    official = json.loads(args.coding_plan.read_text(encoding="utf-8"))
    if official.get("model") != "Qwen2.5-Coder-7B-Instruct":
        raise ValueError("expected the 7B retokenized PLAN")
    kvcomm = build_groups(official["groups"], skip_missing_source=True)
    blend = cacheblend_groups(kvcomm)
    if len(kvcomm) != len(official["groups"]):
        raise ValueError("KVCOMM-style PLAN dropped 7B groups")
    if len(blend) != len(official["groups"]):
        raise ValueError("CacheBlend-style PLAN dropped 7B groups")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_plan(args.output_dir / "PLAN.kvcomm.json", kvcomm, policy="general_shifted_lcs")
    write_plan(
        args.output_dir / "PLAN.cacheblend.json",
        blend,
        policy="cacheblend_style_blend15",
    )
    coding_copied = sum(int(row["copied_tokens"]) for row in official["groups"])
    kvcomm_copied = sum(int(row["copied_tokens"]) for row in kvcomm)
    blend_copied = sum(int(row["copied_tokens"]) for row in blend)
    print(
        json.dumps(
            {
                "groups": len(kvcomm),
                "coding_copied_tokens": coding_copied,
                "kvcomm_copied_tokens": kvcomm_copied,
                "cacheblend_copied_tokens": blend_copied,
                "kvcomm_islands": sum(row["islands"] for row in kvcomm),
                "cacheblend_islands": sum(row["islands"] for row in blend),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
