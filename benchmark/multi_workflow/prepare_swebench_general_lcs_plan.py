#!/usr/bin/env python3
"""Same-token general copier PLAN: longest Δ≠0 exact run, no file-module gate.

Reuses 96092 target/source token ids. Does not rewrite the official PLAN.
Prefetch and ordinary prefix stay off.
"""

from __future__ import annotations

import argparse
import copy
import difflib
import json
from pathlib import Path
from typing import Any

from sglang.srt.mem_cache.kvcomm.types import token_ids_hash

PLAN_96092 = Path(
    "/home/gfy/CodeMAS_Project/kvflow-artifacts/"
    "impactkv_swebench_prerotated_file_modules_20260818/PLAN.json"
)
POLICY = "general_shifted_lcs"
TARGET_USES = 4


def longest_shifted_middle(
    source: list[int], target: list[int]
) -> tuple[int, int, int] | None:
    """Largest exact common run with Δ≠0 and a non-empty dense prefix+suffix."""
    blocks = difflib.SequenceMatcher(
        None, source, target, autojunk=False
    ).get_matching_blocks()
    candidates = []
    for block in blocks:
        if block.size <= 0:
            continue
        source_start = int(block.a)
        target_start = int(block.b)
        length = int(block.size)
        if source_start <= 0:
            length -= 1 - source_start
            source_start = 1
        if target_start <= 0:
            length -= 1 - target_start
            target_start = 1
        if source_start + length >= len(source):
            length = len(source) - source_start - 1
        if target_start + length >= len(target):
            length = min(length, len(target) - target_start - 1)
        if length <= 0:
            continue
        if source_start == target_start:
            continue
        candidates.append((length, source_start, target_start))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    length, source_start, target_start = candidates[0]
    return source_start, target_start, length


def _no_overlap(kept: list[tuple[int, int, int, int, list[int]]], start: int, length: int) -> bool:
    end = start + length
    for _si, t0, n, _ss, _ids in kept:
        t1 = t0 + n
        if start < t1 and t0 < end:
            return False
    return True


def build_groups(
    official: list[dict[str, Any]],
    *,
    skip_missing_source: bool = False,
) -> list[dict[str, Any]]:
    groups = []
    for row in official:
        target_ids = [int(v) for v in row["target_input_ids"]]
        candidates: list[tuple[int, int, int, int, list[int]]] = []
        for source_index, source_ids_raw in enumerate(row["source_input_ids"]):
            source_ids = [int(v) for v in source_ids_raw]
            span = longest_shifted_middle(source_ids, target_ids)
            if span is None:
                if skip_missing_source:
                    continue
                raise ValueError(
                    f"group {row['group_index']} source {source_index} "
                    "has no shifted middle LCS"
                )
            source_start, target_start, length = span
            if source_ids[source_start : source_start + length] != (
                target_ids[target_start : target_start + length]
            ):
                raise ValueError("LCS token ids do not match")
            if target_start - source_start == 0:
                raise ValueError("zero shift leaked into general PLAN")
            candidates.append(
                (source_index, target_start, length, source_start, source_ids)
            )
        candidates.sort(key=lambda item: -item[2])
        kept: list[tuple[int, int, int, int, list[int]]] = []
        for item in candidates:
            if _no_overlap(kept, item[1], item[2]):
                kept.append(item)
        if not kept:
            if skip_missing_source:
                continue
            raise ValueError(f"group {row['group_index']} dropped all overlapping LCS")
        kept.sort(key=lambda item: item[1])
        for left, right in zip(kept, kept[1:]):
            if left[1] + left[2] > right[1]:
                raise ValueError(
                    f"target overlap survived greedy drop: group {row['group_index']}"
                )
        source_rows = []
        case_rows = []
        copied = 0
        for island_index, (source_index, target_start, length, source_start, source_ids) in enumerate(
            kept
        ):
            shift = target_start - source_start
            source_hash = token_ids_hash(source_ids)
            target_hash = token_ids_hash(target_ids)
            segment_hash = token_ids_hash(
                source_ids[source_start : source_start + length]
            )
            source_id = f"general-g{int(row['group_index']):03d}-s{source_index}"
            source_rows.append(
                {
                    "source_id": source_id,
                    "source_prompt_hash": source_hash,
                    "segment_token_hash": segment_hash,
                    "source_prefix_token_hash": token_ids_hash(
                        source_ids[:source_start]
                    ),
                    "source_start": source_start,
                    "length": length,
                    "content_hash": segment_hash,
                    "policy_label": POLICY,
                    "persistent": True,
                    "pre_rotate_delta": shift,
                }
            )
            case_rows.append(
                {
                    "case_id": f"general-g{int(row['group_index']):03d}-i{island_index}",
                    "source_id": source_id,
                    "source_prompt_hash": source_hash,
                    "target_prompt_hash": target_hash,
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
                    "policy_label": POLICY,
                    "target_group_id": f"general-g{int(row['group_index']):03d}",
                    "target_uses": TARGET_USES,
                    "allow_shifted_copy": True,
                }
            )
            copied += length
        group = copy.deepcopy(row)
        group["source_input_ids"] = [item[4] for item in kept]
        group["sources"] = source_rows
        group["cases"] = case_rows
        group["source_prompt_hashes"] = [item["source_prompt_hash"] for item in source_rows]
        group["islands"] = len(case_rows)
        group["copied_tokens"] = copied
        group["pre_rotate_delta"] = case_rows[0]["target_start"] - case_rows[0][
            "source_start"
        ]
        group["policy_label"] = POLICY
        groups.append(group)
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-plan", type=Path, default=PLAN_96092)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    official = json.loads(args.official_plan.read_text(encoding="utf-8"))
    skip_missing = bool(official.get("not_30b_swebench_plan"))
    groups = build_groups(official["groups"], skip_missing_source=skip_missing)
    by_hash = {row["target_prompt_hash"]: row for row in official["groups"]}
    if not skip_missing and len(groups) != len(official["groups"]):
        raise ValueError("group count changed")
    for ours in groups:
        theirs = by_hash[ours["target_prompt_hash"]]
        if ours["target_input_ids"] != theirs["target_input_ids"]:
            raise ValueError("target token ids drifted")
        if ours["target_prompt_hash"] != theirs["target_prompt_hash"]:
            raise ValueError("target hash drifted")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    is_7b = bool(official.get("not_30b_swebench_plan")) or official.get(
        "model"
    ) == "Qwen2.5-Coder-7B-Instruct"
    payload = {
        "model": (
            "Qwen2.5-Coder-7B-Instruct"
            if is_7b
            else "Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit"
        ),
        "not_96092_coding_plan": True,
        "same_token_ids_as_96092": not is_7b,
        "not_30b_swebench_plan": is_7b,
        "policy_label": POLICY,
        "prefetch": False,
        "ordinary_prefix_reuse": False,
        "groups": groups,
    }
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    longer = sum(
        ours["copied_tokens"] > by_hash[ours["target_prompt_hash"]]["copied_tokens"]
        for ours in groups
    )
    print(
        json.dumps(
            {
                "groups": len(groups),
                "islands": sum(row["islands"] for row in groups),
                "groups_with_longer_lcs_than_coding": longer,
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
