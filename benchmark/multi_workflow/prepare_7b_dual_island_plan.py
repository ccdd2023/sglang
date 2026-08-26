#!/usr/bin/env python3
"""Build a 7B-native dual-island PLAN from frozen 30B COLD_CASES text.

COLD_CASES token ids are Qwen3-Coder-30B. Replaying them on
Qwen2.5-Coder-7B-Instruct is invalid. Decode with the 30B tokenizer,
re-encode with the 7B tokenizer, and re-find the shifted island.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer

from sglang.srt.mem_cache.kvcomm.types import token_ids_hash

COLD_CASES = Path(
    "/home/gfy/CodeMAS_Project/kvflow-artifacts/"
    "impactkv_coding_dual_island_v8_cold_20260727/COLD_CASES.json"
)
PLAN_30B = Path(
    "/home/gfy/CodeMAS_Project/kvflow-artifacts/"
    "impactkv_swebench_prerotated_file_modules_20260818/PLAN.json"
)
TOK_30B = Path(
    "/home/gfy/models/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit/tokenizer.json"
)
TOK_7B = Path("/home/gfy/models/Qwen2.5-Coder-7B-Instruct/tokenizer.json")
MODEL_7B = "Qwen2.5-Coder-7B-Instruct"
POLICY = "coding_natural_code_cost"
TARGET_USES = 4


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_subseq(haystack: list[int], needle: list[int]) -> int:
    length = len(needle)
    if length <= 0 or length > len(haystack):
        return -1
    for index in range(len(haystack) - length + 1):
        if haystack[index : index + length] == needle:
            return index
    return -1


def retokenize_case(
    row: dict[str, Any],
    *,
    tok_30b: Tokenizer,
    tok_7b: Tokenizer,
) -> dict[str, Any]:
    source_30 = [int(v) for v in row["source_input_ids"]]
    target_30 = [int(v) for v in row["target_input_ids"]]
    source_start_30 = int(row["source_start"])
    target_start_30 = int(row["target_start"])
    length_30 = int(row["v7_tokens"])
    island_30 = source_30[source_start_30 : source_start_30 + length_30]
    if island_30 != target_30[target_start_30 : target_start_30 + length_30]:
        raise ValueError(f"{row['case_id']} 30B island token ids do not match")
    source_text = tok_30b.decode(source_30, skip_special_tokens=False)
    target_text = tok_30b.decode(target_30, skip_special_tokens=False)
    island_text = tok_30b.decode(island_30, skip_special_tokens=False)
    source_ids = list(tok_7b.encode(source_text).ids)
    target_ids = list(tok_7b.encode(target_text).ids)
    island_ids = list(tok_7b.encode(island_text).ids)
    if source_ids == source_30:
        raise ValueError(f"{row['case_id']} 7B ids identical to 30B; not retokenized")
    source_start = _find_subseq(source_ids, island_ids)
    target_start = _find_subseq(target_ids, island_ids)
    if source_start <= 0 or target_start <= 0:
        raise ValueError(f"{row['case_id']} 7B island missing dense prefix")
    length = len(island_ids)
    if source_ids[source_start : source_start + length] != island_ids:
        raise ValueError(f"{row['case_id']} 7B source island mismatch")
    if target_ids[target_start : target_start + length] != island_ids:
        raise ValueError(f"{row['case_id']} 7B target island mismatch")
    shift = target_start - source_start
    if shift == 0:
        raise ValueError(f"{row['case_id']} zero shift would be prefix cache")
    return {
        "case_id": row["case_id"],
        "source_input_ids": source_ids,
        "target_input_ids": target_ids,
        "source_start": source_start,
        "target_start": target_start,
        "length": length,
        "shift": shift,
        "source_tokens_30b": len(source_30),
        "target_tokens_30b": len(target_30),
        "island_tokens_30b": length_30,
    }


def build_groups(
    cases: list[dict[str, Any]],
    *,
    tok_30b: Tokenizer | None = None,
    tok_7b: Tokenizer | None = None,
) -> list[dict[str, Any]]:
    tok_30b = tok_30b or Tokenizer.from_file(str(TOK_30B))
    tok_7b = tok_7b or Tokenizer.from_file(str(TOK_7B))
    groups: list[dict[str, Any]] = []
    for index, row in enumerate(cases):
        converted = retokenize_case(row, tok_30b=tok_30b, tok_7b=tok_7b)
        source_ids = converted["source_input_ids"]
        target_ids = converted["target_input_ids"]
        source_start = int(converted["source_start"])
        target_start = int(converted["target_start"])
        length = int(converted["length"])
        shift = int(converted["shift"])
        source_hash = token_ids_hash(source_ids)
        target_hash = token_ids_hash(target_ids)
        segment_hash = token_ids_hash(source_ids[source_start : source_start + length])
        source_prefix_hash = token_ids_hash(source_ids[:source_start])
        target_prefix_hash = token_ids_hash(target_ids[:target_start])
        source_id = f"7b-dual-{row['case_id']}"
        source_row = {
            "source_id": source_id,
            "source_prompt_hash": source_hash,
            "segment_token_hash": segment_hash,
            "source_prefix_token_hash": source_prefix_hash,
            "source_start": source_start,
            "length": length,
            "content_hash": segment_hash,
            "policy_label": POLICY,
            "persistent": True,
            "pre_rotate_delta": shift,
            "later_roles_in_protocol": 3,
        }
        case_row = {
            "case_id": f"7b-dual-g{index:03d}-i0",
            "source_id": source_id,
            "source_prompt_hash": source_hash,
            "target_prompt_hash": target_hash,
            "segment_token_hash": segment_hash,
            "source_prefix_token_hash": source_prefix_hash,
            "target_prefix_token_hash": target_prefix_hash,
            "source_start": source_start,
            "target_start": target_start,
            "length": length,
            "content_hash": segment_hash,
            "policy_label": POLICY,
            "target_group_id": f"7b-dual-g{index:03d}",
            "target_uses": TARGET_USES,
            "allow_shifted_copy": True,
            "copy_middle": True,
        }
        groups.append(
            {
                "group_index": index,
                "original_target_group_id": row["case_id"],
                "target_prompt_hash": target_hash,
                "target_input_ids": target_ids,
                "source_prompt_hashes": [source_hash],
                "source_input_ids": [source_ids],
                "sources": [source_row],
                "cases": [case_row],
                "islands": 1,
                "copied_tokens": length,
                "pre_rotate_delta": shift,
                "model": MODEL_7B,
                "retokenized_from_30b": True,
                "source_tokens_30b": converted["source_tokens_30b"],
                "island_tokens_30b": converted["island_tokens_30b"],
            }
        )
    return groups


def assert_not_30b_plan(groups: list[dict[str, Any]]) -> None:
    thirty = _load(PLAN_30B)
    thirty_hashes = {str(row["target_prompt_hash"]) for row in thirty["groups"]}
    ours = {str(row["target_prompt_hash"]) for row in groups}
    overlap = thirty_hashes & ours
    if overlap:
        raise ValueError(f"7B PLAN reuses 30B target hashes: {sorted(overlap)[:3]}")
    if groups[0]["model"] != MODEL_7B:
        raise ValueError("7B PLAN missing model label")
    if not groups[0].get("retokenized_from_30b"):
        raise ValueError("7B PLAN was not retokenized")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cold-cases", type=Path, default=COLD_CASES)
    parser.add_argument("--tok-30b", type=Path, default=TOK_30B)
    parser.add_argument("--tok-7b", type=Path, default=TOK_7B)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = _load(args.cold_cases)["cases"]
    groups = build_groups(
        cases,
        tok_30b=Tokenizer.from_file(str(args.tok_30b)),
        tok_7b=Tokenizer.from_file(str(args.tok_7b)),
    )
    assert_not_30b_plan(groups)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": MODEL_7B,
        "tokenizer": "Qwen2.5-Coder-7B-Instruct",
        "not_30b_swebench_plan": True,
        "retokenized_from_30b_cold_cases": True,
        "groups": groups,
    }
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "groups": len(groups),
                "islands": sum(row["islands"] for row in groups),
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
