#!/usr/bin/env python3
"""7B-native SWE-bench file-module PLAN from frozen 30B 96092 ids.

30B token ids must not be replayed on Qwen2.5-Coder-7B-Instruct.
Decode with the 30B tokenizer, re-encode with 7B, relocate each island
by character offsets. Drop islands that are not a strict Δ≠0 middle.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer

from sglang.srt.mem_cache.kvcomm.types import token_ids_hash

from benchmark.multi_workflow.prepare_swebench_general_lcs_plan import (
    build_groups as build_lcs_groups,
)

def _artifacts() -> Path:
    return Path(
        os.environ.get(
            "IMPACTKV_ARTIFACTS",
            str(Path.home() / "impactkv-artifacts"),
        )
    ).expanduser()


PLAN_30B = (
    _artifacts() / "impactkv_swebench_prerotated_file_modules_20260818/PLAN.json"
)
TOK_30B = Path(
    os.environ.get(
        "IMPACTKV_TOK_30B",
        str(
            Path.home()
            / "models/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit/tokenizer.json"
        ),
    )
).expanduser()
TOK_7B = Path(
    os.environ.get(
        "IMPACTKV_TOK_7B",
        str(Path.home() / "models/Qwen2.5-Coder-7B-Instruct/tokenizer.json"),
    )
).expanduser()
MODEL_7B = "Qwen2.5-Coder-7B-Instruct"
POLICY = "coding_natural_code_cost"
TARGET_USES = 4


def locate_text_span(
    tok: Tokenizer, full_text: str, piece: str, hint_frac: float
) -> tuple[int, int] | None:
    """Map a decoded island onto 7B token ids using char offsets."""
    if not piece or not full_text:
        return None
    starts: list[int] = []
    cursor = 0
    while True:
        found = full_text.find(piece, cursor)
        if found < 0:
            break
        starts.append(found)
        cursor = found + 1
    if not starts:
        return None
    hint_char = int(max(0.0, min(1.0, hint_frac)) * len(full_text))
    pos = min(starts, key=lambda value: abs(value - hint_char))
    end = pos + len(piece)
    encoded = tok.encode(full_text)
    ids = list(encoded.ids)
    offsets = list(encoded.offsets)
    start_tok: int | None = None
    end_tok: int | None = None
    for index, (left, right) in enumerate(offsets):
        if right <= left:
            continue
        if right <= pos or left >= end:
            continue
        if start_tok is None:
            start_tok = index
        end_tok = index + 1
    if start_tok is None or end_tok is None or end_tok <= start_tok:
        return None
    if start_tok <= 0:
        return None
    if end_tok >= len(ids):
        return None
    return start_tok, end_tok - start_tok


def _no_overlap(
    kept: list[tuple[int, int, int, int, list[int]]], start: int, length: int
) -> bool:
    end = start + length
    for _si, t0, n, _ss, _ids in kept:
        t1 = t0 + n
        if start < t1 and t0 < end:
            return False
    return True


def retokenize_group(
    row: dict[str, Any],
    *,
    tok_30b: Tokenizer,
    tok_7b: Tokenizer,
) -> dict[str, Any] | None:
    target_30 = [int(v) for v in row["target_input_ids"]]
    target_text = tok_30b.decode(target_30, skip_special_tokens=False)
    target_ids = list(tok_7b.encode(target_text).ids)
    if target_ids == target_30:
        raise ValueError(f"group {row['group_index']} 7B ids identical to 30B")
    source_cache: dict[int, tuple[list[int], str]] = {}
    candidates: list[tuple[int, int, int, int, list[int]]] = []
    for case in row["cases"]:
        source_id = str(case["source_id"])
        source_index = next(
            index
            for index, source in enumerate(row["sources"])
            if source["source_id"] == source_id
        )
        if source_index not in source_cache:
            source_30 = [int(v) for v in row["source_input_ids"][source_index]]
            source_text = tok_30b.decode(source_30, skip_special_tokens=False)
            source_ids = list(tok_7b.encode(source_text).ids)
            source_cache[source_index] = (source_ids, source_text)
        source_ids, source_text = source_cache[source_index]
        source_start_30 = int(case["source_start"])
        target_start_30 = int(case["target_start"])
        length_30 = int(case["length"])
        island_30 = [int(v) for v in row["source_input_ids"][source_index]][
            source_start_30 : source_start_30 + length_30
        ]
        if island_30 != target_30[target_start_30 : target_start_30 + length_30]:
            continue
        island_text = tok_30b.decode(island_30, skip_special_tokens=False)
        source_span = locate_text_span(
            tok_7b,
            source_text,
            island_text,
            source_start_30 / max(len(row["source_input_ids"][source_index]), 1),
        )
        target_span = locate_text_span(
            tok_7b,
            target_text,
            island_text,
            target_start_30 / max(len(target_30), 1),
        )
        if source_span is None or target_span is None:
            continue
        source_start, length = source_span
        target_start, target_length = target_span
        if length != target_length:
            continue
        if source_ids[source_start : source_start + length] != (
            target_ids[target_start : target_start + length]
        ):
            continue
        if source_start == target_start:
            continue
        candidates.append((source_index, target_start, length, source_start, source_ids))
    if not candidates:
        return None
    candidates.sort(key=lambda item: -item[2])
    kept: list[tuple[int, int, int, int, list[int]]] = []
    for item in candidates:
        if _no_overlap(kept, item[1], item[2]):
            kept.append(item)
    if not kept:
        return None
    kept.sort(key=lambda item: item[1])
    target_hash = token_ids_hash(target_ids)
    source_rows = []
    case_rows = []
    source_ids_out = []
    copied = 0
    for island_index, (source_index, target_start, length, source_start, source_ids) in enumerate(
        kept
    ):
        shift = target_start - source_start
        source_hash = token_ids_hash(source_ids)
        segment_hash = token_ids_hash(source_ids[source_start : source_start + length])
        source_id = f"7b-fm-g{int(row['group_index']):03d}-s{source_index}"
        source_rows.append(
            {
                "source_id": source_id,
                "source_prompt_hash": source_hash,
                "segment_token_hash": segment_hash,
                "source_prefix_token_hash": token_ids_hash(source_ids[:source_start]),
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
                "case_id": f"7b-fm-g{int(row['group_index']):03d}-i{island_index}",
                "source_id": source_id,
                "source_prompt_hash": source_hash,
                "target_prompt_hash": target_hash,
                "segment_token_hash": segment_hash,
                "source_prefix_token_hash": token_ids_hash(source_ids[:source_start]),
                "target_prefix_token_hash": token_ids_hash(target_ids[:target_start]),
                "source_start": source_start,
                "target_start": target_start,
                "length": length,
                "content_hash": segment_hash,
                "policy_label": POLICY,
                "target_group_id": f"7b-fm-g{int(row['group_index']):03d}",
                "target_uses": TARGET_USES,
                "allow_shifted_copy": True,
            }
        )
        source_ids_out.append(source_ids)
        copied += length
    group = copy.deepcopy(row)
    group["target_input_ids"] = target_ids
    group["target_prompt_hash"] = target_hash
    group["source_input_ids"] = source_ids_out
    group["source_prompt_hashes"] = [item["source_prompt_hash"] for item in source_rows]
    group["sources"] = source_rows
    group["cases"] = case_rows
    group["islands"] = len(case_rows)
    group["copied_tokens"] = copied
    group["pre_rotate_delta"] = case_rows[0]["target_start"] - case_rows[0]["source_start"]
    group["policy_label"] = POLICY
    group["model"] = MODEL_7B
    group["retokenized_from_30b"] = True
    group["original_30b_group_index"] = int(row["group_index"])
    return group


def reindex(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for index, row in enumerate(groups):
        group = copy.deepcopy(row)
        group["group_index"] = index
        output.append(group)
    return output


def build_coding_groups(
    official: list[dict[str, Any]],
    *,
    tok_30b: Tokenizer,
    tok_7b: Tokenizer,
) -> list[dict[str, Any]]:
    groups = []
    for row in official:
        converted = retokenize_group(row, tok_30b=tok_30b, tok_7b=tok_7b)
        if converted is None:
            continue
        groups.append(converted)
    return reindex(groups)


def assert_not_30b_plan(groups: list[dict[str, Any]], official: list[dict[str, Any]]) -> None:
    thirty_hashes = {str(row["target_prompt_hash"]) for row in official}
    ours = {str(row["target_prompt_hash"]) for row in groups}
    overlap = thirty_hashes & ours
    if overlap:
        raise ValueError(f"7B PLAN reuses 30B target hashes: {sorted(overlap)[:3]}")
    if not groups:
        raise ValueError("7B PLAN is empty")
    if groups[0]["model"] != MODEL_7B:
        raise ValueError("7B PLAN missing model label")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-plan", type=Path, default=PLAN_30B)
    parser.add_argument("--tok-30b", type=Path, default=TOK_30B)
    parser.add_argument("--tok-7b", type=Path, default=TOK_7B)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    official = json.loads(args.official_plan.read_text(encoding="utf-8"))
    tok_30b = Tokenizer.from_file(str(args.tok_30b))
    tok_7b = Tokenizer.from_file(str(args.tok_7b))
    coding = build_coding_groups(
        official["groups"], tok_30b=tok_30b, tok_7b=tok_7b
    )
    assert_not_30b_plan(coding, official["groups"])
    lcs = build_lcs_groups(coding, skip_missing_source=True)
    coding_by_hash = {row["target_prompt_hash"]: row for row in coding}
    lcs = [row for row in lcs if row["target_prompt_hash"] in coding_by_hash]
    keep = {row["target_prompt_hash"] for row in lcs}
    coding = [row for row in coding if row["target_prompt_hash"] in keep]
    coding = reindex(coding)
    lcs = reindex(lcs)
    if [row["target_prompt_hash"] for row in coding] != [
        row["target_prompt_hash"] for row in lcs
    ]:
        raise ValueError("coding/LCS target sets drifted after intersection")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    coding_payload = {
        "model": MODEL_7B,
        "tokenizer": MODEL_7B,
        "not_30b_swebench_plan": True,
        "same_token_ids_as_96092": False,
        "retokenized_from_30b": True,
        "policy_label": POLICY,
        "prefetch": False,
        "ordinary_prefix_reuse": False,
        "groups": coding,
        "dropped_30b_groups": len(official["groups"]) - len(coding),
    }
    lcs_payload = {
        "model": MODEL_7B,
        "tokenizer": MODEL_7B,
        "not_30b_swebench_plan": True,
        "not_96092_coding_plan": True,
        "same_token_ids_as_96092": False,
        "retokenized_from_30b": True,
        "policy_label": "general_shifted_lcs",
        "prefetch": False,
        "ordinary_prefix_reuse": False,
        "groups": lcs,
        "dropped_30b_groups": len(official["groups"]) - len(lcs),
    }
    coding_path = args.output_dir / "PLAN.coding.json"
    lcs_path = args.output_dir / "PLAN.lcs.json"
    coding_path.write_text(
        json.dumps(coding_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lcs_path.write_text(
        json.dumps(lcs_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "groups": len(coding),
                "coding_islands": sum(row["islands"] for row in coding),
                "lcs_islands": sum(row["islands"] for row in lcs),
                "dropped_30b_groups": len(official["groups"]) - len(coding),
                "coding": str(coding_path),
                "lcs": str(lcs_path),
            }
        )
    )


if __name__ == "__main__":
    main()
