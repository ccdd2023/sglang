#!/usr/bin/env python3
"""Write complete prompts and module boundaries for every frozen candidate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from tokenizers import Tokenizer


ROOT = Path("/home/gfy/CodeMAS_Project")
MODEL = Path(
    "/home/gfy/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-3B-Instruct/"
    "snapshots/488639f1ff808d1d3d0ba301aef8c11461451ec5"
)
DEFAULT_DESIGN = (
    ROOT
    / "kvflow-artifacts/impactkv_module_conditioned_attention_kv_20260807/"
    "task_disjoint20/DESIGN.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "sglang-kvflow-worktrees/coding-aware/docs/kvflow/"
    "MODULE_CONDITIONED_ATTENTION_KV_FULL_PROMPTS_20260807.md"
)
DEFAULT_DATA = (
    ROOT
    / "sglang-kvflow-worktrees/coding-aware/docs/kvflow/assets/"
    "module_conditioned_attention_kv_20260807/FULL_PROMPTS.jsonl"
)
DEFAULT_INDEX = DEFAULT_DATA.with_name("PROMPT_INDEX.csv")


def _token_hash(ids: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for token_id in ids:
        digest.update(int(token_id).to_bytes(8, "little", signed=True))
    return digest.hexdigest()


def _text_hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _fence(value: str) -> str:
    longest = max((len(match.group()) for match in re.finditer(r"~+", value)), default=0)
    marker = "~" * max(4, longest + 1)
    return f"{marker}text\n{value}\n{marker}"


def _validate_blocks(blocks: Sequence[Mapping[str, Any]], total: int) -> None:
    cursor = 0
    for block in blocks:
        if int(block["start"]) != cursor or int(block["end"]) <= cursor:
            raise ValueError("prompt blocks are not a contiguous partition")
        cursor = int(block["end"])
    if cursor != total:
        raise ValueError("prompt blocks do not cover all tokens")


def _block_table(
    blocks: Sequence[Mapping[str, Any]], copied_start: int, copied_end: int
) -> str:
    lines = [
        "| Block | Token interval | Module | Overlaps reused span | Paths |",
        "|---|---:|---|---|---|",
    ]
    for block in blocks:
        start, end = int(block["start"]), int(block["end"])
        overlap = max(start, copied_start) < min(end, copied_end)
        paths = ", ".join(str(value) for value in block.get("paths", []))
        escaped_paths = paths.replace("|", "\\|")
        lines.append(
            f"| `{block['block_id']}` | `[{start},{end})` | "
            f"{block['category']} | {'**yes**' if overlap else 'no'} | "
            f"{escaped_paths} |"
        )
    return "\n".join(lines)


def _annotated(
    tokenizer: Tokenizer,
    ids: Sequence[int],
    *,
    source_start: int,
    target_start: int,
    length: int,
) -> tuple[str, str]:
    target_end = target_start + length
    prefix = tokenizer.decode(ids[:target_start], skip_special_tokens=False)
    reused = tokenizer.decode(ids[target_start:target_end], skip_special_tokens=False)
    suffix = tokenizer.decode(ids[target_end:], skip_special_tokens=False)
    annotation = (
        f"[[DENSE_TARGET_PREFIX tokens=[0,{target_start})]]\n{prefix}\n"
        f"[[LOSSY_REUSE target=[{target_start},{target_end}) "
        f"source=[{source_start},{source_start + length})]]\n{reused}\n"
        f"[[DENSE_TARGET_SUFFIX tokens=[{target_end},{len(ids)})]]\n{suffix}"
    )
    return reused, annotation


def build(design_path: Path, output_path: Path, data_path: Path, index_path: Path) -> dict[str, Any]:
    design = json.loads(design_path.read_text(encoding="utf-8"))
    tokenizer = Tokenizer.from_file(str(MODEL / "tokenizer.json"))
    records = []
    for case in sorted(
        design["cases"], key=lambda row: (str(row["instance_id"]), int(row["request_index"]))
    ):
        source_ids = [int(value) for value in case["source_input_ids"]]
        target_ids = [int(value) for value in case["target_input_ids"]]
        _validate_blocks(case["source_blocks"], len(source_ids))
        _validate_blocks(case["target_blocks"], len(target_ids))
        source_prompt = tokenizer.decode(source_ids, skip_special_tokens=False)
        target_prompt = tokenizer.decode(target_ids, skip_special_tokens=False)
        for candidate in case["candidates"]:
            source_start = int(candidate["source_start"])
            target_start = int(candidate["target_start"])
            length = int(candidate["length"])
            source_slice = source_ids[source_start : source_start + length]
            target_slice = target_ids[target_start : target_start + length]
            if source_slice != target_slice:
                raise ValueError("source/target candidate tokens differ")
            reused, annotation = _annotated(
                tokenizer,
                target_ids,
                source_start=source_start,
                target_start=target_start,
                length=length,
            )
            if tokenizer.decode(source_slice, skip_special_tokens=False) != reused:
                raise ValueError("decoded source/target reused text differs")
            records.append(
                {
                    "record_number": len(records) + 1,
                    "case_id": case["case_id"],
                    "candidate_id": candidate["candidate_id"],
                    "instance_id": case["instance_id"],
                    "request_index": case["request_index"],
                    "source_input_ids": source_ids,
                    "target_input_ids": target_ids,
                    "source_prompt": source_prompt,
                    "target_prompt": target_prompt,
                    "annotated_target_prompt": annotation,
                    "reused_text": reused,
                    "source_start": source_start,
                    "target_start": target_start,
                    "length": length,
                    "rope_position_delta": target_start - source_start,
                    "source_prompt_hash": case["source_prompt_hash"],
                    "target_prompt_hash": case["target_prompt_hash"],
                    "reused_token_hash": _token_hash(source_slice),
                    "reused_text_sha256": _text_hash(reused),
                    "repository_paths": candidate["repository_paths"],
                    "source_blocks": case["source_blocks"],
                    "target_blocks": case["target_blocks"],
                }
            )
    markdown = [
        "# Module-conditioned Attention/KV：完整 Prompt 与复用区间",
        "",
        "本附件逐候选保存机制代理实际看到的完整 source/target prompt。",
        "`[[...]]` 标记只存在于附件，没有发送给模型。",
        "",
        "| # | Task | Request | Candidate | Source interval | Target interval | Paths |",
        "|---:|---|---:|---|---:|---:|---|",
    ]
    for record in records:
        paths = ", ".join(record["repository_paths"]).replace("|", "\\|")
        markdown.append(
            f"| {record['record_number']} | `{record['instance_id']}` | "
            f"{record['request_index']} | `{record['candidate_id']}` | "
            f"`[{record['source_start']},{record['source_start'] + record['length']})` | "
            f"`[{record['target_start']},{record['target_start'] + record['length']})` | {paths} |"
        )
    for record in records:
        markdown.extend(
            [
                "",
                f"## {record['record_number']}. `{record['instance_id']}` / "
                f"`{record['candidate_id']}`",
                "",
                f"RoPE delta: `{record['rope_position_delta']}`; reused token hash: "
                f"`{record['reused_token_hash']}`",
                "",
                "### Source blocks",
                "",
                _block_table(
                    record["source_blocks"],
                    record["source_start"],
                    record["source_start"] + record["length"],
                ),
                "",
                "### Target blocks",
                "",
                _block_table(
                    record["target_blocks"],
                    record["target_start"],
                    record["target_start"] + record["length"],
                ),
                "",
                "### Reused text",
                "",
                _fence(record["reused_text"]),
                "",
                "### Full source prompt",
                "",
                _fence(record["source_prompt"]),
                "",
                "### Full target prompt",
                "",
                _fence(record["target_prompt"]),
                "",
                "### Annotated target prompt",
                "",
                _fence(record["annotated_target_prompt"]),
            ]
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    output_path.chmod(0o644)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    with data_path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    data_path.chmod(0o644)
    index_rows = [
        {
            key: record[key]
            for key in (
                "record_number",
                "case_id",
                "candidate_id",
                "instance_id",
                "request_index",
                "source_start",
                "target_start",
                "length",
                "rope_position_delta",
                "reused_token_hash",
            )
        }
        for record in records
    ]
    with index_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(index_rows[0]))
        writer.writeheader()
        writer.writerows(index_rows)
    index_path.chmod(0o644)
    return {
        "status": "COMPLETE",
        "cases": len({record["case_id"] for record in records}),
        "candidate_records": len(records),
        "all_source_target_reused_tokens_identical": True,
        "markdown": str(output_path),
        "jsonl": str(data_path),
        "index": str(index_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    args = parser.parse_args()
    print(json.dumps(build(args.design, args.output, args.data, args.index), indent=2))


if __name__ == "__main__":
    main()
