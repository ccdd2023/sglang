#!/usr/bin/env python3
"""Build a full-prompt audit appendix for the frozen module-attention cohort."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from tokenizers import Tokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = PROJECT_ROOT.parents[1] / "kvflow-artifacts"
DEFAULT_DESIGN = (
    ARTIFACT_ROOT
    / "impactkv_global_block_attention_20260806/frozen26_r2/DESIGN.json"
)
DEFAULT_RESULT = (
    ARTIFACT_ROOT
    / "impactkv_global_block_attention_20260806/frozen26_r2/RESULT.json"
)
MODEL = Path(
    "/home/gfy/.cache/huggingface/hub/"
    "models--Qwen--Qwen2.5-Coder-3B-Instruct/snapshots/"
    "488639f1ff808d1d3d0ba301aef8c11461451ec5"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "docs/kvflow/PROMPT_MODULE_ATTENTION_KV_FULL_PROMPTS_20260806.md"
)
DEFAULT_DATA = (
    PROJECT_ROOT
    / "docs/kvflow/assets/prompt_module_attention_kv_20260806/FULL_PROMPTS.jsonl"
)
DEFAULT_INDEX = (
    PROJECT_ROOT
    / "docs/kvflow/assets/prompt_module_attention_kv_20260806/PROMPT_INDEX.csv"
)

LABELS = {
    "system_instruction": "System prompt",
    "user_task": "Coding task",
    "compaction_notice": "Context control",
    "assistant_action": "Agent action",
    "copied_observation_island": "Copied repo evidence",
    "read_observation_path_relevant": "Path-relevant repo evidence",
    "read_observation_path_disjoint": "Other repo evidence",
    "other_tool_result": "Tool / runtime feedback",
    "generation_marker": "Next action",
}


def token_hash(ids: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for token_id in ids:
        digest.update(int(token_id).to_bytes(8, "little", signed=True))
    return digest.hexdigest()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def table_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def fence(text: str, language: str = "text") -> str:
    longest = max((len(match.group(0)) for match in re.finditer(r"~+", text)), default=0)
    marker = "~" * max(4, longest + 1)
    return f"{marker}{language}\n{text}\n{marker}"


def validate_blocks(blocks: Sequence[Mapping[str, Any]], total: int) -> None:
    cursor = 0
    for block in blocks:
        if int(block["start"]) != cursor:
            raise ValueError(f"block coverage gap at {block['block_id']}: {cursor}")
        if int(block["tokens"]) != int(block["end"]) - int(block["start"]):
            raise ValueError(f"block token count mismatch: {block['block_id']}")
        cursor = int(block["end"])
    if cursor != total:
        raise ValueError(f"block coverage ends at {cursor}, expected {total}")


def block_table(blocks: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "| Block | Token interval | Tokens | Prompt module | Reused island | Paths / label |",
        "|---|---:|---:|---|---|---|",
    ]
    for block in blocks:
        paths = ", ".join(str(path) for path in block.get("paths", []))
        label = str(block.get("label", ""))
        detail = f"{paths}; {label}" if paths else label
        lines.append(
            "| {block} | `[{start}, {end})` | {tokens} | {module} | {copied} | {detail} |".format(
                block=table_escape(block["block_id"]),
                start=int(block["start"]),
                end=int(block["end"]),
                tokens=int(block["tokens"]),
                module=table_escape(LABELS[str(block["category"])]),
                copied="**yes**" if block.get("copied") else "no",
                detail=table_escape(detail),
            )
        )
    return "\n".join(lines)


def annotated_target(
    tokenizer: Tokenizer,
    ids: Sequence[int],
    *,
    target_start: int,
    source_start: int,
    length: int,
) -> tuple[str, str, str, str]:
    target_end = target_start + length
    prefix = tokenizer.decode(ids[:target_start], skip_special_tokens=False)
    reused = tokenizer.decode(ids[target_start:target_end], skip_special_tokens=False)
    suffix = tokenizer.decode(ids[target_end:], skip_special_tokens=False)
    annotation = (
        f"[[DENSE_TARGET_PREFIX_BEGIN tokens=[0,{target_start})]]\n"
        f"{prefix}"
        f"\n[[DENSE_TARGET_PREFIX_END]]\n"
        f"[[LOSSY_REUSE_BEGIN target=[{target_start},{target_end}) "
        f"source=[{source_start},{source_start + length})]]\n"
        f"{reused}"
        f"\n[[LOSSY_REUSE_END]]\n"
        f"[[DENSE_TARGET_SUFFIX_BEGIN tokens=[{target_end},{len(ids)})]]\n"
        f"{suffix}"
        f"\n[[DENSE_TARGET_SUFFIX_END]]"
    )
    return annotation, prefix, reused, suffix


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    path.chmod(0o644)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    path.chmod(0o644)


def build(
    *,
    design_path: Path,
    result_path: Path,
    output_path: Path,
    data_path: Path,
    index_path: Path,
) -> dict[str, Any]:
    design = json.loads(design_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    tokenizer = Tokenizer.from_file(str(MODEL / "tokenizer.json"))
    representatives = set(result["representative_case_ids"])
    cases = sorted(
        design["cases"],
        key=lambda row: (str(row["instance_id"]), int(row["request_index"])),
    )
    markdown = [
        "# Prompt 模块 Attention–KV 审计：26 例完整 Prompt 附件",
        "",
        "日期：2026-08-06  ",
        "用途：逐例核对模型实际输入、模块 token 边界和有损 KV 复用区间。",
        "",
        "## 阅读约定",
        "",
        "- `Source prompt` 是建立旧 K/V 的完整请求；`Target prompt` 是 Dense 与 Lossy 两臂共同看到的完整请求。",
        "- 两臂的 target `input_ids` 完全相同；Lossy 不删改 prompt 文本，只改变中间 token 区间的 K/V 来源。",
        "- `[[...]]` 是本附件加入的审计标记，**没有发送给模型**。删除这些标记后就是完整 target decoded prompt。",
        "- Special tokens（例如 `<|im_start|>`）被保留，便于核对 chat template。",
        "- `Copied repo evidence` 在 target 中没有重新执行 query rows；它的 K/V 取自 source 对应区间。K 做 RoPE 位置平移，V 原样复制。",
        "",
        "## 执行语义",
        "",
        "```text",
        "Dense target:",
        "  compute target[0 : target_tokens]",
        "",
        "Lossy target:",
        "  compute target[0 : target_start]",
        "  append source_KV[source_start : source_start + length]",
        "    K := RoPE_shift(K, target_start - source_start)",
        "    V := copied unchanged",
        "  compute target[target_start + length : target_tokens] with hybrid cache",
        "```",
        "",
        "## Case 索引",
        "",
        "| # | Instance | Request | Source tokens | Target tokens | Source reuse | Target reuse | Copied tokens | Representative |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    records = []
    index_rows = []
    for number, case in enumerate(cases, 1):
        source_ids = [int(value) for value in case["source_input_ids"]]
        target_ids = [int(value) for value in case["target_input_ids"]]
        source_start = int(case["source_start"])
        target_start = int(case["target_start"])
        length = int(case["length"])
        source_end = source_start + length
        target_end = target_start + length
        validate_blocks(case["source_blocks"], len(source_ids))
        validate_blocks(case["target_blocks"], len(target_ids))
        source_slice_ids = source_ids[source_start:source_end]
        target_slice_ids = target_ids[target_start:target_end]
        if source_slice_ids != target_slice_ids:
            raise ValueError(f"source/target copied tokens differ: {case['case_id']}")
        if token_hash(source_slice_ids) != str(case["segment_token_hash"]):
            raise ValueError(f"copied token hash mismatch: {case['case_id']}")

        source_prompt = tokenizer.decode(source_ids, skip_special_tokens=False)
        target_prompt = tokenizer.decode(target_ids, skip_special_tokens=False)
        annotation, target_prefix, reused_text, target_suffix = annotated_target(
            tokenizer,
            target_ids,
            target_start=target_start,
            source_start=source_start,
            length=length,
        )
        source_reused_text = tokenizer.decode(
            source_slice_ids, skip_special_tokens=False
        )
        if source_reused_text != reused_text:
            raise ValueError(f"decoded reused text differs: {case['case_id']}")
        if target_prefix + reused_text + target_suffix != target_prompt:
            raise ValueError(f"annotated target is not lossless: {case['case_id']}")

        representative = str(case["case_id"]) in representatives
        markdown.append(
            "| {number} | `{instance}` | {request} | {source_tokens} | {target_tokens} | "
            "`[{source_start},{source_end})` | `[{target_start},{target_end})` | "
            "{length} | {representative} |".format(
                number=number,
                instance=table_escape(case["instance_id"]),
                request=int(case["request_index"]),
                source_tokens=len(source_ids),
                target_tokens=len(target_ids),
                source_start=source_start,
                source_end=source_end,
                target_start=target_start,
                target_end=target_end,
                length=length,
                representative="yes" if representative else "no",
            )
        )
        record = {
            "case_number": number,
            "case_id": case["case_id"],
            "instance_id": case["instance_id"],
            "request_index": int(case["request_index"]),
            "source_input_ids": source_ids,
            "target_input_ids": target_ids,
            "source_prompt": source_prompt,
            "target_prompt": target_prompt,
            "annotated_target_prompt": annotation,
            "reused_text": reused_text,
            "source_start": source_start,
            "source_end": source_end,
            "target_start": target_start,
            "target_end": target_end,
            "copied_tokens": length,
            "rope_position_delta": target_start - source_start,
            "source_token_hash": token_hash(source_ids),
            "target_token_hash": token_hash(target_ids),
            "source_text_sha256": text_hash(source_prompt),
            "target_text_sha256": text_hash(target_prompt),
            "reused_token_hash": token_hash(source_slice_ids),
            "reused_text_sha256": text_hash(reused_text),
            "source_blocks": case["source_blocks"],
            "target_blocks": case["target_blocks"],
            "representative": representative,
        }
        records.append(record)
        index_rows.append(
            {
                "case_number": number,
                "case_id": case["case_id"],
                "instance_id": case["instance_id"],
                "request_index": int(case["request_index"]),
                "source_tokens": len(source_ids),
                "target_tokens": len(target_ids),
                "source_start": source_start,
                "source_end": source_end,
                "target_start": target_start,
                "target_end": target_end,
                "copied_tokens": length,
                "rope_position_delta": target_start - source_start,
                "reused_token_hash": token_hash(source_slice_ids),
                "representative": int(representative),
            }
        )

    markdown.extend(["", "## 逐例完整 Prompt", ""])
    for record in records:
        markdown.extend(
            [
                f"## Case {record['case_number']}: `{record['instance_id']}` / request {record['request_index']}",
                "",
                f"Case ID: `{record['case_id']}`  ",
                f"Source tokens: `{len(record['source_input_ids'])}`; target tokens: `{len(record['target_input_ids'])}`; copied tokens: `{record['copied_tokens']}`  ",
                f"Source interval: `[{record['source_start']}, {record['source_end']})`; target interval: `[{record['target_start']}, {record['target_end']})`; RoPE delta: `{record['rope_position_delta']}`  ",
                f"Reused token hash: `{record['reused_token_hash']}`  ",
                f"Representative: `{'yes' if record['representative'] else 'no'}`",
                "",
                "### Source prompt 切分",
                "",
                block_table(record["source_blocks"]),
                "",
                "### Target prompt 切分",
                "",
                block_table(record["target_blocks"]),
                "",
                "### 实际复用文本",
                "",
                fence(record["reused_text"]),
                "",
                "### 完整 Source prompt（原样 decoded）",
                "",
                fence(record["source_prompt"]),
                "",
                "### 完整 Target prompt（原样 decoded）",
                "",
                fence(record["target_prompt"]),
                "",
                "### 完整 Target prompt（加入运行区间标记）",
                "",
                fence(record["annotated_target_prompt"]),
                "",
            ]
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    output_path.chmod(0o644)
    write_jsonl(data_path, records)
    write_csv(index_path, index_rows)
    summary = {
        "status": "COMPLETE",
        "cases": len(records),
        "source_tokens": sum(len(row["source_input_ids"]) for row in records),
        "target_tokens": sum(len(row["target_input_ids"]) for row in records),
        "copied_tokens": sum(int(row["copied_tokens"]) for row in records),
        "all_reused_token_spans_identical": True,
        "all_block_partitions_complete": True,
        "all_annotated_targets_reconstruct_exact_prompt": True,
        "markdown": str(output_path),
        "jsonl": str(data_path),
        "index_csv": str(index_path),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    args = parser.parse_args()
    summary = build(
        design_path=args.design,
        result_path=args.result,
        output_path=args.output,
        data_path=args.data,
        index_path=args.index,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
