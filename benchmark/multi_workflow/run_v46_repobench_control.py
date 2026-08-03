#!/usr/bin/env python3
"""Run a static three-island V46 mechanism/speed control on RepoBench-P."""

from __future__ import annotations

import argparse
import difflib
import json
import statistics
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from benchmark.multi_workflow.run_bridge_reuse_pilot import (
    manifest_case,
    write_json,
)
from benchmark.multi_workflow.run_v40_repobench_control import (
    DEFAULT_WORKLOAD,
    MODEL,
    SOURCE_PREFIX,
    SOURCE_SUFFIX,
    _prediction_line,
    _render,
    _text_span,
    run_arm,
)


ROOT = Path("/home/gfy/CodeMAS_Project")
DEFAULT_OUTPUT = (
    ROOT
    / "kvflow-artifacts/impactkv_v46_observed_path_runtime_20260803"
    / "repobench_three_island"
)
ARM = "coding_observed_path_pool_v46"


def prepare_case(tokenizer: Any, case: dict[str, Any]) -> dict[str, Any]:
    reusable = [
        str(segment["text"])
        for segment in case["segments"]
        if bool(segment["reusable"])
    ]
    repository_text = "".join(reusable)
    source_messages = [
        {
            "role": "system",
            "content": (
                "You inspect repository code returned by read-only tools. "
                "Return only a one-word acknowledgement."
            ),
        },
        {
            "role": "user",
            "content": SOURCE_PREFIX + repository_text + SOURCE_SUFFIX,
        },
    ]
    source_prompt, source_ids, source_offsets = _render(
        tokenizer, source_messages
    )
    target_prompt, target_ids, target_offsets = _render(
        tokenizer, case["messages"]
    )
    candidates = []
    for context_index, text in enumerate(reusable):
        source_span = _text_span(source_prompt, source_offsets, text)
        target_span = _text_span(target_prompt, target_offsets, text)
        if source_span is None or target_span is None:
            continue
        source_start, source_length = source_span
        target_start, target_length = target_span
        if source_length != target_length:
            continue
        length = min(source_length, 4096)
        if length < 128:
            continue
        offset = source_length - length
        span = {
            "source_start": source_start + offset,
            "target_start": target_start + offset,
            "length": length,
            "context_index": context_index,
        }
        if (
            source_ids[span["source_start"] : span["source_start"] + length]
            != target_ids[span["target_start"] : span["target_start"] + length]
        ):
            continue
        candidates.append(span)
    selected = sorted(
        sorted(
            candidates,
            key=lambda row: (row["length"], row["context_index"]),
            reverse=True,
        )[:3],
        key=lambda row: row["target_start"],
    )
    if len(selected) != 3:
        raise ValueError(f"{case['case_id']}: fewer than three eligible islands")
    for left, right in zip(selected, selected[1:]):
        if left["target_start"] + left["length"] > right["target_start"]:
            raise ValueError(f"{case['case_id']}: selected islands overlap")
    return {
        "answers": list(case["metadata"]["answers"]),
        "case_id": str(case["case_id"]),
        "max_new_tokens": int(case["max_new_tokens"]),
        "selected_islands": selected,
        "selected_tokens": sum(row["length"] for row in selected),
        "source_input_ids": source_ids,
        "target_input_ids": target_ids,
    }


def prepare(workload_path: Path, output: Path, limit: int) -> dict[str, Any]:
    workload = json.loads(workload_path.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    source_cases = workload["cases"][:limit] if limit > 0 else workload["cases"]
    cases = [prepare_case(tokenizer, case) for case in source_cases]
    manifest_rows = []
    for case in cases:
        target_group_id = f"v46-static-{case['case_id']}"
        for island_index, span in enumerate(case["selected_islands"]):
            row = manifest_case(
                case_id=f"{case['case_id']}-i{island_index}",
                policy_label=ARM,
                source_ids=case["source_input_ids"],
                target_ids=case["target_input_ids"],
                span=span,
            )
            row.update(
                source_id=f"{case['case_id']}-source-i{island_index}",
                target_group_id=target_group_id,
                target_uses=1,
            )
            manifest_rows.append(row)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "CASES.json", {"cases": cases})
    write_json(
        output / "manifests" / f"{ARM}.json",
        {
            "cache_dtype": "bfloat16",
            "cases": manifest_rows,
            "lease_ttl_s": 900,
            "ledger_path": str(
                output / "server" / ARM / "EXACT_LEDGER.jsonl"
            ),
            "model_id": MODEL,
            "ordinary_prefix_reuse_enabled": False,
            "rope": {
                "base": 1_000_000,
                "is_neox_style": True,
                "rotary_dim": 128,
            },
            "version": 2,
        },
    )
    registration = {
        "status": "REGISTERED_BEFORE_GPU",
        "arm": ARM,
        "cases": len(cases),
        "target_islands_per_case": 3,
        "dataset": "RepoBench-P",
        "model": MODEL,
        "selection": (
            "three largest non-overlapping unique repository-context islands, "
            "128--4096 tokens each"
        ),
        "gates": {
            "copy_events": len(cases) * 3,
            "fallback_events_max": 0,
            "dense_and_reuse_target_count_equal": True,
        },
        "ordinary_prefix_reuse": False,
        "prefetch": False,
        "quality_claimed": "RepoBench next-line control only",
    }
    write_json(output / "REGISTRATION.json", registration)
    return registration


def summarize(output: Path, dense_result: Path | None = None) -> dict[str, Any]:
    cases = {
        case["case_id"]: case
        for case in json.loads((output / "CASES.json").read_text())["cases"]
    }
    dense = {
        row["case_id"]: row
        for row in json.loads(
            (dense_result or output / "dense.json").read_text()
        )["targets"]
    }
    reuse_value = json.loads((output / f"{ARM}.json").read_text())
    reuse = {row["case_id"]: row for row in reuse_value["targets"]}
    sources = {row["case_id"]: row for row in reuse_value["sources"]}
    rows = []
    for case_id, case in cases.items():
        answer = str(case["answers"][0])
        dense_line = _prediction_line(str(dense[case_id].get("output_text", "")))
        reuse_line = _prediction_line(str(reuse[case_id].get("output_text", "")))
        rows.append(
            {
                "case_id": case_id,
                "answer": answer,
                "dense_prediction_line": dense_line,
                "reuse_prediction_line": reuse_line,
                "dense_exact_line": dense_line.strip() == answer.strip(),
                "reuse_exact_line": reuse_line.strip() == answer.strip(),
                "dense_code_sim": difflib.SequenceMatcher(
                    None, dense_line, answer
                ).ratio(),
                "reuse_code_sim": difflib.SequenceMatcher(
                    None, reuse_line, answer
                ).ratio(),
                "dense_ttft_ms": float(dense[case_id]["ttft_ms"]),
                "reuse_ttft_ms": float(reuse[case_id]["ttft_ms"]),
                "source_build_ms": float(sources[case_id]["elapsed_ms"]),
                "selected_tokens": int(case["selected_tokens"]),
                "target_islands": len(case["selected_islands"]),
            }
        )
    copies = [
        row
        for row in reuse_value["ledger_rows"]
        if row.get("event") == "target_copied"
    ]
    fallbacks = [
        row
        for row in reuse_value["ledger_rows"]
        if row.get("event") == "target_fallback"
    ]
    dense_mean = statistics.fmean(row["dense_ttft_ms"] for row in rows)
    reuse_mean = statistics.fmean(row["reuse_ttft_ms"] for row in rows)
    build_mean = statistics.fmean(row["source_build_ms"] for row in rows)
    expected_copies = sum(row["target_islands"] for row in rows)
    value = {
        "status": (
            "COMPLETE"
            if len(copies) == expected_copies and not fallbacks
            else "MECHANISM_FAILURE"
        ),
        "method": ARM,
        "dataset": "RepoBench-P",
        "samples": len(rows),
        "quality": {
            "dense_exact_line": sum(row["dense_exact_line"] for row in rows),
            "reuse_exact_line": sum(row["reuse_exact_line"] for row in rows),
            "dense_code_sim_percent": 100
            * statistics.fmean(row["dense_code_sim"] for row in rows),
            "reuse_code_sim_percent": 100
            * statistics.fmean(row["reuse_code_sim"] for row in rows),
        },
        "latency": {
            "dense_mean_ttft_ms": dense_mean,
            "reuse_mean_ttft_ms": reuse_mean,
            "cache_ready_speedup": dense_mean / reuse_mean,
            "mean_source_build_ms": build_mean,
            "n4_including_build_speedup": dense_mean / (reuse_mean + build_mean / 4),
        },
        "physical_reuse": {
            "copy_events": len(copies),
            "expected_copy_events": expected_copies,
            "fallback_events": len(fallbacks),
            "mean_selected_tokens": statistics.fmean(
                row["selected_tokens"] for row in rows
            ),
        },
        "rows": rows,
    }
    write_json(output / "RESULT.json", value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--workload", type=Path, default=DEFAULT_WORKLOAD)
    prepare_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    prepare_parser.add_argument("--limit", type=int, default=3)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run_parser.add_argument("--arm", required=True)
    run_parser.add_argument("--port", type=int, default=31100)
    summary_parser = subparsers.add_parser("summarize")
    summary_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    summary_parser.add_argument("--dense-result", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        value = prepare(args.workload, args.output, args.limit)
    elif args.command == "run":
        value = run_arm(
            args.output,
            args.arm,
            args.port,
            reuse_arm=ARM,
        )
    else:
        value = summarize(args.output, args.dense_result)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
