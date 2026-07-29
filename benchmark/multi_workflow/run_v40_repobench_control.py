#!/usr/bin/env python3
"""Run the V40 static projection on the frozen RepoBench-P control.

RepoBench-P does not contain an online coding-agent trajectory.  The V40 arm
therefore uses a deliberately narrow projection of the real policy:

* repository context is treated as a successful read-only observation;
* exactly one largest unique middle island is selected (minimum 128 tokens);
* V is copied, K is position-corrected by the native exact-reuse backend;
* all other target tokens are recomputed and Radix prefix reuse is disabled.

This control tests V40's mechanism and next-line accuracy.  It does not test
file-version invalidation and must not replace the SWE-bench Verified result.
"""

from __future__ import annotations

import argparse
import difflib
import json
import statistics
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from benchmark.multi_workflow.run_bridge_reuse_pilot import (
    generate,
    manifest_case,
    stop_server,
    write_json,
)
from benchmark.multi_workflow.run_coding_native_workload_v10 import (
    flush_cache,
    launch_server,
)


ROOT = Path("/home/gfy/CodeMAS_Project")
MODEL = (
    "/home/gfy/.cache/huggingface/hub/"
    "models--Qwen--Qwen2.5-Coder-3B-Instruct/snapshots/"
    "488639f1ff808d1d3d0ba301aef8c11461451ec5"
)
DEFAULT_WORKLOAD = (
    ROOT
    / "kvflow-artifacts/impactkv_three_method_coding_benchmark_20260728"
    / "repobench-p/WORKLOAD.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "kvflow-artifacts/impactkv_three_method_coding_benchmark_20260728"
    / "repobench-p/v40"
)
ARM = "coding_grounded_observation_island_v40"
SOURCE_PREFIX = (
    "A successful read-only repository command returned the ordered code "
    "context below. Inspect it for a later code-completion request.\n\n"
)
SOURCE_SUFFIX = (
    "\n\nA later request will ask for the next missing line. Do not complete "
    "it in this source-materialization request."
)


def _render(tokenizer: Any, messages: list[dict[str, str]]) -> tuple[str, list[int], list[tuple[int, int]]]:
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    encoded = tokenizer(
        prompt, add_special_tokens=False, return_offsets_mapping=True
    )
    return (
        prompt,
        [int(token) for token in encoded["input_ids"]],
        [tuple(int(value) for value in pair) for pair in encoded["offset_mapping"]],
    )


def _text_span(
    prompt: str,
    offsets: list[tuple[int, int]],
    text: str,
) -> tuple[int, int] | None:
    start = prompt.find(text)
    if start < 0 or prompt.find(text, start + 1) >= 0:
        return None
    end = start + len(text)
    positions = [
        index
        for index, (left, right) in enumerate(offsets)
        if right > left and left >= start and right <= end
    ]
    if not positions or positions != list(range(positions[0], positions[-1] + 1)):
        return None
    return positions[0], len(positions)


def prepare_case(
    tokenizer: Any,
    case: dict[str, Any],
    copy_cap: int = 4096,
) -> dict[str, Any]:
    if copy_cap < 128:
        raise ValueError("copy_cap must be at least 128 tokens")
    reusable = [
        str(segment["text"])
        for segment in case["segments"]
        if bool(segment["reusable"])
    ]
    if not reusable:
        raise ValueError(f"{case['case_id']}: no repository context")
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
    for index, text in enumerate(reusable):
        source_span = _text_span(source_prompt, source_offsets, text)
        target_span = _text_span(target_prompt, target_offsets, text)
        if source_span is None or target_span is None:
            continue
        source_start, source_length = source_span
        target_start, target_length = target_span
        if source_length != target_length:
            continue
        length = min(source_length, copy_cap)
        if length < 128:
            continue
        offset = source_length - length
        span = {
            "source_start": source_start + offset,
            "target_start": target_start + offset,
            "length": length,
        }
        if (
            source_ids[span["source_start"] : span["source_start"] + length]
            != target_ids[span["target_start"] : span["target_start"] + length]
        ):
            continue
        candidates.append((length, index, span))
    if not candidates:
        raise ValueError(f"{case['case_id']}: no V40-eligible unique island")
    # V40 selects the largest observation and lets the newest win ties.
    _, selected_index, span = max(candidates, key=lambda row: (row[0], row[1]))
    if span["source_start"] <= 0 or span["target_start"] <= 0:
        raise ValueError(f"{case['case_id']}: selected span is not a middle island")
    if span["source_start"] + span["length"] >= len(source_ids):
        raise ValueError(f"{case['case_id']}: source span reaches prompt end")
    if span["target_start"] + span["length"] >= len(target_ids):
        raise ValueError(f"{case['case_id']}: target span reaches prompt end")
    return {
        "answers": list(case["metadata"]["answers"]),
        "case_id": str(case["case_id"]),
        "max_new_tokens": int(case["max_new_tokens"]),
        "selected_context_index": selected_index,
        "selected_tokens": span["length"],
        "source_input_ids": source_ids,
        "source_start": span["source_start"],
        "target_input_ids": target_ids,
        "target_start": span["target_start"],
    }


def prepare(
    workload_path: Path,
    output: Path,
    limit: int,
    copy_cap: int = 4096,
) -> dict[str, Any]:
    workload = json.loads(workload_path.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    source_cases = workload["cases"]
    if limit > 0:
        source_cases = source_cases[:limit]
    cases = [
        prepare_case(tokenizer, case, copy_cap=copy_cap)
        for case in source_cases
    ]
    manifest_rows = []
    for case in cases:
        row = manifest_case(
            case_id=case["case_id"],
            policy_label=ARM,
            source_ids=case["source_input_ids"],
            target_ids=case["target_input_ids"],
            span={
                "source_start": case["source_start"],
                "target_start": case["target_start"],
                "length": case["selected_tokens"],
            },
        )
        row["target_uses"] = 1
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
        "status": "PREPARED_BEFORE_V40_REPOBENCH_GPU",
        "arm": ARM,
        "cases": len(cases),
        "dataset": "RepoBench-P",
        "model": MODEL,
        "copy_cap": copy_cap,
        "projection_limitations": [
            "no rolling agent trajectory",
            "no file mutation or file-version invalidation",
            "repository chunks are treated as successful read-only observations",
        ],
        "selection": (
            "one largest unique repository-context island, at least 128 "
            f"and at most {copy_cap} tokens, newest wins ties"
        ),
        "target_prompt_identity": "exact WORKLOAD.json messages",
        "prefetch": False,
    }
    write_json(output / "REGISTRATION.json", registration)
    return registration


def run_arm(output: Path, arm: str, port: int) -> dict[str, Any]:
    if arm not in {"dense", ARM}:
        raise ValueError(arm)
    cases = json.loads((output / "CASES.json").read_text())["cases"]
    result_path = output / f"{arm}.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    process, stream, base_url = launch_server(output=output, arm=arm, port=port)
    sources = []
    targets = []
    try:
        generate(
            base_url=base_url,
            input_ids=cases[0]["target_input_ids"][:128],
            key=f"v40-repobench-shape-{arm}",
            max_new_tokens=1,
            stream=True,
        )
        flush_cache(base_url)
        for case in cases:
            flush_cache(base_url)
            if arm != "dense":
                sources.append(
                    {
                        **generate(
                            base_url=base_url,
                            input_ids=case["source_input_ids"],
                            key=f"v40-repobench-source-{case['case_id']}",
                            max_new_tokens=1,
                            stream=False,
                        ),
                        "case_id": case["case_id"],
                    }
                )
            targets.append(
                {
                    **generate(
                        base_url=base_url,
                        input_ids=case["target_input_ids"],
                        key=f"v40-repobench-target-{case['case_id']}",
                        max_new_tokens=case["max_new_tokens"],
                        stream=True,
                    ),
                    "case_id": case["case_id"],
                }
            )
    finally:
        stop_server(process, stream)
    ledger = output / "server" / arm / "EXACT_LEDGER.jsonl"
    ledger_rows = (
        [
            json.loads(line)
            for line in ledger.read_text().splitlines()
            if line.strip()
        ]
        if ledger.exists()
        else []
    )
    value = {
        "arm": arm,
        "sources": sources,
        "targets": targets,
        "ledger_rows": ledger_rows,
    }
    write_json(result_path, value)
    return {
        "arm": arm,
        "targets": len(targets),
        "copy_events": sum(
            row.get("event") == "target_copied" for row in ledger_rows
        ),
        "fallback_events": sum(
            row.get("event") == "target_fallback" for row in ledger_rows
        ),
    }


def _prediction_line(text: str) -> str:
    for line in text.lstrip("\n").splitlines():
        if "`" not in line and "#" not in line and "//" not in line:
            return line
    return ""


def summarize(output: Path) -> dict[str, Any]:
    cases = {
        case["case_id"]: case
        for case in json.loads((output / "CASES.json").read_text())["cases"]
    }
    dense = {
        row["case_id"]: row
        for row in json.loads((output / "dense.json").read_text())["targets"]
    }
    reuse_value = json.loads((output / f"{ARM}.json").read_text())
    reuse = {row["case_id"]: row for row in reuse_value["targets"]}
    sources = {
        row["case_id"]: row for row in reuse_value["sources"]
    }
    if set(cases) != set(dense) or set(cases) != set(reuse):
        raise ValueError("Dense/reuse target coverage differs from frozen cases")
    rows = []
    for case_id, case in cases.items():
        answer = str(case["answers"][0])
        dense_line = _prediction_line(
            str(dense[case_id].get("output_text", ""))
        )
        reuse_line = _prediction_line(
            str(reuse[case_id].get("output_text", ""))
        )
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
            }
        )
    count = len(rows)
    dense_mean = statistics.fmean(row["dense_ttft_ms"] for row in rows)
    reuse_mean = statistics.fmean(row["reuse_ttft_ms"] for row in rows)
    build_mean = statistics.fmean(row["source_build_ms"] for row in rows)
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
    value = {
        "status": (
            "COMPLETE"
            if len(copies) == count and not fallbacks
            else "MECHANISM_FAILURE"
        ),
        "method": ARM,
        "dataset": "RepoBench-P",
        "samples": count,
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
            "n4_including_build_speedup": dense_mean
            / (reuse_mean + build_mean / 4),
        },
        "physical_reuse": {
            "copy_events": len(copies),
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
    prepare_parser.add_argument("--limit", type=int, default=0)
    prepare_parser.add_argument("--copy-cap", type=int, default=4096)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run_parser.add_argument("--arm", choices=("dense", ARM), required=True)
    run_parser.add_argument("--port", type=int, default=31100)
    summary_parser = subparsers.add_parser("summarize")
    summary_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.command == "prepare":
        value = prepare(
            args.workload,
            args.output,
            args.limit,
            copy_cap=args.copy_cap,
        )
    elif args.command == "run":
        value = run_arm(args.output, args.arm, args.port)
    else:
        value = summarize(args.output)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
