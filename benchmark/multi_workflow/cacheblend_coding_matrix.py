#!/usr/bin/env python3
"""Prepare and summarize the native CacheBlend coding comparison lane.

CacheBlend's public artifact is based on vLLM 0.4.1 and cannot load the
Qwen3-8B model used by the QCFuse common-stack lane.  This adapter therefore
keeps the exact retained LCC/RepoBench-P text and sample IDs, runs CacheBlend
with its validated native Qwen2.5-Coder model, and reports reuse only relative
to a matching CacheBlend Dense run.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import statistics
from pathlib import Path
from typing import Iterable


SYSTEM_PROMPT = (
    "You are a precise code-completion assistant. The following text is ordered "
    "code context. Complete exactly the next missing line. Return only that "
    "single line, without Markdown or explanation."
)
AMORTIZATION_COUNTS = (1, 4, 16)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def _answers(row: dict) -> list[str]:
    answers = row.get("answers", [])
    if isinstance(answers, str):
        answers = [answers]
    if not answers:
        raise ValueError("coding row has no answer")
    return [str(answer) for answer in answers]


def prepare_case(row: dict, dataset: str, index: int) -> dict:
    contexts = row.get("context")
    if not isinstance(contexts, list) or not contexts:
        raise ValueError(f"{dataset}[{index}] has no reusable context chunks")
    context_texts = [str(text) for text in contexts if str(text)]
    query = str(row.get("input", ""))
    if not query:
        raise ValueError(f"{dataset}[{index}] has an empty online query")

    user_text = "".join(context_texts) + query
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    instance_id = str(row.get("_id", f"{dataset}-{index}"))
    segments = []
    for chunk_index, text in enumerate(context_texts):
        segments.append(
            {
                "kind": "ordered_code_context",
                "reusable": True,
                "segment_id": f"{instance_id}:context:{chunk_index}",
                "source_position": chunk_index,
                "target_position": chunk_index,
                "text": text,
            }
        )
    segments.append(
        {
            "kind": "online_code_query",
            "reusable": False,
            "segment_id": f"{instance_id}:query",
            "source_position": None,
            "target_position": None,
            "text": query,
        }
    )
    return {
        "case_id": instance_id,
        "max_new_tokens": 64,
        "messages": messages,
        "metadata": {
            "answers": _answers(row),
            "language": row.get("language"),
            "original_source_index": row.get("_qcfuse_coding", {}).get(
                "source_index"
            ),
            "retained_context_chunks": len(context_texts),
            "source_preparation": row.get("_qcfuse_coding", {}),
        },
        "prompt_sha256": _canonical_sha256(messages),
        "segments": segments,
        "source_id": f"longbench:{dataset}:{instance_id}",
        "split": "formal",
        "suite": dataset,
    }


def prepare_workload(
    source: Path,
    dataset: str,
    limit: int,
    case_ids: list[str] | None = None,
) -> dict:
    rows = _read_jsonl(source)
    if case_ids is not None:
        indexed = {
            str(row.get("_id", f"{dataset}-{index}")): row
            for index, row in enumerate(rows)
        }
        missing = set(case_ids).difference(indexed)
        if missing:
            raise ValueError(
                f"{dataset} is missing requested case IDs: {sorted(missing)}"
            )
        rows = [indexed[case_id] for case_id in case_ids]
    elif limit > 0:
        rows = rows[:limit]
    cases = [prepare_case(row, dataset, index) for index, row in enumerate(rows)]
    case_ids = [case["case_id"] for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError(f"{dataset} contains duplicate case IDs")
    return {
        "schema_version": 1,
        "adapter": "cacheblend-native-coding-v1",
        "cases": cases,
        "dataset": dataset,
        "model": "Qwen/Qwen2.5-Coder-3B-Instruct",
        "protocol": {
            "claim_scope": (
                "CacheBlend reuse versus CacheBlend native Dense on identical "
                "retained source text; cross-engine absolute values are descriptive"
            ),
            "generation": {"temperature": 0, "max_new_tokens": 64},
            "reusable_region": "all retained ordered context chunks",
            "online_region": "official code-completion query",
            "source_text_sha256": _file_sha256(source),
        },
    }


def _prediction_line(text: str) -> str:
    for line in text.lstrip("\n").splitlines():
        if "`" not in line and "#" not in line and "//" not in line:
            return line
    return ""


def _code_similarity(prediction: str, answer: str) -> float:
    similarity = difflib.SequenceMatcher(None, prediction, answer).ratio()
    return round(100 * similarity) / 100


def _last_records(records: Iterable[dict]) -> dict[str, dict]:
    by_case = {}
    for record in records:
        if record.get("metadata", {}).get("warmup"):
            continue
        case_id = str(record.get("case_id", ""))
        if not case_id:
            raise ValueError("CacheBlend record has no case_id")
        by_case[case_id] = record
    return by_case


def _mean(values: Iterable[float]) -> float:
    materialized = [float(value) for value in values]
    if not materialized:
        raise ValueError("metric has no values")
    return statistics.fmean(materialized)


def summarize(
    workload: dict,
    dense_records: list[dict],
    reuse_records: list[dict],
    recompute_ratio: float,
) -> dict:
    truth = {
        case["case_id"]: str(case["metadata"]["answers"][0])
        for case in workload["cases"]
    }
    dense = _last_records(dense_records)
    reuse = _last_records(reuse_records)
    expected = set(truth)
    if set(dense) != expected or set(reuse) != expected:
        raise ValueError(
            "Dense/reuse records do not cover exactly the frozen workload: "
            f"dense_missing={len(expected - set(dense))}, "
            f"reuse_missing={len(expected - set(reuse))}"
        )

    rows = []
    for case_id in truth:
        dense_record = dense[case_id]
        reuse_record = reuse[case_id]
        for label, record in (("dense", dense_record), ("reuse", reuse_record)):
            if record.get("error"):
                raise ValueError(f"{case_id} {label} failed: {record['error']}")
            if record.get("ttft_ms") is None:
                raise ValueError(f"{case_id} {label} has no TTFT")
        answer = truth[case_id]
        dense_line = _prediction_line(
            str(dense_record.get("metadata", {}).get("output_text", ""))
        )
        reuse_line = _prediction_line(
            str(reuse_record.get("metadata", {}).get("output_text", ""))
        )
        rows.append(
            {
                "case_id": case_id,
                "answer": answer,
                "dense_prediction_line": dense_line,
                "reuse_prediction_line": reuse_line,
                "dense_code_sim": _code_similarity(dense_line, answer),
                "reuse_code_sim": _code_similarity(reuse_line, answer),
                "dense_exact_line": bool(
                    dense_line.strip() and dense_line.strip() == answer.strip()
                ),
                "reuse_exact_line": bool(
                    reuse_line.strip() and reuse_line.strip() == answer.strip()
                ),
                "dense_ttft_ms": float(dense_record["ttft_ms"]),
                "reuse_ttft_ms": float(reuse_record["ttft_ms"]),
                "cache_build_ms": float(reuse_record.get("cache_build_ms") or 0.0),
                "reused_k_tokens": int(reuse_record.get("reused_k_tokens") or 0),
                "reused_v_tokens": int(reuse_record.get("reused_v_tokens") or 0),
                "recomputed_tokens": int(
                    reuse_record.get("recomputed_tokens") or 0
                ),
            }
        )

    count = len(rows)
    dense_exact = sum(row["dense_exact_line"] for row in rows)
    reuse_exact = sum(row["reuse_exact_line"] for row in rows)
    dense_ttft = _mean(row["dense_ttft_ms"] for row in rows)
    reuse_ttft = _mean(row["reuse_ttft_ms"] for row in rows)
    build_ms = _mean(row["cache_build_ms"] for row in rows)
    amortized = {}
    for reuse_count in AMORTIZATION_COUNTS:
        denominator = reuse_ttft + build_ms / reuse_count
        amortized[str(reuse_count)] = {
            "ttft_ms": denominator,
            "speedup_vs_native_dense": dense_ttft / denominator,
        }

    return {
        "schema_version": 1,
        "method": "CacheBlend",
        "engine": "vLLM-Blend 0.4.1",
        "model": workload["model"],
        "dataset": workload["dataset"],
        "samples": count,
        "recompute_ratio": recompute_ratio,
        "claim_scope": workload["protocol"]["claim_scope"],
        "quality": {
            "dense_code_sim_percent": 100
            * _mean(row["dense_code_sim"] for row in rows),
            "reuse_code_sim_percent": 100
            * _mean(row["reuse_code_sim"] for row in rows),
            "dense_exact_line": dense_exact,
            "reuse_exact_line": reuse_exact,
            "dense_exact_line_percent": 100 * dense_exact / count,
            "reuse_exact_line_percent": 100 * reuse_exact / count,
            "exact_line_delta_pp": 100 * (reuse_exact - dense_exact) / count,
            "dense_pass_reuse_fail": sum(
                row["dense_exact_line"] and not row["reuse_exact_line"]
                for row in rows
            ),
            "dense_fail_reuse_pass": sum(
                not row["dense_exact_line"] and row["reuse_exact_line"]
                for row in rows
            ),
        },
        "latency": {
            "dense_cache_ready_mean_ttft_ms": dense_ttft,
            "reuse_cache_ready_mean_ttft_ms": reuse_ttft,
            "cache_ready_speedup_vs_native_dense": dense_ttft / reuse_ttft,
            "mean_cache_build_ms": build_ms,
            "build_amortized": amortized,
        },
        "physical_reuse": {
            "mean_reused_k_tokens": _mean(
                row["reused_k_tokens"] for row in rows
            ),
            "mean_reused_v_tokens": _mean(
                row["reused_v_tokens"] for row in rows
            ),
            "mean_recomputed_tokens": _mean(
                row["recomputed_tokens"] for row in rows
            ),
        },
        "rows": rows,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--source", type=Path, required=True)
    prepare.add_argument("--dataset", choices=("lcc", "repobench-p"), required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--limit", type=int, default=200)
    prepare.add_argument(
        "--registration",
        type=Path,
        help=(
            "Use the frozen repobench_p_control task IDs from the narrowed "
            "three-method registration, in registered order."
        ),
    )

    score = subparsers.add_parser("summarize")
    score.add_argument("--workload", type=Path, required=True)
    score.add_argument("--dense", type=Path, required=True)
    score.add_argument("--reuse", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--recompute-ratio", type=float, default=0.5)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "prepare":
        case_ids = None
        if args.registration is not None:
            if args.dataset != "repobench-p":
                raise ValueError("--registration currently applies to RepoBench-P")
            frozen = json.loads(
                args.registration.read_text(encoding="utf-8")
            )
            case_ids = list(
                frozen["datasets"]["repobench_p_control"]["task_ids"]
            )
        result = prepare_workload(
            args.source, args.dataset, args.limit, case_ids=case_ids
        )
    else:
        result = summarize(
            json.loads(args.workload.read_text(encoding="utf-8")),
            _read_jsonl(args.dense),
            _read_jsonl(args.reuse),
            args.recompute_ratio,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
