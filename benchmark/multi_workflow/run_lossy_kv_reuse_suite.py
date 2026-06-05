#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
from collections import Counter
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from bench_lossy_kv_reuse import (
    build_local_match_result,
    build_request_payload,
    do_request,
    summarize_payload,
)


BASE_CODE = """from typing import List

def count_up_to(n: int) -> List[int]:
    result = []
    for value in range(2, n):
        is_prime = True
        for factor in range(2, int(value ** 0.5) + 1):
            if value % factor == 0:
                is_prime = False
                break
        if is_prime:
            result.append(value)
    return result
"""


def build_cases() -> list[dict[str, str]]:
    return [
        {"case_id": "exact_same", "candidate_code": BASE_CODE},
        {
            "case_id": "rename_variables",
            "candidate_code": BASE_CODE.replace("result", "primes").replace("value", "candidate"),
        },
        {
            "case_id": "comment_only",
            "candidate_code": "# Count primes below n\n" + BASE_CODE,
        },
        {
            "case_id": "add_helper",
            "candidate_code": "def _identity(x):\n    return x\n\n" + BASE_CODE,
        },
        {
            "case_id": "structure_rewrite",
            "candidate_code": """from typing import List

def count_up_to(n: int) -> List[int]:
    def is_prime(value: int) -> bool:
        if value < 2:
            return False
        for factor in range(2, int(value ** 0.5) + 1):
            if value % factor == 0:
                return False
        return True

    return [value for value in range(2, n) if is_prime(value)]
""",
        },
        {
            "case_id": "different_function",
            "candidate_code": """def reverse_words(text: str) -> str:
    return " ".join(reversed(text.split()))
""",
        },
    ]


async def run_case(args: argparse.Namespace, case: dict[str, str]) -> dict[str, object]:
    warmup_payload = build_request_payload(
        model_path=args.model_path,
        task=args.task,
        code_text=BASE_CODE,
        max_tokens=args.max_tokens,
        reuse_mode="lossless",
        lossy_alignment_method=args.lossy_alignment_method,
        template_task_family=args.template_task_family,
        template_workflow_signature=args.template_workflow_signature,
        template_structural_fingerprint=args.template_structural_fingerprint,
    )
    eval_payload = build_request_payload(
        model_path=args.model_path,
        task=args.task,
        code_text=case["candidate_code"],
        max_tokens=args.max_tokens,
        reuse_mode="lossy",
        lossy_alignment_method=args.lossy_alignment_method,
        template_task_family=args.template_task_family,
        template_workflow_signature=args.template_workflow_signature,
        template_structural_fingerprint=args.template_structural_fingerprint,
    )
    baseline_eval_payload = dict(eval_payload)
    baseline_eval_payload["reuse_mode"] = "lossless"
    local_match = build_local_match_result(warmup_payload, eval_payload)
    warmup_result = None
    baseline_eval_result = None
    eval_result = None
    if not args.skip_http:
        warmup_result = await do_request(args.base_url, warmup_payload)
        eval_result = await do_request(args.base_url, eval_payload)
        baseline_eval_result = await do_request(args.base_url, baseline_eval_payload)
    return {
        "case_id": case["case_id"],
        "warmup_request": summarize_payload(warmup_payload),
        "baseline_eval_request": summarize_payload(baseline_eval_payload),
        "eval_request": summarize_payload(eval_payload),
        "local_match": {
            "reuse_allowed": local_match.reuse_allowed,
            "reuse_confidence": local_match.reuse_confidence,
            "matched_anchor_signature": local_match.matched_anchor_signature,
            "syntax_region_type": local_match.syntax_region_type,
            "match_reason": local_match.match_reason,
            "rejected_reason": local_match.rejected_reason,
        },
        "warmup_result": warmup_result,
        "baseline_eval_result": baseline_eval_result,
        "eval_result": eval_result,
    }


def write_summary(output_dir: Path, records: list[dict[str, object]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "suite_summary.json"
    csv_path = output_dir / "suite_summary.csv"
    md_path = output_dir / "suite_summary.md"

    accepted = sum(1 for record in records if record["local_match"]["reuse_allowed"])
    rejected = len(records) - accepted
    reject_histogram = Counter(
        str(record["local_match"]["rejected_reason"] or "")
        for record in records
        if not record["local_match"]["reuse_allowed"]
    )
    match_histogram = Counter(
        str(record["local_match"]["match_reason"] or "")
        for record in records
        if record["local_match"]["reuse_allowed"]
    )
    server_match_histogram = Counter(
        str(
            ((record.get("eval_result") or {}).get("body") or {})
            .get("metadata", {})
            .get("lossy_reuse", {})
            .get("lossy_first_match_reason", "")
        ) or ""
        for record in records
    )
    server_match_histogram.pop("", None)
    server_reject_histogram = Counter(
        str(
            ((record.get("eval_result") or {}).get("body") or {})
            .get("metadata", {})
            .get("lossy_reuse", {})
            .get("lossy_first_rejected_reason", "")
        ) or ""
        for record in records
    )
    server_reject_histogram.pop("", None)
    server_candidate_histogram = Counter(
        str(
            ((record.get("eval_result") or {}).get("body") or {})
            .get("metadata", {})
            .get("lossy_reuse", {})
            .get("lossy_candidate_count", "")
        )
        for record in records
        if ((record.get("eval_result") or {}).get("body") or {}).get("metadata", {}).get("lossy_reuse", {}).get(
            "lossy_candidate_count"
        ) is not None
    )
    summary = {
        "cases": records,
        "accepted_reuse_count": accepted,
        "rejected_reuse_count": rejected,
        "reject_reason_histogram": dict(reject_histogram),
        "match_reason_histogram": dict(match_histogram),
        "server_match_reason_histogram": dict(server_match_histogram),
        "server_reject_reason_histogram": dict(server_reject_histogram),
        "server_candidate_count_histogram": dict(server_candidate_histogram),
    }
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "case_id",
                "reuse_allowed",
                "reuse_confidence",
                "match_reason",
                "rejected_reason",
                "server_match_reason",
                "server_rejected_reason",
                "server_candidate_count",
                "server_reuse_confidence",
                "baseline_status",
                "baseline_elapsed_ms",
                "eval_status",
                "eval_elapsed_ms",
            ]
        )
        for record in records:
            baseline_result = record.get("baseline_eval_result") or {}
            eval_result = record.get("eval_result") or {}
            server_lossy = ((eval_result.get("body") or {}).get("metadata") or {}).get(
                "lossy_reuse", {}
            )
            writer.writerow(
                [
                    record["case_id"],
                    record["local_match"]["reuse_allowed"],
                    record["local_match"]["reuse_confidence"],
                    record["local_match"]["match_reason"],
                    record["local_match"]["rejected_reason"],
                    server_lossy.get("lossy_first_match_reason"),
                    server_lossy.get("lossy_first_rejected_reason"),
                    server_lossy.get("lossy_candidate_count"),
                    server_lossy.get("lossy_reuse_confidence"),
                    baseline_result.get("status"),
                    baseline_result.get("elapsed_ms"),
                    eval_result.get("status"),
                    eval_result.get("elapsed_ms"),
                ]
            )

    lines = [
        "# Lossy KV Reuse Suite",
        "",
        f"- accepted_reuse_count: {accepted}",
        f"- rejected_reuse_count: {rejected}",
        f"- server_match_reason_histogram: {dict(server_match_histogram)}",
        f"- server_reject_reason_histogram: {dict(server_reject_histogram)}",
        f"- server_candidate_count_histogram: {dict(server_candidate_histogram)}",
        "",
        "| case_id | reuse_allowed | reuse_confidence | match_reason | server_match | server_reject | server_candidates | baseline_ms | lossy_ms |",
        "|---|---:|---:|---|---|---|---:|---:|---:|",
    ]
    for record in records:
        baseline_result = record.get("baseline_eval_result") or {}
        eval_result = record.get("eval_result") or {}
        server_lossy = ((eval_result.get("body") or {}).get("metadata") or {}).get(
            "lossy_reuse", {}
        )
        lines.append(
            f"| {record['case_id']} | {record['local_match']['reuse_allowed']} | "
            f"{record['local_match']['reuse_confidence']} | {record['local_match']['match_reason']} | "
            f"{server_lossy.get('lossy_first_match_reason')} | {server_lossy.get('lossy_first_rejected_reason')} | "
            f"{server_lossy.get('lossy_candidate_count')} | {baseline_result.get('elapsed_ms')} | {eval_result.get('elapsed_ms')} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def main_async(args: argparse.Namespace) -> int:
    records = []
    for case in build_cases():
        records.append(await run_case(args, case))
    write_summary(Path(args.output_dir), records)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small Python-first lossy KV reuse suite")
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model-path", default="/home/gfy/models/Qwen2.5-7B-Instruct")
    parser.add_argument("--task", default="Summarize the function and suggest one safe refactor.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--lossy-alignment-method", default="kvcomm")
    parser.add_argument("--template-task-family", default="code_generation")
    parser.add_argument("--template-workflow-signature", default="agents=planner,implementer,reviewer")
    parser.add_argument("--template-structural-fingerprint", default="loop_for")
    parser.add_argument("--skip-http", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main_async(parse_args())))
