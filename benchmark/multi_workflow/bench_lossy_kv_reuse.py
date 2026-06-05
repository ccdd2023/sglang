#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import aiohttp

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MASCODER_SRC = PROJECT_ROOT / "MAScoder" / "src"
SGLANG_PYTHON = PROJECT_ROOT / "sglang-kvflow" / "python"
for entry in (str(MASCODER_SRC), str(SGLANG_PYTHON)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from mascoder.code_anchor import build_code_anchor_payload
from sglang.srt.mem_cache.anchor_match import (
    AnchorMatchResult,
    build_anchor_metadata,
    match_request_to_candidate,
)


def build_messages(shared_prefix: str, code_context: str, task: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": shared_prefix},
        {"role": "system", "content": code_context},
        {"role": "user", "content": task},
    ]


async def do_request(base_url: str, payload: dict) -> dict:
    started = time.perf_counter()
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180)) as session:
        async with session.post(f"{base_url}/v1/chat/completions", json=payload) as resp:
            body = await resp.json()
    elapsed_ms = (time.perf_counter() - started) * 1000
    return {"status": resp.status, "elapsed_ms": elapsed_ms, "body": body}


def load_code_text(code_file: str, fallback: str) -> str:
    if code_file:
        return Path(code_file).read_text(encoding="utf-8")
    return fallback


def build_request_payload(
    *,
    model_path: str,
    task: str,
    code_text: str,
    max_tokens: int,
    reuse_mode: str,
    lossy_alignment_method: str,
    template_task_family: str,
    template_workflow_signature: str,
    template_structural_fingerprint: str,
) -> dict:
    anchor_payload = build_code_anchor_payload(code_text, language="python")
    messages = build_messages(
        "You are a coding assistant. Reuse stable code anchors when safe.",
        code_text,
        task,
    )
    return {
        "model": model_path,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "priority": 1,
        "role_type": 2,
        "critical_path_distance": 1,
        "code_anchor_signature": anchor_payload.get("ast_anchor_signature", ""),
        "code_content_signature": anchor_payload.get("code_content_signature", ""),
        "code_anchor_spans": anchor_payload.get("code_anchor_spans", []),
        "reuse_mode": reuse_mode,
        "lossy_alignment_method": lossy_alignment_method,
        "template_task_family": template_task_family or None,
        "template_workflow_signature": template_workflow_signature or None,
        "template_structural_fingerprint": template_structural_fingerprint or None,
    }


def summarize_payload(payload: dict) -> dict:
    spans = payload.get("code_anchor_spans", []) or []
    return {
        "code_anchor_signature": payload.get("code_anchor_signature", ""),
        "code_content_signature": payload.get("code_content_signature", ""),
        "num_anchor_spans": len(spans),
        "anchor_types": sorted(
            {
                str(span.get("anchor_type", "") or "")
                for span in spans
                if isinstance(span, dict) and span.get("anchor_type")
            }
        ),
        "reuse_mode": payload.get("reuse_mode", ""),
        "lossy_alignment_method": payload.get("lossy_alignment_method", ""),
        "template_task_family": payload.get("template_task_family", ""),
        "template_workflow_signature": payload.get("template_workflow_signature", ""),
        "template_structural_fingerprint": payload.get("template_structural_fingerprint", ""),
    }


def build_local_match_result(
    base_payload: dict,
    candidate_payload: dict,
) -> AnchorMatchResult:
    request_meta = build_anchor_metadata(
        code_anchor_signature=candidate_payload.get("code_anchor_signature", ""),
        code_content_signature=candidate_payload.get("code_content_signature", ""),
        code_anchor_spans=candidate_payload.get("code_anchor_spans", []),
        reuse_mode=candidate_payload.get("reuse_mode", ""),
        lossy_alignment_method=candidate_payload.get("lossy_alignment_method", ""),
        template_task_family=candidate_payload.get("template_task_family", ""),
        template_workflow_signature=candidate_payload.get("template_workflow_signature", ""),
        template_structural_fingerprint=candidate_payload.get("template_structural_fingerprint", ""),
    )
    candidate_meta = build_anchor_metadata(
        code_anchor_signature=base_payload.get("code_anchor_signature", ""),
        code_content_signature=base_payload.get("code_content_signature", ""),
        code_anchor_spans=base_payload.get("code_anchor_spans", []),
        reuse_mode="lossy",
        lossy_alignment_method=base_payload.get("lossy_alignment_method", ""),
        template_task_family=base_payload.get("template_task_family", ""),
        template_workflow_signature=base_payload.get("template_workflow_signature", ""),
        template_structural_fingerprint=base_payload.get("template_structural_fingerprint", ""),
    )
    return match_request_to_candidate(request_meta, candidate_meta)


async def main_async(args: argparse.Namespace) -> int:
    default_code = "def helper(x):\n    return x + 1\n"
    warmup_code = load_code_text(args.warmup_code_file, default_code)
    candidate_code = load_code_text(args.code_file, default_code)
    warmup_payload = build_request_payload(
        model_path=args.model_path,
        task=args.task,
        code_text=warmup_code,
        max_tokens=args.max_tokens,
        reuse_mode=args.warmup_reuse_mode,
        lossy_alignment_method=args.lossy_alignment_method,
        template_task_family=args.template_task_family,
        template_workflow_signature=args.template_workflow_signature,
        template_structural_fingerprint=args.template_structural_fingerprint,
    )
    candidate_payload = build_request_payload(
        model_path=args.model_path,
        task=args.task,
        code_text=candidate_code,
        max_tokens=args.max_tokens,
        reuse_mode=args.reuse_mode,
        lossy_alignment_method=args.lossy_alignment_method,
        template_task_family=args.template_task_family,
        template_workflow_signature=args.template_workflow_signature,
        template_structural_fingerprint=args.template_structural_fingerprint,
    )
    baseline_eval_payload = dict(candidate_payload)
    baseline_eval_payload["reuse_mode"] = "lossless"
    local_match = build_local_match_result(warmup_payload, candidate_payload)
    warmup_result = None
    baseline_eval_result = None
    eval_result = None
    if not args.skip_http:
        warmup_result = await do_request(args.base_url, warmup_payload)
        eval_result = await do_request(args.base_url, candidate_payload)
        baseline_eval_result = await do_request(args.base_url, baseline_eval_payload)
    out = {
        "config": {
            "base_url": args.base_url,
            "reuse_mode": args.reuse_mode,
            "warmup_reuse_mode": args.warmup_reuse_mode,
            "lossy_alignment_method": args.lossy_alignment_method,
            "template_task_family": args.template_task_family,
            "template_workflow_signature": args.template_workflow_signature,
            "template_structural_fingerprint": args.template_structural_fingerprint,
        },
        "warmup_request": summarize_payload(warmup_payload),
        "baseline_eval_request": summarize_payload(baseline_eval_payload),
        "eval_request": summarize_payload(candidate_payload),
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
    if args.output:
        Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark template + code-anchor lossy KV reuse")
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model-path", default="/home/gfy/models/Qwen2.5-7B-Instruct")
    parser.add_argument("--task", default="Summarize the helper function and suggest one refactor.")
    parser.add_argument("--code-file", default="")
    parser.add_argument("--warmup-code-file", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--reuse-mode", default="lossy")
    parser.add_argument("--warmup-reuse-mode", default="lossless")
    parser.add_argument("--lossy-alignment-method", default="kvcomm")
    parser.add_argument("--template-task-family", default="")
    parser.add_argument("--template-workflow-signature", default="")
    parser.add_argument("--template-structural-fingerprint", default="")
    parser.add_argument("--skip-http", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main_async(parse_args())))
