#!/usr/bin/env python3
"""HumanEval/MBPP codebase-context accuracy sanity for selective AST reuse."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import aiohttp
from transformers import AutoTokenizer

PROJECT = Path(__file__).resolve().parents[2]
for entry in (str(PROJECT.parent / "MAScoder" / "src"), str(PROJECT), str(PROJECT / "python")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from benchmark.multi_workflow.bench_swe_generated_patch_kvcomm import (  # noqa: E402
    DEFAULT_PYTHON,
    CodeSegment,
    build_anchor_fields,
    extract_cached_tokens,
    extract_lossy_meta,
    extract_text,
    kill_port,
    post_chat,
    post_chat_optional_stream,
    wait_ready,
)
from benchmark.multi_workflow.selective_ast_reuse import (  # noqa: E402
    DEFAULT_POLICY_PATH,
    load_selective_policy,
    select_spans,
    split_python_file,
    summarize_selection,
)

DEFAULT_MODEL = "/home/gfy/models/Qwen2.5-7B-Instruct"
OUT_DIR = PROJECT / "results" / "selective_ast_reuse" / "codebase_accuracy"
MODES = ["lossless_full_prefill", "whole_file_reuse_all", "selective_function_method_reuse"]


HELPERS = """
from typing import Iterable, List, Dict, Tuple

def clamp(value, lo, hi):
    return max(lo, min(hi, value))

def flatten_once(items):
    out = []
    for item in items:
        if isinstance(item, list):
            out.extend(item)
        else:
            out.append(item)
    return out

def pairwise(items):
    return list(zip(items, items[1:]))
""".strip()


def launch_server(args: argparse.Namespace) -> subprocess.Popen:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT / "python")
    env["SGLANG_LOSSY_FUZZY_MATCH"] = "1"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        args.python,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model,
        "--port",
        str(args.port),
        "--tp-size",
        "1",
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--max-total-tokens",
        str(args.max_total_tokens),
        "--chunked-prefill-size",
        "8192",
        "--max-prefill-tokens",
        str(args.max_prefill_tokens),
        "--enable-cache-report",
        "--disable-cuda-graph",
        "--allow-auto-truncate",
        "--log-level",
        "error",
    ]
    if args.force_evict:
        env["SGLANG_RADIX_FORCE_EVICT"] = "1"
    if args.max_running_requests is not None:
        cmd += ["--max-running-requests", str(args.max_running_requests)]
    return subprocess.Popen(
        cmd,
        cwd=str(PROJECT),
        env=env,
        stdout=open(args.out_dir / "sglang_server.log", "w"),
        stderr=subprocess.STDOUT,
    )


def extract_code(text: str) -> str:
    match = re.search(r"```(?:python|py)?\n([\s\S]*?)\n```", text or "")
    return (match.group(1) if match else text or "").strip()


def humaneval_body(text: str, prompt: str) -> str:
    code = extract_code(text)
    lines = code.splitlines()
    for idx, line in enumerate(lines):
        if line.lstrip().startswith("def "):
            body = "\n".join(lines[idx + 1 :]).rstrip()
            return body + "\n" if body else "    pass\n"
    # Treat non-def output as a function body.
    out = []
    for line in lines:
        if not line.strip():
            out.append(line)
        elif line.startswith((" ", "\t")):
            out.append(line)
        else:
            out.append("    " + line)
    return "\n".join(out).rstrip() + "\n"


def build_codebase_prompt(task: dict[str, Any], dataset: str) -> tuple[list[CodeSegment], str]:
    if dataset == "humaneval":
        target = str(task["prompt"]).rstrip() + "\n    pass\n"
        instruction = (
            "Complete the target HumanEval function. Return only the function body, "
            "no markdown and no def line."
        )
    else:
        prompt = str(task.get("prompt") or task.get("text") or "").replace("\n", "\n# ")
        tests = "\n".join(str(x) for x in (task.get("test_list") or []))
        target = "# MBPP task\n# " + prompt + "\n\n# Public tests:\n" + "\n".join(f"# {line}" for line in tests.splitlines()) + "\n"
        instruction = (
            "Write the smallest complete Python solution for the MBPP task. "
            "Return only Python code, no markdown, no explanation. "
            "Define the exact function name used by the public tests. "
            "Avoid type annotations unless you also import their names."
        )
    segments = [
        CodeSegment("target.py", target),
        CodeSegment("helpers.py", HELPERS),
    ]
    return segments, instruction


def messages_for_task(task: dict[str, Any], segments: list[CodeSegment], instruction: str, mode: str) -> list[dict[str, str]]:
    blocks = []
    for segment in segments:
        blocks.extend([f"## code_base: {segment.name}", "```python", segment.text, "```", ""])
    return [
        {
            "role": "system",
            "content": "You are a precise Python coding agent. Use the whole code_base context.",
        },
        {
            "role": "user",
            "content": "\n".join([*blocks, "## Instruction", instruction]),
        },
    ]


def selected_for_mode(segments: list[CodeSegment], policy: dict[str, Any], mode: str) -> tuple[list[CodeSegment], dict[str, Any]]:
    spans = []
    for segment in segments:
        spans.extend(split_python_file(segment.name, segment.text, policy))
    selected = select_spans(spans, mode)
    if mode == "whole_file_reuse_all":
        selected_segments = segments
    else:
        selected_segments = [CodeSegment(span.name, span.text) for span in selected]
    return selected_segments, summarize_selection(spans, selected)


def make_payload(args, tokenizer, messages, selected_segments, mode, salt):
    include_anchor = mode != "lossless_full_prefill" and bool(selected_segments)
    payload = {
        "model": args.model,
        "messages": messages,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": 1.0,
        "return_cached_tokens_details": True,
        "reuse_mode": "lossy" if include_anchor else "lossless",
        "lossy_alignment_method": "kvcomm",
        "template_task_family": f"selective_ast_{args.dataset}_accuracy",
        "cache_salt": salt,
    }
    if include_anchor:
        payload.update(build_anchor_fields(tokenizer, messages, selected_segments))
    return payload


async def warm_task(session, args, tokenizer, task, segments, instruction, policy):
    selected, _ = selected_for_mode(segments, policy, "selective_function_method_reuse")
    messages = messages_for_task(task, segments, instruction, "warmup")
    payload = make_payload(args, tokenizer, messages, selected or segments[:1], "selective_function_method_reuse", f"warm:{task['task_id']}")
    payload["max_tokens"] = 1
    await asyncio.wait_for(post_chat_optional_stream(session, args.port, payload, args.emit_ttft), timeout=args.request_timeout)


async def run_task(session, args, tokenizer, task, policy):
    segments, instruction = build_codebase_prompt(task, args.dataset)
    try:
        await warm_task(session, args, tokenizer, task, segments, instruction, policy)
    except Exception as exc:
        print(f"[{args.dataset}] {task['task_id']} warmup failed: {type(exc).__name__}: {exc}", flush=True)
        if args.fail_fast:
            raise
    rows = []
    for mode in MODES:
        selected, selection = selected_for_mode(segments, policy, mode)
        messages = messages_for_task(task, segments, instruction, mode)
        try:
            response = await asyncio.wait_for(
                post_chat_optional_stream(
                    session,
                    args.port,
                    make_payload(args, tokenizer, messages, selected, mode, f"target:{task['task_id']}:{mode}"),
                    args.emit_ttft,
                ),
                timeout=args.request_timeout,
            )
            text = extract_text(response["body"]) or ""
            completion = humaneval_body(text, task.get("prompt", "")) if args.dataset == "humaneval" else extract_code(text)
            meta = extract_lossy_meta(response["body"])
            row = {
                "task_id": task["task_id"],
                "mode": mode,
                "elapsed_ms": response["elapsed_ms"],
                "ttft_ms": response.get("ttft_ms"),
                "cached_tokens": extract_cached_tokens(response["body"]),
                "completion": completion,
                "model_output": text,
                "lossy_match_reason": meta.get("lossy_first_match_reason") or meta.get("lossy_final_match_reason"),
                **selection,
            }
        except Exception as exc:
            if args.fail_fast:
                raise
            row = {
                "task_id": task["task_id"],
                "mode": mode,
                "elapsed_ms": None,
                "cached_tokens": 0,
                "completion": "",
                "model_output": "",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "lossy_match_reason": None,
                **selection,
            }
        rows.append(row)
        print(f"[{args.dataset}] {task['task_id']} {mode} done", flush=True)
    return rows


def load_humaneval(limit: int | None) -> list[dict[str, Any]]:
    from human_eval.data import read_problems

    problems = read_problems()
    rows = [{"task_id": task_id, **problem} for task_id, problem in problems.items()]
    return rows[:limit] if limit else rows


def load_mbpp(limit: int | None) -> list[dict[str, Any]]:
    import datasets

    data = datasets.load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    rows = []
    for idx, item in enumerate(data):
        row = dict(item)
        row["task_id"] = str(row.get("task_id") or idx)
        rows.append(row)
        if limit and len(rows) >= limit:
            break
    return rows


def eval_humaneval(out_dir: Path, rows: list[dict[str, Any]], k: int) -> dict[str, Any]:
    from human_eval.evaluation import evaluate_functional_correctness

    results = {}
    for mode in MODES:
        sample_path = out_dir / f"humaneval_{mode}.jsonl"
        with sample_path.open("w", encoding="utf-8") as f:
            for row in rows:
                if row["mode"] == mode:
                    f.write(json.dumps({"task_id": row["task_id"], "completion": row["completion"]}) + "\n")
        results[mode] = evaluate_functional_correctness(str(sample_path), k=[k], n_workers=4, ignore_incomplete=True)
    return results


def _run_python(source: str, timeout: int = 10) -> subprocess.CompletedProcess:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(source)
        path = f.name
    try:
        return subprocess.run([sys.executable, path], text=True, capture_output=True, timeout=timeout, check=False)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def eval_mbpp(tasks: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {str(task["task_id"]): task for task in tasks}
    results = {}
    for mode in MODES:
        mode_rows = [row for row in rows if row["mode"] == mode]
        passed = 0
        for row in mode_rows:
            task = by_id[str(row["task_id"])]
            tests = "\n".join(str(x) for x in (task.get("test_list") or []))
            setup = "\n".join(str(x) for x in (task.get("test_imports") or []))
            setup_code = str(task.get("test_setup_code") or "")
            proc = _run_python("\n\n".join([setup, setup_code, row["completion"], tests]))
            row["passed"] = proc.returncode == 0
            row["error"] = "" if proc.returncode == 0 else (proc.stderr[-1000:] or proc.stdout[-1000:])
            passed += int(row["passed"])
        results[mode] = {"pass_rate": passed / max(1, len(mode_rows)), "passed": passed, "n": len(mode_rows)}
    return results


def write_report(args, rows, eval_results):
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "rows.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.out_dir / "summary.json").write_text(json.dumps(eval_results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        f"# {args.dataset} codebase-context selective accuracy",
        "",
        "| mode | pass result | avg cached | avg reused toks | avg recomputed toks |",
        "|---|---|---:|---:|---:|",
    ]
    for mode in MODES:
        mode_rows = [row for row in rows if row["mode"] == mode]
        cached = sum(float(r["cached_tokens"]) for r in mode_rows) / max(1, len(mode_rows))
        reused = sum(float(r["estimated_reused_tokens"]) for r in mode_rows) / max(1, len(mode_rows))
        recomputed = sum(float(r["estimated_recomputed_tokens"]) for r in mode_rows) / max(1, len(mode_rows))
        lines.append(f"| `{mode}` | `{eval_results.get(mode)}` | {cached:.1f} | {reused:.1f} | {recomputed:.1f} |")
    lines.append("")
    lines.append("HumanEval/MBPP are accuracy sanity checks; do not pool them with SWE/codebase pass@1.")
    (args.out_dir / "CODEBASE_CONTEXT_ACCURACY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_progress(args: argparse.Namespace, rows: list[dict[str, Any]], status: str, task_id: str | None = None) -> None:
    progress = {
        "dataset": args.dataset,
        "status": status,
        "last_task_id": task_id,
        "rows": len(rows),
        "tasks_completed_estimate": len({str(row["task_id"]) for row in rows}),
        "expected_tasks": args.expected_tasks,
        "expected_rows": args.expected_tasks * len(MODES),
    }
    (args.out_dir / "progress.json").write_text(json.dumps(progress, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


async def run_benchmark(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    policy = load_selective_policy(args.policy)
    tasks = load_humaneval(args.limit) if args.dataset == "humaneval" else load_mbpp(args.limit)
    args.expected_tasks = len(tasks)
    partial_path = args.out_dir / "rows.partial.jsonl"
    partial_path.unlink(missing_ok=True)
    write_progress(args, [], "starting")
    kill_port(args.port)
    proc = launch_server(args)
    try:
        if not await wait_ready(args.port, args.server_timeout):
            raise RuntimeError("server failed to become ready")
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        rows = []
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=args.eval_timeout)) as session:
            for task in tasks:
                task_rows = await run_task(session, args, tokenizer, task, policy)
                rows.extend(task_rows)
                append_jsonl(partial_path, task_rows)
                write_progress(args, rows, "running", str(task["task_id"]))
        eval_results = eval_humaneval(args.out_dir, rows, args.k) if args.dataset == "humaneval" else eval_mbpp(tasks, rows)
        write_report(args, rows, eval_results)
        write_progress(args, rows, "complete")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
        kill_port(args.port)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["humaneval", "mbpp"], default="humaneval")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-total-tokens", type=int, default=65536)
    parser.add_argument("--max-prefill-tokens", type=int, default=16384)
    parser.add_argument("--mem-fraction-static", type=float, default=0.78)
    parser.add_argument("--force-evict", action="store_true")
    parser.add_argument("--max-running-requests", type=int, default=1)
    parser.add_argument("--server-timeout", type=int, default=180)
    parser.add_argument("--eval-timeout", type=int, default=1200)
    parser.add_argument("--request-timeout", type=int, default=300)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--emit-ttft", action="store_true",
                        help="Use streaming post_chat_stream and record per-mode ttft_ms in result rows.")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run_benchmark(parse_args()))
