#!/usr/bin/env python3
"""HumanEval-lite pass@1 benchmark for v44 placeholder_knn_lossy.

Strategy (3h budget per plan B1):
- Read HumanEval from upstream simple_eval_humaneval.read_problems (164 tasks)
- Take first N (default 10) as "lite" subset
- For each task: POST to running sglang server, extract code, run functional_correctness
- Compute pass@1 = sum(passed) / total
- Run twice: baseline (no --enable-placeholder-knn) and v44 (--enable-placeholder-knn)
- Per-case driver: each task gets its own sglang server to avoid _delete_leaf race

Usage:
    # Baseline
    python -m benchmark.multi_workflow.bench_humaneval_pass_at_1 \\
        --max-cases 10 --out-dir results/humaneval_baseline_lite_<date>

    # v44
    python -m benchmark.multi_workflow.bench_humaneval_pass_at_1 \\
        --max-cases 10 --enable-placeholder-knn \\
        --out-dir results/humaneval_v44_lite_<date>
"""
import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "/home/gfy/models/Qwen2.5-3B-Instruct"
DEFAULT_PORT = 30000
HUMANEVAL_DIR = PROJECT / "python/sglang/test"
DEFAULT_OUT_DIR = PROJECT / "results"


def read_problems_lite(num_examples: int = 10):
    """Read HumanEval from upstream simple_eval_humaneval.py.

    Reuses the existing reader to avoid duplicating data.
    """
    sys.path.insert(0, str(PROJECT / "python"))
    from sglang.test.simple_eval_humaneval import read_problems  # type: ignore
    problems = read_problems()
    items = list(problems.values())[:num_examples]
    return items


def find_code(completion: str) -> str:
    """Extract Python code from a markdown-style response."""
    completion = completion or ""
    pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)
    matches = pattern.findall(completion)
    if matches:
        return matches[0]
    # fallback: extract function body
    pattern2 = re.compile(r"def\s+\w+\([^)]*\)[^:]*:(.*?)(?:\n\n|\Z)", re.DOTALL)
    matches2 = pattern2.findall(completion)
    if matches2:
        return completion
    return completion


def evaluate_functional_correctness_lite(problem: dict, completion: str, timeout: int = 10) -> bool:
    """Run HumanEval test against generated code.

    Uses upstream sglang.test.simple_eval_humaneval.evaluate_functional_correctness
    which wraps human_eval.execution.check_correctness.
    """
    sys.path.insert(0, str(PROJECT / "python"))
    from sglang.test.simple_eval_humaneval import (  # type: ignore
        evaluate_functional_correctness,
    )
    passed_list = evaluate_functional_correctness(
        sample=problem,
        completions=[completion],
        n_workers=1,
        timeout=timeout,
    )
    return bool(passed_list[0]) if passed_list else False


async def call_sglang(server_url: str, prompt: str, max_tokens: int = 512) -> str:
    """Call running sglang server and return completion."""
    import aiohttp
    payload = {
        "model": "/home/gfy/models/Qwen2.5-3B-Instruct",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{server_url}/v1/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            data = await resp.json()
            return data["choices"][0]["text"]


def launch_sglang_server(args: argparse.Namespace, log_file: Path) -> subprocess.Popen:
    """Launch sglang server with optional --enable-placeholder-knn env vars."""
    env = os.environ.copy()
    if args.enable_placeholder_knn:
        env["SGLANG_PLACEHOLDER_KNN_MATCH"] = "1"
        env["SGLANG_PLACEHOLDER_KNN_PRE_ROTATE_DELTAS"] = "1"
        env["SGLANG_PLACEHOLDER_KNN_MIN_COSINE"] = str(args.placeholder_knn_min_cosine)
        env["SGLANG_PLACEHOLDER_KNN_TOPK"] = str(args.placeholder_knn_topk)
    cmd = [
        "python", "-m", "sglang.launch_server",
        "--model-path", args.model,
        "--port", str(args.port),
        "--mem-fraction-static", "0.78",
    ]
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_handle = open(log_file, "w")
    return subprocess.Popen(cmd, stdout=log_handle, stderr=subprocess.STDOUT, env=env)


def wait_for_server(port: int, timeout: int = 240) -> bool:
    """Poll /health until ready."""
    import urllib.request
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(2)
    return False


async def run_one_case(problem: dict, server_url: str) -> dict:
    """Generate completion for one HumanEval task, evaluate functional correctness."""
    instruction = (
        "Read the following function signature and docstring, "
        "and fully implement the function described. "
        "Your response should only contain the code for this function.\n"
    )
    prompt = instruction + problem["prompt"]
    try:
        completion = await call_sglang(server_url, prompt)
    except Exception as e:
        return {"task_id": problem["task_id"], "error": str(e), "passed": False}
    passed = evaluate_functional_correctness_lite(problem, completion)
    return {
        "task_id": problem["task_id"],
        "completion_len": len(completion),
        "completion_sha": hashlib_sha1(completion),
        "passed": bool(passed) if passed is not None else False,
    }


def hashlib_sha1(s: str) -> str:
    import hashlib
    return hashlib.sha1(s.encode("utf-8", errors="replace")).hexdigest()[:8]


async def run_benchmark(args: argparse.Namespace) -> dict:
    """Main: launch server, run tasks, kill server, write results."""
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    problems = read_problems_lite(args.max_cases)
    print(f"[bench-humaneval] {len(problems)} HumanEval tasks")

    # Launch server
    server_log = out_dir / "sglang_server.log"
    server_url = f"http://127.0.0.1:{args.port}"
    print(f"[bench-humaneval] launching sglang server at {server_url} (v44={args.enable_placeholder_knn})")
    proc = launch_sglang_server(args, server_log)
    try:
        if not wait_for_server(args.port, args.server_timeout):
            print(f"[bench-humaneval] server failed to start within {args.server_timeout}s; killing")
            proc.kill()
            return {"error": "server_start_failed", "results": []}
        print(f"[bench-humaneval] server ready")

        # Run tasks
        results = []
        for p in problems:
            r = await run_one_case(p, server_url)
            results.append(r)
            print(f"  {p['task_id']}: passed={r['passed']} sha={r.get('completion_sha', '?')}")
    finally:
        proc.kill()
        proc.wait(timeout=10)

    # Aggregate
    n_pass = sum(1 for r in results if r.get("passed"))
    n_total = len(results)
    pass_at_1 = n_pass / n_total if n_total else 0.0
    summary = {
        "model": args.model,
        "max_cases": args.max_cases,
        "enable_placeholder_knn": args.enable_placeholder_knn,
        "placeholder_knn_min_cosine": args.placeholder_knn_min_cosine,
        "placeholder_knn_topk": args.placeholder_knn_topk,
        "n_pass": n_pass,
        "n_total": n_total,
        "pass_at_1": pass_at_1,
        "results": results,
    }
    summary_file = out_dir / "summary.json"
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[bench-humaneval] pass@1 = {pass_at_1:.2%} ({n_pass}/{n_total}) → {summary_file}")
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--max-cases", type=int, default=10, help="Number of HumanEval tasks (lite subset)")
    p.add_argument("--enable-placeholder-knn", action="store_true")
    p.add_argument("--placeholder-knn-min-cosine", type=float, default=0.85)
    p.add_argument("--placeholder-knn-topk", type=int, default=5)
    p.add_argument("--server-timeout", type=int, default=240)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(run_benchmark(parse_args()))