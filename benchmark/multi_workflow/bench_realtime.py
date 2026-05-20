#!/usr/bin/env python3
"""
Real-time KVFlow benchmark: LRU vs Priority eviction strategy.

Tests two sglang servers running simultaneously:
  - LRU server:  http://localhost:30001
  - Priority server: http://localhost:30002

Measures:
  - TTFT: Time To First Token
  - Total latency
  - Cache hit rate (from server logs / load metrics)

Usage:
  python3 bench_realtime.py
"""

import json
import time
import httpx
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

LRU_BASE_URL = "http://localhost:30001"
PRIORITY_BASE_URL = "http://localhost:30002"
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
MAX_TOKENS = 64
NUM_WARMUP = 2
NUM_REQUESTS = 10


@dataclass
class RequestResult:
    success: bool
    ttft_ms: float
    total_latency_ms: float
    tokens_generated: int
    error: Optional[str] = None


def make_chat_request_sync(
    base_url: str,
    messages: list,
    kvflow_hint: dict | None = None,
    max_tokens: int = MAX_TOKENS,
) -> RequestResult:
    """Make a synchronous chat completion request."""
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }
    if kvflow_hint:
        payload["extra_body"] = kvflow_hint

    t0 = time.perf_counter()
    try:
        with httpx.Client(timeout=60.0) as client:
            t_req = time.perf_counter()
            resp = client.post(f"{base_url}/v1/chat/completions", json=payload, headers=headers)
            t_resp = time.perf_counter()

        req_latency_ms = (t_resp - t_req) * 1000

        if resp.status_code != 200:
            return RequestResult(
                success=False,
                ttft_ms=req_latency_ms,
                total_latency_ms=req_latency_ms,
                tokens_generated=0,
                error=f"HTTP {resp.status_code}: {resp.text[:200]}",
            )

        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            return RequestResult(
                success=False,
                ttft_ms=req_latency_ms,
                total_latency_ms=req_latency_ms,
                tokens_generated=0,
                error="No choices in response",
            )

        choice = choices[0]
        message = choice.get("message", {})
        content = message.get("content", "")
        finish_reason = choice.get("finish_reason", "")

        usage = data.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 0)

        ttft_ms = req_latency_ms
        total_latency_ms = (time.perf_counter() - t0) * 1000

        return RequestResult(
            success=True,
            ttft_ms=ttft_ms,
            total_latency_ms=total_latency_ms,
            tokens_generated=completion_tokens,
        )

    except httpx.TimeoutException as e:
        return RequestResult(
            success=False,
            ttft_ms=(time.perf_counter() - t0) * 1000,
            total_latency_ms=(time.perf_counter() - t0) * 1000,
            tokens_generated=0,
            error=f"Timeout: {e}",
        )
    except Exception as e:
        return RequestResult(
            success=False,
            ttft_ms=(time.perf_counter() - t0) * 1000,
            total_latency_ms=(time.perf_counter() - t0) * 1000,
            tokens_generated=0,
            error=str(e),
        )


def get_server_load(base_url: str) -> dict | None:
    """Get server load metrics."""
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(f"{base_url}/v1/loads")
            if resp.status_code == 200:
                return resp.json()
    except Exception:
        pass
    return None


def run_benchmark(
    base_url: str,
    name: str,
    workflows: list[dict],
    concurrent: int = 2,
) -> list[RequestResult]:
    """Run benchmark with a list of workflow configs."""
    print(f"\n{'='*60}")
    print(f"  {name}  ({base_url})")
    print(f"{'='*60}")

    warmup_messages = [
        [{"role": "user", "content": "Hello, how are you?"}],
    ]
    print("  Warming up...")
    for wm in warmup_messages * NUM_WARMUP:
        make_chat_request_sync(base_url, wm)

    results: list[RequestResult] = []

    for i, wf in enumerate(workflows):
        print(f"\n  [Request {i+1}/{len(workflows)}] {wf.get('desc', '')}")
        messages = wf["messages"]
        kvflow_hint = wf.get("kvflow_hint")

        result = make_chat_request_sync(base_url, messages, kvflow_hint)
        results.append(result)

        if result.success:
            print(f"    TTFT: {result.ttft_ms:.1f}ms  |  "
                  f"Latency: {result.total_latency_ms:.1f}ms  |  "
                  f"Tokens: {result.tokens_generated}")
        else:
            print(f"    ERROR: {result.error}")

    return results


def summarize_results(name: str, results: list[RequestResult]) -> dict:
    """Summarize results."""
    successful = [r for r in results if r.success]
    if not successful:
        return {"name": name, "error": "All requests failed"}

    ttfts = [r.ttft_ms for r in successful]
    latencies = [r.total_latency_ms for r in successful]
    tokens = [r.tokens_generated for r in successful]

    return {
        "name": name,
        "total_requests": len(results),
        "successful": len(successful),
        "failed": len(results) - len(successful),
        "ttft_mean_ms": statistics.mean(ttfts),
        "ttft_median_ms": statistics.median(ttfts),
        "ttft_stdev_ms": statistics.stdev(ttfts) if len(ttfts) > 1 else 0,
        "latency_mean_ms": statistics.mean(latencies),
        "latency_median_ms": statistics.median(latencies),
        "tokens_per_sec": (
            sum(tokens) / (sum(latencies) / 1000) if sum(latencies) > 0 else 0
        ),
        "avg_tokens": statistics.mean(tokens) if tokens else 0,
    }


def print_summary(lru_stats: dict, priority_stats: dict):
    """Print comparison summary."""
    print(f"\n{'='*60}")
    print(f"  BENCHMARK RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"\n  {'Metric':<30} {'LRU':>12} {'Priority':>12} {'Improvement':>12}")
    print(f"  {'-'*68}")

    lru_ttft = lru_stats.get("ttft_mean_ms", 0)
    pri_ttft = priority_stats.get("ttft_mean_ms", 0)
    ttft_improve = ((lru_ttft - pri_ttft) / lru_ttft * 100) if lru_ttft > 0 else 0
    print(f"  {'TTFT Mean (ms)':<30} {lru_ttft:>12.1f} {pri_ttft:>12.1f} {ttft_improve:>+11.1f}%")

    lru_lat = lru_stats.get("latency_mean_ms", 0)
    pri_lat = priority_stats.get("latency_mean_ms", 0)
    lat_improve = ((lru_lat - pri_lat) / lru_lat * 100) if lru_lat > 0 else 0
    print(f"  {'Total Latency Mean (ms)':<30} {lru_lat:>12.1f} {pri_lat:>12.1f} {lat_improve:>+11.1f}%")

    lru_tps = lru_stats.get("tokens_per_sec", 0)
    pri_tps = priority_stats.get("tokens_per_sec", 0)
    tps_improve = ((pri_tps - lru_tps) / lru_tps * 100) if lru_tps > 0 else 0
    print(f"  {'Throughput (tok/s)':<30} {lru_tps:>12.1f} {pri_tps:>12.1f} {tps_improve:>+11.1f}%")

    lru_suc = lru_stats.get("successful", 0)
    pri_suc = priority_stats.get("successful", 0)
    lru_tot = lru_stats.get("total_requests", 0)
    pri_tot = priority_stats.get("total_requests", 0)
    print(f"  {'Success Rate':<30} {lru_suc:>12}/{lru_tot:<3} {pri_suc:>12}/{pri_tot:<3}")

    print(f"\n  {'='*68}")
    if ttft_improve > 0:
        print(f"  ✓ Priority is {ttft_improve:.1f}% faster in TTFT")
    elif ttft_improve < 0:
        print(f"  ✗ LRU is {-ttft_improve:.1f}% faster in TTFT")
    else:
        print(f"  = Same TTFT")
    print(f"{'='*68}\n")


def main():
    print("KVFlow Real-Time Benchmark: LRU vs Priority")
    print(f"Model: {MODEL_NAME}")
    print(f"Requests: {NUM_REQUESTS} per server (+ {NUM_WARMUP*2} warmup)")

    # ── Workflow 1: Shared system prompt (Tier-0 cache hit) ──
    # Simulates: multiple agents sharing the same system prompt
    system_prompt = (
        "You are a helpful AI coding assistant. "
        "You help with debugging, code review, and implementation."
    )
    shared_system = [{"role": "system", "content": system_prompt}]

    # ── Workflow 2: Role-specific prompt (Tier-1) ──
    role_prompt = (
        "You are a code implementer. "
        "Write clean, efficient code based on requirements."
    )

    # ── Workflow 3: Task-specific context (Tier-2, private) ──
    task_context = (
        "The user wants to implement a radix tree for KV cache management. "
        "The tree should support prefix matching, LRU eviction, and thread-safe operations."
    )

    # KVFlow hints for Priority server
    KVFLOW_SYS_HINT = {
        "priority": 15,
        "next_agent_prefix": "role_prompt",
        "role_type": 1,           # SYSTEM tier
        "convergence_factor": 0,
        "critical_path_distance": 5,
    }
    KVFLOW_ROLE_HINT = {
        "priority": 13,
        "next_agent_prefix": "task_context",
        "role_type": 2,           # ROLE tier
        "convergence_factor": 20,
        "critical_path_distance": 3,
    }
    KVFLOW_TASK_HINT = {
        "priority": 11,
        "next_agent_prefix": "",
        "role_type": 3,           # TASK tier
        "convergence_factor": 10,
        "critical_path_distance": 1,
    }

    workflows_lru = [
        {
            "desc": "Shared system prompt (Tier-0)",
            "messages": shared_system + [{"role": "user", "content": "Implement a stack in Python."}],
            "kvflow_hint": None,   # LRU server ignores hints
        },
        {
            "desc": "Role-specific prompt (Tier-1)",
            "messages": shared_system
                      + [{"role": "system", "content": role_prompt}]
                      + [{"role": "user", "content": "Write a function to reverse a linked list."}],
            "kvflow_hint": None,
        },
        {
            "desc": "Task context (Tier-2, private)",
            "messages": shared_system
                      + [{"role": "system", "content": role_prompt}]
                      + [{"role": "system", "content": task_context}]
                      + [{"role": "user", "content": "Implement a LRU cache class."}],
            "kvflow_hint": None,
        },
        {
            "desc": "Follow-up with same system (cache hit expected)",
            "messages": shared_system + [{"role": "user", "content": "Add max_size support."}],
            "kvflow_hint": None,
        },
        {
            "desc": "Follow-up with same role (cache hit expected)",
            "messages": shared_system
                      + [{"role": "system", "content": role_prompt}]
                      + [{"role": "user", "content": "Add thread safety."}],
            "kvflow_hint": None,
        },
        {
            "desc": "Fresh request (no cache)",
            "messages": [{"role": "user", "content": "What is the time complexity of quicksort?"}],
            "kvflow_hint": None,
        },
        {
            "desc": "Multi-agent step 1 (system+role)",
            "messages": shared_system
                      + [{"role": "system", "content": role_prompt}]
                      + [{"role": "user", "content": "Debug: IndexError in tree traversal."}],
            "kvflow_hint": None,
        },
        {
            "desc": "Multi-agent step 2 (task context, same prefixes)",
            "messages": shared_system
                      + [{"role": "system", "content": role_prompt}]
                      + [{"role": "system", "content": task_context}]
                      + [{"role": "user", "content": "Fix the bug and add tests."}],
            "kvflow_hint": None,
        },
        {
            "desc": "Same as step 2 (full cache hit)",
            "messages": shared_system
                      + [{"role": "system", "content": role_prompt}]
                      + [{"role": "system", "content": task_context}]
                      + [{"role": "user", "content": "Add edge case handling."}],
            "kvflow_hint": None,
        },
        {
            "desc": "Fresh unrelated request",
            "messages": [{"role": "user", "content": "Explain closure in JavaScript."}],
            "kvflow_hint": None,
        },
    ]

    workflows_priority = [
        {
            "desc": "Shared system prompt (Tier-0)",
            "messages": shared_system + [{"role": "user", "content": "Implement a stack in Python."}],
            "kvflow_hint": KVFLOW_SYS_HINT,
        },
        {
            "desc": "Role-specific prompt (Tier-1)",
            "messages": shared_system
                      + [{"role": "system", "content": role_prompt}]
                      + [{"role": "user", "content": "Write a function to reverse a linked list."}],
            "kvflow_hint": KVFLOW_ROLE_HINT,
        },
        {
            "desc": "Task context (Tier-2, private)",
            "messages": shared_system
                      + [{"role": "system", "content": role_prompt}]
                      + [{"role": "system", "content": task_context}]
                      + [{"role": "user", "content": "Implement a LRU cache class."}],
            "kvflow_hint": KVFLOW_TASK_HINT,
        },
        {
            "desc": "Follow-up with same system (cache hit expected)",
            "messages": shared_system + [{"role": "user", "content": "Add max_size support."}],
            "kvflow_hint": KVFLOW_SYS_HINT,
        },
        {
            "desc": "Follow-up with same role (cache hit expected)",
            "messages": shared_system
                      + [{"role": "system", "content": role_prompt}]
                      + [{"role": "user", "content": "Add thread safety."}],
            "kvflow_hint": KVFLOW_ROLE_HINT,
        },
        {
            "desc": "Fresh request (no cache)",
            "messages": [{"role": "user", "content": "What is the time complexity of quicksort?"}],
            "kvflow_hint": KVFLOW_TASK_HINT,
        },
        {
            "desc": "Multi-agent step 1 (system+role)",
            "messages": shared_system
                      + [{"role": "system", "content": role_prompt}]
                      + [{"role": "user", "content": "Debug: IndexError in tree traversal."}],
            "kvflow_hint": KVFLOW_ROLE_HINT,
        },
        {
            "desc": "Multi-agent step 2 (task context, same prefixes)",
            "messages": shared_system
                      + [{"role": "system", "content": role_prompt}]
                      + [{"role": "system", "content": task_context}]
                      + [{"role": "user", "content": "Fix the bug and add tests."}],
            "kvflow_hint": KVFLOW_TASK_HINT,
        },
        {
            "desc": "Same as step 2 (full cache hit)",
            "messages": shared_system
                      + [{"role": "system", "content": role_prompt}]
                      + [{"role": "system", "content": task_context}]
                      + [{"role": "user", "content": "Add edge case handling."}],
            "kvflow_hint": KVFLOW_TASK_HINT,
        },
        {
            "desc": "Fresh unrelated request",
            "messages": [{"role": "user", "content": "Explain closure in JavaScript."}],
            "kvflow_hint": KVFLOW_TASK_HINT,
        },
    ]

    # Run LRU benchmark
    lru_results = run_benchmark(LRU_BASE_URL, "LRU Server", workflows_lru)

    # Small pause between benchmarks
    time.sleep(2)

    # Run Priority benchmark
    priority_results = run_benchmark(PRIORITY_BASE_URL, "Priority Server", workflows_priority)

    # Summarize
    lru_stats = summarize_results("LRU", lru_results)
    priority_stats = summarize_results("Priority", priority_results)

    print("\n  LRU Detailed Results:")
    for i, r in enumerate(lru_results, 1):
        status = "✓" if r.success else "✗"
        print(f"    [{i}] {status} TTFT={r.ttft_ms:.1f}ms Lat={r.total_latency_ms:.1f}ms "
              f"Tok={r.tokens_generated}  {workflows_lru[i-1]['desc']}")

    print("\n  Priority Detailed Results:")
    for i, r in enumerate(priority_results, 1):
        status = "✓" if r.success else "✗"
        print(f"    [{i}] {status} TTFT={r.ttft_ms:.1f}ms Lat={r.total_latency_ms:.1f}ms "
              f"Tok={r.tokens_generated}  {workflows_priority[i-1]['desc']}")

    print_summary(lru_stats, priority_stats)


if __name__ == "__main__":
    main()
