#!/usr/bin/env python3
"""High-cache-pressure benchmark for KVFlow eviction strategy."""

import time
import re
import httpx
from dataclasses import dataclass
from typing import Optional

LRU_BASE = "http://localhost:30001"
PRIORITY_BASE = "http://localhost:30002"
MODEL = "/home/gfy/models/Qwen2.5-3B-Instruct"
MAX_TOKENS = 16
NUM_REPEATS = 8


LONG_SYS = """You are an expert software engineer. You specialize in writing clean,
efficient, and well-documented code. You follow best practices for code review,
testing, and deployment. Always use type hints, docstrings, and PEP 8 style.
Keep functions small and focused. Write unit tests for all new functionality."""


LONG_CODE = """class TaskService:
    def __init__(self, db_pool):
        self.db = db_pool
        self.cache = {}
    async def get_task(self, task_id):
        if task_id in self.cache: return self.cache[task_id]
        return None
    async def list_tasks(self, project_id, limit=50):
        return []
    async def create_task(self, task):
        self.cache[task.id] = task; return task
    async def update_task(self, tid, **u): return self.cache.get(tid)
    def invalidate_cache(self, tid): self.cache.pop(tid, None)"""


@dataclass
class Result:
    name: str
    latencies: list
    success: bool


def req(base: str, msgs: list, maxt: int = MAX_TOKENS):
    t0 = time.perf_counter()
    try:
        with httpx.Client(timeout=120.0) as c:
            r = c.post(f"{base}/v1/chat/completions", json={
                "model": MODEL, "messages": msgs,
                "max_tokens": maxt, "temperature": 0.7,
            })
        return r.json() if r.status_code == 200 else None, (time.perf_counter() - t0) * 1000
    except Exception:
        return None, (time.perf_counter() - t0) * 1000


def parse_cache_stats(log_path: str) -> dict:
    try:
        with open(log_path) as f:
            content = f.read()
    except Exception:
        return {}
    lines = [l for l in content.split("\n") if "Prefill batch" in l and "#new-token" in l]
    cached, new_tok, throughputs = [], [], []
    for l in lines[-NUM_REPEATS * 3:]:
        m_n = re.search(r"#new-token:\s*(\d+)", l)
        m_c = re.search(r"#cached-token:\s*(\d+)", l)
        m_t = re.search(r"input throughput.*?:\s*([\d.]+)", l)
        if m_n: new_tok.append(int(m_n.group(1)))
        if m_c: cached.append(int(m_c.group(1)))
        if m_t: throughputs.append(float(m_t.group(1)))
    if not new_tok:
        return {}
    total = sum(new_tok) + sum(cached)
    return {
        "cache_hit_ratio": sum(cached) / max(1, total),
        "avg_throughput": sum(throughputs) / max(1, len(throughputs)),
        "total_cached": sum(cached),
        "total_new": sum(new_tok),
    }


def run_test(base: str, name: str) -> Result:
    prefix = [{"role": "system", "content": LONG_SYS}, {"role": "system", "content": LONG_CODE}]
    tasks = [
        "Add error handling for null inputs.",
        "Add type hints to the function.",
        "Write unit tests for it.",
        "Add logging statements.",
        "Optimize for space complexity.",
        "Add docstrings to all methods.",
        "Implement thread safety.",
        "Add support for custom comparators.",
    ]
    warmup = prefix + [{"role": "user", "content": "Implement a stack."}]
    req(base, warmup)
    lats = []
    for t in tasks[:NUM_REPEATS]:
        msgs = prefix + [{"role": "user", "content": t}]
        data, lat = req(base, msgs)
        lats.append(lat)
    return Result(name, lats, all(l > 0 for l in lats))


def main():
    print("=" * 60)
    print("  KVFlow High-Cache-Pressure Benchmark")
    print("  Design: Long prompts + short generation")
    print("=" * 60)

    lru = run_test(LRU_BASE, "LRU")
    pri = run_test(PRIORITY_BASE, "Priority")

    print(f"\n  LRU:      avg={sum(lru.latencies)/len(lru.latencies):.1f}ms  "
          f"min={min(lru.latencies):.1f}ms  max={max(lru.latencies):.1f}ms  "
          f"success={lru.success}")
    print(f"  Priority: avg={sum(pri.latencies)/len(pri.latencies):.1f}ms  "
          f"min={min(pri.latencies):.1f}ms  max={max(pri.latencies):.1f}ms  "
          f"success={pri.success}")

    lru_avg = sum(lru.latencies) / len(lru.latencies)
    pri_avg = sum(pri.latencies) / len(pri.latencies)
    improve = ((lru_avg - pri_avg) / lru_avg * 100) if lru_avg > 0 else 0
    print(f"\n  → TTFT improvement: {improve:+.1f}%")

    print("\n  Per-request TTFT (ms):")
    print(f"  {'#':<4} {'LRU':>10} {'Priority':>10} {'Diff':>10}")
    print(f"  {'-'*36}")
    for i, (la, lp) in enumerate(zip(lru.latencies, pri.latencies), 1):
        diff = ((la - lp) / la * 100) if la > 0 else 0
        print(f"  {i:<4} {la:>10.1f} {lp:>10.1f} {diff:>+9.1f}%")

    print("\n  Server Log Analysis:")
    lru_s = parse_cache_stats("/tmp/sglang_lru_server.log")
    pri_s = parse_cache_stats("/tmp/sglang_priority_server_new.log")
    if lru_s:
        print(f"  LRU:      cache_hit={lru_s['cache_hit_ratio']:.1%}  "
              f"cached={lru_s['total_cached']} tokens  "
              f"avg_throughput={lru_s['avg_throughput']:.1f} tok/s")
    if pri_s:
        print(f"  Priority: cache_hit={pri_s['cache_hit_ratio']:.1%}  "
              f"cached={pri_s['total_cached']} tokens  "
              f"avg_throughput={pri_s['avg_throughput']:.1f} tok/s")

    print("\n" + "=" * 60)
    if improve > 1.0:
        print(f"  ✓ Priority is {improve:.1f}% faster")
    elif improve < -1.0:
        print(f"  ✗ LRU is {-improve:.1f}% faster")
    else:
        print(f"  ≈ No significant difference ({improve:+.1f}%)")
        print("  Note: improvement requires cache hit rate > 50%")
        print("  Check avg_throughput in logs - should be ~48 tok/s for hits")


if __name__ == "__main__":
    main()
