#!/usr/bin/env python3
"""
Adversarial Eviction Pressure Test v2 — Large Request Stress Test

Key insight: use MUCH LARGER unique requests (2000+ tokens each) so fewer
requests fill the cache. With ~30 large unique requests, we exceed the
~40K token cache capacity and force eviction of the shared prefix.

Design:
  Phase 1: Prime with shared System(500t) + Code(1000t) prefix → build Tier-0/1 cache
  Phase 2: Flood N large unique requests (500+1000+500 tokens each) → fill & evict
  Phase 3: Re-request shared prefix → measure TTFT degradation

If Priority Strategy works:
  - Unique requests have low priority (Tier-2, large crit_dist)
  - Shared prefix has high priority (Tier-0/1)
  - When cache is full, unique KV gets evicted first → Phase-3 fast
If LRU:
  - Most recently used wins → shared prefix gets pushed out by unique requests
    that came after it → Phase-3 has to re-prefill → slower
"""

import time
import httpx
import argparse
from dataclasses import dataclass

LRU_BASE = "http://localhost:30001"
PRIORITY_BASE = "http://localhost:30002"
MODEL_PATH = "/home/gfy/models/Qwen2.5-3B-Instruct"
TIMEOUT = 180.0


def make_large_system(idx: int) -> str:
    base = (
        f"You are Developer #{idx}. Your stack: Python 3.11, FastAPI, SQLAlchemy, "
        f"PostgreSQL 15, Redis, Docker, Kubernetes, AWS ECS."
    )
    guidelines = (
        " Follow PEP 8. Add type hints to all functions. Write Google-style docstrings. "
        "Use dataclasses for DTOs. Prefer composition over inheritance. "
        "Use async/await for I/O. Add retry with exponential backoff. "
        "Use structured logging (JSON). Add OpenTelemetry tracing. "
        "Use pydantic for validation. Follow 12-factor app principles. "
        "Use circuit breakers for external calls. Implement rate limiting. "
        "Use connection pooling. Set up health checks. Add metrics (Prometheus). "
        "Use Kubernetes liveness/readiness probes. Follow REST best practices. "
        "Version your APIs. Use environment variables for config. "
        "Write integration tests with pytest. Use factory fixtures in tests."
    )
    return base + guidelines * 2


def make_large_code_context(idx: int) -> str:
    lines = []
    for cls_idx in range(20):
        cidx = idx * 100 + cls_idx
        lines.append(f"""
class Model{cidx}:
    def __init__(self, config: dict):
        self.id = {cidx}
        self.config = config
        self.state = {{}}
        self.registry = []
        self._initialized = False

    def setup(self) -> bool:
        if self._initialized:
            return False
        for key in self.config:
            self.state[key] = self.config[key]
        self._initialized = True
        return True

    def process(self, data: list) -> dict:
        result = {{"id": self.id, "processed": len(data)}}
        for item in data:
            if item.get("active"):
                result["items"] = result.get("items", 0) + 1
        return result

    def validate(self, payload: dict) -> bool:
        required = ["name", "value", "timestamp"]
        return all(k in payload for k in required)

    def reset(self) -> None:
        self.state.clear()
        self.registry.clear()
        self._initialized = False
""")
    return "\n".join(lines)


def make_large_task(idx: int) -> str:
    templates = [
        f"Refactor Service{idx} to use async patterns and add error handling.",
        f"Add retry logic and circuit breaker to Handler{idx}.",
        f"Write unit tests for Model{idx} covering all methods.",
        f"Add OpenTelemetry tracing to Service{idx} methods.",
        f"Implement rate limiting for API{idx} endpoints.",
        f"Add health check endpoints to Application{idx}.",
        f"Refactor Repository{idx} to use connection pooling.",
        f"Add Prometheus metrics to Worker{idx} class.",
        f"Implement graceful shutdown for Server{idx}.",
        f"Add input validation schemas to API{idx} routes.",
    ]
    return templates[idx % len(templates)]


SHARED_SYSTEM = make_large_system(9999)
SHARED_CODE = make_large_code_context(9999)
SHARED_TASK = "Add comprehensive error handling with retry logic."


@dataclass
class RequestResult:
    ttft_ms: float
    success: bool
    response_text: str = ""


def do_request(
    base_url: str,
    system: str,
    code: str,
    task: str,
    max_tokens: int = 32,
    priority: int = 0,
    role_type: int = 2,
    crit_dist: int = 1,
    conv_factor: float = 1.0,
) -> RequestResult:
    messages = [
        {"role": "system", "content": system},
        {"role": "system", "content": code},
        {"role": "user", "content": task},
    ]
    t0 = time.perf_counter()
    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            r = client.post(f"{base_url}/v1/chat/completions", json={
                "model": MODEL_PATH,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "priority": priority,
                "role_type": role_type,
                "critical_path_distance": crit_dist,
                "convergence_factor": conv_factor,
                "next_agent_prefix": "",
            })
        elapsed = (time.perf_counter() - t0) * 1000
        if r.status_code == 200:
            data = r.json()
            text = ""
            if data.get("choices"):
                text = data["choices"][0].get("message", {}).get("content", "")
            return RequestResult(ttft_ms=elapsed, success=True, response_text=text)
        return RequestResult(ttft_ms=elapsed, success=False)
    except Exception as e:
        return RequestResult(ttft_ms=(time.perf_counter() - t0) * 1000, success=False)


def estimate_tokens(text: str) -> int:
    return len(text) // 4


def run_test(
    base_url: str,
    server_name: str,
    num_unique: int,
    large_unique: bool = True,
    verbose: bool = False,
) -> dict:
    print(f"\n  [{server_name}] Phase 1: Prime shared prefix...")
    p1 = do_request(
        base_url, SHARED_SYSTEM, SHARED_CODE, SHARED_TASK,
        max_tokens=32,
        priority=1,
        role_type=1,
        crit_dist=1,
    )
    est_shared = estimate_tokens(SHARED_SYSTEM + SHARED_CODE + SHARED_TASK)
    print(f"    Shared TTFT: {p1.ttft_ms:.1f}ms  success={p1.success}")
    print(f"    Estimated shared prefix size: ~{est_shared} tokens")

    unique_size = estimate_tokens(make_large_system(0) + make_large_code_context(0) + make_large_task(0))
    print(f"\n  [{server_name}] Phase 2: Flood {num_unique} unique requests...")
    print(f"    Estimated unique request size: ~{unique_size} tokens")
    total_unique = num_unique * unique_size
    cache_capacity = 40000
    fill_ratio = total_unique / cache_capacity
    print(f"    Total unique tokens: ~{total_unique} ({fill_ratio*100:.0f}% of {cache_capacity} cache)")
    if fill_ratio < 1.0:
        print(f"    ⚠ WARNING: unique requests only fill {fill_ratio*100:.0f}% of cache!")
        print(f"               This may not trigger eviction. Increase --num-unique")
    print()

    unique_ttfst = []
    for i in range(num_unique):
        sys_p = make_large_system(i)
        code_p = make_large_code_context(i)
        task_p = make_large_task(i)
        p = do_request(
            base_url, sys_p, code_p, task_p,
            max_tokens=32,
            priority=1000 + i,
            role_type=3,
            crit_dist=num_unique - i,
        )
        unique_ttfst.append(p.ttft_ms)
        if verbose or i % 5 == 4 or i == num_unique - 1:
            print(f"    Unique {i+1:4d}/{num_unique}: TTFT={p.ttft_ms:6.1f}ms  "
                  f"success={p.success}  avg={sum(unique_ttfst)/len(unique_ttfst):6.1f}ms")

    print(f"\n  [{server_name}] Phase 3: Re-request shared prefix (after eviction)...")
    p3 = do_request(
        base_url, SHARED_SYSTEM, SHARED_CODE, SHARED_TASK,
        max_tokens=32,
        priority=1,
        role_type=1,
        crit_dist=1,
    )
    print(f"    Shared TTFT: {p3.ttft_ms:.1f}ms  success={p3.success}")

    phase1 = p1.ttft_ms
    phase3 = p3.ttft_ms
    improvement = phase1 - phase3
    improvement_pct = (improvement / phase1 * 100) if phase1 > 0 else 0

    print(f"\n  [{server_name}] SUMMARY:")
    print(f"    Phase-1 shared TTFT : {phase1:.1f}ms  (fresh, full prefill)")
    print(f"    Phase-2 unique avg   : {sum(unique_ttfst)/len(unique_ttfst):.1f}ms")
    print(f"    Phase-3 shared TTFT  : {phase3:.1f}ms  (after eviction flood)")
    print(f"    Degradation          : {-improvement:+.1f}ms  ({-improvement_pct:+.1f}%)")
    print(f"    Interpretation        : ", end="")
    if improvement < -20:
        print("SHARED PREFIX EVICTED (significant re-prefill needed)")
    elif improvement < -5:
        print("SHARED PREFIX PARTIALLY EVICTED")
    elif improvement < 5:
        print("SHARED PREFIX LARGELY PRESERVED (cache not under pressure)")
    else:
        print("SHARED PREFIX FULLY PRESERVED (good cache behavior)")

    return {
        "server": server_name,
        "phase1": phase1,
        "phase2_avg": sum(unique_ttfst) / len(unique_ttfst),
        "phase3": phase3,
        "improvement": improvement,
        "improvement_pct": improvement_pct,
        "num_unique": num_unique,
        "unique_ttfst": unique_ttfst,
        "fill_ratio": fill_ratio,
    }


def main():
    parser = argparse.ArgumentParser(description="Adversarial Eviction Pressure Test v2")
    parser.add_argument("--num-unique", type=int, default=60,
                        help="Number of unique requests to flood (default: 60)")
    parser.add_argument("--base-url", type=str, default=None,
                        help="Single-server mode: test only this base URL")
    parser.add_argument("--server-name", type=str, default="Single",
                        help="Display name for --base-url single-server mode")
    parser.add_argument("--lru-only", action="store_true")
    parser.add_argument("--priority-only", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("  KVFlow Adversarial Eviction Pressure Test v2")
    print("  Key: Large unique requests to fill cache faster")
    print("=" * 70)

    if args.base_url:
        result = run_test(args.base_url, args.server_name, args.num_unique, verbose=args.verbose)
        print("\n" + "=" * 70)
        print("  SINGLE SERVER SUMMARY")
        print("=" * 70)
        print(f"\n  Server:    {result['server']}")
        print(f"  Phase-1:   {result['phase1']:.1f}ms")
        print(f"  Phase-2:   {result['phase2_avg']:.1f}ms")
        print(f"  Phase-3:   {result['phase3']:.1f}ms")
        print(f"  Delta:     {result['improvement']:+.1f}ms  ({result['improvement_pct']:+.1f}%)")
        print("\n" + "=" * 70)
        return

    if not args.priority_only:
        lru = run_test(LRU_BASE, "LRU", args.num_unique, verbose=args.verbose)

    if not args.lru_only:
        pri = run_test(PRIORITY_BASE, "Priority", args.num_unique, verbose=args.verbose)

    print("\n" + "=" * 70)
    print("  COMPARISON SUMMARY")
    print("=" * 70)

    if not args.priority_only and not args.lru_only:
        lru_p3 = lru["phase3"]
        pri_p3 = pri["phase3"]
        delta = lru_p3 - pri_p3
        delta_pct = (delta / lru_p3 * 100) if lru_p3 > 0 else 0

        print(f"\n  Phase-3 Shared Prefix TTFT (after eviction flood):")
        print(f"    LRU:      {lru_p3:.1f}ms")
        print(f"    Priority: {pri_p3:.1f}ms")
        print(f"    Delta:    {delta:+.1f}ms  ({delta_pct:+.1f}% faster with Priority)")

        print(f"\n  Phase-1 → Phase-3 TTFT change:")
        print(f"    LRU:      {lru['improvement']:+.1f}ms  ({lru['improvement_pct']:+.1f}%)")
        print(f"    Priority: {pri['improvement']:+.1f}ms  ({pri['improvement_pct']:+.1f}%)")

        if delta > 10:
            print(f"\n  ✓ Priority is {delta:.1f}ms ({delta_pct:.1f}%) faster")
            print(f"    Shared Tier-0/1 prefix was better preserved under eviction pressure.")
        elif delta > 2:
            print(f"\n  ≈ Slight advantage for Priority ({delta:.1f}ms)")
        elif delta < -10:
            print(f"\n  ✗ LRU is {-delta:.1f}ms faster (unexpected!)")
        else:
            print(f"\n  ≈ No significant difference ({delta:+.1f}ms)")
            if lru["fill_ratio"] < 1.0:
                print(f"    ⚠ Cache fill ratio was only {lru['fill_ratio']*100:.0f}%")
                print(f"    Increase --num-unique to trigger real eviction pressure.")
            else:
                print(f"    Both strategies evicted the shared prefix identically.")

        if lru["fill_ratio"] < 1.0:
            suggested = int(40000 / (lru["fill_ratio"] * lru["num_unique"] / lru["num_unique"])) + 10
            suggested = max(args.num_unique * 2, int(40000 / (estimate_tokens(make_large_system(0) + make_large_code_context(0) + make_large_task(0))) + 5))
            print(f"\n  Note: estimated cache capacity may be {40000} tokens.")
            print(f"  Try: --num-unique {suggested} for stronger eviction pressure.")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
