"""
Multi-Workflow KVFlow Benchmark: Tests Priority vs LRU under realistic
multi-agent multi-workflow scenarios.

Key improvements over bench_priority.py:
  1. Multi-workflow concurrency: truly parallel workflows, not just
     sequential steps within one workflow
  2. Shared + unique prefix design: simulates the system-prompt sharing
     pattern in real MAScoder workflows
  3. Per-workflow agent pools: each workflow has its own agent IDs,
     enabling cross-workflow cache pressure
  4. write_back HiCache: avoids the locked-node deadlock that plagued
     write_through + prefetch in the original benchmarks
  5. Per-round detailed metrics: tracks which prefixes were cached and why

Scenario design:
  - shared_p_len: tokens shared across ALL agent prefixes (e.g. system prompt)
  - unique_p_len: per-agent unique tokens (e.g. agent role description)
  - suffix_len: dynamic tokens per request

Cache pressure calculation:
  - Each workflow puts ~(shared_p_len + unique_p_len) tokens per agent step
  - N workflows × M agents per workflow × N rounds = total cache pressure
  - With shared prefix dedup: shared_p_len + N×M×unique_p_len total unique tokens

Usage:
  # Terminal 1: start server with Priority
  python -m sglang.launch_server \
    --model-path /path/to/Qwen3-8B \
    --port 30300 \
    --radix-eviction-policy priority \
    --max-total-tokens 60000 \
    --enable-cache-report

  # Terminal 2: run multi-workflow benchmark
  python -m benchmark.multi_workflow.bench_multi_workflow \
    --config priority \
    --host 127.0.0.1 --port 30300 \
    --num-workflows 4 --agents-per-workflow 5 \
    --shared-p-len 2048 --unique-p-len 1024 \
    --suffix-len 64 --output-len 64 \
    --num-rounds 5 --warmup-rounds 1 \
    --num-concurrent 1 \
    --output-dir /tmp/kvflow_results

  # Compare with LRU baseline (run on separate server instance):
  # (same command with --config baseline, server with --radix-eviction-policy lru)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AgentConfig:
    """An agent with a two-part prefix: [shared_prefix] + [unique_prefix]"""
    workflow_id: int
    agent_id: str           # e.g. "w0-a0", "w1-a3"
    shared_prefix_tokens: int
    unique_prefix_tokens: int
    unique_prefix_text: str
    shared_prefix_text: str  # reference only (not owned)


@dataclass
class StepResult:
    workflow_id: int
    agent_id: str
    round_idx: int
    step_idx: int
    ttft_ms: float
    e2e_ms: float
    output_tokens: int
    priority: Optional[int] = None


@dataclass
class WorkflowResult:
    workflow_id: int
    round_results: List[List[StepResult]] = field(default_factory=list)




# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

CONFIGS = {
    "hicache": {
        "label": "LRU + HiCache write_through (baseline)",
        "use_priority": False,
        "server_note": "Server: --radix-eviction-policy lru --hicache-write-policy write_through",
    },
    "hicache90k": {
        "label": "LRU + HiCache write_back (90k cache, fair comparison with kvflow)",
        "use_priority": False,
        "server_note": "Server: --radix-eviction-policy lru --hicache-write-policy write_back --enable-hicache-prefetch",
    },
    "kvflow": {
        "label": "Priority + HiCache write_back + proactive prefetch",
        "use_priority": True,
        "server_note": "Server: --radix-eviction-policy priority --hicache-write-policy write_back --enable-hicache-prefetch",
    },
}


# ---------------------------------------------------------------------------
# Tokenizer-calibrated text generation
# ---------------------------------------------------------------------------

def generate_calibrated_text(tokenizer, target_tokens: int) -> Tuple[str, int]:
    """Generate text that tokenizes to approximately target_tokens."""
    vocab_size = tokenizer.vocab_size
    special_ids = set(tokenizer.all_special_ids)
    token_ids = []
    for _ in range(target_tokens + 20):
        tid = random.randint(0, vocab_size - 1)
        if tid not in special_ids:
            token_ids.append(tid)
        if len(token_ids) >= target_tokens + 20:
            break
    text = tokenizer.decode(token_ids[:target_tokens], skip_special_tokens=True)
    actual = len(tokenizer.encode(text, add_special_tokens=False))
    while actual > target_tokens and len(token_ids) > 10:
        token_ids = token_ids[len(token_ids) - 5:]
        text = tokenizer.decode(token_ids, skip_special_tokens=True)
        actual = len(tokenizer.encode(text, add_special_tokens=False))
    return text, actual


def generate_shared_prefix(tokenizer, target_tokens: int) -> str:
    """Generate a realistic shared system prompt of ~target_tokens."""
    system_intro = (
        "You are a helpful AI assistant specialized in software engineering. "
        "You have access to tools for reading files, searching code, running shell commands, "
        "and executing Python code. Follow the instructions provided by the user carefully. "
        "Always think step by step before taking action."
    )
    tokens = tokenizer.encode(system_intro, add_special_tokens=False)
    if len(tokens) >= target_tokens:
        return tokenizer.decode(tokens[:target_tokens])
    # Pad with filler content
    filler, _ = generate_calibrated_text(tokenizer, target_tokens - len(tokens))
    return system_intro + " " + filler


def generate_unique_prefix(tokenizer, target_tokens: int, agent_index: int, workflow_id: int) -> str:
    """Generate a unique per-agent prefix with role-specific vocabulary."""
    role_templates = [
        "You are the PLANNER agent. Your job is to analyze the task, break it down into "
        "subtasks, and create a detailed execution plan with priorities.",
        "You are the ARCHITECT agent. Your job is to design the system architecture, "
        "define module boundaries, and choose appropriate design patterns.",
        "You are the RETRIEVER agent. Your job is to search through the codebase, "
        "find relevant files, and extract information needed for the task.",
        "You are the IMPLEMENTER agent. Your job is to write clean, efficient code "
        "according to the specifications provided by the architect.",
        "You are the TESTER agent. Your job is to write comprehensive unit tests, "
        "integration tests, and verify the correctness of implementations.",
        "You are the REVIEWER agent. Your job is to review code changes, provide "
        "constructive feedback, and ensure code quality standards are met.",
        "You are the OPTIMIZER agent. Your job is to profile performance, identify "
        "bottlenecks, and apply optimizations to improve efficiency.",
        "You are the DEBUGGER agent. Your job is to investigate bugs, trace their "
        "root causes, and implement fixes without introducing regressions.",
    ]
    role = role_templates[agent_index % len(role_templates)]
    tokens = tokenizer.encode(role, add_special_tokens=False)
    if len(tokens) >= target_tokens:
        return tokenizer.decode(tokens[:target_tokens])
    extra, _ = generate_calibrated_text(tokenizer, target_tokens - len(tokens))
    return role + " Additional context: " + extra


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    prompt: str,
    max_tokens: int,
    priority: Optional[int] = None,
    timeout: int = 600,
) -> Tuple[float, float, int]:
    """Send a streaming chat completion request.
    Returns (ttft_ms, e2e_ms, num_output_tokens).
    """
    payload: Dict[str, Any] = {
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": True,
        "top_k": 1,
    }
    if priority is not None:
        payload["priority"] = priority

    start = time.perf_counter()
    ttft: Optional[float] = None
    num_tokens = 0

    try:
        async with session.post(
            f"{url}/v1/chat/completions", json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout, sock_read=300)
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                logger.error(f"Request failed ({resp.status}): {body[:500]}")
                e2e = (time.perf_counter() - start) * 1000
                return (e2e, e2e, 0)

            async for raw_line in resp.content:
                line = raw_line.decode().strip()
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                choices = chunk.get("choices", [])
                for choice in choices:
                    delta = choice.get("delta", {})
                    if delta.get("content"):
                        num_tokens += 1
                        if ttft is None:
                            ttft = (time.perf_counter() - start) * 1000
    except asyncio.TimeoutError:
        e2e = (time.perf_counter() - start) * 1000
        logger.error(f"Request timed out after {timeout}s")
        return (e2e, e2e, 0)

    e2e = (time.perf_counter() - start) * 1000
    return (ttft or e2e, e2e, num_tokens)


async def fetch_loads(session: aiohttp.ClientSession, url: str) -> dict:
    """Fetch load metrics from /v1/loads endpoint."""
    try:
        async with session.get(f"{url}/v1/loads") as resp:
            if resp.status == 200:
                return await resp.json()
    except Exception as e:
        logger.warning(f"Could not fetch /v1/loads: {e}")
    return {}


# ---------------------------------------------------------------------------
# Agent and workflow setup
# ---------------------------------------------------------------------------

def setup_agents(
    tokenizer,
    num_workflows: int,
    agents_per_workflow: int,
    shared_p_len: int,
    unique_p_len: int,
    seed: int,
) -> Tuple[List[AgentConfig], str]:
    """Create all agents with shared + unique prefix structure.
    Returns (agents, shared_prefix_text).
    """
    random.seed(seed)
    shared_prefix = generate_shared_prefix(tokenizer, shared_p_len)

    agents: List[AgentConfig] = []
    for wf_id in range(num_workflows):
        for a_id in range(agents_per_workflow):
            unique_text = generate_unique_prefix(
                tokenizer, unique_p_len, a_id, wf_id
            )
            agents.append(AgentConfig(
                workflow_id=wf_id,
                agent_id=f"w{wf_id}-a{a_id}",
                shared_prefix_tokens=shared_p_len,
                unique_prefix_tokens=unique_p_len,
                unique_prefix_text=unique_text,
                shared_prefix_text=shared_prefix,
            ))

    # Verify token counts
    for a in agents:
        combined = a.shared_prefix_text + " " + a.unique_prefix_text
        actual = len(tokenizer.encode(combined, add_special_tokens=False))
        a._combined_len = actual

    return agents, shared_prefix


# ---------------------------------------------------------------------------
# Workflow runner
# ---------------------------------------------------------------------------

async def run_one_workflow(
    session: aiohttp.ClientSession,
    url: str,
    workflow_id: int,
    agents: List[AgentConfig],
    num_rounds: int,
    output_len: int,
    use_priority: bool,
    tokenizer,
    global_step_counter: int,
    global_lock: asyncio.Lock,
) -> Tuple[WorkflowResult, int]:
    """Run one workflow: execute its agents round-robin for num_rounds.
    Returns (WorkflowResult, final_global_step_counter).
    """
    result = WorkflowResult(workflow_id=workflow_id)
    num_agents = len(agents)

    for round_idx in range(num_rounds):
        round_results: List[StepResult] = []

        for step_idx, agent in enumerate(agents):
            # Full prompt: [shared_prefix] [unique_prefix] [dynamic_suffix]
            suffix_text, _ = generate_calibrated_text(tokenizer, agent.suffix_len)
            prompt = f"{agent.shared_prefix_text} {agent.unique_prefix_text} {suffix_text}"

            # Priority: use the GLOBAL step counter to encode absolute position.
            # Higher value = further from next use = evict first.
            priority = None
            if use_priority:
                async with global_lock:
                    p = global_step_counter + num_agents
                    global_step_counter += 1
                priority = p

            ttft_ms, e2e_ms, n_tokens = await send_request(
                session=session,
                url=url,
                prompt=prompt,
                max_tokens=output_len,
                priority=priority,
            )

            step_result = StepResult(
                workflow_id=workflow_id,
                agent_id=agent.agent_id,
                round_idx=round_idx,
                step_idx=step_idx,
                ttft_ms=ttft_ms,
                e2e_ms=e2e_ms,
                output_tokens=n_tokens,
                priority=priority,
            )
            round_results.append(step_result)

            logger.info(
                f"  wf={workflow_id} round={round_idx} step={step_idx} "
                f"agent={agent.agent_id} ttft={ttft_ms:.1f}ms "
                f"e2e={e2e_ms:.1f}ms tokens={n_tokens}"
                + (f" pri={priority}" if priority is not None else "")
            )

        result.round_results.append(round_results)

    return result, global_step_counter


# ---------------------------------------------------------------------------
# Results printing and JSON output
# ---------------------------------------------------------------------------

def compute_aggregate(
    all_results: List[WorkflowResult],
    args: argparse.Namespace,
) -> Tuple[List[float], List[float], List[float]]:
    """Compute aggregate metrics from all workflow results.
    Returns (ttfts, e2es, round_e2es).
    """
    warmup = args.warmup_rounds
    ttfts, e2es, round_e2es = [], [], []

    for wf in all_results:
        for round_idx, round_steps in enumerate(wf.round_results):
            if round_idx < warmup:
                continue
            for s in round_steps:
                ttfts.append(s.ttft_ms)
                e2es.append(s.e2e_ms)
            round_e2es.append(sum(s.e2e_ms for s in round_steps))

    return ttfts, e2es, round_e2es


def print_and_save_results(
    all_results: List[WorkflowResult],
    loads_before: dict,
    loads_after: dict,
    args: argparse.Namespace,
    elapsed_seconds: float,
) -> Optional[dict]:
    """Print formatted results and return the output dict for comparison.
    
    Cache reuse is measured by comparing TTFT/E2E in measured rounds vs warmup round:
    - warmup round (round 0): cold cache, all tokens must be computed
    - measured rounds (1..N): warm cache, reuse depends on eviction policy
    
    Key metrics:
    - ttft_warmup_avg: average TTFT in warmup round (baseline)
    - ttft_measured_avg: average TTFT in measured rounds
    - ttft_speedup: warmup_TTFT / measured_TTFT (>1 = cache helped)
    - e2e_speedup: warmup_E2E / measured_E2E (>1 = cache helped)
    """
    warmup = args.warmup_rounds
    total_rounds = warmup + args.num_rounds
    config_name = args.config

    ttfts, e2es, round_e2es = compute_aggregate(all_results, args)

    # Collect warmup data for baseline comparison
    warmup_ttft_by_step: Dict[int, List[float]] = {}
    warmup_e2e_by_step: Dict[int, List[float]] = {}
    for wf in all_results:
        if 0 < len(wf.round_results):
            for s in wf.round_results[0]:
                warmup_ttft_by_step.setdefault(s.step_idx, []).append(s.ttft_ms)
                warmup_e2e_by_step.setdefault(s.step_idx, []).append(s.e2e_ms)

    # Compute per-step warmup baselines (avg across all workflows)
    warmup_ttft_per_step = {k: sum(v)/len(v) for k, v in warmup_ttft_by_step.items()}
    warmup_e2e_per_step = {k: sum(v)/len(v) for k, v in warmup_e2e_by_step.items()}

    # Compute warmup round's avg TTFT and E2E as overall baseline
    all_warmup_ttfts = [s.ttft_ms for wf in all_results if len(wf.round_results) > 0
                        for s in wf.round_results[0]]
    all_warmup_e2es = [s.e2e_ms for wf in all_results if len(wf.round_results) > 0
                       for s in wf.round_results[0]]
    warmup_ttft_avg = sum(all_warmup_ttfts)/len(all_warmup_ttfts) if all_warmup_ttfts else 0.0
    warmup_e2e_avg = sum(all_warmup_e2es)/len(all_warmup_e2es) if all_warmup_e2es else 0.0

    logger.info("=" * 72)
    logger.info(f"Multi-Workflow KVFlow Results [{config_name}]")
    logger.info("=" * 72)
    logger.info(
        f"  {args.num_workflows} workflows × {args.agents_per_workflow} agents "
        f"= {args.num_workflows * args.agents_per_workflow} total agents"
    )
    logger.info(
        f"  shared_p_len={args.shared_p_len}, unique_p_len={args.unique_p_len}, "
        f"suffix_len={args.suffix_len}, output_len={args.output_len}"
    )
    logger.info(
        f"  {total_rounds} rounds ({warmup} warmup + {args.num_rounds} measured), "
        f"total_runtime={elapsed_seconds:.1f}s"
    )
    kv_pressure = args.num_workflows * args.agents_per_workflow * (args.shared_p_len + args.unique_p_len)
    logger.info(f"  Total KV pressure per round: {kv_pressure} tokens")
    logger.info("-" * 72)

    # Per-round breakdown
    round_data: Dict[int, Dict] = {}
    for round_idx in range(total_rounds):
        r_ttfts, r_e2es = [], []
        for wf in all_results:
            if round_idx < len(wf.round_results):
                for s in wf.round_results[round_idx]:
                    r_ttfts.append(s.ttft_ms)
                    r_e2es.append(s.e2e_ms)
        if r_ttfts:
            round_data[round_idx] = {
                "avg_ttft": sum(r_ttfts) / len(r_ttfts),
                "avg_e2e": sum(r_e2es) / len(r_e2es),
            }
            tag = " [warmup]" if round_idx < warmup else ""
            logger.info(
                f"  Round {round_idx}: avg TTFT={round_data[round_idx]['avg_ttft']:.1f}ms, "
                f"avg E2E={round_data[round_idx]['avg_e2e']:.1f}ms{tag}"
            )

    logger.info("-" * 72)

    # Aggregate stats
    avg_ttft = avg_e2e = avg_round_e2e = 0.0
    p50_ttft = p90_ttft = p50_e2e = p90_e2e = 0.0

    if ttfts:
        ttfts.sort()
        e2es.sort()
        avg_ttft = sum(ttfts) / len(ttfts)
        avg_e2e = sum(e2es) / len(e2es)
        p50_ttft = ttfts[len(ttfts) // 2]
        p90_ttft = ttfts[int(len(ttfts) * 0.9)]
        p50_e2e = e2es[len(e2es) // 2]
        p90_e2e = e2es[int(len(e2es) * 0.9)]

    if round_e2es:
        round_e2es.sort()
        avg_round_e2e = sum(round_e2es) / len(round_e2es)

    # Speedup vs warmup baseline
    ttft_speedup = (warmup_ttft_avg / avg_ttft) if avg_ttft > 0 else 0.0
    e2e_speedup = (warmup_e2e_avg / avg_e2e) if avg_e2e > 0 else 0.0

    logger.info(
        f"TTFT: avg={avg_ttft:.2f}ms, p50={p50_ttft:.2f}ms, p90={p90_ttft:.2f}ms"
    )
    logger.info(
        f"E2E:  avg={avg_e2e:.2f}ms, p50={p50_e2e:.2f}ms, p90={p90_e2e:.2f}ms"
    )
    logger.info(f"Round E2E: avg={avg_round_e2e:.2f}ms")
    if warmup_ttft_avg > 0:
        logger.info(
            f"Speedup vs warmup: TTFT={ttft_speedup:.2f}x "
            f"(warmup={warmup_ttft_avg:.1f}ms → measured={avg_ttft:.1f}ms), "
            f"E2E={e2e_speedup:.2f}x "
            f"(warmup={warmup_e2e_avg:.1f}ms → measured={avg_e2e:.1f}ms)"
        )

    logger.info("=" * 72)

    # Build output dict
    aggregate: Dict[str, Any] = {
        "ttft_avg_ms": avg_ttft,
        "ttft_p50_ms": p50_ttft,
        "ttft_p90_ms": p90_ttft,
        "e2e_avg_ms": avg_e2e,
        "e2e_p50_ms": p50_e2e,
        "e2e_p90_ms": p90_e2e,
        "round_e2e_avg_ms": avg_round_e2e,
        "warmup_ttft_avg_ms": warmup_ttft_avg,
        "warmup_e2e_avg_ms": warmup_e2e_avg,
        "ttft_speedup_vs_warmup": ttft_speedup,
        "e2e_speedup_vs_warmup": e2e_speedup,
    }

    # Round-by-round data
    round_summaries = {}
    for ridx, rdata in round_data.items():
        round_summaries[f"round_{ridx}"] = rdata

    output: Dict[str, Any] = {
        "config": {
            k: v for k, v in vars(args).items()
            if k not in ("baseline_json",)
        },
        "warmup_rounds": warmup,
        "aggregate": aggregate,
        "round_summaries": round_summaries,
        "loads_before": loads_before,
        "loads_after": loads_after,
        "elapsed_seconds": elapsed_seconds,
        "results": [
            {
                "workflow_id": wf.workflow_id,
                "rounds": [
                    [
                        {
                            "agent_id": s.agent_id,
                            "round_idx": s.round_idx,
                            "step_idx": s.step_idx,
                            "ttft_ms": s.ttft_ms,
                            "e2e_ms": s.e2e_ms,
                            "output_tokens": s.output_tokens,
                            "priority": s.priority,
                        }
                        for s in round_steps
                    ]
                    for round_steps in wf.round_results
                ],
            }
            for wf in all_results
        ],
    }

    # Speedup vs external baseline JSON
    baseline_data = None
    speedup_info = {}
    if args.baseline_json:
        try:
            with open(args.baseline_json, "r") as f:
                baseline_data = json.load(f)
            bl_ttft = baseline_data["aggregate"]["ttft_avg_ms"]
            bl_e2e = baseline_data["aggregate"]["round_e2e_avg_ms"]
            logger.info("-" * 72)
            logger.info("Speedup vs external baseline:")
            if avg_ttft > 0:
                ext_ttft_speedup = bl_ttft / avg_ttft
                speedup_info["ttft_vs_baseline"] = ext_ttft_speedup
                logger.info(
                    f"  TTFT: {ext_ttft_speedup:.2f}x "
                    f"(baseline={bl_ttft:.1f}ms, current={avg_ttft:.1f}ms)"
                )
            if avg_round_e2e > 0:
                ext_e2e_speedup = bl_e2e / avg_round_e2e
                speedup_info["round_e2e_vs_baseline"] = ext_e2e_speedup
                logger.info(
                    f"  Round E2E: {ext_e2e_speedup:.2f}x "
                    f"(baseline={bl_e2e:.1f}ms, current={avg_round_e2e:.1f}ms)"
                )
        except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
            logger.warning(f"Could not load baseline JSON: {e}")

    if speedup_info:
        output["speedup"] = speedup_info

    # Write JSON
    os.makedirs(args.output_dir, exist_ok=True)
    total_agents = args.num_workflows * args.agents_per_workflow
    output_file = os.path.join(
        args.output_dir,
        f"mwf_{config_name}_{total_agents}agents_"
        f"{args.shared_p_len}shr_{args.unique_p_len}uni_"
        f"{args.num_rounds}rounds_{args.num_workflows}wf.json",
    )
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Results written to {output_file}")

    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(
        description="Multi-Workflow KVFlow Benchmark for SGLang Priority vs LRU"
    )
    parser.add_argument(
        "--config", type=str, required=True, choices=list(CONFIGS.keys()),
        help="Test configuration: baseline (LRU) or priority (Priority)",
    )
    parser.add_argument(
        "--num-workflows", type=int, default=4,
        help="Number of concurrent workflows (default: 4)",
    )
    parser.add_argument(
        "--agents-per-workflow", type=int, default=5,
        help="Agents per workflow (default: 5)",
    )
    parser.add_argument(
        "--shared-p-len", type=int, default=2048,
        help="Length of shared prefix in tokens (default: 2048)",
    )
    parser.add_argument(
        "--unique-p-len", type=int, default=1024,
        help="Length of per-agent unique prefix in tokens (default: 1024)",
    )
    parser.add_argument(
        "--suffix-len", type=int, default=64,
        help="Dynamic suffix length in tokens (default: 64)",
    )
    parser.add_argument(
        "--output-len", type=int, default=64,
        help="Max output tokens per step (default: 64)",
    )
    parser.add_argument(
        "--num-rounds", type=int, default=5,
        help="Number of measured rounds (default: 5)",
    )
    parser.add_argument(
        "--warmup-rounds", type=int, default=1,
        help="Number of warmup rounds (default: 1)",
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1",
        help="Server host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port", type=int, default=30300,
        help="Server port (default: 30300)",
    )
    parser.add_argument(
        "--model", type=str,
        default="/home/comp/csgfyu/models/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218",
        help="Model path for tokenizer",
    )
    parser.add_argument(
        "--output-dir", type=str, default="/home/comp/csgfyu/logs/kvflow-multi-workflow",
        help="Output directory for JSON results",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--baseline-json", type=str, default=None,
        help="Path to baseline JSON for speedup comparison",
    )
    parser.add_argument(
        "--agents-seed", type=int, default=42,
        help="Random seed for agent prefix generation (default: 42)",
    )
    args = parser.parse_args()

    cfg = CONFIGS[args.config]
    use_priority = cfg["use_priority"]
    logger.info(f"Config: {args.config} -- {cfg['label']}")
    logger.info(f"  {cfg['server_note']}")

    total_agents = args.num_workflows * args.agents_per_workflow
    total_prefix = args.shared_p_len + args.unique_p_len
    total_kv_pressure = total_agents * total_prefix

    logger.info(f"Scenario:")
    logger.info(f"  {args.num_workflows} workflows × {args.agents_per_workflow} agents = {total_agents} total")
    logger.info(f"  Shared prefix: {args.shared_p_len} tokens (all agents)")
    logger.info(f"  Unique prefix: {args.unique_p_len} tokens (per-agent)")
    logger.info(f"  Total unique KV pressure per round: {total_kv_pressure} tokens")

    random.seed(args.seed)
    url = f"http://{args.host}:{args.port}"

    from transformers import AutoTokenizer
    logger.info(f"Loading tokenizer from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Create agents
    agents, shared_prefix = setup_agents(
        tokenizer,
        args.num_workflows,
        args.agents_per_workflow,
        args.shared_p_len,
        args.unique_p_len,
        args.agents_seed,
    )
    logger.info(
        f"Created {args.num_workflows} workflows × {args.agents_per_workflow} agents "
        f"= {total_agents} total agents"
    )
    logger.info(f"Priority metadata: {'enabled' if use_priority else 'disabled (baseline LRU)'}")
    logger.info(f"Target server: {url}")

    # Attach suffix_len to agents for the runner
    for a in agents:
        a.suffix_len = args.suffix_len

    total_rounds = args.warmup_rounds + args.num_rounds
    global_step_counter = 0
    global_lock = asyncio.Lock()

    async with aiohttp.ClientSession() as session:
        # Connection check
        try:
            async with session.get(f"{url}/health_generate",
                                   timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    logger.error(f"Health check failed: {resp.status}")
                    return
        except Exception as e:
            logger.error(f"Cannot connect to server at {url}: {e}")
            return

        logger.info("Server connection verified")

        loads_before = await fetch_loads(session, url)

        logger.info(
            f"Running {args.num_workflows} concurrent workflows, "
            f"{total_rounds} rounds each "
            f"({args.warmup_rounds} warmup + {args.num_rounds} measured)..."
        )

        start_time = time.perf_counter()

        # Each workflow gets a copy of its own agents
        tasks = []
        for wf_id in range(args.num_workflows):
            wf_agents = [a for a in agents if a.workflow_id == wf_id]
            tasks.append(
                run_one_workflow(
                    session=session,
                    url=url,
                    workflow_id=wf_id,
                    agents=wf_agents,
                    num_rounds=total_rounds,
                    output_len=args.output_len,
                    use_priority=use_priority,
                    tokenizer=tokenizer,
                    global_step_counter=global_step_counter,
                    global_lock=global_lock,
                )
            )

        results_and_counters = await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - start_time

        await asyncio.sleep(1)
        loads_after = await fetch_loads(session, url)

        # Collect results (ignore updated counters)
        all_results = [r for r, _ in results_and_counters]

        print_and_save_results(all_results, loads_before, loads_after, args, elapsed)


if __name__ == "__main__":
    asyncio.run(main())
