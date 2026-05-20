"""
KVFlow Optimal Scenario Benchmark: Tests Priority vs LRU when there are
TRULY SHARED prefixes across multiple workflows.

This is the scenario where KVFlow should excel: a large system prompt shared
by all agents across all workflows.

Key improvements over bench_multi_workflow.py:
  1. Cross-workflow shared system prompt: A single 4k token system prompt
     that ALL agents share. This is realistic - in real MAScoder, all agents
     share the same system instructions.
  2. Group-shared prefixes: Workflow groups (4 workflows per group) share
     a 2k token group prefix. This creates intermediate sharing.
  3. Workflow-aware agent scheduling: Agents within a workflow are executed
     in sequence, creating predictable access patterns.
  4. More rounds: 10 rounds to accumulate cache pressure.

Cache pressure design:
  - System prompt: 4096 tokens (shared by ALL agents)
  - Group prefix: 2048 tokens (shared by 4 workflows)
  - Agent prefix: 1024 tokens (per-agent unique)
  - Round pressure: (4096 + 2048 + 1024) × N agents × N workflows

With 4 workflows × 8 agents = 32 total:
  Per round: ~229,376 tokens
  60k cache: 3.8x pressure
  90k cache: 2.5x pressure

Expected behavior:
  - LRU: System prompt evicted after ~5 rounds, causing cold TTFT
  - Priority: System prompt preserved (highest priority = 0), warm TTFT
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


# ============================================================================
# Data classes
# ============================================================================

@dataclass
class AgentConfig:
    """An agent with three-level prefix hierarchy:
    1. System prompt (shared by ALL agents)
    2. Group prefix (shared by workflows in same group)
    3. Unique prefix (per-agent unique)
    """
    workflow_id: int
    group_id: int
    agent_id: str
    agent_index: int  # 0=planner, 1=implementer, etc.
    system_prompt_tokens: int
    group_prefix_tokens: int
    unique_prefix_tokens: int
    unique_prefix_text: str
    group_prefix_text: str


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


# ============================================================================
# Configs
# ============================================================================

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


# ============================================================================
# Text generation with tokenizer
# ============================================================================

def generate_calibrated_text(tokenizer, target_tokens: int) -> Tuple[str, int]:
    """Generate text that tokenizes to approximately target_tokens."""
    vocab_size = tokenizer.vocab_size
    special_ids = set(tokenizer.all_special_ids)
    token_ids = []
    for _ in range(target_tokens + 50):
        tid = random.randint(0, vocab_size - 1)
        if tid not in special_ids:
            token_ids.append(tid)
        if len(token_ids) >= target_tokens + 50:
            break
    text = tokenizer.decode(token_ids[:target_tokens], skip_special_tokens=True)
    actual = len(tokenizer.encode(text, add_special_tokens=False))
    while actual > target_tokens and len(token_ids) > 10:
        token_ids = token_ids[len(token_ids) - 5:]
        text = tokenizer.decode(token_ids, skip_special_tokens=True)
        actual = len(tokenizer.encode(text, add_special_tokens=False))
    return text, actual


def generate_system_prompt(tokenizer, target_tokens: int) -> str:
    """Generate a realistic system prompt shared by all agents."""
    system_intro = """
You are an expert AI coding assistant specialized in software engineering.
You have access to tools for reading files, searching code, running shell commands,
and executing Python code. You follow a structured workflow to solve tasks:

1. Analyze the task requirements
2. Search for relevant files and code
3. Plan the implementation approach
4. Write clean, efficient code
5. Test and verify the solution
6. Review and optimize if needed

Always think step by step before taking action. Provide detailed explanations
for your decisions. Ensure code quality and test coverage.
""".strip()

    tokens = tokenizer.encode(system_intro, add_special_tokens=False)
    if len(tokens) >= target_tokens:
        return tokenizer.decode(tokens[:target_tokens])

    # Pad with relevant context
    filler, _ = generate_calibrated_text(tokenizer, target_tokens - len(tokens))
    return system_intro + " " + filler


def generate_group_prefix(tokenizer, target_tokens: int, group_id: int) -> str:
    """Generate a group-specific prefix shared by workflows in the same group."""
    group_templates = [
        "You are working in TEAM-A: Data Pipeline Development. Focus on data processing, "
        "ETL pipelines, and database integration. Use tools: find_files, read_file, "
        "write_file, run_shell, sandbox_python.",
        "You are working in TEAM-B: Web Application Development. Focus on web services, "
        "REST APIs, and frontend-backend integration. Use tools: read_file, write_file, "
        "run_shell, sandbox_python, git_commands.",
        "You are working in TEAM-C: Machine Learning Systems. Focus on model training, "
        "data preprocessing, and ML pipelines. Use tools: find_files, read_file, "
        "write_file, run_shell, sandbox_python.",
        "You are working in TEAM-D: Infrastructure & DevOps. Focus on CI/CD, containerization, "
        "and cloud deployment. Use tools: read_file, write_file, run_shell, git_commands.",
    ]
    template = group_templates[group_id % len(group_templates)]
    tokens = tokenizer.encode(template, add_special_tokens=False)
    if len(tokens) >= target_tokens:
        return tokenizer.decode(tokens[:target_tokens])
    extra, _ = generate_calibrated_text(tokenizer, target_tokens - len(tokens))
    return template + " " + extra


def generate_unique_prefix(tokenizer, target_tokens: int, agent_index: int, workflow_id: int, agent_id: str) -> str:
    """Generate a unique per-agent prefix with role-specific vocabulary."""
    role_templates = [
        f"You are the PLANNER for {agent_id}. Analyze tasks, create execution plans, "
        "and coordinate with other agents. Break complex tasks into subtasks.",
        f"You are the ARCHITECT for {agent_id}. Design system architecture, define "
        "module boundaries, and choose appropriate design patterns.",
        f"You are the RETRIEVER for {agent_id}. Search codebase, find relevant files, "
        "extract information needed for the task implementation.",
        f"You are the IMPLEMENTER for {agent_id}. Write clean, efficient code according "
        "to specifications. Follow best practices and coding standards.",
        f"You are the TESTER for {agent_id}. Write comprehensive tests, verify correctness, "
        "and ensure code quality standards are met.",
        f"You are the REVIEWER for {agent_id}. Review code changes, provide feedback, "
        "and ensure code quality standards are maintained.",
        f"You are the DEBUGGER for {agent_id}. Investigate bugs, trace root causes, "
        "and implement fixes without introducing regressions.",
        f"You are the OPTIMIZER for {agent_id}. Profile performance, identify bottlenecks, "
        "and apply optimizations to improve efficiency.",
    ]
    role = role_templates[agent_index % len(role_templates)]
    tokens = tokenizer.encode(role, add_special_tokens=False)
    if len(tokens) >= target_tokens:
        return tokenizer.decode(tokens[:target_tokens])
    extra, _ = generate_calibrated_text(tokenizer, target_tokens - len(tokens))
    return role + " " + extra


# ============================================================================
# HTTP helpers
# ============================================================================

async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    prompt: str,
    max_tokens: int,
    priority: Optional[int] = None,
    timeout: int = 600,
) -> Tuple[float, float, int]:
    """Send a streaming chat completion request."""
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


# ============================================================================
# Agent and workflow setup
# ============================================================================

def setup_agents(
    tokenizer,
    num_workflows: int,
    agents_per_workflow: int,
    system_prompt_tokens: int,
    group_prefix_tokens: int,
    unique_prefix_tokens: int,
    workflows_per_group: int = 4,
    seed: int = 42,
) -> Tuple[List[AgentConfig], str, Dict[int, str], int]:
    """Create all agents with three-level prefix hierarchy.

    Returns:
        agents: List of AgentConfig
        system_prompt: The shared system prompt text
        group_prefixes: Dict[group_id -> prefix_text]
        total_unique_kv: Total unique KV tokens per round
    """
    random.seed(seed)

    # Generate the shared system prompt (shared by ALL agents)
    system_prompt = generate_system_prompt(tokenizer, system_prompt_tokens)

    # Generate group prefixes (shared by workflows in same group)
    num_groups = (num_workflows + workflows_per_group - 1) // workflows_per_group
    group_prefixes: Dict[int, str] = {}
    for g in range(num_groups):
        group_prefixes[g] = generate_group_prefix(tokenizer, group_prefix_tokens, g)

    # Create agents with three-level prefix
    agents: List[AgentConfig] = []
    for wf_id in range(num_workflows):
        group_id = wf_id // workflows_per_group
        for a_idx in range(agents_per_workflow):
            unique_text = generate_unique_prefix(
                tokenizer, unique_prefix_tokens, a_idx, wf_id, f"w{wf_id}-a{a_idx}"
            )
            agents.append(AgentConfig(
                workflow_id=wf_id,
                group_id=group_id,
                agent_id=f"w{wf_id}-a{a_idx}",
                agent_index=a_idx,
                system_prompt_tokens=system_prompt_tokens,
                group_prefix_tokens=group_prefix_tokens,
                unique_prefix_tokens=unique_prefix_tokens,
                unique_prefix_text=unique_text,
                group_prefix_text=group_prefixes[group_id],
            ))

    # Calculate total unique KV pressure per round
    # System prompt: shared, so only 1 copy
    # Group prefixes: num_groups copies
    # Unique prefixes: num_workflows * agents_per_workflow copies
    system_kv = system_prompt_tokens  # 1 copy
    group_kv = len(group_prefixes) * group_prefix_tokens  # num_groups copies
    unique_kv = num_workflows * agents_per_workflow * unique_prefix_tokens
    total_unique_kv = system_kv + group_kv + unique_kv

    return agents, system_prompt, group_prefixes, total_unique_kv


# ============================================================================
# Priority calculation for KVFlow
# ============================================================================

def calculate_priority(
    workflow_id: int,
    agent_index: int,
    round_idx: int,
    total_agents: int,
    total_rounds: int,
) -> int:
    """
    Calculate priority for an agent request.

    Priority encoding:
    - Lower values = higher priority (keep in cache longer)
    - Agent index 0 (planner) has lowest priority (highest importance)
    - Agent index N-1 (last) has highest priority (least important)

    For a workflow with M agents:
      Agent 0: priority = M (last to be reused)
      Agent 1: priority = M-1
      ...
      Agent M-1: priority = 1 (first to be reused next round)

    Across rounds, we add round_offset:
      Round 0: offset = 0
      Round 1: offset = M
      Round 2: offset = 2M
      ...

    This way, in Round 1, Agent 0 from Round 0 has priority = M + M = 2M,
    which is higher than Agent 0 from Round 1 (priority = M).
    So the older Agent 0 will be evicted first if needed.
    """
    base_priority = (total_agents - agent_index)  # 1 to M
    round_offset = round_idx * total_agents
    return round_offset + base_priority


# ============================================================================
# Workflow runner
# ============================================================================

async def run_one_workflow(
    session: aiohttp.ClientSession,
    url: str,
    workflow_id: int,
    agents: List[AgentConfig],
    num_rounds: int,
    output_len: int,
    use_priority: bool,
    tokenizer,
    system_prompt: str,
    total_agents: int,
    total_rounds: int,
) -> WorkflowResult:
    """Run one workflow: execute its agents for num_rounds."""
    result = WorkflowResult(workflow_id=workflow_id)

    for round_idx in range(num_rounds):
        round_results: List[StepResult] = []

        for step_idx, agent in enumerate(agents):
            # Three-level prefix: [system_prompt] [group_prefix] [unique_prefix] [dynamic_suffix]
            suffix_text, _ = generate_calibrated_text(tokenizer, 64)
            prompt = f"{system_prompt} {agent.group_prefix_text} {agent.unique_prefix_text} {suffix_text}"

            # Calculate priority
            priority = None
            if use_priority:
                priority = calculate_priority(
                    workflow_id=workflow_id,
                    agent_index=agent.agent_index,
                    round_idx=round_idx,
                    total_agents=total_agents,
                    total_rounds=total_rounds,
                )

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

            logger.debug(
                f"  wf={workflow_id} round={round_idx} step={step_idx} "
                f"agent={agent.agent_id} ttft={ttft_ms:.1f}ms "
                f"e2e={e2e_ms:.1f}ms tokens={n_tokens}"
                + (f" pri={priority}" if priority is not None else "")
            )

        result.round_results.append(round_results)

    return result


# ============================================================================
# Results printing and JSON output
# ============================================================================

def compute_aggregate(
    all_results: List[WorkflowResult],
    args: argparse.Namespace,
) -> Tuple[List[float], List[float], List[float]]:
    """Compute aggregate metrics."""
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
    total_unique_kv: int,
) -> Optional[dict]:
    """Print formatted results and return the output dict for comparison."""
    warmup = args.warmup_rounds
    total_rounds = warmup + args.num_rounds
    config_name = args.config

    ttfts, e2es, round_e2es = compute_aggregate(all_results, args)

    # Collect warmup data
    all_warmup_ttfts = [s.ttft_ms for wf in all_results if len(wf.round_results) > 0
                        for s in wf.round_results[0]]
    all_warmup_e2es = [s.e2e_ms for wf in all_results if len(wf.round_results) > 0
                       for s in wf.round_results[0]]
    warmup_ttft_avg = sum(all_warmup_ttfts)/len(all_warmup_ttfts) if all_warmup_ttfts else 0.0
    warmup_e2e_avg = sum(all_warmup_e2es)/len(all_warmup_e2es) if all_warmup_e2es else 0.0

    # Per-step warmup baseline
    warmup_ttft_by_step: Dict[int, List[float]] = {}
    for wf in all_results:
        if len(wf.round_results) > 0:
            for s in wf.round_results[0]:
                warmup_ttft_by_step.setdefault(s.step_idx, []).append(s.ttft_ms)
    warmup_ttft_per_step = {k: sum(v)/len(v) for k, v in warmup_ttft_by_step.items()}

    logger.info("=" * 80)
    logger.info(f"KVFlow Optimal Scenario Results [{config_name}]")
    logger.info("=" * 80)
    logger.info(
        f"  {args.num_workflows} workflows × {args.agents_per_workflow} agents "
        f"= {args.num_workflows * args.agents_per_workflow} total agents"
    )
    logger.info(
        f"  system_prompt={args.system_prompt_tokens}, "
        f"group_prefix={args.group_prefix_tokens}, "
        f"unique_prefix={args.unique_prefix_tokens}"
    )
    logger.info(f"  Total unique KV pressure per round: ~{total_unique_kv} tokens")
    logger.info(
        f"  {total_rounds} rounds ({warmup} warmup + {args.num_rounds} measured), "
        f"total_runtime={elapsed_seconds:.1f}s"
    )
    logger.info("-" * 80)

    # Per-round breakdown with per-step analysis
    round_data: Dict[int, Dict] = {}
    for round_idx in range(total_rounds):
        r_ttfts, r_e2es = [], []
        step_ttfts: Dict[int, List[float]] = {}
        for wf in all_results:
            if round_idx < len(wf.round_results):
                for s in wf.round_results[round_idx]:
                    r_ttfts.append(s.ttft_ms)
                    r_e2es.append(s.e2e_ms)
                    step_ttfts.setdefault(s.step_idx, []).append(s.ttft_ms)
        if r_ttfts:
            round_data[round_idx] = {
                "avg_ttft": sum(r_ttfts) / len(r_ttfts),
                "avg_e2e": sum(r_e2es) / len(r_e2es),
                "step_ttfts": {k: sum(v)/len(v) for k, v in step_ttfts.items()},
            }
            tag = " [warmup]" if round_idx < warmup else ""
            logger.info(
                f"  Round {round_idx}: avg TTFT={round_data[round_idx]['avg_ttft']:.1f}ms, "
                f"avg E2E={round_data[round_idx]['avg_e2e']:.1f}ms{tag}"
            )
            # Show per-step TTFT for first and last steps (system prompt vs unique)
            if 0 in round_data[round_idx]['step_ttfts']:
                step0_ttft = round_data[round_idx]['step_ttfts'][0]
                warmup_step0 = warmup_ttft_per_step.get(0, step0_ttft)
                ratio = step0_ttft / warmup_step0 if warmup_step0 > 0 else 1.0
                logger.info(
                    f"    Step 0 (system prompt): TTFT={step0_ttft:.1f}ms "
                    f"(vs warmup {warmup_step0:.1f}ms, {ratio:.2f}x)"
                )

    logger.info("-" * 80)

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

    # Speedup vs warmup
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
    logger.info("=" * 80)

    # Build output dict
    output: Dict[str, Any] = {
        "config": {
            k: v for k, v in vars(args).items()
            if k not in ("baseline_json",)
        },
        "warmup_rounds": warmup,
        "total_unique_kv_per_round": total_unique_kv,
        "aggregate": {
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
        },
        "round_summaries": {
            f"round_{ridx}": {
                "avg_ttft": rdata["avg_ttft"],
                "avg_e2e": rdata["avg_e2e"],
                "step_ttfts": rdata["step_ttfts"],
            }
            for ridx, rdata in round_data.items()
        },
        "warmup_ttft_per_step": warmup_ttft_per_step,
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

    # Speedup vs external baseline
    speedup_info = {}
    if args.baseline_json:
        try:
            with open(args.baseline_json, "r") as f:
                baseline_data = json.load(f)
            bl_ttft = baseline_data["aggregate"]["ttft_avg_ms"]
            bl_e2e = baseline_data["aggregate"]["round_e2e_avg_ms"]
            logger.info("-" * 80)
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
        f"kvflow_opt_{config_name}_{total_agents}agents_"
        f"{args.system_prompt_tokens}sys_{args.group_prefix_tokens}grp_"
        f"{args.unique_prefix_tokens}uni_{args.num_rounds}rounds_{args.num_workflows}wf.json",
    )
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Results written to {output_file}")

    return output


# ============================================================================
# Main
# ============================================================================

async def main():
    parser = argparse.ArgumentParser(
        description="KVFlow Optimal Scenario Benchmark"
    )
    parser.add_argument(
        "--config", type=str, required=True, choices=list(CONFIGS.keys()),
        help="Test configuration",
    )
    parser.add_argument(
        "--num-workflows", type=int, default=8,
        help="Number of concurrent workflows (default: 8)",
    )
    parser.add_argument(
        "--agents-per-workflow", type=int, default=8,
        help="Agents per workflow (default: 8)",
    )
    parser.add_argument(
        "--system-prompt-tokens", type=int, default=4096,
        help="Length of system prompt in tokens (default: 4096)",
    )
    parser.add_argument(
        "--group-prefix-tokens", type=int, default=2048,
        help="Length of group prefix in tokens (default: 2048)",
    )
    parser.add_argument(
        "--unique-prefix-tokens", type=int, default=1024,
        help="Length of unique per-agent prefix in tokens (default: 1024)",
    )
    parser.add_argument(
        "--output-len", type=int, default=64,
        help="Max output tokens per step (default: 64)",
    )
    parser.add_argument(
        "--num-rounds", type=int, default=10,
        help="Number of measured rounds (default: 10)",
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
        "--model", type=str, required=True,
        help="Model path for tokenizer",
    )
    parser.add_argument(
        "--output-dir", type=str, default="/home/comp/25480812/logs/kvflow-optimal",
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
    total_rounds = args.warmup_rounds + args.num_rounds

    random.seed(args.seed)
    url = f"http://{args.host}:{args.port}"

    from transformers import AutoTokenizer
    logger.info(f"Loading tokenizer from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Create agents with three-level prefix hierarchy
    agents, system_prompt, group_prefixes, total_unique_kv = setup_agents(
        tokenizer,
        args.num_workflows,
        args.agents_per_workflow,
        args.system_prompt_tokens,
        args.group_prefix_tokens,
        args.unique_prefix_tokens,
        seed=args.agents_seed,
    )

    logger.info(f"Scenario:")
    logger.info(f"  {args.num_workflows} workflows × {args.agents_per_workflow} agents = {total_agents} total")
    logger.info(f"  System prompt: {args.system_prompt_tokens} tokens (shared by ALL)")
    logger.info(f"  Group prefix: {args.group_prefix_tokens} tokens (shared by {len(group_prefixes)} groups)")
    logger.info(f"  Unique prefix: {args.unique_prefix_tokens} tokens (per-agent)")
    logger.info(f"  Total unique KV pressure per round: ~{total_unique_kv} tokens")
    logger.info(f"  Priority metadata: {'enabled' if use_priority else 'disabled'}")
    logger.info(f"  Target server: {url}")

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

        # Run workflows
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
                    system_prompt=system_prompt,
                    total_agents=total_agents,
                    total_rounds=total_rounds,
                )
            )

        all_results = await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - start_time

        await asyncio.sleep(1)
        loads_after = await fetch_loads(session, url)

        print_and_save_results(
            list(all_results), loads_before, loads_after, args, elapsed, total_unique_kv
        )


if __name__ == "__main__":
    asyncio.run(main())
