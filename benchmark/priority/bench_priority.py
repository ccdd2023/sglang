"""
Priority Benchmark: Multi-Agent Workflow Benchmark for KV Cache Eviction Policies.

Simulates N agents executing in round-robin, each with a unique long prefix
(system prompt) and a short dynamic suffix. Measures how well the eviction
policy preserves prefixes that will be reused soon.

Two configs via --config:
  baseline  - No priority field; server uses default LRU eviction
  priority  - Sends priority = step_counter + steps_to_execution (positive absolute step); server uses PriorityStrategy

Requires the SGLang server to be started with:
  --enable-cache-report          (to report cached_tokens in usage)
  --radix-eviction-policy lru    (for baseline)
  --radix-eviction-policy priority  (for priority)

Usage:
    python -m benchmark.priority.bench_priority --config baseline --port 30200
    python -m benchmark.priority.bench_priority --config priority --port 30200

    # Compare priority results against saved baseline
    python -m benchmark.priority.bench_priority --config priority --port 30200 \
        --baseline-json benchmark_results/priority_bench_baseline_....json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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
    agent_id: str
    prefix_text: str          # fixed system prompt (tokenizer-calibrated)
    prefix_actual_tokens: int  # actual token count after re-encoding
    suffix_len: int           # target dynamic suffix length per call (tokens)


@dataclass
class StepResult:
    agent_id: str
    round_idx: int
    step_idx: int
    ttft_ms: float
    e2e_ms: float
    output_tokens: int
    cached_tokens: int        # from usage.prompt_tokens_details.cached_tokens


@dataclass
class WorkflowResult:
    workflow_id: int
    round_results: List[List[StepResult]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

CONFIGS = {
    "baseline": {
        "label": "SGLang baseline (RadixTree + LRU)",
        "use_priority": False,
        "server_note": "Server must use: --radix-eviction-policy lru",
    },
    "priority": {
        "label": "Priority priority (RadixTree + PriorityStrategy)",
        "use_priority": True,
        "server_note": "Server must use: --radix-eviction-policy priority",
    },
}


# ---------------------------------------------------------------------------
# Tokenizer-calibrated text generation
# ---------------------------------------------------------------------------

def generate_calibrated_text(tokenizer, target_tokens: int) -> tuple[str, int]:
    """Generate text that tokenizes to approximately target_tokens.

    Samples random token IDs from the vocabulary (excluding special tokens),
    decodes to text, then re-encodes to verify the actual count.
    Returns (text, actual_token_count).
    """
    vocab_size = tokenizer.vocab_size
    special_ids = set(tokenizer.all_special_ids)

    # Sample random valid token IDs (overshoot slightly, then trim)
    token_ids = []
    for _ in range(target_tokens + 20):
        tid = random.randint(0, vocab_size - 1)
        if tid not in special_ids:
            token_ids.append(tid)
        if len(token_ids) >= target_tokens + 20:
            break

    # Decode to text, re-encode to measure actual count, trim if needed
    text = tokenizer.decode(token_ids[:target_tokens], skip_special_tokens=True)
    actual = len(tokenizer.encode(text, add_special_tokens=False))

    # If too long, trim by decoding fewer tokens
    while actual > target_tokens and len(token_ids) > 10:
        token_ids = token_ids[: len(token_ids) - 5]
        text = tokenizer.decode(token_ids, skip_special_tokens=True)
        actual = len(tokenizer.encode(text, add_special_tokens=False))

    return text, actual


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    prompt: str,
    max_tokens: int,
    priority: Optional[int] = None,
) -> tuple[float, float, int, int]:
    """Send a streaming chat completion request.

    Returns (ttft_ms, e2e_ms, num_output_tokens, cached_tokens).

    Uses stream_options.include_usage to get cached_tokens in the final
    usage chunk (requires --enable-cache-report on the server).
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
    cached_tokens = 0

    async with session.post(f"{url}/v1/chat/completions", json=payload) as resp:
        if resp.status != 200:
            body = await resp.text()
            logger.error(f"Request failed ({resp.status}): {body[:500]}")
            e2e = (time.perf_counter() - start) * 1000
            return (e2e, e2e, 0, 0)

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

            # Check for the final usage-only chunk (choices=[])
            usage = chunk.get("usage")
            if usage:
                details = usage.get("prompt_tokens_details")
                if details and isinstance(details, dict):
                    cached_tokens = details.get("cached_tokens", 0)

            # Count content-bearing chunks for output token counting
            choices = chunk.get("choices", [])
            for choice in choices:
                delta = choice.get("delta", {})
                if delta.get("content"):
                    num_tokens += 1
                    if ttft is None:
                        ttft = (time.perf_counter() - start) * 1000

    e2e = (time.perf_counter() - start) * 1000
    return (ttft or e2e, e2e, num_tokens, cached_tokens)


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
# Workflow runner
# ---------------------------------------------------------------------------

async def run_one_workflow(
    session: aiohttp.ClientSession,
    url: str,
    agents: List[AgentConfig],
    num_rounds: int,
    output_len: int,
    workflow_id: int,
    use_priority: bool,
    tokenizer,
) -> WorkflowResult:
    """Run a single workflow: round-robin through agents for num_rounds."""
    result = WorkflowResult(workflow_id=workflow_id)
    num_agents = len(agents)
    step_counter = 0  # global step counter across all rounds

    for round_idx in range(num_rounds):
        round_results: List[StepResult] = []
        for step_idx, agent in enumerate(agents):
            # Dynamic suffix changes each round (tokenizer-calibrated)
            suffix_text, _ = generate_calibrated_text(tokenizer, agent.suffix_len)
            prompt = agent.prefix_text + " " + suffix_text

            # Positive absolute step encoding:
            # priority = step_counter + steps_to_execution
            # For round-robin with N agents, every agent is re-used in N steps.
            # Higher value = further from next use = evict first.
            # max() propagation in radix_cache._insert_helper ensures that
            # when an agent re-executes, the old (lower) priority is replaced
            # by the new (higher) one.
            priority = None
            if use_priority:
                priority = step_counter + num_agents

            ttft_ms, e2e_ms, n_tokens, cached = await send_request(
                session=session,
                url=url,
                prompt=prompt,
                max_tokens=output_len,
                priority=priority,
            )

            round_results.append(
                StepResult(
                    agent_id=agent.agent_id,
                    round_idx=round_idx,
                    step_idx=step_idx,
                    ttft_ms=ttft_ms,
                    e2e_ms=e2e_ms,
                    output_tokens=n_tokens,
                    cached_tokens=cached,
                )
            )

            logger.info(
                f"  wf={workflow_id} round={round_idx} step={step_idx} "
                f"agent={agent.agent_id} ttft={ttft_ms:.1f}ms "
                f"e2e={e2e_ms:.1f}ms cached={cached}"
                + (f" priority={priority}" if priority is not None else "")
            )

            step_counter += 1

        result.round_results.append(round_results)

    return result


# ---------------------------------------------------------------------------
# Results printing and JSON output
# ---------------------------------------------------------------------------

def print_and_save_results(
    all_results: List[WorkflowResult],
    loads_before: dict,
    loads_after: dict,
    args: argparse.Namespace,
) -> None:
    """Print formatted results and save JSON output."""
    warmup = args.warmup_rounds
    total_rounds = warmup + args.num_rounds
    config_name = args.config

    logger.info("=" * 70)
    logger.info(f"Priority Benchmark Results [{config_name}]")
    logger.info("=" * 70)
    logger.info(
        f"Config: {args.num_agents} agents, prefix_len={args.prefix_len}, "
        f"suffix_len={args.suffix_len}, output_len={args.output_len}, "
        f"num_rounds={args.num_rounds}, warmup_rounds={warmup}, "
        f"num_concurrent={args.num_concurrent}"
    )
    logger.info("-" * 70)

    # Per-round breakdown (all rounds including warmup)
    for round_idx in range(total_rounds):
        round_ttfts = []
        round_cached = []
        for wf in all_results:
            if round_idx < len(wf.round_results):
                for s in wf.round_results[round_idx]:
                    round_ttfts.append(s.ttft_ms)
                    round_cached.append(s.cached_tokens)
        if round_ttfts:
            avg_ttft = sum(round_ttfts) / len(round_ttfts)
            avg_cached = sum(round_cached) / len(round_cached)
            tag = " [warmup]" if round_idx < warmup else ""
            logger.info(
                f"  Round {round_idx}: avg TTFT = {avg_ttft:.2f} ms, "
                f"avg cached = {avg_cached:.0f} tokens{tag}"
            )

    logger.info("-" * 70)

    # Aggregate stats (excluding warmup rounds)
    measured_ttfts: List[float] = []
    measured_e2es: List[float] = []
    measured_round_e2es: List[float] = []
    measured_cached: List[int] = []

    for wf in all_results:
        for round_idx, round_steps in enumerate(wf.round_results):
            if round_idx < warmup:
                continue
            for s in round_steps:
                measured_ttfts.append(s.ttft_ms)
                measured_e2es.append(s.e2e_ms)
                measured_cached.append(s.cached_tokens)
            measured_round_e2es.append(sum(s.e2e_ms for s in round_steps))

    avg_ttft = 0.0
    avg_round_e2e = 0.0
    avg_cached = 0.0

    if measured_ttfts:
        measured_ttfts.sort()
        avg_ttft = sum(measured_ttfts) / len(measured_ttfts)
        p50_ttft = measured_ttfts[len(measured_ttfts) // 2]
        p90_ttft = measured_ttfts[int(len(measured_ttfts) * 0.9)]
        logger.info(
            f"TTFT (measured, excl. {warmup} warmup): "
            f"avg={avg_ttft:.2f}ms, p50={p50_ttft:.2f}ms, p90={p90_ttft:.2f}ms"
        )

    if measured_round_e2es:
        measured_round_e2es.sort()
        avg_round_e2e = sum(measured_round_e2es) / len(measured_round_e2es)
        logger.info(f"Round E2E (measured): avg={avg_round_e2e:.2f}ms")

    if measured_cached:
        avg_cached = sum(measured_cached) / len(measured_cached)
        cache_hit_rate = (
            avg_cached / args.prefix_len * 100 if args.prefix_len > 0 else 0
        )
        logger.info(
            f"Cache: avg cached_tokens={avg_cached:.0f}, "
            f"approx hit rate={cache_hit_rate:.1f}% (of prefix_len={args.prefix_len})"
        )

    # Speedup vs baseline
    baseline_data = None
    if args.baseline_json:
        try:
            with open(args.baseline_json, "r") as f:
                baseline_data = json.load(f)
            bl_ttft = baseline_data["aggregate"]["ttft_avg_ms"]
            bl_e2e = baseline_data["aggregate"]["round_e2e_avg_ms"]
            logger.info("-" * 70)
            logger.info("Speedup vs baseline:")
            if avg_ttft > 0:
                ttft_speedup = bl_ttft / avg_ttft
                logger.info(
                    f"  TTFT: {ttft_speedup:.2f}x "
                    f"(baseline={bl_ttft:.1f}ms, current={avg_ttft:.1f}ms)"
                )
            if avg_round_e2e > 0:
                e2e_speedup = bl_e2e / avg_round_e2e
                logger.info(
                    f"  Round E2E: {e2e_speedup:.2f}x "
                    f"(baseline={bl_e2e:.1f}ms, current={avg_round_e2e:.1f}ms)"
                )
        except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
            logger.warning(f"Could not load baseline JSON: {e}")

    logger.info("=" * 70)

    # Build JSON output
    aggregate: Dict[str, Any] = {}
    if measured_ttfts:
        aggregate["ttft_avg_ms"] = avg_ttft
        aggregate["ttft_p50_ms"] = measured_ttfts[len(measured_ttfts) // 2]
        aggregate["ttft_p90_ms"] = measured_ttfts[int(len(measured_ttfts) * 0.9)]
    if measured_round_e2es:
        aggregate["round_e2e_avg_ms"] = avg_round_e2e
    if measured_cached:
        aggregate["cached_tokens_avg"] = avg_cached
        aggregate["cache_hit_rate_pct"] = (
            avg_cached / args.prefix_len * 100 if args.prefix_len > 0 else 0
        )

    output: Dict[str, Any] = {
        "config": {
            k: v for k, v in vars(args).items()
            if k != "baseline_json"  # don't leak local paths
        },
        "warmup_rounds": warmup,
        "aggregate": aggregate,
        "loads_before": loads_before,
        "loads_after": loads_after,
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
                            "cached_tokens": s.cached_tokens,
                        }
                        for s in round_steps
                    ]
                    for round_steps in wf.round_results
                ],
            }
            for wf in all_results
        ],
    }

    # Add speedup to JSON if baseline was provided
    if baseline_data and avg_ttft > 0 and avg_round_e2e > 0:
        bl_ttft = baseline_data["aggregate"]["ttft_avg_ms"]
        bl_e2e = baseline_data["aggregate"]["round_e2e_avg_ms"]
        output["speedup"] = {
            "baseline_file": os.path.basename(args.baseline_json),
            "ttft": bl_ttft / avg_ttft,
            "round_e2e": bl_e2e / avg_round_e2e,
        }

    # Write JSON output
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(
        args.output_dir,
        f"priority_bench_{config_name}_{args.num_agents}agents_"
        f"{args.prefix_len}pfx_{args.suffix_len}sfx_"
        f"{args.output_len}out_{args.num_concurrent}conc.json",
    )
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Results written to {output_file}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(
        description="Priority Multi-Agent Workflow Benchmark for SGLang"
    )
    parser.add_argument(
        "--config", type=str, required=True, choices=list(CONFIGS.keys()),
        help="Test configuration: baseline (LRU) or priority (Priority)",
    )
    parser.add_argument(
        "--num-agents", type=int, default=5,
        help="Number of agents in the workflow (default: 5)",
    )
    parser.add_argument(
        "--prefix-len", type=int, default=512,
        help="Length of each agent's fixed prefix in tokens (default: 512)",
    )
    parser.add_argument(
        "--suffix-len", type=int, default=64,
        help="Length of dynamic suffix per step in tokens (default: 64)",
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
        help="Number of warmup rounds excluded from aggregates (default: 1)",
    )
    parser.add_argument(
        "--num-concurrent", type=int, default=1,
        help="Number of concurrent workflows (default: 1)",
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1",
        help="Server host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port", type=int, default=30200,
        help="Server port (default: 30200)",
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3-0.6B",
        help="Model name for tokenizer calibration (default: Qwen/Qwen3-0.6B)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="benchmark_results",
        help="Directory for output JSON files (default: benchmark_results/)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--baseline-json", type=str, default=None,
        help="Path to baseline JSON for speedup comparison",
    )
    args = parser.parse_args()

    # Resolve config
    cfg = CONFIGS[args.config]
    use_priority = cfg["use_priority"]
    logger.info(f"Config: {args.config} -- {cfg['label']}")
    logger.info(f"  {cfg['server_note']}")

    random.seed(args.seed)
    url = f"http://{args.host}:{args.port}"

    # Load tokenizer for calibrated text generation
    from transformers import AutoTokenizer

    logger.info(f"Loading tokenizer for {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Create agents with tokenizer-calibrated prefixes
    agents: List[AgentConfig] = []
    for i in range(args.num_agents):
        prefix_text, actual_tokens = generate_calibrated_text(
            tokenizer, args.prefix_len
        )
        agents.append(
            AgentConfig(
                agent_id=f"agent-{i}",
                prefix_text=prefix_text,
                prefix_actual_tokens=actual_tokens,
                suffix_len=args.suffix_len,
            )
        )

    actual_counts = [a.prefix_actual_tokens for a in agents]
    logger.info(
        f"Created {args.num_agents} agents: target prefix_len={args.prefix_len}, "
        f"actual tokens min={min(actual_counts)} max={max(actual_counts)} "
        f"avg={sum(actual_counts) / len(actual_counts):.0f}"
    )
    logger.info(f"Priority metadata: {'enabled' if use_priority else 'disabled (baseline LRU)'}")
    logger.info(f"Target server: {url}")

    # Long timeout for large prefills on small GPUs
    timeout = aiohttp.ClientTimeout(total=600, sock_read=300)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        # Test connection
        try:
            async with session.get(f"{url}/health_generate") as resp:
                if resp.status != 200:
                    logger.error(
                        f"Server health check failed at {url}/health_generate "
                        f"(status {resp.status})"
                    )
                    return
        except Exception as e:
            logger.error(f"Cannot connect to server at {url}: {e}")
            return

        logger.info("Server connection verified via /health_generate")

        # Get initial load metrics
        loads_before = await fetch_loads(session, url)

        # Run concurrent workflows
        total_rounds = args.warmup_rounds + args.num_rounds
        logger.info(
            f"Running {args.num_concurrent} concurrent workflow(s), "
            f"{total_rounds} rounds each "
            f"({args.warmup_rounds} warmup + {args.num_rounds} measured)..."
        )

        tasks = [
            run_one_workflow(
                session=session,
                url=url,
                agents=agents,
                num_rounds=total_rounds,
                output_len=args.output_len,
                workflow_id=wf_id,
                use_priority=use_priority,
                tokenizer=tokenizer,
            )
            for wf_id in range(args.num_concurrent)
        ]
        all_results = await asyncio.gather(*tasks)

        # Brief pause for stats to settle
        await asyncio.sleep(1)
        loads_after = await fetch_loads(session, url)

        print_and_save_results(list(all_results), loads_before, loads_after, args)


if __name__ == "__main__":
    asyncio.run(main())
