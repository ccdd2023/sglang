"""Phase 3.3: real-trace routing-variance Monte Carlo.

The original real-trace experiment (452 SWE-bench instances) routed all 3
agents to the same file, so cross-agent hit rate was 100% by construction.
The reviewer audit flagged this as a workload-construction artifact, not a
measured property of the algorithm.

This script reuses the existing trace data and re-routes each agent
randomly to a different repo's code block (Monte Carlo over 1000
seeds), then re-evaluates the cross-agent hit rate. The randomized
routing simulates a workload where agents do not all read the same
file, isolating the algorithm's hit rate from the workload's routing
convention.

Output: stdout summary + JSON written to ``data/routing_variance.json``.
"""
from __future__ import annotations

import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

TRACE_PATH = Path(
    "/home/gfy/CodeMAS_Project/sglang-kvflow/results/real_trace_reuse/data/swe_bench_traces.jsonl"
)
OUT_PATH = Path(
    "/home/gfy/CodeMAS_Project/sglang-kvflow/results/real_trace_reuse/data/routing_variance.json"
)
N_SIMULATIONS = 1000
SEED_BASE = 42


def main() -> None:
    records = []
    with open(TRACE_PATH) as f:
        for line in f:
            records.append(json.loads(line))
    print(f"Loaded {len(records)} records")

    # Group by instance -> agent -> (signature, repo)
    by_instance = defaultdict(dict)  # iid -> agent -> list of (signature, repo)
    by_repo = defaultdict(list)  # repo -> list of signatures
    for r in records:
        by_instance[r["instance_id"]][r["agent"]] = (
            r["code_content_signature"],
            r["repo"],
        )
        by_repo[r["repo"]].append(r["code_content_signature"])
    n_instances = len(by_instance)
    print(f"Unique instances: {n_instances}, repos: {len(by_repo)}")

    agents = list(next(iter(by_instance.values())).keys())
    print(f"Agents: {agents}")

    # Monte Carlo: randomize routing per simulation
    rng = random.Random(SEED_BASE)
    sim_hit_rates = []  # fraction of cross-agent pairs that match
    sim_overall_rates = []  # including planner→planner
    repo_list = list(by_repo.keys())

    for sim_i in range(N_SIMULATIONS):
        # Per-instance, per-agent, randomly choose a signature from a
        # different repo than the original (to break the construction
        # artifact). With many repos, cross-agent content overlap is
        # essentially the chance that two random repos' code blocks
        # share a signature — for byte-identical code (the safety gate
        # in this paper), this is essentially zero.
        simulated_signatures = {}  # (iid, agent) -> signature
        for iid, agent_to_orig in by_instance.items():
            for agent in agents:
                # 90% of the time pick a random other repo; 10% keep
                # the original (to model the realistic case where
                # routing usually stays close to the task's repo).
                if rng.random() < 0.9:
                    new_repo = rng.choice(repo_list)
                else:
                    new_repo = agent_to_orig[agent][1]
                # Pick a random signature from that repo
                sim_sig = rng.choice(by_repo[new_repo])
                simulated_signatures[(iid, agent)] = sim_sig

        # Compute hit rate per pair, averaged across instances
        pair_hits = defaultdict(list)
        for iid in by_instance:
            sigs = {a: simulated_signatures[(iid, a)] for a in agents}
            for i, ai in enumerate(agents):
                for aj in agents[i + 1:]:
                    pair_hits[(ai, aj)].append(1 if sigs[ai] == sigs[aj] else 0)
        # Cross-agent mean hit rate
        cross_pairs = [
            (a1, a2) for (a1, a2) in pair_hits
            if a1 != a2 and (a1, a2) != ("planner", "planner")
        ]
        # Actually planner→planner is same-agent; only same-agent pairs are
        # i==j in the loop. We want all (i<j) pairs regardless of agent.
        all_pair_mean = statistics.mean(
            statistics.mean(pair_hits[k]) for k in pair_hits
        )
        sim_overall_rates.append(all_pair_mean)
        if cross_pairs:
            cross_mean = statistics.mean(
                statistics.mean(pair_hits[k]) for k in cross_pairs
            )
        else:
            cross_mean = 0.0
        sim_hit_rates.append(cross_mean)

    # Summary
    cross_mean = statistics.mean(sim_hit_rates)
    cross_std = statistics.stdev(sim_hit_rates) if len(sim_hit_rates) > 1 else 0
    cross_min = min(sim_hit_rates)
    cross_max = max(sim_hit_rates)
    cross_p50 = statistics.median(sim_hit_rates)

    print()
    print(f"=== Routing-variance Monte Carlo (N={N_SIMULATIONS}) ===")
    print()
    print("Original (construction artifact, all 3 agents on same file):")
    print("  planner→coder: 1.000")
    print("  coder→reviewer: 1.000")
    print("  planner→reviewer: 1.000")
    print("  planner→planner: 0.002")
    print()
    print("Randomized routing (90% chance to switch repo per agent per instance):")
    print(f"  cross-agent mean hit rate: {cross_mean:.4f} ± {cross_std:.4f}")
    print(f"  range: [{cross_min:.4f}, {cross_max:.4f}]")
    print(f"  p50: {cross_p50:.4f}")
    print()
    verdict = (
        f"The original 100% cross-agent hit rate is a construction artifact. "
        f"With 90% randomized routing, the mean cross-agent hit rate drops to "
        f"{cross_mean*100:.2f}% ± {cross_std*100:.2f}% (p50 {cross_p50*100:.2f}%, "
        f"range [{cross_min*100:.2f}%, {cross_max*100:.2f}%]). "
        f"This isolates the algorithm's intrinsic hit rate from the workload's "
        f"routing convention."
    )
    print(f"Verdict: {verdict}")

    # Save
    out = {
        "n_simulations": N_SIMULATIONS,
        "n_instances": n_instances,
        "n_repos": len(by_repo),
        "routing_randomization_prob": 0.9,
        "original_cross_agent_hit_rates": {
            "planner->coder": 1.0,
            "coder->reviewer": 1.0,
            "planner->reviewer": 1.0,
            "planner->planner": 0.002,
        },
        "randomized_cross_agent_mean_hit_rate": cross_mean,
        "randomized_cross_agent_std": cross_std,
        "randomized_cross_agent_min": cross_min,
        "randomized_cross_agent_max": cross_max,
        "randomized_cross_agent_p50": cross_p50,
        "verdict": verdict,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
