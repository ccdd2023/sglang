#!/usr/bin/env python3
"""
PriorityStrategy Ablation Experiment.

Verifies and tests the KVFlow eviction strategy under controlled conditions.
This script runs in three modes:

  1. formula_test    - Unit-test the PriorityStrategy formula correctness
  2. simulation      - Simulate cache eviction under 3 pressure levels
  3. formula_battle   - Compare old vs new formula side-by-side with sample nodes

Usage:
  python test_priority_ablation.py formula_test
  python test_priority_ablation.py simulation
  python test_priority_ablation.py formula_battle
  python test_priority_ablation.py all
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import os
from dataclasses import dataclass, field
from typing import Tuple

random.seed(42)

_ROLE_TYPE_SYSTEM = 1
_ROLE_TYPE_ROLE = 2
_ROLE_TYPE_TASK = 3


@dataclass
class SimNode:
    node_id: str
    priority: int
    role_type: int = 0
    critical_path_distance: int = 1
    convergence_factor: int = 0
    last_access_time: float = 0.0
    size_tokens: int = 100


class OldPriorityStrategy:
    """The previous (buggy) PriorityStrategy formula.

    Bug 1: role_type_boost × 100 causes massive numerical dominance.
           role_type_boost=200 → 20000 added, completely swamps base priority (11-15).
    Bug 2: +crit_distance makes LATE-stage nodes (small distance) have HIGHER
           effective priority, meaning they're evicted FIRST — backwards.
    """

    def get_priority(self, node: SimNode) -> Tuple[float, float]:
        role_type_boost = 0
        if node.role_type == _ROLE_TYPE_SYSTEM:
            role_type_boost = 200
        elif node.role_type == _ROLE_TYPE_ROLE:
            role_type_boost = 100
        elif node.role_type == _ROLE_TYPE_TASK:
            role_type_boost = 10

        crit_distance = max(1, node.critical_path_distance)
        crit_boost = crit_distance * 100

        effective_priority = (
            node.priority + role_type_boost * 100 + crit_boost
        )
        return (-effective_priority, -node.last_access_time)


class NewPriorityStrategy:
    """The corrected PriorityStrategy formula.

    Sort key (heapq min-heap, smaller = evicted earlier):
      (role_rank, -crit_dist, -priority, last_access_time)

    This means:
      - Tier: TASK/unknown evicted before ROLE, ROLE before SYSTEM
      - Within Tier: larger crit_dist evicted first (farther from execution)
      - Within same distance: larger priority evicted first (needed later)
      - Final tiebreaker: LRU
    """

    CRIT_WEIGHT = 20
    ROLE_TYPE_BOOST = {
        _ROLE_TYPE_SYSTEM: 10000,
        _ROLE_TYPE_ROLE: 5000,
        _ROLE_TYPE_TASK: 0,
    }

    def get_priority(self, node: SimNode) -> Tuple[int, int, int, float]:
        crit_distance = max(1, node.critical_path_distance)
        if node.role_type == _ROLE_TYPE_SYSTEM:
            role_rank = 2
        elif node.role_type == _ROLE_TYPE_ROLE:
            role_rank = 1
        else:
            role_rank = 0
        return (role_rank, -crit_distance, -node.priority, node.last_access_time)


class LRUStrategy:
    def get_priority(self, node: SimNode) -> Tuple[float, float]:
        return (node.last_access_time, 0.0)


class SimCache:
    def __init__(self, capacity_tokens: int, strategy):
        self.capacity = capacity_tokens
        self.strategy = strategy
        self.nodes: list[SimNode] = []
        self._time = 0.0

    @property
    def used_tokens(self) -> int:
        return sum(n.size_tokens for n in self.nodes)

    @property
    def free_tokens(self) -> int:
        return self.capacity - self.used_tokens

    def add(self, node: SimNode) -> list[SimNode]:
        evicted = []
        while node.size_tokens > self.free_tokens and self.nodes:
            victim = self._select_victim()
            if victim:
                evicted.append(victim)
                self.nodes.remove(victim)
        if node.size_tokens <= self.capacity:
            node.last_access_time = self._time
            self.nodes.append(node)
        self._time += 1.0
        return evicted

    def _select_victim(self) -> SimNode | None:
        if not self.nodes:
            return None
        return min(self.nodes, key=lambda n: self.strategy.get_priority(n))


def formula_test():
    print("\n" + "=" * 70)
    print("FORMULA UNIT TEST")
    print("=" * 70)

    old = OldPriorityStrategy()
    new = NewPriorityStrategy()

    cases = [
        {
            "desc": "System prompt (Tier-0), crit_dist=5 → evict LAST",
            "node": SimNode("sys_0", priority=14, role_type=_ROLE_TYPE_SYSTEM,
                             critical_path_distance=5, last_access_time=10.0),
            "expect_old_evict_first": False,
            "expect_new_evict_first": False,
        },
        {
            "desc": "Task context (Tier-2), crit_dist=1 → protect (evict LAST in Tier-2)",
            "node": SimNode("task_4", priority=11, role_type=_ROLE_TYPE_TASK,
                             critical_path_distance=1, last_access_time=10.0),
            "expect_old_evict_first": True,
            "expect_new_evict_first": False,
        },
        {
            "desc": "Role-based prefix (Tier-1), crit_dist=3 → evict 2nd",
            "node": SimNode("role_2", priority=13, role_type=_ROLE_TYPE_ROLE,
                             critical_path_distance=3, last_access_time=10.0),
            "expect_old_evict_first": False,
            "expect_new_evict_first": False,
        },
        {
            "desc": "Task context (Tier-2), crit_dist=5 → evict FIRST in Tier-2",
            "node": SimNode("task_0", priority=14, role_type=_ROLE_TYPE_TASK,
                             critical_path_distance=5, last_access_time=10.0),
            "expect_old_evict_first": True,
            "expect_new_evict_first": True,
        },
    ]

    all_pass = True
    nodes = [c["node"] for c in cases]
    old_victim = min(nodes, key=lambda x: old.get_priority(x))
    new_victim = min(nodes, key=lambda x: new.get_priority(x))
    for i, c in enumerate(cases, 1):
        n = c["node"]
        old_prio = old.get_priority(n)
        new_prio = new.get_priority(n)

        old_is_first = (n is old_victim)
        new_is_first = (n is new_victim)

        old_correct = (old_is_first == c["expect_old_evict_first"])
        new_correct = (new_is_first == c["expect_new_evict_first"])

        status_old = "PASS" if old_correct else "FAIL"
        status_new = "PASS" if new_correct else "FAIL"

        print(f"\n  [{i}] {c['desc']}")
        print(f"      node: priority={n.priority} role_type={n.role_type} crit_dist={n.critical_path_distance}")
        print(f"      OLD key={old_prio}  [{status_old}]")
        print(f"      NEW key={new_prio}  [{status_new}]")

        if not old_correct:
            print(f"      !! OLD formula gives wrong eviction order")
        if not new_correct:
            print(f"      !! NEW formula gives wrong eviction order")

        if not old_correct or not new_correct:
            all_pass = False

    print("\n" + "-" * 70)
    if all_pass:
        print("  ALL PASS - eviction order matches expectations")
    else:
        print("  SOME FAIL - Review cases above")

    print("\n  KEY INSIGHT: OLD formula has wrong CRIT_SIGN (+ instead of -):")
    print("    OLD: priority + crit_boost (+crit_dist×100) makes LARGE crit_dist = HIGHER priority")
    print("         → Tier separation and DAG ordering are unstable under pressure")
    print("    NEW: sort by (role_rank, -crit_dist, -priority, last_access_time)")

    print("\n  NEW formula numerical breakdown:")
    for c in cases:
        n = c["node"]
        p = new.get_priority(n)
        print(f"    {n.node_id}: key={p}  [evict_first={c['expect_new_evict_first']}]")


def simulation():
    print("\n" + "=" * 70)
    print("CACHE PRESSURE SIMULATION")
    print("=" * 70)

    NUM_ROUNDS = 10
    pressure_levels = {
        "low": {"cache_cap": 50000, "num_wf": 3, "agents": 4, "rounds": 5},
        "medium": {"cache_cap": 30000, "num_wf": 5, "agents": 5, "rounds": 8},
        "high": {"cache_cap": 15000, "num_wf": 8, "agents": 6, "rounds": 12},
    }

    strategies = [
        ("LRU", LRUStrategy()),
        ("Priority(old)", OldPriorityStrategy()),
        ("Priority(new)", NewPriorityStrategy()),
    ]

    for pressure_name, params in pressure_levels.items():
        cap = params["cache_cap"]
        n_wf = params["num_wf"]
        n_agents = params["agents"]
        n_rounds = params["rounds"]

        print(f"\n  [{pressure_name.upper()}] cache={cap:,} tokens | "
              f"{n_wf} workflows × {n_agents} agents × {n_rounds} rounds")

        results = {}
        for strat_name, strat in strategies:
            cache = SimCache(cap, strat)
            total_evicted = 0
            tier0_evicted = 0
            tier1_evicted = 0
            tier2_evicted = 0

            for wf_id in range(n_wf):
                for round_i in range(n_rounds):
                    for agent_i in range(n_agents):
                        crit_dist = n_agents - agent_i
                        role_type = (
                            _ROLE_TYPE_SYSTEM if agent_i == 0
                            else _ROLE_TYPE_ROLE if agent_i == 1
                            else _ROLE_TYPE_TASK
                        )
                        node = SimNode(
                            node_id=f"wf{wf_id}_r{round_i}_a{agent_i}",
                            priority=14 - agent_i,
                            role_type=role_type,
                            critical_path_distance=max(1, crit_dist),
                            size_tokens=random.choices(
                                [512, 1024, 2048],
                                weights=[0.3, 0.5, 0.2]
                            )[0],
                            last_access_time=float(round_i * 10 + agent_i),
                        )
                        evicted = cache.add(node)
                        total_evicted += len(evicted)
                        tier0_evicted += sum(1 for e in evicted if e.role_type == _ROLE_TYPE_SYSTEM)
                        tier1_evicted += sum(1 for e in evicted if e.role_type == _ROLE_TYPE_ROLE)
                        tier2_evicted += sum(1 for e in evicted if e.role_type == _ROLE_TYPE_TASK)

            results[strat_name] = {
                "total_evicted": total_evicted,
                "tier0": tier0_evicted,
                "tier1": tier1_evicted,
                "tier2": tier2_evicted,
                "final_size": cache.used_tokens,
            }

        header = f"  {'Strategy':<16} {'Evicted':>8} {'Tier-0':>8} {'Tier-1':>8} {'Tier-2':>8} {'Final':>10}"
        print(f"  {'-'*60}")
        print(header)
        for strat_name, r in results.items():
            print(f"  {strat_name:<16} {r['total_evicted']:>8} "
                  f"{r['tier0']:>8} {r['tier1']:>8} {r['tier2']:>8} "
                  f"{r['final_size']:>10,}")

        priority_new = results["Priority(new)"]
        lru = results["LRU"]
        priority_old = results["Priority(old)"]

        print(f"\n  Analysis:")
        if priority_new["tier0"] < lru["tier0"]:
            print(f"    ✓ NEW Priority protects {lru['tier0']-priority_new['tier0']} more Tier-0 nodes than LRU")
        else:
            print(f"    ✗ NEW Priority evicts {priority_new['tier0']-lru['tier0']} more Tier-0 nodes than LRU")

        if priority_new["tier0"] < priority_old["tier0"]:
            print(f"    ✓ NEW beats OLD: protects {priority_old['tier0']-priority_new['tier0']} more Tier-0 nodes")
        else:
            print(f"    ✗ NEW worse than OLD on Tier-0 protection")


def formula_battle():
    print("\n" + "=" * 70)
    print("OLD vs NEW FORMULA BATTLE")
    print("=" * 70)

    old = OldPriorityStrategy()
    new = NewPriorityStrategy()

    nodes = [
        SimNode("sys_0", priority=14, role_type=_ROLE_TYPE_SYSTEM, critical_path_distance=5),
        SimNode("role_1", priority=13, role_type=_ROLE_TYPE_ROLE, critical_path_distance=4),
        SimNode("task_2", priority=12, role_type=_ROLE_TYPE_TASK, critical_path_distance=3),
        SimNode("task_3", priority=11, role_type=_ROLE_TYPE_TASK, critical_path_distance=2),
        SimNode("task_4", priority=10, role_type=_ROLE_TYPE_TASK, critical_path_distance=1),
    ]

    old_order = sorted(nodes, key=lambda n: old.get_priority(n))
    new_order = sorted(nodes, key=lambda n: new.get_priority(n))

    print(f"\n  {'Node':<10} {'Priority':>8} {'Role':>6} {'CritDist':>9}")
    print(f"  {'-'*40}")
    for n in nodes:
        print(f"  {n.node_id:<10} {n.priority:>8} {n.role_type:>6} {n.critical_path_distance:>9}")

    print(f"\n  Eviction order (first = evicted first):")
    print(f"  {'Rank':<5} {'OLD formula':<25} {'NEW formula':<25} {'Match?'}")
    print(f"  {'-'*65}")
    for rank, (n_old, n_new) in enumerate(zip(old_order, new_order), 1):
        match = "✓" if n_old.node_id == n_new.node_id else "✗"
        print(f"  {rank:<5} {n_old.node_id:<25} {n_new.node_id:<25} {match}")

    print("\n  OLD formula rank vs expected (Tier-0/1 protected, Tier-2 evicted first):")
    expected = ["task_2", "task_3", "task_4", "role_1", "sys_0"]
    old_correct = sum(1 for o, e in zip([n.node_id for n in old_order], expected) if o == e)
    new_correct = sum(1 for o, e in zip([n.node_id for n in new_order], expected) if o == e)
    print(f"    OLD: {old_correct}/5 nodes in correct eviction order")
    print(f"    NEW: {new_correct}/5 nodes in correct eviction order")

    print("\n  NEW formula numerical breakdown:")
    for n in new_order:
        p = new.get_priority(n)
        print(f"    {n.node_id}: crit_dist={n.critical_path_distance}, -priority={-n.priority}, sort_key=({p[0]},{p[1]},{p[2]:.0f})")


def main():
    parser = argparse.ArgumentParser(description="PriorityStrategy Ablation Experiment")
    parser.add_argument(
        "mode",
        nargs="+",
        choices=["formula_test", "simulation", "formula_battle", "all"],
        default=["all"],
    )
    args = parser.parse_args()

    modes = set(args.mode)
    if "all" in modes:
        modes = {"formula_test", "simulation", "formula_battle"}

    for mode in ["formula_test", "simulation", "formula_battle"]:
        if mode in modes:
            if mode == "formula_test":
                formula_test()
            elif mode == "simulation":
                simulation()
            elif mode == "formula_battle":
                formula_battle()


if __name__ == "__main__":
    main()
