#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

from benchmark.approx_kv.workloads import (
    TraceKind,
    build_interleaved_object_trace,
)
from sglang.srt.mem_cache.approx_kv.scheduling import (
    CacheCandidate,
    CacheObjectKind,
    EvictionPolicy,
    select_victims,
)


@dataclass
class ResidentObject:
    role: str
    last_access_step: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trace",
        choices=[kind.value for kind in TraceKind],
        default=TraceKind.LONG_DISTANCE.value,
    )
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--workflows", type=int, default=1)
    parser.add_argument("--share-role-objects", action="store_true")
    parser.add_argument("--capacity-objects", type=int, default=3)
    parser.add_argument("--object-bytes", type=int, default=1_000_000)
    parser.add_argument("--dense-ms", type=float, required=True)
    parser.add_argument("--recovery-ms", type=float, required=True)
    parser.add_argument("--hit-ms", type=float, default=5.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def simulate(
    *,
    policy: EvictionPolicy,
    roles: tuple[str, ...],
    capacity_objects: int,
    object_bytes: int,
    dense_ms: float,
    recovery_ms: float,
    hit_ms: float,
) -> dict:
    resident: dict[str, ResidentObject] = {}
    latencies = []
    hits = 0
    recoveries = 0
    evictions = 0

    def next_use_step(step: int, role: str) -> int | None:
        return next(
            (future for future in range(step + 1, len(roles)) if roles[future] == role),
            None,
        )

    for step, role in enumerate(roles):
        if role in resident:
            hits += 1
            latencies.append(hit_ms)
            resident[role].last_access_step = step
            continue

        recoveries += 1
        latencies.append(recovery_ms)
        if len(resident) >= capacity_objects:
            candidates = [
                (
                    item,
                    next_use_step(step, item.role),
                )
                for item in resident.values()
            ]
            cache_candidates = [
                CacheCandidate(
                    object_id=item.role,
                    resident_bytes=object_bytes,
                    last_access_step=item.last_access_step,
                    dense_cost_ms=dense_ms,
                    recovery_cost_ms=recovery_ms,
                    kind=CacheObjectKind.CANONICAL_BASE,
                    steps_to_execution=(
                        None if next_step is None else next_step - step
                    ),
                    oracle_next_use_step=next_step,
                    retired=next_step is None,
                )
                for item, next_step in candidates
            ]
            victim = select_victims(
                cache_candidates,
                bytes_to_free=object_bytes,
                policy=policy,
                current_step=step,
            )[0]
            del resident[victim.object_id]
            evictions += 1
        resident[role] = ResidentObject(role, step)

    return {
        "policy": policy.value,
        "requests": len(roles),
        "hits": hits,
        "recoveries": recoveries,
        "evictions": evictions,
        "hit_rate": hits / len(roles),
        "ttft_p50_ms": statistics.median(latencies),
        "ttft_mean_ms": statistics.mean(latencies),
    }


def main() -> int:
    args = parse_args()
    roles = build_interleaved_object_trace(
        kind=TraceKind(args.trace),
        rounds=args.rounds,
        workflows=args.workflows,
        share_roles=args.share_role_objects,
    )
    results = [
        simulate(
            policy=policy,
            roles=roles,
            capacity_objects=args.capacity_objects,
            object_bytes=args.object_bytes,
            dense_ms=args.dense_ms,
            recovery_ms=args.recovery_ms,
            hit_ms=args.hit_ms,
        )
        for policy in EvictionPolicy
    ]
    payload = {
        "config": vars(args) | {"output": str(args.output)},
        "results": results,
    }
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(results, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
