#!/usr/bin/env python3
"""Replay frozen common-agent token IDs through native Dense and reuse modes.

The source ledger is produced by a completed agent run.  Its exact input IDs
are frozen before timing.  One loaded native engine then executes an ABBA
schedule, avoiding both prompt drift and cross-process model-load differences.
Cache construction is reported separately from cache-ready TTFT.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_requests(path: Path, limit: int | None) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        request = row.get("request", row)
        prompt_hash = str(request.get("input_ids_sha256") or "")
        if not prompt_hash or prompt_hash in seen:
            continue
        if not request.get("segments"):
            continue
        ids = request.get("input_ids") or []
        if not ids:
            continue
        selected.append(request)
        seen.add(prompt_hash)
        if limit is not None and len(selected) >= limit:
            break
    if not selected:
        raise ValueError("source ledger contains no reusable exact-token prompts")
    return selected


def issue(url: str, frozen: dict[str, Any], mode: str) -> dict[str, Any]:
    payload = {
        **frozen,
        "benchmark_mode": mode,
        "max_new_tokens": 1,
        "temperature": 0.0,
    }
    response = requests.post(url.rstrip("/") + "/generate", json=payload, timeout=900)
    response.raise_for_status()
    value = response.json()
    if value.get("input_ids_sha256") != frozen["input_ids_sha256"]:
        raise RuntimeError("native backend changed frozen target token IDs")
    if value.get("mode") != mode:
        raise RuntimeError(f"native backend did not execute requested mode: {value}")
    return value


def median(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        raise ValueError(f"no {key} measurements")
    return statistics.median(values)


def replay_target(
    url: str,
    frozen: dict[str, Any],
    cycles: int,
) -> dict[str, Any]:
    warmup = []
    for mode in ("dense", "reuse", "reuse"):
        warmup.append(issue(url, frozen, mode))

    measured: list[dict[str, Any]] = []
    for cycle in range(cycles):
        for order, mode in enumerate(("dense", "reuse", "reuse", "dense")):
            value = issue(url, frozen, mode)
            measured.append(
                {
                    "cycle": cycle,
                    "order": order,
                    "mode": mode,
                    "ttft_ms": value.get("ttft_ms"),
                    "request_elapsed_ms": value.get("request_elapsed_ms"),
                    "cache_build_ms": value.get("cache_build_ms"),
                    "physical_reuse": bool(value.get("physical_reuse")),
                    "reused_k_tokens": int(value.get("reused_k_tokens") or 0),
                    "reused_v_tokens": int(value.get("reused_v_tokens") or 0),
                    "recomputed_tokens": int(value.get("recomputed_tokens") or 0),
                    "fallback_reason": value.get("fallback_reason"),
                }
            )
    dense = [row for row in measured if row["mode"] == "dense"]
    reuse = [row for row in measured if row["mode"] == "reuse"]
    dense_ttft = median(dense, "ttft_ms")
    reuse_ttft = median(reuse, "ttft_ms")
    build_ms = median(reuse, "cache_build_ms")

    def amortized_speedup(uses: int) -> float:
        return dense_ttft / (reuse_ttft + build_ms / uses)

    return {
        "input_ids_sha256": frozen["input_ids_sha256"],
        "prompt_tokens": len(frozen["input_ids"]),
        "reusable_segments": len(frozen.get("segments") or []),
        "reusable_tokens": sum(
            int(span["end"]) - int(span["start"])
            for span in frozen.get("segments") or []
        ),
        "rounds_per_arm": len(dense),
        "warmup": [
            {
                "mode": row.get("mode"),
                "ttft_ms": row.get("ttft_ms"),
                "physical_reuse": row.get("physical_reuse"),
                "fallback_reason": row.get("fallback_reason"),
            }
            for row in warmup
        ],
        "measured": measured,
        "median_dense_ttft_ms": dense_ttft,
        "median_reuse_ttft_ms": reuse_ttft,
        "median_cache_build_ms": build_ms,
        "cache_ready_speedup": dense_ttft / reuse_ttft,
        "n1_including_build_speedup": amortized_speedup(1),
        "n4_including_build_speedup": amortized_speedup(4),
        "n16_including_build_speedup": amortized_speedup(16),
        "physical_reuse_rounds": sum(row["physical_reuse"] for row in reuse),
        "median_reused_k_tokens": median(reuse, "reused_k_tokens"),
        "median_reused_v_tokens": median(reuse, "reused_v_tokens"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cacheblend", "kvcomm"), required=True)
    parser.add_argument("--backend-url", required=True)
    parser.add_argument("--source-ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    frozen_requests = load_requests(args.source_ledger, args.limit)
    args.output.mkdir(parents=True)
    registration = {
        "schema_version": 1,
        "status": "FROZEN_BEFORE_EXACT_PROMPT_REPLAY",
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend": args.backend,
        "source_ledger": str(args.source_ledger.resolve()),
        "source_ledger_sha256": sha256(args.source_ledger),
        "target_prompt_hashes": [
            request["input_ids_sha256"] for request in frozen_requests
        ],
        "schedule": "warmup Dense, Reuse, Reuse; then five ABBA cycles",
        "rounds_per_arm": args.cycles * 2,
        "max_new_tokens": 1,
        "reporting": {
            "cache_ready": "Dense TTFT / Reuse TTFT",
            "n_including_build": "Dense TTFT / (Reuse TTFT + build/N)",
        },
    }
    write_json(args.output / "RUN_REGISTRATION.json", registration)
    targets = [
        replay_target(args.backend_url, request, args.cycles)
        for request in frozen_requests
    ]
    cache_ready = [float(row["cache_ready_speedup"]) for row in targets]
    physical = sum(int(row["physical_reuse_rounds"]) for row in targets)
    result = {
        "status": "PASS" if physical > 0 else "FAIL_NO_PHYSICAL_REUSE",
        "backend": args.backend,
        "targets": targets,
        "summary": {
            "targets": len(targets),
            "measured_rounds_per_arm": args.cycles * 2 * len(targets),
            "median_target_cache_ready_speedup": statistics.median(cache_ready),
            "targets_cache_ready_faster": sum(value > 1 for value in cache_ready),
            "physical_reuse_rounds": physical,
            "input_identity_verified": True,
        },
    }
    write_json(args.output / "RESULT.json", result)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
