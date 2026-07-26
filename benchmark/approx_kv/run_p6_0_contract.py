#!/usr/bin/env python3
"""Freeze the Phase 6 fixed40 workload, schema, and provisional settings."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from benchmark.approx_kv.phase6.manifest import (
    REPRESENTATION_PROFILES,
    build_fixed40_manifest,
)
from benchmark.approx_kv.phase6.runner import source_provenance
from benchmark.approx_kv.phase6.schema import (
    Phase6RunSettings,
    artifact_schema_payload,
    payload_sha256,
    settings_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-git-sha", required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument(
        "--model-revision",
        default="c1899de289a04d12100db370d81485cdf75e47ca",
    )
    parser.add_argument("--chunked-prefill-size", type=int, default=1024)
    parser.add_argument(
        "--chunk-source",
        choices=("cl2", "provisional_worst_case"),
        default="provisional_worst_case",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--central-log", type=Path, required=True)
    return parser.parse_args()


def append_log(path: Path, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")


def build_contract(args: argparse.Namespace) -> dict:
    settings = Phase6RunSettings(
        source_git_sha=args.source_git_sha,
        source_tree_sha=args.source_tree_sha,
        image_digest=args.image_digest,
        model=args.model,
        model_revision=args.model_revision,
        chunked_prefill_size=args.chunked_prefill_size,
        chunk_source=args.chunk_source,
        warmup_repeats=1,
        formal_repeats=2,
        restarts=1,
    )
    workload = build_fixed40_manifest(
        chunked_prefill_size=args.chunked_prefill_size,
        chunk_source=args.chunk_source,
    )
    payload = {
        "schema_version": 1,
        "scope": "phase6-contract-freeze",
        "settings": settings_payload(settings),
        "workload": workload,
        "matched_state": {
            "rebuild_each_round": True,
            "one_measured_target_per_round": True,
            "approx_target_writes_exact": False,
            "exact_baseline_uses_same_round_source": True,
        },
        "cache_outcomes": [
            "exact_gpu_hit",
            "approximate_gpu_recovery",
            "host_demand_load",
            "dense_fallback",
        ],
        "representation_profiles": {
            name: {
                **profile,
                "representation_kinds": list(profile["representation_kinds"]),
            }
            for name, profile in REPRESENTATION_PROFILES.items()
        },
        "artifact_schema": artifact_schema_payload(),
        "performance_ranking_enabled": False,
    }
    payload["contract_sha256"] = payload_sha256(payload)
    return payload


def verify_contract(payload: dict) -> None:
    expected = payload["contract_sha256"]
    content = {
        key: value
        for key, value in payload.items()
        if key not in {"contract_sha256", "run_id"}
    }
    observed = payload_sha256(content)
    if observed != expected:
        raise ValueError(
            f"contract hash mismatch: expected {expected}, observed {observed}"
        )


def main() -> None:
    args = parse_args()
    provenance = source_provenance(args.source_git_sha)
    args.source_tree_sha = provenance["source_tree_sha"]
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    run_id = "phase6-contract-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    append_log(
        args.central_log,
        {
            "run_id": run_id,
            "status": "running",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "output": str(args.output.resolve()),
        },
    )
    try:
        payload = build_contract(args)
        payload["run_id"] = run_id
        verify_contract(payload)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        append_log(
            args.central_log,
            {
                "run_id": run_id,
                "status": "completed",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "output": str(args.output.resolve()),
                "contract_sha256": payload["contract_sha256"],
                "workload_sha256": payload["workload"]["manifest_sha256"],
            },
        )
    except Exception as exc:
        append_log(
            args.central_log,
            {
                "run_id": run_id,
                "status": "failed",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "output": str(args.output.resolve()),
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise


if __name__ == "__main__":
    main()
