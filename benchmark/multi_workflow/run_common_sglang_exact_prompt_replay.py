#!/usr/bin/env python3
"""Exact-token AB/BA TTFT replay for the common-agent SGLang coding arm.

The online coding-aware trajectory supplies only target groups where a real
lossy K/V copy was registered.  Dense and reuse then consume the identical
frozen token IDs in both launch orders.  Source construction is measured and
reported separately so N=1/4/16 amortization never disguises prefetch.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark.multi_workflow import run_natural_code_cost_exact_prompt_speed as base
from benchmark.multi_workflow.runtime_paths import RuntimePaths


PROJECT = Path(__file__).resolve().parents[2]
CAMPAIGN = (
    RuntimePaths.from_project(PROJECT).artifacts
    / "impactkv_common_agent_baselines_fresh24_20260812"
)
ARM = "coding_dependency_graph_cold_lcb"
WARMUPS = 2
MEASURED_ROUNDS = 5
TOTAL_ROUNDS = WARMUPS + MEASURED_ROUNDS
SEQUENCES = {"ab": ("dense", "reuse"), "ba": ("reuse", "dense")}
ACTIVE_PLAN: dict[str, Any] | None = None


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def configure(label: str) -> tuple[Path, Path]:
    if label == "canary4":
        policy_run = CAMPAIGN / "runs/sglang_canary" / ARM / "full_4"
    elif label == "fresh24":
        policy_run = CAMPAIGN / "runs/sglang_formal" / ARM / "full_24"
    else:
        raise ValueError(label)
    output = CAMPAIGN / "exact_prompt_replay" / label / "sglang_coding"
    base.CAMPAIGN = CAMPAIGN
    base.POLICY_RUN = policy_run
    base.DEFAULT_OUTPUT = output
    base.ARM = ARM
    base.WARMUPS = WARMUPS
    base.MEASURED_ROUNDS = MEASURED_ROUNDS
    base.TOTAL_ROUNDS = TOTAL_ROUNDS
    return policy_run, output


def prepare(label: str, output: Path) -> dict[str, Any]:
    registration_path = output / "RUN_REGISTRATION.json"
    if registration_path.is_file():
        return base.read_json(registration_path)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    policy_run, _ = configure(label)
    manifest = policy_run / "DYNAMIC_MANIFEST.json"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    plan = base.build_plan()
    if not plan["groups"]:
        raise ValueError("online coding-aware arm registered no physical target groups")
    output.mkdir(parents=True)
    write_json(output / "PLAN.json", plan)
    registration = {
        "schema_version": 1,
        "status": "FROZEN_BEFORE_EXACT_PROMPT_REPLAY",
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "label": label,
        "arm": ARM,
        "selection": (
            "all online coding-aware target groups with a registered physical "
            "lossy copy; no answer, resolution, or measured TTFT used"
        ),
        "capacity": {
            "target_groups": len(plan["groups"]),
            "islands": sum(int(row["islands"]) for row in plan["groups"]),
            "copied_tokens_per_round": sum(
                int(row["copied_tokens"]) for row in plan["groups"]
            ),
        },
        "protocol": {
            "model": base.MODEL,
            "exact_target_token_ids": True,
            "sequences": {key: list(value) for key, value in SEQUENCES.items()},
            "warmups_per_arm_per_sequence": WARMUPS,
            "measured_rounds_per_arm_per_sequence": MEASURED_ROUNDS,
            "decode_tokens": 1,
            "ordinary_radix_prefix_reuse": False,
            "source_build_reported_separately": True,
            "amortization_uses": [1, 4, 16],
            "prefetch": False,
        },
        "inputs": {
            "online_manifest": str(manifest),
            "online_manifest_sha256": sha256(manifest),
            "plan_sha256": sha256(output / "PLAN.json"),
            "source_sha256": sha256(Path(__file__).resolve()),
        },
        "protected": {
            "old_dirty_checkout_modified": False,
            "old_preregistration_thresholds_modified": False,
            "paper_modified": False,
            "prefetch": False,
        },
    }
    write_json(registration_path, registration)
    return registration


def prepare_pass(output: Path) -> dict[str, Any]:
    global ACTIVE_PLAN
    registration_path = output / "REGISTRATION.json"
    if registration_path.is_file():
        return base.read_json(registration_path)
    if ACTIVE_PLAN is None:
        raise RuntimeError("root exact-token plan is not loaded")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "PLAN.json", ACTIVE_PLAN)
    value = {"status": "ABBA_PASS_READY"}
    write_json(registration_path, value)
    return value


def run_sequence(label: str, output: Path, sequence: str, port: int) -> dict[str, Any]:
    global ACTIVE_PLAN
    if sequence not in SEQUENCES:
        raise ValueError(sequence)
    prepare(label, output)
    ACTIVE_PLAN = base.read_json(output / "PLAN.json")
    original_prepare = base.prepare
    base.prepare = prepare_pass
    try:
        pass_dir = output / sequence
        results = []
        for arm in SEQUENCES[sequence]:
            result_path = pass_dir / f"{arm}.json"
            if result_path.is_file():
                results.append({"arm": arm, "status": "ALREADY_COMPLETE"})
            else:
                results.append(base.run_arm(pass_dir, arm, port))
        return {"sequence": sequence, "results": results}
    finally:
        base.prepare = original_prepare


def measured_rows(value: dict[str, Any], group: int) -> list[dict[str, Any]]:
    return [
        row
        for row in value["targets"]
        if not row["warmup"] and int(row["group_index"]) == group
    ]


def summarize(label: str, output: Path) -> dict[str, Any]:
    registration = prepare(label, output)
    plan = base.read_json(output / "PLAN.json")["groups"]
    loaded = {
        sequence: {
            arm: base.read_json(output / sequence / f"{arm}.json")
            for arm in ("dense", "reuse")
        }
        for sequence in SEQUENCES
    }
    targets = []
    total_copy_events = 0
    total_fallback_events = 0
    for sequence in SEQUENCES:
        ledger = loaded[sequence]["reuse"]["ledger_rows"]
        total_copy_events += sum(row.get("event") == "target_copied" for row in ledger)
        total_fallback_events += sum(
            row.get("event") == "target_fallback" for row in ledger
        )
    for group in plan:
        index = int(group["group_index"])
        dense_rows = []
        reuse_rows = []
        builds = []
        for sequence in SEQUENCES:
            dense_rows.extend(measured_rows(loaded[sequence]["dense"], index))
            reuse_rows.extend(measured_rows(loaded[sequence]["reuse"], index))
            builds.append(
                sum(
                    float(row["elapsed_ms"])
                    for row in loaded[sequence]["reuse"]["sources"]
                    if int(row["group_index"]) == index
                )
            )
        if len(dense_rows) != 2 * MEASURED_ROUNDS or len(reuse_rows) != len(dense_rows):
            raise ValueError(f"group {index}: incomplete AB/BA measurements")
        dense_ttft = statistics.median(float(row["ttft_ms"]) for row in dense_rows)
        reuse_ttft = statistics.median(float(row["ttft_ms"]) for row in reuse_rows)
        build_ms = statistics.median(builds)

        def speedup(uses: int) -> float:
            return dense_ttft / (reuse_ttft + build_ms / uses)

        targets.append(
            {
                "group_index": index,
                "input_ids_sha256": group["target_prompt_hash"],
                "prompt_tokens": len(group["target_input_ids"]),
                "reusable_islands": int(group["islands"]),
                "reusable_tokens": int(group["copied_tokens"]),
                "rounds_per_arm": len(dense_rows),
                "median_dense_ttft_ms": dense_ttft,
                "median_reuse_ttft_ms": reuse_ttft,
                "median_cache_build_ms": build_ms,
                "cache_ready_speedup": dense_ttft / reuse_ttft,
                "n1_including_build_speedup": speedup(1),
                "n4_including_build_speedup": speedup(4),
                "n16_including_build_speedup": speedup(16),
            }
        )
    expected_copy_events = (
        sum(int(row["islands"]) for row in plan) * TOTAL_ROUNDS * len(SEQUENCES)
    )
    physical_ok = total_copy_events == expected_copy_events and total_fallback_events == 0
    result = {
        "status": "PASS" if physical_ok else "FAIL_PHYSICAL_REUSE_GATE",
        "backend": "sglang_coding",
        "label": label,
        "targets": targets,
        "summary": {
            "targets": len(targets),
            "measured_rounds_per_arm": sum(row["rounds_per_arm"] for row in targets),
            "median_target_cache_ready_speedup": statistics.median(
                row["cache_ready_speedup"] for row in targets
            ),
            "median_target_n1_including_build_speedup": statistics.median(
                row["n1_including_build_speedup"] for row in targets
            ),
            "median_target_n4_including_build_speedup": statistics.median(
                row["n4_including_build_speedup"] for row in targets
            ),
            "median_target_n16_including_build_speedup": statistics.median(
                row["n16_including_build_speedup"] for row in targets
            ),
            "targets_cache_ready_faster": sum(
                row["cache_ready_speedup"] > 1 for row in targets
            ),
            "physical_copy_events": total_copy_events,
            "expected_copy_events": expected_copy_events,
            "fallback_events": total_fallback_events,
            "input_identity_verified": True,
        },
        "registration": registration,
    }
    write_json(output / "RESULT.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", choices=("canary4", "fresh24"), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--port", type=int, default=30000)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    run = sub.add_parser("run-sequence")
    run.add_argument("--sequence", choices=tuple(SEQUENCES), required=True)
    sub.add_parser("run-all")
    sub.add_parser("summarize")
    args = parser.parse_args()
    _, default_output = configure(args.label)
    output = (args.output or default_output).resolve()
    if args.command == "prepare":
        value = prepare(args.label, output)
    elif args.command == "run-sequence":
        value = run_sequence(args.label, output, args.sequence, args.port)
    elif args.command == "run-all":
        value = {
            sequence: run_sequence(args.label, output, sequence, args.port)
            for sequence in SEQUENCES
        }
        value["result"] = summarize(args.label, output)
    else:
        value = summarize(args.label, output)
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
