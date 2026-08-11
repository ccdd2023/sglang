#!/usr/bin/env python3
"""AB/BA exact-prompt TTFT replay for all Fresh24 graph-policy targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

from benchmark.multi_workflow import run_natural_code_cost_exact_prompt_speed as base
from benchmark.multi_workflow.runtime_paths import RuntimePaths


PROJECT = Path(__file__).resolve().parents[2]
ARTIFACTS = RuntimePaths.from_project(PROJECT).artifacts
CAMPAIGN = ARTIFACTS / "impactkv_dependency_graph_fresh24_20260811"
POLICY_RUN = CAMPAIGN / "online/coding_dependency_graph_cold_lcb/full_24"
DEFAULT_OUTPUT = CAMPAIGN / "exact_prompt_speed_abba"
ARM = "coding_dependency_graph_cold_lcb"
WARMUPS = 1
MEASURED_ROUNDS = 5
TOTAL_ROUNDS = WARMUPS + MEASURED_ROUNDS
SEQUENCES = {"ab": ("dense", "reuse"), "ba": ("reuse", "dense")}
ACTIVE_PLAN: dict[str, Any] | None = None


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configure_base() -> None:
    base.CAMPAIGN = CAMPAIGN
    base.POLICY_RUN = POLICY_RUN
    base.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    base.ARM = ARM
    base.WARMUPS = WARMUPS
    base.MEASURED_ROUNDS = MEASURED_ROUNDS
    base.TOTAL_ROUNDS = TOTAL_ROUNDS


def prepare(output: Path) -> dict[str, Any]:
    registration_path = output / "REGISTRATION.json"
    if registration_path.exists():
        return base.read_json(registration_path)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    if not (POLICY_RUN / "DYNAMIC_MANIFEST.json").exists():
        raise FileNotFoundError(
            "Fresh24 graph-policy arm must finish before speed registration"
        )
    plan = base.build_plan()
    output.mkdir(parents=True)
    plan_path = output / "PLAN.json"
    base.write_json(plan_path, plan)
    value = {
        "status": "REGISTERED_BEFORE_EXACT_PROMPT_SPEED_GPU",
        "classification": "Fresh24 exact-token cache-ready AB/BA speed validation",
        "selection": (
            "all Fresh24 graph-policy target groups with a physical registered "
            "copy; no answer, task resolution, or measured TTFT used"
        ),
        "capacity": {
            "target_groups": len(plan["groups"]),
            "islands": sum(row["islands"] for row in plan["groups"]),
            "copied_tokens_per_round": sum(
                row["copied_tokens"] for row in plan["groups"]
            ),
        },
        "protocol": {
            "model": base.MODEL,
            "sequences": {key: list(value) for key, value in SEQUENCES.items()},
            "decode_tokens": 1,
            "warmups_per_sequence": WARMUPS,
            "measured_rounds_per_sequence": MEASURED_ROUNDS,
            "exact_target_prompt_tokens": True,
            "ordinary_radix_prefix_reuse": False,
            "source_build_reported_separately": True,
            "synthetic_source_replay_for_measurement": True,
            "agent_prefetch": False,
        },
        "metrics": {
            "primary": "paired cache-ready target TTFT over both AB/BA launches",
            "secondary": "source materialization cost reported separately",
            "accuracy": "not measured; use Fresh24 official agent result",
        },
        "inputs": {
            "online_manifest": str(POLICY_RUN / "DYNAMIC_MANIFEST.json"),
            "online_manifest_sha256": sha256(
                POLICY_RUN / "DYNAMIC_MANIFEST.json"
            ),
            "plan_sha256": sha256(plan_path),
            "source_sha256": sha256(Path(__file__).resolve()),
        },
        "protected": {
            "paper_modified": False,
            "prefetch": False,
            "old_preregistration_thresholds_modified": False,
        },
    }
    base.write_json(registration_path, value)
    return value


def prepare_pass(output: Path) -> dict[str, Any]:
    global ACTIVE_PLAN
    registration_path = output / "REGISTRATION.json"
    if registration_path.exists():
        return base.read_json(registration_path)
    if ACTIVE_PLAN is None:
        raise RuntimeError("root speed plan was not loaded")
    output.mkdir(parents=True, exist_ok=False)
    base.write_json(output / "PLAN.json", ACTIVE_PLAN)
    value = {
        "status": "ABBA_PASS_READY",
        "classification": "internal counterbalanced exact-prompt pass",
    }
    base.write_json(registration_path, value)
    return value


def run_sequence(output: Path, sequence: str, port: int) -> dict[str, Any]:
    global ACTIVE_PLAN
    if sequence not in SEQUENCES:
        raise ValueError(sequence)
    prepare(output)
    ACTIVE_PLAN = base.read_json(output / "PLAN.json")
    base.prepare = prepare_pass
    pass_dir = output / sequence
    results = []
    for arm in SEQUENCES[sequence]:
        result_path = pass_dir / f"{arm}.json"
        if result_path.exists():
            results.append(
                {
                    "arm": arm,
                    "status": "ALREADY_COMPLETE",
                    "path": str(result_path),
                }
            )
        else:
            results.append(base.run_arm(pass_dir, arm, port))
    return {"sequence": sequence, "results": results}


def measured_rows(value: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    return {
        (int(row["group_index"]), int(row["round_index"])): row
        for row in value["targets"]
        if not row["warmup"]
    }


def summarize(output: Path) -> dict[str, Any]:
    registration = prepare(output)
    plan = base.read_json(output / "PLAN.json")["groups"]
    savings = []
    per_sequence = {}
    copy_events = 0
    fallback_events = 0
    source_build_rows = []
    for sequence in SEQUENCES:
        dense = base.read_json(output / sequence / "dense.json")
        reuse = base.read_json(output / sequence / "reuse.json")
        dense_rows = measured_rows(dense)
        reuse_rows = measured_rows(reuse)
        if set(dense_rows) != set(reuse_rows):
            raise ValueError(f"{sequence}: paired targets differ")
        sequence_savings = [
            1
            - float(reuse_rows[key]["ttft_ms"])
            / float(dense_rows[key]["ttft_ms"])
            for key in dense_rows
        ]
        savings.extend(sequence_savings)
        ledger = reuse["ledger_rows"]
        sequence_copies = sum(
            row.get("event") == "target_copied" for row in ledger
        )
        sequence_fallbacks = sum(
            row.get("event") == "target_fallback" for row in ledger
        )
        copy_events += sequence_copies
        fallback_events += sequence_fallbacks
        source_build_rows.extend(
            float(row["elapsed_ms"]) for row in reuse["sources"]
        )
        per_sequence[sequence] = {
            "measured_pairs": len(sequence_savings),
            "paired_ttft_saving_median": statistics.median(sequence_savings),
            "paired_ttft_win_rate": sum(value > 0 for value in sequence_savings)
            / len(sequence_savings),
            "copy_events": sequence_copies,
            "fallback_events": sequence_fallbacks,
        }
    expected = sum(row["islands"] for row in plan) * TOTAL_ROUNDS * 2
    result = {
        "status": "COMPLETE",
        "classification": "counterbalanced exact-target-prompt speed validation",
        "coverage": {
            "target_groups": len(plan),
            "islands": sum(row["islands"] for row in plan),
            "measured_pairs": len(savings),
            "sequences": 2,
        },
        "latency": {
            "paired_ttft_saving_median": statistics.median(savings),
            "paired_ttft_win_rate": sum(value > 0 for value in savings)
            / len(savings),
            "mean_source_materialization_ms": statistics.fmean(source_build_rows),
            "source_build_excluded_from_cache_ready": True,
        },
        "per_sequence": per_sequence,
        "mechanism": {
            "copy_events": copy_events,
            "expected_copy_events": expected,
            "fallback_events": fallback_events,
        },
        "registration": registration,
    }
    base.write_json(output / "RESULT.json", result)
    return result


def main() -> None:
    configure_base()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    run = sub.add_parser("run-sequence")
    run.add_argument("--sequence", choices=tuple(SEQUENCES), required=True)
    run.add_argument("--port", type=int, default=30000)
    sub.add_parser("summarize")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.command == "prepare":
        value = prepare(output)
    elif args.command == "run-sequence":
        value = run_sequence(output, args.sequence, args.port)
    else:
        value = summarize(output)
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
