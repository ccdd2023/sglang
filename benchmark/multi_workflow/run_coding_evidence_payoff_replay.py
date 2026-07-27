#!/usr/bin/env python3
"""Replay natural V7 opportunities from the completed 18-task trajectory.

This diagnostic replays hash-verified consecutive source/target requests from
the completed General-8K agent run.  It changes only the copied tail length:
General copies at most 4096 tokens, while V7 copies the naturally eligible
5320--6144-token tail.  It measures causal request-level speed, not task
accuracy.  No request is added for prefetching.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any

from jinja2 import StrictUndefined, Template
from tokenizers import Tokenizer

from benchmark.multi_workflow.run_bridge_reuse_pilot import (
    CHAT_TEMPLATE,
    MODEL,
    TOKENIZER,
    capped_tail,
    generate,
    launch_server,
    manifest_case,
    render_ids,
    sha256_file,
    stop_server,
    token_ids_hash,
    write_json,
)
from benchmark.multi_workflow.run_coding_evidence_payoff_paired import (
    read_json,
    request_messages,
)


ARTIFACTS = Path("/home/gfy/CodeMAS_Project/kvflow-artifacts")
SOURCE_CAMPAIGN = ARTIFACTS / "impactkv_coding_memory_v5_20260726"
SOURCE_RUN = SOURCE_CAMPAIGN / "general_8k" / "full_18"
SOURCE_REGISTRATION = SOURCE_CAMPAIGN / "RUN_REGISTRATION.json"
OPPORTUNITY_AUDIT = (
    ARTIFACTS
    / "impactkv_coding_evidence_payoff_v7_20260726"
    / "OPPORTUNITY_AUDIT.json"
)
DEFAULT_OUTPUT = (
    ARTIFACTS / "impactkv_coding_evidence_replay_v7_20260726"
)
ARMS = ("general_4k", "coding_evidence_payoff_v7")
REPETITIONS = 5


def percentile(values: list[float], fraction: float) -> float:
    return sorted(values)[math.ceil(fraction * len(values)) - 1]


def _largest_hash_independent_set(
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Avoid cross-role aliases that can consume another case's source lease."""

    def independent(rows: tuple[dict[str, Any], ...]) -> bool:
        sources = [row["source_prompt_hash"] for row in rows]
        targets = [row["target_prompt_hash"] for row in rows]
        return (
            len(sources) == len(set(sources))
            and len(targets) == len(set(targets))
            and not (set(sources) & set(targets))
        )

    selected: tuple[dict[str, Any], ...] = ()
    for size in range(len(candidates), 0, -1):
        valid = [
            rows
            for rows in itertools.combinations(candidates, size)
            if independent(rows)
        ]
        if valid:
            selected = max(
                valid,
                key=lambda rows: sum(row["v7_tokens"] for row in rows),
            )
            break
    selected_ids = {row["case_id"] for row in selected}
    excluded = [
        {
            "case_id": row["case_id"],
            "reason": "cross_role_prompt_hash_alias",
        }
        for row in candidates
        if row["case_id"] not in selected_ids
    ]
    return list(selected), excluded


def prepare(output: Path) -> dict[str, Any]:
    registration = read_json(SOURCE_REGISTRATION)
    instance_order = registration["dataset"]["instances"]
    instance_to_index = {
        instance_id: index + 1
        for index, instance_id in enumerate(instance_order)
    }
    audit = read_json(OPPORTUNITY_AUDIT)
    dynamic = read_json(SOURCE_RUN / "DYNAMIC_MANIFEST.json")
    dynamic_by_request: dict[tuple[int, int], dict[str, Any]] = {}
    for row in dynamic["cases"]:
        match = re.search(r"-m(\d+)-s\d+-q(\d+)-", row["case_id"])
        if match is None:
            continue
        dynamic_by_request[(int(match.group(1)), int(match.group(2)))] = row

    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    template = Template(
        CHAT_TEMPLATE.read_text(encoding="utf-8"),
        undefined=StrictUndefined,
    )
    candidates: list[dict[str, Any]] = []
    for opportunity in audit["rows"]:
        instance_id = opportunity["instance_id"]
        request_index = int(opportunity["target_request_index"])
        model_index = instance_to_index[instance_id]
        original = dynamic_by_request[(model_index, request_index)]
        messages = read_json(
            SOURCE_RUN / instance_id / f"{instance_id}.traj.json"
        )["messages"]
        source_ids = render_ids(
            messages=request_messages(messages, request_index - 1),
            template=template,
            tokenizer=tokenizer,
        )
        target_ids = render_ids(
            messages=request_messages(messages, request_index),
            template=template,
            tokenizer=tokenizer,
        )
        source_hash = token_ids_hash(source_ids)
        target_hash = token_ids_hash(target_ids)
        if source_hash != original["source_prompt_hash"]:
            raise ValueError(f"{instance_id} q{request_index}: source mismatch")
        if target_hash != original["target_prompt_hash"]:
            raise ValueError(f"{instance_id} q{request_index}: target mismatch")
        if int(original["length"]) != int(opportunity["candidate_tokens"]):
            raise ValueError(
                f"{instance_id} q{request_index}: candidate length mismatch"
            )
        v7_tokens = min(int(original["length"]), 6144)
        candidates.append(
            {
                "case_id": f"{instance_id}-q{request_index}",
                "instance_id": instance_id,
                "request_index": request_index,
                "source_input_ids": source_ids,
                "source_prompt_tokens": len(source_ids),
                "source_prompt_hash": source_hash,
                "target_input_ids": target_ids,
                "target_prompt_tokens": len(target_ids),
                "target_prompt_hash": target_hash,
                "source_start": int(original["source_start"]),
                "target_start": int(original["target_start"]),
                "general_tokens": 4096,
                "v7_tokens": v7_tokens,
            }
        )

    cases, excluded = _largest_hash_independent_set(candidates)
    cases_path = output / "REPLAY_CASES.json"
    write_json(
        cases_path,
        {
            "opportunities_in_audit": len(candidates),
            "hash_independent_cases": len(cases),
            "excluded": excluded,
            "cases": cases,
        },
    )
    manifest_hashes = {}
    for arm in ARMS:
        rows = []
        for case in cases:
            length = (
                case["general_tokens"]
                if arm == "general_4k"
                else case["v7_tokens"]
            )
            wide_span = {
                "source_start": case["source_start"],
                "target_start": case["target_start"],
                "length": case["v7_tokens"],
            }
            span = (
                capped_tail(wide_span, length)
                if length < wide_span["length"]
                else wide_span
            )
            row = manifest_case(
                case_id=case["case_id"],
                policy_label=arm,
                source_ids=case["source_input_ids"],
                target_ids=case["target_input_ids"],
                span=span,
            )
            row["target_uses"] = REPETITIONS
            rows.append(row)
        path = output / "manifests" / f"{arm}.json"
        write_json(
            path,
            {
                "cache_dtype": "bfloat16",
                "cases": rows,
                "lease_ttl_s": 900,
                "ledger_path": str(
                    output / "server" / arm / "EXACT_LEDGER.jsonl"
                ),
                "model_id": MODEL,
                "rope": {
                    "base": 10_000_000,
                    "is_neox_style": True,
                    "rotary_dim": 128,
                },
                "version": 2,
            },
        )
        manifest_hashes[arm] = sha256_file(path)

    replay_registration = {
        "registered_before_gpu": True,
        "classification": (
            "fixed-request full-trajectory opportunity replay; causal speed "
            "diagnostic, not task accuracy or end-to-end SOTA evidence"
        ),
        "source_run": str(SOURCE_RUN),
        "model": MODEL,
        "opportunities_in_audit": len(candidates),
        "hash_independent_cases": len(cases),
        "excluded_cross_role_prompt_aliases": excluded,
        "target_repetitions": REPETITIONS,
        "decode_tokens": 1,
        "prefetch": False,
        "arms": {
            "general_4k": "same contiguous overlap capped at 4096 tokens",
            "coding_evidence_payoff_v7": (
                "same overlap widened to the natural V7 5.3K--6.1K length"
            ),
        },
        "inputs": {
            "cases_sha256": sha256_file(cases_path),
            "source_dynamic_manifest_sha256": sha256_file(
                SOURCE_RUN / "DYNAMIC_MANIFEST.json"
            ),
            "opportunity_audit_sha256": sha256_file(OPPORTUNITY_AUDIT),
            "manifest_sha256": manifest_hashes,
            "runner_sha256": sha256_file(Path(__file__)),
        },
        "gate": {
            "median_ttft_reduction_percent_min": 5.0,
            "p95_ttft_reduction_percent_min": 5.0,
            "case_median_win_fraction_min": 0.70,
            "expected_copy_events_per_arm": len(cases) * REPETITIONS,
            "fallback_events_max": 0,
        },
        "protected": {
            "old_dirty_checkout_modified": False,
            "paper_modified": False,
            "existing_preregistration_thresholds_modified": False,
        },
        "status": "REGISTERED_BEFORE_REPLAY_GPU_RUN",
    }
    write_json(output / "REPLAY_REGISTRATION.json", replay_registration)
    return replay_registration


def run_arm(output: Path, arm: str, port: int) -> dict[str, Any]:
    cases = read_json(output / "REPLAY_CASES.json")["cases"]
    process, stream, base_url = launch_server(
        output=output,
        arm=arm,
        port=port,
    )
    sources = []
    rows = []
    try:
        for case in cases:
            sources.append(
                {
                    **generate(
                        base_url=base_url,
                        input_ids=case["source_input_ids"],
                        key=f"replay-source-{arm}-{case['case_id']}",
                        max_new_tokens=1,
                        stream=False,
                    ),
                    "case_id": case["case_id"],
                }
            )
            for repetition in range(REPETITIONS):
                rows.append(
                    {
                        **generate(
                            base_url=base_url,
                            input_ids=case["target_input_ids"],
                            key=(
                                f"replay-target-{arm}-{case['case_id']}-"
                                f"r{repetition}"
                            ),
                            max_new_tokens=1,
                            stream=True,
                        ),
                        "arm": arm,
                        "case_id": case["case_id"],
                        "repetition": repetition,
                    }
                )
    finally:
        stop_server(process, stream)
    ledger_path = output / "server" / arm / "EXACT_LEDGER.jsonl"
    ledger = [
        json.loads(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    result = {
        "arm": arm,
        "rows": rows,
        "source_rows": sources,
        "ledger_rows": ledger,
        "status": "complete",
    }
    write_json(output / "generations" / f"{arm}.json", result)
    return {
        "arm": arm,
        "rows": len(rows),
        "copy_events": sum(
            row.get("event") == "target_copied" for row in ledger
        ),
        "fallback_events": sum(
            row.get("event") == "target_fallback" for row in ledger
        ),
        "status": "complete",
    }


def summarize(output: Path) -> dict[str, Any]:
    cases = read_json(output / "REPLAY_CASES.json")["cases"]
    expected = len(cases) * REPETITIONS
    arms: dict[str, Any] = {}
    for arm in ARMS:
        value = read_json(output / "generations" / f"{arm}.json")
        ttfts = [float(row["ttft_ms"]) for row in value["rows"]]
        copies = [
            row
            for row in value["ledger_rows"]
            if row.get("event") == "target_copied"
        ]
        fallbacks = [
            row
            for row in value["ledger_rows"]
            if row.get("event") == "target_fallback"
        ]
        arms[arm] = {
            "rows": len(ttfts),
            "median_ttft_ms": statistics.median(ttfts),
            "p95_ttft_ms": percentile(ttfts, 0.95),
            "sum_ttft_ms": sum(ttfts),
            "copy_events": len(copies),
            "fallback_events": len(fallbacks),
            "copied_tokens": sum(
                int(row["copied_k_tokens"]) for row in copies
            ),
            "case_median_ttft_ms": {
                case["case_id"]: statistics.median(
                    float(row["ttft_ms"])
                    for row in value["rows"]
                    if row["case_id"] == case["case_id"]
                )
                for case in cases
            },
        }
    general = arms["general_4k"]
    v7 = arms["coding_evidence_payoff_v7"]
    wins = sum(
        v7["case_median_ttft_ms"][case_id]
        < general["case_median_ttft_ms"][case_id]
        for case_id in general["case_median_ttft_ms"]
    )
    comparison = {
        "median_ttft_reduction_percent": 100
        * (1 - v7["median_ttft_ms"] / general["median_ttft_ms"]),
        "p95_ttft_reduction_percent": 100
        * (1 - v7["p95_ttft_ms"] / general["p95_ttft_ms"]),
        "sum_ttft_reduction_percent": 100
        * (1 - v7["sum_ttft_ms"] / general["sum_ttft_ms"]),
        "case_median_wins": wins,
        "case_median_win_fraction": wins / len(cases),
    }
    gate = {
        "median_passed": comparison["median_ttft_reduction_percent"] >= 5,
        "p95_passed": comparison["p95_ttft_reduction_percent"] >= 5,
        "case_wins_passed": comparison["case_median_win_fraction"] >= 0.70,
        "copy_events_passed": all(
            arms[arm]["copy_events"] == expected for arm in ARMS
        ),
        "fallback_passed": all(
            arms[arm]["fallback_events"] == 0 for arm in ARMS
        ),
    }
    result = {
        "classification": (
            "fixed-request full-trajectory opportunity replay; causal speed "
            "diagnostic, not task accuracy or end-to-end SOTA evidence"
        ),
        "cases": len(cases),
        "arms": arms,
        "comparison": comparison,
        "gate": {**gate, "overall_passed": all(gate.values())},
        "prefetch": False,
    }
    write_json(output / "REPLAY_RESULT.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    run = subparsers.add_parser("run-arm")
    run.add_argument("--arm", choices=ARMS, required=True)
    run.add_argument("--port", type=int, default=33300)
    subparsers.add_parser("summarize")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.command == "prepare":
        result = prepare(output)
    elif args.command == "run-arm":
        result = run_arm(output, args.arm, args.port)
    else:
        result = summarize(output)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
