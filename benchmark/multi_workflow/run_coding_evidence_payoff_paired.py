#!/usr/bin/env python3
"""Paired fixed-request speed diagnostic for coding-evidence-payoff V7.

The cases are exact consecutive source/target prompts from the completed V7
agent run.  The source is the natural preceding request, not an extra prefetch
request.  Repeated targets are measurement repetitions only.  This diagnostic
isolates the 4K-versus-6K physical copy effect; it is not task-level accuracy
evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any

from jinja2 import StrictUndefined, Template
from tokenizers import Tokenizer

from benchmark.multi_workflow.run_bridge_reuse_pilot import (
    BASH_TOOL,
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


ARTIFACTS = Path("/home/gfy/CodeMAS_Project/kvflow-artifacts")
SOURCE_CAMPAIGN = (
    ARTIFACTS / "impactkv_coding_evidence_payoff_v7_20260726"
)
SOURCE_RUN_NAME = (
    "canary_astropy__astropy-14995|psf__requests-1142|"
    "sphinx-doc__sphinx-9230"
)
SOURCE_RUN = (
    SOURCE_CAMPAIGN / "coding_evidence_payoff_v7" / SOURCE_RUN_NAME
)
DEFAULT_OUTPUT = (
    ARTIFACTS / "impactkv_coding_evidence_paired_v7_20260726"
)
INSTANCE_ORDER = (
    "astropy__astropy-14995",
    "psf__requests-1142",
    "sphinx-doc__sphinx-9230",
)
ARMS = ("general_4k", "coding_evidence_payoff_v7")
REPETITIONS = 5
ROLLING_NOTICE = (
    '<history_compaction dropped_turn_groups="{dropped}">'
    "Earlier interaction details were omitted to stay within the rolling "
    "history budget. Repository state persists; the most recent complete "
    "interactions follow."
    "</history_compaction>"
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def turn_groups(
    messages: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for message in messages[2:]:
        if message.get("role") == "assistant" and current:
            groups.append(current)
            current = []
        current.append(message)
    if current:
        groups.append(current)
    return groups


def request_messages(
    messages: list[dict[str, Any]], request_index: int
) -> list[dict[str, Any]]:
    """Reconstruct the wrapper input immediately before one agent response."""

    completed = turn_groups(messages)[: request_index - 1]
    dropped = max(0, len(completed) - 6)
    selected = completed[dropped:]
    output = list(messages[:2])
    if dropped:
        output.append(
            {
                "role": "user",
                "content": ROLLING_NOTICE.format(dropped=dropped),
            }
        )
    for group in selected:
        output.extend(group)
    return output


def prepare(output: Path) -> dict[str, Any]:
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    template = Template(
        CHAT_TEMPLATE.read_text(encoding="utf-8"),
        undefined=StrictUndefined,
    )
    dynamic = read_json(SOURCE_RUN / "DYNAMIC_MANIFEST.json")
    wide_cases = [
        row for row in dynamic["cases"] if int(row["length"]) > 4096
    ]
    if len(wide_cases) != 4:
        raise ValueError(
            f"expected four completed V7 wide cases, found {len(wide_cases)}"
        )
    independent_cases = []
    excluded_aliases = []
    selected_target_hashes: dict[str, str] = {}
    for row in wide_cases:
        prior_case = selected_target_hashes.get(row["source_prompt_hash"])
        if prior_case is not None:
            excluded_aliases.append(
                {
                    "case_id": row["case_id"],
                    "reason": "source_hash_equals_prior_selected_target_hash",
                    "prior_case_id": prior_case,
                }
            )
            continue
        independent_cases.append(row)
        selected_target_hashes[row["target_prompt_hash"]] = row["case_id"]
    if len(independent_cases) != 3:
        raise ValueError(
            "expected three hash-independent cases after alias exclusion, "
            f"found {len(independent_cases)}"
        )
    trajectories = {
        instance_id: read_json(
            SOURCE_RUN / instance_id / f"{instance_id}.traj.json"
        )["messages"]
        for instance_id in INSTANCE_ORDER
    }
    cases: list[dict[str, Any]] = []
    manifests: dict[str, list[dict[str, Any]]] = {
        arm: [] for arm in ARMS
    }
    prompt_validations = []
    for original in independent_cases:
        match = re.search(
            r"-m(?P<model_index>\d+)-s\d+-q(?P<request_index>\d+)-",
            original["case_id"],
        )
        if match is None:
            raise ValueError(f"unparseable case id: {original['case_id']}")
        model_index = int(match.group("model_index"))
        request_index = int(match.group("request_index"))
        instance_id = INSTANCE_ORDER[model_index - 1]
        messages = trajectories[instance_id]
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
            raise ValueError(
                f"{original['case_id']}: reconstructed source hash mismatch"
            )
        if target_hash != original["target_prompt_hash"]:
            raise ValueError(
                f"{original['case_id']}: reconstructed target hash mismatch"
            )
        wide_span = {
            "source_start": int(original["source_start"]),
            "target_start": int(original["target_start"]),
            "length": int(original["length"]),
        }
        case_id = f"{instance_id}-q{request_index}"
        cases.append(
            {
                "case_id": case_id,
                "instance_id": instance_id,
                "request_index": request_index,
                "source_input_ids": source_ids,
                "source_prompt_tokens": len(source_ids),
                "target_input_ids": target_ids,
                "target_prompt_tokens": len(target_ids),
                "general_tokens": 4096,
                "v7_tokens": wide_span["length"],
            }
        )
        for arm, span in (
            ("general_4k", capped_tail(wide_span, 4096)),
            ("coding_evidence_payoff_v7", wide_span),
        ):
            row = manifest_case(
                case_id=case_id,
                policy_label=arm,
                source_ids=source_ids,
                target_ids=target_ids,
                span=span,
            )
            row["target_uses"] = REPETITIONS
            manifests[arm].append(row)
        prompt_validations.append(
            {
                "case_id": case_id,
                "source_prompt_hash": source_hash,
                "target_prompt_hash": target_hash,
                "matched_agent_manifest": True,
            }
        )

    cases_path = output / "PAIRED_CASES.json"
    write_json(cases_path, {"cases": cases})
    manifest_hashes = {}
    for arm, rows in manifests.items():
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
    registration = {
        "registered_before_gpu": True,
        "classification": (
            "paired fixed consecutive-request speed diagnostic; not agent "
            "accuracy or end-to-end SOTA evidence"
        ),
        "model": MODEL,
        "arms": {
            "general_4k": "same token-identical contiguous tail capped at 4096",
            "coding_evidence_payoff_v7": (
                "same token-identical contiguous tail at the naturally "
                "selected V7 length, 5326-6144 tokens"
            ),
        },
        "cases": len(cases),
        "excluded_cross_role_prompt_aliases": excluded_aliases,
        "target_repetitions": REPETITIONS,
        "decode_tokens": 1,
        "prefetch": False,
        "source_boundary": (
            "Each source is the hash-verified natural request immediately "
            "preceding its target. Source build is reported separately."
        ),
        "prompt_validations": prompt_validations,
        "inputs": {
            "cases_sha256": sha256_file(cases_path),
            "chat_template_sha256": sha256_file(CHAT_TEMPLATE),
            "source_dynamic_manifest_sha256": sha256_file(
                SOURCE_RUN / "DYNAMIC_MANIFEST.json"
            ),
            "source_trajectory_sha256": {
                instance_id: sha256_file(
                    SOURCE_RUN
                    / instance_id
                    / f"{instance_id}.traj.json"
                )
                for instance_id in INSTANCE_ORDER
            },
            "manifest_sha256": manifest_hashes,
            "runner_sha256": sha256_file(Path(__file__)),
        },
        "motivation_gate": {
            "median_ttft_reduction_percent_min": 5.0,
            "p95_ttft_reduction_percent_min": 5.0,
            "case_median_wins_min": len(cases),
            "expected_copy_events_per_arm": len(cases) * REPETITIONS,
            "fallback_events_max": 0,
        },
        "protected": {
            "old_dirty_checkout_modified": False,
            "paper_modified": False,
            "existing_preregistration_thresholds_modified": False,
        },
        "status": "REGISTERED_BEFORE_DIAGNOSTIC_GPU_RUN",
    }
    write_json(output / "PAIRED_REGISTRATION.json", registration)
    return registration


def run_arm(output: Path, arm: str, port: int) -> dict[str, Any]:
    if not (output / "PAIRED_REGISTRATION.json").exists():
        prepare(output)
    cases = read_json(output / "PAIRED_CASES.json")["cases"]
    process, stream, base_url = launch_server(
        output=output, arm=arm, port=port
    )
    source_rows = []
    rows = []
    try:
        for case in cases:
            source_rows.append(
                {
                    **generate(
                        base_url=base_url,
                        input_ids=case["source_input_ids"],
                        key=f"paired-source-{arm}-{case['case_id']}",
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
                                f"paired-target-{arm}-{case['case_id']}-"
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
        "source_rows": source_rows,
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


def percentile(values: list[float], fraction: float) -> float:
    return sorted(values)[math.ceil(fraction * len(values)) - 1]


def summarize(output: Path) -> dict[str, Any]:
    cases = read_json(output / "PAIRED_CASES.json")["cases"]
    expected_cases = len(cases)
    values = {
        arm: read_json(output / "generations" / f"{arm}.json")
        for arm in ARMS
    }
    arms = {}
    for arm, value in values.items():
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
            "copy_events": len(copies),
            "fallback_events": len(fallbacks),
            "copied_tokens": sum(
                int(row["copied_k_tokens"]) for row in copies
            ),
            "source_build_total_ms": sum(
                float(row["elapsed_ms"]) for row in value["source_rows"]
            ),
            "case_median_ttft_ms": {
                case_id: statistics.median(
                    float(row["ttft_ms"])
                    for row in value["rows"]
                    if row["case_id"] == case_id
                )
                for case_id in sorted(
                    {row["case_id"] for row in value["rows"]}
                )
            },
        }
    general = arms["general_4k"]
    v7 = arms["coding_evidence_payoff_v7"]
    case_wins = sum(
        v7["case_median_ttft_ms"][case_id]
        < general["case_median_ttft_ms"][case_id]
        for case_id in general["case_median_ttft_ms"]
    )
    comparison = {
        "median_ttft_reduction_percent": 100
        * (1 - v7["median_ttft_ms"] / general["median_ttft_ms"]),
        "p95_ttft_reduction_percent": 100
        * (1 - v7["p95_ttft_ms"] / general["p95_ttft_ms"]),
        "case_median_wins": case_wins,
    }
    gate = {
        "median_passed": comparison["median_ttft_reduction_percent"] >= 5,
        "p95_passed": comparison["p95_ttft_reduction_percent"] >= 5,
        "case_wins_passed": case_wins == expected_cases,
        "copy_events_passed": all(
            arms[arm]["copy_events"] == expected_cases * REPETITIONS
            for arm in ARMS
        ),
        "fallback_passed": all(
            arms[arm]["fallback_events"] == 0 for arm in ARMS
        ),
    }
    result = {
        "classification": (
            "paired fixed consecutive-request speed diagnostic; not agent "
            "accuracy or end-to-end SOTA evidence"
        ),
        "arms": arms,
        "comparison": comparison,
        "gate": {**gate, "overall_passed": all(gate.values())},
        "prefetch": False,
    }
    write_json(output / "PAIRED_RESULT.json", result)
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
    campaign = subparsers.add_parser("campaign")
    campaign.add_argument("--port", type=int, default=33300)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.command == "prepare":
        result = prepare(output)
    elif args.command == "run-arm":
        result = run_arm(output, args.arm, args.port)
    elif args.command == "summarize":
        result = summarize(output)
    else:
        prepare(output)
        for arm in ARMS:
            run_arm(output, arm, args.port)
        result = summarize(output)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
