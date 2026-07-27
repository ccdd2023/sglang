#!/usr/bin/env python3
"""Run the V14b-selected head/tail Dense repair on all 225 tasks."""

from __future__ import annotations

import argparse
import ast
import json
import statistics
from pathlib import Path
from typing import Any

from benchmark.multi_workflow.run_bridge_reuse_pilot import (
    generate,
    manifest_case,
    sha256_file,
    stop_server,
    write_json,
)
from benchmark.multi_workflow.run_coding_native_workload_v10 import (
    MODEL,
    flush_cache,
    launch_server,
    read_json,
    read_jsonl,
)
from benchmark.multi_workflow.run_v12_full225_accuracy import (
    BASELINE_ARMS,
    OLD_CASES,
    OLD_RESULT,
    TRUTH,
    WORKLOAD,
    _evaluate,
    _metrics,
    _paired_delta,
    _selected_cases,
    extract_python,
)


ARTIFACTS = Path("/home/gfy/CodeMAS_Project/kvflow-artifacts")
V14B_RESULT = (
    ARTIFACTS
    / "impactkv_v14b_chunk_matched_logit_impact_20260727/V14B_RESULT.json"
)
DEFAULT_OUTPUT = (
    ARTIFACTS / "impactkv_v14b_headtail_full225_20260727"
)
ARM = "v14b_head16_tail16"
CONTROL = "coding_repo_boundary_v12"


def manifest_rows(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for case in cases:
        length = int(case["segment_tokens"]) - 32
        if length <= 0:
            raise ValueError(f"{case['case_id']}: span too short")
        row = manifest_case(
            case_id=f"v14b:{case['original_case_id']}",
            policy_label=ARM,
            source_ids=case["source_input_ids"],
            target_ids=case["target_input_ids"],
            span={
                "source_start": int(case["source_start"]) + 16,
                "target_start": int(case["target_start"]) + 16,
                "length": length,
            },
        )
        row.update(
            allow_target_prefix_bypass=True,
            source_id=f"v14b:{case['original_case_id']}",
            target_uses=1,
        )
        rows.append(row)
    return rows


def register(output: Path) -> dict[str, Any]:
    path = output / "V14B_FULL225_REGISTRATION.json"
    if path.exists():
        value = read_json(path)
        if value["inputs"]["v14b_result_sha256"] != sha256_file(V14B_RESULT):
            raise ValueError("registered V14b result changed")
        return value
    if output.exists():
        raise FileExistsError(output)
    selection = read_json(V14B_RESULT)
    if selection["selected_candidate"] != "repair_head16_tail16":
        raise ValueError("V14b did not select head16_tail16")
    output.mkdir(parents=True)
    cases = _selected_cases()
    manifest_path = output / "manifests" / f"{ARM}.json"
    write_json(
        manifest_path,
        {
            "cache_dtype": "bfloat16",
            "cases": manifest_rows(cases),
            "lease_ttl_s": 900,
            "ledger_path": str(
                output / "server" / ARM / "EXACT_LEDGER.jsonl"
            ),
            "model_id": MODEL,
            "ordinary_prefix_reuse_enabled": True,
            "rope": {
                "base": 1_000_000,
                "is_neox_style": True,
                "rotary_dim": 128,
            },
            "version": 2,
        },
    )
    value = {
        "date": "2026-07-27",
        "experiment": "V14b selected head/tail repair full-225 accuracy",
        "registered_before_gpu": True,
        "model": MODEL,
        "method": (
            "Dense target header and first 16 shared tokens; shifted-copy "
            "shared interior; Dense final 16 shared tokens and target suffix"
        ),
        "protocol": {
            "cases": len(cases),
            "decode": {"max_new_tokens": 512, "temperature": 0},
            "first_real_target_only": True,
            "prefetch": False,
            "controls_reused_byte_for_byte": [
                CONTROL,
                *BASELINE_ARMS,
            ],
        },
        "frozen_gates": {
            "accuracy_gain_vs_v12_pp_min": 1.0,
            "accuracy_vs_kvcomm_pp_min": 0.0,
            "accuracy_vs_cacheblend_pp_min": 0.0,
            "copy_events": len(cases),
            "fallback_events_max": 0,
            "run_long_context_ttft_only_if_accuracy_gain_passes": True,
        },
        "inputs": {
            "cases_sha256": sha256_file(OLD_CASES),
            "manifest_sha256": sha256_file(manifest_path),
            "old_result_sha256": sha256_file(OLD_RESULT),
            "truth_sha256": sha256_file(TRUTH),
            "v14b_result_sha256": sha256_file(V14B_RESULT),
            "workload_sha256": sha256_file(WORKLOAD),
        },
        "protected": {
            "existing_preregistration_thresholds_modified": False,
            "old_dirty_checkout_modified": False,
            "paper_modified": False,
            "prefetch": False,
        },
        "scope": (
            "Development full-coverage audit on already exposed tasks; not a "
            "fresh or confirmatory holdout."
        ),
        "status": "REGISTERED_BEFORE_V14B_FULL225_GPU",
    }
    write_json(path, value)
    return value


def run(output: Path, port: int) -> dict[str, Any]:
    register(output)
    destination = output / "generations" / f"{ARM}.json"
    if destination.exists():
        return {"status": "already_complete"}
    ledger_path = output / "server" / ARM / "EXACT_LEDGER.jsonl"
    if ledger_path.exists():
        raise FileExistsError(ledger_path)
    cases = _selected_cases()
    process, stream, base_url = launch_server(
        output=output, arm=ARM, port=port
    )
    rows = []
    source_rows = []
    try:
        generate(
            base_url=base_url,
            input_ids=[100] * 128,
            key="v14b-full225-server-warmup",
            max_new_tokens=1,
            stream=True,
        )
        flush_cache(base_url)
        for index, case in enumerate(cases, start=1):
            source_rows.append(
                {
                    **generate(
                        base_url=base_url,
                        input_ids=case["source_input_ids"],
                        key=f"v14b-source-{case['case_id']}",
                        max_new_tokens=1,
                        stream=False,
                    ),
                    "case_id": case["original_case_id"],
                }
            )
            rows.append(
                {
                    **generate(
                        base_url=base_url,
                        input_ids=case["target_input_ids"],
                        key=f"v14b-target-{case['case_id']}",
                        max_new_tokens=int(case["max_new_tokens"]),
                        stream=True,
                    ),
                    "arm": ARM,
                    "case_id": case["original_case_id"],
                    "suite": case["suite"],
                }
            )
            if index % 10 == 0:
                write_json(
                    output / "V14B_FULL225_PROGRESS.json",
                    {"complete": index, "total": len(cases)},
                )
    finally:
        stop_server(process, stream)
    ledger = read_jsonl(ledger_path)
    write_json(
        destination,
        {
            "arm": ARM,
            "ledger_rows": ledger,
            "rows": rows,
            "source_rows": source_rows,
            "status": "complete",
        },
    )
    return {
        "copy_events": sum(
            row.get("event") == "target_copied" for row in ledger
        ),
        "fallback_events": sum(
            row.get("event") == "target_fallback" for row in ledger
        ),
        "rows": len(rows),
        "status": "complete",
    }


def score(output: Path) -> dict[str, Any]:
    registration = register(output)
    workload = {
        str(row["case_id"]): row for row in read_json(WORKLOAD)["cases"]
    }
    truth = {str(row["case_id"]): row for row in read_json(TRUTH)["cases"]}
    generation = read_json(output / "generations" / f"{ARM}.json")
    rows = []
    for row in generation["rows"]:
        candidate = extract_python(row["output_text"])
        passed, error = _evaluate(
            workload[row["case_id"]], truth[row["case_id"]], candidate
        )
        try:
            ast.parse(candidate)
            compiled = True
        except SyntaxError:
            compiled = False
        rows.append(
            {
                **row,
                "compiled": compiled,
                "evaluator_error": error,
                "passed": passed,
            }
        )
    old_name = {
        "copy_only_r00": CONTROL,
        **{arm: arm for arm in BASELINE_ARMS},
    }
    for row in read_json(OLD_RESULT)["rows"]:
        if row["arm"] in old_name:
            rows.append({**row, "arm": old_name[row["arm"]]})
    arms = {
        arm: _metrics([row for row in rows if row["arm"] == arm])
        for arm in (ARM, CONTROL, *BASELINE_ARMS)
    }
    outcomes = {
        arm: {
            str(row.get("original_case_id") or row["case_id"]): bool(
                row["passed"]
            )
            for row in rows
            if row["arm"] == arm
        }
        for arm in arms
    }
    comparisons = {
        f"v14b_minus_{control}": _paired_delta(
            outcomes[control], outcomes[ARM], 20260727 + index
        )
        for index, control in enumerate((CONTROL, *BASELINE_ARMS))
    }
    ledger = generation["ledger_rows"]
    mechanism = {
        "copy_events": sum(
            row.get("event") == "target_copied" for row in ledger
        ),
        "fallback_events": sum(
            row.get("event") == "target_fallback" for row in ledger
        ),
    }
    gates = registration["frozen_gates"]
    gain = comparisons[f"v14b_minus_{CONTROL}"][
        "treatment_minus_control_pp"
    ]
    verdict = {
        "accuracy_gain_vs_v12": (
            gain >= gates["accuracy_gain_vs_v12_pp_min"]
        ),
        "accuracy_vs_kvcomm": (
            comparisons["v14b_minus_kvcomm_native_reuse"][
                "treatment_minus_control_pp"
            ]
            >= gates["accuracy_vs_kvcomm_pp_min"]
        ),
        "accuracy_vs_cacheblend": (
            comparisons["v14b_minus_cacheblend_native_reuse"][
                "treatment_minus_control_pp"
            ]
            >= gates["accuracy_vs_cacheblend_pp_min"]
        ),
        "mechanism": (
            mechanism["copy_events"] == gates["copy_events"]
            and mechanism["fallback_events"]
            <= gates["fallback_events_max"]
        ),
    }
    value = {
        "arms": arms,
        "comparisons": comparisons,
        "mechanism": mechanism,
        "rows": rows,
        "status": "V14B_FULL225_COMPLETE",
        "verdict": verdict,
    }
    amendment = output / "V14B_SCORING_AMENDMENT_001.json"
    if (output / "V14B_FULL225_RESULT.json").exists() and not amendment.exists():
        write_json(
            amendment,
            {
                "date": "2026-07-27",
                "change": (
                    "Compute candidate compile_rate with ast.parse instead "
                    "of the initial hard-coded True placeholder."
                ),
                "unchanged": [
                    "generated outputs",
                    "sealed functional pass/fail outcomes",
                    "paired comparisons",
                    "all frozen gates and verdicts",
                ],
                "status": "SCORING_METADATA_CORRECTED",
            },
        )
    write_json(output / "V14B_FULL225_RESULT.json", value)
    return {key: value[key] for key in value if key != "rows"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("register")
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--port", type=int, default=33460)
    sub.add_parser("score")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.command == "register":
        value = register(output)
    elif args.command == "run":
        value = run(output, args.port)
    else:
        value = score(output)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
