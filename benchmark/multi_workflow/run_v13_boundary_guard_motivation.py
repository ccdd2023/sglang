#!/usr/bin/env python3
"""Test KV-drift-motivated dense guards around the shared coding island."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from benchmark.multi_workflow.probe_v13_kv_boundary import (
    DEFAULT_OUTPUT as PROBE_OUTPUT,
    selected_cases,
)
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
    OLD_RESULT,
    TRUTH,
    WORKLOAD,
    _evaluate,
    extract_python,
)


ARTIFACTS = Path("/home/gfy/CodeMAS_Project/kvflow-artifacts")
DEFAULT_OUTPUT = (
    ARTIFACTS / "impactkv_v13_boundary_guard_motivation_20260727"
)
ARMS = ("head16", "tail16", "head16_tail16")
BASELINES = ("dense", "coding_repo_boundary_v12")
GUARDS = {
    "head16": (16, 0),
    "tail16": (0, 16),
    "head16_tail16": (16, 16),
}


def guarded_span(case: dict[str, Any], arm: str) -> dict[str, int]:
    if arm not in GUARDS:
        raise ValueError(arm)
    head, tail = GUARDS[arm]
    length = int(case["segment_tokens"]) - head - tail
    if length <= 0:
        raise ValueError(f"{case['case_id']}: guard consumes shared span")
    return {
        "source_start": int(case["source_start"]) + head,
        "target_start": int(case["target_start"]) + head,
        "length": length,
    }


def manifest_rows(cases: list[dict[str, Any]], arm: str) -> list[dict[str, Any]]:
    rows = []
    for case in cases:
        row = manifest_case(
            case_id=f"v13:{arm}:{case['original_case_id']}",
            policy_label=f"v13_kv_guard_{arm}",
            source_ids=case["source_input_ids"],
            target_ids=case["target_input_ids"],
            span=guarded_span(case, arm),
        )
        row.update(
            allow_target_prefix_bypass=True,
            source_id=f"v13:{arm}:{case['original_case_id']}",
            target_uses=1,
        )
        rows.append(row)
    return rows


def register(output: Path) -> dict[str, Any]:
    path = output / "V13_GUARD_REGISTRATION.json"
    if path.exists():
        value = read_json(path)
        if (
            value["inputs"]["probe_result_sha256"]
            != sha256_file(PROBE_OUTPUT / "V13_KV_PROBE_RESULT.json")
        ):
            raise ValueError("registered KV probe result changed")
        return value
    if output.exists():
        raise FileExistsError(output)
    probe = read_json(PROBE_OUTPUT / "V13_KV_PROBE_RESULT.json")
    if probe["recommended_guard"] != "head16_tail16":
        raise ValueError("probe did not select the registered V13 candidate")
    output.mkdir(parents=True)
    cases = selected_cases()
    manifest_hashes = {}
    for arm in ARMS:
        manifest_path = output / "manifests" / f"{arm}.json"
        write_json(
            manifest_path,
            {
                "cache_dtype": "bfloat16",
                "cases": manifest_rows(cases, arm),
                "lease_ttl_s": 900,
                "ledger_path": str(
                    output / "server" / arm / "EXACT_LEDGER.jsonl"
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
        manifest_hashes[arm] = sha256_file(manifest_path)
    value = {
        "date": "2026-07-27",
        "experiment": "V13 KV-drift boundary-guard motivation accuracy",
        "registered_before_gpu": True,
        "model": MODEL,
        "arms": list(ARMS),
        "protocol": {
            "cases": len(cases),
            "decode": {"max_new_tokens": 512, "temperature": 0},
            "first_real_target_only": True,
            "prefetch": False,
            "controls_reused_byte_for_byte": list(BASELINES),
        },
        "method": {
            "head16": "keep first 16 shared tokens dense",
            "tail16": "keep last 16 shared tokens dense",
            "head16_tail16": (
                "keep first and last 16 shared tokens dense; copy the interior"
            ),
            "probe_recommended": "head16_tail16",
        },
        "frozen_gates": {
            "candidate_accuracy_not_below_v12": True,
            "copy_events_per_arm": len(cases),
            "fallback_events_max": 0,
            "selection": (
                "highest functional accuracy, then most copied tokens; "
                "probe-recommended arm wins an exact tie"
            ),
        },
        "inputs": {
            "manifest_sha256": manifest_hashes,
            "old_result_sha256": sha256_file(OLD_RESULT),
            "probe_result_sha256": sha256_file(
                PROBE_OUTPUT / "V13_KV_PROBE_RESULT.json"
            ),
            "truth_sha256": sha256_file(TRUTH),
            "workload_sha256": sha256_file(WORKLOAD),
        },
        "protected": {
            "existing_preregistration_thresholds_modified": False,
            "old_dirty_checkout_modified": False,
            "paper_modified": False,
            "prefetch": False,
        },
        "scope": (
            "Development motivation subset selected without accuracy labels; "
            "not a formal or fresh-holdout result."
        ),
        "status": "REGISTERED_BEFORE_V13_GUARD_GPU",
    }
    write_json(path, value)
    return value


def run_arm(output: Path, arm: str, port: int) -> dict[str, Any]:
    register(output)
    if arm not in ARMS:
        raise ValueError(arm)
    destination = output / "generations" / f"{arm}.json"
    if destination.exists():
        return {"arm": arm, "status": "already_complete"}
    ledger_path = output / "server" / arm / "EXACT_LEDGER.jsonl"
    if ledger_path.exists():
        raise FileExistsError(ledger_path)
    cases = selected_cases()
    process, stream, base_url = launch_server(
        output=output,
        arm=arm,
        port=port,
    )
    rows = []
    sources = []
    try:
        generate(
            base_url=base_url,
            input_ids=[100] * 128,
            key=f"v13-guard-server-warmup-{arm}",
            max_new_tokens=1,
            stream=True,
        )
        flush_cache(base_url)
        for case in cases:
            sources.append(
                {
                    **generate(
                        base_url=base_url,
                        input_ids=case["source_input_ids"],
                        key=f"v13-guard-source-{arm}-{case['case_id']}",
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
                        key=f"v13-guard-target-{arm}-{case['case_id']}",
                        max_new_tokens=int(case["max_new_tokens"]),
                        stream=True,
                    ),
                    "arm": arm,
                    "case_id": case["original_case_id"],
                    "suite": case["suite"],
                }
            )
    finally:
        stop_server(process, stream)
    ledger = read_jsonl(ledger_path)
    write_json(
        destination,
        {
            "arm": arm,
            "ledger_rows": ledger,
            "rows": rows,
            "source_rows": sources,
            "status": "complete",
        },
    )
    return {
        "arm": arm,
        "copy_events": sum(
            row.get("event") == "target_copied" for row in ledger
        ),
        "fallback_events": sum(
            row.get("event") == "target_fallback" for row in ledger
        ),
        "rows": len(rows),
        "status": "complete",
    }


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "cases": len(rows),
        "pass_rate": statistics.mean(bool(row["passed"]) for row in rows),
        "passed": sum(bool(row["passed"]) for row in rows),
    }


def score(output: Path) -> dict[str, Any]:
    registration = register(output)
    workload = {
        str(row["case_id"]): row for row in read_json(WORKLOAD)["cases"]
    }
    truth = {str(row["case_id"]): row for row in read_json(TRUTH)["cases"]}
    selected_ids = {
        str(row["original_case_id"]) for row in selected_cases()
    }
    rows = []
    mechanism = {}
    for arm in ARMS:
        payload = read_json(output / "generations" / f"{arm}.json")
        ledger = payload["ledger_rows"]
        mechanism[arm] = {
            "copied_tokens": sum(
                int(row.get("copied_k_tokens", 0))
                for row in ledger
                if row.get("event") == "target_copied"
            ),
            "copy_events": sum(
                row.get("event") == "target_copied" for row in ledger
            ),
            "fallback_events": sum(
                row.get("event") == "target_fallback" for row in ledger
            ),
        }
        for row in payload["rows"]:
            candidate = extract_python(row["output_text"])
            passed, error = _evaluate(
                workload[row["case_id"]],
                truth[row["case_id"]],
                candidate,
            )
            rows.append(
                {**row, "evaluator_error": error, "passed": passed}
            )
    old_rows = read_json(OLD_RESULT)["rows"]
    old_name = {
        "dense": "dense",
        "copy_only_r00": "coding_repo_boundary_v12",
    }
    for row in old_rows:
        if (
            row["arm"] in old_name
            and str(row.get("original_case_id") or row["case_id"])
            in selected_ids
        ):
            rows.append({**row, "arm": old_name[row["arm"]]})
    arms = {
        arm: _metrics([row for row in rows if row["arm"] == arm])
        for arm in (*ARMS, *BASELINES)
    }
    v12 = arms["coding_repo_boundary_v12"]["pass_rate"]
    eligible = [
        arm
        for arm in ARMS
        if arms[arm]["pass_rate"] >= v12
        and mechanism[arm]["copy_events"]
        == registration["frozen_gates"]["copy_events_per_arm"]
        and mechanism[arm]["fallback_events"]
        <= registration["frozen_gates"]["fallback_events_max"]
    ]
    selected = (
        max(
            eligible,
            key=lambda arm: (
                arms[arm]["pass_rate"],
                mechanism[arm]["copied_tokens"],
                arm == "head16_tail16",
            ),
        )
        if eligible
        else None
    )
    value = {
        "arms": arms,
        "mechanism": mechanism,
        "rows": rows,
        "selected_candidate": selected,
        "status": "V13_GUARD_MOTIVATION_COMPLETE",
        "verdict": {
            "candidate_advances": selected is not None,
            "probe_recommended_advances": selected == "head16_tail16",
        },
    }
    write_json(output / "V13_GUARD_RESULT.json", value)
    return {key: value[key] for key in value if key != "rows"}


def campaign(output: Path, port: int) -> dict[str, Any]:
    register(output)
    for arm in ARMS:
        run_arm(output, arm, port)
    return score(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("register")
    run = sub.add_parser("run-arm")
    run.add_argument("--arm", choices=ARMS, required=True)
    run.add_argument("--port", type=int, default=33440)
    sub.add_parser("score")
    campaign_parser = sub.add_parser("campaign")
    campaign_parser.add_argument("--port", type=int, default=33440)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.command == "register":
        value = register(output)
    elif args.command == "run-arm":
        value = run_arm(output, args.arm, args.port)
    elif args.command == "score":
        value = score(output)
    else:
        value = campaign(output, args.port)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
