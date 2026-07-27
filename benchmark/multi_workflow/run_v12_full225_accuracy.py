#!/usr/bin/env python3
"""Measure V12 functional accuracy on the frozen 225-case coding workload.

The treatment is the first real target after its registered source request.
This exercises V12's shifted repository-KV copy without a synthetic target
warmup or target prefetch.  Existing byte-frozen Dense, native KVCOMM, and
native CacheBlend generations are rescored/reused as paired controls.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import resource
import statistics
import subprocess
import sys
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
    PROJECT,
    flush_cache,
    launch_server,
    read_json,
    read_jsonl,
)


ARTIFACTS = Path("/home/gfy/CodeMAS_Project/kvflow-artifacts")
OLD_FULL225 = ARTIFACTS / "impactkv_full225_accuracy_audit_20260724"
OLD_CASES = OLD_FULL225 / "FULL225_CASES.json"
OLD_RESULT = OLD_FULL225 / "FULL225_RESULT.json"
FRONTIER = ARTIFACTS / "impactkv_native_frontier_20260719/workload_v2"
WORKLOAD = FRONTIER / "COMMON_WORKLOAD.json"
TRUTH = FRONTIER / "sealed_evaluator/COMMON_TRUTH.json"
DEFAULT_OUTPUT = ARTIFACTS / "impactkv_v12_full225_accuracy_20260727"
ARM = "coding_repo_boundary_v12"
CANARY_ARM = f"{ARM}_canary"
BASELINE_ARMS = (
    "dense",
    "kvcomm_native_reuse",
    "cacheblend_native_reuse",
)
EXPECTED_SUITES = {"humaneval_mas": 129, "mbpp_session": 96}
BOOTSTRAP_SEED = 20260727
BOOTSTRAP_ITERATIONS = 10_000


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _selected_cases() -> list[dict[str, Any]]:
    cases = read_json(OLD_CASES)["cases"]
    counts = {
        suite: sum(row["suite"] == suite for row in cases)
        for suite in EXPECTED_SUITES
    }
    if len(cases) != 225 or counts != EXPECTED_SUITES:
        raise ValueError(f"unexpected frozen case coverage: {counts}")
    if len({str(row["original_case_id"]) for row in cases}) != 225:
        raise ValueError("frozen cases must contain 225 unique original IDs")
    return cases


def _manifest_rows(
    cases: list[dict[str, Any]], policy_label: str
) -> list[dict[str, Any]]:
    rows = []
    for case in cases:
        length = int(case["segment_tokens"])
        row = manifest_case(
            case_id=str(case["case_id"]),
            policy_label=policy_label,
            source_ids=case["source_input_ids"],
            target_ids=case["target_input_ids"],
            span={
                "source_start": int(case["source_start"]),
                "target_start": int(case["target_start"]),
                "length": length,
            },
        )
        row.update(
            allow_target_prefix_bypass=True,
            source_id=str(case["case_id"]),
            target_uses=1,
        )
        rows.append(row)
    return rows


def _manifest(
    output: Path,
    arm: str,
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "cache_dtype": "bfloat16",
        "cases": _manifest_rows(cases, ARM),
        "lease_ttl_s": 900,
        "ledger_path": str(output / "server" / arm / "EXACT_LEDGER.jsonl"),
        "model_id": MODEL,
        "ordinary_prefix_reuse_enabled": True,
        "rope": {
            "base": 1_000_000,
            "is_neox_style": True,
            "rotary_dim": 128,
        },
        "version": 2,
    }


def register(output: Path) -> dict[str, Any]:
    path = output / "V12_ACCURACY_REGISTRATION.json"
    if path.exists():
        value = read_json(path)
        inputs = value["inputs"]
        for name, source in (
            ("old_cases_sha256", OLD_CASES),
            ("old_result_sha256", OLD_RESULT),
            ("truth_sha256", TRUTH),
            ("workload_sha256", WORKLOAD),
        ):
            if inputs[name] != sha256_file(source):
                raise ValueError(f"registered input changed: {name}")
        return value
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    cases = _selected_cases()
    if max(int(row["segment_tokens"]) for row in cases) > 4096:
        raise ValueError("accuracy interpretation assumes every shared span <=4K")

    formal_manifest = output / "manifests" / f"{ARM}.json"
    canary_cases = [
        next(row for row in cases if row["suite"] == suite)
        for suite in EXPECTED_SUITES
    ]
    canary_manifest = output / "manifests" / f"{CANARY_ARM}.json"
    write_json(formal_manifest, _manifest(output, ARM, cases))
    write_json(canary_manifest, _manifest(output, CANARY_ARM, canary_cases))

    registration = {
        "date": "2026-07-27",
        "registered_before_gpu": True,
        "experiment": "V12 frozen full-225 functional-accuracy audit",
        "model": MODEL,
        "arms": [ARM, *BASELINE_ARMS],
        "protocol": {
            "cases": 225,
            "decode": {"max_new_tokens": 512, "temperature": 0},
            "functional_metric": (
                "pass@1 from the frozen sealed HumanEval/MBPP executor"
            ),
            "lifecycle": (
                "one source request then the first real, fully generated target"
            ),
            "prefetch": False,
            "statistics": {
                "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "paired_accuracy_deltas": True,
            },
            "suites": EXPECTED_SUITES,
        },
        "frozen_gates": {
            "accuracy_drop_vs_dense_pp_max": 2.0,
            "fallback_events_max": 0,
            "mechanism_copy_events": 225,
            "v12_minus_kvcomm_native_reuse_pp_min": 0.0,
            "stretch_v12_minus_cacheblend_native_reuse_pp_min": 0.0,
        },
        "interpretation": {
            "all_shared_spans_le_4k": True,
            "cold_first_target_v12_equals_general_span_selection": True,
            "target_prefix_bypass_expected_on_first_target": False,
            "why_first_target": (
                "It measures the lossy shifted-copy semantics without an "
                "artificial repeated-target cache warmup."
            ),
        },
        "inputs": {
            "old_cases_path": str(OLD_CASES),
            "old_cases_sha256": sha256_file(OLD_CASES),
            "old_result_path": str(OLD_RESULT),
            "old_result_sha256": sha256_file(OLD_RESULT),
            "truth_sha256": sha256_file(TRUTH),
            "workload_sha256": sha256_file(WORKLOAD),
        },
        "outputs": {
            "canary_manifest_sha256": sha256_file(canary_manifest),
            "formal_manifest_sha256": sha256_file(formal_manifest),
        },
        "protected": {
            "existing_preregistration_thresholds_modified": False,
            "old_dirty_checkout_modified": False,
            "paper_modified": False,
            "prior_artifacts_modified": False,
        },
        "scope": (
            "Development audit on an already exposed workload. Existing "
            "controls are reused byte-for-byte; this is not a fresh holdout."
        ),
        "status": "REGISTERED_BEFORE_V12_ACCURACY_GPU",
    }
    write_json(path, registration)
    return registration


def run(output: Path, port: int, canary: bool) -> dict[str, Any]:
    register(output)
    arm = CANARY_ARM if canary else ARM
    cases = _selected_cases()
    if canary:
        cases = [
            next(row for row in cases if row["suite"] == suite)
            for suite in EXPECTED_SUITES
        ]
    destination = (
        output / "canary" / "V12_GENERATIONS.json"
        if canary
        else output / "generations" / "V12_GENERATIONS.json"
    )
    if destination.exists():
        return {
            "arm": ARM,
            "cases": len(read_json(destination)["rows"]),
            "status": "already_complete",
        }
    ledger_path = output / "server" / arm / "EXACT_LEDGER.jsonl"
    if ledger_path.exists():
        raise FileExistsError(f"refusing to reuse ledger: {ledger_path}")

    process, stream, base_url = launch_server(
        output=output, arm=arm, port=port
    )
    source_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    try:
        generate(
            base_url=base_url,
            input_ids=[100] * 128,
            key=f"v12-accuracy-server-warmup-{arm}",
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
                        key=f"v12-accuracy-source-{case['case_id']}",
                        max_new_tokens=1,
                        stream=False,
                    ),
                    "case_id": case["case_id"],
                }
            )
            target_rows.append(
                {
                    **generate(
                        base_url=base_url,
                        input_ids=case["target_input_ids"],
                        key=f"v12-accuracy-target-{case['case_id']}",
                        max_new_tokens=int(case["max_new_tokens"]),
                        stream=True,
                    ),
                    "arm": ARM,
                    "case_id": case["case_id"],
                    "original_case_id": case["original_case_id"],
                    "prompt_sha256": case["prompt_sha256"],
                    "suite": case["suite"],
                }
            )
            if not canary and index % 10 == 0:
                write_json(
                    output / "V12_ACCURACY_PROGRESS.json",
                    {"complete": index, "total": len(cases)},
                )
    finally:
        stop_server(process, stream)

    ledger = read_jsonl(ledger_path)
    value = {
        "arm": ARM,
        "ledger_rows": ledger,
        "rows": target_rows,
        "source_rows": source_rows,
        "status": "canary_complete" if canary else "generation_complete",
    }
    write_json(destination, value)
    return {
        "arm": ARM,
        "cases": len(target_rows),
        "copy_events": sum(
            row.get("event") == "target_copied" for row in ledger
        ),
        "fallback_events": sum(
            row.get("event") == "target_fallback" for row in ledger
        ),
        "status": value["status"],
    }


def extract_python(text: str) -> str:
    if "```python" in text:
        return text.split("```python", 1)[1].split("```", 1)[0].strip()
    if "```" in text:
        return text.split("```", 1)[1].split("```", 1)[0].strip()
    return text.strip()


def _limits() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
    resource.setrlimit(
        resource.RLIMIT_AS, (2_147_483_648, 2_147_483_648)
    )
    resource.setrlimit(
        resource.RLIMIT_FSIZE, (1_048_576, 1_048_576)
    )


def _evaluate(
    case: dict[str, Any], truth: dict[str, Any], candidate: str
) -> tuple[bool, str | None]:
    if case["suite"] == "humaneval_mas":
        program = "\n\n".join(
            (
                candidate,
                str(truth["test"]),
                f"check({case['metadata']['official_entry_point']})",
            )
        )
    else:
        program = "\n".join(
            [
                *map(str, truth.get("test_imports") or []),
                candidate,
                *map(str, truth.get("test_list") or []),
            ]
        )
    try:
        result = subprocess.run(
            [sys.executable, "-I", "-c", program],
            capture_output=True,
            text=True,
            timeout=15,
            preexec_fn=_limits,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, "timeout"
    return (
        result.returncode == 0,
        None if result.returncode == 0 else result.stderr[-3000:],
    )


def _wilson(passed: int, count: int) -> list[float]:
    z = 1.959963984540054
    p = passed / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    radius = (
        z
        * ((p * (1 - p) / count + z * z / (4 * count * count)) ** 0.5)
        / denominator
    )
    return [center - radius, center + radius]


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    passed = sum(bool(row["passed"]) for row in rows)
    return {
        "cases": len(rows),
        "compile_rate": statistics.mean(bool(row["compiled"]) for row in rows),
        "pass_rate": passed / len(rows),
        "passed": passed,
        "wilson95": _wilson(passed, len(rows)),
    }


def _paired_delta(
    control: dict[str, bool],
    treatment: dict[str, bool],
    seed: int,
) -> dict[str, Any]:
    ids = sorted(set(control) & set(treatment))
    deltas = [
        100.0 * (float(treatment[case_id]) - float(control[case_id]))
        for case_id in ids
    ]
    rng = random.Random(seed)
    samples = sorted(
        statistics.mean(rng.choice(deltas) for _ in deltas)
        for _ in range(BOOTSTRAP_ITERATIONS)
    )
    transitions = {
        "both_fail": 0,
        "both_pass": 0,
        "control_only": 0,
        "treatment_only": 0,
    }
    for case_id in ids:
        left, right = control[case_id], treatment[case_id]
        key = (
            "both_pass"
            if left and right
            else "control_only"
            if left
            else "treatment_only"
            if right
            else "both_fail"
        )
        transitions[key] += 1
    return {
        "treatment_minus_control_pp": statistics.mean(deltas),
        "ci_low_pp": samples[249],
        "ci_high_pp": samples[9749],
        "pairs": len(ids),
        "transitions": transitions,
    }


def score(output: Path, canary: bool) -> dict[str, Any]:
    registration = register(output)
    generation = read_json(
        output / "canary" / "V12_GENERATIONS.json"
        if canary
        else output / "generations" / "V12_GENERATIONS.json"
    )
    workload = {
        str(row["case_id"]): row for row in read_json(WORKLOAD)["cases"]
    }
    truth = {
        str(row["case_id"]): row for row in read_json(TRUTH)["cases"]
    }
    v12_rows = []
    for row in generation["rows"]:
        candidate = extract_python(row["output_text"])
        passed, error = _evaluate(
            workload[row["original_case_id"]],
            truth[row["original_case_id"]],
            candidate,
        )
        try:
            ast.parse(candidate)
            compiled = True
        except SyntaxError:
            compiled = False
        v12_rows.append(
            {
                **row,
                "compiled": compiled,
                "evaluator_error": error,
                "passed": passed,
            }
        )

    selected_ids = {str(row["original_case_id"]) for row in v12_rows}
    old = read_json(OLD_RESULT)["rows"]
    rows = [*v12_rows]
    for row in old:
        if (
            row["arm"] in BASELINE_ARMS
            and str(row.get("original_case_id") or row["case_id"])
            in selected_ids
        ):
            rows.append(row)

    by_arm = {
        arm: [row for row in rows if row["arm"] == arm]
        for arm in (ARM, *BASELINE_ARMS)
        if any(row["arm"] == arm for row in rows)
    }
    arms = {arm: _metrics(subset) for arm, subset in by_arm.items()}
    outcomes = {
        arm: {
            str(row.get("original_case_id") or row["case_id"]): bool(
                row["passed"]
            )
            for row in subset
        }
        for arm, subset in by_arm.items()
    }
    comparisons = {
        f"v12_minus_{arm}": _paired_delta(
            outcomes[arm], outcomes[ARM], BOOTSTRAP_SEED + index
        )
        for index, arm in enumerate(BASELINE_ARMS)
        if arm in outcomes
    }
    ledger = generation["ledger_rows"]
    copies = sum(row.get("event") == "target_copied" for row in ledger)
    fallbacks = sum(row.get("event") == "target_fallback" for row in ledger)
    gates = registration["frozen_gates"]
    expected = len(v12_rows)
    dense_delta = comparisons.get("v12_minus_dense", {})
    kvcomm_delta = comparisons.get("v12_minus_kvcomm_native_reuse", {})
    cacheblend_delta = comparisons.get(
        "v12_minus_cacheblend_native_reuse", {}
    )
    verdict = {
        "accuracy_safety_vs_dense": (
            dense_delta.get("treatment_minus_control_pp", float("-inf"))
            >= -gates["accuracy_drop_vs_dense_pp_max"]
        ),
        "mechanism": (
            copies == expected
            and fallbacks <= gates["fallback_events_max"]
        ),
        "primary_vs_kvcomm": (
            kvcomm_delta.get("treatment_minus_control_pp", float("-inf"))
            >= gates["v12_minus_kvcomm_native_reuse_pp_min"]
        ),
        "stretch_vs_cacheblend": (
            cacheblend_delta.get(
                "treatment_minus_control_pp", float("-inf")
            )
            >= gates[
                "stretch_v12_minus_cacheblend_native_reuse_pp_min"
            ]
        ),
    }
    result = {
        "arms": arms,
        "comparisons": comparisons,
        "mechanism": {
            "copy_events": copies,
            "fallback_events": fallbacks,
            "target_prefix_bypass_events": sum(
                row.get("event") == "target_prefix_bypass"
                for row in ledger
            ),
        },
        "rows": rows,
        "status": "CANARY" if canary else "V12_ACCURACY_COMPLETE",
        "verdict": verdict,
    }
    destination = (
        output / "canary" / "V12_ACCURACY_RESULT.json"
        if canary
        else output / "V12_ACCURACY_RESULT.json"
    )
    write_json(destination, result)
    return {
        "arms": arms,
        "comparisons": comparisons,
        "mechanism": result["mechanism"],
        "status": result["status"],
        "verdict": verdict,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("register")
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--port", type=int, default=33420)
    run_parser.add_argument("--canary", action="store_true")
    score_parser = sub.add_parser("score")
    score_parser.add_argument("--canary", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.command == "register":
        value = register(output)
    elif args.command == "run":
        value = run(output, args.port, args.canary)
    else:
        value = score(output, args.canary)
    print(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
