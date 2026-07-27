#!/usr/bin/env python3
"""Retrospectively test an online-visible contract guard for V12 outputs."""

from __future__ import annotations

import argparse
import json
import resource
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

from benchmark.multi_workflow.run_bridge_reuse_pilot import (
    sha256_file,
    write_json,
)
from benchmark.multi_workflow.run_coding_native_workload_v10 import read_json
from benchmark.multi_workflow.run_v12_full225_accuracy import (
    OLD_RESULT,
    WORKLOAD,
    extract_python,
)


ARTIFACTS = Path("/home/gfy/CodeMAS_Project/kvflow-artifacts")
V12_RESULT = (
    ARTIFACTS
    / "impactkv_v12_full225_accuracy_20260727/V12_ACCURACY_RESULT.json"
)
V12_SPEED = (
    ARTIFACTS
    / "impactkv_coding_repository_boundary_v12_20260727/V12_RESULT.json"
)
DEFAULT_OUTPUT = ARTIFACTS / "impactkv_v13_visible_guard_20260727"
V12_ARM = "coding_repo_boundary_v12"
DENSE_ARM = "dense"
CACHEBLEND_ARM = "cacheblend_native_reuse"


def shared_text(case: dict[str, Any]) -> str:
    return "\n\n".join(
        str(row["text"]) for row in case["segments"] if row["reusable"]
    )


def _limits() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
    resource.setrlimit(
        resource.RLIMIT_AS, (2_147_483_648, 2_147_483_648)
    )
    resource.setrlimit(
        resource.RLIMIT_FSIZE, (1_048_576, 1_048_576)
    )


VISIBLE_PROGRAM = r"""
import ast
import doctest
import json
import re

candidate = json.loads(CANDIDATE_JSON)
shared = json.loads(SHARED_JSON)
entrypoint = json.loads(ENTRYPOINT_JSON)
result = {
    "compiled": False,
    "doctest_attempted": 0,
    "doctest_failed": 0,
    "interface_ok": False,
    "reason": None,
}
try:
    ast.parse(candidate)
    result["compiled"] = True
    namespace = {}
    exec(compile(candidate, "<candidate>", "exec"), namespace, namespace)
    result["interface_ok"] = callable(namespace.get(entrypoint))
    visible_docstrings = "\n".join(
        body
        for _, body in re.findall(
            r"(\"\"\"|''')([\s\S]*?)\1",
            shared,
        )
        if ">>>" in body
    )
    examples = doctest.DocTestParser().get_examples(visible_docstrings)
    test = doctest.DocTest(
        examples,
        namespace,
        "online_visible_examples",
        "<visible-shared-task>",
        0,
        shared,
    )
    runner = doctest.DocTestRunner(
        optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
    )
    summary = runner.run(test, out=lambda _: None, clear_globs=False)
    result["doctest_attempted"] = summary.attempted
    result["doctest_failed"] = summary.failed
except BaseException as error:
    result["reason"] = type(error).__name__ + ": " + str(error)
print("__IMPACTKV_VISIBLE__" + json.dumps(result, sort_keys=True))
"""


def visible_check(
    case: dict[str, Any], output_text: str
) -> dict[str, Any]:
    candidate = extract_python(output_text)
    entrypoint = str(case["metadata"]["official_entry_point"])
    program = "\n".join(
        (
            f"CANDIDATE_JSON = {json.dumps(json.dumps(candidate))}",
            f"SHARED_JSON = {json.dumps(json.dumps(shared_text(case)))}",
            f"ENTRYPOINT_JSON = {json.dumps(json.dumps(entrypoint))}",
            VISIBLE_PROGRAM,
        )
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
        return {
            "compiled": True,
            "doctest_attempted": 0,
            "doctest_failed": 0,
            "interface_ok": False,
            "passed": False,
            "reason": "timeout",
        }
    marker = "__IMPACTKV_VISIBLE__"
    line = next(
        (
            row
            for row in reversed(result.stdout.splitlines())
            if row.startswith(marker)
        ),
        None,
    )
    if line is None:
        return {
            "compiled": False,
            "doctest_attempted": 0,
            "doctest_failed": 0,
            "interface_ok": False,
            "passed": False,
            "reason": result.stderr[-1000:] or "missing result marker",
        }
    value = json.loads(line.removeprefix(marker))
    value["passed"] = bool(
        value["compiled"]
        and value["interface_ok"]
        and value["doctest_failed"] == 0
    )
    return value


def register(output: Path) -> dict[str, Any]:
    path = output / "V13_VISIBLE_GUARD_REGISTRATION.json"
    if path.exists():
        value = read_json(path)
        if value["inputs"]["v12_result_sha256"] != sha256_file(V12_RESULT):
            raise ValueError("registered V12 result changed")
        return value
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    value = {
        "date": "2026-07-27",
        "experiment": "V13 online-visible contract guarded V12 fallback",
        "registration_timing": (
            "rule frozen before running visible checks; underlying functional "
            "outcomes were already exposed"
        ),
        "policy": {
            "primary": "serve V12-generated candidate",
            "online_visible_checks": [
                "Python syntax",
                "exact required public function name",
                "all prompt-visible doctest examples",
            ],
            "fallback": (
                "if any visible check fails, discard the candidate and rerun "
                "the target dense; no hidden tests or evaluator truth"
            ),
            "prefetch": False,
        },
        "frozen_gates": {
            "accuracy_gain_vs_v12_pp_min": 1.0,
            "accuracy_vs_cacheblend_pp_min": 0.0,
            "fallback_rate_max": 0.05,
            "must_report_final_response_latency_not_only_ttft": True,
        },
        "latency_model": {
            "fallback_final_response": (
                "V12 full decode + visible check + Dense full decode"
            ),
            "ttft_proxy_only": (
                "V12 cache-ready TTFT + fallback_rate * Dense TTFT; excludes "
                "decode and verification and therefore is optimistic"
            ),
        },
        "inputs": {
            "old_result_sha256": sha256_file(OLD_RESULT),
            "v12_result_sha256": sha256_file(V12_RESULT),
            "v12_speed_sha256": sha256_file(V12_SPEED),
            "workload_sha256": sha256_file(WORKLOAD),
        },
        "protected": {
            "existing_preregistration_thresholds_modified": False,
            "old_dirty_checkout_modified": False,
            "paper_modified": False,
            "prefetch": False,
        },
        "scope": (
            "Retrospective development analysis on already exposed functional "
            "outcomes; not confirmatory evidence."
        ),
        "status": "RULE_FROZEN_BEFORE_VISIBLE_CHECK_EXECUTION",
    }
    write_json(path, value)
    return value


def analyze(output: Path) -> dict[str, Any]:
    registration = register(output)
    destination = output / "V13_VISIBLE_GUARD_RESULT.json"
    if destination.exists():
        return read_json(destination)
    workload = {
        str(row["case_id"]): row for row in read_json(WORKLOAD)["cases"]
    }
    source_rows = read_json(V12_RESULT)["rows"]
    by_arm = {
        arm: {
            str(row.get("original_case_id") or row["case_id"]): row
            for row in source_rows
            if row["arm"] == arm
        }
        for arm in (V12_ARM, DENSE_ARM, CACHEBLEND_ARM)
    }
    ids = sorted(by_arm[V12_ARM])
    checks = {}
    selected = {}
    for case_id in ids:
        check = visible_check(
            workload[case_id],
            by_arm[V12_ARM][case_id]["output_text"],
        )
        checks[case_id] = check
        selected[case_id] = (
            by_arm[V12_ARM][case_id]
            if check["passed"]
            else by_arm[DENSE_ARM][case_id]
        )
    fallback_ids = [case_id for case_id in ids if not checks[case_id]["passed"]]
    pass_rates = {
        "v12": statistics.mean(
            bool(by_arm[V12_ARM][case_id]["passed"]) for case_id in ids
        ),
        "dense": statistics.mean(
            bool(by_arm[DENSE_ARM][case_id]["passed"]) for case_id in ids
        ),
        "cacheblend": statistics.mean(
            bool(by_arm[CACHEBLEND_ARM][case_id]["passed"]) for case_id in ids
        ),
        "visible_guard": statistics.mean(
            bool(selected[case_id]["passed"]) for case_id in ids
        ),
    }
    transitions = {
        "v12_fail_to_guard_pass": sum(
            not by_arm[V12_ARM][case_id]["passed"]
            and selected[case_id]["passed"]
            for case_id in ids
        ),
        "v12_pass_to_guard_fail": sum(
            by_arm[V12_ARM][case_id]["passed"]
            and not selected[case_id]["passed"]
            for case_id in ids
        ),
    }
    speed = read_json(V12_SPEED)
    v12_ttft = float(
        speed["arms"][V12_ARM]["mean_ttft_ms"]
    )
    dense_ttft = float(speed["arms"]["dense"]["mean_ttft_ms"])
    fallback_rate = len(fallback_ids) / len(ids)
    optimistic_ttft_proxy = v12_ttft + fallback_rate * dense_ttft
    gates = registration["frozen_gates"]
    gain_pp = 100 * (pass_rates["visible_guard"] - pass_rates["v12"])
    cacheblend_pp = 100 * (
        pass_rates["visible_guard"] - pass_rates["cacheblend"]
    )
    verdict = {
        "accuracy_gain_passed": (
            gain_pp >= gates["accuracy_gain_vs_v12_pp_min"]
        ),
        "cacheblend_accuracy_passed": (
            cacheblend_pp >= gates["accuracy_vs_cacheblend_pp_min"]
        ),
        "fallback_rate_passed": (
            fallback_rate <= gates["fallback_rate_max"]
        ),
    }
    verdict["overall"] = all(verdict.values())
    value = {
        "checks": checks,
        "fallback_case_ids": fallback_ids,
        "fallback_rate": fallback_rate,
        "functional_pass_rates": pass_rates,
        "accuracy_gain_vs_v12_pp": gain_pp,
        "accuracy_vs_cacheblend_pp": cacheblend_pp,
        "transitions": transitions,
        "latency": {
            "dense_mean_ttft_ms": dense_ttft,
            "optimistic_ttft_proxy_ms": optimistic_ttft_proxy,
            "v12_mean_ttft_ms": v12_ttft,
            "warning": (
                "This proxy excludes full first decode, visible execution, "
                "and full fallback decode; it is not final-response latency."
            ),
        },
        "verdict": verdict,
        "status": "V13_VISIBLE_GUARD_COMPLETE",
    }
    write_json(destination, value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("register")
    sub.add_parser("analyze")
    args = parser.parse_args()
    output = args.output.resolve()
    value = register(output) if args.command == "register" else analyze(output)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
