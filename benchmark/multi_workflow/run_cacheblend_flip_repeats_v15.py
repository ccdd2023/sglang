#!/usr/bin/env python3
"""Run and score the preregistered V15 CacheBlend flip repetitions."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import random
import statistics
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from benchmark.multi_workflow.audit_cacheblend_dense_flips_v15 import (
    BOOTSTRAP_ITERATIONS,
    BOOTSTRAP_SEED,
    DEFAULT_OUTPUT,
    DENSE,
    REPEAT_STARTS,
    REUSE,
    TRUTH,
    WORKLOAD,
    paired_bootstrap,
    read_jsonl,
    sha256_file,
    write_json,
)
from benchmark.multi_workflow.run_v12_full225_accuracy import (
    _evaluate,
    extract_python,
)


CACHEBLEND_ROOT = Path(
    "/home/gfy/CodeMAS_Project/kvflow-reproductions/"
    "worktrees/cacheblend-qwen2"
)
CACHEBLEND_RUNNER = CACHEBLEND_ROOT / "example/repro_common.py"
CACHEBLEND_PYTHON = Path(
    "/home/gfy/.conda/envs/cacheblend-repro-20260719/bin/python"
)
MODEL = Path(
    "/home/gfy/.cache/huggingface/hub/"
    "models--Qwen--Qwen2.5-Coder-3B-Instruct/snapshots/"
    "488639f1ff808d1d3d0ba301aef8c11461451ec5"
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )


def engine_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=CACHEBLEND_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def flip_ids(output: Path) -> list[str]:
    registration = read_json(output / "V15_REPEAT_REGISTRATION.json")
    if not registration["registered_before_repeat_gpu"]:
        raise ValueError("V15 repeat protocol is not preregistered")
    case_ids = [str(value) for value in registration["protocol"]["case_ids"]]
    if len(case_ids) != 20 or len(set(case_ids)) != len(case_ids):
        raise ValueError("V15 requires exactly 20 unique historical flips")
    return case_ids


def raw_path(output: Path, start: int, arm: str) -> Path:
    return output / "repeat_runs" / f"start_{start}" / f"{arm}.raw.jsonl"


def scored_path(output: Path, start: int, arm: str) -> Path:
    return output / "repeat_runs" / f"start_{start}" / f"{arm}.scored.jsonl"


def build_command(
    *,
    output: Path,
    start: int,
    arm: str,
) -> list[str]:
    mode = "dense" if arm == "dense" else "reuse"
    command = [
        str(CACHEBLEND_PYTHON),
        str(CACHEBLEND_RUNNER),
        "--workload",
        str(WORKLOAD),
        "--metrics",
        str(raw_path(output, start, arm)),
        "--model",
        str(MODEL),
        "--mode",
        mode,
        "--phase",
        "accuracy",
        "--dtype",
        "native",
        "--split",
        "formal",
        "--limit",
        "0",
        "--recompute-ratio",
        "0.05",
        "--gpu-memory-utilization",
        "0.85",
        "--run-id",
        f"v15-cacheblend-flips-start{start}-{arm}",
    ]
    for case_id in flip_ids(output):
        command.extend(("--case-id", case_id))
    return command


def _historical_index(arm: str) -> dict[str, dict[str, Any]]:
    path = DENSE if arm == "dense" else REUSE
    return {str(row["case_id"]): row for row in read_jsonl(path)}


def validate_raw(
    rows: list[dict[str, Any]],
    *,
    arm: str,
    output: Path,
) -> dict[str, Any]:
    expected = flip_ids(output)
    observed = [str(row["case_id"]) for row in rows]
    if len(observed) != len(set(observed)) or set(observed) != set(expected):
        raise ValueError(f"{arm}: raw case coverage differs")
    reference = _historical_index(arm)
    expected_commit = engine_commit()
    for row in rows:
        case_id = str(row["case_id"])
        if row.get("error") is not None:
            raise ValueError(f"{arm}/{case_id}: {row['error']}")
        for field in (
            "context_tokens",
            "dtype",
            "engine",
            "method",
            "prompt_sha256",
            "request_topology",
            "suite",
            "target_tokens",
            "token_ids_sha256",
        ):
            if row.get(field) != reference[case_id].get(field):
                raise ValueError(f"{arm}/{case_id}: changed {field}")
        if row.get("engine_commit") != expected_commit:
            raise ValueError(f"{arm}/{case_id}: engine commit changed")
        text = str(row.get("metadata", {}).get("output_text") or "")
        if hashlib.sha256(text.encode()).hexdigest() != row["output_sha256"]:
            raise ValueError(f"{arm}/{case_id}: output hash mismatch")
        reused_k = int(row.get("reused_k_tokens") or 0)
        reused_v = int(row.get("reused_v_tokens") or 0)
        blend_layers = int(row.get("blend_layers_executed") or 0)
        if arm == "reuse":
            if reused_k <= 0 or reused_v != reused_k or blend_layers <= 0:
                raise ValueError(f"{arm}/{case_id}: physical reuse not observed")
        elif reused_k or reused_v or blend_layers:
            raise ValueError(f"{arm}/{case_id}: Dense unexpectedly reports reuse")
    return {
        "arm": arm,
        "cases": len(rows),
        "engine_commit": expected_commit,
        "raw_sha256": None,
        "status": "RAW_VALIDATED",
    }


def run_one(output: Path, start: int, arm: str) -> dict[str, Any]:
    if start < 0 or start >= REPEAT_STARTS:
        raise ValueError(f"start must be in [0, {REPEAT_STARTS})")
    destination = raw_path(output, start, arm)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    log_path = destination.with_suffix(".log")
    env = os.environ.copy()
    env.update(
        {
            "HF_HUB_OFFLINE": "1",
            "KVFLOW_ENGINE_COMMIT": engine_commit(),
            "PYTHONNOUSERSITE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    with log_path.open("x", encoding="utf-8") as stream:
        completed = subprocess.run(
            build_command(output=output, start=start, arm=arm),
            cwd=CACHEBLEND_ROOT,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{arm}/start{start} exited {completed.returncode}; inspect {log_path}"
        )
    rows = read_jsonl(destination)
    value = validate_raw(rows, arm=arm, output=output)
    value["raw_sha256"] = sha256_file(destination)
    write_json(destination.with_suffix(".complete.json"), value)
    return value


def score_one(output: Path, start: int, arm: str) -> dict[str, Any]:
    source = raw_path(output, start, arm)
    destination = scored_path(output, start, arm)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    rows = read_jsonl(source)
    validate_raw(rows, arm=arm, output=output)
    workload = {
        str(row["case_id"]): row for row in read_json(WORKLOAD)["cases"]
    }
    truth = {
        str(row["case_id"]): row for row in read_json(TRUTH)["cases"]
    }
    scored = []
    for row in rows:
        case_id = str(row["case_id"])
        candidate = extract_python(str(row["metadata"]["output_text"]))
        passed, error = _evaluate(
            workload[case_id],
            truth[case_id],
            candidate,
        )
        try:
            ast.parse(candidate)
            compiled = True
        except SyntaxError:
            compiled = False
        scored.append(
            {
                **row,
                "compiled": compiled,
                "evaluator_error": error,
                "passed": passed,
            }
        )
    append_jsonl(destination, scored)
    value = {
        "arm": arm,
        "cases": len(scored),
        "passed": sum(bool(row["passed"]) for row in scored),
        "scored_sha256": sha256_file(destination),
        "start": start,
        "status": "SCORED",
    }
    write_json(destination.with_suffix(".complete.json"), value)
    return value


def _first_divergent_token(left: str, right: str) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    left_ids = tokenizer(left, add_special_tokens=False)["input_ids"]
    right_ids = tokenizer(right, add_special_tokens=False)["input_ids"]
    limit = min(len(left_ids), len(right_ids))
    index = next(
        (position for position in range(limit) if left_ids[position] != right_ids[position]),
        limit if len(left_ids) != len(right_ids) else None,
    )
    return {
        "dense_tokens": len(left_ids),
        "first_divergent_token": index,
        "reuse_tokens": len(right_ids),
    }


def summarize(output: Path) -> dict[str, Any]:
    historical = {
        row["case_id"]: row["direction"]
        for row in read_json(output / "V15_BASELINE_AUDIT.json")["flips"]
    }
    start_summaries = []
    stability: dict[str, Counter[str]] = {
        case_id: Counter() for case_id in flip_ids(output)
    }
    pooled_deltas = []
    divergent = []
    exact_matches = 0
    for start in range(REPEAT_STARTS):
        dense = {
            str(row["case_id"]): row
            for row in read_jsonl(scored_path(output, start, "dense"))
        }
        reuse = {
            str(row["case_id"]): row
            for row in read_jsonl(scored_path(output, start, "reuse"))
        }
        transitions: Counter[str] = Counter()
        for case_id in flip_ids(output):
            left, right = dense[case_id], reuse[case_id]
            dense_pass, reuse_pass = bool(left["passed"]), bool(right["passed"])
            direction = (
                "both_pass"
                if dense_pass and reuse_pass
                else "dense_only"
                if dense_pass
                else "reuse_only"
                if reuse_pass
                else "both_fail"
            )
            transitions[direction] += 1
            stability[case_id][direction] += 1
            pooled_deltas.append(
                100.0 * (float(reuse_pass) - float(dense_pass))
            )
            left_text = str(left["metadata"]["output_text"])
            right_text = str(right["metadata"]["output_text"])
            same = left_text == right_text
            exact_matches += int(same)
            if not same:
                divergent.append(
                    {
                        "case_id": case_id,
                        "start": start,
                        **_first_divergent_token(left_text, right_text),
                    }
                )
        start_summaries.append(
            {
                "dense_passed": sum(row["passed"] for row in dense.values()),
                "reuse_minus_dense_tasks": (
                    sum(row["passed"] for row in reuse.values())
                    - sum(row["passed"] for row in dense.values())
                ),
                "reuse_passed": sum(row["passed"] for row in reuse.values()),
                "start": start,
                "transitions": dict(transitions),
            }
        )

    flip_stability = []
    for case_id in flip_ids(output):
        counts = stability[case_id]
        historical_direction = historical[case_id]
        historical_repeats = counts[historical_direction]
        reverse_direction = (
            "reuse_only"
            if historical_direction == "dense_only"
            else "dense_only"
        )
        classification = (
            "historical_direction_stable"
            if historical_repeats >= 4
            else "reverse_direction_stable"
            if counts[reverse_direction] >= 4
            else "unstable_or_converged"
        )
        flip_stability.append(
            {
                "case_id": case_id,
                "classification": classification,
                "historical_direction": historical_direction,
                "repeat_counts": dict(counts),
            }
        )

    positive_starts = sum(
        row["reuse_minus_dense_tasks"] > 0 for row in start_summaries
    )
    ci = paired_bootstrap(
        pooled_deltas,
        seed=BOOTSTRAP_SEED + 15,
        iterations=BOOTSTRAP_ITERATIONS,
    )
    result = {
        "classification": "preregistered_five_start_repeat",
        "dense_preservation": {
            "damage_count": sum(
                row["transitions"].get("dense_only", 0)
                for row in start_summaries
            ),
            "rescue_count": sum(
                row["transitions"].get("reuse_only", 0)
                for row in start_summaries
            ),
        },
        "fidelity": {
            "different_output_pairs": len(divergent),
            "exact_output_agreement": exact_matches
            / (len(flip_ids(output)) * REPEAT_STARTS),
            "first_divergences": divergent,
            "logit_kl_available": False,
        },
        "flip_stability": flip_stability,
        "start_summaries": start_summaries,
        "task_correctness": {
            "positive_start_pairs": positive_starts,
            "pooled_reuse_minus_dense_pp": statistics.mean(pooled_deltas),
            "pooled_reuse_minus_dense_pp_bootstrap95": ci,
        },
        "verdict": {
            "cacheblend_advantage_robust": (
                positive_starts >= 4 and ci[0] > 0
            ),
            "historical_plus_two_requires_reinterpretation": not (
                positive_starts >= 4 and ci[0] > 0
            ),
        },
        "status": "V15_REPEAT_COMPLETE",
    }
    write_json(output / "V15_REPEAT_RESULT.json", result)
    return result


def campaign(output: Path) -> dict[str, Any]:
    completed = []
    for start in range(REPEAT_STARTS):
        order = ("dense", "reuse") if start % 2 == 0 else ("reuse", "dense")
        for arm in order:
            if not raw_path(output, start, arm).exists():
                completed.append(run_one(output, start, arm))
            if not scored_path(output, start, arm).exists():
                completed.append(score_one(output, start, arm))
    result = summarize(output)
    return {
        "actions_completed": len(completed),
        "status": result["status"],
        "verdict": result["verdict"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run-one")
    run_parser.add_argument("--start", type=int, required=True)
    run_parser.add_argument("--arm", choices=("dense", "reuse"), required=True)
    score_parser = sub.add_parser("score-one")
    score_parser.add_argument("--start", type=int, required=True)
    score_parser.add_argument("--arm", choices=("dense", "reuse"), required=True)
    sub.add_parser("summarize")
    sub.add_parser("campaign")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.command == "run-one":
        value = run_one(output, args.start, args.arm)
    elif args.command == "score-one":
        value = score_one(output, args.start, args.arm)
    elif args.command == "summarize":
        value = summarize(output)
    else:
        value = campaign(output)
    print(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
