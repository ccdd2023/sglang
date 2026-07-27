#!/usr/bin/env python3
"""Audit CacheBlend-vs-Dense flips and freeze the V15 repeat protocol.

The historical comparison is explicitly retrospective.  Only the five-start
repeat protocol emitted by this script is preregistered before new GPU work.
Both arms must come from the same CacheBlend engine, commit, dtype, prompts,
token IDs, and evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ARTIFACTS = Path("/home/gfy/CodeMAS_Project/kvflow-artifacts")
NATIVE = (
    ARTIFACTS
    / "impactkv_native_frontier_v3_20260720/runs/cacheblend/native/formal"
)
DENSE = NATIVE / "accuracy.dense.dense.scored.jsonl"
REUSE = NATIVE / "accuracy.reuse.recompute-0.05.scored.jsonl"
WORKLOAD = (
    ARTIFACTS
    / "impactkv_native_frontier_20260719/workload_v2/COMMON_WORKLOAD.json"
)
TRUTH = (
    ARTIFACTS
    / "impactkv_native_frontier_20260719/workload_v2/"
    "sealed_evaluator/COMMON_TRUTH.json"
)
DEFAULT_OUTPUT = (
    ARTIFACTS / "impactkv_v15_cacheblend_flip_audit_20260727"
)
BOOTSTRAP_SEED = 20260727
BOOTSTRAP_ITERATIONS = 10_000
EXPECTED_CASES = 225
REPEAT_STARTS = 5


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _index(rows: list[dict[str, Any]], arm: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        case_id = str(row["case_id"])
        if case_id in indexed:
            raise ValueError(f"{arm}: duplicate case_id {case_id}")
        indexed[case_id] = row
    return indexed


def validate_inputs(
    dense: dict[str, dict[str, Any]],
    reuse: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if len(dense) != EXPECTED_CASES or len(reuse) != EXPECTED_CASES:
        raise ValueError(
            f"expected {EXPECTED_CASES} cases per arm, "
            f"got dense={len(dense)} reuse={len(reuse)}"
        )
    if set(dense) != set(reuse):
        raise ValueError("Dense and CacheBlend case sets differ")

    paired_fields = (
        "case_id",
        "concurrency",
        "context_tokens",
        "dtype",
        "engine",
        "engine_commit",
        "method",
        "phase",
        "prompt_sha256",
        "request_topology",
        "split",
        "suite",
        "target_tokens",
        "token_ids_sha256",
    )
    for case_id in sorted(dense):
        left, right = dense[case_id], reuse[case_id]
        mismatches = [
            field
            for field in paired_fields
            if left.get(field) != right.get(field)
        ]
        if mismatches:
            raise ValueError(f"{case_id}: unpaired fields {mismatches}")
        if left.get("mode") != "dense" or left.get("config_id") != "dense":
            raise ValueError(f"{case_id}: malformed Dense row")
        if (
            right.get("mode") != "reuse"
            or right.get("config_id") != "recompute-0.05"
        ):
            raise ValueError(f"{case_id}: malformed CacheBlend row")
        if left.get("error") is not None or right.get("error") is not None:
            raise ValueError(f"{case_id}: engine error in formal input")
        for row in (left, right):
            output = str(row.get("metadata", {}).get("output_text") or "")
            output_hash = hashlib.sha256(output.encode("utf-8")).hexdigest()
            if not output or output_hash != row.get("output_sha256"):
                raise ValueError(f"{case_id}: output hash mismatch")
        if bool(left.get("physical_reuse_proven")):
            raise ValueError(f"{case_id}: Dense unexpectedly reports reuse")
        if not bool(right.get("physical_reuse_proven")):
            raise ValueError(f"{case_id}: CacheBlend reuse is not proven")

    first = dense[next(iter(sorted(dense)))]
    return {
        field: first.get(field)
        for field in (
            "concurrency",
            "dtype",
            "engine",
            "engine_commit",
            "method",
            "phase",
            "request_topology",
            "split",
        )
    }


def paired_bootstrap(
    deltas: list[float],
    *,
    seed: int = BOOTSTRAP_SEED,
    iterations: int = BOOTSTRAP_ITERATIONS,
) -> list[float]:
    rng = random.Random(seed)
    samples = sorted(
        statistics.mean(rng.choice(deltas) for _ in deltas)
        for _ in range(iterations)
    )
    return [samples[int(0.025 * iterations)], samples[int(0.975 * iterations)]]


def audit_rows(
    dense_rows: list[dict[str, Any]],
    reuse_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    dense = _index(dense_rows, "dense")
    reuse = _index(reuse_rows, "cacheblend")
    identity = validate_inputs(dense, reuse)

    transitions: Counter[str] = Counter()
    suite: dict[str, Counter[str]] = defaultdict(Counter)
    flips = []
    exact_output_matches = 0
    deltas = []
    for case_id in sorted(dense):
        left, right = dense[case_id], reuse[case_id]
        dense_pass = bool(left["passed"])
        reuse_pass = bool(right["passed"])
        transition = (
            "both_pass"
            if dense_pass and reuse_pass
            else "dense_only"
            if dense_pass
            else "reuse_only"
            if reuse_pass
            else "both_fail"
        )
        transitions[transition] += 1
        suite_name = str(left["suite"])
        suite[suite_name][transition] += 1
        dense_output = str(left["metadata"]["output_text"])
        reuse_output = str(right["metadata"]["output_text"])
        same_output = dense_output == reuse_output
        exact_output_matches += int(same_output)
        deltas.append(100.0 * (float(reuse_pass) - float(dense_pass)))
        if dense_pass != reuse_pass:
            flips.append(
                {
                    "case_id": case_id,
                    "context_tokens": int(left["context_tokens"]),
                    "dense_output_chars": len(dense_output),
                    "dense_output_sha256": left["output_sha256"],
                    "direction": transition,
                    "recompute_tokens": int(right["recomputed_tokens"]),
                    "reuse_output_chars": len(reuse_output),
                    "reuse_output_sha256": right["output_sha256"],
                    "reused_k_tokens": int(right["reused_k_tokens"]),
                    "reused_v_tokens": int(right["reused_v_tokens"]),
                    "suite": suite_name,
                }
            )

    dense_passed = transitions["both_pass"] + transitions["dense_only"]
    reuse_passed = transitions["both_pass"] + transitions["reuse_only"]
    dense_failed = EXPECTED_CASES - dense_passed
    return {
        "classification": "retrospective_same_engine_paired_audit",
        "identity": identity,
        "task_correctness": {
            "cacheblend_passed": reuse_passed,
            "cacheblend_rate": reuse_passed / EXPECTED_CASES,
            "dense_passed": dense_passed,
            "dense_rate": dense_passed / EXPECTED_CASES,
            "reuse_minus_dense_pp": statistics.mean(deltas),
            "reuse_minus_dense_pp_bootstrap95": paired_bootstrap(deltas),
        },
        "dense_preservation": {
            "damage_count": transitions["dense_only"],
            "damage_rate_given_dense_pass": (
                transitions["dense_only"] / dense_passed
            ),
            "rescue_count": transitions["reuse_only"],
            "rescue_rate_given_dense_fail": (
                transitions["reuse_only"] / dense_failed
            ),
        },
        "fidelity": {
            "different_outputs": EXPECTED_CASES - exact_output_matches,
            "exact_output_agreement": exact_output_matches / EXPECTED_CASES,
            "exact_output_matches": exact_output_matches,
            "logit_kl_available": False,
        },
        "flips": flips,
        "suite_transitions": {
            name: dict(counts) for name, counts in sorted(suite.items())
        },
        "transitions": dict(transitions),
    }


def build_repeat_registration(
    audit: dict[str, Any],
    *,
    output: Path,
) -> dict[str, Any]:
    flip_ids = [row["case_id"] for row in audit["flips"]]
    return {
        "date": "2026-07-27",
        "experiment": "V15 CacheBlend same-engine flip repeat audit",
        "registered_before_repeat_gpu": True,
        "historical_evidence_status": "retrospective_not_preregistered",
        "question": (
            "Is the historical CacheBlend +2-task point estimate a stable "
            "reuse effect, or an unstable finite-sample/numerical outcome?"
        ),
        "arms": {
            "dense": "CacheBlend native engine with reuse disabled",
            "reuse": "same engine with recompute_ratio=0.05",
        },
        "protocol": {
            "case_ids": flip_ids,
            "cases": len(flip_ids),
            "decode": {"max_new_tokens": "frozen_per_case", "temperature": 0},
            "fresh_model_starts_per_arm": REPEAT_STARTS,
            "paired_start_indices": True,
            "same_engine_commit": True,
            "same_model_dtype_prompt_and_token_ids": True,
            "score_with_frozen_sealed_evaluator": True,
        },
        "predeclared_analysis": {
            "cacheblend_advantage_robust": (
                "reuse-minus-dense accuracy is positive in at least 4/5 "
                "start pairs and the pooled paired-bootstrap 95% lower bound "
                "is greater than zero"
            ),
            "stable_flip": (
                "the historical pass/fail direction repeats in at least 4/5 "
                "paired starts"
            ),
            "unstable_flip": (
                "neither direction appears in at least 4/5 paired starts"
            ),
            "metrics": [
                "task pass rate",
                "Dense-pass to reuse-fail damage rate",
                "Dense-fail to reuse-pass rescue rate",
                "exact output agreement",
                "first divergent generated token",
                "Dense-reference top-1 agreement and KL when logits exist",
            ],
        },
        "inputs": {
            "dense_scored_path": str(DENSE),
            "dense_scored_sha256": sha256_file(DENSE),
            "historical_audit_sha256": hashlib.sha256(
                json.dumps(
                    audit,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest(),
            "reuse_scored_path": str(REUSE),
            "reuse_scored_sha256": sha256_file(REUSE),
            "truth_sha256": sha256_file(TRUTH),
            "workload_sha256": sha256_file(WORKLOAD),
        },
        "outputs": {
            "directory": str(output / "repeat_runs"),
            "overwrite_existing_runs": False,
        },
        "protected": {
            "existing_preregistration_thresholds_modified": False,
            "old_dirty_checkout_modified": False,
            "paper_modified": False,
            "prior_artifacts_modified": False,
        },
        "status": "REGISTERED_BEFORE_V15_REPEAT_GPU",
    }


def run(output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite audit: {output}")
    output.mkdir(parents=True)
    audit = audit_rows(read_jsonl(DENSE), read_jsonl(REUSE))
    write_json(output / "V15_BASELINE_AUDIT.json", audit)
    write_json(
        output / "V15_FLIP_CASES.json",
        {
            "case_ids": [row["case_id"] for row in audit["flips"]],
            "cases": len(audit["flips"]),
            "rows": audit["flips"],
        },
    )
    registration = build_repeat_registration(audit, output=output)
    write_json(output / "V15_REPEAT_REGISTRATION.json", registration)
    return {
        "audit": {
            key: audit[key]
            for key in (
                "task_correctness",
                "dense_preservation",
                "fidelity",
                "suite_transitions",
                "transitions",
            )
        },
        "flip_cases": len(audit["flips"]),
        "output": str(output),
        "repeat_status": registration["status"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.output.resolve()),
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
