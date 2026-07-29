#!/usr/bin/env python3
"""Register and validate the V40/KVCOMM/CacheBlend fair comparison.

The comparison has two intentionally separate layers:

* ``controlled`` requires identical model input token IDs.  Results may be
  ranked directly for quality and latency.
* ``native`` preserves each upstream request topology.  Reuse is paired only
  with Dense from the same engine; absolute cross-engine TTFT is descriptive.

This module contains no GPU execution and never enables prefetch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path("/home/gfy/CodeMAS_Project")
ARTIFACT_ROOT = (
    ROOT / "kvflow-artifacts/impactkv_v40_sota_fair_v2_20260729"
)
SOURCE_REGISTRATION = (
    ROOT
    / "kvflow-artifacts/impactkv_three_method_coding_benchmark_20260728"
    / "REGISTRATION.json"
)
SWE_POPULATION = (
    ROOT
    / "sglang-kvflow/results/repo_level_datasets/"
    "swe_verified_500_instances.json"
)
STATIC_SOURCE = (
    ROOT
    / "kvflow-reproductions/qcfuse-official/data/"
    "coding_200_qwen3_8b_5k"
)
MODEL_SNAPSHOT = (
    Path("/home/gfy/.cache/huggingface/hub")
    / "models--Qwen--Qwen2.5-Coder-3B-Instruct/snapshots"
    / "488639f1ff808d1d3d0ba301aef8c11461451ec5"
)
MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"
SELECTION_SALT = "impactkv-v40-sota-fair-v2-holdout-20260729"
METHODS = ("v40", "cacheblend", "kvcomm")
REQUIRED_RECORD_FIELDS = {
    "case_id",
    "config_id",
    "engine",
    "error",
    "method",
    "mode",
    "prompt_sha256",
    "token_ids_sha256",
    "ttft_ms",
}


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def _repository(instance_id: str) -> str:
    if "__" not in instance_id:
        raise ValueError(f"not a SWE-bench instance id: {instance_id}")
    return instance_id.split("__", 1)[0]


def select_hash_holdout(
    population: Sequence[dict[str, Any]],
    *,
    excluded_ids: Iterable[str],
    size: int,
    per_repository_cap: int = 2,
    salt: str = SELECTION_SALT,
) -> list[dict[str, Any]]:
    """Select an outcome-independent, repository-capped SWE-bench holdout."""

    excluded = set(excluded_ids)
    candidates = []
    seen: set[str] = set()
    for row in population:
        instance_id = str(row["instance_id"])
        if instance_id in seen:
            raise ValueError(f"duplicate SWE-bench instance: {instance_id}")
        seen.add(instance_id)
        if instance_id in excluded:
            continue
        candidates.append(row)
    ranked = sorted(
        candidates,
        key=lambda row: (
            hashlib.sha256(
                f"{salt}:{row['instance_id']}".encode("utf-8")
            ).hexdigest(),
            str(row["instance_id"]),
        ),
    )
    selected: list[dict[str, Any]] = []
    repository_counts: dict[str, int] = {}
    for row in ranked:
        repository = _repository(str(row["instance_id"]))
        if repository_counts.get(repository, 0) >= per_repository_cap:
            continue
        selected.append(row)
        repository_counts[repository] = (
            repository_counts.get(repository, 0) + 1
        )
        if len(selected) == size:
            break
    if len(selected) != size:
        raise ValueError(
            f"only {len(selected)} holdout tasks satisfy the repository cap"
        )
    return selected


def _static_case_ids(path: Path) -> list[str]:
    rows = _read_jsonl(path)
    result = [
        str(row.get("_id", f"{path.stem}-{index}"))
        for index, row in enumerate(rows)
    ]
    if len(result) != len(set(result)):
        raise ValueError(f"{path} contains duplicate case IDs")
    return result


def _model_fingerprint(snapshot: Path) -> dict[str, Any]:
    files = {}
    for name in (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ):
        path = snapshot / name
        if path.exists():
            files[name] = file_sha256(path)
    if "tokenizer.json" not in files or "config.json" not in files:
        raise FileNotFoundError(
            f"incomplete local model snapshot at {snapshot}"
        )
    return {
        "model_id": MODEL_ID,
        "snapshot": str(snapshot),
        "snapshot_revision": snapshot.name,
        "file_sha256": files,
    }


def build_registration(
    *,
    source_registration_path: Path = SOURCE_REGISTRATION,
    population_path: Path = SWE_POPULATION,
    static_source: Path = STATIC_SOURCE,
    model_snapshot: Path = MODEL_SNAPSHOT,
    holdout_size: int = 20,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source_registration = json.loads(
        source_registration_path.read_text(encoding="utf-8")
    )
    population = json.loads(population_path.read_text(encoding="utf-8"))
    mechanism = source_registration["datasets"][
        "swebench_verified_mechanism"
    ]["tasks"]
    mechanism_ids = [str(row["instance_id"]) for row in mechanism]
    holdout = select_hash_holdout(
        population,
        excluded_ids=mechanism_ids,
        size=holdout_size,
    )
    holdout_ids = [str(row["instance_id"]) for row in holdout]
    repobench = static_source / "repobench-p.jsonl"
    lcc = static_source / "lcc.jsonl"
    registration = {
        "schema_version": 2,
        "registration_id": "impactkv-v40-sota-fair-v2-20260729",
        "status": "REGISTERED_BEFORE_V2_TREATMENT",
        "objective": (
            "Test whether coding-aware V40 improves the official coding-task "
            "accuracy/TTFT tradeoff over KVCOMM and CacheBlend without "
            "prefetch."
        ),
        "comparison_layers": {
            "controlled": {
                "rankable": True,
                "require_identical_token_ids": True,
                "role": (
                    "same-model frozen-request TTFT and shared-agent "
                    "SWE-bench quality"
                ),
            },
            "native": {
                "rankable": False,
                "require_identical_token_ids": False,
                "role": (
                    "reuse versus the matching Dense arm inside each "
                    "upstream engine"
                ),
            },
        },
        "scope": {
            "methods": list(METHODS),
            "normalization": "one matching Dense arm per native engine",
            "prefetch": False,
            "excluded_headline_methods": [
                "tail",
                "general",
                "QCFuse",
                "FUSE-RAG",
            ],
        },
        "model": _model_fingerprint(model_snapshot),
        "fixed_protocol": {
            "temperature": 0.0,
            "concurrency": 1,
            "max_new_tokens_static": 64,
            "agent_step_limit": 20,
            "agent_prompt_token_limit": 28000,
            "agent_wall_time_limit_seconds": 1200,
            "tool_observation_chars": 6000,
            "assistant_reasoning_chars": 3000,
        },
        "parameter_sweeps": {
            "v40": {
                "copy_cap": [2048, 4096, 8192],
                "minimum_tokens": 128,
            },
            "cacheblend": {"recompute_ratio": [0.25, 0.5, 0.75]},
            "kvcomm": {
                "threshold": [0.3, 0.5, 0.7],
                "max_anchor_num": 20,
                "window_size": 5,
            },
        },
        "operating_point_rule": (
            "maximize official resolved count on the 12-task development "
            "cohort; break ties by N=4 build-amortized speedup"
        ),
        "datasets": {
            "swebench_verified_development": {
                "task_count": len(mechanism_ids),
                "task_ids": mechanism_ids,
                "selection": (
                    "existing outcome-free V40-opportunity mechanism cohort"
                ),
                "population_claim": False,
            },
            "swebench_verified_holdout": {
                "task_count": len(holdout_ids),
                "task_ids": holdout_ids,
                "selection": (
                    "salted hash over the remaining official population, "
                    "at most two tasks per repository"
                ),
                "selection_salt": SELECTION_SALT,
                "uses_method_output": False,
                "population_claim": True,
            },
            "repobench_p": {
                "task_count": len(_static_case_ids(repobench)),
                "source": str(repobench),
                "source_sha256": file_sha256(repobench),
            },
            "lcc": {
                "task_count": len(_static_case_ids(lcc)),
                "source": str(lcc),
                "source_sha256": file_sha256(lcc),
            },
        },
        "metrics": {
            "primary_accuracy": "official SWE-bench resolved/pass@1",
            "primary_speed": (
                "paired geometric-mean cache-ready TTFT speedup on "
                "identical frozen token IDs"
            ),
            "secondary": [
                "N=4 source-build-amortized TTFT speedup",
                "p50 and p95 TTFT",
                "end-to-end task wall time",
                "Dense-fail/reuse-pass and Dense-pass/reuse-fail transitions",
                "code similarity on static controls",
            ],
        },
        "pins": {
            "coding_branch_base": "13671eb70",
            "cacheblend": "a798011319c1bdb59ff6b8a9da06fa5028a3292b",
            "kvcomm": "3bf7410ca3fd63930241f9332e0c396c91fc05ed",
        },
        "protected": {
            "old_dirty_checkouts_modified": False,
            "paper_modified": False,
            "old_preregistration_gates_modified": False,
        },
        "provenance": {
            "source_registration": str(source_registration_path),
            "source_registration_sha256": file_sha256(
                source_registration_path
            ),
            "population": str(population_path),
            "population_sha256": file_sha256(population_path),
        },
    }
    return registration, holdout


def materialize_registration(
    output: Path = ARTIFACT_ROOT,
    **kwargs: Any,
) -> dict[str, Any]:
    registration, holdout = build_registration(**kwargs)
    registration_path = output / "COMPARISON_REGISTRATION.json"
    if registration_path.exists():
        existing = json.loads(registration_path.read_text(encoding="utf-8"))
        if existing != registration:
            raise FileExistsError(
                f"refusing to replace a different registration: "
                f"{registration_path}"
            )
    _write_json(registration_path, registration)
    _write_json(output / "swebench_verified/HOLDOUT.json", holdout)
    dataset = output / "swebench_verified/minisweagent_dataset/test.jsonl"
    dataset.parent.mkdir(parents=True, exist_ok=True)
    serialized = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in holdout
    )
    if dataset.exists() and dataset.read_text(encoding="utf-8") != serialized:
        raise FileExistsError(f"refusing to replace dataset: {dataset}")
    dataset.write_text(serialized, encoding="utf-8")
    return registration


def validate_workload(
    workload: dict[str, Any],
    *,
    expected_case_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    cases = workload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("workload must contain non-empty cases")
    case_ids: list[str] = []
    for case in cases:
        case_id = str(case.get("case_id", ""))
        if not case_id:
            raise ValueError("workload case has no case_id")
        case_ids.append(case_id)
        messages = case.get("messages")
        segments = case.get("segments")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"{case_id}: messages are missing")
        if not isinstance(segments, list) or not segments:
            raise ValueError(f"{case_id}: segments are missing")
        expected_hash = canonical_sha256(messages)
        if case.get("prompt_sha256") != expected_hash:
            raise ValueError(f"{case_id}: message manifest hash mismatch")
        user_text = "".join(
            str(message.get("content", ""))
            for message in messages
            if message.get("role") == "user"
        )
        cursor = 0
        for segment in segments:
            text = str(segment.get("text", ""))
            if not text:
                raise ValueError(f"{case_id}: empty segment")
            position = user_text.find(text, cursor)
            if position < 0:
                raise ValueError(
                    f"{case_id}: segment order/text differs from user prompt"
                )
            cursor = position + len(text)
        if int(case.get("max_new_tokens", 0)) <= 0:
            raise ValueError(f"{case_id}: invalid max_new_tokens")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("workload contains duplicate case IDs")
    if expected_case_ids is not None and case_ids != list(expected_case_ids):
        raise ValueError("workload case order differs from frozen registration")
    return {
        "cases": len(cases),
        "dataset": workload.get("dataset"),
        "message_manifest_sha256": canonical_sha256(
            [
                {
                    "case_id": case["case_id"],
                    "prompt_sha256": case["prompt_sha256"],
                }
                for case in cases
            ]
        ),
    }


def _target_records(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in records
        if not row.get("metadata", {}).get("warmup")
        and not row.get("metadata", {}).get("source_observation")
    ]


def validate_ledger(
    workload: dict[str, Any],
    records: Sequence[dict[str, Any]],
    *,
    expected_method: str | None = None,
    expected_mode: str | None = None,
) -> dict[str, Any]:
    validate_workload(workload)
    cases = {str(case["case_id"]): case for case in workload["cases"]}
    targets = _target_records(records)
    by_case: dict[str, list[dict[str, Any]]] = {}
    for row in targets:
        missing = REQUIRED_RECORD_FIELDS.difference(row)
        if missing:
            raise ValueError(f"ledger row is missing fields: {sorted(missing)}")
        if expected_method is not None and row["method"] != expected_method:
            raise ValueError("ledger method differs from requested method")
        if expected_mode is not None and row["mode"] != expected_mode:
            raise ValueError("ledger mode differs from requested mode")
        case_id = str(row["case_id"])
        if case_id not in cases:
            raise ValueError(f"unknown ledger case: {case_id}")
        if row["prompt_sha256"] != cases[case_id]["prompt_sha256"]:
            raise ValueError(f"{case_id}: ledger prompt manifest mismatch")
        if row.get("error"):
            raise ValueError(f"{case_id}: ledger error: {row['error']}")
        if not isinstance(row.get("ttft_ms"), (int, float)):
            raise ValueError(f"{case_id}: missing TTFT")
        if float(row["ttft_ms"]) <= 0:
            raise ValueError(f"{case_id}: non-positive TTFT")
        if row.get("token_ids_sha256") in {None, "", "error"}:
            raise ValueError(f"{case_id}: missing token IDs hash")
        if row["mode"] == "reuse":
            physical = int(row.get("reused_k_tokens", 0)) + int(
                row.get("reused_v_tokens", 0)
            )
            if physical <= 0 and not row.get("fallback_reason"):
                raise ValueError(
                    f"{case_id}: reuse has neither physical tokens nor fallback"
                )
        by_case.setdefault(case_id, []).append(row)
    missing_cases = set(cases).difference(by_case)
    extra_repeats = {
        case_id: len(rows) for case_id, rows in by_case.items() if len(rows) > 1
    }
    if missing_cases:
        raise ValueError(f"ledger is missing cases: {sorted(missing_cases)}")
    return {
        "cases": len(by_case),
        "records": len(targets),
        "repeated_cases": extra_repeats,
        "physical_reuse_records": sum(
            int(row.get("reused_k_tokens", 0))
            + int(row.get("reused_v_tokens", 0))
            > 0
            for row in targets
        ),
        "fallback_records": sum(bool(row.get("fallback_reason")) for row in targets),
    }


def token_identity_audit(
    ledgers: dict[str, Sequence[dict[str, Any]]],
) -> dict[str, Any]:
    """Determine whether method ledgers are eligible for direct comparison."""

    hashes: dict[str, dict[str, str]] = {}
    for method, records in ledgers.items():
        for row in _target_records(records):
            case_id = str(row["case_id"])
            value = str(row.get("token_ids_sha256", ""))
            if not value or value == "error":
                raise ValueError(f"{method}/{case_id}: invalid token hash")
            hashes.setdefault(case_id, {})[method] = value
    expected_methods = set(ledgers)
    missing = {
        case_id: sorted(expected_methods.difference(values))
        for case_id, values in hashes.items()
        if set(values) != expected_methods
    }
    mismatched = {
        case_id: values
        for case_id, values in hashes.items()
        if len(set(values.values())) > 1
    }
    return {
        "controlled_rankable": not missing and not mismatched,
        "cases": len(hashes),
        "missing_methods": missing,
        "token_hash_mismatches": mismatched,
        "classification": (
            "controlled" if not missing and not mismatched else "native_only"
        ),
    }


def choose_operating_point(
    summaries: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Choose accuracy first, then N=4 speed, with stable config-id ties."""

    if not summaries:
        raise ValueError("no operating points")
    for row in summaries:
        if "resolved" not in row or "n4_speedup" not in row:
            raise ValueError("operating point lacks resolved or n4_speedup")
    return max(
        summaries,
        key=lambda row: (
            int(row["resolved"]),
            float(row["n4_speedup"]),
            str(row["config_id"]),
        ),
    )


def discordant_task_ids(
    outcomes: dict[str, dict[str, bool]],
) -> list[str]:
    """Return tasks requiring two prespecified stability repeats."""

    return sorted(
        task_id
        for task_id, by_arm in outcomes.items()
        if len(set(bool(value) for value in by_arm.values())) > 1
    )


def paired_geometric_speedup(
    dense: Sequence[float], reuse: Sequence[float]
) -> float:
    if len(dense) != len(reuse) or not dense:
        raise ValueError("paired latency vectors must have equal non-zero length")
    ratios = []
    for dense_value, reuse_value in zip(dense, reuse):
        if dense_value <= 0 or reuse_value <= 0:
            raise ValueError("latencies must be positive")
        ratios.append(float(dense_value) / float(reuse_value))
    return statistics.geometric_mean(ratios)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    register_parser = subparsers.add_parser("register")
    register_parser.add_argument("--output", type=Path, default=ARTIFACT_ROOT)
    register_parser.add_argument("--holdout-size", type=int, default=20)
    workload_parser = subparsers.add_parser("validate-workload")
    workload_parser.add_argument("path", type=Path)
    ledger_parser = subparsers.add_parser("validate-ledger")
    ledger_parser.add_argument("--workload", type=Path, required=True)
    ledger_parser.add_argument("--ledger", type=Path, required=True)
    ledger_parser.add_argument("--method")
    ledger_parser.add_argument("--mode")
    args = parser.parse_args()
    if args.command == "register":
        value = materialize_registration(
            output=args.output,
            holdout_size=args.holdout_size,
        )
    elif args.command == "validate-workload":
        value = validate_workload(
            json.loads(args.path.read_text(encoding="utf-8"))
        )
    else:
        value = validate_ledger(
            json.loads(args.workload.read_text(encoding="utf-8")),
            _read_jsonl(args.ledger),
            expected_method=args.method,
            expected_mode=args.mode,
        )
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
