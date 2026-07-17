#!/usr/bin/env python3
"""Build and gate the formal 4,960-row V11 development causal atlas."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from benchmark.multi_workflow.analyze_sessiongraph_atlas import (
    DISTURBANCES,
    DOSES,
    Observation,
    analyze,
)
from benchmark.multi_workflow.sessiongraph_v11 import read_jsonl, write_jsonl


MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
REMAINING_DISTURBANCES = {
    "position_only",
    "module_reorder",
    "same_task",
    "cross_task",
}
ROLE_SPEC = {
    "negative_controls": ({"identity", "change_after"}, 1280),
    "upstream": ({"upstream_edit"}, 640),
    "semantic_prefix": ({"semantic_prefix"}, 640),
    "remaining_base": (REMAINING_DISTURBANCES, 2220),
    "remaining_delta": (REMAINING_DISTURBANCES, 180),
}
REQUIRED_OBSERVATION_FIELDS = {
    "session_id",
    "module_id",
    "module_type",
    "cache_scope",
    "disturbance",
    "recompute_fraction",
    "token_count",
    "position_norm",
    "rope_delta",
    "prefix_changed_tokens",
    "graph_distance",
    "k_deviation",
    "v_deviation",
    "attention_mass",
    "attention_mass_measured",
    "teacher_logit_js",
    "teacher_top1_changed",
    "causal_splice_logit_js",
    "lookup_ms",
    "source_tokens",
    "target_tokens",
    "source_prompt_hash",
    "target_prompt_hash",
    "measurement_model",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _key(row: Mapping[str, Any]) -> tuple[str, str, str, float]:
    return (
        str(row["session_id"]),
        str(row["module_id"]),
        str(row["disturbance"]),
        float(row["recompute_fraction"]),
    )


def _executor_summary(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("passed") is not True or value.get("errors"):
        raise ValueError(f"executor summary is not clean: {path}")
    if value.get("model") != MODEL:
        raise ValueError(f"executor model mismatch: {path}")
    return value


def aggregate(
    *,
    design_path: Path,
    role_paths: Mapping[str, Path],
    summary_paths: Sequence[Path],
    executor_amendment_path: Path,
    forbidden_paths: Sequence[Path],
    aggregate_output: Path,
    manifest_output: Path,
    gate_output: Path,
    bootstrap: int,
) -> dict[str, Any]:
    inputs = [design_path, *role_paths.values(), *summary_paths, executor_amendment_path]
    forbidden = {
        ("path", str(path.resolve())) for path in forbidden_paths if path.exists()
    } | {
        ("sha256", _sha(path)) for path in forbidden_paths if path.exists()
    }
    for path in inputs:
        if ("path", str(path.resolve())) in forbidden or ("sha256", _sha(path)) in forbidden:
            raise ValueError(f"forbidden unchunked artifact selected: {path}")

    amendment = json.loads(executor_amendment_path.read_text(encoding="utf-8"))
    expected_amendment = {
        "accepted": True,
        "attention_implementation": "sdpa",
        "dtype": "bfloat16",
        "splice_suffix_chunk_size": 512,
        "thresholds_changed": False,
    }
    mismatched_amendment = {
        key: (amendment.get(key), expected)
        for key, expected in expected_amendment.items()
        if amendment.get(key) != expected
    }
    if mismatched_amendment:
        raise ValueError(f"executor amendment mismatch: {mismatched_amendment}")
    summaries = [_executor_summary(path) for path in summary_paths]
    delta_summary = summaries[-1]
    for key, expected in (
        ("dtype", "bfloat16"),
        ("attention_implementation", "sdpa"),
        ("splice_suffix_chunk_size", 512),
    ):
        if delta_summary.get(key) != expected:
            raise ValueError(f"delta executor mismatch for {key}")

    role_rows: dict[str, list[dict[str, Any]]] = {}
    for role, (allowed_disturbances, expected_rows) in ROLE_SPEC.items():
        rows = read_jsonl(role_paths[role])
        observed_disturbances = {str(row["disturbance"]) for row in rows}
        if len(rows) != expected_rows:
            raise ValueError(f"{role} rows {len(rows)} != {expected_rows}")
        if not observed_disturbances <= allowed_disturbances:
            raise ValueError(
                f"{role} has unexpected disturbances: "
                f"{sorted(observed_disturbances - allowed_disturbances)}"
            )
        role_rows[role] = rows

    rows = [
        row
        for role in ROLE_SPEC
        for row in role_rows[role]
    ]
    violations: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        missing = sorted(REQUIRED_OBSERVATION_FIELDS - set(row))
        if missing:
            violations.append(
                {"kind": "observation_schema", "row": index, "missing": missing}
            )
        if row.get("status", "ok") != "ok":
            violations.append({"kind": "row_status", "row": index})
        if row.get("cohort") != "development":
            violations.append({"kind": "cohort", "row": index})
        if row.get("measurement_model") != MODEL:
            violations.append({"kind": "measurement_model", "row": index})
        if row.get("attention_mass") is not None or row.get(
            "attention_mass_measured"
        ) is not False:
            violations.append({"kind": "attention_proxy", "row": index})

    design = [
        row
        for row in read_jsonl(design_path)
        if row.get("cohort") == "development"
    ]
    design_keys = {_key(row) for row in design}
    row_keys = [_key(row) for row in rows]
    counts = Counter(row_keys)
    duplicates = sorted(key for key, count in counts.items() if count != 1)
    missing_keys = sorted(design_keys - set(row_keys))
    extra_keys = sorted(set(row_keys) - design_keys)
    if len(design) != 4960 or len(design_keys) != 4960:
        violations.append(
            {
                "kind": "design_coverage",
                "rows": len(design),
                "unique_keys": len(design_keys),
            }
        )
    if duplicates:
        violations.append({"kind": "duplicate_design_keys", "count": len(duplicates)})
    if missing_keys:
        violations.append({"kind": "missing_design_keys", "count": len(missing_keys)})
    if extra_keys:
        violations.append({"kind": "extra_design_keys", "count": len(extra_keys)})

    cells: dict[tuple[str, str, str], set[float]] = defaultdict(set)
    for row in rows:
        cells[_key(row)[:3]].add(float(row["recompute_fraction"]))
    incomplete = sorted(key for key, doses in cells.items() if doses != DOSES)
    if incomplete:
        violations.append({"kind": "incomplete_doses", "count": len(incomplete)})
    sessions = {str(row["session_id"]) for row in rows}
    disturbances = {str(row["disturbance"]) for row in rows}
    if len(rows) != 4960 or len(sessions) != 32 or disturbances != DISTURBANCES:
        violations.append(
            {
                "kind": "formal_shape",
                "rows": len(rows),
                "sessions": len(sessions),
                "disturbances": sorted(disturbances),
            }
        )
    if violations:
        statistical = {
            "passed": False,
            "status": "INVALID_ARTIFACT",
            "reasons": ["formal artifact validation failed"],
        }
    else:
        statistical = analyze(
            [Observation.from_row(row) for row in rows],
            iterations=bootstrap,
        )

    manifest = {
        "formal_rows": len(rows),
        "formal_design_sha256": _sha(design_path),
        "inputs": {
            role: {
                "path": str(path),
                "sha256": _sha(path),
                "rows": len(role_rows[role]),
            }
            for role, path in role_paths.items()
        },
        "executor_summaries": [
            {"path": str(path), "sha256": _sha(path)} for path in summary_paths
        ],
        "executor_amendment": {
            "path": str(executor_amendment_path),
            "sha256": _sha(executor_amendment_path),
        },
        "forbidden_inputs": [
            {"path": str(path), "sha256": _sha(path)}
            for path in forbidden_paths
            if path.exists()
        ],
        "executor": {
            "model": MODEL,
            "dtype": "bfloat16",
            "attention_implementation": "sdpa",
            "splice_suffix_chunk_size": 512,
        },
        "duplicate_design_keys": len(duplicates),
        "missing_design_keys": len(missing_keys),
        "extra_design_keys": len(extra_keys),
        "incomplete_dose_cells": len(incomplete),
    }
    result = {
        **statistical,
        "artifact_validation_passed": not violations,
        "artifact_violations": violations,
        "manifest": manifest,
    }
    write_jsonl(aggregate_output, rows)
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    gate_output.parent.mkdir(parents=True, exist_ok=True)
    gate_output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    for role in ROLE_SPEC:
        parser.add_argument(f"--{role.replace('_', '-')}", type=Path, required=True)
    parser.add_argument(
        "--executor-summary", type=Path, action="append", required=True
    )
    parser.add_argument("--executor-amendment", type=Path, required=True)
    parser.add_argument(
        "--forbidden-input", type=Path, action="append", default=[]
    )
    parser.add_argument("--aggregate-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--gate-output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    args = parser.parse_args()
    role_paths = {
        role: getattr(args, role)
        for role in ROLE_SPEC
    }
    result = aggregate(
        design_path=args.design,
        role_paths=role_paths,
        summary_paths=args.executor_summary,
        executor_amendment_path=args.executor_amendment,
        forbidden_paths=args.forbidden_input,
        aggregate_output=args.aggregate_output,
        manifest_output=args.manifest_output,
        gate_output=args.gate_output,
        bootstrap=args.bootstrap,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
