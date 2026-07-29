#!/usr/bin/env python3
"""Materialize the frozen SWE-bench mechanism cohort for the narrowed study."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path("/home/gfy/CodeMAS_Project")
DEFAULT_REGISTRATION = (
    ROOT
    / "kvflow-artifacts/impactkv_three_method_coding_benchmark_20260728"
    / "REGISTRATION.json"
)
DEFAULT_POPULATION = (
    ROOT
    / "sglang-kvflow/results/repo_level_datasets/"
    "swe_verified_500_instances.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "kvflow-artifacts/impactkv_three_method_coding_benchmark_20260728"
    / "swebench_verified"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(
    registration_path: Path,
    population_path: Path,
    output: Path,
) -> dict[str, Any]:
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    task_ids = [
        row["instance_id"]
        for row in registration["datasets"]["swebench_verified_mechanism"][
            "tasks"
        ]
    ]
    population = json.loads(population_path.read_text(encoding="utf-8"))
    indexed = {row["instance_id"]: row for row in population}
    missing = set(task_ids).difference(indexed)
    if missing:
        raise ValueError(f"SWE-bench population is missing: {sorted(missing)}")
    selected = [indexed[task_id] for task_id in task_ids]
    output.mkdir(parents=True, exist_ok=True)
    snapshot = output / "FROZEN_SUBSET.json"
    dataset_dir = output / "minisweagent_dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dataset = dataset_dir / "test.jsonl"
    evaluation = output / "EVAL_REGISTRATION.json"
    snapshot.write_text(
        json.dumps(selected, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    dataset.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in selected
        ),
        encoding="utf-8",
    )
    evaluation_value = {
        "schema_version": 1,
        "registration_id": "three-method-swebench-mechanism-20260728",
        "dataset": {
            "name": "princeton-nlp/SWE-bench_Verified",
            "split": "test",
            "population_size": len(population),
            "local_snapshot": str(snapshot),
            "local_snapshot_sha256": _sha256(snapshot),
        },
        "instances": [{"instance_id": task_id} for task_id in task_ids],
        "selection_uses_accuracy_or_method_outputs": False,
        "claim_scope": "reuse-rich development/mechanism cohort",
    }
    evaluation.write_text(
        json.dumps(
            evaluation_value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "status": "PREPARED_FROM_FROZEN_THREE_METHOD_REGISTRATION",
        "count": len(selected),
        "instances": task_ids,
        "source_registration": str(registration_path),
        "source_registration_sha256": _sha256(registration_path),
        "population": str(population_path),
        "population_sha256": _sha256(population_path),
        "snapshot": str(snapshot),
        "snapshot_sha256": _sha256(snapshot),
        "dataset": str(dataset),
        "dataset_sha256": _sha256(dataset),
        "evaluation_registration": str(evaluation),
        "evaluation_registration_sha256": _sha256(evaluation),
    }
    (output / "DATASET_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registration", type=Path, default=DEFAULT_REGISTRATION
    )
    parser.add_argument("--population", type=Path, default=DEFAULT_POPULATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(args.registration, args.population, args.output),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
