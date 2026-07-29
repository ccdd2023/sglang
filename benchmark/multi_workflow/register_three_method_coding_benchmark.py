#!/usr/bin/env python3
"""Register the narrowed V40/KVCOMM/CacheBlend coding comparison.

The benchmark has two deliberately different cohorts:

* ``mechanism`` contains SWE-bench Verified tasks whose already-frozen Dense
  trajectories expose substantial V40-eligible repository observations.  It
  is useful for development and mechanism validation, but not for an
  unqualified population claim.
* ``control`` is selected from RepoBench-P without looking at any method
  output.  It checks whether conclusions survive in static repository code
  completion, where V40's file-version signal is largely absent.

Dense is a normalization arm for every native engine, not a fourth competing
method.  This registration is additive and never changes an older experiment
or preregistration gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/home/gfy/CodeMAS_Project")
DEFAULT_MOTIVATION = (
    ROOT
    / "kvflow-artifacts/impactkv_v40_grounded_observation_motivation_20260728"
    / "V40_MOTIVATION_RESULT.json"
)
DEFAULT_REPOBENCH = (
    ROOT
    / "kvflow-reproductions/qcfuse-official/data/coding_200_qwen3_8b_5k"
    / "repobench-p.jsonl"
)
DEFAULT_OUTPUT = (
    ROOT
    / "kvflow-artifacts/impactkv_three_method_coding_benchmark_20260728"
    / "REGISTRATION.json"
)
METHODS = (
    "coding_grounded_observation_island_v40",
    "KVCOMM",
    "CacheBlend",
)
MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct"
SELECTION_SALT = "three-method-coding-benchmark-20260728-v1"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )


def _repo(instance_id: str) -> str:
    if "__" not in instance_id:
        raise ValueError(f"not a SWE-bench instance id: {instance_id}")
    return instance_id.split("__", 1)[0]


def opportunity_rows(motivation: dict[str, Any]) -> list[dict[str, Any]]:
    """Return outcome-free per-task reuse-opportunity measurements."""

    rows: list[dict[str, Any]] = []
    for cohort_rows in motivation["cohorts"].values():
        for raw in cohort_rows:
            eligible = int(raw["eligible_target_requests"])
            selected = [int(value) for value in raw["selected_tokens"]]
            requests = int(raw["requests_with_source"])
            expected_tokens = sum(selected) / eligible if eligible else 0.0
            rows.append(
                {
                    "instance_id": str(raw["instance_id"]),
                    "repo": _repo(str(raw["instance_id"])),
                    "eligible_target_requests": eligible,
                    "requests_with_source": requests,
                    "source_request_rate": requests / eligible if eligible else 0.0,
                    "selected_tokens": sum(selected),
                    "expected_copied_tokens_per_target": expected_tokens,
                    "version_invalidated_observations": int(
                        raw["version_invalidated_observations"]
                    ),
                }
            )
    ids = [row["instance_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("motivation result contains duplicate task IDs")
    return rows


def select_mechanism_cohort(
    rows: list[dict[str, Any]],
    *,
    size: int,
    per_repo_cap: int = 2,
    minimum_source_requests: int = 4,
) -> list[dict[str, Any]]:
    """Select high-opportunity tasks without reading accuracy outcomes."""

    eligible = [
        row
        for row in rows
        if row["requests_with_source"] >= minimum_source_requests
    ]
    ranked = sorted(
        eligible,
        key=lambda row: (
            -float(row["expected_copied_tokens_per_target"]),
            -float(row["source_request_rate"]),
            hashlib.sha256(
                f"{SELECTION_SALT}:{row['instance_id']}".encode("utf-8")
            ).hexdigest(),
            row["instance_id"],
        ),
    )
    selected: list[dict[str, Any]] = []
    repo_counts: dict[str, int] = {}
    for row in ranked:
        repo = str(row["repo"])
        if repo_counts.get(repo, 0) >= per_repo_cap:
            continue
        selected.append(row)
        repo_counts[repo] = repo_counts.get(repo, 0) + 1
        if len(selected) == size:
            break
    if len(selected) != size:
        raise ValueError(
            f"only {len(selected)} mechanism tasks satisfy the frozen rule; "
            f"requested {size}"
        )
    return selected


def select_repobench_ids(
    source: Path, *, size: int
) -> tuple[list[str], int]:
    ids: list[str] = []
    with source.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            ids.append(str(row.get("_id", f"repobench-p-{index}")))
    if len(ids) != len(set(ids)):
        raise ValueError("RepoBench-P source contains duplicate IDs")
    ranked = sorted(
        ids,
        key=lambda case_id: (
            hashlib.sha256(
                f"{SELECTION_SALT}:repobench:{case_id}".encode("utf-8")
            ).hexdigest(),
            case_id,
        ),
    )
    if size > len(ranked):
        raise ValueError(f"requested {size} RepoBench-P cases from {len(ranked)}")
    return ranked[:size], len(ranked)


def registration(
    *,
    motivation_path: Path,
    repobench_path: Path,
    mechanism_size: int,
    control_size: int,
) -> dict[str, Any]:
    motivation = json.loads(motivation_path.read_text(encoding="utf-8"))
    mechanism = select_mechanism_cohort(
        opportunity_rows(motivation), size=mechanism_size
    )
    controls, control_population = select_repobench_ids(
        repobench_path, size=control_size
    )
    value: dict[str, Any] = {
        "schema_version": 1,
        "status": "REGISTERED_BEFORE_THREE_METHOD_TREATMENT",
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "competing_methods": list(METHODS),
            "normalization": (
                "Each native engine has a matched Dense arm; Dense is not "
                "counted as a competing reuse method."
            ),
            "deferred_methods": [
                "QCFuse",
                "FUSE-RAG",
                "ProphetKV",
                "tail repair",
                "generic contiguous reuse",
            ],
            "prefetch": False,
        },
        "fairness": {
            "model": MODEL,
            "same_task_ids_within_dataset": True,
            "temperature": 0,
            "accuracy": {
                "swebench_verified": "official container resolved/pass@1",
                "repobench_p": "exact next line and code similarity",
            },
            "latency": {
                "primary": "cache-ready online TTFT speedup versus native Dense",
                "secondary": "N=4 TTFT including one quarter of cache build",
                "fixed_concurrency": 1,
                "source_build_excluded_from_cache_ready": True,
            },
            "cross_engine_reporting": (
                "Report absolute accuracy and TTFT, but rank reuse methods "
                "primarily by paired accuracy delta and speedup versus their "
                "own native Dense because runtime topology differs."
            ),
        },
        "datasets": {
            "swebench_verified_mechanism": {
                "role": "headline development/mechanism cohort",
                "selection_basis": (
                    "Frozen Dense trajectories only: expected V40-eligible "
                    "copied tokens per target, then source-request rate; "
                    "minimum four source-bearing requests and at most two "
                    "tasks per repository."
                ),
                "selection_uses_accuracy_or_method_outputs": False,
                "population_claim": False,
                "task_count": len(mechanism),
                "tasks": mechanism,
                "source": str(motivation_path),
                "source_sha256": file_sha256(motivation_path),
            },
            "repobench_p_control": {
                "role": "mainstream static repository-completion control",
                "selection_basis": "salted hash of source case ID",
                "selection_uses_accuracy_or_method_outputs": False,
                "population_size": control_population,
                "task_count": len(controls),
                "task_ids": controls,
                "source": str(repobench_path),
                "source_sha256": file_sha256(repobench_path),
            },
        },
        "promotion_gates": {
            "canary_before_scale": [
                "all three reuse arms execute physical KV reuse",
                "all matched Dense arms execute",
                "no task ID or prompt mismatch",
                "official or deterministic accuracy evaluator completes",
                "TTFT is measured at first generated token",
            ],
            "success_target": (
                "V40 preserves or improves paired accuracy relative to its "
                "Dense arm and exceeds both baseline accuracy deltas while "
                "retaining positive cache-ready TTFT speedup."
            ),
            "no_post_outcome_replacement": True,
        },
        "protected": {
            "old_dirty_checkout_modified": False,
            "paper_modified": False,
            "old_preregistration_gates_modified": False,
        },
        "selection_salt": SELECTION_SALT,
    }
    value["registration_sha256"] = canonical_sha256(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motivation", type=Path, default=DEFAULT_MOTIVATION)
    parser.add_argument("--repobench", type=Path, default=DEFAULT_REPOBENCH)
    parser.add_argument("--mechanism-size", type=int, default=12)
    parser.add_argument("--control-size", type=int, default=50)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    value = registration(
        motivation_path=args.motivation,
        repobench_path=args.repobench,
        mechanism_size=args.mechanism_size,
        control_size=args.control_size,
    )
    if args.output.exists():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        # Timestamp is allowed to differ only before the first write.
        if existing != value:
            raise FileExistsError(
                f"refusing to modify existing registration: {args.output}"
            )
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
