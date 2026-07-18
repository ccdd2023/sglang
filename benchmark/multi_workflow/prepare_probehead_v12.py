#!/usr/bin/env python3
"""Register and freeze ProbeHead V12 from explicit, immutable V11 inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from benchmark.multi_workflow.probehead_v12 import (
    BOOTSTRAP_ITERATIONS,
    HEAD_CANDIDATES,
    JS_LIMIT,
    MAX_COPY_ISLANDS,
    MAX_PROBE_P95_MS,
    MIN_HARM_REDUCTION,
    MIN_PROMPT_COPY_FRACTION,
    PROFILE,
    SHUFFLE_SEED,
)
from benchmark.multi_workflow.sessiongraph_v11 import read_jsonl, write_jsonl


MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_turn(
    turns: list[dict[str, Any]], current: dict[str, Any], module_id: str
) -> int:
    target = next(
        module for module in current["modules"] if module["module_id"] == module_id
    )
    candidates = []
    for turn in turns:
        if int(turn["turn_id"]) >= int(current["turn_id"]):
            continue
        for module in turn["modules"]:
            if (
                module["module_id"] == module_id
                and module["content_hash"] == target["content_hash"]
            ):
                candidates.append(int(turn["turn_id"]))
    if not candidates:
        raise ValueError(
            f"selected module has no exact earlier source: "
            f"{current['session_id']}:{current['turn_id']}:{module_id}"
        )
    return max(candidates)


def prepare(
    *,
    replay_path: Path,
    capacity_path: Path,
    split_path: Path,
    v11_design_path: Path,
    v11_registration_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    capacity = read_jsonl(capacity_path)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    v11_registration = json.loads(v11_registration_path.read_text(encoding="utf-8"))
    if v11_registration.get("policy") != "fileversion-sessiongraphkv-v11":
        raise ValueError("the supplied registration is not V11")
    development = set(map(str, split.get("development", ())))
    holdout = set(map(str, split.get("holdout", ())))
    if len(development) != 32 or len(holdout) != 32 or development & holdout:
        raise ValueError("V11 split must be disjoint 32 development / 32 holdout")
    if len(capacity) != 192:
        raise ValueError("V11 capacity must cover exactly 192 later-turn requests")

    by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_turn = {}
    for turn in replay:
        session_id = str(turn["session_id"])
        by_session[session_id].append(turn)
        by_turn[(session_id, int(turn["turn_id"]))] = turn
    for turns in by_session.values():
        turns.sort(key=lambda value: int(value["turn_id"]))

    workflow_rows: list[dict[str, Any]] = []
    for capacity_row in capacity:
        session_id = str(capacity_row["session_id"])
        turn_id = int(capacity_row["turn_id"])
        current = by_turn[(session_id, turn_id)]
        modules = {
            str(module["module_id"]): module for module in current["modules"]
        }
        cohort = "development" if session_id in development else "holdout"
        if session_id not in development | holdout:
            raise ValueError(f"session is absent from frozen split: {session_id}")
        for module_id in map(str, capacity_row["copied_module_ids"]):
            module = modules[module_id]
            source_turn_id = _source_turn(
                by_session[session_id], current, module_id
            )
            for head_tokens in HEAD_CANDIDATES:
                workflow_rows.append(
                    {
                        "case_kind": "workflow",
                        "cohort": cohort,
                        "session_id": session_id,
                        "turn_id": turn_id,
                        "source_turn_id": source_turn_id,
                        "module_id": module_id,
                        "module_type": str(module["module_type"]),
                        "cache_scope": str(module["cache_scope"]),
                        "disturbance": "same_task",
                        "head_tokens": head_tokens,
                        "prompt_tokens": int(capacity_row["prompt_tokens"]),
                    }
                )

    stress_rows: list[dict[str, Any]] = []
    seen_stress = set()
    for row in read_jsonl(v11_design_path):
        key = (
            str(row["cohort"]),
            str(row["session_id"]),
            int(row["turn_id"]),
            str(row["module_id"]),
            str(row["disturbance"]),
        )
        if float(row["recompute_fraction"]) != 0.0 or key in seen_stress:
            continue
        seen_stress.add(key)
        for head_tokens in HEAD_CANDIDATES:
            stress_rows.append(
                {
                    "case_kind": "stress",
                    "cohort": key[0],
                    "session_id": key[1],
                    "turn_id": key[2],
                    "source_turn_id": None,
                    "module_id": key[3],
                    "module_type": str(row["module_type"]),
                    "cache_scope": str(row["cache_scope"]),
                    "disturbance": key[4],
                    "head_tokens": head_tokens,
                    "prompt_tokens": None,
                    "negative_control": bool(row["negative_control"]),
                }
            )

    design = [*workflow_rows, *stress_rows]
    keys = {
        (
            row["case_kind"],
            row["cohort"],
            row["session_id"],
            row["turn_id"],
            row["module_id"],
            row["disturbance"],
            row["head_tokens"],
        )
        for row in design
    }
    if len(keys) != len(design):
        raise ValueError("V12 design contains duplicate keys")

    output_dir.mkdir(parents=True, exist_ok=True)
    design_path = output_dir / "P0_DESIGN.jsonl"
    write_jsonl(design_path, design)
    registration = {
        "policy": PROFILE,
        "claim_scope": "same-session exact-module reuse gated by observed head-KV deviation",
        "stage_scope": "offline P0 only; P1 accuracy, TTFT, and runtime executor closed",
        "model": MODEL,
        "dtype": "bfloat16",
        "attention_implementation": "sdpa",
        "max_context": 32768,
        "splice_suffix_chunk_size": 512,
        "head_candidates": list(HEAD_CANDIDATES),
        "probe_score": "max(mean_cosine_deviation(rope_shift(K)), mean_cosine_deviation(V))",
        "max_copy_islands": MAX_COPY_ISLANDS,
        "shuffled_seed": SHUFFLE_SEED,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "gates": {
            "negative_control_max_js_lte": JS_LIMIT,
            "composed_splice_p95_js_lte": JS_LIMIT,
            "median_cost_positive_copy_fraction_gte": MIN_PROMPT_COPY_FRACTION,
            "copy_all_harm_reduction_gte": MIN_HARM_REDUCTION,
            "shuffled_harm_reduction_gte": MIN_HARM_REDUCTION,
            "harm_reduction_ci_low_gt": 0.0,
            "splice_top1_changes": 0,
            "probe_p95_ms_lt": MAX_PROBE_P95_MS,
            "invalid_rows": 0,
        },
        "calibration_rule": (
            "maximize development median request copy fraction among feasible "
            "(head, threshold) pairs; tie-break by lower p95 harm, smaller head, "
            "then lower threshold"
        ),
        "development_sessions": 32,
        "holdout_sessions": 32,
        "workflow_design_rows": len(workflow_rows),
        "stress_design_rows": len(stress_rows),
        "design_sha256": _sha(design_path),
        "source_artifacts": {
            "replay": {"path": str(replay_path), "sha256": _sha(replay_path)},
            "capacity": {
                "path": str(capacity_path),
                "sha256": _sha(capacity_path),
            },
            "split": {"path": str(split_path), "sha256": _sha(split_path)},
            "v11_design": {
                "path": str(v11_design_path),
                "sha256": _sha(v11_design_path),
            },
            "v11_registration": {
                "path": str(v11_registration_path),
                "sha256": _sha(v11_registration_path),
            },
        },
        "v11_thresholds_changed": False,
        "v11_verdict_changed": False,
        "paper_modified": False,
        "holdout_measurements_read": False,
    }
    registration_path = output_dir / "EXPERIMENT_REGISTRATION.json"
    registration_path.write_text(
        json.dumps(registration, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    status = {
        "policy": PROFILE,
        "registration": "FROZEN",
        "development": "READY_TO_MEASURE",
        "calibration_lock": "NOT_CREATED",
        "holdout": "SEALED",
        "p1": "CLOSED",
    }
    (output_dir / "STAGE_STATUS.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "passed": True,
        "design_rows": len(design),
        "workflow_rows": len(workflow_rows),
        "stress_rows": len(stress_rows),
        "development_sessions": len(development),
        "holdout_sessions": len(holdout),
        "registration": str(registration_path),
        "design": str(design_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--capacity", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--v11-design", type=Path, required=True)
    parser.add_argument("--v11-registration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                replay_path=args.replay,
                capacity_path=args.capacity,
                split_path=args.split,
                v11_design_path=args.v11_design,
                v11_registration_path=args.v11_registration,
                output_dir=args.output_dir,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
