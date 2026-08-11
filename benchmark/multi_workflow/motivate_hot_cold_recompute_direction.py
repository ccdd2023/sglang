#!/usr/bin/env python3
"""Test whether coding dependency should protect hot KV and copy cold KV.

The natural-module study established that path/symbol-linked consumers attend
to their real repository source.  That is an importance result, not a reuse
safety result.  This experiment freezes same-task hot/cold repository-code
pairs before opening new physical continuations.  Every treatment copies the
same 128-token tail of a natural code module, so copied-token budget is exact.

Hot means the frozen design contains an online-visible later consumer relation
(path, directory, symbol, or interpretation grounding).  Cold means no such
consumer was found in the rendered target prompt.  The primary diagnostic is
64-token greedy continuation divergence from each arm's own Dense prompt.  It
is not official task accuracy and cannot by itself deploy an online selector.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from benchmark.multi_workflow import validate_single_island_action_divergence as action


ROOT = Path("/home/gfy/CodeMAS_Project")
SOURCE = (
    ROOT
    / "kvflow-artifacts/impactkv_natural_module_attention_20260808/"
    "attention_initial20_r1/DESIGN.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "kvflow-artifacts/impactkv_hot_cold_recompute_direction_20260810/"
    "same_task_fixed128"
)
COPY_TOKENS = 128
MIN_PAIRS = 8
MIN_TASKS = 7
MAX_PAIRS_PER_TASK = 2
SELECTION_SALT = "impactkv-hot-cold-protection-20260810-v1"
BOOTSTRAP_SEED = 2026081001
BOOTSTRAP_SAMPLES = 10_000


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable(value: str) -> str:
    return hashlib.sha256(f"{SELECTION_SALT}:{value}".encode()).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    path.chmod(0o644)


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_key(case: Mapping[str, Any], candidate: Mapping[str, Any]) -> str:
    return f"{case['case_id']}::{candidate['candidate_id']}"


def _is_hot(candidate: Mapping[str, Any]) -> bool:
    """A frozen later source-consumer relation marks a protected hot source."""

    return candidate.get("relation_control") is not None


def _record(case: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    length = int(candidate["natural_length"])
    return {
        "key": _candidate_key(case, candidate),
        "case": case,
        "candidate": candidate,
        "instance_id": str(case["instance_id"]),
        "case_id": str(case["case_id"]),
        "candidate_id": str(candidate["candidate_id"]),
        "hot": _is_hot(candidate),
        "natural_length": length,
        "prompt_length": len(case["target_input_ids"]),
        "target_position": int(candidate["target_start"])
        / max(len(case["target_input_ids"]), 1),
    }


def _pair_cost(cold: Mapping[str, Any], hot: Mapping[str, Any]) -> float:
    length = abs(math.log(cold["natural_length"] / hot["natural_length"]))
    prompt = abs(math.log(cold["prompt_length"] / hot["prompt_length"]))
    position = abs(float(cold["target_position"]) - float(hot["target_position"]))
    return length + 0.5 * prompt + 0.5 * position


def pair_same_task_candidates(
    source_design: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Greedily freeze unique same-task hot/cold pairs without outcomes."""

    by_task: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: {"cold": [], "hot": []}
    )
    for case in source_design["cases"]:
        for candidate in case["candidates"]:
            if str(candidate["module_type"]) != "repository_code":
                continue
            if int(candidate["natural_length"]) < COPY_TOKENS:
                continue
            row = _record(case, candidate)
            by_task[row["instance_id"]]["hot" if row["hot"] else "cold"].append(row)

    pairs: list[dict[str, Any]] = []
    for task in sorted(by_task, key=_stable):
        cold = by_task[task]["cold"]
        hot = by_task[task]["hot"]
        edges = sorted(
            (
                _pair_cost(left, right),
                _stable(f"{left['key']}::{right['key']}"),
                left,
                right,
            )
            for left in cold
            for right in hot
        )
        used_cold: set[str] = set()
        used_hot: set[str] = set()
        task_pairs = 0
        for cost, _, left, right in edges:
            if task_pairs >= MAX_PAIRS_PER_TASK:
                break
            if left["key"] in used_cold or right["key"] in used_hot:
                continue
            pair_id = f"p{len(pairs):02d}-{_stable(left['key'] + right['key'])[:8]}"
            pairs.append(
                {
                    "pair_id": pair_id,
                    "instance_id": task,
                    "cost": cost,
                    "cold": left,
                    "hot": right,
                }
            )
            used_cold.add(left["key"])
            used_hot.add(right["key"])
            task_pairs += 1
    return pairs


def _fixed_tail_candidate(
    row: Mapping[str, Any], pair_id: str, arm: str
) -> dict[str, Any]:
    candidate = copy.deepcopy(row["candidate"])
    natural_length = int(candidate["natural_length"])
    candidate.update(
        {
            "candidate_id": f"{candidate['candidate_id']}::{arm}",
            "source_start": int(candidate["source_start"])
            + natural_length
            - COPY_TOKENS,
            "target_start": int(candidate["target_start"])
            + natural_length
            - COPY_TOKENS,
            "length": COPY_TOKENS,
            "original_candidate_id": str(row["candidate_id"]),
            "original_natural_length": natural_length,
            "pair_id": pair_id,
            "arm": arm,
            "dependency_hot": arm == "hot",
        }
    )
    return candidate


def prepare(source: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        registration = output / "REGISTRATION.json"
        if registration.exists():
            return _read(registration)
        raise FileExistsError(output)
    source_design = _read(source)
    pairs = pair_same_task_candidates(source_design)
    cases = []
    frozen_pairs = []
    for pair in pairs:
        pair_value = {
            "pair_id": pair["pair_id"],
            "instance_id": pair["instance_id"],
            "cost": pair["cost"],
            "arms": {},
        }
        for arm in ("cold", "hot"):
            row = pair[arm]
            case = copy.deepcopy(row["case"])
            candidate = _fixed_tail_candidate(row, pair["pair_id"], arm)
            case["candidates"] = [candidate]
            case["hot_cold_pair_id"] = pair["pair_id"]
            case["hot_cold_arm"] = arm
            cases.append(case)
            pair_value["arms"][arm] = {
                "case_id": case["case_id"],
                "candidate_id": candidate["candidate_id"],
                "original_candidate_id": candidate["original_candidate_id"],
                "natural_length": candidate["original_natural_length"],
                "target_prompt_tokens": len(case["target_input_ids"]),
                "target_position": int(candidate["target_start"]),
            }
        frozen_pairs.append(pair_value)

    tasks = {str(row["instance_id"]) for row in frozen_pairs}
    gates = {
        "at_least_8_pairs": len(frozen_pairs) >= MIN_PAIRS,
        "at_least_7_tasks": len(tasks) >= MIN_TASKS,
        "same_task_within_every_pair": all(
            pair["instance_id"] in pair["arms"]["cold"]["case_id"]
            and pair["instance_id"] in pair["arms"]["hot"]["case_id"]
            for pair in frozen_pairs
        ),
        "all_equal_128_token_budget": all(
            int(candidate["length"]) == COPY_TOKENS
            for case in cases
            for candidate in case["candidates"]
        ),
    }
    output.mkdir(parents=True)
    design_path = output / "DESIGN.json"
    _write(
        design_path,
        {
            "model": str(action.MODEL),
            "continuation_tokens": action.CONTINUATION_TOKENS,
            "copy_tokens": COPY_TOKENS,
            "pairs": frozen_pairs,
            "cases": cases,
        },
    )
    registration = {
        "status": (
            "REGISTERED_BEFORE_ACTION_OUTCOMES"
            if all(gates.values())
            else "STOPPED_BEFORE_ACTION_OUTCOMES"
        ),
        "purpose": (
            "test the directional hypothesis that online-visible coding dependency "
            "should protect/recompute hot code while lossy-copying cold code"
        ),
        "source_design": str(source),
        "source_design_sha256": _sha(source),
        "script_sha256": _sha(Path(__file__)),
        "design_sha256": _sha(design_path),
        "selection_used_new_physical_or_action_outcomes": False,
        "selection": {
            "same_task": True,
            "unique_candidates": True,
            "maximum_pairs_per_task": MAX_PAIRS_PER_TASK,
            "copy_tokens_per_arm": COPY_TOKENS,
            "cost": "abs(log natural length ratio) + 0.5*abs(log prompt length ratio) + 0.5*abs(target position fraction)",
        },
        "capacity": {
            "pairs": len(frozen_pairs),
            "tasks": len(tasks),
            "cases": len(cases),
        },
        "capacity_gates": gates,
        "frozen_outcome_gates": {
            "cold_normalized_edit_median_below_hot": True,
            "cold_pairwise_win_fraction_above_half": True,
            "cold_exact_match_rate_at_least_hot": True,
            "minimum_behaviorally_informative_pairs": 3,
        },
        "metric_scope": (
            "64-token continuation divergence is a motivation diagnostic, not "
            "official execution accuracy or TTFT"
        ),
        "protected": {
            "paper_modified": False,
            "prefetch": False,
            "old_preregistration_thresholds_modified": False,
        },
    }
    _write(output / "REGISTRATION.json", registration)
    return registration


def _quantiles(values: Sequence[float]) -> list[float]:
    return [float(value) for value in np.quantile(values, [0.025, 0.5, 0.975])]


def summarize(output: Path) -> dict[str, Any]:
    registration = _read(output / "REGISTRATION.json")
    design = _read(output / "DESIGN.json")
    if registration["design_sha256"] != _sha(output / "DESIGN.json"):
        raise ValueError("design changed after registration")
    rows = action._jsonl(output / "ACTION_OUTCOMES.jsonl")
    by_case = {str(row["case_id"]): row for row in rows}
    arm_index = {
        str(case["case_id"]): {
            "pair_id": str(case["hot_cold_pair_id"]),
            "arm": str(case["hot_cold_arm"]),
        }
        for case in design["cases"]
    }
    values: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for case_id, row in by_case.items():
        meta = arm_index[case_id]
        candidate = row["candidates"][0]
        values[meta["pair_id"]][meta["arm"]] = {
            "case_id": case_id,
            "instance_id": str(row["instance_id"]),
            "exact_match": bool(candidate["exact_match"]),
            "normalized_token_edit_distance": float(
                candidate["normalized_token_edit_distance"]
            ),
            "common_prefix_fraction": float(candidate["common_prefix_fraction"]),
        }
    pairs = []
    for pair in design["pairs"]:
        pair_id = str(pair["pair_id"])
        if set(values[pair_id]) != {"cold", "hot"}:
            raise RuntimeError(f"incomplete pair: {pair_id}")
        cold = values[pair_id]["cold"]
        hot = values[pair_id]["hot"]
        pairs.append(
            {
                "pair_id": pair_id,
                "instance_id": pair["instance_id"],
                "cold": cold,
                "hot": hot,
                "cold_edit_minus_hot": cold["normalized_token_edit_distance"]
                - hot["normalized_token_edit_distance"],
                "cold_prefix_minus_hot": cold["common_prefix_fraction"]
                - hot["common_prefix_fraction"],
            }
        )
    cold_edits = [row["cold"]["normalized_token_edit_distance"] for row in pairs]
    hot_edits = [row["hot"]["normalized_token_edit_distance"] for row in pairs]
    informative = [
        row
        for row in pairs
        if row["cold"]["normalized_token_edit_distance"]
        != row["hot"]["normalized_token_edit_distance"]
    ]
    task_diffs: dict[str, list[float]] = defaultdict(list)
    for row in pairs:
        task_diffs[str(row["instance_id"])].append(float(row["cold_edit_minus_hot"]))
    tasks = sorted(task_diffs)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sampled = rng.choice(tasks, len(tasks), replace=True)
        pooled = [value for task in sampled for value in task_diffs[str(task)]]
        boot.append(statistics.median(pooled))
    exact_cold = sum(row["cold"]["exact_match"] for row in pairs) / len(pairs)
    exact_hot = sum(row["hot"]["exact_match"] for row in pairs) / len(pairs)
    gates = {
        "cold_normalized_edit_median_below_hot": statistics.median(cold_edits)
        < statistics.median(hot_edits),
        "cold_pairwise_win_fraction_above_half": (
            sum(row["cold_edit_minus_hot"] < 0 for row in informative)
            / max(len(informative), 1)
            > 0.5
        ),
        "cold_exact_match_rate_at_least_hot": exact_cold >= exact_hot,
        "minimum_behaviorally_informative_pairs": len(informative)
        >= int(
            registration["frozen_outcome_gates"][
                "minimum_behaviorally_informative_pairs"
            ]
        ),
    }
    result = {
        "status": "PASS_DIRECTION" if all(gates.values()) else "NO_DIRECTIONAL_PASS",
        "pairs": len(pairs),
        "tasks": len(tasks),
        "informative_pairs": len(informative),
        "copy_tokens_per_arm": COPY_TOKENS,
        "cold": {
            "exact_match_rate": exact_cold,
            "normalized_edit_median": statistics.median(cold_edits),
            "normalized_edit_mean": statistics.fmean(cold_edits),
        },
        "hot": {
            "exact_match_rate": exact_hot,
            "normalized_edit_median": statistics.median(hot_edits),
            "normalized_edit_mean": statistics.fmean(hot_edits),
        },
        "paired": {
            "cold_edit_minus_hot_q025_q50_q975": _quantiles(boot),
            "cold_wins_among_informative": sum(
                row["cold_edit_minus_hot"] < 0 for row in informative
            ),
            "hot_wins_among_informative": sum(
                row["cold_edit_minus_hot"] > 0 for row in informative
            ),
        },
        "gates": gates,
        "pair_results": pairs,
        "next_action": (
            "test_cold_copy_in_same_history_official_fork"
            if all(gates.values())
            else "do_not_deploy_dependency_direction_from_this_proxy"
        ),
    }
    _write(output / "RESULT.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "measure", "summarize"))
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-cases", type=int, default=0)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.source, args.output)
    elif args.command == "measure":
        registration = prepare(args.source, args.output)
        if registration["status"] != "REGISTERED_BEFORE_ACTION_OUTCOMES":
            raise RuntimeError("capacity gates did not pass")
        result = action.measure(args.output, args.max_cases)
    else:
        result = summarize(args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
