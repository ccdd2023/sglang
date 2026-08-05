#!/usr/bin/env python3
"""Task-disjoint test of risk-filtered, path-ranked single-island reuse."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmark.multi_workflow import motivate_v52_path_dependency as m52
from benchmark.multi_workflow.coding_reuse_policy import (
    critical_coding_event_reasons,
    repository_paths,
)
from benchmark.multi_workflow.motivate_v48_attention_kv_risk import (
    _candidate_internal_metrics,
    _compose_splice,
    _dense_source,
    _first_token_nll,
    _model_theta,
    _target_forward,
)
from benchmark.multi_workflow.motivate_v49_probe_proxy import _probe_score
from benchmark.multi_workflow.motivate_v50_coding_provenance import (
    _balanced_select,
    _sha256,
    _turn_groups,
)
from benchmark.multi_workflow.run_bridge_reuse_pilot import write_json


ROOT = Path("/home/gfy/CodeMAS_Project")
FRESH_ROOT = ROOT / "kvflow-artifacts/impactkv_m55_v40_task_disjoint_20260805"
DEFAULT_OUTPUT = ROOT / "kvflow-artifacts/impactkv_m55_two_stage_20260805/fresh13"
PROBE_LAYER = 17
PROBE_HEAD_TOKENS = 16
PROBE_THRESHOLD = 0.011477339267730712
RANDOM_SEED = 202608055
MIN_CASES = 16
MIN_TASKS = 8
FRESH_TASKS = (
    "astropy__astropy-13033",
    "django__django-12406",
    "django__django-16560",
    "psf__requests-6028",
    "pydata__xarray-3095",
    "pydata__xarray-3305",
    "pydata__xarray-6992",
    "pylint-dev__pylint-4551",
    "pylint-dev__pylint-4661",
    "pytest-dev__pytest-5787",
    "scikit-learn__scikit-learn-14087",
    "sphinx-doc__sphinx-7590",
    "sphinx-doc__sphinx-8120",
)


def _trajectory_paths(root: Path) -> dict[str, Path]:
    selected: dict[str, Path] = {}
    for path in sorted((root / "tasks").glob("**/dense/**/*.traj.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        instance_id = str(value.get("instance_id") or "")
        if instance_id in FRESH_TASKS and instance_id not in selected:
            selected[instance_id] = path
    return selected


def _candidate_pool_with_audit(
    tokenizer: Any, root: Path
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    paths = _trajectory_paths(root)
    original = m52._trajectory_paths
    try:
        m52._trajectory_paths = lambda: paths
        raw = m52._candidate_pool(tokenizer)
    finally:
        m52._trajectory_paths = original
    groups = {
        instance_id: _turn_groups(
            json.loads(path.read_text(encoding="utf-8"))["messages"][2:]
        )
        for instance_id, path in paths.items()
    }
    rows = []
    for row in raw:
        target_completed = int(row["target_request_index"]) - 1
        task_groups = groups[str(row["instance_id"])]
        if not all(
            _version_valid_at_target(candidate, task_groups, target_completed)
            for candidate in row["candidates"]
        ):
            continue
        rows.append(
            {
                **row,
                "candidates": [
                    {**candidate, "version_valid_at_target": True}
                    for candidate in row["candidates"]
                ],
                "version_guard_checked": True,
            }
        )
    return rows, {
        "path_matched_pairs_before_version_guard": len(raw),
        "pairs_removed_by_target_version_guard": len(raw) - len(rows),
        "version_valid_pairs": len(rows),
    }


def _candidate_pool(tokenizer: Any, root: Path) -> list[dict[str, Any]]:
    return _candidate_pool_with_audit(tokenizer, root)[0]


def _version_valid_at_target(
    candidate: Mapping[str, Any],
    groups: Sequence[Sequence[dict[str, Any]]],
    target_completed: int,
) -> bool:
    source_paths = {str(value) for value in candidate.get("repository_paths", ())}
    if not source_paths:
        return False
    source_index = int(candidate["group_index"])
    for later in groups[source_index + 1 : target_completed]:
        if "repository_mutation_command" not in critical_coding_event_reasons(later):
            continue
        changed_paths = repository_paths(later)
        if not changed_paths or not source_paths.isdisjoint(changed_paths):
            return False
    return True


def _gpu_eligible(cases: int, tasks: int) -> bool:
    return cases >= MIN_CASES and tasks >= MIN_TASKS


def prepare(output: Path, trajectory_root: Path, limit: int) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    task_registration = trajectory_root / "M55_TASK_REGISTRATION.json"
    task_result = trajectory_root / "M55_TASK_RESULT.json"
    if not task_registration.exists():
        raise FileNotFoundError("fresh-13 task registration is required")
    tokenizer = AutoTokenizer.from_pretrained(m52.MODEL, local_files_only=True)
    eligible, candidate_audit = _candidate_pool_with_audit(
        tokenizer, trajectory_root
    )
    selected = _balanced_select(eligible, min(limit, len(eligible)))
    selected_tasks = len({row["instance_id"] for row in selected})
    output.mkdir(parents=True)
    design_path = output / "DESIGN.json"
    write_json(
        design_path,
        {
            "cases": selected,
            "eligible_cases_before_sampling": len(eligible),
            "eligible_tasks_before_sampling": len(
                {row["instance_id"] for row in eligible}
            ),
            "candidate_audit": candidate_audit,
            "model": str(m52.MODEL),
        },
    )
    registration = {
        "status": "REGISTERED_BEFORE_FRESH13_CAUSAL_LABELS",
        "purpose": (
            "test a lexicographic single-island selector that filters by "
            "frozen K/V probe risk before ranking online path dependency"
        ),
        "design_sha256": _sha256(design_path),
        "task_registration": str(task_registration),
        "task_registration_sha256": _sha256(task_registration),
        "task_result": str(task_result),
        "task_result_sha256": _sha256(task_result) if task_result.exists() else None,
        "cases": len(selected),
        "tasks": selected_tasks,
        "eligible_cases_before_sampling": len(eligible),
        "eligible_tasks_before_sampling": len(
            {row["instance_id"] for row in eligible}
        ),
        "candidate_audit": candidate_audit,
        "model": str(m52.MODEL),
        "cohort_contract": {
            "task_disjoint_from_m52_m53_m54": True,
            "same_source_target_prompt_within_pair": True,
            "candidate_tokens_each": 128,
            "version_valid_grounded_observations": True,
            "one_case_per_task_before_second_case": True,
            "maximum_cases": limit,
            "minimum_cases": MIN_CASES,
            "minimum_tasks": MIN_TASKS,
        },
        "frozen_probe": {
            "layer_zero_based": PROBE_LAYER,
            "head_tokens": PROBE_HEAD_TOKENS,
            "risk_threshold": PROBE_THRESHOLD,
            "source": (
                "/home/gfy/CodeMAS_Project/kvflow-artifacts/"
                "impactkv_m49_probe_proxy_20260805/PROXY_LOCK.json"
            ),
        },
        "frozen_selectors": {
            "v40_fixed_budget_recency": "larger group_index wins",
            "path_only": "path_relevant",
            "probe_only": "minimum probe score",
            "seeded_random": f"sha256({RANDOM_SEED}:case_id:candidate_id)",
            "two_stage": (
                "discard score above threshold; choose path_relevant if "
                "eligible, otherwise minimum-risk eligible; Dense if none"
            ),
        },
        "frozen_gates": {
            "minimum_cases": MIN_CASES,
            "minimum_tasks": MIN_TASKS,
            "two_stage_coverage_vs_v40_min": 0.70,
            "common_cases_min": 12,
            "common_tasks_min": 6,
            "attention_ratio_vs_probe_min": 1.10,
            "attention_higher_fraction_vs_probe_min": 0.60,
            "js_ratio_vs_probe_max": 1.00,
            "js_ratio_vs_path_max": 0.90,
            "js_lower_fraction_vs_path_min": 0.60,
            "pareto_vs_v40": (
                "attention ratio >=1.10 and JS ratio <=1.00, or attention "
                "ratio >=1.00 and JS ratio <=0.90"
            ),
        },
        "interpretation_limits": [
            "Qwen2.5-Coder-3B causal measurement on Qwen3-Coder agent histories",
            "single 128-token island motivation, not task accuracy or TTFT",
            "Dense target attention is an oracle label, not runtime overhead",
            "passing does not promote V46 three-island composition",
        ],
        "gpu_measurement_eligible": _gpu_eligible(
            len(selected), selected_tasks
        ),
    }
    write_json(output / "REGISTRATION.json", registration)
    if not registration["gpu_measurement_eligible"]:
        write_json(
            output / "RESULT.json",
            {
                "status": "STOPPED_BEFORE_GPU_CAUSAL_LABELS",
                "decision": "INSUFFICIENT_TASK_DISJOINT_COHORT",
                "cases": len(selected),
                "tasks": selected_tasks,
                "minimum_cases": MIN_CASES,
                "minimum_tasks": MIN_TASKS,
                "candidate_audit": candidate_audit,
                "next_step": (
                    "do not measure, tune the threshold, or implement the "
                    "selector; revise the path-utility opportunity definition "
                    "under a separately registered experiment"
                ),
                "scope": (
                    "frozen task-disjoint capacity gate; no attention/JS GPU "
                    "labels were opened"
                ),
            },
        )
    return registration


def _complete(row: Mapping[str, Any]) -> bool:
    if row.get("status") != "ok" or len(row.get("candidates", [])) != 2:
        return False
    return all(
        math.isfinite(float(candidate[key]))
        for candidate in row["candidates"]
        for key in ("attention_mean", "probe_score", "causal_splice_logit_js")
    )


def measure(output: Path, max_cases: int) -> dict[str, Any]:
    design_path = output / "DESIGN.json"
    registration = json.loads((output / "REGISTRATION.json").read_text())
    if not registration.get("gpu_measurement_eligible", False):
        raise RuntimeError("capacity gate failed; GPU causal labels stay sealed")
    if registration["design_sha256"] != _sha256(design_path):
        raise ValueError("design changed after registration")
    cases = json.loads(design_path.read_text())["cases"]
    if max_cases > 0:
        cases = cases[:max_cases]
    destination = output / "OBSERVATIONS.jsonl"
    completed = set()
    if destination.exists():
        for line in destination.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if _complete(row):
                    completed.add(str(row["case_id"]))
    pending = [row for row in cases if str(row["case_id"]) not in completed]
    if not pending:
        return {"status": "COMPLETE", "cases": len(cases), "new_cases": 0}
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU substitution is forbidden")
    model = AutoModelForCausalLM.from_pretrained(
        m52.MODEL,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    ).to("cuda").eval()
    theta = _model_theta(model.config)
    errors = []
    written = 0
    for index, case in enumerate(pending, 1):
        try:
            source_cache, _ = _dense_source(model, case["source_input_ids"])
            target_cache, dense_logits, attention = _target_forward(
                model=model,
                target_ids=case["target_input_ids"],
                candidates=case["candidates"],
            )
            dense_nll = _first_token_nll(
                dense_logits, int(case["answer_first_token_id"])
            )
            measured = []
            for candidate in case["candidates"]:
                candidate_id = str(candidate["candidate_id"])
                internal = _candidate_internal_metrics(
                    candidate=candidate,
                    attention=attention[candidate_id],
                    source_cache=source_cache,
                    target_cache=target_cache,
                    theta=theta,
                    num_attention_heads=int(model.config.num_attention_heads),
                )
                probe = _probe_score(
                    source_cache=source_cache,
                    target_cache=target_cache,
                    candidate=candidate,
                    layer=PROBE_LAYER,
                    head_tokens=PROBE_HEAD_TOKENS,
                    theta=theta,
                )
                splice_logits = _compose_splice(
                    model=model,
                    target_ids=case["target_input_ids"],
                    target_cache=target_cache,
                    source_cache=source_cache,
                    candidates=[candidate],
                    theta=theta,
                )
                splice_nll = _first_token_nll(
                    splice_logits, int(case["answer_first_token_id"])
                )
                measured.append(
                    {
                        **candidate,
                        **internal,
                        "probe_score": probe["score"],
                        "position_fraction": candidate["target_start"]
                        / len(case["target_input_ids"]),
                        "causal_splice_logit_js": m52._js(
                            dense_logits, splice_logits
                        ),
                        "answer_first_token_nll_delta": splice_nll - dense_nll,
                    }
                )
                del splice_logits
            row = {
                "status": "ok",
                "case_id": case["case_id"],
                "instance_id": case["instance_id"],
                "candidates": measured,
            }
            if not _complete(row):
                raise RuntimeError("case produced incomplete metrics")
            with destination.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
            written += 1
            print(json.dumps({"case": index, "case_id": case["case_id"]}), flush=True)
            del source_cache, target_cache, dense_logits, attention
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as error:
            errors.append(
                {"case_id": case["case_id"], "error": f"{type(error).__name__}: {error}"}
            )
            break
    status = {
        "status": "COMPLETE" if not errors and written == len(pending) else "PARTIAL",
        "selected_cases": len(cases),
        "previously_completed_cases": len(completed),
        "new_cases": written,
        "errors": errors,
    }
    write_json(output / "MEASUREMENT_STATUS.json", status)
    return status


def _candidate(row: Mapping[str, Any], candidate_id: str) -> Mapping[str, Any]:
    return next(item for item in row["candidates"] if item["candidate_id"] == candidate_id)


def select(row: Mapping[str, Any], arm: str) -> Mapping[str, Any] | None:
    candidates = list(row["candidates"])
    if arm == "v40_fixed_budget_recency":
        return max(candidates, key=lambda item: (int(item["group_index"]), int(item["target_start"])))
    if arm == "path_only":
        return _candidate(row, "path_relevant")
    if arm == "probe_only":
        return min(candidates, key=lambda item: (float(item["probe_score"]), str(item["candidate_id"])))
    if arm == "seeded_random":
        return min(
            candidates,
            key=lambda item: hashlib.sha256(
                f"{RANDOM_SEED}:{row['case_id']}:{item['candidate_id']}".encode()
            ).hexdigest(),
        )
    if arm == "two_stage":
        eligible = [
            item for item in candidates if float(item["probe_score"]) <= PROBE_THRESHOLD
        ]
        if not eligible:
            return None
        relevant = [item for item in eligible if item["candidate_id"] == "path_relevant"]
        return relevant[0] if relevant else min(eligible, key=lambda item: float(item["probe_score"]))
    raise ValueError(f"unknown arm: {arm}")


def _geometric_ratio(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        return math.nan
    return math.exp(
        statistics.fmean(
            math.log(a + 1e-10) - math.log(b + 1e-10)
            for a, b in zip(left, right, strict=True)
        )
    )


def _direction_fraction(left: Sequence[float], right: Sequence[float], *, higher: bool) -> float:
    score = 0.0
    for a, b in zip(left, right, strict=True):
        if a == b:
            score += 0.5
        elif (a > b) == higher:
            score += 1.0
    return score / len(left)


def _comparison(rows: Sequence[Mapping[str, Any]], other: str) -> dict[str, float]:
    two = [select(row, "two_stage") for row in rows]
    control = [select(row, other) for row in rows]
    pairs = [(a, b) for a, b in zip(two, control, strict=True) if a is not None and b is not None]
    attention_left = [float(a["attention_mean"]) for a, _ in pairs]
    attention_right = [float(b["attention_mean"]) for _, b in pairs]
    js_left = [float(a["causal_splice_logit_js"]) for a, _ in pairs]
    js_right = [float(b["causal_splice_logit_js"]) for _, b in pairs]
    return {
        "cases": len(pairs),
        "attention_geometric_ratio": _geometric_ratio(attention_left, attention_right),
        "attention_higher_pair_fraction": _direction_fraction(attention_left, attention_right, higher=True),
        "js_geometric_ratio": _geometric_ratio(js_left, js_right),
        "js_lower_pair_fraction": _direction_fraction(js_left, js_right, higher=False),
    }


def analyze(output: Path) -> dict[str, Any]:
    existing = output / "RESULT.json"
    if existing.exists():
        value = json.loads(existing.read_text(encoding="utf-8"))
        if value.get("decision") == "INSUFFICIENT_TASK_DISJOINT_COHORT":
            return value
    rows = [
        json.loads(line)
        for line in (output / "OBSERVATIONS.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if not rows or any(not _complete(row) for row in rows):
        raise ValueError("observations are missing or incomplete")
    registration = json.loads((output / "REGISTRATION.json").read_text())
    comparisons = {
        arm: _comparison(rows, arm)
        for arm in ("v40_fixed_budget_recency", "path_only", "probe_only", "seeded_random")
    }
    selected_two = [select(row, "two_stage") for row in rows]
    selected_rows = [row for row, item in zip(rows, selected_two, strict=True) if item is not None]
    coverage = len(selected_rows) / len(rows)
    probe = comparisons["probe_only"]
    path = comparisons["path_only"]
    v40 = comparisons["v40_fixed_budget_recency"]
    pareto = (
        v40["attention_geometric_ratio"] >= 1.10
        and v40["js_geometric_ratio"] <= 1.00
    ) or (
        v40["attention_geometric_ratio"] >= 1.00
        and v40["js_geometric_ratio"] <= 0.90
    )
    tasks = len({str(row["instance_id"]) for row in rows})
    common_tasks = len({str(row["instance_id"]) for row in selected_rows})
    gates = {
        "minimum_cases": len(rows) >= MIN_CASES,
        "minimum_tasks": tasks >= MIN_TASKS,
        "two_stage_coverage_vs_v40": coverage >= 0.70,
        "common_cases": len(selected_rows) >= 12,
        "common_tasks": common_tasks >= 6,
        "attention_ratio_vs_probe": probe["attention_geometric_ratio"] >= 1.10,
        "attention_higher_fraction_vs_probe": probe["attention_higher_pair_fraction"] >= 0.60,
        "js_ratio_vs_probe": probe["js_geometric_ratio"] <= 1.00,
        "js_ratio_vs_path": path["js_geometric_ratio"] <= 0.90,
        "js_lower_fraction_vs_path": path["js_lower_pair_fraction"] >= 0.60,
        "pareto_vs_v40": pareto,
        "equal_copy_budget_on_common_cases": all(
            int(select(row, arm)["length"]) == 128
            for row in selected_rows
            for arm in (
                "v40_fixed_budget_recency",
                "path_only",
                "probe_only",
                "seeded_random",
                "two_stage",
            )
        ),
    }
    eligibility = len(rows) >= MIN_CASES and tasks >= MIN_TASKS
    decision = (
        "INSUFFICIENT_TASK_DISJOINT_COHORT"
        if not eligibility
        else "SUPPORTED_MOTIVATION"
        if all(gates.values())
        else "NOT_SUPPORTED"
    )
    value = {
        "status": "COMPLETE",
        "decision": decision,
        "cases": len(rows),
        "tasks": tasks,
        "two_stage_selected_cases": len(selected_rows),
        "two_stage_coverage": coverage,
        "comparisons": comparisons,
        "frozen_gate_results": gates,
        "next_step": (
            "freeze a runtime design; do not silently activate it"
            if decision == "SUPPORTED_MOTIVATION"
            else "do not implement this selector or tune the frozen threshold"
        ),
        "scope": "task-disjoint single-island causal motivation; not task accuracy or TTFT",
    }
    write_json(output / "RESULT.json", value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    prepare_parser.add_argument("--trajectory-root", type=Path, default=FRESH_ROOT)
    prepare_parser.add_argument("--limit", type=int, default=24)
    measure_parser = subparsers.add_parser("measure")
    measure_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    measure_parser.add_argument("--max-cases", type=int, default=0)
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.command == "prepare":
        value = prepare(args.output, args.trajectory_root, args.limit)
    elif args.command == "measure":
        value = measure(args.output, args.max_cases)
    else:
        value = analyze(args.output)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
