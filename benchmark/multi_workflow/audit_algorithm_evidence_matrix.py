#!/usr/bin/env python3
"""Build a hash-backed adoption/falsification matrix for coding-aware KV work."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path("/home/gfy/CodeMAS_Project")
ARTIFACTS = ROOT / "kvflow-artifacts"
REPORTS = ROOT / "kvflow-reports"
DEFAULT_OUTPUT = ARTIFACTS / "impactkv_algorithm_evidence_matrix_20260805"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _fresh_accuracy_evidence_status(result: dict[str, Any]) -> str:
    """Do not turn an all-zero equality into positive accuracy evidence."""
    aggregate = result.get("aggregate", {})
    resolved = aggregate.get("resolved", {})
    complete = int(aggregate.get("complete_tasks", 0))
    if complete > 0 and resolved and max(int(value) for value in resolved.values()) == 0:
        return "INCONCLUSIVE_ZERO_POWER"
    return str(result["status"])


def build() -> dict[str, Any]:
    invalid_m51 = (
        ARTIFACTS
        / "impactkv_m51_file_version_risk_20260805/matched18/INVALID_DESIGN.json"
    )
    valid_m51 = (
        ARTIFACTS
        / "impactkv_m51_file_version_risk_20260805/matched18_v2/RESULT.json"
    )
    if not invalid_m51.exists():
        raise FileNotFoundError("M51 invalid-design tombstone is missing")
    if _json(valid_m51).get("decision") != "NOT_SUPPORTED":
        raise AssertionError("valid M51 verdict changed")

    paths = {
        "early_audit": REPORTS
        / "weekly_reports_20260718/2026-07-21_IMPACTKV_KVFLOW_WEEKLY_RESEARCH_AUDIT_REVISION.md",
        "v11": ARTIFACTS
        / "impactkv_sessiongraph_v11_20260717/P0_FINAL_VERDICT.md",
        "v12": ARTIFACTS
        / "impactkv_probehead_v12_20260717/DEVELOPMENT_CALIBRATION_REPORT.json",
        "e2": ARTIFACTS
        / "impactkv_exact_middle_e2_20260718/server/E2_RESULT.json",
        "p_failure": REPORTS
        / "weekly_reports_20260722/2026-07-22_IMPACTKV_CODING_AWARE_WEEKLY_FAILURE_ANALYSIS.md",
        "p27c": ARTIFACTS
        / "impactkv_task_capsule_p27c_budget_grid_20260722/P27C_DEVELOPMENT_RESULT.json",
        "p27e": ARTIFACTS
        / "impactkv_task_capsule_p27e_confirmatory_20260722/P27E_CONFIRMATORY_RESULT.json",
        "v44": ARTIFACTS
        / "impactkv_v44_dense_sensitive_v40_20260728/V44_RESULT.json",
        "v45": ARTIFACTS
        / "impactkv_v45_versioned_evidence_20260803/V45_DECISION.json",
        "v46": ARTIFACTS
        / "impactkv_v46_accuracy_speed_20260803/V46_ACCURACY_SPEED_RESULT.json",
        "m47": ARTIFACTS
        / "impactkv_m47_task_conditioned_pool_20260805/full50/RESULT.json",
        "m49": ARTIFACTS
        / "impactkv_m49_probe_proxy_20260805/FINAL_RESULT.json",
        "m50": ARTIFACTS
        / "impactkv_m50_coding_provenance_20260805/matched20/RESULT.json",
        "m51": valid_m51,
        "m52": ARTIFACTS
        / "impactkv_m52_path_dependency_20260805/matched20/RESULT.json",
        "m53": ARTIFACTS
        / "impactkv_m53_path_dependency_holdout_20260805/request_disjoint19/RESULT.json",
        "m54": ARTIFACTS
        / "impactkv_m54_dependency_drift_hybrid_20260805/untouched14/RESULT.json",
        "v88_v92": ARTIFACTS
        / "impactkv_codemas_v2_controlled_sota_20260729/v92_online_kv_risk_speed_route/FINAL_RESULT.json",
    }
    optional_paths = {
        "m55_fresh_accuracy": ARTIFACTS
        / "impactkv_m55_v40_task_disjoint_20260805/M55_TASK_RESULT.json",
        "m55_two_stage": ARTIFACTS
        / "impactkv_m55_two_stage_20260805/fresh13/RESULT.json",
        "m56_same_prompt": ARTIFACTS
        / "impactkv_m56_v40_same_prompt_20260805/fresh13/RESULT.json",
    }
    paths.update(
        {name: path for name, path in optional_paths.items() if path.exists()}
    )
    sources = {name: _source(path) for name, path in paths.items()}
    m49 = _json(paths["m49"])
    m50 = _json(paths["m50"])
    m51 = _json(paths["m51"])
    m52 = _json(paths["m52"])
    m53 = _json(paths["m53"])
    m54 = _json(paths["m54"])
    if m49.get("status") != "FALSIFIED_PROXY":
        raise AssertionError("M49 request-level verdict changed")
    if m50.get("decision") != "NOT_SUPPORTED":
        raise AssertionError("M50 verdict changed")
    if m51.get("decision") != "NOT_SUPPORTED":
        raise AssertionError("M51 verdict changed")
    if m52.get("dependency_decision") != "SUPPORTED":
        raise AssertionError("M52 dependency verdict changed")
    if m53.get("decision") != "NOT_REPLICATED":
        raise AssertionError("M53 combined verdict changed")
    if m54.get("decision") != "NOT_SUPPORTED":
        raise AssertionError("M54 verdict changed")

    rows = [
        {
            "family": "TaskCone / early AST runtime",
            "hypothesis": "task-local or AST-local placement directly gives safe, fast KV reuse",
            "status": "INVALID_MEASUREMENT",
            "direct_evidence": "early body-offset/RoPE/zero-gap and launcher defects invalidate positive headlines",
            "later_validation": "none; results remain withdrawn",
            "sources": ["early_audit"],
        },
        {
            "family": "ASTSpanKV / AST-IslandKV",
            "hypothesis": "control-flow criticality predicts repair utility",
            "status": "LATENCY_FAIL",
            "direct_evidence": "ASTSpan fragmentation dominates TTFT; bounded islands still do not beat Dense speed",
            "later_validation": "M47 coding-symbol selection loses to recency at equal copy budget",
            "sources": ["p_failure", "m47"],
        },
        {
            "family": "V9/V10 workflow and session modules",
            "hypothesis": "immutable non-prefix modules provide enough reuse capacity",
            "status": "CAPACITY_FAIL",
            "direct_evidence": "corrected eligible capacity is below frozen gates",
            "later_validation": "no later experiment rescues the capacity premise",
            "sources": ["v11"],
        },
        {
            "family": "V11/V45 FileVersion",
            "hypothesis": "path-scoped versioning prevents semantically stale source views",
            "status": "SUPPORTED_COMPONENT_ONLY",
            "direct_evidence": "capacity and target-time invalidation are useful correctness guards",
            "later_validation": "M51 rejects mutation as a physical K/V-risk score; guard remains correctness-only",
            "sources": ["v11", "v45", "m51"],
        },
        {
            "family": "V12 ProbeHead",
            "hypothesis": "small dynamic K/V probes can identify safe reusable bodies",
            "status": "CAPACITY_FAIL",
            "direct_evidence": "4,639 configurations contain no capacity/harm jointly feasible point",
            "later_validation": "M49 supports the probe concept for single-island ranking, but falsifies request-level three-island aggregation",
            "sources": ["v12", "m49"],
        },
        {
            "family": "E2 shifted middle-span executor",
            "hypothesis": "K can be RoPE-shifted and V copied into a middle span correctly",
            "status": "SUPPORTED_COMPONENT_ONLY",
            "direct_evidence": "120/120 identity observations, zero logprob difference and zero fallback",
            "later_validation": "all later physical splice experiments depend on this mechanism",
            "sources": ["e2"],
        },
        {
            "family": "P3-P26 static coding repair",
            "hypothesis": "syntax/symbol/component/layer heuristics beat equal-cost tail",
            "status": "FALSIFIED_MECHANISM",
            "direct_evidence": "no stable equal-cost win; P23 oracle ceiling is below its gate",
            "later_validation": "M48 shows model-state drift is more predictive than attention or syntax alone; M47 again rejects lexical selection",
            "sources": ["p_failure", "m47"],
        },
        {
            "family": "P27 function capsule",
            "hypothesis": "a few complete retrieved functions are sufficient context",
            "status": "GENERALIZATION_FAIL",
            "direct_evidence": "development gain reverses on independent P27E; reduced-context Dense is the failing component",
            "later_validation": "M52 supports interaction-level path dependency, not function-only context sufficiency",
            "sources": ["p27c", "p27e", "m52"],
        },
        {
            "family": "V31-V38 lifecycle abstention",
            "hypothesis": "mutation/test/diff phases directly identify unsafe target requests",
            "status": "FALSIFIED_MECHANISM",
            "direct_evidence": "rules increase Dense abstention without stable accuracy gain over General",
            "later_validation": "M50/M51 reject block type and mutation as uniform physical-risk proxies",
            "sources": ["p_failure", "m50", "m51"],
        },
        {
            "family": "V40 grounded single observation",
            "hypothesis": "natural successful read-only observations form a useful conservative reuse boundary",
            "status": "RESEARCH_BASELINE",
            "direct_evidence": "V44 point estimate 4/12 versus Dense/General 3/12 with fewer copied tokens",
            "later_validation": "M50 removes the automatic-safety interpretation; M49 single-island risk and M52/M53 path utility motivate refinement",
            "sources": ["v44", "m49", "m50", "m52", "m53"],
        },
        {
            "family": "V46 bounded multi-observation pool",
            "hypothesis": "more valid natural observations improve speed without unacceptable quality loss",
            "status": "NOT_PROMOTED",
            "direct_evidence": "pool/executor increases copy opportunity and TTFT speed, but preservation canary is 2/3 versus V40/Dense 3/3",
            "later_validation": "M47 explains speed by contiguity/position; M49 and M54 do not validate composed-risk control",
            "sources": ["v46", "m47", "m49", "m54"],
        },
        {
            "family": "M52/M53 path dependency",
            "hypothesis": "recent path overlap identifies observations the model currently uses",
            "status": "SUPPORTED_COMPONENT_ONLY",
            "direct_evidence": "attention direction replicates on request-disjoint data",
            "later_validation": "safety directions do not fully replicate; path is utility, not a Dense guard",
            "sources": ["m52", "m53"],
        },
        {
            "family": "M54 path-weighted drift",
            "hypothesis": "dependency times drift is a better scalar risk score",
            "status": "FALSIFIED_MECHANISM",
            "direct_evidence": "hybrid JS Spearman is below probe-only and pair accuracy is unchanged",
            "later_validation": "forces lexicographic utility/risk separation in M55",
            "sources": ["m54"],
        },
        {
            "family": "V88-V92 CacheBlend-derived route",
            "hypothesis": "coding-conditioned layer/ratio routing can improve the controlled baseline",
            "status": "SUPPORTED_COMPONENT_ONLY",
            "direct_evidence": "positive point estimates but no conventional paired significance and high source-build break-even",
            "later_validation": "online state risk is retained as mechanism evidence; architecture is not the SGLang V40 line",
            "sources": ["v88_v92"],
        },
    ]
    if "m55_fresh_accuracy" in paths:
        result = _json(paths["m55_fresh_accuracy"])
        raw_status = str(result["status"])
        evidence_status = _fresh_accuracy_evidence_status(result)
        rows.append(
            {
                "family": "M55 fresh-13 V40 task rationale",
                "hypothesis": (
                    "V40 preserves official task quality at lower exposure on "
                    "a task-disjoint coding-agent cohort"
                ),
                "status": evidence_status,
                "direct_evidence": (
                    "fresh official container evaluation with frozen Dense, "
                    f"General, and V40 arms; protocol status={raw_status}"
                ),
                "later_validation": (
                    "all-zero arms have no accuracy-identifying power; speed "
                    "is delegated to M56"
                ),
                "sources": ["m55_fresh_accuracy"],
            }
        )
    if "m55_two_stage" in paths:
        result = _json(paths["m55_two_stage"])
        rows.append(
            {
                "family": "M55 risk-filtered path selector",
                "hypothesis": (
                    "filtering by frozen probe risk before path-utility ranking "
                    "beats either signal alone at equal single-island budget"
                ),
                "status": str(result["decision"]),
                "direct_evidence": (
                    "task-disjoint 128-token physical-splice attention/JS audit"
                ),
                "later_validation": (
                    "motivation only; cannot promote a multi-island runtime"
                ),
                "sources": ["m55_two_stage"],
            }
        )
    if "m56_same_prompt" in paths:
        result = _json(paths["m56_same_prompt"])
        rows.append(
            {
                "family": "M56 V40 same-prompt speed replay",
                "hypothesis": (
                    "V40 reduces TTFT when Dense and reuse receive identical "
                    "frozen prompt token IDs"
                ),
                "status": str(result["decision"]),
                "direct_evidence": (
                    "one-token paired TTFT/fidelity replay with source build "
                    "reported separately"
                ),
                "later_validation": "speed evidence only; not task accuracy",
                "sources": ["m56_same_prompt"],
            }
        )
    return {
        "status": "COMPLETE",
        "schema_version": 1,
        "rows": rows,
        "sources": sources,
        "protected": {
            "invalid_m51_excluded": True,
            "invalid_m51_tombstone": _source(invalid_m51),
            "paper_modified": False,
            "old_preregistration_thresholds_modified": False,
        },
        "interpretation": (
            "A failed row falsifies the frozen tested claim, not every possible "
            "future algorithm in the broad family."
        ),
    }


def render_markdown(value: dict[str, Any]) -> str:
    lines = [
        "# Coding-aware KV algorithm evidence matrix",
        "",
        value["interpretation"],
        "",
        "| Algorithm family | Frozen claim | Status | Later motivation effect |",
        "|---|---|---|---|",
    ]
    for row in value["rows"]:
        lines.append(
            f"| {row['family']} | {row['hypothesis']} | `{row['status']}` | "
            f"{row['later_validation']} |"
        )
    lines.extend(["", "## Hash-backed sources", ""])
    for name, source in value["sources"].items():
        lines.append(f"- `{name}`: `{source['sha256']}` — `{source['path']}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    value = build()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "ALGORITHM_EVIDENCE_MATRIX.json").write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output / "ALGORITHM_EVIDENCE_MATRIX.md").write_text(
        render_markdown(value), encoding="utf-8"
    )
    print(json.dumps({"status": value["status"], "rows": len(value["rows"])}, indent=2))


if __name__ == "__main__":
    main()
