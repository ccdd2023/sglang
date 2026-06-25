#!/usr/bin/env python3
"""Aggregate prompt-fair lossy reuse runs into a per-case Pareto calibration table."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _rule_shape(shape: Any) -> dict[str, int]:
    if not isinstance(shape, dict):
        return {}
    normalized = {str(k): int(v) for k, v in shape.items()}
    if normalized.get("bridge_prefix", 0) > 0:
        return {"bridge_prefix": normalized["bridge_prefix"]}
    if normalized.get("bridge_window", 0) > 0:
        return {"bridge_window": normalized["bridge_window"]}
    return normalized


def _current_shape(row: dict[str, Any]) -> dict[str, int]:
    return _rule_shape(
        row.get("current_selected_span_count_by_granularity")
        or row.get("best_acceptable_selected_span_count_by_granularity")
        or {}
    )


def _current_estimated_reused_tokens(row: dict[str, Any]) -> float:
    return _safe_float(
        row.get("current_estimated_reused_tokens")
        if row.get("current_estimated_reused_tokens") is not None
        else row.get("best_acceptable_estimated_reused_tokens")
    )


def _matches_rule_window(
    row: dict[str, Any],
    shape: dict[str, int],
    min_tokens: int,
    max_tokens: int,
    require_path_mentioned: bool = False,
    anchor_name_regexes: list[str] | None = None,
) -> bool:
    row_shape = _current_shape(row)
    if row_shape != shape:
        return False
    if require_path_mentioned and not row.get("current_any_anchor_path_mentioned"):
        return False
    if anchor_name_regexes:
        anchor_names = [str(name) for name in row.get("current_selected_anchor_names") or []]
        if not any(
            re.search(pattern, name)
            for pattern in anchor_name_regexes
            for name in anchor_names
        ):
            return False
    estimated = _current_estimated_reused_tokens(row)
    return min_tokens <= estimated <= max_tokens


def _exact_anchor_name_regexes(row: dict[str, Any]) -> list[str]:
    names = [str(name) for name in row.get("current_selected_anchor_names") or [] if name]
    return [f"^{re.escape(name)}$" for name in names]


def _is_known_risky(row: dict[str, Any], threshold: float) -> bool:
    if row.get("cap_sensitive"):
        return True
    fastest_unsafe_f1 = _safe_float(row.get("fastest_unsafe_f1"), default=1.0)
    return fastest_unsafe_f1 < threshold


def _case_rows(summary_path: Path, label: str, mode: str) -> list[dict[str, Any]]:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    out: list[dict[str, Any]] = []
    for case in data.get("cases", []):
        rows = {row.get("mode"): row for row in case.get("rows", [])}
        ref = rows.get("lossless_full_prefill")
        cand = rows.get(mode)
        if not ref or not cand:
            continue
        ref_ttft = _safe_float(ref.get("ttft_ms"))
        cand_ttft = _safe_float(cand.get("ttft_ms"))
        speedup = ref_ttft / cand_ttft if ref_ttft > 0 and cand_ttft > 0 else 0.0
        out.append(
            {
                "run_label": label,
                "summary_path": str(summary_path),
                "instance_id": case.get("instance_id"),
                "repo": case.get("repo"),
                "prompt_fair_ok": bool(case.get("prompt_fair_ok", cand.get("prompt_fair_ok", True))),
                "ttft_ms": cand.get("ttft_ms"),
                "lossless_ttft_ms": ref.get("ttft_ms"),
                "paired_speedup": round(speedup, 4),
                "token_f1": cand.get("output_token_f1_vs_lossless"),
                "accuracy_bucket": cand.get("accuracy_bucket"),
                "exact_output_match": cand.get("output_exact_match_vs_lossless"),
                "suffix_copy_len": cand.get("lossy_anchor_suffix_copy_len") or 0,
                "suffix_copy_planned_len": cand.get("lossy_anchor_suffix_copy_planned_len") or 0,
                "gap_recompute_len": cand.get("lossy_anchor_gap_recompute_len") or 0,
                "estimated_reused_tokens": cand.get("estimated_reused_tokens") or 0,
                "payload_anchor_count": cand.get("payload_anchor_count") or 0,
                "payload_anchor_token_count": cand.get("payload_anchor_token_count") or 0,
                "payload_anchor_max_total_rejected": cand.get("payload_anchor_max_total_rejected"),
                "selected_span_count_by_granularity": cand.get("selected_span_count_by_granularity") or {},
                "decision_reason_counts": cand.get("decision_reason_counts") or {},
                "hybrid_graph_token_estimate": cand.get("hybrid_graph_token_estimate"),
                "hybrid_risk_gate_rejected": cand.get("hybrid_risk_gate_rejected"),
                "hybrid_risk_gate_reason": cand.get("hybrid_risk_gate_reason"),
            }
        )
    return out


def _load_selection_features(path: Path | None, mode: str) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, Any]] = {}
    for row in data.get("rows", []):
        if row.get("mode") != mode or row.get("status") != "ok":
            continue
        out[str(row.get("instance_id"))] = row
    return out


def _load_rows_by_instance(path: Path | None, key: str = "instance_id") -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        return {str(row.get(key)): row for row in csv.DictReader(f) if row.get(key)}


def _load_labeled_rows(specs: list[str] | None, key: str = "instance_id") -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for spec in specs or []:
        if "=" not in spec:
            raise SystemExit(f"labeled row input must be LABEL=PATH, got {spec!r}")
        label, raw_path = spec.split("=", 1)
        rows = _load_rows_by_instance(Path(raw_path), key=key)
        for instance_id, row in rows.items():
            out[(label, instance_id)] = row
    return out


def _attach_posthoc(
    row: dict[str, Any],
    code_action_row: dict[str, Any],
    gold_intent_row: dict[str, Any],
    require_code_action_threshold: float,
    require_no_gold_intent_regression: bool,
    max_gold_intent_regression: float,
) -> None:
    if code_action_row:
        row.update(
            {
                "code_action_score": _safe_float(code_action_row.get("code_action_score")),
                "code_action_file_containment": _safe_float(code_action_row.get("file_containment")),
                "code_action_identifier_containment": _safe_float(code_action_row.get("identifier_containment")),
            }
        )
    if gold_intent_row:
        row.update(
            {
                "gold_intent_delta": _safe_float(gold_intent_row.get("gold_intent_delta")),
                "candidate_gold_intent_score": _safe_float(gold_intent_row.get("candidate_gold_intent_score")),
                "candidate_gold_file_containment": _safe_float(gold_intent_row.get("candidate_gold_file_containment")),
                "candidate_gold_identifier_containment": _safe_float(gold_intent_row.get("candidate_gold_identifier_containment")),
            }
        )
    sanity_reasons = []
    if require_code_action_threshold > 0:
        score = row.get("code_action_score")
        if score is None or _safe_float(score) < require_code_action_threshold:
            sanity_reasons.append("code_action_below_threshold")
    if require_no_gold_intent_regression:
        delta = row.get("gold_intent_delta")
        if delta is None or _safe_float(delta) < -max_gold_intent_regression:
            sanity_reasons.append("gold_intent_regression")
    row["posthoc_sanity_ok"] = not sanity_reasons
    row["posthoc_sanity_reasons"] = sanity_reasons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help="LABEL=path/to/summary.json")
    parser.add_argument("--mode", default="hybrid_code_aware_lossy")
    parser.add_argument("--token-f1-threshold", type=float, default=0.90)
    parser.add_argument("--min-speedup", type=float, default=1.05)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--emit-policy", action="store_true")
    parser.add_argument("--emit-rule-policy", action="store_true",
                        help="Emit a conservative shape/token-window policy that does not match on instance_id.")
    parser.add_argument("--selection-features", type=Path,
                        help="selection_features.json from --dry-run-selection-features; enables task/anchor overlap rule features.")
    parser.add_argument("--rule-token-margin-ratio", type=float, default=0.05,
                        help="Relative margin around best acceptable estimated_reused_tokens for generated rules.")
    parser.add_argument("--rule-anchor-name-regex", action="store_true",
                        help="Add exact selected-anchor-name regexes to generated rule matches.")
    parser.add_argument("--reject-cap-sensitive", action="store_true")
    parser.add_argument("--allow-cap-sensitive-case", action="append", default=[])
    parser.add_argument("--code-action-rows", type=Path,
                        help="Optional code_action_overlap_rows.csv. When provided, per-case code-action scores are added to calibration rows.")
    parser.add_argument("--gold-intent-rows", type=Path,
                        help="Optional gold_patch_intent_rows.csv. When provided, per-case gold-intent deltas are added to calibration rows.")
    parser.add_argument("--code-action-run", action="append", default=[],
                        help="Optional LABEL=code_action_overlap_rows.csv. Use when aggregating multiple run-specific posthoc files.")
    parser.add_argument("--gold-intent-run", action="append", default=[],
                        help="Optional LABEL=gold_patch_intent_rows.csv. Use when aggregating multiple run-specific posthoc files.")
    parser.add_argument("--require-code-action-threshold", type=float, default=0.0,
                        help="If >0, only rows with code_action_score >= threshold can be recommended as safe-speedup.")
    parser.add_argument("--require-no-gold-intent-regression", action="store_true",
                        help="Require gold_intent_delta >= -max-gold-intent-regression for safe-speedup recommendations.")
    parser.add_argument("--max-gold-intent-regression", type=float, default=0.10)
    args = parser.parse_args()
    selection_features = _load_selection_features(args.selection_features, args.mode)
    code_action_rows = _load_rows_by_instance(args.code_action_rows)
    gold_intent_rows = _load_rows_by_instance(args.gold_intent_rows)
    code_action_rows_by_run = _load_labeled_rows(args.code_action_run)
    gold_intent_rows_by_run = _load_labeled_rows(args.gold_intent_run)

    all_rows: list[dict[str, Any]] = []
    for spec in args.run:
        if "=" not in spec:
            raise SystemExit(f"--run must be LABEL=PATH, got {spec!r}")
        label, raw_path = spec.split("=", 1)
        all_rows.extend(_case_rows(Path(raw_path), label, args.mode))

    for row in all_rows:
        label = str(row.get("run_label"))
        instance_id = str(row.get("instance_id"))
        _attach_posthoc(
            row,
            code_action_rows_by_run.get((label, instance_id)) or code_action_rows.get(instance_id) or {},
            gold_intent_rows_by_run.get((label, instance_id)) or gold_intent_rows.get(instance_id) or {},
            args.require_code_action_threshold,
            args.require_no_gold_intent_regression,
            args.max_gold_intent_regression,
        )

    by_case: dict[str, list[dict[str, Any]]] = {}
    for row in all_rows:
        by_case.setdefault(str(row["instance_id"]), []).append(row)

    summaries: list[dict[str, Any]] = []
    for instance_id, rows in sorted(by_case.items()):
        acceptable = [
            row
            for row in rows
            if row["prompt_fair_ok"]
            and _safe_float(row.get("token_f1")) >= args.token_f1_threshold
            and row.get("posthoc_sanity_ok", True)
        ]
        acceptable_before_posthoc = [
            row
            for row in rows
            if row["prompt_fair_ok"]
            and _safe_float(row.get("token_f1")) >= args.token_f1_threshold
        ]
        copied = [row for row in rows if _safe_float(row.get("suffix_copy_len")) > 0]
        unsafe = [
            row
            for row in copied
            if _safe_float(row.get("token_f1")) < args.token_f1_threshold
        ]
        best_acceptable = max(acceptable, key=lambda row: _safe_float(row.get("paired_speedup")), default=None)
        best_acceptable_before_posthoc = max(
            acceptable_before_posthoc,
            key=lambda row: _safe_float(row.get("paired_speedup")),
            default=None,
        )
        fastest = max(rows, key=lambda row: _safe_float(row.get("paired_speedup")), default=None)
        fastest_unsafe = max(unsafe, key=lambda row: _safe_float(row.get("paired_speedup")), default=None)
        cap_sensitive = False
        copied_by_len: dict[int, list[float]] = {}
        for row in copied:
            copied_by_len.setdefault(int(row.get("suffix_copy_len") or 0), []).append(_safe_float(row.get("token_f1")))
        if len(copied_by_len) > 1:
            max_f1 = max(score for scores in copied_by_len.values() for score in scores)
            min_f1 = min(score for scores in copied_by_len.values() for score in scores)
            cap_sensitive = max_f1 - min_f1 >= 0.20
        summary_row = {
            "instance_id": instance_id,
            "repo": rows[0].get("repo"),
            "n_runs": len(rows),
            "n_copied_runs": len(copied),
            "n_acceptable_runs": len(acceptable),
            "n_acceptable_runs_before_posthoc": len(acceptable_before_posthoc),
            "best_acceptable_before_posthoc_speedup": (
                best_acceptable_before_posthoc.get("paired_speedup")
                if best_acceptable_before_posthoc
                else 0
            ),
            "best_acceptable_run": best_acceptable.get("run_label") if best_acceptable else "",
            "best_acceptable_speedup": best_acceptable.get("paired_speedup") if best_acceptable else 0,
            "best_acceptable_f1": best_acceptable.get("token_f1") if best_acceptable else "",
            "best_acceptable_copy_len": best_acceptable.get("suffix_copy_len") if best_acceptable else 0,
            "best_acceptable_estimated_reused_tokens": best_acceptable.get("estimated_reused_tokens") if best_acceptable else 0,
            "best_acceptable_selected_span_count_by_granularity": best_acceptable.get("selected_span_count_by_granularity") if best_acceptable else {},
            "fastest_run": fastest.get("run_label") if fastest else "",
            "fastest_speedup": fastest.get("paired_speedup") if fastest else 0,
            "fastest_f1": fastest.get("token_f1") if fastest else "",
            "fastest_copy_len": fastest.get("suffix_copy_len") if fastest else 0,
            "fastest_unsafe_run": fastest_unsafe.get("run_label") if fastest_unsafe else "",
            "fastest_unsafe_speedup": fastest_unsafe.get("paired_speedup") if fastest_unsafe else 0,
            "fastest_unsafe_f1": fastest_unsafe.get("token_f1") if fastest_unsafe else "",
            "cap_sensitive": cap_sensitive,
            "recommended_bucket": (
                "safe-speedup"
                if best_acceptable and _safe_float(best_acceptable.get("paired_speedup")) >= args.min_speedup
                else "safe-no-speedup"
                if best_acceptable
                else "no-acceptable-copy"
            ),
        }
        feature_row = selection_features.get(instance_id) or {}
        if feature_row:
            summary_row.update(
                {
                    "current_selected_span_count_by_granularity": feature_row.get("selected_span_count_by_granularity") or {},
                    "current_estimated_reused_tokens": feature_row.get("estimated_reused_tokens") or 0,
                    "current_selected_anchor_names": feature_row.get("selected_anchor_names") or [],
                    "current_any_anchor_path_mentioned": bool(feature_row.get("any_anchor_path_mentioned")),
                    "current_any_anchor_basename_mentioned": bool(feature_row.get("any_anchor_basename_mentioned")),
                    "current_max_anchor_lexical_overlap": feature_row.get("max_anchor_lexical_overlap") or 0,
                    "current_max_anchor_symbol_overlap": feature_row.get("max_anchor_symbol_overlap") or 0,
                }
            )
        code_action_row = code_action_rows.get(instance_id) or {}
        if best_acceptable:
            summary_row.update(
                {
                    "code_action_score": best_acceptable.get("code_action_score"),
                    "code_action_file_containment": best_acceptable.get("code_action_file_containment"),
                    "code_action_identifier_containment": best_acceptable.get("code_action_identifier_containment"),
                }
            )
        elif code_action_row:
            summary_row.update(
                {
                    "code_action_score": _safe_float(code_action_row.get("code_action_score")),
                    "code_action_file_containment": _safe_float(code_action_row.get("file_containment")),
                    "code_action_identifier_containment": _safe_float(code_action_row.get("identifier_containment")),
                }
            )
        gold_intent_row = gold_intent_rows.get(instance_id) or {}
        if best_acceptable:
            summary_row.update(
                {
                    "gold_intent_delta": best_acceptable.get("gold_intent_delta"),
                    "candidate_gold_intent_score": best_acceptable.get("candidate_gold_intent_score"),
                    "candidate_gold_file_containment": best_acceptable.get("candidate_gold_file_containment"),
                    "candidate_gold_identifier_containment": best_acceptable.get("candidate_gold_identifier_containment"),
                }
            )
        elif gold_intent_row:
            summary_row.update(
                {
                    "gold_intent_delta": _safe_float(gold_intent_row.get("gold_intent_delta")),
                    "candidate_gold_intent_score": _safe_float(gold_intent_row.get("candidate_gold_intent_score")),
                    "candidate_gold_file_containment": _safe_float(gold_intent_row.get("candidate_gold_file_containment")),
                    "candidate_gold_identifier_containment": _safe_float(gold_intent_row.get("candidate_gold_identifier_containment")),
                }
            )
        sanity_reasons = []
        if args.require_code_action_threshold > 0:
            score = summary_row.get("code_action_score")
            if score is None or _safe_float(score) < args.require_code_action_threshold:
                sanity_reasons.append("code_action_below_threshold")
        if args.require_no_gold_intent_regression:
            delta = summary_row.get("gold_intent_delta")
            if delta is None or _safe_float(delta) < -args.max_gold_intent_regression:
                sanity_reasons.append("gold_intent_regression")
        summary_row["posthoc_sanity_ok"] = not sanity_reasons
        summary_row["posthoc_sanity_reasons"] = sanity_reasons
        if (
            summary_row["recommended_bucket"] in {"safe-no-speedup", "no-acceptable-copy"}
            and best_acceptable_before_posthoc
            and _safe_float(best_acceptable_before_posthoc.get("paired_speedup")) >= args.min_speedup
            and len(acceptable_before_posthoc) > len(acceptable)
        ):
            summary_row["recommended_bucket"] = "safe-speedup-posthoc-reject"
        summaries.append(summary_row)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "mode": args.mode,
        "token_f1_threshold": args.token_f1_threshold,
        "min_speedup": args.min_speedup,
        "n_cases": len(summaries),
        "safe_speedup_cases": sum(1 for row in summaries if row["recommended_bucket"] == "safe-speedup"),
        "posthoc_rejected_speedup_cases": sum(1 for row in summaries if row["recommended_bucket"] == "safe-speedup-posthoc-reject"),
        "safe_no_speedup_cases": sum(1 for row in summaries if row["recommended_bucket"] == "safe-no-speedup"),
        "no_acceptable_copy_cases": sum(1 for row in summaries if row["recommended_bucket"] == "no-acceptable-copy"),
        "cap_sensitive_cases": [row["instance_id"] for row in summaries if row["cap_sensitive"]],
        "posthoc_sanity_required": {
            "require_code_action_threshold": args.require_code_action_threshold,
            "require_no_gold_intent_regression": args.require_no_gold_intent_regression,
            "max_gold_intent_regression": args.max_gold_intent_regression,
        },
        "runs": args.run,
    }
    (args.out_dir / "pareto_calibration_summary.json").write_text(
        json.dumps({"summary": report, "cases": summaries, "rows": all_rows}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if args.emit_policy:
        cases: dict[str, dict[str, Any]] = {}
        cap_sensitive_allowlist = set(args.allow_cap_sensitive_case or [])
        for row in summaries:
            copy_len = int(row.get("best_acceptable_copy_len") or 0)
            speedup = _safe_float(row.get("best_acceptable_speedup"))
            cap_sensitive_rejected = (
                args.reject_cap_sensitive
                and bool(row.get("cap_sensitive"))
                and row["instance_id"] not in cap_sensitive_allowlist
            )
            if (
                not cap_sensitive_rejected
                and row.get("recommended_bucket") == "safe-speedup"
                and copy_len > 0
                and speedup >= args.min_speedup
            ):
                cases[row["instance_id"]] = {
                    "action": "cap",
                    "max_suffix_copy_len": copy_len,
                    "source_run": row.get("best_acceptable_run"),
                    "source_speedup": speedup,
                    "source_token_f1": row.get("best_acceptable_f1"),
                    "required_selected_span_count_by_granularity": row.get("best_acceptable_selected_span_count_by_granularity") or {},
                    "reason": "best_acceptable_prompt_fair_calibration",
                }
            else:
                cases[row["instance_id"]] = {
                    "action": "reject",
                    "source_run": row.get("best_acceptable_run"),
                    "source_speedup": speedup,
                    "source_token_f1": row.get("best_acceptable_f1"),
                    "reason": "cap_sensitive_default_reject" if cap_sensitive_rejected else "no_calibrated_safe_speedup",
                }
        policy = {
            "policy_name": "hybrid_prompt_fair_pareto_calibrated_diagnostic",
            "mode": args.mode,
            "token_f1_threshold": args.token_f1_threshold,
            "min_speedup": args.min_speedup,
            "diagnostic_only": True,
            "reject_cap_sensitive": args.reject_cap_sensitive,
            "cap_sensitive_allowlist": sorted(cap_sensitive_allowlist),
            "notes": (
                "Generated from existing prompt-fair runs. Use only as calibration/upper diagnostic; "
                "do not report as held-out generalization."
            ),
            "runs": args.run,
            "cases": cases,
        }
        (args.out_dir / "hybrid_calibration_policy.json").write_text(
            json.dumps(policy, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    if args.emit_rule_policy:
        rules: list[dict[str, Any]] = []
        skipped_conflicting_rules: list[dict[str, Any]] = []
        cap_sensitive_allowlist = set(args.allow_cap_sensitive_case or [])
        for row in summaries:
            copy_len = int(row.get("best_acceptable_copy_len") or 0)
            estimated_reused = _current_estimated_reused_tokens(row)
            speedup = _safe_float(row.get("best_acceptable_speedup"))
            shape = _current_shape(row)
            if row.get("recommended_bucket") != "safe-speedup":
                continue
            if copy_len <= 0 or estimated_reused <= 0 or not shape:
                continue
            if (
                args.reject_cap_sensitive
                and row.get("cap_sensitive")
                and row["instance_id"] not in cap_sensitive_allowlist
            ):
                continue
            margin = max(0.0, args.rule_token_margin_ratio)
            min_tokens = max(1, int(math.floor(estimated_reused * (1.0 - margin))))
            max_tokens = int(math.ceil(estimated_reused * (1.0 + margin)))
            require_path_mentioned = bool(row.get("current_any_anchor_path_mentioned"))
            anchor_name_regexes = _exact_anchor_name_regexes(row) if args.rule_anchor_name_regex else []
            conflicts = [
                other["instance_id"]
                for other in summaries
                if other["instance_id"] != row["instance_id"]
                and _is_known_risky(other, args.token_f1_threshold)
                and _matches_rule_window(
                    other,
                    shape,
                    min_tokens,
                    max_tokens,
                    require_path_mentioned,
                    anchor_name_regexes,
                )
            ]
            if conflicts:
                skipped_conflicting_rules.append(
                    {
                        "source_case": row["instance_id"],
                        "conflicts": conflicts,
                        "shape": shape,
                        "min_estimated_reused_tokens": min_tokens,
                        "max_estimated_reused_tokens": max_tokens,
                        "require_anchor_path_mentioned": require_path_mentioned,
                        "selected_anchor_name_any_regex": anchor_name_regexes,
                    }
                )
                continue
            match = {
                "selected_span_count_by_granularity": shape,
                "min_estimated_reused_tokens": min_tokens,
                "max_estimated_reused_tokens": max_tokens,
            }
            if require_path_mentioned:
                match["require_anchor_path_mentioned"] = True
            if anchor_name_regexes:
                match["selected_anchor_name_any_regex"] = anchor_name_regexes
            rules.append(
                {
                    "name": f"calibrated_{row['instance_id']}",
                    "action": "cap",
                    "max_suffix_copy_len": copy_len,
                    "reason": "shape_token_window_calibration",
                    "source_case": row["instance_id"],
                    "source_run": row.get("best_acceptable_run"),
                    "source_speedup": speedup,
                    "source_token_f1": row.get("best_acceptable_f1"),
                    "match": match,
                }
            )
        rule_policy = {
            "policy_name": "hybrid_prompt_fair_shape_token_window_calibrated",
            "mode": args.mode,
            "token_f1_threshold": args.token_f1_threshold,
            "min_speedup": args.min_speedup,
            "diagnostic_only": True,
            "not_instance_id_matched": True,
            "default_action": "reject",
            "default_reason": "no_shape_token_window_rule",
            "rule_token_margin_ratio": args.rule_token_margin_ratio,
            "rule_anchor_name_regex": args.rule_anchor_name_regex,
            "reject_cap_sensitive": args.reject_cap_sensitive,
            "cap_sensitive_allowlist": sorted(cap_sensitive_allowlist),
            "notes": (
                "Generated from existing prompt-fair runs. Rules match only observable selection "
                "shape, estimated reused-token windows, and optional task-anchor overlap; "
                "evaluate on held-out cases before reporting."
            ),
            "runs": args.run,
            "rules": rules,
            "skipped_conflicting_rules": skipped_conflicting_rules,
        }
        (args.out_dir / "hybrid_rule_calibration_policy.json").write_text(
            json.dumps(rule_policy, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    with (args.out_dir / "pareto_calibration_cases.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "instance_id",
            "repo",
            "n_runs",
            "n_copied_runs",
            "n_acceptable_runs",
            "best_acceptable_run",
            "best_acceptable_speedup",
            "best_acceptable_f1",
            "best_acceptable_copy_len",
            "best_acceptable_estimated_reused_tokens",
            "best_acceptable_selected_span_count_by_granularity",
            "fastest_run",
            "fastest_speedup",
            "fastest_f1",
            "fastest_copy_len",
            "fastest_unsafe_run",
            "fastest_unsafe_speedup",
            "fastest_unsafe_f1",
            "cap_sensitive",
            "code_action_score",
            "gold_intent_delta",
            "posthoc_sanity_ok",
            "posthoc_sanity_reasons",
            "recommended_bucket",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
