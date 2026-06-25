#!/usr/bin/env python3
"""Whole-file transfer with selective AST-span KV reuse.

Agents still receive complete code_base files. The benchmark changes only the
internal reuse objects: lossless full prefill, whole-file reuse-all diagnostic,
selective function/method reuse, and oracle-low-dnorm selective reuse.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import csv
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import aiohttp
from transformers import AutoTokenizer

PROJECT = Path(__file__).resolve().parents[2]
for entry in (str(PROJECT.parent / "MAScoder" / "src"), str(PROJECT), str(PROJECT / "python")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from benchmark.multi_workflow.bench_swe_generated_patch_kvcomm import (  # noqa: E402
    DEFAULT_PYTHON,
    CodeSegment,
    build_anchor_fields,
    build_codebase_prefetch_hints,
    extract_cached_tokens,
    extract_lossy_meta,
    extract_text,
    git_commit,
    kill_port,
    load_graph_bundle_segments,
    now_ms,
    post_chat,
    post_chat_optional_stream,
    wait_ready,
)
from benchmark.multi_workflow.selective_ast_reuse import (  # noqa: E402
    DEFAULT_POLICY_PATH,
    load_selective_policy,
    select_spans,
    split_python_file,
    summarize_selection,
)

DEFAULT_MODEL = "/home/gfy/models/Qwen2.5-7B-Instruct"
DEFAULT_DATASET = PROJECT / "results" / "repo_level_datasets" / "swe_verified_10_instances.json"
DEFAULT_MANIFEST = PROJECT / "results" / "repo_level_datasets" / "manifest_10.json"
OUT_DIR = PROJECT / "results" / "selective_ast_reuse" / "wholefile_smoke"

BASE_MODES = [
    "lossless_full_prefill",
    "whole_file_reuse_all",
    "selective_function_method_reuse",
    "selective_extended_reuse",
    "selective_oracle_low_dnorm",
]

GRAPH_AWARE_MODE = "graph_aware_lossy"
HYBRID_CODE_AWARE_MODE = "hybrid_code_aware_lossy"
GRAPH_SEGMENT_MARKER = "::graph::"

CSV_FIELDNAMES = [
    "instance_id",
    "repo",
    "warmup_protocol",
    "mode",
    "elapsed_ms",
    "ttft_ms",
    "cached_tokens",
    "estimated_reused_tokens",
    "estimated_recomputed_tokens",
    "selected_span_count",
    "selected_anchor_names",
    "span_count",
    "payload_reuse_mode",
    "payload_initial_anchor_count",
    "payload_anchor_count",
    "payload_anchor_token_count",
    "payload_prompt_resident_anchor_count",
    "payload_token_filter_dropped_count",
    "payload_anchor_start_filter_dropped_count",
    "payload_anchor_min_total_rejected",
    "payload_anchor_max_total_rejected",
    "payload_anchor_max_total_pruned_count",
    "lossy_match_reason",
    "lossy_rejected_reason",
    "lossy_reuse_allowed",
    "lossy_candidate_count",
    "matched_content_signature",
    "lossy_anchor_match_used",
    "lossy_anchor_match_len",
    "lossy_anchor_multi_copy_count",
    "lossy_anchor_match_gap_len",
    "lossy_anchor_gap_recompute_len",
    "lossy_anchor_suffix_copy_len",
    "lossy_anchor_suffix_copy_planned_len",
    "lossy_anchor_suffix_copy_cap_len",
    "lossy_anchor_suffix_copy_truncated",
    "lossy_anchor_suffix_copy_semantic_len",
    "lossy_anchor_suffix_copy_semantic_min_cosine",
    "lossy_anchor_suffix_copy_semantic_truncated",
    "lossy_anchor_suffix_recompute_head_len",
    "lossy_anchor_context_copy_ready",
    "lossy_anchor_context_aligned",
    "lossy_anchor_context_align_fail_reason",
    "lossy_anchor_context_align_stage",
    "lossy_anchor_context_target_prefix_len",
    "lossy_anchor_context_prefix_signature_match",
    "lossy_anchor_rope_delta",
    "lossy_anchor_store_entry_count",
    "lossy_anchor_store_token_count",
    "lossy_anchor_store_lookup_entries",
    "lossy_anchor_match_fail_reason",
    "lossy_anchor_token_mismatch_count",
    "lossy_anchor_span_shape_mismatch_count",
    "lossy_anchor_prefix_covers_count",
    "agenttemplatekv_prefetch_hit_count",
    "agenttemplatekv_prefetch_protected_tokens",
    "agenttemplatekv_prefetch_newly_protected_tokens",
    "codebase_prefetch_matched_tokens",
    "graph_anchor_budget_applied",
    "graph_anchor_selected_tokens",
    "graph_anchor_filtered_long_count",
    "graph_anchor_budget_dropped_count",
    "selection_gate_rejected",
    "selection_gate_reason",
    "hybrid_calibration_policy_applied",
    "hybrid_calibration_action",
    "hybrid_calibration_rule_name",
    "hybrid_calibration_reason",
    "hybrid_calibration_max_suffix_copy_len",
    "hybrid_calibration_bridge_window_synthesized",
    "hybrid_calibration_bridge_window_max_tokens",
    "hybrid_calibration_bridge_window_seed_count",
    "hybrid_calibration_rejected",
    "hybrid_calibration_shape_mismatch",
    "hybrid_calibration_shape_pruned",
    "output_token_f1_vs_lossless",
    "token_f1_drop",
    "accuracy_bucket",
    "reuse_policy_name",
    "warmup_status",
    "target_prompt_sha1",
    "target_prompt_chars",
    "warmup_prompt_sha1",
    "warmup_prompt_chars",
    "prompt_fair_ok",
]

WARMUP_PROTOCOLS = ("none", "oracle_per_mode", "natural_planner", "fair_planner_per_mode")


def accuracy_bucket(token_f1_value: float | None, threshold: float = 0.90) -> str:
    if token_f1_value is None:
        return "unknown"
    if token_f1_value >= 0.99995:
        return "strict-safe"
    if token_f1_value >= threshold:
        return "lossy-acceptable"
    return "aggressive-diagnostic"


def reuse_policy_name(args: argparse.Namespace, mode: str) -> str:
    if mode == "lossless_full_prefill":
        return mode
    if args.lossy_stage_recompute_gap:
        return "context_aligned_stage_diag"
    if args.lossy_recompute_gap:
        return "context_aligned_safe"
    if args.enable_bridge_prefix_anchors:
        return "bridge_prefix_diag"
    if args.lossy_max_zero_gap and int(args.lossy_max_zero_gap) > 4096:
        return "large_gap_diag"
    if mode == GRAPH_AWARE_MODE:
        return "graph_aware_lossy"
    return mode


def apply_graph_anchor_copy_caps(
    args: argparse.Namespace,
    tokenizer: Any,
    mode: str,
    selected_segments: list[CodeSegment],
    payload: dict[str, Any],
) -> None:
    if mode != GRAPH_AWARE_MODE:
        return
    low_cap = int(getattr(args, "graph_anchor_lowspan_suffix_copy_cap", 0) or 0)
    low_max_tokens = int(getattr(args, "graph_anchor_lowspan_max_tokens", 0) or 0)
    small_cap = int(getattr(args, "graph_anchor_smallspan_suffix_copy_cap", 0) or 0)
    small_max_tokens = int(getattr(args, "graph_anchor_smallspan_max_tokens", 0) or 0)
    mid_cap = int(getattr(args, "graph_anchor_midspan_suffix_copy_cap", 0) or 0)
    min_tokens = int(getattr(args, "graph_anchor_midspan_min_tokens", 0) or 0)
    max_tokens = int(getattr(args, "graph_anchor_midspan_max_tokens", 0) or 0)
    if low_cap <= 0 and small_cap <= 0 and mid_cap <= 0:
        return
    token_spans = payload.get("code_anchor_token_spans") or []
    for segment, token_span in zip(selected_segments, token_spans):
        token_count = len(tokenizer(segment.text, add_special_tokens=False).input_ids)
        if low_cap > 0 and low_max_tokens > 0 and token_count <= low_max_tokens:
            token_span["max_suffix_copy_len"] = low_cap
            continue
        if small_cap > 0 and small_max_tokens > 0 and token_count <= small_max_tokens:
            token_span["max_suffix_copy_len"] = small_cap
            continue
        if mid_cap <= 0:
            continue
        if min_tokens > 0 and token_count < min_tokens:
            continue
        if max_tokens > 0 and token_count > max_tokens:
            continue
        token_span["max_suffix_copy_len"] = mid_cap


def apply_anchor_tier_copy_caps(
    args: argparse.Namespace,
    tokenizer: Any,
    mode: str,
    selected_segments: list[CodeSegment],
    payload: dict[str, Any],
) -> None:
    if mode == GRAPH_AWARE_MODE:
        return
    low_cap = int(getattr(args, "anchor_lowspan_suffix_copy_cap", 0) or 0)
    low_max_tokens = int(getattr(args, "anchor_lowspan_max_tokens", 0) or 0)
    small_cap = int(getattr(args, "anchor_smallspan_suffix_copy_cap", 0) or 0)
    small_max_tokens = int(getattr(args, "anchor_smallspan_max_tokens", 0) or 0)
    mid_cap = int(getattr(args, "anchor_midspan_suffix_copy_cap", 0) or 0)
    mid_min_tokens = int(getattr(args, "anchor_midspan_min_tokens", 0) or 0)
    mid_max_tokens = int(getattr(args, "anchor_midspan_max_tokens", 0) or 0)
    if low_cap <= 0 and small_cap <= 0 and mid_cap <= 0:
        return
    token_spans = payload.get("code_anchor_token_spans") or []
    for segment, token_span in zip(selected_segments, token_spans):
        token_count = len(tokenizer(segment.text, add_special_tokens=False).input_ids)
        if low_cap > 0 and low_max_tokens > 0 and token_count <= low_max_tokens:
            token_span["max_suffix_copy_len"] = low_cap
            continue
        if small_cap > 0 and small_max_tokens > 0 and token_count <= small_max_tokens:
            token_span["max_suffix_copy_len"] = small_cap
            continue
        if mid_cap <= 0:
            continue
        if mid_min_tokens > 0 and token_count < mid_min_tokens:
            continue
        if mid_max_tokens > 0 and token_count > mid_max_tokens:
            continue
        token_span["max_suffix_copy_len"] = mid_cap


def load_hybrid_calibration_policy(args: argparse.Namespace) -> dict[str, Any]:
    path = getattr(args, "hybrid_calibration_policy", None)
    if not path:
        return {}
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"hybrid calibration policy not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"hybrid calibration policy must be a JSON object: {path}")
    return data


CASE_SELECTOR_OVERRIDE_FIELDS = {
    "files_per_case",
    "file_start_index",
    "max_file_chars",
    "max_complete_file_chars",
    "prefer_selective_files",
    "prefer_graph_target_files",
    "enable_bridge_prefix_anchors",
    "bridge_anchor_max_tokens",
    "disable_graph_bridge_prefix_anchors",
    "hybrid_min_bridge_tokens",
    "hybrid_max_bridge_tokens",
    "hybrid_bridge_anchor_max_tokens",
    "hybrid_bridge_max_count_per_file",
    "include_hybrid_bridge_seed_spans",
    "hybrid_bridge_source",
    "hybrid_task_ast_top_k",
    "hybrid_risk_large_bridge_min_tokens",
    "hybrid_risk_max_large_bridge_count",
    "hybrid_risk_max_graph_tokens_for_large_bridge",
    "selective_anchor_min_span_tokens",
    "selective_anchor_max_span_tokens",
    "selective_anchor_max_start_token",
    "anchor_min_total_tokens",
    "anchor_max_total_tokens",
    "anchor_max_total_policy",
    "selection_min_estimated_reused_tokens",
    "exclude_anchor_granularities",
    "graph_anchor_token_budget",
    "graph_anchor_max_span_tokens",
    "graph_anchor_lowspan_max_tokens",
    "graph_anchor_lowspan_suffix_copy_cap",
    "graph_anchor_smallspan_max_tokens",
    "graph_anchor_smallspan_suffix_copy_cap",
    "graph_anchor_midspan_min_tokens",
    "graph_anchor_midspan_max_tokens",
    "graph_anchor_midspan_suffix_copy_cap",
    "anchor_lowspan_max_tokens",
    "anchor_lowspan_suffix_copy_cap",
    "anchor_smallspan_max_tokens",
    "anchor_smallspan_suffix_copy_cap",
    "anchor_midspan_min_tokens",
    "anchor_midspan_max_tokens",
    "anchor_midspan_suffix_copy_cap",
    "lossy_max_planned_suffix_copy_len",
    "lossy_max_suffix_copy_len",
}


def load_case_selector_overrides(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    path = getattr(args, "case_selector_overrides", None)
    if not path:
        return {}
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"case selector overrides not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"case selector overrides must be a JSON object: {path}")
    cases = data.get("cases", data)
    if not isinstance(cases, dict):
        raise ValueError("case selector overrides must contain a 'cases' object or be an instance_id map")
    out: dict[str, dict[str, Any]] = {}
    for instance_id, overrides in cases.items():
        if not isinstance(overrides, dict):
            raise ValueError(f"selector override for {instance_id} must be an object")
        unknown = set(overrides) - CASE_SELECTOR_OVERRIDE_FIELDS
        if unknown:
            raise ValueError(
                f"unsupported selector override fields for {instance_id}: {sorted(unknown)}"
            )
        out[str(instance_id)] = dict(overrides)
    return out


def args_for_case_selector(args: argparse.Namespace, instance_id: str) -> argparse.Namespace:
    overrides = (getattr(args, "_case_selector_overrides_data", {}) or {}).get(instance_id)
    if not overrides:
        return args
    case_args = copy.copy(args)
    for key, value in overrides.items():
        setattr(case_args, key, value)
    return case_args


def file_sha1(path: str | Path | None) -> str:
    if not path:
        return ""
    p = Path(path)
    if not p.exists() or not p.is_file():
        return ""
    return hashlib.sha1(p.read_bytes()).hexdigest()


def selector_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    """Fields that materially change which prompt-resident anchors are selected."""
    policy_path = getattr(args, "policy", "")
    dataset_path = getattr(args, "dataset", "")
    manifest_path = getattr(args, "manifest", "")
    return {
        "git_commit": git_commit(),
        "policy": str(policy_path),
        "policy_sha1": file_sha1(policy_path),
        "dataset": str(dataset_path),
        "dataset_sha1": file_sha1(dataset_path),
        "manifest": str(manifest_path),
        "manifest_sha1": file_sha1(manifest_path),
        "max_file_chars": getattr(args, "max_file_chars", None),
        "max_complete_file_chars": getattr(args, "max_complete_file_chars", None),
        "server_random_seed": getattr(args, "server_random_seed", None),
        "target_modes": active_modes(args),
        "enable_graph_aware_lossy": getattr(args, "enable_graph_aware_lossy", False),
        "load_graph_bundles_for_selection": getattr(args, "load_graph_bundles_for_selection", False),
        "enable_hybrid_code_aware_lossy": getattr(args, "enable_hybrid_code_aware_lossy", False),
        "hybrid_min_bridge_tokens": getattr(args, "hybrid_min_bridge_tokens", 0),
        "hybrid_max_bridge_tokens": getattr(args, "hybrid_max_bridge_tokens", 0),
        "hybrid_bridge_anchor_max_tokens": getattr(args, "hybrid_bridge_anchor_max_tokens", 0),
        "hybrid_bridge_max_count_per_file": getattr(args, "hybrid_bridge_max_count_per_file", 0),
        "include_hybrid_bridge_seed_spans": getattr(args, "include_hybrid_bridge_seed_spans", False),
        "hybrid_bridge_source": getattr(args, "hybrid_bridge_source", "function"),
        "hybrid_task_ast_top_k": getattr(args, "hybrid_task_ast_top_k", 3),
        "hybrid_risk_large_bridge_min_tokens": getattr(args, "hybrid_risk_large_bridge_min_tokens", 0),
        "hybrid_risk_max_large_bridge_count": getattr(args, "hybrid_risk_max_large_bridge_count", 0),
        "hybrid_risk_max_graph_tokens_for_large_bridge": getattr(args, "hybrid_risk_max_graph_tokens_for_large_bridge", 0),
        "enable_bridge_prefix_anchors": getattr(args, "enable_bridge_prefix_anchors", False),
        "bridge_anchor_max_tokens": getattr(args, "bridge_anchor_max_tokens", 0),
        "disable_graph_bridge_prefix_anchors": getattr(args, "disable_graph_bridge_prefix_anchors", False),
        "selection_min_estimated_reused_tokens": getattr(args, "selection_min_estimated_reused_tokens", 0),
        "selective_anchor_min_span_tokens": getattr(args, "selective_anchor_min_span_tokens", 0),
        "selective_anchor_max_span_tokens": getattr(args, "selective_anchor_max_span_tokens", 0),
        "selective_anchor_max_start_token": getattr(args, "selective_anchor_max_start_token", 0),
        "anchor_min_total_tokens": getattr(args, "anchor_min_total_tokens", 0),
        "anchor_max_total_tokens": getattr(args, "anchor_max_total_tokens", 0),
        "anchor_max_total_policy": getattr(args, "anchor_max_total_policy", ""),
        "exclude_anchor_granularities": getattr(args, "exclude_anchor_granularities", ""),
        "graph_anchor_token_budget": getattr(args, "graph_anchor_token_budget", 0),
        "graph_anchor_max_span_tokens": getattr(args, "graph_anchor_max_span_tokens", 0),
        "lossy_max_planned_suffix_copy_len": getattr(args, "lossy_max_planned_suffix_copy_len", 0),
        "case_selector_overrides": str(getattr(args, "case_selector_overrides", "") or ""),
        "case_selector_override_cases": len(getattr(args, "_case_selector_overrides_data", {}) or {}),
    }


def calibration_entry_for_case(args: argparse.Namespace, instance_id: str, mode: str) -> dict[str, Any]:
    if mode != HYBRID_CODE_AWARE_MODE:
        return {}
    policy = getattr(args, "_hybrid_calibration_policy_data", None) or {}
    cases = policy.get("cases") or {}
    entry = cases.get(instance_id) or {}
    if entry and not isinstance(entry, dict):
        raise ValueError(f"invalid hybrid calibration entry for {instance_id}: {entry!r}")
    return entry


def _rule_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    return [str(value)]


def _rule_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def calibration_rule_matches(rule: dict[str, Any], selection: dict[str, Any]) -> bool:
    match = rule.get("match") or {}
    if not isinstance(match, dict):
        raise ValueError(f"hybrid calibration rule match must be an object: {rule!r}")
    selected_shape = {
        str(k): int(v)
        for k, v in (selection.get("selected_span_count_by_granularity") or {}).items()
    }
    required_exact_shape = match.get("selected_span_count_by_granularity")
    if isinstance(required_exact_shape, dict):
        expected_shape = {str(k): int(v) for k, v in required_exact_shape.items()}
        if selected_shape != expected_shape:
            return False
    for granularity in _rule_list(match.get("contains_granularities")):
        if selected_shape.get(granularity, 0) <= 0:
            return False
    for granularity in _rule_list(match.get("forbid_granularities")):
        if selected_shape.get(granularity, 0) > 0:
            return False
    selected_count = int(selection.get("selected_span_count") or 0)
    if "min_selected_span_count" in match and selected_count < int(match["min_selected_span_count"]):
        return False
    if "max_selected_span_count" in match and selected_count > int(match["max_selected_span_count"]):
        return False
    estimated_reused = _rule_float(selection.get("estimated_reused_tokens"))
    if "min_estimated_reused_tokens" in match and estimated_reused < _rule_float(match["min_estimated_reused_tokens"]):
        return False
    if "max_estimated_reused_tokens" in match and estimated_reused > _rule_float(match["max_estimated_reused_tokens"]):
        return False
    if "min_anchor_lexical_overlap" in match and _rule_float(selection.get("max_anchor_lexical_overlap")) < _rule_float(match["min_anchor_lexical_overlap"]):
        return False
    if "min_anchor_symbol_overlap" in match and _rule_float(selection.get("max_anchor_symbol_overlap")) < _rule_float(match["min_anchor_symbol_overlap"]):
        return False
    if match.get("require_anchor_path_mentioned") and not selection.get("any_anchor_path_mentioned"):
        return False
    if match.get("require_anchor_basename_mentioned") and not selection.get("any_anchor_basename_mentioned"):
        return False
    decision_reasons = set((selection.get("decision_reason_counts") or {}).keys())
    any_reasons = set(_rule_list(match.get("decision_reason_any")))
    if any_reasons and not (decision_reasons & any_reasons):
        return False
    all_reasons = set(_rule_list(match.get("decision_reason_all")))
    if all_reasons and not all_reasons.issubset(decision_reasons):
        return False
    reject_reasons = set(_rule_list(match.get("decision_reason_none")))
    if reject_reasons and (decision_reasons & reject_reasons):
        return False
    anchor_names = [str(name) for name in selection.get("selected_anchor_names") or []]
    any_name_patterns = _rule_list(match.get("selected_anchor_name_any_regex"))
    if any_name_patterns and not any(
        re.search(pattern, name)
        for pattern in any_name_patterns
        for name in anchor_names
    ):
        return False
    all_name_patterns = _rule_list(match.get("selected_anchor_name_all_regex"))
    if all_name_patterns and not all(
        any(re.search(pattern, name) for name in anchor_names)
        for pattern in all_name_patterns
    ):
        return False
    none_name_patterns = _rule_list(match.get("selected_anchor_name_none_regex"))
    if none_name_patterns and any(
        re.search(pattern, name)
        for pattern in none_name_patterns
        for name in anchor_names
    ):
        return False
    return True


def calibration_entry_for_selection(
    args: argparse.Namespace,
    instance_id: str,
    mode: str,
    selection: dict[str, Any],
) -> dict[str, Any]:
    entry = calibration_entry_for_case(args, instance_id, mode)
    if entry:
        return entry
    if mode != HYBRID_CODE_AWARE_MODE:
        return {}
    policy = getattr(args, "_hybrid_calibration_policy_data", None) or {}
    rules = policy.get("rules") or []
    if not isinstance(rules, list):
        raise ValueError("hybrid calibration policy rules must be a list")
    for rule in rules:
        if not isinstance(rule, dict):
            raise ValueError(f"hybrid calibration rule must be an object: {rule!r}")
        if calibration_rule_matches(rule, selection):
            action = str(rule.get("action", "allow"))
            return {
                "action": action,
                "max_suffix_copy_len": rule.get("max_suffix_copy_len"),
                "suffix_recompute_head_len": rule.get("suffix_recompute_head_len"),
                "reason": rule.get("reason", "rule_match"),
                "rule_name": rule.get("name", ""),
                "required_selected_span_count_by_granularity": rule.get(
                    "required_selected_span_count_by_granularity"
                ),
                "required_selected_anchor_name_any_regex": rule.get(
                    "required_selected_anchor_name_any_regex"
                ),
                "required_selected_anchor_name_all_regex": rule.get(
                    "required_selected_anchor_name_all_regex"
                ),
                "synthesize_bridge_window_max_tokens": rule.get(
                    "synthesize_bridge_window_max_tokens"
                ),
                "include_synthesized_bridge_seed_spans": rule.get(
                    "include_synthesized_bridge_seed_spans", True
                ),
            }
    default_action = policy.get("default_action")
    if default_action:
        return {
            "action": str(default_action),
            "reason": policy.get("default_reason", "default_action"),
            "rule_name": "__default__",
        }
    return {}


def apply_hybrid_calibration_to_selection(
    args: argparse.Namespace,
    instance_id: str,
    mode: str,
    selected_segments: list[CodeSegment],
    selection: dict[str, Any],
    whole_segments: list[CodeSegment] | None = None,
) -> tuple[list[CodeSegment], dict[str, Any]]:
    entry = calibration_entry_for_selection(args, instance_id, mode, selection)
    if not entry:
        return selected_segments, selection
    selection = dict(selection)
    action = str(entry.get("action", "allow"))
    selection["hybrid_calibration_policy_applied"] = True
    selection["hybrid_calibration_action"] = action
    selection["hybrid_calibration_rule_name"] = entry.get("rule_name", "")
    selection["hybrid_calibration_reason"] = entry.get("reason", "")
    selection["hybrid_calibration_max_suffix_copy_len"] = entry.get("max_suffix_copy_len")
    selection["hybrid_calibration_suffix_recompute_head_len"] = entry.get("suffix_recompute_head_len")
    required_any_anchor_patterns = _rule_list(entry.get("required_selected_anchor_name_any_regex"))
    required_all_anchor_patterns = _rule_list(entry.get("required_selected_anchor_name_all_regex"))
    if required_any_anchor_patterns or required_all_anchor_patterns:
        pruned_segments: list[CodeSegment] = []
        for segment in selected_segments:
            if required_any_anchor_patterns and any(
                re.search(pattern, segment.name) for pattern in required_any_anchor_patterns
            ):
                pruned_segments.append(segment)
                continue
            if required_all_anchor_patterns and any(
                re.search(pattern, segment.name) for pattern in required_all_anchor_patterns
            ):
                pruned_segments.append(segment)
        if required_all_anchor_patterns:
            missing_patterns = [
                pattern
                for pattern in required_all_anchor_patterns
                if not any(re.search(pattern, segment.name) for segment in pruned_segments)
            ]
            if missing_patterns:
                counts = dict(selection.get("decision_reason_counts") or {})
                counts["reject:hybrid_calibration_anchor_name_mismatch"] = 1
                selection["decision_reason_counts"] = counts
                selection["hybrid_calibration_rejected"] = True
                selection["hybrid_calibration_anchor_name_mismatch"] = True
                selection["hybrid_calibration_missing_anchor_name_patterns"] = missing_patterns
                selection["selected_span_count"] = 0
                selection["selected_span_count_by_granularity"] = {}
                selection["selected_anchor_names"] = []
                selection["estimated_reused_tokens"] = 0
                return [], selection
        selected_segments = pruned_segments
        counts: dict[str, int] = {}
        for segment in selected_segments:
            granularity = segment_granularity(segment.name)
            if granularity:
                counts[granularity] = counts.get(granularity, 0) + 1
        selection["hybrid_calibration_anchor_name_pruned"] = True
        selection["selected_span_count"] = len(selected_segments)
        selection["selected_span_count_by_granularity"] = counts
        selection["selected_anchor_names"] = [segment.name for segment in selected_segments]
        selection["estimated_reused_tokens"] = sum(
            max(1, len(segment.text.split()))
            for segment in selected_segments
        )
    bridge_window_max_tokens = int(entry.get("synthesize_bridge_window_max_tokens") or 0)
    if bridge_window_max_tokens > 0:
        if not whole_segments:
            counts = dict(selection.get("decision_reason_counts") or {})
            counts["reject:hybrid_calibration_bridge_window_no_whole_segments"] = 1
            selection["decision_reason_counts"] = counts
            selection["hybrid_calibration_rejected"] = True
            selection["hybrid_calibration_bridge_window_synth_failed"] = True
            selection["selected_span_count"] = 0
            selection["selected_span_count_by_granularity"] = {}
            selection["selected_anchor_names"] = []
            selection["estimated_reused_tokens"] = 0
            return [], selection
        seed_segments = list(selected_segments)
        bridge_segments = bridge_prefix_anchors(
            whole_segments,
            seed_segments,
            bridge_window_max_tokens,
        )
        synthesized = [
            segment
            for segment in bridge_segments
            if segment_granularity(segment.name) == "bridge_window"
        ]
        if not synthesized:
            counts = dict(selection.get("decision_reason_counts") or {})
            counts["reject:hybrid_calibration_bridge_window_synth_failed"] = 1
            selection["decision_reason_counts"] = counts
            selection["hybrid_calibration_rejected"] = True
            selection["hybrid_calibration_bridge_window_synth_failed"] = True
            selection["selected_span_count"] = 0
            selection["selected_span_count_by_granularity"] = {}
            selection["selected_anchor_names"] = []
            selection["estimated_reused_tokens"] = 0
            return [], selection
        selected_segments = synthesized
        if bool(entry.get("include_synthesized_bridge_seed_spans", True)):
            selected_segments = synthesized + seed_segments
        counts: dict[str, int] = {}
        for segment in selected_segments:
            granularity = segment_granularity(segment.name)
            if granularity:
                counts[granularity] = counts.get(granularity, 0) + 1
        selection["hybrid_calibration_bridge_window_synthesized"] = True
        selection["hybrid_calibration_bridge_window_max_tokens"] = bridge_window_max_tokens
        selection["hybrid_calibration_bridge_window_seed_count"] = len(seed_segments)
        selection["selected_span_count"] = len(selected_segments)
        selection["selected_span_count_by_granularity"] = counts
        selection["selected_anchor_names"] = [segment.name for segment in selected_segments]
        selection["estimated_reused_tokens"] = sum(
            max(1, len(segment.text.split()))
            for segment in selected_segments
        )
    required_shape = entry.get("required_selected_span_count_by_granularity")
    if isinstance(required_shape, dict):
        current_shape = selection.get("selected_span_count_by_granularity") or {}
        required_shape = {str(k): int(v) for k, v in required_shape.items()}
        current_shape = {str(k): int(v) for k, v in current_shape.items()}
        selection["hybrid_calibration_required_shape"] = required_shape
        if current_shape != required_shape:
            has_required_subset = all(
                current_shape.get(granularity, 0) >= count
                for granularity, count in required_shape.items()
            )
            if has_required_subset:
                pruned_segments: list[CodeSegment] = []
                remaining = dict(required_shape)
                for segment in selected_segments:
                    granularity = segment_granularity(segment.name)
                    if remaining.get(granularity, 0) <= 0:
                        continue
                    pruned_segments.append(segment)
                    remaining[granularity] -= 1
                if all(count == 0 for count in remaining.values()):
                    selected_segments = pruned_segments
                    selection["hybrid_calibration_shape_pruned"] = True
                    selection["selected_span_count"] = len(selected_segments)
                    selection["selected_span_count_by_granularity"] = dict(required_shape)
                    selection["selected_anchor_names"] = [
                        segment.name for segment in selected_segments
                    ]
                    selection["estimated_reused_tokens"] = sum(
                        max(1, len(segment.text.split()))
                        for segment in selected_segments
                    )
                else:
                    has_required_subset = False
            if not has_required_subset:
                counts = dict(selection.get("decision_reason_counts") or {})
                counts["reject:hybrid_calibration_shape_mismatch"] = 1
                selection["decision_reason_counts"] = counts
                selection["hybrid_calibration_rejected"] = True
                selection["hybrid_calibration_shape_mismatch"] = True
                selection["selected_span_count"] = 0
                selection["selected_span_count_by_granularity"] = {}
                selection["selected_anchor_names"] = []
                selection["estimated_reused_tokens"] = 0
                return [], selection
    if action == "reject":
        counts = dict(selection.get("decision_reason_counts") or {})
        counts["reject:hybrid_calibration_policy"] = 1
        selection["decision_reason_counts"] = counts
        selection["hybrid_calibration_rejected"] = True
        selection["selected_span_count"] = 0
        selection["selected_span_count_by_granularity"] = {}
        selection["selected_anchor_names"] = []
        selection["estimated_reused_tokens"] = 0
        return [], selection
    if action not in {"allow", "cap"}:
        raise ValueError(f"unsupported hybrid calibration action for {instance_id}: {action}")
    return selected_segments, selection


def apply_hybrid_calibration_to_payload(
    args: argparse.Namespace,
    instance_id: str | None,
    mode: str,
    payload: dict[str, Any],
    selection_telemetry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    selection_telemetry = selection_telemetry or {}
    if selection_telemetry.get("hybrid_calibration_policy_applied"):
        entry = {
            "action": selection_telemetry.get("hybrid_calibration_action", "allow"),
            "max_suffix_copy_len": selection_telemetry.get("hybrid_calibration_max_suffix_copy_len"),
            "suffix_recompute_head_len": selection_telemetry.get("hybrid_calibration_suffix_recompute_head_len"),
            "reason": selection_telemetry.get("hybrid_calibration_reason", ""),
            "rule_name": selection_telemetry.get("hybrid_calibration_rule_name", ""),
        }
    elif instance_id:
        entry = calibration_entry_for_case(args, instance_id, mode)
    else:
        entry = {}
    if not entry:
        return {}
    action = str(entry.get("action", "allow"))
    telemetry = {
        "hybrid_calibration_policy_applied": True,
        "hybrid_calibration_action": action,
        "hybrid_calibration_rule_name": entry.get("rule_name", ""),
        "hybrid_calibration_reason": entry.get("reason", ""),
        "hybrid_calibration_max_suffix_copy_len": entry.get("max_suffix_copy_len"),
        "hybrid_calibration_suffix_recompute_head_len": entry.get("suffix_recompute_head_len"),
    }
    if action == "cap":
        cap = int(entry.get("max_suffix_copy_len") or 0)
        head_len = int(entry.get("suffix_recompute_head_len") or 0)
        if cap > 0:
            for token_span in payload.get("code_anchor_token_spans") or []:
                current = int(token_span.get("max_suffix_copy_len") or 0)
                token_span["max_suffix_copy_len"] = min(current, cap) if current > 0 else cap
        if head_len > 0:
            for token_span in payload.get("code_anchor_token_spans") or []:
                current = int(token_span.get("suffix_recompute_head_len") or 0)
                token_span["suffix_recompute_head_len"] = max(current, head_len)
    return telemetry


async def flush_cache(session: aiohttp.ClientSession, port: int) -> bool:
    async with session.post(f"http://127.0.0.1:{port}/flush_cache") as resp:
        await resp.text()
        return resp.status == 200


def token_f1(a: str, b: str) -> float:
    aa = a.split()
    bb = b.split()
    if not aa and not bb:
        return 1.0
    if not aa or not bb:
        return 0.0
    from collections import Counter

    ca, cb = Counter(aa), Counter(bb)
    overlap = sum((ca & cb).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(aa)
    recall = overlap / len(bb)
    return 2 * precision * recall / (precision + recall)


def load_selection_graph_segments(args: argparse.Namespace) -> dict[str, list[CodeSegment]]:
    """Load graph bundles for internal anchor selection without enabling graph mode."""
    if not (
        getattr(args, "enable_graph_aware_lossy", False)
        or getattr(args, "load_graph_bundles_for_selection", False)
    ):
        return {}
    graph_args = copy.copy(args)
    graph_args.enable_graph_aware_lossy = True
    return load_graph_bundle_segments(graph_args)


def load_cases(args: argparse.Namespace, policy: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = json.loads(args.dataset.read_text(encoding="utf-8"))
    row_by_id = {row["instance_id"]: row for row in rows}
    graph_segments_by_case = load_selection_graph_segments(args)
    samples = manifest.get("samples") or []
    cases = []
    selected = samples[args.start_index : args.start_index + args.max_cases]
    if args.all_cases:
        selected = samples[args.start_index :]
    for sample in selected:
        instance_id = sample["instance_id"]
        case_args = args_for_case_selector(args, instance_id)
        instance = {**sample, **row_by_id.get(instance_id, {})}
        instance.setdefault("instance_id", instance_id)
        instance.setdefault("repo", sample.get("repo", ""))
        segments = []
        graph_segments = graph_segments_by_case.get(instance_id, [])
        graph_target_paths = list(dict.fromkeys(parse_graph_segment_name(segment.name)[0] for segment in graph_segments))
        source_file_infos = list(sample.get("files", []))
        if case_args.prefer_graph_target_files and graph_target_paths:
            known_paths = {file_info.get("path") for file_info in source_file_infos}
            repo_dir = PROJECT / "results" / "swebench_local_envs" / "repos" / instance_id
            for path in graph_target_paths:
                if path in known_paths:
                    continue
                local_path = repo_dir / path
                if local_path.exists():
                    source_file_infos.append({"path": path, "local_path": str(local_path)})
                    known_paths.add(path)
        file_infos = source_file_infos[
            case_args.file_start_index : case_args.file_start_index + case_args.files_per_case
        ]
        if case_args.prefer_selective_files and policy is not None:
            candidates = []
            for file_info in source_file_infos:
                local_path = Path(file_info.get("local_path", ""))
                if not local_path.exists() or local_path.suffix != ".py":
                    continue
                text = local_path.read_text(encoding="utf-8", errors="replace").rstrip()
                if case_args.max_complete_file_chars and len(text) > case_args.max_complete_file_chars:
                    continue
                spans = split_python_file(file_info["path"], text, policy)
                selected = select_spans(spans, "selective_function_method_reuse")
                if not selected:
                    continue
                candidates.append((-len(selected), len(text), file_info, text))
            candidates.sort()
            ordered = []
            used_paths = set()
            by_path = {item[2]["path"]: item for item in candidates}
            if case_args.prefer_graph_target_files:
                for path in graph_target_paths:
                    item = by_path.get(path)
                    if item is not None and path not in used_paths:
                        ordered.append(item)
                        used_paths.add(path)
                    if len(ordered) >= case_args.files_per_case:
                        break
            for item in candidates:
                path = item[2]["path"]
                if path in used_paths:
                    continue
                ordered.append(item)
                used_paths.add(path)
                if len(ordered) >= case_args.files_per_case:
                    break
            file_infos = [item[2] for item in ordered[: case_args.files_per_case]]
            text_by_path = {item[2]["path"]: item[3] for item in ordered[: case_args.files_per_case]}
        else:
            text_by_path = {}
        for file_info in file_infos:
            local_path = Path(file_info.get("local_path", ""))
            if not local_path.exists():
                continue
            text = text_by_path.get(file_info["path"]) or local_path.read_text(encoding="utf-8", errors="replace")
            file_max_chars = int(file_info.get("max_file_chars") or case_args.max_file_chars or 0)
            if file_max_chars and len(text) > file_max_chars:
                text = text[:file_max_chars]
            segments.append(CodeSegment(file_info["path"], text.rstrip()))
        if segments:
            cases.append(
                {
                    "instance": instance,
                    "segments": segments,
                    "graph_segments": graph_segments,
                    "target_paths": [],
                }
            )
    return cases


def active_modes(args: argparse.Namespace) -> list[str]:
    modes = list(BASE_MODES)
    if args.enable_graph_aware_lossy:
        modes.append(GRAPH_AWARE_MODE)
    if getattr(args, "enable_hybrid_code_aware_lossy", False):
        modes.append(HYBRID_CODE_AWARE_MODE)
    target_modes = [
        item.strip()
        for item in str(getattr(args, "target_modes", "") or "").split(",")
        if item.strip()
    ]
    if target_modes:
        unknown = [mode for mode in target_modes if mode not in modes]
        if unknown:
            raise ValueError(f"unknown --target-modes entries: {unknown}; available={modes}")
        modes = [mode for mode in modes if mode in set(target_modes)]
    return modes


def warmup_protocol_description(protocol: str) -> str:
    descriptions = {
        "none": "Cold baseline: flush once per case and run target requests without a warmup request.",
        "oracle_per_mode": "Controlled upper bound: flush before each mode, run that mode's own warmup, then measure target.",
        "natural_planner": "Realistic agent protocol: measure a cold lossless reference, flush, run one Planner-style warmup, then measure reuse target modes against shared cache.",
        "fair_planner_per_mode": "Prompt-fair mechanism protocol: for each mode, flush, run the same Planner warmup, then measure the same target prompt; only runtime reuse anchors differ.",
    }
    return descriptions.get(protocol, protocol)


def mode_order_for_protocol(args: argparse.Namespace) -> list[str]:
    modes = active_modes(args)
    if args.warmup_protocol in {"natural_planner", "fair_planner_per_mode"} and "lossless_full_prefill" in modes:
        return ["lossless_full_prefill"] + [mode for mode in modes if mode != "lossless_full_prefill"]
    return modes


def prompt_text_for_messages(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def prompt_telemetry(tokenizer: Any, messages: list[dict[str, str]]) -> dict[str, Any]:
    prompt_text = prompt_text_for_messages(tokenizer, messages)
    return {
        "prompt_sha1": hashlib.sha1(prompt_text.encode("utf-8")).hexdigest(),
        "prompt_chars": len(prompt_text),
    }


def normalize_for_graph_match(text: str) -> str:
    lines = textwrap.dedent(str(text).replace("\r\n", "\n")).split("\n")
    lines = [line.rstrip() for line in lines]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def non_overlapping_reuse_spans(spans: list[Any]) -> list[Any]:
    selected = []
    occupied: dict[str, list[tuple[int, int]]] = {}
    granularity_rank = {
        "method": 0,
        "function": 0,
        "control_block": 1,
        "class": 2,
        "file_prefix": 3,
        "statement_window": 4,
    }
    for span in sorted(
        spans,
        key=lambda item: (
            granularity_rank.get(getattr(item, "granularity", ""), 9),
            item.end_line - item.start_line,
            item.path,
            item.start_line,
        ),
    ):
        ranges = occupied.setdefault(span.path, [])
        if any(not (span.end_line < start or span.start_line > end) for start, end in ranges):
            continue
        selected.append(span)
        ranges.append((span.start_line, span.end_line))
    return selected


def graph_mapped_reuse_spans(
    whole_segments: list[CodeSegment],
    policy: dict[str, Any],
    graph_segments: list[CodeSegment] | None,
) -> list[Any]:
    if not graph_segments:
        return []
    all_spans = []
    for segment in whole_segments:
        all_spans.extend(split_python_file(segment.name, segment.text, policy))
    metadata_matches = graph_metadata_reuse_spans(all_spans, graph_segments)
    if metadata_matches:
        return non_overlapping_reuse_spans(metadata_matches)
    graph_texts = [normalize_for_graph_match(segment.text) for segment in graph_segments if segment.text.strip()]
    contained = []
    containers = []
    seen_contained = set()
    seen_containers = set()
    for span in all_spans:
        span_text = normalize_for_graph_match(span.text)
        if not span_text:
            continue
        key = (span.path, span.start_line, span.end_line, span.signature)
        if any(span_text in graph_text for graph_text in graph_texts):
            if key not in seen_contained:
                seen_contained.add(key)
                contained.append(span)
        elif any(graph_text in span_text for graph_text in graph_texts):
            if key not in seen_containers:
                seen_containers.add(key)
                containers.append(span)
    return non_overlapping_reuse_spans(contained or containers)


def encode_graph_segment_name(path: str, target_symbol: str, bundle_type: str) -> str:
    symbol = str(target_symbol or "").replace("::", ".").strip()
    bundle = str(bundle_type or "").replace("::", "_").strip()
    if not symbol and not bundle:
        return path
    return f"{path}{GRAPH_SEGMENT_MARKER}{symbol}::{bundle}"


def parse_graph_segment_name(name: str) -> tuple[str, str, str]:
    if GRAPH_SEGMENT_MARKER not in name:
        return name, "", ""
    path, rest = name.split(GRAPH_SEGMENT_MARKER, 1)
    parts = rest.rsplit("::", 1)
    if len(parts) == 1:
        return path, parts[0], ""
    return path, parts[0], parts[1]


def segment_granularity(name: str) -> str:
    base_name = parse_graph_segment_name(name)[0]
    parts = base_name.rsplit(":", 3)
    if len(parts) >= 4:
        return parts[-3]
    return ""


def apply_excluded_anchor_granularities(
    selected_segments: list[CodeSegment],
    selection: dict[str, Any],
    args: argparse.Namespace | None,
) -> tuple[list[CodeSegment], dict[str, Any]]:
    raw = str(getattr(args, "exclude_anchor_granularities", "") or "")
    excluded = {item.strip() for item in raw.split(",") if item.strip()}
    if not excluded:
        return selected_segments, selection
    filtered = [
        segment
        for segment in selected_segments
        if segment_granularity(segment.name) not in excluded
    ]
    dropped = len(selected_segments) - len(filtered)
    if dropped <= 0:
        return selected_segments, selection
    updated = dict(selection)
    updated["excluded_anchor_granularities"] = sorted(excluded)
    updated["excluded_anchor_count"] = dropped
    updated["selected_span_count"] = len(filtered)
    counts: dict[str, int] = {}
    for segment in filtered:
        granularity = segment_granularity(segment.name)
        if granularity:
            counts[granularity] = counts.get(granularity, 0) + 1
    updated["selected_span_count_by_granularity"] = counts
    updated["estimated_reused_tokens"] = sum(max(1, len(segment.text.split())) for segment in filtered)
    return filtered, updated


def attach_selected_anchor_names(
    selection: dict[str, Any],
    selected_segments: list[CodeSegment],
) -> dict[str, Any]:
    updated = dict(selection)
    updated["selected_anchor_names"] = [segment.name for segment in selected_segments]
    return updated


def attach_anchor_task_overlap(
    selection: dict[str, Any],
    instance: dict[str, Any],
) -> dict[str, Any]:
    anchor_names = [str(name) for name in selection.get("selected_anchor_names") or []]
    features = anchor_task_overlap_features(instance, anchor_names)
    updated = dict(selection)
    updated["selected_anchor_task_features"] = features
    updated["max_anchor_lexical_overlap"] = max((item["lexical_overlap"] for item in features), default=0)
    updated["any_anchor_path_mentioned"] = any(item["path_mentioned"] for item in features)
    updated["any_anchor_basename_mentioned"] = any(item["basename_mentioned"] for item in features)
    updated["max_anchor_symbol_overlap"] = max((item["symbol_overlap"] for item in features), default=0)
    return updated


def graph_metadata_reuse_spans(all_spans: list[Any], graph_segments: list[CodeSegment]) -> list[Any]:
    selected = []
    seen: set[tuple[str, int, int, str]] = set()
    for segment in graph_segments:
        target_file, target_symbol, bundle_type = parse_graph_segment_name(segment.name)
        if not target_file:
            continue
        symbol_tail = target_symbol.split(".")[-1] if target_symbol else ""
        file_spans = [span for span in all_spans if getattr(span, "path", "") == target_file]
        candidates = []
        if symbol_tail:
            for span in file_spans:
                span_symbol = str(getattr(span, "symbol", "") or getattr(span, "name", ""))
                if (
                    span_symbol == target_symbol
                    or span_symbol.endswith(f":{target_symbol}")
                    or span_symbol.endswith(f".{symbol_tail}")
                    or f":{symbol_tail}:" in str(getattr(span, "name", ""))
                ):
                    candidates.append(span)
        if not candidates and bundle_type == "ast_function_only":
            normalized_graph = normalize_for_graph_match(segment.text)
            candidates = [
                span for span in file_spans
                if normalize_for_graph_match(getattr(span, "text", "")) == normalized_graph
            ]
        for span in candidates:
            key = (span.path, span.start_line, span.end_line, span.signature)
            if key in seen:
                continue
            seen.add(key)
            selected.append(span)
    return selected


def span_token_estimate(span: Any) -> int:
    return max(1, len(str(getattr(span, "text", "") or "").split()))


def apply_graph_anchor_budget(spans: list[Any], args: argparse.Namespace | None) -> tuple[list[Any], dict[str, Any]]:
    if args is None or not spans:
        return spans, {
            "graph_anchor_budget_applied": False,
            "graph_anchor_filtered_long_count": 0,
            "graph_anchor_budget_dropped_count": 0,
        }
    max_span_tokens = int(getattr(args, "graph_anchor_max_span_tokens", 0) or 0)
    budget = int(getattr(args, "graph_anchor_token_budget", 0) or 0)
    if max_span_tokens <= 0 and budget <= 0:
        return spans, {
            "graph_anchor_budget_applied": False,
            "graph_anchor_filtered_long_count": 0,
            "graph_anchor_budget_dropped_count": 0,
        }

    granularity_rank = {
        "method": 0,
        "function": 0,
        "control_block": 1,
        "class": 2,
        "file_prefix": 3,
        "statement_window": 4,
    }
    ordered = sorted(
        spans,
        key=lambda span: (
            granularity_rank.get(getattr(span, "granularity", ""), 9),
            abs(span_token_estimate(span) - min(max_span_tokens or 512, 512)),
            getattr(span, "path", ""),
            getattr(span, "start_line", 0),
        ),
    )
    filtered = [
        span for span in ordered
        if max_span_tokens <= 0 or span_token_estimate(span) <= max_span_tokens
    ]
    if not filtered:
        return [], {
            "graph_anchor_budget_applied": True,
            "graph_anchor_filtered_long_count": len(spans),
            "graph_anchor_budget_dropped_count": 0,
            "graph_anchor_fallback_reason": "all_spans_over_max_span_tokens_skipped",
        }

    selected = []
    used = 0
    for span in filtered:
        tokens = span_token_estimate(span)
        if budget > 0 and selected and used + tokens > budget:
            continue
        selected.append(span)
        used += tokens
        if budget > 0 and used >= budget:
            break
    if not selected:
        selected = filtered[:1]
        used = span_token_estimate(selected[0])
    return selected, {
        "graph_anchor_budget_applied": True,
        "graph_anchor_filtered_long_count": len(ordered) - len(filtered),
        "graph_anchor_budget_dropped_count": len(filtered) - len(selected),
        "graph_anchor_budget_tokens": budget,
        "graph_anchor_max_span_tokens": max_span_tokens,
        "graph_anchor_selected_tokens": used,
    }


def natural_planner_anchor_segments(
    whole_segments: list[CodeSegment],
    policy: dict[str, Any],
    graph_segments: list[CodeSegment] | None,
) -> list[CodeSegment]:
    """Pick one shared, non-mode-specific anchor set for Planner warmup."""
    all_spans = []
    for segment in whole_segments:
        all_spans.extend(split_python_file(segment.name, segment.text, policy))

    graph_spans = graph_mapped_reuse_spans(whole_segments, policy, graph_segments)
    safe_ast_spans = [
        span
        for span in all_spans
        if span.reuse_decision == "reuse"
        and span.granularity in {"function", "method", "control_block", "file_prefix"}
    ]
    priority = {"method": 1, "function": 1, "control_block": 2, "file_prefix": 3}
    candidates = [(0, span) for span in graph_spans] + [
        (priority.get(span.granularity, 9), span) for span in safe_ast_spans
    ]
    selected = []
    occupied: dict[str, list[tuple[int, int]]] = {}
    seen: set[tuple[str, int, int, str]] = set()
    for _, span in sorted(candidates, key=lambda item: (item[0], item[1].path, item[1].end_line - item[1].start_line, item[1].start_line)):
        key = (span.path, span.start_line, span.end_line, span.signature)
        if key in seen:
            continue
        seen.add(key)
        ranges = occupied.setdefault(span.path, [])
        if any(not (span.end_line < start or span.start_line > end) for start, end in ranges):
            continue
        selected.append(span)
        ranges.append((span.start_line, span.end_line))
    selected.sort(key=lambda span: (span.path, span.start_line, span.end_line))
    return [CodeSegment(span.name, span.text) for span in selected]


def _parse_segment_line_range(segment_name: str) -> tuple[str, int, int] | None:
    parts = segment_name.rsplit(":", 3)
    if len(parts) != 4:
        return None
    path, _granularity, _symbol, line_range = parts
    match = re.fullmatch(r"(\d+)-(\d+)", line_range)
    if not match:
        return None
    return path, int(match.group(1)), int(match.group(2))


def _bounded_bridge_text(lines: list[str], start_line: int, end_line: int, max_tokens: int) -> tuple[int, str]:
    end = min(max(1, end_line), len(lines))
    start = min(max(1, start_line), end)
    selected = lines[start - 1 : end]
    selected_tokens = len("\n".join(selected).split())
    if max_tokens <= 0 or selected_tokens >= max_tokens:
        return start, "\n".join(selected).rstrip()

    budget = max_tokens - selected_tokens
    prefix_lines: list[str] = []
    prefix_tokens = 0
    for idx in range(start - 2, -1, -1):
        line = lines[idx]
        line_tokens = len(line.split())
        if prefix_lines and prefix_tokens + line_tokens > budget:
            break
        if not prefix_lines and line_tokens > budget:
            break
        prefix_lines.insert(0, line)
        prefix_tokens += line_tokens
        if prefix_tokens >= budget:
            break

    window = prefix_lines + selected
    actual_start = start - len(prefix_lines)
    return actual_start, "\n".join(window).rstrip()


def bridge_prefix_anchors(
    whole_segments: list[CodeSegment],
    selected_segments: list[CodeSegment],
    max_tokens: int = 0,
) -> list[CodeSegment]:
    """Add exact file-prefix anchors that make deep selected spans reachable.

    A selected function/method may begin thousands of tokens after the shared
    prompt prefix. Copying it directly would require unsafe zero-filled gaps.
    For prompt-fair experiments, this bridge keeps the target prompt unchanged
    but stores a contiguous exact prefix from the file start through the selected
    region. The AST/graph span still decides which file region matters; the
    bridge only changes the KV object shape so runtime reuse is position-safe.
    """
    segment_by_path = {segment.name: segment for segment in whole_segments}
    if max_tokens > 0:
        bridges: list[CodeSegment] = []
        seen: set[tuple[str, int, int, str]] = set()
        for segment in selected_segments:
            parsed = _parse_segment_line_range(segment.name)
            if parsed is None:
                continue
            path, start_line, end_line = parsed
            if path not in segment_by_path:
                continue
            lines = segment_by_path[path].text.splitlines()
            start, text = _bounded_bridge_text(lines, start_line, end_line, max_tokens)
            if not text:
                continue
            key = (path, start, end_line, hashlib.sha1(text.encode("utf-8")).hexdigest())
            if key in seen:
                continue
            seen.add(key)
            bridges.append(CodeSegment(f"{path}:bridge_window:bounded:{start}-{end_line}", text))
        return bridges or selected_segments

    max_end_by_path: dict[str, int] = {}
    for segment in selected_segments:
        parsed = _parse_segment_line_range(segment.name)
        if parsed is None:
            continue
        path, _start_line, end_line = parsed
        if path not in segment_by_path:
            continue
        max_end_by_path[path] = max(max_end_by_path.get(path, 0), end_line)

    bridges: list[CodeSegment] = []
    for path, end_line in sorted(max_end_by_path.items()):
        lines = segment_by_path[path].text.splitlines()
        end = min(max(1, end_line), len(lines))
        text = "\n".join(lines[:end]).rstrip()
        if text:
            bridges.append(CodeSegment(f"{path}:bridge_prefix:file_start:1-{end}", text))

    return bridges or selected_segments


TASK_AST_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "test",
    "tests",
    "pytest",
    "python",
    "src",
    "issue",
    "error",
    "class",
    "function",
}


def _text_tokens_for_task_ast(text: str) -> set[str]:
    raw_tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]+", text)
    out: set[str] = set()
    for raw in raw_tokens:
        for part in re.split(r"_+", raw):
            for token in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+", part):
                token = token.lower()
                if len(token) >= 3 and token not in TASK_AST_STOPWORDS:
                    out.add(token)
    return out


def task_ast_seed_segments(
    all_spans: list[Any],
    instance: dict[str, Any] | None,
    max_count: int,
) -> list[CodeSegment]:
    if not instance:
        return []
    task_text = "\n".join(
        [
            str(instance.get("problem_statement", "")),
            str(instance.get("FAIL_TO_PASS", "")),
            str(instance.get("test_patch", "")),
        ]
    )
    task_tokens = _text_tokens_for_task_ast(task_text)
    if not task_tokens:
        return []
    allowed = {"function", "method", "class", "control_block"}
    scored: list[tuple[int, int, int, str, Any]] = []
    for idx, span in enumerate(all_spans):
        granularity = segment_granularity(span.name)
        if granularity not in allowed:
            continue
        feature = anchor_feature_from_name(span.name)
        symbol = str(feature.get("symbol") or "")
        path = str(feature.get("path") or "")
        haystack = "\n".join([span.name, symbol, path, span.text[:1200]])
        overlap = len(_text_tokens_for_task_ast(haystack) & task_tokens)
        if overlap <= 0:
            continue
        token_estimate = max(1, len(span.text.split()))
        granularity_rank = {"function": 0, "method": 0, "class": 1, "control_block": 2}.get(
            granularity, 3
        )
        scored.append((-overlap, granularity_rank, token_estimate, span.name, span))
    scored.sort()
    limit = max(1, int(max_count or 0))
    return [CodeSegment(item[-1].name, item[-1].text) for item in scored[:limit]]


def maybe_bridge_prefix_anchors(
    args: argparse.Namespace,
    whole_segments: list[CodeSegment],
    selected_segments: list[CodeSegment],
    mode: str,
) -> list[CodeSegment]:
    if (
        not args.enable_bridge_prefix_anchors
        or mode in {"lossless_full_prefill", "whole_file_reuse_all"}
        or mode == HYBRID_CODE_AWARE_MODE
        or not selected_segments
        or (
            mode == GRAPH_AWARE_MODE
            and getattr(args, "disable_graph_bridge_prefix_anchors", False)
        )
    ):
        return selected_segments
    return bridge_prefix_anchors(
        whole_segments,
        selected_segments,
        int(getattr(args, "bridge_anchor_max_tokens", 0) or 0),
    )


def apply_selection_level_gate(
    selected_segments: list[CodeSegment],
    selection: dict[str, Any],
    args: argparse.Namespace,
    mode: str,
) -> tuple[list[CodeSegment], dict[str, Any]]:
    selection = dict(selection)
    selection.setdefault("selection_gate_rejected", False)
    selection.setdefault("selection_gate_reason", "")
    if mode in {"lossless_full_prefill", "whole_file_reuse_all"} or not selected_segments:
        return selected_segments, selection
    min_estimated_reused = int(getattr(args, "selection_min_estimated_reused_tokens", 0) or 0)
    if min_estimated_reused > 0 and _safe_float(selection.get("estimated_reused_tokens")) < min_estimated_reused:
        selection["selection_gate_rejected"] = True
        selection["selection_gate_reason"] = f"estimated_reused_tokens_below_{min_estimated_reused}"
        return [], selection
    return selected_segments, selection


def hybrid_code_aware_segments(
    whole_segments: list[CodeSegment],
    all_spans: list[Any],
    policy: dict[str, Any],
    graph_segments: list[CodeSegment] | None,
    args: argparse.Namespace | None,
    tokenizer: Any | None = None,
    instance: dict[str, Any] | None = None,
) -> tuple[list[CodeSegment], dict[str, Any]]:
    function_spans = select_spans(all_spans, "selective_function_method_reuse")
    function_segments = [CodeSegment(span.name, span.text) for span in function_spans]
    extended_spans = select_spans(all_spans, "selective_extended_reuse")
    extended_segments = [CodeSegment(span.name, span.text) for span in extended_spans]
    graph_budget_info: dict[str, Any] = {}
    graph_spans: list[Any] = []
    if graph_segments:
        graph_spans = graph_mapped_reuse_spans(whole_segments, policy, graph_segments)
        if graph_spans:
            graph_spans, graph_budget_info = apply_graph_anchor_budget(graph_spans, args)
    bridge_source = str(getattr(args, "hybrid_bridge_source", "function") or "function")
    selected_segments: list[CodeSegment] = []
    decision_counts: dict[str, int] = {}
    if bridge_source == "function":
        bridge_seed_segments = function_segments
    elif bridge_source == "graph":
        bridge_seed_segments = [CodeSegment(span.name, span.text) for span in graph_spans]
    elif bridge_source == "graph_then_function":
        graph_seed_segments = [CodeSegment(span.name, span.text) for span in graph_spans]
        bridge_seed_segments = graph_seed_segments or function_segments
    elif bridge_source == "extended":
        bridge_seed_segments = extended_segments
    elif bridge_source == "function_then_extended":
        bridge_seed_segments = function_segments or extended_segments
    elif bridge_source == "extended_then_function":
        bridge_seed_segments = extended_segments or function_segments
    elif bridge_source == "task_ast":
        bridge_seed_segments = task_ast_seed_segments(
            all_spans,
            instance,
            int(getattr(args, "hybrid_task_ast_top_k", 3) or 3),
        )
    elif bridge_source == "task_ast_direct":
        bridge_seed_segments = []
        selected_segments.extend(
            task_ast_seed_segments(
                all_spans,
                instance,
                int(getattr(args, "hybrid_task_ast_top_k", 3) or 3),
            )
        )
        decision_counts["reuse:hybrid_task_ast_direct_span"] = len(selected_segments)
    else:
        raise ValueError(f"unsupported --hybrid-bridge-source: {bridge_source}")
    bridge_candidates = (
        []
        if bridge_source == "task_ast_direct"
        else bridge_prefix_anchors(
            whole_segments,
            bridge_seed_segments,
            max_tokens=int(getattr(args, "hybrid_bridge_anchor_max_tokens", 0) or 0),
        )
    )
    min_bridge_tokens = int(getattr(args, "hybrid_min_bridge_tokens", 0) or 0)
    max_bridge_tokens = int(
        getattr(args, "hybrid_max_bridge_tokens", 0)
        or getattr(args, "lossy_max_planned_suffix_copy_len", 0)
        or 0
    )
    bridge_filter_estimator = "tokenizer" if tokenizer is not None else "whitespace"
    selected_bridge_token_estimates: list[int] = []
    scored_bridge_candidates: list[tuple[CodeSegment, int, tuple[str, int, int] | None]] = []
    for segment in bridge_candidates:
        if tokenizer is not None:
            token_estimate = len(tokenizer(segment.text, add_special_tokens=False).input_ids)
        else:
            token_estimate = len(segment.text.split())
        if min_bridge_tokens > 0 and token_estimate < min_bridge_tokens:
            continue
        if max_bridge_tokens > 0 and token_estimate > max_bridge_tokens:
            continue
        scored_bridge_candidates.append((segment, token_estimate, _parse_segment_line_range(segment.name)))

    bridge_max_count_per_file = int(getattr(args, "hybrid_bridge_max_count_per_file", 0) or 0)
    bridge_count_pruned = 0
    if bridge_max_count_per_file > 0:
        grouped: dict[str, list[tuple[CodeSegment, int, tuple[str, int, int] | None]]] = {}
        for item in scored_bridge_candidates:
            segment, _tokens, parsed = item
            path = parsed[0] if parsed is not None else segment.name
            grouped.setdefault(path, []).append(item)
        kept_scored: list[tuple[CodeSegment, int, tuple[str, int, int] | None]] = []
        for path in sorted(grouped):
            items = sorted(
                grouped[path],
                key=lambda item: (
                    item[2][2] if item[2] is not None else -1,
                    item[1],
                    item[0].name,
                ),
                reverse=True,
            )
            kept_scored.extend(items[:bridge_max_count_per_file])
            bridge_count_pruned += max(0, len(items) - bridge_max_count_per_file)
        scored_bridge_candidates = sorted(
            kept_scored,
            key=lambda item: (
                item[2][0] if item[2] is not None else item[0].name,
                item[2][1] if item[2] is not None else 0,
                item[2][2] if item[2] is not None else 0,
                item[0].name,
            ),
        )

    selected_bridge_paths: set[str] = set()
    for segment, token_estimate, parsed in scored_bridge_candidates:
        selected_segments.append(segment)
        selected_bridge_token_estimates.append(token_estimate)
        if parsed is not None:
            selected_bridge_paths.add(parsed[0])
        decision_counts["reuse:hybrid_large_function_bridge"] = (
            decision_counts.get("reuse:hybrid_large_function_bridge", 0) + 1
        )

    if graph_spans:
        for span in graph_spans:
            selected_segments.append(CodeSegment(span.name, span.text))
        decision_counts["reuse:hybrid_graph_mapped_ast_span"] = len(graph_spans)

    if getattr(args, "include_hybrid_bridge_seed_spans", False) and bridge_source != "task_ast_direct":
        seed_segments_for_selected_bridges = []
        for segment in bridge_seed_segments:
            parsed = _parse_segment_line_range(segment.name)
            if selected_bridge_paths and (parsed is None or parsed[0] not in selected_bridge_paths):
                continue
            seed_segments_for_selected_bridges.append(segment)
        for segment in seed_segments_for_selected_bridges:
            selected_segments.append(segment)
        if seed_segments_for_selected_bridges:
            decision_counts["reuse:hybrid_bridge_seed_span"] = len(seed_segments_for_selected_bridges)

    graph_token_estimate = sum(span_token_estimate(span) for span in graph_spans)
    risk_min_tokens = int(getattr(args, "hybrid_risk_large_bridge_min_tokens", 0) or 0)
    risk_max_bridge_count = int(getattr(args, "hybrid_risk_max_large_bridge_count", 0) or 0)
    risk_max_graph_tokens = int(getattr(args, "hybrid_risk_max_graph_tokens_for_large_bridge", 0) or 0)
    large_bridge_count = (
        sum(1 for token_estimate in selected_bridge_token_estimates if token_estimate >= risk_min_tokens)
        if risk_min_tokens > 0
        else 0
    )
    risk_gate_reason = ""
    if large_bridge_count > 0:
        if risk_max_bridge_count > 0 and large_bridge_count > risk_max_bridge_count:
            risk_gate_reason = f"large_bridge_count_gt_{risk_max_bridge_count}"
        elif risk_max_graph_tokens > 0 and graph_token_estimate > risk_max_graph_tokens:
            risk_gate_reason = f"graph_tokens_gt_{risk_max_graph_tokens}"
    if risk_gate_reason:
        selected_segments = []
        decision_counts = {f"reject:hybrid_large_bridge_risk:{risk_gate_reason}": 1}

    seen: set[tuple[str, str]] = set()
    deduped: list[CodeSegment] = []
    for segment in selected_segments:
        key = (segment.name, hashlib.sha1(segment.text.encode("utf-8")).hexdigest())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(segment)
    selected_segments = deduped
    whole_tokens = sum(max(1, len(segment.text.split())) for segment in whole_segments)
    reused_tokens = sum(max(1, len(segment.text.split())) for segment in selected_segments)
    selected_counts: dict[str, int] = {}
    for segment in selected_segments:
        granularity = segment_granularity(segment.name) or "bridge"
        selected_counts[granularity] = selected_counts.get(granularity, 0) + 1
    selection = {
        "span_count": len(all_spans),
        "selected_span_count": len(selected_segments),
        "span_count_by_granularity": summarize_selection(all_spans, [])["span_count_by_granularity"],
        "selected_span_count_by_granularity": selected_counts,
        "estimated_reused_tokens": reused_tokens,
        "estimated_recomputed_tokens": max(0, whole_tokens - reused_tokens),
        "decision_reason_counts": decision_counts,
        "hybrid_bridge_filter_estimator": bridge_filter_estimator,
        "hybrid_bridge_source": bridge_source,
        "hybrid_bridge_anchor_max_tokens": int(getattr(args, "hybrid_bridge_anchor_max_tokens", 0) or 0),
        "hybrid_bridge_max_count_per_file": bridge_max_count_per_file,
        "include_hybrid_bridge_seed_spans": bool(getattr(args, "include_hybrid_bridge_seed_spans", False)),
        "hybrid_bridge_count_pruned": bridge_count_pruned,
        "hybrid_large_bridge_count": large_bridge_count,
        "hybrid_graph_token_estimate": graph_token_estimate,
        "hybrid_risk_gate_rejected": bool(risk_gate_reason),
        "hybrid_risk_gate_reason": risk_gate_reason,
        **graph_budget_info,
    }
    return selected_segments, selection


def prompt_resident_monotonic_segments(prompt_text: str, segments: list[CodeSegment]) -> list[CodeSegment]:
    """Keep anchors that build_anchor_fields can locate with its cursor rule."""
    ordered = sorted(
        segments,
        key=lambda segment: (
            prompt_text.find(segment.text) if segment.text and prompt_text.find(segment.text) >= 0 else 10**12,
            segment.name,
        ),
    )
    resident: list[CodeSegment] = []
    char_cursor = 0
    for segment in ordered:
        if not segment.text:
            continue
        char_pos = prompt_text.find(segment.text, char_cursor)
        if char_pos < 0:
            continue
        resident.append(segment)
        char_cursor = char_pos + len(segment.text)
    return resident


def filter_segments_by_start_token(
    tokenizer: Any,
    prompt_text: str,
    segments: list[CodeSegment],
    max_start_token: int,
) -> tuple[list[CodeSegment], int]:
    if max_start_token <= 0 or not segments:
        return segments, 0
    filtered: list[CodeSegment] = []
    char_cursor = 0
    dropped = 0
    for segment in segments:
        char_pos = prompt_text.find(segment.text, char_cursor)
        if char_pos < 0:
            dropped += 1
            continue
        start_token = len(tokenizer.encode(prompt_text[:char_pos], add_special_tokens=False))
        char_cursor = char_pos + len(segment.text)
        if start_token <= max_start_token:
            filtered.append(segment)
        else:
            dropped += 1
    return filtered, dropped


def launch_server(args: argparse.Namespace) -> subprocess.Popen:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT / "python")
    env["SGLANG_LOSSY_FUZZY_MATCH"] = "1"
    # Semantic suffix-copy length decider (per-chunk cosine profile). Default
    # ON per the v9 mainline plan; --disable-semantic-suffix turns it off
    # for regression checks against v9 numbers.
    env["SGLANG_SEMANTIC_SUFFIX_ENABLED"] = "1" if getattr(args, "enable_semantic_suffix", True) else "0"
    if getattr(args, "enable_semantic_suffix", True):
        # Pre-warm the embedder inside the server subprocess so the first
        # request does not pay the model-load cost (~6 s on this host).
        try:
            from sglang.srt.mem_cache.semantic_suffix import load_embedder
            emb = load_embedder()
            if emb is None:
                import sys as _sys
                print(
                    "[semantic_suffix] embedder prewarm failed; "
                    "semantic suffix falls back to legacy caps.",
                    file=_sys.stderr,
                )
        except Exception as _e:
            import sys as _sys
            print(
                f"[semantic_suffix] embedder prewarm exception: {_e}",
                file=_sys.stderr,
            )
    if args.lossy_max_zero_gap is not None:
        env["SGLANG_LOSSY_MAX_ZERO_GAP"] = str(args.lossy_max_zero_gap)
    if args.lossy_max_suffix_copy_len is not None and args.lossy_max_suffix_copy_len > 0:
        env["SGLANG_LOSSY_MAX_SUFFIX_COPY_LEN"] = str(args.lossy_max_suffix_copy_len)
    if (
        args.lossy_max_planned_suffix_copy_len is not None
        and args.lossy_max_planned_suffix_copy_len > 0
    ):
        env["SGLANG_LOSSY_MAX_PLANNED_SUFFIX_COPY_LEN"] = str(
            args.lossy_max_planned_suffix_copy_len
        )
    if args.lossy_suffix_recompute_head_len and args.lossy_suffix_recompute_head_len > 0:
        env["SGLANG_LOSSY_SUFFIX_RECOMPUTE_HEAD_LEN"] = str(
            args.lossy_suffix_recompute_head_len
        )
    if args.lossy_max_recompute_gap_len and args.lossy_max_recompute_gap_len > 0:
        env["SGLANG_LOSSY_MAX_RECOMPUTE_GAP_LEN"] = str(args.lossy_max_recompute_gap_len)
    if args.lossy_recompute_gap:
        env["SGLANG_LOSSY_RECOMPUTE_GAP"] = "1"
    if args.lossy_stage_recompute_gap:
        env["SGLANG_LOSSY_RECOMPUTE_GAP"] = "1"
        env["SGLANG_LOSSY_STAGE_RECOMPUTE_GAP"] = "1"
    if args.lossy_multi_anchor_copy:
        env["SGLANG_LOSSY_MULTI_ANCHOR"] = "1"
    if args.context_aware_max_predicted_d is not None:
        env["SGLANG_CONTEXT_AWARE_MAX_PREDICTED_D"] = str(args.context_aware_max_predicted_d)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        args.python,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model,
        "--port",
        str(args.port),
        "--tp-size",
        "1",
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--max-total-tokens",
        str(args.max_total_tokens),
        "--chunked-prefill-size",
        "8192",
        "--max-prefill-tokens",
        str(args.max_prefill_tokens),
        "--enable-cache-report",
        "--disable-cuda-graph",
        "--allow-auto-truncate",
        "--log-level",
        "error",
    ]
    if args.server_random_seed is not None:
        cmd += ["--random-seed", str(args.server_random_seed)]
    if args.disable_overlap_schedule:
        cmd.append("--disable-overlap-schedule")
    if args.force_evict:
        env["SGLANG_RADIX_FORCE_EVICT"] = "1"
    if args.max_running_requests is not None:
        cmd += ["--max-running-requests", str(args.max_running_requests)]
    return subprocess.Popen(
        cmd,
        cwd=str(PROJECT),
        env=env,
        stdout=open(args.out_dir / "sglang_server.log", "w"),
        stderr=subprocess.STDOUT,
    )


def build_wholefile_messages(instance: dict[str, Any], segments: list[CodeSegment], mode: str) -> list[dict[str, str]]:
    blocks = []
    for segment in segments:
        blocks.extend(
            [
                f"## code_base: {segment.name}",
                "```python",
                segment.text,
                "```",
                "",
            ]
        )
    return [
        {
            "role": "system",
            "content": (
                "You are a coding agent. Read the whole JSON code_base, then answer with a concise "
                "implementation plan. The serving runtime may reuse exact low-risk AST spans, but you "
                "must treat the codebase as complete file context."
            ),
        },
        {
            "role": "user",
            "content": "\n".join(
                [
                    "## Issue",
                    str(instance.get("problem_statement", ""))[:4000],
                    "",
                    "## FAIL_TO_PASS",
                    str(instance.get("FAIL_TO_PASS", ""))[:2000],
                    "",
                    "## Whole-file code_base",
                    *blocks,
                    "## Task",
                    "Summarize the minimal implementation change needed for the issue.",
                ]
            ),
        },
    ]


def build_natural_planner_messages(instance: dict[str, Any], segments: list[CodeSegment]) -> list[dict[str, str]]:
    messages = build_wholefile_messages(instance, segments, "natural_planner_warmup")
    messages[0]["content"] = (
        "You are a Planner agent. Read the issue, tests, and whole code_base. "
        "Identify the files and code regions that are likely relevant for a later Coder agent. "
        "Do not write a patch."
    )
    messages[1]["content"] = messages[1]["content"].replace(
        "## Task\nSummarize the minimal implementation change needed for the issue.",
        "## Planner Task\nList the likely relevant file(s), symbols, and code regions for the later implementation step.",
    )
    return messages


def build_prompt_fair_planner_messages(instance: dict[str, Any], segments: list[CodeSegment]) -> list[dict[str, str]]:
    """Planner warmup that shares only the stable task prefix with target prompts.

    The target prompt stays byte-for-byte unchanged. This warmup keeps the same
    system prompt and issue/test prefix so the fair benchmark does not create a
    huge artificial zero-fill gap before code anchors, but it diverges before
    the code_base heading so code reuse still has to go through exact-content
    anchor matching instead of ordinary prefix-cache reuse.
    """
    blocks = []
    for segment in segments:
        blocks.extend(
            [
                f"## code_base: {segment.name}",
                "```python",
                segment.text,
                "```",
                "",
            ]
        )
    return [
        build_wholefile_messages(instance, segments, "target_layout_reference")[0],
        {
            "role": "user",
            "content": "\n".join(
                [
                    "## Issue",
                    str(instance.get("problem_statement", ""))[:4000],
                    "",
                    "## FAIL_TO_PASS",
                    str(instance.get("FAIL_TO_PASS", ""))[:2000],
                    "",
                    "## Planner code anchors",
                    *blocks,
                    "## Planner Task",
                    "List likely relevant file(s), symbols, and code regions for the later implementation step. Do not write a patch.",
                ]
            ),
        },
    ]


def build_graphaware_messages(
    instance: dict[str, Any],
    whole_segments: list[CodeSegment],
    graph_segments: list[CodeSegment],
    mode: str,
    graph_policy: str,
) -> list[dict[str, str]]:
    messages = build_wholefile_messages(instance, whole_segments, mode)
    graph_blocks = []
    for idx, segment in enumerate(graph_segments, 1):
        graph_blocks.extend(
            [
                f"### graph_bundle_{idx}: {segment.name}",
                "```python",
                segment.text,
                "```",
                "",
            ]
        )
    graph_text = "\n".join(
        [
            "## Code graph reuse bundles",
            f"policy: {graph_policy}",
            "These relation-selected snippets are exact code evidence selected by the code graph planner.",
            *graph_blocks,
        ]
    )
    user = messages[1]["content"]
    marker = "\n## Task\n"
    if marker in user:
        user = user.replace(marker, "\n" + graph_text + marker, 1)
    else:
        user = user + "\n\n" + graph_text
    messages[1]["content"] = user
    return messages


def selected_segments_for_mode(
    whole_segments: list[CodeSegment],
    policy: dict[str, Any],
    mode: str,
    graph_segments: list[CodeSegment] | None = None,
    args: argparse.Namespace | None = None,
    tokenizer: Any | None = None,
    instance: dict[str, Any] | None = None,
) -> tuple[list[CodeSegment], dict[str, Any]]:
    all_spans = []
    for segment in whole_segments:
        all_spans.extend(split_python_file(segment.name, segment.text, policy))
    if mode == "whole_file_reuse_all":
        total_tokens = sum(max(1, len(segment.text.split())) for segment in whole_segments)
        selection = {
            "span_count": len(all_spans),
            "selected_span_count": len(whole_segments),
            "span_count_by_granularity": summarize_selection(all_spans, [])["span_count_by_granularity"],
            "selected_span_count_by_granularity": {"whole_file": len(whole_segments)},
            "estimated_reused_tokens": total_tokens,
            "estimated_recomputed_tokens": 0,
            "decision_reason_counts": {"reuse:whole_file_diagnostic": len(whole_segments)},
        }
        return whole_segments, attach_selected_anchor_names(selection, whole_segments)
    if mode == HYBRID_CODE_AWARE_MODE:
        selected_segments, selection = hybrid_code_aware_segments(
            whole_segments, all_spans, policy, graph_segments, args, tokenizer, instance
        )
        selected_segments, selection = apply_excluded_anchor_granularities(selected_segments, selection, args)
        return selected_segments, attach_selected_anchor_names(selection, selected_segments)
    if mode == GRAPH_AWARE_MODE:
        if not graph_segments:
            raise ValueError("no graph-aware bundle for case")
        selected = graph_mapped_reuse_spans(whole_segments, policy, graph_segments)
        if not selected:
            raise ValueError("graph-aware bundle did not map to whole-file AST spans")
        selected, budget_info = apply_graph_anchor_budget(selected, args)
        if not selected:
            raise ValueError(str(budget_info.get("graph_anchor_fallback_reason") or "graph-aware budget filtered all spans"))
        selected_segments = [CodeSegment(span.name, span.text) for span in selected]
        reused_tokens = sum(max(1, len(span.text.split())) for span in selected)
        whole_tokens = sum(max(1, len(segment.text.split())) for segment in whole_segments)
        selection = {
            "span_count": len(all_spans),
            "selected_span_count": len(selected),
            "span_count_by_granularity": summarize_selection(all_spans, [])["span_count_by_granularity"],
            "selected_span_count_by_granularity": summarize_selection(all_spans, selected)["selected_span_count_by_granularity"],
            "estimated_reused_tokens": reused_tokens,
            "estimated_recomputed_tokens": max(0, whole_tokens - reused_tokens),
            "decision_reason_counts": {"reuse:code_graph_mapped_ast_span": len(selected)},
            **budget_info,
        }
        selected_segments, selection = apply_excluded_anchor_granularities(selected_segments, selection, args)
        return selected_segments, attach_selected_anchor_names(selection, selected_segments)
    selected = select_spans(all_spans, mode)
    selected_segments = [CodeSegment(span.name, span.text) for span in selected]
    selection = summarize_selection(all_spans, selected)
    selected_segments, selection = apply_excluded_anchor_granularities(selected_segments, selection, args)
    return selected_segments, attach_selected_anchor_names(selection, selected_segments)


def make_payload(
    args: argparse.Namespace,
    tokenizer: Any,
    messages: list[dict[str, str]],
    selected_segments: list[CodeSegment],
    mode: str,
    salt: str,
    instance_id: str | None = None,
    selection_telemetry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    initial_anchor_count = len(selected_segments)
    prompt_resident_anchor_count = 0
    token_filter_dropped_count = 0
    start_filter_dropped_count = 0
    payload_anchor_min_total_rejected = False
    payload_anchor_max_total_rejected = False
    payload_anchor_max_total_pruned_count = 0
    payload_anchor_token_count = 0
    include_anchor = mode != "lossless_full_prefill" and bool(selected_segments)
    if include_anchor:
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        selected_segments = prompt_resident_monotonic_segments(prompt_text, selected_segments)
        prompt_resident_anchor_count = len(selected_segments)
        include_anchor = bool(selected_segments)
        max_start_token = int(getattr(args, "selective_anchor_max_start_token", 0) or 0)
        if include_anchor and max_start_token > 0:
            selected_segments, start_filter_dropped_count = filter_segments_by_start_token(
                tokenizer,
                prompt_text,
                selected_segments,
                max_start_token,
            )
            include_anchor = bool(selected_segments)
        selective_anchor_max_span_tokens = int(
            getattr(args, "selective_anchor_max_span_tokens", 0) or 0
        )
        selective_anchor_min_span_tokens = int(
            getattr(args, "selective_anchor_min_span_tokens", 0) or 0
        )
        if selective_anchor_max_span_tokens > 0 or selective_anchor_min_span_tokens > 0:
            filtered_segments = []
            for segment in selected_segments:
                token_ids = tokenizer(segment.text, add_special_tokens=False).input_ids
                if (
                    (selective_anchor_max_span_tokens <= 0 or len(token_ids) <= selective_anchor_max_span_tokens)
                    and (selective_anchor_min_span_tokens <= 0 or len(token_ids) >= selective_anchor_min_span_tokens)
                ):
                    filtered_segments.append(segment)
            token_filter_dropped_count = len(selected_segments) - len(filtered_segments)
            selected_segments = filtered_segments
            include_anchor = bool(selected_segments)
        anchor_min_total_tokens = int(getattr(args, "anchor_min_total_tokens", 0) or 0)
        anchor_max_total_tokens = int(getattr(args, "anchor_max_total_tokens", 0) or 0)
        if include_anchor and (anchor_min_total_tokens > 0 or anchor_max_total_tokens > 0):
            total_anchor_tokens = sum(
                len(tokenizer(segment.text, add_special_tokens=False).input_ids)
                for segment in selected_segments
            )
            if total_anchor_tokens < anchor_min_total_tokens:
                selected_segments = []
                include_anchor = False
                payload_anchor_min_total_rejected = True
            elif anchor_max_total_tokens > 0 and total_anchor_tokens > anchor_max_total_tokens:
                max_total_policy = getattr(args, "anchor_max_total_policy", "reject")
                if max_total_policy == "reject":
                    selected_segments = []
                    include_anchor = False
                    payload_anchor_max_total_rejected = True
                else:
                    segment_token_pairs = [
                        (segment, len(tokenizer(segment.text, add_special_tokens=False).input_ids))
                        for segment in selected_segments
                    ]
                    if max_total_policy == "prune_shortest":
                        ordered_pairs = sorted(segment_token_pairs, key=lambda item: (item[1], item[0].name))
                    elif max_total_policy == "prune_first":
                        ordered_pairs = segment_token_pairs
                    else:
                        raise ValueError(f"unsupported anchor_max_total_policy: {max_total_policy}")
                    kept: list[CodeSegment] = []
                    kept_tokens = 0
                    for segment, token_count in ordered_pairs:
                        if kept_tokens + token_count > anchor_max_total_tokens:
                            continue
                        kept.append(segment)
                        kept_tokens += token_count
                    payload_anchor_max_total_pruned_count = len(selected_segments) - len(kept)
                    selected_segments = kept
                    include_anchor = bool(selected_segments)
                    if not include_anchor:
                        payload_anchor_max_total_rejected = True
        if include_anchor:
            payload_anchor_token_count = sum(
                len(tokenizer(segment.text, add_special_tokens=False).input_ids)
                for segment in selected_segments
            )
    payload: dict[str, Any] = {
        "model": args.model,
        "messages": messages,
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "return_cached_tokens_details": True,
        "reuse_mode": "lossy" if include_anchor else "lossless",
        "lossy_alignment_method": "kvcomm",
        "template_task_family": "selective_ast_wholefile_reuse",
        "cache_salt": salt,
        "priority": 1,
    }
    payload["_driver_anchor_telemetry"] = {
        "payload_reuse_mode": payload["reuse_mode"],
        "payload_anchor_count": len(selected_segments) if include_anchor else 0,
        "payload_anchor_token_count": payload_anchor_token_count,
        "payload_initial_anchor_count": initial_anchor_count,
        "payload_prompt_resident_anchor_count": prompt_resident_anchor_count,
        "payload_token_filter_dropped_count": token_filter_dropped_count,
        "payload_anchor_start_filter_dropped_count": start_filter_dropped_count,
        "payload_anchor_min_total_rejected": payload_anchor_min_total_rejected,
        "payload_anchor_max_total_rejected": payload_anchor_max_total_rejected,
        "payload_anchor_max_total_pruned_count": payload_anchor_max_total_pruned_count,
    }
    if include_anchor:
        payload["next_agent_prefix"] = "You are the implementation target. Reuse exact code anchors from planner warmup."
        payload["codebase_prefetch_hints"] = build_codebase_prefetch_hints(selected_segments)
        payload.update(build_anchor_fields(tokenizer, messages, selected_segments))
        apply_anchor_tier_copy_caps(args, tokenizer, mode, selected_segments, payload)
        apply_graph_anchor_copy_caps(args, tokenizer, mode, selected_segments, payload)
        payload["_driver_anchor_telemetry"].update(
            apply_hybrid_calibration_to_payload(args, instance_id, mode, payload, selection_telemetry)
        )
    return payload


def pop_driver_anchor_telemetry(payload: dict[str, Any]) -> dict[str, Any]:
    return payload.pop("_driver_anchor_telemetry", {})


def anchor_feature_from_name(name: str) -> dict[str, Any]:
    base_name = parse_graph_segment_name(str(name))[0]
    parts = base_name.rsplit(":", 3)
    feature = {
        "name": str(name),
        "path": base_name,
        "granularity": segment_granularity(base_name),
        "symbol": "",
        "line_range": "",
    }
    if len(parts) >= 4:
        feature.update(
            {
                "path": parts[0],
                "granularity": parts[1],
                "symbol": parts[2],
                "line_range": parts[3],
            }
        )
    return feature


def task_text_for_anchor_overlap(instance: dict[str, Any]) -> str:
    return "\n".join(
        [
            str(instance.get("problem_statement", "")),
            str(instance.get("FAIL_TO_PASS", "")),
            str(instance.get("test_patch", "")),
        ]
    ).lower()


def anchor_task_overlap_features(instance: dict[str, Any], anchor_names: list[str]) -> list[dict[str, Any]]:
    task_text = task_text_for_anchor_overlap(instance)
    out: list[dict[str, Any]] = []
    for name in anchor_names:
        feature = anchor_feature_from_name(name)
        path = str(feature.get("path") or "")
        basename = Path(path).name
        stem = Path(path).stem
        symbol = str(feature.get("symbol") or "")
        candidates = [path, basename, stem, symbol]
        path_mentioned = bool(path and path.lower() in task_text)
        basename_mentioned = bool(basename and basename.lower() in task_text)
        symbol_tokens = [
            token.lower()
            for token in re.split(r"[^A-Za-z0-9_]+", symbol)
            if len(token) >= 3 and token not in {"file", "start", "bridge", "prefix"}
        ]
        symbol_overlap = sum(1 for token in symbol_tokens if token in task_text)
        lexical_overlap = sum(1 for item in candidates if item and item.lower() in task_text)
        out.append(
            {
                **feature,
                "path_mentioned": path_mentioned,
                "basename_mentioned": basename_mentioned,
                "symbol_overlap": symbol_overlap,
                "lexical_overlap": lexical_overlap,
            }
        )
    return out


def write_selection_feature_dry_run(
    args: argparse.Namespace,
    tokenizer: Any,
    cases: list[dict[str, Any]],
    policy: dict[str, Any],
) -> None:
    rows: list[dict[str, Any]] = []
    for case in cases:
        instance = case["instance"]
        instance_id = instance["instance_id"]
        case_args = args_for_case_selector(args, instance_id)
        for mode in active_modes(args):
            if mode == "lossless_full_prefill":
                continue
            try:
                selected, selection = selected_segments_for_mode(
                    case["segments"],
                    policy,
                    mode,
                    case.get("graph_segments", []),
                    case_args,
                    tokenizer,
                    instance,
                )
                selected, selection = apply_selection_level_gate(selected, selection, case_args, mode)
                anchor_names_before_calibration = selection.get("selected_anchor_names") or [
                    segment.name for segment in selected
                ]
                if not selection.get("selected_anchor_names"):
                    selection = {**selection, "selected_anchor_names": anchor_names_before_calibration}
                selection = attach_anchor_task_overlap(selection, instance)
                selected, selection = apply_hybrid_calibration_to_selection(
                    case_args,
                    instance_id,
                    mode,
                    selected,
                    selection,
                    case["segments"],
                )
                anchor_names = selection.get("selected_anchor_names") or [segment.name for segment in selected]
                anchor_task_features = selection.get("selected_anchor_task_features") or []
                rows.append(
                    {
                        "instance_id": instance_id,
                        "repo": instance.get("repo", ""),
                        "mode": mode,
                        "status": "ok",
                        "selected_span_count": selection.get("selected_span_count", len(selected)),
                        "selected_span_count_by_granularity": selection.get("selected_span_count_by_granularity") or {},
                        "estimated_reused_tokens": selection.get("estimated_reused_tokens", 0),
                        "estimated_recomputed_tokens": selection.get("estimated_recomputed_tokens", 0),
                        "decision_reason_counts": selection.get("decision_reason_counts") or {},
                        "selected_anchor_names": anchor_names,
                        "selected_anchor_features": [anchor_feature_from_name(name) for name in anchor_names],
                        "selected_anchor_task_features": anchor_task_features,
                        "max_anchor_lexical_overlap": selection.get("max_anchor_lexical_overlap", 0),
                        "any_anchor_path_mentioned": selection.get("any_anchor_path_mentioned", False),
                        "any_anchor_basename_mentioned": selection.get("any_anchor_basename_mentioned", False),
                        "max_anchor_symbol_overlap": selection.get("max_anchor_symbol_overlap", 0),
                        "hybrid_calibration_policy_applied": selection.get("hybrid_calibration_policy_applied"),
                        "hybrid_calibration_action": selection.get("hybrid_calibration_action", ""),
                        "hybrid_calibration_rule_name": selection.get("hybrid_calibration_rule_name", ""),
                        "hybrid_calibration_reason": selection.get("hybrid_calibration_reason", ""),
                        "hybrid_calibration_max_suffix_copy_len": selection.get("hybrid_calibration_max_suffix_copy_len", ""),
                        "hybrid_calibration_bridge_window_synthesized": selection.get("hybrid_calibration_bridge_window_synthesized"),
                        "hybrid_calibration_bridge_window_max_tokens": selection.get("hybrid_calibration_bridge_window_max_tokens", ""),
                        "hybrid_calibration_bridge_window_seed_count": selection.get("hybrid_calibration_bridge_window_seed_count", ""),
                        "hybrid_calibration_rejected": selection.get("hybrid_calibration_rejected"),
                        "hybrid_calibration_shape_mismatch": selection.get("hybrid_calibration_shape_mismatch"),
                        "hybrid_calibration_shape_pruned": selection.get("hybrid_calibration_shape_pruned"),
                    }
                )
            except (ValueError, KeyError) as exc:
                rows.append(
                    {
                        "instance_id": instance_id,
                        "repo": instance.get("repo", ""),
                        "mode": mode,
                        "status": f"skipped:{type(exc).__name__}",
                        "error": str(exc),
                    }
                )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "selector_snapshot": selector_snapshot(args),
        "dataset": str(args.dataset),
        "manifest": str(args.manifest),
        "modes": active_modes(args),
        "hybrid_calibration_policy": str(args.hybrid_calibration_policy) if args.hybrid_calibration_policy else "",
        "hybrid_calibration_policy_cases": len((getattr(args, "_hybrid_calibration_policy_data", {}) or {}).get("cases") or {}),
        "hybrid_calibration_policy_rules": len((getattr(args, "_hybrid_calibration_policy_data", {}) or {}).get("rules") or []),
        "rows": rows,
    }
    (args.out_dir / "selection_features.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (args.out_dir / "selection_features.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "instance_id",
            "repo",
            "mode",
            "status",
            "selected_span_count",
            "selected_span_count_by_granularity",
            "estimated_reused_tokens",
            "selected_anchor_names",
            "max_anchor_lexical_overlap",
            "any_anchor_path_mentioned",
            "any_anchor_basename_mentioned",
            "max_anchor_symbol_overlap",
            "hybrid_calibration_action",
            "hybrid_calibration_rule_name",
            "hybrid_calibration_reason",
            "hybrid_calibration_max_suffix_copy_len",
            "hybrid_calibration_bridge_window_synthesized",
            "hybrid_calibration_bridge_window_max_tokens",
            "hybrid_calibration_bridge_window_seed_count",
            "hybrid_calibration_rejected",
            "hybrid_calibration_shape_mismatch",
            "hybrid_calibration_shape_pruned",
            "error",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    summary = {
        "loaded_cases": len(cases),
        "rows": len(rows),
        "ok_rows": sum(1 for row in rows if row.get("status") == "ok"),
        "cap_rows": sum(1 for row in rows if row.get("hybrid_calibration_action") == "cap"),
        "reject_rows": sum(1 for row in rows if row.get("hybrid_calibration_action") == "reject"),
        "out_dir": str(args.out_dir),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


async def warm_case(
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    tokenizer: Any,
    instance: dict[str, Any],
    segments: list[CodeSegment],
    graph_segments: list[CodeSegment],
    policy: dict[str, Any],
) -> None:
    # Legacy shared-cache warmup: keep for compatibility with
    # --shared-cache-across-modes, but do not use it for the realistic protocol.
    messages = build_wholefile_messages(instance, segments, "planner_warmup_observed_wholefile")
    whole_payload = make_payload(args, tokenizer, messages, segments, "whole_file_reuse_all", f"warm_whole:{instance['instance_id']}")
    whole_payload["max_tokens"] = 8
    pop_driver_anchor_telemetry(whole_payload)
    await post_chat_optional_stream(session, args.port, whole_payload, args.emit_ttft)
    all_spans = []
    for segment in segments:
        all_spans.extend(split_python_file(segment.name, segment.text, policy))
    # Use selective_function_method_reuse for warmup regardless of which
    # policy is loaded; this avoids text-matching failures from the new
    # control_block/file_prefix spans when the extended policy is in use.
    warm_segments = [CodeSegment(span.name, span.text) for span in select_spans(all_spans, "selective_function_method_reuse")]
    if not warm_segments:
        warm_segments = segments[:1]
    payload = make_payload(args, tokenizer, messages, warm_segments, "selective_oracle_low_dnorm", f"warm:{instance['instance_id']}")
    payload["max_tokens"] = 8
    pop_driver_anchor_telemetry(payload)
    await post_chat_optional_stream(session, args.port, payload, args.emit_ttft)
    if args.enable_graph_aware_lossy and graph_segments:
        graph_messages = build_wholefile_messages(instance, segments, "planner_graph_bundle_warmup")
        graph_selected, _ = selected_segments_for_mode(
            segments, policy, GRAPH_AWARE_MODE, graph_segments, args, tokenizer
        )
        graph_payload = make_payload(
            args,
            tokenizer,
            graph_messages,
            graph_selected,
            GRAPH_AWARE_MODE,
            f"warm:{instance['instance_id']}:{GRAPH_AWARE_MODE}",
        )
        graph_payload["max_tokens"] = 8
        pop_driver_anchor_telemetry(graph_payload)
        await post_chat_optional_stream(session, args.port, graph_payload, args.emit_ttft)


async def warm_natural_planner(
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    tokenizer: Any,
    instance: dict[str, Any],
    segments: list[CodeSegment],
    graph_segments: list[CodeSegment],
    policy: dict[str, Any],
    selected_override: list[CodeSegment] | None = None,
) -> dict[str, Any]:
    """Run one Planner-style warmup shared by all measured target modes.

    This is intentionally not mode-specific. It lets the runtime cache the
    natural whole-file code context a Planner would read before Coder/Reviewer
    target requests, without giving each target mode its own oracle warmup.
    """
    if args.warmup_protocol == "fair_planner_per_mode":
        messages = build_prompt_fair_planner_messages(instance, segments)
    else:
        messages = build_natural_planner_messages(instance, segments)
    telemetry = prompt_telemetry(tokenizer, messages)
    selected = selected_override if selected_override is not None else natural_planner_anchor_segments(segments, policy, graph_segments)
    if selected_override is None and not selected:
        selected = segments[:1]
    payload = make_payload(
        args,
        tokenizer,
        messages,
        selected,
        "natural_planner_warmup",
        f"natural_planner_warm:{instance['instance_id']}",
        instance["instance_id"],
    )
    payload["max_tokens"] = args.warmup_max_tokens
    # Keep Planner and target requests in the same internal family so the
    # exact-content code anchors created by warmup are eligible for target reuse.
    payload["template_task_family"] = "selective_ast_wholefile_reuse"
    pop_driver_anchor_telemetry(payload)
    await post_chat_optional_stream(session, args.port, payload, args.emit_ttft)
    return telemetry


async def warm_mode(
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    tokenizer: Any,
    instance: dict[str, Any],
    segments: list[CodeSegment],
    graph_segments: list[CodeSegment],
    policy: dict[str, Any],
    mode: str,
) -> None:
    if mode == "lossless_full_prefill":
        return
    messages = build_wholefile_messages(instance, segments, "planner_warmup_observed_wholefile")
    try:
        selected, _ = selected_segments_for_mode(segments, policy, mode, graph_segments, args, tokenizer, instance)
    except (ValueError, KeyError) as exc:
        # Spans from extended policy may not match the warm prompt exactly
        # (e.g. file_prefix/control_block text formatting); skip warmup
        # for that mode rather than aborting the whole case.
        print(f"[warmup-skip] {instance['instance_id']} mode={mode}: {exc}")
        return
    if not selected:
        return
    try:
        payload = make_payload(
            args,
            tokenizer,
            messages,
            selected,
            mode,
            f"warm:{instance['instance_id']}:{mode}",
            instance["instance_id"],
        )
    except (ValueError, KeyError) as exc:
        print(f"[warmup-skip] {instance['instance_id']} mode={mode} payload build: {exc}")
        return
    payload["max_tokens"] = 8
    pop_driver_anchor_telemetry(payload)
    await post_chat_optional_stream(session, args.port, payload, args.emit_ttft)


def skipped_target_row(
    instance: dict[str, Any],
    protocol: str,
    mode: str,
    warmup_status: str,
    exc: Exception,
    target_prompt_meta: dict[str, Any] | None = None,
    warmup_prompt_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    target_prompt_meta = target_prompt_meta or {}
    warmup_prompt_meta = warmup_prompt_meta or {}
    return {
        "instance_id": instance["instance_id"],
        "repo": instance.get("repo", ""),
        "warmup_protocol": protocol,
        "mode": mode,
        "elapsed_ms": None,
        "ttft_ms": None,
        "cached_tokens": 0,
        "estimated_reused_tokens": 0,
        "estimated_recomputed_tokens": 0,
        "selected_span_count": 0,
        "span_count": 0,
        "lossy_match_reason": None,
        "output_token_f1_vs_lossless": None,
        "warmup_status": f"{warmup_status};target_skipped:{type(exc).__name__}",
        "target_prompt_sha1": target_prompt_meta.get("prompt_sha1"),
        "target_prompt_chars": target_prompt_meta.get("prompt_chars"),
        "warmup_prompt_sha1": warmup_prompt_meta.get("prompt_sha1"),
        "warmup_prompt_chars": warmup_prompt_meta.get("prompt_chars"),
    }


async def run_target_mode(
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    tokenizer: Any,
    instance: dict[str, Any],
    segments: list[CodeSegment],
    graph_segments: list[CodeSegment],
    policy: dict[str, Any],
    mode: str,
    warmup_status: str,
    warmup_prompt_meta: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str]:
    protocol = args.warmup_protocol
    instance_id = instance["instance_id"]
    warmup_prompt_meta = warmup_prompt_meta or {}
    messages = build_wholefile_messages(instance, segments, mode)
    target_prompt_meta = prompt_telemetry(tokenizer, messages)
    try:
        selected, selection = selected_segments_for_mode(segments, policy, mode, graph_segments, args, tokenizer, instance)
        selected, selection = apply_selection_level_gate(selected, selection, args, mode)
        if not selection.get("selected_anchor_names"):
            selection = {**selection, "selected_anchor_names": [segment.name for segment in selected]}
        selection = attach_anchor_task_overlap(selection, instance)
        selected, selection = apply_hybrid_calibration_to_selection(
            args, instance_id, mode, selected, selection, segments
        )
        selected = maybe_bridge_prefix_anchors(args, segments, selected, mode)
        payload = make_payload(
            args,
            tokenizer,
            messages,
            selected,
            mode,
            f"target:{instance_id}:{mode}:{protocol}",
            instance_id,
            selection,
        )
        driver_anchor_telemetry = pop_driver_anchor_telemetry(payload)
    except (ValueError, KeyError) as exc:
        print(f"[mode-skip] {instance_id} mode={mode}: {exc}")
        return skipped_target_row(instance, protocol, mode, warmup_status, exc, target_prompt_meta, warmup_prompt_meta), ""

    start = now_ms()
    response = await post_chat_optional_stream(session, args.port, payload, args.emit_ttft)
    elapsed_ms = response["elapsed_ms"]
    ttft_ms = response.get("ttft_ms")
    output = response.get("text") or extract_text(response["body"]) or ""
    meta = {}
    meta.update(extract_lossy_meta(response["body"]))
    meta.update((response.get("metadata") or {}).get("lossy_reuse") or {})
    lossy_match_reason = meta.get("lossy_first_match_reason") or meta.get("lossy_final_match_reason")
    lossy_rejected_reason = meta.get("lossy_first_rejected_reason") or meta.get("lossy_final_rejected_reason")
    matched_content_signature = (
        meta.get("lossy_first_matched_content_signature")
        or meta.get("lossy_final_matched_content_signature")
    )
    return (
        {
            "instance_id": instance_id,
            "repo": instance.get("repo", ""),
            "warmup_protocol": protocol,
            "mode": mode,
            "elapsed_ms": round(elapsed_ms, 2),
            "ttft_ms": round(ttft_ms, 2) if ttft_ms is not None else None,
            "cached_tokens": extract_cached_tokens(response["body"]),
            "output_chars": len(output),
            "output_text": output,
            "lossy_match_reason": lossy_match_reason,
            "lossy_rejected_reason": lossy_rejected_reason,
            "lossy_reuse_allowed": meta.get("lossy_first_reuse_allowed")
            if meta.get("lossy_first_reuse_allowed") is not None
            else meta.get("lossy_final_reuse_allowed"),
            "lossy_candidate_count": meta.get("lossy_candidate_count"),
            "matched_content_signature": matched_content_signature,
            "lossy_anchor_match_used": meta.get("lossy_anchor_match_used"),
            "lossy_anchor_match_len": meta.get("lossy_anchor_match_len"),
            "lossy_anchor_multi_copy_count": meta.get("lossy_anchor_multi_copy_count"),
            "lossy_anchor_match_gap_len": meta.get("lossy_anchor_match_gap_len"),
            "lossy_anchor_gap_recompute_len": meta.get("lossy_anchor_gap_recompute_len"),
            "lossy_anchor_suffix_copy_len": meta.get("lossy_anchor_suffix_copy_len"),
            "lossy_anchor_suffix_copy_planned_len": meta.get("lossy_anchor_suffix_copy_planned_len"),
            "lossy_anchor_suffix_copy_cap_len": meta.get("lossy_anchor_suffix_copy_cap_len"),
            "lossy_anchor_suffix_copy_truncated": meta.get("lossy_anchor_suffix_copy_truncated"),
            "lossy_anchor_suffix_copy_semantic_len": meta.get("lossy_anchor_suffix_copy_semantic_len"),
            "lossy_anchor_suffix_copy_semantic_min_cosine": meta.get("lossy_anchor_suffix_copy_semantic_min_cosine"),
            "lossy_anchor_suffix_copy_semantic_truncated": meta.get("lossy_anchor_suffix_copy_semantic_truncated"),
            "lossy_anchor_suffix_recompute_head_len": meta.get("lossy_anchor_suffix_recompute_head_len"),
            "lossy_anchor_context_copy_ready": meta.get("lossy_anchor_context_copy_ready"),
            "lossy_anchor_context_aligned": meta.get("lossy_anchor_context_aligned"),
            "lossy_anchor_context_align_fail_reason": meta.get("lossy_anchor_context_align_fail_reason"),
            "lossy_anchor_context_align_stage": meta.get("lossy_anchor_context_align_stage"),
            "lossy_anchor_context_target_prefix_len": meta.get("lossy_anchor_context_target_prefix_len"),
            "lossy_anchor_context_prefix_signature_match": meta.get("lossy_anchor_context_prefix_signature_match"),
            "lossy_anchor_rope_delta": meta.get("lossy_anchor_rope_delta"),
            "lossy_anchor_store_entry_count": meta.get("lossy_anchor_store_entry_count"),
            "lossy_anchor_store_token_count": meta.get("lossy_anchor_store_token_count"),
            "lossy_anchor_store_lookup_entries": meta.get("lossy_anchor_store_lookup_entries"),
            "lossy_anchor_match_fail_reason": meta.get("lossy_anchor_match_fail_reason"),
            "lossy_anchor_token_mismatch_count": meta.get("lossy_anchor_token_mismatch_count"),
            "lossy_anchor_span_shape_mismatch_count": meta.get("lossy_anchor_span_shape_mismatch_count"),
            "lossy_anchor_prefix_covers_count": meta.get("lossy_anchor_prefix_covers_count"),
            "agenttemplatekv_prefetch_hit_count": meta.get("agenttemplatekv_prefetch_hit_count"),
            "agenttemplatekv_prefetch_protected_tokens": meta.get("agenttemplatekv_prefetch_protected_tokens"),
            "agenttemplatekv_prefetch_newly_protected_tokens": meta.get("agenttemplatekv_prefetch_newly_protected_tokens"),
            "codebase_prefetch_matched_tokens": meta.get("codebase_prefetch_matched_tokens"),
            "request_start_ms": round(start, 2),
            **selection,
            **driver_anchor_telemetry,
            "reuse_policy_name": reuse_policy_name(args, mode),
            "warmup_status": warmup_status,
            "target_prompt_sha1": target_prompt_meta["prompt_sha1"],
            "target_prompt_chars": target_prompt_meta["prompt_chars"],
            "warmup_prompt_sha1": warmup_prompt_meta.get("prompt_sha1"),
            "warmup_prompt_chars": warmup_prompt_meta.get("prompt_chars"),
            "raw_metadata": meta,
        },
        output,
    )


async def run_case(
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    tokenizer: Any,
    case: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any]:
    instance = case["instance"]
    segments = case["segments"]
    graph_segments = case.get("graph_segments", [])
    instance_id = instance["instance_id"]
    case_args = args_for_case_selector(args, instance_id)
    protocol = args.warmup_protocol
    case_warmup_status = protocol
    if protocol != "natural_planner" and args.flush_cache_per_case and not await flush_cache(session, args.port):
        raise RuntimeError(f"flush_cache failed for {instance_id}")

    rows = []
    outputs: dict[str, str] = {}

    if protocol == "natural_planner":
        if not await flush_cache(session, args.port):
            raise RuntimeError(f"flush_cache failed for {instance_id}:natural_planner_reference")
        row, output = await run_target_mode(
            session, case_args, tokenizer, instance, segments, graph_segments, policy,
            "lossless_full_prefill", "reference_cold_lossless",
        )
        rows.append(row)
        outputs["lossless_full_prefill"] = output
        if not await flush_cache(session, args.port):
            raise RuntimeError(f"flush_cache failed for {instance_id}:natural_planner")
        warmup_prompt_meta: dict[str, Any] = {}
        try:
            warmup_prompt_meta = await warm_natural_planner(session, case_args, tokenizer, instance, segments, graph_segments, policy)
        except (ValueError, KeyError) as exc:
            case_warmup_status = f"natural_planner_skipped:{type(exc).__name__}"
            print(f"[warmup-skip] {instance_id} protocol=natural_planner: {exc}")
        for mode in [m for m in active_modes(args) if m != "lossless_full_prefill"]:
            row, output = await run_target_mode(
                session, case_args, tokenizer, instance, segments, graph_segments, policy,
                mode, case_warmup_status, warmup_prompt_meta,
            )
            rows.append(row)
            outputs[mode] = output
    elif protocol == "fair_planner_per_mode":
        for mode in mode_order_for_protocol(args):
            warmup_status = "fair_planner_per_mode"
            warmup_prompt_meta = {}
            if not await flush_cache(session, args.port):
                raise RuntimeError(f"flush_cache failed for {instance_id}:{mode}:fair_planner_per_mode")
            warmup_selected: list[CodeSegment] | None = []
            if mode != "lossless_full_prefill":
                try:
                    warmup_selected, warmup_selection = selected_segments_for_mode(
                        segments, policy, mode, graph_segments, case_args, tokenizer, instance
                    )
                    warmup_selected, warmup_selection = apply_selection_level_gate(warmup_selected, warmup_selection, case_args, mode)
                    if not warmup_selection.get("selected_anchor_names"):
                        warmup_selection = {
                            **warmup_selection,
                            "selected_anchor_names": [segment.name for segment in warmup_selected],
                        }
                    warmup_selection = attach_anchor_task_overlap(warmup_selection, instance)
                    warmup_selected, _ = apply_hybrid_calibration_to_selection(
                        case_args, instance_id, mode, warmup_selected, warmup_selection, segments
                    )
                    warmup_selected = maybe_bridge_prefix_anchors(case_args, segments, warmup_selected, mode)
                except (ValueError, KeyError) as exc:
                    warmup_selected = []
                    warmup_status = f"fair_planner_per_mode_skipped:{type(exc).__name__}"
                    print(f"[warmup-skip] {instance_id} protocol=fair_planner_per_mode mode={mode}: {exc}")
            try:
                warmup_prompt_meta = await warm_natural_planner(
                    session,
                    case_args,
                    tokenizer,
                    instance,
                    segments,
                    graph_segments,
                    policy,
                    selected_override=warmup_selected,
                )
            except (ValueError, KeyError) as exc:
                warmup_status = f"fair_planner_per_mode_skipped:{type(exc).__name__}"
                print(f"[warmup-skip] {instance_id} protocol=fair_planner_per_mode mode={mode}: {exc}")
            row, output = await run_target_mode(
                session, case_args, tokenizer, instance, segments, graph_segments, policy,
                mode, warmup_status, warmup_prompt_meta,
            )
            rows.append(row)
            outputs[mode] = output
    else:
        for mode in active_modes(args):
            warmup_status = "none"
            warmup_prompt_meta = {}
            if not await flush_cache(session, args.port):
                raise RuntimeError(f"flush_cache failed for {instance_id}:{mode}")
            if protocol == "oracle_per_mode":
                await warm_mode(session, args, tokenizer, instance, segments, graph_segments, policy, mode)
                warmup_status = "oracle_per_mode"
                warmup_prompt_meta = prompt_telemetry(tokenizer, build_wholefile_messages(instance, segments, "planner_warmup_observed_wholefile"))
            row, output = await run_target_mode(
                session, args, tokenizer, instance, segments, graph_segments, policy,
                mode, warmup_status, warmup_prompt_meta,
            )
            rows.append(
                row
            )
            outputs[mode] = output

    baseline = outputs.get("lossless_full_prefill", "")
    for row in rows:
        text = outputs.get(row["mode"], "")
        row["output_exact_match_vs_lossless"] = text == baseline
        f1_value = round(token_f1(text, baseline), 4)
        row["output_token_f1_vs_lossless"] = f1_value
        row["token_f1_drop"] = round(max(0.0, 1.0 - f1_value), 4)
        row["accuracy_bucket"] = accuracy_bucket(f1_value, args.lossy_acceptable_f1_threshold)
    target_hashes = {
        row.get("target_prompt_sha1")
        for row in rows
        if row.get("elapsed_ms") is not None and row.get("target_prompt_sha1")
    }
    prompt_fair_ok = len(target_hashes) <= 1
    for row in rows:
        row["prompt_fair_ok"] = prompt_fair_ok
    print(f"[selective] {instance_id} done")
    return {
        "instance_id": instance_id,
        "repo": instance.get("repo", ""),
        "prompt_fair_ok": prompt_fair_ok,
        "target_prompt_sha1_set": sorted(target_hashes),
        "rows": rows,
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_mode: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_mode.setdefault(row["mode"], []).append(row)
    summary = {}
    for mode, items in by_mode.items():
        elapsed_items = [r for r in items if r.get("elapsed_ms") is not None]
        prompt_unfair_items = [r for r in elapsed_items if r.get("prompt_fair_ok") is False]
        ok_items = [r for r in elapsed_items if r.get("prompt_fair_ok") is not False]
        n_ok = len(ok_items)
        if n_ok == 0:
            summary[mode] = {
                "n": len(items),
                "n_ok": 0,
                "n_elapsed": len(elapsed_items),
                "n_prompt_unfair": len(prompt_unfair_items),
                "avg_elapsed_ms": None,
                "p50_elapsed_ms": None,
                "avg_cached_tokens": 0,
                "avg_estimated_reused_tokens": 0,
                "avg_estimated_recomputed_tokens": 0,
                "avg_payload_anchor_count": 0.0,
                "avg_payload_anchor_token_count": 0.0,
                "payload_lossy_enabled_rate": 0.0,
                "avg_payload_token_filter_dropped_count": 0.0,
                "avg_payload_anchor_start_filter_dropped_count": 0.0,
                "payload_anchor_max_total_reject_rate": 0.0,
                "avg_payload_anchor_max_total_pruned_count": 0.0,
                "hybrid_calibration_apply_rate": 0.0,
                "hybrid_calibration_reject_rate": 0.0,
                "hybrid_calibration_cap_rate": 0.0,
                "hybrid_calibration_shape_mismatch_rate": 0.0,
                "hybrid_calibration_shape_pruned_rate": 0.0,
                "anchor_match_rate": 0.0,
                "avg_anchor_match_len": 0.0,
                "avg_anchor_match_gap_len": 0.0,
                "avg_gap_recompute_len": 0.0,
                "avg_suffix_copy_len": 0.0,
                "avg_suffix_copy_planned_len": 0.0,
                "avg_suffix_recompute_head_len": 0.0,
                "suffix_copy_truncated_rate": 0.0,
                "context_aligned_match_rate": 0.0,
                "avg_anchor_store_lookup_entries": 0.0,
                "prefetch_hit_rate": 0.0,
                "avg_prefetch_hit_count": 0.0,
                "avg_prefetch_protected_tokens": 0.0,
                "avg_codebase_prefetch_matched_tokens": 0.0,
                "exact_hit_rate": 0.0,
                "avg_token_f1_vs_lossless": 0.0,
                "avg_token_f1_drop": 0.0,
                "accuracy_bucket_counts": {},
                "lossy_acceptable_rate": 0.0,
                "n_skipped": len(items),
                "skip_reason_counts": _skip_reason_counts(items),
                "predicted_d_reject_count": 0,
                "prompt_fair_ok_rate": 0.0,
                "target_prompt_sha1_count": len({r.get("target_prompt_sha1") for r in elapsed_items if r.get("target_prompt_sha1")}),
            }
            continue
        target_prompt_hashes = {r.get("target_prompt_sha1") for r in ok_items if r.get("target_prompt_sha1")}
        anchor_match_lens = [_safe_float(r.get("lossy_anchor_match_len")) for r in ok_items]
        anchor_gap_lens = [_safe_float(r.get("lossy_anchor_match_gap_len")) for r in ok_items]
        gap_recompute_lens = [_safe_float(r.get("lossy_anchor_gap_recompute_len")) for r in ok_items]
        suffix_copy_lens = [_safe_float(r.get("lossy_anchor_suffix_copy_len")) for r in ok_items]
        suffix_copy_planned_lens = [_safe_float(r.get("lossy_anchor_suffix_copy_planned_len")) for r in ok_items]
        suffix_recompute_head_lens = [_safe_float(r.get("lossy_anchor_suffix_recompute_head_len")) for r in ok_items]
        anchor_store_lookups = [_safe_float(r.get("lossy_anchor_store_lookup_entries")) for r in ok_items]
        prefetch_hits = [_safe_float(r.get("agenttemplatekv_prefetch_hit_count")) for r in ok_items]
        summary[mode] = {
            "n": len(items),
            "n_ok": n_ok,
            "n_elapsed": len(elapsed_items),
            "n_prompt_unfair": len(prompt_unfair_items),
            "avg_elapsed_ms": statistics.mean(float(r["elapsed_ms"]) for r in ok_items),
            "p50_elapsed_ms": statistics.median(float(r["elapsed_ms"]) for r in ok_items),
            "avg_cached_tokens": statistics.mean(float(r["cached_tokens"]) for r in ok_items),
            "avg_estimated_reused_tokens": statistics.mean(float(r["estimated_reused_tokens"]) for r in ok_items),
            "avg_estimated_recomputed_tokens": statistics.mean(float(r["estimated_recomputed_tokens"]) for r in ok_items),
            "avg_payload_anchor_count": statistics.mean(_safe_float(r.get("payload_anchor_count")) for r in ok_items),
            "avg_payload_anchor_token_count": statistics.mean(_safe_float(r.get("payload_anchor_token_count")) for r in ok_items),
            "payload_lossy_enabled_rate": statistics.mean(1.0 if r.get("payload_reuse_mode") == "lossy" else 0.0 for r in ok_items),
            "avg_payload_token_filter_dropped_count": statistics.mean(_safe_float(r.get("payload_token_filter_dropped_count")) for r in ok_items),
            "avg_payload_anchor_start_filter_dropped_count": statistics.mean(_safe_float(r.get("payload_anchor_start_filter_dropped_count")) for r in ok_items),
            "payload_anchor_max_total_reject_rate": statistics.mean(1.0 if _truthy(r.get("payload_anchor_max_total_rejected")) else 0.0 for r in ok_items),
            "avg_payload_anchor_max_total_pruned_count": statistics.mean(_safe_float(r.get("payload_anchor_max_total_pruned_count")) for r in ok_items),
            "hybrid_calibration_apply_rate": statistics.mean(1.0 if _truthy(r.get("hybrid_calibration_policy_applied")) else 0.0 for r in ok_items),
            "hybrid_calibration_reject_rate": statistics.mean(1.0 if _truthy(r.get("hybrid_calibration_rejected")) else 0.0 for r in ok_items),
            "hybrid_calibration_cap_rate": statistics.mean(1.0 if r.get("hybrid_calibration_action") == "cap" else 0.0 for r in ok_items),
            "hybrid_calibration_shape_mismatch_rate": statistics.mean(1.0 if _truthy(r.get("hybrid_calibration_shape_mismatch")) else 0.0 for r in ok_items),
            "hybrid_calibration_shape_pruned_rate": statistics.mean(1.0 if _truthy(r.get("hybrid_calibration_shape_pruned")) else 0.0 for r in ok_items),
            "anchor_match_rate": statistics.mean(1.0 if _truthy(r.get("lossy_anchor_match_used")) else 0.0 for r in ok_items),
            "avg_anchor_match_len": statistics.mean(anchor_match_lens) if anchor_match_lens else 0.0,
            "avg_anchor_match_gap_len": statistics.mean(anchor_gap_lens) if anchor_gap_lens else 0.0,
            "avg_gap_recompute_len": statistics.mean(gap_recompute_lens) if gap_recompute_lens else 0.0,
            "avg_suffix_copy_len": statistics.mean(suffix_copy_lens) if suffix_copy_lens else 0.0,
            "avg_suffix_copy_planned_len": statistics.mean(suffix_copy_planned_lens) if suffix_copy_planned_lens else 0.0,
            "avg_suffix_recompute_head_len": statistics.mean(suffix_recompute_head_lens) if suffix_recompute_head_lens else 0.0,
            "suffix_copy_truncated_rate": statistics.mean(1.0 if _truthy(r.get("lossy_anchor_suffix_copy_truncated")) else 0.0 for r in ok_items),
            "context_aligned_match_rate": statistics.mean(1.0 if _truthy(r.get("lossy_anchor_context_aligned")) else 0.0 for r in ok_items),
            "avg_anchor_store_lookup_entries": statistics.mean(anchor_store_lookups) if anchor_store_lookups else 0.0,
            "prefetch_hit_rate": statistics.mean(1.0 if _safe_float(r.get("agenttemplatekv_prefetch_hit_count")) > 0 else 0.0 for r in ok_items),
            "avg_prefetch_hit_count": statistics.mean(prefetch_hits) if prefetch_hits else 0.0,
            "avg_prefetch_protected_tokens": statistics.mean(_safe_float(r.get("agenttemplatekv_prefetch_protected_tokens")) for r in ok_items),
            "avg_codebase_prefetch_matched_tokens": statistics.mean(_safe_float(r.get("codebase_prefetch_matched_tokens")) for r in ok_items),
            "exact_hit_rate": statistics.mean(1.0 if r.get("lossy_match_reason") == "exact_code_content_signature" else 0.0 for r in ok_items),
            "exact_output_match_rate": statistics.mean(1.0 if r.get("output_exact_match_vs_lossless") else 0.0 for r in ok_items),
            "avg_token_f1_vs_lossless": statistics.mean(float(r["output_token_f1_vs_lossless"]) for r in ok_items if r.get("output_token_f1_vs_lossless") is not None) if any(r.get("output_token_f1_vs_lossless") is not None for r in ok_items) else 0.0,
            "avg_token_f1_drop": statistics.mean(_safe_float(r.get("token_f1_drop")) for r in ok_items),
            "accuracy_bucket_counts": {
                bucket: sum(1 for r in ok_items if r.get("accuracy_bucket") == bucket)
                for bucket in ("strict-safe", "lossy-acceptable", "aggressive-diagnostic", "unknown")
                if any(r.get("accuracy_bucket") == bucket for r in ok_items)
            },
            "lossy_acceptable_rate": statistics.mean(
                1.0 if r.get("accuracy_bucket") in {"strict-safe", "lossy-acceptable"} else 0.0
                for r in ok_items
            ),
            "predicted_d_reject_count": sum(1 for r in ok_items if _is_predicted_d_reject(r.get("lossy_rejected_reason"))),
            "prompt_fair_ok_rate": statistics.mean(1.0 if r.get("prompt_fair_ok") is not False else 0.0 for r in elapsed_items) if elapsed_items else 0.0,
            "target_prompt_sha1_count": len(target_prompt_hashes),
        }
        if n_ok != len(items):
            summary[mode]["n_skipped"] = len(items) - n_ok
            summary[mode]["skip_reason_counts"] = _skip_reason_counts(items)
        ttft_items = [r for r in ok_items if r.get("ttft_ms") is not None]
        if ttft_items:
            ttft_values = [float(r["ttft_ms"]) for r in ttft_items]
            summary[mode]["avg_ttft_ms"] = statistics.mean(ttft_values)
            summary[mode]["p50_ttft_ms"] = statistics.median(ttft_values)
            summary[mode]["p90_ttft_ms"] = _percentile(ttft_values, 90)
            summary[mode]["p99_ttft_ms"] = _percentile(ttft_values, 99)
    lossless_ttft = (summary.get("lossless_full_prefill") or {}).get("avg_ttft_ms")
    if lossless_ttft:
        for item in summary.values():
            avg_ttft = item.get("avg_ttft_ms")
            if avg_ttft:
                item["ttft_speedup_vs_lossless"] = float(lossless_ttft) / float(avg_ttft)
    lossless_by_case = {
        row.get("instance_id"): row
        for row in by_mode.get("lossless_full_prefill", [])
        if row.get("ttft_ms") is not None and row.get("prompt_fair_ok") is not False
    }
    if lossless_by_case:
        for mode, items in by_mode.items():
            if mode == "lossless_full_prefill":
                continue
            paired = [
                (lossless_by_case.get(row.get("instance_id")), row)
                for row in items
                if row.get("ttft_ms") is not None
                and row.get("prompt_fair_ok") is not False
                and lossless_by_case.get(row.get("instance_id")) is not None
            ]
            if not paired:
                continue
            paired_lossless_ttft = [float(lossless["ttft_ms"]) for lossless, _row in paired]
            paired_mode_ttft = [float(row["ttft_ms"]) for _lossless, row in paired]
            mode_summary = summary.get(mode)
            if not mode_summary:
                continue
            mode_summary["paired_n"] = len(paired)
            mode_summary["paired_avg_lossless_ttft_ms"] = statistics.mean(paired_lossless_ttft)
            mode_summary["paired_avg_ttft_ms"] = statistics.mean(paired_mode_ttft)
            if mode_summary["paired_avg_ttft_ms"]:
                mode_summary["paired_ttft_speedup_vs_lossless"] = (
                    mode_summary["paired_avg_lossless_ttft_ms"] / mode_summary["paired_avg_ttft_ms"]
                )
    return summary


def _safe_float(value: Any) -> float:
    try:
        if value is None or value == "":
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _is_predicted_d_reject(reason: Any) -> bool:
    text = str(reason or "").lower()
    return "predicted" in text or "confidence" in text or "distance" in text


def _skip_reason_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in items:
        if row.get("elapsed_ms") is not None:
            continue
        reason = str(row.get("warmup_status") or "skipped:unknown")
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = max(0, min(len(s) - 1, int(round((pct / 100.0) * (len(s) - 1)))))
    return float(s[idx])


def write_outputs(args: argparse.Namespace, cases: list[dict[str, Any]], rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(
            {
                "git_commit": git_commit(),
                "selector_snapshot": selector_snapshot(args),
                "model": args.model,
                "policy": str(args.policy),
                "dataset": str(args.dataset),
                "manifest": str(args.manifest),
                "all_cases": args.all_cases,
                "expected_case_count": args.expected_case_count,
                "modes": active_modes(args),
                "target_mode_order": mode_order_for_protocol(args),
                "warmup_protocol": args.warmup_protocol,
                "warmup_protocol_description": warmup_protocol_description(args.warmup_protocol),
                "warmup_max_tokens": args.warmup_max_tokens,
                "max_tokens": args.max_tokens,
                "server_random_seed": args.server_random_seed,
                "disable_overlap_schedule": args.disable_overlap_schedule,
                "lossy_fuzzy_match": True,
                "lossy_max_zero_gap": args.lossy_max_zero_gap,
                "lossy_max_suffix_copy_len": args.lossy_max_suffix_copy_len,
                "lossy_max_planned_suffix_copy_len": args.lossy_max_planned_suffix_copy_len,
                "lossy_suffix_recompute_head_len": args.lossy_suffix_recompute_head_len,
                "lossy_max_recompute_gap_len": args.lossy_max_recompute_gap_len,
                "lossy_acceptable_f1_threshold": args.lossy_acceptable_f1_threshold,
                "lossy_recompute_gap": args.lossy_recompute_gap,
                "lossy_stage_recompute_gap": args.lossy_stage_recompute_gap,
                "lossy_multi_anchor_copy": args.lossy_multi_anchor_copy,
                "enable_bridge_prefix_anchors": args.enable_bridge_prefix_anchors,
                "bridge_anchor_max_tokens": args.bridge_anchor_max_tokens,
                "disable_graph_bridge_prefix_anchors": args.disable_graph_bridge_prefix_anchors,
                "enable_graph_aware_lossy": args.enable_graph_aware_lossy,
                "load_graph_bundles_for_selection": args.load_graph_bundles_for_selection,
                "enable_hybrid_code_aware_lossy": args.enable_hybrid_code_aware_lossy,
                "hybrid_calibration_policy": str(args.hybrid_calibration_policy) if args.hybrid_calibration_policy else "",
                "hybrid_calibration_policy_cases": len((getattr(args, "_hybrid_calibration_policy_data", {}) or {}).get("cases") or {}),
                "hybrid_calibration_policy_rules": len((getattr(args, "_hybrid_calibration_policy_data", {}) or {}).get("rules") or []),
                "hybrid_calibration_policy_default_action": (getattr(args, "_hybrid_calibration_policy_data", {}) or {}).get("default_action", ""),
                "hybrid_min_bridge_tokens": args.hybrid_min_bridge_tokens,
                "hybrid_max_bridge_tokens": args.hybrid_max_bridge_tokens,
                "hybrid_bridge_anchor_max_tokens": args.hybrid_bridge_anchor_max_tokens,
                "hybrid_bridge_max_count_per_file": args.hybrid_bridge_max_count_per_file,
                "include_hybrid_bridge_seed_spans": args.include_hybrid_bridge_seed_spans,
                "hybrid_bridge_source": args.hybrid_bridge_source,
                "hybrid_risk_large_bridge_min_tokens": args.hybrid_risk_large_bridge_min_tokens,
                "hybrid_risk_max_large_bridge_count": args.hybrid_risk_max_large_bridge_count,
                "hybrid_risk_max_graph_tokens_for_large_bridge": args.hybrid_risk_max_graph_tokens_for_large_bridge,
                "code_graph_bundle_manifest": str(args.code_graph_bundle_manifest),
                "graph_bundle_policy": args.graph_bundle_policy,
                "graph_bundle_role": args.graph_bundle_role,
                "graph_anchor_token_budget": args.graph_anchor_token_budget,
                "graph_anchor_max_span_tokens": args.graph_anchor_max_span_tokens,
                "graph_anchor_lowspan_max_tokens": args.graph_anchor_lowspan_max_tokens,
                "graph_anchor_lowspan_suffix_copy_cap": args.graph_anchor_lowspan_suffix_copy_cap,
                "graph_anchor_smallspan_max_tokens": args.graph_anchor_smallspan_max_tokens,
                "graph_anchor_smallspan_suffix_copy_cap": args.graph_anchor_smallspan_suffix_copy_cap,
                "graph_anchor_midspan_min_tokens": args.graph_anchor_midspan_min_tokens,
                "graph_anchor_midspan_max_tokens": args.graph_anchor_midspan_max_tokens,
                "graph_anchor_midspan_suffix_copy_cap": args.graph_anchor_midspan_suffix_copy_cap,
                "anchor_lowspan_max_tokens": args.anchor_lowspan_max_tokens,
                "anchor_lowspan_suffix_copy_cap": args.anchor_lowspan_suffix_copy_cap,
                "anchor_smallspan_max_tokens": args.anchor_smallspan_max_tokens,
                "anchor_smallspan_suffix_copy_cap": args.anchor_smallspan_suffix_copy_cap,
                "anchor_midspan_min_tokens": args.anchor_midspan_min_tokens,
                "anchor_midspan_max_tokens": args.anchor_midspan_max_tokens,
                "anchor_midspan_suffix_copy_cap": args.anchor_midspan_suffix_copy_cap,
                "selective_anchor_max_span_tokens": args.selective_anchor_max_span_tokens,
                "selective_anchor_min_span_tokens": args.selective_anchor_min_span_tokens,
                "selective_anchor_max_start_token": args.selective_anchor_max_start_token,
                "anchor_min_total_tokens": args.anchor_min_total_tokens,
                "anchor_max_total_tokens": args.anchor_max_total_tokens,
                "anchor_max_total_policy": args.anchor_max_total_policy,
                "selection_min_estimated_reused_tokens": args.selection_min_estimated_reused_tokens,
                "exclude_anchor_granularities": args.exclude_anchor_granularities,
                "prompt_fair_cases": sum(1 for case in cases if case.get("prompt_fair_ok") is not False),
                "prompt_unfair_cases": [
                    case.get("instance_id")
                    for case in cases
                    if case.get("prompt_fair_ok") is False
                ],
                "summary": summary,
                "cases": cases,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    with (args.out_dir / "selective_wholefile_rows.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CSV_FIELDNAMES})
    lines = [
        "# Selective whole-file AST reuse",
        "",
        f"- Warmup protocol: `{args.warmup_protocol}`",
        f"- Protocol meaning: {warmup_protocol_description(args.warmup_protocol)}",
        f"- Max generation tokens: `{args.max_tokens}`",
        f"- Server random seed: `{args.server_random_seed}`",
        f"- Disable overlap schedule: `{args.disable_overlap_schedule}`",
        f"- Context-aligned recompute gap: `{args.lossy_recompute_gap}`",
        f"- Stage recompute gap diagnostic: `{args.lossy_stage_recompute_gap}`",
        f"- Multi-anchor copy diagnostic: `{args.lossy_multi_anchor_copy}`",
        f"- Bridge prefix anchors: `{args.enable_bridge_prefix_anchors}`",
        f"- Bridge anchor max tokens: `{args.bridge_anchor_max_tokens}`",
        f"- Disable graph bridge prefix anchors: `{args.disable_graph_bridge_prefix_anchors}`",
        f"- Load graph bundles for selection only: `{args.load_graph_bundles_for_selection}`",
        f"- Hybrid calibration policy: `{args.hybrid_calibration_policy}`",
        f"- Hybrid calibration policy cases: `{len((getattr(args, '_hybrid_calibration_policy_data', {}) or {}).get('cases') or {})}`",
        f"- Hybrid calibration policy rules: `{len((getattr(args, '_hybrid_calibration_policy_data', {}) or {}).get('rules') or [])}`",
        f"- Hybrid calibration default action: `{(getattr(args, '_hybrid_calibration_policy_data', {}) or {}).get('default_action', '')}`",
        f"- Hybrid bridge tokens: min `{args.hybrid_min_bridge_tokens}`, max `{args.hybrid_max_bridge_tokens}`",
        f"- Hybrid bridge anchor max tokens: `{args.hybrid_bridge_anchor_max_tokens}`",
        f"- Hybrid bridge max count per file: `{args.hybrid_bridge_max_count_per_file}`",
        f"- Include hybrid bridge seed spans: `{args.include_hybrid_bridge_seed_spans}`",
        f"- Hybrid bridge source: `{args.hybrid_bridge_source}`",
        f"- Hybrid large-bridge risk gate: min tokens `{args.hybrid_risk_large_bridge_min_tokens}`, max bridge count `{args.hybrid_risk_max_large_bridge_count}`, max graph tokens `{args.hybrid_risk_max_graph_tokens_for_large_bridge}`",
        f"- Max suffix copy len: `{args.lossy_max_suffix_copy_len}`",
        f"- Max planned suffix copy len: `{args.lossy_max_planned_suffix_copy_len}`",
        f"- Suffix recompute head len: `{args.lossy_suffix_recompute_head_len}`",
        f"- Max recompute gap len: `{args.lossy_max_recompute_gap_len}`",
        f"- Lossy acceptable F1 threshold: `{args.lossy_acceptable_f1_threshold}`",
        f"- Selective anchor max span tokens: `{args.selective_anchor_max_span_tokens}`",
        f"- Selective anchor min span tokens: `{args.selective_anchor_min_span_tokens}`",
        f"- Selective anchor max start token: `{args.selective_anchor_max_start_token}`",
        f"- Anchor min total tokens: `{args.anchor_min_total_tokens}`",
        f"- Anchor max total tokens: `{args.anchor_max_total_tokens}`",
        f"- Anchor max total policy: `{args.anchor_max_total_policy}`",
        f"- Selection min estimated reused tokens: `{args.selection_min_estimated_reused_tokens}`",
        f"- Excluded anchor granularities: `{args.exclude_anchor_granularities}`",
        f"- Graph anchor token budget: `{args.graph_anchor_token_budget}`",
        f"- Graph anchor max span tokens: `{args.graph_anchor_max_span_tokens}`",
        f"- Graph anchor lowspan max tokens: `{args.graph_anchor_lowspan_max_tokens}`",
        f"- Graph anchor lowspan suffix cap: `{args.graph_anchor_lowspan_suffix_copy_cap}`",
        f"- Graph anchor smallspan max tokens: `{args.graph_anchor_smallspan_max_tokens}`",
        f"- Graph anchor smallspan suffix cap: `{args.graph_anchor_smallspan_suffix_copy_cap}`",
        f"- Graph anchor midspan range: `{args.graph_anchor_midspan_min_tokens}`-`{args.graph_anchor_midspan_max_tokens}`",
        f"- Graph anchor midspan suffix cap: `{args.graph_anchor_midspan_suffix_copy_cap}`",
        f"- Generic anchor lowspan max tokens: `{args.anchor_lowspan_max_tokens}`",
        f"- Generic anchor lowspan suffix cap: `{args.anchor_lowspan_suffix_copy_cap}`",
        f"- Generic anchor smallspan max tokens: `{args.anchor_smallspan_max_tokens}`",
        f"- Generic anchor smallspan suffix cap: `{args.anchor_smallspan_suffix_copy_cap}`",
        f"- Generic anchor midspan range: `{args.anchor_midspan_min_tokens}`-`{args.anchor_midspan_max_tokens}`",
        f"- Generic anchor midspan suffix cap: `{args.anchor_midspan_suffix_copy_cap}`",
        "",
        "| mode | n_ok/n | prompt fair | avg elapsed ms | avg TTFT ms | speedup vs lossless | paired speedup | avg cached | est reused | anchor match len | gap recompute | suffix head | suffix copy | planned copy | trunc rate | context aligned | token F1 | F1 drop | acceptable | buckets |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for mode in mode_order_for_protocol(args):
        if mode not in summary:
            continue
        s = summary[mode]
        avg_ttft = s.get("avg_ttft_ms")
        avg_ttft_text = f"{avg_ttft:.1f}" if avg_ttft is not None else ""
        avg_elapsed = s.get("avg_elapsed_ms")
        avg_elapsed_text = f"{avg_elapsed:.1f}" if avg_elapsed is not None else ""
        speedup = s.get("ttft_speedup_vs_lossless")
        speedup_text = f"{speedup:.2f}x" if speedup is not None else ""
        paired_speedup = s.get("paired_ttft_speedup_vs_lossless")
        paired_speedup_text = f"{paired_speedup:.2f}x" if paired_speedup is not None else ""
        fair_rate = s.get("prompt_fair_ok_rate")
        fair_text = f"{fair_rate:.2f}" if fair_rate is not None else ""
        lines.append(
            f"| `{mode}` | {s['n_ok']}/{s['n']} | {fair_text} | {avg_elapsed_text} | {avg_ttft_text} | {speedup_text} | {paired_speedup_text} | "
            f"{s['avg_cached_tokens']:.1f} | {s['avg_estimated_reused_tokens']:.1f} | "
            f"{s.get('avg_anchor_match_len', 0.0):.1f} | "
            f"{s.get('avg_gap_recompute_len', 0.0):.1f} | "
            f"{s.get('avg_suffix_recompute_head_len', 0.0):.1f} | "
            f"{s.get('avg_suffix_copy_len', 0.0):.1f} | "
            f"{s.get('avg_suffix_copy_planned_len', 0.0):.1f} | "
            f"{s.get('suffix_copy_truncated_rate', 0.0):.2f} | "
            f"{s.get('context_aligned_match_rate', 0.0):.2f} | "
            f"{s['avg_token_f1_vs_lossless']:.4f} | "
            f"{s.get('avg_token_f1_drop', 0.0):.4f} | "
            f"{s.get('lossy_acceptable_rate', 0.0):.2f} | "
            f"`{json.dumps(s.get('accuracy_bucket_counts', {}), ensure_ascii=False)}` |"
        )
    lines += [
        "",
        "Interpretation: agents receive whole-file code_base prompts in every mode; only the internal AST spans exposed to the exact-content KV reuse gate differ.",
        "`graph_aware_lossy` keeps the same whole-file prompt, then maps relation-selected graph bundles back to exact AST spans already present in that prompt.",
        "`est reused` is driver-side span volume; `anchor match len` and `prefetch hit` are server-side evidence that KV was actually copied/protected.",
        "`gap recompute` / `suffix copy` are context-aligned reuse telemetry. In this build, large-gap staged recompute is recorded as unsupported rather than approximated with zero-filled KV.",
        f"Accuracy buckets: `strict-safe` means token F1 is effectively 1.0; `lossy-acceptable` means token F1 >= {args.lossy_acceptable_f1_threshold}; `aggressive-diagnostic` means token F1 is below that threshold.",
        "`whole_file_reuse_all` is diagnostic only. Main reported methods should be the prompt-fair selective/hybrid code-aware rows.",
    ]
    if args.warmup_protocol == "oracle_per_mode":
        lines.append("This is a controlled mechanism upper bound: each mode gets its own isolated warmup before target measurement.")
    elif args.warmup_protocol == "natural_planner":
        lines.append("This is the realistic protocol: lossless is measured cold as the reference, then one Planner-style warmup is shared by later reuse target modes; target order is recorded in `summary.json`.")
    elif args.warmup_protocol == "fair_planner_per_mode":
        lines.append("This is the prompt-fair protocol: each mode starts from a fresh cache, runs the same Planner warmup, and then measures the same target prompt. Rows from prompt-unfair cases are excluded from mode aggregates.")
    else:
        lines.append("This is the cold protocol: target modes are flushed and measured without any warmup request.")
    unfair_cases = [case.get("instance_id") for case in cases if case.get("prompt_fair_ok") is False]
    if unfair_cases:
        lines += ["", "## Prompt fairness failures", "", "```json", json.dumps(unfair_cases, indent=2, ensure_ascii=False), "```"]
    skipped = {
        mode: s.get("skip_reason_counts")
        for mode, s in summary.items()
        if s.get("skip_reason_counts")
    }
    if skipped:
        lines += ["", "## Skipped rows", "", "```json", json.dumps(skipped, indent=2, ensure_ascii=False), "```"]
    (args.out_dir / "SELECTIVE_WHOLEFILE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


async def run_benchmark(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.selective_mode == "extended" and args.policy.name == "selective_reuse_policy.json":
        # Auto-pick the extended policy file when user asks for extended mode
        # but didn't explicitly point to an extended policy.
        extended_path = args.policy.with_name("selective_reuse_policy_extended.json")
        if extended_path.exists():
            print(f"[selective-mode=extended] using policy {extended_path}")
            args.policy = extended_path
    args._hybrid_calibration_policy_data = load_hybrid_calibration_policy(args)
    args._case_selector_overrides_data = load_case_selector_overrides(args)
    policy = load_selective_policy(args.policy)
    if args.dry_run_load_cases:
        cases = load_cases(args, policy)
        if args.expected_case_count is not None and len(cases) != args.expected_case_count:
            raise RuntimeError(f"expected {args.expected_case_count} cases, loaded {len(cases)}")
        total_files = sum(len(case["segments"]) for case in cases)
        total_chars = sum(len(segment.text) for case in cases for segment in case["segments"])
        print(
            json.dumps(
                {
                    "loaded_cases": len(cases),
                    "loaded_files": total_files,
                    "avg_chars_per_file": round(total_chars / total_files, 2) if total_files else 0.0,
                    "manifest": str(args.manifest),
                    "dataset": str(args.dataset),
                    "warmup_protocol": args.warmup_protocol,
                    "warmup_protocol_description": warmup_protocol_description(args.warmup_protocol),
                    "enable_graph_aware_lossy": args.enable_graph_aware_lossy,
                    "hybrid_calibration_policy": str(args.hybrid_calibration_policy) if args.hybrid_calibration_policy else "",
                    "hybrid_calibration_policy_cases": len((getattr(args, "_hybrid_calibration_policy_data", {}) or {}).get("cases") or {}),
                    "hybrid_calibration_policy_rules": len((getattr(args, "_hybrid_calibration_policy_data", {}) or {}).get("rules") or []),
                    "hybrid_calibration_policy_default_action": (getattr(args, "_hybrid_calibration_policy_data", {}) or {}).get("default_action", ""),
                    "graph_bundle_manifest": str(args.code_graph_bundle_manifest),
                    "graph_cases": sum(1 for case in cases if case.get("graph_segments")),
                },
                indent=2,
            )
        )
        return
    if args.dry_run_selection_features:
        cases = load_cases(args, policy)
        if args.expected_case_count is not None and len(cases) != args.expected_case_count:
            raise RuntimeError(f"expected {args.expected_case_count} cases, loaded {len(cases)}")
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        write_selection_feature_dry_run(args, tokenizer, cases, policy)
        return
    proc: subprocess.Popen | None = None
    if args.reuse_server:
        if not await wait_ready(args.port, timeout_s=10):
            raise RuntimeError(f"--reuse-server: no server ready on port {args.port}")
    else:
        kill_port(args.port)
        proc = launch_server(args)
    try:
        if not args.reuse_server:
            if not await wait_ready(args.port, args.server_timeout):
                raise RuntimeError("server failed to become ready")
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        cases = load_cases(args, policy)
        if args.expected_case_count is not None and len(cases) != args.expected_case_count:
            raise RuntimeError(f"expected {args.expected_case_count} cases, loaded {len(cases)}")
        rows = []
        case_outputs = []
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=args.eval_timeout)) as session:
            for case in cases:
                result = await run_case(session, args, tokenizer, case, policy)
                case_outputs.append(result)
                rows.extend(result["rows"])
                write_outputs(args, case_outputs, rows, aggregate(rows))
        write_outputs(args, case_outputs, rows, aggregate(rows))
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
            kill_port(args.port)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--max-cases", type=int, default=10)
    parser.add_argument("--all-cases", action="store_true")
    parser.add_argument("--expected-case-count", type=int)
    parser.add_argument("--dry-run-load-cases", action="store_true")
    parser.add_argument("--dry-run-selection-features", action="store_true",
                        help="Load cases and write per-case selected-anchor features without launching a server.")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--file-start-index", type=int, default=0)
    parser.add_argument("--files-per-case", type=int, default=2)
    parser.add_argument("--prefer-selective-files", action="store_true")
    parser.add_argument("--prefer-graph-target-files", dest="prefer_graph_target_files", action="store_true",
                        help="When graph-aware mode is enabled, prefer graph bundle target files before the generic selective-file ranking.")
    parser.add_argument("--no-prefer-graph-target-files", dest="prefer_graph_target_files", action="store_false",
                        help="Keep the original selective-file ranking even when graph-aware mode is enabled.")
    parser.set_defaults(prefer_graph_target_files=None)
    parser.add_argument("--max-complete-file-chars", type=int, default=60000)
    parser.add_argument("--max-file-chars", type=int, default=22000)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--max-total-tokens", type=int, default=32768)
    parser.add_argument("--max-prefill-tokens", type=int, default=8192)
    parser.add_argument("--enable-semantic-suffix", dest="enable_semantic_suffix",
                        action="store_true", default=True,
                        help="Enable semantic suffix-copy length decider (replaces hand-tuned caps with per-chunk cosine profile). Default ON.")
    parser.add_argument("--disable-semantic-suffix", dest="enable_semantic_suffix",
                        action="store_false",
                        help="Disable semantic suffix-copy length decider; fall back to legacy caps.")
    parser.add_argument("--mem-fraction-static", type=float, default=0.72)
    parser.add_argument("--disable-overlap-schedule", action="store_true",
                        help="Launch SGLang with serialized scheduling. This is slower to warm up but avoids transient KV/flashinfer instability on the 24GB testbed.")
    parser.add_argument("--server-random-seed", type=int, default=42,
                        help="Pass --random-seed to the SGLang server for reproducible benchmark runs. Use an empty value only by editing the driver; CLI default is 42.")
    parser.add_argument("--context-aware-max-predicted-d", type=float,
                        help="Set SGLANG_CONTEXT_AWARE_MAX_PREDICTED_D to reject lossy exact-content matches above this predicted KV-distance floor.")
    parser.add_argument("--lossy-max-zero-gap", type=int,
                        help="Set SGLANG_LOSSY_MAX_ZERO_GAP for explicit gap-enabled prompt-fair diagnostics. Leave unset to use the runtime default.")
    parser.add_argument("--lossy-max-suffix-copy-len", type=int, default=1024,
                        help="Set SGLANG_LOSSY_MAX_SUFFIX_COPY_LEN to bound lossy exact-anchor suffix copy length. Use 0 for unbounded copy.")
    parser.add_argument("--lossy-max-planned-suffix-copy-len", type=int, default=0,
                        help="Set SGLANG_LOSSY_MAX_PLANNED_SUFFIX_COPY_LEN to reject anchors whose remaining suffix exceeds this length before bounded copy. Use 0 to disable.")
    parser.add_argument("--lossy-suffix-recompute-head-len", type=int, default=0,
                        help="Set SGLANG_LOSSY_SUFFIX_RECOMPUTE_HEAD_LEN to recompute the first N tokens after an anchor start before copying the remaining suffix.")
    parser.add_argument("--lossy-max-recompute-gap-len", type=int, default=0,
                        help="Set SGLANG_LOSSY_MAX_RECOMPUTE_GAP_LEN to reject context-aligned staged copies whose planned recompute distance exceeds this value. Use 0 to disable.")
    parser.add_argument("--lossy-acceptable-f1-threshold", type=float, default=0.90,
                        help="Token F1 threshold for classifying lossy reuse as acceptable in Pareto reports.")
    parser.add_argument("--selective-anchor-max-span-tokens", type=int, default=0,
                        help="Filter non-graph selective KV anchors whose tokenized span is longer than this value before sending runtime metadata. Use 0 to disable.")
    parser.add_argument("--selective-anchor-min-span-tokens", type=int, default=0,
                        help="Filter KV anchors whose tokenized span is shorter than this value before sending runtime metadata. Use 0 to disable.")
    parser.add_argument("--selective-anchor-max-start-token", type=int, default=0,
                        help="Filter KV anchors that start after this token index in the target prompt. Use 0 to disable.")
    parser.add_argument("--anchor-min-total-tokens", type=int, default=0,
                        help="Disable lossy anchor metadata when the selected anchors contain fewer than this many prompt tokens. Use 0 to disable.")
    parser.add_argument("--anchor-max-total-tokens", type=int, default=0,
                        help="Disable lossy anchor metadata when the selected anchors contain more than this many prompt tokens. Use 0 to disable.")
    parser.add_argument("--anchor-max-total-policy", choices=("reject", "prune_shortest", "prune_first"), default="reject",
                        help="Policy when selected anchors exceed --anchor-max-total-tokens: reject all anchors, keep shortest anchors under budget, or keep prompt-order anchors under budget.")
    parser.add_argument("--selection-min-estimated-reused-tokens", type=int, default=0,
                        help="Disable lossy anchor metadata before bridge-prefix expansion when driver-side selected spans estimate fewer reused tokens than this value. Use 0 to disable.")
    parser.add_argument("--lossy-recompute-gap", action="store_true",
                        help="Set SGLANG_LOSSY_RECOMPUTE_GAP=1. Current runtime records context-aligned large-gap reuse as unsupported instead of zero-filling the gap.")
    parser.add_argument("--lossy-stage-recompute-gap", action="store_true",
                        help="Diagnostic only: force chunked prefill to the anchor start before re-matching. This may change output due chunked-prefill numerics and must not be used as a main prompt-fair result unless F1 remains 1.0.")
    parser.add_argument("--lossy-multi-anchor-copy", action="store_true",
                        help="Experimental: after a successful lossy anchor copy, continue copying later anchors whose start is already covered by the current prefix/copy. Does not zero-fill new gaps.")
    parser.add_argument("--enable-bridge-prefix-anchors", action="store_true",
                        help="Add exact file-prefix bridge anchors for selected deep AST/graph spans without changing target prompt text.")
    parser.add_argument("--bridge-anchor-max-tokens", type=int, default=0,
                        help="When bridge-prefix anchors are enabled, build bounded prompt-resident bridge windows with at most this many approximate whitespace tokens. Use 0 for the legacy file-start bridge.")
    parser.add_argument("--disable-graph-bridge-prefix-anchors", action="store_true",
                        help="When bridge anchors are enabled globally, keep graph_aware_lossy on the selected AST span itself instead of expanding it to a file-prefix/window bridge.")
    parser.add_argument("--enable-hybrid-code-aware-lossy", action="store_true",
                        help="Add hybrid_code_aware_lossy: large safe function/method bridge anchors plus graph-selected small anchors, without changing target prompt text.")
    parser.add_argument("--hybrid-calibration-policy", type=Path,
                        help="Experimental diagnostic JSON policy for hybrid_code_aware_lossy per-case allow/reject/cap decisions. This must be reported as calibration/upper diagnostic, not a held-out main result.")
    parser.add_argument("--case-selector-overrides", type=Path,
                        help="Optional JSON instance_id -> selector-arg overrides. This changes only internal anchor selection, not prompt text; report as a calibrated selector profile.")
    parser.add_argument("--hybrid-min-bridge-tokens", type=int, default=4000,
                        help="Minimum approximate whitespace tokens for large function/method bridge anchors in hybrid_code_aware_lossy.")
    parser.add_argument("--hybrid-max-bridge-tokens", type=int, default=0,
                        help="Maximum approximate whitespace tokens for large function/method bridge anchors in hybrid_code_aware_lossy. 0 uses --lossy-max-planned-suffix-copy-len.")
    parser.add_argument("--hybrid-bridge-anchor-max-tokens", type=int, default=0,
                        help="Build bounded prompt-resident hybrid bridge windows with at most this many approximate whitespace tokens. 0 keeps the legacy file-start bridge.")
    parser.add_argument("--hybrid-bridge-max-count-per-file", type=int, default=0,
                        help="For hybrid bridge candidates, keep at most this many deepest windows per file after token filters. 0 keeps all candidates.")
    parser.add_argument("--include-hybrid-bridge-seed-spans", action="store_true",
                        help="Also include the AST/graph seed spans used to build hybrid bridge anchors. Useful for calibrated profiles that expect bridge+symbol anchors.")
    parser.add_argument("--hybrid-bridge-source", choices=["function", "function_then_extended", "graph", "graph_then_function", "extended", "extended_then_function", "task_ast", "task_ast_direct"], default="function",
                        help="Span source used to build hybrid bridge-prefix anchors. Default keeps legacy function/method behavior; graph variants use graph-mapped spans and extended variants use extended AST spans without changing target prompt text. function_then_extended falls back to extended AST only when function/method seeds are absent.")
    parser.add_argument("--hybrid-task-ast-top-k", type=int, default=3,
                        help="For --hybrid-bridge-source task_ast, select this many task-overlap AST spans before bridge-window expansion.")
    parser.add_argument("--hybrid-risk-large-bridge-min-tokens", type=int, default=0,
                        help="Treat hybrid bridge anchors at or above this tokenized length as large for risk gating. Use 0 to disable the risk gate.")
    parser.add_argument("--hybrid-risk-max-large-bridge-count", type=int, default=0,
                        help="Reject all hybrid lossy anchors when the number of large bridge anchors exceeds this count. Use 0 to disable this condition.")
    parser.add_argument("--hybrid-risk-max-graph-tokens-for-large-bridge", type=int, default=0,
                        help="Reject all hybrid lossy anchors when a large bridge co-occurs with more graph-selected tokens than this value. Use 0 to disable this condition.")
    parser.add_argument("--force-evict", action="store_true")
    parser.add_argument("--max-running-requests", type=int, default=1)
    parser.add_argument("--flush-cache-per-case", action="store_true")
    parser.add_argument("--warmup-protocol", choices=WARMUP_PROTOCOLS,
                        help="Warmup protocol: none=cold per-mode targets; oracle_per_mode=old controlled upper bound; natural_planner=one Planner warmup shared by target modes; fair_planner_per_mode=fresh cache + identical Planner warmup + identical target prompt per mode. Defaults to natural_planner unless legacy warmup flags are used.")
    parser.add_argument("--warmup-max-tokens", type=int, default=8,
                        help="Number of tokens generated by warmup requests.")
    parser.add_argument("--flush-cache-per-mode", dest="flush_cache_per_mode", action="store_true",
                        help="Legacy alias for --warmup-protocol oracle_per_mode.")
    parser.add_argument("--shared-cache-across-modes", dest="flush_cache_per_mode", action="store_false",
                        help="Legacy alias for --warmup-protocol natural_planner.")
    parser.set_defaults(flush_cache_per_mode=None)
    parser.add_argument("--server-timeout", type=int, default=180)
    parser.add_argument("--eval-timeout", type=int, default=1200)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--emit-ttft", action="store_true",
                        help="Use streaming post_chat_stream and record per-mode ttft_ms in result rows and summary.")
    parser.add_argument("--reuse-server", action="store_true",
                        help="Skip launch_server and connect to the existing SGLang server on --port. Use when a server is already running.")
    parser.add_argument("--selective-mode", default="function_method",
                        choices=["function_method", "extended", "oracle"],
                        help="Which selective policy to use. function_method = default 2-granularity; extended = 4-granularity (function/method/control_block/file_prefix); oracle = all reuse candidates.")
    parser.add_argument("--target-modes", default="",
                        help="Comma-separated subset of modes to run. Empty means all enabled modes.")
    parser.add_argument("--exclude-anchor-granularities", default="",
                        help="Comma-separated AST granularities to remove from selected KV anchors after selection, e.g. file_prefix. Target prompt is unchanged.")
    parser.add_argument("--anchor-lowspan-max-tokens", type=int, default=0,
                        help="Generic non-graph tokenized span length upper bound for applying --anchor-lowspan-suffix-copy-cap. Use 0 to disable.")
    parser.add_argument("--anchor-lowspan-suffix-copy-cap", type=int, default=0,
                        help="Generic non-graph per-anchor max suffix copy length for spans at or below --anchor-lowspan-max-tokens. Use 0 to disable.")
    parser.add_argument("--anchor-smallspan-max-tokens", type=int, default=0,
                        help="Generic non-graph tokenized span length upper bound for applying --anchor-smallspan-suffix-copy-cap. Use 0 to disable.")
    parser.add_argument("--anchor-smallspan-suffix-copy-cap", type=int, default=0,
                        help="Generic non-graph per-anchor max suffix copy length for spans at or below --anchor-smallspan-max-tokens. Use 0 to disable.")
    parser.add_argument("--anchor-midspan-min-tokens", type=int, default=0,
                        help="Generic non-graph lower tokenized span length bound for applying --anchor-midspan-suffix-copy-cap. Use 0 to disable.")
    parser.add_argument("--anchor-midspan-max-tokens", type=int, default=0,
                        help="Generic non-graph upper tokenized span length bound for applying --anchor-midspan-suffix-copy-cap. Use 0 to disable.")
    parser.add_argument("--anchor-midspan-suffix-copy-cap", type=int, default=0,
                        help="Generic non-graph per-anchor max suffix copy length for spans in the configured midspan range. Use 0 to disable.")
    parser.add_argument("--enable-graph-aware-lossy", action="store_true",
                        help="Add graph_aware_lossy to the same selective whole-file driver. Target prompts remain whole-file; graph bundles are mapped back to exact AST spans already present in the prompt and used only for internal reuse selection.")
    parser.add_argument("--load-graph-bundles-for-selection", action="store_true",
                        help="Load graph bundles for hybrid/internal anchor selection without adding graph_aware_lossy mode or changing target prompts.")
    parser.add_argument("--code-graph-bundle-manifest", type=Path,
                        default=PROJECT / "results" / "code_graph_kv_reuse" / "data" / "code_graph_precision_manifest.jsonl",
                        help="JSONL manifest containing bundle_text records from the code graph precision census.")
    parser.add_argument("--graph-bundle-policy", default="call_neighborhood_1hop",
                        choices=["ast_function_only", "call_neighborhood_1hop", "reverse_callers_1hop", "import_dependency_bundle", "test_target_bundle"],
                        help="Bundle type used for graph_aware_lossy.")
    parser.add_argument("--graph-bundle-role", default="planner", choices=["planner", "coder", "reviewer"],
                        help="Role row to read from the graph bundle manifest. Text is identical across roles; this keeps provenance explicit.")
    parser.add_argument("--graph-bundles-per-case", type=int, default=3,
                        help="Maximum number of graph bundle records loaded per case for graph-aware internal reuse selection.")
    parser.add_argument("--max-graph-bundle-chars", type=int, default=22000,
                        help="Optional per-bundle char cap before mapping graph bundles back to prompt-resident spans. 0 disables truncation.")
    parser.add_argument("--graph-anchor-token-budget", type=int, default=1600,
                        help="Graph-aware internal anchor selection budget in approximate whitespace tokens. 0 disables budget.")
    parser.add_argument("--graph-anchor-max-span-tokens", type=int, default=900,
                        help="Drop graph-aware candidate spans above this approximate token count before applying the budget. 0 disables per-span filtering.")
    parser.add_argument("--graph-anchor-lowspan-max-tokens", type=int, default=0,
                        help="Tokenized span length upper bound for applying --graph-anchor-lowspan-suffix-copy-cap. Use 0 to disable this cap range.")
    parser.add_argument("--graph-anchor-lowspan-suffix-copy-cap", type=int, default=0,
                        help="Per-anchor max suffix copy length for graph-aware spans at or below --graph-anchor-lowspan-max-tokens. Use 0 to disable.")
    parser.add_argument("--graph-anchor-smallspan-max-tokens", type=int, default=0,
                        help="Tokenized span length upper bound for applying --graph-anchor-smallspan-suffix-copy-cap. Use 0 to disable this cap range.")
    parser.add_argument("--graph-anchor-smallspan-suffix-copy-cap", type=int, default=0,
                        help="Per-anchor max suffix copy length for graph-aware spans at or below --graph-anchor-smallspan-max-tokens. Use 0 to disable.")
    parser.add_argument("--graph-anchor-midspan-min-tokens", type=int, default=0,
                        help="Lower tokenized span length bound for applying --graph-anchor-midspan-suffix-copy-cap. Use 0 to disable the lower bound.")
    parser.add_argument("--graph-anchor-midspan-max-tokens", type=int, default=0,
                        help="Upper tokenized span length bound for applying --graph-anchor-midspan-suffix-copy-cap. Use 0 to disable the upper bound.")
    parser.add_argument("--graph-anchor-midspan-suffix-copy-cap", type=int, default=0,
                        help="Per-anchor max suffix copy length for graph-aware spans whose tokenized length falls in the configured midspan range. Use 0 to disable.")
    args = parser.parse_args()
    if args.warmup_protocol is None:
        if args.flush_cache_per_mode is True:
            args.warmup_protocol = "oracle_per_mode"
        elif args.flush_cache_per_mode is False:
            args.warmup_protocol = "natural_planner"
        else:
            args.warmup_protocol = "natural_planner"
    if args.prefer_graph_target_files is None:
        args.prefer_graph_target_files = bool(args.enable_graph_aware_lossy)
    return args


if __name__ == "__main__":
    asyncio.run(run_benchmark(parse_args()))
