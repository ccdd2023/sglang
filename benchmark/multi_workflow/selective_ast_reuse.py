#!/usr/bin/env python3
"""Selective AST/graph-aware reuse helpers.

The prompt can still carry whole files, but the reuse object is an internal
AST span. AST metadata selects candidate spans; exact content signatures remain
the safety gate used by the serving runtime.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_ALLOWED_GRANULARITIES = {"function", "method"}
EXTENDED_ALLOWED_GRANULARITIES = {"function", "method", "control_block", "file_prefix"}
DEFAULT_POLICY_PATH = (
    Path(__file__).resolve().parents[2]
    / "results/selective_ast_reuse/data/selective_reuse_policy.json"
)
DEFAULT_AST_DISTANCE_PATH = (
    Path(__file__).resolve().parents[2]
    / "results/ast_granularity_kv_sensitivity/data/ast_granularity_distance_7b.json"
)


@dataclass(frozen=True)
class CodeFile:
    path: str
    text: str


@dataclass(frozen=True)
class ReuseSpan:
    name: str
    path: str
    text: str
    granularity: str
    start_line: int
    end_line: int
    signature: str
    reuse_decision: str
    decision_reason: str
    risk_p90: float | None = None
    tail_rate: float | None = None


def normalize_code_text(text: str) -> str:
    lines = [line.rstrip() for line in str(text).replace("\r\n", "\n").split("\n")]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def content_signature(text: str) -> str:
    return hashlib.sha1(normalize_code_text(text).encode("utf-8")).hexdigest()[:16]


def _line_slice(lines: list[str], start_line: int, end_line: int) -> str:
    start = max(1, start_line)
    end = max(start, end_line)
    return "\n".join(lines[start - 1 : end]).rstrip()


def _node_name(node: ast.AST) -> str:
    return str(getattr(node, "name", node.__class__.__name__))


def _class_ranges(tree: ast.AST) -> list[tuple[int, int]]:
    ranges = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and getattr(node, "end_lineno", None):
            ranges.append((int(node.lineno), int(node.end_lineno)))
    return ranges


def _inside_class(node: ast.AST, class_ranges: list[tuple[int, int]]) -> bool:
    lineno = int(getattr(node, "lineno", 0) or 0)
    return any(start < lineno <= end for start, end in class_ranges)


def load_selective_policy(path: str | Path | None = None) -> dict[str, Any]:
    policy_path = Path(path) if path else DEFAULT_POLICY_PATH
    if not policy_path.exists():
        return build_selective_policy(read_ast_granularity_summary(DEFAULT_AST_DISTANCE_PATH))
    return json.loads(policy_path.read_text(encoding="utf-8"))


def read_ast_granularity_summary(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    records = [
        row
        for row in data.get("records", [])
        if row.get("agent_role") != "planner" and row.get("granularity") and "d_norm" in row
    ]
    if records:
        by_granularity: dict[str, list[dict[str, Any]]] = {}
        for row in records:
            by_granularity.setdefault(str(row["granularity"]), []).append(row)
        summary = {}
        for granularity, rows in by_granularity.items():
            vals = sorted(float(row.get("d_norm", 0.0) or 0.0) for row in rows)
            if not vals:
                continue
            p50 = vals[int(round((len(vals) - 1) * 0.50))]
            p90 = vals[int(round((len(vals) - 1) * 0.90))]
            summary[granularity] = {
                "count": len(vals),
                "mean": sum(vals) / len(vals),
                "p50": p50,
                "p90": p90,
                "max": vals[-1],
                "tail_rate_gt_0_5": sum(1 for val in vals if val > 0.5) / len(vals),
                "device_retention_cost_tokens": sum(
                    int(row.get("span_tokens", row.get("approx_tokens", 0)) or 0)
                    for row in rows
                    if row.get("agent_role") == "coder"
                ),
            }
        return summary
    return (data.get("summary") or {}).get("by_granularity") or data.get("summary") or {}


def build_selective_policy(
    by_granularity: dict[str, Any],
    *,
    p90_threshold: float = 0.45,
    max_tail_rate: float = 0.10,
    extended: bool = False,
) -> dict[str, Any]:
    allowed_set = EXTENDED_ALLOWED_GRANULARITIES if extended else DEFAULT_ALLOWED_GRANULARITIES
    policy: dict[str, Any] = {
        "schema_version": "selective_ast_reuse_policy_v1" + ("_extended" if extended else ""),
        "p90_threshold": p90_threshold,
        "max_tail_rate": max_tail_rate,
        "default_allowed_granularities": sorted(allowed_set),
        "granularities": {},
    }
    for granularity, stats in sorted(by_granularity.items()):
        p90 = float(stats.get("p90", 1.0) or 1.0)
        count = int(stats.get("count", 0) or 0)
        max_v = float(stats.get("max", 0.0) or 0.0)
        tail_rate = float(stats.get("tail_rate_gt_0_5", 0.0) or stats.get("tail_gt_0_5", 0.0) or 0.0)
        default_allow = granularity in allowed_set and p90 < p90_threshold
        oracle_allow = p90 < p90_threshold and tail_rate <= max_tail_rate
        if default_allow:
            decision = "reuse"
            reason = "default_function_method_low_p90" if not extended else "extended_safe_granularities"
        elif oracle_allow:
            decision = "oracle_reuse_only"
            reason = "low_p90_low_tail_oracle"
        else:
            decision = "recompute"
            reason = "granularity_risk"
        policy["granularities"][granularity] = {
            "decision": decision,
            "reason": reason,
            "oracle_allow": oracle_allow,
            "count": count,
            "mean": float(stats.get("mean", 0.0) or 0.0),
            "p50": float(stats.get("p50", 0.0) or 0.0),
            "p90": p90,
            "max": max_v,
            "tail_rate_gt_0_5": tail_rate,
            "retention_tokens": int(stats.get("device_retention_cost_tokens", 0) or 0),
        }
    return policy


def extract_codebase_files(prompt_text: str) -> list[CodeFile]:
    """Extract whole-file codebase objects from JSON or fenced markdown.

    Accepted JSON shape: {"code_base": [{"path": "...", "content": "..."}]}.
    Fallback markdown shape: a marker line containing code_base/path followed by
    a fenced Python block.
    """
    text = str(prompt_text or "")
    files: list[CodeFile] = []
    try:
        parsed = json.loads(text)
        for item in parsed.get("code_base", []) if isinstance(parsed, dict) else []:
            if isinstance(item, dict) and item.get("content"):
                files.append(CodeFile(str(item.get("path") or f"code_base{len(files)+1}.py"), str(item["content"])))
    except Exception:
        pass
    if files:
        return files

    marker_re = re.compile(
        r"(?P<marker>(?:^|\n)#{0,3}\s*(?:code_base|file|path)[:\s]+(?P<path>[^\n`]+))\n```(?:python|py)?\n(?P<code>[\s\S]*?)\n```",
        re.IGNORECASE,
    )
    for match in marker_re.finditer(text):
        path = match.group("path").strip().strip("`")
        files.append(CodeFile(path or f"code_base{len(files)+1}.py", match.group("code").strip()))
    if files:
        return files

    for idx, match in enumerate(re.finditer(r"```(?:python|py)?\n([\s\S]*?)\n```", text), start=1):
        files.append(CodeFile(f"fenced_block_{idx}.py", match.group(1).strip()))
    return files


def split_python_file(path: str, text: str, policy: dict[str, Any] | None = None) -> list[ReuseSpan]:
    policy = policy or load_selective_policy()
    lines = str(text).splitlines()
    spans: list[tuple[str, str, int, int]] = []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        tree = None
    if tree is not None:
        class_ranges = _class_ranges(tree)
        for node in ast.walk(tree):
            end = getattr(node, "end_lineno", None)
            if not end or not getattr(node, "lineno", None):
                continue
            if isinstance(node, ast.ClassDef):
                spans.append(("class", _node_name(node), int(node.lineno), int(end)))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                gran = "method" if _inside_class(node, class_ranges) else "function"
                spans.append((gran, _node_name(node), int(node.lineno), int(end)))
            elif isinstance(node, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With, ast.AsyncWith)):
                spans.append(("control_block", node.__class__.__name__.lower(), int(node.lineno), int(end)))

    if lines:
        prefix_end = min(len(lines), 200)
        spans.append(("file_prefix", Path(path).name, 1, prefix_end))
    if not spans and lines:
        for start in range(1, len(lines) + 1, 20):
            spans.append(("statement_window", f"lines_{start}_{min(start + 19, len(lines))}", start, min(start + 19, len(lines))))

    seen: set[tuple[str, int, int]] = set()
    out: list[ReuseSpan] = []
    for granularity, symbol, start_line, end_line in sorted(spans, key=lambda item: (item[2], item[3], item[0])):
        key = (granularity, start_line, end_line)
        if key in seen:
            continue
        seen.add(key)
        span_text = _line_slice(lines, start_line, end_line)
        if not span_text.strip():
            continue
        info = (policy.get("granularities") or {}).get(granularity, {})
        decision = str(info.get("decision") or ("reuse" if granularity in DEFAULT_ALLOWED_GRANULARITIES else "recompute"))
        reason = str(info.get("reason") or "default")
        out.append(
            ReuseSpan(
                name=f"{path}:{granularity}:{symbol}:{start_line}-{end_line}",
                path=path,
                text=span_text,
                granularity=granularity,
                start_line=start_line,
                end_line=end_line,
                signature=content_signature(span_text),
                reuse_decision=decision,
                decision_reason=reason,
                risk_p90=float(info["p90"]) if "p90" in info else None,
                tail_rate=float(info["tail_rate_gt_0_5"]) if "tail_rate_gt_0_5" in info else None,
            )
        )
    return out


def _non_overlapping_spans(spans: list[ReuseSpan]) -> list[ReuseSpan]:
    selected: list[ReuseSpan] = []
    occupied: dict[str, list[tuple[int, int]]] = {}
    for span in spans:
        ranges = occupied.setdefault(span.path, [])
        if any(start <= span.start_line and span.end_line <= end for start, end in ranges):
            continue
        selected.append(span)
        ranges.append((span.start_line, span.end_line))
    return selected


def select_spans(spans: list[ReuseSpan], mode: str) -> list[ReuseSpan]:
    if mode == "whole_file_reuse_all":
        return [span for span in spans if span.granularity == "file_prefix"] or spans[:1]
    if mode == "selective_oracle_low_dnorm":
        return _non_overlapping_spans([span for span in spans if span.reuse_decision in {"reuse", "oracle_reuse_only"}])
    if mode == "selective_function_method_reuse":
        return _non_overlapping_spans(
            [span for span in spans if span.reuse_decision == "reuse" and span.granularity in DEFAULT_ALLOWED_GRANULARITIES]
        )
    if mode == "selective_extended_reuse":
        return _non_overlapping_spans(
            [span for span in spans if span.reuse_decision == "reuse" and span.granularity in EXTENDED_ALLOWED_GRANULARITIES]
        )
    return []


def summarize_selection(spans: list[ReuseSpan], selected: list[ReuseSpan]) -> dict[str, Any]:
    selected_ids = {id(span) for span in selected}
    total_tokens = sum(max(1, len(span.text.split())) for span in spans)
    reused_tokens = sum(max(1, len(span.text.split())) for span in selected)
    counts: dict[str, int] = {}
    selected_counts: dict[str, int] = {}
    reasons: dict[str, int] = {}
    for span in spans:
        counts[span.granularity] = counts.get(span.granularity, 0) + 1
        reason = span.decision_reason if id(span) not in selected_ids else f"reuse:{span.decision_reason}"
        reasons[reason] = reasons.get(reason, 0) + 1
        if id(span) in selected_ids:
            selected_counts[span.granularity] = selected_counts.get(span.granularity, 0) + 1
    return {
        "span_count": len(spans),
        "selected_span_count": len(selected),
        "span_count_by_granularity": counts,
        "selected_span_count_by_granularity": selected_counts,
        "estimated_reused_tokens": reused_tokens,
        "estimated_recomputed_tokens": max(0, total_tokens - reused_tokens),
        "decision_reason_counts": reasons,
    }
