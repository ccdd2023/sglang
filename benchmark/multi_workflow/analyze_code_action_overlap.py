#!/usr/bin/env python3
"""Post-hoc code-action overlap analysis for prompt-fair KV reuse runs.

Token F1 is useful but can over-penalize semantically equivalent prose changes.
This script compares lossy outputs against the lossless reference on code-task
signals: mentioned files, backtick/code identifiers, code fence tokens, and
simple edit-action words.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


FILE_RE = re.compile(r"(?<![\w/.-])(?:[\w.-]+/)*[\w.-]+\.(?:py|pyi|toml|cfg|ini|txt|rst|md|yaml|yml|json)(?![\w/.-])")
BACKTICK_RE = re.compile(r"`([^`\n]{1,160})`")
CODE_FENCE_RE = re.compile(r"```(?:\w+)?\n(.*?)```", re.DOTALL)
IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*\b")

ACTION_TERMS = {
    "add",
    "allow",
    "change",
    "check",
    "comment",
    "convert",
    "ensure",
    "fix",
    "handle",
    "ignore",
    "locate",
    "modify",
    "prevent",
    "remove",
    "replace",
    "return",
    "set",
    "skip",
    "update",
    "validate",
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "be",
    "by",
    "for",
    "from",
    "here",
    "in",
    "is",
    "it",
    "line",
    "needed",
    "of",
    "or",
    "should",
    "string",
    "the",
    "this",
    "to",
    "with",
}


def _norm(value: str) -> str:
    return value.strip().strip(".,:;()[]{}").lower()


def _identifier_terms(text: str) -> set[str]:
    terms = set()
    for match in IDENT_RE.finditer(text):
        value = _norm(match.group(0))
        if len(value) < 3 or value in STOPWORDS:
            continue
        if value in ACTION_TERMS:
            continue
        terms.add(value)
    return terms


def extract_signals(text: str) -> dict[str, set[str]]:
    files = {_norm(match.group(0)) for match in FILE_RE.finditer(text)}
    backticks = {_norm(match.group(1)) for match in BACKTICK_RE.finditer(text)}
    code_tokens: set[str] = set()
    for fence in CODE_FENCE_RE.finditer(text):
        code_tokens |= _identifier_terms(fence.group(1))
    identifiers = _identifier_terms(" ".join(backticks)) | code_tokens
    actions = {
        term
        for term in ACTION_TERMS
        if re.search(rf"\b{re.escape(term)}(?:ed|ing|s)?\b", text, flags=re.IGNORECASE)
    }
    return {
        "files": files,
        "backticks": backticks,
        "code_tokens": code_tokens,
        "identifiers": identifiers,
        "actions": actions,
    }


def jaccard(reference: set[str], candidate: set[str]) -> float:
    if not reference and not candidate:
        return 1.0
    if not reference or not candidate:
        return 0.0
    return len(reference & candidate) / len(reference | candidate)


def containment(reference: set[str], candidate: set[str]) -> float:
    if not reference:
        return 1.0
    return len(reference & candidate) / len(reference)


def analyze_case(case: dict[str, Any], mode: str) -> dict[str, Any] | None:
    rows = {row.get("mode"): row for row in case.get("rows", [])}
    ref = rows.get("lossless_full_prefill")
    cand = rows.get(mode)
    if not ref or not cand:
        return None
    ref_text = ref.get("output_text") or ""
    cand_text = cand.get("output_text") or ""
    ref_sig = extract_signals(ref_text)
    cand_sig = extract_signals(cand_text)

    file_containment = containment(ref_sig["files"], cand_sig["files"])
    identifier_containment = containment(ref_sig["identifiers"], cand_sig["identifiers"])
    action_containment = containment(ref_sig["actions"], cand_sig["actions"])
    backtick_containment = containment(ref_sig["backticks"], cand_sig["backticks"])
    code_action_score = (
        0.35 * file_containment
        + 0.30 * identifier_containment
        + 0.20 * action_containment
        + 0.15 * backtick_containment
    )
    return {
        "instance_id": case.get("instance_id"),
        "mode": mode,
        "token_f1": cand.get("output_token_f1_vs_lossless"),
        "accuracy_bucket": cand.get("accuracy_bucket"),
        "ttft_ms": cand.get("ttft_ms"),
        "suffix_copy_len": cand.get("lossy_anchor_suffix_copy_len"),
        "file_containment": round(file_containment, 4),
        "identifier_containment": round(identifier_containment, 4),
        "action_containment": round(action_containment, 4),
        "backtick_containment": round(backtick_containment, 4),
        "code_action_score": round(code_action_score, 4),
        "ref_files": sorted(ref_sig["files"]),
        "cand_files": sorted(cand_sig["files"]),
        "ref_identifiers": sorted(ref_sig["identifiers"])[:40],
        "cand_identifiers": sorted(cand_sig["identifiers"])[:40],
        "ref_actions": sorted(ref_sig["actions"]),
        "cand_actions": sorted(cand_sig["actions"]),
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--mode", default="hybrid_code_aware_lossy")
    parser.add_argument("--token-f1-threshold", type=float, default=0.90)
    parser.add_argument("--code-action-threshold", type=float, default=0.90)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()

    data = json.loads(args.summary.read_text(encoding="utf-8"))
    out_dir = args.out_dir or args.summary.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        row
        for case in data.get("cases", [])
        for row in [analyze_case(case, args.mode)]
        if row is not None
    ]
    scores = [float(row["code_action_score"]) for row in rows]
    token_f1s = [
        float(row["token_f1"])
        for row in rows
        if row.get("token_f1") is not None
    ]
    composite_acceptable = [
        row
        for row in rows
        if (
            row.get("token_f1") is not None
            and float(row["token_f1"]) >= args.token_f1_threshold
        )
        or float(row["code_action_score"]) >= args.code_action_threshold
    ]
    summary = {
        "source_summary": str(args.summary),
        "mode": args.mode,
        "n": len(rows),
        "token_f1_threshold": args.token_f1_threshold,
        "code_action_threshold": args.code_action_threshold,
        "avg_token_f1": round(mean(token_f1s), 4),
        "avg_code_action_score": round(mean(scores), 4),
        "code_action_ge_090": sum(1 for score in scores if score >= 0.90),
        "code_action_ge_080": sum(1 for score in scores if score >= 0.80),
        "composite_acceptable_count": len(composite_acceptable),
        "composite_acceptable_rate": round(len(composite_acceptable) / len(rows), 4) if rows else 0.0,
        "composite_rejects": [
            row["instance_id"]
            for row in rows
            if row not in composite_acceptable
        ],
        "low_token_high_code_action": [
            row["instance_id"]
            for row in rows
            if row.get("token_f1") is not None
            and float(row["token_f1"]) < 0.90
            and float(row["code_action_score"]) >= 0.90
        ],
    }

    (out_dir / "code_action_overlap_summary.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (out_dir / "code_action_overlap_rows.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "instance_id",
            "mode",
            "token_f1",
            "accuracy_bucket",
            "ttft_ms",
            "suffix_copy_len",
            "file_containment",
            "identifier_containment",
            "action_containment",
            "backtick_containment",
            "code_action_score",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
