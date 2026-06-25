#!/usr/bin/env python3
"""Compare prompt-fair reuse outputs against gold SWE-bench patch intent.

This is a lightweight task-level sanity check.  It does not try to apply a
generated patch; instead, it asks whether the generated explanation/code still
mentions the gold changed files and code identifiers from the reference patch.
That makes it useful when token F1 drops because prose changes, but the model is
still pointing at the same repair target.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from analyze_code_action_overlap import extract_signals, jaccard, mean


DIFF_FILE_RE = re.compile(r"^diff --git a/(.*?) b/(.*?)$", re.MULTILINE)
HUNK_RE = re.compile(r"^@@.*?@@\s*(.*)$", re.MULTILINE)
IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*\b")

PATCH_STOPWORDS = {
    "assert",
    "class",
    "def",
    "false",
    "for",
    "from",
    "if",
    "import",
    "none",
    "not",
    "raise",
    "return",
    "self",
    "true",
    "with",
}


def _norm(value: str) -> str:
    return value.strip().strip(".,:;()[]{}'\"").lower()


def _identifiers_from_text(text: str) -> set[str]:
    terms: set[str] = set()
    for match in IDENT_RE.finditer(text):
        value = _norm(match.group(0))
        if len(value) < 3 or value in PATCH_STOPWORDS:
            continue
        terms.add(value)
    return terms


def extract_patch_intent(patch: str, include_tests: bool = False, test_patch: str = "") -> dict[str, set[str]]:
    text = patch + ("\n" + test_patch if include_tests else "")
    files = {_norm(dst) for _, dst in DIFF_FILE_RE.findall(text)}
    identifiers: set[str] = set()
    hunk_symbols: set[str] = set()
    for line in text.splitlines():
        if line.startswith("@@"):
            symbol_text = HUNK_RE.match(line)
            if symbol_text:
                hunk_symbols |= _identifiers_from_text(symbol_text.group(1))
            continue
        if not line.startswith(("+", "-")) or line.startswith(("+++", "---")):
            continue
        identifiers |= _identifiers_from_text(line[1:])
    return {
        "gold_files": files,
        "gold_identifiers": identifiers,
        "gold_hunk_symbols": hunk_symbols,
    }


def containment(reference: set[str], candidate: set[str]) -> float:
    if not reference:
        return 1.0
    return len(reference & candidate) / len(reference)


def output_gold_scores(text: str, gold: dict[str, set[str]]) -> dict[str, Any]:
    sig = extract_signals(text)
    output_files = sig["files"]
    output_identifiers = sig["identifiers"] | sig["backticks"] | sig["code_tokens"]
    gold_ids = gold["gold_identifiers"] | gold["gold_hunk_symbols"]
    file_containment = containment(gold["gold_files"], output_files)
    identifier_containment = containment(gold_ids, output_identifiers)
    hunk_symbol_containment = containment(gold["gold_hunk_symbols"], output_identifiers)
    score = 0.55 * file_containment + 0.35 * identifier_containment + 0.10 * hunk_symbol_containment
    return {
        "gold_file_containment": round(file_containment, 4),
        "gold_identifier_containment": round(identifier_containment, 4),
        "gold_hunk_symbol_containment": round(hunk_symbol_containment, 4),
        "gold_intent_score": round(score, 4),
        "output_files": sorted(output_files),
        "output_identifiers": sorted(output_identifiers)[:60],
    }


def analyze_case(case: dict[str, Any], instance: dict[str, Any], mode: str, include_tests: bool) -> dict[str, Any] | None:
    rows = {row.get("mode"): row for row in case.get("rows", [])}
    ref = rows.get("lossless_full_prefill")
    cand = rows.get(mode)
    if not ref or not cand:
        return None
    gold = extract_patch_intent(
        instance.get("patch", ""),
        include_tests=include_tests,
        test_patch=instance.get("test_patch", ""),
    )
    ref_scores = output_gold_scores(ref.get("output_text") or "", gold)
    cand_scores = output_gold_scores(cand.get("output_text") or "", gold)
    ref_sig = extract_signals(ref.get("output_text") or "")
    cand_sig = extract_signals(cand.get("output_text") or "")
    return {
        "instance_id": case.get("instance_id"),
        "repo": case.get("repo"),
        "mode": mode,
        "token_f1": cand.get("output_token_f1_vs_lossless"),
        "accuracy_bucket": cand.get("accuracy_bucket"),
        "ttft_ms": cand.get("ttft_ms"),
        "suffix_copy_len": cand.get("lossy_anchor_suffix_copy_len"),
        "gold_files": sorted(gold["gold_files"]),
        "gold_identifiers": sorted(gold["gold_identifiers"] | gold["gold_hunk_symbols"])[:80],
        "lossless_gold_intent_score": ref_scores["gold_intent_score"],
        "candidate_gold_intent_score": cand_scores["gold_intent_score"],
        "gold_intent_delta": round(cand_scores["gold_intent_score"] - ref_scores["gold_intent_score"], 4),
        "candidate_gold_file_containment": cand_scores["gold_file_containment"],
        "candidate_gold_identifier_containment": cand_scores["gold_identifier_containment"],
        "lossless_gold_file_containment": ref_scores["gold_file_containment"],
        "lossless_gold_identifier_containment": ref_scores["gold_identifier_containment"],
        "lossless_candidate_file_jaccard": round(jaccard(ref_sig["files"], cand_sig["files"]), 4),
        "lossless_candidate_identifier_jaccard": round(
            jaccard(ref_sig["identifiers"] | ref_sig["backticks"], cand_sig["identifiers"] | cand_sig["backticks"]),
            4,
        ),
        "candidate_files": cand_scores["output_files"],
        "candidate_identifiers": cand_scores["output_identifiers"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--mode", default="hybrid_code_aware_lossy")
    parser.add_argument("--gold-intent-threshold", type=float, default=0.70)
    parser.add_argument("--max-gold-intent-regression", type=float, default=0.10)
    parser.add_argument("--token-f1-threshold", type=float, default=0.90)
    parser.add_argument("--include-test-patch", action="store_true")
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()

    summary_data = json.loads(args.summary.read_text(encoding="utf-8"))
    instances = {
        item["instance_id"]: item
        for item in json.loads(args.instances.read_text(encoding="utf-8"))
    }
    out_dir = args.out_dir or args.summary.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for case in summary_data.get("cases", []):
        instance = instances.get(case.get("instance_id"))
        if not instance:
            continue
        row = analyze_case(case, instance, args.mode, args.include_test_patch)
        if row is not None:
            rows.append(row)

    scores = [float(row["candidate_gold_intent_score"]) for row in rows]
    deltas = [float(row["gold_intent_delta"]) for row in rows]
    f1s = [float(row["token_f1"]) for row in rows if row.get("token_f1") is not None]
    gold_ok = [
        row
        for row in rows
        if float(row["candidate_gold_intent_score"]) >= args.gold_intent_threshold
    ]
    no_gold_regression = [
        row
        for row in rows
        if float(row["gold_intent_delta"]) >= -args.max_gold_intent_regression
    ]
    composite_ok = [
        row
        for row in rows
        if (
            row.get("token_f1") is not None
            and float(row["token_f1"]) >= args.token_f1_threshold
        )
        or float(row["candidate_gold_intent_score"]) >= args.gold_intent_threshold
    ]
    report = {
        "source_summary": str(args.summary),
        "source_instances": str(args.instances),
        "mode": args.mode,
        "n": len(rows),
        "include_test_patch": args.include_test_patch,
        "token_f1_threshold": args.token_f1_threshold,
        "gold_intent_threshold": args.gold_intent_threshold,
        "max_gold_intent_regression": args.max_gold_intent_regression,
        "avg_token_f1": round(mean(f1s), 4),
        "avg_candidate_gold_intent_score": round(mean(scores), 4),
        "avg_gold_intent_delta_vs_lossless": round(mean(deltas), 4),
        "gold_intent_ge_threshold_count": len(gold_ok),
        "gold_intent_ge_threshold_rate": round(len(gold_ok) / len(rows), 4) if rows else 0.0,
        "no_gold_intent_regression_count": len(no_gold_regression),
        "no_gold_intent_regression_rate": round(len(no_gold_regression) / len(rows), 4) if rows else 0.0,
        "gold_intent_regressions": [
            row["instance_id"]
            for row in rows
            if row not in no_gold_regression
        ],
        "composite_acceptable_count": len(composite_ok),
        "composite_acceptable_rate": round(len(composite_ok) / len(rows), 4) if rows else 0.0,
        "composite_rejects": [
            row["instance_id"]
            for row in rows
            if row not in composite_ok
        ],
        "low_token_high_gold_intent": [
            row["instance_id"]
            for row in rows
            if row.get("token_f1") is not None
            and float(row["token_f1"]) < args.token_f1_threshold
            and float(row["candidate_gold_intent_score"]) >= args.gold_intent_threshold
        ],
    }

    (out_dir / "gold_patch_intent_summary.json").write_text(
        json.dumps({"summary": report, "rows": rows}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (out_dir / "gold_patch_intent_rows.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "instance_id",
            "repo",
            "mode",
            "token_f1",
            "accuracy_bucket",
            "ttft_ms",
            "suffix_copy_len",
            "lossless_gold_intent_score",
            "candidate_gold_intent_score",
            "gold_intent_delta",
            "candidate_gold_file_containment",
            "candidate_gold_identifier_containment",
            "lossless_candidate_file_jaccard",
            "lossless_candidate_identifier_jaccard",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
