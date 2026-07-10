#!/usr/bin/env python3
"""R21 verdict scoring.

Three metrics:
1. Verdict accuracy vs ground truth (FAIL=1.0 rate; "model FAILs when patch is FAIL")
2. Pass-rate deviation from lossless baseline:
   lossless PASS-rate vs lossy PASS-rate (consistency-of-judgement)
3. Failure-type agreement vs ground-truth patch category
"""
from __future__ import annotations
import argparse, json, re
from collections import Counter
from pathlib import Path

ROOT = Path("/home/gfy/CodeMAS_Project/sglang-kvflow")
GT_PATH = ROOT / "results/lossy_alg_round21/ground_truth.json"

FAIL_PATTERNS = [
    ("missing_error_handling", ["error handling", "proper error", "raises", "exception", "raise TypeError", "raise ValueError", "check.*before", "input validation"]),
    ("type_check_missing", ["type check", "isinstance", "type of", "non-numeric", "empty", "non-contiguous", "data type"]),
    ("missing_import", ["missing import", "undefined name", "is not defined", "import error", "syntax error", "is used instead"]),
    ("guard_missing", ["sentinel", "out-of-bounds", "guard", "boundary check", "out of range", "null check"]),
    ("logic_error", ["logical error", "incorrect", "wrong.*function", "should.*return", "does not correctly"]),
    ("safety_risk", ["buffer leak", "memory management", "race condition", "consistency", "code duplication", "redundant", "inconsistent"]),
]
PASS_PATTERN = re.compile(r"\bVERDICT:\s*PASS\b", re.IGNORECASE)
FAIL_PATTERN = re.compile(r"\bVERDICT:\s*FAIL\b", re.IGNORECASE)


def parse_verdict(text: str) -> str:
    if PASS_PATTERN.search(text or ""):
        return "PASS"
    if FAIL_PATTERN.search(text or ""):
        return "FAIL"
    return "UNKNOWN"


def classify_fail_reason(text: str) -> str:
    """Return the first matching failure pattern (returns 'other' if none)."""
    if not text:
        return "other"
    low = text.lower()
    for name, pats in FAIL_PATTERNS:
        for pat in pats:
            if re.search(pat, low):
                return name
    return "other"


def classify_patch(patch: str) -> str:
    """Map a patch into a failure-type category using the same vocabulary as classify_fail_reason."""
    p = patch or ""
    low = p.lower()
    if "raise" in low or "error handling" in low:
        return "missing_error_handling"
    if "isinstance" in low and "+" in p:
        return "type_check_missing"
    if "missing import" in low or "name 'ge' is not defined" in low:
        return "missing_import"
    if "sentinel" in low or "out-of-bounds" in low or "out_of_range" in low:
        return "guard_missing"
    return "other"


def score_run(out_path: Path, gt: dict, label: str) -> dict:
    rows = [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()]
    n_total = len(rows)
    verdicts = [(r['case_id'], parse_verdict(r['output_text']), r['output_text']) for r in rows]

    # 1. Per-row accuracy vs ground truth (only meaningful for PASS because gt is ALL FAIL here)
    gt_fail = sum(1 for k, v, _ in verdicts if gt.get(k) == "FAIL")
    gt_pass = sum(1 for k, v, _ in verdicts if gt.get(k) == "PASS")
    model_pass = sum(1 for _, v, _ in verdicts if v == "PASS")
    model_fail = sum(1 for _, v, _ in verdicts if v == "FAIL")
    model_unk = sum(1 for _, v, _ in verdicts if v == "UNKNOWN")

    # Accuracy vs ALL-FAIL ground truth = % model FAILs when expected FAIL
    fail_correct = sum(1 for k, v, _ in verdicts if gt.get(k) == "FAIL" and v == "FAIL")
    pass_correct = sum(1 for k, v, _ in verdicts if gt.get(k) == "PASS" and v == "PASS")

    # 2. Failure type agreement
    type_agree = 0
    type_total = 0
    for r in rows:
        v = parse_verdict(r["output_text"])
        if v == "FAIL":
            model_type = classify_fail_reason(r["output_text"])
            gt_type = classify_patch(_patch_lookup(r["case_id"]))
            type_total += 1
            if model_type == gt_type:
                type_agree += 1

    return {
        "label": label,
        "n_rows": n_total,
        "model_pass": model_pass,
        "model_fail": model_fail,
        "model_unknown": model_unk,
        "pass_rate": model_pass / n_total if n_total else 0,
        "fail_rate": model_fail / n_total if n_total else 0,
        "fail_correct_when_gt_fail": fail_correct,
        "fail_correct_pct": fail_correct / n_total if n_total else 0,
        "failure_type_agree": type_agree,
        "failure_type_total": type_total,
        "failure_type_agree_pct": (type_agree / type_total) if type_total else 0,
    }


_PATCH_CACHE = None


def _patch_lookup(case_id: str) -> str:
    global _PATCH_CACHE
    if _PATCH_CACHE is None:
        m = Path("/home/gfy/CodeMAS_Project/sglang-kvflow/results/giant_codebase/tasks/pandas__pandas__1000/manifest.jsonl")
        _PATCH_CACHE = {}
        for line in m.read_text().splitlines():
            if not line.strip(): continue
            d = json.loads(line)
            _PATCH_CACHE[d["instance_id"]] = d.get("patch", "") or ""
    return _PATCH_CACHE.get(case_id, "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", help="Output JSONL files from each run")
    parser.add_argument("--labels", nargs="+", help="Labels matching runs")
    args = parser.parse_args()

    gt = json.loads(GT_PATH.read_text())

    runs = list(zip(args.labels or [Path(p).parent.name for p in args.runs], args.runs))
    print(f"{'config':<25} {'n':>3} {'pass%':>6} {'fail%':>6} {'FAIL_acc%':>10} {'type_agree%':>12}")
    print("-" * 80)
    for label, path in runs:
        r = score_run(Path(path), gt, label)
        print(f"{label:<25} {r['n_rows']:>3} {r['pass_rate']*100:>5.1f}% {r['fail_rate']*100:>5.1f}% "
              f"{r['fail_correct_pct']*100:>9.1f}% {r['failure_type_agree_pct']*100:>11.1f}%")


if __name__ == "__main__":
    main()
