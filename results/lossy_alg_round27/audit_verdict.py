#!/usr/bin/env python3
"""Audit script: extended metrics + per-case stability + UNK sub-classification.

Adds metrics the original score_verdict.py missed:
- meaningful_verdict_rate = (PASS+FAIL) / total  → "can the model produce a verdict?"
- pass_tendency = PASS / (PASS+FAIL)            → "when it can, which way does it lean?"
- fail_recall_overall = FAIL / total            → old FAIL_acc
- fail_recall_among_meaningful = FAIL/(PASS+FAIL)
- per-case verdict agreement (5 cases × N agents)
- UNK sub-classification (template-match / C-leakage / too-short / bad-format)
- output length stats (median, p10, p90)
- template-leakage detection ("<one-sentence reason>" placeholder in output)

Reads 6 outputs.jsonl files (R19/R17 lossy + R19 lossless + R26 lossy/lossless
+ R27 lossy/lossless) and prints a side-by-side comparison table + per-case
detail. No GPU, no server — pure JSONL parse.

Usage:
    python results/lossy_alg_round27/audit_verdict.py
"""
from __future__ import annotations
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

ROOT = Path("/home/gfy/Project/sglang-kvflow")
if not ROOT.exists():
    ROOT = Path("/home/gfy/CodeMAS_Project/sglang-kvflow")

GT_PATH = ROOT / "results/lossy_alg_round21/ground_truth.json"

# Regex for verdict extraction (same as score_verdict.py)
PASS_RE = re.compile(r"\bVERDICT:\s*PASS\b", re.IGNORECASE)
FAIL_RE = re.compile(r"\bVERDICT:\s*FAIL\b", re.IGNORECASE)
TEMPLATE_LEAK_RE = re.compile(r"<one-sentence reason>", re.IGNORECASE)
C_CODE_LEAK_RE = re.compile(r"\bis_dictionary\b|\bColumnNullType\b|\bArrowSchema\b")

RUNS = [
    # (label, jsonl path, mode)
    ("R19 lossy (7B × 5)",    ROOT / "results/lossy_alg_round21/r19_verdict/outputs.jsonl",                  "lossy"),
    ("R19 lossless (7B × 5)", ROOT / "results/lossy_alg_round21/lossless_verdict/outputs.jsonl",           "lossless"),
    ("R17 lossy (7B × 5)",    ROOT / "results/lossy_alg_round21/r17_verdict/outputs.jsonl",                  "lossy"),
    ("R26 lossy (3B-Gen × 3)", ROOT / "results/lossy_alg_round26/r26_3b_3agent/outputs.jsonl",             "lossy"),
    ("R26 lossless (3B-Gen × 3)", ROOT / "results/lossy_alg_round26/r26_3b_3agent_lossless/outputs.jsonl", "lossless"),
    ("R27 lossy (3B-Cod × 3)", ROOT / "results/lossy_alg_round27/r27_coder3b_3agent/outputs.jsonl",        "lossy"),
    ("R27 lossless (3B-Cod × 3)", ROOT / "results/lossy_alg_round27/r27_coder3b_3agent_lossless/outputs.jsonl", "lossless"),
]


def parse_verdict(text: str) -> str:
    if PASS_RE.search(text or ""):
        return "PASS"
    if FAIL_RE.search(text or ""):
        return "FAIL"
    return "UNKNOWN"


def classify_unk(text: str, verdict: str) -> str:
    """Sub-classify UNKNOWN outputs so we know what 'UNK' really means."""
    if verdict != "UNKNOWN":
        return ""
    if not text or len(text.strip()) < 30:
        return "too_short"
    if TEMPLATE_LEAK_RE.search(text):
        return "template_placeholder"
    if C_CODE_LEAK_RE.search(text):
        return "c_code_leakage"
    if PASS_RE.search(text) and FAIL_RE.search(text):
        return "both_verdicts"
    return "other_format"


def load_run(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def metrics_for_run(rows: list[dict], gt: dict) -> dict:
    n = len(rows)
    verdicts = []
    for r in rows:
        v = parse_verdict(r.get("output_text", ""))
        u = classify_unk(r.get("output_text", ""), v)
        verdicts.append({
            "case_id": r.get("case_id"),
            "agent_idx": r.get("agent_idx"),
            "role": r.get("role"),
            "verdict": v,
            "unk_subtype": u,
            "text": r.get("output_text", ""),
            "len": len(r.get("output_text", "")),
        })

    n_pass = sum(1 for v in verdicts if v["verdict"] == "PASS")
    n_fail = sum(1 for v in verdicts if v["verdict"] == "FAIL")
    n_unk  = sum(1 for v in verdicts if v["verdict"] == "UNKNOWN")
    meaningful = n_pass + n_fail

    # Ground-truth FAIL_acc (= fraction of all outputs that say FAIL when GT is FAIL)
    # Note: ALL 5 cases in our subset have GT = FAIL
    fail_correct = n_fail  # since all gt = FAIL, every FAIL is correct

    # Per-case verdict stability
    by_case = defaultdict(list)
    for v in verdicts:
        by_case[v["case_id"]].append(v["verdict"])
    case_unanimous = sum(1 for vs in by_case.values() if len(set(vs)) == 1)
    case_split = n - case_unanimous  # cases where agents disagreed

    # UNK sub-classification
    unk_subtypes = Counter(v["unk_subtype"] for v in verdicts if v["verdict"] == "UNKNOWN")
    n_template = unk_subtypes.get("template_placeholder", 0)
    n_c_leak   = unk_subtypes.get("c_code_leakage", 0)
    n_short    = unk_subtypes.get("too_short", 0)

    # Template-leakage detection (outputs containing "<one-sentence reason>")
    n_template_leak = sum(1 for r in rows if TEMPLATE_LEAK_RE.search(r.get("output_text", "") or ""))

    return {
        "n": n,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "n_unk": n_unk,
        "meaningful_rate": meaningful / n if n else 0,
        "fail_recall_overall": n_fail / n if n else 0,            # old FAIL_acc
        "fail_recall_among_meaningful": n_fail / meaningful if meaningful else 0,
        "pass_tendency": n_pass / meaningful if meaningful else 0,
        "case_unanimous": case_unanimous,
        "case_split": case_split,
        "unk_subtypes": dict(unk_subtypes),
        "n_template_leak": n_template_leak,
        "n_c_leak": n_c_leak,
        "n_short": n_short,
        "median_len": median([v["len"] for v in verdicts]) if verdicts else 0,
        "min_len": min((v["len"] for v in verdicts), default=0),
        "max_len": max((v["len"] for v in verdicts), default=0),
        "verdicts": verdicts,
    }


def main():
    gt = json.loads(GT_PATH.read_text()) if GT_PATH.exists() else {}

    print(f"Ground truth total: {len(gt)} cases (all FAIL)")
    # Find which 5 cases our runs actually use
    sample = load_run(RUNS[0][1])
    used_cases = sorted({r.get("case_id") for r in sample}) if sample else []
    print(f"Sample uses {len(used_cases)} cases (all in subset)")
    for c in used_cases:
        print(f"  {c}: gt={gt.get(c, '?')}")
    print()

    print(f"{'config':<28} {'n':>3} {'PASS':>5} {'FAIL':>5} {'UNK':>5} | "
          f"{'mean%':>5} {'FAIL_acc':>8} {'FAIL_mean':>9} {'PASS_tend':>9} | "
          f"{'unanim':>6} {'split':>5} | "
          f"{'tmpl_leak':>9} {'c_leak':>6} {'short':>5} | "
          f"{'med_len':>7} {'min':>4} {'max':>4}")
    print("-" * 145)

    all_metrics = {}
    for label, path, mode in RUNS:
        rows = load_run(path)
        if not rows:
            print(f"{label:<28} (no file)")
            continue
        m = metrics_for_run(rows, gt)
        all_metrics[label] = m
        sub = m["unk_subtypes"]
        print(f"{label:<28} {m['n']:>3} {m['n_pass']:>5} {m['n_fail']:>5} {m['n_unk']:>5} | "
              f"{m['meaningful_rate']*100:>4.0f}% {m['fail_recall_overall']*100:>7.1f}% "
              f"{m['fail_recall_among_meaningful']*100:>8.1f}% {m['pass_tendency']*100:>8.1f}% | "
              f"{m['case_unanimous']:>6} {m['case_split']:>5} | "
              f"{m['n_template_leak']:>9} {m['n_c_leak']:>6} {m['n_short']:>5} | "
              f"{m['median_len']:>7} {m['min_len']:>4} {m['max_len']:>4}")

    print()
    print("=" * 80)
    print("Per-case verdict detail (for spot-check of stability)")
    print("=" * 80)
    for label, path, mode in RUNS:
        if label not in all_metrics:
            continue
        m = all_metrics[label]
        by_case = defaultdict(list)
        for v in m["verdicts"]:
            by_case[v["case_id"]].append(v["verdict"])
        print(f"\n## {label}")
        for case_id in sorted(by_case.keys()):
            vs = by_case[case_id]
            short_id = case_id.split("__")[-1][:12]
            unique = "+".join(sorted(set(vs)))
            print(f"  {short_id:<14} agents={len(vs)} unique={unique:<20} ({','.join(vs)})")

    # Save per-row detail for spot-check
    out_path = ROOT / "results/lossy_alg_round27/audit_verdict_detail.json"
    out_data = {label: [
        {k: v[k] for k in ("case_id", "agent_idx", "verdict", "unk_subtype", "len")}
        for v in m["verdicts"]
    ] for label, m in all_metrics.items()}
    out_path.write_text(json.dumps(out_data, indent=2, ensure_ascii=False))
    print(f"\nSaved per-row detail: {out_path}")

    # Print UNK sub-classification breakdown
    print("\n" + "=" * 80)
    print("UNK sub-classification (what does 'garbage' actually look like?)")
    print("=" * 80)
    for label, m in all_metrics.items():
        if m["n_unk"] == 0:
            continue
        print(f"\n## {label} (n_unk={m['n_unk']})")
        for sub, count in m["unk_subtypes"].items():
            print(f"  {sub:<25} {count}")


if __name__ == "__main__":
    main()