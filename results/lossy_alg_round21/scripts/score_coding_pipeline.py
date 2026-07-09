#!/usr/bin/env python3
"""R40 (2026-07-08): Score coding_pipeline outputs.

5-agent pipeline (coder / tester / reviewer / refactorer / integrator).
Reads `outputs.jsonl` produced by `bench_giant_codebase_reuse.py
--task-mode coding_pipeline`. For each case:

  - agent 1 (coder) output: must contain a unified git diff
    → `git apply` against results/giant_codebase/pandas_src
  - agent 2 (tester) output: must contain `<test_result>PASS/FAIL</test_result>`
  - agent 5 (integrator) output: must contain `<final_verdict>PASS/FAIL</final_verdict>`

Accuracy metric: per case, the case is "correct" if
  (git apply succeeded for agent 1's diff)
  AND (tester parses as PASS OR integrator parses as PASS).

For each config (lossless / R38b / etc.), produces:
  - apply_rate (fraction of cases where git apply succeeded)
  - tester_pass_rate (fraction of cases where tester said PASS)
  - integrator_pass_rate
  - combined_pass_rate (apply AND (tester OR integrator) all PASS)
  - per-stage TTFT breakdown avg (pulled from rows.csv sidecar if present)

Usage:
  python results/lossy_alg_round21/scripts/score_coding_pipeline.py \
      results/baseline_ours_r38b_coding_pipeline_5x5_verdict/outputs.jsonl \
      --label R38b \
      --manifest results/giant_codebase/tasks/pandas__pandas__1000/manifest.jsonl \
      --repo-root results/giant_codebase/pandas_src \
      --rows-csv results/baseline_ours_r38b_coding_pipeline_5x5_verdict/rows.csv
"""
import argparse
import csv
import json
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path


DIFF_RE = re.compile(r"^diff --git .+$", re.MULTILINE)
TEST_RESULT_RE = re.compile(r"<test_result>\s*(PASS|FAIL)\s*(?:—\s*(.+?))?\s*</test_result>", re.IGNORECASE | re.DOTALL)
FINAL_VERDICT_RE = re.compile(r"<final_verdict>\s*(PASS|FAIL)\s*(?:—\s*(.+?))?\s*</final_verdict>", re.IGNORECASE | re.DOTALL)


def extract_diff(text: str) -> str | None:
    """Return the diff text starting at the first 'diff --git' line. None if not found."""
    m = DIFF_RE.search(text)
    if not m:
        return None
    return text[m.start():].rstrip() + "\n"


def try_apply(diff_text: str, repo_root: Path) -> tuple[bool, str]:
    """Write diff to a temp file, attempt `git apply --check`, return (success, stderr)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False) as f:
        f.write(diff_text)
        patch_path = f.name
    try:
        result = subprocess.run(
            ["git", "apply", "--check", patch_path],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=15,
        )
        return (result.returncode == 0, result.stderr)
    except subprocess.TimeoutExpired:
        return (False, "git apply timed out")
    finally:
        Path(patch_path).unlink(missing_ok=True)


def parse_test_result(text: str) -> tuple[str | None, str | None]:
    """Return (verdict, reason) from <test_result>...</test_result> tag, or (None, None)."""
    m = TEST_RESULT_RE.search(text)
    if not m:
        return (None, None)
    return (m.group(1).upper(), (m.group(2) or "").strip() or None)


def parse_final_verdict(text: str) -> tuple[str | None, str | None]:
    m = FINAL_VERDICT_RE.search(text)
    if not m:
        return (None, None)
    return (m.group(1).upper(), (m.group(2) or "").strip() or None)


def load_outputs(path: Path) -> list[dict]:
    """Load outputs.jsonl into a list of dicts."""
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  WARN: skipping malformed JSON line: {e}", file=sys.stderr)
    return rows


def aggregate_per_case(rows: list[dict], repo_root: Path) -> dict:
    """Group rows by case_id, compute per-case apply/tester/integrator results."""
    by_case = defaultdict(dict)  # case_id -> {agent_idx: text}
    for r in rows:
        by_case[r.get("case_id", "")][int(r.get("agent_idx", 0))] = r.get("output_text", "")

    cases = []
    for case_id, agents in by_case.items():
        coder_text = agents.get(1, "")
        tester_text = agents.get(2, "")
        integrator_text = agents.get(5, "")
        reviewer_text = agents.get(3, "")
        refactorer_text = agents.get(4, "")

        diff = extract_diff(coder_text)
        apply_ok = False
        apply_err = ""
        if diff is not None:
            apply_ok, apply_err = try_apply(diff, repo_root)

        tester_verdict, tester_reason = parse_test_result(tester_text)
        integrator_verdict, integrator_reason = parse_final_verdict(integrator_text)

        # Combined accuracy: apply succeeds AND (tester PASS OR integrator PASS)
        any_pass = (
            (tester_verdict == "PASS")
            or (integrator_verdict == "PASS")
        )
        combined_ok = apply_ok and any_pass

        cases.append({
            "case_id": case_id,
            "apply_ok": apply_ok,
            "apply_err": apply_err.strip()[:200],
            "diff_found": diff is not None,
            "tester_verdict": tester_verdict,
            "tester_reason": tester_reason or "",
            "integrator_verdict": integrator_verdict,
            "integrator_reason": integrator_reason or "",
            "reviewer_text_len": len(reviewer_text),
            "refactorer_text_len": len(refactorer_text),
            "combined_ok": combined_ok,
        })
    return {"cases": cases}


def avg_ttft_breakdown(rows_csv: Path | None) -> dict[str, float]:
    """Read rows.csv and compute the avg of the 8 TTFT-breakdown columns."""
    if rows_csv is None or not rows_csv.exists():
        return {}
    cols = [
        "ttft_tokenize_ms", "ttft_radix_prefix_ms", "ttft_chunk_plan_ms",
        "ttft_copy_ms", "ttft_gap_prefill_ms",
        "ttft_head_recompute_early_ms", "ttft_head_recompute_late_ms",
        "ttft_decode_first_token_ms",
    ]
    sums = {c: 0.0 for c in cols}
    n = 0
    with rows_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n += 1
            for c in cols:
                try:
                    sums[c] += float(row.get(c, 0) or 0)
                except (ValueError, TypeError):
                    pass
    if n == 0:
        return {}
    return {c: round(sums[c] / n, 3) for c in cols}


def render_report(label: str, agg: dict, breakdown: dict[str, float]) -> str:
    cases = agg["cases"]
    n = len(cases)
    if n == 0:
        return f"## {label}: no cases\n"
    diff_found_n = sum(1 for c in cases if c["diff_found"])
    apply_n = sum(1 for c in cases if c["apply_ok"])
    tester_pass_n = sum(1 for c in cases if c["tester_verdict"] == "PASS")
    tester_fail_n = sum(1 for c in cases if c["tester_verdict"] == "FAIL")
    tester_unk_n = n - tester_pass_n - tester_fail_n
    integ_pass_n = sum(1 for c in cases if c["integrator_verdict"] == "PASS")
    integ_fail_n = sum(1 for c in cases if c["integrator_verdict"] == "FAIL")
    integ_unk_n = n - integ_pass_n - integ_fail_n
    combined_n = sum(1 for c in cases if c["combined_ok"])

    out = []
    out.append(f"## {label} ({n} cases)")
    out.append("")
    out.append("| metric | count | rate |")
    out.append("|---|---|---|")
    out.append(f"| diff_found (coder emitted a diff) | {diff_found_n}/{n} | {100*diff_found_n/n:.1f}% |")
    out.append(f"| git apply success | {apply_n}/{n} | {100*apply_n/n:.1f}% |")
    out.append(f"| tester PASS | {tester_pass_n}/{n} | {100*tester_pass_n/n:.1f}% |")
    out.append(f"| tester FAIL | {tester_fail_n}/{n} | {100*tester_fail_n/n:.1f}% |")
    out.append(f"| tester UNKNOWN (no <test_result> tag) | {tester_unk_n}/{n} | {100*tester_unk_n/n:.1f}% |")
    out.append(f"| integrator PASS | {integ_pass_n}/{n} | {100*integ_pass_n/n:.1f}% |")
    out.append(f"| integrator FAIL | {integ_fail_n}/{n} | {100*integ_fail_n/n:.1f}% |")
    out.append(f"| integrator UNKNOWN (no <final_verdict> tag) | {integ_unk_n}/{n} | {100*integ_unk_n/n:.1f}% |")
    out.append(f"| **combined (apply AND (tester OR integrator) PASS)** | **{combined_n}/{n}** | **{100*combined_n/n:.1f}%** |")
    out.append("")

    if breakdown:
        total = sum(breakdown.values())
        out.append("### Per-stage TTFT (avg ms)")
        out.append("")
        out.append("| stage | ms | % |")
        out.append("|---|---|---|")
        labels = {
            "ttft_tokenize_ms": "tokenize",
            "ttft_radix_prefix_ms": "radix_prefix",
            "ttft_chunk_plan_ms": "chunk_plan",
            "ttft_copy_ms": "copy",
            "ttft_gap_prefill_ms": "gap_prefill",
            "ttft_head_recompute_early_ms": "head_recompute_early",
            "ttft_head_recompute_late_ms": "head_recompute_late",
            "ttft_decode_first_token_ms": "decode_first_token",
        }
        for k, label in labels.items():
            ms = breakdown.get(k, 0.0)
            pct = 100 * ms / total if total > 0 else 0
            out.append(f"| {label} | {ms:.1f} | {pct:.1f}% |")
        out.append(f"| **total summed** | **{total:.1f}** | 100.0% |")
        out.append("")

    # Per-case detail
    out.append("### Per-case")
    out.append("")
    out.append("| case_id | apply | tester | integrator | combined |")
    out.append("|---|---|---|---|---|")
    for c in cases:
        apply_mark = "✓" if c["apply_ok"] else ("-" if not c["diff_found"] else "✗")
        tester_mark = c["tester_verdict"] or "UNK"
        integ_mark = c["integrator_verdict"] or "UNK"
        combined_mark = "✓" if c["combined_ok"] else "✗"
        out.append(
            f"| {c['case_id'][:40]} | {apply_mark} | {tester_mark} | {integ_mark} | {combined_mark} |"
        )
    out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description="Score coding_pipeline outputs")
    ap.add_argument("outputs_jsonl", type=Path, help="Path to outputs.jsonl")
    ap.add_argument("--label", default="config", help="Label for the report header")
    ap.add_argument("--manifest", type=Path, default=None, help="manifest.jsonl (for case metadata)")
    ap.add_argument("--repo-root", type=Path, default=Path("results/giant_codebase/pandas_src"),
                    help="Repo root to apply diffs against")
    ap.add_argument("--rows-csv", type=Path, default=None, help="rows.csv (for TTFT breakdown)")
    args = ap.parse_args()

    if not args.outputs_jsonl.exists():
        print(f"ERROR: {args.outputs_jsonl} not found", file=sys.stderr)
        sys.exit(1)

    rows = load_outputs(args.outputs_jsonl)
    if not rows:
        print(f"ERROR: no rows in {args.outputs_jsonl}", file=sys.stderr)
        sys.exit(1)

    agg = aggregate_per_case(rows, args.repo_root)
    breakdown = avg_ttft_breakdown(args.rows_csv)
    print(render_report(args.label, agg, breakdown))


if __name__ == "__main__":
    main()