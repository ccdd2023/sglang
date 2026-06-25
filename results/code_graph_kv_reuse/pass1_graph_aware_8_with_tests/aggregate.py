#!/usr/bin/env python3
"""Aggregate per-mode per-case results for the 8-case graph-aware pass@1 with candidate tests.

Reads summary.json and writes:
- pass1_8_with_tests_diagnostics.csv  (per-row)
- pass1_8_with_tests_summary.md       (per-mode summary)
- pass1_8_with_tests_summary.json     (machine-readable per-mode counts)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

DEFAULT_RESULT_DIR = Path("results/code_graph_kv_reuse/pass1_graph_aware_8_with_tests")


def _parse_summary(d: dict) -> list[dict]:
    rows: list[dict] = []
    for r in d.get("results", []):
        iid = r.get("instance_id")
        if not iid:
            continue
        for m in r.get("modes", []):
            mode = m.get("mode")
            if not mode:
                continue
            apply = m.get("apply_check") or {}
            cand = m.get("candidate_test") or {}
            synth = m.get("patch_synthesis") or {}
            gen_error = m.get("generation_error") or ""
            synth_error = synth.get("error") or gen_error
            synth_ok = bool(synth.get("ok"))
            cand_rc = cand.get("returncode")
            cand_skipped = (cand_rc is None) or ("skipped by --skip-candidate-tests" in (cand.get("stderr_tail") or ""))
            cand_text = f"{cand.get('stdout_tail') or ''}\n{cand.get('stderr_tail') or ''}".lower()
            if cand_skipped:
                failure_class = "not_run"
            elif cand_rc == 0:
                failure_class = "pass"
            elif any(marker in cand_text for marker in ["failed to build", "metadata-generation-failed", "pip install failed", "install failed"]):
                if any(marker in cand_text for marker in ["syntaxerror", "indentationerror", "return outside function", "expected an indented block"]):
                    failure_class = "candidate_patch_syntax_or_install_failure"
                else:
                    failure_class = "env_install_failure"
            elif any(marker in cand_text for marker in ["syntaxerror", "indentationerror", "return outside function", "expected an indented block"]):
                failure_class = "candidate_patch_syntax_failure"
            else:
                failure_class = "real_pytest_failure"
            rows.append({
                "instance_id": iid,
                "mode": mode,
                "apply_ok": bool(apply.get("returncode") == 0),
                "apply_returncode": apply.get("returncode"),
                "synthesis_ok": synth_ok,
                "synthesis_error": (synth_error or "")[:160],
                "search_not_found": "search not found" in (synth_error or "").lower(),
                "json_parse_failed": "json" in (synth_error or "").lower() and "parse" in (synth_error or "").lower(),
                "candidate_test_skipped": cand_skipped,
                "candidate_test_pass": (cand_rc == 0) if not cand_skipped else False,
                "candidate_test_returncode": cand_rc,
                "failure_class": failure_class,
                "cached_tokens": m.get("cached_tokens"),
                "elapsed_ms": m.get("elapsed_ms"),
                "match_reason": m.get("match_reason", ""),
                "graph_segments": len(m.get("graph_segments", []) or []) if mode == "graph_aware_lossy" else 0,
            })
    return rows


def _write_csv(rows: list[dict]) -> None:
    if not rows:
        return
    cols = list(rows[0].keys())
    with CSV_OUT.open("w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(repr(r.get(c, "")) for c in cols) + "\n")


def _per_mode_summary(rows: list[dict]) -> dict:
    modes = sorted({r["mode"] for r in rows})
    out = {}
    for m in modes:
        mrows = [r for r in rows if r["mode"] == m]
        n = len(mrows)
        cand_runs = [r for r in mrows if not r["candidate_test_skipped"]]
        cand_pass = sum(1 for r in cand_runs if r["candidate_test_pass"])
        out[m] = {
            "n": n,
            "apply_ok": sum(1 for r in mrows if r["apply_ok"]),
            "synthesis_ok": sum(1 for r in mrows if r["synthesis_ok"]),
            "search_not_found": sum(1 for r in mrows if r["search_not_found"]),
            "json_parse_failed": sum(1 for r in mrows if r["json_parse_failed"]),
            "candidate_test_runs": len(cand_runs),
            "candidate_test_pass": cand_pass,
            "candidate_test_pass_rate": (cand_pass / len(cand_runs)) if cand_runs else None,
            "failure_class_counts": {
                klass: sum(1 for r in mrows if r["failure_class"] == klass)
                for klass in sorted({r["failure_class"] for r in mrows})
            },
            "mean_cached_tokens": (sum(r["cached_tokens"] for r in mrows if r["cached_tokens"] is not None) / max(1, sum(1 for r in mrows if r["cached_tokens"] is not None))) if mrows else 0,
        }
    return out


def _write_summary_md(summary: dict, rows: list[dict], table_out: Path) -> None:
    lines: list[str] = []
    lines.append("# 8-case graph-aware pass@1 with candidate tests summary\n")
    lines.append(f"rows={len(rows)}\n")
    lines.append("\n| mode | n | apply_ok | synthesis_ok | search_not_found | json_parse_failed | candidate_test_runs | candidate_test_pass | mean_cached_tokens |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for m, s in summary.items():
        lines.append(
            f"| {m} | {s['n']} | {s['apply_ok']}/{s['n']} | {s['synthesis_ok']}/{s['n']} | "
            f"{s['search_not_found']}/{s['n']} | {s['json_parse_failed']}/{s['n']} | "
            f"{s['candidate_test_runs']} | {s['candidate_test_pass']}/{s['candidate_test_runs']} | "
            f"{s['mean_cached_tokens']:.0f} |"
        )
    lines.append("\n## Failure classes\n")
    lines.append("| mode | failure_class_counts |")
    lines.append("|---|---|")
    for m, s in summary.items():
        lines.append(f"| {m} | `{json.dumps(s['failure_class_counts'], sort_keys=True)}` |")
    table_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> int:
    result_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_RESULT_DIR
    summary_path = result_dir / "summary.json"
    csv_out = result_dir / "pass1_8_with_tests_diagnostics.csv"
    table_out = result_dir / "pass1_8_with_tests_summary.md"
    json_out = result_dir / "pass1_8_with_tests_summary.json"
    if not summary_path.exists():
        print(f"missing {summary_path}", file=sys.stderr)
        return 1
    d = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = _parse_summary(d)
    global CSV_OUT
    CSV_OUT = csv_out
    _write_csv(rows)
    summary = _per_mode_summary(rows)
    json_out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    _write_summary_md(summary, rows, table_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
