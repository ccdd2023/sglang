#!/usr/bin/env python3
"""Summarize pass@1 repair-attempt sweeps.

The generator writes one ``summary.json`` per run. This helper merges runs such
as repair-attempts=0/1/2 and reports generation, synthesis, apply, and test
success rates per reuse mode.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[2]
MODES = ("lossless", "lossy", "lossy_prefetch")


def git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(PROJECT),
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def load_summary(path: Path) -> dict[str, Any]:
    if path.is_dir() and (path / "summary.json").exists():
        summary_path = path / "summary.json"
    elif path.is_dir() and (path / "passrate_table.csv").exists():
        return load_passrate_csv(path / "passrate_table.csv")
    elif path.suffix == ".csv":
        return load_passrate_csv(path)
    else:
        summary_path = path
    return json.loads(summary_path.read_text(encoding="utf-8"))


def load_passrate_csv(path: Path) -> dict[str, Any]:
    results_by_case: dict[str, dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            case = results_by_case.setdefault(
                row["instance_id"],
                {
                    "instance_id": row["instance_id"],
                    "repo": row.get("repo", ""),
                    "modes": [],
                },
            )
            case["modes"].append(
                {
                    "mode": row["mode"],
                    "elapsed_ms": float(row["elapsed_ms"]) if row.get("elapsed_ms") else None,
                    "repair_elapsed_ms": float(row["repair_elapsed_ms"]) if row.get("repair_elapsed_ms") else None,
                    "cached_tokens": int(float(row["cached_tokens"])) if row.get("cached_tokens") else 0,
                    "diff_extracted": row.get("diff_extracted") == "True",
                    "patch_synthesis": {
                        "ok": row.get("synthesis_ok") == "True",
                        "error": "" if row.get("synthesis_ok") == "True" else "csv_synthesis_failed",
                    },
                    "apply_check": {
                        "returncode": 0 if row.get("apply_clean") == "True" else 1,
                    },
                    "repair_attempted": bool(row.get("repair_elapsed_ms")),
                    "generation_error": "",
                    "candidate_test": {
                        "returncode": 0 if row.get("pass1") == "True" else 1,
                    },
                }
            )
    return {
        "model": "",
        "timestamp": "",
        "source_csv": str(path),
        "results": list(results_by_case.values()),
    }


def classify_synthesis_error(mode_result: dict[str, Any]) -> str:
    if mode_result.get("generation_error"):
        return "generation_error"
    if mode_result.get("diff_extracted"):
        return ""
    synthesis = mode_result.get("patch_synthesis") or {}
    error = str(synthesis.get("error") or "").lower()
    if "json parse failed" in error:
        return "json_parse_failed"
    if "no json object" in error:
        return "no_json_object"
    if "search not found" in error:
        return "search_not_found"
    if "file not found" in error:
        return "file_not_found"
    if "ellipsis" in error or "placeholder" in error:
        return "placeholder"
    if error:
        return "other_synthesis_error"
    return "no_diff"


def summarize(summary: dict[str, Any], label: str) -> list[dict[str, Any]]:
    by_mode: dict[str, list[dict[str, Any]]] = {mode: [] for mode in MODES}
    for case in summary.get("results", []):
        for mode_result in case.get("modes", []):
            mode = mode_result.get("mode")
            if mode in by_mode:
                by_mode[mode].append({"case": case, "mode_result": mode_result})

    rows = []
    for mode, items in by_mode.items():
        n = len(items)
        if n == 0:
            continue
        generated = 0
        diff_extracted = 0
        repaired = 0
        apply_ok = 0
        test_ok = 0
        json_fail = 0
        search_fail = 0
        no_diff = 0
        generation_error = 0
        failures: list[str] = []
        for item in items:
            case = item["case"]
            mode_result = item["mode_result"]
            if not mode_result.get("generation_error"):
                generated += 1
            else:
                generation_error += 1
            if mode_result.get("diff_extracted"):
                diff_extracted += 1
            if mode_result.get("repair_attempted"):
                repaired += 1
            if (mode_result.get("apply_check") or {}).get("returncode") == 0:
                apply_ok += 1
            if (mode_result.get("candidate_test") or {}).get("returncode") == 0:
                test_ok += 1
            err_class = classify_synthesis_error(mode_result)
            if err_class == "json_parse_failed":
                json_fail += 1
            elif err_class == "search_not_found":
                search_fail += 1
            elif err_class:
                no_diff += 1
            if (mode_result.get("candidate_test") or {}).get("returncode") != 0:
                failures.append(f"{case.get('instance_id')}:{err_class or 'test_failed'}")
        rows.append(
            {
                "run": label,
                "mode": mode,
                "n": n,
                "generated": generated,
                "diff_extracted": diff_extracted,
                "repair_attempted": repaired,
                "apply_ok": apply_ok,
                "pass_at_1": test_ok,
                "pass_at_1_rate": round(test_ok / n, 4),
                "json_parse_failed": json_fail,
                "search_not_found": search_fail,
                "other_no_diff": no_diff,
                "generation_error": generation_error,
                "failure_cases": ";".join(failures),
            }
        )
    return rows


def write_outputs(out_dir: Path, rows: list[dict[str, Any]], inputs: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["run", "mode"]
    with (out_dir / "repair_sweep_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    report = [
        "# Pass@1 Repair Sweep Summary",
        "",
        f"- Git commit: `{git_commit()}`",
        f"- Input runs: `{', '.join(inputs)}`",
        "",
        "| run | mode | n | diff extracted | apply ok | pass@1 | JSON parse fail | search not found | repaired |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        report.append(
            f"| {row['run']} | {row['mode']} | {row['n']} | {row['diff_extracted']} | "
            f"{row['apply_ok']} | {row['pass_at_1']} ({row['pass_at_1_rate']:.3f}) | "
            f"{row['json_parse_failed']} | {row['search_not_found']} | {row['repair_attempted']} |"
        )
    report.extend(
        [
            "",
            "## Per-Run Failure Cases",
            "",
        ]
    )
    for row in rows:
        report.append(f"- `{row['run']} / {row['mode']}`: {row['failure_cases']}")
    (out_dir / "REPAIR_SWEEP_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path,
                        help="Run directories or summary.json files to merge.")
    parser.add_argument("--labels", nargs="*", default=None,
                        help="Optional labels matching runs, e.g. repair0 repair1 repair2.")
    parser.add_argument("--out-dir", type=Path,
                        default=PROJECT / "results" / "swe_generated_patch_kvcomm" / "repair_sweep")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = args.labels or [path.name for path in args.runs]
    if len(labels) != len(args.runs):
        raise SystemExit("--labels length must match runs length")
    rows: list[dict[str, Any]] = []
    for label, path in zip(labels, args.runs):
        rows.extend(summarize(load_summary(path), label))
    write_outputs(args.out_dir, rows, [str(path) for path in args.runs])


if __name__ == "__main__":
    main()
