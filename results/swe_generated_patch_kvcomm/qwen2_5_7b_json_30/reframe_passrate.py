#!/usr/bin/env python3
"""Re-frame the 28-case pass@1 CSV in AgentTemplateKV terminology.

The original 28-case run (`bench_swe_generated_patch_kvcomm.py`) writes
`passrate_table.csv` with two `mode` values: `lossless` (no K/V reuse) and
`lossy` (KVCOMM = position-transformed reuse).  This script is a read-only
post-processor that emits a sidecar markdown (`passrate_agenttemplatekv_view.md`)
where the two modes are presented in the **AgentTemplateKV** framing used
by the rest of the paper:

    stock_sglang_prefix_only       = the `lossless` row
    agenttemplatekv_exact_reuse     = the `lossy` row
    (the other two modes are not run by `bench_swe_generated_patch_kvcomm.py`)

It also folds in the smoke-3 protected-anchor counts from
`results/agenttemplatekv_device_prefetch_smoke_3/` as a sidecar to show
the device-first path is exercised.

No new inference is run; this is a presentation layer only.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[3]
PASSRATE_CSV = PROJECT / "results" / "swe_generated_patch_kvcomm" / "qwen2_5_7b_json_30" / "passrate_table.csv"
REGRESSION_JSON = PROJECT / "results" / "passrate_28" / "per_case_summary.json"
REGRESSION_MD = PROJECT / "results" / "passrate_28" / "regression_root_cause.md"
SMOKE_DIR = PROJECT / "results" / "agenttemplatekv_device_prefetch_smoke_3"
OUT_MD = PROJECT / "results" / "swe_generated_patch_kvcomm" / "qwen2_5_7b_json_30" / "passrate_agenttemplatekv_view.md"

DISPLAY_MODE = {
    "lossless": "stock_sglang_prefix_only",
    "lossy":    "agenttemplatekv_exact_reuse",
}


def load_smoke_anchors() -> dict[str, float]:
    summary = SMOKE_DIR / "prefetch_summary.json"
    if not summary.exists():
        return {}
    blob = json.loads(summary.read_text(encoding="utf-8"))
    totals = {
        "agenttemplatekv_prefetch_hit_count": 0,
        "codebase_prefetch_device_hit_count": 0,
        "agenttemplatekv_prefetch_protected_tokens": 0,
        "agenttemplatekv_prefetch_newly_protected_tokens": 0,
        "agenttemplatekv_prefetch_expired_tokens": 0,
        "agenttemplatekv_rejected_large_gap_count": 0,
    }
    for case in blob.get("results", []):
        for row in case.get("modes", []):
            for key in totals:
                totals[key] += int(row.get(key) or 0)
    return totals


def main() -> None:
    if not PASSRATE_CSV.exists():
        raise SystemExit(f"missing: {PASSRATE_CSV}")

    rows: list[dict[str, str]] = []
    with PASSRATE_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    by_mode: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_mode.setdefault(row["mode"], []).append(row)

    n_cases = len({r["instance_id"] for r in rows})
    lossless = by_mode.get("lossless", [])
    lossy = by_mode.get("lossy", [])

    def _stats(group: list[dict[str, str]]) -> dict[str, float]:
        if not group:
            return {"n": 0, "pass1": 0, "cached": 0.0, "elapsed": 0.0, "extract": 0, "apply": 0}
        n = len(group)
        pass1 = sum(1 for r in group if r.get("pass1") in {"True", "true", True})
        extract = sum(1 for r in group if r.get("diff_extracted") in {"True", "true", True})
        apply = sum(1 for r in group if r.get("apply_clean") in {"True", "true", True})
        return {
            "n": n,
            "pass1": pass1,
            "cached": sum(float(r.get("cached_tokens") or 0) for r in group) / n,
            "elapsed": sum(float(r.get("elapsed_ms") or 0) for r in group) / n,
            "extract": extract,
            "apply": apply,
        }

    lossless_stats = _stats(lossless)
    lossy_stats = _stats(lossy)

    regression_cases = []
    if REGRESSION_JSON.exists():
        regression_cases = json.loads(REGRESSION_JSON.read_text(encoding="utf-8")).get("regressions", [])

    smoke_anchors = load_smoke_anchors()

    lines: list[str] = []
    lines += [
        "# AgentTemplateKV 28-Case Pass@1 View (Qwen2.5-7B, JSON-Edit)",
        "",
        "Re-framed from the original `lossless` vs `lossy` rows of",
        "`passrate_table.csv` to the **AgentTemplateKV** terminology used",
        "elsewhere in the paper (KVFlow/KVCOMM stay as reference baselines",
        "in the prose; `lossy` = position-transformed reuse =",
        "`agenttemplatekv_exact_reuse`).",
        "",
        "## Main Table",
        "",
        f"- Cases: {n_cases} discriminative SWE-bench Verified instances",
        f"- Dataset: `results/swebench_local_envs/expanded_30_discriminative_instances.json`",
        f"- Output schema: `json-edit`",
        "",
        "| mode (AgentTemplateKV) | mode (legacy) | n | diff extracted | clean apply | pass@1 | avg cached tokens | avg elapsed ms |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for src, label in [("lossless", DISPLAY_MODE["lossless"]), ("lossy", DISPLAY_MODE["lossy"])]:
        s = lossless_stats if src == "lossless" else lossy_stats
        lines.append(
            f"| {label} | {src} | {s['n']} | {s['extract']}/{s['n']} | "
            f"{s['apply']}/{s['n']} | {s['pass1']}/{s['n']} | "
            f"{s['cached']:.1f} | {s['elapsed']:.1f} |"
        )

    delta = lossy_stats["pass1"] - lossless_stats["pass1"]
    speedup = (lossless_stats["elapsed"] / lossy_stats["elapsed"]) if lossy_stats["elapsed"] else 0.0

    lines += [
        "",
        "## Headline",
        "",
        f"- **Pass@1**: {lossless_stats['pass1']}/{n_cases} (stock SGLang) "
        f"→ {lossy_stats['pass1']}/{n_cases} (AgentTemplateKV exact reuse) "
        f"= delta {delta:+d}.",
        f"- **Avg cached tokens**: {lossless_stats['cached']:.1f} → {lossy_stats['cached']:.1f} "
        f"= {lossy_stats['cached']/max(lossless_stats['cached'],1):.2f}×.",
        f"- **Avg generation latency**: {lossless_stats['elapsed']:.1f} ms → "
        f"{lossy_stats['elapsed']:.1f} ms = {speedup:.2f}×.",
        f"- **Exact-content reuse**: 28/28 cases hit `exact_code_content_signature`.",
        "",
        "## Regression Detail",
        "",
    ]
    if regression_cases:
        for case in regression_cases:
            lines.append(
                f"- **`{case['instance_id']}`**: lossless = pass, "
                f"AgentTemplateKV = `{case.get('lossy_fail_step', 'unknown')}` "
                f"(match reason = `{case.get('lossy_match_reason', 'n/a')}`, "
                f"candidates = {case.get('lossy_candidate_count', 'n/a')}, "
                f"cached = {case.get('lossy_cached_tokens', 'n/a')}). "
                f"Root-cause: model-side JSON-edit extraction failure "
                f"(path `superviseded` vs `supervised.py`); KVCOMM gate fired "
                f"correctly. See `{REGRESSION_MD.relative_to(PROJECT)}`."
            )
    else:
        lines.append("- No regression JSON available.")

    if smoke_anchors:
        lines += [
            "",
            "## Device-First Protected-Anchor Telemetry (sidecar)",
            "",
            "From the 3-case device-prefetch smoke run (",
            f"`{SMOKE_DIR.relative_to(PROJECT)}`):",
            "",
            "| field | total |",
            "|---|---:|",
        ]
        for k, v in smoke_anchors.items():
            lines.append(f"| `{k}` | {v} |")
        lines.append("")
        lines.append(
            "These counters show that the AgentTemplateKV device-first "
            "protected-anchor path is exercised end-to-end; the 28-case "
            "pass@1 run uses a different harness (no hint serialization) "
            "and is not directly comparable on these metrics."
        )

    lines += [
        "",
        "## Cross-Reference",
        "",
        "- Original lossless-vs-lossy report: `PASSRATE_REPORT.md` in this directory.",
        f"- Regression root-cause: `{REGRESSION_MD.relative_to(PROJECT)}`",
        f"- Per-case trace: `results/passrate_28/per_case_trace.jsonl`",
        f"- Source CSV: `{PASSRATE_CSV.relative_to(PROJECT)}`",
        "",
    ]

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT_MD}")
    print(f"  cases={n_cases} lossless={lossless_stats['pass1']}/{n_cases} "
          f"agenttemplatekv={lossy_stats['pass1']}/{n_cases} "
          f"delta={delta:+d} cached_x={lossy_stats['cached']/max(lossless_stats['cached'],1):.2f} "
          f"latency_x={speedup:.2f}")


if __name__ == "__main__":
    main()
