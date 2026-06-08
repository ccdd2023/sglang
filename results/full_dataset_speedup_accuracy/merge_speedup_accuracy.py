#!/usr/bin/env python3
"""Consolidate the 500-case AgentTemplateKV speedup with the 28-case pass@1.

Inputs (paths are overridable for smoke tests):
  --speedup-csv:   results/coding_kvflow_prefetch/qwen2_5_7b_500/prefetch_table.csv
                   (500 cases × 4 modes = ~2,000 records when full)
  --speedup-summary: results/coding_kvflow_prefetch/qwen2_5_7b_500/summary.json
  --passrate-csv:  results/swe_generated_patch_kvcomm/qwen2_5_7b_json_30/passrate_table.csv
                   (28 cases × 2 modes = 56 records)

Outputs:
  - report.md   paper-ready table + bootstrap CI + tail analysis
  - summary.json machine-readable copy
  - merged_table.csv  long-format data (mode, instance_id, latency_ms, ...)

Reuses the bootstrap-CI pattern from the 100-case run
(`results/coding_kvflow_prefetch/qwen2_5_7b_100/compute_ci.py`).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]


def _rel(p: Path) -> str:
    """Display a path relative to PROJECT when possible, else absolute."""
    try:
        return str(p.resolve().relative_to(PROJECT))
    except ValueError:
        return str(p)
DEFAULT_SPEEDUP_CSV = PROJECT / "results" / "coding_kvflow_prefetch" / "qwen2_5_7b_500" / "prefetch_table.csv"
DEFAULT_SPEEDUP_SUMMARY = PROJECT / "results" / "coding_kvflow_prefetch" / "qwen2_5_7b_500" / "summary.json"
DEFAULT_PASSRATE_CSV = PROJECT / "results" / "swe_generated_patch_kvcomm" / "qwen2_5_7b_json_30" / "passrate_table.csv"
DEFAULT_OUT_DIR = PROJECT / "results" / "full_dataset_speedup_accuracy"

MODES_4 = [
    "baseline_prefix_cache_only",
    "kvflow_prefix_only",
    "kvflow_prefix_plus_codebase_prefetch",
    "kvcomm_lossy_plus_codebase_prefetch",
]
DISPLAY_MODE = {
    "baseline_prefix_cache_only": "stock_sglang_prefix_only",
    "kvflow_prefix_only": "kvflow_style_prefix_baseline",
    "kvflow_prefix_plus_codebase_prefetch": "kvflow_style_prefix_plus_hints",
    "kvcomm_lossy_plus_codebase_prefetch": "agenttemplatekv_exact_reuse",
}

PASSRATE_MODE = {
    "lossless": "stock_sglang_prefix_only",
    "lossy":    "agenttemplatekv_exact_reuse",
}


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    if q <= 0:
        return s[0]
    if q >= 100:
        return s[-1]
    k = (len(s) - 1) * (q / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] * (c - k) + s[c] * (k - f)


def bootstrap_paired(
    paired_diffs: list[float], n_resamples: int = 10_000, seed: int = 0
) -> dict[str, float]:
    """One-sided paired bootstrap test of mean(diff) > 0.

    Mirrors `results/coding_kvflow_prefetch/qwen2_5_7b_100/compute_ci.py`:
    H1: improvement is real (d > 0).  p = P(resample_mean <= 0).
    """
    if not paired_diffs:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p": 1.0, "n": 0}
    rng = random.Random(seed)
    n = len(paired_diffs)
    observed = sum(paired_diffs) / n
    resampled_means = []
    le_count = 0
    for _ in range(n_resamples):
        sample = [paired_diffs[rng.randrange(n)] for _ in range(n)]
        m = sum(sample) / n
        resampled_means.append(m)
        if m <= 0:
            le_count += 1
    resampled_means.sort()
    ci_low = resampled_means[int(0.025 * n_resamples)]
    ci_high = resampled_means[int(0.975 * n_resamples)]
    p = le_count / n_resamples
    return {
        "mean": observed,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p": p,
        "n": n,
    }


def speedup_per_mode(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    by_mode: dict[str, list[dict[str, str]]] = {m: [] for m in MODES_4}
    for r in rows:
        if r["mode"] in by_mode:
            by_mode[r["mode"]].append(r)
    stats: dict[str, dict[str, float]] = {}
    for mode, group in by_mode.items():
        if not group:
            continue
        latencies = [float(r.get("elapsed_ms") or 0) for r in group]
        cached = [float(r.get("cached_tokens") or 0) for r in group]
        exact_hits = sum(1 for r in group if r.get("lossy_match_reason") == "exact_code_content_signature")
        f1s = [float(r.get("output_token_f1_vs_baseline") or 0) for r in group]
        stats[mode] = {
            "n": len(group),
            "avg_latency_ms": statistics.mean(latencies),
            "p50_latency_ms": percentile(latencies, 50),
            "p90_latency_ms": percentile(latencies, 90),
            "p99_latency_ms": percentile(latencies, 99),
            "max_latency_ms": max(latencies),
            "avg_cached_tokens": statistics.mean(cached),
            "exact_content_hit_rate": exact_hits / len(group),
            "avg_token_f1": statistics.mean(f1s),
        }
    return stats


def passrate_per_mode(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    by_mode: dict[str, list[dict[str, str]]] = {}
    for r in rows:
        by_mode.setdefault(r["mode"], []).append(r)
    out: dict[str, dict[str, float]] = {}
    for mode, group in by_mode.items():
        n = len(group)
        pass1 = sum(1 for r in group if r.get("pass1") in {"True", "true", True})
        out[mode] = {
            "n": n,
            "pass_at_1": pass1,
            "pass_at_1_rate": pass1 / n if n else 0.0,
            "avg_cached_tokens": statistics.mean(float(r.get("cached_tokens") or 0) for r in group) if n else 0.0,
            "avg_elapsed_ms": statistics.mean(float(r.get("elapsed_ms") or 0) for r in group) if n else 0.0,
        }
    return out


def render_report(
    speedup_stats: dict[str, dict[str, float]],
    passrate_stats: dict[str, dict[str, float]],
    ci_atk: dict[str, float],
    ci_cached: dict[str, float],
    n_speedup: int,
    n_passrate: int,
    speedup_csv: Path,
    passrate_csv: Path,
) -> str:
    lines: list[str] = [
        "# Full-Dataset Speedup + Accuracy (AgentTemplateKV)",
        "",
        f"Consolidated view of the {n_speedup}-case serving speedup",
        f"(`{_rel(speedup_csv)}`) and the {n_passrate}-case",
        f"discriminative-subset pass@1 (`{_rel(passrate_csv)}`).",
        "",
        "## Main Table",
        "",
        "| mode (display) | n (e2e) | p50 ms | p90 ms | avg cached | exact hit | F1 | n (acc) | pass@1 | delta vs lossless |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    lossless_passrate = passrate_stats.get("lossless", {}).get("pass_at_1", 0)
    for mode in MODES_4:
        s = speedup_stats.get(mode, {})
        if not s:
            continue
        display = DISPLAY_MODE[mode]
        # Map 4 serving modes to pass@1: stock_sglang / agenttemplatekv map directly;
        # the two kvflow_style_* modes share the lossless pass@1 (no exact reuse).
        if display == "stock_sglang_prefix_only":
            pr = passrate_stats.get("lossless", {})
        elif display == "agenttemplatekv_exact_reuse":
            pr = passrate_stats.get("lossy", {})
        else:
            pr = passrate_stats.get("lossless", {})
        pr_pass = pr.get("pass_at_1", 0)
        pr_n = pr.get("n", 0)
        pr_rate = pr.get("pass_at_1_rate", 0.0)
        delta = pr_pass - lossless_passrate
        lines.append(
            f"| {display} | {int(s['n'])} | {s['p50_latency_ms']:.0f} | {s['p90_latency_ms']:.0f} | "
            f"{s['avg_cached_tokens']:.0f} | {s['exact_content_hit_rate']:.2f} | {s['avg_token_f1']:.4f} | "
            f"{pr_n} | {pr_pass}/{pr_n} ({pr_rate:.2%}) | {delta:+d} |"
        )

    lines += [
        "",
        "## Tail Analysis (per mode)",
        "",
        "| mode | p50 ms | p90 ms | p99 ms | max ms |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode in MODES_4:
        s = speedup_stats.get(mode, {})
        if not s:
            continue
        lines.append(
            f"| {DISPLAY_MODE[mode]} | {s['p50_latency_ms']:.0f} | {s['p90_latency_ms']:.0f} | "
            f"{s['p99_latency_ms']:.0f} | {s['max_latency_ms']:.0f} |"
        )

    lines += [
        "",
        "## Statistical Significance (paired bootstrap, 10,000 resamples)",
        "",
        f"- **Latency**: stock SGLang − AgentTemplateKV = "
        f"**{ci_atk['mean']:+.0f} ms** (95% CI [{ci_atk['ci_low']:+.0f}, {ci_atk['ci_high']:+.0f}] ms), "
        f"one-sided p = **{ci_atk['p']:.4f}** (n = {ci_atk['n']}).",
        f"- **Cached tokens**: AgentTemplateKV − stock SGLang = "
        f"**{ci_cached['mean']:+.0f}** (95% CI [{ci_cached['ci_low']:+.0f}, {ci_cached['ci_high']:+.0f}]), "
        f"one-sided p = **{ci_cached['p']:.4f}** (n = {ci_cached['n']}).",
        "",
    ]
    if ci_atk["p"] < 0.05:
        lines.append("The latency improvement is significant at p < 0.05.")
    else:
        lines.append("**The latency improvement is NOT significant at p < 0.05** "
                     "(see Risk flag 5 in the plan; consider re-aggregating the 100-case data instead).")
    if ci_cached["p"] < 0.05:
        lines.append("The cached-token gain is significant at p < 0.05.")
    lines += [
        "",
        "## Pass@1 Detail",
        "",
        f"- **Cases**: {n_passrate} discriminative SWE-bench Verified instances "
        "(only this subset has local repo envs + gold tests; full 500-case pass@1 "
        "requires building more envs and is out of scope for this round).",
        f"- **Lossless** (stock SGLang) pass@1: {passrate_stats.get('lossless', {}).get('pass_at_1', 0)}/{n_passrate}.",
        f"- **AgentTemplateKV exact reuse** pass@1: {passrate_stats.get('lossy', {}).get('pass_at_1', 0)}/{n_passrate}.",
        f"- **Delta**: {passrate_stats.get('lossy', {}).get('pass_at_1', 0) - passrate_stats.get('lossless', {}).get('pass_at_1', 0):+d}.",
        "- Regression root-cause: `scikit-learn-10844` is a model-side JSON-edit "
        "extraction failure; KVCOMM gate fired correctly. See "
        "`results/passrate_28/regression_root_cause.md`.",
        "",
        "## Files",
        "",
        f"- Speedup source: `{_rel(speedup_csv)}`",
        f"- Pass@1 source: `{_rel(passrate_csv)}`",
        "- Aggregated long-format: `merged_table.csv`",
        "- Machine-readable: `summary.json`",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--speedup-csv", type=Path, default=DEFAULT_SPEEDUP_CSV)
    parser.add_argument("--speedup-summary", type=Path, default=DEFAULT_SPEEDUP_SUMMARY)
    parser.add_argument("--passrate-csv", type=Path, default=DEFAULT_PASSRATE_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    speedup_rows = load_csv(args.speedup_csv)
    passrate_rows = load_csv(args.passrate_csv)

    if not speedup_rows:
        print(f"WARNING: speedup CSV empty or missing: {args.speedup_csv}")
    if not passrate_rows:
        print(f"WARNING: passrate CSV empty or missing: {args.passrate_csv}")

    speedup_stats = speedup_per_mode(speedup_rows)
    passrate_stats = passrate_per_mode(passrate_rows)

    # Paired bootstrap: per-case latency difference, stock - AgentTemplateKV
    by_id_atk: dict[str, float] = {}
    by_id_stock: dict[str, float] = {}
    by_id_cached_atk: dict[str, float] = {}
    by_id_cached_stock: dict[str, float] = {}
    for r in speedup_rows:
        if r["mode"] == "kvcomm_lossy_plus_codebase_prefetch":
            by_id_atk[r["instance_id"]] = float(r.get("elapsed_ms") or 0)
            by_id_cached_atk[r["instance_id"]] = float(r.get("cached_tokens") or 0)
        elif r["mode"] == "baseline_prefix_cache_only":
            by_id_stock[r["instance_id"]] = float(r.get("elapsed_ms") or 0)
            by_id_cached_stock[r["instance_id"]] = float(r.get("cached_tokens") or 0)
    common_ids = sorted(set(by_id_atk) & set(by_id_stock))
    latency_diffs = [by_id_stock[i] - by_id_atk[i] for i in common_ids]
    cached_diffs = [by_id_cached_atk[i] - by_id_cached_stock[i] for i in common_ids]
    ci_atk = bootstrap_paired(latency_diffs, n_resamples=args.n_resamples, seed=args.seed)
    ci_cached = bootstrap_paired(cached_diffs, n_resamples=args.n_resamples, seed=args.seed + 1)

    n_speedup = len({r["instance_id"] for r in speedup_rows})
    n_passrate = len({r["instance_id"] for r in passrate_rows})

    md = render_report(
        speedup_stats=speedup_stats,
        passrate_stats=passrate_stats,
        ci_atk=ci_atk,
        ci_cached=ci_cached,
        n_speedup=n_speedup,
        n_passrate=n_passrate,
        speedup_csv=args.speedup_csv,
        passrate_csv=args.passrate_csv,
    )
    (args.out_dir / "report.md").write_text(md, encoding="utf-8")
    print(f"wrote {args.out_dir / 'report.md'}")

    # Long-format merged table
    merged_path = args.out_dir / "merged_table.csv"
    fieldnames = [
        "source", "mode_legacy", "mode_display", "instance_id", "repo",
        "elapsed_ms", "cached_tokens", "exact_content_hit", "token_f1_vs_baseline",
        "pass_at_1", "diff_extracted", "apply_clean", "lossy_match_reason",
    ]
    with merged_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in speedup_rows:
            w.writerow({
                "source": "speedup",
                "mode_legacy": r.get("mode", ""),
                "mode_display": DISPLAY_MODE.get(r.get("mode", ""), r.get("mode", "")),
                "instance_id": r.get("instance_id", ""),
                "repo": r.get("repo", ""),
                "elapsed_ms": r.get("elapsed_ms", ""),
                "cached_tokens": r.get("cached_tokens", ""),
                "exact_content_hit": 1 if r.get("lossy_match_reason") == "exact_code_content_signature" else 0,
                "token_f1_vs_baseline": r.get("output_token_f1_vs_baseline", ""),
                "pass_at_1": "",
                "diff_extracted": "",
                "apply_clean": "",
                "lossy_match_reason": r.get("lossy_match_reason", ""),
            })
        for r in passrate_rows:
            w.writerow({
                "source": "passrate",
                "mode_legacy": r.get("mode", ""),
                "mode_display": PASSRATE_MODE.get(r.get("mode", ""), r.get("mode", "")),
                "instance_id": r.get("instance_id", ""),
                "repo": r.get("repo", ""),
                "elapsed_ms": r.get("elapsed_ms", ""),
                "cached_tokens": r.get("cached_tokens", ""),
                "exact_content_hit": 1 if r.get("match_reason") == "exact_code_content_signature" else 0,
                "token_f1_vs_baseline": "",
                "pass_at_1": r.get("pass1", ""),
                "diff_extracted": r.get("diff_extracted", ""),
                "apply_clean": r.get("apply_clean", ""),
                "lossy_match_reason": r.get("match_reason", ""),
            })
    print(f"wrote {merged_path}")

    # Machine-readable summary
    summary = {
        "speedup_csv": str(args.speedup_csv),
        "passrate_csv": str(args.passrate_csv),
        "n_speedup_cases": n_speedup,
        "n_passrate_cases": n_passrate,
        "speedup_per_mode": speedup_stats,
        "passrate_per_mode": passrate_stats,
        "ci_latency_stock_minus_agenttemplatekv": ci_atk,
        "ci_cached_agenttemplatekv_minus_stock": ci_cached,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.out_dir / 'summary.json'}")

    # Final stdout summary
    print(
        f"speedup cases={n_speedup} passrate cases={n_passrate} "
        f"latency_diff={ci_atk['mean']:+.0f}ms p={ci_atk['p']:.4f} "
        f"cached_diff={ci_cached['mean']:+.0f} p={ci_cached['p']:.4f}"
    )


if __name__ == "__main__":
    main()
