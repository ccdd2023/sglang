#!/usr/bin/env python3
"""FAIR-MEASUREMENT (B2): cross-config A/B analyzer for the giant-codebase bench.

The giant-codebase driver (``bench_giant_codebase_reuse.py``) runs ONE mode
per invocation (one server, one ``--mode``), so a fair A/B compares the
``rows.csv`` from a baseline run (``--mode prefix_cache_only``) against the
``rows.csv`` from an experimental run (``--mode placeholder_knn_reuse``).

This analyzer:

1. Loads two (or more) ``rows.csv`` files, keyed by a config label.
2. **Warmup-parity gate (B2):** for every ``(case_id, agent_id)``, compares
   ``radix_prefix_tokens = cached_tokens - codeaware_reused_tokens`` across
   configs. The radix L1 prefix is the ONLY part the ``prefix_cache_only``
   baseline also sees, so it MUST cancel (delta <= 15%) or the measured
   speedup is confounded with radix prefix warmth — the exact bug that
   invalidated the 1.448x result. On violation it writes
   ``PARITY_VIOLATIONS.txt`` and exits non-zero (unless ``--allow``).
3. **Decomposed speedup (B3):** speedup = baseline.p50_TTFT / exp.p50_TTFT
   over the reuser agents (agent 1 / "implementer" excluded as the source),
   with the radix / code-aware token split printed alongside so the
   attribution is honest.

Usage::

    python -m benchmark.multi_workflow.analyze_fair_ab \\
        --baseline results/.../prefix_cache_only/rows.csv \\
        --experimental results/.../placeholder_knn_reuse/rows.csv \\
        [--lossless results/.../lossless_full_prefill/rows.csv] \\
        [--l3-general results/.../l3_general/rows.csv] \\
        --out-dir results/.../fair_ab_report

The ``--lossless`` config (if given) is the accuracy reference: token-F1 of
each config vs lossless is reported (when the bench recorded an
``output_token_f1_vs_baseline`` or when raw output text sidecars are
available). The ``--l3-general`` config is the "general algorithm" baseline
for the user's accuracy bar (code-aware F1 >= general L3 F1).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# Reuse the bench's token-F1 helper so the offline A/B F1 matches the in-run
# definition exactly (bag-of-whitespace-tokens F1).
from benchmark.multi_workflow.bench_kvcomm_ttft_stress import token_f1


SOURCE_ROLE = "implementer"  # agent 1 — the source, not a reuse beneficiary


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_outputs(path: Path) -> dict[tuple[str, int], str]:
    """Load an outputs.jsonl sidecar into {(case_id, agent_idx): output_text}.

    The giant driver dumps one row per (case, agent) with keys case_id,
    task_index, agent_idx, role, output_text (bench_giant_codebase_reuse.py).
    Keying by (case_id, agent_idx) aligns a config's output against the
    lossless reference's output of the SAME (case, agent) for a real F1.
    """
    out: dict[tuple[str, int], str] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = rec.get("case_id", "")
            aidx = int(rec.get("agent_idx", 0) or 0)
            out[(cid, aidx)] = rec.get("output_text", "") or ""
    return out


def real_f1_map(
    config_outputs: dict[tuple[str, int], str],
    lossless_outputs: dict[tuple[str, int], str],
) -> dict[tuple[str, int], float]:
    """Per-(case,agent) token-F1 of config output vs lossless output."""
    f1s: dict[tuple[str, int], float] = {}
    for key, cfg_text in config_outputs.items():
        ref = lossless_outputs.get(key)
        if ref is None:
            continue
        f1s[key] = token_f1(cfg_text, ref)
    return f1s


def _f(row: dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key) or 0)
    except (TypeError, ValueError):
        return 0.0


def percentile(xs: list[float], pct: float) -> float:
    if not xs:
        return 0.0
    vals = sorted(xs)
    idx = min(len(vals) - 1, max(0, int(round((len(vals) - 1) * pct))))
    return vals[idx]


def safe_mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def reuser_rows(rows: list[dict[str, Any]], exclude_source: bool) -> list[dict[str, Any]]:
    if not exclude_source:
        return list(rows)
    out = [r for r in rows if r.get("agent_id") != SOURCE_ROLE]
    return out if out else list(rows)


def config_stats(
    rows: list[dict[str, Any]],
    exclude_source: bool,
    real_f1: dict[tuple[str, int], float] | None = None,
) -> dict[str, Any]:
    rr = reuser_rows(rows, exclude_source)
    ttft = [_f(r, "ttft_ms") for r in rr]
    # REAL accuracy (Step 3): if a real-F1 map (config vs lossless, from
    # outputs.jsonl) is available, use it — the CSV's output_token_f1_vs_baseline
    # defaults to 1.0 when no in-run baseline exists (single-mode runs), which
    # is a fake number. Fall back to the CSV column only if no real F1.
    if real_f1:
        f1_vals = [
            real_f1.get((r.get("case_id", ""), int(float(r.get("agent_idx", 0) or 0))))
            for r in rr
        ]
        f1_vals = [v for v in f1_vals if v is not None]
        f1_source = "real (vs lossless, from outputs.jsonl)"
    else:
        f1_vals = [
            _f(r, "output_token_f1_vs_baseline")
            for r in rr
            if r.get("output_token_f1_vs_baseline") not in (None, "", "None")
        ]
        f1_source = "csv default (1.0 if no in-run baseline — likely fake)"
    return {
        "n": len(rr),
        "p50_ttft": percentile(ttft, 0.5),
        "p90_ttft": percentile(ttft, 0.9),
        "avg_ttft": safe_mean(ttft),
        "avg_cached": safe_mean([_f(r, "cached_tokens") for r in rr]),
        "avg_radix_prefix": safe_mean([_f(r, "radix_prefix_tokens") for r in rr]),
        "avg_codeaware_reused": safe_mean([_f(r, "codeaware_reused_tokens") for r in rr]),
        "avg_l2_wholeslot_reused": safe_mean([_f(r, "l2_wholeslot_reused_tokens") for r in rr]),
        "avg_l3_offset_reused": safe_mean([_f(r, "l3_offset_reused_tokens") for r in rr]),
        "avg_c2_chunk_reused": safe_mean([_f(r, "c2_chunk_reused_tokens") for r in rr]),
        "avg_f1": safe_mean(f1_vals),
        "f1_source": f1_source,
        "f1_n": len(f1_vals),
        "rows": rr,
    }


def parity_check(
    baseline_rows: list[dict[str, Any]],
    exp_rows: list[dict[str, Any]],
    tol: float = 0.15,
    exclude_source: bool = True,
    abs_floor: int = 64,
) -> tuple[list[str], bool]:
    """Compare radix_prefix_tokens per (case_id, agent_id). Return
    (violation_messages, all_ok).

    The source agent (implementer) is excluded by default — it is excluded
    from the speedup average too, and its tiny cold-start radix (system
    prompt only) would otherwise fire false 100%% deltas. An absolute floor
    (``abs_floor`` tokens) further suppresses noise: when BOTH configs' radix
    prefix for a (case,agent) is below the floor, a relative delta is
    meaningless (it is just system-prompt/BOS jitter)."""
    def index(rows: list[dict[str, Any]]) -> dict[tuple[str, str], float]:
        out: dict[tuple[str, str], float] = {}
        for r in rows:
            if exclude_source and r.get("agent_id") == SOURCE_ROLE:
                continue
            key = (r.get("case_id", ""), r.get("agent_id", ""))
            out[key] = _f(r, "radix_prefix_tokens")
        return out

    bi, ei = index(baseline_rows), index(exp_rows)
    violations: list[str] = []
    for key in sorted(set(bi) | set(ei)):
        b, e = bi.get(key, 0.0), ei.get(key, 0.0)
        # Absolute floor: tiny radix prefixes (system prompt only) are noise.
        if max(abs(b), abs(e)) < abs_floor:
            continue
        denom = max(abs(b), abs(e), 1.0)
        delta = abs(e - b) / denom
        if delta > tol:
            violations.append(
                f"  case={key[0]} agent={key[1]}: baseline_radix={b:.0f} "
                f"exp_radix={e:.0f} delta={delta*100:.0f}% > {tol*100:.0f}%"
            )
    return violations, len(violations) == 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", type=Path, required=True, help="L2 whole-slot (general KVCOMM) rows.csv")
    ap.add_argument("--experimental", type=Path, required=True, help="L4+C2 (AST-chunked KVCOMM) rows.csv")
    ap.add_argument("--lossless", type=Path, default=None, help="lossless_full_prefill rows.csv (accuracy reference)")
    ap.add_argument("--l3-general", type=Path, default=None, help="L3 general MiniLM rows.csv (accuracy bar)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--include-source", action="store_true", help="Do not exclude agent 1 (implementer) from averages")
    ap.add_argument("--allow", action="store_true", help="Do not exit non-zero on parity violation")
    ap.add_argument("--tol", type=float, default=0.15, help="radix-prefix parity tolerance (fraction)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    exclude_source = not args.include_source

    configs: dict[str, list[dict[str, Any]]] = {
        "baseline(L2 whole-slot)": load_rows(args.baseline),
        "experimental(L4+C2 AST)": load_rows(args.experimental),
    }
    if args.lossless:
        configs["lossless(reference)"] = load_rows(args.lossless)
    if args.l3_general:
        configs["l3_general"] = load_rows(args.l3_general)

    # REAL accuracy (Step 3): load each config's outputs.jsonl (same dir as its
    # rows.csv) and compute token-F1 vs the lossless reference's outputs.jsonl.
    # This replaces the fake CSV F1 (1.0 default for single-mode runs).
    lossless_outputs: dict[tuple[str, int], str] = {}
    if args.lossless:
        lossless_outputs = load_outputs(args.lossless.parent / "outputs.jsonl")
    real_f1_by_cfg: dict[str, dict[tuple[str, int], float]] = {}
    if lossless_outputs:
        for name, rows in configs.items():
            if name == "lossless(reference)":
                continue
            csv_path = {
                "baseline(L2 whole-slot)": args.baseline,
                "experimental(L4+C2 AST)": args.experimental,
                "l3_general": args.l3_general,
            }.get(name)
            if csv_path is None:
                continue
            cfg_out = load_outputs(csv_path.parent / "outputs.jsonl")
            real_f1_by_cfg[name] = real_f1_map(cfg_out, lossless_outputs)

    stats = {
        name: config_stats(rows, exclude_source, real_f1=real_f1_by_cfg.get(name))
        for name, rows in configs.items()
    }

    lines: list[str] = ["# Fair A/B Report", ""]
    lines.append(f"- exclude_source_agent: {exclude_source} (source role = `{SOURCE_ROLE}`)")
    lines.append(f"- parity tolerance: {args.tol*100:.0f}%")
    lines.append("")

    BASE = "baseline(L2 whole-slot)"
    EXP = "experimental(L4+C2 AST)"

    # Decomposed per-config table.
    lines.append("## Per-config (reusers only)")
    lines.append("")
    lines.append("| config | n | p50_TTFT | avg_cached | avg_radix_prefix | avg_codeaware_reused | avg_l2 | avg_c2 | avg_F1 | F1_source |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for name, s in stats.items():
        lines.append(
            f"| {name} | {s['n']} | {s['p50_ttft']:.1f} | {s['avg_cached']:.0f} | "
            f"{s['avg_radix_prefix']:.0f} | {s['avg_codeaware_reused']:.0f} | "
            f"{s['avg_l2_wholeslot_reused']:.0f} | {s['avg_c2_chunk_reused']:.0f} | "
            f"{s['avg_f1']:.3f} | {s['f1_n']} |"
        )
    lines.append(f"- F1 source: {stats[EXP]['f1_source']}")
    lines.append("")

    # Parity gate.
    base_rows = configs[BASE]
    exp_rows = configs[EXP]
    violations, ok = parity_check(base_rows, exp_rows, tol=args.tol, exclude_source=exclude_source)
    lines.append("## Warmup-parity gate (B2)")
    lines.append("")
    base_radix = stats[BASE]["avg_radix_prefix"]
    exp_radix = stats[EXP]["avg_radix_prefix"]
    radix_delta = exp_radix - base_radix
    lines.append(
        f"- avg radix_prefix_tokens: baseline(L2)={base_radix:.0f}, experimental(L4+C2)={exp_radix:.0f}, "
        f"delta={radix_delta:.0f}"
    )
    if ok:
        lines.append(f"- **PARITY OK** — radix L1 prefix cancels ({len(violations)} per-agent violations).")
    else:
        lines.append(f"- **PARITY VIOLATION** — {len(violations)} per-agent (case,agent) pairs exceed {args.tol*100:.0f}%:")
        lines.append("")
        lines.append("```")
        lines.extend(violations[:50])
        if len(violations) > 50:
            lines.append(f"  ... ({len(violations)-50} more)")
        lines.append("```")
        lines.append("")
        lines.append("The measured speedup is CONFOUNDED with radix prefix warmth and is NOT a clean")
        lines.append("code-aware contribution. Fix the warmup/salt parity before trusting the speedup.")
    lines.append("")

    # Decomposed speedup.
    lines.append("## Speed bar (L4+C2 vs L2, radix-isolated)")
    lines.append("")
    b_p50 = stats[BASE]["p50_ttft"]
    e_p50 = stats[EXP]["p50_ttft"]
    speedup = (b_p50 / e_p50) if e_p50 > 0 else 0.0
    lines.append(f"- speedup = L2.p50 / L4+C2.p50 = {b_p50:.1f} / {e_p50:.1f} = **{speedup:.3f}x**")
    lines.append(f"- L2 avg_radix_prefix = {base_radix:.0f}; L4+C2 avg_radix_prefix = {exp_radix:.0f} (delta {radix_delta:.0f}, should be ~0)")
    lines.append(f"- L2 avg_codeaware_reused = {stats[BASE]['avg_codeaware_reused']:.0f} (l2_wholeslot={stats[BASE]['avg_l2_wholeslot_reused']:.0f})")
    lines.append(f"- L4+C2 avg_codeaware_reused = {stats[EXP]['avg_codeaware_reused']:.0f} (l2={stats[EXP]['avg_l2_wholeslot_reused']:.0f}, c2={stats[EXP]['avg_c2_chunk_reused']:.0f})")
    speed_verdict = "MET (AST not slower than whole-slot)" if speedup >= 1.0 else "NOT MET (AST slower than whole-slot)"
    lines.append(f"- speed bar (>=1.0x): **{speed_verdict}**")
    if not ok:
        lines.append("")
        lines.append(f"Speedup {speedup:.3f}x is NOT trustworthy (parity violated).")

    # Accuracy bar.
    lines.append("")
    lines.append("## Accuracy bar (L4+C2 vs L2, both vs lossless)")
    lines.append("")
    exp_f1 = stats[EXP]["avg_f1"]
    base_f1 = stats[BASE]["avg_f1"]
    lines.append(f"- L4+C2 avg_F1 vs lossless = {exp_f1:.3f}")
    lines.append(f"- L2   avg_F1 vs lossless = {base_f1:.3f}")
    acc_verdict = "MET (AST not less accurate than whole-slot)" if exp_f1 >= base_f1 else "NOT MET (AST less accurate than whole-slot)"
    lines.append(f"- accuracy bar (L4+C2 F1 >= L2 F1): **{acc_verdict}**")
    if args.l3_general:
        l3_f1 = stats["l3_general"]["avg_f1"]
        verdict = "MET" if exp_f1 >= l3_f1 else "NOT MET"
        lines.append(f"- (legacy) l3_general avg_F1 = {l3_f1:.3f}  ->  L4+C2 >= general MiniLM: **{verdict}** ({exp_f1:.3f} vs {l3_f1:.3f})")
    else:
        lines.append("- (legacy) l3_general config not provided; the bar above (vs L2) is the primary one.")

    report = args.out_dir / "FAIR_AB_REPORT.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[analyze_fair_ab] wrote {report}")
    print(f"[analyze_fair_ab] speedup={speedup:.3f}x radix_delta={radix_delta:.0f} parity={'OK' if ok else 'VIOLATION'}")

    if not ok and not args.allow:
        (args.out_dir / "PARITY_VIOLATIONS.txt").write_text("\n".join(violations) + "\n", encoding="utf-8")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
