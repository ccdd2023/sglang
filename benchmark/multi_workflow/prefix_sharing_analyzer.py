#!/usr/bin/env python3
"""
prefix_sharing_analyzer.py

Analyze KVFlow benchmark results to quantify:
  1. Multi-tier sharing rates (Tier-0/1/2/3 breakdown)
  2. Priority vs LRU per-step speedup contribution
  3. Cross-workflow KV reuse effectiveness
  4. Stable-state vs warmup performance gap
  5. Theoretical vs measured KV savings

Usage:
  python prefix_sharing_analyzer.py \\
      --baseline /path/to/results/lru_nocache.json \\
      --lru-wb /path/to/results/lru_wb_only.json \\
      --lru-pf /path/to/results/lru_wb_pf.json \\
      --pri /path/to/results/priority_wb_only.json \\
      --kvflow /path/to/results/kvflow.json \\
      --output /path/to/results/sharing_analysis.html
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from pathlib import Path


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    name: str
    config_label: str
    num_workflows: int
    agents_per_workflow: int
    num_rounds: int
    warmup_rounds: int
    # Tier lengths
    tier0_tokens: int
    tier1_tokens: int
    tier2_tokens: int
    tier3_tokens: int
    # Aggregate metrics
    stable_ttft_ms: float
    stable_e2e_ms: float
    warmup_ttft_ms: float
    warmup_e2e_ms: float
    est_ttft_hit_rate: float
    stable_speedup_ttft: float
    stable_speedup_e2e: float
    # Per-step speedup
    per_step_stable_speedup_ttft: Dict[str, float]
    per_step_stable_speedup_e2e: Dict[str, float]
    # Round-by-round data
    round_data: Dict[str, Dict]
    theoretical_kv_saving_pct: float


def load_result(path: str, name: str) -> Optional[BenchmarkResult]:
    """Load and parse a single benchmark JSON result."""
    if not os.path.exists(path):
        print(f"  [WARN] File not found: {path}", file=sys.stderr)
        return None
    with open(path) as f:
        raw = json.load(f)

    cfg = raw.get("config", {})
    agg = raw.get("aggregate", {})
    steps = raw.get("per_step_stable_speedup", {})

    return BenchmarkResult(
        name=name,
        config_label=agg.get("label", name),
        num_workflows=cfg.get("num_workflows", 4),
        agents_per_workflow=cfg.get("agents_per_workflow", 5),
        num_rounds=cfg.get("num_rounds", 5),
        warmup_rounds=cfg.get("warmup_rounds", 1),
        tier0_tokens=agg.get("tier0_tokens", cfg.get("shared_p_len", 512)),
        tier1_tokens=agg.get("tier1_tokens", cfg.get("unique_p_len", 1024)),
        tier2_tokens=agg.get("tier2_tokens", cfg.get("tier2_len", 512)),
        tier3_tokens=agg.get("tier3_tokens", cfg.get("suffix_len", 64)),
        stable_ttft_ms=agg.get("stable_ttft_avg_ms", 0),
        stable_e2e_ms=agg.get("stable_e2e_avg_ms", 0),
        warmup_ttft_ms=agg.get("warmup_ttft_avg_ms", 0),
        warmup_e2e_ms=agg.get("warmup_e2e_avg_ms", 0),
        est_ttft_hit_rate=agg.get("est_ttft_hit_rate", 0),
        stable_speedup_ttft=agg.get("stable_speedup_ttft", 1.0),
        stable_speedup_e2e=agg.get("stable_speedup_e2e", 1.0),
        per_step_stable_speedup_ttft={
            k: v for k, v in (steps.get("ttft", {}).items())
        },
        per_step_stable_speedup_e2e={
            k: v for k, v in (steps.get("e2e", {}).items())
        },
        round_data=raw.get("round_summaries", {}),
        theoretical_kv_saving_pct=agg.get("theoretical_kv_saving_pct", 0),
    )


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def compute_tier_fraction(result: BenchmarkResult) -> Dict[str, float]:
    """Compute what fraction of total prefix each tier represents."""
    total = result.tier0_tokens + result.tier1_tokens + result.tier2_tokens
    if total == 0:
        return {"tier0": 0, "tier1": 0, "tier2": 0, "tier3": 0}
    t3 = result.tier3_tokens
    return {
        "tier0": result.tier0_tokens / total,
        "tier1": result.tier1_tokens / total,
        "tier2": result.tier2_tokens / total,
        "tier3": t3 / (total + t3) if (total + t3) > 0 else 0,
    }


def compute_component_contribution(
    baseline: BenchmarkResult,
    config: BenchmarkResult,
) -> Dict[str, float]:
    """Break down KVFlow's benefit into per-component contributions.

    Returns dict with keys:
      - write_back_ttft_pct: % improvement from write_back
      - prefetch_ttft_pct: % improvement from prefetch
      - priority_ttft_pct: % improvement from Priority strategy
    """
    def improvement(baseline_ms, config_ms):
        if baseline_ms <= 0 or config_ms <= 0:
            return 0.0
        return (baseline_ms - config_ms) / baseline_ms * 100

    return {
        "write_back_ttft_pct": improvement(baseline.stable_ttft_ms, config.stable_ttft_ms),
        "write_back_e2e_pct": improvement(baseline.stable_e2e_ms, config.stable_e2e_ms),
    }


def bar(val: float, width: int = 40) -> str:
    """Render a float as an ASCII bar."""
    filled = int(round(val * width))
    return "█" * filled + "░" * (width - filled)


def fmt_ms(v: float) -> str:
    if v <= 0:
        return "N/A"
    return f"{v:.1f}ms"


def fmt_pct(v: float) -> str:
    if v <= 0:
        return "N/A"
    return f"{v:.1f}%"


def fmt_speedup(v: float) -> str:
    if v <= 0:
        return "N/A"
    return f"{v:.2f}x"


# ---------------------------------------------------------------------------
# Text report generator
# ---------------------------------------------------------------------------

def print_report(results: Dict[str, BenchmarkResult]) -> None:
    """Print a plain-text analysis report to stdout."""
    print("=" * 80)
    print("KVFlow Prefix Sharing Analysis Report")
    print("=" * 80)

    # Sort by TTFT for comparison
    ordered = sorted(
        results.values(),
        key=lambda r: r.stable_ttft_ms if r.stable_ttft_ms > 0 else 999999
    )

    # ── 1. Summary table ──────────────────────────────────────────────────
    print("\n## 1. Summary: Stable-State Performance\n")
    header = f"{'Config':<22} {'Stable TTFT':>12} {'Stable E2E':>12} {'Hit Rate':>10} {'Speedup':>8}"
    print(header)
    print("-" * len(header))
    for r in ordered:
        hit = fmt_pct(r.est_ttft_hit_rate * 100)
        sp = fmt_speedup(r.stable_speedup_ttft)
        print(f"{r.name:<22} {fmt_ms(r.stable_ttft_ms):>12} {fmt_ms(r.stable_e2e_ms):>12} {hit:>10} {sp:>8}")

    # ── 2. Multi-tier sharing breakdown ───────────────────────────────────
    print("\n## 2. Multi-Tier Prefix Structure\n")
    if not ordered:
        print("  No data.")
    else:
        ref = ordered[0]
        total_prefix = ref.tier0_tokens + ref.tier1_tokens + ref.tier2_tokens
        print(f"  Total KV prefix per agent: {total_prefix} tokens")
        print(f"  Tier-0 (universal):       {ref.tier0_tokens:>6} tokens  ({100*ref.tier0_tokens/total_prefix:.0f}%) -- shared by ALL agents, ALL workflows")
        print(f"  Tier-1 (role-based):       {ref.tier1_tokens:>6} tokens  ({100*ref.tier1_tokens/total_prefix:.0f}%) -- shared by same role across workflows")
        print(f"  Tier-2 (workflow-specific):{ref.tier2_tokens:>6} tokens  ({100*ref.tier2_tokens/total_prefix:.0f}%) -- unique per workflow")
        print(f"  Tier-3 (dynamic suffix):   {ref.tier3_tokens:>6} tokens  -- always unique per request")
        print()
        # Theoretical saving
        if ref.theoretical_kv_saving_pct > 0:
            print(f"  Theoretical KV saving (cross-workflow sharing): {ref.theoretical_kv_saving_pct:.1f}%")
            print(f"    = Tier-0 (100% saved) + Tier-1 (role-sharing) contribution")
        else:
            # Fallback for old configs
            tier1_share = 100 * ref.tier1_tokens / total_prefix * (1 - 1 / min(ref.agents_per_workflow, 5))
            tier0_share = 100 * ref.tier0_tokens / total_prefix
            saving = tier0_share + tier1_share
            print(f"  Estimated KV saving (cross-workflow sharing): {saving:.1f}%")
            print(f"    = Tier-0 ({tier0_share:.1f}% fully shared) + Tier-1 (~{tier1_share:.1f}% role-shared)")

    # ── 3. Component breakdown ─────────────────────────────────────────────
    print("\n## 3. Component Ablation Analysis\n")
    # Identify available configs for ablation
    baseline = results.get("lru_nocache") or results.get("baseline")
    lru_wb = results.get("lru_wb_only")
    lru_pf = results.get("lru_wb_pf")
    pri_wb = results.get("priority_wb_only")
    kvflow = results.get("kvflow")

    def show_contribution(name: str, before: Optional[BenchmarkResult],
                           after: Optional[BenchmarkResult], component: str):
        if before is None or after is None:
            return
        if before.stable_ttft_ms <= 0 or after.stable_ttft_ms <= 0:
            return
        pct = (before.stable_ttft_ms - after.stable_ttft_ms) / before.stable_ttft_ms * 100
        e2e_pct = (before.stable_e2e_ms - after.stable_e2e_ms) / before.stable_e2e_ms * 100
        print(f"  {name}:")
        print(f"    TTFT improvement: {pct:+.1f}%  ({fmt_ms(before.stable_ttft_ms)} → {fmt_ms(after.stable_ttft_ms)})")
        print(f"    E2E improvement:  {e2e_pct:+.1f}%  ({fmt_ms(before.stable_e2e_ms)} → {fmt_ms(after.stable_e2e_ms)})")
        print(f"    -> {component}")

    show_contribution("write_back (LRU)", baseline, lru_wb,
                      "LRU + write_back isolates write_back contribution")
    show_contribution("prefetch (LRU+WB)", lru_wb, lru_pf,
                      "LRU + WB + prefetch isolates prefetch contribution")
    show_contribution("priority (no prefetch)", lru_wb, pri_wb,
                      "Priority + WB (no prefetch) isolates Priority contribution")
    show_contribution("KVFlow full", baseline, kvflow,
                      "Full KVFlow = write_back + prefetch + Priority")

    # Show bar chart
    if baseline and kvflow:
        if baseline.stable_ttft_ms > 0:
            kv_improvement = (baseline.stable_ttft_ms - kvflow.stable_ttft_ms) / baseline.stable_ttft_ms * 100
            print(f"\n  KVFlow TTFT improvement over no-cache baseline: {kv_improvement:+.1f}%")
            print(f"  {bar(max(0, kv_improvement) / 100)} {kv_improvement:.1f}%")

    # ── 4. Per-step speedup ────────────────────────────────────────────────
    print("\n## 4. Per-Step Stable-State Speedup\n")
    step_keys = set()
    for r in ordered:
        step_keys.update(r.per_step_stable_speedup_ttft.keys())

    if step_keys:
        sorted_steps = sorted(step_keys, key=lambda k: int(k.split("_")[1]) if "_" in k else 0)
        header = f"{'Step':<10}"
        for r in ordered[:5]:  # Show top 5 configs
            header += f" {r.name[:10]:>10}"
        print(header)
        print("-" * len(header))
        for step in sorted_steps[:10]:
            row = f"{step:<10}"
            for r in ordered[:5]:
                v = r.per_step_stable_speedup_ttft.get(step, 0)
                row += f" {fmt_speedup(v):>10}"
            print(row)
    else:
        print("  (No per-step data available)")

    # ── 5. Cache hit rate comparison ──────────────────────────────────────
    print("\n## 5. Estimated Cache Hit Rate\n")
    for r in ordered:
        rate = r.est_ttft_hit_rate
        if rate > 0:
            print(f"  {r.name:<22}: {rate:.1%} {bar(rate)}")
        else:
            print(f"  {r.name:<22}: N/A")

    # ── 6. Warmup vs Stable-State ─────────────────────────────────────────
    print("\n## 6. Warmup vs Stable-State TTFT Gap\n")
    for r in ordered:
        if r.warmup_ttft_ms > 0 and r.stable_ttft_ms > 0:
            gap = r.warmup_ttft_ms - r.stable_ttft_ms
            gap_pct = gap / r.warmup_ttft_ms * 100
            print(f"  {r.name:<22}: warmup={fmt_ms(r.warmup_ttft_ms)}  "
                  f"stable={fmt_ms(r.stable_ttft_ms)}  "
                  f"gap={gap:.1f}ms ({gap_pct:.0f}%)")

    print("\n" + "=" * 80)


# ---------------------------------------------------------------------------
# HTML report generator
# ---------------------------------------------------------------------------

def generate_html_report(results: Dict[str, BenchmarkResult], output_path: str) -> None:
    """Generate a self-contained HTML analysis report."""
    ordered = sorted(
        results.values(),
        key=lambda r: r.stable_ttft_ms if r.stable_ttft_ms > 0 else 999999
    )

    ref = ordered[0] if ordered else None
    total_prefix = (ref.tier0_tokens + ref.tier1_tokens + ref.tier2_tokens) if ref else 0

    # Build tier data for pie chart
    tier_data = ""
    if ref:
        tiers = [
            ("Tier-0 Universal", ref.tier0_tokens, "#4CAF50"),
            ("Tier-1 Role-Based", ref.tier1_tokens, "#2196F3"),
            ("Tier-2 Workflow", ref.tier2_tokens, "#FF9800"),
            ("Tier-3 Dynamic", ref.tier3_tokens, "#9E9E9E"),
        ]
        tier_data = ",".join(
            f'{{label:"{l}",value:{v},color:"{c}"}}'
            for l, v, c in tiers if v > 0
        )

    # Build comparison table rows
    rows = ""
    for r in ordered:
        rows += f"""<tr>
  <td>{r.name}</td>
  <td>{fmt_ms(r.stable_ttft_ms)}</td>
  <td>{fmt_ms(r.stable_e2e_ms)}</td>
  <td>{fmt_pct(r.est_ttft_hit_rate * 100)}</td>
  <td>{fmt_speedup(r.stable_speedup_ttft)}</td>
  <td>{fmt_pct(r.theoretical_kv_saving_pct)}</td>
</tr>"""

    # Component breakdown
    comp_rows = ""
    configs_pairs = [
        ("write_back", "lru_nocache", "lru_wb_only"),
        ("prefetch", "lru_wb_only", "lru_wb_pf"),
        ("priority", "lru_wb_only", "priority_wb_only"),
        ("kvflow_full", "lru_nocache", "kvflow"),
    ]
    for label, before_key, after_key in configs_pairs:
        b = results.get(before_key)
        a = results.get(after_key)
        if b and a and b.stable_ttft_ms > 0:
            ttft_imp = (b.stable_ttft_ms - a.stable_ttft_ms) / b.stable_ttft_ms * 100
            e2e_imp = (b.stable_e2e_ms - a.stable_e2e_ms) / b.stable_e2e_ms * 100
            comp_rows += f"""<tr>
  <td>{label}</td>
  <td>{ttft_imp:+.1f}%</td>
  <td>{fmt_ms(b.stable_ttft_ms)} → {fmt_ms(a.stable_ttft_ms)}</td>
  <td>{e2e_imp:+.1f}%</td>
  <td>{fmt_ms(b.stable_e2e_ms)} → {fmt_ms(a.stable_e2e_ms)}</td>
</tr>"""

    # Per-step data
    step_headers = list(ordered[:6])
    step_cols = ""
    for r in step_headers:
        step_cols += f"<th>{r.name}</th>"

    step_keys = set()
    for r in ordered:
        step_keys.update(r.per_step_stable_speedup_ttft.keys())
    sorted_steps = sorted(step_keys, key=lambda k: int(k.split("_")[1]) if "_" in k else 0)

    step_rows = ""
    for step in sorted_steps[:15]:
        step_rows += "<tr>"
        step_rows += f'<td class="step-name">{step}</td>'
        for r in step_headers:
            v = r.per_step_stable_speedup_ttft.get(step, 0)
            color = "#4CAF50" if v > 1.5 else "#FF9800" if v > 1.0 else "#666"
            step_rows += f'<td style="color:{color}">{fmt_speedup(v)}</td>'
        step_rows += "</tr>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>KVFlow Prefix Sharing Analysis</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 40px; background: #fafafa; color: #333; }}
  h1 {{ color: #1a1a2e; border-bottom: 3px solid #2196F3; padding-bottom: 8px; }}
  h2 {{ color: #16213e; margin-top: 2em; }}
  h3 {{ color: #0f3460; }}
  .card {{ background: white; border-radius: 8px; padding: 20px; margin: 16px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
  th {{ background: #1a1a2e; color: white; padding: 10px; text-align: left; }}
  td {{ padding: 8px 10px; border-bottom: 1px solid #eee; }}
  tr:hover {{ background: #f5f5f5; }}
  .metric {{ font-weight: bold; color: #2196F3; }}
  .speedup {{ font-weight: bold; }}
  .positive {{ color: #4CAF50; }}
  .negative {{ color: #f44336; }}
  .tag {{ display: inline-block; background: #e3f2fd; color: #1565C0; border-radius: 4px; padding: 2px 8px; font-size: 0.85em; margin: 2px; }}
  .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin: 16px 0; }}
  .summary-item {{ background: white; border-radius: 8px; padding: 16px; box-shadow: 0 1px 4px rgba(0,0,0,0.06); text-align: center; }}
  .summary-value {{ font-size: 2em; font-weight: bold; color: #2196F3; }}
  .summary-label {{ color: #666; margin-top: 4px; font-size: 0.9em; }}
  .bar-cell {{ min-width: 120px; }}
  .step-name {{ font-family: monospace; color: #555; }}
  .info-box {{ background: #e8f5e9; border-left: 4px solid #4CAF50; padding: 12px 16px; margin: 12px 0; border-radius: 0 4px 4px 0; }}
  .warn-box {{ background: #fff3e0; border-left: 4px solid #FF9800; padding: 12px 16px; margin: 12px 0; border-radius: 0 4px 4px 0; }}
  .config-tag {{ display: inline-block; background: #1a1a2e; color: white; border-radius: 12px; padding: 2px 10px; font-size: 0.8em; margin: 2px; }}
  .improvement {{ font-weight: bold; }}
  .improvement.good {{ color: #4CAF50; }}
  .improvement.bad {{ color: #f44336; }}
  footer {{ margin-top: 40px; color: #999; font-size: 0.85em; text-align: center; }}
</style>
</head>
<body>

<h1>KVFlow Prefix Sharing Analysis</h1>
<p>Generated automatically from benchmark JSON results</p>

<!-- Summary KPIs -->
<div class="summary-grid">
  <div class="summary-item">
    <div class="summary-value">{len(ordered)}</div>
    <div class="summary-label">Configurations Analyzed</div>
  </div>
  <div class="summary-item">
    <div class="summary-value">{ref.num_workflows if ref else '?'}x{ref.agents_per_workflow if ref else '?'}</div>
    <div class="summary-label">Workflows x Agents</div>
  </div>
  <div class="summary-item">
    <div class="summary-value">{total_prefix}</div>
    <div class="summary-label">Tokens per Agent Prefix</div>
  </div>
  <div class="summary-item">
    <div class="summary-value">{fmt_pct(ref.theoretical_kv_saving_pct) if ref else 'N/A'}</div>
    <div class="summary-label">Theoretical KV Saving</div>
  </div>
</div>

<!-- 1. Stable-state performance table -->
<div class="card">
<h2>1. Stable-State Performance Comparison</h2>
<p class="info-box">
  Stable-state TTFT = average TTFT from rounds 1+ (excludes warmup round 0).
  This is the true steady-state latency after cache is populated.
</p>
<table>
  <thead>
    <tr>
      <th>Config</th>
      <th>Stable TTFT</th>
      <th>Stable E2E</th>
      <th>Est. Cache Hit Rate</th>
      <th>Speedup vs Warmup</th>
      <th>Theoretical KV Saving</th>
    </tr>
  </thead>
  <tbody>
    {rows}
  </tbody>
</table>
</div>

<!-- 2. Multi-tier structure -->
<div class="card">
<h2>2. Multi-Tier Prefix Sharing Structure</h2>
<div class="info-box">
  <strong>Why multi-tier sharing works:</strong> In real MAS coder workflows,
  agents share prefixes at different granularities:
  <ul style="margin: 8px 0">
    <li><strong>Tier-0 (universal):</strong> System instructions shared by ALL agents → 100% KV reuse</li>
    <li><strong>Tier-1 (role-based):</strong> Code imports/signatures shared by same role (e.g. IMPLEMENTER) across workflows → ~100% reuse for same role</li>
    <li><strong>Tier-2 (workflow):</strong> Task context unique to each workflow → no cross-workflow sharing, but Priority helps</li>
    <li><strong>Tier-3 (dynamic):</strong> Per-request suffix → always unique, no sharing possible</li>
  </ul>
</div>
<table>
  <thead>
    <tr><th>Tier</th><th>Tokens</th><th>% of Prefix</th><th>Sharing Pattern</th><th>KV Reuse</th></tr>
  </thead>
  <tbody>
    <tr>
      <td><span class="tag">Tier-0</span> Universal</td>
      <td>{ref.tier0_tokens if ref else 0}</td>
      <td>{100*ref.tier0_tokens/total_prefix:.0f}% if ref else 0%</td>
      <td>ALL agents, ALL workflows</td>
      <td class="improvement good">100% reuse</td>
    </tr>
    <tr>
      <td><span class="tag">Tier-1</span> Role-Based</td>
      <td>{ref.tier1_tokens if ref else 0}</td>
      <td>{100*ref.tier1_tokens/total_prefix:.0f}% if ref else 0%</td>
      <td>Same role across workflows (e.g. all IMPLEMENTERs)</td>
      <td class="improvement good">~100% reuse (same role)</td>
    </tr>
    <tr>
      <td><span class="tag">Tier-2</span> Workflow-Specific</td>
      <td>{ref.tier2_tokens if ref else 0}</td>
      <td>{100*ref.tier2_tokens/total_prefix:.0f}% if ref else 0%</td>
      <td>Unique per workflow</td>
      <td class="improvement">0% cross-workflow</td>
    </tr>
    <tr>
      <td><span class="tag">Tier-3</span> Dynamic</td>
      <td>{ref.tier3_tokens if ref else 0}</td>
      <td>per-request</td>
      <td>Always unique per request</td>
      <td class="improvement bad">0% (unavoidable)</td>
    </tr>
  </tbody>
</table>
<p><strong>Net theoretical KV saving:</strong> {fmt_pct(ref.theoretical_kv_saving_pct) if ref else 'N/A'}
(Saving = Tier-0 fully saved + Tier-1 role-sharing savings)</p>
</div>

<!-- 3. Component ablation -->
<div class="card">
<h2>3. Component Ablation Breakdown</h2>
<p class="info-box">
  Each row shows the isolated contribution of one KVFlow component,
  computed by comparing two configs that differ by exactly that component.
</p>
<table>
  <thead>
    <tr><th>Component</th><th>TTFT Improvement</th><th>TTFT (before→after)</th><th>E2E Improvement</th><th>E2E (before→after)</th></tr>
  </thead>
  <tbody>
    {comp_rows if comp_rows else '<tr><td colspan="5" style="text-align:center;color:#999">No ablation data available. Run ablation experiments first.</td></tr>'}
  </tbody>
</table>
</div>

<!-- 4. Per-step speedup -->
<div class="card">
<h2>4. Per-Step Stable-State Speedup</h2>
<p>Speedup = stable_ttft / warmup_ttft for each step. Steps with high speedup
benefit most from KVFlow (likely Tier-0 and Tier-1 prefixes with cache hits).</p>
<table>
  <thead>
    <tr><th>Step</th>{step_cols}</tr>
  </thead>
  <tbody>
    {step_rows if step_rows else '<tr><td colspan="99" style="text-align:center;color:#999">No per-step data available</td></tr>'}
  </tbody>
</table>
</div>

<!-- 5. Cache hit rate -->
<div class="card">
<h2>5. Estimated Cache Hit Rate</h2>
<p>Estimated from: <code>est_hit_rate = 1 - stable_ttft / warmup_ttft</code>.
A higher hit rate means more KV compute was avoided through caching.</p>
<table>
  <thead>
    <tr><th>Config</th><th>Hit Rate</th><th>Visual</th></tr>
  </thead>
  <tbody>
    {''.join(f'''<tr>
  <td>{r.name}</td>
  <td>{fmt_pct(r.est_ttft_hit_rate * 100)}</td>
  <td class="bar-cell">{bar(r.est_ttft_hit_rate) if r.est_ttft_hit_rate > 0 else 'N/A'}</td>
</tr>''' for r in ordered if r.est_ttft_hit_rate > 0)}
  </tbody>
</table>
</div>

<footer>
KVFlow Prefix Sharing Analyzer &mdash; Auto-generated<br>
Configs: {', '.join(r.name for r in ordered)}
</footer>

</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nHTML report written to: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze KVFlow benchmark results for prefix sharing analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--baseline", type=str, default=None,
                        help="Path to lru_nocache (no-cache baseline) JSON")
    parser.add_argument("--lru-wb", type=str, default=None,
                        help="Path to lru_wb_only JSON")
    parser.add_argument("--lru-pf", type=str, default=None,
                        help="Path to lru_wb_pf JSON")
    parser.add_argument("--pri", type=str, default=None,
                        help="Path to priority_wb_only JSON")
    parser.add_argument("--kvflow", type=str, default=None,
                        help="Path to kvflow (full) JSON")
    parser.add_argument("--configs", type=str, nargs="+", default=[],
                        help="Additional config JSON files: NAME=path pairs")
    parser.add_argument("--output", type=str, default="prefix_sharing_analysis.html",
                        help="Output HTML path (default: prefix_sharing_analysis.html)")
    parser.add_argument("--text-only", action="store_true",
                        help="Only print text report, skip HTML generation")
    args = parser.parse_args()

    results: Dict[str, BenchmarkResult] = {}

    def add_result(name: str, path: str):
        r = load_result(path, name)
        if r:
            results[name] = r

    if args.baseline:
        add_result("lru_nocache", args.baseline)
    if args.lru_wb:
        add_result("lru_wb_only", args.lru_wb)
    if args.lru_pf:
        add_result("lru_wb_pf", args.lru_pf)
    if args.pri:
        add_result("priority_wb_only", args.pri)
    if args.kvflow:
        add_result("kvflow", args.kvflow)

    for item in args.configs:
        if "=" in item:
            name, path = item.split("=", 1)
            add_result(name, path)

    if not results:
        print("Error: No result files provided. Use --baseline, --lru-wb, --lru-pf, --pri, --kvflow, or --configs.", file=sys.stderr)
        sys.exit(1)

    print_report(results)

    if not args.text_only:
        generate_html_report(results, args.output)


if __name__ == "__main__":
    main()
