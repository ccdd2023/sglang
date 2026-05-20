#!/usr/bin/env python3
# =============================================================================
# Ablation Experiment Result Analyzer
# 分析 run_dag_ablation.sh 生成的实验结果
# =============================================================================

import json
import glob
import os
import sys
from datetime import datetime

RESULT_DIR = "/home/comp/25480812/CodeMAS_Project/logs/kvflow-multi-workflow/results"

CONFIGS = {
    'hicache90k': {
        'label': 'LRU Baseline',
        'priority': False,
        'prefetch': False,
        'color': '#FF6B6B',
    },
    'priority_wb_only': {
        'label': 'Priority (no prefetch)',
        'priority': True,
        'prefetch': False,
        'color': '#4ECDC4',
    },
    'lru_wb_pf': {
        'label': 'LRU + Prefetch',
        'priority': False,
        'prefetch': True,
        'color': '#45B7D1',
    },
    'kvflow': {
        'label': 'KVFlow (full)',
        'priority': True,
        'prefetch': True,
        'color': '#96CEB4',
    },
}

def find_result(config, wf_count):
    """Find the most recent DAG result file for a given config and workflow count.
    DAG: 5 agents/workflow, so 4wf=20agents, 16wf=80agents
    Linear: 10 agents/workflow, so 4wf=40agents, 16wf=160agents
    """
    agents = wf_count * 5  # DAG agents per workflow
    patterns = [
        f"{RESULT_DIR}/mwf_{config}_*_{agents}agents*_{wf_count}wf*.json",
    ]
    for pattern in patterns:
        files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
        if files:
            return files[0]
    return None


def parse_result(filepath):
    """Parse a result JSON file."""
    try:
        with open(filepath) as f:
            data = json.load(f)
        agg = data.get('aggregate', {})
        return {
            'ttft_warmup': agg.get('warmup_ttft_avg_ms', 0),
            'ttft_stable': agg.get('stable_ttft_avg_ms', 0),
            'e2e_stable': agg.get('stable_e2e_avg_ms', 0),
            'hit_rate': agg.get('est_ttft_hit_rate', 0) * 100,
            'rounds': agg.get('num_rounds', 0),
            'total_agents': data.get('config', {}).get('total_agents', 0),
        }
    except Exception as e:
        print(f"  Error parsing {filepath}: {e}")
        return None


def analyze_ablation(wf_count, label):
    """Analyze ablation results for a given pressure level."""
    print(f"\n{'='*80}")
    print(f"ABLATION RESULTS: {label} ({wf_count} workflows)")
    print(f"{'='*80}")

    results = {}
    for cfg, info in CONFIGS.items():
        filepath = find_result(cfg, wf_count)
        if filepath:
            result = parse_result(filepath)
            if result:
                results[cfg] = {**info, **result, 'filepath': filepath}
                print(f"\n  {info['label']}: {filepath.split('/')[-1]}")
                print(f"    Warmup TTFT: {result['ttft_warmup']:.1f} ms")
                print(f"    Stable TTFT: {result['ttft_stable']:.1f} ms")
                print(f"    E2E Stable:  {result['e2e_stable']:.1f} ms")
                print(f"    Hit Rate:    {result['hit_rate']:.1f}%")
        else:
            print(f"\n  {info['label']}: NOT FOUND")

    if not results:
        print("  No results found!")
        return

    # Calculate speedups vs baseline
    baseline_ttft = results.get('hicache90k', {}).get('ttft_stable', 1)
    if baseline_ttft == 0:
        baseline_ttft = 1

    print(f"\n{'-'*80}")
    print(f"ABLATION ANALYSIS: Speedup vs hicache90k baseline")
    print(f"{'-'*80}")
    print(f"\n{'Config':<28} {'Stable TTFT':>12} {'Speedup':>8} {'vs Baseline':>12} {'Hit Rate':>10}")
    print(f"{'-'*80}")

    for cfg in ['hicache90k', 'priority_wb_only', 'lru_wb_pf', 'kvflow']:
        if cfg not in results:
            continue
        r = results[cfg]
        speedup = baseline_ttft / r['ttft_stable'] if r['ttft_stable'] > 0 else 0
        diff_pct = (baseline_ttft - r['ttft_stable']) / baseline_ttft * 100 if baseline_ttft > 0 else 0
        diff_str = f'+{diff_pct:.1f}%' if diff_pct >= 0 else f'{diff_pct:.1f}%'
        print(f"{r['label']:<28} {r['ttft_stable']:>10.1f}ms {speedup:>7.2f}x {diff_str:>12} {r['hit_rate']:>9.1f}%")

    # Component contribution analysis
    print(f"\n{'-'*80}")
    print(f"COMPONENT CONTRIBUTION ANALYSIS")
    print(f"{'-'*80}")

    l_ttft = results.get('hicache90k', {}).get('ttft_stable', 0)
    p_ttft = results.get('priority_wb_only', {}).get('ttft_stable', 0)
    pf_ttft = results.get('lru_wb_pf', {}).get('ttft_stable', 0)
    kv_ttft = results.get('kvflow', {}).get('ttft_stable', 0)

    if l_ttft and p_ttft:
        print(f"  Priority alone (LRU->Priority):     {l_ttft/p_ttft:.2f}x = {(l_ttft-p_ttft)/l_ttft*100:+.1f}%")
    if l_ttft and pf_ttft:
        print(f"  Prefetch alone (LRU->LRU+PF):       {l_ttft/pf_ttft:.2f}x = {(l_ttft-pf_ttft)/l_ttft*100:+.1f}%")
    if p_ttft and kv_ttft:
        print(f"  Prefetch over Priority (P->KVFlow): {p_ttft/kv_ttft:.2f}x = {(p_ttft-kv_ttft)/p_ttft*100:+.1f}%")
    if pf_ttft and kv_ttft:
        print(f"  Priority over LRU+PF (PF->KVFlow): {pf_ttft/kv_ttft:.2f}x = {(pf_ttft-kv_ttft)/pf_ttft*100:+.1f}%")
    if l_ttft and kv_ttft:
        print(f"  Combined (LRU->KVFlow):             {l_ttft/kv_ttft:.2f}x = {(l_ttft-kv_ttft)/l_ttft*100:+.1f}%")

    # Root cause diagnosis
    print(f"\n{'-'*80}")
    print(f"ROOT CAUSE DIAGNOSIS")
    print(f"{'-'*80}")

    winner = min(results.items(), key=lambda x: x[1]['ttft_stable']) if results else (None, None)
    winner_label = winner[0] if winner[0] else 'N/A'
    winner_ttft = winner[1]['ttft_stable'] if winner[1] else 0

    if 'hicache90k' not in results:
        print("  Cannot diagnose: baseline (hicache90k) not found")
    else:
        if winner_label == 'hicache90k':
            print(f"  Winner: hicache90k (LRU baseline)")
            print(f"  Diagnosis: Priority is HARMFUL in this scenario")
            if p_ttft and l_ttft and p_ttft < l_ttft:
                print(f"    - Priority alone reduces performance by {(l_ttft-p_ttft)/l_ttft*100:.1f}%")
            if pf_ttft and l_ttft and pf_ttft < l_ttft:
                print(f"    - Prefetch alone reduces performance by {(l_ttft-pf_ttft)/l_ttft*100:.1f}%")
        elif winner_label == 'kvflow':
            print(f"  Winner: kvflow (Priority + Prefetch)")
            print(f"  Diagnosis: Both Priority and Prefetch contribute positively")
        elif winner_label == 'priority_wb_only':
            print(f"  Winner: Priority only (no prefetch)")
            print(f"  Diagnosis: Priority helps, but Prefetch is ineffective or harmful")
        elif winner_label == 'lru_wb_pf':
            print(f"  Winner: LRU + Prefetch")
            print(f"  Diagnosis: Prefetch helps, but Priority is ineffective or harmful")

    return results


def generate_chart_data():
    """Generate Chart.js compatible data for ablation results."""
    print("\n" + "="*80)
    print("CHART DATA (for kvflow_experiment_charts.html)")
    print("="*80 + "\n")

    chart_configs = [
        ('low_pressure', 4, 'Low Pressure (4 workflows)'),
        ('high_pressure', 16, 'High Pressure (16 workflows)'),
    ]

    for key, wf, label in chart_configs:
        results = {}
        for cfg in CONFIGS:
            fp = find_result(cfg, wf)
            if fp:
                r = parse_result(fp)
                if r:
                    results[cfg] = r

        if not results:
            print(f"// {label}: No data yet")
            continue

        baseline = results.get('hicache90k', {}).get('ttft_stable', 1)
        ttft_data = [results.get(c, {}).get('ttft_stable', 0) for c in CONFIGS]
        hit_data = [results.get(c, {}).get('hit_rate', 0) for c in CONFIGS]

        print(f"// {label}")
        print(f"ablation_{key}_ttft: {ttft_data}")
        print(f"ablation_{key}_hits: {hit_data}")
        print()


def main():
    print(f"KVFlow Ablation Experiment Analyzer")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Result dir: {RESULT_DIR}")

    # Analyze low pressure
    analyze_ablation(4, "Low Pressure (4 workflows)")

    # Analyze high pressure
    analyze_ablation(16, "High Pressure (16 workflows)")

    # Generate chart data
    generate_chart_data()

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
