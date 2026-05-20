#!/usr/bin/env python3
# =============================================================================
# vLLM vs SGLang vs KVFlow Triple Comparison Result Analyzer
# 分析 run_pipeline_triple_compare.sh 生成的实验结果
# =============================================================================

import json
import glob
import os
import sys
from datetime import datetime

RESULT_DIR = "/home/comp/25480812/CodeMAS_Project/logs/kvflow-multi-workflow/results"

CONFIGS = {
    'vllm_triple': {
        'label': 'vLLM (LRU + CPU offload)',
        'server': 'vLLM',
        'eviction': 'LRU',
        'hicache': False,
        'priority': False,
        'prefetch': False,
        'color': '#FF6B6B',
    },
    'sglang_triple': {
        'label': 'SGLang (LRU, no HiCache)',
        'server': 'SGLang',
        'eviction': 'LRU',
        'hicache': False,
        'priority': False,
        'prefetch': False,
        'color': '#4ECDC4',
    },
    'sglang_hicache_triple': {
        'label': 'SGLang (LRU + HiCache)',
        'server': 'SGLang',
        'eviction': 'LRU',
        'hicache': True,
        'priority': False,
        'prefetch': False,
        'color': '#45B7D1',
    },
    'kvflow_triple': {
        'label': 'KVFlow (Priority + HiCache)',
        'server': 'SGLang',
        'eviction': 'Priority',
        'hicache': True,
        'priority': True,
        'prefetch': True,
        'color': '#96CEB4',
    },
}


def find_result(config):
    """Find the most recent result file for a given config."""
    patterns = [
        f"{RESULT_DIR}/mwf_{config}_*.json",
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
            'cache_hits': agg.get('cache_hits', 0),
            'cache_misses': agg.get('cache_misses', 0),
            'total_prefills': agg.get('total_prefills', 0),
            'round_data': agg.get('round_data', []),
        }
    except Exception as e:
        print(f"  Error parsing {filepath}: {e}")
        return None


def analyze_triple_compare():
    """Analyze all triple comparison results."""
    print("=" * 80)
    print("vLLM vs SGLang vs KVFlow TRIPLE COMPARISON ANALYSIS")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Result dir: {RESULT_DIR}")
    print("=" * 80)

    results = {}
    for cfg, info in CONFIGS.items():
        filepath = find_result(cfg)
        if filepath:
            result = parse_result(filepath)
            if result:
                results[cfg] = {**info, **result, 'filepath': filepath}
                print(f"\n  {info['label']}: {filepath.split('/')[-1]}")
                print(f"    Warmup TTFT: {result['ttft_warmup']:.1f} ms")
                print(f"    Stable TTFT: {result['ttft_stable']:.1f} ms")
                print(f"    E2E Stable:  {result['e2e_stable']:.1f} ms")
                print(f"    Hit Rate:    {result['hit_rate']:.1f}%")
                if result.get('cache_hits', 0) + result.get('cache_misses', 0) > 0:
                    total = result['cache_hits'] + result['cache_misses']
                    print(f"    Cache: {result['cache_hits']}/{total} hits ({result['cache_hits']/total*100:.1f}%)")
        else:
            print(f"\n  {info['label']}: NOT FOUND")

    if not results:
        print("\n  No results found!")
        return

    # Calculate speedups
    baseline_ttft = results.get('vllm_triple', {}).get('ttft_stable', 1)
    if baseline_ttft == 0:
        baseline_ttft = 1

    print(f"\n" + "-" * 80)
    print("SPEEDUP ANALYSIS (vs vLLM baseline)")
    print(f"-" * 80)
    print(f"\n{'Config':<35} {'TTFT':>10} {'vs vLLM':>12} {'Speedup':>8} {'Hit Rate':>10}")
    print(f"{'-'*80}")

    for cfg in ['vllm_triple', 'sglang_triple', 'sglang_hicache_triple', 'kvflow_triple']:
        if cfg not in results:
            continue
        r = results[cfg]
        ttft = r['ttft_stable']
        speedup = baseline_ttft / ttft if ttft > 0 else 0
        diff_pct = (baseline_ttft - ttft) / baseline_ttft * 100 if baseline_ttft > 0 else 0
        diff_str = f'+{diff_pct:.1f}%' if diff_pct >= 0 else f'{diff_pct:.1f}%'
        print(f"{r['label']:<35} {ttft:>8.1f}ms {diff_str:>12} {speedup:>7.2f}x {r['hit_rate']:>9.1f}%")

    # Component contribution analysis
    print(f"\n" + "-" * 80)
    print("COMPONENT CONTRIBUTION ANALYSIS")
    print(f"-" * 80)

    vllm_ttft = results.get('vllm_triple', {}).get('ttft_stable', 0)
    sglang_ttft = results.get('sglang_triple', {}).get('ttft_stable', 0)
    sglang_hc_ttft = results.get('sglang_hicache_triple', {}).get('ttft_stable', 0)
    kvflow_ttft = results.get('kvflow_triple', {}).get('ttft_stable', 0)

    if vllm_ttft and sglang_ttft:
        print(f"  SGLang vs vLLM (LRU baseline):     {vllm_ttft/sglang_ttft:.2f}x = {(vllm_ttft-sglang_ttft)/vllm_ttft*100:+.1f}%")

    if sglang_ttft and sglang_hc_ttft:
        print(f"  HiCache contribution (SGLang):     {sglang_ttft/sglang_hc_ttft:.2f}x = {(sglang_ttft-sglang_hc_ttft)/sglang_ttft*100:+.1f}%")

    if sglang_hc_ttft and kvflow_ttft:
        print(f"  Priority contribution (HiCache):   {sglang_hc_ttft/kvflow_ttft:.2f}x = {(sglang_hc_ttft-kvflow_ttft)/sglang_hc_ttft*100:+.1f}%")

    if vllm_ttft and kvflow_ttft:
        print(f"  KVFlow vs vLLM (total):            {vllm_ttft/kvflow_ttft:.2f}x = {(vllm_ttft-kvflow_ttft)/vllm_ttft*100:+.1f}%")

    # Winner diagnosis
    print(f"\n" + "-" * 80)
    print("WINNER ANALYSIS")
    print(f"-" * 80)

    winner = min(results.items(), key=lambda x: x[1]['ttft_stable']) if results else (None, None)
    if winner[0]:
        print(f"  Winner: {winner[1]['label']}")
        print(f"  Stable TTFT: {winner[1]['ttft_stable']:.1f} ms")
        print(f"  Hit Rate: {winner[1]['hit_rate']:.1f}%")

    # Round-by-round analysis
    print(f"\n" + "-" * 80)
    print("ROUND-BY-ROUND ANALYSIS")
    print(f"-" * 80)

    for cfg in ['vllm_triple', 'kvflow_triple']:
        if cfg not in results:
            continue
        r = results[cfg]
        round_data = r.get('round_data', [])
        if round_data:
            print(f"\n  {r['label']}:")
            for i, rd in enumerate(round_data):
                if i == 0:
                    label = "Warmup"
                else:
                    label = f"Round {i}"
                ttft = rd.get('ttft_avg_ms', 0)
                hits = rd.get('cache_hits', 0)
                total = rd.get('cache_hits', 0) + rd.get('cache_misses', 0)
                hit_rate = hits/total*100 if total > 0 else 0
                print(f"    {label:>8}: TTFT={ttft:>8.1f}ms, Hit Rate={hit_rate:>5.1f}%")

    return results


def generate_chart_data():
    """Generate Chart.js compatible data for triple comparison results."""
    print("\n" + "=" * 80)
    print("CHART DATA (for HTML visualization)")
    print("=" * 80 + "\n")

    chart_configs = ['vllm_triple', 'sglang_triple', 'sglang_hicache_triple', 'kvflow_triple']
    labels = ["vLLM", "SGLang", "SGLang+HiCache", "KVFlow"]
    ttft_data = []
    hit_data = []

    for cfg in chart_configs:
        fp = find_result(cfg)
        if fp:
            r = parse_result(fp)
            if r:
                ttft_data.append(r.get('ttft_stable', 0))
                hit_data.append(r.get('hit_rate', 0))
            else:
                ttft_data.append(0)
                hit_data.append(0)
        else:
            ttft_data.append(0)
            hit_data.append(0)

    print(f"chart_labels: {labels}")
    print(f"ttft_data: {ttft_data}")
    print(f"hit_data: {hit_data}")
    print()

    # Calculate speedup data
    if ttft_data[0] > 0:
        speedup_data = [ttft_data[0] / x if x > 0 else 0 for x in ttft_data]
        print(f"speedup_data: {speedup_data}")


def main():
    # Analyze results
    results = analyze_triple_compare()

    # Generate chart data
    generate_chart_data()

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
