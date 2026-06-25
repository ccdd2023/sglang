# AgentTemplateKV TTFT P1 Rollup

## Summary

- Strongest result: 8k single-segment agent-scaling shows p50 TTFT speedups of 5.10x (2 agents) and 4.24x (3 agents) for `exact_reuse_plus_code_hints` vs `prefix_cache_only`.
- Output F1 is 1.0 across reported shards because max_tokens=1 and the benchmark is TTFT-dominant.
- Device-hit/anchor reuse is present in the strong 8k shards, but `consumed_count` remains a counter gap in many rows; report this as a metadata/implementation caveat, not as a full consumed-path closure.
- 16k and 32k are boundary results: exact-content hit remains 1.0, but anchor device-hit rate falls and TTFT speedup disappears.

## Main Groups

| shard | prefix p50/p90 | exact+hints p50/p90 | speedup p50/p90 | exact hit | device hit | status |
|---|---:|---:|---:|---:|---:|---|
| p1_l8000_a2_s1 | 334.7/421.4 | 65.7/445.4 | 5.10x/0.95x | 1.00 | 0.70 | {'anchor_reuse_device_hit_consumed_counter_gap': 7, 'protected_not_consumed:no_anchor_match': 3} |
| p1_l8000_a3_s1 | 331.9/493.8 | 78.3/447.6 | 4.24x/1.10x | 1.00 | 0.53 | {'anchor_reuse_device_hit_consumed_counter_gap': 8, 'protected_not_consumed:no_anchor_match': 7} |
| p1_l8000_a2_s2 | 639.5/673.8 | 613.9/651.4 | 1.04x/1.03x | 1.00 | 0.30 | {'anchor_reuse_device_hit_consumed_counter_gap': 3, 'protected_not_consumed:no_anchor_match': 7} |
| p1_l16000_a2_s1 | 940.1/1289.3 | 888.0/1188.7 | 1.06x/1.08x | 1.00 | 0.20 | {'anchor_reuse_device_hit_consumed_counter_gap': 2, 'protected_not_consumed:no_anchor_match': 8} |
| p1_l32000_a2_s1 | 1796.6/2013.9 | 1914.0/2168.8 | 0.94x/0.93x | 1.00 | 0.00 | {'protected_not_consumed:no_anchor_match': 10} |

## Paper Guidance

- Use 8k single-segment p50 TTFT as the positive micro/stress result.
- Do not claim robust long-context TTFT acceleration from this run; 16k/32k show an anchor-match stability limitation.
- Phrase fast-path metadata as: exact-content anchor reuse and device hits are observed, while the prefetch consumed counter is not fully closed for all rows.
