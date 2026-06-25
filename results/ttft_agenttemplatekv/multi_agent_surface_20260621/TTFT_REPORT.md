# KVCOMM TTFT Stress Report

This run is prefill/TTFT dominant and is intended to complement the realistic 100-case E2E table.

## Output Schema

`ttft_ms`, `e2e_ms`, `prompt_tokens`, `cached_tokens`, `cached_ratio`, `exact_hit`, `match_reason`, `matched_content_signature`, `output_exact_match_vs_baseline`, `output_token_f1_vs_baseline`, `mode`, `case_id`, `agent_id`, `segment_count`, `max_file_chars`.

## Row Counts

- agent_scaling: 60
- agent_scaling_workflow: 20

## Summary Groups

- `agent_scaling_workflow|exact_reuse_no_hints|8000|1|1|1`: n=1, avg TTFT=105.7 ms, p50=105.7 ms, p90=105.7 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_no_hints|8000|1|2|1`: n=1, avg TTFT=202.0 ms, p50=202.0 ms, p90=202.0 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_no_hints|8000|1|3|1`: n=1, avg TTFT=672.1 ms, p50=672.1 ms, p90=672.1 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_no_hints|8000|1|4|1`: n=1, avg TTFT=1910.1 ms, p50=1910.1 ms, p90=1910.1 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_no_hints|8000|1|5|1`: n=1, avg TTFT=1862.8 ms, p50=1862.8 ms, p90=1862.8 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_plus_code_hints|8000|1|1|1`: n=1, avg TTFT=82.6 ms, p50=82.6 ms, p90=82.6 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_plus_code_hints|8000|1|2|1`: n=1, avg TTFT=220.0 ms, p50=220.0 ms, p90=220.0 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_plus_code_hints|8000|1|3|1`: n=1, avg TTFT=1185.5 ms, p50=1185.5 ms, p90=1185.5 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_plus_code_hints|8000|1|4|1`: n=1, avg TTFT=1675.2 ms, p50=1675.2 ms, p90=1675.2 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_plus_code_hints|8000|1|5|1`: n=1, avg TTFT=3034.9 ms, p50=3034.9 ms, p90=3034.9 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|hints_no_exact|8000|1|1|1`: n=1, avg TTFT=235.1 ms, p50=235.1 ms, p90=235.1 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|hints_no_exact|8000|1|2|1`: n=1, avg TTFT=513.6 ms, p50=513.6 ms, p90=513.6 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|hints_no_exact|8000|1|3|1`: n=1, avg TTFT=771.6 ms, p50=771.6 ms, p90=771.6 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|hints_no_exact|8000|1|4|1`: n=1, avg TTFT=1030.2 ms, p50=1030.2 ms, p90=1030.2 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|hints_no_exact|8000|1|5|1`: n=1, avg TTFT=1295.7 ms, p50=1295.7 ms, p90=1295.7 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|prefix_cache_only|8000|1|1|1`: n=1, avg TTFT=263.5 ms, p50=263.5 ms, p90=263.5 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|prefix_cache_only|8000|1|2|1`: n=1, avg TTFT=300.7 ms, p50=300.7 ms, p90=300.7 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|prefix_cache_only|8000|1|3|1`: n=1, avg TTFT=346.2 ms, p50=346.2 ms, p90=346.2 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|prefix_cache_only|8000|1|4|1`: n=1, avg TTFT=972.1 ms, p50=972.1 ms, p90=972.1 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|prefix_cache_only|8000|1|5|1`: n=1, avg TTFT=1210.4 ms, p50=1210.4 ms, p90=1210.4 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling|exact_reuse_no_hints|8000|1|1|1`: n=1, avg TTFT=105.7 ms, p50=105.7 ms, p90=105.7 ms, exact hit=1.00, device hit=1.00, consumed=1.00, protected=0.0, F1=1.0000, status={'consumed': 1}
- `agent_scaling|exact_reuse_no_hints|8000|1|2|1`: n=2, avg TTFT=101.0 ms, p50=92.7 ms, p90=109.3 ms, exact hit=1.00, device hit=1.00, consumed=0.50, protected=0.0, F1=1.0000, status={'consumed': 1, 'device_hit_without_consumed': 1}
- `agent_scaling|exact_reuse_no_hints|8000|1|3|1`: n=3, avg TTFT=224.0 ms, p50=103.6 ms, p90=469.2 ms, exact hit=1.00, device hit=0.67, consumed=0.33, protected=0.0, F1=1.0000, status={'consumed': 1, 'no_fast_path': 1, 'device_hit_without_consumed': 1}
- `agent_scaling|exact_reuse_no_hints|8000|1|4|1`: n=4, avg TTFT=477.5 ms, p50=582.8 ms, p90=653.1 ms, exact hit=1.00, device hit=0.25, consumed=0.25, protected=0.0, F1=1.0000, status={'no_fast_path': 3, 'consumed': 1}
- `agent_scaling|exact_reuse_no_hints|8000|1|5|1`: n=5, avg TTFT=372.6 ms, p50=93.9 ms, p90=800.3 ms, exact hit=1.00, device hit=0.60, consumed=0.20, protected=0.0, F1=1.0000, status={'consumed': 1, 'no_fast_path': 2, 'device_hit_without_consumed': 2}
- `agent_scaling|exact_reuse_plus_code_hints|8000|1|1|1`: n=1, avg TTFT=82.6 ms, p50=82.6 ms, p90=82.6 ms, exact hit=1.00, device hit=1.00, consumed=1.00, protected=4490.0, F1=1.0000, status={'consumed': 1}
- `agent_scaling|exact_reuse_plus_code_hints|8000|1|2|1`: n=2, avg TTFT=110.0 ms, p50=98.1 ms, p90=121.8 ms, exact hit=1.00, device hit=1.00, consumed=1.00, protected=4490.0, F1=1.0000, status={'consumed': 2}
- `agent_scaling|exact_reuse_plus_code_hints|8000|1|3|1`: n=3, avg TTFT=395.2 ms, p50=519.8 ms, p90=565.9 ms, exact hit=1.00, device hit=1.00, consumed=0.33, protected=4490.0, F1=1.0000, status={'protected_not_consumed:no_anchor_match': 2, 'consumed': 1}
- `agent_scaling|exact_reuse_plus_code_hints|8000|1|4|1`: n=4, avg TTFT=418.8 ms, p50=680.6 ms, p90=773.6 ms, exact hit=1.00, device hit=1.00, consumed=0.50, protected=4490.0, F1=1.0000, status={'consumed': 2, 'protected_not_consumed:no_anchor_match': 2}
- `agent_scaling|exact_reuse_plus_code_hints|8000|1|5|1`: n=5, avg TTFT=607.0 ms, p50=899.8 ms, p90=993.3 ms, exact hit=1.00, device hit=1.00, consumed=0.40, protected=4490.0, F1=1.0000, status={'protected_not_consumed:no_anchor_match': 3, 'consumed': 2}
- `agent_scaling|hints_no_exact|8000|1|1|1`: n=1, avg TTFT=235.1 ms, p50=235.1 ms, p90=235.1 ms, exact hit=0.00, device hit=1.00, consumed=0.00, protected=2245.0, F1=1.0000, status={'protected_not_consumed:no_anchor_match': 1}
- `agent_scaling|hints_no_exact|8000|1|2|1`: n=2, avg TTFT=256.8 ms, p50=256.4 ms, p90=257.2 ms, exact hit=0.00, device hit=1.00, consumed=0.00, protected=2245.0, F1=1.0000, status={'protected_not_consumed:no_anchor_match': 2}
- `agent_scaling|hints_no_exact|8000|1|3|1`: n=3, avg TTFT=257.2 ms, p50=256.8 ms, p90=258.5 ms, exact hit=0.00, device hit=1.00, consumed=0.00, protected=2245.0, F1=1.0000, status={'protected_not_consumed:no_anchor_match': 3}
- `agent_scaling|hints_no_exact|8000|1|4|1`: n=4, avg TTFT=257.5 ms, p50=257.8 ms, p90=258.8 ms, exact hit=0.00, device hit=1.00, consumed=0.00, protected=2245.0, F1=1.0000, status={'protected_not_consumed:no_anchor_match': 4}
- `agent_scaling|hints_no_exact|8000|1|5|1`: n=5, avg TTFT=259.1 ms, p50=258.1 ms, p90=263.7 ms, exact hit=0.00, device hit=1.00, consumed=0.00, protected=2245.0, F1=1.0000, status={'protected_not_consumed:no_anchor_match': 5}
- `agent_scaling|prefix_cache_only|8000|1|1|1`: n=1, avg TTFT=263.5 ms, p50=263.5 ms, p90=263.5 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 1}
- `agent_scaling|prefix_cache_only|8000|1|2|1`: n=2, avg TTFT=150.3 ms, p50=47.2 ms, p90=253.4 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 2}
- `agent_scaling|prefix_cache_only|8000|1|3|1`: n=3, avg TTFT=115.4 ms, p50=46.5 ms, p90=253.2 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 3}
- `agent_scaling|prefix_cache_only|8000|1|4|1`: n=4, avg TTFT=243.0 ms, p50=252.4 ms, p90=253.3 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 4}
- `agent_scaling|prefix_cache_only|8000|1|5|1`: n=5, avg TTFT=242.1 ms, p50=234.8 ms, p90=254.6 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 5}

## Exact-Reuse Speedup vs Prefix

- `('agent_scaling', '8000', '1', '1', '1')`: p50=3.19x, p90=3.19x (prefix=263.5/263.5 ms, exact+hints=82.6/82.6 ms)
- `('agent_scaling', '8000', '1', '2', '1')`: p50=0.48x, p90=2.08x (prefix=47.2/253.4 ms, exact+hints=98.1/121.8 ms)
- `('agent_scaling', '8000', '1', '3', '1')`: p50=0.09x, p90=0.45x (prefix=46.5/253.2 ms, exact+hints=519.8/565.9 ms)
- `('agent_scaling', '8000', '1', '4', '1')`: p50=0.37x, p90=0.33x (prefix=252.4/253.3 ms, exact+hints=680.6/773.6 ms)
- `('agent_scaling', '8000', '1', '5', '1')`: p50=0.26x, p90=0.26x (prefix=234.8/254.6 ms, exact+hints=899.8/993.3 ms)
- `('agent_scaling_workflow', '8000', '1', '1', '1')`: p50=3.19x, p90=3.19x (prefix=263.5/263.5 ms, exact+hints=82.6/82.6 ms)
- `('agent_scaling_workflow', '8000', '1', '2', '1')`: p50=1.37x, p90=1.37x (prefix=300.7/300.7 ms, exact+hints=220.0/220.0 ms)
- `('agent_scaling_workflow', '8000', '1', '3', '1')`: p50=0.29x, p90=0.29x (prefix=346.2/346.2 ms, exact+hints=1185.5/1185.5 ms)
- `('agent_scaling_workflow', '8000', '1', '4', '1')`: p50=0.58x, p90=0.58x (prefix=972.1/972.1 ms, exact+hints=1675.2/1675.2 ms)
- `('agent_scaling_workflow', '8000', '1', '5', '1')`: p50=0.40x, p90=0.40x (prefix=1210.4/1210.4 ms, exact+hints=3034.9/3034.9 ms)
