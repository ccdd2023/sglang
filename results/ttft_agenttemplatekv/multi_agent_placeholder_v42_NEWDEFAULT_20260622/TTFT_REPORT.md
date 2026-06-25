# KVCOMM TTFT Stress Report

This run is prefill/TTFT dominant and is intended to complement the realistic 100-case E2E table.

## Output Schema

`ttft_ms`, `e2e_ms`, `prompt_tokens`, `cached_tokens`, `cached_ratio`, `exact_hit`, `match_reason`, `matched_content_signature`, `output_exact_match_vs_baseline`, `output_token_f1_vs_baseline`, `mode`, `case_id`, `agent_id`, `segment_count`, `max_file_chars`.

## Row Counts

- agent_scaling: 75
- agent_scaling_workflow: 25

## Summary Groups

- `agent_scaling_workflow|exact_reuse_no_hints|8000|1|1|1`: n=1, avg TTFT=93.4 ms, p50=93.4 ms, p90=93.4 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_no_hints|8000|1|2|1`: n=1, avg TTFT=189.2 ms, p50=189.2 ms, p90=189.2 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_no_hints|8000|1|3|1`: n=1, avg TTFT=346.2 ms, p50=346.2 ms, p90=346.2 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_no_hints|8000|1|4|1`: n=1, avg TTFT=797.5 ms, p50=797.5 ms, p90=797.5 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_no_hints|8000|1|5|1`: n=1, avg TTFT=2301.2 ms, p50=2301.2 ms, p90=2301.2 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_plus_code_hints|8000|1|1|1`: n=1, avg TTFT=93.4 ms, p50=93.4 ms, p90=93.4 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_plus_code_hints|8000|1|2|1`: n=1, avg TTFT=163.0 ms, p50=163.0 ms, p90=163.0 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_plus_code_hints|8000|1|3|1`: n=1, avg TTFT=278.1 ms, p50=278.1 ms, p90=278.1 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_plus_code_hints|8000|1|4|1`: n=1, avg TTFT=1498.8 ms, p50=1498.8 ms, p90=1498.8 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_plus_code_hints|8000|1|5|1`: n=1, avg TTFT=2870.3 ms, p50=2870.3 ms, p90=2870.3 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|hints_no_exact|8000|1|1|1`: n=1, avg TTFT=231.4 ms, p50=231.4 ms, p90=231.4 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|hints_no_exact|8000|1|2|1`: n=1, avg TTFT=511.1 ms, p50=511.1 ms, p90=511.1 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|hints_no_exact|8000|1|3|1`: n=1, avg TTFT=766.6 ms, p50=766.6 ms, p90=766.6 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|hints_no_exact|8000|1|4|1`: n=1, avg TTFT=1032.7 ms, p50=1032.7 ms, p90=1032.7 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|hints_no_exact|8000|1|5|1`: n=1, avg TTFT=1284.4 ms, p50=1284.4 ms, p90=1284.4 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|placeholder_knn_reuse|8000|1|1|1`: n=1, avg TTFT=69.1 ms, p50=69.1 ms, p90=69.1 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|placeholder_knn_reuse|8000|1|2|1`: n=1, avg TTFT=152.9 ms, p50=152.9 ms, p90=152.9 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|placeholder_knn_reuse|8000|1|3|1`: n=1, avg TTFT=236.2 ms, p50=236.2 ms, p90=236.2 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|placeholder_knn_reuse|8000|1|4|1`: n=1, avg TTFT=339.5 ms, p50=339.5 ms, p90=339.5 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|placeholder_knn_reuse|8000|1|5|1`: n=1, avg TTFT=1055.2 ms, p50=1055.2 ms, p90=1055.2 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|prefix_cache_only|8000|1|1|1`: n=1, avg TTFT=261.8 ms, p50=261.8 ms, p90=261.8 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|prefix_cache_only|8000|1|2|1`: n=1, avg TTFT=286.3 ms, p50=286.3 ms, p90=286.3 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|prefix_cache_only|8000|1|3|1`: n=1, avg TTFT=337.6 ms, p50=337.6 ms, p90=337.6 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|prefix_cache_only|8000|1|4|1`: n=1, avg TTFT=393.7 ms, p50=393.7 ms, p90=393.7 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|prefix_cache_only|8000|1|5|1`: n=1, avg TTFT=427.1 ms, p50=427.1 ms, p90=427.1 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling|exact_reuse_no_hints|8000|1|1|1`: n=1, avg TTFT=93.4 ms, p50=93.4 ms, p90=93.4 ms, exact hit=1.00, device hit=1.00, consumed=1.00, protected=0.0, F1=1.0000, status={'consumed': 1}
- `agent_scaling|exact_reuse_no_hints|8000|1|2|1`: n=2, avg TTFT=94.6 ms, p50=92.4 ms, p90=96.8 ms, exact hit=1.00, device hit=1.00, consumed=0.50, protected=0.0, F1=1.0000, status={'consumed': 1, 'device_hit_without_consumed': 1}
- `agent_scaling|exact_reuse_no_hints|8000|1|3|1`: n=3, avg TTFT=115.4 ms, p50=117.8 ms, p90=133.7 ms, exact hit=1.00, device hit=1.00, consumed=0.33, protected=0.0, F1=1.0000, status={'consumed': 1, 'device_hit_without_consumed': 2}
- `agent_scaling|exact_reuse_no_hints|8000|1|4|1`: n=4, avg TTFT=199.4 ms, p50=87.5 ms, p90=542.2 ms, exact hit=1.00, device hit=0.75, consumed=0.25, protected=0.0, F1=1.0000, status={'consumed': 1, 'device_hit_without_consumed': 2, 'no_fast_path': 1}
- `agent_scaling|exact_reuse_no_hints|8000|1|5|1`: n=5, avg TTFT=460.2 ms, p50=677.3 ms, p90=724.3 ms, exact hit=1.00, device hit=0.40, consumed=0.20, protected=0.0, F1=1.0000, status={'no_fast_path': 3, 'consumed': 1, 'device_hit_without_consumed': 1}
- `agent_scaling|exact_reuse_plus_code_hints|8000|1|1|1`: n=1, avg TTFT=93.4 ms, p50=93.4 ms, p90=93.4 ms, exact hit=1.00, device hit=1.00, consumed=1.00, protected=4490.0, F1=1.0000, status={'consumed': 1}
- `agent_scaling|exact_reuse_plus_code_hints|8000|1|2|1`: n=2, avg TTFT=81.5 ms, p50=77.2 ms, p90=85.8 ms, exact hit=1.00, device hit=1.00, consumed=1.00, protected=4490.0, F1=1.0000, status={'consumed': 2}
- `agent_scaling|exact_reuse_plus_code_hints|8000|1|3|1`: n=3, avg TTFT=92.7 ms, p50=85.5 ms, p90=115.1 ms, exact hit=1.00, device hit=1.00, consumed=1.00, protected=4490.0, F1=1.0000, status={'consumed': 3}
- `agent_scaling|exact_reuse_plus_code_hints|8000|1|4|1`: n=4, avg TTFT=374.7 ms, p50=602.2 ms, p90=701.0 ms, exact hit=1.00, device hit=1.00, consumed=0.50, protected=4490.0, F1=1.0000, status={'consumed': 2, 'protected_not_consumed:no_anchor_match': 2}
- `agent_scaling|exact_reuse_plus_code_hints|8000|1|5|1`: n=5, avg TTFT=574.1 ms, p50=830.9 ms, p90=892.0 ms, exact hit=1.00, device hit=1.00, consumed=0.40, protected=4490.0, F1=1.0000, status={'protected_not_consumed:no_anchor_match': 3, 'consumed': 2}
- `agent_scaling|hints_no_exact|8000|1|1|1`: n=1, avg TTFT=231.4 ms, p50=231.4 ms, p90=231.4 ms, exact hit=0.00, device hit=1.00, consumed=0.00, protected=2245.0, F1=1.0000, status={'protected_not_consumed:no_anchor_match': 1}
- `agent_scaling|hints_no_exact|8000|1|2|1`: n=2, avg TTFT=255.5 ms, p50=254.1 ms, p90=257.0 ms, exact hit=0.00, device hit=1.00, consumed=0.00, protected=2245.0, F1=1.0000, status={'protected_not_consumed:no_anchor_match': 2}
- `agent_scaling|hints_no_exact|8000|1|3|1`: n=3, avg TTFT=255.5 ms, p50=256.3 ms, p90=258.3 ms, exact hit=0.00, device hit=1.00, consumed=0.00, protected=2245.0, F1=1.0000, status={'protected_not_consumed:no_anchor_match': 3}
- `agent_scaling|hints_no_exact|8000|1|4|1`: n=4, avg TTFT=258.2 ms, p50=259.3 ms, p90=263.5 ms, exact hit=0.00, device hit=1.00, consumed=0.00, protected=2245.0, F1=1.0000, status={'protected_not_consumed:no_anchor_match': 4}
- `agent_scaling|hints_no_exact|8000|1|5|1`: n=5, avg TTFT=256.9 ms, p50=256.1 ms, p90=259.9 ms, exact hit=0.00, device hit=1.00, consumed=0.00, protected=2245.0, F1=1.0000, status={'protected_not_consumed:no_anchor_match': 5}
- `agent_scaling|placeholder_knn_reuse|8000|1|1|1`: n=1, avg TTFT=69.1 ms, p50=69.1 ms, p90=69.1 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 1}
- `agent_scaling|placeholder_knn_reuse|8000|1|2|1`: n=2, avg TTFT=76.5 ms, p50=69.2 ms, p90=83.7 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 2}
- `agent_scaling|placeholder_knn_reuse|8000|1|3|1`: n=3, avg TTFT=78.7 ms, p50=80.9 ms, p90=82.2 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 3}
- `agent_scaling|placeholder_knn_reuse|8000|1|4|1`: n=4, avg TTFT=84.9 ms, p50=81.5 ms, p90=111.5 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 4}
- `agent_scaling|placeholder_knn_reuse|8000|1|5|1`: n=5, avg TTFT=211.0 ms, p50=277.4 ms, p90=298.6 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 5}
- `agent_scaling|prefix_cache_only|8000|1|1|1`: n=1, avg TTFT=261.8 ms, p50=261.8 ms, p90=261.8 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 1}
- `agent_scaling|prefix_cache_only|8000|1|2|1`: n=2, avg TTFT=143.1 ms, p50=42.2 ms, p90=244.1 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 2}
- `agent_scaling|prefix_cache_only|8000|1|3|1`: n=3, avg TTFT=112.5 ms, p50=48.7 ms, p90=247.5 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 3}
- `agent_scaling|prefix_cache_only|8000|1|4|1`: n=4, avg TTFT=98.4 ms, p50=47.3 ms, p90=254.5 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 4}
- `agent_scaling|prefix_cache_only|8000|1|5|1`: n=5, avg TTFT=85.4 ms, p50=46.1 ms, p90=245.7 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 5}

## Exact-Reuse Speedup vs Prefix

- `('agent_scaling', '8000', '1', '1', '1')`: p50=2.80x, p90=2.80x (prefix=261.8/261.8 ms, exact+hints=93.4/93.4 ms)
- `('agent_scaling', '8000', '1', '2', '1')`: p50=0.55x, p90=2.85x (prefix=42.2/244.1 ms, exact+hints=77.2/85.8 ms)
- `('agent_scaling', '8000', '1', '3', '1')`: p50=0.57x, p90=2.15x (prefix=48.7/247.5 ms, exact+hints=85.5/115.1 ms)
- `('agent_scaling', '8000', '1', '4', '1')`: p50=0.08x, p90=0.36x (prefix=47.3/254.5 ms, exact+hints=602.2/701.0 ms)
- `('agent_scaling', '8000', '1', '5', '1')`: p50=0.06x, p90=0.28x (prefix=46.1/245.7 ms, exact+hints=830.9/892.0 ms)
- `('agent_scaling_workflow', '8000', '1', '1', '1')`: p50=2.80x, p90=2.80x (prefix=261.8/261.8 ms, exact+hints=93.4/93.4 ms)
- `('agent_scaling_workflow', '8000', '1', '2', '1')`: p50=1.76x, p90=1.76x (prefix=286.3/286.3 ms, exact+hints=163.0/163.0 ms)
- `('agent_scaling_workflow', '8000', '1', '3', '1')`: p50=1.21x, p90=1.21x (prefix=337.6/337.6 ms, exact+hints=278.1/278.1 ms)
- `('agent_scaling_workflow', '8000', '1', '4', '1')`: p50=0.26x, p90=0.26x (prefix=393.7/393.7 ms, exact+hints=1498.8/1498.8 ms)
- `('agent_scaling_workflow', '8000', '1', '5', '1')`: p50=0.15x, p90=0.15x (prefix=427.1/427.1 ms, exact+hints=2870.3/2870.3 ms)
