# KVCOMM TTFT Stress Report

This run is prefill/TTFT dominant and is intended to complement the realistic 100-case E2E table.

## Output Schema

`ttft_ms`, `e2e_ms`, `prompt_tokens`, `cached_tokens`, `cached_ratio`, `exact_hit`, `match_reason`, `matched_content_signature`, `output_exact_match_vs_baseline`, `output_token_f1_vs_baseline`, `mode`, `case_id`, `agent_id`, `segment_count`, `max_file_chars`.

## Row Counts

- agent_scaling: 75
- agent_scaling_workflow: 25

## Summary Groups

- `agent_scaling_workflow|exact_reuse_no_hints|8000|1|1|1`: n=1, avg TTFT=77.3 ms, p50=77.3 ms, p90=77.3 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_no_hints|8000|1|2|1`: n=1, avg TTFT=169.2 ms, p50=169.2 ms, p90=169.2 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_no_hints|8000|1|3|1`: n=1, avg TTFT=241.3 ms, p50=241.3 ms, p90=241.3 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_no_hints|8000|1|4|1`: n=1, avg TTFT=1369.9 ms, p50=1369.9 ms, p90=1369.9 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_no_hints|8000|1|5|1`: n=1, avg TTFT=1943.8 ms, p50=1943.8 ms, p90=1943.8 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_plus_code_hints|8000|1|1|1`: n=1, avg TTFT=80.4 ms, p50=80.4 ms, p90=80.4 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_plus_code_hints|8000|1|2|1`: n=1, avg TTFT=194.2 ms, p50=194.2 ms, p90=194.2 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_plus_code_hints|8000|1|3|1`: n=1, avg TTFT=254.6 ms, p50=254.6 ms, p90=254.6 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_plus_code_hints|8000|1|4|1`: n=1, avg TTFT=1651.0 ms, p50=1651.0 ms, p90=1651.0 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_plus_code_hints|8000|1|5|1`: n=1, avg TTFT=3797.2 ms, p50=3797.2 ms, p90=3797.2 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|hints_no_exact|8000|1|1|1`: n=1, avg TTFT=235.2 ms, p50=235.2 ms, p90=235.2 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|hints_no_exact|8000|1|2|1`: n=1, avg TTFT=508.2 ms, p50=508.2 ms, p90=508.2 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|hints_no_exact|8000|1|3|1`: n=1, avg TTFT=764.1 ms, p50=764.1 ms, p90=764.1 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|hints_no_exact|8000|1|4|1`: n=1, avg TTFT=1021.0 ms, p50=1021.0 ms, p90=1021.0 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|hints_no_exact|8000|1|5|1`: n=1, avg TTFT=1282.2 ms, p50=1282.2 ms, p90=1282.2 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|placeholder_knn_reuse|8000|1|1|1`: n=1, avg TTFT=269.5 ms, p50=269.5 ms, p90=269.5 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|placeholder_knn_reuse|8000|1|2|1`: n=1, avg TTFT=150.8 ms, p50=150.8 ms, p90=150.8 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|placeholder_knn_reuse|8000|1|3|1`: n=1, avg TTFT=268.1 ms, p50=268.1 ms, p90=268.1 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|placeholder_knn_reuse|8000|1|4|1`: n=1, avg TTFT=499.3 ms, p50=499.3 ms, p90=499.3 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|placeholder_knn_reuse|8000|1|5|1`: n=1, avg TTFT=859.6 ms, p50=859.6 ms, p90=859.6 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|prefix_cache_only|8000|1|1|1`: n=1, avg TTFT=257.1 ms, p50=257.1 ms, p90=257.1 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|prefix_cache_only|8000|1|2|1`: n=1, avg TTFT=292.9 ms, p50=292.9 ms, p90=292.9 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|prefix_cache_only|8000|1|3|1`: n=1, avg TTFT=338.0 ms, p50=338.0 ms, p90=338.0 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|prefix_cache_only|8000|1|4|1`: n=1, avg TTFT=381.2 ms, p50=381.2 ms, p90=381.2 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|prefix_cache_only|8000|1|5|1`: n=1, avg TTFT=432.2 ms, p50=432.2 ms, p90=432.2 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling|exact_reuse_no_hints|8000|1|1|1`: n=1, avg TTFT=77.3 ms, p50=77.3 ms, p90=77.3 ms, exact hit=1.00, device hit=1.00, consumed=1.00, protected=0.0, F1=1.0000, status={'consumed': 1}
- `agent_scaling|exact_reuse_no_hints|8000|1|2|1`: n=2, avg TTFT=84.6 ms, p50=80.6 ms, p90=88.6 ms, exact hit=1.00, device hit=1.00, consumed=0.50, protected=0.0, F1=1.0000, status={'consumed': 1, 'device_hit_without_consumed': 1}
- `agent_scaling|exact_reuse_no_hints|8000|1|3|1`: n=3, avg TTFT=80.4 ms, p50=80.9 ms, p90=82.8 ms, exact hit=1.00, device hit=1.00, consumed=0.33, protected=0.0, F1=1.0000, status={'consumed': 1, 'device_hit_without_consumed': 2}
- `agent_scaling|exact_reuse_no_hints|8000|1|4|1`: n=4, avg TTFT=342.5 ms, p50=585.3 ms, p90=618.9 ms, exact hit=1.00, device hit=0.50, consumed=0.25, protected=0.0, F1=1.0000, status={'consumed': 1, 'no_fast_path': 2, 'device_hit_without_consumed': 1}
- `agent_scaling|exact_reuse_no_hints|8000|1|5|1`: n=5, avg TTFT=388.8 ms, p50=96.7 ms, p90=865.9 ms, exact hit=1.00, device hit=0.60, consumed=0.20, protected=0.0, F1=1.0000, status={'consumed': 1, 'no_fast_path': 2, 'device_hit_without_consumed': 2}
- `agent_scaling|exact_reuse_plus_code_hints|8000|1|1|1`: n=1, avg TTFT=80.4 ms, p50=80.4 ms, p90=80.4 ms, exact hit=1.00, device hit=1.00, consumed=1.00, protected=4490.0, F1=1.0000, status={'consumed': 1}
- `agent_scaling|exact_reuse_plus_code_hints|8000|1|2|1`: n=2, avg TTFT=97.1 ms, p50=95.9 ms, p90=98.3 ms, exact hit=1.00, device hit=1.00, consumed=1.00, protected=4490.0, F1=1.0000, status={'consumed': 2}
- `agent_scaling|exact_reuse_plus_code_hints|8000|1|3|1`: n=3, avg TTFT=84.9 ms, p50=84.2 ms, p90=87.0 ms, exact hit=1.00, device hit=1.00, consumed=1.00, protected=4490.0, F1=1.0000, status={'consumed': 3}
- `agent_scaling|exact_reuse_plus_code_hints|8000|1|4|1`: n=4, avg TTFT=412.7 ms, p50=733.1 ms, p90=738.3 ms, exact hit=1.00, device hit=1.00, consumed=0.50, protected=4490.0, F1=1.0000, status={'consumed': 2, 'protected_not_consumed:no_anchor_match': 2}
- `agent_scaling|exact_reuse_plus_code_hints|8000|1|5|1`: n=5, avg TTFT=759.4 ms, p50=924.4 ms, p90=953.4 ms, exact hit=1.00, device hit=1.00, consumed=0.20, protected=4490.0, F1=1.0000, status={'protected_not_consumed:no_anchor_match': 4, 'consumed': 1}
- `agent_scaling|hints_no_exact|8000|1|1|1`: n=1, avg TTFT=235.2 ms, p50=235.2 ms, p90=235.2 ms, exact hit=0.00, device hit=1.00, consumed=0.00, protected=2245.0, F1=1.0000, status={'protected_not_consumed:no_anchor_match': 1}
- `agent_scaling|hints_no_exact|8000|1|2|1`: n=2, avg TTFT=254.1 ms, p50=254.1 ms, p90=254.1 ms, exact hit=0.00, device hit=1.00, consumed=0.00, protected=2245.0, F1=1.0000, status={'protected_not_consumed:no_anchor_match': 2}
- `agent_scaling|hints_no_exact|8000|1|3|1`: n=3, avg TTFT=254.7 ms, p50=255.1 ms, p90=255.2 ms, exact hit=0.00, device hit=1.00, consumed=0.00, protected=2245.0, F1=1.0000, status={'protected_not_consumed:no_anchor_match': 3}
- `agent_scaling|hints_no_exact|8000|1|4|1`: n=4, avg TTFT=255.2 ms, p50=255.7 ms, p90=256.3 ms, exact hit=0.00, device hit=1.00, consumed=0.00, protected=2245.0, F1=1.0000, status={'protected_not_consumed:no_anchor_match': 4}
- `agent_scaling|hints_no_exact|8000|1|5|1`: n=5, avg TTFT=256.4 ms, p50=256.4 ms, p90=257.0 ms, exact hit=0.00, device hit=1.00, consumed=0.00, protected=2245.0, F1=1.0000, status={'protected_not_consumed:no_anchor_match': 5}
- `agent_scaling|placeholder_knn_reuse|8000|1|1|1`: n=1, avg TTFT=269.5 ms, p50=269.5 ms, p90=269.5 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 1}
- `agent_scaling|placeholder_knn_reuse|8000|1|2|1`: n=2, avg TTFT=75.4 ms, p50=72.1 ms, p90=78.8 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 2}
- `agent_scaling|placeholder_knn_reuse|8000|1|3|1`: n=3, avg TTFT=89.4 ms, p50=93.1 ms, p90=96.6 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 3}
- `agent_scaling|placeholder_knn_reuse|8000|1|4|1`: n=4, avg TTFT=124.8 ms, p50=78.2 ms, p90=276.3 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 4}
- `agent_scaling|placeholder_knn_reuse|8000|1|5|1`: n=5, avg TTFT=171.9 ms, p50=106.4 ms, p90=301.8 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 5}
- `agent_scaling|prefix_cache_only|8000|1|1|1`: n=1, avg TTFT=257.1 ms, p50=257.1 ms, p90=257.1 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 1}
- `agent_scaling|prefix_cache_only|8000|1|2|1`: n=2, avg TTFT=146.4 ms, p50=44.6 ms, p90=248.3 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 2}
- `agent_scaling|prefix_cache_only|8000|1|3|1`: n=3, avg TTFT=112.7 ms, p50=46.1 ms, p90=248.3 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 3}
- `agent_scaling|prefix_cache_only|8000|1|4|1`: n=4, avg TTFT=95.3 ms, p50=44.8 ms, p90=249.2 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 4}
- `agent_scaling|prefix_cache_only|8000|1|5|1`: n=5, avg TTFT=86.4 ms, p50=45.1 ms, p90=251.2 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 5}

## Exact-Reuse Speedup vs Prefix

- `('agent_scaling', '8000', '1', '1', '1')`: p50=3.20x, p90=3.20x (prefix=257.1/257.1 ms, exact+hints=80.4/80.4 ms)
- `('agent_scaling', '8000', '1', '2', '1')`: p50=0.46x, p90=2.53x (prefix=44.6/248.3 ms, exact+hints=95.9/98.3 ms)
- `('agent_scaling', '8000', '1', '3', '1')`: p50=0.55x, p90=2.85x (prefix=46.1/248.3 ms, exact+hints=84.2/87.0 ms)
- `('agent_scaling', '8000', '1', '4', '1')`: p50=0.06x, p90=0.34x (prefix=44.8/249.2 ms, exact+hints=733.1/738.3 ms)
- `('agent_scaling', '8000', '1', '5', '1')`: p50=0.05x, p90=0.26x (prefix=45.1/251.2 ms, exact+hints=924.4/953.4 ms)
- `('agent_scaling_workflow', '8000', '1', '1', '1')`: p50=3.20x, p90=3.20x (prefix=257.1/257.1 ms, exact+hints=80.4/80.4 ms)
- `('agent_scaling_workflow', '8000', '1', '2', '1')`: p50=1.51x, p90=1.51x (prefix=292.9/292.9 ms, exact+hints=194.2/194.2 ms)
- `('agent_scaling_workflow', '8000', '1', '3', '1')`: p50=1.33x, p90=1.33x (prefix=338.0/338.0 ms, exact+hints=254.6/254.6 ms)
- `('agent_scaling_workflow', '8000', '1', '4', '1')`: p50=0.23x, p90=0.23x (prefix=381.2/381.2 ms, exact+hints=1651.0/1651.0 ms)
- `('agent_scaling_workflow', '8000', '1', '5', '1')`: p50=0.11x, p90=0.11x (prefix=432.2/432.2 ms, exact+hints=3797.2/3797.2 ms)
