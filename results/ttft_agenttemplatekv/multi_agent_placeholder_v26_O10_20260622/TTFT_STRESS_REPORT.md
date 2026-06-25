# KVCOMM TTFT Stress Report

This run is prefill/TTFT dominant and is intended to complement the realistic 100-case E2E table.

## Output Schema

`ttft_ms`, `e2e_ms`, `prompt_tokens`, `cached_tokens`, `cached_ratio`, `exact_hit`, `match_reason`, `matched_content_signature`, `output_exact_match_vs_baseline`, `output_token_f1_vs_baseline`, `mode`, `case_id`, `agent_id`, `segment_count`, `max_file_chars`.

## Row Counts

- agent_scaling: 75
- agent_scaling_workflow: 25

## Summary Groups

- `agent_scaling_workflow|exact_reuse_no_hints|8000|1|1|1`: n=1, avg TTFT=71.7 ms, p50=71.7 ms, p90=71.7 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_no_hints|8000|1|2|1`: n=1, avg TTFT=205.4 ms, p50=205.4 ms, p90=205.4 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_no_hints|8000|1|3|1`: n=1, avg TTFT=247.8 ms, p50=247.8 ms, p90=247.8 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_no_hints|8000|1|4|1`: n=1, avg TTFT=1484.7 ms, p50=1484.7 ms, p90=1484.7 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_no_hints|8000|1|5|1`: n=1, avg TTFT=1903.9 ms, p50=1903.9 ms, p90=1903.9 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_plus_code_hints|8000|1|1|1`: n=1, avg TTFT=83.8 ms, p50=83.8 ms, p90=83.8 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_plus_code_hints|8000|1|2|1`: n=1, avg TTFT=195.9 ms, p50=195.9 ms, p90=195.9 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_plus_code_hints|8000|1|3|1`: n=1, avg TTFT=272.6 ms, p50=272.6 ms, p90=272.6 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_plus_code_hints|8000|1|4|1`: n=1, avg TTFT=1471.6 ms, p50=1471.6 ms, p90=1471.6 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_plus_code_hints|8000|1|5|1`: n=1, avg TTFT=3736.8 ms, p50=3736.8 ms, p90=3736.8 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|hints_no_exact|8000|1|1|1`: n=1, avg TTFT=234.8 ms, p50=234.8 ms, p90=234.8 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|hints_no_exact|8000|1|2|1`: n=1, avg TTFT=509.0 ms, p50=509.0 ms, p90=509.0 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|hints_no_exact|8000|1|3|1`: n=1, avg TTFT=771.9 ms, p50=771.9 ms, p90=771.9 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|hints_no_exact|8000|1|4|1`: n=1, avg TTFT=1029.4 ms, p50=1029.4 ms, p90=1029.4 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|hints_no_exact|8000|1|5|1`: n=1, avg TTFT=1286.6 ms, p50=1286.6 ms, p90=1286.6 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|placeholder_knn_reuse|8000|1|1|1`: n=1, avg TTFT=66.7 ms, p50=66.7 ms, p90=66.7 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|placeholder_knn_reuse|8000|1|2|1`: n=1, avg TTFT=331.6 ms, p50=331.6 ms, p90=331.6 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|placeholder_knn_reuse|8000|1|3|1`: n=1, avg TTFT=441.1 ms, p50=441.1 ms, p90=441.1 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|placeholder_knn_reuse|8000|1|4|1`: n=1, avg TTFT=487.7 ms, p50=487.7 ms, p90=487.7 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|placeholder_knn_reuse|8000|1|5|1`: n=1, avg TTFT=1339.8 ms, p50=1339.8 ms, p90=1339.8 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|prefix_cache_only|8000|1|1|1`: n=1, avg TTFT=254.9 ms, p50=254.9 ms, p90=254.9 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|prefix_cache_only|8000|1|2|1`: n=1, avg TTFT=297.0 ms, p50=297.0 ms, p90=297.0 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|prefix_cache_only|8000|1|3|1`: n=1, avg TTFT=343.4 ms, p50=343.4 ms, p90=343.4 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|prefix_cache_only|8000|1|4|1`: n=1, avg TTFT=385.1 ms, p50=385.1 ms, p90=385.1 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|prefix_cache_only|8000|1|5|1`: n=1, avg TTFT=431.0 ms, p50=431.0 ms, p90=431.0 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling|exact_reuse_no_hints|8000|1|1|1`: n=1, avg TTFT=71.7 ms, p50=71.7 ms, p90=71.7 ms, exact hit=1.00, device hit=1.00, consumed=1.00, protected=0.0, F1=1.0000, status={'consumed': 1}
- `agent_scaling|exact_reuse_no_hints|8000|1|2|1`: n=2, avg TTFT=102.7 ms, p50=97.6 ms, p90=107.8 ms, exact hit=1.00, device hit=1.00, consumed=0.50, protected=0.0, F1=1.0000, status={'consumed': 1, 'device_hit_without_consumed': 1}
- `agent_scaling|exact_reuse_no_hints|8000|1|3|1`: n=3, avg TTFT=82.6 ms, p50=82.4 ms, p90=85.9 ms, exact hit=1.00, device hit=1.00, consumed=0.33, protected=0.0, F1=1.0000, status={'consumed': 1, 'device_hit_without_consumed': 2}
- `agent_scaling|exact_reuse_no_hints|8000|1|4|1`: n=4, avg TTFT=371.2 ms, p50=633.0 ms, p90=668.5 ms, exact hit=1.00, device hit=0.50, consumed=0.25, protected=0.0, F1=1.0000, status={'consumed': 1, 'no_fast_path': 2, 'device_hit_without_consumed': 1}
- `agent_scaling|exact_reuse_no_hints|8000|1|5|1`: n=5, avg TTFT=380.8 ms, p50=98.2 ms, p90=861.3 ms, exact hit=1.00, device hit=0.60, consumed=0.20, protected=0.0, F1=1.0000, status={'consumed': 1, 'no_fast_path': 2, 'device_hit_without_consumed': 2}
- `agent_scaling|exact_reuse_plus_code_hints|8000|1|1|1`: n=1, avg TTFT=83.8 ms, p50=83.8 ms, p90=83.8 ms, exact hit=1.00, device hit=1.00, consumed=1.00, protected=4490.0, F1=1.0000, status={'consumed': 1}
- `agent_scaling|exact_reuse_plus_code_hints|8000|1|2|1`: n=2, avg TTFT=98.0 ms, p50=96.0 ms, p90=100.0 ms, exact hit=1.00, device hit=1.00, consumed=1.00, protected=4490.0, F1=1.0000, status={'consumed': 2}
- `agent_scaling|exact_reuse_plus_code_hints|8000|1|3|1`: n=3, avg TTFT=90.9 ms, p50=90.6 ms, p90=98.5 ms, exact hit=1.00, device hit=1.00, consumed=1.00, protected=4490.0, F1=1.0000, status={'consumed': 3}
- `agent_scaling|exact_reuse_plus_code_hints|8000|1|4|1`: n=4, avg TTFT=367.9 ms, p50=621.1 ms, p90=669.8 ms, exact hit=1.00, device hit=1.00, consumed=0.50, protected=4490.0, F1=1.0000, status={'consumed': 2, 'protected_not_consumed:no_anchor_match': 2}
- `agent_scaling|exact_reuse_plus_code_hints|8000|1|5|1`: n=5, avg TTFT=747.4 ms, p50=884.1 ms, p90=959.1 ms, exact hit=1.00, device hit=1.00, consumed=0.20, protected=4490.0, F1=1.0000, status={'protected_not_consumed:no_anchor_match': 4, 'consumed': 1}
- `agent_scaling|hints_no_exact|8000|1|1|1`: n=1, avg TTFT=234.8 ms, p50=234.8 ms, p90=234.8 ms, exact hit=0.00, device hit=1.00, consumed=0.00, protected=2245.0, F1=1.0000, status={'protected_not_consumed:no_anchor_match': 1}
- `agent_scaling|hints_no_exact|8000|1|2|1`: n=2, avg TTFT=254.5 ms, p50=253.0 ms, p90=256.1 ms, exact hit=0.00, device hit=1.00, consumed=0.00, protected=2245.0, F1=1.0000, status={'protected_not_consumed:no_anchor_match': 2}
- `agent_scaling|hints_no_exact|8000|1|3|1`: n=3, avg TTFT=257.3 ms, p50=256.4 ms, p90=259.6 ms, exact hit=0.00, device hit=1.00, consumed=0.00, protected=2245.0, F1=1.0000, status={'protected_not_consumed:no_anchor_match': 3}
- `agent_scaling|hints_no_exact|8000|1|4|1`: n=4, avg TTFT=257.4 ms, p50=257.4 ms, p90=257.6 ms, exact hit=0.00, device hit=1.00, consumed=0.00, protected=2245.0, F1=1.0000, status={'protected_not_consumed:no_anchor_match': 4}
- `agent_scaling|hints_no_exact|8000|1|5|1`: n=5, avg TTFT=257.3 ms, p50=257.2 ms, p90=258.7 ms, exact hit=0.00, device hit=1.00, consumed=0.00, protected=2245.0, F1=1.0000, status={'protected_not_consumed:no_anchor_match': 5}
- `agent_scaling|placeholder_knn_reuse|8000|1|1|1`: n=1, avg TTFT=66.7 ms, p50=66.7 ms, p90=66.7 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 1}
- `agent_scaling|placeholder_knn_reuse|8000|1|2|1`: n=2, avg TTFT=165.8 ms, p50=64.7 ms, p90=266.9 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 2}
- `agent_scaling|placeholder_knn_reuse|8000|1|3|1`: n=3, avg TTFT=147.0 ms, p50=86.5 ms, p90=270.2 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 3}
- `agent_scaling|placeholder_knn_reuse|8000|1|4|1`: n=4, avg TTFT=121.9 ms, p50=77.0 ms, p90=266.0 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 4}
- `agent_scaling|placeholder_knn_reuse|8000|1|5|1`: n=5, avg TTFT=268.0 ms, p50=268.2 ms, p90=271.9 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 5}
- `agent_scaling|prefix_cache_only|8000|1|1|1`: n=1, avg TTFT=254.9 ms, p50=254.9 ms, p90=254.9 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 1}
- `agent_scaling|prefix_cache_only|8000|1|2|1`: n=2, avg TTFT=148.5 ms, p50=45.5 ms, p90=251.5 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 2}
- `agent_scaling|prefix_cache_only|8000|1|3|1`: n=3, avg TTFT=114.5 ms, p50=45.9 ms, p90=252.1 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 3}
- `agent_scaling|prefix_cache_only|8000|1|4|1`: n=4, avg TTFT=96.3 ms, p50=45.3 ms, p90=250.3 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 4}
- `agent_scaling|prefix_cache_only|8000|1|5|1`: n=5, avg TTFT=86.2 ms, p50=44.7 ms, p90=252.0 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 5}

## Exact-Reuse Speedup vs Prefix

- `('agent_scaling', '8000', '1', '1', '1')`: p50=3.04x, p90=3.04x (prefix=254.9/254.9 ms, exact+hints=83.8/83.8 ms)
- `('agent_scaling', '8000', '1', '2', '1')`: p50=0.47x, p90=2.52x (prefix=45.5/251.5 ms, exact+hints=96.0/100.0 ms)
- `('agent_scaling', '8000', '1', '3', '1')`: p50=0.51x, p90=2.56x (prefix=45.9/252.1 ms, exact+hints=90.6/98.5 ms)
- `('agent_scaling', '8000', '1', '4', '1')`: p50=0.07x, p90=0.37x (prefix=45.3/250.3 ms, exact+hints=621.1/669.8 ms)
- `('agent_scaling', '8000', '1', '5', '1')`: p50=0.05x, p90=0.26x (prefix=44.7/252.0 ms, exact+hints=884.1/959.1 ms)
- `('agent_scaling_workflow', '8000', '1', '1', '1')`: p50=3.04x, p90=3.04x (prefix=254.9/254.9 ms, exact+hints=83.8/83.8 ms)
- `('agent_scaling_workflow', '8000', '1', '2', '1')`: p50=1.52x, p90=1.52x (prefix=297.0/297.0 ms, exact+hints=195.9/195.9 ms)
- `('agent_scaling_workflow', '8000', '1', '3', '1')`: p50=1.26x, p90=1.26x (prefix=343.4/343.4 ms, exact+hints=272.6/272.6 ms)
- `('agent_scaling_workflow', '8000', '1', '4', '1')`: p50=0.26x, p90=0.26x (prefix=385.1/385.1 ms, exact+hints=1471.6/1471.6 ms)
- `('agent_scaling_workflow', '8000', '1', '5', '1')`: p50=0.12x, p90=0.12x (prefix=431.0/431.0 ms, exact+hints=3736.8/3736.8 ms)
