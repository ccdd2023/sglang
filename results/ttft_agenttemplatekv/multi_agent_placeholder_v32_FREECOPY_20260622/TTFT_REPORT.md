# KVCOMM TTFT Stress Report

This run is prefill/TTFT dominant and is intended to complement the realistic 100-case E2E table.

## Output Schema

`ttft_ms`, `e2e_ms`, `prompt_tokens`, `cached_tokens`, `cached_ratio`, `exact_hit`, `match_reason`, `matched_content_signature`, `output_exact_match_vs_baseline`, `output_token_f1_vs_baseline`, `mode`, `case_id`, `agent_id`, `segment_count`, `max_file_chars`.

## Row Counts

- agent_scaling: 75
- agent_scaling_workflow: 25

## Summary Groups

- `agent_scaling_workflow|exact_reuse_no_hints|8000|1|1|1`: n=1, avg TTFT=105.6 ms, p50=105.6 ms, p90=105.6 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_no_hints|8000|1|2|1`: n=1, avg TTFT=200.3 ms, p50=200.3 ms, p90=200.3 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_no_hints|8000|1|3|1`: n=1, avg TTFT=266.5 ms, p50=266.5 ms, p90=266.5 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_no_hints|8000|1|4|1`: n=1, avg TTFT=1369.5 ms, p50=1369.5 ms, p90=1369.5 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_no_hints|8000|1|5|1`: n=1, avg TTFT=2623.0 ms, p50=2623.0 ms, p90=2623.0 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_plus_code_hints|8000|1|1|1`: n=1, avg TTFT=110.1 ms, p50=110.1 ms, p90=110.1 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_plus_code_hints|8000|1|2|1`: n=1, avg TTFT=218.2 ms, p50=218.2 ms, p90=218.2 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_plus_code_hints|8000|1|3|1`: n=1, avg TTFT=358.4 ms, p50=358.4 ms, p90=358.4 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_plus_code_hints|8000|1|4|1`: n=1, avg TTFT=1514.7 ms, p50=1514.7 ms, p90=1514.7 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_plus_code_hints|8000|1|5|1`: n=1, avg TTFT=3384.6 ms, p50=3384.6 ms, p90=3384.6 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|hints_no_exact|8000|1|1|1`: n=1, avg TTFT=238.7 ms, p50=238.7 ms, p90=238.7 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|hints_no_exact|8000|1|2|1`: n=1, avg TTFT=510.7 ms, p50=510.7 ms, p90=510.7 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|hints_no_exact|8000|1|3|1`: n=1, avg TTFT=781.1 ms, p50=781.1 ms, p90=781.1 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|hints_no_exact|8000|1|4|1`: n=1, avg TTFT=1002.5 ms, p50=1002.5 ms, p90=1002.5 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|hints_no_exact|8000|1|5|1`: n=1, avg TTFT=1300.4 ms, p50=1300.4 ms, p90=1300.4 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|placeholder_knn_reuse|8000|1|1|1`: n=1, avg TTFT=79.5 ms, p50=79.5 ms, p90=79.5 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|placeholder_knn_reuse|8000|1|2|1`: n=1, avg TTFT=196.8 ms, p50=196.8 ms, p90=196.8 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|placeholder_knn_reuse|8000|1|3|1`: n=1, avg TTFT=208.3 ms, p50=208.3 ms, p90=208.3 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|placeholder_knn_reuse|8000|1|4|1`: n=1, avg TTFT=511.8 ms, p50=511.8 ms, p90=511.8 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|placeholder_knn_reuse|8000|1|5|1`: n=1, avg TTFT=1123.2 ms, p50=1123.2 ms, p90=1123.2 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|prefix_cache_only|8000|1|1|1`: n=1, avg TTFT=252.4 ms, p50=252.4 ms, p90=252.4 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|prefix_cache_only|8000|1|2|1`: n=1, avg TTFT=287.9 ms, p50=287.9 ms, p90=287.9 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|prefix_cache_only|8000|1|3|1`: n=1, avg TTFT=341.3 ms, p50=341.3 ms, p90=341.3 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|prefix_cache_only|8000|1|4|1`: n=1, avg TTFT=368.0 ms, p50=368.0 ms, p90=368.0 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|prefix_cache_only|8000|1|5|1`: n=1, avg TTFT=434.7 ms, p50=434.7 ms, p90=434.7 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling|exact_reuse_no_hints|8000|1|1|1`: n=1, avg TTFT=105.6 ms, p50=105.6 ms, p90=105.6 ms, exact hit=1.00, device hit=1.00, consumed=1.00, protected=0.0, F1=1.0000, status={'consumed': 1}
- `agent_scaling|exact_reuse_no_hints|8000|1|2|1`: n=2, avg TTFT=100.1 ms, p50=99.7 ms, p90=100.6 ms, exact hit=1.00, device hit=1.00, consumed=0.50, protected=0.0, F1=1.0000, status={'consumed': 1, 'device_hit_without_consumed': 1}
- `agent_scaling|exact_reuse_no_hints|8000|1|3|1`: n=3, avg TTFT=88.8 ms, p50=83.6 ms, p90=102.3 ms, exact hit=1.00, device hit=1.00, consumed=0.33, protected=0.0, F1=1.0000, status={'consumed': 1, 'device_hit_without_consumed': 2}
- `agent_scaling|exact_reuse_no_hints|8000|1|4|1`: n=4, avg TTFT=342.4 ms, p50=592.9 ms, p90=603.7 ms, exact hit=1.00, device hit=0.50, consumed=0.25, protected=0.0, F1=1.0000, status={'consumed': 1, 'no_fast_path': 2, 'device_hit_without_consumed': 1}
- `agent_scaling|exact_reuse_no_hints|8000|1|5|1`: n=5, avg TTFT=524.6 ms, p50=787.8 ms, p90=830.6 ms, exact hit=1.00, device hit=0.40, consumed=0.20, protected=0.0, F1=1.0000, status={'no_fast_path': 3, 'consumed': 1, 'device_hit_without_consumed': 1}
- `agent_scaling|exact_reuse_plus_code_hints|8000|1|1|1`: n=1, avg TTFT=110.1 ms, p50=110.1 ms, p90=110.1 ms, exact hit=1.00, device hit=1.00, consumed=1.00, protected=4490.0, F1=1.0000, status={'consumed': 1}
- `agent_scaling|exact_reuse_plus_code_hints|8000|1|2|1`: n=2, avg TTFT=109.1 ms, p50=75.6 ms, p90=142.6 ms, exact hit=1.00, device hit=1.00, consumed=1.00, protected=4490.0, F1=1.0000, status={'consumed': 2}
- `agent_scaling|exact_reuse_plus_code_hints|8000|1|3|1`: n=3, avg TTFT=119.5 ms, p50=124.7 ms, p90=137.4 ms, exact hit=1.00, device hit=1.00, consumed=1.00, protected=4490.0, F1=1.0000, status={'consumed': 3}
- `agent_scaling|exact_reuse_plus_code_hints|8000|1|4|1`: n=4, avg TTFT=378.7 ms, p50=602.7 ms, p90=719.2 ms, exact hit=1.00, device hit=1.00, consumed=0.50, protected=4490.0, F1=1.0000, status={'consumed': 2, 'protected_not_consumed:no_anchor_match': 2}
- `agent_scaling|exact_reuse_plus_code_hints|8000|1|5|1`: n=5, avg TTFT=676.9 ms, p50=812.9 ms, p90=835.8 ms, exact hit=1.00, device hit=1.00, consumed=0.20, protected=4490.0, F1=1.0000, status={'consumed': 1, 'protected_not_consumed:no_anchor_match': 4}
- `agent_scaling|hints_no_exact|8000|1|1|1`: n=1, avg TTFT=238.7 ms, p50=238.7 ms, p90=238.7 ms, exact hit=0.00, device hit=1.00, consumed=0.00, protected=2245.0, F1=1.0000, status={'protected_not_consumed:no_anchor_match': 1}
- `agent_scaling|hints_no_exact|8000|1|2|1`: n=2, avg TTFT=255.4 ms, p50=254.0 ms, p90=256.7 ms, exact hit=0.00, device hit=1.00, consumed=0.00, protected=2245.0, F1=1.0000, status={'protected_not_consumed:no_anchor_match': 2}
- `agent_scaling|hints_no_exact|8000|1|3|1`: n=3, avg TTFT=260.4 ms, p50=262.9 ms, p90=263.5 ms, exact hit=0.00, device hit=1.00, consumed=0.00, protected=2245.0, F1=1.0000, status={'protected_not_consumed:no_anchor_match': 3}
- `agent_scaling|hints_no_exact|8000|1|4|1`: n=4, avg TTFT=250.6 ms, p50=251.2 ms, p90=251.4 ms, exact hit=0.00, device hit=1.00, consumed=0.00, protected=2245.0, F1=1.0000, status={'protected_not_consumed:no_anchor_match': 4}
- `agent_scaling|hints_no_exact|8000|1|5|1`: n=5, avg TTFT=260.1 ms, p50=256.8 ms, p90=275.1 ms, exact hit=0.00, device hit=1.00, consumed=0.00, protected=2245.0, F1=1.0000, status={'protected_not_consumed:no_anchor_match': 5}
- `agent_scaling|placeholder_knn_reuse|8000|1|1|1`: n=1, avg TTFT=79.5 ms, p50=79.5 ms, p90=79.5 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 1}
- `agent_scaling|placeholder_knn_reuse|8000|1|2|1`: n=2, avg TTFT=98.4 ms, p50=84.1 ms, p90=112.6 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 2}
- `agent_scaling|placeholder_knn_reuse|8000|1|3|1`: n=3, avg TTFT=69.4 ms, p50=67.5 ms, p90=76.0 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 3}
- `agent_scaling|placeholder_knn_reuse|8000|1|4|1`: n=4, avg TTFT=127.9 ms, p50=103.0 ms, p90=250.4 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 4}
- `agent_scaling|placeholder_knn_reuse|8000|1|5|1`: n=5, avg TTFT=224.6 ms, p50=259.6 ms, p90=272.4 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 5}
- `agent_scaling|prefix_cache_only|8000|1|1|1`: n=1, avg TTFT=252.4 ms, p50=252.4 ms, p90=252.4 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 1}
- `agent_scaling|prefix_cache_only|8000|1|2|1`: n=2, avg TTFT=144.0 ms, p50=42.5 ms, p90=245.4 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 2}
- `agent_scaling|prefix_cache_only|8000|1|3|1`: n=3, avg TTFT=113.8 ms, p50=47.3 ms, p90=247.8 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 3}
- `agent_scaling|prefix_cache_only|8000|1|4|1`: n=4, avg TTFT=92.0 ms, p50=41.7 ms, p90=246.9 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 4}
- `agent_scaling|prefix_cache_only|8000|1|5|1`: n=5, avg TTFT=86.9 ms, p50=46.9 ms, p90=249.4 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 5}

## Exact-Reuse Speedup vs Prefix

- `('agent_scaling', '8000', '1', '1', '1')`: p50=2.29x, p90=2.29x (prefix=252.4/252.4 ms, exact+hints=110.1/110.1 ms)
- `('agent_scaling', '8000', '1', '2', '1')`: p50=0.56x, p90=1.72x (prefix=42.5/245.4 ms, exact+hints=75.6/142.6 ms)
- `('agent_scaling', '8000', '1', '3', '1')`: p50=0.38x, p90=1.80x (prefix=47.3/247.8 ms, exact+hints=124.7/137.4 ms)
- `('agent_scaling', '8000', '1', '4', '1')`: p50=0.07x, p90=0.34x (prefix=41.7/246.9 ms, exact+hints=602.7/719.2 ms)
- `('agent_scaling', '8000', '1', '5', '1')`: p50=0.06x, p90=0.30x (prefix=46.9/249.4 ms, exact+hints=812.9/835.8 ms)
- `('agent_scaling_workflow', '8000', '1', '1', '1')`: p50=2.29x, p90=2.29x (prefix=252.4/252.4 ms, exact+hints=110.1/110.1 ms)
- `('agent_scaling_workflow', '8000', '1', '2', '1')`: p50=1.32x, p90=1.32x (prefix=287.9/287.9 ms, exact+hints=218.2/218.2 ms)
- `('agent_scaling_workflow', '8000', '1', '3', '1')`: p50=0.95x, p90=0.95x (prefix=341.3/341.3 ms, exact+hints=358.4/358.4 ms)
- `('agent_scaling_workflow', '8000', '1', '4', '1')`: p50=0.24x, p90=0.24x (prefix=368.0/368.0 ms, exact+hints=1514.7/1514.7 ms)
- `('agent_scaling_workflow', '8000', '1', '5', '1')`: p50=0.13x, p90=0.13x (prefix=434.7/434.7 ms, exact+hints=3384.6/3384.6 ms)
