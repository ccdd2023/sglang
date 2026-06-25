# KVCOMM TTFT Stress Report

This run is prefill/TTFT dominant and is intended to complement the realistic 100-case E2E table.

## Output Schema

`ttft_ms`, `e2e_ms`, `prompt_tokens`, `cached_tokens`, `cached_ratio`, `exact_hit`, `match_reason`, `matched_content_signature`, `output_exact_match_vs_baseline`, `output_token_f1_vs_baseline`, `mode`, `case_id`, `agent_id`, `segment_count`, `max_file_chars`.

## Row Counts

- agent_scaling: 75
- agent_scaling_workflow: 25

## Summary Groups

- `agent_scaling_workflow|exact_reuse_no_hints|2000|1|1|1`: n=1, avg TTFT=70.7 ms, p50=70.7 ms, p90=70.7 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_no_hints|2000|1|2|1`: n=1, avg TTFT=145.8 ms, p50=145.8 ms, p90=145.8 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_no_hints|2000|1|3|1`: n=1, avg TTFT=303.9 ms, p50=303.9 ms, p90=303.9 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_no_hints|2000|1|4|1`: n=1, avg TTFT=344.1 ms, p50=344.1 ms, p90=344.1 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_no_hints|2000|1|5|1`: n=1, avg TTFT=450.8 ms, p50=450.8 ms, p90=450.8 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_plus_code_hints|2000|1|1|1`: n=1, avg TTFT=76.7 ms, p50=76.7 ms, p90=76.7 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_plus_code_hints|2000|1|2|1`: n=1, avg TTFT=140.9 ms, p50=140.9 ms, p90=140.9 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_plus_code_hints|2000|1|3|1`: n=1, avg TTFT=270.3 ms, p50=270.3 ms, p90=270.3 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_plus_code_hints|2000|1|4|1`: n=1, avg TTFT=335.0 ms, p50=335.0 ms, p90=335.0 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_plus_code_hints|2000|1|5|1`: n=1, avg TTFT=455.3 ms, p50=455.3 ms, p90=455.3 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|hints_no_exact|2000|1|1|1`: n=1, avg TTFT=87.0 ms, p50=87.0 ms, p90=87.0 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|hints_no_exact|2000|1|2|1`: n=1, avg TTFT=179.7 ms, p50=179.7 ms, p90=179.7 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|hints_no_exact|2000|1|3|1`: n=1, avg TTFT=267.7 ms, p50=267.7 ms, p90=267.7 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|hints_no_exact|2000|1|4|1`: n=1, avg TTFT=361.5 ms, p50=361.5 ms, p90=361.5 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|hints_no_exact|2000|1|5|1`: n=1, avg TTFT=472.2 ms, p50=472.2 ms, p90=472.2 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|placeholder_knn_reuse|2000|1|1|1`: n=1, avg TTFT=58.4 ms, p50=58.4 ms, p90=58.4 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|placeholder_knn_reuse|2000|1|2|1`: n=1, avg TTFT=159.5 ms, p50=159.5 ms, p90=159.5 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|placeholder_knn_reuse|2000|1|3|1`: n=1, avg TTFT=241.3 ms, p50=241.3 ms, p90=241.3 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|placeholder_knn_reuse|2000|1|4|1`: n=1, avg TTFT=295.2 ms, p50=295.2 ms, p90=295.2 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|placeholder_knn_reuse|2000|1|5|1`: n=1, avg TTFT=348.7 ms, p50=348.7 ms, p90=348.7 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|prefix_cache_only|2000|1|1|1`: n=1, avg TTFT=89.4 ms, p50=89.4 ms, p90=89.4 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|prefix_cache_only|2000|1|2|1`: n=1, avg TTFT=127.4 ms, p50=127.4 ms, p90=127.4 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|prefix_cache_only|2000|1|3|1`: n=1, avg TTFT=174.2 ms, p50=174.2 ms, p90=174.2 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|prefix_cache_only|2000|1|4|1`: n=1, avg TTFT=213.1 ms, p50=213.1 ms, p90=213.1 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|prefix_cache_only|2000|1|5|1`: n=1, avg TTFT=254.4 ms, p50=254.4 ms, p90=254.4 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling|exact_reuse_no_hints|2000|1|1|1`: n=1, avg TTFT=70.7 ms, p50=70.7 ms, p90=70.7 ms, exact hit=1.00, device hit=1.00, consumed=1.00, protected=0.0, F1=1.0000, status={'consumed': 1}
- `agent_scaling|exact_reuse_no_hints|2000|1|2|1`: n=2, avg TTFT=72.9 ms, p50=68.0 ms, p90=77.8 ms, exact hit=1.00, device hit=1.00, consumed=0.50, protected=0.0, F1=1.0000, status={'consumed': 1, 'device_hit_without_consumed': 1}
- `agent_scaling|exact_reuse_no_hints|2000|1|3|1`: n=3, avg TTFT=101.3 ms, p50=86.5 ms, p90=144.7 ms, exact hit=1.00, device hit=1.00, consumed=0.33, protected=0.0, F1=1.0000, status={'consumed': 1, 'device_hit_without_consumed': 2}
- `agent_scaling|exact_reuse_no_hints|2000|1|4|1`: n=4, avg TTFT=86.0 ms, p50=89.0 ms, p90=101.9 ms, exact hit=1.00, device hit=1.00, consumed=0.25, protected=0.0, F1=1.0000, status={'consumed': 1, 'device_hit_without_consumed': 3}
- `agent_scaling|exact_reuse_no_hints|2000|1|5|1`: n=5, avg TTFT=90.2 ms, p50=88.4 ms, p90=105.3 ms, exact hit=1.00, device hit=1.00, consumed=0.20, protected=0.0, F1=1.0000, status={'consumed': 1, 'device_hit_without_consumed': 4}
- `agent_scaling|exact_reuse_plus_code_hints|2000|1|1|1`: n=1, avg TTFT=76.7 ms, p50=76.7 ms, p90=76.7 ms, exact hit=1.00, device hit=1.00, consumed=1.00, protected=1234.0, F1=1.0000, status={'consumed': 1}
- `agent_scaling|exact_reuse_plus_code_hints|2000|1|2|1`: n=2, avg TTFT=70.5 ms, p50=69.4 ms, p90=71.5 ms, exact hit=1.00, device hit=1.00, consumed=1.00, protected=1234.0, F1=1.0000, status={'consumed': 2}
- `agent_scaling|exact_reuse_plus_code_hints|2000|1|3|1`: n=3, avg TTFT=90.1 ms, p50=94.0 ms, p90=96.7 ms, exact hit=1.00, device hit=1.00, consumed=1.00, protected=1234.0, F1=1.0000, status={'consumed': 3}
- `agent_scaling|exact_reuse_plus_code_hints|2000|1|4|1`: n=4, avg TTFT=83.7 ms, p50=87.1 ms, p90=91.6 ms, exact hit=1.00, device hit=1.00, consumed=1.00, protected=1234.0, F1=1.0000, status={'consumed': 4}
- `agent_scaling|exact_reuse_plus_code_hints|2000|1|5|1`: n=5, avg TTFT=91.1 ms, p50=95.1 ms, p90=105.8 ms, exact hit=1.00, device hit=1.00, consumed=1.00, protected=1234.0, F1=1.0000, status={'consumed': 5}
- `agent_scaling|hints_no_exact|2000|1|1|1`: n=1, avg TTFT=87.0 ms, p50=87.0 ms, p90=87.0 ms, exact hit=0.00, device hit=1.00, consumed=0.00, protected=617.0, F1=1.0000, status={'protected_not_consumed:no_anchor_match': 1}
- `agent_scaling|hints_no_exact|2000|1|2|1`: n=2, avg TTFT=89.9 ms, p50=89.8 ms, p90=89.9 ms, exact hit=0.00, device hit=1.00, consumed=0.00, protected=617.0, F1=1.0000, status={'protected_not_consumed:no_anchor_match': 2}
- `agent_scaling|hints_no_exact|2000|1|3|1`: n=3, avg TTFT=89.2 ms, p50=89.3 ms, p90=89.3 ms, exact hit=0.00, device hit=1.00, consumed=0.00, protected=617.0, F1=1.0000, status={'protected_not_consumed:no_anchor_match': 3}
- `agent_scaling|hints_no_exact|2000|1|4|1`: n=4, avg TTFT=90.4 ms, p50=91.7 ms, p90=92.5 ms, exact hit=0.00, device hit=1.00, consumed=0.00, protected=617.0, F1=1.0000, status={'protected_not_consumed:no_anchor_match': 4}
- `agent_scaling|hints_no_exact|2000|1|5|1`: n=5, avg TTFT=94.4 ms, p50=93.3 ms, p90=96.7 ms, exact hit=0.00, device hit=1.00, consumed=0.00, protected=617.0, F1=1.0000, status={'protected_not_consumed:no_anchor_match': 5}
- `agent_scaling|placeholder_knn_reuse|2000|1|1|1`: n=1, avg TTFT=58.4 ms, p50=58.4 ms, p90=58.4 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 1}
- `agent_scaling|placeholder_knn_reuse|2000|1|2|1`: n=2, avg TTFT=79.8 ms, p50=56.4 ms, p90=103.1 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 2}
- `agent_scaling|placeholder_knn_reuse|2000|1|3|1`: n=3, avg TTFT=80.4 ms, p50=67.5 ms, p90=118.6 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 3}
- `agent_scaling|placeholder_knn_reuse|2000|1|4|1`: n=4, avg TTFT=73.8 ms, p50=58.4 ms, p90=123.0 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 4}
- `agent_scaling|placeholder_knn_reuse|2000|1|5|1`: n=5, avg TTFT=69.7 ms, p50=60.2 ms, p90=113.8 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 5}
- `agent_scaling|prefix_cache_only|2000|1|1|1`: n=1, avg TTFT=89.4 ms, p50=89.4 ms, p90=89.4 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 1}
- `agent_scaling|prefix_cache_only|2000|1|2|1`: n=2, avg TTFT=63.7 ms, p50=36.0 ms, p90=91.4 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 2}
- `agent_scaling|prefix_cache_only|2000|1|3|1`: n=3, avg TTFT=58.1 ms, p50=45.2 ms, p90=90.9 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 3}
- `agent_scaling|prefix_cache_only|2000|1|4|1`: n=4, avg TTFT=53.3 ms, p50=41.4 ms, p90=89.7 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 4}
- `agent_scaling|prefix_cache_only|2000|1|5|1`: n=5, avg TTFT=50.9 ms, p50=41.2 ms, p90=90.1 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 5}

## Exact-Reuse Speedup vs Prefix

- `('agent_scaling', '2000', '1', '1', '1')`: p50=1.17x, p90=1.17x (prefix=89.4/89.4 ms, exact+hints=76.7/76.7 ms)
- `('agent_scaling', '2000', '1', '2', '1')`: p50=0.52x, p90=1.28x (prefix=36.0/91.4 ms, exact+hints=69.4/71.5 ms)
- `('agent_scaling', '2000', '1', '3', '1')`: p50=0.48x, p90=0.94x (prefix=45.2/90.9 ms, exact+hints=94.0/96.7 ms)
- `('agent_scaling', '2000', '1', '4', '1')`: p50=0.47x, p90=0.98x (prefix=41.4/89.7 ms, exact+hints=87.1/91.6 ms)
- `('agent_scaling', '2000', '1', '5', '1')`: p50=0.43x, p90=0.85x (prefix=41.2/90.1 ms, exact+hints=95.1/105.8 ms)
- `('agent_scaling_workflow', '2000', '1', '1', '1')`: p50=1.17x, p90=1.17x (prefix=89.4/89.4 ms, exact+hints=76.7/76.7 ms)
- `('agent_scaling_workflow', '2000', '1', '2', '1')`: p50=0.90x, p90=0.90x (prefix=127.4/127.4 ms, exact+hints=140.9/140.9 ms)
- `('agent_scaling_workflow', '2000', '1', '3', '1')`: p50=0.64x, p90=0.64x (prefix=174.2/174.2 ms, exact+hints=270.3/270.3 ms)
- `('agent_scaling_workflow', '2000', '1', '4', '1')`: p50=0.64x, p90=0.64x (prefix=213.1/213.1 ms, exact+hints=335.0/335.0 ms)
- `('agent_scaling_workflow', '2000', '1', '5', '1')`: p50=0.56x, p90=0.56x (prefix=254.4/254.4 ms, exact+hints=455.3/455.3 ms)
