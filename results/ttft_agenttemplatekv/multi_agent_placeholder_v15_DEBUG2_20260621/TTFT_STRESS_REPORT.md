# KVCOMM TTFT Stress Report

This run is prefill/TTFT dominant and is intended to complement the realistic 100-case E2E table.

## Output Schema

`ttft_ms`, `e2e_ms`, `prompt_tokens`, `cached_tokens`, `cached_ratio`, `exact_hit`, `match_reason`, `matched_content_signature`, `output_exact_match_vs_baseline`, `output_token_f1_vs_baseline`, `mode`, `case_id`, `agent_id`, `segment_count`, `max_file_chars`.

## Row Counts

- agent_scaling: 25
- agent_scaling_workflow: 5

## Summary Groups

- `agent_scaling_workflow|exact_reuse_no_hints|8000|1|5|1`: n=1, avg TTFT=399.6 ms, p50=399.6 ms, p90=399.6 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_plus_code_hints|8000|1|5|1`: n=1, avg TTFT=445.0 ms, p50=445.0 ms, p90=445.0 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|hints_no_exact|8000|1|5|1`: n=1, avg TTFT=1274.0 ms, p50=1274.0 ms, p90=1274.0 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|placeholder_knn_reuse|8000|1|5|1`: n=1, avg TTFT=585.5 ms, p50=585.5 ms, p90=585.5 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|prefix_cache_only|8000|1|5|1`: n=1, avg TTFT=1257.2 ms, p50=1257.2 ms, p90=1257.2 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling|exact_reuse_no_hints|8000|1|5|1`: n=5, avg TTFT=79.9 ms, p50=76.0 ms, p90=90.8 ms, exact hit=1.00, device hit=1.00, consumed=0.20, protected=0.0, F1=1.0000, status={'consumed': 1, 'device_hit_without_consumed': 4}
- `agent_scaling|exact_reuse_plus_code_hints|8000|1|5|1`: n=5, avg TTFT=89.0 ms, p50=94.0 ms, p90=95.1 ms, exact hit=1.00, device hit=1.00, consumed=1.00, protected=4490.0, F1=1.0000, status={'consumed': 5}
- `agent_scaling|hints_no_exact|8000|1|5|1`: n=5, avg TTFT=254.8 ms, p50=254.8 ms, p90=255.6 ms, exact hit=0.00, device hit=1.00, consumed=0.00, protected=2245.0, F1=1.0000, status={'protected_not_consumed:no_anchor_match': 5}
- `agent_scaling|placeholder_knn_reuse|8000|1|5|1`: n=5, avg TTFT=117.1 ms, p50=76.4 ms, p90=281.3 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 5}
- `agent_scaling|prefix_cache_only|8000|1|5|1`: n=5, avg TTFT=251.4 ms, p50=250.2 ms, p90=255.9 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 5}

## Exact-Reuse Speedup vs Prefix

- `('agent_scaling', '8000', '1', '5', '1')`: p50=2.66x, p90=2.69x (prefix=250.2/255.9 ms, exact+hints=94.0/95.1 ms)
- `('agent_scaling_workflow', '8000', '1', '5', '1')`: p50=2.83x, p90=2.83x (prefix=1257.2/1257.2 ms, exact+hints=445.0/445.0 ms)
