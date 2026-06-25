# KVCOMM TTFT Stress Report

This run is prefill/TTFT dominant and is intended to complement the realistic 100-case E2E table.

## Output Schema

`ttft_ms`, `e2e_ms`, `prompt_tokens`, `cached_tokens`, `cached_ratio`, `exact_hit`, `match_reason`, `matched_content_signature`, `output_exact_match_vs_baseline`, `output_token_f1_vs_baseline`, `mode`, `case_id`, `agent_id`, `segment_count`, `max_file_chars`.

## Row Counts

- agent_scaling: 25
- agent_scaling_workflow: 5

## Summary Groups

- `agent_scaling_workflow|exact_reuse_no_hints|8000|1|5|1`: n=1, avg TTFT=430.0 ms, p50=430.0 ms, p90=430.0 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_plus_code_hints|8000|1|5|1`: n=1, avg TTFT=471.1 ms, p50=471.1 ms, p90=471.1 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|hints_no_exact|8000|1|5|1`: n=1, avg TTFT=1274.9 ms, p50=1274.9 ms, p90=1274.9 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|placeholder_knn_reuse|8000|1|5|1`: n=1, avg TTFT=615.6 ms, p50=615.6 ms, p90=615.6 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|prefix_cache_only|8000|1|5|1`: n=1, avg TTFT=1260.4 ms, p50=1260.4 ms, p90=1260.4 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling|exact_reuse_no_hints|8000|1|5|1`: n=5, avg TTFT=86.0 ms, p50=79.6 ms, p90=106.3 ms, exact hit=1.00, device hit=1.00, consumed=0.20, protected=0.0, F1=1.0000, status={'consumed': 1, 'device_hit_without_consumed': 4}
- `agent_scaling|exact_reuse_plus_code_hints|8000|1|5|1`: n=5, avg TTFT=94.2 ms, p50=91.1 ms, p90=115.7 ms, exact hit=1.00, device hit=1.00, consumed=1.00, protected=4490.0, F1=1.0000, status={'consumed': 5}
- `agent_scaling|hints_no_exact|8000|1|5|1`: n=5, avg TTFT=255.0 ms, p50=254.9 ms, p90=255.6 ms, exact hit=0.00, device hit=1.00, consumed=0.00, protected=2245.0, F1=1.0000, status={'protected_not_consumed:no_anchor_match': 5}
- `agent_scaling|placeholder_knn_reuse|8000|1|5|1`: n=5, avg TTFT=123.1 ms, p50=77.9 ms, p90=281.3 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 5}
- `agent_scaling|prefix_cache_only|8000|1|5|1`: n=5, avg TTFT=252.1 ms, p50=251.7 ms, p90=256.8 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 5}

## Exact-Reuse Speedup vs Prefix

- `('agent_scaling', '8000', '1', '5', '1')`: p50=2.76x, p90=2.22x (prefix=251.7/256.8 ms, exact+hints=91.1/115.7 ms)
- `('agent_scaling_workflow', '8000', '1', '5', '1')`: p50=2.68x, p90=2.68x (prefix=1260.4/1260.4 ms, exact+hints=471.1/471.1 ms)
