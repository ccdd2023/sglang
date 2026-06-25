# KVCOMM TTFT Stress Report

This run is prefill/TTFT dominant and is intended to complement the realistic 100-case E2E table.

## Output Schema

`ttft_ms`, `e2e_ms`, `prompt_tokens`, `cached_tokens`, `cached_ratio`, `exact_hit`, `match_reason`, `matched_content_signature`, `output_exact_match_vs_baseline`, `output_token_f1_vs_baseline`, `mode`, `case_id`, `agent_id`, `segment_count`, `max_file_chars`.

## Row Counts

- agent_scaling: 10
- agent_scaling_workflow: 5

## Summary Groups

- `agent_scaling_workflow|exact_reuse_no_hints|8000|1|2|1`: n=1, avg TTFT=194.4 ms, p50=194.4 ms, p90=194.4 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_plus_code_hints|8000|1|2|1`: n=1, avg TTFT=231.0 ms, p50=231.0 ms, p90=231.0 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|hints_no_exact|8000|1|2|1`: n=1, avg TTFT=507.4 ms, p50=507.4 ms, p90=507.4 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|placeholder_knn_reuse|8000|1|2|1`: n=1, avg TTFT=369.4 ms, p50=369.4 ms, p90=369.4 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|prefix_cache_only|8000|1|2|1`: n=1, avg TTFT=507.7 ms, p50=507.7 ms, p90=507.7 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling|exact_reuse_no_hints|8000|1|2|1`: n=2, avg TTFT=97.2 ms, p50=90.3 ms, p90=104.1 ms, exact hit=1.00, device hit=1.00, consumed=0.50, protected=0.0, F1=1.0000, status={'consumed': 1, 'device_hit_without_consumed': 1}
- `agent_scaling|exact_reuse_plus_code_hints|8000|1|2|1`: n=2, avg TTFT=115.5 ms, p50=111.3 ms, p90=119.7 ms, exact hit=1.00, device hit=1.00, consumed=1.00, protected=4490.0, F1=1.0000, status={'consumed': 2}
- `agent_scaling|hints_no_exact|8000|1|2|1`: n=2, avg TTFT=253.7 ms, p50=251.4 ms, p90=256.0 ms, exact hit=0.00, device hit=1.00, consumed=0.00, protected=2245.0, F1=1.0000, status={'protected_not_consumed:no_anchor_match': 2}
- `agent_scaling|placeholder_knn_reuse|8000|1|2|1`: n=2, avg TTFT=184.7 ms, p50=89.6 ms, p90=279.9 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 2}
- `agent_scaling|prefix_cache_only|8000|1|2|1`: n=2, avg TTFT=253.9 ms, p50=248.5 ms, p90=259.2 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 2}

## Exact-Reuse Speedup vs Prefix

- `('agent_scaling', '8000', '1', '2', '1')`: p50=2.23x, p90=2.17x (prefix=248.5/259.2 ms, exact+hints=111.3/119.7 ms)
- `('agent_scaling_workflow', '8000', '1', '2', '1')`: p50=2.20x, p90=2.20x (prefix=507.7/507.7 ms, exact+hints=231.0/231.0 ms)
