# KVCOMM TTFT Stress Report

This run is prefill/TTFT dominant and is intended to complement the realistic 100-case E2E table.

## Output Schema

`ttft_ms`, `e2e_ms`, `prompt_tokens`, `cached_tokens`, `cached_ratio`, `exact_hit`, `match_reason`, `matched_content_signature`, `output_exact_match_vs_baseline`, `output_token_f1_vs_baseline`, `mode`, `case_id`, `agent_id`, `segment_count`, `max_file_chars`.

## Row Counts

- agent_scaling: 10
- agent_scaling_workflow: 5

## Summary Groups

- `agent_scaling_workflow|exact_reuse_no_hints|8000|1|2|1`: n=1, avg TTFT=222.3 ms, p50=222.3 ms, p90=222.3 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_plus_code_hints|8000|1|2|1`: n=1, avg TTFT=194.2 ms, p50=194.2 ms, p90=194.2 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|hints_no_exact|8000|1|2|1`: n=1, avg TTFT=512.7 ms, p50=512.7 ms, p90=512.7 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|placeholder_knn_reuse|8000|1|2|1`: n=1, avg TTFT=553.0 ms, p50=553.0 ms, p90=553.0 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|prefix_cache_only|8000|1|2|1`: n=1, avg TTFT=510.9 ms, p50=510.9 ms, p90=510.9 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling|exact_reuse_no_hints|8000|1|2|1`: n=2, avg TTFT=111.2 ms, p50=108.5 ms, p90=113.8 ms, exact hit=1.00, device hit=1.00, consumed=0.50, protected=0.0, F1=1.0000, status={'consumed': 1, 'device_hit_without_consumed': 1}
- `agent_scaling|exact_reuse_plus_code_hints|8000|1|2|1`: n=2, avg TTFT=97.1 ms, p50=95.7 ms, p90=98.4 ms, exact hit=1.00, device hit=1.00, consumed=1.00, protected=4490.0, F1=1.0000, status={'consumed': 2}
- `agent_scaling|hints_no_exact|8000|1|2|1`: n=2, avg TTFT=256.3 ms, p50=255.5 ms, p90=257.1 ms, exact hit=0.00, device hit=1.00, consumed=0.00, protected=2245.0, F1=1.0000, status={'protected_not_consumed:no_anchor_match': 2}
- `agent_scaling|placeholder_knn_reuse|8000|1|2|1`: n=2, avg TTFT=276.5 ms, p50=271.3 ms, p90=281.7 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 2}
- `agent_scaling|prefix_cache_only|8000|1|2|1`: n=2, avg TTFT=255.5 ms, p50=252.4 ms, p90=258.5 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 2}

## Exact-Reuse Speedup vs Prefix

- `('agent_scaling', '8000', '1', '2', '1')`: p50=2.64x, p90=2.63x (prefix=252.4/258.5 ms, exact+hints=95.7/98.4 ms)
- `('agent_scaling_workflow', '8000', '1', '2', '1')`: p50=2.63x, p90=2.63x (prefix=510.9/510.9 ms, exact+hints=194.2/194.2 ms)
