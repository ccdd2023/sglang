# KVCOMM TTFT Stress Report

This run is prefill/TTFT dominant and is intended to complement the realistic 100-case E2E table.

## Output Schema

`ttft_ms`, `e2e_ms`, `prompt_tokens`, `cached_tokens`, `cached_ratio`, `exact_hit`, `match_reason`, `matched_content_signature`, `output_exact_match_vs_baseline`, `output_token_f1_vs_baseline`, `mode`, `case_id`, `agent_id`, `segment_count`, `max_file_chars`.

## Row Counts

- agent_scaling: 24
- agent_scaling_workflow: 12

## Summary Groups

- `agent_scaling_workflow|exact_reuse_no_hints|8000|1|2|1`: n=3, avg TTFT=222.5 ms, p50=124.2 ms, p90=421.5 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 3}
- `agent_scaling_workflow|exact_reuse_plus_code_hints|8000|1|2|1`: n=3, avg TTFT=247.3 ms, p50=121.8 ms, p90=504.6 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 3}
- `agent_scaling_workflow|hints_no_exact|8000|1|2|1`: n=3, avg TTFT=677.5 ms, p50=687.7 ms, p90=853.7 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 3}
- `agent_scaling_workflow|prefix_cache_only|8000|1|2|1`: n=3, avg TTFT=631.4 ms, p50=490.2 ms, p90=917.5 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 3}
- `agent_scaling|exact_reuse_no_hints|8000|1|2|1`: n=6, avg TTFT=111.3 ms, p50=64.2 ms, p90=73.5 ms, exact hit=1.00, device hit=0.83, consumed=0.17, protected=0.0, F1=1.0000, status={'consumed': 1, 'device_hit_without_consumed': 4, 'no_fast_path': 1}
- `agent_scaling|exact_reuse_plus_code_hints|8000|1|2|1`: n=6, avg TTFT=123.7 ms, p50=57.8 ms, p90=65.9 ms, exact hit=1.00, device hit=0.83, consumed=0.00, protected=2245.0, F1=1.0000, status={'protected_not_consumed:position_mismatch': 5, 'protected_not_consumed:no_anchor_match': 1}
- `agent_scaling|hints_no_exact|8000|1|2|1`: n=6, avg TTFT=338.8 ms, p50=247.8 ms, p90=449.5 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 6}
- `agent_scaling|prefix_cache_only|8000|1|2|1`: n=6, avg TTFT=315.7 ms, p50=246.9 ms, p90=424.2 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 6}

## Exact-Reuse Speedup vs Prefix

- `('agent_scaling', '8000', '1', '2', '1')`: p50=4.27x, p90=6.44x (prefix=246.9/424.2 ms, exact+hints=57.8/65.9 ms)
- `('agent_scaling_workflow', '8000', '1', '2', '1')`: p50=4.02x, p90=1.82x (prefix=490.2/917.5 ms, exact+hints=121.8/504.6 ms)
