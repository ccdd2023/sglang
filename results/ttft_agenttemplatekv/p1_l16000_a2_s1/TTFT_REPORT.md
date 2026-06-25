# KVCOMM TTFT Stress Report

This run is prefill/TTFT dominant and is intended to complement the realistic 100-case E2E table.

## Output Schema

`ttft_ms`, `e2e_ms`, `prompt_tokens`, `cached_tokens`, `cached_ratio`, `exact_hit`, `match_reason`, `matched_content_signature`, `output_exact_match_vs_baseline`, `output_token_f1_vs_baseline`, `mode`, `case_id`, `agent_id`, `segment_count`, `max_file_chars`.

## Row Counts

- agent_scaling: 40
- agent_scaling_workflow: 20

## Summary Groups

- `agent_scaling_workflow|exact_reuse_no_hints|16000|1|2|1`: n=5, avg TTFT=750.5 ms, p50=693.4 ms, p90=1540.1 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 5}
- `agent_scaling_workflow|exact_reuse_plus_code_hints|16000|1|2|1`: n=5, avg TTFT=1675.3 ms, p50=2010.6 ms, p90=2117.8 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 5}
- `agent_scaling_workflow|hints_no_exact|16000|1|2|1`: n=5, avg TTFT=1813.6 ms, p50=1641.3 ms, p90=2407.7 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 5}
- `agent_scaling_workflow|prefix_cache_only|16000|1|2|1`: n=5, avg TTFT=1973.1 ms, p50=2176.8 ms, p90=2229.2 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 5}
- `agent_scaling|exact_reuse_no_hints|16000|1|2|1`: n=10, avg TTFT=375.2 ms, p50=77.1 ms, p90=627.5 ms, exact hit=1.00, device hit=0.50, consumed=0.10, protected=0.0, F1=1.0000, status={'consumed': 1, 'device_hit_without_consumed': 4, 'no_fast_path': 5}
- `agent_scaling|exact_reuse_plus_code_hints|16000|1|2|1`: n=10, avg TTFT=837.7 ms, p50=888.0 ms, p90=1188.7 ms, exact hit=1.00, device hit=0.20, consumed=0.00, protected=6172.0, F1=1.0000, status={'anchor_reuse_device_hit_consumed_counter_gap': 2, 'protected_not_consumed:no_anchor_match': 8}
- `agent_scaling|hints_no_exact|16000|1|2|1`: n=10, avg TTFT=906.8 ms, p50=911.5 ms, p90=1145.6 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 10}
- `agent_scaling|prefix_cache_only|16000|1|2|1`: n=10, avg TTFT=986.6 ms, p50=940.1 ms, p90=1289.3 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 10}

## Exact-Reuse Speedup vs Prefix

- `('agent_scaling', '16000', '1', '2', '1')`: p50=1.06x, p90=1.08x (prefix=940.1/1289.3 ms, exact+hints=888.0/1188.7 ms)
- `('agent_scaling_workflow', '16000', '1', '2', '1')`: p50=1.08x, p90=1.05x (prefix=2176.8/2229.2 ms, exact+hints=2010.6/2117.8 ms)
