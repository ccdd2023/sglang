# KVCOMM TTFT Stress Report

This run is prefill/TTFT dominant and is intended to complement the realistic 100-case E2E table.

## Output Schema

`ttft_ms`, `e2e_ms`, `prompt_tokens`, `cached_tokens`, `cached_ratio`, `exact_hit`, `match_reason`, `matched_content_signature`, `output_exact_match_vs_baseline`, `output_token_f1_vs_baseline`, `mode`, `case_id`, `agent_id`, `segment_count`, `max_file_chars`.

## Row Counts

- agent_scaling: 200
- agent_scaling_workflow: 80

## Summary Groups

- `agent_scaling_workflow|exact_reuse_no_hints|8000|1|2|1`: n=10, avg TTFT=512.0 ms, p50=592.7 ms, p90=704.1 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 10}
- `agent_scaling_workflow|exact_reuse_no_hints|8000|1|3|1`: n=10, avg TTFT=589.4 ms, p50=588.3 ms, p90=623.1 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 10}
- `agent_scaling_workflow|exact_reuse_plus_code_hints|8000|1|2|1`: n=10, avg TTFT=432.9 ms, p50=508.7 ms, p90=525.7 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 10}
- `agent_scaling_workflow|exact_reuse_plus_code_hints|8000|1|3|1`: n=10, avg TTFT=884.6 ms, p50=883.8 ms, p90=902.2 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 10}
- `agent_scaling_workflow|hints_no_exact|8000|1|2|1`: n=10, avg TTFT=744.3 ms, p50=759.6 ms, p90=851.6 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 10}
- `agent_scaling_workflow|hints_no_exact|8000|1|3|1`: n=10, avg TTFT=1028.3 ms, p50=1028.0 ms, p90=1032.9 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 10}
- `agent_scaling_workflow|prefix_cache_only|8000|1|2|1`: n=10, avg TTFT=672.6 ms, p50=672.0 ms, p90=759.8 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 10}
- `agent_scaling_workflow|prefix_cache_only|8000|1|3|1`: n=10, avg TTFT=1113.5 ms, p50=1105.8 ms, p90=1124.8 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 10}
- `agent_scaling|exact_reuse_no_hints|8000|1|2|1`: n=20, avg TTFT=256.0 ms, p50=345.2 ms, p90=356.3 ms, exact hit=1.00, device hit=0.30, consumed=0.05, protected=0.0, F1=1.0000, status={'consumed': 1, 'device_hit_without_consumed': 5, 'no_fast_path': 14}
- `agent_scaling|exact_reuse_no_hints|8000|1|3|1`: n=30, avg TTFT=196.5 ms, p50=74.5 ms, p90=448.0 ms, exact hit=1.00, device hit=0.67, consumed=0.00, protected=0.0, F1=1.0000, status={'device_hit_without_consumed': 20, 'no_fast_path': 10}
- `agent_scaling|exact_reuse_plus_code_hints|8000|1|2|1`: n=20, avg TTFT=216.4 ms, p50=78.5 ms, p90=447.3 ms, exact hit=1.00, device hit=0.60, consumed=0.00, protected=2245.0, F1=1.0000, status={'anchor_reuse_device_hit_consumed_counter_gap': 12, 'protected_not_consumed:no_anchor_match': 8}
- `agent_scaling|exact_reuse_plus_code_hints|8000|1|3|1`: n=30, avg TTFT=294.9 ms, p50=356.1 ms, p90=467.5 ms, exact hit=1.00, device hit=0.33, consumed=0.00, protected=2245.0, F1=1.0000, status={'protected_not_consumed:no_anchor_match': 20, 'anchor_reuse_device_hit_consumed_counter_gap': 10}
- `agent_scaling|hints_no_exact|8000|1|2|1`: n=20, avg TTFT=372.2 ms, p50=390.4 ms, p90=456.0 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 20}
- `agent_scaling|hints_no_exact|8000|1|3|1`: n=30, avg TTFT=342.8 ms, p50=337.5 ms, p90=359.9 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 30}
- `agent_scaling|prefix_cache_only|8000|1|2|1`: n=20, avg TTFT=336.3 ms, p50=335.3 ms, p90=421.5 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 20}
- `agent_scaling|prefix_cache_only|8000|1|3|1`: n=30, avg TTFT=371.2 ms, p50=336.2 ms, p90=465.5 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 30}

## Exact-Reuse Speedup vs Prefix

- `('agent_scaling', '8000', '1', '2', '1')`: p50=4.27x, p90=0.94x (prefix=335.3/421.5 ms, exact+hints=78.5/447.3 ms)
- `('agent_scaling', '8000', '1', '3', '1')`: p50=0.94x, p90=1.00x (prefix=336.2/465.5 ms, exact+hints=356.1/467.5 ms)
- `('agent_scaling_workflow', '8000', '1', '2', '1')`: p50=1.32x, p90=1.45x (prefix=672.0/759.8 ms, exact+hints=508.7/525.7 ms)
- `('agent_scaling_workflow', '8000', '1', '3', '1')`: p50=1.25x, p90=1.25x (prefix=1105.8/1124.8 ms, exact+hints=883.8/902.2 ms)
