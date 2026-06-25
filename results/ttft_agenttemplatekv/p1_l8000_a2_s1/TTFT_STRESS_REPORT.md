# KVCOMM TTFT Stress Report

This run is prefill/TTFT dominant and is intended to complement the realistic 100-case E2E table.

## Output Schema

`ttft_ms`, `e2e_ms`, `prompt_tokens`, `cached_tokens`, `cached_ratio`, `exact_hit`, `match_reason`, `matched_content_signature`, `output_exact_match_vs_baseline`, `output_token_f1_vs_baseline`, `mode`, `case_id`, `agent_id`, `segment_count`, `max_file_chars`.

## Row Counts

- agent_scaling: 40
- agent_scaling_workflow: 20

## Summary Groups

- `agent_scaling_workflow|exact_reuse_no_hints|8000|1|2|1`: n=5, avg TTFT=392.4 ms, p50=405.0 ms, p90=703.5 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 5}
- `agent_scaling_workflow|exact_reuse_plus_code_hints|8000|1|2|1`: n=5, avg TTFT=356.7 ms, p50=510.6 ms, p90=517.7 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 5}
- `agent_scaling_workflow|hints_no_exact|8000|1|2|1`: n=5, avg TTFT=722.9 ms, p50=788.4 ms, p90=856.0 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 5}
- `agent_scaling_workflow|prefix_cache_only|8000|1|2|1`: n=5, avg TTFT=654.1 ms, p50=692.2 ms, p90=912.3 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 5}
- `agent_scaling|exact_reuse_no_hints|8000|1|2|1`: n=10, avg TTFT=196.2 ms, p50=65.4 ms, p90=355.1 ms, exact hit=1.00, device hit=0.50, consumed=0.10, protected=0.0, F1=1.0000, status={'consumed': 1, 'device_hit_without_consumed': 4, 'no_fast_path': 5}
- `agent_scaling|exact_reuse_plus_code_hints|8000|1|2|1`: n=10, avg TTFT=178.4 ms, p50=65.7 ms, p90=445.4 ms, exact hit=1.00, device hit=0.70, consumed=0.00, protected=2245.0, F1=1.0000, status={'anchor_reuse_device_hit_consumed_counter_gap': 7, 'protected_not_consumed:no_anchor_match': 3}
- `agent_scaling|hints_no_exact|8000|1|2|1`: n=10, avg TTFT=361.4 ms, p50=338.2 ms, p90=468.0 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 10}
- `agent_scaling|prefix_cache_only|8000|1|2|1`: n=10, avg TTFT=327.1 ms, p50=334.7 ms, p90=421.4 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 10}

## Exact-Reuse Speedup vs Prefix

- `('agent_scaling', '8000', '1', '2', '1')`: p50=5.10x, p90=0.95x (prefix=334.7/421.4 ms, exact+hints=65.7/445.4 ms)
- `('agent_scaling_workflow', '8000', '1', '2', '1')`: p50=1.36x, p90=1.76x (prefix=692.2/912.3 ms, exact+hints=510.6/517.7 ms)
