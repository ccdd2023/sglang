# KVCOMM TTFT Stress Report

This run is prefill/TTFT dominant and is intended to complement the realistic 100-case E2E table.

## Output Schema

`ttft_ms`, `e2e_ms`, `prompt_tokens`, `cached_tokens`, `cached_ratio`, `exact_hit`, `match_reason`, `matched_content_signature`, `output_exact_match_vs_baseline`, `output_token_f1_vs_baseline`, `mode`, `case_id`, `agent_id`, `segment_count`, `max_file_chars`.

## Row Counts

- agent_scaling: 40
- agent_scaling_workflow: 20

## Summary Groups

- `agent_scaling_workflow|exact_reuse_no_hints|8000|1|2|2`: n=5, avg TTFT=969.9 ms, p50=887.6 ms, p90=1536.2 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 5}
- `agent_scaling_workflow|exact_reuse_plus_code_hints|8000|1|2|2`: n=5, avg TTFT=1004.4 ms, p50=1079.7 ms, p90=1298.5 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 5}
- `agent_scaling_workflow|hints_no_exact|8000|1|2|2`: n=5, avg TTFT=1179.6 ms, p50=1258.9 ms, p90=1270.4 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 5}
- `agent_scaling_workflow|prefix_cache_only|8000|1|2|2`: n=5, avg TTFT=1186.8 ms, p50=1113.3 ms, p90=1512.5 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 5}
- `agent_scaling|exact_reuse_no_hints|8000|1|2|2`: n=10, avg TTFT=484.9 ms, p50=257.4 ms, p90=809.3 ms, exact hit=1.00, device hit=0.50, consumed=0.10, protected=0.0, F1=1.0000, status={'consumed': 1, 'device_hit_without_consumed': 4, 'no_fast_path': 5}
- `agent_scaling|exact_reuse_plus_code_hints|8000|1|2|2`: n=10, avg TTFT=502.2 ms, p50=613.9 ms, p90=651.4 ms, exact hit=1.00, device hit=0.30, consumed=0.00, protected=4287.4, F1=1.0000, status={'anchor_reuse_device_hit_consumed_counter_gap': 3, 'protected_not_consumed:no_anchor_match': 7}
- `agent_scaling|hints_no_exact|8000|1|2|2`: n=10, avg TTFT=589.8 ms, p50=614.9 ms, p90=644.0 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 10}
- `agent_scaling|prefix_cache_only|8000|1|2|2`: n=10, avg TTFT=593.4 ms, p50=639.5 ms, p90=673.8 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 10}

## Exact-Reuse Speedup vs Prefix

- `('agent_scaling', '8000', '1', '2', '2')`: p50=1.04x, p90=1.03x (prefix=639.5/673.8 ms, exact+hints=613.9/651.4 ms)
- `('agent_scaling_workflow', '8000', '1', '2', '2')`: p50=1.03x, p90=1.16x (prefix=1113.3/1512.5 ms, exact+hints=1079.7/1298.5 ms)
