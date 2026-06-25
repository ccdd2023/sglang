# KVCOMM TTFT Stress Report

This run is prefill/TTFT dominant and is intended to complement the realistic 100-case E2E table.

## Output Schema

`ttft_ms`, `e2e_ms`, `prompt_tokens`, `cached_tokens`, `cached_ratio`, `exact_hit`, `match_reason`, `matched_content_signature`, `output_exact_match_vs_baseline`, `output_token_f1_vs_baseline`, `mode`, `case_id`, `agent_id`, `segment_count`, `max_file_chars`.

## Row Counts

- agent_scaling: 60
- agent_scaling_workflow: 20

## Summary Groups

- `agent_scaling_workflow|exact_reuse_no_hints|8000|1|3|1`: n=5, avg TTFT=568.3 ms, p50=589.1 ms, p90=962.2 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 5}
- `agent_scaling_workflow|exact_reuse_plus_code_hints|8000|1|3|1`: n=5, avg TTFT=679.8 ms, p50=864.1 ms, p90=901.8 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 5}
- `agent_scaling_workflow|hints_no_exact|8000|1|3|1`: n=5, avg TTFT=969.5 ms, p50=1032.0 ms, p90=1038.8 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 5}
- `agent_scaling_workflow|prefix_cache_only|8000|1|3|1`: n=5, avg TTFT=996.8 ms, p50=1126.2 ms, p90=1258.6 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 5}
- `agent_scaling|exact_reuse_no_hints|8000|1|3|1`: n=15, avg TTFT=189.4 ms, p50=73.2 ms, p90=470.7 ms, exact hit=1.00, device hit=0.67, consumed=0.07, protected=0.0, F1=1.0000, status={'consumed': 1, 'device_hit_without_consumed': 9, 'no_fast_path': 5}
- `agent_scaling|exact_reuse_plus_code_hints|8000|1|3|1`: n=15, avg TTFT=226.6 ms, p50=78.3 ms, p90=447.6 ms, exact hit=1.00, device hit=0.53, consumed=0.00, protected=2245.0, F1=1.0000, status={'anchor_reuse_device_hit_consumed_counter_gap': 8, 'protected_not_consumed:no_anchor_match': 7}
- `agent_scaling|hints_no_exact|8000|1|3|1`: n=15, avg TTFT=323.2 ms, p50=337.1 ms, p90=362.8 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 15}
- `agent_scaling|prefix_cache_only|8000|1|3|1`: n=15, avg TTFT=332.3 ms, p50=331.9 ms, p90=493.8 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 15}

## Exact-Reuse Speedup vs Prefix

- `('agent_scaling', '8000', '1', '3', '1')`: p50=4.24x, p90=1.10x (prefix=331.9/493.8 ms, exact+hints=78.3/447.6 ms)
- `('agent_scaling_workflow', '8000', '1', '3', '1')`: p50=1.30x, p90=1.40x (prefix=1126.2/1258.6 ms, exact+hints=864.1/901.8 ms)
