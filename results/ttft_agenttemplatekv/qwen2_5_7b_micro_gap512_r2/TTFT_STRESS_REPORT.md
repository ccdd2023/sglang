# KVCOMM TTFT Stress Report

This run is prefill/TTFT dominant and is intended to complement the realistic 100-case E2E table.

## Output Schema

`ttft_ms`, `e2e_ms`, `prompt_tokens`, `cached_tokens`, `cached_ratio`, `exact_hit`, `match_reason`, `matched_content_signature`, `output_exact_match_vs_baseline`, `output_token_f1_vs_baseline`, `mode`, `case_id`, `agent_id`, `segment_count`, `max_file_chars`.

## Row Counts

- agent_scaling: 24
- agent_scaling_workflow: 12

## Summary Groups

- `agent_scaling_workflow|exact_reuse_no_hints|8000|1|2|1`: n=3, avg TTFT=217.8 ms, p50=125.0 ms, p90=411.7 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 3}
- `agent_scaling_workflow|exact_reuse_plus_code_hints|8000|1|2|1`: n=3, avg TTFT=251.8 ms, p50=119.4 ms, p90=522.6 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 3}
- `agent_scaling_workflow|hints_no_exact|8000|1|2|1`: n=3, avg TTFT=667.0 ms, p50=680.4 ms, p90=830.1 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 3}
- `agent_scaling_workflow|prefix_cache_only|8000|1|2|1`: n=3, avg TTFT=632.1 ms, p50=494.0 ms, p90=918.0 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 3}
- `agent_scaling|exact_reuse_no_hints|8000|1|2|1`: n=6, avg TTFT=108.9 ms, p50=61.9 ms, p90=72.4 ms, exact hit=1.00, device hit=0.00, consumed=0.17, protected=0.0, F1=1.0000, status={'consumed': 1, 'no_fast_path': 5}
- `agent_scaling|exact_reuse_plus_code_hints|8000|1|2|1`: n=6, avg TTFT=125.9 ms, p50=59.2 ms, p90=65.2 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=2245.0, F1=1.0000, status={'protected_not_consumed:position_mismatch': 5, 'protected_not_consumed:no_anchor_match': 1}
- `agent_scaling|hints_no_exact|8000|1|2|1`: n=6, avg TTFT=333.5 ms, p50=249.2 ms, p90=448.7 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 6}
- `agent_scaling|prefix_cache_only|8000|1|2|1`: n=6, avg TTFT=316.0 ms, p50=244.7 ms, p90=422.8 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 6}

## Exact-Reuse Speedup vs Prefix

- `('agent_scaling', '8000', '1', '2', '1')`: p50=4.13x, p90=6.48x (prefix=244.7/422.8 ms, exact+hints=59.2/65.2 ms)
- `('agent_scaling_workflow', '8000', '1', '2', '1')`: p50=4.14x, p90=1.76x (prefix=494.0/918.0 ms, exact+hints=119.4/522.6 ms)
