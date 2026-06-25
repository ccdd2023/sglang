# KVCOMM TTFT Stress Report

This run is prefill/TTFT dominant and is intended to complement the realistic 100-case E2E table.

## Output Schema

`ttft_ms`, `e2e_ms`, `prompt_tokens`, `cached_tokens`, `cached_ratio`, `exact_hit`, `match_reason`, `matched_content_signature`, `output_exact_match_vs_baseline`, `output_token_f1_vs_baseline`, `mode`, `case_id`, `agent_id`, `segment_count`, `max_file_chars`.

## Row Counts

- agent_scaling: 24
- agent_scaling_workflow: 12

## Summary Groups

- `agent_scaling_workflow|exact_reuse_no_hints|8000|1|2|1`: n=3, avg TTFT=521.2 ms, p50=518.3 ms, p90=528.7 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 3}
- `agent_scaling_workflow|exact_reuse_plus_code_hints|8000|1|2|1`: n=3, avg TTFT=521.4 ms, p50=524.9 ms, p90=528.4 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 3}
- `agent_scaling_workflow|hints_no_exact|8000|1|2|1`: n=3, avg TTFT=481.7 ms, p50=482.0 ms, p90=482.3 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 3}
- `agent_scaling_workflow|prefix_cache_only|8000|1|2|1`: n=3, avg TTFT=503.2 ms, p50=481.0 ms, p90=548.7 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 3}
- `agent_scaling|exact_reuse_no_hints|8000|1|2|1`: n=6, avg TTFT=260.6 ms, p50=258.3 ms, p90=270.6 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 6}
- `agent_scaling|exact_reuse_plus_code_hints|8000|1|2|1`: n=6, avg TTFT=260.7 ms, p50=258.8 ms, p90=263.8 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=2245.0, F1=1.0000, status={'protected_not_consumed:no_anchor_match': 6}
- `agent_scaling|hints_no_exact|8000|1|2|1`: n=6, avg TTFT=240.8 ms, p50=240.8 ms, p90=241.2 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 6}
- `agent_scaling|prefix_cache_only|8000|1|2|1`: n=6, avg TTFT=251.6 ms, p50=240.1 ms, p90=254.6 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 6}

## Exact-Reuse Speedup vs Prefix

- `('agent_scaling', '8000', '1', '2', '1')`: p50=0.93x, p90=0.97x (prefix=240.1/254.6 ms, exact+hints=258.8/263.8 ms)
- `('agent_scaling_workflow', '8000', '1', '2', '1')`: p50=0.92x, p90=1.04x (prefix=481.0/548.7 ms, exact+hints=524.9/528.4 ms)
