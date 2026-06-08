# KVCOMM TTFT Stress Report

This run is prefill/TTFT dominant and is intended to complement the realistic 100-case E2E table.

## Output Schema

`ttft_ms`, `e2e_ms`, `prompt_tokens`, `cached_tokens`, `cached_ratio`, `exact_hit`, `match_reason`, `matched_content_signature`, `output_exact_match_vs_baseline`, `output_token_f1_vs_baseline`, `mode`, `case_id`, `agent_id`, `segment_count`, `max_file_chars`.

## Row Counts

- agent_scaling: 12
- agent_scaling_workflow: 6

## Summary Groups

- `agent_scaling_workflow|exact_reuse_plus_code_hints|8000|1|2|1`: n=3, avg TTFT=115.5 ms, p50=111.8 ms, exact hit=1.00, F1=1.0000
- `agent_scaling_workflow|prefix_cache_only|8000|1|2|1`: n=3, avg TTFT=503.3 ms, p50=505.9 ms, exact hit=0.00, F1=1.0000
- `agent_scaling|exact_reuse_plus_code_hints|8000|1|2|1`: n=6, avg TTFT=57.7 ms, p50=56.0 ms, exact hit=1.00, F1=1.0000
- `agent_scaling|prefix_cache_only|8000|1|2|1`: n=6, avg TTFT=251.7 ms, p50=252.9 ms, exact hit=0.00, F1=1.0000
