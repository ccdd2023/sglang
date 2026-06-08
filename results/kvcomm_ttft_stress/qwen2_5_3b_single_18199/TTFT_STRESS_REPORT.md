# KVCOMM TTFT Stress Report

This run is prefill/TTFT dominant and is intended to complement the realistic 100-case E2E table.

## Output Schema

`ttft_ms`, `e2e_ms`, `prompt_tokens`, `cached_tokens`, `cached_ratio`, `exact_hit`, `match_reason`, `matched_content_signature`, `output_exact_match_vs_baseline`, `output_token_f1_vs_baseline`, `mode`, `case_id`, `agent_id`, `segment_count`, `max_file_chars`.

## Row Counts

- ttft_stress: 4

## Summary Groups

- `ttft_stress|exact_reuse_no_hints|16000|1|1|3`: n=1, avg TTFT=627.7 ms, p50=627.7 ms, exact hit=1.00, F1=1.0000
- `ttft_stress|exact_reuse_plus_code_hints|16000|1|1|3`: n=1, avg TTFT=894.9 ms, p50=894.9 ms, exact hit=1.00, F1=1.0000
- `ttft_stress|no_reuse_fresh_salt|16000|1|1|3`: n=1, avg TTFT=897.3 ms, p50=897.3 ms, exact hit=0.00, F1=1.0000
- `ttft_stress|prefix_cache_only|16000|1|1|3`: n=1, avg TTFT=897.1 ms, p50=897.1 ms, exact hit=0.00, F1=1.0000
