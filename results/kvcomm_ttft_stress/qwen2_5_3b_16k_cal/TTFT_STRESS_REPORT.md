# KVCOMM TTFT Stress Report

This run is prefill/TTFT dominant and is intended to complement the realistic 100-case E2E table.

## Output Schema

`ttft_ms`, `e2e_ms`, `prompt_tokens`, `cached_tokens`, `cached_ratio`, `exact_hit`, `match_reason`, `matched_content_signature`, `output_exact_match_vs_baseline`, `output_token_f1_vs_baseline`, `mode`, `case_id`, `agent_id`, `segment_count`, `max_file_chars`.

## Row Counts

- ttft_stress: 40

## Summary Groups

- `ttft_stress|exact_reuse_no_hints|16000|1|1|3`: n=10, avg TTFT=869.4 ms, p50=888.7 ms, exact hit=1.00, F1=1.0000
- `ttft_stress|exact_reuse_plus_code_hints|16000|1|1|3`: n=10, avg TTFT=899.9 ms, p50=899.9 ms, exact hit=1.00, F1=1.0000
- `ttft_stress|no_reuse_fresh_salt|16000|1|1|3`: n=10, avg TTFT=894.5 ms, p50=895.0 ms, exact hit=0.00, F1=1.0000
- `ttft_stress|prefix_cache_only|16000|1|1|3`: n=10, avg TTFT=891.9 ms, p50=892.7 ms, exact hit=0.00, F1=1.0000
