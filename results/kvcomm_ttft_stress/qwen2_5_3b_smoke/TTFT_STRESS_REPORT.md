# KVCOMM TTFT Stress Report

This run is prefill/TTFT dominant and is intended to complement the realistic 100-case E2E table.

## Output Schema

`ttft_ms`, `e2e_ms`, `prompt_tokens`, `cached_tokens`, `cached_ratio`, `exact_hit`, `match_reason`, `matched_content_signature`, `output_exact_match_vs_baseline`, `output_token_f1_vs_baseline`, `mode`, `case_id`, `agent_id`, `segment_count`, `max_file_chars`.

## Row Counts

- ttft_stress: 36

## Summary Groups

- `ttft_stress|exact_reuse_no_hints|16000|1|1|3`: n=3, avg TTFT=889.8 ms, p50=887.3 ms, exact hit=1.00, F1=1.0000
- `ttft_stress|exact_reuse_no_hints|32000|1|1|3`: n=3, avg TTFT=2285.8 ms, p50=2284.5 ms, exact hit=0.00, F1=1.0000
- `ttft_stress|exact_reuse_no_hints|8000|1|1|3`: n=3, avg TTFT=267.7 ms, p50=267.7 ms, exact hit=1.00, F1=1.0000
- `ttft_stress|exact_reuse_plus_code_hints|16000|1|1|3`: n=3, avg TTFT=803.3 ms, p50=883.2 ms, exact hit=1.00, F1=1.0000
- `ttft_stress|exact_reuse_plus_code_hints|32000|1|1|3`: n=3, avg TTFT=2293.0 ms, p50=2294.3 ms, exact hit=1.00, F1=1.0000
- `ttft_stress|exact_reuse_plus_code_hints|8000|1|1|3`: n=3, avg TTFT=267.4 ms, p50=267.1 ms, exact hit=1.00, F1=1.0000
- `ttft_stress|no_reuse_fresh_salt|16000|1|1|3`: n=3, avg TTFT=888.1 ms, p50=887.6 ms, exact hit=0.00, F1=1.0000
- `ttft_stress|no_reuse_fresh_salt|32000|1|1|3`: n=3, avg TTFT=2297.6 ms, p50=2298.3 ms, exact hit=0.00, F1=1.0000
- `ttft_stress|no_reuse_fresh_salt|8000|1|1|3`: n=3, avg TTFT=343.8 ms, p50=340.3 ms, exact hit=0.00, F1=1.0000
- `ttft_stress|prefix_cache_only|16000|1|1|3`: n=3, avg TTFT=888.6 ms, p50=888.9 ms, exact hit=0.00, F1=1.0000
- `ttft_stress|prefix_cache_only|32000|1|1|3`: n=3, avg TTFT=2302.2 ms, p50=2307.2 ms, exact hit=0.00, F1=1.0000
- `ttft_stress|prefix_cache_only|8000|1|1|3`: n=3, avg TTFT=338.4 ms, p50=339.6 ms, exact hit=0.00, F1=1.0000
