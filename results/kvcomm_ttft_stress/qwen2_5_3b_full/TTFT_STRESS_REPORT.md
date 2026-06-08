# KVCOMM TTFT Stress Report

This run is prefill/TTFT dominant and is intended to complement the realistic 100-case E2E table.

## Output Schema

`ttft_ms`, `e2e_ms`, `prompt_tokens`, `cached_tokens`, `cached_ratio`, `exact_hit`, `match_reason`, `matched_content_signature`, `output_exact_match_vs_baseline`, `output_token_f1_vs_baseline`, `mode`, `case_id`, `agent_id`, `segment_count`, `max_file_chars`.

## Row Counts

- ttft_stress: 400

## Summary Groups

- `ttft_stress|exact_reuse_no_hints|16000|1|1|3`: n=50, avg TTFT=844.2 ms, p50=910.5 ms, exact hit=1.00, F1=0.9400
- `ttft_stress|exact_reuse_no_hints|8000|1|1|3`: n=50, avg TTFT=341.7 ms, p50=346.5 ms, exact hit=1.00, F1=0.8600
- `ttft_stress|exact_reuse_plus_code_hints|16000|1|1|3`: n=50, avg TTFT=867.6 ms, p50=907.6 ms, exact hit=1.00, F1=0.9600
- `ttft_stress|exact_reuse_plus_code_hints|8000|1|1|3`: n=50, avg TTFT=338.8 ms, p50=348.5 ms, exact hit=1.00, F1=0.8400
- `ttft_stress|no_reuse_fresh_salt|16000|1|1|3`: n=50, avg TTFT=864.6 ms, p50=903.6 ms, exact hit=0.00, F1=0.9200
- `ttft_stress|no_reuse_fresh_salt|8000|1|1|3`: n=50, avg TTFT=352.6 ms, p50=357.4 ms, exact hit=0.00, F1=0.8200
- `ttft_stress|prefix_cache_only|16000|1|1|3`: n=50, avg TTFT=866.1 ms, p50=903.1 ms, exact hit=0.00, F1=1.0000
- `ttft_stress|prefix_cache_only|8000|1|1|3`: n=50, avg TTFT=351.0 ms, p50=345.2 ms, exact hit=0.00, F1=1.0000
