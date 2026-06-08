# KVCOMM TTFT Stress Report

This run is prefill/TTFT dominant and is intended to complement the realistic 100-case E2E table.

## Output Schema

`ttft_ms`, `e2e_ms`, `prompt_tokens`, `cached_tokens`, `cached_ratio`, `exact_hit`, `match_reason`, `matched_content_signature`, `output_exact_match_vs_baseline`, `output_token_f1_vs_baseline`, `mode`, `case_id`, `agent_id`, `segment_count`, `max_file_chars`.

## Row Counts

- ttft_stress: 40

## Summary Groups

- `ttft_stress|exact_reuse_no_hints|16000|1|1|3`: n=5, avg TTFT=1456.9 ms, p50=1692.7 ms, exact hit=1.00, F1=0.6000
- `ttft_stress|exact_reuse_no_hints|32000|1|1|3`: n=5, avg TTFT=4124.9 ms, p50=4125.4 ms, exact hit=1.00, F1=1.0000
- `ttft_stress|exact_reuse_plus_code_hints|16000|1|1|3`: n=5, avg TTFT=1449.7 ms, p50=1691.1 ms, exact hit=1.00, F1=0.6000
- `ttft_stress|exact_reuse_plus_code_hints|32000|1|1|3`: n=5, avg TTFT=4128.8 ms, p50=4125.3 ms, exact hit=1.00, F1=1.0000
- `ttft_stress|no_reuse_fresh_salt|16000|1|1|3`: n=5, avg TTFT=1698.5 ms, p50=1700.0 ms, exact hit=0.00, F1=1.0000
- `ttft_stress|no_reuse_fresh_salt|32000|1|1|3`: n=5, avg TTFT=4142.0 ms, p50=4145.2 ms, exact hit=0.00, F1=1.0000
- `ttft_stress|prefix_cache_only|16000|1|1|3`: n=5, avg TTFT=1692.0 ms, p50=1685.4 ms, exact hit=0.00, F1=1.0000
- `ttft_stress|prefix_cache_only|32000|1|1|3`: n=5, avg TTFT=4136.4 ms, p50=4138.8 ms, exact hit=0.00, F1=1.0000
