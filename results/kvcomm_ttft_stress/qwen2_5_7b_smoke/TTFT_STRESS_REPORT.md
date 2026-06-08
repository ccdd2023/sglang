# KVCOMM TTFT Stress Report

This run is prefill/TTFT dominant and is intended to complement the realistic 100-case E2E table.

## Output Schema

`ttft_ms`, `e2e_ms`, `prompt_tokens`, `cached_tokens`, `cached_ratio`, `exact_hit`, `match_reason`, `matched_content_signature`, `output_exact_match_vs_baseline`, `output_token_f1_vs_baseline`, `mode`, `case_id`, `agent_id`, `segment_count`, `max_file_chars`.

## Row Counts

- ttft_stress: 24

## Summary Groups

- `ttft_stress|exact_reuse_no_hints|16000|1|1|3`: n=3, avg TTFT=1499.9 ms, p50=1704.7 ms, exact hit=1.00, F1=0.6667
- `ttft_stress|exact_reuse_no_hints|8000|1|1|3`: n=3, avg TTFT=543.3 ms, p50=494.8 ms, exact hit=1.00, F1=0.0000
- `ttft_stress|exact_reuse_plus_code_hints|16000|1|1|3`: n=3, avg TTFT=1702.8 ms, p50=1705.5 ms, exact hit=1.00, F1=1.0000
- `ttft_stress|exact_reuse_plus_code_hints|8000|1|1|3`: n=3, avg TTFT=600.5 ms, p50=658.8 ms, exact hit=1.00, F1=0.3333
- `ttft_stress|no_reuse_fresh_salt|16000|1|1|3`: n=3, avg TTFT=1693.6 ms, p50=1696.2 ms, exact hit=0.00, F1=1.0000
- `ttft_stress|no_reuse_fresh_salt|8000|1|1|3`: n=3, avg TTFT=671.5 ms, p50=657.7 ms, exact hit=0.00, F1=0.0000
- `ttft_stress|prefix_cache_only|16000|1|1|3`: n=3, avg TTFT=1705.3 ms, p50=1706.0 ms, exact hit=0.00, F1=1.0000
- `ttft_stress|prefix_cache_only|8000|1|1|3`: n=3, avg TTFT=665.3 ms, p50=664.9 ms, exact hit=0.00, F1=1.0000
