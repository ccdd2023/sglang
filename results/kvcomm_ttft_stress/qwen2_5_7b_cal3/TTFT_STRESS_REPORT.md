# KVCOMM TTFT Stress Report

This run is prefill/TTFT dominant and is intended to complement the realistic 100-case E2E table.

## Output Schema

`ttft_ms`, `e2e_ms`, `prompt_tokens`, `cached_tokens`, `cached_ratio`, `exact_hit`, `match_reason`, `matched_content_signature`, `output_exact_match_vs_baseline`, `output_token_f1_vs_baseline`, `mode`, `case_id`, `agent_id`, `segment_count`, `max_file_chars`.

## Row Counts

- ttft_stress: 20

## Summary Groups

- `ttft_stress|exact_reuse_no_hints|32000|1|1|1`: n=5, avg TTFT=1413.1 ms, p50=1415.2 ms, exact hit=1.00, F1=1.0000
- `ttft_stress|exact_reuse_plus_code_hints|32000|1|1|1`: n=5, avg TTFT=1411.9 ms, p50=1407.2 ms, exact hit=1.00, F1=1.0000
- `ttft_stress|no_reuse_fresh_salt|32000|1|1|1`: n=5, avg TTFT=1419.1 ms, p50=1417.6 ms, exact hit=0.00, F1=1.0000
- `ttft_stress|prefix_cache_only|32000|1|1|1`: n=5, avg TTFT=1408.0 ms, p50=1409.5 ms, exact hit=0.00, F1=1.0000
