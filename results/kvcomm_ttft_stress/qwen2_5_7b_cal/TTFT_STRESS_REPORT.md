# KVCOMM TTFT Stress Report

This run is prefill/TTFT dominant and is intended to complement the realistic 100-case E2E table.

## Output Schema

`ttft_ms`, `e2e_ms`, `prompt_tokens`, `cached_tokens`, `cached_ratio`, `exact_hit`, `match_reason`, `matched_content_signature`, `output_exact_match_vs_baseline`, `output_token_f1_vs_baseline`, `mode`, `case_id`, `agent_id`, `segment_count`, `max_file_chars`.

## Row Counts

- ttft_stress: 320

## Summary Groups

- `ttft_stress|exact_reuse_no_hints|16000|1|1|3`: n=10, avg TTFT=1013.6 ms, p50=1098.8 ms, exact hit=1.00, F1=0.1000
- `ttft_stress|exact_reuse_no_hints|16000|32|1|3`: n=10, avg TTFT=1009.0 ms, p50=1088.7 ms, exact hit=1.00, F1=0.1565
- `ttft_stress|exact_reuse_no_hints|32000|1|1|3`: n=10, avg TTFT=4140.4 ms, p50=4136.7 ms, exact hit=0.30, F1=1.0000
- `ttft_stress|exact_reuse_no_hints|32000|32|1|3`: n=10, avg TTFT=4140.5 ms, p50=4142.0 ms, exact hit=1.00, F1=0.6305
- `ttft_stress|exact_reuse_no_hints|48000|1|1|3`: n=10, avg TTFT=96.9 ms, p50=99.3 ms, exact hit=0.00, F1=1.0000
- `ttft_stress|exact_reuse_no_hints|48000|32|1|3`: n=10, avg TTFT=100.7 ms, p50=103.6 ms, exact hit=0.00, F1=1.0000
- `ttft_stress|exact_reuse_no_hints|8000|1|1|3`: n=10, avg TTFT=626.5 ms, p50=657.7 ms, exact hit=1.00, F1=0.5000
- `ttft_stress|exact_reuse_no_hints|8000|32|1|3`: n=10, avg TTFT=61.3 ms, p50=61.1 ms, exact hit=1.00, F1=0.3809
- `ttft_stress|exact_reuse_plus_code_hints|16000|1|1|3`: n=10, avg TTFT=1181.4 ms, p50=1107.6 ms, exact hit=1.00, F1=0.4000
- `ttft_stress|exact_reuse_plus_code_hints|16000|32|1|3`: n=10, avg TTFT=1011.1 ms, p50=1091.6 ms, exact hit=1.00, F1=0.1592
- `ttft_stress|exact_reuse_plus_code_hints|32000|1|1|3`: n=10, avg TTFT=4143.9 ms, p50=4139.4 ms, exact hit=1.00, F1=1.0000
- `ttft_stress|exact_reuse_plus_code_hints|32000|32|1|3`: n=10, avg TTFT=4143.9 ms, p50=4145.5 ms, exact hit=1.00, F1=0.6341
- `ttft_stress|exact_reuse_plus_code_hints|48000|1|1|3`: n=10, avg TTFT=102.0 ms, p50=103.5 ms, exact hit=0.00, F1=1.0000
- `ttft_stress|exact_reuse_plus_code_hints|48000|32|1|3`: n=10, avg TTFT=96.9 ms, p50=97.0 ms, exact hit=0.00, F1=1.0000
- `ttft_stress|exact_reuse_plus_code_hints|8000|1|1|3`: n=10, avg TTFT=643.4 ms, p50=659.5 ms, exact hit=1.00, F1=0.7000
- `ttft_stress|exact_reuse_plus_code_hints|8000|32|1|3`: n=10, avg TTFT=62.3 ms, p50=61.8 ms, exact hit=1.00, F1=0.4148
- `ttft_stress|no_reuse_fresh_salt|16000|1|1|3`: n=10, avg TTFT=1706.6 ms, p50=1707.2 ms, exact hit=0.00, F1=1.0000
- `ttft_stress|no_reuse_fresh_salt|16000|32|1|3`: n=10, avg TTFT=1473.9 ms, p50=1704.2 ms, exact hit=0.00, F1=0.4462
- `ttft_stress|no_reuse_fresh_salt|32000|1|1|3`: n=10, avg TTFT=4149.7 ms, p50=4148.7 ms, exact hit=0.00, F1=1.0000
- `ttft_stress|no_reuse_fresh_salt|32000|32|1|3`: n=10, avg TTFT=4141.4 ms, p50=4141.5 ms, exact hit=0.00, F1=0.5375
- `ttft_stress|no_reuse_fresh_salt|48000|1|1|3`: n=10, avg TTFT=84.7 ms, p50=84.2 ms, exact hit=0.00, F1=1.0000
- `ttft_stress|no_reuse_fresh_salt|48000|32|1|3`: n=10, avg TTFT=87.9 ms, p50=86.0 ms, exact hit=0.00, F1=1.0000
- `ttft_stress|no_reuse_fresh_salt|8000|1|1|3`: n=10, avg TTFT=677.2 ms, p50=657.1 ms, exact hit=0.00, F1=0.4000
- `ttft_stress|no_reuse_fresh_salt|8000|32|1|3`: n=10, avg TTFT=60.0 ms, p50=60.0 ms, exact hit=0.00, F1=0.3891
- `ttft_stress|prefix_cache_only|16000|1|1|3`: n=10, avg TTFT=1710.6 ms, p50=1709.2 ms, exact hit=0.00, F1=1.0000
- `ttft_stress|prefix_cache_only|16000|32|1|3`: n=10, avg TTFT=1709.4 ms, p50=1709.5 ms, exact hit=0.00, F1=1.0000
- `ttft_stress|prefix_cache_only|32000|1|1|3`: n=10, avg TTFT=4145.0 ms, p50=4140.2 ms, exact hit=0.00, F1=1.0000
- `ttft_stress|prefix_cache_only|32000|32|1|3`: n=10, avg TTFT=4141.1 ms, p50=4142.9 ms, exact hit=0.00, F1=1.0000
- `ttft_stress|prefix_cache_only|48000|1|1|3`: n=10, avg TTFT=92.0 ms, p50=86.1 ms, exact hit=0.00, F1=1.0000
- `ttft_stress|prefix_cache_only|48000|32|1|3`: n=10, avg TTFT=87.6 ms, p50=85.2 ms, exact hit=0.00, F1=1.0000
- `ttft_stress|prefix_cache_only|8000|1|1|3`: n=10, avg TTFT=664.0 ms, p50=664.1 ms, exact hit=0.00, F1=1.0000
- `ttft_stress|prefix_cache_only|8000|32|1|3`: n=10, avg TTFT=60.4 ms, p50=60.3 ms, exact hit=0.00, F1=1.0000
