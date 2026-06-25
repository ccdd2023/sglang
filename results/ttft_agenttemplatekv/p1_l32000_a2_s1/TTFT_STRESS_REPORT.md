# KVCOMM TTFT Stress Report

This run is prefill/TTFT dominant and is intended to complement the realistic 100-case E2E table.

## Output Schema

`ttft_ms`, `e2e_ms`, `prompt_tokens`, `cached_tokens`, `cached_ratio`, `exact_hit`, `match_reason`, `matched_content_signature`, `output_exact_match_vs_baseline`, `output_token_f1_vs_baseline`, `mode`, `case_id`, `agent_id`, `segment_count`, `max_file_chars`.

## Row Counts

- agent_scaling: 40
- agent_scaling_workflow: 20

## Summary Groups

- `agent_scaling_workflow|exact_reuse_no_hints|32000|1|2|1`: n=5, avg TTFT=4146.9 ms, p50=4162.8 ms, p90=4179.7 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 5}
- `agent_scaling_workflow|exact_reuse_plus_code_hints|32000|1|2|1`: n=5, avg TTFT=4051.2 ms, p50=4054.3 ms, p90=4072.7 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 5}
- `agent_scaling_workflow|hints_no_exact|32000|1|2|1`: n=5, avg TTFT=3916.0 ms, p50=3932.0 ms, p90=3943.9 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 5}
- `agent_scaling_workflow|prefix_cache_only|32000|1|2|1`: n=5, avg TTFT=3601.2 ms, p50=3799.8 ms, p90=3810.5 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 5}
- `agent_scaling|exact_reuse_no_hints|32000|1|2|1`: n=10, avg TTFT=2073.5 ms, p50=2000.7 ms, p90=2185.1 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 10}
- `agent_scaling|exact_reuse_plus_code_hints|32000|1|2|1`: n=10, avg TTFT=2025.6 ms, p50=1914.0 ms, p90=2168.8 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=13105.0, F1=1.0000, status={'protected_not_consumed:no_anchor_match': 10}
- `agent_scaling|hints_no_exact|32000|1|2|1`: n=10, avg TTFT=1958.0 ms, p50=1948.3 ms, p90=2005.3 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 10}
- `agent_scaling|prefix_cache_only|32000|1|2|1`: n=10, avg TTFT=1800.6 ms, p50=1796.6 ms, p90=2013.9 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 10}

## Exact-Reuse Speedup vs Prefix

- `('agent_scaling', '32000', '1', '2', '1')`: p50=0.94x, p90=0.93x (prefix=1796.6/2013.9 ms, exact+hints=1914.0/2168.8 ms)
- `('agent_scaling_workflow', '32000', '1', '2', '1')`: p50=0.94x, p90=0.94x (prefix=3799.8/3810.5 ms, exact+hints=4054.3/4072.7 ms)
