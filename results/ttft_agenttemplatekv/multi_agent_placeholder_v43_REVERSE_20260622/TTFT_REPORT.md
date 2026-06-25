# KVCOMM TTFT Stress Report

This run is prefill/TTFT dominant and is intended to complement the realistic 100-case E2E table.

## Output Schema

`ttft_ms`, `e2e_ms`, `prompt_tokens`, `cached_tokens`, `cached_ratio`, `exact_hit`, `match_reason`, `matched_content_signature`, `output_exact_match_vs_baseline`, `output_token_f1_vs_baseline`, `mode`, `case_id`, `agent_id`, `segment_count`, `max_file_chars`.

## Row Counts

- agent_scaling: 75
- agent_scaling_workflow: 25

## Summary Groups

- `agent_scaling_workflow|exact_reuse_no_hints|8000|1|1|1`: n=1, avg TTFT=748.6 ms, p50=748.6 ms, p90=748.6 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_no_hints|8000|1|2|1`: n=1, avg TTFT=1406.4 ms, p50=1406.4 ms, p90=1406.4 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_no_hints|8000|1|3|1`: n=1, avg TTFT=1395.2 ms, p50=1395.2 ms, p90=1395.2 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_no_hints|8000|1|4|1`: n=1, avg TTFT=1282.8 ms, p50=1282.8 ms, p90=1282.8 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_no_hints|8000|1|5|1`: n=1, avg TTFT=455.0 ms, p50=455.0 ms, p90=455.0 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_plus_code_hints|8000|1|1|1`: n=1, avg TTFT=874.5 ms, p50=874.5 ms, p90=874.5 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_plus_code_hints|8000|1|2|1`: n=1, avg TTFT=1657.7 ms, p50=1657.7 ms, p90=1657.7 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_plus_code_hints|8000|1|3|1`: n=1, avg TTFT=1538.8 ms, p50=1538.8 ms, p90=1538.8 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_plus_code_hints|8000|1|4|1`: n=1, avg TTFT=1438.8 ms, p50=1438.8 ms, p90=1438.8 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_plus_code_hints|8000|1|5|1`: n=1, avg TTFT=493.9 ms, p50=493.9 ms, p90=493.9 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|hints_no_exact|8000|1|1|1`: n=1, avg TTFT=229.5 ms, p50=229.5 ms, p90=229.5 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|hints_no_exact|8000|1|2|1`: n=1, avg TTFT=513.8 ms, p50=513.8 ms, p90=513.8 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|hints_no_exact|8000|1|3|1`: n=1, avg TTFT=756.4 ms, p50=756.4 ms, p90=756.4 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|hints_no_exact|8000|1|4|1`: n=1, avg TTFT=1014.2 ms, p50=1014.2 ms, p90=1014.2 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|hints_no_exact|8000|1|5|1`: n=1, avg TTFT=1279.5 ms, p50=1279.5 ms, p90=1279.5 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|placeholder_knn_reuse|8000|1|1|1`: n=1, avg TTFT=77.2 ms, p50=77.2 ms, p90=77.2 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|placeholder_knn_reuse|8000|1|2|1`: n=1, avg TTFT=139.9 ms, p50=139.9 ms, p90=139.9 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|placeholder_knn_reuse|8000|1|3|1`: n=1, avg TTFT=247.7 ms, p50=247.7 ms, p90=247.7 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|placeholder_knn_reuse|8000|1|4|1`: n=1, avg TTFT=354.8 ms, p50=354.8 ms, p90=354.8 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|placeholder_knn_reuse|8000|1|5|1`: n=1, avg TTFT=426.1 ms, p50=426.1 ms, p90=426.1 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|prefix_cache_only|8000|1|1|1`: n=1, avg TTFT=45.0 ms, p50=45.0 ms, p90=45.0 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|prefix_cache_only|8000|1|2|1`: n=1, avg TTFT=78.4 ms, p50=78.4 ms, p90=78.4 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|prefix_cache_only|8000|1|3|1`: n=1, avg TTFT=319.6 ms, p50=319.6 ms, p90=319.6 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|prefix_cache_only|8000|1|4|1`: n=1, avg TTFT=776.0 ms, p50=776.0 ms, p90=776.0 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|prefix_cache_only|8000|1|5|1`: n=1, avg TTFT=1254.5 ms, p50=1254.5 ms, p90=1254.5 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling|exact_reuse_no_hints|8000|1|1|1`: n=1, avg TTFT=748.6 ms, p50=748.6 ms, p90=748.6 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 1}
- `agent_scaling|exact_reuse_no_hints|8000|1|2|1`: n=2, avg TTFT=703.2 ms, p50=700.1 ms, p90=706.3 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 2}
- `agent_scaling|exact_reuse_no_hints|8000|1|3|1`: n=3, avg TTFT=465.1 ms, p50=629.8 ms, p90=680.4 ms, exact hit=1.00, device hit=0.33, consumed=0.33, protected=0.0, F1=1.0000, status={'no_fast_path': 2, 'consumed': 1}
- `agent_scaling|exact_reuse_no_hints|8000|1|4|1`: n=4, avg TTFT=320.7 ms, p50=502.9 ms, p90=601.0 ms, exact hit=1.00, device hit=0.50, consumed=0.25, protected=0.0, F1=1.0000, status={'consumed': 1, 'no_fast_path': 2, 'device_hit_without_consumed': 1}
- `agent_scaling|exact_reuse_no_hints|8000|1|5|1`: n=5, avg TTFT=91.0 ms, p50=89.4 ms, p90=101.3 ms, exact hit=1.00, device hit=1.00, consumed=0.20, protected=0.0, F1=1.0000, status={'consumed': 1, 'device_hit_without_consumed': 4}
- `agent_scaling|exact_reuse_plus_code_hints|8000|1|1|1`: n=1, avg TTFT=874.5 ms, p50=874.5 ms, p90=874.5 ms, exact hit=1.00, device hit=1.00, consumed=0.00, protected=4490.0, F1=1.0000, status={'protected_not_consumed:no_anchor_match': 1}
- `agent_scaling|exact_reuse_plus_code_hints|8000|1|2|1`: n=2, avg TTFT=828.8 ms, p50=806.7 ms, p90=851.0 ms, exact hit=1.00, device hit=1.00, consumed=0.00, protected=4490.0, F1=1.0000, status={'protected_not_consumed:no_anchor_match': 2}
- `agent_scaling|exact_reuse_plus_code_hints|8000|1|3|1`: n=3, avg TTFT=512.9 ms, p50=696.5 ms, p90=735.4 ms, exact hit=1.00, device hit=1.00, consumed=0.33, protected=4490.0, F1=1.0000, status={'protected_not_consumed:no_anchor_match': 2, 'consumed': 1}
- `agent_scaling|exact_reuse_plus_code_hints|8000|1|4|1`: n=4, avg TTFT=359.7 ms, p50=603.7 ms, p90=607.5 ms, exact hit=1.00, device hit=1.00, consumed=0.50, protected=4490.0, F1=1.0000, status={'consumed': 2, 'protected_not_consumed:no_anchor_match': 2}
- `agent_scaling|exact_reuse_plus_code_hints|8000|1|5|1`: n=5, avg TTFT=98.8 ms, p50=96.3 ms, p90=118.7 ms, exact hit=1.00, device hit=1.00, consumed=1.00, protected=4490.0, F1=1.0000, status={'consumed': 5}
- `agent_scaling|hints_no_exact|8000|1|1|1`: n=1, avg TTFT=229.5 ms, p50=229.5 ms, p90=229.5 ms, exact hit=0.00, device hit=1.00, consumed=0.00, protected=2245.0, F1=1.0000, status={'protected_not_consumed:no_anchor_match': 1}
- `agent_scaling|hints_no_exact|8000|1|2|1`: n=2, avg TTFT=256.9 ms, p50=252.4 ms, p90=261.4 ms, exact hit=0.00, device hit=1.00, consumed=0.00, protected=2245.0, F1=1.0000, status={'protected_not_consumed:no_anchor_match': 2}
- `agent_scaling|hints_no_exact|8000|1|3|1`: n=3, avg TTFT=252.1 ms, p50=251.1 ms, p90=257.1 ms, exact hit=0.00, device hit=1.00, consumed=0.00, protected=2245.0, F1=1.0000, status={'protected_not_consumed:no_anchor_match': 3}
- `agent_scaling|hints_no_exact|8000|1|4|1`: n=4, avg TTFT=253.6 ms, p50=253.0 ms, p90=259.6 ms, exact hit=0.00, device hit=1.00, consumed=0.00, protected=2245.0, F1=1.0000, status={'protected_not_consumed:no_anchor_match': 4}
- `agent_scaling|hints_no_exact|8000|1|5|1`: n=5, avg TTFT=255.9 ms, p50=254.8 ms, p90=259.7 ms, exact hit=0.00, device hit=1.00, consumed=0.00, protected=2245.0, F1=1.0000, status={'protected_not_consumed:no_anchor_match': 5}
- `agent_scaling|placeholder_knn_reuse|8000|1|1|1`: n=1, avg TTFT=77.2 ms, p50=77.2 ms, p90=77.2 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 1}
- `agent_scaling|placeholder_knn_reuse|8000|1|2|1`: n=2, avg TTFT=69.9 ms, p50=60.0 ms, p90=79.8 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 2}
- `agent_scaling|placeholder_knn_reuse|8000|1|3|1`: n=3, avg TTFT=82.6 ms, p50=87.4 ms, p90=96.0 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 3}
- `agent_scaling|placeholder_knn_reuse|8000|1|4|1`: n=4, avg TTFT=88.7 ms, p50=88.0 ms, p90=96.4 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 4}
- `agent_scaling|placeholder_knn_reuse|8000|1|5|1`: n=5, avg TTFT=85.2 ms, p50=83.3 ms, p90=96.4 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 5}
- `agent_scaling|prefix_cache_only|8000|1|1|1`: n=1, avg TTFT=45.0 ms, p50=45.0 ms, p90=45.0 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 1}
- `agent_scaling|prefix_cache_only|8000|1|2|1`: n=2, avg TTFT=39.2 ms, p50=37.8 ms, p90=40.6 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 2}
- `agent_scaling|prefix_cache_only|8000|1|3|1`: n=3, avg TTFT=106.5 ms, p50=46.5 ms, p90=228.0 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 3}
- `agent_scaling|prefix_cache_only|8000|1|4|1`: n=4, avg TTFT=194.0 ms, p50=247.8 ms, p90=250.9 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 4}
- `agent_scaling|prefix_cache_only|8000|1|5|1`: n=5, avg TTFT=250.9 ms, p50=251.1 ms, p90=257.6 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 5}

## Exact-Reuse Speedup vs Prefix

- `('agent_scaling', '8000', '1', '1', '1')`: p50=0.05x, p90=0.05x (prefix=45.0/45.0 ms, exact+hints=874.5/874.5 ms)
- `('agent_scaling', '8000', '1', '2', '1')`: p50=0.05x, p90=0.05x (prefix=37.8/40.6 ms, exact+hints=806.7/851.0 ms)
- `('agent_scaling', '8000', '1', '3', '1')`: p50=0.07x, p90=0.31x (prefix=46.5/228.0 ms, exact+hints=696.5/735.4 ms)
- `('agent_scaling', '8000', '1', '4', '1')`: p50=0.41x, p90=0.41x (prefix=247.8/250.9 ms, exact+hints=603.7/607.5 ms)
- `('agent_scaling', '8000', '1', '5', '1')`: p50=2.61x, p90=2.17x (prefix=251.1/257.6 ms, exact+hints=96.3/118.7 ms)
- `('agent_scaling_workflow', '8000', '1', '1', '1')`: p50=0.05x, p90=0.05x (prefix=45.0/45.0 ms, exact+hints=874.5/874.5 ms)
- `('agent_scaling_workflow', '8000', '1', '2', '1')`: p50=0.05x, p90=0.05x (prefix=78.4/78.4 ms, exact+hints=1657.7/1657.7 ms)
- `('agent_scaling_workflow', '8000', '1', '3', '1')`: p50=0.21x, p90=0.21x (prefix=319.6/319.6 ms, exact+hints=1538.8/1538.8 ms)
- `('agent_scaling_workflow', '8000', '1', '4', '1')`: p50=0.54x, p90=0.54x (prefix=776.0/776.0 ms, exact+hints=1438.8/1438.8 ms)
- `('agent_scaling_workflow', '8000', '1', '5', '1')`: p50=2.54x, p90=2.54x (prefix=1254.5/1254.5 ms, exact+hints=493.9/493.9 ms)
