# KVCOMM TTFT Stress Report

This run is prefill/TTFT dominant and is intended to complement the realistic 100-case E2E table.

## Output Schema

`ttft_ms`, `e2e_ms`, `prompt_tokens`, `cached_tokens`, `cached_ratio`, `exact_hit`, `match_reason`, `matched_content_signature`, `output_exact_match_vs_baseline`, `output_token_f1_vs_baseline`, `mode`, `case_id`, `agent_id`, `segment_count`, `max_file_chars`.

## Row Counts

- agent_scaling: 75
- agent_scaling_workflow: 25

## Summary Groups

- `agent_scaling_workflow|exact_reuse_no_hints|2000|1|1|1`: n=1, avg TTFT=97.9 ms, p50=97.9 ms, p90=97.9 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_no_hints|2000|1|2|1`: n=1, avg TTFT=140.2 ms, p50=140.2 ms, p90=140.2 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_no_hints|2000|1|3|1`: n=1, avg TTFT=238.7 ms, p50=238.7 ms, p90=238.7 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_no_hints|2000|1|4|1`: n=1, avg TTFT=328.9 ms, p50=328.9 ms, p90=328.9 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_no_hints|2000|1|5|1`: n=1, avg TTFT=474.5 ms, p50=474.5 ms, p90=474.5 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_plus_code_hints|2000|1|1|1`: n=1, avg TTFT=80.5 ms, p50=80.5 ms, p90=80.5 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_plus_code_hints|2000|1|2|1`: n=1, avg TTFT=206.5 ms, p50=206.5 ms, p90=206.5 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_plus_code_hints|2000|1|3|1`: n=1, avg TTFT=248.2 ms, p50=248.2 ms, p90=248.2 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_plus_code_hints|2000|1|4|1`: n=1, avg TTFT=377.5 ms, p50=377.5 ms, p90=377.5 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_plus_code_hints|2000|1|5|1`: n=1, avg TTFT=487.0 ms, p50=487.0 ms, p90=487.0 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|hints_no_exact|2000|1|1|1`: n=1, avg TTFT=86.5 ms, p50=86.5 ms, p90=86.5 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|hints_no_exact|2000|1|2|1`: n=1, avg TTFT=177.1 ms, p50=177.1 ms, p90=177.1 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|hints_no_exact|2000|1|3|1`: n=1, avg TTFT=268.7 ms, p50=268.7 ms, p90=268.7 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|hints_no_exact|2000|1|4|1`: n=1, avg TTFT=354.9 ms, p50=354.9 ms, p90=354.9 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|hints_no_exact|2000|1|5|1`: n=1, avg TTFT=474.2 ms, p50=474.2 ms, p90=474.2 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|placeholder_knn_reuse|2000|1|1|1`: n=1, avg TTFT=57.6 ms, p50=57.6 ms, p90=57.6 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|placeholder_knn_reuse|2000|1|2|1`: n=1, avg TTFT=124.5 ms, p50=124.5 ms, p90=124.5 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|placeholder_knn_reuse|2000|1|3|1`: n=1, avg TTFT=199.4 ms, p50=199.4 ms, p90=199.4 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|placeholder_knn_reuse|2000|1|4|1`: n=1, avg TTFT=266.6 ms, p50=266.6 ms, p90=266.6 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|placeholder_knn_reuse|2000|1|5|1`: n=1, avg TTFT=338.1 ms, p50=338.1 ms, p90=338.1 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|prefix_cache_only|2000|1|1|1`: n=1, avg TTFT=97.3 ms, p50=97.3 ms, p90=97.3 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|prefix_cache_only|2000|1|2|1`: n=1, avg TTFT=127.9 ms, p50=127.9 ms, p90=127.9 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|prefix_cache_only|2000|1|3|1`: n=1, avg TTFT=176.7 ms, p50=176.7 ms, p90=176.7 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|prefix_cache_only|2000|1|4|1`: n=1, avg TTFT=197.3 ms, p50=197.3 ms, p90=197.3 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|prefix_cache_only|2000|1|5|1`: n=1, avg TTFT=245.9 ms, p50=245.9 ms, p90=245.9 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling|exact_reuse_no_hints|2000|1|1|1`: n=1, avg TTFT=97.9 ms, p50=97.9 ms, p90=97.9 ms, exact hit=1.00, device hit=1.00, consumed=1.00, protected=0.0, F1=1.0000, status={'consumed': 1}
- `agent_scaling|exact_reuse_no_hints|2000|1|2|1`: n=2, avg TTFT=70.1 ms, p50=63.0 ms, p90=77.2 ms, exact hit=1.00, device hit=1.00, consumed=0.50, protected=0.0, F1=1.0000, status={'consumed': 1, 'device_hit_without_consumed': 1}
- `agent_scaling|exact_reuse_no_hints|2000|1|3|1`: n=3, avg TTFT=79.6 ms, p50=78.9 ms, p90=81.3 ms, exact hit=1.00, device hit=1.00, consumed=0.33, protected=0.0, F1=1.0000, status={'consumed': 1, 'device_hit_without_consumed': 2}
- `agent_scaling|exact_reuse_no_hints|2000|1|4|1`: n=4, avg TTFT=82.2 ms, p50=89.0 ms, p90=91.9 ms, exact hit=1.00, device hit=1.00, consumed=0.25, protected=0.0, F1=1.0000, status={'consumed': 1, 'device_hit_without_consumed': 3}
- `agent_scaling|exact_reuse_no_hints|2000|1|5|1`: n=5, avg TTFT=94.9 ms, p50=86.8 ms, p90=128.2 ms, exact hit=1.00, device hit=1.00, consumed=0.20, protected=0.0, F1=1.0000, status={'consumed': 1, 'device_hit_without_consumed': 4}
- `agent_scaling|exact_reuse_plus_code_hints|2000|1|1|1`: n=1, avg TTFT=80.5 ms, p50=80.5 ms, p90=80.5 ms, exact hit=1.00, device hit=1.00, consumed=1.00, protected=1234.0, F1=1.0000, status={'consumed': 1}
- `agent_scaling|exact_reuse_plus_code_hints|2000|1|2|1`: n=2, avg TTFT=103.2 ms, p50=85.5 ms, p90=121.0 ms, exact hit=1.00, device hit=1.00, consumed=1.00, protected=1234.0, F1=1.0000, status={'consumed': 2}
- `agent_scaling|exact_reuse_plus_code_hints|2000|1|3|1`: n=3, avg TTFT=82.8 ms, p50=84.9 ms, p90=88.0 ms, exact hit=1.00, device hit=1.00, consumed=1.00, protected=1234.0, F1=1.0000, status={'consumed': 3}
- `agent_scaling|exact_reuse_plus_code_hints|2000|1|4|1`: n=4, avg TTFT=94.4 ms, p50=99.4 ms, p90=99.4 ms, exact hit=1.00, device hit=1.00, consumed=1.00, protected=1234.0, F1=1.0000, status={'consumed': 4}
- `agent_scaling|exact_reuse_plus_code_hints|2000|1|5|1`: n=5, avg TTFT=97.4 ms, p50=98.8 ms, p90=119.8 ms, exact hit=1.00, device hit=1.00, consumed=1.00, protected=1234.0, F1=1.0000, status={'consumed': 5}
- `agent_scaling|hints_no_exact|2000|1|1|1`: n=1, avg TTFT=86.5 ms, p50=86.5 ms, p90=86.5 ms, exact hit=0.00, device hit=1.00, consumed=0.00, protected=617.0, F1=1.0000, status={'protected_not_consumed:no_anchor_match': 1}
- `agent_scaling|hints_no_exact|2000|1|2|1`: n=2, avg TTFT=88.5 ms, p50=86.7 ms, p90=90.4 ms, exact hit=0.00, device hit=1.00, consumed=0.00, protected=617.0, F1=1.0000, status={'protected_not_consumed:no_anchor_match': 2}
- `agent_scaling|hints_no_exact|2000|1|3|1`: n=3, avg TTFT=89.6 ms, p50=89.7 ms, p90=91.4 ms, exact hit=0.00, device hit=1.00, consumed=0.00, protected=617.0, F1=1.0000, status={'protected_not_consumed:no_anchor_match': 3}
- `agent_scaling|hints_no_exact|2000|1|4|1`: n=4, avg TTFT=88.7 ms, p50=89.5 ms, p90=89.7 ms, exact hit=0.00, device hit=1.00, consumed=0.00, protected=617.0, F1=1.0000, status={'protected_not_consumed:no_anchor_match': 4}
- `agent_scaling|hints_no_exact|2000|1|5|1`: n=5, avg TTFT=94.8 ms, p50=94.8 ms, p90=96.0 ms, exact hit=0.00, device hit=1.00, consumed=0.00, protected=617.0, F1=1.0000, status={'protected_not_consumed:no_anchor_match': 5}
- `agent_scaling|placeholder_knn_reuse|2000|1|1|1`: n=1, avg TTFT=57.6 ms, p50=57.6 ms, p90=57.6 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 1}
- `agent_scaling|placeholder_knn_reuse|2000|1|2|1`: n=2, avg TTFT=62.2 ms, p50=56.7 ms, p90=67.8 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 2}
- `agent_scaling|placeholder_knn_reuse|2000|1|3|1`: n=3, avg TTFT=66.5 ms, p50=69.4 ms, p90=69.5 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 3}
- `agent_scaling|placeholder_knn_reuse|2000|1|4|1`: n=4, avg TTFT=66.7 ms, p50=70.1 ms, p90=71.0 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 4}
- `agent_scaling|placeholder_knn_reuse|2000|1|5|1`: n=5, avg TTFT=67.6 ms, p50=65.0 ms, p90=77.6 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 5}
- `agent_scaling|prefix_cache_only|2000|1|1|1`: n=1, avg TTFT=97.3 ms, p50=97.3 ms, p90=97.3 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 1}
- `agent_scaling|prefix_cache_only|2000|1|2|1`: n=2, avg TTFT=63.9 ms, p50=36.8 ms, p90=91.1 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 2}
- `agent_scaling|prefix_cache_only|2000|1|3|1`: n=3, avg TTFT=58.9 ms, p50=41.8 ms, p90=93.2 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 3}
- `agent_scaling|prefix_cache_only|2000|1|4|1`: n=4, avg TTFT=49.3 ms, p50=38.0 ms, p90=87.7 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 4}
- `agent_scaling|prefix_cache_only|2000|1|5|1`: n=5, avg TTFT=49.2 ms, p50=40.4 ms, p90=90.0 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 5}

## Exact-Reuse Speedup vs Prefix

- `('agent_scaling', '2000', '1', '1', '1')`: p50=1.21x, p90=1.21x (prefix=97.3/97.3 ms, exact+hints=80.5/80.5 ms)
- `('agent_scaling', '2000', '1', '2', '1')`: p50=0.43x, p90=0.75x (prefix=36.8/91.1 ms, exact+hints=85.5/121.0 ms)
- `('agent_scaling', '2000', '1', '3', '1')`: p50=0.49x, p90=1.06x (prefix=41.8/93.2 ms, exact+hints=84.9/88.0 ms)
- `('agent_scaling', '2000', '1', '4', '1')`: p50=0.38x, p90=0.88x (prefix=38.0/87.7 ms, exact+hints=99.4/99.4 ms)
- `('agent_scaling', '2000', '1', '5', '1')`: p50=0.41x, p90=0.75x (prefix=40.4/90.0 ms, exact+hints=98.8/119.8 ms)
- `('agent_scaling_workflow', '2000', '1', '1', '1')`: p50=1.21x, p90=1.21x (prefix=97.3/97.3 ms, exact+hints=80.5/80.5 ms)
- `('agent_scaling_workflow', '2000', '1', '2', '1')`: p50=0.62x, p90=0.62x (prefix=127.9/127.9 ms, exact+hints=206.5/206.5 ms)
- `('agent_scaling_workflow', '2000', '1', '3', '1')`: p50=0.71x, p90=0.71x (prefix=176.7/176.7 ms, exact+hints=248.2/248.2 ms)
- `('agent_scaling_workflow', '2000', '1', '4', '1')`: p50=0.52x, p90=0.52x (prefix=197.3/197.3 ms, exact+hints=377.5/377.5 ms)
- `('agent_scaling_workflow', '2000', '1', '5', '1')`: p50=0.51x, p90=0.51x (prefix=245.9/245.9 ms, exact+hints=487.0/487.0 ms)
