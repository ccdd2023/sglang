# KVCOMM TTFT Stress Report

This run is prefill/TTFT dominant and is intended to complement the realistic 100-case E2E table.

## Output Schema

`ttft_ms`, `e2e_ms`, `prompt_tokens`, `cached_tokens`, `cached_ratio`, `exact_hit`, `match_reason`, `matched_content_signature`, `output_exact_match_vs_baseline`, `output_token_f1_vs_baseline`, `mode`, `case_id`, `agent_id`, `segment_count`, `max_file_chars`.

## Row Counts

- agent_scaling: 75
- agent_scaling_workflow: 25

## Summary Groups

- `agent_scaling_workflow|exact_reuse_no_hints|8000|1|1|1`: n=1, avg TTFT=91.6 ms, p50=91.6 ms, p90=91.6 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_no_hints|8000|1|2|1`: n=1, avg TTFT=212.7 ms, p50=212.7 ms, p90=212.7 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_no_hints|8000|1|3|1`: n=1, avg TTFT=260.1 ms, p50=260.1 ms, p90=260.1 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_no_hints|8000|1|4|1`: n=1, avg TTFT=1550.4 ms, p50=1550.4 ms, p90=1550.4 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_no_hints|8000|1|5|1`: n=1, avg TTFT=3547.8 ms, p50=3547.8 ms, p90=3547.8 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_plus_code_hints|8000|1|1|1`: n=1, avg TTFT=83.8 ms, p50=83.8 ms, p90=83.8 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_plus_code_hints|8000|1|2|1`: n=1, avg TTFT=209.3 ms, p50=209.3 ms, p90=209.3 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_plus_code_hints|8000|1|3|1`: n=1, avg TTFT=312.3 ms, p50=312.3 ms, p90=312.3 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_plus_code_hints|8000|1|4|1`: n=1, avg TTFT=1721.7 ms, p50=1721.7 ms, p90=1721.7 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|exact_reuse_plus_code_hints|8000|1|5|1`: n=1, avg TTFT=3143.7 ms, p50=3143.7 ms, p90=3143.7 ms, exact hit=1.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|hints_no_exact|8000|1|1|1`: n=1, avg TTFT=238.4 ms, p50=238.4 ms, p90=238.4 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|hints_no_exact|8000|1|2|1`: n=1, avg TTFT=513.5 ms, p50=513.5 ms, p90=513.5 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|hints_no_exact|8000|1|3|1`: n=1, avg TTFT=763.6 ms, p50=763.6 ms, p90=763.6 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|hints_no_exact|8000|1|4|1`: n=1, avg TTFT=1022.2 ms, p50=1022.2 ms, p90=1022.2 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|hints_no_exact|8000|1|5|1`: n=1, avg TTFT=1286.2 ms, p50=1286.2 ms, p90=1286.2 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|placeholder_knn_reuse|8000|1|1|1`: n=1, avg TTFT=261.2 ms, p50=261.2 ms, p90=261.2 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|placeholder_knn_reuse|8000|1|2|1`: n=1, avg TTFT=166.6 ms, p50=166.6 ms, p90=166.6 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|placeholder_knn_reuse|8000|1|3|1`: n=1, avg TTFT=268.5 ms, p50=268.5 ms, p90=268.5 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|placeholder_knn_reuse|8000|1|4|1`: n=1, avg TTFT=744.9 ms, p50=744.9 ms, p90=744.9 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|placeholder_knn_reuse|8000|1|5|1`: n=1, avg TTFT=1063.4 ms, p50=1063.4 ms, p90=1063.4 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|prefix_cache_only|8000|1|1|1`: n=1, avg TTFT=256.4 ms, p50=256.4 ms, p90=256.4 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|prefix_cache_only|8000|1|2|1`: n=1, avg TTFT=296.4 ms, p50=296.4 ms, p90=296.4 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|prefix_cache_only|8000|1|3|1`: n=1, avg TTFT=346.6 ms, p50=346.6 ms, p90=346.6 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|prefix_cache_only|8000|1|4|1`: n=1, avg TTFT=391.9 ms, p50=391.9 ms, p90=391.9 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling_workflow|prefix_cache_only|8000|1|5|1`: n=1, avg TTFT=439.6 ms, p50=439.6 ms, p90=439.6 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'': 1}
- `agent_scaling|exact_reuse_no_hints|8000|1|1|1`: n=1, avg TTFT=91.6 ms, p50=91.6 ms, p90=91.6 ms, exact hit=1.00, device hit=1.00, consumed=1.00, protected=0.0, F1=1.0000, status={'consumed': 1}
- `agent_scaling|exact_reuse_no_hints|8000|1|2|1`: n=2, avg TTFT=106.3 ms, p50=92.7 ms, p90=119.9 ms, exact hit=1.00, device hit=1.00, consumed=0.50, protected=0.0, F1=1.0000, status={'consumed': 1, 'device_hit_without_consumed': 1}
- `agent_scaling|exact_reuse_no_hints|8000|1|3|1`: n=3, avg TTFT=86.7 ms, p50=83.9 ms, p90=93.7 ms, exact hit=1.00, device hit=1.00, consumed=0.33, protected=0.0, F1=1.0000, status={'consumed': 1, 'device_hit_without_consumed': 2}
- `agent_scaling|exact_reuse_no_hints|8000|1|4|1`: n=4, avg TTFT=387.6 ms, p50=665.9 ms, p90=670.4 ms, exact hit=1.00, device hit=0.50, consumed=0.25, protected=0.0, F1=1.0000, status={'consumed': 1, 'no_fast_path': 2, 'device_hit_without_consumed': 1}
- `agent_scaling|exact_reuse_no_hints|8000|1|5|1`: n=5, avg TTFT=709.6 ms, p50=832.1 ms, p90=905.5 ms, exact hit=1.00, device hit=0.20, consumed=0.20, protected=0.0, F1=1.0000, status={'no_fast_path': 4, 'consumed': 1}
- `agent_scaling|exact_reuse_plus_code_hints|8000|1|1|1`: n=1, avg TTFT=83.8 ms, p50=83.8 ms, p90=83.8 ms, exact hit=1.00, device hit=1.00, consumed=1.00, protected=4490.0, F1=1.0000, status={'consumed': 1}
- `agent_scaling|exact_reuse_plus_code_hints|8000|1|2|1`: n=2, avg TTFT=104.6 ms, p50=95.5 ms, p90=113.8 ms, exact hit=1.00, device hit=1.00, consumed=1.00, protected=4490.0, F1=1.0000, status={'consumed': 2}
- `agent_scaling|exact_reuse_plus_code_hints|8000|1|3|1`: n=3, avg TTFT=104.1 ms, p50=102.2 ms, p90=109.1 ms, exact hit=1.00, device hit=1.00, consumed=1.00, protected=4490.0, F1=1.0000, status={'consumed': 3}
- `agent_scaling|exact_reuse_plus_code_hints|8000|1|4|1`: n=4, avg TTFT=430.4 ms, p50=729.5 ms, p90=795.8 ms, exact hit=1.00, device hit=1.00, consumed=0.50, protected=4490.0, F1=1.0000, status={'consumed': 2, 'protected_not_consumed:no_anchor_match': 2}
- `agent_scaling|exact_reuse_plus_code_hints|8000|1|5|1`: n=5, avg TTFT=628.7 ms, p50=924.5 ms, p90=1044.2 ms, exact hit=1.00, device hit=1.00, consumed=0.40, protected=4490.0, F1=1.0000, status={'protected_not_consumed:no_anchor_match': 3, 'consumed': 2}
- `agent_scaling|hints_no_exact|8000|1|1|1`: n=1, avg TTFT=238.4 ms, p50=238.4 ms, p90=238.4 ms, exact hit=0.00, device hit=1.00, consumed=0.00, protected=2245.0, F1=1.0000, status={'protected_not_consumed:no_anchor_match': 1}
- `agent_scaling|hints_no_exact|8000|1|2|1`: n=2, avg TTFT=256.7 ms, p50=256.4 ms, p90=257.0 ms, exact hit=0.00, device hit=1.00, consumed=0.00, protected=2245.0, F1=1.0000, status={'protected_not_consumed:no_anchor_match': 2}
- `agent_scaling|hints_no_exact|8000|1|3|1`: n=3, avg TTFT=254.5 ms, p50=256.8 ms, p90=257.5 ms, exact hit=0.00, device hit=1.00, consumed=0.00, protected=2245.0, F1=1.0000, status={'protected_not_consumed:no_anchor_match': 3}
- `agent_scaling|hints_no_exact|8000|1|4|1`: n=4, avg TTFT=255.6 ms, p50=257.7 ms, p90=257.9 ms, exact hit=0.00, device hit=1.00, consumed=0.00, protected=2245.0, F1=1.0000, status={'protected_not_consumed:no_anchor_match': 4}
- `agent_scaling|hints_no_exact|8000|1|5|1`: n=5, avg TTFT=257.2 ms, p50=257.9 ms, p90=259.1 ms, exact hit=0.00, device hit=1.00, consumed=0.00, protected=2245.0, F1=1.0000, status={'protected_not_consumed:no_anchor_match': 5}
- `agent_scaling|placeholder_knn_reuse|8000|1|1|1`: n=1, avg TTFT=261.2 ms, p50=261.2 ms, p90=261.2 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 1}
- `agent_scaling|placeholder_knn_reuse|8000|1|2|1`: n=2, avg TTFT=83.3 ms, p50=79.7 ms, p90=86.9 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 2}
- `agent_scaling|placeholder_knn_reuse|8000|1|3|1`: n=3, avg TTFT=89.5 ms, p50=88.5 ms, p90=96.3 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 3}
- `agent_scaling|placeholder_knn_reuse|8000|1|4|1`: n=4, avg TTFT=186.2 ms, p50=279.3 ms, p90=287.4 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 4}
- `agent_scaling|placeholder_knn_reuse|8000|1|5|1`: n=5, avg TTFT=212.7 ms, p50=283.6 ms, p90=293.5 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 5}
- `agent_scaling|prefix_cache_only|8000|1|1|1`: n=1, avg TTFT=256.4 ms, p50=256.4 ms, p90=256.4 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 1}
- `agent_scaling|prefix_cache_only|8000|1|2|1`: n=2, avg TTFT=148.2 ms, p50=44.2 ms, p90=252.2 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 2}
- `agent_scaling|prefix_cache_only|8000|1|3|1`: n=3, avg TTFT=115.5 ms, p50=47.2 ms, p90=253.5 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 3}
- `agent_scaling|prefix_cache_only|8000|1|4|1`: n=4, avg TTFT=98.0 ms, p50=46.8 ms, p90=252.2 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 4}
- `agent_scaling|prefix_cache_only|8000|1|5|1`: n=5, avg TTFT=87.9 ms, p50=47.2 ms, p90=253.0 ms, exact hit=0.00, device hit=0.00, consumed=0.00, protected=0.0, F1=1.0000, status={'no_fast_path': 5}

## Exact-Reuse Speedup vs Prefix

- `('agent_scaling', '8000', '1', '1', '1')`: p50=3.06x, p90=3.06x (prefix=256.4/256.4 ms, exact+hints=83.8/83.8 ms)
- `('agent_scaling', '8000', '1', '2', '1')`: p50=0.46x, p90=2.22x (prefix=44.2/252.2 ms, exact+hints=95.5/113.8 ms)
- `('agent_scaling', '8000', '1', '3', '1')`: p50=0.46x, p90=2.32x (prefix=47.2/253.5 ms, exact+hints=102.2/109.1 ms)
- `('agent_scaling', '8000', '1', '4', '1')`: p50=0.06x, p90=0.32x (prefix=46.8/252.2 ms, exact+hints=729.5/795.8 ms)
- `('agent_scaling', '8000', '1', '5', '1')`: p50=0.05x, p90=0.24x (prefix=47.2/253.0 ms, exact+hints=924.5/1044.2 ms)
- `('agent_scaling_workflow', '8000', '1', '1', '1')`: p50=3.06x, p90=3.06x (prefix=256.4/256.4 ms, exact+hints=83.8/83.8 ms)
- `('agent_scaling_workflow', '8000', '1', '2', '1')`: p50=1.42x, p90=1.42x (prefix=296.4/296.4 ms, exact+hints=209.3/209.3 ms)
- `('agent_scaling_workflow', '8000', '1', '3', '1')`: p50=1.11x, p90=1.11x (prefix=346.6/346.6 ms, exact+hints=312.3/312.3 ms)
- `('agent_scaling_workflow', '8000', '1', '4', '1')`: p50=0.23x, p90=0.23x (prefix=391.9/391.9 ms, exact+hints=1721.7/1721.7 ms)
- `('agent_scaling_workflow', '8000', '1', '5', '1')`: p50=0.14x, p90=0.14x (prefix=439.6/439.6 ms, exact+hints=3143.7/3143.7 ms)
