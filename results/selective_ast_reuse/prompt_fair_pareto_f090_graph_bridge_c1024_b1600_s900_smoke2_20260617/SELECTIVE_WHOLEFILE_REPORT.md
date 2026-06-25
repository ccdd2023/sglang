# Selective whole-file AST reuse

- Warmup protocol: `fair_planner_per_mode`
- Protocol meaning: Prompt-fair mechanism protocol: for each mode, flush, run the same Planner warmup, then measure the same target prompt; only runtime reuse anchors differ.
- Context-aligned recompute gap: `True`
- Stage recompute gap diagnostic: `True`
- Bridge prefix anchors: `True`
- Max suffix copy len: `1024`
- Lossy acceptable F1 threshold: `0.9`
- Graph anchor token budget: `1600`
- Graph anchor max span tokens: `900`

| mode | n_ok/n | prompt fair | avg elapsed ms | avg TTFT ms | speedup vs lossless | avg cached | est reused | anchor match len | gap recompute | suffix copy | planned copy | trunc rate | context aligned | token F1 | F1 drop | acceptable | buckets |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `lossless_full_prefill` | 2/2 | 1.00 | 2553.1 | 967.5 | 1.00x | 917.5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.00 | 0.00 | 1.0000 | 0.0000 | 1.00 | `{"strict-safe": 2}` |
| `whole_file_reuse_all` | 2/2 | 1.00 | 2504.8 | 917.0 | 1.06x | 1941.5 | 3854.5 | 1024.0 | 4938.0 | 1024.0 | 2880.0 | 1.00 | 1.00 | 0.9767 | 0.0233 | 1.00 | `{"strict-safe": 1, "lossy-acceptable": 1}` |
| `selective_function_method_reuse` | 2/2 | 1.00 | 2488.3 | 901.2 | 1.07x | 1941.5 | 1669.0 | 1024.0 | 4947.0 | 1024.0 | 4351.5 | 1.00 | 1.00 | 1.0000 | 0.0000 | 1.00 | `{"strict-safe": 2}` |
| `selective_extended_reuse` | 2/2 | 1.00 | 2480.6 | 892.9 | 1.08x | 1941.5 | 2588.5 | 1024.0 | 4938.0 | 1024.0 | 2880.0 | 1.00 | 1.00 | 0.9767 | 0.0233 | 1.00 | `{"strict-safe": 1, "lossy-acceptable": 1}` |
| `selective_oracle_low_dnorm` | 2/2 | 1.00 | 2480.3 | 893.0 | 1.08x | 1941.5 | 2588.5 | 1024.0 | 4938.0 | 1024.0 | 2880.0 | 1.00 | 1.00 | 0.9767 | 0.0233 | 1.00 | `{"strict-safe": 1, "lossy-acceptable": 1}` |
| `graph_aware_lossy` | 2/2 | 1.00 | 2528.6 | 941.4 | 1.03x | 1642.0 | 149.5 | 724.5 | 4947.0 | 724.5 | 1973.5 | 0.50 | 1.00 | 1.0000 | 0.0000 | 1.00 | `{"strict-safe": 2}` |

Interpretation: agents receive whole-file code_base prompts in every mode; only the internal AST spans exposed to the exact-content KV reuse gate differ.
`graph_aware_lossy` keeps the same whole-file prompt, then maps relation-selected graph bundles back to exact AST spans already present in that prompt.
`est reused` is driver-side span volume; `anchor match len` and `prefetch hit` are server-side evidence that KV was actually copied/protected.
`gap recompute` / `suffix copy` are context-aligned reuse telemetry. In this build, large-gap staged recompute is recorded as unsupported rather than approximated with zero-filled KV.
Accuracy buckets: `strict-safe` means token F1 is effectively 1.0; `lossy-acceptable` means token F1 >= 0.9; `aggressive-diagnostic` means token F1 is below that threshold.
`whole_file_reuse_all` is diagnostic only. The main method is `selective_function_method_reuse`.
This is the prompt-fair protocol: each mode starts from a fresh cache, runs the same Planner warmup, and then measures the same target prompt. Rows from prompt-unfair cases are excluded from mode aggregates.
