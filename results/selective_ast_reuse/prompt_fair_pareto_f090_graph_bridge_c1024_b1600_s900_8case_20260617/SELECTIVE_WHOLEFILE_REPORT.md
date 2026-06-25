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
| `lossless_full_prefill` | 8/8 | 1.00 | 2307.8 | 737.4 | 1.00x | 420.4 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.00 | 0.00 | 1.0000 | 0.0000 | 1.00 | `{"strict-safe": 8}` |
| `whole_file_reuse_all` | 8/8 | 1.00 | 2237.4 | 665.6 | 1.11x | 1444.4 | 3270.5 | 1024.0 | 2977.0 | 1024.0 | 4118.8 | 1.00 | 1.00 | 0.9436 | 0.0564 | 0.88 | `{"strict-safe": 5, "lossy-acceptable": 2, "aggressive-diagnostic": 1}` |
| `selective_function_method_reuse` | 8/8 | 1.00 | 2257.9 | 686.6 | 1.07x | 1188.4 | 1176.4 | 768.0 | 2353.8 | 768.0 | 2940.2 | 0.75 | 0.75 | 0.9846 | 0.0154 | 1.00 | `{"strict-safe": 6, "lossy-acceptable": 2}` |
| `selective_extended_reuse` | 8/8 | 1.00 | 2253.9 | 682.3 | 1.08x | 1280.2 | 1967.9 | 859.9 | 2972.6 | 859.9 | 2693.5 | 0.75 | 0.88 | 0.9139 | 0.0861 | 0.75 | `{"strict-safe": 4, "lossy-acceptable": 2, "aggressive-diagnostic": 2}` |
| `selective_oracle_low_dnorm` | 8/8 | 1.00 | 2248.1 | 676.6 | 1.09x | 1280.2 | 1967.9 | 859.9 | 2972.6 | 859.9 | 2693.5 | 0.75 | 0.88 | 0.9139 | 0.0861 | 0.75 | `{"strict-safe": 4, "lossy-acceptable": 2, "aggressive-diagnostic": 2}` |
| `graph_aware_lossy` | 6/8 | 1.00 | 2248.7 | 676.8 | 1.09x | 1334.2 | 618.3 | 820.0 | 3138.3 | 820.0 | 2396.2 | 0.67 | 1.00 | 0.9046 | 0.0954 | 0.83 | `{"strict-safe": 3, "lossy-acceptable": 2, "aggressive-diagnostic": 1}` |

Interpretation: agents receive whole-file code_base prompts in every mode; only the internal AST spans exposed to the exact-content KV reuse gate differ.
`graph_aware_lossy` keeps the same whole-file prompt, then maps relation-selected graph bundles back to exact AST spans already present in that prompt.
`est reused` is driver-side span volume; `anchor match len` and `prefetch hit` are server-side evidence that KV was actually copied/protected.
`gap recompute` / `suffix copy` are context-aligned reuse telemetry. In this build, large-gap staged recompute is recorded as unsupported rather than approximated with zero-filled KV.
Accuracy buckets: `strict-safe` means token F1 is effectively 1.0; `lossy-acceptable` means token F1 >= 0.9; `aggressive-diagnostic` means token F1 is below that threshold.
`whole_file_reuse_all` is diagnostic only. The main method is `selective_function_method_reuse`.
This is the prompt-fair protocol: each mode starts from a fresh cache, runs the same Planner warmup, and then measures the same target prompt. Rows from prompt-unfair cases are excluded from mode aggregates.

## Skipped rows

```json
{
  "graph_aware_lossy": {
    "fair_planner_per_mode_skipped:ValueError;target_skipped:ValueError": 2
  }
}
```
