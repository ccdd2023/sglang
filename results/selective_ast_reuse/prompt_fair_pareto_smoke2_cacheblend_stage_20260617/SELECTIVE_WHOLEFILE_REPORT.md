# Selective whole-file AST reuse

- Warmup protocol: `fair_planner_per_mode`
- Protocol meaning: Prompt-fair mechanism protocol: for each mode, flush, run the same Planner warmup, then measure the same target prompt; only runtime reuse anchors differ.
- Context-aligned recompute gap: `True`
- Stage recompute gap diagnostic: `True`
- Bridge prefix anchors: `False`

| mode | n_ok/n | prompt fair | avg elapsed ms | avg TTFT ms | speedup vs lossless | avg cached | est reused | anchor match len | gap recompute | suffix copy | context aligned | token F1 | F1 drop | acceptable | buckets |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `lossless_full_prefill` | 2/2 | 1.00 | 2563.0 | 977.9 | 1.00x | 917.5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.00 | 1.0000 | 0.0000 | 1.00 | `{"strict-safe": 2}` |
| `whole_file_reuse_all` | 2/2 | 1.00 | 2603.9 | 1016.5 | 0.96x | 917.5 | 3854.5 | 0.0 | 2322.5 | 0.0 | 0.00 | 0.8054 | 0.1946 | 0.00 | `{"aggressive-diagnostic": 2}` |
| `selective_function_method_reuse` | 2/2 | 1.00 | 2628.3 | 1014.7 | 0.96x | 917.5 | 1669.0 | 0.0 | 9239.5 | 0.0 | 0.00 | 0.7058 | 0.2942 | 0.00 | `{"aggressive-diagnostic": 2}` |
| `selective_extended_reuse` | 0/2 | 0.00 |  |  |  | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.00 | 0.0000 | 0.0000 | 0.00 | `{}` |
| `selective_oracle_low_dnorm` | 0/2 | 0.00 |  |  |  | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.00 | 0.0000 | 0.0000 | 0.00 | `{}` |
| `graph_aware_lossy` | 2/2 | 1.00 | 2601.0 | 1013.0 | 0.97x | 917.5 | 821.5 | 0.0 | 6711.5 | 0.0 | 0.00 | 0.9722 | 0.0278 | 0.50 | `{"strict-safe": 1, "aggressive-diagnostic": 1}` |

Interpretation: agents receive whole-file code_base prompts in every mode; only the internal AST spans exposed to the exact-content KV reuse gate differ.
`graph_aware_lossy` keeps the same whole-file prompt, then maps relation-selected graph bundles back to exact AST spans already present in that prompt.
`est reused` is driver-side span volume; `anchor match len` and `prefetch hit` are server-side evidence that KV was actually copied/protected.
`gap recompute` / `suffix copy` are context-aligned reuse telemetry. In this build, large-gap staged recompute is recorded as unsupported rather than approximated with zero-filled KV.
Accuracy buckets: `strict-safe` means token F1 is effectively 1.0; `lossy-acceptable` means token F1 >= 0.95; `aggressive-diagnostic` means token F1 < 0.95.
`whole_file_reuse_all` is diagnostic only. The main method is `selective_function_method_reuse`.
This is the prompt-fair protocol: each mode starts from a fresh cache, runs the same Planner warmup, and then measures the same target prompt. Rows from prompt-unfair cases are excluded from mode aggregates.

## Skipped rows

```json
{
  "selective_extended_reuse": {
    "fair_planner_per_mode_skipped:ValueError;target_skipped:ValueError": 2
  },
  "selective_oracle_low_dnorm": {
    "fair_planner_per_mode_skipped:ValueError;target_skipped:ValueError": 2
  }
}
```
