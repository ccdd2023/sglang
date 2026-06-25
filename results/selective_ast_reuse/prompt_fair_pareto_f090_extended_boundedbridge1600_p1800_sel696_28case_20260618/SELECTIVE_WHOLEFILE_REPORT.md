# Selective whole-file AST reuse

- Warmup protocol: `fair_planner_per_mode`
- Protocol meaning: Prompt-fair mechanism protocol: for each mode, flush, run the same Planner warmup, then measure the same target prompt; only runtime reuse anchors differ.
- Context-aligned recompute gap: `True`
- Stage recompute gap diagnostic: `True`
- Bridge prefix anchors: `True`
- Bridge anchor max tokens: `1600`
- Max suffix copy len: `0`
- Max planned suffix copy len: `1800`
- Suffix recompute head len: `0`
- Max recompute gap len: `0`
- Lossy acceptable F1 threshold: `0.9`
- Selective anchor max span tokens: `0`
- Anchor min total tokens: `0`
- Selection min estimated reused tokens: `696`
- Graph anchor token budget: `1600`
- Graph anchor max span tokens: `900`

| mode | n_ok/n | prompt fair | avg elapsed ms | avg TTFT ms | speedup vs lossless | avg cached | est reused | anchor match len | gap recompute | suffix head | suffix copy | planned copy | trunc rate | context aligned | token F1 | F1 drop | acceptable | buckets |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `lossless_full_prefill` | 28/28 | 1.00 | 2096.0 | 536.1 | 1.00x | 602.9 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.00 | 0.00 | 1.0000 | 0.0000 | 1.00 | `{"strict-safe": 28}` |
| `selective_extended_reuse` | 28/28 | 1.00 | 2056.7 | 496.7 | 1.08x | 1096.2 | 936.1 | 493.4 | 5.4 | 0.0 | 493.4 | 560.5 | 0.00 | 0.32 | 0.9484 | 0.0516 | 0.86 | `{"strict-safe": 21, "lossy-acceptable": 3, "aggressive-diagnostic": 4}` |

Interpretation: agents receive whole-file code_base prompts in every mode; only the internal AST spans exposed to the exact-content KV reuse gate differ.
`graph_aware_lossy` keeps the same whole-file prompt, then maps relation-selected graph bundles back to exact AST spans already present in that prompt.
`est reused` is driver-side span volume; `anchor match len` and `prefetch hit` are server-side evidence that KV was actually copied/protected.
`gap recompute` / `suffix copy` are context-aligned reuse telemetry. In this build, large-gap staged recompute is recorded as unsupported rather than approximated with zero-filled KV.
Accuracy buckets: `strict-safe` means token F1 is effectively 1.0; `lossy-acceptable` means token F1 >= 0.9; `aggressive-diagnostic` means token F1 is below that threshold.
`whole_file_reuse_all` is diagnostic only. The main method is `selective_function_method_reuse`.
This is the prompt-fair protocol: each mode starts from a fresh cache, runs the same Planner warmup, and then measures the same target prompt. Rows from prompt-unfair cases are excluded from mode aggregates.
