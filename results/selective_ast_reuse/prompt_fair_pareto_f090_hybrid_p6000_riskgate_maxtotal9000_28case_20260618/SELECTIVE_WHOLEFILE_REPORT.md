# Selective whole-file AST reuse

- Warmup protocol: `fair_planner_per_mode`
- Protocol meaning: Prompt-fair mechanism protocol: for each mode, flush, run the same Planner warmup, then measure the same target prompt; only runtime reuse anchors differ.
- Context-aligned recompute gap: `False`
- Stage recompute gap diagnostic: `True`
- Bridge prefix anchors: `True`
- Bridge anchor max tokens: `0`
- Disable graph bridge prefix anchors: `True`
- Hybrid bridge tokens: min `4000`, max `6000`
- Hybrid large-bridge risk gate: min tokens `5200`, max bridge count `1`, max graph tokens `100`
- Max suffix copy len: `6000`
- Max planned suffix copy len: `6000`
- Suffix recompute head len: `0`
- Max recompute gap len: `0`
- Lossy acceptable F1 threshold: `0.9`
- Selective anchor max span tokens: `0`
- Selective anchor min span tokens: `200`
- Selective anchor max start token: `0`
- Anchor min total tokens: `0`
- Anchor max total tokens: `9000`
- Selection min estimated reused tokens: `200`
- Excluded anchor granularities: ``
- Graph anchor token budget: `1600`
- Graph anchor max span tokens: `900`
- Graph anchor lowspan max tokens: `0`
- Graph anchor lowspan suffix cap: `0`
- Graph anchor smallspan max tokens: `0`
- Graph anchor smallspan suffix cap: `0`
- Graph anchor midspan range: `0`-`0`
- Graph anchor midspan suffix cap: `0`
- Generic anchor lowspan max tokens: `399`
- Generic anchor lowspan suffix cap: `256`
- Generic anchor smallspan max tokens: `599`
- Generic anchor smallspan suffix cap: `128`
- Generic anchor midspan range: `600`-`900`
- Generic anchor midspan suffix cap: `0`

| mode | n_ok/n | prompt fair | avg elapsed ms | avg TTFT ms | speedup vs lossless | paired speedup | avg cached | est reused | anchor match len | gap recompute | suffix head | suffix copy | planned copy | trunc rate | context aligned | token F1 | F1 drop | acceptable | buckets |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `lossless_full_prefill` | 28/28 | 1.00 | 2381.9 | 806.9 | 1.00x |  | 602.9 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.00 | 0.00 | 1.0000 | 0.0000 | 1.00 | `{"strict-safe": 28}` |
| `hybrid_code_aware_lossy` | 28/28 | 1.00 | 2214.4 | 638.6 | 1.26x | 1.26x | 2483.9 | 1018.5 | 1881.0 | 365.2 | 0.0 | 1881.0 | 1895.1 | 0.07 | 0.50 | 0.9833 | 0.0167 | 0.96 | `{"strict-safe": 18, "lossy-acceptable": 9, "aggressive-diagnostic": 1}` |

Interpretation: agents receive whole-file code_base prompts in every mode; only the internal AST spans exposed to the exact-content KV reuse gate differ.
`graph_aware_lossy` keeps the same whole-file prompt, then maps relation-selected graph bundles back to exact AST spans already present in that prompt.
`est reused` is driver-side span volume; `anchor match len` and `prefetch hit` are server-side evidence that KV was actually copied/protected.
`gap recompute` / `suffix copy` are context-aligned reuse telemetry. In this build, large-gap staged recompute is recorded as unsupported rather than approximated with zero-filled KV.
Accuracy buckets: `strict-safe` means token F1 is effectively 1.0; `lossy-acceptable` means token F1 >= 0.9; `aggressive-diagnostic` means token F1 is below that threshold.
`whole_file_reuse_all` is diagnostic only. The main method is `selective_function_method_reuse`.
This is the prompt-fair protocol: each mode starts from a fresh cache, runs the same Planner warmup, and then measures the same target prompt. Rows from prompt-unfair cases are excluded from mode aggregates.
