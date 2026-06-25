# Selective whole-file AST reuse

- Warmup protocol: `fair_planner_per_mode`
- Protocol meaning: Prompt-fair mechanism protocol: for each mode, flush, run the same Planner warmup, then measure the same target prompt; only runtime reuse anchors differ.
- Max generation tokens: `96`
- Context-aligned recompute gap: `False`
- Stage recompute gap diagnostic: `True`
- Bridge prefix anchors: `False`
- Bridge anchor max tokens: `0`
- Disable graph bridge prefix anchors: `False`
- Hybrid calibration policy: `results/selective_ast_reuse/manual_p8000_cap_sweep_policies_20260618/cap3500.json`
- Hybrid calibration policy cases: `0`
- Hybrid calibration policy rules: `8`
- Hybrid calibration default action: `reject`
- Hybrid bridge tokens: min `1000`, max `8000`
- Hybrid large-bridge risk gate: min tokens `7800`, max bridge count `1`, max graph tokens `100`
- Max suffix copy len: `0`
- Max planned suffix copy len: `8000`
- Suffix recompute head len: `0`
- Max recompute gap len: `0`
- Lossy acceptable F1 threshold: `0.9`
- Selective anchor max span tokens: `0`
- Selective anchor min span tokens: `200`
- Selective anchor max start token: `0`
- Anchor min total tokens: `0`
- Anchor max total tokens: `12000`
- Anchor max total policy: `reject`
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
- Generic anchor lowspan max tokens: `0`
- Generic anchor lowspan suffix cap: `0`
- Generic anchor smallspan max tokens: `0`
- Generic anchor smallspan suffix cap: `0`
- Generic anchor midspan range: `0`-`0`
- Generic anchor midspan suffix cap: `0`

| mode | n_ok/n | prompt fair | avg elapsed ms | avg TTFT ms | speedup vs lossless | paired speedup | avg cached | est reused | anchor match len | gap recompute | suffix head | suffix copy | planned copy | trunc rate | context aligned | token F1 | F1 drop | acceptable | buckets |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `lossless_full_prefill` | 8/8 | 1.00 | 2180.6 | 615.9 | 1.00x |  | 469.8 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.00 | 0.00 | 1.0000 | 0.0000 | 1.00 | `{"strict-safe": 8}` |
| `hybrid_code_aware_lossy` | 8/8 | 1.00 | 1888.5 | 322.8 | 1.91x | 1.91x | 3969.8 | 2856.9 | 3500.0 | 16.8 | 0.0 | 3500.0 | 5993.6 | 1.00 | 1.00 | 0.8595 | 0.1405 | 0.62 | `{"strict-safe": 2, "lossy-acceptable": 3, "aggressive-diagnostic": 3}` |

Interpretation: agents receive whole-file code_base prompts in every mode; only the internal AST spans exposed to the exact-content KV reuse gate differ.
`graph_aware_lossy` keeps the same whole-file prompt, then maps relation-selected graph bundles back to exact AST spans already present in that prompt.
`est reused` is driver-side span volume; `anchor match len` and `prefetch hit` are server-side evidence that KV was actually copied/protected.
`gap recompute` / `suffix copy` are context-aligned reuse telemetry. In this build, large-gap staged recompute is recorded as unsupported rather than approximated with zero-filled KV.
Accuracy buckets: `strict-safe` means token F1 is effectively 1.0; `lossy-acceptable` means token F1 >= 0.9; `aggressive-diagnostic` means token F1 is below that threshold.
`whole_file_reuse_all` is diagnostic only. Main reported methods should be the prompt-fair selective/hybrid code-aware rows.
This is the prompt-fair protocol: each mode starts from a fresh cache, runs the same Planner warmup, and then measures the same target prompt. Rows from prompt-unfair cases are excluded from mode aggregates.
