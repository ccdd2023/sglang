# Selective whole-file AST reuse

- Warmup protocol: `fair_planner_per_mode`
- Protocol meaning: Prompt-fair mechanism protocol: for each mode, flush, run the same Planner warmup, then measure the same target prompt; only runtime reuse anchors differ.
- Max generation tokens: `96`
- Context-aligned recompute gap: `False`
- Stage recompute gap diagnostic: `True`
- Bridge prefix anchors: `True`
- Bridge anchor max tokens: `0`
- Disable graph bridge prefix anchors: `True`
- Hybrid calibration policy: `results/selective_ast_reuse/pareto_calibration_p5500_semantic_gated_f0895_20260618/hybrid_calibration_policy.json`
- Hybrid calibration policy cases: `28`
- Hybrid calibration policy rules: `0`
- Hybrid calibration default action: ``
- Hybrid bridge tokens: min `4000`, max `5500`
- Hybrid bridge anchor max tokens: `0`
- Hybrid bridge max count per file: `0`
- Hybrid bridge source: `function`
- Hybrid large-bridge risk gate: min tokens `5200`, max bridge count `1`, max graph tokens `100`
- Max suffix copy len: `5500`
- Max planned suffix copy len: `5500`
- Suffix recompute head len: `0`
- Max recompute gap len: `0`
- Lossy acceptable F1 threshold: `0.9`
- Selective anchor max span tokens: `0`
- Selective anchor min span tokens: `200`
- Selective anchor max start token: `0`
- Anchor min total tokens: `0`
- Anchor max total tokens: `9000`
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
- Generic anchor lowspan max tokens: `399`
- Generic anchor lowspan suffix cap: `256`
- Generic anchor smallspan max tokens: `599`
- Generic anchor smallspan suffix cap: `128`
- Generic anchor midspan range: `0`-`0`
- Generic anchor midspan suffix cap: `0`

| mode | n_ok/n | prompt fair | avg elapsed ms | avg TTFT ms | speedup vs lossless | paired speedup | avg cached | est reused | anchor match len | gap recompute | suffix head | suffix copy | planned copy | trunc rate | context aligned | token F1 | F1 drop | acceptable | buckets |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `lossless_full_prefill` | 5/5 | 1.00 | 3195.8 |  |  |  | 870.8 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.00 | 0.00 | 1.0000 | 0.0000 | 1.00 | `{"strict-safe": 5}` |
| `hybrid_code_aware_lossy` | 5/5 | 1.00 | 2983.5 |  |  |  | 2932.8 | 1632.2 | 2062.0 | 11415.8 | 0.0 | 2062.0 | 2216.6 | 0.60 | 1.00 | 0.9070 | 0.0930 | 0.60 | `{"strict-safe": 3, "aggressive-diagnostic": 2}` |

Interpretation: agents receive whole-file code_base prompts in every mode; only the internal AST spans exposed to the exact-content KV reuse gate differ.
`graph_aware_lossy` keeps the same whole-file prompt, then maps relation-selected graph bundles back to exact AST spans already present in that prompt.
`est reused` is driver-side span volume; `anchor match len` and `prefetch hit` are server-side evidence that KV was actually copied/protected.
`gap recompute` / `suffix copy` are context-aligned reuse telemetry. In this build, large-gap staged recompute is recorded as unsupported rather than approximated with zero-filled KV.
Accuracy buckets: `strict-safe` means token F1 is effectively 1.0; `lossy-acceptable` means token F1 >= 0.9; `aggressive-diagnostic` means token F1 is below that threshold.
`whole_file_reuse_all` is diagnostic only. Main reported methods should be the prompt-fair selective/hybrid code-aware rows.
This is the prompt-fair protocol: each mode starts from a fresh cache, runs the same Planner warmup, and then measures the same target prompt. Rows from prompt-unfair cases are excluded from mode aggregates.
