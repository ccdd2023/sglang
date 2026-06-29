# Selective whole-file AST reuse

- Warmup protocol: `natural_planner`
- Protocol meaning: Realistic agent protocol: measure a cold lossless reference, flush, run one Planner-style warmup, then measure reuse target modes against shared cache.
- Max generation tokens: `96`
- Server random seed: `42`
- Disable overlap schedule: `True`
- Context-aligned recompute gap: `True`
- Stage recompute gap diagnostic: `True`
- Multi-anchor copy diagnostic: `False`
- Bridge prefix anchors: `False`
- Bridge anchor max tokens: `0`
- Disable graph bridge prefix anchors: `False`
- Load graph bundles for selection only: `False`
- Hybrid calibration policy: `None`
- Hybrid calibration policy cases: `0`
- Hybrid calibration policy rules: `0`
- Hybrid calibration default action: ``
- Hybrid bridge tokens: min `4000`, max `0`
- Hybrid bridge anchor max tokens: `0`
- Hybrid bridge max count per file: `0`
- Include hybrid bridge seed spans: `False`
- Hybrid bridge source: `function`
- Hybrid large-bridge risk gate: min tokens `0`, max bridge count `0`, max graph tokens `0`
- Max suffix copy len: `2048`
- Max planned suffix copy len: `0`
- Suffix recompute head len: `0`
- Max recompute gap len: `12000`
- Lossy acceptable F1 threshold: `0.9`
- Selective anchor max span tokens: `0`
- Selective anchor min span tokens: `0`
- Selective anchor max start token: `0`
- Anchor min total tokens: `0`
- Anchor max total tokens: `0`
- Anchor max total policy: `reject`
- Selection min estimated reused tokens: `0`
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
| `lossless_full_prefill` | 3/3 | 1.00 | 2676.2 |  |  |  | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.00 | 0.00 | 1.0000 | 0.0000 | 1.00 | `{"strict-safe": 3}` |
| `selective_function_method_reuse` | 3/3 | 1.00 | 2771.8 |  |  |  | 64.7 | 687.3 | 58.7 | 3301.7 | 0.0 | 58.7 | 58.7 | 0.00 | 0.33 | 0.9422 | 0.0578 | 0.67 | `{"strict-safe": 1, "lossy-acceptable": 1, "aggressive-diagnostic": 1}` |

Interpretation: agents receive whole-file code_base prompts in every mode; only the internal AST spans exposed to the exact-content KV reuse gate differ.
`graph_aware_lossy` keeps the same whole-file prompt, then maps relation-selected graph bundles back to exact AST spans already present in that prompt.
`est reused` is driver-side span volume; `anchor match len` and `prefetch hit` are server-side evidence that KV was actually copied/protected.
`gap recompute` / `suffix copy` are context-aligned reuse telemetry. In this build, large-gap staged recompute is recorded as unsupported rather than approximated with zero-filled KV.
Accuracy buckets: `strict-safe` means token F1 is effectively 1.0; `lossy-acceptable` means token F1 >= 0.9; `aggressive-diagnostic` means token F1 is below that threshold.
`whole_file_reuse_all` is diagnostic only. Main reported methods should be the prompt-fair selective/hybrid code-aware rows.
This is the realistic protocol: lossless is measured cold as the reference, then one Planner-style warmup is shared by later reuse target modes; target order is recorded in `summary.json`.
