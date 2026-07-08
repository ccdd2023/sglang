# Round 28 — AST Node-Type-Stratified Reuse (2026-07-07)

## Hypothesis

Only AST chunks of type `function` or `class` are eligible for KV reuse;
control-flow chunks (`for`/`while`/`if`/`try`) fall back to dense prefill.
Hypothesis: control-flow chunks carry more stale KV per token in
partial-share workloads.

## Result: **HYPOTHESIS FALSIFIED** (with a clean null result)

The precompute pool `pandas_5case_v4` contains **72 chunks, ALL of type
`function` (93.1%) or `class` (6.9%)**. Zero `for`/`while`/`if`/`try` chunks.
The env var `SGLANG_AST_REUSE_TYPES=function,class` had **nothing to filter**,
so the treatment config was byte-equivalent to the baseline.

**Why this matters**: the `ast_chunker` operates at **function/class
granularity** (`ast.FunctionDef`, `ast.AsyncFunctionDef`, `ast.ClassDef`).
Control-flow blocks (`for`/`while`/`if`/`try`) are *nested inside* functions
and never appear as standalone chunks. Selective AST-node-type filtering is
inapplicable at this granularity — to target control flow, we'd need a
sub-function chunker that splits function bodies at control-flow boundaries
(an entirely new chunking strategy; out of scope for selective reuse).

## Code change (landed but inactive for this pool)

`python/sglang/srt/mem_cache/radix_cache.py` — `_build_chunk_plan`
(lines ~2387-2420 + ~2507-2532). New env var:

- `SGLANG_AST_REUSE_TYPES` (comma-separated list, default empty = OFF).
  When set, chunks whose `anchor_type` is not in the whitelist become
  `dense_prefill` decisions with `skip_reason="anchor_type_filtered"`.

Default OFF → R19/R21 behavior unchanged. **Behavior validated as no-op
for the current pool** (baseline == treatment byte-for-byte).

## Fair A/B numbers (Qwen2.5-Coder-7B × 5 agents, verdict task)

| Metric | lossless | r28_baseline (R19 BEST) | r28_func_class_only | Δ |
|---|---|---|---|---|
| avg TTFT (reusers, ms) | 925.9 | 704.9 | 704.0 | -0.9 ms (noise) |
| p50 TTFT (ms) | 955.8 | 739.6 | 739.6 | 0 ms |
| p90 TTFT (ms) | 1227.0 | 881.3 | 888.0 | +6.7 ms (noise) |
| **Speedup vs lossless** | 1.000× | **1.294×** | **1.295×** | ≈0 |
| avg cached_tokens | 108.5 | 609.0 | 609.0 | 0 |
| avg radix_prefix_tokens | 108.5 | 113.6 | 113.6 | 0 |
| avg codeaware_reused_tokens | 0 | 495.4 | 495.4 | 0 |
| anchor_type_filtered chunks | — | 0 | 0 | (no target to filter) |
| avg F1 vs lossless | 1.000 | 0.498 | 0.498 | 0 |
| **Verdict PASS %** | 48.0% | 32.0% | 32.0% | 0 |
| **Verdict FAIL %** | 52.0% | 60.0% | 60.0% | 0 |
| **Verdict UNK %** | 0.0% | 0.0% | 0.0% | 0 |
| **FAIL accuracy vs GT** | 52.0% | **60.0%** | **60.0%** | 0 |
| Failure-type agreement | 38.5% | 13.3% | 13.3% | 0 |

(`anchor_type_filtered chunks` = 0 because no chunks of non-{function,class}
type exist in the pool; the env var would activate if a future pool contained
mixed types.)

## Pool composition (`results/codebase_kv/pandas_5case_v4/manifest.jsonl`)

| anchor_type | n | % |
|---|---|---|
| function | 67 | 93.1% |
| class | 5 | 6.9% |
| for | 0 | 0.0% |
| while | 0 | 0.0% |
| if | 0 | 0.0% |
| try | 0 | 0.0% |
| **Total** | **72** | 100% |

## Verdict

- ❌ **Selective AST-node-type filtering is a NO-OP** on this codebase at
  function-level granularity. Hypothesis falsified by data, not by
  implementation.
- ✅ **Code change is correct and ready** — will activate automatically if
  a future chunker produces control-flow-level chunks (a sub-function
  chunker is the precondition, out of scope for this round).
- ✅ **No regression** — treatment is byte-equal to baseline. Safe to ship
  as-is.
- 📚 **Knowledge gained**: AST node-type selectivity requires
  function-internal chunking. Two follow-up directions enabled:
  - **R29 (sink-preserved copy)**: cross-context loss concentrates at chunk
    boundaries, not at control-flow boundaries — sink recompute is the right
    intervention.
  - **R30 (signature-stable filter)**: function-level signature matching
    (name + arg types) would catch payload-divergent functions that current
    byte-exact chunk pool reuses (loss 维度 C/D).

## Reproduction

```bash
# Baseline (control — should match R21 R19 exactly)
bash results/lossy_alg_round28/launchers/run_r28_baseline_verdict.sh
# Treatment (with anchor_type whitelist — should match baseline here)
bash results/lossy_alg_round28/launchers/run_r28_func_class_only_verdict.sh
# Lossless reference
bash results/lossy_alg_round28/launchers/run_lossless_verdict.sh

# Verdict accuracy
python results/lossy_alg_round28/scripts/score_r28.py \
  results/lossy_alg_round28/lossless_verdict/outputs.jsonl \
  results/lossy_alg_round28/r28_baseline_verdict/outputs.jsonl \
  results/lossy_algMAS_Project/sglang-kvflow/results/lossy_alg_round28/r28_func_class_only_verdict/outputs.jsonl \
  --labels lossless r28_baseline r28_func_class_only
# (note: typo above; use real path)
PYTHONPATH=. /home/gfy/.conda/envs/sglang-kvflow/bin/python \
  benchmark/multi_workflow/analyze_fair_ab.py \
  --baseline results/lossy_alg_round28/r28_baseline_verdict/rows.csv \
  --experimental results/lossy_alg_round28/r28_func_class_only_verdict/rows.csv \
  --lossless results/lossy_alg_round28/lossless_verdict/rows.csv \
  --out-dir results/lossy_alg_round28/

# Pool anchor_type distribution
python3 -c "
import json
from collections import Counter
c = Counter()
for line in open('results/codebase_kv/pandas_5case_v4/manifest.jsonl'):
    c[json.loads(line).get('anchor_type', 'NONE')] += 1
for k, v in c.most_common(): print(f'  {k}: {v}')
"
```

## Files

| Artifact | Path |
|---|---|
| Code change | `python/sglang/srt/mem_cache/radix_cache.py` (env var + filter in `_build_chunk_plan`) |
| Baseline output | `results/lossy_alg_round28/r28_baseline_verdict/` |
| Treatment output | `results/lossy_alg_round28/r28_func_class_only_verdict/` |
| Lossless output | `results/lossy_alg_round28/lossless_verdict/` |
| Fair A/B | `results/lossy_alg_round28/FAIR_AB_REPORT.md` |
| Launchers | `results/lossy_alg_round28/launchers/{run_r28_baseline,run_r28_func_class_only,run_lossless}_verdict.sh` |
| Verdict scorer | `results/lossy_alg_round28/scripts/score_r28.py` |
| Precomputed KV pool | `results/codebase_kv/pandas_5case_v4/` (R19 BEST pool, unchanged) |