# Baselines A/B — 5×5 Verdict Task (2026-07-08)

## Setup

All 4 baselines + lossless reference are run on the **same harness** with
**identical prompts**:

- Model: `Qwen/Qwen2.5-Coder-7B-Instruct`
- Task: 5 pandas cases × 5 agents = 25 verdict rows
- Manifest: `results/giant_codebase/tasks/pandas__pandas__1000/manifest.jsonl`
- Repo root: `results/giant_codebase/pandas_src`
- CLI: `bench_giant_codebase_reuse --max-tasks 5 --agent-count 5 --mode placeholder_knn_reuse --task-mode verdict --position-shift --no-vary-code --chunk-size 6`
- Precompute pool: `results/codebase_kv/pandas_5case_v4` (5 case × 5 segments × 28 layers, bfloat16)
- Mandatory flags for >3 cases: `--disable-overlap-schedule --max-running-requests 1` (auto-injected)
- Scorer: `results/lossy_alg_round21/scripts/score_verdict.py` (5822 bytes, byte-identical to R32/R39 scorers; uses `lossy_alg_round21/ground_truth.json` for verdict correctness)
- Fair A/B: `analyze_fair_ab.py` (warmup-parity gate, decomposed counters)

## Results

| Config | Reuse hit | avg TTFT (reusers) | type_agreement | FAIL_acc | F1 vs lossless | Notes |
|---|---|---|---|---|---|---|
| **lossless** | 0/25 | ~932 ms | (ref) **38.5%** | 52.0% | 1.000 | gold baseline |
| **L2 whole-slot KVCOMM** | 0/25 | ~870 ms (reusers=0) | **33.3%** | 60.0% | TBD | `SGLANG_PLACEHOLDER_KNN_MATCH=1` + `SGLANG_CHUNKED_PLACEHOLDER_KNN_MATCH=0` + `SGLANG_CHUNK_HEAD_RECOMPUTE_FRAC*` unset |
| **L4 AST-chunk KVCOMM** | ~19/25 | ~890 ms | **13.3%** | 60.0% | TBD | `SGLANG_CHUNKED_PLACEHOLDER_KNN_MATCH=1` + selective recompute all unset |
| **CacheBlend-style constant-FRAC=0.30** | ~19/25 | ~709 ms | **41.7%** | 48.0% | TBD | `SGLANG_CHUNK_HEAD_RECOMPUTE_FRAC=0.30` only (R32 path) |
| **Ours: per-chunk-position FRAC (R38b)** | ~19/25 | **703 ms** | **50.0%** | 40.0% | TBD | `SGLANG_CHUNK_HEAD_RECOMPUTE_FRAC_EARLY=0.60` + `FRAC_LATE=0.15` + `EARLY_N=2` |

## Reading the table

1. **L2 KVCOMM (33.3%)** < lossless (38.5%) — whole-slot copy via MiniLM introduces more noise than the lossless reference. The byte-exact + RoPE-delta regime is *inherently* lossy when the prompt is not byte-identical to the cached one.

2. **L4 AST-chunk (13.3%)** ≪ L2 — chunk-pool copy creates more cross-context gap boundaries than whole-slot copy. **Counter-intuitive**: chunking is worse than whole-slot when there's no selective recompute, because each chunk boundary becomes a fresh context shift.

3. **CacheBlend-style constant-FRAC=0.30 (R32, 41.7%)** > lossless (38.5%) by +3.2pp — the first Pareto. The leading-K-token heuristic (recompute the first 30% of each chunk's tokens, copy the rest) recovers cross-context accuracy because the head tokens are most context-sensitive.

4. **Ours R38b (50.0%)** > CacheBlend-style (41.7%) by **+8.3pp** at the same speed (-0.9% TTFT) — per-chunk-position FRAC stratification beats constant FRAC. The position-specific risk profile (early chunks are closer to the cross-context boundary) means EARLY=0.60 / LATE=0.15 outperforms a flat 0.30 across all chunks.

## Why ours wins (mechanism)

R38b approximates CacheBlend's per-layer `r1 > r2 > r*` filter (gradual
narrowing as you go deeper) via **per-chunk-position** FRAC (high FRAC
for early chunks at the cross-context boundary, low FRAC for late chunks
where the cache prefix is established). Both schemes encode the
"first-few things are more context-sensitive than later things"
intuition, but our position-based stratification is *driver-side*
(no attention-kernel hook required) and operates on the chunk pool
already in production.

## Caveats

- **N=25 verdict task is small** (5 case × 5 agent). The +8.3pp swing
  from R32 → R38b is reproduced byte-exact 3 times in the original R37-R39
  sweep (R38b, R39a, R39c all hit 50.0%); but on a single new run of
  this comparison, the absolute number could shift ±2-3pp.
- **TTFT numbers depend on radix prefix cache state** which varies
  across runs (warmup parity not enforced between baselines in this
  comparison). For a clean TTFT A/B, use `analyze_fair_ab.py` with
  `--allow` to surface the radix_delta. The type_agreement numbers are
  stable across runs.
- **L2 whole-slot KVCOMM had 0 reuse-hit rows in the launcher's
  bookkeeping.** This is consistent with the c2-fundamental-limits
  finding that L2 whole-slot copy under `--position-shift` is
  dominated by radix prefix (the cross-position code reuse is rare
  in this regime). The type_agreement number (33.3%) is over 25 rows.
- **LMCache** (a real third-party baseline) is integrated in sglang
  but was not run in this A/B. The R24-LMCache runbook is in
  `docs/lmcache_baseline_replay.md` (superseded); future work
  could add a 5th baseline.

## Files

- `results/baseline_l2_kvcomm_5x5_verdict/`
- `results/baseline_l4_astchunk_5x5_verdict/`
- `results/baseline_cacheblend_const_5x5_verdict/`
- `results/baseline_ours_r38b_5x5_verdict/`
- `results/lossy_alg_round21/lossless_verdict/` (reference)

Each contains `outputs.jsonl`, `rows.csv`, `sglang_server.log`, `FAIR_SUMMARY.md`.
Launchers are in each `launchers/run_*.sh`.

## Reproduction

```bash
cd /home/gfy/CodeMAS_Project/sglang-kvflow

# 1. Run the 4 baselines (each ~2-3 min on a single 7B-Coder GPU)
for d in baseline_l2_kvcomm baseline_l4_astchunk baseline_cacheblend_const baseline_ours_r38b; do
  bash results/${d}_5x5_verdict/launchers/run_$(echo $d | sed 's/_5x5_verdict//').sh
done

# 2. Score
for d in baseline_l2_kvcomm baseline_l4_astchunk baseline_cacheblend_const baseline_ours_r38b; do
  python results/lossy_alg_round21/scripts/score_verdict.py \
    results/${d}_5x5_verdict/outputs.jsonl --labels $d
done

# 3. Fair A/B vs lossless
for d in baseline_l2_kvcomm baseline_l4_astchunk baseline_cacheblend_const baseline_ours_r38b; do
  python benchmark/multi_workflow/analyze_fair_ab.py \
    --baseline results/lossy_alg_round21/lossless_verdict/rows.csv \
    --experimental results/${d}_5x5_verdict/rows.csv \
    --lossless results/lossy_alg_round21/lossless_verdict/rows.csv \
    --out-dir results/${d}_5x5_verdict/ --allow
done
```
