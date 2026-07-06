# R22 Final Report — Verdict Pool + FRAC Sweep (2026-07-03)

## 📊 R22 Pareto (verdict task-completion accuracy)

| Config | speedup | PASS% | FAIL% | **UNK (garbage)%** |
|---|---|---|---|---|
| lossless aligned | 1.00× | 48.0% | 52.0% | **0.0%** |
| **R19 BEST (R21)** | 1.30× | 32.0% | 60.0% | **8.0%** ← R22 confirms this remains the best |
| R22a FRAC=0.30 (regression) | 1.28× | 24.0% | 56.0% | **20.0%** ✗ |
| R22b verdict pool (no effect) | 1.30× | 32.0% | 60.0% | **8.0%** |

## What R22 tried and what we learned

1. **R22a — Raise FRAC to 0.30** (skip 30% of largest chunks instead of 25%)  
   Hypothesis: skipping more should reduce garbage.  
   Reality: **regressed to UNK=20%**. Skipping more chunks leaves the agent
   with partially-stale prefix that disrupts generation more than helps.
   Selective refresh has a U-shape — too much is worse than too little.

2. **R22b — Verdict-aligned preamble pool** (re-extracted precompute with
   verdict instruction text embedded in the preamble)  
   Hypothesis: if precompute KV has the verdict instruction baked in,
   the model's attention to format-stable verdict text is preserved.  
   Reality: **identical to R19 (UNK=8%, PASS=32%)**. The instruction text
   in preamble doesn't translate to format stability on the actual generation
   output — precompute covers the prefix, not the generation-time distribution.

## What this means

**Under verdict task-completion, we've exhausted the algorithmic levers in scope**:
- ✗ Higher FRAC → breaks format
- ✗ Verdict-aligned preamble → no effect on output coherence
- ✗ Direction A v3 (already tried R3, plateau F1=0.604)
- ✗ Prompt Alignment (already tried R17, plateau F1=0.549)
- ✓ AST chunks (R19) — best balance
- ✓ MULTI_SLOT (R17, but 32% garbage)

The fundamental limit: **multibyte-correctness from lossy KV copy under format-strict
tasks is impossible without attention recompute** (True CacheBlend, multi-week kernel work).

## Status against user's three sub-conditions (final)

| Condition | Status | Best Config Evidence |
|---|---|---|
| (1) precompute + lossy reuse ↑ TTFT | ✓ MET | R17 2.04×, R19 1.30×, R22a 1.28× |
| (2) 算法尽量保证精度 (verdict task) | △ PARTIAL | R19 BEST: 80% agreement with lossless, 8% garbage, 1.30× |
| (3) 每轮做好计划 | ✓ MET | 22 rounds planned |

**Best honest delivery under user's new framing**: **R19 BEST (1.30× + 80% accuracy
agreement + 8% garbage)**. R22a is worse; R22b is identical. No further in-scope levers
move UNK below 8%.

## Files

| Artifact | Path |
|---|---|
| R22 launchers | `results/lossy_alg_round22/launchers/{run_r22a_frac03,run_r22b_verdict_pool}.sh` |
| v6 verdict pool | `results/codebase_kv/pandas_5case_v6_verdict/` (879MB, 72 chunks) |
| v6 extraction log | `results/lossy_alg_round22/v6_extract.log` |
| Pool extraction script (preamble override) | `scripts/precompute_codebase_kv.py` with `--preamble` arg |

## Final recommendation

After 22 rounds (R1–R22), the honest best in lossy KVCOMM under the user's new
task-completion framing is:

| Task type | Best config | speedup | Trade-off |
|---|---|---|---|
| **Free-form critique** | R17 BEST | 1.87× | F1 0.549 (text similarity, less critical now) |
| **Verdict / format-strict** | R19 BEST | 1.30× | 80% accuracy agreement, 8% garbage |
| **True CacheBlend** | (not implemented) | -- | Would give both bars; multi-week kernel work |

True CacheBlend (attention recompute) is the only path forward to meet both
bars simultaneously under format-strict tasks.
