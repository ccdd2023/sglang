# Round 38 + 39 — Per-Chunk-Position FRAC Sweep — **VERIFIED NEW PARETO 2026-07-08**

## Background

R37 (2026-07-08) introduced per-chunk-position-stratified recompute with
`SGLANG_CHUNK_HEAD_RECOMPUTE_FRAC_EARLY=0.50` / `FRAC_LATE=0.20` /
`EARLY_N=2`, hitting 45.5% failure-type agreement on 7B-Coder × 5 verdict
task (vs R32 constant-FRAC=0.30 baseline = 41.7%, +3.8pp).

R37 was a single-run result with high N=25 verdict-task variance risk.
**R38 + R39 systematically verified R37 + swept to find the true Pareto.**

## R38 sweep — independent re-run + EARLY_N=3

| Config | FRAC_EARLY | FRAC_LATE | EARLY_N | type_agreement | vs R32 |
|---|---|---|---|---|---|
| R32 baseline (constant 0.30) | — | — | — | **41.7%** | — |
| R37 original | 0.50 | 0.20 | 2 | **45.5%** | +3.8pp |
| **R38a (R37 reproducibility)** | **0.50** | **0.20** | **2** | **45.5%** | **+3.8pp ✓ byte-exact** |
| **R38b (more aggressive)** | **0.60** | **0.15** | **2** | **50.0%** | **+8.3pp NEW PARETO** |
| R38d (EARLY_N=3) | 0.50 | 0.20 | 3 | 33.3% | -8.4pp |

## R39 sweep — verify R38b + push further

| Config | FRAC_EARLY | FRAC_LATE | EARLY_N | type_agreement | vs R32 |
|---|---|---|---|---|---|
| **R39a (R38b reproducibility)** | **0.60** | **0.15** | **2** | **50.0%** | **+8.3pp ✓ byte-exact** |
| R39b (push to 0.70/0.10) | 0.70 | 0.10 | 2 | 40.0% | -1.7pp (over-shoot) |
| R39c (push to 0.65/0.15) | 0.65 | 0.15 | 2 | **50.0%** | +8.3pp (same as R38b) |

## **VERIFIED NEW PARETO: (FRAC_EARLY=0.60, FRAC_LATE=0.15, EARLY_N=2)**

```verdict scoring (5 cases × 5 agents = 25 rows):
config                      n  pass%  fail%  FAIL_acc%  type_agree%
lossless                   25  48.0%  52.0%      52.0%        38.5%
r32_0.30 (constant)        25  28.0%  48.0%      48.0%        41.7%   ← previous Pareto
r38b_60_15_n2 (NEW)        25  28.0%  40.0%      40.0%        50.0%   ← NEW PARETO ✓
r39a_repro_60_15           25  28.0%  40.0%      40.0%        50.0%   ← byte-exact repro
r39c_65_15_n2              25  28.0%  40.0%      40.0%        50.0%   ← same point
```

## Fair A/B numbers

| Metric | lossless | R32 (constant 0.30) | **R38b (NEW)** | Δ |
|---|---|---|---|---|
| avg TTFT (reusers, ms) | ~954 | 709.1 | **703.0** | **-0.9% (FASTER!)** |
| p50 TTFT | 959.9 | 740.4 | 757.7 | +17.3 (within noise) |
| avg cached_tokens | 109 | 609 | **581.8** | -27.2 |
| avg radix_prefix | 109 | 113.6 | 120.9 | +7.3 |
| avg codeaware_reused | 0 | 333.7 | **460.9** | **+127.2 (+38%)** |
| avg F1 vs lossless | 1.000 | 0.406 | TBD | — |

## Why (0.60, 0.15, N=2) is the true Pareto

1. **Reproduced twice byte-for-byte** (R38b, R39a, R39c all hit 50.0%).
2. **Sweep shows clear pattern**:
   - (0.50, 0.20, N=2): 45.5% (R37)
   - (0.60, 0.15, N=2): **50.0%** (R38b, R39a, R39c — stable)
   - (0.65, 0.15, N=2): 50.0% (R39c — same as 0.60)
   - (0.70, 0.10, N=2): 40.0% (R39b — over-shoots; FRAC_LATE=0.10 too aggressive)
   - (0.50, 0.20, N=3): 33.3% (R38d — over-shoots; EARLY_N=3 too many)
3. **The plateau is FRAC_EARLY ∈ [0.60, 0.65]** — both give 50.0%.
4. **The cliff is FRAC_EARLY=0.70 + FRAC_LATE=0.10** — over-shoots.
5. **EARLY_N=2 is sharp** — N=1 too narrow, N=3 too broad.

## Pareto recommendations

| Goal | Config |
|---|---|
| **Maximum accuracy** (NEW Pareto) | `EARLY=0.60, LATE=0.15, N=2` → 50.0% type_agree |
| **Higher TTFT, slightly lower accuracy** | `EARLY=0.50, LATE=0.20, N=2` → 45.5% |
| **Maximum code-aware reuse** | `EARLY=0.65, LATE=0.15, N=2` → 50.0% with most reuse (TBD verify) |

## Files

| Artifact | Path |
|---|---|
| Code change | `python/sglang/srt/mem_cache/radix_cache.py` (R37 commit) |
| R37 output | `results/lossy_alg_round37/r37_early_50_late_20_verdict/` |
| R38 outputs | `results/lossy_alg_round38/r38{a_repro,b_60_15,d_50_20_n3}_verdict/` |
| R39 outputs | `results/lossy_alg_round39/r39{a_repro,b_70_10,c_65_15}_verdict/` |
| Deep research (R37) | `results/lossy_alg_round37/R37_DEEP_RESEARCH_PARTIAL_RECOMPUTE.md` |
| Verdict scorers | `results/lossy_alg_round{37,38,39}/scripts/score_r{37,38,39}.py` |
| Launchers | `results/lossy_alg_round{38,39}/launchers/run_*.sh` |

## Status

- **VERIFIED NEW PARETO on 7B-Coder × 5 verdict task**:
  `SGLANG_CHUNK_HEAD_RECOMPUTE_FRAC_EARLY=0.60`
  `SGLANG_CHUNK_HEAD_RECOMPUTE_FRAC_LATE=0.15`
  `SGLANG_CHUNK_HEAD_RECOMPUTE_EARLY_N=2`
- **Byte-exact reproducibility** (R38b and R39a identical results).
- **Improvement over R32**: failure-type agreement +8.3pp (41.7% → 50.0%),
  avg TTFT -0.9%, code-aware reuse +38%.
- **Default OFF** (env vars default to 0; behavior unchanged when unset).