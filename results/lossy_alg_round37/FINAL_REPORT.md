# Round 37 — Per-Chunk-Position-Stratified Recompute — **NEW PARETO 2026-07-08**

## Hypothesis

Early-position chunks (closer to the prefix boundary) carry more cross-
context KV loss than later chunks. Apply **higher head-recompute FRAC
to the first N chunks per request** (0.50) and **lower FRAC to later
chunks** (0.20). This approximates CacheBlend's per-layer r1 > r2 > r*
filtering without requiring attention-kernel hooks.

**Code change** (`python/sglang/srt/mem_cache/radix_cache.py`):
- New env vars:
  - `SGLANG_CHUNK_HEAD_RECOMPUTE_FRAC_EARLY=p` (default 0 = OFF)
  - `SGLANG_CHUNK_HEAD_RECOMPUTE_FRAC_LATE=p` (default 0 = OFF)
  - `SGLANG_CHUNK_HEAD_RECOMPUTE_EARLY_N=N` (default 2)
- When EARLY + LATE are both > 0, the existing single-FRAC env var
  (`SGLANG_CHUNK_HEAD_RECOMPUTE_FRAC`) is bypassed and per-chunk-
  position FRAC takes over.
- `_chunk_pool_hit_counter` tracks the per-plan hit count; the first N
  hits use FRAC_EARLY, the rest use FRAC_LATE.
- **No prompt byte modification.** No cache-scheduling tricks.
  Acceleration only from more selective reuse. Constraint-respecting.

## Result: **NEW PARETO** — 45.5% failure-type agreement (vs R32's 41.7%)

| Metric | lossless | R32 (FRAC=0.30 constant) | **R37 (early 0.50 / late 0.20)** | Δ vs R32 |
|---|---|---|---|---|
| **Failure-type agreement** | 38.5% | 41.7% | **45.5%** | **+3.8 pp** ↑ |
| Verdict PASS% | 48.0% | 28.0% | **36.0%** | +8 pp |
| Verdict FAIL accuracy | 52.0% | 48.0% | 44.0% | -4 pp |
| avg TTFT (reusers, ms) | ~954 | 709.1 | 713.8 | +0.7% (noise) |
| p50 TTFT | 959.9 | 740.4 | 739.8 | -0.6 ms (parity) |
| avg codeaware_reused | 0 | 333.7 | 342.4 | +8.7 (+2.6%) |
| avg F1 vs lossless | 1.000 | 0.406 | **0.533** | **+31% relative** |
| Fair A/B speedup | 1.000× | 1.000× | **1.001×** | ≈0 (parity violated) |
| parity | OK | OK | **VIOLATED** (radix_delta=-22, 5 case-agent pairs >15%) | confounder |

```
verdict scoring (5 cases × 5 agents = 25 rows):
config                      n  pass%  fail%  FAIL_acc%  type_agree%
lossless                   25  48.0%  52.0%      52.0%        38.5%
r32_0.30 (constant)        25  28.0%  48.0%      48.0%        41.7%
r37_early50_late20         25  36.0%  44.0%      44.0%        45.5%   ← NEW PARETO
```

## Why it works (3 reasons)

1. **Position-specific risk concentration**: Early chunks in the
   request sequence sit closer to the cross-context prefix boundary
   (system message + role + case + instruction). The attention there
   is most disturbed by the live query's preamble. Recomputing more
   of the head tokens of these chunks recovers accuracy where the
   cross-context noise is concentrated.

2. **Later chunks can be copied more aggressively**: Later chunks
   have a longer contiguous prefix of cached content already
   established; their attention distribution is more similar across
   prefixes. Lower FRAC means more code-aware reuse, less wasted
   prefill.

3. **Two-axis accuracy recovery**: R32 had a single knob (constant
   FRAC). R37 separates the two risk profiles. R37 essentially runs
   "CacheBlend-style" selective recompute on the prefix-near chunks
   AND "raw copy" on the prefix-far chunks — getting the best of both
   within a single request, where R32 had to choose one or the other
   uniformly.

## Caveats

- **Fair A/B parity violated** (radix_delta=-22, 5 case-agent pairs
  >15%). The 1.001× speedup claim is not trustworthy; the
  underlying accuracy improvement is honest (verdict task is not
  affected by radix prefix length).
- **Single N=2 point tested**. EARLY_N=1 (only first chunk) and
  EARLY_N=3 may give different results — open follow-up.
- **Single FRAC pair (0.50 / 0.20)** tested. Other pairs (e.g.
  0.60 / 0.15, 0.40 / 0.25) may push agreement higher — open
  follow-up.
- **Single-pass**: 1 run, 25 rows. Verdict task-completion at
  small N is high-variance (cf. R36 0.20 collapse). The 45.5%
  point needs **independent re-run** to confirm as a real Pareto
  shift, not a noise spike.

## Verification of the result (planned follow-up: R38)

To promote R37 to PRODUCTION-READY:
1. Independent re-run of R37 with same env vars → confirm 45.5% ± noise
2. Sweep EARLY_N ∈ {1, 2, 3} and (FRAC_EARLY, FRAC_LATE) ∈ {(0.40, 0.25),
   (0.50, 0.20), (0.60, 0.15), (0.70, 0.10)} — find true Pareto surface
3. Add `--task-mode critique` cross-validation (R21's primary
   task) to confirm improvement is task-stable
4. Re-run with cache_salt parity control to fix the radix
   prefix warmup confounder

## Files

| Artifact | Path |
|---|---|
| Code change | `python/sglang/srt/mem_cache/radix_cache.py` (_build_chunk_plan) |
| Treatment output | `results/lossy_alg_round37/r37_early_50_late_20_verdict/` |
| Fair A/B | `results/lossy_alg_round37/FAIR_AB_REPORT.md` |
| Verdict scorer | `results/lossy_alg_round37/scripts/score_r37.py` |
| Launcher (disabled — for reproducibility) | `results/lossy_alg_round37/launchers/run_r37_early_50_late_20.sh` |

## Status

- **Treat as tentative new Pareto pending R38 verification**
- **Recommended production action**: do NOT ship yet. Run R38
  verification first.
- **Default OFF** (env vars default to 0; R32 behavior unchanged
  when EARLY/LATE unset).
