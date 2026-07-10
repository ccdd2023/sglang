# Round 35 — Aggregate R26 (3B × 3) + R32 (head_recompute_30)— **NEGATIVE**

## Hypothesis

Combine two proven winners from previous rounds:
- **R26** (`r26-r27-3b-speedup-2026-07-06`): Qwen2.5-3B-Instruct × 3 agents
  with coarse chunks + MULTI_SLOT → **2.014× speedup** baseline (R26
  family).
- **R32** (`r32-cacheblend-head-recompute`): `SGLANG_CHUNK_HEAD_RECOMPUTE_FRAC=0.30`
  on top of R19 baseline → **failure-type agreement 41.7%** (Pareto).

Hypothesis: stacking both could yield the algorithmic ceiling
"speed-first × accuracy-first combined" — both bars met simultaneously.

## Result: **NEGATIVE / MIXED** — neither bar clearly met

| Metric | lossless 3B | R26 repro (control) | **R35** (3B + head_recompute) | Δ vs R26 |
|---|---|---|---|---|
| **Failure-type agreement** | 0.0% (UNKs) | 25.0% | **25.0%** | 0 (no change) |
| **Verdict PASS%** | 86.7% | 53.3% | **73.3%** | **+20 pp** |
| Verdict FAIL accuracy | 13.3% | 26.7% | 26.7% | 0 |
| avg TTFT (reusers, ms) | (lossless baseline) | 295.9 | **347.4** | **+51.5 ms (+17.4%)** |
| p50 TTFT | — | 256.4 | 376.6 | +120.2 |
| avg codeaware_reused | 0 | **1924.3** | **677.4** | **-1247 (-65%)** |
| avg radix_prefix | ~109 | 135.7 | 157.5 | +21.8 |
| Fair A/B speedup vs control | — | 1.849× | **0.602×** | **-67% (parity violated)** |

```
verdict scoring (n=15: 5 cases × 3 agents):
config                      n  pass%  fail%  FAIL_acc%  type_agree%
lossless_3B                15  86.7%  13.3%      13.3%         0.0%   (note: 0% type-agr because lossless 3B has different failure vocabulary)
r26_repro                  15  53.3%  26.7%      26.7%        25.0%
r35_3B_head_recompute      15  73.3%  26.7%      26.7%        25.0%   ← agreement unchanged from R26; PASS closer to lossless
```

## Why it failed (honest analysis)

1. **Head recompute on top of high reuse = net loss**.
   R26 reaches 1924 tokens of code-aware reuse (highest of any
   configuration). R32 head recompute (FRAC=0.30) cuts that to 677 tokens
   (-65%) because each chunk contributes ~30% to dense prefill. Net cost
   of head recompute **exceeds** accuracy benefit on the 3B × 3 regime.

2. **3B-Instruct has different failure vocabulary** from 7B-Coder.
   The "type_agree" column measures whether the model and the gold
   patch describe the same FAIL category (e.g. "type check missing").
   7B-Coder has stable category vocabulary (R32 hits 41.7%); 3B-Instruct
   has noisier/looser vocabulary and never exceeds ~25% agreement on
   the same gold patches (R26 baseline = 25.0%).

3. **Pareto per architecture differs**:
   - **7B × 5**: R32 head_recompute_30 wins (1.30× speed, 41.7% agreement)
     — small model's residual aliasing vs 7B's stronger pattern
     recognition makes type-categorization a high-leverage signal.
   - **3B × 3**: R26 no head recompute wins (2.014× speed, 25% agreement)
     — small model already runs hot on tokens; head recompute cost
     dominates.

4. **PASS% is closer to lossless but FAIL accuracy unchanged**:
   The 73.3% PASS% (vs R26 53.3%) is a +20pp improvement — but FAIL
   accuracy stays at 26.7%, identical to R26. This is **distribution
   restoration**, not **agreement recovery**. The 3B model previously
   under-called PASS (53.3%); head recompute nudges it back toward the
   86.7% lossless rate. But it doesn't make the FAILs that ARE called
   more accurate — agreement is unchanged.

## Decision: NEGATIVE

- **Do NOT ship R35 as production config.**
- **R26 alone remains the speed-first Pareto** (2.014×, no head_recompute).
- **R32 alone remains the accuracy-first Pareto** (1.30×, head_recompute_30
  on 7B × 5).
- Combining them works **only if you have separate deployment targets**
  (use R26 for latency-critical; R32 for accuracy-critical). They are
  NOT compatible on a single config.

## What this iteration taught us

The strongest gains come from **finding the right Pareto per architecture
shape**, not from naive composition. R26 × R32 didn't compose — they
each found their own Pareto point.

Specifically:
- **3B × 3** regime is dominated by **serving cost** (chunk pool capacity,
  KV footprint). Any selective recompute intervention reduces effective
  reuse below the model's accuracy-measurement noise floor.

- **7B × 5** regime is dominated by **cross-context reasoning accuracy**.
  Selective recompute (R32 head) recovers accuracy that the larger
  model's reasoning chain can actually use.

## Files

| Artifact | Path |
|---|---|
| Control (= R26) | `results/lossy_alg_round35/r26_repro_verdict/` |
| Treatment | `results/lossy_alg_round35/r35_3b_head_recompute_verdict/` |
| Lossless reference (3B) | `results/lossy_alg_round35/lossless_3b_verdict/` |
| Fair A/B | `results/lossy_alg_round35/FAIR_AB_REPORT.md` |
| Launchers (disabled) | `results/lossy_alg_round35/launchers/run_*.sh` |
| Verdict scorer | `results/lossy_alg_round35/scripts/score_r35.py` |
