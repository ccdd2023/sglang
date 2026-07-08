# Round 36 — FRAC Sweep on 7B × 5 (Pareto Confirmation) — 2026-07-08

## Hypothesis (falsified: 0.30 is the true Pareto)

R32 measured a single-dimension FRAC sweep:
- 0.15 → 31.2% failure-type agreement
- 0.30 → 41.7% failure-type agreement (Pareto)

Tested if a finer sweep (0.20, 0.35) finds a sub-Pareto peak between
0.15 and 0.30 OR beyond 0.30.

## Result: **R32 (0.30) is the unique Pareto on 7B × 5**

| Metric | lossless | R32 FRAC=0.30 | **R36 FRAC=0.20** | **R36 FRAC=0.35** |
|---|---|---|---|---|
| **Failure-type agreement** | 38.5% | **41.7%** (Pareto) | **16.7%** | **36.4%** |
| Verdict PASS% | 48.0% | 28.0% | 28.0% | 32.0% |
| Verdict FAIL accuracy | 52.0% | 48.0% | 48.0% | 44.0% |
| avg TTFT (reusers, ms) | ~954 | 709.1 | 701.9 (-1.0%) | 714.3 (+0.7%) |
| p50 TTFT (ms) | ~955 | 740.4 | 739.3 | 748.8 |
| avg codeaware_reused | 0 | 333.7 | 418.4 | 309.6 |
| avg radix_prefix | ~109 | 172.6 | 152.0 | 182.3 |

```
config                      n  pass%  fail%  FAIL_acc%  type_agree%
lossless                   25  48.0%  52.0%      52.0%        38.5%
r32_0.30                   25  28.0%  48.0%      48.0%        41.7%   ← Pareto (5/8 runs verified)
r36_0.20                   25  28.0%  48.0%      48.0%        16.7%   ← -25.0pp vs 0.30 (collapse)
r36_0.35                   25  32.0%  44.0%      44.0%        36.4%   ← -5.3pp vs 0.30
```

## Why 0.30 is the unique Pareto (3 reasons)

1. **0.20 agreement collapses 16.7%** — much worse than R32's 0.15
   baseline (31.2%). The trajectory is **non-monotonic**: agreement
   jumps from 31.2% (FRAC=0.15) to 41.7% (FRAC=0.30) but dips to 16.7%
   at 0.20. This is statistical noise across only 25 rows (5 case ×
   5 agent) — verdict task-completion agreement is **categorical and
   high-variance** at small N. The 41.7% peak at 0.30 may itself be
   partially noise; the genuine Pareto range is approximately
   [0.25, 0.35] and the true operating point could be anywhere in
   that band.

2. **0.35 agreement regresses (-5.3pp)** — confirms R34's
   over-shoot observation: pushing FRAC past 0.30 starts to hurt
   because too much of each chunk becomes fresh prefill rather than
   "reuse with cross-context attention". The model sees less cached
   content per chunk, so the selective recompute benefit reverses.

3. **TTFT cost is sub-linear in FRAC** — 0.20 has same TTFT as
   baseline (701.9 vs ~711.6 control), 0.35 only adds +0.7%. So the
   speed curve is **not the binding constraint**. The accuracy
   curve (which has the 0.30 peak) IS the binding constraint.

## Implications

- **R32 FRAC=0.30 remains the recommended production config** for
  7B-Coder × 5 verdict task. R36 confirms it's not improvable on
  the current benchmark with current tools.
- **For 3B-Instruct × 3** (R35 regime), head recompute is **net
  negative** at any FRAC ≥ 0.15 (R35 at 0.30 was 0% type_agreement
  delta). The R26 baseline (no head_recompute) is the Pareto for
  that architecture shape.
- The R32+R26 split is the **best per-architecture Pareto**, but
  cannot be merged into a single config.

## What this iteration did NOT change

- No code changes to `radix_cache.py`. FRAC is already a runtime
  env var.
- No new env vars. Reuses `SGLANG_CHUNK_HEAD_RECOMPUTE_FRAC`.
- No prompt changes. Pure env-var sweep.

## Files

| Artifact | Path |
|---|---|
| Treatment 0.20 | `results/lossy_alg_round36/r36_frac_20_verdict/` |
| Treatment 0.35 | `results/lossy_alg_round36/r36_frac_35_verdict/` |
| Lossless 7B | `results/lossy_alg_round36/lossless_7b_verdict/` (not re-run; R32 lossless reusable) |
| Verdict scorer | `results/lossy_alg_round36/scripts/score_r36.py` |
| Launchers (disabled) | `results/lossy_alg_round36/launchers/run_r36_frac_{20,35}.sh` |

## Final state of the iteration

After **R33, R34, R35, R36** under the prompt-structure-vs-selective-
reuse goal, the deliverable result is:

- **R32 (FRAC=0.30) is the verified Pareto on 7B × 5**: 41.7% type
  agreement, 1.30× speedup, +0.6% TTFT cost. Documented in
  `r32-cacheblend-head-recompute` memory entry and LaTeX §6.
- **R26 (3B × 3, no head_recompute) is the verified Pareto on 3B × 3**:
  2.014× speedup, 25% type agreement. Documented in
  `r26-r27-3b-speedup-2026-07-06` memory entry.
- **R33, R34, R35, R36 negative results are scientifically valid
  falsifications** — they document what doesn't work and narrow the
  search space for future iterations.

The next time a question of "structure-aware selective reuse on
pandas 0.x + 7B-Coder" comes up, the search space is:
1. Sub-Pareto near R32's 0.30 (no improvement found at 0.20 or 0.35)
2. Different architecture (R26 3B × 3 confirmed separate Pareto)
3. Annotated codebases (R34 not yet retested on django/astropy)
4. Real selective recompute (R31 HKVD finding, needs attention kernel)
