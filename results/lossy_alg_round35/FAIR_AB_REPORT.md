# Fair A/B Report

- exclude_source_agent: True (source role = `implementer`)
- parity tolerance: 15%

## Per-config (reusers only)

| config | n | p50_TTFT | avg_cached | avg_radix_prefix | avg_codeaware_reused | avg_l2 | avg_c2 | avg_F1 | F1_source |
|---|---|---|---|---|---|---|---|---|---|
| baseline(L2 whole-slot) | 10 | 224.4 | 3036 | 149 | 2886 | 0 | 2886 | 0.435 | 10 |
| experimental(L4+C2 AST) | 10 | 373.1 | 1198 | 182 | 1016 | 0 | 1016 | 0.530 | 10 |
| lossless(reference) | 10 | 408.6 | 129 | 129 | 0 | 0 | 0 | 1.000 | 10 |
- F1 source: real (vs lossless, from outputs.jsonl)

## Warmup-parity gate (B2)

- avg radix_prefix_tokens: baseline(L2)=149, experimental(L4+C2)=182, delta=33
- **PARITY VIOLATION** — 7 per-agent (case,agent) pairs exceed 15%:

```
  case=pandas-dev__pandas.95280573.combine_file__11s6papj agent=debugger: baseline_radix=152 exp_radix=109 delta=28% > 15%
  case=pandas-dev__pandas.95280573.combine_file__11s6papj agent=reviewer: baseline_radix=153 exp_radix=109 delta=29% > 15%
  case=pandas-dev__pandas.95280573.combine_file__1eilbetv agent=debugger: baseline_radix=158 exp_radix=322 delta=51% > 15%
  case=pandas-dev__pandas.95280573.combine_file__1eilbetv agent=reviewer: baseline_radix=173 exp_radix=322 delta=46% > 15%
  case=pandas-dev__pandas.95280573.combine_file__2p4yneeo agent=debugger: baseline_radix=160 exp_radix=133 delta=17% > 15%
  case=pandas-dev__pandas.95280573.combine_file__3ddy6d59 agent=debugger: baseline_radix=133 exp_radix=293 delta=55% > 15%
  case=pandas-dev__pandas.95280573.combine_file__3ra8xqln agent=debugger: baseline_radix=165 exp_radix=134 delta=19% > 15%
```

The measured speedup is CONFOUNDED with radix prefix warmth and is NOT a clean
code-aware contribution. Fix the warmup/salt parity before trusting the speedup.

## Speed bar (L4+C2 vs L2, radix-isolated)

- speedup = L2.p50 / L4+C2.p50 = 224.4 / 373.1 = **0.602x**
- L2 avg_radix_prefix = 149; L4+C2 avg_radix_prefix = 182 (delta 33, should be ~0)
- L2 avg_codeaware_reused = 2886 (l2_wholeslot=0)
- L4+C2 avg_codeaware_reused = 1016 (l2=0, c2=1016)
- speed bar (>=1.0x): **NOT MET (AST slower than whole-slot)**

Speedup 0.602x is NOT trustworthy (parity violated).

## Accuracy bar (L4+C2 vs L2, both vs lossless)

- L4+C2 avg_F1 vs lossless = 0.530
- L2   avg_F1 vs lossless = 0.435
- accuracy bar (L4+C2 F1 >= L2 F1): **MET (AST not less accurate than whole-slot)**
- (legacy) l3_general config not provided; the bar above (vs L2) is the primary one.
