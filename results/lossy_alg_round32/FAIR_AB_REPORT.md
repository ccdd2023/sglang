# Fair A/B Report

- exclude_source_agent: True (source role = `implementer`)
- parity tolerance: 15%

## Per-config (reusers only)

| config | n | p50_TTFT | avg_cached | avg_radix_prefix | avg_codeaware_reused | avg_l2 | avg_c2 | avg_F1 | F1_source |
|---|---|---|---|---|---|---|---|---|---|
| baseline(L2 whole-slot) | 20 | 740.7 | 738 | 119 | 619 | 0 | 619 | 0.498 | 20 |
| experimental(L4+C2 AST) | 20 | 738.5 | 761 | 161 | 600 | 0 | 600 | 0.408 | 20 |
| lossless(reference) | 20 | 959.9 | 109 | 109 | 0 | 0 | 0 | 1.000 | 20 |
- F1 source: real (vs lossless, from outputs.jsonl)

## Warmup-parity gate (B2)

- avg radix_prefix_tokens: baseline(L2)=119, experimental(L4+C2)=161, delta=42
- **PARITY VIOLATION** — 12 per-agent (case,agent) pairs exceed 15%:

```
  case=pandas-dev__pandas.95280573.combine_file__11s6papj agent=auditor: baseline_radix=92 exp_radix=243 delta=62% > 15%
  case=pandas-dev__pandas.95280573.combine_file__11s6papj agent=debugger: baseline_radix=89 exp_radix=314 delta=72% > 15%
  case=pandas-dev__pandas.95280573.combine_file__11s6papj agent=verifier: baseline_radix=92 exp_radix=243 delta=62% > 15%
  case=pandas-dev__pandas.95280573.combine_file__1eilbetv agent=auditor: baseline_radix=115 exp_radix=170 delta=32% > 15%
  case=pandas-dev__pandas.95280573.combine_file__1eilbetv agent=debugger: baseline_radix=122 exp_radix=158 delta=23% > 15%
  case=pandas-dev__pandas.95280573.combine_file__1eilbetv agent=reviewer: baseline_radix=122 exp_radix=158 delta=23% > 15%
  case=pandas-dev__pandas.95280573.combine_file__1eilbetv agent=verifier: baseline_radix=122 exp_radix=158 delta=23% > 15%
  case=pandas-dev__pandas.95280573.combine_file__3ddy6d59 agent=debugger: baseline_radix=113 exp_radix=180 delta=37% > 15%
  case=pandas-dev__pandas.95280573.combine_file__3ddy6d59 agent=reviewer: baseline_radix=113 exp_radix=180 delta=37% > 15%
  case=pandas-dev__pandas.95280573.combine_file__3ra8xqln agent=auditor: baseline_radix=89 exp_radix=118 delta=25% > 15%
  case=pandas-dev__pandas.95280573.combine_file__3ra8xqln agent=debugger: baseline_radix=337 exp_radix=114 delta=66% > 15%
  case=pandas-dev__pandas.95280573.combine_file__3ra8xqln agent=verifier: baseline_radix=92 exp_radix=226 delta=59% > 15%
```

The measured speedup is CONFOUNDED with radix prefix warmth and is NOT a clean
code-aware contribution. Fix the warmup/salt parity before trusting the speedup.

## Speed bar (L4+C2 vs L2, radix-isolated)

- speedup = L2.p50 / L4+C2.p50 = 740.7 / 738.5 = **1.003x**
- L2 avg_radix_prefix = 119; L4+C2 avg_radix_prefix = 161 (delta 42, should be ~0)
- L2 avg_codeaware_reused = 619 (l2_wholeslot=0)
- L4+C2 avg_codeaware_reused = 600 (l2=0, c2=600)
- speed bar (>=1.0x): **MET (AST not slower than whole-slot)**

Speedup 1.003x is NOT trustworthy (parity violated).

## Accuracy bar (L4+C2 vs L2, both vs lossless)

- L4+C2 avg_F1 vs lossless = 0.408
- L2   avg_F1 vs lossless = 0.498
- accuracy bar (L4+C2 F1 >= L2 F1): **NOT MET (AST less accurate than whole-slot)**
- (legacy) l3_general config not provided; the bar above (vs L2) is the primary one.
