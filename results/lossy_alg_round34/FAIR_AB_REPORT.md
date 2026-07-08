# Fair A/B Report

- exclude_source_agent: True (source role = `implementer`)
- parity tolerance: 15%

## Per-config (reusers only)

| config | n | p50_TTFT | avg_cached | avg_radix_prefix | avg_codeaware_reused | avg_l2 | avg_c2 | avg_F1 | F1_source |
|---|---|---|---|---|---|---|---|---|---|
| baseline(L2 whole-slot) | 20 | 756.8 | 610 | 193 | 417 | 0 | 417 | 0.406 | 20 |
| experimental(L4+C2 AST) | 20 | 763.1 | 630 | 262 | 368 | 0 | 368 | 0.427 | 20 |
| lossless(reference) | 20 | 955.1 | 109 | 109 | 0 | 0 | 0 | 1.000 | 20 |
- F1 source: real (vs lossless, from outputs.jsonl)

## Warmup-parity gate (B2)

- avg radix_prefix_tokens: baseline(L2)=193, experimental(L4+C2)=262, delta=69
- **PARITY VIOLATION** — 10 per-agent (case,agent) pairs exceed 15%:

```
  case=pandas-dev__pandas.95280573.combine_file__11s6papj agent=auditor: baseline_radix=393 exp_radix=594 delta=34% > 15%
  case=pandas-dev__pandas.95280573.combine_file__11s6papj agent=debugger: baseline_radix=371 exp_radix=89 delta=76% > 15%
  case=pandas-dev__pandas.95280573.combine_file__11s6papj agent=verifier: baseline_radix=393 exp_radix=594 delta=34% > 15%
  case=pandas-dev__pandas.95280573.combine_file__2p4yneeo agent=debugger: baseline_radix=113 exp_radix=167 delta=32% > 15%
  case=pandas-dev__pandas.95280573.combine_file__3ddy6d59 agent=debugger: baseline_radix=224 exp_radix=290 delta=23% > 15%
  case=pandas-dev__pandas.95280573.combine_file__3ddy6d59 agent=reviewer: baseline_radix=224 exp_radix=290 delta=23% > 15%
  case=pandas-dev__pandas.95280573.combine_file__3ra8xqln agent=auditor: baseline_radix=148 exp_radix=89 delta=40% > 15%
  case=pandas-dev__pandas.95280573.combine_file__3ra8xqln agent=debugger: baseline_radix=150 exp_radix=563 delta=73% > 15%
  case=pandas-dev__pandas.95280573.combine_file__3ra8xqln agent=reviewer: baseline_radix=150 exp_radix=563 delta=73% > 15%
  case=pandas-dev__pandas.95280573.combine_file__3ra8xqln agent=verifier: baseline_radix=359 exp_radix=538 delta=33% > 15%
```

The measured speedup is CONFOUNDED with radix prefix warmth and is NOT a clean
code-aware contribution. Fix the warmup/salt parity before trusting the speedup.

## Speed bar (L4+C2 vs L2, radix-isolated)

- speedup = L2.p50 / L4+C2.p50 = 756.8 / 763.1 = **0.992x**
- L2 avg_radix_prefix = 193; L4+C2 avg_radix_prefix = 262 (delta 69, should be ~0)
- L2 avg_codeaware_reused = 417 (l2_wholeslot=0)
- L4+C2 avg_codeaware_reused = 368 (l2=0, c2=368)
- speed bar (>=1.0x): **NOT MET (AST slower than whole-slot)**

Speedup 0.992x is NOT trustworthy (parity violated).

## Accuracy bar (L4+C2 vs L2, both vs lossless)

- L4+C2 avg_F1 vs lossless = 0.427
- L2   avg_F1 vs lossless = 0.406
- accuracy bar (L4+C2 F1 >= L2 F1): **MET (AST not less accurate than whole-slot)**
- (legacy) l3_general config not provided; the bar above (vs L2) is the primary one.
