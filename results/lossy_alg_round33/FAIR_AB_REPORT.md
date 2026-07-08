# Fair A/B Report

- exclude_source_agent: True (source role = `implementer`)
- parity tolerance: 15%

## Per-config (reusers only)

| config | n | p50_TTFT | avg_cached | avg_radix_prefix | avg_codeaware_reused | avg_l2 | avg_c2 | avg_F1 | F1_source |
|---|---|---|---|---|---|---|---|---|---|
| baseline(L2 whole-slot) | 20 | 744.2 | 610 | 193 | 417 | 0 | 417 | 0.406 | 20 |
| experimental(L4+C2 AST) | 20 | 778.2 | 850 | 310 | 540 | 0 | 540 | 0.303 | 20 |
| lossless(reference) | 20 | 954.7 | 109 | 109 | 0 | 0 | 0 | 1.000 | 20 |
- F1 source: real (vs lossless, from outputs.jsonl)

## Warmup-parity gate (B2)

- avg radix_prefix_tokens: baseline(L2)=193, experimental(L4+C2)=310, delta=118
- **PARITY VIOLATION** — 9 per-agent (case,agent) pairs exceed 15%:

```
  case=pandas-dev__pandas.95280573.combine_file__11s6papj agent=auditor: baseline_radix=393 exp_radix=701 delta=44% > 15%
  case=pandas-dev__pandas.95280573.combine_file__1eilbetv agent=auditor: baseline_radix=152 exp_radix=228 delta=33% > 15%
  case=pandas-dev__pandas.95280573.combine_file__1eilbetv agent=verifier: baseline_radix=152 exp_radix=228 delta=33% > 15%
  case=pandas-dev__pandas.95280573.combine_file__2p4yneeo agent=auditor: baseline_radix=113 exp_radix=334 delta=66% > 15%
  case=pandas-dev__pandas.95280573.combine_file__2p4yneeo agent=debugger: baseline_radix=113 exp_radix=367 delta=69% > 15%
  case=pandas-dev__pandas.95280573.combine_file__2p4yneeo agent=verifier: baseline_radix=146 exp_radix=367 delta=60% > 15%
  case=pandas-dev__pandas.95280573.combine_file__3ddy6d59 agent=auditor: baseline_radix=113 exp_radix=621 delta=82% > 15%
  case=pandas-dev__pandas.95280573.combine_file__3ddy6d59 agent=verifier: baseline_radix=113 exp_radix=214 delta=47% > 15%
  case=pandas-dev__pandas.95280573.combine_file__3ra8xqln agent=debugger: baseline_radix=150 exp_radix=737 delta=80% > 15%
```

The measured speedup is CONFOUNDED with radix prefix warmth and is NOT a clean
code-aware contribution. Fix the warmup/salt parity before trusting the speedup.

## Speed bar (L4+C2 vs L2, radix-isolated)

- speedup = L2.p50 / L4+C2.p50 = 744.2 / 778.2 = **0.956x**
- L2 avg_radix_prefix = 193; L4+C2 avg_radix_prefix = 310 (delta 118, should be ~0)
- L2 avg_codeaware_reused = 417 (l2_wholeslot=0)
- L4+C2 avg_codeaware_reused = 540 (l2=0, c2=540)
- speed bar (>=1.0x): **NOT MET (AST slower than whole-slot)**

Speedup 0.956x is NOT trustworthy (parity violated).

## Accuracy bar (L4+C2 vs L2, both vs lossless)

- L4+C2 avg_F1 vs lossless = 0.303
- L2   avg_F1 vs lossless = 0.406
- accuracy bar (L4+C2 F1 >= L2 F1): **NOT MET (AST less accurate than whole-slot)**
- (legacy) l3_general config not provided; the bar above (vs L2) is the primary one.
