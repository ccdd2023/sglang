# Fair A/B Report

- exclude_source_agent: True (source role = `implementer`)
- parity tolerance: 15%

## Per-config (reusers only)

| config | n | p50_TTFT | avg_cached | avg_radix_prefix | avg_codeaware_reused | avg_l2 | avg_c2 | avg_F1 | F1_source |
|---|---|---|---|---|---|---|---|---|---|
| baseline(L2 whole-slot) | 20 | 739.6 | 738 | 119 | 619 | 0 | 619 | 0.498 | 20 |
| experimental(L4+C2 AST) | 20 | 739.6 | 738 | 119 | 619 | 0 | 619 | 0.498 | 20 |
| lossless(reference) | 20 | 955.8 | 109 | 109 | 0 | 0 | 0 | 1.000 | 20 |
- F1 source: real (vs lossless, from outputs.jsonl)

## Warmup-parity gate (B2)

- avg radix_prefix_tokens: baseline(L2)=119, experimental(L4+C2)=119, delta=0
- **PARITY OK** — radix L1 prefix cancels (0 per-agent violations).

## Speed bar (L4+C2 vs L2, radix-isolated)

- speedup = L2.p50 / L4+C2.p50 = 739.6 / 739.6 = **1.000x**
- L2 avg_radix_prefix = 119; L4+C2 avg_radix_prefix = 119 (delta 0, should be ~0)
- L2 avg_codeaware_reused = 619 (l2_wholeslot=0)
- L4+C2 avg_codeaware_reused = 619 (l2=0, c2=619)
- speed bar (>=1.0x): **MET (AST not slower than whole-slot)**

## Accuracy bar (L4+C2 vs L2, both vs lossless)

- L4+C2 avg_F1 vs lossless = 0.498
- L2   avg_F1 vs lossless = 0.498
- accuracy bar (L4+C2 F1 >= L2 F1): **MET (AST not less accurate than whole-slot)**
- (legacy) l3_general config not provided; the bar above (vs L2) is the primary one.
