# Iteration → Tech-Name Index (sglang-kvflow, 2026-07-08)

This index maps our internal iteration numbers (R##) to descriptive
**technique / algorithm** names so external readers can map the work to the
published literature without learning our internal numbering scheme.

| R## | Tech Name | Status | One-liner |
|---|---|---|---|
| R19 | AST-chunked placeholder pool (no selective recompute) | baseline | 1.30× speed, 80% type_agreement, 8% garbage |
| R21 | Verdict-task scoring framework | reference | introduced verdict PASS/FAIL/UNK + failure-type agreement metric |
| R26 | 3B-Instruct × 3 (speed-prioritized) | separate Pareto | 2.014× speedup, 25% type_agreement on 3B × 3 |
| R28 | anchor-type filter | null | Pool was already 100% function/class; env var kept |
| R29 | attention-sink recompute | declined | needs attention-kernel hook (multi-week work) |
| R30 | type-sig filter | declined | subsumed by R34 |
| R31 | CacheBlend deep research | 17 confirmed | 99-agent fan-out, HKVD-selected |
| R32 | constant-FRAC head_recompute | prior Pareto | FRAC=0.30, 41.7% type_agreement, leading-K-token heuristic |
| R33 | imports-prelude | RETRACTED | violates "prompt-byte-identical" constraint |
| R34 | type-annotation gate | RETIRED | pandas 0.x untyped, gate fires on nothing |
| R35 | 3B×3 + head_recompute stack | NEGATIVE | naive composition fails, different Pareto |
| R36 | constant-FRAC sweep | confirms R32 | 0.15/0.20/0.35 all regress; 0.30 unique |
| R37 | per-chunk-position FRAC (0.50/0.20) | first version | 45.5% type_agreement, EARLY=0.50/LATE=0.20/N=2 |
| R38a | per-chunk-position FRAC repro | byte-exact | 45.5% ✓ (R37 reproducible) |
| **R38b** | **per-chunk-position FRAC (0.60/0.15)** | **NEW PARETO** | **50.0% type_agreement, byte-exact 3×, -0.9% TTFT, +38% reuse** |
| R38d | EARLY_N=3 over-shoot | regress | 33.3% (EARLY_N too broad) |
| R39a | R38b reproducibility check | byte-exact | 50.0% ✓ |
| R39b | 0.70/0.10 over-shoot | regress | 40.0% (FRAC_LATE=0.10 too aggressive) |
| R39c | 0.65/0.15 plateau check | same point | 50.0% (plateau confirmed) |

## Algorithm-to-citation mapping (for external readers)

| Our R## | Closest published work | Notes |
|---|---|---|
| R28 (anchor-type filter) | — | our own AST-chunk design |
| R31 (deep research) | CacheBlend (arXiv:2405.16444) | identified as the only published algorithm that can recover cross-context accuracy |
| R32 (constant-FRAC head_recompute) | CacheBlend-style constant-FRAC | **leading-K-token heuristic**, NOT the full per-layer HKVD filter |
| R33 (imports-prelude) | — | FALSIFIED |
| R37 / R38b (per-chunk-position FRAC) | per-layer r1 > r2 > r* in CacheBlend §4.3 | our per-CHUNK-POSITION approximation; no attention-kernel hook |
| R34 (type-annotation gate) | — | FALSIFIED (no production code) |
| R35 (3B×3 + head_recompute) | — | NEGATIVE: different architecture has different Pareto |
| R36 (constant-FRAC sweep) | — | R32's 0.30 confirmed unique Pareto |
| R39 (R38b repro + sweep) | — | 50.0% reproduced 3× byte-exact |

## How to refer to a result in conversation

Instead of "R38b", say **"per-chunk-position FRAC (R38b) — FRAC_EARLY=0.60, FRAC_LATE=0.15, EARLY_N=2"**.

Instead of "R32", say **"constant-FRAC head_recompute (R32) — FRAC=0.30"**.

R## is retained in the deck for cross-referencing the results/ directory
tree and git commit history; tech name leads.
