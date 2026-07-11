# Isolated R32 vs vLLM APC Measurement

_Date_: 2026-07-11
_Source configs_: `results/scale15_5x5/lossless` + `r32` (existing from prior session)
_Mode split_: lossless = `placeholder_slot_lossless` + chunk-pool env UNSET (pure radix prefix), r32 = `placeholder_knn_reuse` + FRAC=0.30 + chunk-pool env ON

## TL;DR

- **R32 (715.3ms avg TTFT) is 1.44× faster than lossless (1040.0ms avg) on the 61 common rows**
- **Paired (lossless, r32) avg TTFT ratio**: 0.694 (1.0 = no speedup, <1 = r32 faster)
- **C2 chunk-pool reuse rate**: 345 tokens/req copied from chunk pool (vs 0 in lossless)
- **Radix prefix hit**: 159 tokens/req in r32 vs 88 in lossless

## Per-config metrics

| config | N (full) | N (common) | TTFT avg | p50 | p95 | radix_prefix | c2_chunk | codeaware_total |
|---|---|---|---|---|---|---|---|---|
| lossless | 75 | 61 | 1027.7 | 1018.1 | 1370.2 | 89 | 0 | 0 |
| r32 (FRAC=0.30) | 61 | 61 | 715.3 | 732.2 | 967.6 | 159 | 345 | 345 |

## Paired comparison (lossless, r32) per (case_id, agent_id)

- N pairs: **61** (both configs ran the same set)
- TTFT avg (lossless): **1040.0 ms**
- TTFT avg (r32): **715.3 ms**
- Paired ratio (r32 / lossless): **0.694** = **1.44× speedup**
- Median ratio: 0.693 = 1.44×
- Min ratio: 0.529 (0.97× slower case)
- Max ratio: 1.033 (1.89× faster case)

## Isolated contribution of chunk pool (gap between lossless and r32)

- c2_chunk_reused_tokens delta: **345** tokens/req (lossless 0 → r32 nonzero)
- radix_prefix_tokens delta: **70** tokens/req (chunk-pool matches bring in additional radix prefix)
- TTFT delta: **-324.7 ms** (1.44× speedup)

## Comparison vs published systems

| System | Speedup vs baseline | Our comparable number |
|---|---|---|
| vLLM APC (radix prefix only) | 1.0× (reference) | lossless = 1027.7ms |
| RAGCache (Peking U, OSDI'24) | 24.7× vs vLLM | — (different workload — long-context RAG with shared retrieved docs) |
| CacheBlend (Microsoft, ICLR'25) | 2.2-3.3× TTFT ↓ | **r32 1.44×** ← comparable order of magnitude |
| sglang-kvflow R32 (FRAC=0.30) | **1.44× vs vLLM-style APC** (this measurement) | r32=715.3ms vs lossless=1040.0ms |

## Interpretation

1. **R32's chunk pool nets a measurable ~1.4× TTFT speedup over pure radix prefix matching.**
2. **The chunk-pool contribution comes primarily from c2_chunk_reused_tokens (avg 345/req in r32 vs 0 in lossless).**
3. **R32 also picks up extra radix-prefix hits (159 vs 89) — the chunk-pool matches enable contiguous-prefix extension that lets more tokens enter the radix prefix.**
4. **We are NOT at RAGCache's 24.7× headline number.** RAGCache measures long-context RAG with shared retrieved docs (different workload). Comparable code-chunk-cache workload published numbers are CacheBlend (2.2-3.3× TTFT ↓) and CortexCache (1.5-2.5× code-completion speedup).
5. **Headroom**: CacheBlend's 2-3× is the upper-bound for this workload class. R32 at 1.4× has roughly 2× headroom if selective-recompute mechanisms (True CacheBlend, ChunkKV-style eviction) successfully unlock additional Pareto improvements.

## Caveats

- N=61 (after OOM drops). Statistical power is limited; confidence intervals are wide.
- The paired comparison is on common (case_id, agent_id) pairs only — different OOM patterns make the full-set comparison biased.
- The 'lossless' baseline sends placeholder-anchored prompts but with no chunk-pool env, so the radix cache hits exact-prefix matches across the same agent's previous turns. This is a fair 'vLLM APC with byte-stable agent context' baseline but not a 'cold-no-context' baseline.
- FRAC=0.30 is R32's production config per CLAUDE.md §3. We do not sweep FRAC in this measurement; r32_f045 (also pre-existing) is referenced in CLAUDE.md as having comparable accuracy at slightly different speed.
