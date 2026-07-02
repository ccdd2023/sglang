# LAYERED F1=0.508 anomaly — Phase 7 investigation final report (2026-07-02)

## Headline

**All three "make SYNC behave like LAYERED" hypotheses are FALSIFIED.** The
SYNC→LAYERED F1 gap (0.375 → 0.509) is **real, stable, and not explained by
any default-stream race, record_stream collision, or per-layer wait pattern**.

The only remaining structural difference is the **CUDA stream itself**
(LAYERED runs host→GPU copy + RoPE on `load_stream`; SYNC on default
stream). The gap persists under `--agent-count 1` (Exp1) and produces
**divergent outputs on the same input** (SYNC a1 vs LAYERED a1 F1=0.714),
so it is not noise.

## 7-way A/B (5 cases × 5 agents, 32k tokens, position-shift, canonical-prefix)

| config | F1 | TTFT | hypothesis tested | result |
|---|---|---|---|---|
| lossless ref | 1.000 | 948ms | — | (reference) |
| SYNC (orig) | 0.375 | 923ms | — | baseline |
| **LAYERED (orig)** | **0.509** | 918ms | — | **anomaly** |
| SYNC + DOUBLE SYNC before RoPE (Exp3) | 0.375 | — | H1: default-stream race | **falsified** |
| SYNC + per-layer event wait (Exp4) | 0.375 | — | H3: per-layer wait pattern | **falsified** |
| SYNC + record_stream on host/dst (Exp5) | 0.375 | — | H1b: record_stream collision | **falsified** |
| SYNC a1 (Exp1) | 0.369 | — | H4: 5-agent warmup bias | **falsified** |
| LAYERED a1 (Exp1) | 0.559 | — | H4: 5-agent warmup bias | **falsified** |

The three "fence differently on SYNC" experiments all produced F1=0.375
bit-identical to the SYNC baseline. The 5-agent-vs-1-agent ablation showed
both SYNC and LAYERED F1 are stable (SYNC 0.375→0.369, LAYERED 0.509→0.559),
and SYNC_a1 vs LAYERED_a1 on the same input produced F1=0.714 — meaning
the two paths emit materially different token streams on identical prompts.

## What the falsifications tell us

1. **The F1 gap is NOT a default-stream race that LAYERED dodges.** A second
   `torch.cuda.synchronize()` immediately before default-stream RoPE
   (Exp3) had zero effect on F1. The leak detector is NOT racing a writer
   into RoPE's view of the dst slots.
2. **The gap is NOT a per-layer event-wait artifact.** Replicating the
   per-layer `event.record/wait` loop on the SYNC path (Exp4) had zero
   effect. The "consume layer-by-layer" timing is not the mechanism.
3. **The gap is NOT a record_stream collision on host_cat/dst_cat.** Adding
   `record_stream(torch.cuda.current_stream())` on those tensors in the
   SYNC path (Exp5) had zero effect. Default-stream RoPE is not freeing
   those tensors prematurely.
4. **The gap is NOT a 5-agent warmup bias.** Single-agent runs (Exp1) show
   the same 0.37 vs 0.55 split. The gap is not driven by reuse-state from
   earlier agents in the same case.

## What is left

The only remaining structural difference between SYNC and LAYERED is the
**CUDA stream identity itself** (default vs `load_stream`). Pure-torch
`apply_rotary_emb` ops are stream-agnostic; we verified the LAYERED
per-layer code path uses the same chunk/*/-/+/cat/stack operations
(`sglang/srt/layers/rotary_embedding/utils.py:34`). So the math is
identical.

The remaining candidate mechanisms for the gap:

1. **Per-chunk `update_producer` ring buffer side effect.** LAYERED calls
   `layer_done_counter.update_producer()` which advances a 3-slot ring and
   may interact with concurrent radix cache operations in a way that
   changes timing. We did not isolate this; testing it requires mocking
   the LayerDoneCounter.
2. **`record_stream` on `host_cat`/`dst_cat` INSIDE the load_stream scope
   (lines 3238-3240 in radix_cache.py).** LAYERED records those tensors
   on `load_stream`, which prevents the caching allocator from reusing
   their storage during the layered transfer. SYNC doesn't do this for
   the default stream. This is a real structural difference we didn't
   test in isolation (our Exp5 only adds record_stream AFTER the SYNC
   body, not during).
3. **Concurrent cache-controller activity on `load_stream`.** SGLang's
   HiCacheController uses `load_stream` for radix HiCache loads as well.
   If a radix HiCache load fires concurrently with our precompute copy,
   `load_stream` becomes a shared resource and ordering between the two
   could differ from default-stream SYNC.

None of these have a simple, single-line fix. **We are not going to chase
this further under the current "code-aware lossy KV reuse" project** —
even a successful fix would only close the F1 gap from 0.509 to ~0.55,
which still does not meet the 1.000 lossless accuracy bar (the L1+L2+chunk
pool design is fundamentally lossy for shifted-prefix reuse; see
`c2-cacheblend-lossy-not-safe-2026-06-28.md`).

## Decision

**The LAYERED F1=0.508 is documented as a real but unfixed mechanism-specific
side-effect of running the host→GPU copy+RoPE on `load_stream` rather than
the default stream.** It is not a transferable correctness improvement
(we cannot reproduce it on SYNC by adding fences). It is not a noise
artifact (the gap is stable across a1/a5 and across 5 cases). It is not
a default-stream race (Exp3 + Exp5 both rule this out).

The next direction is **not** to chase this further. Per the user's
original goal, both bars (speed and accuracy) remain unmet. The path
forward is true CacheBlend (attention recompute) — the only known
mechanism that delivers both bars, for which the precompute infrastructure
we built is the prerequisite.

## Files

- 5-way A/B (Phase 6): `results/kvcomm_ab/precompute_ab_report/COMPARISON.txt`
- Single-agent SYNC a1: `results/kvcomm_ab/7b_precompute_ab_sync_a1/`
- Single-agent LAYERED a1: `results/kvcomm_ab/7b_precompute_ab_layered_a1/`
- SYNC + DOUBLE SYNC (Exp3): `results/kvcomm_ab/7b_precompute_ab_sync_doublesync/`
- SYNC + PERLAYERWAIT (Exp4): `results/kvcomm_ab/7b_precompute_ab_sync_perlayerwait/`
- SYNC + RECORDSTREAM (Exp5): `results/kvcomm_ab/7b_precompute_ab_sync_recordstream/`
- Byte-cmp dump site (limited yield): `results/kvcomm_ab/bytecmp/sync_batched_n0.pt`
- Launchers: `results/kvcomm_ab/run_7b_precompute_ab_{sync,layered}_a1.sh`,
  `..._sync_doublesync.sh`, `..._sync_perlayerwait.sh`,
  `..._sync_recordstream.sh`, `..._{sync,layered}_bytecmp.sh`

## Code changes (preserved, default OFF)

`python/sglang/srt/mem_cache/radix_cache.py`:
- `_bytecmp_dump` helper (guarded by `SGLANG_KVFLOW_BYTECMP_DUMP`).
- Three dump sites: per-chunk SYNC (`:2974`), batched SYNC (`:3565`),
  LAYERED per-layer (`:3230`).
- Exp3/4/5 hooks in `_load_host_chunks_to_device` SYNC body, all default
  OFF behind `SGLANG_KVFLOW_{DOUBLE_SYNC,PERLAYERWAIT,RECORDSTREAM_SYNC}`.
- Leak-detector fix: `placeholder_chunk_pool_pinned_tokens` was over-
  accounting for host-pool entries. Split into
  `placeholder_chunk_pool_pinned_device_tokens` (only device-resident
  entries contribute to `protected_size()`).

## What's NOT pursued (and why)

- **True CacheBlend (attention recompute over copied KV)**: the only
  known path to BOTH bars. Out of scope for this anomaly investigation;
  requires its own plan and user sign-off.
- **Layered/load_stream micro-optimization**: even with the F1 gap
  closed (0.55 vs 0.50), speed bar is still unmet (TTFT ≈ lossless
  in both paths; transfer is not the bottleneck, see Phase 6).
