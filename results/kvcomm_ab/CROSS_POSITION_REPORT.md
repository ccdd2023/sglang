# Cross-Position Fix — Results Report (2026-06-30)

## TL;DR

The approved **cross-position `slot_id` fix unblocked byte-exact KVCOMM reuse** (was 0 reuse
across all paths → now 7–13/16 reusers hit). Two gap-handling mechanisms were measured:

- **C2-direct gap-ZEROING**: 2.5× speed but **F1=0.008 (garbage)** — zeroing the task
  instruction destroys the output. Unacceptable.
- **STAGED gap-prefill** (non-lossy gap): **1.085× speed, F1=0.376** (valid-but-different
  outputs). Best balance.

In the same-code position-shift regime (full sharing), **whole-slot (L2) beats AST-chunk
(L4) on speed**; AST chunking's benefit is in partial-sharing, not here.

## The Fix

`bench_kvcomm_ttft_stress.py:build_slot_messages` (~line 464):

```python
# was: slot_id=f"code_base{idx}"            # POSITIONAL → cross-position miss
slot_id=f"code_base:{segment.name}"         # CONTENT-DERIVED → cross-position hit
```

`segment.name` (file path) is stable across positions and unique per task. This makes the
L4 chunk-pool key `(slot_id, chunk_signature)` and the L2 `content_signature` stable across
positions, so the same code at a different position shares a pool key → lookup hits. The
RoPE delta (`radix_cache._apply_rope_delta_to_keys`) handles the absolute-position shift.

## Setup

giant-codebase (pandas), 4 tasks × 5 agents, `--position-shift --no-vary-code` (same code,
cyclically rotated slot order per agent), MiniLM L3 + offset-gate OFF. Distinct per-agent
`cache_salt`. Fair analyzer (`analyze_fair_ab.py`): source agent excluded, radix-prefix
parity gate (OK, delta=0), real token-F1 from `outputs.jsonl` vs lossless.

## Results (reusers only, agent 1 excluded)

| config | reuse hit | avg codeaware | speedup vs lossless (avg / p50) | F1 vs lossless |
|---|---|---|---|---|
| L2 whole-slot, gap-ZERO (`SGLANG_CACHEBLEND_DIRECT=1`) | 7/16 | 2751 | 2.533× / 1.015× | 0.106 (reuse cases 0.008) |
| L4 AST-chunk, gap-ZERO | 5/16 | 386 | 1.015× / 0.987× | 0.197 |
| **L2 whole-slot, STAGED** (DIRECT OFF) | **13/16** | **1252** | **1.085× / 1.052×** | **0.376** |
| L4 AST-chunk, STAGED | 11/16 | 642 | 0.998× / 0.958× | 0.388 |
| lossless (reference) | — | 0 | 1.0× | 1.000 |

## Key findings

### 1. Gap-zeroing destroys accuracy; staging fixes it
The gap = the header between the radix prefix and the first code chunk (~30 tokens, including
the task instruction "Inspect the repeated repository code and answer with one concise
implementation risk").

- **Gap-ZEROING** (C2-direct) zeroes the gap KV → the model loses the instruction →
  garbage output (F1=0.008 for reuse cases) despite 2.5× speed.
- **STAGED gap-prefill** prefills the gap with real KV (scheduler recompute-gap round,
  `schedule_policy.py:784`), then copies the contiguous chunk run. Non-lossy gap → F1=0.376.

### 2. F1=0.376 is cross-context KV loss, NOT garbage
Inspected `outputs.jsonl`: staged outputs are **coherent, on-topic, valid implementation-risk
answers** that differ from lossless only in *specifics* (which file / which risk). Example
(case 11s6papj, agent 2):
- lossless: "A key implementation risk is the potential for memory leaks if the buffer memory
  is not properly managed..."
- staged: "The repeated code in `buffer.py` and `dataframe.py` for handling buffer types and
  null representations introduces a risk of inconsistent memory management..."

Both valid. BoW-F1 vs a single reference is harsh for this open-ended task. The loss is the
fundamental cross-context KV dependency (KV at layers>0 encodes the preceding prefix;
reordered segments → stale KV), confirming `c2-cacheblend-lossy-not-safe` with data — but
the loss is "different-but-valid", not "garbage", when the gap is prefilled.

### 3. Whole-slot (L2) > AST-chunk (L4) for speed in full-sharing position-shift
- L2-staged: 1.085× speed, F1=0.376.
- L4-staged: 0.998× (no speedup), F1=0.388.

AST chunking hurts here because (a) the AST chunker only emits `FunctionDef`/`ClassDef`
chunks, **missing the leading import/module-level block** → first chunk far from the prefix
→ gap-cascade (some segments get 0 reuse); (b) per-chunk copy overhead. Whole-slot
(`SGLANG_CHUNK_COARSE=1`, byte_start=0) captures the whole segment in one copy.

**AST chunking's benefit is in PARTIAL-sharing** (some functions shared, others new), NOT
full-sharing position-shift. Accuracy bar (L4 F1 ≥ L2 F1): MET (0.388 ≥ 0.376). Speed bar
(L4 ≥ L2): NOT MET in this regime.

### 4. The fundamental KVCOMM tension
Non-lossy reuse requires prefilling the gap (staged) → only the chunk run beyond the gap is
"saved" → speedup only when the leading gap is small and the first chunk run is large. The
3 slow staged outliers (800–1200ms) are cases that staged a gap-prefill round but the first
chunk run was small → staging overhead without copy benefit.

## Bars vs the user's goal (speedup + acceptable accuracy, not worse than general)

- **Speed:** L2-staged 1.085× (MET, modest). L4-staged 0.998× (NOT MET).
- **Accuracy (not worse than general L2):** L4-staged 0.388 ≥ L2-staged 0.376 (MET).
- **Both bars by the AST path in this regime:** NOT simultaneously met (AST meets accuracy
  but not speed; whole-slot meets speed).

## Next options (need user direction)

- **(A)** Accept L2-staged (whole-slot, 1.085×, F1=0.376) as the code-aware mechanism; frame
  AST chunking as a partial-sharing refinement.
- **(B)** Test a **partial-sharing scenario** (e.g., one function modified per agent) where
  AST chunking beats whole-slot — the regime where AST's granularity pays off.
- **(C)** **True CacheBlend** (attention recompute for the copied chunks) to push accuracy
  toward 1.0 — expensive (the `c2-fundamental-limits` cost).
- **(D)** **Larger model** — 3B prefill is cheap so reuse speedup is modest; a larger model
  would amplify the copy-vs-prefill delta.
- **(E)** Prompt redesign so the shared instruction precedes the differing role → radix
  prefix covers the instruction → smaller gap → less staging overhead / higher speedup.

## Artifacts

- Launchers: `results/kvcomm_ab/run_{lossless,l2_coarse,l4_c2,l2_coarse_staged,l4_staged}.sh`
- Per-config rows + outputs: `results/kvcomm_ab/{lossless,l2_coarse,l4_c2,l2_coarse_staged,l4_staged}/`
- Analyzer reports: `results/kvcomm_ab/report*/FAIR_AB_REPORT.md`
- This report: `results/kvcomm_ab/CROSS_POSITION_REPORT.md`

---

## 7B-Coder follow-up (larger-model direction, 2026-06-30)

User chose "larger model for more speedup". Re-ran the staged configs with
`Qwen/Qwen2.5-Coder-7B-Instruct` (28 layers, hidden 3584) on the RTX 4090
(`--mem-fraction-static 0.85 --max-total-tokens 32768`). 7B lossless p50=932ms
vs 3B's 446ms (~2× slower prefill → bigger copy-vs-prefill delta).

| model | config | reuse hit | avg codeaware | per-case speedup (avg / p50) | F1 vs lossless |
|---|---|---|---|---|---|
| 3B | L2 whole-slot STAGED | 13/16 | 1252 | 1.085× / 1.052× | 0.376 |
| 3B | L4 AST STAGED | 11/16 | 642 | 0.998× / 0.958× | 0.388 |
| **7B** | **L2 whole-slot STAGED** | **13/16** | **1259** | **1.136× / 1.144×** | **0.461** |
| 7B | L4 AST STAGED | 16/16 | 1069 | 1.067× / 1.044× | 0.399 |

**Larger model confirmed the direction** (per-case speedup, the fair same-case metric):
- **Speedup amplified**: L2 1.052→1.144×; L4 0.958→1.044× (L4 now shows positive speedup).
- **F1 improved**: L2 0.376→0.461; L4 0.388→0.399. The larger/coder model is more robust to
  the cross-context KV loss.
- Trajectory: bigger model → more speedup + better accuracy. (3B prefill is cheap so reuse
  saves little; 7B prefill is ~2× costlier so reuse saves more.)

**But AST (L4) is still dominated by whole-slot (L2) in full-sharing position-shift** — now
on BOTH axes for 7B: speed 1.044× < 1.144×, accuracy F1 0.399 < 0.461. The analyzer's
L4-vs-L2 speed bar = 0.871× (AST slower); accuracy bar NOT MET (L4 < L2). This is structural,
not model-dependent: the AST chunker misses the leading import/module block (gap-cascade) and
adds per-chunk overhead. A larger model amplifies the absolute speedup but cannot make AST
beat whole-slot under full sharing.

**Bars vs user's goal (7B):**
- Speed (vs lossless): L2-staged 1.144× MET; L4-staged 1.044× MET (per-case).
- Accuracy (L4 not worse than general L2): L4 0.399 < L2 0.461 → NOT MET.

**Conclusion:** the larger-model direction raised the absolute speedup and F1 (good), but did
not close the AST-vs-whole-slot gap, which is fundamental to the full-sharing position-shift
regime. To demonstrate AST's advantage we need either a **partial-sharing scenario** (where
granularity pays off) or an **AST-chunker fix** to capture module-level code (so AST ≥
whole-slot under full sharing, then beats it under partial sharing).

### 7B artifacts
- Launchers: `results/kvcomm_ab/run_7b_{lossless,l2_staged,l4_staged}.sh`
- Rows + outputs: `results/kvcomm_ab/7b_{lossless,l2_staged,l4_staged}/`
- Analyzer: `results/kvcomm_ab/report_7b/FAIR_AB_REPORT.md`

### AST chunker gap-fill fix (7B, `SGLANG_CHUNK_FILL_GAPS=1`)
Added a `_build_module_chunk_span` to `ast_chunker.py` that fills non-anchor gaps
(leading imports, inter-anchor code, trailing) with "module" chunks, so the AST
chunker covers the WHOLE slot (was: only FunctionDef/ClassDef → missed imports →
gap-cascade). Gated by `SGLANG_CHUNK_FILL_GAPS=1` (default OFF). 43 chunker unit
tests still pass.

| 7B config | reuse | avg_ca | per-case sp (p50) | F1 |
|---|---|---|---|---|
| L2 whole-slot STAGED | 13/16 | 1259 | 1.144× | 0.461 |
| L4 AST STAGED (no fill) | 16/16 | 1069 | 1.044× | 0.399 |
| **L4 AST STAGED + FILL_GAPS** | 14/16 | 1312 | 1.074× | 0.414 |

Gap-filling IMPROVED AST (reuse 1069→1312, speedup 1.044→1.074×, F1 0.399→0.414 —
now captures the module-level code). But L4-fill (1.074×, F1=0.414) is STILL below
whole-slot L2 (1.144×, F1=0.461). Reason: AST with fill has MORE chunks (module +
functions) → more per-chunk alloc+move+RoPE overhead than whole-slot's single copy.
In full-sharing, granularity is a DISADVANTAGE (N copies > 1 copy); AST's granularity
only pays off under PARTIAL sharing (where whole-slot can't reuse a partially-changed
slot but AST can reuse the unchanged functions).

**Robust conclusion (3 data points: 3B, 7B, 7B+gap-fill):** in same-code position-shift
(full sharing), whole-slot (L2) is optimal and AST (L4) cannot beat it — on both speed
and accuracy. The cross-position fix + larger model + gap-fill raised the absolute
numbers (best: 7B L2-staged 1.144× speedup, F1=0.461), but the AST-vs-whole-slot gap is
structural to the full-sharing regime. To demonstrate AST's advantage requires a
partial-sharing + position-rotation scenario (some functions shared at different
positions, others new) — the realistic coding scenario where whole-slot fails (slot
differs) but AST reuses the shared functions cross-position.

---

## Partial-sharing + rotation scenario (2026-07-01, 7B-Coder)

User chose "Partial-sharing + rotation". Added `--partial-share` to
`bench_giant_codebase_reuse.py`: each agent DROPS one different top-level function
(FunctionDef/ClassDef) from each segment (guarded `len(defs) >= 2`), so the segment text
differs per agent (whole-slot byte-exact match FAILS) while the remaining functions are
byte-identical (AST reuses them cross-position via content-derived slot_id + per-chunk
signature). Combined with `--position-shift --no-vary-code`.

| 7B config | reuse hit | avg_ca | p50 TTFT | F1 vs lossless |
|---|---|---|---|---|
| L2 whole-slot STAGED | 13/16 | 1259 | 832ms | 0.534 |
| **L4 AST + FILL_GAPS STAGED** | 9/16 | 507 | **691ms** | **0.694** |
| lossless | — | 0 | 864ms | 1.000 |

**Both bars MET (on average) — the first regime where AST beats whole-slot:**
- **Speed:** L4 1.205× faster than L2 (832/691); ~1.25× vs lossless (p50 ratio). MET.
- **Accuracy:** L4 F1=0.694 > L2 F1=0.534. MET (AST not worse than general).

**Why AST wins here (and not in full-sharing):** whole-slot copies LARGE import blocks
cross-context (lossy + staging overhead → slow, F1=0.534). AST copies only the SHARED
FUNCTIONS (smaller, fewer tokens cross-context → faster, less loss → F1=0.694). The
granularity pays off because the slot text differs (whole-slot can't match) but
functions are shared (AST matches per-function).

**HONEST CAVEAT — high variance, cross-context loss remains:**
Per-row F1 for L4 reuse cases ranges 0.00–1.00. When L4 reuses SUBSTANTIALLY (1015, 1874
tokens) the F1 crashes to 0.00–0.06 (cross-context KV loss = near-garbage); when it
reuses little or the prefix difference is small, F1 is 0.88–1.00. The average 0.694 is
inflated by several 0-reuse cases (F1=1.00). So:
- The bars are met ON AVERAGE but the per-case accuracy is brittle.
- The fundamental cross-context loss ([[c2-cacheblend-lossy-not-safe]]) is NOT fixed by
  partial-sharing — it's still lossy when reuse is substantial.
- The result is also noisy (only 4 tasks × 4 reusers = 16 samples; some tasks like
  2p4yneeo are slow for both configs).

**Net:** partial-sharing is the correct regime to demonstrate AST's advantage (it beats
whole-slot on both bars), confirming the mechanism works. But the cross-context accuracy
loss remains the limiting factor for substantial reuse — closing it needs true CacheBlend
(attention recompute) or a scenario with smaller prefix differences. The 16-sample result
should be re-run with more tasks for a robust average.

### Partial-share artifacts
- Code: `--partial-share` flag in `bench_giant_codebase_reuse.py` (run_one_task ~:355)
- Launchers: `results/kvcomm_ab/run_7b_ps_{lossless,l2_staged,l4_fill}.sh`
- Rows + outputs: `results/kvcomm_ab/7b_ps_{lossless,l2_staged,l4_fill}/`
- Analyzer: `results/kvcomm_ab/report_7b_ps/FAIR_AB_REPORT.md`

### ROBUST 12-task partial-share (7B, 2026-07-01) — supersedes the 4-task run

Re-ran with `--max-tasks 12` for a robust average (the 4-task result was 16 samples,
noisy). L2 had 42 reusers, L4 had 48 (some tasks skipped by L2 for too-few defs);
42 aligned common reuser cases for a fair speed comparison:

| metric | L2 whole-slot | L4 AST+fill | verdict |
|---|---|---|---|
| p50 TTFT (aligned 42 cases) | 927ms | 966ms | L4/L2 = **0.96× (AST slightly SLOWER)** |
| per-case speedup vs lossless (p50) | 1.046× | 1.004× | both barely beat lossless |
| avg codeaware reused | 1399 | 516 | AST reuses less (only shared functions) |
| F1 vs lossless | 0.513 | 0.622 | **AST more accurate** |

Analyzer (`report_7b_ps12`): speed bar **0.952× — NOT MET** (AST slower); accuracy bar
**MET** (L4 F1 0.622 > L2 0.513).

**Honest conclusion (ROBUST, supersedes the 4-task "1.205× AST win"):**
- The 4-task "AST 1.205× faster" was NOISE — at 12 tasks/42 cases, AST is slightly
  *slower* than whole-slot (0.96×). AST's per-chunk alloc+move+RoPE overhead roughly
  cancels its reuse advantage at this scale, even in the partial-sharing regime where
  AST's granularity *should* help.
- AST IS more accurate (F1 0.622 vs 0.513) — it copies only shared functions (smaller
  cross-context copies) while whole-slot copies large import blocks (lossier). This bar
  holds robustly.
- **So: AST meets the accuracy bar but NOT the speed bar** in partial-sharing. The
  remaining lever for AST speed is eliminating per-chunk overhead (batch all chunk
  copies into one alloc+move+RoPE — option C from the earlier question), so AST's
  smaller/cheaper reuse actually translates to a speed win.

### 12-task partial-share artifacts
- Launchers: `results/kvcomm_ab/run_7b_ps12_{lossless,l2_staged,l4_fill}.sh`
- Rows + outputs: `results/kvcomm_ab/7b_ps12_{lossless,l2_staged,l4_fill}/`
- Analyzer: `results/kvcomm_ab/report_7b_ps12/FAIR_AB_REPORT.md`
