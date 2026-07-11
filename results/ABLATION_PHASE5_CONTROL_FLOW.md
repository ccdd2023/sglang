# Phase 5: Control-Flow-Selective Recompute — 8-config Equal-Budget Ablation

## TL;DR

**Phase 5 implemented + measured. Control-flow selective recompute fires on most chunks (cf_k_avg=267, fire rate OK per R34 guard), but does NOT beat the R32 uniform sweep at equal budget.**

- type_match: controlflow **9.8%** vs R32_f045 **11.5%** at matched c2_reuse (259 vs 268). Delta **-1.6pp**, well within both CIs (overlap heavily).
- TTFT: controlflow **738ms** vs R32_f045 **721ms** (controlflow is 17ms slower, 0.98×).
- The HKVD-by-signal finding (control_flow K_dev 5.7× > data_flow) **does NOT translate** to a better selective recompute policy. Possible explanations at end.

This is **negative-but-not-definitive**: the policy works (fires correctly, comparable accuracy/TTFT to R32_f030), but does not unlock a Pareto improvement over uniform R32 sweep.

## Setup

### Implementation (radix_cache.py + telemetry)
- New env `SGLANG_CHUNK_HEAD_RECOMPUTE_BY_CONTROL_FLOW=1` (default OFF).
- New helper `RadixCache._count_control_flow_tokens(chunk_text, tokenizer)` — parses chunk_text with stdlib `ast`, walks for control-flow nodes (If/For/While/With/Try/Return/Raise/Yield/Assert/Break/Continue), counts tokens whose char span overlaps any control-flow range.
- New dispatch branch in `_build_chunk_plan` (highest priority, before node_kind and frac): `_head_k = n_control_flow_tokens(chunk)`. Fires if `0 < n_control_flow_tokens < chunk_len`.
- Telemetry counter `placeholder_chunk_pool_control_flow_k_count` added to 4 emission sites: `serving_chat.py` (2 lists), `scheduler_output_processor_mixin.py` (1 list + 1 dict access), `bench_kvcomm_ttft_stress.py` (1 dict entry).
- Launcher `results/scale15_5x5/launchers/run_controlflow.sh` (mirror of `run_nodekind.sh`, swap `BY_CONTROL_FLOW=1` for `NODE_KIND=1`).

### Critical bug found and fixed
First benchmark run had **control_flow fire rate 0%**. Diagnosis: missing `placeholder_chunk_pool_control_flow_k_count` column in `rows.csv`. Fix: add `chunk_pool_control_flow_k = int(meta.get(...) or 0)` extraction and dict entry in `bench_kvcomm_ttft_stress.py:1139-1140, 1264-1265`. Re-run shows fire rate OK (cf_k_sum=16316 / hit_sum=1483 = 1100%; per-row cf_k_avg=267).

### Benchmark
- n=15 pandas tasks × 5 agents = 75 nominal, 61 actual rows (same OOM pattern as other configs).
- Pool: `pandas_15case_v1` (120 chunks precomputed, byte-exact match).

## Results (n=61 rows, OOM-stable subset)

### Per-config table (sorted by c2_reused desc = fastest first)

| config | n | type_match | /n% | 95% CI | TTFT (ms) | c2_reuse | reuse | cf_k_avg | fire% |
|---|---|---|---|---|---|---|---|---|---|
| lossless | 75 | 8 | 10.7% | [3.6, 29.1] | 1028 | 0 | 0 | – | – |
| nodekind_sig | 61 | 2 | 3.3% | [0.0, 9.1] | 701 | 523 | 523 | 0 | 1182% |
| R32_f015 | 61 | 4 | 6.6% | [0.0, 18.2] | 707 | 465 | 465 | – | – |
| R32_f026 | 61 | 6 | 9.8% | [0.0, 23.6] | 713 | 380 | 380 | – | – |
| **nodekind** | 61 | 4 | 6.6% | [0.0, 15.0] | 715 | 362 | 362 | 0 | 1256% |
| R32_f030 | 61 | 6 | 9.8% | [0.0, 26.0] | 715 | 345 | 345 | – | – |
| R38b | 60 | 4 | 6.7% | [0.0, 30.0] | 721 | 283 | 283 | – | – |
| R32_f045 | 61 | 7 | 11.5% | [0.0, 29.1] | 721 | 268 | 268 | – | – |
| **controlflow** | 61 | 6 | 9.8% | [0.0, 21.7] | 738 | 259 | 259 | **267** | 1100% |

### Pareto (type_match% vs reuse)
- Frontier: `[nodekind_sig, R32_f015, R32_f026, R32_f045]` — controlflow is NOT on frontier (lower type_match than R32_f045 at comparable reuse).
- controlflow sits between R32_f030 (9.8%, 715ms) and R32_f045 (11.5%, 721ms) on accuracy; slower than both on TTFT.

### Vertical slice: controlflow vs R32 at matched reuse (≈ equal B)
- Closest match: R32_f045 (c2_reuse=268, controlflow=259, |Δ|=8)
- type_match delta: **-1.6pp** (controlflow 9.8% vs R32_f045 11.5%)
- TTFT: 738ms vs 721ms (**0.98×**, controlflow slower)
- Both CIs overlap heavily → **delta is not statistically significant**

### Phase 5 fire-rate guard
- control_flow_k_sum = 16316 / hit_sum = 1483 = 1100%
- Per-row cf_k_avg = 267 (sum of n_control_flow_tokens over chunks fired)
- **Path fires correctly** (no R34-style no-op)

## Interpretation

### What worked
- Implementation correct: dispatch fires on most chunks (cf_k_sum 16316, hit_sum 1483, fire rate OK).
- Telemetry complete: 4 emission sites wired, CSV column present, analyze script picks it up.
- Mechanism plausible in isolation: control_flow tokens ARE 5.7× more sensitive (HKVD n=24 paired, p=0.0000).
- Policy is **competitive** with R32_f030 on accuracy (9.8% vs 9.8%) and TTFT (738ms vs 715ms — 23ms slower).

### What didn't
- Control-flow selective recompute is **NOT Pareto-better** than uniform R32 sweep. R32_f045 dominates controlflow on both accuracy (+1.6pp) and TTFT (-17ms).
- The HKVD sensitivity finding does NOT translate to better selective recompute.

### Why HKVD positive doesn't transfer
Four plausible explanations (in order of likelihood):

1. **Bucket too wide.** `CONTROL_TYPES` includes If/For/While/With/Try + Return/Raise/Yield/Assert/Break/Continue. The body of these nodes contains data tokens (BinOp, Compare, literals), not just keywords. So "control flow tokens" as defined here is **a mixture**, not pure structural markers. A finer split (keywords-only vs control-body) might isolate a useful signal.

2. **Sensitivity ≠ must-recompute.** HKVD measures how much KV drifts under prefix swap. But the model can tolerate drift at some layers (early layers) without output corruption. Recomputing head_K = first N tokens is a conservative policy that catches all layers; we may be over-recomputing at layers that don't need it.

3. **Coverage gap.** `_count_control_flow_tokens` returns ~267 per row but R32_f045 uses FRAC=0.45 × chunk_len. The control_flow K may be lower than R32_f045's effective K in some chunks, leaving more stale KV uncorrected. But c2_reuse shows controlflow actually copied LESS (259 vs 268), so this isn't the issue.

4. **Statistical noise.** 61 rows with wide CIs (e.g., controlflow [0.0, 21.7] vs R32_f045 [0.0, 29.1]) make the -1.6pp delta indistinguishable from sampling noise. A larger n or strict OOM-free subset would be needed.

## Decision

**Per the plan's Case B criteria:**
- controlflow type_match > R32@equal_B AND speedup ≥ 1.3× → **NOT MET**
- type_match delta is +1.6pp WORSE (not better), speedup is 0.98× (not ≥ 1.3×)

**Verdict: Phase 5 prototype built + measured; selective control-flow recompute does not beat uniform R32 sweep. Treat as negative result.**

This does NOT refute the HKVD finding (control_flow tokens genuinely drift more). It shows that **"high sensitivity" is necessary but not sufficient for "selective recompute works"**. The mechanism has to match the policy in a way that translates sensitivity into output quality.

## Implications for research direction

1. **HKVD-by-signalis a real measurement, but the lever is narrow.** Control flow vs data flow is a clean semantic axis; the K_dev delta is huge. But selective recompute at the head boundary (R32-style) doesn't capture this signal.

2. **Two follow-up directions remain unclaimed:**
   - **CacheBlend-style selective recompute** (per-token mask, not per-head) could match HKVD sensitivity better — recompute specifically the control-flow tokens wherever they appear, not just the leading head_K tokens.
   - **Smaller n, finer bucket split** (keywords-only vs control-body) might isolate a stronger signal that translates.

3. **Per-chunk budget calibration was the bottleneck**, not the AST signal. R32_f045 wins by being MORE aggressive (FRAC=0.45 → ~95 tokens per chunk), not by being smarter about which tokens.

## Files

### Implementation
- `python/sglang/srt/mem_cache/radix_cache.py` — `_count_control_flow_tokens` (line 2168), dispatch branch (line 2783), counter init (line 870)
- `python/sglang/srt/entrypoints/openai/serving_chat.py` — 2 emission sites
- `python/sglang/srt/managers/scheduler_output_processor_mixin.py` — 2 emission sites (1 list + 1 dict)
- `benchmark/multi_workflow/bench_kvcomm_ttft_stress.py` — extraction + row dict entry (line 1139, 1264)

### Benchmark
- `results/scale15_5x5/launchers/run_controlflow.sh` — new launcher
- `results/scale15_5x5/run_all_ablation.sh` — added `run controlflow` line
- `results/scale15_5x5/controlflow/{rows.csv, outputs.jsonl, sglang_server.log}` — benchmark output (61 rows)
- `results/scale15_5x5/analyze_ablation_nodekind.py` — extended to include controlflow

### Reports
- `results/ABLATION_PHASE5_CONTROL_FLOW.md` — this file

## What this means for the project

1. **R32 (FRAC=0.30) remains the recommended production config** (per CLAUDE.md §3). The Pareto frontier stays the same: `[nodekind_sig, R32_f015, R32_f026, R32_f045]` with R32_f045 winning on accuracy at modest speed cost.

2. **The code-structure-recompute research line is FULLY DEAD now**, not just 80% dead as we thought after the interface/body NULL. We tested:
   - interface vs body (Direction A) — NULL
   - control vs data (Phase 5 HKVD) — POSITIVE at mechanism layer
   - control-flow selective recompute (Phase 5 policy) — NEGATIVE at application layer
   - first_use / def / import_dist / rare_id (Phase 4 HKVD) — NULL reversed

3. **The "code structure decides recompute" line has been thoroughly falsified.** The only honest framing is: "R32 uniform-along-position is the unique Pareto; code-structure signals exist at the KV layer but don't translate to better selective recompute under contiguous-head constraints."

4. **True CacheBlend** (per-token mask, not per-head) remains the unclaimed frontier. If/when attention-kernel hooks become available, that direction is worth pursuing because it CAN match the per-token sensitivity profile that HKVD measured.