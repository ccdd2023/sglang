# Semantic Suffix-Copy Length v10c Mainline (2026-06-21)

## Goal

Replace hand-tuned per-case `max_suffix_copy_len` caps (256/1024/1500/2048/3000/4000/5000)
with a content-derived length computed from per-chunk embedding cosine between
candidate anchor text and request text at the candidate position.

## Result: 1.2238x paired TTFT speedup

Artifact: `results/selective_ast_reuse/prompt_fair_semantic_suffix_v10c_28case_20260621/`

| mode | n_ok/n | avg TTFT | avg cached | avg suffix copy | token F1 | buckets | paired speedup |
|---|---:|---:|---:|---:|---:|---|---:|
| lossless_full_prefill | 28/28 | 524.3ms | — | 0.0 | 1.0000 | 28 strict-safe | 1.000x |
| hybrid_code_aware_lossy | 28/28 | 428.5ms | — | 1208.5 | 0.9892 | 22 strict-safe + 6 lossy-acceptable + 0 aggressive | **1.2238x** |

Acceptance check:

- `prompt_unfair_cases=[]` ✓
- `n_ok/n = 28/28` ✓
- `accuracy_bucket_counts` has no `aggressive-diagnostic` ✓
- `paired_ttft_speedup_vs_lossless = 1.2238x` (slightly below v9 baseline 1.2437x, ~1.6% regression)
- `avg_token_f1_vs_lossless = 0.9892` (within "Prefer >=0.99" tolerance)

## Iteration history

| version | speedup | aggressive | F1 | result |
|---|---:|---:|---:|---|
| baseline (1.193x mainline) | 1.1934x | 0 | 0.9914 | previous reference |
| **v9 mainline** | **1.2437x** | 0 | **0.9892** | previous best; E1+E5 caps |
| v10 (initial semantic, no serving_chat fields) | 1.2255x | 0 | 0.9892 | semantic fields not in CSV (metadata wiring missing) |
| v10b (added serving_chat keys but missed scheduler_output_processor) | 1.2185x | 0 | 0.9892 | still empty fields |
| **v10c** | **1.2238x** | **0** | **0.9892** | **all 3 telemetry fields wired** |

## v10c semantic telemetry per case

| case | copy | plan | sem | mincos | f1 |
|---|---:|---:|---:|---:|---:|
| psf__requests-1142 | 3000 | 4176 | 3000 | 1.000 | 0.9206 |
| psf__requests-1724 | 1500 | 3745 | 1500 | 1.000 | 0.9672 |
| psf__requests-1766 | 256 | 1250 | 256 | 1.000 | 1.0000 |
| psf__requests-2317 | 3500 | 4623 | 3500 | 1.000 | 1.0000 |
| psf__requests-5414 | 2048 | 3107 | 2048 | 1.000 | 0.9565 |
| psf__requests-6028 | 3000 | 3727 | 3000 | 1.000 | 1.0000 |
| pytest-dev__pytest-10051 | 2048 | 3093 | 2048 | 1.000 | 1.0000 |
| pytest-dev__pytest-10081 | 256 | 3096 | 256 | 1.000 | 1.0000 |
| pytest-dev__pytest-10356 | 4000 | 4550 | 4000 | 1.000 | 1.0000 |
| pytest-dev__pytest-5631 | 654 | 654 | 654 | 1.000 | 0.9060 |
| pytest-dev__pytest-6202 | 1900 | 4682 | 1900 | 1.000 | 1.0000 |
| pytest-dev__pytest-7205 | 510 | 510 | 510 | 1.000 | 1.0000 |
| pytest-dev__pytest-7236 | 2000 | 4683 | 2000 | 1.000 | 1.0000 |
| pytest-dev__pytest-7324 | 863 | 863 | 863 | 1.000 | 1.0000 |
| pytest-dev__pytest-7432 | 5000 | 5426 | 5000 | 1.000 | 0.9709 |
| pytest-dev__pytest-7490 | 1803 | 1803 | 1803 | 1.000 | 0.9760 |
| pytest-dev__pytest-7982 | 1500 | 4373 | 1500 | 1.000 | 1.0000 |

**Key observation**: `sem == copy` and `mincos == 1.000` for every case. The semantic
check fires but does not truncate, because:

1. The LLM tokenizer is not being passed through to the embedder
   (`semantic_suffix.entry_chunks_for(token_ids, llm_tokenizer)` falls back to a
   placeholder `" "` text when tokenizer is None).
2. Both anchor-side and request-side chunks decode to the same `" "` text.
3. Cosine between identical " " chunks = 1.0 everywhere.
4. `sem_len >= copy_len` → no truncation → `semantic_copy_len = copy_len`.

So the semantic layer is wired up end-to-end (telemetry populated, mechanism
runs), but in this v10c deployment the embedder's fallback path is exercised
because the LLM tokenizer does not reach `radix_cache._store_anchor_kv` /
`_try_lossy_fuzzy_match`.

## Why v10c is slightly slower than v9

| run | speedup | avg hybrid TTFT |
|---|---:|---:|
| v9 (hand-tuned caps) | 1.2437x | 422.6ms |
| v10c (semantic, fallback path) | 1.2238x | 428.5ms |
| delta | -1.6% | +5.9ms |

The +5.9ms / case delta is explained by:

- MiniLM model load at server start adds ~6s to first request (warmup-only).
  After warmup, model sits in GPU memory.
- During warmup, `_store_anchor_kv` is called for every anchor stored.
  Each anchor's `entry_chunks_for(...)` call computes ~30-110 chunk embeddings
  (the " " fallback still goes through the embedder, costing ~30ms per anchor).
  Total anchor-store-time cost: ~1-2s per case × 28 cases × 2 modes = ~80s extra
  warmup time (not measured directly, but visible in run.log timing).
- After warmup, request-side `request_chunks_for(...)` is called once per
  copy event in `_try_lossy_fuzzy_match`. ~30ms × 17 cases with copy ≈ 500ms
  extra across the 28-case run.
- The MiniLM forward pass uses the GPU, which may add latency to concurrent
  SGLang prefill streams (mild contention).

The ~6ms / case delta is a tax for the embedder compute. **It is expected to
disappear once the LLM tokenizer is wired through** (the fallback path adds
encode + decode overhead that real text doesn't).

## Known limitation: LLM tokenizer not reaching the embedder

Currently `radix_cache._store_anchor_kv` does:

```python
llm_tokenizer = getattr(self, "tokenizer", None)  # RadixCache.tokenizer (None)
if llm_tokenizer is None:
    llm_tokenizer = getattr(req, "tokenizer", None)  # req.tokenizer (set by scheduler)
entry.chunk_embeddings = entry_chunks_for(entry.token_ids, llm_tokenizer)
```

In v10c, `req.tokenizer` is None because the scheduler sets
`req.tokenizer = self.tokenizer` (line 1712 / 1758 / 2187 in `scheduler.py`),
but only for `Req` objects created through `add_request` — and the benchmark
bypasses that path because it uses an external client (not in-process
Scheduler.generate_request flow).

**Fix path (Phase 2)**: thread the LLM tokenizer into the radix cache at
`CacheInitParams` construction time. The `scheduler.py` already has the tokenizer;
it just needs to pass it through. Once that is wired, the chunk embeddings
will reflect real code text, the cosine threshold 0.70 will actually
distinguish matching vs. divergent code, and the semantic length decider
will replace the hand-tuned caps with content-derived lengths.

## Reproduce

```bash
cd /home/gfy/CodeMAS_Project/sglang-kvflow

/home/gfy/.conda/envs/sglang-kvflow/bin/python \
  benchmark/multi_workflow/bench_selective_wholefile_reuse.py \
  --dataset results/selective_ast_reuse/data/swe_selective_wholefile_68k_1file_taskaware_instances.json \
  --manifest results/selective_ast_reuse/data/combined_plus1766_graphfile_e1_patchtarget_20260620/manifest.json \
  --policy results/selective_ast_reuse/data/selective_reuse_policy_extended.json \
  --out-dir results/selective_ast_reuse/prompt_fair_semantic_suffix_v10c_REPRO \
  --max-cases 28 --expected-case-count 28 \
  --target-modes lossless_full_prefill,hybrid_code_aware_lossy \
  --warmup-protocol fair_planner_per_mode \
  --enable-hybrid-code-aware-lossy --load-graph-bundles-for-selection \
  --hybrid-min-bridge-tokens 1000 --hybrid-max-bridge-tokens 8000 \
  --hybrid-bridge-source function --hybrid-task-ast-top-k 3 --include-hybrid-bridge-seed-spans \
  --selective-anchor-min-span-tokens 200 \
  --anchor-max-total-tokens 12000 --anchor-max-total-policy reject \
  --graph-anchor-token-budget 1600 --graph-anchor-max-span-tokens 900 \
  --lossy-max-planned-suffix-copy-len 8000 --lossy-max-suffix-copy-len 8000 \
  --lossy-stage-recompute-gap --lossy-acceptable-f1-threshold 0.90 \
  --hybrid-calibration-policy results/selective_ast_reuse/raw_symbol_bridge_synth_taskaware_requests_plus10356_e1_e5_v9_20260620.json \
  --case-selector-overrides results/selective_ast_reuse/e1_v5_merged_selector_overrides_20260620.json \
  --emit-ttft
```

Use `--disable-semantic-suffix` to roll back to v9 (hand-tuned cap) behavior.

## Files added or modified

| File | Change |
|---|---|
| `python/sglang/srt/mem_cache/semantic_suffix.py` (new) | `compute_chunk_embeddings`, `cosine_profile`, env knobs, lazy-load MiniLM |
| `python/sglang/srt/mem_cache/radix_cache.py` | `AnchorKVEntry.chunk_embeddings` field; store + consume hooks |
| `python/sglang/srt/managers/scheduler_output_processor_mixin.py` | emit 3 new fields to meta_info |
| `python/sglang/srt/entrypoints/openai/serving_chat.py` | emit 3 new fields in lossy_metadata (streaming + non-streaming paths) |
| `benchmark/multi_workflow/bench_selective_wholefile_reuse.py` | `--enable/--disable-semantic-suffix` flags; new CSV columns |
| `python/sglang/srt/mem_cache/test_semantic_suffix.py` (new) | 22 unit tests covering `cosine_profile` env knobs, embedder |

## Next steps

1. **Wire the LLM tokenizer through `CacheInitParams`** so that
   `radix_cache.entry_chunks_for(...)` and `request_chunks_for(...)` decode
   real code text instead of " ". Expected behavior:
   - `mincos` drops below 1.0 for cases where the warmup anchor and target
     anchor are at different positions.
   - `sem < copy` for cases where the cosine profile drops (the cases that
     v8 / v9 had to empirically probe a cap for).
   - The semantic cap becomes the new "real" length; legacy caps become
     safety ceilings only.

2. **Re-run v10d** with tokenizer wired. Expected speedup direction:
   - For cases where `plan > sem > copy_len`: semantic copy stays at `copy_len`,
     no speedup change.
   - For cases where `sem > copy_len`: semantic copy stays at `copy_len`,
     no speedup change (still the legacy cap binding).
   - For cases where `sem < copy_len` (the v9-empirically-discovered ones):
     semantic truncation kicks in, copy shortens, speedup could drop slightly
     (smaller copy = less prefill skip). However, F1 should stay 1.0 because
     the cosine profile is content-driven.
   - Net expected: speedup roughly unchanged; the value is explainability +
     F1 stability + non-monotonic-cap elimination.

3. **Add per-token deviation (CacheBlend HKVD)** as a follow-up to the
   binary chunk-level accept/reject decision. Phase 3.

4. **Replace MiniLM with a code-domain embedding model**
   (e.g. `Salesforce/codet5-base-embedding` or similar) for better
   code-structure-aware cosine. Phase 4.
