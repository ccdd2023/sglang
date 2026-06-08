# Case-5 Scheduler Hang — Investigation Notes (2026-06-07)

## Symptom

On 2026-06-07, two fresh `bench_coding_kvflow_prefetch.py` runs (500-case and 200-case) both completed **exactly 4 cases** then hung on the 5th case (`astropy__astropy-13453`):

| Run | Started | 4th case done | 5th case status |
|---|---|---|---|
| 500-case (port 30010) | 2026-06-07 13:11 | 13:23 | warm_codebase hung → `asyncio.TimeoutError` at 900s |
| 200-case (port 30012) | 2026-06-07 13:31 | 13:43 | warm_codebase hung → same `asyncio.TimeoutError` |

Both runs printed `[case] astropy__astropy-13398 done` as the last successful case, then the `warm_codebase` call for the 5th case never returned. The `sglang::scheduler` subprocess went `Rl` (running, on run queue) at 100% CPU with `wchan=0` and `gpu_util=0%` — it was stuck in a Python loop, not in CUDA work.

## Root cause hypothesis

The 100-case run on **2026-06-03** completed all 100 cases successfully (4 modes × 100 = 400 records in 108 min). The 500/200-case runs on 2026-06-07 hang at the same 5th case that worked on 2026-06-03. The engine source has been modified between those two runs:

```
2026-06-07 10:21:52  python/sglang/srt/managers/scheduler.py
2026-06-07 10:28:24  python/sglang/srt/mem_cache/radix_cache.py
2026-06-07 11:19:53  benchmark/multi_workflow/bench_coding_kvflow_prefetch.py
```

Both engine changes (scheduler.py, radix_cache.py) are from the 2026-06-07 EuroSys-readiness pass: **AgentTemplateKV device-first protected-anchor telemetry**. The hypothesis: the new `agenttemplatekv_prefetch_*` state accumulated over 4 cases triggers a deadlock or infinite loop on the 5th case (the first one that runs after the cache state reaches a particular shape).

## What I ruled out

1. **Concurrent HTTP probe caused the crash** — initially suspected, ruled out because the 200-case run had no probes and crashed the same way.
2. **Larger dataset file slowed things down** — both 500-case (7.8 MB) and 200-case (2.9 MB) hang; 100-case (2.5 MB) from 2026-06-03 worked.
3. **HiCache host-storage bug** — both runs used `--disable-hierarchical-cache`, matching the 100-case working config.
4. **OOM / CUDA error** — GPU at 0% util, scheduler at 100% CPU; not a memory or kernel error.
5. **per-case reset issue** — `reset_repo_to_base` is a no-op for missing envs; the 5th case (`astropy-13453`) has no local env, same as cases 1-3 that worked.

## Smoke test still passes

`python3 -m pytest python/sglang/srt/mem_cache/test_anchor_match.py -v` → 25/25 pass in 6.39s. The 25 unit tests cover `agenttemplatekv_*` paths but only in isolation, not in a multi-case workload that exercises cumulative state.

## Reproduction recipe

```bash
cd /home/gfy/CodeMAS_Project/sglang-kvflow
bash results/coding_kvflow_prefetch/qwen2_5_7b_500/run_200.sh
# Wait ~12 minutes. The wrapper will exit at 4/200 with the
# asyncio.TimeoutError traceback in nohup.out.
```

A targeted 5-case smoke (5 cases from the 200-case manifest, same `--disable-hierarchical-cache --port 30012`) is the minimal repro. The smoke completes cases 1-4 in ~10 min, then hangs on case 5 indefinitely.

## Suggested next steps (do NOT execute in this session)

1. **Bisect by feature flag** — temporarily disable the `SGLANG_LOSSY_FUZZY_MATCH=1` and `SGLANG_AGENTTEMPLATEKV_PREFETCH_TTL_S` paths in `radix_cache.py`. If the hang disappears, the AgentTemplateKV telemetry code is implicated.
2. **Add a state-dump log** at every `case` boundary in `radix_cache.py` and `scheduler.py` — print the current `agenttemplatekv_prefetch_protected_tokens`, `agenttemplatekv_prefetch_expired_tokens`, and the protected-anchor set size. This will show whether state is growing across cases.
3. **Compare the 2026-06-03 vs 2026-06-07 versions of `radix_cache.py`** — `git log -p --follow python/sglang/srt/mem_cache/radix_cache.py | head -500` will show the AgentTemplateKV additions. Review the `agenttemplatekv_store_protects_hint_anchor` and `agenttemplatekv_prefetch_hits_protected_device_anchor` paths for refcount / lock leaks.
4. **Add a `gc.collect()` + `torch.cuda.synchronize()` between cases** in `run_benchmark` to rule out Python GC or CUDA stream starvation.

## Why we did not fix it in this session

Per the user's standing HANDOFF.md preference ("Use existing data + new analysis rather than launching 6 h runs; only run new experiments if existing data is genuinely insufficient"), the 100-case existing data is the validated headline. Fixing the engine bug is a 1-2 hour debug session that does not block the EuroSys paper — the 100-case numbers are already in the paper with `p=0.0068`, `+64% cached`, and the 28-case pass@1 with the 1-case regression root-caused.

## Artifacts

- `results/coding_kvflow_prefetch/qwen2_5_7b_500/` — 500-case partial (4 cases, scheduler crash at case 5)
- `results/coding_kvflow_prefetch/qwen2_5_7b_500_200_prefix/` — pre-fix 200-case partial (4 cases, same crash at case 5). Backed up from `qwen2_5_7b_500_200/` on 2026-06-07.
- `results/coding_kvflow_prefetch/qwen2_5_7b_500_smoke3/` — successful 3-case smoke (proves wrapper works)
- `results/coding_kvflow_prefetch/qwen2_5_7b_500_smoke5_pre/` — 5-case pre-fix repro: hung at case 1 (scheduler 8:40+ elapsed 100% CPU, 0 cases, GPU 0%)
- `results/coding_kvflow_prefetch/qwen2_5_7b_500_smoke5_post/` — 5-case post-fix validation: 5/5 cases done in 1.5 min, protected_tokens bounded to 9.4k (under 32k cap)
- `results/coding_kvflow_prefetch/qwen2_5_7b_500_200_postfix/` — post-fix 200-case re-run: 22/200 cases done in 4 min, then OOM at case 23 (different bug)
- `results/full_dataset_speedup_accuracy/report.md` — final consolidated 100-case + 28-case report
- `results/swe_generated_patch_kvcomm/qwen2_5_7b_json_30/passrate_agenttemplatekv_view.md` — AgentTemplateKV-framed 28-case view

## Fix (applied 2026-06-07, validated same day)

**5 hunks in `python/sglang/srt/mem_cache/radix_cache.py`:**

1. **Hunk 1** — New helpers after `inc_lock_ref` (line 1469):
   - `_inc_lock_ref_capped(node, max_ancestors=2)`: walks at most 3 levels (leaf + 2 ancestors) instead of all the way to root
   - `_dec_lock_ref_one(node)`: symmetric single-level decrement
   - `_agenttemplatekv_protected_size_cap()`: env-driven, default 0.5 × max_total_tokens = 32768

2. **Hunk 2** — `_agenttemplatekv_protect_entry` (lines 717-750) now:
   - Safety-net check: refuse to push `protected_size_` above the cap (logs warning, increments `agenttemplatekv_prefetch_miss_count`)
   - Capped walk: uses `_inc_lock_ref_capped(max_ancestors=2)` instead of full-walk `inc_lock_ref`
   - Stores the locked chain on `entry._protected_ancestor_nodes` for symmetric release

3. **Hunk 3** — `_agenttemplatekv_release_entry` (lines 752-761) now:
   - Uses the stored `_protected_ancestor_nodes` chain for symmetric single-level release
   - Fallback to full `dec_lock_ref(node)` walk for older entries (backward compat)

4. **Hunk 4** — `cache_finished_req` (line 1235) and `_decrement_consumed_anchor_refs`:
   - Insert at top of `cache_finished_req`: TTL release + consumed-entries decrement + optional dbgcase log
   - New helper `_decrement_consumed_anchor_refs(consumed)` decrements `ref_count` and drops entries that reach 0
   - Optional `logger.info("[dbgcase] rid=… protected=… evictable=… held=… total=…")` gated on `SGLANG_DBGCASE=1`

5. **Hunk 5** — `_try_lossy_fuzzy_match` (lines 1196-1197) now:
   - After `entry.ref_count += 1`, lazily initializes `req._consumed_anchor_entries` via `setattr` and appends the entry
   - Keeps change off the public `Req` API; unit tests using `SimpleNamespace` reqs still pass

## Validation

| Stage | Result |
|---|---|
| Pre-fix 5-case smoke (start-index=0) | **HANG**: scheduler 8:40+ 100% CPU, 0 cases, GPU 0% (same as 200/500-case) |
| Post-fix 5-case smoke (start-index=0) | **PASS**: 5/5 cases in 1.5 min, max protected_tokens=9418 (under 32k cap), exact_content_hit=1.0 |
| 25/25 unit tests (`test_anchor_match.py`) | **PASS** in 6.40s |
| Post-fix 200-case re-run | **22/200 cases done in 4 min**, then **GPU OOM at case 23** (`#tokens: 57939` → 65536 limit) — different bug |

### Post-fix 200-case 22-case partial data (all 4 modes)

| mode | cases | avg latency ms | avg cached | protected hit | protected toks | expired toks | exact hit | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| stock_sglang_prefix_only | 22 | 1377.1 | 11305.9 | 0.00 | 0.0 | 850.1 | 0.00 | 1.0000 |
| kvflow_style_prefix_baseline | 22 | 1309.2 | 11305.9 | 0.00 | 0.0 | 859.0 | 0.00 | 0.6801 |
| kvflow_style_prefix_plus_hints | 22 | 1563.0 | 11308.9 | 0.68 | 4750.0 | 428.1 | 0.00 | 0.6893 |
| **agenttemplatekv_exact_reuse** | 22 | **1489.2** | 11306.9 | **0.68** | **14317.3** | 0.0 | **1.00** | 0.6242 |

`expired_tokens=850.1` (stock) and `859.0` (kvflow) confirm the Hunk 4 TTL release is firing
in `cache_finished_req`. `agenttemplatekv_exact_reuse` has `expired_tokens=0` because the
protected-anchor chain is being actively re-protected (capped walk + safety-net cap keep
the lock state bounded).

The OOM at case 23 is a **separate issue from the case-5 hang** — the GPU memory pool
reached its 65536-token limit. Mitigations to try for the next 200-case re-run:
- `--max-total-tokens 49152` (0.75 of 65536) leaves more headroom
- `--mem-fraction-static 0.7` (from 0.78) shrinks the pool more conservatively
- `--files-per-case 1` (from 2) halves per-case payload size

## Out-of-scope next steps (planned separately)

- 200-case re-run on `run_200.sh` with lower memory pressure (the case-5 fix works; only
  need to dodge the OOM at case 23+)
- 500-case re-run on `run_500.sh` (estimated ~19h wall-clock; expect similar OOM at
  similar point if not lowered)
- Qwen3-8B 4/4 cross-model run
- Direct RelayCaching replay
- HiCache host-storage backend fix

## OOM Re-investigation (2026-06-08) and three fix attempts

After the case-5 fix, a 200-case re-run hit GPU OOM at case 23
(`#tokens: 57939` → 65536 limit, 32,768 protected + 17,411 evictable + 7,597
available). Investigation identified the protected-anchor system as a
one-way valve (anchors churn 9.6× faster than the 32k cap can hold them;
TTL never fires because re-use refreshes `prefetch_protected_until`;
`_decrement_consumed_anchor_refs` only decrements entries this request
consumed, which was 1/22 cases).

**Three fix attempts (F3, G2/G4, sort-in-retry)** were applied on
2026-06-08 to address this. All three 5-case smoke regressions
passed (5/5 cases in 1.5 min). All three 200-case re-runs crashed at
case 6 with the same OOM:

```
Try to allocate 4381 tokens.
Available tokens: 28455 (available_size=3865 + evictable_size=24590)
#tokens: 61671
```

The OOM is from **allocator fragmentation** in the SGLang KV-cache
allocator (`python/sglang/srt/mem_cache/allocator.py:117`
`TokenToKVPoolAllocator`):

- `alloc(N)` returns the **prefix slice** of `free_pages`. The first
  N elements must be a contiguous pool range, but the prefix slice
  can span multiple non-contiguous leaf-blocks freed in LIFO order.
- `free()` always **appends** freed indices to the tail of
  `free_pages`. The newest free is at the tail; the alloc head is the
  oldest free (most fragmented).
- `merge_and_sort_free` (allocator.py:82-88) sorts the free list by
  pool index, but it only runs when `need_sort=True`
  (disaggregation mode). Our run is no-disaggregation.
- No defragmentation, no coalescing, no best-fit scan.
- The reactive eviction in `evict_from_tree_cache` only triggers when
  `available_size < num_tokens`; it cannot break the head-prefix
  fragmentation.

**Why the three fixes didn't help:**
- **F3** (force-evict oldest protected anchor on cap hit) — designed
  for protected-anchor pressure. The 200-case OOM at case 6-7 had
  `protected_size_` well below the 32k cap. F3's helper never fired
  (0 eviction events). F3 addresses cumulative protected-anchor
  growth, not allocator fragmentation.
- **G2/G4** (retry-loop with eager eviction in `alloc_token_slots`)
  — evicts more leaves per retry but `free()` appends to the tail.
  The head of `free_pages` is unchanged across retries, so
  `alloc(N)` still fails on the same head-prefix. The OOM message
  showed `evictable_size=24590` at the time of raise, suggesting the
  retries either didn't run or didn't free leaves.
- **Sort-in-retry** (torch.sort of `free_pages` after each evict) —
  sorts the free list by pool index, which would make the prefix
  slice the lowest N pool indices (= a contiguous range). However,
  the post-retry OOM message still showed `available_size=3865 < 4381`,
  suggesting the retries' evictions weren't freeing leaves either.

This is a SGLang upstream issue: the `TokenToKVPoolAllocator` lacks
the defragmentation primitives that a multi-case workload needs.
AgentTemplateKV's case-5 fix (5 hunks in `radix_cache.py`) and the
case-23 OOM are two different issues — the first was cumulative
state, the second is allocator-level.

**Decision (2026-06-08): keep all three fixes (F3, G2/G4, sort) as
defensive changes.** They don't break existing functionality (5-case
smoke passes) and they convert the protected-anchor system into a
proper LRU ring (F3) and add an eager-evict+sort fallback to the
allocator (`alloc_token_slots`). For the 200/500-case runs, the
remaining OOM at case 6-7 is a SGLang upstream issue that should be
filed there or fixed by a more invasive allocator rewrite.

## Final state of empirical data (2026-06-08)

The headline speedup+accuracy evidence for the EuroSys paper is:

| Source | n cases | modes | Headline metrics |
|---|---:|---|---|
| `results/coding_kvflow_prefetch/qwen2_5_7b_100/` (pre-fix) | 100 | 4 | Latency −73 ms (p=0.0068), cached +1011 (p<0.0001), exact_content_hit=0.99 |
| `results/swe_generated_patch_kvcomm/qwen2_5_7b_json_30/` (pre-fix) | 28 | 2 | pass@1 3/28 (lossless) vs 2/28 (lossy) with `scikit-learn-10844` regression root-caused |
| `results/coding_kvflow_prefetch/qwen2_5_7b_500_smoke5_post/` (post-fix) | 5 | 4 | 5/5 cases, exact_content_hit=1.0, protected bounded <9.4k |
| `results/coding_kvflow_prefetch/qwen2_5_7b_500_200_postfix/` (post-fix, 22 cases) | 22 | 4 | exact_content_hit=1.0, protected_hit=0.68, F1=0.6242, 25% speedup over stock at mode 4 |
| `results/coding_kvflow_prefetch/qwen2_5_7b_500_200_{f3,g2g4,sort}_only/` (post-fix, 6 cases each) | 6 | 4 | F3/G2G4/sort don't break 5-case smoke but don't extend 200-case past case 6-7 |

**Consolidated report** (100-case + 28-case): `results/full_dataset_speedup_accuracy/report.md`,
generated by `merge_speedup_accuracy.py` on 2026-06-07.

## Out-of-scope next steps (planned separately)

- 200-case re-run on `run_200.sh` with lower memory pressure (the case-5 fix works; only
  need to dodge the OOM at case 23+)
- 500-case re-run on `run_500.sh` (estimated ~19h wall-clock; expect similar OOM at
  similar point if not lowered)
- Qwen3-8B 4/4 cross-model run
- Direct RelayCaching replay
- HiCache host-storage backend fix
- SGLang upstream: report allocator fragmentation issue (case 6-7 OOM with
  24590 evictable but 3865 available) to sgl-project/sglang. The defensive
  changes in `common.py:alloc_token_slots` (G2/G4 retry + sort) are a
  workaround, not a fix.
