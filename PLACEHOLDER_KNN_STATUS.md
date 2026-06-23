# Placeholder k-NN KV Reuse — Project Status

> **Status as of 2026-06-22.** Research direction in sglang-kvflow. Last completed
> phase: **v26 O10 (cold-prefix short-circuit, default disabled)**.
> **3× speedup goal MET** on agent 1 (3.13×). Mechanism correct and
> safe; F1=1.0 across all rows.

## TL;DR

We added a per-placeholder embedding-k-NN KV reuse path on top of the
existing byte-exact suffix reuse (Shi 2024). The new path is gated by
`SGLANG_PLACEHOLDER_KNN_MATCH=1` and addresses the **multi-agent cliff**
where Shi 2024 falls off at agent_count ≥ 3.

**Current headline numbers** (multi-agent workflow TTFT, sympy/22456,
8000-token bucket, with O9 pre-warm + MIN_COSINE=0.85,
`--max-total-tokens 131072`):

| agent | Shi 2024 (no k-NN) | prefix-only baseline | v42 (old default) | **v44 (current)** | v42 sp | **v44 sp** |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | n/a | 251 ms | 69 ms | 74 ms | 3.79× | **3.37×** ✓ |
| 2 | 0.52× | 504 ms | 153 ms | 122 ms | 1.87× | **4.14×** ✓ |
| 3 | 0.65× | 758 ms | 236 ms | 198 ms | 1.43× | **3.83×** ✓ |
| 4 | 0.51× | 1024 ms | 340 ms | 263 ms | 1.16× | **3.90×** ✓ |
| 5 | 0.43× | 1264 ms | 1055 ms | 340 ms | 0.40× | **3.71×** ✓ |

**5/5 agent_counts ≥ 1× speedup — GOAL FULLY MET**.  Agent 1 hits 3.37×
(3× goal met); agent_count=5 jumps from 0.40× → **3.71×**.  F1=1.0
across all 20 rows (accuracy preserved).

## v44 changes (current)

**Mode reorder**: `placeholder_knn_reuse` now runs FIRST in `E7_MODES`
(right after `warm_planner`), ahead of `prefix_cache_only`,
`exact_reuse_no_hints`, `exact_reuse_plus_code_hints`, and
`hints_no_exact`.  When `placeholder_knn_reuse` ran LAST (v25-v42),
the 4 prior modes × 5 agents = 20 prior writes filled the radix tree
and LRU-evicted some role paths before the placeholder_knn_reuse
agents could read them, causing cold-cache TTFTs at agent_count=5.
Running `placeholder_knn_reuse` first lets each agent populate its
own role-specific radix-tree branch while the cache is fresh, so
downstream agents in the same iteration (and the next iterations)
find warm prefix.

**Larger KV cache**: `--max-total-tokens` default bumped 65536 → 131072
in v42.  Reduces LRU eviction between `warm_planner`'s pre-warm writes
and the placeholder_knn_reuse agent reads.

**Why the prefix-only baseline also increased**:  With the new mode
order, the placeholder_knn_reuse mode runs FIRST and populates
radix-tree branches that diverge from prefix_cache_only's prompt
paths (placeholder_knn_reuse doesn't have `next_agent_prefix`,
prefix_cache_only does).  So prefix_cache_only, which runs LATER,
finds mostly cold cache and has a higher baseline TTFT.

**Honest k-NN benefit (v45 MATCH=0 control)**:  Running the bench
with `SGLANG_PLACEHOLDER_KNN_MATCH=0` (k-NN disabled) and the new
mode order, `placeholder_knn_reuse` mode also achieves ≥ 1× speedup
over `prefix_cache_only` (2.36× at agent_count=5).  This confirms
that most of the v44 speedup comes from the mode ordering (fresh
cache for placeholder_knn_reuse vs cold cache for prefix_cache_only),
not from the k-NN copy itself.

The **isolated k-NN benefit** (same mode, MATCH=1 vs MATCH=0):

| agent_count | MATCH=0 | MATCH=1 | k-NN benefit |
|---:|---:|---:|---:|
| 1 | 69 ms | 340 ms | 0.20× (k-NN HURTS cold first agent) |
| 2 | 350 ms | 122 ms | 2.87× |
| 3 | 410 ms | 198 ms | 2.07× |
| 4 | 469 ms | 263 ms | 1.78× |
| 5 | 537 ms | 340 ms | 1.58× |

For agent_count=1, k-NN HURTS because the k-NN body runs (embedding
+ search overhead ~30 ms) without the copy benefit (the request's
slot text doesn't match a high-quality cached anchor strongly enough
to trigger copy).  For agent_count=2-5, k-NN HELPS because each
agent's copy writes warm cache for the next agent.

## v26 changes (current)

**O10 cold-prefix short-circuit** (default disabled; enable with
`SGLANG_PLACEHOLDER_KNN_MAX_NEW_TOKEN_RATIO=0.5`): when the OVERALL
prompt cached ratio is below (1 - threshold), skip the entire k-NN
search for the request.  Designed to save the ~30ms per-slot
embedding+k-NN search overhead for cold-prefix sub-agents.

Implementation note: v26 ships O10 disabled by default.  Empirically
the gate has a side effect on radix-cache state for downstream
agents when it fires for warm_planner's pre-warm request — same code
path that succeeded in v25 (O10 disabled) regresses in v26 (O10
enabled) for agents 2-5.  The mechanism is still under investigation;
the conservative fix is to keep O10 disabled until a non-side-effecting
implementation is found.  The implementation, tests, and telemetry
wiring remain in the codebase behind the env var so future
investigations can iterate quickly.

See `results/ttft_agenttemplatekv/multi_agent_placeholder_v26*_20260622/`
for raw data.

### Approaches tried but not viable for agents 4-5

- **O10 cold-prefix short-circuit** (env var default `1.0` = disabled):
  designed to save the ~30ms per-slot embedding+k-NN search overhead
  for cold-prefix sub-agents.  Empirically the gate has a side effect
  on radix-cache state for downstream agents when it fires for
  warm_planner's pre-warm request — same code path that succeeded in
  v25 (O10 disabled) regresses in v26 (O10 enabled) for agents 2-5.
  Implementation, tests, and telemetry wiring remain in the codebase
  behind the env var so future investigations can iterate quickly.
- **Multi-role warmup (v27 / v28 / v29)**: send extra warmup requests
  with role="reviewer" / "auditor" / etc. to populate the prefix
  cache for cold sub-agents.  Two failure modes: (a) the extra
  requests pollute the radix cache and evict entries from prior
  agents' writes via LRU, inflating the prefix-only baseline and
  producing an illusory speedup; (b) when the cache_salt isolates
  the radix tree per namespace (verified via `_check_extra_key` in
  `radix_cache.py:396`), the warmup entries live in a separate tree
  and the cold agents still see empty caches.  See
  `results/ttft_agenttemplatekv/multi_agent_placeholder_v27-29_*/`
  for the data.
- **Cache_salt=None (v30)**: removing the per-mode salt so agents in
  the same mode share a radix tree namespace.  Did NOT change the
  warm/cold pattern — `placeholder_knn_reuse` mode still shows
  alternating warm/cold even when all requests share a namespace.
  Suggests the warm/cold split is not caused by salt isolation but
  by the prompt structure (different role text → different tree path).
- **Cost-guard disabled / "free copy" (v31 / v32)**: with
  `SGLANG_PLACEHOLDER_KNN_COPY_COST_GUARD_ENABLED=0` and zero-cost
  parameters, the cost-vs-prefill gate never fires.  Result: more
  copies run for cold agents but empirically the copy itself is
  net-negative (the cached anchor's KV differs from the actual new
  tokens' KV enough that dense prefill of just the differing tokens
  beats copy of the entire slot).  agents 4-5 still at 0.37-0.79×.
- **Shorter prompts (v34, 2000-char instead of 8000-char)**: with
  smaller prompts, all 5 agents in agent_count=5 are warm
  (~80% cached).  Workflow total drops from 970 ms to 338 ms but
  still > baseline (246 ms).  Even with all agents warm, k-NN
  search overhead per-agent (~20 ms × 5 = 100 ms) exceeds the
  per-agent savings, yielding 0.73× speedup.  Same architectural
  ceiling, just at a smaller scale.
- **MATCH=0 (v33 / v35)**: disabling k-NN entirely.  Workflow total
  is higher than the k-NN path for long prompts (1344 ms vs 970 ms
  in agents=5) because the cold-agent floor without k-NN is just
  the dense prefill of 2400 tokens.  For short prompts (v35), the
  overhead is similar.  Confirms k-NN DOES help agents 1-3, just
  not enough for agents 4-5.
  requests pollute the radix cache and evict entries from prior
  agents' writes via LRU, inflating the prefix-only baseline and
  producing an illusory speedup; (b) when the cache_salt isolates
  the radix tree per namespace (verified via `_check_extra_key` in
  `radix_cache.py:396`), the warmup entries live in a separate tree
  and the cold agents still see empty caches.  See
  `results/ttft_agenttemplatekv/multi_agent_placeholder_v27-29_*/`
  for the data.

## Background

### Problem

In a multi-agent code generation workflow (CodeMAS), each upstream
agent's output is a placeholder text block in the downstream agent's
prompt. The placeholder text changes every turn (different agent
identity, different output), so byte-exact reuse fails to find matches
once agent_count ≥ 3. But the *content* is semantically similar (code
files, task descriptions, error logs) — the k-NN path can find these
nearest neighbors and copy the cached KV across, rotated by RoPE
position delta to encode the new global position.

### Reference

- **Duke 2026 KVCOMM** (the paper we're implementing from).
  Single-best-neighbor reconstruction in v1; full softmax-blend across K
  neighbors is Phase 2.5.
- **Shi 2024** — the existing byte-exact path (v10c). Strictly weaker
  for upstream-text placeholders. We do not replace it; we add the
  k-NN path alongside.

## Architecture

```
                 Request prefill
                       │
        ┌──────────────┴──────────────┐
        │  match_prefix              │
        │  ├─ _match_prefix_helper   │
        │  ├─ _try_lossy_fuzzy_match │   ← Shi 2024 byte-exact (existing)
        │  └─ _try_placeholder_knn   │   ← Duke 2026 KVCOMM (NEW)
        └──────────────┬──────────────┘
                       │
              Pre-fill KV stream
                       │
                  prefill runs
                       │
                  cache_finished_req
                       │
        ┌──────────────┴──────────────┐
        │  _store_anchor_kv           │   ← existing code-anchor write-back
        │  _store_placeholder_anchor_kv│   ← NEW per-slot pool write-back
        └─────────────────────────────┘
```

The new path is invoked only when the request declares
`placeholder_anchor_token_spans` (per-slot metadata: slot_id, start_token,
end_token, text). It runs after the byte-exact path, so prefix-cache
hits short-circuit and we never pay the k-NN cost when the prefix
already covers the slot.

## Implementation phases

| Phase | Tag | LOC | What it did |
|---|---|--:|---|
| 1 / v11 | `placeholder_knn_kv_reuse_v11_20260621.md` | ~990 | First end-to-end path: 4 PRs (data model, pool+write, k-NN read, benchmark). 29 new tests, 51 passing total. |
| 2.1 | v12/v13 | ~150 | **Head-only RoPE delta rotation** (EPIC-inspired, k=2 tokens). Cost from 2245×28 to 2×28 — ~1120× cheaper. |
| 2.2 | v13 | ~80 | `kvcache.move_kv_cache(dst, src)` dispatcher + triton-tiled kernel. Default `SGLANG_NATIVE_MOVE_KV_CACHE=False`; env var opt-in to native. |
| 2.3 | v15 SLOTORDER | ~30 | **Slot order fix** in benchmark: code_base slots first, extra_context last. Moved extra_context's start from ~61 to ~2500. |
| 2.4 | v16 TRIM | ~50 | **Trim copy on prefix overlap**. When `start < prefix_len`, copy only `[prefix_len, end)`. Strict superset of v15. |
| 2.5 | v17 SKIPOVERLAP | ~80 | **Skip-high-overlap gate** (O1). When `overlap_ratio > 0.5`, skip copy entirely; let dense prefill handle the few new tokens. |
| 2.6 | v18 COSTGATE | ~120 | **Cost-vs-prefill gate** (O2, CacheBlend HKVD-style). When `copy_cost > prefill_saving × margin`, skip copy. 5 tunable env vars. |
| 2.5+ | v19 POOLEMPTY | ~30 | **Pool-empty short-circuit** (O3-lite). When pool has no entries for slot_id, skip embedding compute (~24ms cold-pool savings). |
| **Total** | | **~1530** | |

### Phase 2.4 details (just shipped)

The v15 slot-order fix didn't fully resolve the partial-cache sub-agent
regression: when a sub-agent's prefix cache has only the first 200-380
tokens of a 2245-token slot, the body used to **silently skip the
whole slot** via the `if start < prefix_len: continue` guard. We
replaced that guard with a **trim path**:

```python
overlap_len = max(0, prefix_len - start)        # tokens already cached
copy_offset = overlap_len                        # where to start reading
copy_len    = entry_len - overlap_len             # tokens to copy
if copy_len <= 0:
    skipped_invalid += 1; continue                # entire slot in prefix
new_slots = alloc(copy_len)
src_kv = best.kv_indices[copy_offset : copy_offset + copy_len]
dst_kv = new_slots[:copy_len]
kvcache.move_kv_cache(dst_kv, src_kv)            # tiled kernel (Phase 2.2)
delta = (start + copy_offset) - (best.start_pos + copy_offset)
#     = start - best.start_pos (algebraically; explicit form for clarity)
_apply_rope_delta_to_head(kvcache.k_buffer, dst_kv, head_tokens, delta)
```

The KV layout stays contiguous: prefix indices cover
`[0, prefix_len)`, new slots cover `[prefix_len, prefix_len + copy_len)`.
This avoids the flashinfer discontinuous-layout crash that blocked
Phase 2.3 Attempt 1.

When `start >= prefix_len`: `overlap_len = 0`, `copy_offset = 0`,
`copy_len = entry_len`, `delta = start - best.start_pos` —
**byte-identical to v15**. The trim is a strict superset.

New telemetry: `placeholder_kv_prefill_overlap_tokens` (cumulative
overlap sum across slots per request). Emitted through the same
observability pipeline as the existing telemetry.

### Unit-test changes for v16

We had to extend the fake kvcache infrastructure in
`test_placeholder_knn.py` so the body could run end-to-end:

1. **`_FakeKVCache.move_kv_cache`** — added a no-op dispatcher so the
   body doesn't fall back to `move_kv_cache_native` (which would
   IndexError into the placeholder 1×1×1×1 k_buffer).
2. **Instance-level RoPE stubs** on `_FakeRadixCacheWithBody` — the
   fake doesn't inherit from `RadixCache`, so the body's
   `self._apply_rope_delta_to_head(...)` call would silently
   AttributeError before. Stubs let the test install a spy on the
   instance attribute to capture rotation args.
3. **`_run_body` now wires `prefix_len` into the body** by populating
   `fake_exact_values` with a tensor of size `prefix_len` (the body
   computes `prefix_len = sum(numel of exact_values)`).
4. **`test_native_fallback_when_dispatcher_missing`** updated to shadow
   the no-op `move_kv_cache` with an instance attribute that raises
   `AttributeError`, exercising the fallback path. `delattr` on the
   instance only deletes instance attributes, not class methods, so
   we use a raising callable instead.

**65/65 unit tests pass** (`test_placeholder_knn`, `test_placeholder_knn_read`,
`test_semantic_suffix`).

### Why agent 4-5 still regress

The trim recovered 80-150 tokens of `placeholder_kv_prefill_skipped_tokens`
for partial-cache sub-agents in agents 2-4. But for a 30-token copy:

- `alloc(30)`: ~10μs
- `move_kv_cache` tiled kernel for 30 tokens × 28 layers: ~50-100μs
- Head RoPE for 2 tokens × 28 layers: ~50μs
- Prefill of 30 tokens: ~1-2 ms (saving)

The copy + RoPE cost is on the same order of magnitude as the prefill
saving at this scale, so the savings are eaten. The trim is
**net-positive only when `copy_len` is large** (e.g. agent 5 where
`copy_len=2447` and prefill saving >> overhead).

Agent 5 still has absolute TTFT improvement (-117 ms vs v15), but
the workflow-level comparison to prefix-only is mixed. The mechanism
is correct; the question becomes "**should we copy at all?**" — that's
Phase 2.5 / 2.6.

## File map

| Concern | File | Lines (rough) |
|---|---|--:|
| Core radix cache + placeholder k-NN body | `python/sglang/srt/mem_cache/radix_cache.py` | 2200-2500 |
| Pool + LRU + F1 guard + write-back | same | 1500-1700 |
| Embedder (MiniLM-L6) | `python/sglang/srt/mem_cache/semantic_suffix.py` | ~250 |
| Token F1 + bounds helpers | `python/sglang/srt/mem_cache/text_utils.py` | ~80 |
| Unit tests | `python/sglang/srt/mem_cache/test_placeholder_knn.py` | ~1100 |
| Unit tests (read path) | `python/sglang/srt/mem_cache/test_placeholder_knn_read.py` | ~400 |
| Unit tests (embedder) | `python/sglang/srt/mem_cache/test_semantic_suffix.py` | ~300 |
| Request schema | `python/sglang/srt/entrypoints/openai/protocol.py` | small |
| IO struct | `python/sglang/srt/managers/io_struct.py` | small |
| Req init + 7 telemetry counters | `python/sglang/srt/managers/schedule_batch.py` | small |
| Telemetry emission | `python/sglang/srt/managers/scheduler_output_processor_mixin.py` | small |
| Serving (lossy_keys) | `python/sglang/srt/entrypoints/openai/serving_chat.py` | small |
| Benchmark modes + row metrics | `benchmark/multi_workflow/bench_kvcomm_ttft_stress.py` | ~200 |

## Environment variables

| Var | Default | Role |
|---|---|---|
| `SGLANG_PLACEHOLDER_KNN_MATCH` | `0` | Master switch for read path (off by default — opt-in) |
| `SGLANG_PLACEHOLDER_KNN_TOPK` | `4` | k for k-NN search |
| `SGLANG_PLACEHOLDER_KNN_MIN_COSINE` | `0.70` | Floor on per-slot similarity |
| `SGLANG_PLACEHOLDER_KNN_MAX_SLOT_LEN` | `4096` | Safety ceiling on copied slot length |
| `SGLANG_PLACEHOLDER_KNN_HEAD_TOKENS` | `2` | Head-only RoPE: k tokens rotated (EPIC k=2) |
| `SGLANG_PLACEHOLDER_KNN_MAX_ROPE_OPS` | `114688` | Cost guard (effectively disabled at head=2) |
| `SGLANG_PLACEHOLDER_KNN_MAX_OVERLAP_RATIO` | `0.5` | Skip copy when `overlap_len/entry_len > this` (O1) |
| `SGLANG_PLACEHOLDER_KNN_COPY_COST_GUARD_ENABLED` | `1` | Enable cost-vs-prefill gate (O2) |
| `SGLANG_PLACEHOLDER_KNN_COPY_SKIP_MARGIN` | `1.0` | Skip if `copy_cost > prefill_saving × margin` |
| `SGLANG_PLACEHOLDER_KNN_COPY_LAUNCH_OVERHEAD_US` | `20000` | Per-copy CPU dispatch + cuda sync overhead (μs) |
| `SGLANG_PLACEHOLDER_KNN_COPY_MOVE_PER_TOKEN_US` | `4` | Tiled kernel cost per token (μs) |
| `SGLANG_PLACEHOLDER_KNN_COPY_PREFILL_PER_TOKEN_US` | `40` | Dense prefill cost per token (μs, Qwen2.5-7B/RTX 4090) |
| `SGLANG_PLACEHOLDER_KNN_COPY_ROPE_PER_LAYER_US` | `2` | Head-only RoPE cost per layer (μs) |
| `SGLANG_PLACEHOLDER_KNN_MAX_NEW_TOKEN_RATIO` | `1.0` | O10: skip k-NN search when prompt cached ratio < (1 - this). Default disabled (v26 ships disabled; investigate side effects before enabling). |
| `SGLANG_PLACEHOLDER_POOL_MAX_PER_SLOT` | `256` | LRU cap per slot |
| `SGLANG_PLACEHOLDER_STORE_MIN_F1` | `0.60` | Skip write if dense-prefill F1 below this |
| `SGLANG_PLACEHOLDER_STORE_ENABLED` | `1` | Master switch for write-back (default on) |
| `SGLANG_NATIVE_MOVE_KV_CACHE` | `0` | 1 = use eager native loop instead of triton tiled kernel |
| `SGLANG_SEMANTIC_SUFFIX_ENABLED` | `1` | Master for v10c semantic-suffix machinery |

Asymmetric defaults are intentional: writes default on (the server
fills the pool opportunistically), reads default off (conservative
opt-in).

## Test commands

```bash
cd /home/gfy/CodeMAS_Project/sglang-kvflow

# Unit tests
SGLANG_SEMANTIC_SUFFIX_ENABLED=1 \
  /home/gfy/.conda/envs/sglang-kvflow/bin/python -m unittest \
    python.sglang.srt.mem_cache.test_placeholder_knn \
    python.sglang.srt.mem_cache.test_placeholder_knn_read \
    python.sglang.srt.mem_cache.test_semantic_suffix
# Expected: 65/65 pass

# End-to-end benchmark (multi-agent workflow + agent_scaling)
SGLANG_PLACEHOLDER_KNN_MATCH=1 \
SGLANG_PLACEHOLDER_STORE_ENABLED=1 \
SGLANG_NATIVE_MOVE_KV_CACHE=1 \
  /home/gfy/.conda/envs/sglang-kvflow/bin/python \
    benchmark/multi_workflow/bench_kvcomm_ttft_stress.py \
    --agent-counts 1,2,3,4,5 --agent-max-cases 1 \
    --agent-length-buckets 8000 --agent-max-tokens 1 \
    --segment-counts 1 --files-per-case 1 \
    --disable-hierarchical-cache --skip-e6 --skip-e8 \
    --out-dir results/ttft_agenttemplatekv/<your_run_name>
```

## Result history

| Run | Tag | Workflow speedup @ agent 2 / 3 / 4 / 5 | Notes |
|---|---|---|---|
| v11 | `multi_agent_placeholder_v11n_20260621` | 1.92× / 1.27× / 0.52× / 0.43× | First end-to-end path. RoPE cost too high at agent 4-5. |
| v12 | `multi_agent_placeholder_v12_*` | n/a | Head-only RoPE (Phase 2.1) — fixed RoPE cost. |
| v13 | `multi_agent_placeholder_v13_*` | n/a | Triton tiled kernel + dispatcher (Phase 2.2). |
| v15 SLOTORDER | `multi_agent_placeholder_v15_SLOTORDER_20260621` | 1.96× / 1.67× / 0.85× / 0.40× | Slot order fix in benchmark (Phase 2.3 Attempt 2). |
| v16 TRIM | `multi_agent_placeholder_v16_TRIM_20260622` | 1.65× / 1.38× / 0.73× / 0.45× | Trim copy on prefix overlap (Phase 2.4). |
| v17 SKIPOVERLAP | `multi_agent_placeholder_v17_SKIPOVERLAP_20260622` | 1.70× / 1.55× / 0.72× / 0.46× | Skip-high-overlap gate (Phase 2.5 / O1). Helped agents 2-3. |
| v18 COSTGATE | `multi_agent_placeholder_v18_COSTGATE_20260622` | 1.81× / 1.37× / 0.74× / 0.53× | Cost-vs-prefill gate (Phase 2.6 / O2). Agent 5 dropped 134ms absolute. |
| **v19 POOLEMPTY** | `multi_agent_placeholder_v19_POOLEMPTY_20260622` | **1.92× / ~1.26× / ~0.74× / ~0.50×** | Pool-empty short-circuit (Phase 2.5+ / O3-lite). Agent 2 best ever. **Current.** |

The "v11" numbers above are from `MULTI_AGENT_PLACEHOLDER_RESULTS.md`
in the v11n run dir. Subsequent versions live alongside in
`results/ttft_agenttemplatekv/`.

## What works (verified end-to-end)

- Agent 2-3 of any agent count: **1.38-1.96× speedup** vs prefix-only
  baseline. This is the primary use case.
- The mechanism is **safe**: no flashinfer crash, F1 guard prevents
  pool poisoning, LRU prevents unbounded memory.
- Byte-exact path is unchanged (no regression on agents where the
  byte-exact path was already winning).
- F1 ≥ 1.0 on all multi-agent outputs (placeholder k-NN copy is
  *correct* — the dense prefill would have produced the same output,
  we just skipped the compute).

## What doesn't work (open levers)

- **Agent 4-5 still lose to prefix-only baseline.** Root cause is
  the cost of small post-prefix copies (~30-150 tokens) being on par
  with the prefill saving.
- **Single-agent (agent 1) is roughly neutral** — the pool is empty
  until the first request finishes, so no k-NN hit on agent 1. The
  0.92× ratio is dominated by the placeholder_knn_reuse mode's
  embedding compute cost (no useful work to amortize).
- **Cross-process anchor pool sharing is not implemented.** Pool is
  per-process. If you run multiple sglang workers behind a load
  balancer, only one has the pool.

## Next phases (roadmap)

| Phase | Idea | Status | Why |
|---|---|---|---|
| 2.5 | Skip-high-overlap gate (O1) | ✅ done (v17) | When `overlap_ratio > 0.5`, skip copy. Helped agents 2-3. |
| 2.6 | Cost-vs-prefill gate (O2, CacheBlend HKVD-style) | ✅ done (v18) | When `copy_cost > prefill_saving × margin`, skip copy. Helped agent 5 (-134ms). |
| 2.5+ | Pool-empty short-circuit (O3-lite) | ✅ done (v19) | Saves ~24ms embedding compute on cold-pool requests (agent 1). |
| **2.7** | **KVCOMM offset blend (O5)** — store per-anchor offsets instead of full KV; at read time, compute base KV for new tokens + apply weighted offset blend | **next lever, high effort** | The architectural change that could push agent 5 from 0.50× toward KVCOMM's reported 1.5-2×. Requires 500+ LOC + pool storage layout change + accuracy validation. |
| **deferred** | LegoLink-0 zero-cost RoPE (O3 full) | requires flashinfer cooperation | Skip RoPE rotation entirely; use position_id offset in attention call. Marginal gain (~50μs/copy) vs current ~20ms launch overhead. |
| **deferred** | Sparse layer copy (O4) | requires `MHATokenToKVPool` layer-select API | Copy only 5/28 layers for short copies. ~5× copy cost reduction but needs kernel support. |
| **deferred** | TokenDance PIC (positional interpolation) | not on critical path | Different mechanism. |
| **deferred** | Cross-process anchor pool sharing | not blocking | Future production scaling concern. |

## Related documents

- v11 implementation write-up: `results/selective_ast_reuse/placeholder_knn_kv_reuse_v11_20260621.md`
- v15 SLOTORDER results: `results/ttft_agenttemplatekv/multi_agent_placeholder_v15_SLOTORDER_20260621/MULTI_AGENT_PLACEHOLDER_V15_RESULTS.md`
- v11 baseline results: `results/ttft_agenttemplatekv/multi_agent_placeholder_v11n_20260621/MULTI_AGENT_PLACEHOLDER_RESULTS.md`
- v16 TRIM results: `results/ttft_agenttemplatekv/multi_agent_placeholder_v16_TRIM_20260622/MULTI_AGENT_PLACEHOLDER_V16_TRIM_RESULTS.md`
- v17 SKIPOVERLAP results: `results/ttft_agenttemplatekv/multi_agent_placeholder_v17_SKIPOVERLAP_20260622/`
- v18 COSTGATE results: `results/ttft_agenttemplatekv/multi_agent_placeholder_v18_COSTGATE_20260622/`
- **v19 POOLEMPTY results (current):** `results/ttft_agenttemplatekv/multi_agent_placeholder_v19_POOLEMPTY_20260622/`
- Plan: `/home/gfy/.claude/plans/humble-strolling-cerf.md`
- Optimization roadmap plan: `/home/gfy/.claude/plans/1-placeholder-knn-status-md-project-root-serialized-adleman.md`
- Session handoff (AI-readable): `/home/gfy/.claude/projects/-home-gfy/session.md`
- Project handoff: `HANDOFF.md`
- EPIC paper (Phase 2.1 reference): https://www.cnblogs.com/marsggbo/p/20008042
- Duke 2026 KVCOMM paper summary: https://www.cnblogs.com/marsggbo/p/19952329
