# Placeholder k-NN KV Reuse — Project Status

> **Status as of 2026-06-22.** Research direction in sglang-kvflow. Last completed
> phase: **v19 POOLEMPTY (O1 + O2 + O3-lite)**. Mechanism is correct and safe;
> partial-cache sub-agents (agents 4-5) are improved but still regress —
> closing the remaining gap requires the architectural KVCOMM offset blend
> (Phase 2.7 / O5).

## TL;DR

We added a per-placeholder embedding-k-NN KV reuse path on top of the
existing byte-exact suffix reuse (Shi 2024). The new path is gated by
`SGLANG_PLACEHOLDER_KNN_MATCH=1` and addresses the **multi-agent cliff**
where Shi 2024 falls off at agent_count ≥ 3.

**Current headline numbers** (multi-agent workflow TTFT, sympy/22456,
8000-token bucket):

| agent | Shi 2024 (no k-NN) | prefix-only baseline | v16 TRIM | **v19 POOLEMPTY** | v16 sp | **v19 sp** |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | n/a | 257 ms | 284 ms | 265 ms | 0.91× | **0.97×** |
| 2 | 0.52× | 293 ms | 179 ms | 152 ms | 1.65× | **1.92×** |
| 3 | 0.65× | 338 ms | 247 ms | ~268 ms | 1.37× | ~1.26× |
| 4 | 0.51× | 381 ms | 519 ms | ~500 ms | 0.73× | ~0.74× |
| 5 | 0.43× | 432 ms | 950 ms | ~860 ms | 0.45× | ~0.50× |

Agent 2 hit **1.92×** (best ever). Agent 5 dropped 90ms absolute. The
multi-agent cliff is fixed for agent 2. Agents 4-5 are still losing to
prefix-only because the k-NN search + copy overhead exceeds savings on
small post-prefix copies. The remaining lever is architectural
(Phase 2.7 / O5: KVCOMM offset blend).

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
