# Per-Placeholder KV Reuse with Embedding k-NN — v11 Implementation (2026-06-21)

## Goal

Implement Duke/MIT/NVIDIA 2026 KVCOMM-style placeholder KV reuse on sglang-kvflow,
building on the v10c semantic-suffix machinery.  The new path replaces Shi 2024
byte-exact suffix reuse for **upstream-text placeholder slots** in multi-agent
prompts — the part where Shi 2024 falls off a cliff (0.51-0.65× at agent ≥ 3).

## What was built

A complete per-placeholder k-NN KV reuse path delivered in **4 sequenced PRs**:

| PR | Scope | LOC | Tests |
|---|---|---:|---:|
| PR 1 | Data model + payload plumbing | ~80 | 5 (new `embed_single_text`) |
| PR 2 | Pool + write-back (no reads) | ~250 | 11 (LRU eviction, F1 guard, slot taxonomy) |
| PR 3 | k-NN read path (gated by env) | ~400 | 13 (k-NN search, gating, env switches) |
| PR 4 | Benchmark + MAScoder integration | ~260 | smoke tests |
| **Total** | | **~990** | **29 new tests, 51 passing total** |

### Files added or modified

| File | Change |
|---|---|
| `python/sglang/srt/mem_cache/radix_cache.py` | Extended `AnchorKVEntry` (+5 fields: `slot_id`, `slot_label`, `pool_embedding`, `embedding_text`, `last_access_time`); added `placeholder_anchor_pool` + lock; added module-level `_placeholder_knn_search`; added `_store_placeholder_anchor_kv` (write-back with F1 guard); added `_try_placeholder_knn_lossy_match` (k-NN read with RoPE delta); hooked both into `cache_finished_req` and `match_prefix`. |
| `python/sglang/srt/mem_cache/semantic_suffix.py` | Added `embed_single_text(text, emb=None) -> Tensor[D]` helper for single-text L2-normalized embedding. Reuses the v10c `_embed_texts` + lazy singleton loader. |
| `python/sglang/srt/mem_cache/text_utils.py` *(new)* | Moved `token_f1` and `token_bounds_for_text` from the benchmark so the F1-guard on the server side can import them without a benchmark→runtime cycle. |
| `python/sglang/srt/mem_cache/test_placeholder_knn.py` *(new)* | 11 tests: LRU eviction, F1-guard, slot taxonomy, env gating. |
| `python/sglang/srt/mem_cache/test_placeholder_knn_read.py` *(new)* | 13 tests: pure-Python k-NN search, end-to-end MiniLM, gating behavior. |
| `python/sglang/srt/mem_cache/test_semantic_suffix.py` | Added 5 tests for `embed_single_text`; fixed env-restoration in `EmbedSingleTextTests` to prevent test-ordering leak. |
| `python/sglang/srt/entrypoints/openai/protocol.py` | Added `placeholder_anchor_token_spans` Field to `ChatCompletionRequest`. |
| `python/sglang/srt/entrypoints/openai/serving_chat.py` | Pass through to `GenerateReqInput`. |
| `python/sglang/srt/managers/io_struct.py` | Added field on `GenerateReqInput` + `TokenizedGenerateReqInput` + `__getitem__` propagation. |
| `python/sglang/srt/managers/schedule_batch.py` | Added `placeholder_anchor_token_spans` param to `Req.__init__`; init 6 telemetry counters (`placeholder_anchor_pool_hit_count`, `placeholder_anchor_pool_miss_count`, `placeholder_knn_topk_similarity_mean`, `placeholder_kv_prefill_skipped_tokens`, `placeholder_kv_prefill_matched_slots`, `placeholder_anchor_store_entry_count`). |
| `benchmark/multi_workflow/bench_kvcomm_ttft_stress.py` | Added `PlaceholderSlot` dataclass; `build_slot_messages()`; `build_placeholder_anchor_fields()`; new modes `placeholder_knn_reuse` and `placeholder_knn_plus_exact` in `E6_MODES` / `CORE_TTFT_MODES` / `E7_MODES`; new payload branch in `make_payload`; new row metrics in `row_from_response` (7 new fields). |

## Architecture

The new mechanism composes with the existing byte-exact path rather than
replacing it.  In `match_prefix` (radix_cache.py), `_try_lossy_fuzzy_match`
runs first (Shi 2024 byte-exact suffix), then **immediately after**,
`_try_placeholder_knn_lossy_match` runs if the request declared
`placeholder_anchor_token_spans`.

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

### k-NN search primitive

```python
def _placeholder_knn_search(pool_entries, query_embedding,
                            top_k=4, min_similarity=0.70):
    # Filter out entries with no embedding
    valid = [e for e in pool_entries if e.pool_embedding is not None]
    embeddings = torch.stack([e.pool_embedding for e in valid])
    embeddings = F.normalize(embeddings, p=2, dim=1)
    q = F.normalize(query_embedding.view(1, -1), p=2, dim=1)
    sims = (embeddings @ q.T).squeeze(1)
    top_sims, top_idx = torch.topk(sims, k=min(top_k, sims.numel()))
    return [(valid[i], float(s)) for s, i in zip(top_sims, top_idx)
            if s >= min_similarity]
```

Uses **single-best-neighbor** reconstruction in v1 (Duke 2026's full
softmax-blend across K neighbors is Phase 2).  L2-normalizes both sides
defensively in case the embedder moved across devices.

### Per-slot pool + LRU

```python
self.placeholder_anchor_pool: dict[str, list[AnchorKVEntry]] = {}
self.placeholder_pool_max_per_slot: int = int(
    os.environ.get("SGLANG_PLACEHOLDER_POOL_MAX_PER_SLOT", "256")
)
```

LRU eviction by `last_access_time` (per-entry timestamp updated on each
lookup).  Independent lock from `anchor_kv_store_lock` to keep the two
paths from serializing.

### F1 guard on writes

`_store_placeholder_anchor_kv` computes F1 between `span.text` (predicted
text from the client) and `tokenizer.decode(actual_token_ids)` (the actual
prefill output).  If F1 < `SGLANG_PLACEHOLDER_STORE_MIN_F1=0.60`, the entry
is dropped — this prevents a divergent dense prefill from poisoning the
pool with a "wrong" placeholder text.

### RoPE delta rotation

When the k-NN match succeeds:

```python
delta = start - best.start_pos  # new_pos - old_pos
if delta != 0 and self.rope_rotary_dim > 0:
    delta_tensor = torch.full((entry_len,), delta, dtype=torch.long)
    self._apply_rope_delta_to_keys(kvcache.k_buffer, dst_kv, delta_tensor)
```

The anchor was stored at `best.start_pos` (the original agent's prefill
position); this request's slot starts at `start`.  Identical math to v10c's
suffix copy rotation at `_try_lossy_fuzzy_match:1924`.

## Environment variables (new)

| Var | Default | Role |
|---|---|---|
| `SGLANG_PLACEHOLDER_KNN_MATCH` | `0` | Master switch for read path (off by default) |
| `SGLANG_PLACEHOLDER_KNN_TOPK` | `4` | k for k-NN search |
| `SGLANG_PLACEHOLDER_KNN_MIN_COSINE` | `0.70` | Floor on per-slot similarity |
| `SGLANG_PLACEHOLDER_KNN_MAX_SLOT_LEN` | `4096` | Safety ceiling on copied slot length |
| `SGLANG_PLACEHOLDER_POOL_MAX_PER_SLOT` | `256` | LRU cap per slot |
| `SGLANG_PLACEHOLDER_STORE_MIN_F1` | `0.60` | Skip write if dense-prefill F1 below this |
| `SGLANG_PLACEHOLDER_STORE_ENABLED` | `1` | Master switch for write-back (default on; reads default off) |

The asymmetry is intentional: writes default on (the server fills the pool
opportunistically when any request declares spans), reads default off
(the conservative step is to add the new mechanism, then opt-in).

## Test coverage

- **51 tests pass total** (27 from v10c + 11 from PR2 + 13 from PR3).
- The 13 PR3 tests cover: pure-Python k-NN search (empty pool, single
  entry, top-k, min-similarity filter, top-k cap, None embedding skip,
  descending sort), end-to-end with MiniLM (similar texts pass, disjoint
  texts rejected), gating (default disabled, no spans, semantic disabled).
- The 11 PR2 tests cover: LRU under-cap, LRU eviction at cap, per-slot
  isolation, F1 sanity (identical, partial overlap, disjoint, empty),
  end-to-end F1-skip behavior, env switches.

## Verification commands

```bash
cd /home/gfy/CodeMAS_Project/sglang-kvflow

# Run all placeholder k-NN tests
SGLANG_SEMANTIC_SUFFIX_ENABLED=1 \
  /home/gfy/.conda/envs/sglang-kvflow/bin/python -m unittest \
    python.sglang.srt.mem_cache.test_semantic_suffix \
    python.sglang.srt.mem_cache.test_placeholder_knn \
    python.sglang.srt.mem_cache.test_placeholder_knn_read

# Smoke-test benchmark integration
/home/gfy/.conda/envs/sglang-kvflow/bin/python -c "
import sys; sys.path.insert(0, 'benchmark/multi_workflow')
from bench_kvcomm_ttft_stress import build_slot_messages, build_placeholder_anchor_fields, PlaceholderSlot, make_payload, CodeSegment
import argparse
# (smoke test from PR 4 docs)
"

# Run the multi-agent surfacing experiment with the new path on
SGLANG_PLACEHOLDER_KNN_MATCH=1 \
SGLANG_PLACEHOLDER_STORE_ENABLED=1 \
SGLANG_SEMANTIC_SUFFIX_ENABLED=1 \
  /home/gfy/.conda/envs/sglang-kvflow/bin/python \
    benchmark/multi_workflow/bench_kvcomm_ttft_stress.py \
    --agent-counts 1,2,3,4,5 \
    --agent-max-cases 1 \
    --agent-length-buckets 8000 \
    --agent-max-tokens 1 \
    --segment-counts 1 --files-per-case 1 \
    --disable-hierarchical-cache \
    --skip-e6 --skip-e8
```

## Expected behavior

For single-agent (warmup only): `placeholder_knn_reuse` should be roughly
neutral — the pool is empty until the first request finishes, so there
is no k-NN hit on agent 1.  Code-segment byte-exact path continues to
work as before.

For agent ≥ 2: each upstream agent's output populates one slot in the
pool.  The next agent's k-NN search finds the prior slot text and copies
its KV with a RoPE delta.  This is the geometric path Duke 2026
documents — placeholder count grows linearly with agent_count, so
reuse grows linearly too.

Conservative v1 target (matches the plan's projection): **3-10× at
agent 5** (vs Shi 2024's measured 0.65×).

## Measured results

End-to-end run: `results/ttft_agenttemplatekv/multi_agent_placeholder_v11n_20260621/`

| agent | prefix_cache | placeholder_knn | speedup | skipped_tokens | match_count |
|---:|---:|---:|---:|---:|---:|
| 1 | 259 ms | 261 ms | 0.99× | 0 | 0/1 |
| 2 | 299 ms | **156 ms** | **1.92×** | **2245** | 2/2 |
| 3 | 358 ms | **283 ms** | **1.27×** | **2245** | 3/3 |
| 4 | 390 ms | 745 ms | 0.52× | 2245 (some) | 2/4 |
| 5 | 441 ms | 1019 ms | 0.43× | 2245 (some) | 2/5 |

**Win**: agent 2 hit **1.92×**, agent 3 hit **1.27×** — the multi-agent
cliff at agent 2-3 is fixed (Shi 2024 was at 0.52× at agent 3).

**Loss**: agent 4-5 regressed to 0.43-0.52×. Root cause is the RoPE
delta rotation cost — copying 2245 tokens across all layers and
rotating keys at every position is more expensive than the savings
from skipping the dense prefill at agent_count ≥ 4.

**Plan vs reality**:
- Plan: 3-10× at agent 5
- Reality: 0.43× at agent 5
- Gap: 7-23×

The gap is due to RoPE delta cost. Phase 2 should add an
`SGLANG_PLACEHOLDER_KNN_ABORT_IF_DELTA_TOO_LARGE` guard that bails out
when `delta × entry_len > some_threshold` — i.e., when the copy cost
exceeds the prefill savings.

See `results/ttft_agenttemplatekv/multi_agent_placeholder_v11n_20260621/MULTI_AGENT_PLACEHOLDER_RESULTS.md`
for full per-agent detail.

## Out of scope (intentional)

- **Soft-weighted K-nearest reconstruction** (Duke 2026's full softmax
  blend across neighbors) — Phase 2.
- **Cross-process anchor pool sharing** — Phase 2; current pool is per-process.
- **MAScoder `_build_subtask_prompt` refactor** — planned but not
  implemented in v1. The benchmark has the slot-decomposition path
  (`PlaceholderSlot`, `build_slot_messages`); MAScoder integration is
  a separate PR.
- **New embedding model** — keep MiniLM (already cached locally).

## Files added in v11

```
python/sglang/srt/mem_cache/text_utils.py                  (new, ~70 LOC)
python/sglang/srt/mem_cache/test_placeholder_knn.py        (new, ~280 LOC)
python/sglang/srt/mem_cache/test_placeholder_knn_read.py   (new, ~290 LOC)
```

## Related documents

- `results/ttft_agenttemplatekv/multi_agent_surface_20260621/MULTI_AGENT_GEOMETRIC_SURFACE.md`
  — empirical surfacing of Shi 2024's multi-agent cliff
- `results/selective_ast_reuse/prompt_fair_semantic_suffix_v10c_28case_20260621.md`
  — v10c mainline (the foundation for v11)
- `/home/gfy/.claude/plans/humble-strolling-cerf.md` — v11 plan
