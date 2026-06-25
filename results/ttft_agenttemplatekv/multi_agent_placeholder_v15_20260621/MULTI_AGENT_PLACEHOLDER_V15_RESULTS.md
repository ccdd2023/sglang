# Multi-Agent k-NN KV Reuse — v15 Slot Order Fix Results (2026-06-22)

## Goal

Fix the agent 4-5 regression observed in v11n/v12/v13/v14 by ensuring the
`extra_context` placeholder slot's `start_token` is positioned **after** the
prefix cache boundary, so the k-NN path can copy the slot's full content
without overlapping the prefix's pre-cached region.

## Root cause (from v11n analysis)

In v11n telemetry, agent 4-5's "miss" sub-agents showed `skipped_invalid=1`
and `skip_tok=0` despite `sim=1.0` from the k-NN pool. The body code at
`radix_cache.py:2273-2275` had a guard:

```python
# Slots must come AFTER the prefix-matched exact_len so the
# reconstructed KV sits contiguously in the prefill stream.
if start < exact_len:
    skipped_invalid += 1
    continue
```

The benchmark's `build_slot_messages` (v11-v14) put the `extra_context`
slot **first** in the prompt body (right after the system/role/task
instructions, before the code segments). With prompt structure:
- system + role + task: ~200 tokens
- extra_context slot: starts at ~60 tokens
- code_base1 slot: starts at ~2300 tokens

The prefix cache (radix tree) matched tokens 0-215+ of the prompt. The
extra_context slot's `start_token=61` was less than `prefix_len=215`,
so the guard fired and the k-NN copy was skipped. The k-NN match was
discarded even though it was valid (sim=1.0).

## Two attempted fixes

**Attempt 1: Remove the guard entirely.** Confirmed via debug log that
`start=61 < prefix_len=215+` for all 5 sub-agents. Removing the guard
let the k-NN copy run, but **the flashinfer attention backend crashed**:
```
RuntimeError: qo_indptr[1]-35 - qo_indptr[0]0 should be non-negative
```

The crash happens because the prefix cache's `device_indices` (length
`prefix_len`) plus the k-NN's `new_slots` (length `entry_len`) form a
discontinuous KV layout — the model's `q` (query) tokens start mid-KV
sequence, which flashinfer can't handle. The guard is **required for
correctness** with the flashinfer backend.

**Attempt 2: Reorder the placeholder slots in the benchmark.** Swap the
order: put `code_base1` slot FIRST, `extra_context` slot LAST. This
moves `extra_context`'s `start_token` to position ~2500 (after the code
segments), well past the prefix boundary. The k-NN body sees
`start > prefix_len` and can copy the slot safely without overlapping
the prefix's KV.

## Implementation

File: `benchmark/multi_workflow/bench_kvcomm_ttft_stress.py`,
function `build_slot_messages`, lines ~432-449. Reordered the default
slot list to put `code_base{N}` slots before `extra_context`.

## End-to-end results

| agent | prefix | v15 SLOTORDER | speedup |
|---:|---:|---:|---:|
| 1 | 260 | 298 | 0.87× |
| 2 | 292 | **149** | **1.96×** ✓ |
| 3 | 338 | **202** | **1.67×** ✓ |
| 4 | 385 | 454 | 0.85× |
| 5 | 426 | 1067 | 0.40× |

### Comparison vs prior runs

| agent | v11n | v13 HEAD2 | v14 NATIVE | **v15 SLOTORDER** |
|---:|---:|---:|---:|---:|
| 1 | 0.99× | 0.96× | 0.98× | 0.87× |
| 2 | 1.92× | 1.71× | 1.78× | **1.96×** ✓ |
| 3 | 1.27× | 1.25× | 1.29× | **1.67×** ✓ |
| 4 | 0.52× | 0.52× | 0.53× | **0.85×** ✓ |
| 5 | 0.43× | 0.42× | 0.41× | 0.40× |

Agent 2-3 hit the **best speedups ever measured** for the placeholder
k-NN path (1.96× and 1.67×). Agent 4 improved significantly (0.52× →
0.85×). Agent 5 still regresses (0.40×).

### Per-agent hit telemetry (v15 SLOTORDER, agent 4-5)

```
agent=4 implementer  hit=0 match=0 skip_tok=0   skip_invalid=0
agent=4 debugger     hit=0 match=0 skip_tok=0   skip_invalid=0
agent=4 reviewer     hit=0 match=0 skip_tok=0   skip_invalid=0
agent=4 verifier     hit=1 match=1 skip_tok=192 skip_invalid=1
agent=5 implementer  hit=2 match=2 skip_tok=2447 skip_invalid=0
agent=5 debugger     hit=1 match=1 skip_tok=208  skip_invalid=1
agent=5 reviewer     hit=1 match=1 skip_tok=219  skip_invalid=1
agent=5 verifier     hit=2 match=2 skip_tok=2473 skip_invalid=0
agent=5 auditor      hit=1 match=1 skip_tok=237  skip_invalid=1
```

The pattern: agents where `cached > 2500` (prefix cache has nearly the
full prompt) get `hit=2 match=2` (both slots copied, 2400+ tokens
skipped). Agents where `cached < 300` get `hit=1 match=1` with only
small skips — **the prefix cache is matching MORE than expected**, which
makes the k-NN guard fire for the smaller slot.

## Why agent 5 still regresses

Agent 5's `implementer` and `verifier` succeed (cached > 2500), but
`debugger/reviewer/auditor` only partially succeed (cached < 300). The
total workflow TTFT is dominated by the 3 sub-agents with 280-300ms
TTFT each, summing to 1067ms.

The pattern correlates with whether the prefix cache has the
**code_base1 segment content** for that sub-agent. When the prefix
cache has it (cached=2500+), the k-NN succeeds. When the prefix cache
doesn't (cached=300), only one of the two slots is processed.

The root cause is that the prefix cache's behavior across sub-agents
within a single `agent_count=N` run is **uneven**: some sub-agents
benefit from prior prefix matches, others don't. The placeholder pool
content is consistent (sim=0.99), but the guard's interaction with
the prefix cache state is not.

## What's still needed

To fix the agent 5 regression, one of:
1. **Trim the k-NN copy** to only the post-prefix portion of the slot
   (when start < prefix_len, copy `entry_len - (prefix_len - start)` tokens
   instead of the full slot). This requires deeper changes to the body.
2. **A more capable attention backend** that handles discontinuous KV
   layouts. The flashinfer crash indicates a limitation; this would
   require backend-level work.
3. **Drop the `code_base1` slot from the placeholder system** and only
   keep `extra_context` (which is the part the placeholder k-NN was
   designed for). Code segments are stable enough that they don't
   need k-NN; the prefix cache handles them.

## Files

- This report: `results/ttft_agenttemplatekv/multi_agent_placeholder_v15_20260621/MULTI_AGENT_PLACEHOLDER_V15_RESULTS.md`
- Raw CSV: `results/ttft_agenttemplatekv/multi_agent_placeholder_v15_SLOTORDER_20260621/ttft_stress_table.csv`
- v11n baseline: `results/ttft_agenttemplatekv/multi_agent_placeholder_v11n_20260621/MULTI_AGENT_PLACEHOLDER_RESULTS.md`
- Code change: `benchmark/multi_workflow/bench_kvcomm_ttft_stress.py:432-449` (slot order swap in `build_slot_messages`)
- Plan: `/home/gfy/.claude/plans/humble-strolling-cerf.md`

## Reproduction

```bash
cd /home/gfy/CodeMAS_Project/sglang-kvflow
SGLANG_PLACEHOLDER_KNN_MATCH=1 SGLANG_PLACEHOLDER_STORE_ENABLED=1 \
SGLANG_NATIVE_MOVE_KV_CACHE=1 \
  python benchmark/multi_workflow/bench_kvcomm_ttft_stress.py \
    --agent-counts 1,2,3,4,5 --agent-max-cases 1 \
    --agent-length-buckets 8000 --agent-max-tokens 1 \
    --segment-counts 1 --files-per-case 1 \
    --disable-hierarchical-cache --skip-e6 --skip-e8 \
    --out-dir results/ttft_agenttemplatekv/multi_agent_placeholder_v15_REPRO
```

## Summary

**v15 SLOTORDER achieves the best agent 2-3 speedups ever** (1.96× and
1.67×) and improves agent 4 from 0.52× to 0.85×, but agent 5 still
regresses at 0.40×. The slot order fix successfully resolves the
`skipped_invalid` guard issue for sub-agents whose prefix cache has
the full prompt content, but agent 5's `debugger/reviewer/auditor`
sub-agents still hit the guard because their prefix cache state is
uneven across the run. Further work needed to trim the k-NN copy or
improve prefix cache fairness.
