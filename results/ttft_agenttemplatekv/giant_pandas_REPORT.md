# Giant-Codebase Reuse Benchmark — Report

Headline reuse metrics across the persistent-server multi-agent runs. Each run loaded N SWE-Smith pandas tasks × 5 agents from a single sglang server (chunked at `chunk-size` tasks per chunk to dodge the `_delete_leaf` race).

## Per-Run Overview

| Run | Tasks | Rows | Total Prompt Tok | Total Cached Tok | **Reuse Ratio** | Avg Workflow TTFT (ms) | Anchor Hits | Matched Slots | Max Pool Size |
|-----|------:|-----:|-----------------:|-----------------:|----------------:|-----------------------:|------------:|--------------:|--------------:|
| `giant_pandas_pilot5_v4_20260626` | 5 | 25 | 174,740 | 1,187 | **0.0068** | 2563 | 0 | 0 | 0 |

## Per-Task Reuse Trend — `giant_pandas_pilot5_v4_20260626` (N=5)

| Task Idx | Case ID | Avg Cached Ratio | Reuse Ratio | Avg Agent TTFT (ms) | Sum Workflow TTFT (ms) | Pool Size | Anchor Hits | Matched |
|---------:|---------|-----------------:|------------:|---------------------:|------------------------:|----------:|------------:|--------:|
| 0 | `combine_file__11s6papj` | 0.004 | 0.004 | 372 | 1859 | 0 | 0 | 0 |
| 1 | `combine_file__1eilbetv` | 0.008 | 0.008 | 555 | 2773 | 0 | 0 | 0 |
| 2 | `combine_file__2p4yneeo` | 0.006 | 0.006 | 721 | 3607 | 0 | 0 | 0 |
| 3 | `combine_file__3ddy6d59` | 0.011 | 0.011 | 323 | 1615 | 0 | 0 | 0 |
| 4 | `combine_file__3ra8xqln` | 0.007 | 0.007 | 592 | 2962 | 0 | 0 | 0 |

## Pool Growth Interpretation

- `placeholder_anchor_store_entry_count` is the cumulative pool size at the end of each task's last agent. Monotonic non-decreasing growth within a chunk indicates the placeholder k-NN body is writing new anchors.
- `placeholder_anchor_pool_hit_count` is the per-task sum of k-NN body matches. A non-zero value means downstream agents found a similar (cos ≥ min_cosine) anchor in the pool from a prior request.
- `placeholder_kv_prefill_matched_slots` is the count of slots whose KV was successfully copied from a pool entry instead of dense-prefilled. This is the operation that produces real TTFT savings.

**Note**: A run may show non-zero `reuse_ratio` from prefix-cache reuse alone even when anchor hits = 0. This is the **cache-ordering** contribution described in the v44 plan §3.1 — the KNN body adds additional speedup on top.
