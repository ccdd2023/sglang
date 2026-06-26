# Giant-Codebase Reuse Benchmark — Report

Headline reuse metrics across the persistent-server multi-agent runs. Each run loaded N SWE-Smith pandas tasks × 5 agents from a single sglang server (chunked at `chunk-size` tasks per chunk to dodge the `_delete_leaf` race).

## Per-Run Overview

| Run | Tasks | Rows | Total Prompt Tok | Total Cached Tok | **Reuse Ratio** | Avg Workflow TTFT (ms) | Anchor Hits | Matched Slots | Max Pool Size |
|-----|------:|-----:|-----------------:|-----------------:|----------------:|-----------------------:|------------:|--------------:|--------------:|
| `giant_pandas_50_20260626` | 50 | 250 | 1,724,770 | 350,782 | **0.2034** | 2220 | 0 | 0 | 0 |

## Per-Task Reuse Trend — `giant_pandas_50_20260626` (N=50)

| Task Idx | Case ID | Avg Cached Ratio | Reuse Ratio | Avg Agent TTFT (ms) | Sum Workflow TTFT (ms) | Pool Size | Anchor Hits | Matched |
|---------:|---------|-----------------:|------------:|---------------------:|------------------------:|----------:|------------:|--------:|
| 0 | `combine_file__11s6papj` | 0.201 | 0.201 | 306 | 1532 | 0 | 0 | 0 |
| 1 | `combine_file__1eilbetv` | 0.204 | 0.204 | 393 | 1967 | 0 | 0 | 0 |
| 2 | `combine_file__2p4yneeo` | 0.203 | 0.203 | 560 | 2802 | 0 | 0 | 0 |
| 3 | `combine_file__3ddy6d59` | 0.206 | 0.205 | 282 | 1411 | 0 | 0 | 0 |
| 4 | `combine_file__3ra8xqln` | 0.204 | 0.203 | 522 | 2610 | 0 | 0 | 0 |
| 5 | `combine_file__4qxpwmzm` | 0.203 | 0.202 | 627 | 3134 | 0 | 0 | 0 |
| 6 | `combine_file__4z3kyuwy` | 0.203 | 0.203 | 488 | 2439 | 0 | 0 | 0 |
| 7 | `combine_file__5hf3mzz8` | 0.205 | 0.204 | 360 | 1800 | 0 | 0 | 0 |
| 8 | `combine_file__5n84dyfn` | 0.203 | 0.203 | 568 | 2841 | 0 | 0 | 0 |
| 9 | `combine_file__6jxwun44` | 0.203 | 0.203 | 494 | 2470 | 0 | 0 | 0 |
| 10 | `combine_file__6syhgyxn` | 0.204 | 0.204 | 392 | 1958 | 0 | 0 | 0 |
| 11 | `combine_file__7lw4rdof` | 0.204 | 0.203 | 502 | 2509 | 0 | 0 | 0 |
| 12 | `combine_file__90rzhqlp` | 0.204 | 0.203 | 473 | 2366 | 0 | 0 | 0 |
| 13 | `combine_file__9409pn6x` | 0.204 | 0.203 | 472 | 2362 | 0 | 0 | 0 |
| 14 | `combine_file__9d3km1v8` | 0.205 | 0.204 | 362 | 1812 | 0 | 0 | 0 |
| 15 | `combine_file__a1eww6g6` | 0.203 | 0.203 | 575 | 2875 | 0 | 0 | 0 |
| 16 | `combine_file__a1i5j5x9` | 0.203 | 0.203 | 520 | 2601 | 0 | 0 | 0 |
| 17 | `combine_file__aeazsphe` | 0.203 | 0.203 | 522 | 2612 | 0 | 0 | 0 |
| 18 | `combine_file__ajampw57` | 0.204 | 0.203 | 484 | 2419 | 0 | 0 | 0 |
| 19 | `combine_file__akmsx8we` | 0.204 | 0.204 | 388 | 1938 | 0 | 0 | 0 |
| 20 | `combine_file__bj6sx8t8` | 0.207 | 0.206 | 246 | 1232 | 0 | 0 | 0 |
| 21 | `combine_file__bnhc8507` | 0.203 | 0.203 | 553 | 2766 | 0 | 0 | 0 |
| 22 | `combine_file__bp7kl060` | 0.203 | 0.202 | 605 | 3023 | 0 | 0 | 0 |
| 23 | `combine_file__d20m1tdx` | 0.205 | 0.204 | 371 | 1855 | 0 | 0 | 0 |
| 24 | `combine_file__dyzef8r1` | 0.204 | 0.204 | 512 | 2562 | 0 | 0 | 0 |
| 25 | `combine_file__e3qrzqzo` | 0.203 | 0.203 | 511 | 2553 | 0 | 0 | 0 |
| 26 | `combine_file__eeendq0s` | 0.203 | 0.203 | 497 | 2484 | 0 | 0 | 0 |
| 27 | `combine_file__el0nskpi` | 0.206 | 0.205 | 314 | 1568 | 0 | 0 | 0 |
| 28 | `combine_file__fe1h3bzf` | 0.203 | 0.203 | 494 | 2471 | 0 | 0 | 0 |
| 29 | `combine_file__g9thdmoi` | 0.204 | 0.204 | 439 | 2196 | 0 | 0 | 0 |
| 30 | `combine_file__hd6urze7` | 0.204 | 0.204 | 411 | 2056 | 0 | 0 | 0 |
| 31 | `combine_file__hliqxhn6` | 0.204 | 0.203 | 554 | 2770 | 0 | 0 | 0 |
| 32 | `combine_file__hzt3rk4z` | 0.204 | 0.203 | 487 | 2435 | 0 | 0 | 0 |
| 33 | `combine_file__i3dkumyn` | 0.203 | 0.203 | 488 | 2441 | 0 | 0 | 0 |
| 34 | `combine_file__i846e6z3` | 0.204 | 0.204 | 382 | 1911 | 0 | 0 | 0 |
| 35 | `combine_file__i8daem98` | 0.206 | 0.205 | 328 | 1641 | 0 | 0 | 0 |
| 36 | `combine_file__innqdri0` | 0.206 | 0.205 | 393 | 1966 | 0 | 0 | 0 |
| 37 | `combine_file__j9e0j8z9` | 0.207 | 0.206 | 246 | 1229 | 0 | 0 | 0 |
| 38 | `combine_file__ja8zv65r` | 0.207 | 0.206 | 277 | 1386 | 0 | 0 | 0 |
| 39 | `combine_file__jd2k9on0` | 0.203 | 0.203 | 563 | 2815 | 0 | 0 | 0 |
| 40 | `combine_file__jj4xjw2c` | 0.205 | 0.204 | 321 | 1606 | 0 | 0 | 0 |
| 41 | `combine_file__jj51a1dt` | 0.205 | 0.204 | 425 | 2125 | 0 | 0 | 0 |
| 42 | `combine_file__l5t8z395` | 0.206 | 0.205 | 303 | 1513 | 0 | 0 | 0 |
| 43 | `combine_file__l5wqhuke` | 0.203 | 0.203 | 593 | 2966 | 0 | 0 | 0 |
| 44 | `combine_file__leh2t9n7` | 0.204 | 0.204 | 387 | 1937 | 0 | 0 | 0 |
| 45 | `combine_file__lhketo08` | 0.203 | 0.203 | 547 | 2734 | 0 | 0 | 0 |
| 46 | `combine_file__m3gnmry9` | 0.203 | 0.203 | 523 | 2616 | 0 | 0 | 0 |
| 47 | `combine_file__m3gu6j4n` | 0.204 | 0.204 | 394 | 1972 | 0 | 0 | 0 |
| 48 | `combine_file__mmfdyzv0` | 0.203 | 0.203 | 535 | 2676 | 0 | 0 | 0 |
| 49 | `combine_file__mtuy10j5` | 0.207 | 0.206 | 210 | 1052 | 0 | 0 | 0 |

## Pool Growth Interpretation

- `placeholder_anchor_store_entry_count` is the cumulative pool size at the end of each task's last agent. Monotonic non-decreasing growth within a chunk indicates the placeholder k-NN body is writing new anchors.
- `placeholder_anchor_pool_hit_count` is the per-task sum of k-NN body matches. A non-zero value means downstream agents found a similar (cos ≥ min_cosine) anchor in the pool from a prior request.
- `placeholder_kv_prefill_matched_slots` is the count of slots whose KV was successfully copied from a pool entry instead of dense-prefilled. This is the operation that produces real TTFT savings.

**Note**: A run may show non-zero `reuse_ratio` from prefix-cache reuse alone even when anchor hits = 0. This is the **cache-ordering** contribution described in the v44 plan §3.1 — the KNN body adds additional speedup on top.
