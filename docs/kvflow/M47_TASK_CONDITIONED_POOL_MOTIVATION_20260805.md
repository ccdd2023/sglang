# M47 motivation: does task-conditioned coding selection improve V46?

Date: 2026-08-05

Branch: `research/coding-aware-v45-multi-observation-20260803`
Status: complete negative/diagnostic motivation result; not a promoted V47 policy

## Question

V46 currently chooses up to three valid repository-observation islands by
length and recency.  Inspired by dynamic input-dependent selection in
MInference and query-dependent block importance in RcLLM, this experiment asks
a narrower causal question:

> With the prompt, copied-token budget, island count, executor, model and
> decoding fixed, does a coding-task signal choose safer or more useful V46
> islands than recency or random selection?

The tested coding signal is deliberately simple and answer-blind.  It extracts
identifiers from the last 2,048 visible characters before the RepoBench-P
completion cursor, weights rare identifiers within the seven repository
observations more strongly, and selects the three observations with the
largest weighted overlap.  The missing answer line is never read by the
selector.

This is a **selector motivation experiment**, not a new production policy and
not functional coding-task accuracy.

## Controlled design

Dataset: the existing frozen RepoBench-P 50-case workload.  Model:
Qwen2.5-Coder-3B-Instruct.  Every reuse arm uses the exact same target token IDs
and copies exactly three 512-token middle islands, for 1,536 copied tokens per
target.

| Arm | Selection rule | Copied tokens | Islands | Prefix reuse | Prefetch |
|---|---|---:|---:|---|---|
| Dense | no lossy KV copy | 0 | 0 | off | off |
| V46 recency | longest eligible observations; newest wins ties | 1,536 | 3 | off | off |
| Coding symbol | cursor-local identifier/IDF overlap | 1,536 | 3 | off | off |
| Matched random | frozen per-case seed `20260805` | 1,536 | 3 | off | off |

The 12-case canary was registered before GPU execution.  It expanded to 50
only after all three reuse arms produced every expected copy with zero
fallback and the coding selector differed sufficiently from V46.  On the full
set, coding-symbol selection differs from V46 on 43/50 cases and from random on
48/50 cases, so the comparison is not an accidental selector equivalence.

## Full-50 result

All reuse arms completed 150/150 physical copy events with zero fallback.

| Arm | Exact line | CodeSim | Mean TTFT | Cache-ready speedup | N=4 incl. build |
|---|---:|---:|---:|---:|---:|
| Dense | **5/50** | 49.99% | 287.27 ms | 1.000x | — |
| V46 recency | 4/50 | **52.54%** | **217.29 ms** | **1.322x** | **1.048x** |
| Coding symbol | 4/50 | 51.35% | 244.53 ms | 1.175x | 0.954x |
| Matched random | 4/50 | 51.79% | 250.01 ms | 1.149x | 0.937x |

The simple coding selector does **not** improve quality:

- versus V46 recency: -1.18 CodeSim points, paired bootstrap 95% CI
  `[-4.18, +1.14]`, and no exact-line gain;
- versus matched random: -0.44 CodeSim points, paired bootstrap 95% CI
  `[-3.46, +2.93]`, and no exact-line gain;
- Dense has one exact-line pass that all three lossy arms damage.  None of the
  three selectors rescues a Dense exact failure.

The result therefore does not support promoting cursor-local lexical overlap
as an island-admission signal.

## Why V46 recency is faster even at the same copied-token count

Equal token count is not equal saved attention work in a causal transformer.
A token later in the prompt attends to a longer prefix, so skipping its Dense
prefill saves more attention work than skipping an equally long early token.
Scattered islands also introduce Dense gaps and additional Dense/copy stage
boundaries.

| Selector | Mean context index | Mean target-position fraction | Causal attention-work proxy covered | Mean Dense gap between islands |
|---|---:|---:|---:|---:|
| V46 recency | **3.88** | **0.461** | **36.65%** | **0 token** |
| Coding symbol | 2.38 | 0.289 | 24.75% | 593.92 token |
| Matched random | 2.27 | 0.277 | 23.92% | 675.84 token |

V46 is faster than coding-symbol selection by 27.23 ms per target (paired
bootstrap 95% CI `[-32.45, -22.10]` for `V46 - symbol`) and faster than random
by 32.72 ms (`[-37.66, -27.79]`).  The confidence intervals exclude zero.
This is evidence that V46's length/recency tie-break is an effective **speed
heuristic** on this workload, not evidence that recency is a generally safer
quality heuristic.

## Research decision

Adopt the following conclusions:

1. Keep V46's preference for late, contiguous observations in the speed/cost
   model.  A future risk selector must account for causal position and Dense
   gaps, not only copied-token count.
2. Do not promote raw lexical identifier overlap.  It is too weak a proxy for
   the contextual error introduced by stale K/V.
3. The next motivation probe should use the model's own signal—target-query
   attention and K/V drift—while matching not only copied tokens and island
   count, but also target-position/work budget.  Otherwise a “quality-aware”
   selector can lose speed merely by moving reuse earlier in the prompt.
4. RepoBench exact-line and CodeSim remain mechanism metrics.  Any promoted
   runtime policy still requires frozen functional execution accuracy.

## Reproduction and artifacts

Implementation:

```text
benchmark/multi_workflow/motivate_v47_task_conditioned_pool.py
benchmark/multi_workflow/test_motivate_v47_task_conditioned_pool.py
```

Machine-readable registration, per-case selections, raw generations, ledgers
and final result:

```text
/home/gfy/CodeMAS_Project/kvflow-artifacts/
  impactkv_m47_task_conditioned_pool_20260805/
    canary12/
    full50/
      REGISTRATION.json
      SELECTION_AUDIT.json
      dense.json
      v46_recency_m47.json
      coding_symbol_overlap_m47.json
      matched_random_m47.json
      RESULT.json
```

Verification: `3 passed`; full50 has 150/150 copies and zero fallback for each
reuse arm.  GPU is idle after completion.
