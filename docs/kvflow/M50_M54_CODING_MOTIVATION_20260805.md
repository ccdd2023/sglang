# M50–M54 技术附件：coding 信息究竟预测复用价值还是复用风险

Date: 2026-08-05

Branch: `research/coding-aware-v45-multi-observation-20260803`
Runtime decision: do not change V40/V46 yet

> 阅读定位：这是 M50–M54 的冻结协议与逐项数据附件，不是面向汇报的主叙事。建议先读 [技术迭代大报告](TECHNICAL_ITERATION_EVIDENCE_REVIEW_20260805.md) 第一部分；只有在核查配对设计、gate 或原始数值时再读本文。

## 结论先行

这五项实验没有为 V46 找到一个更复杂的统一风险分数，反而澄清了过去一直混淆的两个概念。一个 observation 与当前代码路径相关，说明模型更可能依赖它；但这并不保证它在旧 prompt 中形成的 K/V 可以安全搬到新 prompt。反过来，一段 K/V 的局部 drift 很小，只表示它可能较安全，也不说明复用它对当前任务有价值。

M50 和 M51 分别否定了两个直觉解释：grounded tool result 并不稳定地比 assistant decision 更安全，same-file mutation 也没有稳定增加实测 splice harm。M52/M53 则在两批请求中复现了 path-relevant observation 获得更高 target attention，给出了目前最强的 coding-specific dependency 证据。M54 再证明，直接把 path 权重乘到 16-token drift probe 上会让风险排序变差，而不是变好。

因此，这组实验推出的是一个二维决策，而不是另一个 heuristic：

```text
coding path dependency  -> expected utility of retaining/reusing an island
small K/V probe drift   -> expected lossy-reuse risk of that island
```

下一版应先拒绝或 Dense 计算高风险 island，再在低风险候选中优先选择 path dependency 高的 observation。本文后续部分保留英文实验定义，是为了与 registration、artifact 字段和分析脚本逐项对应。

## Why the old motivation was incomplete

V40 established that successful read-only observations provide enough
contiguous copy capacity and that they can be guarded by repository version.
That was a policy/capacity audit.  It did not establish either of these causal
claims:

- grounded observations have lower cross-prefix K/V distortion than assistant
  reasoning;
- observations become more damaging to reuse after their file is modified.

M48 later established that model-internal `attention × K/V drift` correlates
with physical splice harm.  M49 established that a 16-token layer-18 probe can
rank individual-island harm, but its request-level maximum cannot predict
three-island composition.  M50–M54 connect those findings to actual coding
provenance and path dependencies in real coding-agent histories.

## Common causal measurement

All candidates are exact 128-token spans present in both a source prompt and a
later target prompt.  The measurement:

1. runs Dense source and Dense target prefill with
   `Qwen2.5-Coder-3B-Instruct` in BF16;
2. takes source K/V for the candidate and applies the exact RoPE position
   correction;
3. physically splices that K/V into the Dense target prefix;
4. recomputes the remaining target suffix;
5. compares the final logits with Dense target logits using Jensen–Shannon
   divergence (JS);
6. where applicable, measures target-query attention and frozen next-action
   NLL.

This is real lossy KV reuse, not copy-only token reuse and not exact-prefix
reuse.  It is an offline mechanism measurement, not task accuracy or TTFT.

## M50: grounded fact versus assistant decision

### Hypothesis

A successful read-only tool observation is an externally grounded repository
fact, while assistant reasoning and its tool call are prefix-conditioned model
decisions.  At equal length and under the same rolling-history transition, the
grounded block should therefore be safer to reuse.

### Controls

- 20 real Dense coding-agent requests from 20 different tasks;
- 128 tokens per candidate;
- same source prompt, target prompt, and rolling transition within each pair;
- nearest available assistant block, at most 512 tokens or 5% of the target
  prompt away;
- position-difference regression reported in addition to raw paired results.

### Result

| Metric | Grounded tool | Assistant decision | Grounded lower pairs | Equal-position ratio |
|---|---:|---:|---:|---:|
| Final-logit JS mean | 0.0003266 | 0.0004579 | **50.0%** | 0.750 |
| Final-logit JS median | 0.0002564 | 0.0003588 | — | — |
| K/V cosine drift mean | 0.01869 | 0.01694 | 60.0% | 0.633 |
| K/V cosine drift median | 0.00790 | 0.01364 | — | — |
| Next-action NLL delta mean | 0.00358 | 0.00889 | 50.0% | — |

The average and position-adjusted JS favor grounded observations, but only 10
of 20 individual pairs do.  The preregistered 65% consistency gate fails.
The distribution is heterogeneous and dominated by a few large assistant
errors; provenance alone is not a dependable admission rule.

Decision: `NOT_SUPPORTED`.

## M51: same-file version transition

### Hypothesis

For the same exact old observation, reuse after a real mutation to its file
should be more damaging than reuse after a noncritical interaction that still
references the same file.

### Controls

- 18 exact-observation pairs across eight tasks;
- treatment contains only a localized repository mutation event;
- control references the same path but has no mutation, diff, executable
  failure, or other critical event;
- same 128-token observation in both contexts;
- position, prefix-shift, and prompt-length differences included as
  covariates.

### Result

| Metric | Same-path mutation | Same-path noncritical | Mutation higher pairs | Adjusted ratio |
|---|---:|---:|---:|---:|
| Final-logit JS mean | 0.0003987 | 0.0003883 | **44.4%** | 0.819 |
| Final-logit JS median | 0.0001348 | 0.0003609 | — | — |
| K/V cosine drift mean | 0.01724 | 0.02323 | 55.6% | 0.713 |
| Next-action NLL delta mean | -0.00024 | 0.00478 | 44.4% | — |

Neither the direction-consistency gate nor the adjusted-effect gate passes.
This does not make repository version guards incorrect: stale repository facts
remain semantically unsafe.  It does mean the current data do not support the
stronger claim that same-path mutation itself predicts larger cross-prefix KV
splice error.

Decision: `NOT_SUPPORTED`.

The first `matched18` preparation repeated case IDs because of a sampler bug
and is explicitly marked invalid.  The sampler was fixed, tests were added,
and the unchanged thresholds were rerun in `matched18_v2`.  Only the v2 result
above is valid.

## M52: path overlap as model dependency

### Hypothesis

Within the same target prompt, an old grounded observation whose repository
path is referenced by the latest completed coding interaction should receive
more target-query attention than a path-disjoint observation.

M52 separately preregistered a stronger guard hypothesis: if higher dependency
also produced larger splice harm, path overlap could trigger Dense protection.

### Controls

- 20 requests across 12 tasks;
- one path-relevant and one path-disjoint grounded observation in the same
  source/target prompt;
- 128 tokens per candidate;
- target-position distance capped at 25% and regression-adjusted;
- final 32 prompt queries and five model layers for attention measurement.

### Result

| Metric | Path relevant | Path disjoint | Relevant higher pairs | Position-adjusted ratio |
|---|---:|---:|---:|---:|
| Attention mean | **0.03251** | 0.01231 | **70.0%** | **1.623** |
| K/V drift mean | 0.00436 | 0.01395 | 25.0% | 0.381 |
| Attention × drift mean | 0.000128 | 0.000112 | 60.0% | 0.614 |
| Final-logit JS mean | 0.0004627 | 0.0005374 | 30.0% | 0.391 |

The dependency gates pass: path overlap predicts substantially higher model
attention.  The predeclared Dense-protection guard gates fail because relevant
blocks are not more damaging; in this cohort they are less distorted and less
damaging.

Decisions:

- path dependency: `SUPPORTED`;
- path-overlap Dense guard: `NOT_SUPPORTED`.

## M53: request-disjoint replication

M53 froze the direction discovered by M52 and tested 19 unused request IDs,
keeping at most one request per candidate-pair identity.  It covers nine tasks,
but tasks and some individual observations overlap M52; it is not a fully
independent dataset.

| Metric | Path relevant | Path disjoint | Relevant higher pairs | Position-adjusted ratio |
|---|---:|---:|---:|---:|
| Attention mean | **0.02334** | 0.01429 | **89.5%** | **1.413** |
| K/V drift mean | 0.01398 | 0.02609 | 36.8% | 0.969 |
| Attention × drift mean | 0.000240 | 0.000236 | 63.2% | 1.811 |
| Final-logit JS mean | 0.0003537 | 0.0003331 | 47.4% | 0.408 |

The attention finding replicates strongly.  Lower drift passes only the raw
pair-direction gate, not the adjusted ratio gate.  Lower JS passes the
adjusted ratio gate but not the pair-consistency gate.  Therefore the complete
M52 reverse-safety pattern does not replicate.

Decision: `NOT_REPLICATED` as a combined dependency-and-safety claim.  The
path-dependency subclaim remains supported by both cohorts.

## M54: multiplicative dependency × drift risk

### Hypothesis

Use the M49 frozen layer-18, 16-token probe as drift and multiply it by M52's
frozen `1.623` path-attention ratio for relevant observations.  This should
rank single-island JS better than the probe alone.

M54 uses all 14 eligible requests not opened by M52 or M53, across six tasks.
No M54 causal labels were read before registration.

| Score | Global Spearman with JS | Within-request pair ranking accuracy |
|---|---:|---:|
| 16-token probe only | **0.506** | 42.9% |
| Path-weighted probe | 0.477 | 42.9% |
| Change | **-0.030** | 0.0 pp |

The hybrid remains correlated with JS, but path weighting makes it worse than
the probe and does not improve pair ordering.  It fails four of six frozen
gates.

Decision: `NOT_SUPPORTED`; do not implement this multiplicative score in
SGLang.

## What is now proved, suggested, and disproved

### Supported

- K/V drift is a real single-island distortion signal (M48/M49).
- The latest coding interaction's path is a stable indicator of which old
  repository observation the model uses (M52/M53).
- Coding metadata therefore carries information that recency alone does not:
  it estimates reuse **utility/dependency**.

### Suggested but not yet independently proved

- Path-relevant blocks may often be stable enough to be attractive reuse
  candidates; average JS is favorable in M52 and adjusted JS remains favorable
  in M53, but pairwise consistency is insufficient.
- A constrained selector may benefit from combining a dependency objective
  with an independent distortion constraint.

### Disproved for the tested protocols

- `grounded observation => uniformly safe`;
- `same-path mutation => larger KV splice harm`;
- `path overlap => use Dense protection`;
- `path_weight × probe_drift => better scalar risk`;
- M49's `max probe risk over three islands => request abstention`.

## Consequence for V40, V45, and V46

V40's grounded-observation rule remains useful as a conservative way to avoid
copying assistant decisions, but it is not a causal safety proof.  V45's
version guard remains a semantic correctness rule, not an empirically proven
KV-distortion predictor.  V46's multi-observation path pool remains the right
data structure for keeping several version-valid candidates, but its recency/
size ranking does not exploit the now-supported dependency signal.

The next method should rank pool members with two separate axes:

| Probe risk | Path dependency | Action |
|---|---|---|
| High | High | Dense/recompute; important but unsafe |
| Low | High | Prefer for lossy KV reuse |
| High | Low | Reject; low value and unsafe |
| Low | Low | Use only as spare capacity |

This preserves the distinction that M54 showed is necessary.  Dependency is a
benefit, not a multiplier on risk.

## Proposed M55 before any runtime change

Collect a genuinely task-disjoint coding-agent cohort first.  Because M49
falsified the existing three-island request-level risk aggregation, M55 must
first isolate one 128-token island per target.  On that frozen cohort compare
five equal-budget selectors:

1. fixed-budget recency, representing the age bias shared by V40/V46;
2. path dependency only;
3. probe-risk only;
4. two-stage constrained selection;
5. seeded random control.

The two-stage selector should be specified as:

```text
eligible_i = version_valid_i and probe_risk_i <= frozen_threshold
if no island is eligible: use Dense
otherwise choose the path-relevant eligible island;
if it is unsafe, choose the minimum-risk eligible island
```

The motivation gate should require the two-stage selector to capture more
target attention than probe-only while keeping single-island JS no worse, and
to reduce JS versus path-only while copying the same 128 tokens on common
non-abstained cases.  It must also cover at least 70% of the recency control.
Only after this single-island gate passes may a separately registered
multi-island composition rule be designed; M55 itself cannot promote V46's
three-island request policy.

## M55/M56 outcome update

The preregistered M55 quality comparison did not open GPU causal labels.  The
fresh-13 trajectories produced 31 path-matched pairs; target-time version
validation removed two, leaving 29 eligible pairs and 24 balanced cases.  They
covered only five tasks, below the frozen eight-task minimum.  The decision is
`INSUFFICIENT_TASK_DISJOINT_COHORT`: this falsifies the breadth of the strict
paired opportunity definition, not the unmeasured two-stage selector.

The independent official task campaign also had no accuracy-identifying power:
Dense, General, and V40 all resolved 0/13.  V40 nevertheless physically copied
196,704 tokens in 253 requests across all 13 tasks with zero fallback, 69.5%
fewer copied tokens than General.

M56 then replayed 383 exact-same-prompt requests, including 244 V40 targets
across all 13 tasks.  All 244 targets physically copied with zero fallback.
Median target TTFT improved from 316.18 ms to 286.74 ms (1.103x); N=4 including
source build remained 1.102x, and first-token agreement was 97.54%.  This is
positive V40 speed/fidelity evidence, not task-accuracy evidence.  The frozen
run used one Dense-then-V40 server order, so a separately registered reverse-
order replication remains necessary before a publication-level speed claim.

## Artifacts

Implementation and tests:

```text
benchmark/multi_workflow/motivate_v50_coding_provenance.py
benchmark/multi_workflow/motivate_v51_file_version_risk.py
benchmark/multi_workflow/motivate_v52_path_dependency.py
benchmark/multi_workflow/motivate_v53_path_dependency_holdout.py
benchmark/multi_workflow/motivate_v54_dependency_drift_hybrid.py
benchmark/multi_workflow/motivate_v55_two_stage_selector.py
benchmark/multi_workflow/run_m55_v40_task_disjoint_campaign.py
benchmark/multi_workflow/run_m56_v40_same_prompt_replay.py
benchmark/multi_workflow/audit_algorithm_evidence_matrix.py
benchmark/multi_workflow/test_motivate_v50_coding_provenance.py
benchmark/multi_workflow/test_motivate_v51_file_version_risk.py
benchmark/multi_workflow/test_motivate_v52_path_dependency.py
benchmark/multi_workflow/test_motivate_v53_path_dependency_holdout.py
benchmark/multi_workflow/test_motivate_v54_dependency_drift_hybrid.py
benchmark/multi_workflow/test_motivate_v55_two_stage_selector.py
benchmark/multi_workflow/test_run_m55_v40_task_disjoint_campaign.py
benchmark/multi_workflow/test_run_m56_v40_same_prompt_replay.py
benchmark/multi_workflow/test_audit_algorithm_evidence_matrix.py
```

Results:

```text
/home/gfy/CodeMAS_Project/kvflow-artifacts/impactkv_m50_coding_provenance_20260805/matched20
/home/gfy/CodeMAS_Project/kvflow-artifacts/impactkv_m51_file_version_risk_20260805/matched18_v2
/home/gfy/CodeMAS_Project/kvflow-artifacts/impactkv_m52_path_dependency_20260805/matched20
/home/gfy/CodeMAS_Project/kvflow-artifacts/impactkv_m53_path_dependency_holdout_20260805/request_disjoint19
/home/gfy/CodeMAS_Project/kvflow-artifacts/impactkv_m54_dependency_drift_hybrid_20260805/untouched14
/home/gfy/CodeMAS_Project/kvflow-artifacts/impactkv_m55_v40_task_disjoint_20260805
/home/gfy/CodeMAS_Project/kvflow-artifacts/impactkv_m55_two_stage_20260805/fresh13
/home/gfy/CodeMAS_Project/kvflow-artifacts/impactkv_m56_v40_same_prompt_20260805/fresh13
/home/gfy/CodeMAS_Project/kvflow-artifacts/impactkv_algorithm_evidence_matrix_20260805_final
```

Verification: 35 focused tests pass across the motivation, evidence-audit,
fresh-campaign, and same-prompt replay helpers.  No V40/V46 runtime code, paper,
prefetch branch, old dirty checkout, or prior preregistration threshold was
modified.
