# M48–M49: model-internal risk motivation and cheap-proxy falsification

Date: 2026-08-05

Branch: `research/coding-aware-v45-multi-observation-20260803`
Decision: keep the oracle finding as motivation; do not add the tested probe to
SGLang

## Why these experiments were run

M47 showed that a cursor-local lexical selector did not improve V46 quality.
The next question was whether the model's own state contains a more causal
signal of lossy-reuse risk:

1. Does target-query attention combined with cross-prefix K/V drift explain
   which observation island changes the model most?
2. If yes, can that oracle be approximated before copying by recomputing only
   a small head of each island?

These are two separate questions.  M48 answers the first with a full Dense
target oracle.  M49 freezes a cheap approximation on M48 development data and
tests it once on 50 disjoint cases.

## M48 oracle experiment

### Measurement

For every eligible 512-token repository observation, M48 computes:

- target attention mass from the last 32 prompt query tokens;
- RoPE-corrected source-versus-target K cosine drift;
- source-versus-target V cosine and relative-L2 drift;
- a head-aware `attention × max(K drift, V drift)` score;
- the causal final-logit JS after physically splicing that one source island
  into the target and recomputing the suffix;
- the corresponding answer-first-token NLL delta.

Layers 1, 9, 18, 27 and 36 (zero-based 0, 8, 17, 26, 35) are sampled.  V46's
three selected islands are also composed left-to-right, so the request-level
label includes interaction between copied islands rather than summing three
independent single-island labels.

This uses full Dense target K/V.  It is an oracle motivation measurement, not
an online controller and not a latency result.

### Full-50 result

There are 294 equal-length candidate observations across 50 cases.

| Signal | Global Spearman with single-island JS | Mean within-case Spearman | Fixed-position Spearman |
|---|---:|---:|---:|
| Attention only | 0.158 | 0.164 | 0.138 |
| Mean K/V cosine drift | 0.526 | **0.577** | 0.156 |
| Mean attention × drift | **0.570** | 0.545 | **0.257** |
| q90 attention × drift | 0.569 | 0.528 | 0.251 |
| Position fraction | -0.503 | -0.550 | -0.052 |

The product improves the global and position-controlled relationship over
attention or K/V drift alone.  Layer 18 is the strongest single sampled layer:
its mean attention×drift has Spearman 0.590 with causal JS.

At V46 request level, the maximum q90 oracle risk among three selected islands
has Spearman 0.469 with composed-logit JS and 0.327 with answer-first-token NLL
delta.  The only M47 case where V46 damages a Dense exact-line pass is the
highest-risk request among all 50 M48 cases.  This is promising causal
motivation, but request-level separation is moderate rather than perfect.

## Why M48 cannot be implemented directly

Deep-layer target attention from the completion query and full target K/V are
available only after densely computing the relevant target.  Paying that cost
would erase the reuse speedup.  M48 therefore supports the existence of a risk
signal, not the deployability of its exact formula.

Layer-1 attention is cheap because it can be formed from token embeddings, but
it is not useful here: layer-1 attention×drift has Spearman -0.137 with causal
JS.  The useful attention geometry appears only after contextual processing.

## M49 cheap-probe experiment

### Frozen approximation

M49 tests a ProbeHead-style online approximation.  Before copying the body of
an otherwise valid V46 observation, recompute its first `H` target tokens,
compare their K/V with the cached source, and score the island as:

```text
max(mean RoPE-corrected K cosine deviation,
    mean V cosine deviation)
```

Development searches only:

- zero-based layers 8, 17 and 26;
- `H ∈ {8, 16, 32, 64}`.

The registered selection rule maximizes request-level correlation with
three-island composed JS.  Configurations within 0.02 of the best use the
smallest H, then the shallower layer.  This locks:

| Parameter | Frozen value |
|---|---:|
| Layer | zero-based 17 / human layer 18 |
| Probe head | 16 token per 512-token island |
| Probe fraction | 3.125% |
| Request score | maximum island score among V46's three islands |
| High-risk threshold | 0.0114773393, development p90 |

The independent validation set is the first 50 eligible cases in source order
from the existing `v66_final_repobench_holdout100` workload.  Its case-ID
overlap with M48 development is zero.  The layer, H, score, threshold and gates
were locked before holdout GPU execution.

### Independent holdout result

| Metric | Development | Independent holdout | Frozen gate |
|---|---:|---:|---:|
| Single-island global JS Spearman | 0.469 | **0.530** | descriptive |
| Single-island mean within-case JS Spearman | 0.518 | **0.489** | ≥0.30 |
| Request three-island composed-JS Spearman | 0.305 | **0.193** | ≥0.30 |
| Request composed-NLL Spearman | 0.239 | 0.169 | descriptive |
| High-risk / low-risk mean composed-JS ratio | — | **1.281** | ≥1.50 |

Only the single-island gate passes.  The two request-level gates fail.  The
frozen threshold marks four requests high-risk; their mean composed JS is
0.002004 versus 0.001564 for the other 46.  Several of the largest causal-harm
requests remain below threshold.

The final registered result is `FALSIFIED_PROXY`.

## What the failure means

The failure does not contradict M48.  It localizes the gap:

- a short layer-18 K/V probe can rank **individual island** risk;
- taking the maximum of three independent island scores cannot reliably
  predict **multi-island composed** harm;
- request harm includes interaction and accumulation after the first stale
  island changes the states used to process later gaps/islands;
- threshold tuning cannot manufacture the missing interaction signal.

This is also why adding this probe directly to SGLang would be premature.  It
would spend 16 Dense tokens per island while failing the metric that actually
controls V46's three-island admission decision.

## Development decision

Adopt:

1. M48's model-internal K/V drift is legitimate causal motivation.
2. Layer 18 is a useful place to study cross-prefix divergence.
3. A 16-token probe has real per-island ranking value and may be reused in a
   future *per-island* budget policy.

Reject for runtime:

1. `max(probe risk over three islands) >= threshold => abstain`;
2. further post-hoc adjustment of the 0.011477 threshold;
3. presenting M48 oracle measurement as online overhead.

The next experiment must test a causal **per-island intervention**, for example
dropping the highest-risk island while retaining the other two, and must
measure all three two-island combinations because independent single-island
scores may not compose linearly.  That policy must be developed on the already
opened M48/M49 data and validated on a new disjoint cohort, such as the unused
second half of `v66_final_repobench_holdout100`.  Only if that intervention
passes should the 16-token probe be implemented in SGLang and evaluated with
functional accuracy and TTFT.

## Artifacts

Implementation:

```text
benchmark/multi_workflow/motivate_v48_attention_kv_risk.py
benchmark/multi_workflow/motivate_v49_probe_proxy.py
benchmark/multi_workflow/test_motivate_v48_attention_kv_risk.py
benchmark/multi_workflow/test_motivate_v49_probe_proxy.py
```

Results:

```text
/home/gfy/CodeMAS_Project/kvflow-artifacts/
  impactkv_m48_attention_kv_risk_20260805/
    canary8/
    full50/RESULT.json
  impactkv_m49_probe_proxy_20260805/
    REGISTRATION.json
    DEV_PROXIES.jsonl
    PROXY_LOCK.json
    HOLDOUT_OBSERVATIONS.jsonl
    FINAL_RESULT.json
```

No production V46 policy, old dirty checkout, paper, prefetch branch, or prior
preregistration threshold was modified.
