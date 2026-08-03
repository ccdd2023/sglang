# Coding-aware V46 development record

## Outcome

V46 converts V45's single transient observation island into a bounded pool of
up to three persistent coding observations. It reaches substantially more
reuse without prefetch or ordinary exact-prefix reuse, while retaining
target-time file-version checks.

The strongest current evidence is narrow:

- Lifecycle-safe offline replay: 236/331 requests have at least one copy,
  303,600/1,064,801 prompt tokens are copied (28.51%), and no target references
  a source released in the same prepared request.
- Matched three-case RepoBench-P control: 9/9 copy events, zero fallbacks,
  1,536 copied tokens per target, 1.309x cache-ready TTFT speedup, and 1.059x
  speedup at four target uses including source construction. The generated
  line is identical to Dense on all three cases.
- Repaired SWE-bench canary: 15 physical island copies across 16 requests,
  3,376 copied tokens, zero fallbacks, and the official evaluator resolves
  `pytest-dev__pytest-7982` (1/1).

These are mechanism and canary results. Three static completions and one agent
task do not establish population accuracy or superiority over CacheBlend.

## What is coding-aware

V46 does not copy an arbitrary old tail. It admits only completed tool
interactions that contain a substantial read-only repository observation.
The online policy extracts literal source/config/document paths from the
command and from that command's tool output. This matters for commands such as
`find`, `grep`, and compound shell reads whose command text alone does not
identify every observed file.

Each admitted source stores:

- the exact token slice and its prompt/token hashes;
- the observation group's content identity;
- observed repository paths and code symbols;
- whether the observation depends on a repository-wide search;
- the repository request/version boundary at which it was observed.

At every later target, the policy locates the same observation group in the
current rolling prompt and scans later tool interactions for writes. A
same-file write invalidates the observation unless the existing conservative
symbol evidence proves it safe. A repository-wide `find` or recursive `grep`
observation invalidates after any repository mutation. Missing or ambiguous
group identity also abstains.

Among valid sources, the bridge selects at most three token islands in prompt
order. Islands must be exact token slices, strictly in the middle of the
target prompt, and non-overlapping. All nonselected tokens are recomputed
dense. K is RoPE-adjusted from its source position to the target position; V
is copied without rotation. This remains lossy because the copied hidden KV
states came from a different prefix context even though the island's visible
tokens are identical.

## Runtime changes

The SGLang exact-middle controller now accepts a target bundle rather than one
case. The scheduler drains adjacent ready islands with a loop, allowing Dense
prefill gaps between copied islands. Persistent dynamic source leases survive
multiple target uses and are released explicitly when the coding evidence is
invalidated or the bounded pool evicts them.

The first real agent canary exposed a source-lifetime ordering bug: request 7
planned three target islands, then admitted a future source and released one
of those three target sources in the same atomic manifest update. The server
correctly fell back with `missing_source`. V46 now protects all source IDs used
by the current target while ranking the next pool. If all three slots are
protected, a new candidate waits for a later request. The repaired canary has
zero fallback.

## Matched comparison

| Method | Cache-ready speedup | N=4 including build | Quality on three matched cases | Comparison status |
|---|---:|---:|---|---|
| Dense | 1.000x | 1.000x | 56.42% code similarity | SGLang reference |
| V45 | 1.091x | 0.912x | 49.12% | identical V45/V46 target IDs |
| V46 | 1.309x | 1.059x | 56.42%, same output as Dense | controlled SGLang lane |
| CacheBlend | 1.306x | 0.593x | 49.33% vs its Dense 56.67% | same target IDs, native-Dense normalized |
| KVCOMM | 14.234x | 7.987x | native result only | prompt/token IDs differ; not controlled-rankable |

V46 and CacheBlend are effectively tied in cache-ready speed on this tiny
control. V46 has the better result after four-use source-build amortization
and preserved Dense output here, but the sample is too small for a claim.
KVCOMM copies roughly 18K tokens per target in a different native graph and
prompt protocol, so its much larger speedup is descriptive until a common
prompt/backend comparison exists.

## Next admissible experiment

Run a frozen multi-task SWE-bench cohort with paired Dense and V46 under the
same agent protocol. Report official resolved rate, per-request TTFT, end-to-
end task time, copy/fallback telemetry, source-build cost, and trajectory
divergence before the first copy. CacheBlend must be compared through its
native engine normalized to its own Dense lane unless an identical SGLang
implementation is available. Do not promote V46 from canary status until the
cohort has enough tasks to distinguish accuracy from agent nondeterminism.

Artifacts live outside the repository under:

- `kvflow-artifacts/impactkv_v46_observed_path_runtime_20260803/`
- `kvflow-artifacts/impactkv_v46_agent_canary_20260803/`
- `kvflow-artifacts/impactkv_v46_agent_canary_fix1_20260803/`
