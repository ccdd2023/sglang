# Coding-aware V45: target-time version evidence

Date: 2026-08-03

Branch: `research/coding-aware-v45-versioned-evidence-20260803`

Parent: V40 research tip `525a03c6b`

## Outcome first

V45 found and closes one real V40 safety gap: V40 checks whether a repository
observation is current when it registers a cache source, but it did not check
again after the next tool interaction and before consuming that source. A
write in that interval could make the pending KV evidence stale.

The proposed symbol-level expansion did not earn promotion. Across 12 frozen
V40 trajectories and 270 rolling windows, an experimental symbol-aware policy
admitted exactly the same 662 candidate instances as V40. It recovered zero
same-file, different-symbol observations. Consequently the active V45 arm does
not use symbol disjointness to admit or retain reuse.

V45 is therefore a strict V40 continuation, not a CacheBlend variant:

1. Keep V40's grounded, tool-observation-only source policy.
2. Require a localized repository path for every V45 source.
3. Bind the source to its observation hash and visible file evidence.
4. Revalidate that evidence immediately before the next target reuse.
5. Release the source and run Dense for that span after any overlapping or
   unlocalized write. Even an explicitly different symbol does not bypass the
   active guard.

No V45 GPU speed or accuracy claim is made yet. The combined symbol-and-guard
proposal failed its promotion gate and was narrowed. A second, production
planner audit shows that the guard-only V45 is eligible for a separately
registered Dense/V40/V45 GPU canary.

## The V40 gap in a concrete example

Suppose the agent completes these interactions:

```text
request q:     read pkg/parser.py -> tool returns the Parser implementation
after q:       agent patches pkg/parser.py
request q+1:   the old read text is still present in the rolling prompt
```

V40 could register the read at request `q`, before the patch existed. At
request `q+1`, the token text still matched, so the pending source could pass
the segment-match checks even though its coding evidence referred to the old
file version.

V45 records that the source came from `pkg/parser.py`. Before registering the
target case at `q+1`, it locates the exact tool observation in the current
rolling groups and scans every later coding interaction. The patch overlaps
the source path, so V45 releases the pending source. The ordinary model prompt
is unchanged; only the unsafe KV copy is omitted.

## Active V45 algorithm

The configured arm is `coding_versioned_evidence_guard_v45`.

### Source registration

The bridge keeps the same six-interaction rolling history as V40. When a
future request will roll out the oldest interaction, V45 considers the other
five groups and applies V40's source rules:

- the interaction must be a successful, substantial, read-only repository
  observation;
- only tool observations are eligible—assistant reasoning and tool-call text
  are not copied;
- a later visible same-file mutation in the retained window invalidates the
  observation;
- validation commands, failures, diffs, and mutation outputs are not source
  observations.

The bridge first applies V40's exact largest-candidate rule, 128-token minimum,
4096-token cap, unique occurrence requirement, and strict-middle requirement.
V45 then abstains if that selected observation has no localized path because
it cannot be checked against later writes. It does not choose a pathful
runner-up; this ensures every V45 target is either the same V40 token segment
or no reuse at all.

The pending source stores local-only evidence:

```text
tool-observation SHA-256
repository path(s)
visible Python symbols (telemetry only)
```

These fields do not change the runtime manifest schema.

### Target-time guard

Before consuming a pending source, V45:

1. finds exactly one current rolling group with the recorded observation hash;
2. rejects missing or duplicate evidence;
3. scans all later repository mutation events, including shell redirection
   writes that older mutation classifiers did not cover;
4. permits writes to disjoint files;
5. rejects an overlapping file write, an unlocalized write, or any ambiguity;
6. releases the rejected source through the existing sidecar lifecycle.

Only after this guard passes does the existing exact token-segment check run.
The SGLang runtime then performs the same V40 middle-span KV operation: K is
position-adjusted for the target location, V is copied, and the surrounding
prefix/suffix is computed normally. This remains lossy contextual reuse; it is
not exact prefix caching and it adds no prefetch request.

## What the frozen offline audit measured

Input: the 12 `coding_grounded_observation_island_v40` trajectories from the
V44 development cohort. The audit used only prompts, tool commands, and tool
outputs already visible online. It did not use reference patches, evaluator
answers, new model calls, or GPU inference.

| Quantity | Result |
|---|---:|
| V40 trajectories | 12 |
| Six-group source windows | 270 |
| V40 candidate instances | 662 |
| Experimental symbol-aware candidate instances | 662 |
| Candidates recovered by symbol disjointness | 0 |
| Candidate next-target checks | 640 |
| Next-target checks rejected | 71 |
| Rejected because observation was duplicated | 55 |
| Rejected after ambiguous same-file write | 13 |
| Rejected after explicit same-symbol write | 3 |

The 55 duplicate-observation cases are not evidence of a new version bug: the
existing unique token-segment gate would also reject them. The remaining 16
cases establish the cross-request write window at candidate level. A further
tokenizer-accurate audit is required to determine how many correspond to the
single candidate V40 would actually register and later physically copy.

That second audit was then completed using the production Qwen tokenizer,
chat template, rolling compaction, and the bridge's real
`prepare_reuse_query` planner on identical frozen trajectory prompts:

| Planner result | V40 | narrowed V45 |
|---|---:|---:|
| Registered sources | 213 | 199 |
| Runtime-eligible targets | 203 | 183 |
| Planned copied target tokens | 171,139 | 157,516 |

V45 retained 90.1% of V40 target opportunities. It removed 20 V40 targets:
eight because a same-file write became visible before target reuse, and twelve
because the V40-selected source had no safe V45 pending source. All 183 shared
targets used exactly the same segment, V45 introduced no target absent from
V40, and every prompt hash was identical. The initial exact-planner audit had
one different shared segment because pathless observations were filtered
before ranking; that confound was recorded, corrected by strict V40-first
selection, and rerun under an explicit amendment rather than overwritten.

Frozen artifacts:

- `/home/gfy/CodeMAS_Project/kvflow-artifacts/impactkv_v45_versioned_evidence_20260803/V45_REGISTRATION.json`
- `/home/gfy/CodeMAS_Project/kvflow-artifacts/impactkv_v45_versioned_evidence_20260803/V45_MOTIVATION_RESULT.json`
- `/home/gfy/CodeMAS_Project/kvflow-artifacts/impactkv_v45_versioned_evidence_20260803/V45_STRICT_SOURCE_AUDIT_REGISTRATION.json`
- `/home/gfy/CodeMAS_Project/kvflow-artifacts/impactkv_v45_versioned_evidence_20260803/V45_SELECTED_TARGET_STRICT_RESULT.json`

## Claims that are not supported

- V45 has not yet improved SWE-bench accuracy.
- V45 has not yet improved TTFT or end-to-end latency.
- The prototype Python-symbol parser is not a general language-aware version
  graph, and symbol relaxation is disabled in the active arm.
- Candidate-level offline counts are not physical-copy counts.
- This audit does not compare V45 with KVCOMM or CacheBlend.

## Next registered milestone

The tokenizer-accurate prerequisite passed: eight actual V40-planned targets
were removed because of a newly visible same-file write, while 90.1% of target
opportunities remained. The next milestone may therefore register a small
Dense/V40/V45 GPU canary on the affected frozen tasks. It must use identical
prompts and report task accuracy, physical copy count, copied tokens,
cache-ready TTFT, and build-inclusive latency separately. The canary must not
be described as a KVCOMM or CacheBlend comparison; those remain later matched
native-engine baselines after V45 demonstrates a useful V40 delta.

Development remains confined to this SGLang branch. The old dirty checkout,
paper tree, prefetch branch, and existing preregistration thresholds are out
of scope.
