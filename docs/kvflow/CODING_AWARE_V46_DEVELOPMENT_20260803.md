# Coding-aware lossy KV reuse V46

This document is the review entry point for V46. It explains the research
idea, the implementation in SGLang, the differences from V40, the evidence we
have, and the claims the evidence does **not** support. It is written for a
collaborator who knows SGLang but has not followed the V40--V46 experiments.

## Review coordinates

| Item | Value |
|---|---|
| Review branch | `research/coding-aware-v45-multi-observation-20260803` |
| V46 implementation commit | `f940fe76e` |
| Accuracy bookkeeping and result commit | `1887e69ab` |
| Runtime arm | `coding_observed_path_pool_v46` |
| Prefetch | disabled |
| Ordinary Radix prefix reuse | disabled in controlled V46 experiments |
| Maximum live coding sources | 3 |
| Maximum copied islands per target | 3 |

The branch name contains `v45` because V46 was developed as the lifecycle-safe
continuation of the V45 target-version guard. The implemented and evaluated
runtime arm is V46.

## Executive summary

V40 proved that a coding agent can reuse the KV of one old, successful,
read-only repository observation instead of copying an arbitrary prompt tail.
Its main limitation is opportunity: it keeps only one transient source and can
copy only one island on the next request.

V46 keeps a bounded pool of up to three repository observations and can copy
up to three non-overlapping islands in one later prompt. It also strengthens
provenance: paths are extracted from both the tool command and its output, and
every source is revalidated against the current rolling history immediately
before it is used. Persistent source leases are explicitly invalidated,
evicted, or released at session reset.

The speed improvement is real. On the controlled 50-case RepoBench-P
mechanism test, cache-ready TTFT improves from V40's 1.089x to V46's 1.326x
relative to SGLang Dense. V46 is also above break-even at four target uses when
source construction is included (1.050x).

The current accuracy evidence is negative for promotion. V46 produces 4/50
exact next lines, versus 5/50 for its Dense lane, and preserves only 2/3 of a
small frozen set of SWE-bench tasks previously solved by Dense and V40. The
full-12 campaign was not promoted after its canary failed. V46 is a faster
mechanism, not an accuracy-ready final method.

## The problem V46 is trying to solve

A coding-agent prompt grows by repeatedly adding assistant commands and tool
results. A repository read such as the output of `sed`, `cat`, `rg`, or `find`
may remain verbatim in several later prompts. Dense inference recomputes KV for
that old text on every request.

General middle-span reuse can copy any repeated prompt span, but it does not
know whether the span describes code that has since changed. V46 uses coding
semantics to answer two questions:

1. Which repeated spans are grounded, read-only repository observations?
2. Is each observation still valid in the repository version visible in the
   current agent trajectory?

V46 does not predict the solution patch and does not inspect the reference
answer or evaluator result. Its decision uses only commands, tool outputs, and
history already visible to the online agent.

## Plain-language example

Suppose an agent performs these interactions:

1. `sed -n '1,240p' src/parser.py` returns a long source listing.
2. `rg 'load_config' -n src config` returns paths and matching lines.
3. `cat config/default.toml` returns a configuration file.
4. The next model request contains all three old tool outputs.

V40 can retain only one of these observations for one next request. V46 may
retain all three, verify that the referenced files have not been changed, and
copy three separate KV islands. Prompt text before, between, and after those
islands is still computed densely.

If a later `apply_patch` changes `src/parser.py`, the parser observation is
invalidated. A direct read of `config/default.toml` may remain valid if the
mutation is clearly disjoint. A recursive `find` or `grep -R` observation is
repository-scoped, so any later repository mutation invalidates it. Ambiguous
paths, duplicated observations, overlapping islands, and missing source KV all
fail closed.

## Algorithm, step by step

### 1. Form rolling coding groups

The bridge groups an assistant tool call and its tool result into one completed
coding interaction. The default online window keeps six recent groups. When a
full window rolls forward, the oldest group is excluded from future-source
selection because it will not exist in the next prompt.

Only completed history is considered. Assistant reasoning, future messages,
reference patches, and evaluator outcomes are never candidate KV sources.

### 2. Admit grounded read-only observations

`is_successful_readonly_evidence()` applies a mechanical, answer-blind filter:

- the command must contain a supported read/search operation such as `rg`,
  `grep`, `find`, `sed`, `cat`, `head`, or `tail`;
- execution, test, diff, and mutation commands are rejected;
- all visible return codes must be zero;
- the tool observation must contain at least 400 characters;
- only tool-result messages become the copied span; assistant reasoning and
  tool-call tokens remain Dense.

The policy extracts path provenance from both the command and the returned
text. This is important for `find` and search commands: their command line may
name only a directory while their output identifies the actual files the
model observed.

### 3. Apply source-time version checks

For each candidate observation, V46 scans later completed groups already in
the rolling window:

- a later write to the same path invalidates the observation;
- an unlocalized or ambiguous write fails closed;
- a repository-scoped search is invalidated by any later repository mutation;
- the existing symbol extractor is telemetry only; same-file,
  symbol-disjoint relaxation is deliberately disabled.

An observation without localized path evidence is not registered.

### 4. Register a persistent KV source

The exact tool-result text is rendered with the active chat template and
tokenized. It must occur exactly once inside its coding group, have at least
`reuse_min_tokens` tokens, and be strictly inside the prompt rather than at the
prefix or suffix boundary. The tail of the observation is capped by
`reuse_copy_cap`.

The bridge records an auditable identity containing:

- source ID and request index;
- source prompt and prefix hashes;
- exact segment token hash and content hash;
- full coding-group hash and tool-observation hash;
- observed paths, visible symbols, and repository-scope dependency;
- source start, length, model identity, and cache dtype.

The source is marked persistent and materialized by the local SGLang manifest
controller. Persistent here means reusable across later target requests, not
immortal: the policy can invalidate it and the bounded pool can evict it.

### 5. Revalidate every source at target time

Before planning a target, `observed_path_target_guard()` rechecks each pool
entry against the **current** rolling history:

1. the original coding group must occur exactly once;
2. its tool-observation hash must still match;
3. its paths must still be localized;
4. every later repository mutation is evaluated again;
5. the exact segment must occur exactly once inside the matched group.

Any failed check removes and releases the source. This target-time guard closes
the one-request gap between selecting a source and consuming it.

### 6. Select up to three target islands

Valid target candidates are ranked by copied length and then recency. V46
greedily keeps at most three non-overlapping candidates and finally orders them
by target position. Each island must be strictly in the middle of the target.

The pool update protects every source referenced by the current target before
ranking new future sources. The remaining slots are filled by the longest,
most recent candidates. This ordering fixes a lifecycle bug found in the first
agent canary, where registering a future source evicted a source already
referenced by the same request.

In compact pseudocode:

```text
pool = validate_current_version(pool, rolling_history)
targets = longest_recent_nonoverlap(pool, limit=3)

new_sources = grounded_observed_path_candidates(rolling_history_for_next_turn)
protected = source_ids(targets)
next_pool = protected + longest_recent(pool + new_sources, remaining_slots)

atomic_manifest_append(
    target_islands=targets,
    new_sources=next_pool - pool,
    releases=pool - next_pool,
)
```

### 7. Alternate Dense prefill and physical KV copy

The SGLang controller attaches all islands for one target as an ordered bundle.
The scheduler then executes this sequence:

```text
Dense prefix -> copy island 1 -> Dense gap -> copy island 2
             -> Dense gap -> copy island 3 -> Dense suffix
```

For a copied span whose source starts at position `s` and target starts at
position `t`, V46 copies V unchanged and adjusts K by the RoPE position delta
`t - s`. Mechanical validation requires the planned token hashes, positions,
model identity, cache dtype, source handle, and copied-token counts to match.
Otherwise the request falls back to Dense.

No prefetch participates in this path. Controlled measurements also disable
ordinary Radix prefix reuse, so the reported saving comes from middle-span KV
copy rather than hidden prefix warming.

## Why this is still lossy reuse

The visible tokens inside an island are identical, but their source KV was
computed under an older prefix context. Let `x` be the repeated token span,
`C_s` its source prefix, and `C_t` its target prefix. Dense target inference
would compute:

```text
KV_dense = TransformerKV(C_t, x)
```

V46 instead uses the cached hidden state from:

```text
KV_reuse = TransformerKV(C_s, x)
```

RoPE adjustment corrects K's absolute position, but it cannot reconstruct the
different attention history induced by `C_t`. Therefore `KV_reuse` is not
generally equal to `KV_dense`. V46 is exact in token identity and lossy in
contextual hidden state. It is neither exact-prefix reuse nor prefetch.

## V40 versus V46

| Dimension | V40 | V46 | Consequence |
|---|---|---|---|
| Runtime arm | `coding_grounded_observation_island_v40` | `coding_observed_path_pool_v46` | Separate, selectable policies |
| Source type | Successful read-only tool observation | Same grounded source class | Both are coding-aware and exclude reasoning |
| Path provenance | Primarily literal paths in command text | Command plus paths printed in tool output | V46 understands `find`/search outputs better |
| Search scope | No explicit repository-scope dependency | Recursive search invalidated after any mutation | Stronger stale-evidence protection |
| Source-time validity | Checks writes visible when source is selected | Stronger observed-path check | V46 rejects unlocalized evidence |
| Target-time validity | No dedicated next-request evidence revalidation | Group identity, observation hash, paths, and later writes rechecked | Closes the selection-to-use gap |
| Live source state | One transient pending source | Persistent bounded pool of at most 3 | More reuse opportunities |
| Source lifetime | Normally one next target | Multiple later targets until invalidation, eviction, or session reset | Amortizes source construction |
| Islands per target | At most 1 | At most 3, ordered and non-overlapping | More copied tokens and more accuracy risk |
| Selection | Largest eligible observation, recency tie-break | Longest/recent pool entries, greedy non-overlap | V46 can combine independent reads |
| SGLang request state | Single case/source/lease | Bundled cases/sources/leases with island cursor | Enables Dense gaps between copies |
| K/V operation | RoPE-shift K, copy V | Same operation for every island | Loss model is unchanged per island |
| Prefetch | None | None | Speed is reuse-only |
| Ordinary prefix reuse in controls | Disabled | Disabled | Middle-copy speed is isolated |

V45 is the bridge between the two: it preserved V40's single-candidate order
and added target-time file-version validation. V46 retains that safety idea,
widens path provenance, and adds the persistent multi-source/multi-island
runtime.

## Implementation map

| Layer | Main implementation | Responsibility |
|---|---|---|
| Coding policy | `benchmark/multi_workflow/coding_reuse_policy.py` | Read-only admission, path/scope provenance, version invalidation, target guard |
| Agent bridge | `benchmark/multi_workflow/bridge_reuse_litellm_model.py` | Pool ranking, token-span identity, target bundles, atomic source/release updates |
| Experiment registration | `benchmark/multi_workflow/run_bridge_reuse_agent_experiment.py` | Frozen agent protocol and telemetry requirements |
| SGLang executor | `python/sglang/srt/mem_cache/kvcomm_exact.py` | Persistent sources, leases, bundled islands, RoPE-corrected K/V transfer, fallback |
| Scheduler integration | `python/sglang/srt/managers/schedule_batch.py`, `schedule_policy.py` | Drain adjacent ready islands and stage intervening Dense ranges |
| Offline lifecycle audit | `benchmark/multi_workflow/audit_v46_runtime_parity.py` | Replay production planner without model/GPU and detect release conflicts |
| Static speed/quality control | `benchmark/multi_workflow/run_v46_repobench_control.py` | Matched Dense/V46 RepoBench-P mechanism test |
| Unit coverage | `test_coding_reuse_policy.py`, `test_bridge_reuse_litellm_model.py`, `test_kvcomm_exact.py` | Policy, bridge, lifecycle, bundle, and copy invariants |

The dynamic manifest is append-only and local. External HTTP requests cannot
choose arbitrary spans. The scheduler reloads the sidecar and validates hashes
and bundle ordering before attaching reuse state.

## Evidence hierarchy

The results below answer different questions and must not be merged into one
accuracy claim.

### A. Offline production-planner replay: opportunity and lifecycle only

Frozen V40 agent trajectories were replayed through the production V46 bridge:

- 331 requests and 1,064,801 prompt tokens;
- 236 requests contain at least one planned copy;
- 303,600 tokens are copied, or 28.51% of prompt tokens;
- no target references a source released in the same atomic update;
- at most three live sources and three target islands.

This run issues zero model and zero GPU requests. It establishes opportunity
and planner/lifecycle consistency, not latency or task accuracy.

### B. RepoBench-P 50-case control: mechanism, TTFT, and next-line quality

All 50 V46 targets executed three physical copies: 150/150 copy events, 1,536
copied tokens per target, and zero fallback. In this static control, the runner
selects three unique, non-overlapping repository-context islands. It exercises
the V46 SGLang executor and loss mechanism, but it does not by itself validate
the online agent's path-selection policy.

| Method | Cache-ready | N=4 including build | N=16 including build | Exact line | Code similarity |
|---|---:|---:|---:|---:|---:|
| SGLang Dense | 1.000x | 1.000x | 1.000x | 5/50 | 49.99% |
| V40 | 1.089x | 0.897x | not measured | 4/50 | 52.32% |
| V46 | **1.326x** | **1.050x** | 1.244x | 4/50 | 52.54% |
| CacheBlend | **1.501x** | 0.827x | 1.247x | 4/50 | **55.66%** |
| KVCOMM | 13.849x | 8.636x | 12.033x | 5/50 | 56.54% |

Definitions:

- **Cache-ready** excludes construction of the source KV and isolates target
  request TTFT.
- **N=4/N=16 including build** amortizes one source-construction cost over four
  or sixteen target uses.
- **Exact line** requires the predicted next line, after outer whitespace
  normalization, to match the reference line.
- **Code similarity** is the mean character-sequence similarity between the
  predicted and reference next lines. It is descriptive and is not task
  completion accuracy.

For V46, mean Dense TTFT is 286.67 ms, reuse TTFT is 216.22 ms, and mean source
construction is 226.76 ms. Source construction breaks even after about 3.22
target uses. Compared with Dense, 42/50 outputs are identical, four improve in
similarity, and four regress. One Dense exact pass is lost and no exact pass is
gained.

CacheBlend and V46 share frozen target IDs but run on their native engines and
are normalized to their respective native Dense lanes. CacheBlend's native
Dense lane is 5/50 exact with 51.44% similarity. KVCOMM changes the native
multi-agent prompt and token IDs and copies roughly 18K tokens per target; its
large speedup is descriptive and is not controlled-rankable against V46.

### C. Official SWE-bench preservation: task accuracy, but a tiny cohort

The strongest accuracy metric is the official SWE-bench `resolved` result,
which executes the submitted patch in the official task container.

| Task | V46 physical copies | Copied tokens | Fallback | V46 official result |
|---|---:|---:|---:|---|
| `astropy__astropy-7671` | 0 | 0 | 0 | resolved |
| `pytest-dev__pytest-10051` | 54 | 30,223 | 0 | **unresolved** |
| `scikit-learn__scikit-learn-10297` | 33 | 21,599 | 0 | resolved |

V46 preserves 2/3 tasks, compared with Dense 3/3, V40 3/3, and the historical
General arm 2/3. Only two tasks actually consume copied KV; one passes and one
fails. The failing task has a nonempty submitted patch and no runtime fallback,
so the failure is not explained by a missing source or empty-submission bug.

This is a preservation characterization, not an unbiased population estimate:
the three tasks were selected because they were prior Dense/V40 passes, and
the cohort registration occurred after two V46 outcomes were already visible.
It can falsify the claim that V46 trivially preserves known passes; it cannot
estimate general SWE-bench accuracy with useful confidence.

An earlier empty terminal-submission bookkeeping issue was fixed by capturing
an existing tracked `git diff` only when mini-SWE-agent submits an empty result.
The fix makes no additional model request and never replaces a nonempty model
submission. Its regression suite passes.

### D. Full-12 promotion status

No valid full-12 campaign completed. An initial launch processed one task, but
that output was abandoned after exposing the empty terminal-submission
bookkeeping problem described above. After the fix, the registered combined
canary included `pytest-dev__pytest-10051`, which failed official accuracy. The
full-12 campaign was therefore not restarted. There is no honest full-12 V46
accuracy number to report.

## Current conclusion

What is established:

- V46 physically performs multi-island, shifted, lossy KV reuse in SGLang.
- The production planner exposes substantially more copy opportunity than V40.
- Persistent source lifecycle and current-target protection work with zero
  fallback in the repaired canaries and the 50-case static control.
- V46 improves cache-ready speed and four-use amortized speed over V40.

What is not established:

- V46 does not currently beat CacheBlend on cache-ready speed or next-line
  similarity.
- V46 does not preserve all known Dense/V40 SWE-bench passes.
- KVCOMM has not yet been compared under identical prompts and token IDs.
- No population-level SWE-bench, multi-dataset, or SOTA claim is justified.

The likely failure mode is no longer lack of reuse opportunity. V46 copies up
to three contextual hidden-state approximations simultaneously. Path-version
validity proves that the visible repository text is not stale; it does not
prove that KV computed under an older reasoning prefix is harmless for the
current decision. More copied islands increase speed but also compound this
context mismatch.

## Next admissible development step

The next version should add an answer-blind target quality guard before
changing the low-level transfer mechanism. Candidate inputs include:

- total copied-token budget and copied fraction of the prompt;
- number and age of simultaneous islands;
- repository-scope versus direct-file provenance;
- distance from the most recent mutation, test failure, or patch inspection;
- whether the current turn is in an edit/validation/commit-sensitive phase.

The guard should reduce island count or copied-token budget on risky targets,
not switch to exact reuse or prefetch. It must first repeat the same Dense-pass
preservation gate. Only after passing that accuracy gate should it run a larger
SWE-bench cohort and a matched CacheBlend comparison.

## Reproduction and artifacts

Core regression suites:

```bash
cd /home/gfy/CodeMAS_Project/sglang-kvflow-worktrees/coding-aware

PYTHONPATH=.:python:/home/gfy/.venvs/mini-swe-agent-v2.3.0/lib/python3.12/site-packages \
  /home/gfy/.conda/envs/sglang-kvflow/bin/python -m pytest -q \
  benchmark/multi_workflow/test_coding_reuse_policy.py \
  benchmark/multi_workflow/test_bridge_reuse_litellm_model.py \
  benchmark/multi_workflow/test_run_swebench_with_limit_patch_capture.py

# Keep mini-SWE-agent's site-packages out of the SGLang runtime test. Its
# tokenizers dependency is intentionally isolated from the SGLang environment.
PYTHONPATH=python \
  /home/gfy/.conda/envs/sglang-kvflow/bin/python -m pytest -q \
  python/sglang/srt/mem_cache/test_kvcomm_exact.py
```

Human- and machine-readable V46 speed/accuracy summary:

```text
/home/gfy/CodeMAS_Project/kvflow-artifacts/
  impactkv_v46_accuracy_speed_20260803/
    V46_ACCURACY_SPEED_RESULT.md
    V46_ACCURACY_SPEED_RESULT.json
    repobench_full50/RESULT.json
```

Earlier mechanism and lifecycle artifacts:

```text
/home/gfy/CodeMAS_Project/kvflow-artifacts/
  impactkv_v46_observed_path_runtime_20260803/
  impactkv_v46_agent_canary_20260803/
  impactkv_v46_agent_canary_fix1_20260803/
```

Large artifacts remain outside Git. This branch does not modify the paper,
historical dirty checkout, or existing preregistration thresholds.
