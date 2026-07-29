# Coding-aware V40 review request

## Status

This branch is the coding owner's review handoff:

```text
review/coding-aware-v40-prefetch-20260729
```

It is frozen from `research/coding-aware-lossy` at `525a03c6b`. It contains
the current coding-aware method and its tests. It does **not** contain or merge
`research/prefetch-p8-async-20260722`.

Please review this branch before composing it with prefetch. Do not merge
either owner branch into the other owner branch. The accepted coding changes
and the prefetch branch should meet only in a fresh integration branch based
on `kvflow/shared-core`.

## What V40 does

V40 treats recent coding-agent history as versioned repository evidence.
Among the retained interactions, it selects at most one sufficiently large
tool observation that:

1. came from a successful read-only repository command;
2. contains no assistant reasoning or tool-call text;
3. can be located exactly once in the next prompt;
4. has not been invalidated by a later write to the same repository path;
5. remains byte/token identical between source and target.

If path provenance is missing or ambiguous, V40 fails closed and recomputes
the tokens. For the accepted island, the runtime copies V and applies the
source-to-target RoPE position delta to K. All tokens outside that island are
computed densely.

This is lossy attention-context reuse, not prefetch: copied KV came from a
preceding real agent request. V40 does not issue a synthetic replay or warming
request.

## Primary review scope

Please start with these files:

| Area | Files | Review question |
|---|---|---|
| V40 selection | `benchmark/multi_workflow/coding_reuse_policy.py` | Are read-only detection and file-version invalidation fail-closed? |
| Request adapter | `benchmark/multi_workflow/bridge_reuse_litellm_model.py` | Is the chosen observation unique, large enough, and sourced from a real preceding request? |
| Selector tests | `benchmark/multi_workflow/test_coding_reuse_policy.py` | Do tests cover failed reads, writes, path overlap, unknown paths and version invalidation? |
| Adapter tests | `benchmark/multi_workflow/test_bridge_reuse_litellm_model.py` | Do tests cover unique placement, minimum size, copy caps and fallback? |
| Runtime policy seam | `python/sglang/srt/mem_cache/coding_aware/policy.py` | Does policy construction preserve exact identity and dense coverage? |
| Runtime seam tests | `python/sglang/srt/mem_cache/coding_aware/test_policy.py` | Are mismatch, critical-region and head-budget cases dense? |

The active V40 entry points are:

```text
grounded_observation_candidates(...)
reuse_arm="coding_grounded_observation_island_v40"
build_coding_reuse_plan(...)
```

The branch also retains historical experiments needed to explain how V40 was
reached. Those files are evidence, not the proposed composition payload.
Do not merge every V11--V39 driver, report helper or obsolete preregistration
script into the combined implementation merely because it is present here.

## Required invariants

Please block approval if any of these invariants is violated:

- token identity, model identity, source generation or segment bounds can be
  bypassed;
- an unknown path is treated as proof that a read survived a later write;
- a failed or mutating tool interaction can become a reusable observation;
- assistant reasoning tokens are copied by V40;
- prefetch can widen, shrink or replace the V40-selected span;
- missing or late residency does not fall back to V40's declared dense ranges;
- coding-only behavior depends on a hidden cache warm-up.

## Focused verification

From this worktree, run:

```bash
PYTHONPATH=python /home/gfy/.conda/envs/sglang-kvflow/bin/python -m pytest -q \
  python/sglang/srt/mem_cache/coding_aware/test_policy.py \
  benchmark/multi_workflow/test_coding_reuse_policy.py

PYTHONPATH=python:/home/gfy/.venvs/mini-swe-agent-v2.3.0/lib/python3.12/site-packages \
  /home/gfy/.conda/envs/sglang-kvflow/bin/python -m pytest -q \
  benchmark/multi_workflow/test_bridge_reuse_litellm_model.py
```

The split is intentional: the SGLang environment contains `pytest`, while the
mini-SWE-agent environment supplies the adapter's `litellm` and
`minisweagent` dependencies.

Confirm independently that the prefetch tip is not part of this branch:

```bash
git merge-base --is-ancestor \
  research/prefetch-p8-async-20260722 \
  review/coding-aware-v40-prefetch-20260729
test "$?" -eq 1
```

## Evidence interpretation

The current measurements support feasibility, not a final superiority claim:

- RepoBench-P static control: cache-ready TTFT speedup was 1.089x; exact-line
  agreement changed from 10% to 8%.
- Twelve-task SWE-bench development cohort: Dense passed 6/12 and V40 passed
  4/12; median TTFT changed from 295.5 ms to 258.3 ms.
- The two initially observed Dense-pass/V40-fail tasks were not stable under
  Dense repeats, so that single-run gap is not a causal accuracy estimate.
- Native KVCOMM and CacheBlend results are not interchangeable with this
  bridge unless model, prompt, task order, limits and accuracy harness are
  controlled.

The detailed machine-readable audit remains outside Git under:

```text
/home/gfy/CodeMAS_Project/kvflow-artifacts/
  impactkv_three_method_coding_benchmark_20260728/
    THREE_METHOD_AUDIT.json
    THREE_METHOD_AUDIT.md
```

## After approval

Create a new integration branch from `kvflow/shared-core`, bring in the
reviewed active coding payload, then merge the collaborator's prefetch branch
there. Validate four explicit modes with the same task and prompt:

| Mode | Coding-aware | Prefetch |
|---|---:|---:|
| Dense | 0 | 0 |
| Coding-only | 1 | 0 |
| Prefetch-only | 0 | 1 |
| Combined | 1 | 1 |

The combined mode is acceptable only when its selected token span is identical
to coding-only and disabling prefetch restores coding-only behavior exactly.
