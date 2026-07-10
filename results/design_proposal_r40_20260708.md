# R40 Design Proposal — Multi-agent coding task + type-aware KV selection

**Date:** 2026-07-08
**Author:** sglang-kvflow session
**Status:** Awaiting user sign-off

---

## Context

The current verdict-task benchmark (R19-R39, 5×5 = 25 rows, pandas 0.x) has these structural problems:

1. **All 5 agents (implementer/debugger/reviewer/verifier/auditor) see the SAME 5 code slices** and all output `VERDICT: PASS/FAIL`. Real multi-agent coding systems (MetaGPT, SWE-Agent, OpenHands) use different agents doing DIFFERENT tasks, each with a DIFFERENT context slice.
2. **Accuracy is subjective** — 5 agents vote PASS/FAIL; `type_agreement` measures consistency between configs, not correctness against ground truth. Real systems judge by `git apply + pytest` (objective test pass).
3. **KV selection is heuristic** — R38b's "position proxy" (FRAC_EARLY vs FRAC_LATE based on chunk order) is empirically Pareto-optimal but conceptually weak. Real coding-aware signals exist (type signature, call graph, import closure).
4. **TTFT breakdown is estimated**, not measured at sub-stage granularity (radix / chunk_plan / head_recompute_EARLY / gap / head_recompute_LATE / copy).

This proposal addresses all four with concrete code changes + new benchmarks.

---

## Proposal A — New 5-agent distribution (modeled after MetaGPT + SWE-Agent)

**OLD:** 5 agents all see same 5 code slices + output `VERDICT: PASS/FAIL`.

**NEW:** 5 different tasks, each with different context slice:

| # | Role | Task | Context slice | Output |
|---|---|---|---|---|
| 1 | **coder** | Write a patch for the case | full repo state + function signature | `<patch>...</patch>` |
| 2 | **tester** | Run test against the patch | patch + test file | `<test_result>PASS/FAIL</test_result>` (objective) |
| 3 | **reviewer** | Review diff vs original | original code + patched code | `<review>...</review>` + verdict |
| 4 | **refactorer** | Suggest refactoring | patch + style guide | `<refactor>...</refactor>` |
| 5 | **integrator** | Finalize | all 4 prior outputs + test result | `<final_verdict>PASS/FAIL</final_verdict>` |

**Accuracy judge:** `git apply <patch> && pytest <test_file>` → binary pass/fail (matches SWE-bench). Compare lossless vs R38b on this objective metric, not on 5-vote agreement.

**Cross-agent KV reuse:** All 5 agents share the SAME codebase chunks (chunk pool), but each has a different system prompt + different output format → byte-exact L4 chunk reuse works across all 5 agents (different prefix, same chunks). Expected reuse rate similar to verdict task (~19/25 case hit).

**Implementation cost:** ~200 lines in `make_payload` + new `task_mode="coding_pipeline"` + new scorer that runs `git apply` + `pytest`. Estimated 4-6 hours including new benchmark runs.

---

## Proposal B — Type-aware KV selection (replaces position proxy)

**OLD:** `FRAC_EARLY=0.60, FRAC_LATE=0.15, EARLY_N=2` — position-based, principled only as "first few chunks need more recompute."

**NEW:** Per-chunk-type FRAC driven by AST analysis:

- Extract each AST chunk's **type signature** (def name + arg types + return type via `ast.unparse` + `typing.get_type_hints`).
- Compute **type complexity score** per chunk: number of generic types, number of dynamic dispatch sites, recursion depth, polymorphism.
- `FRAC_type_aware = base_FRAC × (1 + type_complexity_weight)`.
- For pandas 0.x (untyped): complexity ≈ 0 for most chunks → reverts to R38b behavior. **No regression on current benchmark.**
- For SWE-bench django/astropy (typed): measurable per-chunk FRAC differentiation.

**Implementation cost:** ~150 lines in `ast_chunker.py` + new env var `SGLANG_CHUNK_TYPE_AWARE_FRAC=1` (default OFF). Need a TYPED codebase to evaluate — would require running on django/astropy instead of pandas 0.x.

---

## Proposal C — Measured TTFT breakdown (instrument, don't estimate)

**OLD:** Slide (6) estimates TTFT segments from `FAIR_SUMMARY.md` + perf counters.

**NEW:** Add per-stage timing instrumentation:

```python
@dataclass
class TTFTBreakdown:
    radix_prefix_ms: float
    chunk_plan_ms: float
    head_recompute_early_ms: float
    head_recompute_late_ms: float
    gap_prefill_ms: float
    copy_ms: float
    decode_first_token_ms: float
```

Log this in every request's `rows.csv`. The `analyze_fair_ab.py` script gets a new `--breakdown` mode that aggregates ms per stage and reports % contribution.

**Implementation cost:** ~50 lines instrumentation + new column in `rows.csv`. **This is the lowest-hanging fruit and gives the user immediate visibility into where time goes.**

---

## Decision matrix

| Proposal | Effort | Reversibility | Risk | Impact |
|---|---|---|---|---|
| A — new agent distribution | 4-6h | reversible (keep verdict mode) | medium (refactor `make_payload`) | high — gets us to objective accuracy |
| B — type-aware FRAC | 3-4h | reversible (default OFF) | low (pandas reverts to R38b) | medium — principled replacement of position proxy |
| C — measured TTFT breakdown | 1-2h | fully reversible | very low (additive logging) | high — clarity on speedup source |

---

## Recommendation

**Start with C (TTFT breakdown)** — lowest cost, highest clarity. Then **A (new agent distribution)** — biggest payoff on accuracy validity. Then **B (type-aware FRAC)** if user wants a more principled replacement for position proxy.

OR: Run all three in parallel as separate worktrees if user wants speed.

---

## What I need from user

1. **Authorize which proposals to implement** (A / B / C / all / none / other)
2. **Choose agent distribution** (above table vs alternative — e.g., 3-agent simpler pipeline, or keep verdict task as sub-mode)
3. **Choose accuracy judge** (git apply + pytest vs SWE-bench leaderboard eval vs unit-test pass)
4. **Choose codebase** (pandas 0.x verdict task continues OR move to SWE-bench django/astropy for type-aware signals)

I will not start code changes until you confirm.