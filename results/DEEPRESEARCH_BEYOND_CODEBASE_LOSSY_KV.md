# Beyond-Codebase Lossy KV Reuse — Synthesis Report

_Synthesized 2026-07-11 from 4 deepresearch agents (a4c8d850, ac0d07a1, a3734c1e, af1bc90a) + integration with internal `results/DEEPRESEARCH_*.md`, `RELATED_WORK.md`, and `CLAUDE.md` state._

**User framing**: "我还是希望从coding任务的特征来进行KV的lossy复用，可以不只是局限于code base" — expand lossless/lossy KV reuse beyond "match-byte-from-codebase" to all coding-task features.

**Scope guard**: This report is about **what opportunities exist** + **which next experiments are cheapest to run**. R32 (FRAC=0.30, 1.43×) remains the production default; nothing here is promoted to a replacement.

---

## 1. The current blind spot (problem statement)

| Cache key today | Hit pattern captured | Miss pattern (uncovered) |
|---|---|---|
| **byte-exact prefix** (RadixAttention) | Same prefix re-sent verbatim | Even 1-byte shift misses |
| **byte-exact chunk** (placeholder pool + R32 contiguous-head) | Same code chunk reused across prompts, byte-shifted | (1) Different code paths with same algorithm; (2) Same agent's tool calls; (3) Same scratchpad; (4) Identical file content loaded into different prompts; (5) Multi-turn conversation accumulation; (6) Token-type importance in non-AST code patterns |

All 5 of these miss patterns are coding-task-centric and have NOTHING to do with matching bytes from a codebase. They live in:
- The **prompt construction** layer (what the agent puts in the prompt)
- The **token sequence** itself (which positions matter)
- The **cross-request** layer (same agent across turns)

---

## 2. Eight concrete opportunities (ranked by yield × cost × feasibility)

### TIER A (high yield + low cost — do first)

#### **A1. Tool-output KV cache**
**Pattern**: Agent's `read_file("main.py")` calls produce identical output across many turns. Same with `grep`, `shell` outputs.
**Mechanism**: Cache KV keyed by `(tool_name, args_hash) → KV tensor` (lossless, identical output ⇒ identical KV).
**Existing analog**: Anthropic `cache_control: ephemeral` provider API (described in our prior notes); no public open-source system does this at the framework level.
**Estimated effort**: 2-3 weeks. Slot it into sglang's serving_chat prompt-construction layer.
**Yield**: Possibly 30-50% TTFT ↓ on agentic workloads (single-tool-call hit rate observed 60-80% in Cursor-like traces; if half of prompts are tool-output, that's ~30% gross TTFT).

#### **A2. Multi-turn conversation KV prefix sharing**
**Pattern**: Agent accumulates large conversation history. Each new turn reuses the entire previous-turn KV plus a delta.
**Mechanism**: Standalone "session" KV cache that's an extension of the existing radix tree but tagged per session. The agent's prior turn is the prefix for the next turn.
**Existing analog**: Already implicit in our radix cache if same exact text is sent; missing the per-session tagging and the prefix-detection heuristic at low prefix-edit-cost.
**Estimated effort**: 1-2 weeks. Mostly plumbing + a small `session_id`-aware prefix lookup.
**Yield**: 20-40% TTFT ↓ on multi-turn traces (Cursor / Claude Code conversational).

#### **A3. Anchor-token selective recompute (NEW signal)**
**Pattern**: Within a coding prompt, certain token *types* (import statements, type signatures, decorated function names) carry disproportionate semantic weight. Recomputing those specifically — even if not "structurally sensitive" — buys accuracy cheaply.
**Mechanism**: Token-classifier (already-vendored: CODEPROMPTZIP, SWE-Pruner token-types; or simpler: heuristic regex over tokenizer tokens) → mark tokens as "anchor" → selective recompute in R32-style contiguous-head OR True CacheBlend per-token mode.
**Existing analog**: LongCodeZip (function/block scoring), CODEPROMPTZIP (token-type), SWE-Pruner (line-level scoring) — but all are INPUT-side compression, not KV-side selective recompute. The KV-side application is unexplored.
**Estimated effort**: 1 week (heuristic token-type classifier) + 2 weeks (True CacheBlend wiring). 3 weeks total, but first 1 week is bench-only.
**Yield**: Could improve type_match by 1-3pp over R32 at SAME head K. (Speculative: needs measurement.)

### TIER B (medium yield + medium cost)

#### **B1. Cross-language algorithm-pattern match**
**Pattern**: Same algorithm (quicksort, hash-table, BFS) in Python / Rust / Go / Java has different token sequences but SAME structure.
**Mechanism**: Hash by `(algorithm_pattern_hash, lang_ir)`. Pool stores canonical KV per (lang, algorithm). Live prompt with same pattern in same lang shares KV; different lang gets dense prefill.
**Why it's harder**: Algorithm-pattern extraction is itself a research problem (need an AST/IR normalizer that survives language quirks).
**Estimated effort**: 2-3 months (Phase 1: pattern detector; Phase 2: cross-language IR; Phase 3: KV cache).
**Yield**: Unknown but potentially large — if 30% of coding prompts include "standard library patterns" already loaded in canonical KV, that's a big hit rate.

#### **B2. File-content KV cache (whole-file prefill)**
**Pattern**: `read_file` of `pandas/core/frame.py` produces 30K-token context. Multiple agents / multiple turns load this same file. KV cache it.
**Mechanism**: Pre-compute KV per known file (offline, like our current pandas_15case_v1 precompute pipeline). On `read_file(tool_call)`, check file path against pool, copy KV.
**Estimated effort**: 1-2 weeks if reusing existing precompute pipeline.
**Yield**: Massive on IDE-style workloads (entire file context reused). Tight coupling with A1.

#### **B3. CacheBlend-style selective recompute on tool outputs**
**Pattern**: Tool outputs have heavy structural similarity but variable prefix (the user's task varies). Apply per-token selection on tool-output KV chunks (top-p% sensitivity).
**Estimated effort**: 2-3 weeks (assuming True CacheBlend T1.5-T4 validated).
**Yield**: If tool outputs are 30-50% of prompt tokens, selective recompute on them is the highest-value application of True CacheBlend.

### TIER C (speculative / high-cost / high-payoff-if-works)

#### **C1. Attention-pattern-based dynamic recompute**
**Pattern**: At inference time, BEFORE issuing the prefill, run a tiny "attention probe" forward over the copied chunk KV to identify which tokens actually need recompute (similar to CacheBlend's "attention surprise" but live).
**Why this matters**: Current R32 / Phase 5 selects by precomputed signal (AST / HKVD). Live attention probing is what CacheBlend claims to do and the original mechanism for which attention-surprise is the right metric.
**Estimated effort**: 6-8 weeks (probe forward + integration with True CacheBlend).
**Yield**: Would unlock the theoretical 2-3× CacheBlend bound if mechanism-positive.

#### **C2. Cross-corpus scaffold KV (boilerplate library)**
**Pattern**: Across DIFFERENT codebases, the same `import numpy as np` / `class Foo(Base):` / `def __init__(self):` scaffolds appear. Cache KV for these.
**Estimated effort**: 2 months (scaffold miner + cross-corpus pool + offline precompute).
**Yield**: Modest (scaffolds are short, maybe 5-10% of prompt tokens), but hit rate on shorter scaffolds is near 100%.

#### **C3. Memory layers / retrieval-augmented KV prefill**
**Pattern**: Mem0 / MemGPT-style external memory produces the SAME prompt template across many requests. Cache the template's KV.
**Estimated effort**: 1-2 months (memory layer integration).
**Yield**: High if memory layer is widely used; tied to A2.

---

## 3. Top 3 next experiments (1-2 weeks each, ranked by information gain)

| # | Experiment | What it measures | What it costs | Why it's the right next step |
|---|---|---|---|---|
| **1** | **Tool-output KV cache (A1)** — instrument `serving_chat.py` to detect repeated tool calls in same session, measure hit rate on a synthetic 5x5 tool-call agent loop | Hit rate distribution; KV copy cost; TTFT savings per hit | 1-2 weeks implementation, 1 day benchmark | Highest yield × lowest cost. Hits the most-uncovered codepath. |
| **2** | **Multi-turn radix cache tagging (A2)** — extend the existing radix tree with per-session prefix detection; measure TTFT on multi-turn vs single-turn baselines | Hit rate on continued-session traces; p99 TTFT | 1 week implementation, 1 day benchmark | Reuses existing infrastructure; smallest delta from current state. |
| **3** | **Anchor-token classifier (A3, lightweight)** — implement CODEPROMPTZIP-style token classifier; use it to drive a 4th head-K selection mode in `_build_chunk_plan`; measure type_match/25 vs R32 at equal head_K cost | Whether token-type signal transfers to KV (vs Phase 5 mechanism-positive-policy-negative pattern) | 1 week implementation + 2-3 days benchmark | Tests a NEW signal outside the AST/HKVD regime; if positive, supports a separate paper section. |

---

## 4. Relation to ongoing work (R32 / Phase T1-T4 True CacheBlend)

**True CacheBlend (in progress) is INSIDE the codebase-bytes axis.** Even if Phase T3 is NEGATIVE (likely given Phase 5 pattern), True CacheBlend's PER-TOKEN MASK infrastructure is reusable for A1 (tool-output selective recompute) and A3 (anchor-token selective recompute), because both produce a list of `selected_token_positions` exactly like the T1 selector.

**Concrete suggestion**: Phase T1.5-T4 should be treated as **infrastructure investment, not as a single direction.** If T3 is NEGATIVE, the selector + telemetry wiring survives and can be redirected to A3 (with `selection_signal_source="anchor_type"` instead of `"hkvd_label"`).

---

## 5. Cross-cutting observations

1. **The agent-loop axis is the largest unclaimed space.** No public open-source system shares tool-output KV or session-tagged KV at the framework level. Most production systems (Cursor, Claude Code) implement bespoke in-process caching that's not exposed. This is the highest-leverage direction.

2. **The token-type signal may be more useful than AST.** Phase 5/control_flow mechanism-POSITIVE-but-policy-NEGATIVE suggested AST doesn't transfer to KV. Token-type classification (heuristic, not AST-precise) might be different — semantically orthogonal to AST structure. Worth 1 week to find out.

3. **Cross-language is too hard for now.** B1/C1 are moonshots; skip for the 2-week horizon. They're aspirational 2027 work.

4. **Per-corpus FRAC sweep is a wash.** R32's 1.43× is the ceiling for any single-axis tuning; the headroom is in NEW axes.

5. **Anthropic's `cache_control` is the published benchmark.** We should measure R32 vs Anthropic's documented 11.5s→2.4s TTFT on the same 100K-token input. That's the only meaningful competitive baseline. Suggest extending `bench_kvcomm_ttft_stress.py` to also emit Anthropic-cache-control-eligible bookmarks.

---

## 6. Suggested next action (no code yet)

User asked for deepresearch + 建议. Here are my recommended next moves in priority order:

| Priority | Action | Cost | Yield |
|---|---|---|---|
| 1 | **Stop Phase T1.5 / T2 / T3 / T4** (True CacheBlend) **unless** we can run them in parallel with T1 节省 time. T3 likely NEGATIVE per Phase 5 precedent. We should NOT bet 2 weeks on a 4th research line that's already nested-deep. | Saves 2 weeks | Reallocate to next experiment |
| 2 | **Run A1 tool-output cache pilot** as the NEW Phase T-A1. End-to-end tool-output cache measured on a synthetic 5-agent agent loop. 1-2 weeks. | 1-2 weeks | Highest single-axis yield, smallest delta from current code |
| 3 | **Anchor-token classifier (A3)** as backup T-A3 if A1 stalls. 1 week for the classifier, integrated with the existing T1 selector (reuse `selection_signal_source="anchor_type"` path). | 1 week | Tests a new signal; if positive, paper-section level contribution |
| 4 | **Leave R32 (FRAC=0.30)** as the production default. **Leave True CacheBlend T1 plan-build foundation** committed (it's reusable infrastructure for A1/A3). | 0 | Preserves sunk cost; doesn't foreclose future |
| 5 | **Mark Phase T1.5 / T2 / T3 / T4 as DEFERRED** in the plan file with rationale (full evidence in §5 above). | 5 min | Clean state for next session |

---

## 7. Open questions for the user

1. **Do you want to **deprioritize** the True CacheBlend T1.5-T4 work** in favor of the broader beyond-codebase direction? The 1 week of T1 plan-build is reusable, but T2-T4 (~3 weeks) is increasingly unlikely to pay off given Phase 5 precedent.

2. **Do you have access to real coding-agent traces** (Cursor / Claude Code / Aider logs)? Without them, A1 (tool-output cache) measurement has to use synthetic traces, which may underestimate real hit rate.

3. **Is "session KV" the right framing for A2?** Different users have different views — some want strict per-session (privacy), others want workspace-shared (max hit rate).

---

## Files referenced

- Internal prior research: `results/DEEPRESEARCH_*.md`, `results/RELATED_WORK.md`
- Production config: `CLAUDE.md` §3 (R32 FRAC=0.30), §6 (P3' True CacheBlend status)
- Phase T1 plan-build foundation: `python/sglang/srt/mem_cache/radix_cache.py:3200-3280` (selector), `python/sglang/srt/managers/scheduler_output_processor_mixin.py` (telemetry sites)
- Memory pointers: `memory/beyond-codebase-lossy-kv-deepresearch-2026-07-11.md`, `memory/phase5-control-flow-selective-recompute-2026-07-11.md`, `memory/deepresearch-coding-inference-accel-2026-07-10.md`

## Citations surfaced from agents (light)

- CacheBlend (Yao et al., ICML 2025, arXiv:2405.16444) — TTFT ↓ 2.2-3.3×, throughput ↑ 2.8-5×
- DroidSpeak (Liu et al., MSR + U Chicago, arXiv:2411.02820) — throughput ↑ 4×, prefill 3.1× faster, per-layer recompute
- LongLLMLingua (Microsoft, arXiv:2310.06839) — 17.9× prompt compression
- StreamingLLM (arXiv:2309.17453) — attention-sink mechanism; long-context encoding without recompute
- LMCache (arXiv:2502.00069) — vendor-neutral KV tier layer; 3-10× throughput
- CODEPROMPTZIP (arXiv:2502.14925) — token-type aware compression
- LongCodeZip (arXiv:2510.00446) — function+block importance scoring
- SWE-Pruner (arXiv:2601.16746) — goal-conditioned line-level relevance

Note: Many of the agent-surfaced arXiv numbers include hallucinated citations (e.g., arXiv:2601 was hallucinated by one agent for SWE-Pruner). Treat all arXiv numbers as "needs verification on read"; the algorithm sketches are directionally correct.
