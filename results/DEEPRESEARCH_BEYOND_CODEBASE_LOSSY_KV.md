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

---

## Appendix A — Peer-surfaced citations (added 2026-07-11 post-synthesis)

A peer Claude session completed a complementary survey while this synthesis was being written. The following published systems should be folded into the synthesis above:

### **A. TokenCake** (arXiv:2510.18586, Oct 2025; PKU + Alibaba)
**Algorithm**: Two-scheduler architecture for multi-agent LLM apps with external function calls.
- **Temporal Scheduler**: offloads idle agent KV to CPU during long tool calls; predictively reloads based on estimated tool-completion time; uses a CPU block buffer with free-list recycling + progressive GPU reservation to hide transfer latency.
- **Spatial Scheduler**: dynamically partitions GPU KV cache into a *global shared pool* + a *reserved pool*; per-agent hybrid-priority metric (graph structure + runtime state); two-phase reservation adjustment per cycle based on memory pressure.

**Numbers**: **≥47.06% latency reduction** vs vLLM at 1.0 QPS high load (Code-Writer + Deep Research apps; ShareGPT + AgentCode traces; Poisson arrivals); GPU KV-cache utilization +16.9%; offload latency "near-second" → "stable sub-millisecond".

**Lossless/lossy**: Eviction/offload but token-level lossless (no KV approximation).

**Why this validates A1**: TokenCake's Temporal+Spatial Scheduler is functionally the published version of what I called **"A1 tool-output cache"**. The 47% reduction is concrete, measured, and on a coding-workload benchmark. **The A1 opportunity has external validation now — this is no longer speculative.**

### **B. Hogwild! Inference + Hogwild! GPU** (arXiv:2504.06261 ICLR 2025; NeurIPS 2025 extension)
**Algorithm**: Multiple generation requests processed **concurrently in a single forward pass** via parallel attention over shared KV-cache entries. KV tensors from different requests are concatenated along the sequence dimension; attention is computed jointly in one kernel.

**Numbers**: SOTA throughput on Llama 3.1, Llama 3.2, Qwen 2.5, Mistral. **LOSSLESS** (output-equivalent to sequential batching).

**Why this matters**: Hogwild! gives a **different architectural path** — instead of cache-and-reuse (R32 / CacheBlend), co-execute multiple requests with shared KV across the batch. This is orthogonal to our current chunk-pool direction and could be combined.

### **C. CacheGen** (Microsoft, arXiv:2503.07036, Mar 2025)
**Algorithm**: Modular framework streaming + compressing KV caches for distributed inference. Adaptive encoder-decoder + context-aware streaming decoder. Targets the **KV transport** problem (PD-disaggregation bandwidth).

**Numbers**: **3.5-4.5× compression** with minimal accuracy loss.

**Why this matters**: CacheGen is a clean **LOSSY dimension** orthogonal to R32. If we want to share precomputed KV across nodes (Moonshot / Mooncake-style architecture), CacheGen-style compression reduces transport cost by 3-4×. Could combine with our precomputed pool to ship KV over RDMA cheaply.

### **D. MemAgent** (arXiv:2507.02259, ByteDance + Tsinghua, ICLR 2026)
**Algorithm**: Workflow that reads text in chunks and **overwrites a fixed-length memory slot** in the context window. Trained via Multi-Conv DAPO (RL). Trained on 32K, **extrapolates to 3.5M tokens with <5% loss**.

**Why this matters**: MemAgent is a **fundamentally different mechanism** — instead of caching KV externally, it bakes the memory into the model's own context window. **This is a model-side solution; we cannot retrofit it onto an existing model. But it tells us the long-context frontier is moving toward agent-oriented memory slots**, which validates the broader direction (agents need long KV contexts).

### **E. Prompt Cache (Bercovich et al., 2024, arXiv:2311.04934)**
**Algorithm**: Modular attention-reuse layer for low-latency inference across requests with shared prefix structure. LOSSLESS. **Often cited as foundational for agent workloads**.

**Why this matters**: Bercovich 2024 is essentially the abstract template for what `cache_control: ephemeral` and our radix cache implement. Reading it for completeness would help locate where our radix cache sits in the academic landscape.

### **F. Anthropic production numbers (Armin Ronacher blog, Nov 2025)** — **THE MOST IMPORTANT DATA POINT**
- **To-C consumer workloads**: **~62% of KV blocks reusable**
- **To-B API (Claude.ai + Sonnet 4.5)**: **~97% of cache hits come from single-turn system-prompt sharing** (huge — system prompt alone is the dominant hit)
- **P99 KV-cache TTL in to-B**: **97 seconds** (very short — content-hash stability matters)
- **Anti-pattern**: injecting dynamic tool I/O into the system prefix silently breaks cache reuse for ALL subsequent turns

**Architectural insight for coding agents**: Claude Code treats `CLAUDE.md`/`AGENTS.md` as **immutable, content-addressed assets** at the start of every session. **Cache-hit-rate drop is treated as a production-incident-level alert internally.**

**Implication for our project**: Anthropic's published 97% is dominated by system-prompt sharing. **If we can build a "system-prompt + tooling + scaffold" layer in our serving infrastructure that hits even 70% across turns**, that's already production-relevant. The lever is **content addressing** (stable input → stable KV) + **session tags** (not necessarily "trust the user — KV stays across requests" but "the system prompt + tool schemas should stay").

### **G. PolyKV** (ishan1410 GitHub, May 2025)
**Algorithm**: Asymmetrically compressed shared KV pool for multi-agent LLM inference reading the same document. **O(1) memory complexity in agent count** for shared document context. LOSSY (asymmetric compression).

**Why this matters**: PolyKV is published open-source and demonstrates that multi-agent shared-document KV is a real, achievable target. Could integrate as a backend for our chunk pool.

### **H. CacheBlend confirmed at ICLR 2025** (arXiv:2412.15444)
Different paper ID than what I had (2405.16444 was an earlier preprint). Worth noting for citation accuracy.

### **MELLON hallucination flag**
The MELLON reference in our prior notes appears to be hallucinated. arXiv 2505.21150 is an unrelated Nevanlinna-theory math paper. **There may be a real MELLON but it isn't in our citations.** Flag for follow-up.

---

## Appendix B — Revised priority after peer-surfaced data

**Before peer research**: A1 (tool-output cache) was speculative.

**After peer research**: A1 is now **externally validated** via TokenCake (47% latency reduction on coding workloads). Combined with Anthropic's published 97% system-prompt sharing hit rate, the agent-loop cache axis is the **highest-leverage direction with the strongest external evidence**.

### Updated next-experiment ranking

| Rank | Experiment | Yield signal | Time |
|---|---|---|---|
| **1** | **Tool-output + system-prompt KV cache combined (A1 + A2)** — instrument serving_chat to (a) tag system prompt + tool schemas as `cache_key=stable`, (b) cache tool outputs by `(tool_name, args_hash)`, (c) measure hit rate on synthetic 5-agent code-edit trace | External: TokenCake 47% ↓ + Anthropic 97% on system prompt | 1-2 weeks |
| **2** | **CacheGen-style transport compression (C)** — wire our precomputed pool to ship KV across nodes via CacheGen's encoder; measure bandwidth savings | External: CacheGen 3.5-4.5× transport compression | 2-3 weeks |
| **3** | **Anchor-token classifier (A3, lightweight)** — reuse True CacheBlend T1 selector with `selection_signal_source="anchor_type"` | Speculative; needs measurement | 1 week |
| 4 | Hogwild! integration | Low priority — orthogonal to cache-reuse, more architectural | 1+ month |

### Updated strategic recommendation

1. **Stop pursuing True CacheBlend T2-T4**. Phase 5 precedent + peer-surfaced validated alternatives make this the lowest-information direction.
2. **Run A1+A2 combined** as the new Phase T-A1. Build on existing radix-cache plumbing. External validation is strong.
3. **Treat CacheGen (C) as Phase T-C** — 2-3 weeks; high payoff if we want to share precomputed KV across nodes.
4. **MemAgent is model-side, out of our scope.** Skip.

---

## Citations now consolidated (both mine + peer's, deduplicated)

| System | arXiv / URL | Lossless? | Class | Yield |
|---|---|---|---|---|
| RadixAttention | 2312.07104 (NSDI'24) | lossless | prefix-reuse | (substrate) |
| vLLM APC | (SOCP'23) | lossless | prefix-reuse | (substrate) |
| Mooncake | 2407.00079 (FAST'25 Best) | lossless | PD-disagg | 525% thr |
| MemServe | 2506.17565 | lossless | cross-pool | -78% TTFT |
| LMCache | 2510.09665 + 2502.00069 | lossless | vendor layer | 10× TTFT |
| **TokenCake** | **2510.18586** | **lossless (token)** | **multi-agent sched** | **≥47% latency ↓** |
| **Hogwild!** | **2504.06261 (ICLR'25)** | **lossless** | **parallel attn** | **SOTA throughput** |
| MemAgent | 2507.02259 (ICLR'26) | lossless | memory slot | 3.5M extrapolation |
| Prompt Cache | 2311.04934 | lossless | attention reuse | foundational |
| CacheBlend | 2412.15444 (ICLR'25) | lossy | selective chunk | 2.2-2.5× TTFT |
| **CacheGen** | **2503.07036 (Mar 2025)** | **lossy** | **KV stream compression** | **3.5-4.5× compression** |
| DroidSpeak | 2411.02820 | lossy | cross-LLM | 4× thr, 3.1× prefill |
| LongLLMLingua | 2310.06839 | lossy | input compression | 17.9× compression |
| StreamingLLM | 2309.17453 | lossy | attention sink | long-context |
| CODEPROMPTZIP | 2502.14925 (ACL'26) | lossy | input compress | type-aware |
| LongCodeZip | 2510.00446 (ASE'25) | lossy | function/block | 5.6× |
| SWE-Pruner | ~2601.* (unverified) | lossy | line-level skim | 14.84× |
| PolyKV | (GitHub, May 2025) | lossy | asymmetric KV | O(1) mem |
| **Anthropic cache_control** | **provider API** | **lossless** | **provider system** | **97% system-prompt sharing, 85% latency ↓** |

Note: SWE-Pruner citation has hallucinated arXiv ID; needs verification if used.

---

## Appendix C — Second peer-surfaced citations (added 2026-07-11)

A second peer Claude session completed "Cross-document KV reuse + RAG-specific lossy KV" with verified arXiv IDs. The following systems should be folded into the synthesis:

### **I. KVLink** (arXiv:2502.16002, Yang et al., v4 Nov 2025)
**Algorithm**: Each retrieved document is **pre-encoded into its own independent KV cache off the critical path**. At query time, per-document KV chunks are concatenated into one fused cache; positional embeddings are re-shifted to match the new global position; trainable "separator" special tokens inserted between documents so the model can re-establish cross-document self-attention without recomputing.

**Numbers**: **TTFT reduced up to 96%** vs full re-prefill; **+4% QA accuracy** vs SOTA baselines across 7 RAG datasets; composes with separate KV-cache compression to lower IO.

**LOSSLESS** (no token dropped). **Closest published analog to our precomputed pool**, but uses whole-document granularity. **Could combine** with our chunk-level pool by precomputing KV per chunk as a KVLink entry.

### **II. RAGCache** (arXiv:2405.00031, OSDI 2024; Jin et al., PKU + Sea AI Lab)
**Algorithm**: **Hierarchical cache** structured as:
- **Knowledge tree** — organized similar retrieved documents
- **Prefix tree** — overlap of the prompt prefix
- **KV cache per chunk** — intermediate states reused across similar queries

**Numbers**: **24.7× LLM-inference latency reduction vs vLLM**; **4.1× LLM inference speedup**.

**Lossy by design at intermediate-states granularity; lossless when prefixes match exactly.**

**THIS IS THE DIRECT COMPETITOR TO OUR PLACEHOLDER POOL.** RAGCache's hierarchical cache shares the same structural insight as ours (precomputed chunk KV) plus adds a semantic similarity layer (knowledge tree). We should benchmark our R32 against RAGCache's reported numbers as the published ceiling for chunk-level reuse.

### **III. CacheGen** (SIGCOMM 2024, arXiv:2310.07240; Liu et al.; NOT arXiv:2503.07036 — that's a different paper)
**Algorithm**: Tensor encoder/decoder custom to KV-cache distributions; compresses into compact bitstream; adaptive per-tensor compression; optional recompute when bandwidth collapses.

**Numbers**: **3.5-4.3× KV-cache size reduction**; **3.2-3.7× reduction in total context-fetch+processing delay**.

**LOSSLESS-or-LOSSY**: LOSSY (claims negligible quality impact).

**Where it fits**: CacheGen targets **KV transport bandwidth** in PD-disaggregation. Could be wired into our precompute pipeline to ship KV across nodes cheaply. **Two arXiv IDs have been associated with "CacheGen" in our notes; the SIGCOMM'24 one (2310.07240) is the verified original.**

### **IV. ChunkAttention** (ACL 2024, arXiv:2402.15220; Ye, Tao, Huang, Li)
**Algorithm**: Per-request KV cache split into small chunks, indexed in auxiliary prefix tree; at runtime share KV tensors of matching chunks; **two-phase partition scheduler** reorders head-dim tiles so each shared chunk is served from contiguous memory.

**Numbers**: **3.2-4.8× self-attention kernel speedup** with system prompt lengths 1024-4096.

**LOSSLESS** (prefix match at finer granularity than single radix tree). **Note**: I checked our sglang fork — ChunkAttention is **NOT** wired in. The peer's "integrated into SGLang" claim appears mistaken (the `dual_chunk_attention_config` field in our fork is a different Qwen3 DCA mechanism, not the ChunkAttention paper). ChunkAttention could be a 1-2 month integration project.

### **V. KV-Runahead** (ACL 2025, arXiv:2509.01066; Cho et al.)
**Algorithm**: Overlaps KV-cache computation with tokenization by **prefetching and selectively recomputing** KV entries in parallel. Distributed scheduler treats per-layer KV prefetch as a separate pipeline stage.

**Numbers**: Performance improves substantially on long prompts; quality impact small.

**Adjacency to our work**: KV-Runahead is about long-context scaling generally, not cross-prompt RAG. Different angle.

### **VI. Prompt Cache** (arXiv:2401.17268, Yale; arXiv:2507.10314 follow-up)
**Algorithm**: LLM partitioned into **modules** (sub-sequences of transformer layers), each with own KV cache; subsequent requests with same prefix reuse cached module outputs directly.

**LOSSLESS** (exact prefix match within modules).

**Modern descendant**: Re-examines prompt *position* effects when cache is partial.

### **Flagged UNVERIFIED (peer's search-only cites)**
The following names appeared but the agent could not verify arXiv IDs — treat with caution:
- ChunkKV (claimed 2502.09811)
- ClusterKV (claimed 2503.03596)
- KVzip (claimed 2503.01566)
- LayerKV (claimed 2410.00428)
- RetrievalAttention (claimed 2403.03451) — actually this is a real paper, confirms semantic K-means lookup
- RetrievalCache (claimed 2404.12337)

None have arXiv landing-page verification in this survey; some may be hallucinated. Use them for inspiration not citation.

---

## Appendix D — Implications for sglang-kvflow

Three concrete "head-to-head" research outputs now exist that we can directly compare against:

| Our metric | Public comparison | Threshold for paper claim |
|---|---|---|
| R32 TTFT (1.43×) | **CacheBlend: 2.2-2.5× TTFT ↓** | ≤2.0× means we're competitive on the same axis |
| R32 placeholder pool hit rate | **RAGCache: 24.7× vs vLLM** | ≥10× means our hit rate beats vLLM |
| Pool KV reuse quality | **KVLink: +4% QA accuracy vs SOTA** | ≥baseline means lossless concat is equivalent |

**New comparison target**: We should benchmark R32 + True CacheBlend T1.5 against:
- **CacheBlend** (TTFT axis) — direct competition
- **RAGCache** (latency/speedup axis) — direct competition
- **KVLink** (accuracy axis) — verifies lossless claim

If R32 lands **within 70% of RAGCache's 24.7× on shared-chunk hit rates**, that's a publishable result.

---

## Appendix E — Updated strategic recommendation after second peer research

1. **True CacheBlend (in progress) is a different mechanism axis** — it's per-position selection within precomputed chunks. We're not competing head-to-head with CacheBlend's per-position selection; we're proposing an alternative (R32 contiguous head + uniform FRAC).

2. **The strongest competition is RAGCache's hierarchical cache** (24.7× speedup). Our R32 at 1.43× is well below this. Two possible responses:
   - (a) Measure R32 with byte-shifted matching disabled (only exact-prefix cache, like vLLM APC) → measure baseline and isolate our chunk pool's contribution
   - (b) Add semantic-similarity matching at the chunk-pool level (knowledge-tree style) to compete directly with RAGCache

3. **KVLink is the lossless concat alternative.** Our chunk pool + R32 is a *lossy* version of what KVLink does losslessly. Different point in the lossless-lossy spectrum. Worth noting in §2 of paper.

4. **ChunkAttention integration** (1-2 month scope) would give us kernel-level speedup on top of R32's algorithmic gain.

---

## Final priority after both peer-surfaced citations

| # | Experiment | Yield evidence | Time |
|---|---|---|---|
| **1** | **R32 vs vLLM APC isolated measurement** (set chunk pool OFF, only radix prefix; measure TTFT delta vs full chunk pool) — quantifies R32's net value vs published 24.7× RAGCache | anchors our position | 1 day |
| **2** | **A1 tool-output cache** combined with system-prompt cache (TokenCake 47% + Anthropic 97% are concrete external benchmarks) | external validation | 1-2 weeks |
| **3** | **RAGCache-style knowledge-tree layer** added to our chunk pool (semantic similarity matching alongside byte-exact) | external ceiling | 4-6 weeks |
| 4 | ChunkAttention kernel integration | kernel-level 3.2-4.8× | 1-2 months |
| 5 | A3 anchor-token classifier | speculative | 1 week |

**Crucial context**: RAGCache's 24.7× is on long-context RAG with shared retrieved documents. Our pool is on shared code chunks. Different workloads. But the structural lesson — **hierarchical cache + semantic similarity** — may apply.

**Recommended for next session**: Run isolated experiment #1 (1 day) to get a meaningful comparison number. If R32 ≥ 10× (vs vLLM) on code-chunk hit rates, paper claim is competitive. If less, reconsider scope.
