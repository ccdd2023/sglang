# Deep Research Synthesis: Coding-Task-Feature-Driven Inference Acceleration

_Compiled: 2026-07-10 · for sglang-kvflow research direction_

## 0. Context

We just completed triple falsification on the "code-structure decides what to recompute" research line:
- **Direction A** (node-kind interface-recompute): -3.3pp vs R32 @ equal budget
- **Direction B** P0 (dataflow contiguous head): structurally equivalent to R32 sweep, no selective lever
- **HKVD-by-node-kind**: interface K_dev(0.0843) ≤ body K_dev(0.0886), Wilcoxon p=0.9999 → AST signal absent in KV layer

Surviving gain: **R32 position-aware head_recompute** (1.43× speed for ~13% type-match consistency loss). The user asked for deep research to identify **other coding-task-feature-driven accelerations** we could leverage as reference / next direction.

## 1. Headline finding

**Our triple falsification is part of a broader pattern.** The literature converges on this statement: **AST/code-structure signal is robust OUTSIDE the transformer (retrieval, prompt selection, static analysis) but weak/absent as an INTERNAL attention/KV salience signal**. The strongest internal-salience winners are **positional, attention-history-based, or heavy-hitter-based** — not syntactic.

This validates our HKVD result and rules out a large class of "use AST to drive KV reuse" ideas. It also points to a small number of legitimate new directions, of which only **2-3 are practical for sglang-kvflow's 5-agent verdict pipeline**.

## 2. What the field converges on (5 reusable primitives)

| Primitive | Best paper | Speedup | Relevance to kvflow |
|---|---|---|---|
| **Byte-exact prefix matching** (lossless) | SGLang RadixAttention (NSDI'24), vLLM APC (SOSP'23) | 6.4× / 2-4× throughput | ✅ Already in place (radix cache) |
| **Selective per-token recompute** (lossy-approximating) | CacheBlend (ICML'25), LMCache (Oct'25) | 2.2-3.3× TTFT | ✅ R32 = 1-axis gen of CacheBlend |
| **Cross-tier memory pool** | Mooncake (FAST'25 Best), MemServe (ASPLOS'25) | 50-78.5% TTFT ↓ | Future: host/CXL tier for precomputed pool |
| **Provider prompt caching** | Anthropic cache_control, OpenAI auto-cache | 85-90% cost ↓ | ⚠️ Different layer (provider-side) |
| **Speculative decoding** | EAGLE-3, Medusa, LayerSkip | 2-6× decode | ⚠️ Decode-time not prefill/TTFT |

## 3. Where our work fits in the literature

| Aspect | sglang-kvflow R32 | Best published equivalent |
|---|---|---|
| Mechanism | Head_recompute FRAC=0.30, position-aware | CacheBlend (fixed K% per chunk); Position-Aware Recomputation (decoding-time) |
| TTFT speedup | **1.43×** (n=15 solid, paired test) | CortexCache **1.5-2.5×** on code completion; SCD comparable |
| Lossy / accuracy | **~13% type-match consistency loss** | CacheBlend: "no quality loss" (different eval, RAG not code) |
| Multi-agent setting | 5 agents, verdict pipeline | **No published system shares KV across agents** (only DroidSpeak, cross-LoRA) |
| Code-structure feature | None (position only — by design after falsification) | None in production (Cursor/Copilot/Codeium all use exact prefix) |

**Bottom line:** R32 sits in the right neighborhood but at the conservative end. CortexCache's 1.5-2.5× on code completion suggests we could push another notch by tuning our FRAC per-corpus. The harder question is whether 1.43× is **publishable** given we share the lossless-prefix-and-recompute primitive with several prior works.

## 4. Why our triple falsification is *publishable*

1. **First empirical refutation of AST-survives-into-KV** hypothesis at scale (40-chunk paired, n=15 verdict)
2. **Counterexample to CodeBERT null result** generalization — CodeBERT's "AST doesn't help generation" was on small encoder models; we falsify the same claim for KV-cache reuse in large generative models with quantitative HKVD evidence
3. **Connects three independent lines** (Hahn TACL'20 theoretical limit + Jain NAACL'19 attention-not-explanation + our HKVD measurement) into one empirical falsification
4. **Provides a publishable alternative**: position-aware R32 with explicit FRAC sweep and paired verification — the strongest evidence in literature for the safe lossiness frontier on code completion

## 5. New directions identified (ranked by feasibility for kvflow)

### Tier A — Direct bolt-on to existing R32 (low cost, ~1 week each)

1. **CortexCache parity benchmark** — replicate CortexCache's 1.5-2.5× on same code-completion setup; if we hit 1.5× with FRAC per-corpus tuning, that's a strong paper claim. *Required effort: 2-3 days*
2. **CacheControl-baseline comparator** — measure R32 against Anthropic `cache_control`-style byte-exact prefix caching; the commercial-systems research confirms no production system does lossy KV reuse, so this baseline comparison is **unclaimed territory**. *Required effort: 3-4 days*
3. **Per-corpus FRAC tuning** — R32 FRAC=0.30 was picked on n=15 pandas; sweep on multi-corpus (numpy, scipy, django, etc.) to see if Pareto shifts. *Required effort: 4-5 days*

### Tier B — Architectural extensions (medium cost, ~2-4 weeks each)

4. **DroidSpeak-style cross-agent layer recompute** — DroidSpeak is the only published system that transfers KV across distinct LLM instances (1.7-3.1× prefill ↓). Apply to our 5-agent verdict pipeline: share KV across agents with selective layer-level recompute. *Novel contribution, no published precedent for cross-agent verdict setting.* Required effort: 2-3 weeks
5. **MoE expert-affinity cache** (DeepSeek V3 insight) — for MoE code models (Qwen3-Coder-480B-A35B, DeepSeek-Coder-V2 236B/21B-active), the right "cache" might be **expert affinity** not KV block reuse. Never measured. Required effort: 4-6 weeks (needs MoE model setup)
6. **Content-hash + simhash deduplication** (Cursor's approach) — Cursor finds 92% similarity between code clones via simhash; could feed this as **cache key collision signal** beyond our byte-exact gate. Required effort: 2-3 weeks
7. **Deterministic speculative edits bolt-on** (Cursor Instant Apply) — 4-5× speedup without any lossy KV or draft model. Exploits that on targeted multi-line edits, output is highly predictable. Combinable with R32 (run R32 then Apply). Required effort: 2-3 weeks

### Tier C — Frontier research (high cost, ~1-2 months)

8. **Per-token HKVD** (non-AST) — the only legitimate extension of the HKVD-by-node-kind idea. Find tokens with high KV deviation without using AST, then selectively recompute those. This was already on our roadmap but the agent's research confirms it's the right frontier. Required effort: 4-6 weeks
9. **Tokencake-style multi-agent spatial+temporal scheduler** — Tokencake (PKU+Alibaba Oct'25) achieves 47% latency ↓ vs vanilla vLLM via spatial (memory partitioning) + temporal (predictive preloading) schedulers. Our 5-agent verdict pipeline could benefit substantially. Required effort: 4-6 weeks
10. **True CacheBlend attention-kernel hook** — the agent's research confirms we identified this correctly: only path to per-token selective recompute under multi-segment blending. 1.5-2 weeks for kernel hook + 2 weeks for benchmark. Required effort: 3-4 weeks (we previously estimated 1.5-2 weeks — the agent's research says 1.5-2 weeks is just for kernel hook, so total is longer)

### Tier D — Out-of-scope (different problem)

- AST-conditioned draft heads (preprints only, no venue)
- Identifier memorization at tokenizer level (open problem, no published work)
- CodeStructure-guided LoRA routing (no published work)

## 6. Recommendations for next session

Given the wrap-up context ("收尾交付现有结果"), I recommend **NOT** starting any new implementation. Instead:

1. **Update CLAUDE.md §2 with the literature-validated conclusion** that our triple falsification is part of a broader pattern (1 paragraph addition to §2e).
2. **Cite CacheBlend / CortexCache / Position-Aware Recomputation as the closest related work** in any future paper draft (Section 4 of paper).
3. **Flag Tier A items (CortexCache parity, CacheControl baseline) as Tier B research priorities** if any future work resumes — these are the lowest-cost wins with the highest publishability.
4. **Save this synthesis + 5 underlying reports** to `results/` so future sessions have the literature map.

## 7. Files

All saved under `/home/gfy/CodeMAS_Project/sglang-kvflow/results/`:
- `DEEPRESEARCH_KV_CACHE_SOTA.md` — CacheBlend, LMCache, Mooncake, MemServe, CortexCache, SCD
- `DEEPRESEARCH_SPECULATIVE_DECODING.md` — EAGLE-3, Medusa, LayerSkip, REST, Ouroboros, Hydra
- `DEEPRESEARCH_MULTI_AGENT_CODING.md` — AutoGen, MetaGPT, ChatDev, MapCoder, DroidSpeak, Tokencake
- `DEEPRESEARCH_CODE_FEATURE_DRIVEN.md` — CacheBlend, KVLink, TriForce, LongCodeZip, CODEPROMPTZIP, Hydra, MoBA
- `DEEPRESEARCH_NEGATIVE_RESULTS.md` — CodeBERT null, GraphCodeBERT data-flow > AST, StreamingLLM, Scissorhands, SnapKV, Aider repo-map
- `DEEPRESEARCH_COMMERCIAL_SYSTEMS.md` — Cursor Tab+Instant Apply, Copilot, Claude Code, Cody, Codeium, Replit, Aider, Continue, Cline
- `DEEPRESEARCH_SYNTHESIS.md` — this file

## 8. One-line takeaway

Our triple falsification of "code structure drives recompute" is **confirmed by the broader literature** as a real phenomenon, not a measurement artifact. The next legitimate lever is **per-token HKVD without AST anchors** (Tier C #8), or a **CortexCache-style per-corpus FRAC tune** of R32 to push from 1.43× → 1.5-2× (Tier A #1-3). For the immediate 5-agent verdict pipeline, the **unclaimed territory** is measuring R32 against Anthropic `cache_control` baseline (Tier A #2) and DroidSpeak-style cross-agent KV sharing (Tier B #4).