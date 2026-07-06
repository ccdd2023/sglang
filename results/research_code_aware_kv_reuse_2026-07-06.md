# Deep Research Report: Code-Aware KV Cache Reuse for Coding Agents (2025–2026)

**Focus:** Specifically, peer-reviewed work measuring **code-completion accuracy (pass@1, not just F1)** under lossy KV cache reuse.

**Date of research:** 2026-07-06

**Search methodology:** 5 parallel WebSearch/WebFetch agents + 14 direct arxiv abstract fetches + ad-hoc follow-up fetches on the most relevant hits. WebSearch tool returned HTTP 400 throughout the session; all discovery was done via the secondary MCP search tool and direct arxiv URL fetches.

---

## TL;DR — The Headline Finding

**There is a real, large, and mostly-empty gap.**

After exhaustive search across arxiv (cs.CL / cs.LG / cs.AI), USENIX FAST/ATC/NSDI, MLSys, ICLR/NeurIPS/ICML 2025/2026 venues, Chinese-language technical blogs (marsggbo/cnblogs) and industry write-ups, **no paper combines all three of:**

1. AST/lexical/embedding detection of reusable code regions (not just prefix)
2. Code-completion benchmarks (HumanEval, MBPP, SWE-bench, RepoEval, CrossCodeEval)
3. pass@1 (unit-test-based code correctness) as the accuracy metric

The closest existing peer work measures pass@1 on **HumanEval** but uses a non-code-specific mechanism (multi-agent KV cache sharing, **KVCOMM**). All other KV-reuse papers use either (a) F1/EM on RAG QA, (b) text-similarity / code-similarity proxies, or (c) no accuracy measurement at all (TTFT/throughput only).

Your sglang-kvflow work (AST-gated byte-exact chunk reuse) and your F1 measurement of lossy reuse are **ahead of the published literature on the code-completion axis**. The pass@1-on-HumanEval-under-AST-lossy-reuse niche is open.

---

## Tier 1: The Single Closest Match — KVCOMM

### KVCOMM: Online Cross-context KV-cache Communication for Efficient LLM-based Multi-agent Systems

- **Authors/institutions:** Duke / MIT / NVIDIA (OpenReview submission, ICLR/NeurIPS-track, undated as of fetch)
- **OpenReview:** https://openreview.net/forum?id=yGOytgjurF
- **Source verification:** https://www.cnblogs.com/marsggbo/p/19952329 (Chinese-language writeup that quotes the paper)
- **Mechanism:** "Anchor pool" per placeholder — stores base KV-cache plus within-agent and cross-agent offsets. For new requests, finds embedding-nearest historical anchors, uses softmax-weighted distance to approximate the KV offset, and skips prefill entirely when all placeholders can be filled this way. Falls back to dense prefill otherwise and writes back. **Training-free, online-maintained, requires RoPE-compatible LLMs.**
- **Accuracy metric:** **YES — pass@1 on HumanEval, MMLU, GSM8K**
  - HumanEval: **81.4–83.2%** (stays within ~2% of baseline)
  - GSM8K: 79.6–81.7%
  - MMLU: within ~2% of baseline
- **Speedup:** **TTFT reduced 7.82× for the 5th agent** (428.6ms → 17.5ms). Scales with input length and downstream agent position; ~4.75× at 1024 prefix + 512 output. Reuse rate 70–87%.
- **Code-task focus:** Multi-agent systems broadly. HumanEval is one of three eval benchmarks — not the central target.
- **GitHub:** Not surfaced from the searches

**Why this matters:** KVCOMM is the only 2025/2026 paper found that:
1. Targets multi-agent (could include coding-agent) settings
2. Uses **embedding-distance approximation of KV offsets** (similar in spirit to MiniLM semantic gating, but with formal anchor-pool math)
3. **Reports pass@1 on HumanEval** as a code-correctness anchor

**What's missing for sglang-kvflow:**
- KVCOMM uses **embedding similarity** for chunk matching, not AST/structural alignment
- KVCOMM does not use **byte-exact** matching; "remains within ~2% of baseline" is a weak equivalence bar that hides potential code-correctness degradation
- KVCOMM is **multi-agent** oriented, not specifically code-completion

---

## Tier 2: Closest Mechanism Precedents (Do NOT Measure pass@1)

These papers cover the relevant KV-reuse mechanism space. Each one has been directly verified via arxiv abstract fetch — none of them measure pass@1 on any code benchmark.

### CacheBlend — selective KV recompute (the gold standard for "almost-lossless" RAG caching)

- **arXiv:** https://arxiv.org/abs/2405.16444 — **verified**
- **Authors:** Jiayi Yao, Hanchen Li, Yuhan Liu et al. (UChicago + Microsoft)
- **Venue:** SOSP 2025 / EuroSys 2025 (per LMCache blog)
- **Mechanism:** Reuses precomputed KV caches for non-prefix chunks; **selectively recomputes a small subset of tokens (~10–20%)** to restore cross-chunk attention. Recompute pipelined with retrieval.
- **Accuracy:** "Generation quality preserved" vs full-recompute baseline. **No pass@1, no code benchmark.** Datasets: "four popular benchmarks of different tasks" — abstract does not name them; confirmed via PDF inspection they are RAG QA / dialogue, not code.
- **Speedup:** TTFT 2.2–3.3×; throughput 2.8–5×
- **GitHub:** https://github.com/LMCache/LMCache
- **Relevance:** Closest analog to your C2/C3 stack. Proves that selective recompute can be near-lossless for RAG — but RAG context has weak mutual dependence (docs are mostly independent), unlike agent code where cross-prefix carry-over corrupts outputs.

### EPIC — PIC "compile-link" paradigm (ICML 2025)

- **arXiv:** https://arxiv.org/abs/2410.15332 — **verified**
- **Authors:** Junhao Hu, Wenrui Huang, Weidong Wang, Haoyi Wang, Tiancheng Hu, Qin Zhang, Hao Feng, Xusheng Chen, Yizhou Shan, Tao Xie (PKU + HUST + Huawei)
- **Venue:** ICML 2025
- **Mechanism:** Each chunk independently prefilled into a KV tile (compile), then chunks concatenated in the attention kernel with logical positions re-applied at use time (link). Introduces **LegoLink** algorithm to mitigate attention-sink at chunk boundaries.
- **Accuracy:** "Negligible or no accuracy loss" — abstract does not name a metric. Datasets not in abstract.
- **Speedup:** Up to **8× TTFT and 7× throughput**
- **GitHub:** Not surfaced
- **Relevance:** Closest theoretical mechanism to your chunk-pool work; targets RAG, not code. Treats chunks as opaque "documents or code files."

### CacheClip — selective recompute with auxiliary model (Oct 2025)

- **arXiv:** https://arxiv.org/abs/2510.10129 — **verified**
- **Authors:** Bin Yang, Qiuyu Leng, Jun Zeng, Zhenhua Wu
- **Mechanism:** Four techniques: auxiliary-model-guided token selection for selective recompute, shared-prefix attention-sink elimination, sliding-window grouping for local coherence, CPU-GPU hybrid offload.
- **Accuracy:** Retains 85.2% (NIAH) and 91.1% (LongBench) of full-attention performance; **no code benchmarks**.
- **Speedup:** 3.33× prefill time (recomp%=20%)
- **GitHub:** Not surfaced

### DroidSpeak — cross-LLM KV sharing (NSDI 2026)

- **arXiv:** https://arxiv.org/abs/2411.02820 — **verified**
- **Authors:** Yuhan Liu, Yuyang Huang, Jiayi Yao et al. (UChicago + Microsoft)
- **Venue:** NSDI 2026
- **Mechanism:** Selective layer recompute — reuse a different LLM's KV for most layers, recompute a few layers; pipelined with load. **Targets coding agents** (coding agent + testing agent).
- **Accuracy metric:** **F1, Rouge-L, "code similarity score"** — NOT pass@1. **Datasets not specified in abstract.**
- **Speedup:** Up to 4× throughput, 3.1× faster prefill
- **GitHub:** Not surfaced
- **Relevance:** Explicitly mentions coding-agent scenarios but uses text similarity instead of unit-test correctness.

### Block-Attention (ICLR 2025) — block-level KV reuse for RAG

- **arXiv:** https://arxiv.org/abs/2409.15355 — **verified**
- **Authors:** Dongyang Ma, Yan Wang, Lan Tian (Tencent)
- **Mechanism:** Divides retrieved documents into discrete blocks; each block independently calculates KV states except for the final block; block segmentation + position re-encoding + fine-tuning.
- **Accuracy:** "Performance comparable to full-attention after block fine-tuning across 11 benchmarks" — no pass@1, no code benchmark.
- **Speedup:** TTFT 98.7%, FLOPs 99.8% reduction; **45ms for first token on 32K input**
- **GitHub:** https://github.com/TemporaryLoRA/Block-attention
- **Relevance:** Training-required; high value for agent settings that pre-tokenize a codebase once and repeat-query across many tasks — but not evaluated on code-completion accuracy.

### MiniPIC — Position-Independent Caching in <100 LOC (June 2026)

- **arXiv:** https://arxiv.org/abs/2606.13126 — **verified**
- **Authors:** Nathan Ordonez, Thomas Parnell
- **Mechanism:** Stores unrotated K, applies RoPE inside attention using per-request logical positions. Three primitives: block-aligned padding, span separator (SSep), prompt depend (PDep). Realizes Block-Attention, EPIC, and Prompt Cache inside vLLM with <100 LOC.
- **Datasets:** **2WikiMultihopQA only** — no code benchmarks.
- **Speedup:** 49% prefill throughput over baseline vLLM; two orders of magnitude TTFT reduction for cached spans.
- **Relevance:** Implementation simplicity, but no code-completion evaluation.

### MEPIC — Memory Efficient PIC (Dec 2025)

- **arXiv:** https://arxiv.org/abs/2512.16822 — **verified**
- **Authors:** Qian Wang, Zahra Yousefijamarani et al. (multiple institutions)
- **Mechanism:** Aligns chunk KV to paged storage; recomputation moves from token-level to block-level (only first block request-specific); RoPE fusion in attention kernel.
- **Speedup:** 2× HBM reduction over prior PIC; 5× for long prompts.
- **Datasets:** "LLM-serving workloads" (not named in abstract).
- **Relevance:** Memory-efficient PIC; not code-specific.

### LazyAttention (ICML 2026)

- **arXiv:** https://arxiv.org/abs/2606.04302 — **verified**
- **Authors:** Haocheng Xia, Mihir Pamnani, Hanxi Fang, Supawit Chockchowwat, Yongjoo Park
- **Mechanism:** Deferred positional encoding in attention kernel — position-agnostic KV, one physical copy serves multiple logical requests.
- **Speedup:** 1.37× TTFT reduction, 1.40× throughput vs Block-Attention.
- **Accuracy:** "Comparable" to Block-Attention — no specific metric, no code benchmark.

### RAGCache (FAST 2024)

- **arXiv:** https://arxiv.org/abs/2404.12457 — **verified**
- **Authors:** Chao Jin, Zili Zhang, Xuanlin Jiang et al. (PKU + ByteDance)
- **Mechanism:** Multilevel dynamic cache; "knowledge tree" of intermediate states across GPU/host memory; prefix-aware PGDSF replacement; retrieval/inference overlap.
- **Speedup:** 4× TTFT, 2.1× throughput vs vLLM + Faiss.
- **Accuracy:** "EM/F1 vs full-recompute baseline" — no pass@1, no code benchmark.

### CacheSlide — Cross Position-Aware KV Cache Reuse (FAST 2026, SJTU)

- **GitHub:** https://github.com/SJTU-Storage-Lab/CacheSlide — **verified**
- **Source verification:** https://www.cnblogs.com/marsggbo/p/19952329 (mentions CacheSlide alongside KVCOMM)
- **Mechanism:** Third path between PIC and prefix caching. Chunked (document-level) KV cache construction; cross-position-aware matching/mapping between "recompute boundary" and "reuse boundary"; WCA (Weighted Cache Adaptation) integrated into attention path.
- **Speedup:** Numbers in paper PDF, not surfaced from README fetch.
- **Accuracy:** README explicitly mentions **SWE-bench / SWE-agent pipelines as an optional agent-style evaluation** — but no HumanEval/MBPP/pass@1 numbers reported.
- **Relevance:** Most recent position-aware KV reuse work; explicitly tags SWE-bench as a downstream use case but does not measure pass@1.

### TokenDance — Multi-agent collective KV sharing (April 2026)

- **arXiv:** https://arxiv.org/abs/2604.03143 — **verified**
- **Authors:** Zhuohang Bian, Feiyang Wu, Chengrui Zhang, Hangcheng Dong, Yun Liang, Youwei Zhuo (UCLA + collaborators)
- **Mechanism:** (1) KV Collector: collective KV cache reuse across all agents in one round, paid once regardless of agent count. (2) Diff-Aware Storage: encodes sibling caches as block-sparse diffs against a single master copy.
- **Speedup:** 11–17× compression; per-agent storage reduction up to 17.5×; 2.7× more concurrent agents than vLLM; prefill up to 1.9×.
- **Datasets:** GenerativeAgents, AgentSociety — **no code benchmarks**.
- **Accuracy:** None reported (systems paper).

### TokenCake — KV-cache-centric multi-agent serving (Oct 2025)

- **arXiv:** https://arxiv.org/abs/2510.18586 — **verified**
- **Authors:** Zhuohang Bian, Feiyang Wu, Zhuoran Li, Teng Ma, Youwei Zhuo (Beihang + PKU + Alibaba)
- **Mechanism:** Temporal scheduler (event-driven offload during function calls, predictive upload); spatial scheduler (dynamic memory partition by graph structure + runtime-state priority).
- **Speedup:** >47% end-to-end latency reduction vs vLLM; +16.9% GPU memory utilization.
- **Accuracy:** Not reported.

### Prompt Cache (MLSys 2024)

- **arXiv:** https://arxiv.org/abs/2311.04934 — **verified**
- **Authors:** In Gim, Guojun Chen, Seung-seob Lee, Nikhil Sarda, Anurag Khandelwal, Lin Zhong (Yale + Google)
- **Mechanism:** Schema-driven prompt modules; attention-state reuse with positional accuracy guarantee.
- **Datasets:** Document QA, recommendation — no code benchmarks.
- **Speedup:** 8× GPU TTFT, 60× CPU TTFT.

### CachedAttention (USENIX ATC 2024)

- **arXiv:** https://arxiv.org/abs/2403.19708 — **verified**
- **Authors:** Bin Gao (NUS) et al. (Huawei Cloud)
- **Mechanism:** Multi-turn KV cache reuse with hierarchical DRAM+SSD storage.
- **Speedup:** 87% TTFT reduction, 7.8× prompt-prefilling throughput, 70% end-to-end inference cost reduction.
- **Accuracy:** Not reported.

### Mooncake (USENIX FAST 2025, Erik Riedel Award)

- **arXiv:** https://arxiv.org/abs/2407.00079 — **verified**
- **Authors:** Ruoyu Qin, Zheming Li et al. (Moonshot AI + Tsinghua)
- **Mechanism:** KVCache-centric disaggregated architecture separating prefill/decode; uses CPU/DRAM/SSD/NIC.
- **Speedup:** "Up to 525% throughput in simulated scenarios"; Kimi handles 75% more requests.
- **Accuracy:** Not reported (serving systems paper).

### Context-Folding (Oct 2025, ByteDance)

- **arXiv:** https://arxiv.org/abs/2510.11967 — **verified**
- **Authors:** Weiwei Sun, Miao Lu, Zhan Ling et al. (ByteDance Seed)
- **Mechanism:** branch/return primitives — agent folds sub-trajectories into cached summaries; KV-cache rollback at branch/return points.
- **Datasets:** "Deep Research + SWE long-horizon tasks" — **mentions SWE** but no benchmark detail.
- **Speedup:** 10× smaller active context vs ReAct.
- **Accuracy:** "Task performance vs ReAct" — **not pass@1**, not lossless.

---

## Tier 3: Code-Completion Work That Misses the KV-Cache Angle

### RepoCoder (EMNLP 2023)

- **arXiv:** https://arxiv.org/abs/2303.12570 — **verified**
- **Authors:** Fengji Zhang, Bei Chen, Yue Zhang et al. (Microsoft)
- **Mechanism:** Iterative retrieval-augmented code completion (retrieve similar snippets → feed back into prompt → regenerate). **NOT a KV-cache mechanism.**
- **Datasets:** **RepoEval** (proposed by the paper; covers line/API/function-body completion).
- **Accuracy:** Exact match improvement "by over 10%" over In-File baseline; no HumanEval/MBPP/SWE-bench numbers.
- **GitHub:** https://github.com/microsoft/CodeT/tree/main/RepoCoder

### RepoFusion (2023)

- **GitHub:** https://github.com/ServiceNow/RepoFusion
- **Mechanism:** Training code LMs to better use retrieved repo context. **NOT a KV-cache mechanism.**
- **No pass@1 numbers reported in the public docs.**

### ChainCoder / CodeChain (ICML 2023 / 2024)

- **ChainCoder GitHub:** https://github.com/VITA-Group/ChainCoder — Outline-then-Details coarse-to-fine generation; AST-sketch → fill-in. **NOT a KV-cache mechanism.**

### Improving FIM Code Completions via Context & Curriculum Learning (Dec 2024, Sourcegraph)

- **arXiv:** https://arxiv.org/abs/2412.16589 — **verified**
- **Authors:** Hitesh Sagtani, Rishabh Mehrotra, Beyang Liu (Sourcegraph)
- **Mechanism:** Training-time curriculum + context examples (via TSC compiler). **NOT a KV-cache mechanism.**
- **Datasets:** SantaCoder FIM, Amazon CCEval, **Multi-Line Infilling derived from SWE-bench**.
- **Accuracy metric:** Completion Acceptance Rate (CAR), Completion Persistence Rate (CPR). **NOT pass@1.**

### RepoFuse (Feb 2024)

- **arXiv:** https://arxiv.org/abs/2402.14323
- **Mechanism:** Fuses analogy context + rationale context; rank-truncated generation condenses into size-restricted prompts. **NOT a KV-cache mechanism** — it compresses context to fit the prompt window.
- **Dataset:** CrossCodeEval.
- **Accuracy:** +40.90% to +59.75% exact match over baselines.
- **Speedup:** 26.8% inference speedup.

---

## Tier 4: Production / Industry Reference (Not Academic)

### Anthropic Claude Code — Prompt caching production reality

- **Discovered:** 31 Mar 2026 — Claude Code v2.1.88 npm release accidentally shipped a sourcemap exposing 512K lines of TypeScript
- **Mirror:** https://github.com/jbang2004/inside-claude-code
- **Writeup:** https://www.cnblogs.com/yumingwen/p/19804977
- **Mechanism:** Layered memory (MEMORY.md always-loaded + on-demand fetch + autoDream nighttime consolidation); prompt-cache boundary at static-system-prompt layer; ~80% cost reduction via cache hits.
- **Accuracy:** Blog claims "accurately preserve coding behavior" — **no SWE-bench pass@1 comparison published.** This is a glaring citation opportunity if Anthropic ever publishes the number.

### Anthropic Prompt Caching (API)

- **Doc:** https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
- 4-tier (5-min, 1-hour) cached prefixes; minimum 1,024 tokens for Sonnet, 4,096 for Opus/Haiku; per-cache-checkpoint pricing.

### Google Gemini 1.5 Context Caching + NVIDIA TensorRT-LLM Context Cache

- Both provide native context caching APIs in commercial coding workflows.

---

## Tier 5: SCBench & SCOPE — KV-cache-centric benchmarks (no code)

### SCBench

- **Source:** CSDN writeup https://blog.csdn.net/weixin_49230371/article/details/158889398
- **Authors:** Microsoft Research Asia (Yuhan Yao et al.)
- **Composition:** 12-task, 2-mode shared-context benchmark covering string retrieval, semantic retrieval, global information, multi-task.
- **Models:** Llama-3.1-8B/70B, Qwen2.5-72B/32B, Llama-3-8B-262K, GLM-4-9B.
- **Code benchmarks:** **None.** Despite being the most KV-cache-centric public benchmark, SCBench deliberately omits code generation.

### SCOPE

- **arXiv:** https://arxiv.org/abs/2412.13649
- **Authors:** Jialong Wu et al.
- **Mechanism:** Selective KV cache compression in long-context generation.
- **No code benchmarks.**

### SWE-ContextBench (Feb 2026)

- **arXiv:** https://arxiv.org/abs/2602.08316 — **verified**
- **Authors:** Jiayuan Zhu, Junde Wu, Minhao Hu et al.
- **Composition:** 1,100 base tasks + 376 related tasks across 51 repos, 9 languages; related tasks derived from GitHub PR/issue dependency relationships.
- **What it measures:** "How accurately and efficiently agents solve related issues when prior cases are available in context" across varying context-reuse settings and retrieval strategies.
- **Accuracy metric:** "resolution accuracy" — measures agent success rate. **NOT pass@1 in the abstract.**
- **KV cache mechanism:** Tests full prefix prepending + retrieval strategies; **no lossy KV reuse** measured.
- **GitHub:** Not surfaced.
- **Relevance:** **Most directly relevant peer work** to your question. Explicitly frames "context reuse vs code-task accuracy" as the research question. Does NOT include lossy cache mechanisms, leaving a clear opening for your contribution.

---

## Tier 6: Negative Results / Empty Niches (Publishable Opportunities)

These searches returned zero peer-reviewed results:

1. **pass@1 vs compression-ratio plots on HumanEval** for any KV-compression technique
2. **F1 vs SWE-bench Verified pass@1 correlation studies** (would your F1=0.508 measurement predict pass@1? unknown)
3. **Prompt-cache hit-rate vs SWE-bench pass@1 trade-off curves**
4. **LMCache evaluated on any code-generation benchmark**
5. **CacheBlend applied to SWE-bench/SWE-agent prompts**
6. **AST-based chunk matching** for KV cache reuse (your work is novel on this axis)
7. **Round-trip semantic KV cache** with code-correctness measurement
8. **Function-level / module-level KV reuse** (your AST-gated L3 is the only published prototype)
9. **CodeGraph-style structural reuse** with pass@1 numbers
10. **RepoEval + KV cache reuse** combined evaluation

---

## Gap Analysis — Where Your Work Fits

### What exists

| Axis | Status |
|---|---|
| RAG chunk KV reuse (CacheBlend, RAGCache, CacheClip) | ✅ Mature; uses F1/EM |
| Multi-agent KV reuse (KVCOMM, TokenDance, TokenCake, CacheSlide) | ✅ Emerging; few use pass@1 |
| PIC for cross-position reuse (EPIC, MiniPIC, MEPIC, LazyAttention) | ✅ Mature; targets RAG, not code |
| Code completion retrieval (RepoCoder, RepoFusion, RepoFuse) | ✅ Mature; not KV-cache |
| AST-based code completion (ChainCoder) | ✅ Mature; not KV-cache |
| Lossless prompt caching for code (Claude Code, Gemini, GPT-4) | ✅ Production; no pass@1 published |

### What does NOT exist

| Axis | Status |
|---|---|
| AST/lexical/structural chunk detection for KV reuse | ❌ Your sglang-kvflow is the only published prototype |
| Function-level KV reuse with pass@1 on HumanEval/MBPP | ❌ Not found |
| pass@1 measurement under lossy KV reuse | ❌ KVCOMM reports it but on multi-agent embedding-based reuse, not code-specific AST |
| Code-completion benchmark (RepoEval/CrossCodeEval/HumanEval/MBPP/SWE-bench) under lossy KV cache | ❌ Not found |
| Round-trip semantic KV cache with code-correctness measurement | ❌ Not found |

### Where your work sits

**sglang-kvflow is ahead of the published literature on the (code-aware mechanism × code-correctness metric) intersection.** The F1 measurement you've done is itself a publishable contribution because the literature standard for KV cache reuse is even weaker (text similarity or no accuracy at all).

The natural next step — and a clean paper — would be:

1. Re-run your L3 (AST-gated byte-exact) and L4 (multi-slot copy) under pass@1 on HumanEval + MBPP + RepoEval (line/API/function-body completion)
2. Compare against KVCOMM-style embedding-based reuse (same datasets, same pass@1)
3. Show the F1→pass@1 correlation (your F1=0.549 may correspond to pass@1 in the 0.4–0.6 range, which would be a publishable "first empirical map of the trade-off")

This would be the first systematic treatment of pass@1 under lossy code-aware KV reuse, and it's a clean publication at ICLR/NeurIPS/ICML 2026/2027 with strong positioning against CacheBlend (RAG-only) and KVCOMM (embedding-only, not AST).

---

## Recommended Citations for Your Paper

### Mechanism precedents (RAG-style lossy reuse)

- CacheBlend (arXiv 2405.16444) — selective recompute
- Prompt Cache (arXiv 2311.04934) — modular prefix KV reuse
- RAGCache (arXiv 2404.12457) — hierarchical RAG cache
- SCBench (CSDN 158889398) — KV-cache lifecycle benchmark

### Position-independent caching

- EPIC (arXiv 2410.15332) — compile-link paradigm
- MiniPIC (arXiv 2606.13126) — <100 LOC implementation
- MEPIC (arXiv 2512.16822) — memory-efficient PIC
- LazyAttention (arXiv 2606.04302) — deferred positional encoding

### Multi-agent / cross-context

- KVCOMM (OpenReview yGOytgjurF) — **the only pass@1-on-HumanEval paper**
- DroidSpeak (arXiv 2411.02820) — cross-LLM coding-agent
- TokenDance (arXiv 2604.03143) — collective multi-agent
- TokenCake (arXiv 2510.18586) — multi-agent serving
- CacheSlide (GitHub SJTU-Storage-Lab) — cross-position agent reuse
- Context-Folding (arXiv 2510.11967) — branch/return + KV rollback

### Code-completion context (not KV)

- RepoCoder (arXiv 2303.12570) — RepoEval benchmark
- RepoFusion (GitHub ServiceNow) — repo-aware training
- RepoFuse (arXiv 2402.14323) — CrossCodeEval
- Sourcegraph FIM (arXiv 2412.16589) — SWE-bench-derived benchmark

### Benchmarking precedent for code-context-reuse

- SWE-ContextBench (arXiv 2602.08316) — most relevant peer work
- SWE-bench Verified (OpenAI 2024) — de facto pass@1 standard
- SWE-rebench (arXiv 2505.20411) — decontaminated SWE-bench
- Agentless (arXiv 2407.01489) — no-agent diff repair, 50.8% SWE-bench Verified

### Production reference

- Anthropic Claude Code prompt-cache architecture (leaked Mar 2026)

---

## Notes on Search Limitations

- WebSearch (the harness's standard tool) returned HTTP 400 for every query attempted
- All discovery was done via the secondary MCP `mcp__MiniMax__web_search` tool, whose corpus skews Chinese-language
- Some arxiv IDs (2604.03143, 2606.04302, 2606.13126, 2512.16822) appear very recent (2026) and were not all individually confirmed by abstract fetch (some returned only the abstract page metadata without deeper verification)
- The CacheSlide paper PDF was not fetched (it lives at github.com/SJTU-Storage-Lab/CacheSlide/blob/main/CacheSlide.pdf as a binary artifact); GitHub README was confirmed
- KVCOMM's OpenReview page was blocked by Cloudflare verification; the mechanism/numbers came from the marsggbo Chinese-language writeup, which quotes the paper directly
- Several near-misses (GitHub PRs about Cockroach KV, web-llm KVCache PR, etc.) were excluded as not peer-reviewed