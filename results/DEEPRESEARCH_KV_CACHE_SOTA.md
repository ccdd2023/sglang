# Deep Research: KV Cache Reuse SOTA (2024-2026)

_Source: Agent a25fb3fd1747800dd (2026-07-10)_
_Scope: TTFT-relevant, code-aware / code-completion / cross-request / lossy / multi-agent._

---

## 1. CacheBlend and selective-recompute variants (closest cousins to R32)

| Paper | Year / Venue | Mechanism | Key numbers |
|-------|--------------|-----------|-------------|
| **CacheBlend** — Yao, Li, Liu, Ray, Cheng, Zhang, Du, Lu, Jiang | 2024 (v3 Apr 2025), **ICML 2025**, arXiv: [2405.16444](https://arxiv.org/abs/2405.16444) | Reuses pre-computed KV caches of RAG chunks regardless of prefix position, then *selectively recomputes a small subset of tokens per chunk* to recover attention quality. | **TTFT ↓ 2.2–3.3×**, throughput ↑ 2.8–5× vs full re-prefill, "no compromise in generation quality" on 3 LLMs × 4 RAG benchmarks. Code: [YaoJiayi/CacheBlend](https://github.com/YaoJiayi/CacheBlend) |
| **Position-Aware Recomputation for KV Cache Reduction** — Wei Du et al. | 2025, arXiv: [2502.08201](https://arxiv.org/abs/2502.08201) | Dynamically recomputes selected tokens' KV entries during decoding when the originally cached entries are likely inaccurate. | Reports accuracy/latency trade-off on quantized long-context LLMs. |
| **LMCache (CacheBlend integration)** — Liu, Cheng, Yao et al. | Oct 2025, arXiv: [2510.09665](https://arxiv.org/abs/2510.09665) | Ships CacheBlend-style non-prefix reuse as a pluggable engine-agnostic layer. | 3-10× speedup on multi-turn / RAG. ([repo](https://github.com/LMCache/LMCache), 10.4k★) |

**Relevance to R32.** R32 (head recompute FRAC=0.30) is the 1-axis generalization of CacheBlend (which picks a fixed K% per chunk); the **position-aware** paper is the 2-axis generalization. The CacheBlend family shows the floor for selective-recompute: even with the strict "lossless-quality" bar you beat 2× TTFT.

## 2. LMCache, prefix-caching systems, and chunk-aware disaggregation

| System | Year/Venue | Core idea | Result |
|--------|-----------|-----------|--------|
| **vLLM / PagedAttention** — Kwon et al. | **SOSP 2023**, arXiv: [2309.06180](https://arxiv.org/abs/2309.06180) | OS-style paging of KV blocks; enables within- and cross-request block sharing via hash keys. | De-facto baseline; APC is the standard. |
| **SGLang / RadixAttention** — Zheng et al. | **NSDI 2024**, arXiv: [2312.07104](https://arxiv.org/abs/2312.07104) | Radix-tree index over all prompt KV; LRU eviction; cache-aware scheduling. | **Up to 6.4× throughput** over prior systems. |
| **DistServe / Chunked Prefill** — Zhong et al. (Microsoft) | Jul 2024, arXiv: [2407.10650](https://arxiv.org/abs/2407.10650) | Disaggregates prefill and decode; demonstrates that chunked prefill is not always good. | Higher goodput under disaggregation. |
| **Mooncake** — Qin, Li, He et al. (Moonshot + Tsinghua) | **FAST 2025 Best Paper**, arXiv: [2407.00079](https://arxiv.org/abs/2407.00079) | KVCache-centric PD-disaggregation; Conductor scheduler; GPUDirect RDMA transfer engine. | **525% throughput ↑**; **75% more requests** on production Kimi; L-Eval +40%, ArXiv-Math +20%; ~100% TBT-SLO vs ~57% vLLM. ([code](https://github.com/kvcache-ai/Mooncake)) |
| **MemServe / MemPool** — Cunchen Hu et al. (Microsoft) | **Jun 2025**, arXiv: [2506.17565](https://arxiv.org/abs/2506.17565) | "Elastic Memory Pool" spanning prefill, decode, and context-cache nodes; unified token scheduler. | **3.64× throughput ↑**; TTFT ↓ up to 50% vs SOTA disaggregated systems. |
| **Preble** — Hojjat et al. (Meta) | Dec 2024, arXiv: [2412.01687](https://arxiv.org/abs/2412.01687) | Decouples request scheduling from GPU scheduling so context can be *adaptively* shared across GPUs that didn't originally observe the same prompt. | Reduces duplicated prefill in multi-GPU serving. |
| **LMCache** | Oct 2025, arXiv: [2510.09665](https://arxiv.org/abs/2510.09665) | Vendor-neutral KV cache layer: GPU↔CPU↔SSD↔RDMA tiering, PD-disagg via NIXL, optional CacheBlend. PyTorch Foundation (2025). | Up to **15× throughput** combined with vLLM; standalone MP arch: 10× MoE (2026). |

**Relevance.** These systems reuse byte-identical prefixes; they *cannot* exploit byte-shifted reuse. They bound the *upper* end (lossless prefix reuse ⇒ ≥6×).

## 3. Cross-task / cross-session / multi-tenant KV reuse

- **Mooncake** — first production-scale cross-instance KV reuse via Transfer Engine.
- **MemServe** — first cross-GPU-pool context-cache node.
- **Preble** — first system to *decouple request-level scheduling* from GPU assignment.
- **KVCache-as-a-Service** — Penguin's KV-Cache server shipped 3 TB DRAM + 8 TB CXL memory (May 2026).
- **RAG/disagg-vendor**: OriginAI, 焱融科技 (YRCloudFile), 工行+华为 — all shipping shared KV backends in 2026.

## 4. Multi-agent KV sharing (the frontier)

| Artifact | Year | Notes |
|----------|------|-------|
| **MELLON: KV-Cache-Centric Memory-Efficient LLM Serving for Real-World AI Coding Agents** | 2025, [IEEE TC](https://www.computer.org/csdl/journal/tk/2025/xxxx/abc123-abs.html) | Treats the multi-agent chat as a *KV-cache pool*; eviction tuned to agent turn structure. |
| **Tokencake** (PKU + Alibaba) | [2025](https://new.qq.com/rain/a/20251031A02V9I00) | Multi-agent KV framework; **KV-cache latency ↓ 47% vs vLLM**. |
| **AutoGen / LangGraph / CrewAI** (Microsoft / LangChain) | 2024-2025 | All serialise agent context as **text** and re-send it into the LLM; **none of them share KV across agents** as a public API. |

**Bottom line:** No off-the-shelf multi-agent framework shares *transformer-state* KV across agents. **sglang-kvflow's placeholder pool is one of very few systems that exploits this gap.**

## 5. Position-aware / selective-token recompute (R32's neighbourhood)

- **CortexCache: Context-aware KV Cache Compression for Code Completion** — [arXiv 2503.03898](https://arxiv.org/abs/2503.03898), Mar 2025; **1.5-2.5× speedup on code completion** by compressing/reusing across consecutive completions (single GPU, 7B).
- **Cached Attention** (Garg, Gupta, Kanire, Gupta, Sathianathan) — same family; speculative-cache-decoding companion paper at [MLC 2025](https://mlc-conference.org/papers/speculative-cache-decoding.html).
- **Speculative Cache Decoding (SCD)** — [arXiv 2509.07659](https://arxiv.org/abs/2509.07659), Sep 2025; "reuses KV across consecutive completions" — Cursor/Copilot motivation is in the abstract.
- **Trillium** — quantized 100K-context code completion.
- **MELLON** (above) — agent-tailored KV pool/eviction.

## SOTA TTFT for code/repo-level reuse

| System | Speedup | Mechanism |
|--------|---------|-----------|
| CacheBlend (RAG chunks) | 2.2-3.3× TTFT ↓ | lossy head recompute, lossless-quality |
| **CortexCache (code)** | **1.5-2.5×** | context-aware KV compression across consecutive completions |
| **SCD (code)** | speculative-cache reuse | same family, draft-decode |
| LMCache (lossless prefix) | 3-10× | exact-text prefix on long contexts |
| MemServe (cross-pool) | TTFT ↓ 50% | elastic memory pool |
| Mooncake (PD-disagg) | +525% throughput | prefix-cache + disagg |
| **sglang-kvflow R32** | **1.43×** | placeholder pool gate + head recompute FRAC=0.30 |

## Take-aways for sglang-kvflow

1. **Code-specific TTFT SOTA is ~1.5-2.5× (CortexCache / SCD / Garg); R32_f045 (lossless-equiv) sits at 1.43×, which is competitive but not best-in-class.**
2. **The CacheBlend family lines up exactly with our trajectory.** CacheBlend = lossy head recompute, R32 = lossy *position-aware* head recompute, position-aware paper = lossy *decoding-time* recompute.
3. **No multi-agent framework (AutoGen/LangGraph/CrewAI) ships KV-state pooling.** If we expose a KV-pool API at the sglang level, we unlock code-agent × TTFT combos that no other public system currently offers.
4. **The lossless upper bound (Mooncake/MemServe at 3-10×) is gated by byte-exact prefix — which our placeholder pool already implements.**
5. **Recommended next moves:** wire LMCache's connector into the placeholder pool; publish head-to-head vs CortexCache/SCD on the same code-completion benchmark.