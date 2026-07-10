# Deep Research: Multi-Agent Coding System TTFT/Latency (2023-2026)

_Source: Agent a063d836ecf0c99cc (2026-07-10)_

---

## 1. Multi-Agent Coding Frameworks

| Paper | Authors | Venue / Year | Headline Result | Relevance |
|---|---|---|---|---|
| **AutoGen** | Wu et al. (Microsoft) | arXiv:2308.08155; COLM 2024 | Generic conversable-agent framework | Closest orchestrator substrate for 5-agent voting — but defaults to passing full chat history |
| **MetaGPT** | Hong et al. (DeepWisdom) | arXiv:2308.00352; **ICLR 2024 Oral** | HumanEval 85.9%, MBPP 87.7%; SOP-driven 4-role pipeline | **Role specialization beats consensus voting** |
| **ChatDev** | Qian et al. (Tsinghua, OpenBMB) | arXiv:2307.07924; **ACL 2024** | 70-task waterfall; avg 13.23 hallucination bugs caught; 409s, $0.30 | Phase-gated review template |
| **CrewAI** | João Moura | OSS library (no paper), 2023-2025 | Hierarchical Process = LLM-driven manager | Direct analogue to inter-agent verdict |
| **AgentCoder** | Huang et al. (HKU + SJTU) | arXiv:2312.13010 (2023) | HumanEval 96.3%, MBPP 91.8%; **2-3× lower token cost than single-agent** | **Shield test designer from coder** to prevent shared blind spots |
| **MapCoder** | Islam, Ali, Parvez (BUET) | arXiv:2405.11403; **ACL 2024** | HumanEval 93.9%, MBPP 83.1%; 4-agent retriever/planner/coder/debugger | **Most relevant prior art** — debugger agent resolves disagreements |
| **Self-Collaboration Code Generation** | Dong et al. (Peking U) | arXiv:2304.07590; TOSEM | +29.9-47.1% Pass@1; 3 roles on **one model** | Direct precedent for one-LLM × N-roles, byte-exact KV reuse trivially correct |
| **Self-Refine** | Madaan et al. (CMU) | arXiv:2303.17651; **NeurIPS 2023** | ~20% absolute gain across 7 tasks | Generator → feedback → refiner as 3 prompt-switches on one model |
| **Reflexion** | Shinn et al. (NEU/MIT/Princeton) | arXiv:2303.11366; **NeurIPS 2023** | HumanEval 91% (GPT-4) | Maps to "5-agent disagreement → round-2 re-vote with reasoning" |
| **AutoCodeRover** | Zhang et al. (NUS) | arXiv:2404.05427; **ISSTA 2024** | SWE-bench-Lite 19%; $0.43/issue; AST-search single-agent | Contrast case: AST-aware single agent > 5 redundant voters |

## 2. Cross-Agent Context Sharing — the critical angle

**Verdict on KV-level cross-agent sharing:** **Exactly one published system** transfers actual KV tensors between distinct LLM instances: **DroidSpeak** (Liu et al., Microsoft Research, arXiv:2411.02820, Nov 2024). It targets cross-LoRA serving (Llama-3-8B ↔ fingpt-llama-3-8B), recomputes ~11% "critical" layers, and reports **1.7-3.1× prefill latency reduction** with negligible quality loss. Coding↔tester agent pairs explicitly named as motivating use case.

**Tier B — Text/prefix sharing, not raw KV:**
- **CacheBlend** (Yao et al., NeurIPS 2024 / arXiv:2405.16444): reuses chunk KV at *non-prefix* positions, recomputes 10-20% tokens → **TTFT 2.2-3.3×, throughput 2.8-5×**. **Caveat**: byte-exact text ≠ KV-exact when surrounding prefix differs.
- **Parrot / ParrotServe** (Microsoft, OSDI 2024): Semantic Variables expose inter-request dependencies
- **Prompt Cache** (Gim et al., Yale, MLSys 2024, arXiv:2311.04934): schema-declared reusable modules → **8× TTFT (GPU), 60× (CPU)**
- **SGLang RadixAttention** (Zheng et al., NeurIPS 2024, arXiv:2312.07104): compressed-trie prefix sharing, up to **6.4× throughput**
- **vLLM APC / PagedAttention** (Kwon et al., SOSP 2023, arXiv:2309.06180)
- **Anthropic Prompt Caching** (Aug 2024): `cache_control: ephemeral`, ~10% input price for cached tokens, 1h TTL
- **OpenAI Prompt Caching** (2024): auto-cache for prompts >1024 tokens, 50% discount

**Tier C — Shared memory/scratchpad (no KV):**
- **MemGPT / Letta** (UC Berkeley, arXiv:2310.08560)
- **MemoryBank** (arXiv:2305.10250)
- **A-MEM** (Alibaba, arXiv:2502.12110; **NeurIPS 2025**) — Zettelkasten-style structured notes
- **LangChain / LlamaIndex** memory types — buffered/summarized chat history

**Key gap:** No published work shares KV cache between agents in a *verdict-style* coding pipeline (5 agents on same canonical prefix, agreement → final verdict).

## 3. TTFT / Latency Optimization for Agent Systems

| System | Venue | Headline Result | Cross-request? |
|---|---|---|---|
| **MemServe / MemPool** | Hu et al. (Huawei), ASPLOS 2025 (arXiv:2406.17565) | **−78.5% avg / −84.9% P99 TTFT on ReAct agent workload** | Yes (token-radix + prompt trees) |
| **LMCache** | arXiv:2510.09665 | Single-node 1.9-8.1× TTFT; HyperPod 100 sessions × 2K shared × Llama-70B P90 1.21× | Yes (L0 GPU → L3 remote) |
| **Mooncake** | **FAST 2025 Best Paper** | 525% throughput (sim); **50% real-trace prefix reuse**; Kimi production at 128×H200 | Yes |
| **CacheBlend** | NeurIPS 2024 | 2.2-3.3× TTFT, breaks prefix-only barrier (lossy) | Yes |
| **ChunkAttention** | Ye et al., **ACL 2024** | 3.2-4.8× self-attention kernel speedup for shared 1024-4096-token system prompts | Yes |
| **AttentionStore** | NUS + Huawei, **USENIX ATC 2024** | TTFT −87-88% (70B/40B), prefilling 7.8-8.2× | Yes (tiered HBM→DRAM→SSD) |
| **LLMCompiler** | Berkeley, **ICML 2024** | **3.7× latency, 6.7× cost** via DAG-parallel tool calls | No (composable with KV reuse) |
| **DistServe** | PKU + UCSD, **OSDI 2024** | 7.4× more requests within SLO via PD disaggregation | Yes |
| **Splitwise** | Microsoft, **ISCA 2024** | 2.35× throughput at same cost via PD split | No |
| **SARATHI-Serve** | Microsoft, **OSDI 2024** | 2.6-5.6× serving capacity via chunked prefill | No |
| **EAGLE-2** | arXiv:2406.16858 | 3.05-4.26× lossless per-step speedup | No (per-token) |

## 4. Coding-Specific Multi-Agent Benchmarks

**Direct finding:** **No benchmark I found measures both (a) 3+ agent agreement on the same coding task AND (b) TTFT/latency as a first-class metric.** This combination is an open niche.

- **MapCoder** (ACL 2024) — 4-agent pipeline, debugger verifies via execution
- **MapCoder-Lite** (EACL Findings 2026) — 4 LoRA-specialised agents distilled into 7B base
- **ChatDev / MetaGPT** — 5-role systems, no formal latency metric
- **SWE-Gym** (**ICML 2025**, arXiv:2412.21139) — trains verifier for best-of-N; verifier-as-gate pattern
- **Self-Consistency** (Wang et al., 2022) — canonical majority-vote agreement metric
- **SWE-bench Verified** (500 tasks, OpenAI Aug 2024) — cleanest substrate
- **CodeArena** (HF Space, 2025+) — 5 agents head-to-head via Elo ranking
- **Multi-Agent Debate for Java Code** (arXiv:2503.17912, 2025) — explicit consensus via debate

## 5. Cross-Turn KV Reuse in Chat / Agent Systems

- **SGLang RadixAttention** (NeurIPS 2024): radix tree over KV blocks; LRU eviction; **3-5× higher hit rate than vLLM v0.3 prefix caching**; up to **6.4× throughput** on multi-turn chat
- **vLLM APC** (SOSP 2023): block-hash + parent-link chain; 2-4× throughput
- **Prompt Cache** (MLSys 2024): schema-declared modules; 8× TTFT (GPU), 60× (CPU)
- **Tokencake** (Beihang + PKU + Alibaba, arXiv Oct 2025): **KV-cache-centric multi-agent serving framework; 47% latency reduction vs vanilla vLLM** via spatial + temporal schedulers — most directly relevant paper
- **CacheBlend** (NeurIPS 2024): only published system that breaks prefix-only barrier; 2.2-3.3× TTFT, lossy
- **H2O** (NeurIPS 2023): heavy-hitter eviction, 29× throughput vs DeepSpeed-ZI at 20% budget
- **StreamingLLM** (ICLR 2024): attention sinks + sliding window, 22.2× speedup, 4M+ tokens

## Synthesis: What This Means for Our 5-Agent Verdict Pipeline

**The published cross-request KV reuse landscape has converged on two primitives:**
1. **Lossless byte-exact prefix matching** — SGLang RadixAttention, vLLM APC, ChunkAttention, MemServe. Safe; ~50% real-trace reuse ceiling (Mooncake production data).
2. **Lossy partial-chunk reuse** — CacheBlend. Up to 3.3× TTFT but accuracy not preserved for prefix-shifted text.

**Recommended stack for the 5-agent pipeline:**
1. **Substrate:** SGLang RadixAttention (already in place)
2. **Latency ceiling reference:** MemServe/MemPool's **−78.5% avg TTFT on ReAct**
3. **Production anchor:** LMCache's HyperPod case study
4. **Multi-agent scheduling innovation:** Tokencake's spatial + temporal scheduler split
5. **Research frontier:** DroidSpeak's selective layer recomputation applied to agent-variant (LoRA-differentiated role) settings — publishable and novel
6. **Hard constraint inherited:** exact-text-match is the gate for KV reuse. Per-agent content in canonical preamble → 0% hit rate.

**Benchmarking gap to publish:** A (5-agent agreement %, TTFT) joint-metric benchmark on SWE-bench Verified would be a new evaluation methodology.

## Primary Sources (URLs)

- AutoGen — arxiv.org/abs/2308.08155
- MetaGPT — arxiv.org/abs/2308.00352
- ChatDev — arxiv.org/abs/2307.07924
- AgentCoder — arxiv.org/abs/2312.13010
- MapCoder — arxiv.org/abs/2405.11403
- Self-Collaboration — arxiv.org/abs/2304.07590
- Self-Refine — arxiv.org/abs/2303.17651
- Reflexion — arxiv.org/abs/2303.11366
- AutoCodeRover — arxiv.org/abs/2404.05427
- **DroidSpeak (cross-agent KV)** — arxiv.org/abs/2411.02820
- CacheBlend — arxiv.org/abs/2405.16444
- Prompt Cache — arxiv.org/abs/2311.04934
- MemServe — arxiv.org/abs/2406.17565
- LMCache — arxiv.org/abs/2510.09665
- Mooncake — arxiv.org/abs/2407.00079
- AttentionStore — arxiv.org/abs/2403.19708
- ChunkAttention — arxiv.org/abs/2402.15220
- LLMCompiler — arxiv.org/abs/2312.04511
- DistServe — arxiv.org/abs/2401.09670
- SARATHI-Serve — arxiv.org/abs/2403.02310
- SGLang — arxiv.org/abs/2312.07104
- vLLM — arxiv.org/abs/2309.06180
- SWE-Gym — arxiv.org/abs/2412.21139
- A-MEM — arxiv.org/abs/2502.12110

## Unverified / Flagged

- **NVIDIA Dynamo 30× claim** — GTC 2025 marketing, no peer-reviewed paper
- **LCKV / CacheCraft** — codenames referenced but no canonical paper
- **Tokencake** — primary arXiv URL not directly fetched; cited from secondary sources
- **LMCache 15× throughput headline** — arXiv abstract only