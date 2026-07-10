# Deep Research: Practical Caching & Acceleration in Production Coding AI Systems (2023-2026)

_Source: Agent a93706dcb54c6df67 (2026-07-10)_

## 1. Executive summary: where production time is saved

Five orthogonal acceleration layers appear in production:

| Layer | Headline mechanism | Best evidence |
|---|---|---|
| **Provider-side prompt caching** | `cache_control` on static prefix | Anthropic: 11.5s→2.4s TTFT, 90% cost cut |
| **Custom small autocomplete models** | Per-keystroke tiny model + RL | Cursor Tab: 260ms p50; 400M req/day |
| **Speculative / deterministic-draft decoding** | Lossless token prediction | EAGLE-2 5× on HumanEval; Cursor 4–5× instant-apply |
| **Paged KV + prefix caching + RadixAttention** | Server-side prompt reuse | vLLM 2–4×, SGLang 5–6.4×, DeepSeek V3 56.3% KV hit |
| **Workload-specific model routing** | Tab vs Chat vs Agent | Codeium/Cody/Replit/Copilot all do this |

**Critical gap relative to R32 head_recompute:** No production coding AI system does *code-structure-aware lossy KV reuse*. Theirs is "keep the whole prefix verbatim or start fresh." The kvflow opportunity is the under-served middle.

## 2. Commercial systems

### Cursor (Anysphere)
- **Tab autocomplete** is a *custom from-scratch* 400M req/day model with **online RL** rolling new checkpoints every 1.5-2 hours from on-policy user data. Fusion Tab (Jan 2025): p50 **475ms→260ms**, context 5.5k→13k tokens, 25% more accepted edits.
- **Instant Apply** uses a custom `llama-3-70b-ft` with **deterministic speculative edits** — no draft model, exploits code-edit locality; **4-5× faster than next-fastest model**; trained specifically for full-file rewrites since "models struggle with diff-formatted edits."
- **Merkle-tree codebase indexing** (Jan 2026) splits files into syntactic chunks; **embeddings cached by chunk content (content-hash)**, not path; **simhash** dedup finds 92% similarity between code clones; semantic search boosts agent accuracy by **+12.5%**.
- **Tri-gram inverted indexes for grep** (Mar 2026) cut ripgrep from 15s+ to ms-scale on monorepos.

### GitHub Copilot
- **Prompt caching + tool search** (Jun 2026): "Copilot reuses model state for repeated prompt prefixes… and loads tool definitions on demand."
- **Next Edit Suggestions (NES)** is a *custom-trained* model using **SFT + RL** on internal edit sessions.
- **Repository-scoped memory** (Jan 2026): `store_memory` tool with verified-at-retrieval citations.

### Claude Code / Anthropic
- **Prominent cost saver is cache_control**, not lossy reuse. `cache_control: ephemeral` on tools→system→messages blocks; 5min default / 1h `extended` TTL; 4-breakpoint cap; **-90% cost, -85% latency** on 100k-token book (11.5s→2.4s TTFT).
- **Sub-agents run in separate context windows** with separate prompt caches ("a named subagent has a separate prompt cache from the main session; a fork shares the parent's prompt cache").
- **Orchestrator + parallel subagent** pattern at Anthropic: orchestrator saves plan to memory, launches parallel subagents, synthesizes.
- **`/compact` auto-compaction at ~92% context** — invalidates messages-tier cache prefix; tools + system prefix remain cached.

### Cody (Sourcegraph, open-source)
- **ContextMixer with Reciprocal Rank Fusion** (k=60, same as Azure Cognitive Search). Pluggable retrievers: JaccardSimilarity (100ms precompute on cursor move), `lsp-light` (Language Server Protocol), `tsc` (TypeScript compiler), `cached-retriever`.
- **Fast-path client** bypasses Sourcegraph server, talks directly to Cody Gateway → Fireworks AI for SSE streaming.
- **Hybrid `starcoder-hybrid` routing** (16b for multiline, 7b for single-line).

### Codeium / Windsurf
- **"Latency, the ultimate constraint"** (Feb 2024): root-caused autocomplete latency — embedding ~100ms, network 250ms California→India, inference scales with parameter count.
- **Workload-specific model strategy** (Aug 2024): autocomplete is a from-scratch small model; Command uses task-specific finetuning; Chat uses frontier models.
- **In-line FIM** — model predicts repairs to *contiguous tokens within a single line*; training masks out subsets of contiguous tokens. Codeium explicitly notes GitHub Copilot did *not* enable this in production.
- **Riptide** (proprietary LLM reranker) replaces brute-force "shove in retrieved chunks" retrieval.

### Replit Ghostwriter
- **Trains models from scratch for completion** with custom tokenizer, MosaicML training, WandB monitoring, **Flash Attention v2 Triton kernel**.
- **Code-repair distillation** from synthetic (code, LSP-diagnostic) pairs.
- **Progressive classification** pipeline: behavioral info → string matching → AST/LSP parsing → LLMs only when needed, controlling inference cost on petabyte-scale code.

### Tabnine
- Kubernetes-clustered GPU inference; **Universal models** (Pro) + customer-finetuned (Enterprise); Chat can fan out to Claude/GPT/Gemini/Devstral/Qwen-Coder. Enterprise Context Engine explicitly **critiques "reading tax"** ("brute-force prompting" where AI reads 50K irrelevant tokens).

## 3. Open-source coding agents

| Tool | Distinctive caching technique |
|---|---|
| **Aider** | 3 prompt-cache breakpoints + **5-min keepalive thread** `warm_cache()` pinging provider every 5 min to refresh 5-min TTL |
| **Continue.dev** | 25/25/50 hybrid retrieval (recent-edit / FTS / embeddings); content-hash embedding cache (cross-workspace); role-based router `{chat, autocomplete, apply, edit, embed, rerank}` |
| **Cline** | 3 cache markers on system + last 2 user msgs; 8-section structured `summarize_task` + `continuationPrompt` two-phase preservation; **ripgrep 256KB cap + mention dedup** to avoid context blow-up |
| **OpenHands** | BuildKit local cache + `keep_runtime_alive` container reuse (5min→<1min test instance startup); `LLMSummarizingCondenser` (claims 2× per-turn cost cut); `MultimodalRouter`/`RandomRouter`/`FallbackStrategy`; Anthropic cache on system + last user/tool; OpenAI auto-cache as 24h-retention pass-through |
| **Roo Code** (Cline fork, archived 2026-05-15) | Boomerang Tasks — clean context isolation between orchestrator and sub-task |
| **Kilo Code** (Cline fork) | Mid-task model swap (cheaper model for exploration, expensive for synthesis) |
| **Claude Code** | Sub-agent vs fork cache sharing; CLAUDE.md hierarchy (user/project/local) forms stable cacheable prefix |

## 4. Inference engines

### vLLM (UC Berkeley, PagedAttention, SOSP 2023)
- **2-4× throughput at same latency** vs FasterTransformer/Orca; Automatic Prefix Caching merged Jan 2024; production prefix-cache-hit case studies show first-token **1.4s → 380ms** (3.2×).

### SGLang (LMSYS, RadixAttention, NeurIPS 2024)
- **5-6.4× throughput** via radix-tree KV reuse; **automatically detects implicit sharing across unrelated requests** without manual hints (no overhead on cache miss); EAGLE-3 day-1 integration adds 1.38× at bs=64.

### TensorRT-LLM (NVIDIA)
- FP8 Llama 3.1 405B: 4,804 tok/s peak; FP4: 7,497 tok/s. **ReDrafter speculative decoding on Llama 3.1 405B**: 33→121 tok/s = **3.6×**.

### Speculative decoding — production options
- **EAGLE-2**: **5.00×** HumanEval on LLaMA2-Chat 13B @ T=0; 3.05-4.26× MT-bench.
- **EAGLE-3**: up to **6.5×**; 1.4× over EAGLE-2.
- **Medusa**: **3.29×** MT-Bench *coding* category on Vicuna-7B.
- **Lookahead**: **4× code completion** with strong multi-GPU scaling, no auxiliary model.

### Prompt caching (provider-level)
- **Anthropic**: 90% cost / 85% latency for cache reads; 4 breakpoint cap; `extended` 1h TTL.
- **OpenAI**: 50% off cached input (1.25× writes); 1024-token minimum prefix; up to 24h retention.

### Quantization for code
- **AWQ** (MLSys 2024 Best Paper): on MBPP/CodeLlama-7B-Instruct INT4, AWQ *beats* FP16 (40.64 vs 38.53 pass@1) and GPTQ (31.97). **GPTQ underperforms FP16 on code** — reconstruction loss hurts reasoning.

## 5. Code-specific model serving — DeepSeek at scale

DeepSeek's own V3/R1 inference disclosure is the most concrete:
- **Per H800 node:** prefill ~73.7k tok/s, decode ~14.8k tok/s.
- **Per user output: 20-22 tok/s;** avg KV-cache length 4,989 tokens/output token; **56.3% on-disk KV-cache hit rate** — repetitive code prompts.
- Prefill: routed-expert **EP32** + MLA/shared **DP32**; decode: **EP144**.
- **FP8** matmuls + BF16 core MLA; **$87k/day** fleet cost.
- SGLang reproduction on **12×8 H100** (~half the nodes): prefill 52.3k tok/s, decode 22.3k tok/s; **TTFT 2-5s, ITL ~100ms**; EPLB 1.49× prefill / 2.54× decode; cost **~$0.20/1M output tokens**.

## 6. Latency benchmarks & UX thresholds

- **Copilot <200ms median completion** (2022; replicated 2024).
- **Cursor Tab 100-200ms** target, p50 **260ms** Fusion.
- **OpenHands SWE-bench: ~14.2 min/task** (avg; full SWE-Bench Lite with gpt-4o ~$600).
- **Cursor cloud agents: 50M actions/day, 7M workflows, 40% of internal PRs**.
- **Anthropic's published cache hit numbers** (11.5→2.4s TTFT, 86% cost cut on 10k-token many-shot).
- **vLLM cache hit case study:** 1.4s cold → 380ms warm = 3.2× speedup, 60-70% token-cost savings.
- **Nielsen's 100ms / 1s / 10s** rules applied across all blogs: sub-200ms for inline; 1.2s for "fluent feel" copilots.

## 7. What kvflow R32 head_recompute has NOT considered

1. **Provider-side prompt caching** (`cache_control: ephemeral`) — Anthropic/OAI give 90%/50% cost cut for *no lossy accuracy*. **Make sure to measure against a `cache_control`-baseline R32 condition.** Aider's 5-min keepalive ping explicitly counteracts 5-min TTL.

2. **Deterministic speculative edits** (Cursor instant-apply) — 4-5× speedup without any lossy KV or draft model. Exploits that on *targeted multi-line edits*, output is highly predictable. Replaces draft-model training with explicit heuristic. **CacheBlend/head_recompute ignore this; could combine.**

3. **Custom small autocomplete model tier** — Cursor Tab 260ms p50, Replit custom 7B, Codeium in-line FIM, Cody `starcoder-hybrid` 7b/16b. Bypass KV reuse entirely by being small enough to prefill in tens of ms. **If a 0.5-3B code model gives acceptable output for the target task, TTFT 200ms parity beats any lossy-KV innovation at 100B+.**

4. **Online RL on acceptance/skip policy** — Cursor rolls Tab every 1.5-2h from on-policy user data; GitHub Copilot NES uses RL with learned grader. RL learns *when NOT to suggest* — complementary signal to lossy-KV "when NOT to reuse this prefix."

5. **Content-hash embedding/code cache** — Cursor chunk-content caching + simhash (92% clone similarity), Continue.dev content-hash dedup. **Cache key collisions across users' similar code** is underexploited relative to path-keyed caches.

6. **Tri-gram inverted indexes for grep** (Cursor fast-regex) — agent-side search latency, not inference. 15s+ → ms is bigger than any inference win.

7. **BuildKit / container warm-up** (OpenHands 5min→<1min test instance startup) — orthogonal to LLM KV but biggest end-to-end win for agent test loops.

8. **Architecture-specific MoE serving patterns** (DeepSeek V3: 56.3% real-world KV hit, PD-disaggregation via Mooncake, DeepGEMM grouped-GEMM, EPLB). For MoE code models (Qwen3-Coder-480B-A35B, DeepSeek-Coder-V2 236B/21B-active), the right "cache" might be *expert affinity* not KV block reuse — never measured.

9. **Reciprocal Rank Fusion for context retrieval** (Cody ContextMixer, k=60) — production-grade retrieval fusion is prerequisite to using retrieved context with lossy KV.

10. **AWS/Azure/Alibaba deployment-specific tuning** (FP8 vs FP4 vs INT4 for code) — quant-noise-as-regularization finding (AWQ > FP16 on MBPP) suggests **quantization-aware head recompute** could add FRAC beyond our current R32 0.30.

## 8. Key references (top 25, deduplicated)

| # | Topic | URL |
|---|---|---|
| 1 | Anthropic prompt caching (latency/cost numbers) | https://www.anthropic.com/news/prompt-caching |
| 2 | Cursor Tab Fusion 260ms p50 | https://cursor.com/blog/tab-update |
| 3 | Cursor Tab online RL 400M req/day | https://cursor.com/blog/tab-rl |
| 4 | Cursor deterministic speculative edits 4-5× | https://cursor.com/blog/instant-apply |
| 5 | Cursor Merkle + simhash codebase indexing | https://cursor.com/blog/secure-codebase-indexing |
| 6 | Cursor fast-regex trigram indexes | https://cursor.com/blog/fast-regex-search |
| 7 | Copilot prompt caching + tool search | https://github.blog/ai-and-ml/github-copilot/getting-more-from-each-token-how-copilot-improves-context-handling-and-model-routing/ |
| 8 | Copilot NES custom SFT+RL | https://github.blog/ai-and-ml/github-copilot/evolving-github-copilots-next-edit-suggestions-through-custom-model-training/ |
| 9 | Claude Code sub-agent cache separation | https://code.claude.com/docs/en/sub-agents |
| 10 | Anthropic orchestrator + subagent research | https://www.anthropic.com/engineering/built-multi-agent-research-system |
| 11 | Cody ContextMixer + RRF source | https://github.com/sourcegraph/cody-public-snapshot |
| 12 | Codeium "latency is the constraint" | https://codeium.com/blog/latency-the-ultimate-constraint |
| 13 | Codeium workload-specific model strategy | https://codeium.com/blog/our-model-strategy |
| 14 | Replit custom from-scratch code LLM | https://replit.com/blog/llm-training |
| 15 | vLLM PagedAttention (SOSP 2023) | https://arxiv.org/abs/2309.06180 |
| 16 | SGLang RadixAttention 5-6.4× | https://lmsys.org/blog/2024-01-17-sglang/ |
| 17 | EAGLE-2 5× HumanEval | https://arxiv.org/abs/2406.16858 |
| 18 | Lookahead 4× code completion | https://arxiv.org/abs/2402.02057 |
| 19 | Medusa 3.29× coding | https://arxiv.org/abs/2401.10774 |
| 20 | AWQ beats FP16 on MBPP pass@1 | https://arxiv.org/abs/2306.00978 |
| 21 | DeepSeek V3 serving numbers (56.3% KV hit) | https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md |
| 22 | SGLang DeepSeek V3 reproduction | https://lmsys.org/blog/2025-05-05-large-scale-ep/ |
| 23 | Aider prompt caching keepalive | https://aider.chat/docs/usage/caching.html |
| 24 | TensorRT-LLM FP8/FP4 perf | https://nvidia.github.io/TensorRT-LLM/performance/perf-overview.html |
| 25 | Artificial Analysis leaderboard | https://artificialanalysis.ai/leaderboards/models |

## Synthesis verdict

R32 head_recompute (1.43×, position-aware) is well-placed relative to lossy-KV alternatives (CacheBlend variants; all FALSIFIED), but **no production coding AI does code-aware lossy KV reuse at all** — they pick one of three regimes:
1. Keep the prefix verbatim → cache_control/speculative decode
2. Smaller model → fast prefill, no reuse
3. Async workflow → hours of latency tolerated

The genuine kvflow gap is *interactive agents with mid-sized context* (10-60k tokens) where cache_control alone doesn't suffice (less than 1024 tokens of *change*), but where a frontier MoE/Coder model is required. Target that window with: (a) cache_control baseline comparator, (b) deterministic speculative edits bolt-on, (c) content-hash chunk embeddings, (d) per-tenant MoE expert affinity.