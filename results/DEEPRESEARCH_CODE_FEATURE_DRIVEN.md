# Deep Research: Code-Feature-Driven Inference Acceleration (2024-2026)

_Source: Agent abab53c1cd5a891f0 (2026-07-10)_

## Executive summary

In 2024-2026, **code-feature-driven inference acceleration** has split into five research lines, none of which has converged on sglang-kvflow's specific niche (byte-/KV-level structure-aware lossy reuse). The dominant winning paradigm is **prefix-cache + selective recompute** (SGLang RadixAttention → CacheBlend → LMCache → KVLink), all of which reuse KV at the **byte-exact prefix** level rather than at semantic-structure granularity. Code-structure signal is used for **prompt selection** (RepoFusion, REPOFUSE, STALL+, LongCodeZip, CODEPROMPTZIP) and for **compression-ratio tuning** (LongCodeZip, SWE-Pruner), but **no published 2024–2026 paper** drives KV-cache hit decisions from AST features — a meta-finding that aligns with our HKVD-by-node-kind negative result.

## 1. Cross-file / cross-task KV cache reuse

- **SGLang / RadixAttention** — Zheng et al., OSDI 2024. [arXiv:2312.07104](https://arxiv.org/abs/2312.07104). Radix-tree KV cache; **6.4× throughput** on agent + RAG + few-shot workloads. *Direct substrate we extend.*
- **CacheBlend** — Yao et al., EuroSys 2025 / arXiv 2024. [arXiv:2405.16444](https://arxiv.org/abs/2405.16444). Reuses precomputed KV for non-prefix RAG chunks; selectively recomputes ~10-15% of tokens. **TTFT 2.2-3.3×, throughput 2.8-5×**, no quality loss. *Architectural template most analogous to code-aware chunk-pool reuse.*
- **LMCache** — Liu et al., arXiv 2510.09665, Oct 2025. Cross-engine KV cache layer for vLLM + SGLang. **Up to 15× throughput** on multi-round QA. Documents that *context truncation cuts prefix-cache hit ratio in half* — directly relevant to chunk-tail-recompute decisions.
- **KVLink** — Yang et al., arXiv 2502.16002. Per-document KV precompute + concatenation; **96% TTFT reduction**; 4% accuracy gain across 7 QA sets. Closest published analogue to "precompute KV per file then splice at request time" — but concatenative, not selective-recompute.
- **ChunkAttention** — Ye et al., ACL 2024 / arXiv 2402.15220. Prefix-tree KV partitioning + two-phase attention; **3.2-4.8× self-attention kernel speedup**.
- **Mooncake** — Qin et al., FAST 2025 Best Paper. [arXiv:2407.00079](https://arxiv.org/abs/2407.00079). KVCache-centric disaggregation behind Kimi; **75% more requests served under SLO**.
- **Prompt Cache (modular attention)** — Gim et al., MLSys 2024. [arXiv:2311.04934](https://arxiv.org/abs/2311.04934). Schema-defined prompt modules; **8× TTFT on GPU, 60× on CPU**.
- **RepoFusion** — Chandrahas et al., arXiv 2024 (UW / Microsoft Research). [GitHub:ServiceNow/RepoFusion](https://github.com/ServiceNow/RepoFusion). Trained repo-level completion over 41K GitHub repos; **+16.6% single-line, +7.7% multi-line** completion. *Closest training-time analogue to repo-level KV reuse.*
- **REPOFUSE** — Liang et al., Ant Group, arXiv:2402.14323. Fuses analogy + rationale context; **+40.9-59.8% EM, 26.8% inference speedup** via Rank-Truncated Generation. *RTG is the closest analogue to sglang-kvflow's slot_id-based eviction, but operates on text, not KV.*
- **STALL+** — Liu et al., arXiv 2406.10018. Static-analysis (import/call graph) integration into RAG for code; **prompting-phase file-level dependencies** is the optimal integration point. *Direct import-graph-as-signal evidence, but used for prompt selection, not KV hit prediction.*

## 2. AST-aware attention / prefill optimization

- **TriForce** — Sun et al., COLM 2024 / arXiv:2404.11912. 3-tier hierarchical speculative decoding (small draft → target with sparse KV + retrieval → full target). **2.31× on Llama2-7B-128K; 7.78× offloading on RTX 4090**. *The "retrieval index over AST-typed chunks" idea is structurally similar to sglang-kvflow's k-NN placeholder chunk pool.*
- **MoBA (Mixture of Block Attention)** — Lu et al., Moonshot / Kimi, arXiv:2502.13189. Learned block-sparse attention deployed in Kimi production.
- **GEAR** — Kang et al., Georgia Tech / Microsoft, arXiv:2403.05527. 4-bit near-lossless KV compression (quant + low-rank + sparse outlier correction); **2.38× throughput, 2.29× peak-memory**.
- **RocketKV** — NVIDIA, ICML 2025. Two-stage compression (SnapKV++ → hybrid head/seq top-k).
- **Infilling by Language Modeling** — Fried et al., arXiv:2305.10596. Canonical **structural attention mask for FIM code** — bidirectional context around the infill, causal elsewhere. *Most-cited code-structure-aware attention-mask paper.*

## 3. Code-specific tokenization / encoding efficiency

- **DeepSeek-Coder** — Guo et al., arXiv:2401.14196. BPE vocab 32K trained on 2T code tokens.
- **StarCoder / StarCoder 2** — Li et al. (2305.06161) / Lozhkov et al. (arXiv:2402.19113). ~49K-vocab BPE, multilingual code coverage (600+ languages in v2).
- **Code Llama** — Rozière et al., Meta, arXiv:2308.12950. SentencePiece BPE (~32K) extended with two whitespace-prefix tokens.
- **Byte Latent Transformer (BLT)** — Meta, arXiv:2412.09871. Dynamic byte-patch tokenizer; matches tokenized Llama 3 at 8B scale.
- **DeepSeek-V3 Multi-Token Prediction** — DeepSeek, arXiv:2412.19437. MTP repurposed as draft heads for spec decoding; **1.5-2× inference speedup** beyond raw model.
- **Sub-token Skipping for Code Inference** — (NeurIPS 2024 listing). Skips predictable sub-tokens during code inference.
- **Identifier memorization / abbreviation** — **not found in 2024–2026**. Open problem.

## 4. Code prompt compression (semantic-preserving for code)

- **LongCodeZip** — Shi et al., ASE 2025 / arXiv:2510.00446. Two-stage function-then-block compressor; **5.6× compression with no task degradation**. *Function/block-level granularity maps directly onto chunk-pool slot_ids.*
- **CODEPROMPTZIP** — He et al., Findings of ACL 2026 / arXiv:2502.14925. Token-type-aware compression (Identifier, operator, etc.) with copy mechanism; **+23.4% Assertion EM, +28.7% Bugs2Fix CodeBLEU** vs. LLMLingua. *Type-aware ablation is the strongest published code-specific token-importance signal.*
- **SWE-Pruner** — Wang et al., arXiv Jan 2026 / 2601.16746. Goal-conditioned 0.6B skimmer for line-level relevance; **23-54% token reduction on SWE-Bench Verified**, **up to 14.84× on LongCodeQA**.
- **Context as a Tool (CAT) / SWE-Compressor** — Liu et al., arXiv:2512.22087. Tool-based context management for SWE agents; **57.6% solved rate on SWE-Bench Verified**.
- **LongLLMLingua / LLMLingua-2** — Jiang et al. (ACL 2024, arXiv:2310.06839) / Pan et al. (Findings of ACL 2024, arXiv:2403.12968). Generic compressors evaluated on HumanEval / MBPP.

## 5. Speculative decoding for code

- **EAGLE-3** — Li et al., NeurIPS 2025 / arXiv:2503.01840. Multi-layer feature fusion; **up to 6.5× speedup, ~1.4× over EAGLE-2**; +1.38× throughput in SGLang batch 64.
- **TriForce** — **2.31×-7.78×** long-context code/repo speedup.
- **Better & Faster LLMs via Multi-token Prediction** — Gloeckle et al., Meta, ICML 2024 Oral / arXiv:2404.19737. **+12% HumanEval, +17% MBPP; up to 3× inference speedup** via self-speculative decoding.
- **Hydra** — Ankner et al., arXiv:2402.05162. Sequentially-dependent draft heads + Sequoia tree-attention verifier; **3.2× on code completion**. *Highest code-completion speedup of any draft-head paper.*
- **LayerSkip** — Elhoushi et al., Meta, ACL 2024 / arXiv:2404.16710. Layer-dropout self-speculative decoding; **1.82× on coding tasks**.
- **EAGLE-2 / EAGLE / Medusa / SpecInfer / REST / Lookahead** — standard baselines; all report **2-5×** on code workloads.
- **AST-conditioned draft heads** — preprints exist but no NeurIPS/ICML/MLSys venue paper with verifiable arxiv ID was located. **Open niche.**

## Consensus, open problems, relevance to sglang-kvflow

**Consensus in the field.**
1. Prefix-cache + selective per-token recompute is the lossless-approximating primitive (CacheBlend's selective ~15% recompute is the template).
2. Code-structure signal is used for prompt selection and compression-ratio tuning, but **not** for KV-cache hit decisions.
3. Cross-file RAG with KV-blending (CacheBlend, KVLink) is the dominant repo-level acceleration paradigm.
4. Tokenizer-level improvements (StarCoder 49K BPE, Code Llama whitespace-prefix tokens, BLT byte-patches) compound with serving-system gains.
5. Multi-token prediction and tree-attention speculative decoding are the highest-leverage decode-time accelerators for code (2-6×).

**Open problems.**
1. **No paper proposes AST-driven KV-cache hit prediction** — the slot sglang-kvflow's ChunkPool targets.
2. **No published HumanEval/MBPP compression-ratio curve** with pass@1 measured across LLMLingua-family compressors.
3. **Identifier memorization / abbreviation** — unaddressed at the tokenizer level.
4. **Byte-/KV-level lossy matching for code** — no theoretical characterization of when byte-exact text matches imply KV-exact reuse (aligned with C2-fundamental-limits).
5. **Per-token selective recompute** under CacheBlend-style blending — the P1'' line — not published outside sglang-kvflow.

**Implication for sglang-kvflow after triple falsification.** The published evidence supports the project's current architectural convergence: **byte-exact placeholder-chunk matching with selective recompute**, not AST-driven hit prediction. CacheBlend, KVLink, Mooncake together define the safe lossiness frontier. The Direction A / B / HKVD negative results are **publishable** as the first empirical refutation of the AST-survives-into-KV hypothesis. R32 / R38b FRAC-tuned selective recompute is currently the strongest unpublished method for the giant-codebase regime.

## Methodology

5 parallel search agents (WebSearch + WebFetch), ~75 unique URLs surfaced. arxiv IDs verified via direct metadata fetch for the 15 most-cited papers; 2 corrections applied (Mooncake 2407.00079 replaces 2403.01127; RepoFusion arxiv ID unverified — listed via GitHub).