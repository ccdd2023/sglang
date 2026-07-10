# Deep Research: Negative Results & Falsifications in Code-Structure-Aware Inference

_Source: Agent a1f24e20f72840a2b (2026-07-10)_

## Executive takeaway

Our triple falsification is consistent with a broader pattern: **explicit code structure often helps when used OUTSIDE the transformer as retrieval/planning/static-analysis context, but it is weak or absent as an internal attention/KV-cache salience signal**. The strongest practical KV-cache wins are position- or attention-history-based, not AST/node-kind-based. Few papers publish "code-aware KV reuse failed" directly; most negative evidence is indirect: AST linearization null results, explicit avoidance of AST in favor of sparse data-flow, attention interpretability failures, and production systems treating structure as a coarse retrieval index rather than a token-level reuse prior.

## 1. Direct code-structure-aware null/negative evidence

### CodeBERT — Zhangyin Feng et al., Findings of EMNLP 2020
**Title:** *CodeBERT: A Pre-Trained Model for Programming and Natural Languages*  
**URL:** https://arxiv.org/abs/2002.08155

**Negative/null finding:** CodeBERT's authors report that a version trained by traversing the AST "does not bring improvements on generation tasks." On C# code-to-NL generation, CodeBERT reaches **22.36 BLEU**, while **code2seq reaches 23.04 BLEU**, and the authors explicitly attribute code2seq's edge to compositional AST paths. Yet in code search, the plain token-sequence CodeBERT is very strong: **0.7603 macro MRR** on CodeSearchNet, beating sequence baselines.

**Explanation:** AST information is not automatically useful when serialized into the same sequence channel. Compositional AST paths can help generation, but naive AST traversal may destroy the locality/ordering statistics that transformer pretraining exploits.

### GraphCodeBERT — Daya Guo et al., ICLR 2021
**Title:** *GraphCodeBERT: Pre-training Code Representations with Data Flow*  
**URL:** https://arxiv.org/abs/2009.08366

**Finding:** This is a counterexample, but a revealing one. The paper deliberately chooses **data flow rather than AST**, arguing AST has an "unnecessarily deep hierarchy." Data flow helps, but modestly: code-search overall MRR drops from **0.713 to 0.693** when data flow is removed.

**Explanation:** Sparse semantic relations such as "where the value comes from" are more aligned with program behavior than dense syntactic tree boundaries. This supports a narrower claim: **some symbolic structure helps if sparse and semantic; AST node-kind boundaries are a poor KV-salience prior**.

## 2. Theory: why AST structure may not appear in attention/KV

### Michael Hahn, TACL 2020
**Title:** *Theoretical Limitations of Self-Attention in Neural Sequence Models*  
**URL:** https://arxiv.org/abs/1906.06755

**Finding:** Fixed-size self-attention cannot model certain periodic finite-state languages or hierarchical structures unless layers/heads grow with input length.

**Explanation:** Standard transformers are fixed-depth sequence processors; a single token or boundary can have diluted influence over long contexts. This gives a theoretical reason why AST boundaries might not survive as clean, reusable KV features.

### Sarthak Jain and Byron C. Wallace, NAACL 2019
**Title:** *Attention is not Explanation*  
**URL:** https://aclanthology.org/N19-1357/

**Finding:** Attention weights are often uncorrelated with gradient-based feature importance, and radically different attention distributions can yield the same prediction.

**Explanation:** Even if an attention head seems to focus on a syntactic region, that does not mean the model's computation causally depends on that structure. This aligns with our HKVD-by-node-kind result: **node role need not correspond to KV importance**.

## 3. Position-aware wins versus structure-aware losses

### StreamingLLM — Guangxuan Xiao et al., ICLR 2024
**Title:** *Efficient Streaming Language Models with Attention Sinks*  
**URL:** https://arxiv.org/abs/2309.17453

**Finding:** Keeping **initial tokens plus recent tokens** stabilizes long streaming inference. Supports **4M+ tokens** and reports up to **22.2×** speedup over sliding-window recomputation.

**Explanation:** The key signal is positional: early tokens act as "attention sinks" even when not semantically important.

### Scissorhands — Zichang Liu et al., NeurIPS 2023
**Title:** *Scissorhands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression at Test Time*  
**URL:** https://neurips.cc/virtual/2023/poster/72050

**Finding:** Retaining historically pivotal tokens yields up to **5× KV-cache memory reduction** with no quality loss, and up to **20×** combined with 4-bit quantization.

**Explanation:** Token importance is persistent in attention history; no AST or syntactic role is required.

### SnapKV — Yuhong Li et al., NeurIPS 2024
**Title:** *SnapKV: LLM Knows What You are Looking for Before Generation*  
**URL:** https://arxiv.org/abs/2404.14469

**Finding:** Uses an end-of-prompt observation window to select important KV positions per head, reporting **3.6× faster generation**, **8.2× memory efficiency**, and up to **380K context tokens** on one A100-80GB.

**Explanation:** The winning feature is per-head attention/position behavior, not code structure.

## 4. Production and adjacent falsifications

### Prompt caching: exact-prefix reality
- Anthropic requires **100% identical prompt segments** up to the cache breakpoint
- OpenAI requires **exact prefix matches** and reads the longest matching prefix

**Lesson:** Production prompt caching validates our "exact text match" constraint. AST-aware reuse cannot safely substitute for prefix/KV identity unless the underlying prompt prefix is identical.

### Aider repo map
**URL:** https://aider.chat/docs/repomap.html

Aider's repo map uses a graph-ranked symbol/signature map, defaulting to about **1K tokens**, but documents large-repo limits: even the map can exceed context, and it samples rather than includes every class/function.

**Lesson:** Structure is useful as **retrieval/navigation metadata**, not as a fine-grained inference-time KV reuse signal.

### STALL+ and RepoBench
**URLs:** https://arxiv.org/abs/2406.10018, https://iclr.cc/virtual/2024/poster/17776

RepoBench formalizes repository-level retrieval as a separate task; STALL+ finds static analysis helps most when injected during prompting, while post-processing is weakest and Python static analysis is constrained.

**Lesson:** Code structure helps upstream context selection more than downstream token recompute selection.

### vLLM speculative decoding caveat
**URL:** https://docs.vllm.ai/en/latest/features/speculative_decoding/

vLLM's speculative-decoding docs frame gains as workload-dependent: medium/low-QPS, memory-bound regimes; real gains depend on model, traffic, hardware, and sampling.

**Lesson:** Adjacent inference optimizations also fail when a clean algorithmic prior does not match runtime bottlenecks.

## Bottom line

The literature supports a refined version of our falsification: **code structure is not useless, but AST/node-kind structure is not a reliable internal KV/attention salience signal**. The robust pattern is:
- External symbolic structure can help retrieve or prompt
- Internal inference optimization succeeds when driven by exact prefix identity, position, attention sinks, or historical heavy hitters

## Sources

- [CodeBERT: A Pre-Trained Model for Programming and Natural Languages](https://arxiv.org/abs/2002.08155)
- [GraphCodeBERT: Pre-training Code Representations with Data Flow](https://arxiv.org/abs/2009.08366)
- [Theoretical Limitations of Self-Attention in Neural Sequence Models](https://arxiv.org/abs/1906.06755)
- [Attention is not Explanation](https://aclanthology.org/N19-1357/)
- [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453)
- [Scissorhands: Exploiting the Persistence of Importance Hypothesis](https://neurips.cc/virtual/2023/poster/72050)
- [SnapKV: LLM Knows What You are Looking for Before Generation](https://arxiv.org/abs/2404.14469)
- [RepoBench: Benchmarking Repository-Level Code Auto-Completion Systems](https://iclr.cc/virtual/2024/poster/17776)
- [STALL+: Boosting LLM-based Repository-level Code Completion with Static Analysis](https://arxiv.org/abs/2406.10018)
- [Aider repo map documentation](https://aider.chat/docs/repomap.html)
- [Anthropic prompt caching documentation](https://platform.claude.com/docs/en/docs/build-with-claude/prompt-caching)
- [OpenAI prompt caching documentation](https://developers.openai.com/api/docs/guides/prompt-caching)
- [vLLM speculative decoding documentation](https://docs.vllm.ai/en/latest/features/speculative_decoding/)