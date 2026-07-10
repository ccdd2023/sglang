# Deep Research: Speculative Decoding (2024-2026)

_Source: Agent a93277b1f2858d64e (2026-07-10)_

---

## Section A — Required Foundational Papers

### A1. EAGLE (v1)
- **Title:** EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty
- **Authors:** Yuhui Li, Fangyun Wei, Chao Zhang, Hongyang Zhang
- **Venue/Year:** arXiv cs.LG, Jan 2024 (v3 Mar 2025); accepted ICML 2024
- **URL:** https://arxiv.org/abs/2401.15077 | Code: https://github.com/SafeAILab/EAGLE
- **Draft mechanism:** Lightweight auto-regression at the **feature level (second-to-top-layer)**, conditioned on the target LLM's last hidden state. Uses a one-step-advanced token sequence to mitigate feature-level uncertainty.
- **Acceptance criterion:** Distribution-preserving speculative sampling.
- **Speedup:** LLaMA2-Chat 70B latency speedup **2.7×–3.5×**; throughput ~2×.

### A2. EAGLE-2
- **Title:** EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees
- **Venue/Year:** arXiv, Jun 2024; EMNLP 2024
- **URL:** https://arxiv.org/abs/2406.16858
- **Draft mechanism:** **Context-aware dynamic draft tree** — draft tree shape recomputed each step using draft model's calibrated confidence.
- **Speedup:** **3.05×–4.26×** speedup; **20%–40% faster than EAGLE-1**.

### A3. EAGLE-3
- **Title:** EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test
- **Venue/Year:** arXiv, Mar 2025 (v3 Apr 2025); NeurIPS 2025
- **URL:** https://arxiv.org/abs/2503.01840
- **Draft mechanism:** Abandons top-layer feature prediction; uses **multi-layer feature fusion** via "training-time test" paradigm.
- **Speedup:** Up to **6.5×** across five tasks; **~1.4× over EAGLE-2**; **+38% throughput in SGLang at batch size 64**.

### A4. Medusa / Medusa-1 / Medusa-2
- **Title:** Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads
- **Authors:** Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason D. Lee, Deming Chen, Tri Dao
- **Venue/Year:** arXiv, Jan 2024 (v3 Jun 2024)
- **URL:** https://arxiv.org/abs/2401.10774 | Code: https://github.com/FasterDecoding/medusa
- **Draft mechanism:** Adds **multiple decoding heads** to the LLM; each head predicts a future token position. Medusa-2 fine-tunes heads + backbone jointly.
- **Speedup:** Medusa-1 **>2.2×** lossless; Medusa-2 **2.3×–3.6×**.

### A5. Lookahead Decoding
- **Title:** Break the Sequential Dependency of LLM Inference Using Lookahead Decoding
- **Authors:** Yichao Fu, Peter Bailis, Ion Stoica, Hao Zhang
- **Venue/Year:** arXiv, Feb 2024 (UC Berkeley / UCSD / Hao AI Lab)
- **URL:** https://arxiv.org/abs/2402.02057 | Code: https://github.com/hao-ai-lab/LookaheadDecoding
- **Draft mechanism:** **No draft model.** Uses **Jacobi iteration** to generate multiple lookahead steps in parallel.
- **Speedup:** Up to **1.8× on MT-bench**; up to **4× with strong multi-GPU scaling** on code completion.

### A6. SpecInfer
- **Title:** SpecInfer: Accelerating Generative LLM Serving with Speculative Inference and Token Tree Verification
- **Venue/Year:** ASPLOS '24; arXiv:2305.09781
- **URL:** https://arxiv.org/abs/2305.09781
- **Draft mechanism:** **Token tree verification** — small speculative models emit a token tree; target LLM verifies the entire tree in one parallel forward pass. **Collective boost** combines multiple draft models.
- **Speedup:** Distributed LLM serving **1.5×–2.8×**; offloading-based serving **2.6×–3.5×**.

### A7. REST
- **Title:** REST: Retrieval-Based Speculative Decoding
- **Venue/Year:** arXiv:2311.08252 (Nov 2023, v2 Apr 2024); NAACL 2024
- **URL:** https://arxiv.org/abs/2311.08252 | Code: https://github.com/FasterDecoding/REST
- **Draft mechanism:** **No draft model, no extra training.** Retrieves draft tokens from a datastore of n-gram continuations indexed by current context.
- **Speedup:** **1.62×–2.36×** on 7B/13B models, single-batch, code or text generation.

### A8. Ouroboros
- **Title:** Ouroboros: Generating Longer Drafts Phrase by Phrase for Faster Speculative Decoding
- **Venue/Year:** arXiv, Feb 2024 (revised Oct 2024); EMNLP 2024
- **URL:** https://arxiv.org/abs/2402.13720 | Code: https://github.com/thunlp/Ouroboros
- **Draft mechanism:** **Phrase-by-phrase draft generation** — uses the target LLM itself to generate candidate "draft phrases" (contiguous chunks). **Training-free**.
- **Speedup:** **Up to 2.8× over speculative decoding; up to 3.9× over vanilla decoding.**

## Section B — 2024–2025 Follow-ups

### B1. MagicDec (ICLR 2025)
- **Title:** MagicDec: Breaking Throughput-Latency Trade-off for Long Context Generation with Speculative Decoding
- **URL:** https://github.com/Infini-AI-Lab/MagicDec
- **Draft mechanism:** Both **standalone draft** and **self-speculation** modes; SnapKV-based drafting and StreamingLLM-style KV-budget drafting for long-context inference.
- **Hardware:** 8×A100, 8×H100, 8×L40.

### B2. PEARL (ICLR 2025)
- **Title:** PEARL: Parallel Speculative Decoding with Adaptive Draft Length
- **URL:** https://arxiv.org/abs/2408.11850 | Code: https://github.com/smart-lty/ParallelSpeculativeDecoding
- **Draft mechanism:** **Pre-verify** + **Post-verify** overlap of drafting and verification phases; adaptive draft length per scenario.
- **Speedup:** Up to **4.43× over auto-regressive**; up to **1.50× over vanilla speculative decoding**.

### B3. LayerSkip (ACL 2024)
- **Title:** LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding
- **URL:** https://arxiv.org/abs/2404.16710 | Code: https://github.com/facebookresearch/LayerSkip
- **Draft mechanism:** **Self-speculative decoding** — same model as both drafter (early layers) and verifier (remaining layers). Trained with layer dropout + early-exit loss.
- **Speedup:** Up to **2.16× on CNN/DM summarization**, **1.82× on coding**, **2.0× on TOPv2** semantic parsing.
- **Agent relevance:** No auxiliary draft model → minimal memory footprint per request, attractive for high-concurrency agent serving.

### B4-B7: Other 2025 references
- **SpecBranch** — ICLR 2026 (referenced; not fetched).
- **FastGRPO** (Sep 2025) — concurrency-aware speculative decoding for RL training.
- **Speculative Actions** (Oct 2025) — "lossless framework for agentic systems" (referenced; URL not retrieved).
- **"Speculative Thinking: How LLMs Think on the Fly"** — arXiv 2504.03591 (Apr 2025); fetch failed, content not retrieved.

## Section C — Agentic / Multi-Turn Applications

- **C1. Speculative Tool Use in Reinforcement Learning Agents** — Semantic Scholar listing only (2025). **Flagged.**
- **C2. "Speculative Decoding for LLM Inference Acceleration"** — arXiv 2509.17048 (Sep 2025). **Flagged.**
- **C3. Multi-Turn Speculative Decoding for LLM Agents** — direction exists but no primary paper retrieved.

## Section D — Cross-Cutting Notes on Tree Attention

Tree attention is the dominant verification primitive across modern speculative decoding:
- **SpecInfer** — token tree + tree attention mask
- **EAGLE-2 / EAGLE-3** — dynamic draft tree + tree attention
- **Medusa** — multi-head tree + tree attention
- **AdaServe** — SLO-customized speculative decoding with tree attention (EuroSys 2026)

Custom attention masks let the target LLM validate W×D candidate sequences in ~one forward-pass cost. When implemented with rejection-sampling, output distribution exactly matches the target model (lossless).

## Section F — Summary Table

| System | Year/Venue | Mechanism | Speedup |
|---|---|---|---|
| EAGLE | ICML'24 | Feature-level draft | 2.7–3.5× (70B) |
| EAGLE-2 | EMNLP'24 | Dynamic draft tree | 3.05–4.26× |
| EAGLE-3 | NeurIPS'25 | Multi-layer fusion + training-time test | up to 6.5×; +38% SGLang |
| Medusa-1 | 2024 | Multi-head + frozen backbone | >2.2× lossless |
| Medusa-2 | 2024 | Multi-head + joint FT | 2.3–3.6× |
| Lookahead | 2024 | Jacobi iteration (no draft model) | 1.8× MT-bench; 4× code |
| SpecInfer | ASPLOS'24 | Token tree + collective boost | 1.5–3.5× |
| REST | NAACL'24 | Retrieval draft (no training) | 1.62–2.36× |
| Ouroboros | EMNLP'24 | Phrase-by-phrase self-draft (training-free) | up to 2.8× over SD |
| MagicDec | ICLR'25 | Long-context SD (SnapKV/StreamingLLM) | n/a (PDF only) |
| PEARL | ICLR'25 | Adaptive draft length + pre/post-verify | up to 4.43× |
| LayerSkip | ACL'24 | Self-speculative (early-exit + layer dropout) | 1.82–2.16× |

## Section G — Agent-Workflow Relevance

- **Agent loops with repeated system prompts + tool schemas** → high acceptance-rate candidates for retrieval/REST-style drafts and Ouroboros phrase-drafting (common-prefix-heavy).
- **Multi-turn KV-cache reuse** is *not* the same as cross-turn speculative drafting. The above papers all assume single-request drafting; the "agentic speculative decoding" literature is sparse.
- **Self-speculative methods (LayerSkip, Kangaroo)** are most attractive for agent serving because no separate draft-model memory slot is needed per concurrent agent.

## Sources NOT Successfully Accessed

| Item | Issue |
|---|---|
| SpecInfer FlexFlow repo | Page is for FlexFlow-Train, not SpecInfer code |
| "Speculative Thinking: How LLMs Think on the Fly" (full text) | Fetch returned different paper |
| "Speculative Thinking: When LLMs Think Too Fast for Tool Use" | Fetch returned different paper |
| MagicDec numeric speedups | README doesn't include speedup figures; arXiv PDF needed |
| SpecBranch / FastGRPO / Speculative Actions | Only referenced via awesome-list; no primary URL |
| Kangaroo | Search returned no primary source |
| EAGLE/Medusa/Lookahead per-benchmark GPU specs | Not in abstracts |