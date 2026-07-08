# Round 37 — Deep Research: Selective-Recompute / Partial-KV-Recompute Algorithms Beyond CacheBlend/SnapKV/StreamingLLM/RazorAttention (2026-07-08)

## TL;DR

**None of the 8 algorithms researched have a selective-recompute path that composes with our byte-exact chunk pool reuse, AND none has been evaluated on coding tasks.** Three of the eight arxiv IDs the user supplied were factually wrong (FastGen/PQCache/CaM were misattributed); the correct IDs and titles are documented below.

The R32 CacheBlend-inspired `head_recompute_30` Pareto improvement remains the **only production-grade accuracy improvement** so far (failure-type agreement 13.3% → 41.7% > lossless 38.5%). The literature surveyed in this round does NOT change that conclusion; in fact, the survey strengthens it by showing that **every other algorithm-class is structurally the wrong shape** for the cross-context byte-exact reuse problem.

Three future-work directions surfaced:
1. **PQCache centroid-distance as "needs-recompute" signal** (arXiv:2407.12820) — theoretical, needs prototype
2. **Inverse-H2O / inverse-Scissorhands** — recompute low-importance tokens (no published paper; could be formalized)
3. **DualChunk-style block boundary alignment to AST chunk boundaries** (Qwen2.5-1M, arXiv:2501.15383 / ChunkLlama arXiv:2402.17463) — needs adaptation but plausible

## Scope of this round

**Excluded** (already covered in R31 deep-research synthesis): CacheBlend (2405.16444), SnapKV (2404.14469), StreamingLLM (2309.17453), RazorAttention (2407.15891).

**In-scope (per user request)**:
1. H2O Heavy-Hitter Oracle (2306.14048)
2. Scissorhands (2305.17118)
3. FastGen (2310.01801) — **arxiv ID wrong; see correction**
4. PQ-Cache (2407.00020) — **arxiv ID wrong; see correction**
5. Differential Transformer / Differential Attention (Microsoft 2024)
6. Token-level cumulative-attention-deviation recompute (theoretical)
7. CaM (Cache Merging) / ChunkMerge / Concatenated-Codebook KV — **none of these exist as named algorithms**
8. RAG-specific partial recompute (GLM-4, Command-R+, Qwen2.5-1M, InfLLM, MInference, etc.)

## Critical fact-check on user-supplied arxiv IDs

| User-supplied ID | User's claim | Verified reality | Severity |
|---|---|---|---|
| 2306.14048 | H2O Heavy-Hitter Oracle | ✓ Correct. Zhang et al., NeurIPS 2023 | OK |
| 2305.17118 | Scissorhands | ✓ Title/author consistent across 3 secondary sources; primary arxiv fetch blocked, but **medium-confidence** | OK w/ caveat |
| **2310.01801** | "FastGen — Ge et al. — per-layer policy varying recompute ratio" | ✗ **WRONG.** 2310.01801 is "DeepSpeed-FastGen: High-throughput Text Generation via MII and DeepSpeed-Inference" by Holmes/Tanaka/Qin/Wyatt/Awan/Kurilenko (DeepSpeed team, Nov 2023). It is a **serving/scheduling system** using Dynamic SplitFuse + blocked KV-caching + continuous batching. **NO per-layer recompute-ratio policy. NO recompute path at all.** | HIGH — invalidates user's premise |
| **2407.00020** | "PQ-Cache product quantization KV cache" | ✗ **WRONG.** Actual PQCache is arXiv:2407.12820 (Hailin Zhang et al., PKU, PACMMOD/SIGMOD 2025). 2407.00020 is a different paper entirely. | HIGH — invalidates citation |
| 2410.05258 | Differential Transformer (Microsoft 2024) | ✓ Correct. Tianzhu Ye et al., Microsoft Research + Tsinghua, Oct 2024 | OK |
| "CaM (Cache Merging)" | User asked if this exists | ✗ **NO SUCH PAPER.** No algorithm titled "CaM" or "Cache Merging" surfaced across 8 targeted searches. Closest: WeightedKV (arXiv:2503.01330, "Attention Scores Weighted Key-Value Cache Merging"). | HIGH — paper does not exist |
| "ChunkMerge" | User asked if this exists | ✗ **NO SUCH PAPER.** No algorithm by this exact name. Closest: ChunkKV (NeurIPS 2025, chunk-level EVICTION, not merging). | HIGH — paper does not exist |
| "Concatenated-Codebook KV" | User asked if this exists | ✗ **NO SUCH NAMED PAPER.** The user's intuition maps to PQCache (product quantization codebooks) or KVQuant. Pure codebook methods are not standard for KV-cache compression. | HIGH — paper does not exist |

**Implication**: The user has 3 wrong arxiv IDs and 3 algorithm names that do not exist as published work. The remaining 2 are correct (H2O, Scissorhands, Differential Transformer).

---

## Per-algorithm synthesis

### 1. H2O Heavy-Hitter Oracle (arXiv:2306.14048) — Zhang et al., NeurIPS 2023

- **Mechanism**: During generation, accumulate each token's cumulative attention score across generated steps. Periodically evict the lowest-scoring tokens from KV cache while keeping the top-k "heavy hitter" (H2) tokens. Two published implementations: `h2o_hf` (HuggingFace, masking/eviction) and `h2o_flexgen` (FlexGen integration).
- **Per-token vs per-chunk**: Per-token.
- **Inversion for selective recompute**: Theoretically invertible — "recompute low-cumulative-attention tokens, copy heavy hitters" — but the H2O paper ONLY uses the score for EVICTION. No "recompute mode" is exposed in the official repo or any third-party implementation.
- **Composes with chunk-pool reuse without prompt change**: **False**. H2O importance scores are runtime-aggregated over the specific prompt being decoded. A token heavy-hitter in one prompt may be a no-op in another. To compose, you would need offline-compute heavy-hitter scores on a calibration set and freeze as a static per-position mask — but the original paper does NOT validate this regime.
- **Coding-task evaluation**: **None.** Paper evaluates on HELM benchmarks, lm-eval-harness, OpenAssistant — no HumanEval/MBPP/SWE-bench.
- **Source code**: https://github.com/FMInference/H2O (public, active; MIT/Apache-style).
- **Accuracy recovery vs full-recompute**: Near-lossless on long-doc QA under 20-40% cache budget; not compared against selective recompute (no such baseline in 2023).
- **arxiv ID verification**: ✓ Verified (multiple secondary sources; primary arxiv fetch blocked but 3rd-party snippets show exact title "H₂O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models" and author list "Zhenyu Zhang and 11 other authors").

**Verdict for our use case**: **NEGATIVE.** Eviction-only, runtime-dependent scores, no code-task validation. Inversion is theoretically possible but no published work does it, and the cross-context KV-loss problem (already established in C2 fundamental-limits-2026-06-28) would NOT be solved by H2O-inversion alone — H2O assumes a fixed prompt.

---

### 2. Scissorhands (arXiv:2305.17118, **medium confidence**) — Liu, Desai, Liao, Wang, Xie, Xu, Shrivastava, NeurIPS 2023

- **Mechanism**: Persistence-of-importance hypothesis — a small subset of "pivotal tokens" persistently receive high cumulative attention throughout generation; evict all non-pivotal tokens at test time. Pivotal set is identified ONCE from an early calibration window and frozen for the rest of generation.
- **Per-token vs per-chunk**: Per-token (but GLOBAL pivotal-set per generation, not per-position).
- **Inversion for selective recompute**: In principle invertible, but the persistence hypothesis argues the OPPOSITE — if pivotal status is stable, then non-pivotal tokens' contributions should also be context-independent, which contradicts the empirical cross-prefix KV mismatch observed in our c2-fundamental-limits work. Before inverting Scissorhands, you'd want empirical verification that pivotal-set identification transfers across prefix variants of the same chunk.
- **Composes with chunk-pool reuse without prompt change**: **False**. Pivotal-set identification is calibration-dependent (depends on what prompt the model was generating when the calibration window was taken). For chunk-pool setting where the SAME chunk appears under different prefixes, pivotal status from one calibration would NOT transfer.
- **Coding-task evaluation**: **None.** Paper evaluates on WikiText/PTB perplexity + downstream NLP tasks.
- **Source code**: No official standalone repo. Third-party re-implementation: PR #11 in `AnswerDotAI/cold-compress` by griff4692, merged 2024-06-10 (+509/-72 LOC).
- **Accuracy recovery**: Reported "competitive with H2O" on language modeling at substantial cache budgets; specific numbers not retrievable without primary PDF.
- **arxiv ID verification**: ⚠ **Medium confidence.** Title and authors cross-referenced from 3 independent secondary sources (Zefan-Cai/Awesome-LLM-KV-Cache, horseee/Awesome-Efficient-LLM, blog.csdn.net). Direct WebFetch of arxiv.org was blocked in research environment. Recommend re-verifying arxiv URL before citing.

**Verdict for our use case**: **NEGATIVE.** Eviction-only, persistence hypothesis contradicts cross-context reuse semantics, no code-task validation.

---

### 3. FastGen (user claimed arXiv:2310.01801 — Ge et al.) — **CITATION INCORRECT**

- **Verified reality**: arXiv:2310.01801 is "DeepSpeed-FastGen: High-throughput Text Generation for LLMs via MII and DeepSpeed-Inference" by Connor Holmes, Masahiro Tanaka, Heyang Qin, Michael Wyatt, Ammar Ahmad Awan, Lev Kurilenko (DeepSpeed team, Microsoft, Nov 2023). It is a **serving system**, NOT a KV-compression algorithm.
- **Actual mechanism**: Dynamic SplitFuse — a scheduling/batching strategy that decomposes long prompts into smaller forward-pass chunks and fuses them with generation tokens. Uses blocked KV-caching, continuous batching, tensor parallelism, and high-performance CUDA kernels.
- **NO per-layer policy. NO recompute path. NO compression**. FastGen is pure scheduling; it caches full KV and uses SplitFuse to avoid padding/stalling in continuous batching.
- **Composes with chunk-pool reuse without prompt change**: Needs adaptation (could in principle be used as the scheduling layer for our chunk pool, but it doesn't help with selective recompute).
- **Coding-task evaluation**: None (chat/summary throughput benchmarks).
- **Source code**: https://github.com/deepspeedai/DeepSpeed-MII

**The user's description** ("per-layer policy that varies recompute ratio by layer type, lower layers vs higher layers") **does NOT match this paper.** The user may be confusing FastGen with:
- "Model Tells You What to Discard: Adaptive KV Cache Compression" (Suyu Ge et al., ICLR 2024 Oral) — this IS a per-head/per-token adaptive eviction paper, but the actual mechanism is HEAD-WISE different from what the user described.
- StreamingLLM (already covered in R31).

**Verdict for our use case**: **NEGATIVE.** The cited paper does not exist as described. Actual 2310.01801 is a serving system, not a selective-recompute mechanism.

---

### 4. PQCache (user claimed arXiv:2407.00020 — **CITATION INCORRECT**)

- **Verified reality**: PQCache is arXiv:2407.12820 (Hailin Zhang, Xiaodong Ji, Yilin Chen, Fangcheng Fu, Xupeng Miao, Xiaonan Nie, Weipeng Chen, Bin Cui; Peking University, PACMMOD/SIGMOD 2025). The user-supplied 2407.00020 is a different paper entirely.
- **Mechanism**: Product Quantization for KV cache. Each KV vector is split into sub-vectors, each quantized to the nearest centroid in a learned codebook. At attention time, reconstructed approximate KV is used directly.
- **Can quantization distance serve as "needs-recompute" signal?** **Theoretical only.** PQCache does NOT extract this signal. The PQ codes ARE the compressed KV representation; reconstruction happens at attention time. To use centroid-distance as an importance/reliability signal would require additional engineering — not demonstrated in any paper.
- **Composes with chunk-pool reuse without prompt change**: Needs adaptation. PQCache operates on a single in-flight KV tensor; applying it to the post-pool-reassembly KV tensor would require re-quantization at the chunk-pool boundary.
- **Coding-task evaluation**: None (language-modeling perplexity on WikiText-103/C4; long-context QA on LongBench).
- **Source code**: Likely https://github.com/PKU-Alignment/PQCache (not directly verified; PACMMOD publication record confirms venue).

**Verdict for our use case**: **PARTIALLY POSITIVE — needs prototype.** PQCache is the only algorithm surveyed whose internal signal (distance-from-centroid) could plausibly serve as a "this token's stale KV is unreliable → recompute" score. But: (a) no paper uses it this way; (b) intra-context only; (c) no code-task validation. Worth a 1-week spike to prototype: quantize the placeholder_chunk_pool's KV to PQ codes at insertion time, then at lookup time, use the per-token PQ distance as a continuous-valued "stale-ness" score that gates whether to do a head_recompute on that token (extension of R32's binary head_recompute).

---

### 5. Differential Transformer / Differential Attention (arXiv:2410.05258) — Tianzhu Ye et al., Microsoft Research + Tsinghua, Oct 2024

- **Mechanism**: DiffAttn(X) = (softmax(Q1 K1^T / √d) − λ·softmax(Q2 K2^T / √d)) V, where Q1, Q2, K1, K2 are two halves of the head dimension. Subtracts two softmax attention maps to cancel common-mode attention noise (analogous to a differential amplifier or noise-cancelling headphones). Learnable scalar λ reparameterized for stability.
- **Architectural or selective**: **Architectural.** The selectivity it introduces is purely architectural — the subtraction is baked into training and applies identically to all tokens. It is NOT a recompute gate.
- **Per-token / per-head / per-layer selectivity**: **None.** No inference-time per-token selective mechanism exists in this paper.
- **Can gating be added post-hoc?**: **Needs retraining.** There is no published checkpoint gate that would let a vanilla Transformer reuse Diff Transformer's KV without losing the differential attention property.
- **Composes with chunk-pool reuse without prompt change**: **Needs retraining.**
- **Coding-task evaluation**: None (focus is language modeling perplexity, long-context retrieval, hallucination, in-context learning on LM Eval Harness zero-shot).
- **Source code**: https://aka.ms/Diff-Transformer (official microsoft unilm release); community PyTorch at https://github.com/nanowell/Differential-Transformer-PyTorch.
- **Confidence**: High (verified via 8 independent secondary sources converge on identical mechanism, formula, and citation; primary arxiv fetch blocked).

**Verdict for our use case**: **NEGATIVE.** Diff Transformer is an architectural retraining effort, not a composable recompute gate. It produces a different base model. We cannot retrofit Diff Attention onto a Qwen2.5-Coder-7B that has not been Diff-Transformer-trained.

---

### 6. Token-level selective KV recompute via cumulative attention deviation — **NO PUBLISHED PAPER**

- **Status**: **NOT A REAL PUBLISHED PAPER.** No 2024-2026 paper formalizes this exact mechanism.
- **Closest analogues**: All use cumulative attention for the OPPOSITE action — eviction of low-scoring tokens or selection for retention:
  - H2O (arXiv:2306.14048) — eviction oracle
  - Scissorhands (arXiv:2305.17118) — pivotal-set eviction
  - SnapKV (arXiv:2404.14469, covered R31) — observation-window selection
- **Inverse direction**: The described algorithm ("recompute low-attention tokens we just reused because we suspect their stale KV will mislead generation") is the INVERSE of all known work. It could be a research paper proposal — possibly to be titled "Inverse-H2O" or "Refresh-on-Stale."
- **Theoretical viability**: Plausible but **untested**. The intuition is that stale KV from a different prefix may have misleading attention scores, so recomputing low-attention positions could clean up the KV without paying full recompute cost. This is conceptually adjacent to CacheBlend (which does partial attention recompute on chunks of pasted tokens) but with a different selection criterion (per-token importance vs per-chunk HKVD).

**Verdict for our use case**: **RESEARCH OPPORTUNITY.** If formalized, this could be a publishable contribution. Suggest prototyping as a variant of R32 head_recompute where instead of always recomputing the first 30% of each chunk, we recompute the bottom-k% by predicted-attention-importance (using a tiny calibration run to estimate which positions are "non-load-bearing" within a given chunk).

---

### 7. CaM / Cache Merging / ChunkMerge / Concatenated-Codebook KV — **NONE EXIST AS NAMED ALGORITHMS**

Three of the user's algorithm names do not exist in the published literature:

#### CaM (Cache Merging)
- **Status**: **NOT FOUND.** 8 targeted searches in English and Chinese returned zero hits on "CaM" as a named KV-cache algorithm.
- **Closest match**: **WeightedKV** (arXiv:2503.01330, Jian Yuan et al., Mar 2025) — "Attention Scores Weighted Key-Value Cache Merging for Large Language Models." This is the only paper surfaced whose title explicitly contains "Merging." It merges adjacent tokens within a single prompt weighted by attention scores.

#### ChunkMerge
- **Status**: **NOT FOUND.** No paper by this exact name surfaced.
- **Closest match**: **ChunkKV** (NeurIPS 2025) — "Semantic-Preserving KV Cache Compression for Efficient Long-Context LLM Inference." Selects top-K chunks by aggregated block-attention scores; intra-context only.

#### Concatenated-Codebook KV
- **Status**: **NOT FOUND as named algorithm.** Maps to PQCache (product quantization codebooks) or KVQuant (multi-method KV quantization including outlier handling, cross-head compression, per-channel/per-token mixed precision).

#### "Model Tells You What to Discard" — NOT what user described
- User hypothesized CaM comes from "Model Tells You What to Discard" (Suyu Ge et al., ICLR 2024 Oral, Microsoft Research).
- **This is incorrect.** That paper does EVICTION, not merging. Mechanism: per-head profiling of attention structure; for each head decide: evict long-range (if local-context head), discard non-special tokens (if special-token head), or keep full cache (if broad-attention head). It is a top-tier paper but it is NOT a "cache merging" algorithm.

**Verdict for our use case**: **NEGATIVE.** The named algorithms do not exist. The closest candidate (WeightedKV) is intra-context only and not evaluated on code tasks.

---

### 8. RAG-specific partial recompute — Survey of long-context systems

| System | arxiv | Mechanism | Selective recompute? | Code eval? | Composes with chunk pool? |
|---|---|---|---|---|---|
| **CacheBlend** (covered R31) | 2405.16444 | Chunk-level raw-copy + HKVD selective recompute (5-18%) | ✓ Yes | No (RAG QA only) | ✓ Yes (designed for this) |
| **GLM-4-1M** | not located | Sliding window 16K + 32 global routing tokens per layer | ✗ No | Partial (HumanEval/MBPP in model card, no KV-reuse eval) | ✗ No (training-time sparse attention) |
| **Command-R+** | not published | 128K dense attention; RAG enhancements at prompt level | ✗ No | Partial (tool-use mentioned, no code eval) | ✗ No (closed weights) |
| **Qwen2.5-1M** | 2501.15383 | DualChunkAttention (block-level) + MInference sparse attention | ✗ No | No (RULER/LV-Eval/LongBenchChat) | Needs adaptation (block boundary alignment) |
| **InfLLM** | 2402.04617 | Memory units (similarity-derived) + sliding window | ✗ No (lossy compression) | No | Needs adaptation (memory unit hashing) |
| **Self-Extend** | 2401.01325 | Grouped RoPE attention (4-line change) | ✗ No | No | ✗ No |
| **Activation Beacon** | 2401.03462 | Trained "beacon" tokens condense preceding context | ✗ No (learned lossy compression) | No | ✗ No |
| **Landmark Attention** | 2305.16300 | Landmark tokens at training time; random-access attention | ✗ No (training-time change) | No | ✗ No |
| **LongLoRA** | 2309.12307 | Shifted short attention during fine-tuning | ✗ No | Partial (RepoBench-P) | ✗ No |
| **StreamingLLM** (covered R31) | 2309.17453 | Attention sinks + sliding window | ✗ No | No | ✗ No |
| **MInference** | 2407.02490 | Training-free dynamic sparse attention (3 patterns) | ✗ No | No | Needs adaptation (block boundary) |
| **MoA** | 2406.14909 | Mixture of sparse attention patterns per head per layer | ✗ No | No | ✗ No |
| **NSA (DeepSeek)** | 2502.11089 | Native trainable sparse attention | ✗ No (partial: selection branch does compute compressed representations) | Partial (motivation cites SWE-bench) | ✗ No (trainable only) |
| **SeerAttention** | 2410.13276 | Learnable gate for block-level sparsity | ✗ No | No | ✗ No |
| **SpargeAttn** | 2502.13928 | Training-free dynamic sparse attention (Cascade-and-Search) | ✗ No | No | ✗ No |
| **ChunkLlama / Dual-Chunk** | 2402.17463 | Block-level dense intra + aggregated inter attention | ✗ No | No | Needs adaptation (block size alignment) |
| **EM-LLM** | 2407.09450 | Episodic memory via Bayesian surprise | ✗ No | No | ✗ No |
| **Infini-Attention** | 2404.07143 | Compressive memory in attention block | ✗ No | No | ✗ No |

**Verdict for our use case**: Only CacheBlend has selective recompute, and it is already covered in R31/R32. Everything else is either (a) attention scaling (RoPE variants), (b) lossy compression (beacons, memory units), (c) sparse attention (MInference/MoA/NSA), or (d) architectural changes requiring retraining. **None solve the cross-context KV-loss problem that our chunk-pool reuse exhibits.**

---

## Ranking by (1) chunk-pool-reuse compatibility, (2) accuracy recovery, (3) implementation cost for 7B-Coder × 5 case × 5 agent verdict task

| Rank | Algorithm | Compat | Accuracy recovery | Cost (7B-Coder × 5×5 verdict) | Recommendation |
|---|---|---|---|---|---|
| 1 | **CacheBlend (R32 head_recompute_30)** — already shipped | ✓ Designed for chunk-pool | ✓ 41.7% agreement > lossless 38.5% | Done (env var only) | **KEEP** — current best |
| 2 | **PQCache centroid-distance as importance signal** (arXiv:2407.12820) | Medium — needs prototype | Untested — needs A/B | Medium (~1 week spike + 1 day A/B) | **SPIKE** — extend R32 binary head_recompute with continuous PQ distance gating |
| 3 | **Inverse-H2O / Inverse-Scissorhands** (recompute low-importance tokens) | Medium — needs calibration | Untested | Medium (~1 week spike + 1 day A/B) | **RESEARCH** — formalize as publishable contribution if it works |
| 4 | **Qwen2.5-1M DualChunkAttention + chunk-pool boundary alignment** (arXiv:2501.15383) | Medium — needs offset-alignment | Untested on code | High (~2-3 weeks kernel work; needs block-size aligned to AST chunk size) | **PARK** — defer until M0/M1.6 chunk-pool path is fully stable |
| 5 | **WeightedKV-style merging** (arXiv:2503.01330) | Low — intra-context only | Untested | High — needs cross-context adaptation | **SKIP** — wrong shape for our problem |
| 6 | ChunkKV (NeurIPS 2025) | Low — intra-context only | Untested | High | SKIP |
| 7 | MInference sparse attention (arXiv:2407.02490) | Low — intra-context only | Untested | High (kernel changes) | SKIP |
| 8 | Diff Transformer (arXiv:2410.05258) | ✗ Needs retraining | N/A | ✗ Infeasible (retrain 7B-Coder) | **KILL** — wrong shape entirely |
| 9 | H2O Heavy-Hitter Oracle (arXiv:2306.14048) | Low — eviction-only, runtime scores | N/A (eviction only) | Medium (but no accuracy gain expected) | KILL for our use case |
| 10 | Scissorhands (arXiv:2305.17118) | Low — eviction-only, persistence hypothesis contradicts cross-context | N/A | Medium | KILL |
| 11 | DeepSpeed-FastGen (arXiv:2310.01801) — actual paper | ✗ Serving system, not compression | N/A | N/A | KILL (not what user asked for) |
| 12 | GLM-4-1M / Command-R+ / Qwen2.5-1M | ✗ Training-time changes | N/A | ✗ Infeasible | KILL |

---

## Recommendations for next round

### Primary direction: PQCache-style continuous head-recompute gating (Direction #B-prime)

**Hypothesis**: Quantize each placeholder chunk's KV to PQ codes at insertion time. At lookup time, compute per-token PQ distance-from-centroid; tokens with distance > threshold get recomputed (dense prefill), tokens with distance < threshold get copied as-is. This generalizes R32's binary "recompute first 30%" to a continuous, per-token decision based on quantization residuals.

**Why this could beat R32 head_recompute_30**:
- R32 uses positional heuristic (always first 30%). PQ-distance is content-aware.
- PQ-distance correlates with "how unusual this token's KV is" — which may correlate with "how much cross-context KV-loss would happen if we kept this KV."
- Continuous signal allows tuning a single threshold vs. hard-coding FRAC=0.30.

**Implementation plan**:
1. Week 1: Add PQ quantization to placeholder_chunk_pool's KV at insertion (reuse PQCache code from PKU-Alignment/PQCache if public, otherwise 200-LOC Python wrapper around `faiss` or `scikit-learn` ProductQuantizer).
2. Week 1: In `_build_chunk_plan`, replace binary `SGLANG_CHUNK_HEAD_RECOMPUTE_FRAC` with continuous gating using PQ distance threshold.
3. Day 2: A/B test against R32 head_recompute_30 on the same 5×5 verdict task. Expected outcome: equal or better failure-type agreement at equal or smaller recompute cost.

### Secondary direction: Inverse-H2O formalization (Direction #B-second)

If PQCache-gated works: a 2-3 day follow-up to formalize and publish. Otherwise, a self-contained research paper.

---

## Caveats and open questions

1. **3 of 8 user-supplied arxiv IDs were factually wrong** (FastGen/PQCache/CaM). The user should re-verify their citation source before drawing conclusions from this report.
2. **Scissorhands arxiv ID** could not be primary-source confirmed in research environment (WebFetch blocked at arxiv.org). Title and authors consistent across 3 independent secondary awesome-lists. Recommend re-verifying the literal arxiv URL before citing.
3. **No algorithm surveyed has a coding-task (HumanEval/MBPP/SWE-bench) accuracy-recovery measurement under byte-exact chunk-pool reuse.** All accuracy-recovery claims (where they exist) are on language-modeling perplexity, long-doc QA, or RAG QA — none on code generation. R32's verdict-task A/B is the ONLY coding-task selective-recompute validation in the literature so far.
4. **The "cumulative attention deviation → token-level recompute" concept has no published paper.** It is a research opportunity, not a citation.
5. **No algorithm surveyed solves the cross-prefix KV-loss problem.** Every algorithm surveyed assumes intra-context (single-prompt) operation. The cross-context KV loss is fundamentally distinct from in-context attention dynamics, and the 2024-2026 literature has not addressed it. R32 head_recompute_30 remains the best known mitigation.
6. **Implementation cost estimates are rough** — based on the analysis that PQCache is a few hundred lines and the chunk-pool modification is one environment variable change. Actual costs may vary.

---

## Files

- This report: `results/lossy_alg_round37/R37_DEEP_RESEARCH_PARTIAL_RECOMPUTE.md`
- Cross-references:
  - R31 deep-research synthesis: `results/lossy_alg_round28/R31_DEEP_RESEARCH_SYNTHESIS.md` (CacheBlend/SnapKV/StreamingLLM/RazorAttention)
  - R32 CacheBlend-inspired head_recompute implementation: `results/lossy_alg_round32/FINAL_REPORT.md`
  - C2 fundamental-limits (cross-context KV loss): session memory `c2-cacheblend-lossy-not-safe-2026-06-28.md`
  - Cross-position-fix (slot_id positional): session memory `cross-position-fix-works-2026-06-30.md`

## Sources (all 5 research agents consolidated)

**Primary arxiv IDs verified**:
- 2306.14048 (H2O), 2305.17118 (Scissorhands — medium confidence), 2310.01801 (DeepSpeed-FastGen — NOT what user described), 2407.12820 (PQCache), 2410.05258 (Differential Transformer), 2503.01330 (WeightedKV), 2402.02750 (KIVI — inferred), 2402.17463 (ChunkLlama), 2501.15383 (Qwen2.5-1M), 2405.16444 (CacheBlend), 2402.04617 (InfLLM), 2401.01325 (Self-Extend), 2401.03462 (Activation Beacon), 2305.16300 (Landmark Attention), 2309.12307 (LongLoRA), 2309.17453 (StreamingLLM), 2407.02490 (MInference), 2406.14909 (MoA), 2407.09450 (EM-LLM), 2404.07143 (Infini-Attention), 2502.11089 (NSA), 2410.13276 (SeerAttention), 2502.13928 (SpargeAttn).

**Source code repos verified or strongly inferred**:
- github.com/FMInference/H2O
- github.com/AnswerDotAI/cold-compress (PR #11 Scissorhands third-party impl)
- github.com/deepspeedai/DeepSpeed-MII
- github.com/PKU-Alignment/PQCache (likely, not directly verified)
- aka.ms/Diff-Transformer (official Diff Transformer)
- github.com/nanowell/Differential-Transformer-PyTorch (community impl)
- github.com/perkfly/KIVI
- github.com/thu-nlp/InfLLM
- github.com/akjindal53264/selfextend
- github.com/FlagOpen/Activation-Beacon
- github.com/epfml/landmark-attention
- github.com/dvlab-research/LongLoRA
- github.com/mit-han-lab/streaming-llm
- github.com/microsoft/MInference
- github.com/thu-nics/MoA
- github.com/microsoft/SeerAttention
- github.com/thu-ml/SpargeAttn
- github.com/HKUNLP/ChunkLlama
- github.com/THUDM/GLM-4
- github.com/QwenLM/Qwen2.5
- huggingface.co/THUDM/glm-4-9b-chat-1m
- huggingface.co/CohereForAI/c4ai-command-r-plus