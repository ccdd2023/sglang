# Related Work: Code-Aware & Cross-Request Inference Acceleration (2023-2026)

_Compiled 2026-07-10 · for sglang-kvflow paper Related Work section_

This document gives algorithm-level detail for the systems most relevant to sglang-kvflow's R32 / placeholder chunk pool work. Each section: (1) precise algorithm mechanism (with pseudocode or condition where applicable), (2) quantitative result, (3) applicability to our work.

---

## Group 1 — Lossless cross-request KV reuse (byte-exact prefix)

### 1.1 SGLang / RadixAttention — Zheng et al., NeurIPS 2024
**Paper:** arXiv:2312.07104

**Algorithm (RadixAttention):**
- A **radix tree** indexes all prompt KV blocks (page-level granularity). Each tree node = a unique token prefix; leaf = the request that owns it.
- Insertion walks token-by-token from root, branching when a divergence appears. Children share a parent until their first differing token.
- Eviction: LRU over leaf counts (whole subtrees evicted together; no partial eviction).
- **Cache-aware scheduling:** before admitting a new request, scheduler checks max-prefix overlap with currently-runnable set; reorders run queue to maximize tree hits.

```
def try_reuse(prompt_tokens):
    node = radix.root
    hit_len = 0
    for tok in prompt_tokens:
        child = node.children.get(tok)
        if child is None: break
        node = child; hit_len += 1
    return node.kv_block if hit_len > 0 else None  # lossless prefix
```

**Result:** Up to **6.4× throughput** on multi-turn chat / agent / RAG / few-shot workloads. Hit rate 3-5× higher than vLLM v0.3 prefix caching on the same trace.

**Applicability to kvflow:** This is the **substrate we extend**. Our placeholder chunk pool builds on RadixAttention as the L1 (byte-exact prefix). The radix tree gives us agent-level KV sharing within a request family; chunk pool gives us non-prefix byte-shifted reuse.

### 1.2 vLLM / PagedAttention — Kwon et al., SOSP 2023
**Paper:** arXiv:2309.06180

**Algorithm (PagedAttention + APC):**
- KV blocks stored as **non-contiguous pages** with a **block table** mapping logical → physical block IDs (like OS virtual memory).
- **Automatic Prefix Caching (APC):** block hash = hash(parent_block_hash || tokens_in_block); same hash chain → block-level reuse.
- Match granularity: page (typically 16 tokens); not single-token.

```
block_hash(t) = SHA256(parent_block_hash || tokens[t*page : (t+1)*page])
# Match found when block_hash(t) == block_hash(t') for all t ≤ match_len
```

**Result:** 2-4× throughput vs FasterTransformer/Orca. Production case study: first-token 1.4s → 380ms (3.2×) on cache-hit workloads.

**Applicability:** vLLM is the parallel ecosystem. Our chunk pool can be wired as a vLLM custom block store; recent LMCache integration does exactly this. Different page size means hash collisions are common if page != our chunk size.

### 1.3 Mooncake — Qin et al., FAST 2025 Best Paper
**Paper:** arXiv:2407.00079

**Architecture (3 components):**
- **Conductor scheduler:** predicts cache-reuse opportunities across requests and balances prefill load across GPU nodes.
- **Transfer Engine:** GPUDirect RDMA between prefill node → decode node; KVCache transferred in chunks over NVLink/PCIe/RDMA.
- **Early-rejection policy:** drop requests whose context cannot be served by current cache cluster (predictive admission).

**Algorithm (Conductor scheduling heuristic):**
- For incoming batch B, compute reuse_budget = Σ max-prefix-length(req) over all req ∈ B
- Greedily pick the subset of waiting requests that maximizes reuse_budget subject to SLO and load-balance constraints
- KVCache at prefill node is moved to decode node *only after* the request is admitted (lazy migration)

**Result:** **525% throughput increase** in simulation; **75% more requests served under SLO** on Kimi production (128×H200); L-Eval +40%, ArXiv-Math +20% quality; TBT-SLO compliance ~100% (vs ~57% vLLM).

**Applicability:** Mooncake's PD-disaggregation is **the right architecture for our precomputed pool** — our pool is naturally "prefill-decoupled" (canonical prefix KV is precomputed offline; only delta is filled live). Mooncake's Transfer Engine is the right transport for shipping the precomputed pool to the live decode node.

### 1.4 MemServe / MemPool — Hu et al., ASPLOS 2025
**Paper:** arXiv:2406.17565 (revised)

**Architecture (Elastic Memory Pool):**
- 3 node types: **prefill nodes** (compute-heavy), **decode nodes** (memory-heavy), **context-cache nodes** (CPU/SSD-resident).
- Unified **token scheduler** assigns each request's tokens to whichever node holds the optimal KV. A request may have its prompt KV on a context-cache node, run prefill on a prefill node, and decode on a decode node — with KV shuttling in between.

**Algorithm (token-level scheduling):**
- For each request r at token position t:
  - best_node(r, t) = argmin(cost(node) + transfer_cost(node → next_node))
  - Where cost includes SLO penalty, compute availability, and KVCache hit probability
- Hysteresis to avoid thrashing.

**Result:** **3.64× throughput** increase; TTFT ↓ up to **50%** vs SOTA disaggregated systems. **-78.5% avg / -84.9% P99 TTFT** on ReAct agent workload.

**Applicability:** MemServe's **context-cache node tier** maps directly onto our precomputed pool — our precomputed canonical-prefix KV is naturally a context-cache tier resident.

### 1.5 LMCache — Liu et al., Oct 2025
**Paper:** arXiv:2510.09665

**Architecture:**
- **Vendor-neutral KV cache layer** between model and serving engine. Pluggable backends: vLLM, SGLang, NVIDIA Dynamo, CoreWeave+Cohere.
- **Tier hierarchy:** L0 GPU HBM → L1 host DRAM → L2 NVMe SSD → L3 remote (RDMA).
- **Optional CacheBlend integration:** enables lossy reuse at non-prefix positions.
- **PyTorch Foundation project** (2025).

**Algorithm (lookup + tiered placement):**
- Lookup: match query tokens against indexed chunks (byte-exact prefix + optionally byte-shifted)
- On hit at tier Ti: copy to L0, evict L0 victim if full
- Background: prefetch predicted-hot chunks to L0 based on access pattern

**Result:** **Up to 15× throughput** combined with vLLM; **3-10× standalone**. HyperPod case study: 100 sessions × 2K shared tokens × Llama-70B → P90 TTFT 1.21×. Documents that *context truncation cuts prefix-cache hit ratio in half* — directly relevant to our chunk-tail-recompute decisions.

**Applicability:** **This is the integration target for our placeholder chunk pool.** Wiring our pool as an LMCache backend gives us engine-agnostic deployment. Already on PyTorch Foundation → low integration friction.

---

## Group 2 — Lossy selective-token recompute (the R32 family)

### 2.1 CacheBlend — Yao et al., ICML 2025
**Paper:** arXiv:2405.16444

**Algorithm (the foundational selective-recompute scheme):**
- Given: RAG-style context with K chunks {C1, ..., CK} each with precomputed KV (computed at canonical prefix).
- Naive: copy all KV, get stale attention since prefix mismatches live prompt.
- CacheBlend: copy all chunks' KV, then **selectively recompute a small subset S ⊂ tokens of each chunk**.

**Selection heuristic (recompute set S):**
- For each token in chunk Ci, compute **attention surprise**: deviation between attention received at canonical prefix vs. estimated attention at live prefix.
- Top-p% surprise tokens (p ≈ 10-15%) form the recompute set S.
- Re-prefill only S ∪ live-prompt tokens; reuse the rest.

```
def select_recompute(chunk_kv, live_prompt, attention_oracle, p=0.15):
    scores = attention_oracle.score(chunk_kv, live_prompt)  # per-token "stale" score
    threshold = np.percentile(scores, 100 * (1 - p))
    return [t for t, s in enumerate(scores) if s >= threshold]
```

**Result:** **TTFT ↓ 2.2-3.3×, throughput ↑ 2.8-5×** vs full re-prefill. "No compromise in generation quality" on 3 LLMs × 4 RAG benchmarks.

**Applicability to R32:** **R32 is the 1-axis (uniform) generalization of CacheBlend.** CacheBlend picks a fixed FRAC per chunk; R32 picks FRAC uniformly across chunks. CacheBlend's "no quality loss" claim is for RAG (chunks share small prefix); for code completion with byte-shifted chunks, **our prior C2-fundamental-limits work showed byte-exact text ≠ KV-exact when surrounding prefix differs** — which is exactly why CacheBlend-style selective recompute is risky on code.

### 2.2 CortexCache — Mar 2025
**Paper:** arXiv:2503.03898

**Algorithm (context-aware KV compression across consecutive completions):**
- Target setting: **IDE autocomplete** (single GPU, 7B model). User types → completion → user types more → completion.
- After completion N, **compress the KV of completion N** into a smaller representation (learned projection or attention-replay summary).
- For completion N+1, **reuse the compressed KV** as warm-start; only recompute the prefix that changed.

**Algorithm sketch:**
```
# After completion N (prompt P_N + output O_N):
compressed_kv_N = compress(kv(P_N + O_N))

# For completion N+1 (new prompt P_{N+1}):
common_prefix = longest_common_prefix(P_N, P_{N+1})
warm_kv = compressed_kv_N[:common_prefix_len]
new_kv = prefill(P_{N+1}[common_prefix:])
final_kv = concat(warm_kv, new_kv)
```

**Result:** **1.5-2.5× speedup on code completion** (single GPU, 7B). Quality: comparable to full re-prefill on HumanEval + repo-level completion.

**Applicability to R32:** **Directly competitive.** CortexCache's 1.5-2.5× vs our R32 1.43× suggests we have ~0.5-1× headroom. The mechanism is similar (cross-completion reuse) but CortexCache operates at the completion level, not chunk-pool level. **A head-to-head benchmark on the same code-completion setup is the cheapest Tier-A win** (per deepresearch synthesis #1).

### 2.3 Position-Aware Recomputation — Du et al., Feb 2025
**Paper:** arXiv:2502.08201

**Algorithm (position-conditioned decoding-time recompute):**
- During decoding, **dynamically identify tokens whose cached KV is likely inaccurate** (stale due to long context).
- Recompute only those tokens' KV during the current step.

**Selection condition (decoding-time):**
- For each cached token at position p with cached KV_kp:
  - Estimate staleness = ||attention_pattern(p, current_step) - attention_pattern(p, when_cached)||₂
  - If staleness > τ, mark for recompute this step
- Recompute set is dynamic per step.

**Result:** Reports accuracy/latency trade-off on quantized long-context LLMs. No headline speedup number — paper is in spirit rather than headline.

**Applicability to R32:** **Position-Aware Recomputation is R32's 2-axis generalization** — R32 is fixed-FRAC; this is per-token-decoding-time-FRAC. Their measurement (HKVD-style positional deviation) **directly validates our position-aware hypothesis** (pos1 K_dev +7.2% > pos5 in our scale15 measurement).

### 2.4 KVLink — Yang et al., Feb 2025
**Paper:** arXiv:2502.16002

**Algorithm (per-document concatenative precompute):**
- For long-context QA, **precompute KV per source document** offline.
- At inference time, **concatenate** document KVs in order, then continue prefill from the live question.
- No selective recompute — purely concatenative.

**Algorithm sketch:**
```
# Offline precompute:
for doc in corpus:
    store[kv_hash(doc.text)] = prefill(doc.text).kv

# Inference:
chunks_kv = [store[kv_hash(d.text)] for d in retrieved_docs]
prefix_kv = concat(chunks_kv)
final_kv = prefill(question, init_kv=prefix_kv)
```

**Result:** **96% TTFT reduction**; 4% accuracy gain across 7 QA sets.

**Applicability:** **Concatenative, not selective** — therefore safe (no lossy approximation) but doesn't apply when document order doesn't match canonical prefix. Our placeholder chunk pool is byte-shifted and selective; KVLink is byte-exact-order and concatenative. **KVLink validates the precompute-then-splice pattern**; our chunk pool extends it to byte-shifted contexts.

---

## Group 3 — Cross-agent / cross-LLM KV transfer (the frontier)

### 3.1 DroidSpeak — Liu et al., Microsoft Research, Nov 2024
**Paper:** arXiv:2411.02820

**Algorithm (cross-LoRA selective layer recompute):**
- Two LLM instances: base (Llama-3-8B) and LoRA-adapted (fingpt-llama-3-8B).
- They share the **first N-K transformer layers** but diverge in the last K layers (LoRA adapter).
- **KV transfer with selective layer recompute:**
  1. Compute KV on base model for prefix P.
  2. Transfer prefix KV to LoRA-adapted model.
  3. **Identify "critical" layers** (last K layers + a small set of "important" early layers where LoRA-adaptation creates drift).
  4. Recompute only critical layers; reuse the rest.

**Algorithm sketch:**
```
def critical_layers(base_kv, lora_model, threshold=0.11):
    """Recompute ~11% of layers; ~89% reuse."""
    recompute = set()
    for layer_idx, layer_kv in enumerate(base_kv):
        # Run forward through lora_model's corresponding layer with this KV
        out = lora_model.layers[layer_idx].forward(layer_kv)
        # Compute deviation between out and base_model.layers[layer_idx].forward(layer_kv)
        dev = cosine_distance(out, baseline_out)
        if dev > threshold: recompute.add(layer_idx)
    return recompute  # ~11% of layers typically
```

**Result:** **1.7-3.1× prefill latency reduction** with negligible quality loss.

**Applicability to kvflow:** **This is the most directly relevant unclaimed territory for our 5-agent verdict pipeline.** DroidSpeak is the **only published system** that transfers actual KV tensors between distinct LLM instances. Our 5 agents share the same base model but diverge in role-specific prompts → the DroidSpeak technique applies if we can identify which layers are "role-agnostic" vs "role-specific." **Tier-B novel contribution.**

### 3.2 Tokencake — Beihang + PKU + Alibaba, Oct 2025
**Paper:** arXiv not directly retrieved; cited via news.qq.com secondary source

**Architecture (multi-agent KV serving framework):**
- Two scheduling dimensions:
  - **Spatial scheduler**: partitions GPU memory across agents, preventing KVCache thrashing.
  - **Temporal scheduler**: predictively preloads likely-needed KV based on agent turn structure (e.g., agent N+1 likely needs agent N's KV).

**Algorithm (predictive preloading):**
- For each agent's turn t, learn a probability distribution over (agent_id, token_position) → likely next access.
- Schedule preloads so that 95%+ of agent-t-1's KV is resident in L0 when agent-t starts.

**Result:** **KV-cache latency ↓ 47% vs vanilla vLLM** on multi-agent workload.

**Applicability to kvflow:** **Tokencake's spatial+temporal scheduler split** is the right abstraction for our 5-agent pipeline. Not yet adopted in sglang-kvflow. Tier-B novel extension (4-6 weeks).

---

## Group 4 — Speculative decoding for code (decode-time, not prefill)

### 4.1 EAGLE-3 — Li et al., NeurIPS 2025
**Paper:** arXiv:2503.01840

**Algorithm (multi-layer feature fusion + training-time test):**
- Train a small **draft model** that predicts the target LLM's hidden states at multiple layers (not just top layer).
- **Training-time test:** during draft training, predict not just next-token but multi-layer token-level features.
- Draft = one-step-advanced multi-layer feature; verify with target LLM using tree-attention.

**Algorithm sketch:**
```
# Draft (cheap):
draft_features = eagle3_draft(target_features_at_layer[l1, l2, ..., lk])
draft_tokens = draft_features.argmax(-1)  # top-k candidates

# Verify (expensive, one forward pass on tree):
target_output = target_llm.forward(prompt + tree(draft_tokens))
# Tree attention mask verifies W×D candidates in one pass
```

**Result:** Up to **6.5× speedup**; ~1.4× over EAGLE-2; +38% throughput in SGLang at batch 64.

**Applicability:** Decode-time only, not TTFT. **Not applicable to our prefill-axis work**; would compound with R32 if both enabled.

### 4.2 Medusa — Cai et al., Jan 2024
**Paper:** arXiv:2401.10774

**Algorithm (multiple decoding heads + tree verification):**
- Add **K parallel decoding heads** to LLM, each predicting token at position t+k.
- At decoding step, all K heads produce candidate continuations; tree-attention verifies all candidates in one forward pass.

```
# Medusa forward:
hidden = backbone(input_embeds)  # shared
candidates = [medusa_head_k(hidden) for k in range(K)]  # K candidates
# Tree attention verifies W^K candidates in one pass
```

**Result:** Medusa-1 **>2.2× lossless** (frozen backbone); Medusa-2 **2.3-3.6×** (joint FT). **3.29× on MT-Bench coding** category (Vicuna-7B).

**Applicability:** Same — decode-time only.

### 4.3 Lookahead Decoding — Fu et al., UC Berkeley, Feb 2024
**Paper:** arXiv:2402.02057

**Algorithm (Jacobi iteration, no draft model):**
- Treats decoding as solving a fixed-point equation x = f(x).
- **Jacobi iteration** generates multiple lookahead steps in parallel: x_{t+1}, x_{t+2}, ... = f(x_t) iteratively.
- Verifies via n-gram match against Jacobi-generated candidates.

**Result:** Up to **1.8× on MT-bench**; up to **4× on code completion** with strong multi-GPU scaling.

**Applicability:** **No auxiliary model needed** → minimal memory footprint. Attractive for high-concurrency agent serving. Decode-time only.

### 4.4 Hydra — Ankner et al., Feb 2024
**Paper:** arXiv:2402.05162

**Algorithm (sequentially-dependent draft heads + Sequoia tree attention):**
- Draft heads predict not just position t+1 but **t+1 conditioned on t** (sequential dependency).
- Verify with **Sequoia tree attention** (balanced tree, ~log W depth).

**Result:** **3.2× on code completion** (highest code-completion speedup of any draft-head paper).

**Applicability:** Decode-time; structurally interesting as **sequential dependency** captures same intuition as our HKVD positional signal.

### 4.5 LayerSkip — Elhoushi et al., Meta, ACL 2024
**Paper:** arXiv:2404.16710

**Algorithm (self-speculative decoding via early exit):**
- Same LLM is both drafter (early layers, e.g., layers 0-4) and verifier (all layers).
- Trained with **layer dropout** + **early-exit loss** so early layers produce useful draft.
- No auxiliary model.

**Result:** Up to **2.16× on CNN/DM summarization**, **1.82× on coding**, **2.0× on TOPv2** semantic parsing.

**Applicability:** **No separate draft-model memory slot needed** → attractive for high-concurrency agent serving. Decode-time.

---

## Group 5 — Code-aware prompt compression (input side)

### 5.1 LongCodeZip — Shi et al., ASE 2025
**Paper:** arXiv:2510.00446

**Algorithm (two-stage function-then-block compressor):**
1. **Function-level pass:** identify function definitions; rank by importance (call-graph centrality, recency).
2. **Block-level pass:** within kept functions, rank code blocks by importance (data-dependency, control-flow).
3. **Top-K selection:** keep top-K functions then top-K blocks within those.

**Algorithm sketch:**
```
importance_function(f) = α * pagerank(f) + β * recency(f) + γ * cross_refs(f)
importance_block(b, f) = importance_function(f) + δ * control_flow_complexity(b)
keep_top_k(functions, k_fn) → keep_top_k(blocks_within, k_blk)
```

**Result:** **5.6× compression with no task degradation.**

**Applicability:** **Function/block-level granularity maps directly onto our chunk-pool slot_ids.** LongCodeZip is **input-side** (compress prompt text); our pool is **KV-side** (reuse precomputed KV). Combining: LongCodeZip-style function ranking → guide which chunks to precompute.

### 5.2 CODEPROMPTZIP — He et al., Findings of ACL 2026
**Paper:** arXiv:2502.14925

**Algorithm (token-type-aware compression):**
- Classify each token into types: Identifier, Operator, Literal, Keyword, Punctuation.
- Type-specific importance scores (e.g., Identifiers near type signatures weighted higher).
- **Copy mechanism:** preserve type structure while compressing within-type.

**Result:** **+23.4% Assertion EM, +28.7% Bugs2Fix CodeBLEU** vs LLMLingua (generic compressor). Strongest published code-specific token-importance signal.

**Applicability:** **Type-aware ablation is the closest published analogue to our R40 type-aware FRAC** (which we retired). The published signal exists but is input-side; our work was KV-side.

### 5.3 SWE-Pruner — Wang et al., Jan 2026
**Paper:** arXiv:2601.16746

**Algorithm (goal-conditioned 0.6B skimmer):**
- Train a small (0.6B) model to predict **line-level relevance** to the user's goal.
- At inference, score every line; keep top-K lines.

**Result:** **23-54% token reduction on SWE-Bench Verified**; **up to 14.84× on LongCodeQA**.

**Applicability:** Lightweight enough to invoke per chunk. **Analog of a per-chunk KV-retention scorer.** Direct connection: SWE-Pruner-style skim could feed our pool's slot importance weights.

---

## Group 6 — Code-structure pretraining (the null-result context)

### 6.1 CodeBERT — Feng et al., Findings of EMNLP 2020
**Paper:** arXiv:2002.08155

**Negative finding (precise):**
- CodeBERT with **AST traversal training** does NOT improve generation tasks.
- C# code-to-NL generation: CodeBERT 22.36 BLEU vs code2seq 23.04 BLEU (code2seq wins via compositional AST paths).
- Code search: plain token-sequence CodeBERT = 0.7603 macro MRR (CodeSearchNet), beating sequence baselines.

**Why it matters:** **First published null result** on AST-as-input-signal for code transformers. Directly precedes our triple falsification.

### 6.2 GraphCodeBERT — Guo et al., ICLR 2021
**Paper:** arXiv:2009.08366

**Refined finding (precise):**
- Adding **data flow edges** (not full AST) to CodeBERT pretraining: code-search MRR 0.713 → 0.693 when data flow removed.
- The paper **deliberately rejects AST** in favor of sparse data-flow: "AST has unnecessarily deep hierarchy."

**Why it matters:** **Confirms AST is the wrong structural signal** even for pretraining, not just KV. Data flow (sparse, semantic) is preferred over AST (dense, syntactic).

---

## Group 7 — Production systems (what commercial code AI actually does)

### 7.1 Cursor Tab + Instant Apply — Anysphere, 2024-2026
**Sources:** cursor.com/blog/tab-update, tab-rl, instant-apply

**Algorithm (Instant Apply — deterministic speculative edits):**
- Custom `llama-3-70b-ft` trained on **full-file rewrites** (not diff-formatted edits — "models struggle with diff-formatted edits").
- At inference, model predicts a **complete edited file** directly; diff is computed post-hoc.
- **Deterministic speculative decoding:** the edit pattern is highly predictable from the surrounding context (cursor position, language, prior edits).
- Tree-attention over speculative edit candidates.

**Result:** **4-5× faster than next-fastest model**; 260ms p50 Fusion Tab; 400M req/day.

**Applicability:** **Cursor's content-hash + simhash (92% clone similarity) cache key collision signal** is underexploited relative to our path-keyed cache. Bolt-on Tier-B.

### 7.2 Anthropic prompt caching — 2024
**Source:** anthropic.com/news/prompt-caching

**Algorithm:**
- `cache_control: ephemeral` on message blocks (system → tools → messages).
- 5-min default TTL; 1-hour `extended` TTL; 4-breakpoint cap.
- Exact-prefix match within the same org/workspace.
- **Sub-agents get separate prompt caches**; "forks" share parent's cache.

**Result:** **-90% cost, -85% latency** on 100k-token book (11.5s → 2.4s TTFT).

**Applicability:** **R32 must be measured against this baseline.** No production lossy-KV system has been benchmarked against `cache_control` — unclaimed Tier-A win.

### 7.3 Claude Code sub-agent / orchestrator — Anthropic, 2025
**Source:** code.claude.com/docs/en/sub-agents, anthropic.com/engineering/built-multi-agent-research-system

**Algorithm (orchestrator + parallel subagent):**
- Main agent saves plan to **memory** (survives 200K truncation).
- Launches **parallel sub-agents** in **separate context windows** (separate prompt caches).
- Sub-agents return results; main synthesizes.
- `/compact` at ~92% context invalidates messages-tier prefix but keeps tools+system prefix cached.

**Applicability:** **The orchestrator-subagent pattern is structurally analogous to our 5-agent verdict pipeline**, but Anthropic's implementation shares only TEXT not KV. Our chunk-pool approach shares KV across agents with role-specific deltas — **the unclaimed combination.**

---

## Cross-cutting summary table

| System | Lossy? | Layer | TTFT speedup | Closest to kvflow? |
|---|---|---|---|---|
| SGLang RadixAttention | No | Substrate | 6.4× throughput | **Substrate** |
| vLLM APC | No | Substrate | 2-4× | Parallel substrate |
| Mooncake | No | PD-disagg | 525% throughput | Architecture pattern |
| MemServe | No | Tier | -78.5% TTFT ReAct | Agent analogy |
| LMCache | Both | Vendor layer | 3-10× | Integration target |
| **CacheBlend** | Yes (selective) | Chunk | 2.2-3.3× | **Closest lossless-quality** |
| **CortexCache** | Yes (compressed) | Cross-completion | 1.5-2.5× code | **Direct competitor** |
| **Position-Aware Recompute** | Yes (per-token) | Decoding-time | n/a (paper in spirit) | 2-axis generalization |
| **KVLink** | No (concatenative) | Per-doc | 96% TTFT ↓ | Safe precompute pattern |
| **DroidSpeak** | Yes (per-layer) | Cross-LLM | 1.7-3.1× prefill | **Cross-agent novel** |
| **Tokencake** | No | Multi-agent sched | 47% latency ↓ | Multi-agent scheduler |
| EAGLE-3 | N/A | Decode | 6.5× | Decode-only |
| Medusa | N/A | Decode | 3.29× coding | Decode-only |
| Lookahead | N/A | Decode | 4× code | Decode-only |
| Hydra | N/A | Decode | 3.2× code | Decode-only |
| LayerSkip | N/A | Decode (self-spec) | 1.82× coding | Decode-only |
| LongCodeZip | Yes (text) | Input | 5.6× compression | Input-side analog |
| CODEPROMPTZIP | Yes (text) | Input | +23.4% over baseline | Type-aware analog |
| SWE-Pruner | Yes (text) | Input | 14.84× LongCodeQA | Per-chunk scorer analog |
| CodeBERT (AST null) | n/a | Pretraining | Negative | First AST null |
| GraphCodeBERT | n/a | Pretraining | AST > data flow | Confirms AST wrong signal |
| **Cursor Instant Apply** | N/A (speculative) | Edit prediction | 4-5× | Bolt-on target |
| **Anthropic cache_control** | No | Provider | 85% latency ↓ | Baseline comparator |

---

## Position of sglang-kvflow R32 in this landscape

1. **Substrate level:** SGLang RadixAttention is in place (L1 byte-exact prefix).
2. **Lossy selective recompute level:** R32 = 1-axis uniform version of CacheBlend; achieves 1.43× where CortexCache reports 1.5-2.5× on similar code-completion settings — **0.5-1× headroom via per-corpus FRAC tuning**.
3. **Multi-agent level:** **No published system shares KV across agents in a verdict pipeline** — our 5-agent verdict pool is the **unclaimed contribution**.
4. **Code-structure feature:** **Confirmed null** (Direction A/B/HKVD) — code structure is not a usable internal KV signal; closest related CodeBERT/GraphCodeBERT results concur.

## Recommended paper-related-work structure

```
2. Related Work
  2.1 Cross-Request KV Reuse (lossless prefix)
       [SGLang RadixAttention] [vLLM PagedAttention] [Mooncake] [MemServe] [LMCache]
  2.2 Lossy Selective-Token Recompute
       [CacheBlend] [CortexCache] [Position-Aware Recompute] [KVLink]
  2.3 Cross-Agent and Cross-LLM KV Transfer
       [DroidSpeak] [Tokencake]
  2.4 Speculative Decoding (decode-time, orthogonal)
       [EAGLE-3] [Medusa] [Lookahead] [Hydra] [LayerSkip]
  2.5 Code-Aware Prompt Compression (input-side, orthogonal)
       [LongCodeZip] [CODEPROMPTZIP] [SWE-Pruner]
  2.6 Code-Structure Signal Null Results (context for our falsification)
       [CodeBERT AST null] [GraphCodeBERT data-flow > AST]
  2.7 Production Coding AI Caching Strategies (positioning)
       [Cursor Tab+Instant Apply] [Anthropic cache_control] [Claude Code sub-agents]
```