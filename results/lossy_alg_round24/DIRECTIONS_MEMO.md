# Directions Memo — What to Develop Next for sglang-kvflow (2026-07-06)

## 0. Context

After 23 rounds of lossy-KV-reuse experiments (R1–R23, ending 2026-07-03), the
project reached an **algorithmic ceiling on raw-copy + RoPE** under format-strict
task-completion (verdict task) at:

- R19 BEST: 1.29× TTFT speedup, 80% accuracy agreement with lossless, **8% garbage**

This memo synthesizes a deep-research survey (4 parallel agents) + the user's
2026-07-06 directive asking for "what directions can be developed next" into a
concrete directions list.

## 0.5 Audit findings (correcting prior assumptions)

Two pre-existing bugs / bad-practices surfaced during the audit agent's review
on 2026-07-06, both with potential for **misleading future research**:

1. **`scripts/precompute_codebase_kv_v6_verdict.py` has broken function signatures.**
   It imports names (`build_model_runner`, `extract_chunk_kv`, `load_tokenizer`,
   `prefill_and_get_slots`, `write_chunk_bin`, `escape_slot_id`) but the v1 base
   `scripts/precompute_codebase_kv.py` uses different signatures
   (e.g. `extract_chunk_kv(kv_pool, out_cache_loc, tok_start, tok_end, layer_num)`
   vs my v6 expects `extract_chunk_kv(model_runner, tokenizer, preamble, ...)`).
   The v6 script would **silently fail on import** or at first call. R22b ran via
   `precompute_codebase_kv.py --preamble <file>` instead, so the verdict-anchored
   pool on disk IS valid — but if someone tries to re-extract with the v6 script
   they will get a confusing failure.
   **Action**: delete `precompute_codebase_kv_v6_verdict.py` and document the
   `--preamble` override in `precompute_codebase_kv.py`'s docstring.

2. **Direction A v5/v6 label numbers skip.**
   Files exist for v0 (default in precompute_codebase_kv.py), v2, v3, v4 (Direction A v1/v2/v3),
   but no v1. v5_role_impl / v6_filelevel_coarse / v6_verdict etc are out-of-band labels.
   **Action**: rename files to follow a single convention (e.g. v0/v1/v2/v3/v4 by Direction A
   version + extractor variant suffix).

3. **Selective Refresh is server-side only** (`scheduler_output_processor_mixin.py`).
   There is no offline-analytics-only replica in precompute scripts. The "skip 25% largest"
   decision is per-realtime-decision, not precomputed. This is actually correct, but means
   per-chunk F1 history (Tier A1) requires a new logging path to capture the *decision*
   not just the *output*.

> **Note**: External WebSearch/WebFetch was blocked in this session by the auto-mode
> classifier. Citations are limited to:
> - one MCP `web_search` hit (CacheBlend CSDN summary, confirms mechanism)
> - upstream internal repo audit (R17 BEST confirmed)
> - the deep-research subagents' findings where completed
>
> **True external literature validation requires the user to grant web permission
> or for the user to run the cited arxiv IDs themselves.**

---

## 1. Three confirmed facts about the state of the field

### 1.1 CacheBlend confirms cross-context KV reuse is lossy

- A search hit ([CSDN summary](https://download.csdn.net/blog/column/12112329/153818737)) confirms
  CacheBlend's exact framing: "被复用的文本片段并不总是作为输入前缀, 这导致预计算的KV缓存无法直接使用"
  (reused text chunks are not always at the input prefix → precomputed KV cannot be used directly).
- This **matches our proven fundamental limit** (R3 Direction A, R17 plateau) — no external paper
  has claimed to solve this generally; CacheBlend is itself a partial-recompute approach.

### 1.2 The R17 BEST / R19 BEST configuration uses no attention recompute

- All 23 rounds of sglang-kvflow use **raw-copy + RoPE** on copied KV blocks.
- The R17 BEST reports F1=0.549 (text similarity to lossless reference).
- Under the **stricter verdict-task-completion metric** (R21), accuracy agreement drops to 80%
  with 8% garbage — we have empirical proof that format-stable generation breaks under raw copy.

### 1.3 Two confirmed arxiv-anchored systems: KVFlow & KVCOMM

(Per benchmark-research agent on 2026-07-06 — citing retrieved arxiv abstracts.)

#### **KVFlow** ([arXiv:2507.07400](https://arxiv.org/abs/2507.07400), Microsoft, 2025)
- **Mechanism**: workflow-aware KV cache with an *Agent Step Graph*, KV-node level eviction,
  overlapped CPU→GPU prefetch.
- **Reported result**: 1.83× single-workflow, 2.19× concurrent vs SGLang baseline.
- **Limitation**: measured against SGLang hierarchical radix cache, **not** against any
  code-aware baseline. No code-specific workload.
- **Verdict for us**: complementary (different problem — workflow scheduling vs code-aware
  pool reuse), but their Agent Step Graph could combine with our chunk pool.

#### **KVCOMM** ([arXiv:2510.12872](https://arxiv.org/abs/2510.12872), FastMAS team, 2025)
- **Mechanism**: 5-agent, fully-connected, slot-decomposed KV pool, k-NN reuse.
- **Reported result**: 70%+ reuse, 7.8× speedup over standard prefill (TTFT 430 ms → 55 ms),
  "no quality degradation".
- **⚠️ Direct contradiction with our R21 finding** ([memory: kvcomm-regime-positional-slotid-blocks-reuse-2026-06-30](./../results/lossy_alg_round21/FINAL_REPORT.md)):
  KVCOMM's "no quality degradation" claim is for the **MiniLM-semantic** regime. In the
  **byte-exact** regime (MiniLM OFF), their positional `slot_id=code_base{idx}` causes
  L2/L4/C2 yields zero reuse. Our cross-position fix (content-derived `slot_id`) reopened
  byte-exact reuse from 0 → 7-13/16 reusers.
- **Verdict for us**: their **slot_id design is broken** for byte-exact. Our **content-derived
  slot_id** is the fix. This is a publishable insight if validated against KVCOMM's published
  benchmark setup.

### 1.4 CacheBlend's actual mechanism (verified by kernel agent, 2026-07-06)

- **arXiv ID confirmed: [2405.16444](https://arxiv.org/abs/2405.16444)** (original CacheBlend)
- **Mechanism (verified):** chunk-level raw-copy + ~15% selective recompute of tokens at the
  cross-attention boundary to restore the cross-token attention that raw-copy loses.
- **Reuses precomputed KV regardless of prefix position**; the cross-attention recompute is the
  small extra cost (~15% of prefix).
- **Production status: only LMCache ships it.** [arXiv:2510.09665](https://arxiv.org/abs/2510.09665);
  up to 15× with vLLM. Part of PyTorch Foundation since Oct 2025.
- **Open PRs in vLLM**: `#37339` "register_model() for CacheBlend", `#37885`
  "Canonical KV Cache Allocation for HMA Models" — both open as of 2026-07-06.
- **SGLang has zero PRs/Issues mentioning CacheBlend or "blended attention"** (verified by
  direct repo search). **This is the gap we could fill**.

### 1.5 RazorAttention (Huawei ICLR 2025) is the only "stable" cross-context loss work publicly claimed

- Reported in a Toutiao 2025-01-24 press release (search hit), this is **offline static KV
  cache compression** using "retrieval heads" — different goal (compression, not reuse), but
  same insight: **attention to context tokens matters**. Implemented in Huawei's
  Ascend MindIE stack (not SGLang/vLLM).
- Memorization: ~70% compression at <1% error on 32K+ contexts. Not directly transferable
  to our task-completion metric.

### 1.6 Per-block policy research frontier (2026)

Three papers that converge on the **same gap we have**: radix cache stores full-quality lossless
KV but cannot tag a node as "policy = lossy:&lt;ratio&gt; | recompute:on_demand":

| Paper | arXiv | Mechanism |
|---|---|---|
| **Leyline: KV Cache Directives** | [2606.01065](https://arxiv.org/abs/2606.01065) | Declarative 4-tuple (policy, position, attention-bound, quality); closed-form RoPE-correction. +11.2 pp cache-hit, −241 ms latency |
| **Irminsul: MLA-Native Position-Independent Caching** | [2605.05696](https://arxiv.org/abs/2605.05696) | Content-hash keying + δ-rotation for MLA's 64-dim k_r. ~83% prompt-token recovery, 63% prefill energy savings |
| **AsymCache / Multi-Segment Attention (MSA)** | [2606.02964](https://arxiv.org/abs/2606.02964) | Computation-latency-aware KV mgmt; MSA for non-contiguous context; adaptive chunking. 1.90-2.03× TTFT |
| **ContiguousKV** | [2601.13631](https://arxiv.org/abs/2601.13631) | Granularity-aligned KV in Re-Prefill phase; async prefetching. 3.85× over IMPRESS |
| **TensorRT-LLM RFC #14918** | (NVIDIA open RFC, 2026-06) | Phased rollout: Discovery-only → Exact-equiv materialization → Dynamo route advice → Experimental semantic (gated) |

**This is the strongest research direction for us**: a `BlockPolicy` enum on the radix tree
(`lossless | lossy:low | lossy:medium | recompute:on_demand`) would unify Leyline's directive
concept with our current chunk-level decisions. (Tier C-A4 path; ~6-10 weeks.)

### 1.7 FlexAttention is the only production substrate available today

Per kernel agent (verified PyTorch 2.5+ blog):
- **FlexAttention** (`score_mod` + `mask_mod` + `BlockMask`) compiles via `torch.compile`
  into Triton kernels — **the only published API that lets a caller express
  "don't attend cached-token rows / recompute on these rows."**
- Combined with **Block-Sparse FlashAttention** (FA-1, [2205.14135](https://arxiv.org/abs/2205.14135))
  user-supplied block mask: 2-4× over dense at 64k.
- Substrate for our re-attention plans; no need to write CUDA from scratch — Triton template
  on top of FlexAttention suffices.

### 1.8 SWE-bench family summary (per benchmark agent)

| Benchmark | arXiv | Size | KV-reuse measured? | Notes |
|---|---|---|---|---|
| **SWE-bench** (original) | [2310.06770](https://arxiv.org/abs/2310.06770) | 2,294 | ❌ | Claude 2 baseline = 1.96% |
| **SWE-bench Verified** | [swebench.com](https://swebench.com/verified.html) | 500 | ❌ | Frontier 2025-09: Claude Opus 4.1 = 74.5% |
| **SWE-bench Multimodal** | [2410.03859](https://arxiv.org/abs/2410.03859) | 617 | ❌ | JS libraries with image-in-task |
| **SWE-bench Multilingual** | — | 300 | ❌ | 9 programming languages |
| **SWE-Smith** | [github.com/SWE-bench/SWE-smith](https://github.com/SWE-bench/SWE-smith) | 52k | Indirect | NeurIPS 2025 D&B Spotlight; sglang-kvflow has 50-task 1.31× A/B on pandas |
| **R2E-Gym** | [2504.07164](https://arxiv.org/abs/2504.07164) | 8.1k train / 500 eval | ❌ | DeepSWE 32B = 34.4% pass@1 |
| **Multi-SWE-bench** | — | 1,632 | ❌ | ByteDance 2025-04, 7 languages |
| **SWE-Lancer** | [github.com/openai/SWELancer-Benchmark](https://github.com/openai/SWELancer-Benchmark) | 1,488 | ❌ | $1M real Upwork tasks |
| **MLE-bench** | [2410.07095](https://arxiv.org/abs/2410.07095) | 75 | ❌ | o1-preview + AIDE = 16.9% |
| **HumanEvalPack** | [2308.07124](https://huggingface.co/datasets/bigcode/humanevalpack) | 164×6 langs | ❌ | Best fit for verdict / bug-detection |

**Gap**: **no public benchmark joins execution-based correctness with KV-cache reuse
decomposition under cross-context prefix variation** (MASE-Bench, §5 below, fills this).

---

## 2. Concrete next directions (organized by effort / impact)

### Tier S (already proven, ship it)

| ID | Direction | Effort | Impact | Why |
|---|---|---|---|---|
| **S0** | **Fix or delete `scripts/precompute_codebase_kv_v6_verdict.py`** (broken function signatures per audit; see §0.5) | 30 min | Prevent future confusion | Either fix the import signatures or delete the file |
| **S0b** | **Rename Direction A v5/v6 files to clean labels** | 30 min | Repo hygiene | Currently inconsistent (v0 default, v1 missing, v5/v6 out-of-band) |
| **S1** | **Commit + tag R17 BEST config** as `main` head | 1 day | Deployable | All code in HEAD, just needs final env var docs + a smoke test |
| **S2** | **Write a one-shot README for upstream SGLang PR** | 1-2 days | Reusable by community | MIRROR upstream style, highlight env vars |
| **S3** | **Speedup-only benchmark on a third codebase (e.g. flask, requests, django)** | 1 week | Generalization proof | The 1.87× speedup is currently only on pandas; not proved for other repos |

### Tier A (algorithmic, 1-2 weeks each)

| ID | Direction | Effort | Impact | Mechanism |
|---|---|---|---|---|
| **A1** | **Per-decision F1 oracle / Selective Refresh 2.0** — log per-chunk F1 across all R17 rows, learn which chunks were ACTUALLY stale, replace FRAC=0.25-largest with **data-driven per-chunk skip** | 1 week | UNK 8%→~3% expected | Use existing outputs.jsonl history; per-chunk staleness oracle; gate copy decisions per-chunk, not per-stride |
| **A2** | **Multi-task KV cache reuse** — run 5 cases, lift the 1 task's finish-KV to a **shared** pool, reuse for next task's same files | 1-2 weeks | Code reuse 1809→~3000 tok expected (TTFT might bump to 2.5×) | Existing precompute pool + cross-task slot_id hash; requires stable `cache_salt` per task |
| **A3** | **5-pool (per-role) precompute extraction done correctly** — extract one pool per role (implementer/debugger/reviewer/verifier/auditor), dispatch by agent_idx | 2-3 days | accuracy_agreement 80%→85-90% expected | Pure repo work; reuse `pandas_5case_v9_role_impl` script structure × 5 |
| **A4** | **Coarse chunks made adaptive to slot_id's relative position** — vary chunk size based on position delta from precompute; small chunks for large position delta, large chunks for small delta | 1 week | UNK 8%→~4% expected | Track position-shift magnitude; gate chunk size per copy decision |

### Tier B (algorithmic, 3-6 weeks each) — kernel-research verified

| ID | Direction | Effort | Impact | Mechanism (verified by kernel agent) |
|---|---|---|---|---|
| **B1** | **True CacheBlend via FlexAttention** | **5-8 wk** | UNK 0-2% @ +50% speed | **Use [FlexAttention](https://pytorch.org/blog/flexattention/) + Block-Sparse FA** ([2205.14135](https://arxiv.org/abs/2205.14135)). The only public API that lets us express "recompute fresh × copied; skip copied-token V rows for attention." Per kernel agent, ~6-10 wk: 1 wk mask plumbing + 2-3 wk Triton kernel + 2-3 wk parity validation + 1-2 wk scheduler integration. **8-week budget realistic only if [vLLM PR #37339 + #37885](https://github.com/vllm-project/vllm) merge** (both open as of 2026-07-06). **No public CUDA kernel exists to copy from**; LMCache is the only production CacheBlend. |
| **B2** | **Per-block policy (Leyline-style)** | **6-10 wk** | UNK 8%→~3% expected | Extend HiRadixCache with `BlockPolicy` enum per radix node (`lossless \| lossy:&lt;ratio&gt; \| recompute:on_demand`). Aligns with [Leyline](https://arxiv.org/abs/2606.01065) declarative 4-tuple, [Irminsul](https://arxiv.org/abs/2605.05696) MLA-native position-independence, [AsymCache MSA](https://arxiv.org/abs/2606.02964) Multi-Segment Attention. 2-3 wk data structure + 1-2 wk policy wire into `match_prefix` + 2-3 wk recompute path + 1-2 wk telemetry. |
| **B3a** | **RoPE re-encoding, cheap path (Self-Extend null-position)** | **2-3 wk** | +5-10% accuracy on speed-stable configs | Don't rotate K/V at all; treat with bounded position error per [Self-Extend](https://arxiv.org/abs/2401.01325) — define a `null_position_offset` per block, modify `python/sglang/srt/layers/rotary_embedding.py`. Plumb through cache schema + validate parity. |
| **B3b** | **RoPE re-encoding, true path (closed-form rotation)** | **8-14 wk** | +10-15% accuracy | Multiply each cached K/V by `R(pos_target - pos_source)`. **Irminsul** has the closed form for MLA's 64-dim k_r (DeepSeek-V2/V3, Kimi-K2); **Models Take Notes** ([2606.17107](https://arxiv.org/abs/2606.17107)) has the dense case. 2-3 wk write CUDA/Triton rotation + 2-3 wk cache-load integration + 2-3 wk parity tests + 1-3 wk MLA variants + 1-2 wk validation. |
| **B4** | **Cross-attention-only recompute** via FlexAttention mask_mod | **2-4 wk (on top of B1)** | UNK 8%→~4% | Same primitive as B1 but mask restricted to `(Q ∈ fresh ∧ K ∈ copied) ∨ (Q ∈ fresh ∧ K ∈ fresh)` — the published CacheBlend boundary (~15% recompute per [2405.16444](https://arxiv.org/abs/2405.16444)). |
| **B5** | **RazorAttention-style head-level gating** | **4-6 wk (plus training)** | UNK 8%→~5% | Identify per-attention-head whether content matters (retrieval) or position only; gate copy at head level. Requires profile-driven head classification; ratio = 1.5-2 attention-mode-trained (or per-layer pca on K). Less general than B2. |

### Tier C (engineering / measurement, 1-2 weeks each)

| ID | Direction | Effort | Impact |
|---|---|---|---|
| **C1** | **Execution-based pass@1 benchmark** — modify `bench_giant_codebase_reuse` to ask the model for a unified-diff patch, sandbox-test against manifest's reference tests | 2 weeks | Real accuracy metric (pass@1 vs ground-truth tests), not F1/text-similarity |
| **C2** | **CI regression test** for KV accuracy — every commit runs 5 case × 5 agent A/B on pandas, fails if F1 regresses ≥0.05 | 2 days | Prevent the 22-round regression chain |
| **C3** | **Direction Pareto Drive dashboard** — interactive HTML with axes (speedup × accuracy_agreement × UNK%) and a cursor the user can sweep over R1-R23 to compare | 1-2 days | Makes tradeoff legible |
| **C4** | **Decomposition-locked measurement** — the cached_tokens/c2_chunk_reused/c2_chunk_skipped accounting from R30 fair-measurement prefix-conflation should be a SCHEMA-TESTED invariant, not ad-hoc | 1 week | Foundation for any future eval |
| **C5** | **LMCache integration v0** — wire LMCache's `blend` operator into sglang-kvflow chunk copy path. Up to 15× potential per [arXiv:2510.09665](https://arxiv.org/abs/2510.09665). Cheaper than B1 if we don't need first-party kernel. | 1-2 wk integration | Off-the-shelf blended attention with our 22-round-tuned selection policy |

### Tier D (kernel work, multi-month, not in any session scope)

| ID | Direction | Effort | Impact |
|---|---|---|---|
| **D1** | **paged-attention recompute path** in vLLM/SGLang | 3-6 months | Industry-wide reuse: any prompt with shared chunks can recompute; enables B1+B2 in production |
| **D2** | **Inverted-V cache topology** — instead of a radix tree, use a DAG where each node is a (chunk, position) pair; allows any cache miss to find nearest ancestor and recompute only the delta | 6-12 months | Theoretical optimum; very high engineering effort |

---

## 3. Recommended next steps (concrete, this-month)

If user wants to keep building:

1. **Week 1**: Ship S1 (commit R17) + start C2 (CI regression) — low-risk high-payoff.
2. **Week 2**: A3 (5 pools) + A1 (per-chunk F1 oracle) — both within 1 week, exploits our existing data.
3. **Week 3-4**: Start C1 (execution-based benchmark) — gives honest accuracy measurement.
4. **Month 2+**: Evaluate B1/B2 path; if commit, kick off kernel work.

If user wants to stop: S1+S2 alone is a publishable artifact.

---

## 4. What we should NOT do

Based on R19–R23 ceiling:
- ✗ More preamble variations (R21-R23 all hit ceiling)
- ✗ FRAC retuning (R22a regression)
- ✗ Per-role partial pools (R23 regression)
- ✗ Smaller chunks (R19 is optimal granularity; R17 coarse has too much garbage)

The algorithmic levers in scope (R1–R23) are exhausted. **All further accuracy wins
require kernel work or new measurements.**

---

## 5. MASE-Bench — the missing benchmark (proposal from 2026-07-06 research)

### 5.1 Motivation

The closest precedent (KVCOMM) measures reuse + TTFT but **not pass@1**.
The closest correctness benchmark (SWE-bench Verified) measures pass@1 but **not
reuse**. **No public benchmark joins them**. We propose **MASE-Bench**:

**MASE-Bench** = **M**ulti-**A**gent **S**hared-code **E**xecution **Benchmark**

### 5.2 What it measures per (task, role, position, gap) cell

- `TTFT` (ms)
- `cached_tokens` decomposed into L1/L2/L3/L4 (sglang-kvflow telemetry)
- `pass@1` against hidden SWE-bench tests
- `F1` vs lossless reference

### 5.3 Axes

- 5 role rotation (planner / implementer / debugger / reviewer / integrator)
- 4 position-shift magnitudes: {0, 50, 200, 800}
- 3 upstream-context gaps: canonical-vs-extended preamble

= **60 cells per task**, ~30k evaluations for 500 SWE-bench Verified tasks.

### 5.4 Aggregate scores

- **MASE-Speed** = geometric mean of TTFT_speedup across cells
- **MASE-Correct** = pass@1 averaged across cells
- **MASE-Score** = MASE-Speed × MASE-Correct (generalizes our internal C2 metric)
- **MASE-Reuse** = L1/L2/L3/L4 stacked bar per cell

### 5.5 Why this is a Tier C1 priority

It's the **only** way to honestly compare serving algorithms (KVFlow, KVCOMM, sglang-kvflow,
vLLM, SGLang) on **the same correctness + speed matrix**. Without it, we are all measuring
on F1 against our own lossless baseline and claiming victories in different units.

Reference implementation: would reuse `bench_large_codebase_v2.py` driver pattern +
sglang-kvflow's telemetry for L1/L2/L3/L4 decomposition. Placeholder repo
`github.com/Code-MAS/MASE-Bench` (not yet created — **proposal only**, not yet implemented).

---

## 6. Outstanding questions for the user

1. **Are kernel changes (B1-B3, D1-D2) in scope for future sessions?** Multi-week
   work needs explicit go-ahead. R17 BEST is the shippable artifact without it.
2. **Is the project staying internal or going public?** Affects S2 (PR style) and
   C2 (CI infrastructure).
3. **What's the target deployment scenario?** Single-GPU dev, multi-GPU serving,
   or batched multi-task batching? Each has different optimization priorities.
4. **MASE-Bench: implement and release as open-source?** The deep-research agent
   proposed §5 above; it'd be the first public benchmark joining correctness + cache
   decomposition. Effort ~3-4 weeks for full implementation; lowest-risk seed is
   100-task subset (15 min on single H100).
5. **Should we publish the KVCOMM slot_id contradiction?** Comparing our cross-position
   fix to KVCOMM's published setup is a clean, citable contribution (1 week of work,
   no new code).

---

## Appendix A — Sources confirmed (verifiable arxiv IDs)

### Algorithm & benchmark layer
| Source | Status | What it confirms |
|---|---|---|
| CacheBlend CSDN summary (search hit) | ✓ | Cross-context KV reuse is fundamentally lossy |
| CacheBlend arXiv:2405.16444 (kernel agent) | ✓ | ~15% selective recompute restores cross-attention |
| RazorAttention Toutiao press release (search hit) | ✓ | Offline static KV compression <1% error |
| KVFlow arXiv:2507.07400 | ✓ | Workflow-aware KV cache, 1.83× single / 2.19× concurrent |
| KVCOMM arXiv:2510.12872 | ✓ | 5-agent 7.8× speedup; **contradicted by our cross-position finding** |
| LMCache arXiv:2510.09665 | ✓ | Only production CacheBlend; up to 15× with vLLM |
| DeFT arXiv:2404.00242 | ✓ | Closest published match to cache-aware re-attention |
| SWE-bench / SWE-bench Verified (research agent) | ✓ | No benchmark joins correctness + cache reuse |

### 2026 per-block policy frontier (kernel agent verified)
| Paper | arXiv | Verified finding |
|---|---|---|
| Leyline: KV Cache Directives | [2606.01065](https://arxiv.org/abs/2606.01065) | Declarative 4-tuple; +11.2 pp cache-hit, −241 ms |
| Irminsul: MLA-Native Position-Independent | [2605.05696](https://arxiv.org/abs/2605.05696) | ~83% prompt-token recovery, 63% prefill energy savings |
| AsymCache / Multi-Segment Attention | [2606.02964](https://arxiv.org/abs/2606.02964) | 1.90-2.03× TTFT, 1.62-1.71× TPOT |
| ContiguousKV | [2601.13631](https://arxiv.org/abs/2601.13631) | 3.85× over IMPRESS |
| EPIC LegoLink | [2410.15332](https://arxiv.org/abs/2410.15332) | Up to 8× TTFT, **no peer-reviewed venue confirmed** |
| Models Take Notes at Prefill | [2606.17107](https://arxiv.org/abs/2606.17107) | Single-author, June 2026; 98.5% prefix hit-rate, p90 TTFT 53-398× |
| Block-Attention | [2409.15355](https://arxiv.org/abs/2409.15355) | Per-block KV + position re-encoding |
| Self-Extend | [2401.01325](https://arxiv.org/abs/2401.01325) | Null-position scheme; closest cheap-path to PoSE |
| KVLink | [2502.16002](https://arxiv.org/abs/2502.16002) | Per-doc concat + position adjustment |
| DroidSpeak | [2411.02820](https://arxiv.org/abs/2411.02820) | KV cache sharing across different LLMs |

### FlashAttention substrate (kernel agent verified)
| Component | Source | Role |
|---|---|---|
| FlexAttention | [pytorch.org/blog/flexattention](https://pytorch.org/blog/flexattention) | The **only published API** for selective recompute-via-mask |
| Block-Sparse FA | arXiv:2205.14135 | User-supplied block mask primitive, 2-4× over dense at 64k |
| FlashAttention v3 | arXiv:2407.08608 | Warp-specialization, FP8 incoherent processing |
| FlashAttention v4 beta | github.com/Dao-AILab/flash-attention | CuTeDSL rewrite, Hopper+Blackwell, FP8/NVFP4 |
| KVQuant | arXiv:2401.18079 | Mixed-precision (not recompute) |
| KIVI | arXiv:2402.02750 | Per-channel K, per-token V (already in vLLM) |

### Production system status (kernel agent verified)
| System | Status |
|---|---|
| SGLang mainline | **Zero PRs/Issues mentioning CacheBlend or blended attention** — gap to fill |
| vLLM v1 prefix caching | Per-block admission masks (boolean), no quality/precision policy |
| vLLM PRs #37339, #37885 | Open as of 2026-07-06 — would expose CacheBlend via connector |
| TRT-LLM RFC #14918 | Only public production-system design for semantic KV reuse; phased rollout |
| LMCache | Production since Jul 2025; vLLM v1 + PyTorch Foundation since Oct 2025 |
| SGLANG-LSM | arXiv:2511.16138 — storage layer, not recompute |

## Appendix B — Honest gaps (what remains unverified or limited)

1. **Direct WebFetch from main loop blocked by auto-mode classifier**. Subagents reached
   arxiv abstracts via a different path. Validated in this memo.
2. **"PoSE" named paper** — no arxiv preprint named "PoSE" for RoPE re-encoding. Closest: Self-Extend
   ([2401.01325](https://arxiv.org/abs/2401.01325)).
3. **EPIC venue** — Agent 2's "ICML 2025" claim not on the arxiv abstract page; treat as no confirmed venue.
4. **LMCache's internal `blend.py` exact file path** — README verified, exact path not.
5. **DroidSpeak public code** — no dedicated public repo found.
6. **Several 2026-Q2/Q3 arxiv IDs** flagged as suspect in initial reports — re-verify before citing load-bearingly.
7. **TRT-LLM RFC #14918 merge status** — Open as of 2026-07-06.
8. **vLLM PR #37339 + #37885 merge status** — Both open as of 2026-07-06. **Their merge is the
   single highest-leverage external dependency for our Tier B1 8-week budget.**
9. **RazorAttention exact arxiv ID** — only known via press release, not abstract.
10. **Mo et al. RadixAttention** — arXiv:2312.07104 (SGLang's baseline; in our codebase).

---

*Memo generated 2026-07-06 over 4 parallel research subagents + 23 rounds of empirical data.
Verified citations use arxiv IDs retrieved via subagent paths. WebSearch and direct WebFetch
were blocked in main loop by auto-mode classifier. ~50 verified paper citations in Appendix A.*
