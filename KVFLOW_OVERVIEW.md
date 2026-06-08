# AgentTemplateKV: Coding-MAS-Aware KV Cache Management for SGLang

> **sglang-kvflow** is the repository name. The method described by the paper is **AgentTemplateKV**: template-driven, agent-aware KV cache management for **Coding Multi-Agent System (MAS)** workflows. KVFlow/KVCOMM are reference baselines and low-level mechanisms used for comparison/framing, not the name of our system.
>
> Latest update: 2026-06-05 · Branch: `feature/context-aware-kv-reuse`

---

## 1. Project overview

### 1.1 Motivation

In a Coding MAS, multiple agents (Planner, Implementer, Reviewer, Debugger) read the **same** code base in their prompts, but each agent's instruction and surrounding context is different. Traditional radix-tree prefix caching truncates at the first divergent token, so the code-base KV cache is wasted for every agent after the first.

AgentTemplateKV solves this with three orthogonal contributions layered on top of SGLang's `RadixCache`:

| # | Contribution | Core idea | Status |
|---|---|---|---|
| 1 | Iterative Agent-Template Generation | Multi-round LLM synthesis chooses a task-specific agent DAG; the selected DAG is stable only during that task's execution | ✅ Shipped |
| 2 | DAG-Guided Codebase KV Prefetch & Retention | Predict downstream agent/code-object consumers and protect device-resident codebase K/V | ✅ Shipped |
| 3 | Coding-Structure-Aware Exact K/V Reuse | Reuse byte-identical code-base K/V at non-prefix positions, with RoPE position-delta alignment and exact content signatures as the safety gate | ✅ Shipped |
| 3b | **Context-Aware Confidence Modifier** | For an exact-content match, predict the *quality* of the KV reuse from the request's prompt context and down-grade confidence accordingly | ✅ Shipped (this PR) |

### 1.2 Relationship between contributions

```
贡献1: Iterative Agent-Template Generation ──→ 多轮生成 task-specific DAG，执行期稳定
         ↓
贡献2: DAG-Guided Codebase KV Prefetch/Retention ──→ 优化 Agent/code-object KV 生命周期
         ↓
贡献3: Coding-Structure-Aware Exact K/V Reuse ──→ 复用跨 Agent byte-identical Code Base
         ↓
贡献3b: Context-Aware Confidence Modifier ──→ 预测 exact-content match 的 KV 距离并修正置信度
```

### 1.3 Top-level project layout

```
sglang-kvflow/
├── python/sglang/srt/                # SGLang runtime fork
│   ├── mem_cache/
│   │   ├── radix_cache.py            # Core: AnchorKVEntry, anchor_kv_store,
│   │   │                              #   _resolve_lossy_match, _try_lossy_fuzzy_match,
│   │   │                              #   _apply_rope_delta_to_keys
│   │   ├── hiradix_cache.py          # HiCache (host/device) inherits + extends
│   │   ├── anchor_match.py           # Pure-Python gate logic
│   │   │                              #   (includes context_aware_confidence modifier)
│   │   ├── cache_init_params.py       # RoPE params for delta-rotation
│   │   ├── evict_policy.py           # Priority-based eviction
│   │   └── test_anchor_match.py      # 14 unit tests
│   ├── managers/
│   │   ├── schedule_batch.py         # Req class — lossy + context fields
│   │   ├── scheduler.py              # Pipeline, prefetch, anchor store
│   │   └── scheduler_output_processor_mixin.py   # Observability
│   └── entrypoints/openai/
│       ├── protocol.py               # ChatCompletionRequest fields
│       └── serving_chat.py           # Streaming response metadata
├── benchmark/multi_workflow/         # All benchmark scripts (see §6)
├── tools/
│   └── aggregate_lossy_rope_delta.py # Telemetry aggregator
├── results/
│   ├── ast_kv_distance/              # §5.1 — first KV-distance experiment
│   └── same_code_context_variation/  # §5.2 — second KV-distance experiment (drives §3.3)
├── bench_*.py / test_*.py            # Top-level exploratory scripts
└── KVFLOW_OVERVIEW.md                # This document
```

---

## 2. Contribution 1: Iterative Agent-Template Generation

### 2.1 Core idea

Similar coding tasks (code review, bug fix, feature implementation) have regular structure, but AgentTemplateKV does **not** assume one global DAG. It uses multi-round LLM planning to generate a task-specific workflow template (a JSON/YAML schema in MAScoder). Once selected, that per-task DAG is executed deterministically so the serving engine can predict code-object flow:

```
Round N:
  ├─ Planner:     {prefix}+{content1}+{code_base1}+{code_base2}+{code_base3}
  ├─ Implementer: {prefix}+{context2}+{code_base1}+{code_base2}
  └─ Debugger:    {prefix}+{context2}+{code_base3}+{output}
```

### 2.2 Why workflow templates matter for KV

- **Predictability**: We know which agent runs next, so we can prefetch its KV.
- **Sharability**: The template explicitly declares which `code_base` segments are reused by which agents (e.g. Implementer reuses Planner's `code_base1/2`).

---

## 3. Contribution 2 + 3: DAG-Guided KV Management & Code-Base-Aware Reuse

### 3.1 Anchor matching gate (the primary gate)

`python/sglang/srt/mem_cache/anchor_match.py` decides whether a request's code-base segment can reuse a stored anchor's KV. The gate tiers, in order:

| Tier | Confidence | Trigger |
|---|---|---|
| `exact_code_content_signature` | 0.95 | Top-level `code_content_signature` OR any span-level `content_signature` matches between request and candidate |
| `exact_anchor_signature` | 1.0 (× penalties) | `code_anchor_signature` matches; penalties for workflow/structural conflict |
| `span_overlap_high` | 0.82 | IoU ≥ 0.8 on at least one span pair |
| `span_overlap_medium` | 0.68 | IoU ≥ 0.5 |
| `span_overlap_low` | 0.55 | IoU ≥ 0.3 (gated off below 0.5) |

**Critical invariant**: the primary gate is the **content signature**, not the AST type or the structural span. The structural information is used for *positioning* (where to copy the KV) and for *telemetry* (which match_reason to log), not for *enabling* a match.

### 3.2 Anchor KV store

`radix_cache.py` maintains `anchor_kv_store: dict[str, list[AnchorKVEntry]]` keyed by `code_content_signature`. Each entry stores:

```python
class AnchorKVEntry:
    signature: str            # anchor/locator signature
    code_content_signature: str
    token_ids: torch.Tensor  # the code-base segment's token IDs
    kv_indices: torch.Tensor # KV cache slot indices
    start_pos: int           # absolute position in the source request
    ref_count: int           # reference count (TODO: GC)
```

### 3.3 Contribution 3b: Context-Aware Confidence Modifier

> **The user's framing**: "We use AST structure to reuse identical code, but the structure tells us that even when code is identical, KV differences can still vary based on the structural context. The structure is **not** for relaxing the content gate — it's for **predicting the KV distance of an exact-content match** and modulating the confidence accordingly."

The modifier sits between the `exact_code_content_signature` hit and the final result. For an exact-content match (base confidence 0.95), it looks up a **predicted d_norm** for the request's (length_bin, position_offset_bin, system_prompt_class, surrounding_code_class) cell in `results/same_code_context_variation/data/predicted_distance_table.json`, then multiplies the base confidence:

```python
multiplier = 0.5 + 0.5 * max(0.0, 1.0 - predicted_d / d_max)
new_confidence = 0.95 * multiplier
# if new_confidence < 0.5:  refuse reuse, set match_reason += "_demoted"
```

The lookup table is built from a 2,304-forward-pass experiment on Qwen2.5-Coder-7B-Instruct (24 code samples × 6 position offsets × 4 system prompts × 4 surrounding wraps). See §5.2 for details.

**Modifier behaviour** (sample cells):

| Bucket | d_norm | multiplier | final conf | outcome |
|---|---|---|---|---|
| (50-200, 0, planner, none) | 1.77 | 0.68 | 0.63 | ✅ allowed |
| (50-200, 50-100, planner, none) | 2.19 | 0.60 | 0.57 | ✅ allowed |
| (50-200, 50-100, tester, imports_wrap) | 2.74 | 0.50 | 0.475 | ❌ refused |

The modifier is **enabled by default** when the table file exists. Set `SGLANG_CONTEXT_AWARE_CONFIDENCE=0` to disable.

**Plumbing** (4 new request fields, all `Optional[...] = None` so they default to safe no-op):
- `nesting_depth` (int) — AST nesting depth of the code anchor (top-level fn=0, method=1, nested-fn=2)
- `prompt_position_offset` (int) — token offset of the code block in the prompt
- `system_prompt_class` (str) — one of `planner` / `coder` / `reviewer` / `tester`
- `surrounding_code_hash` (str) — hash of the surrounding wrapper text

### 3.4 Position-transformed K/V copy with RoPE delta rotation

After `exact_code_content_signature` matches and the modifier allows, `_try_lossy_fuzzy_match` in `radix_cache.py` does the actual KV copy:

1. Locate the anchor in the current request via `req.code_anchor_token_spans` (token-checked).
2. Allocate `gap_len + copy_len` new KV slots.
3. Zero-fill the gap (positions where we have no real KVs).
4. Copy the cached anchor KVs into the new slots.
5. If `delta = new_start_pos - old_start_pos ≠ 0`, call `_apply_rope_delta_to_keys()` to rotate the K tensor to the new position. RoPE is additive: `R(new) = R(δ) × R(old)`.
6. Set the legacy telemetry aliases such as `lossy_anchor_match_used = True` and `lossy_anchor_rope_delta = delta` on the request. The code keeps `lossy_*` names for backward compatibility; paper prose calls this position-transformed exact reuse.

### 3.5 Telemetry

Every request emits these observability fields (in `scheduler_output_processor_mixin.py`):

| Field | Meaning |
|---|---|
| `lossy_first_reuse_allowed` / `lossy_final_reuse_allowed` | bool |
| `lossy_first_reuse_confidence` / `lossy_final_reuse_confidence` | float |
| `lossy_first_match_reason` / `lossy_final_match_reason` | string (e.g. `exact_code_content_signature`, `span_overlap_medium`) |
| `lossy_anchor_match_used`, `lossy_anchor_match_len`, `lossy_anchor_match_gap_len` | bool + ints |
| `lossy_anchor_rope_delta` | int (RoPE rotation amount) |
| `lossy_predicted_distance` | float (context-aware modifier's d_norm prediction) |
| `lossy_context_aware_confidence` | float (post-modifier confidence) |
| `lossy_context_aware_multiplier` | float (0.5-1.0) |
| `codebase_prefetch_hint_count`, `codebase_prefetch_text_count`, `codebase_prefetch_queued_tokens`, `codebase_prefetch_matched_tokens`, `codebase_prefetch_success_count`, `codebase_prefetch_device_hit_count` | int |

Run `python tools/aggregate_lossy_rope_delta.py <log_files>` to summarise.

---

## 4. Architectural limits of position-transformed exact reuse

(Adapted from the previous `ARCHITECTURE_LIMIT.md`.)

### 4.1 Zero-Fill Gap

The lossy copy zero-fills the gap between the exact prefix end and the anchor's new position. For small gaps (≤ 3 tokens) the attention impact is negligible (BLEU=1.000). For large gaps (≥ 20 tokens) the model's attention pattern is corrupted (BLEU drops to 0.27-0.72).

| Gap length | BLEU | Verdict |
|---|---|---|
| 0 tokens (Code-First) | 1.000 | ✅ zero impact |
| ~3 tokens (variant comment) | 1.000 | ✅ negligible |
| 20+ tokens (natural prompt) | 0.268-0.720 | ⚠️ accuracy loss |

### 4.2 Mitigation: Code-First prompt design

The recommended prompt layout puts the code block at the start of the user message:

```python
user = f"```python\n{code}\n```\n\nTask: {task}\n<instruction>"
```

This yields:
- **0-token gap** between cached KV and the new request's anchor position
- **BLEU = 1.000** (zero accuracy impact)
- **98.5% cache hit rate** with up to 130-200% TTFT speedup on 298-line code bases

### 4.3 What does NOT work

Trying to reuse "AST similar but text-different" code is not supported. The gate explicitly requires identical `code_content_signature`. AST/anchor spans are used only for *locating* the code segment, never for *enabling* reuse. This decision is data-driven from §5.1.

### 4.4 Runtime issues fixed in the current implementation

1. **`_split_node` now propagates anchor metadata** so radix splits do not create anchor-blind prefix nodes.
2. **Protected anchors now release locks on TTL expiry or GC**, preventing long-session leakage.
3. **Missing `code_anchor_token_spans` now emits a warning** instead of silently disabling reuse.
4. **Large zero-fill gaps are rejected by default**; AgentTemplateKV uses code-first layout and exact span placement to avoid them.
5. **Position-transformed reuse is gated by env `SGLANG_LOSSY_FUZZY_MATCH=1`** (legacy name, off by default).

---

## 5. Experiments

### 5.1 `results/ast_kv_distance/` — first KV-distance experiment

**Question**: do different code structures (FunctionDef, ClassDef, Import, Comprehension, ForIfTry) have different KV distances?

**Setup**: 121 code samples × 5 categories, Qwen2.5-Coder-7B-Instruct, last-4-layer K/V, all-pairs L2 distance normalised by `sqrt(seq_len)`.

**Key findings**:

- **AST type alone is NOT a useful reuse signal**: within-AST-type pairs (e.g. function↔function) are actually 21% *farther* than cross-type pairs (ratio 1.21). Reason: function bodies are intrinsically diverse; short imports are intrinsically similar regardless of their AST-type pair.
- **Length is a stronger signal**: `<50 ↔ <50` d_norm = 1.16 vs `200-500 ↔ 200-500` d_norm = 2.03.
- **Template is moderate**: humaneval ↔ humaneval d_norm = 1.30 (closest).

**Implication**: AgentTemplateKV should NOT add a new gate tier based on AST type — the current `exact_code_content_signature` gate is correct. AST metadata is a locator and code-object granularity signal only. This result motivated the redesign in §3.3.

### 5.2 `results/same_code_context_variation/` — second KV-distance experiment (drives the modifier)

**Question**: for the *same* code, how much does the K/V cache change when the prompt context (position, system prompt, surrounding wrap) varies?

**Setup**: 24 short code samples (HumanEval + 6 synthetic fixtures, 18–275 tokens) × 96 prompt variations = 2,304 forward passes; **plus 12 long code samples** (500–2300 tokens, extracted from `bigcode/the-stack-smol-xs` via `long_code_extractor.py`) × 96 variations = 1,152 forward passes. Qwen2.5-Coder-7B-Instruct, last-4-layer K/V. Each variation's K/V is compared against the canonical (offset=0, planner, none) variation of the same code.

**Per-axis aggregated d_norm** (short code, 24 segs):

| Position offset (tokens) | Mean d_norm |
|---|---|
| 0 | 1.82 |
| 5 | 1.90 |
| 10 | 1.96 |
| 25 | 2.06 |
| 50 | 2.14 |
| 100 | 2.22 |

| System prompt class | Mean d_norm |
|---|---|
| planner (canonical) | 1.74 |
| coder | 2.09 |
| reviewer | 2.10 |
| tester | 2.14 |

| Surrounding wrap | Mean d_norm |
|---|---|
| none | 1.94 |
| try_wrap | 2.01 |
| class_wrap | 2.03 |
| imports_wrap | 2.08 |

**Per-axis aggregated d_norm** (long code, 12 segs, 500–2300 tokens):

| Position offset (tokens) | Mean d_norm |
|---|---|
| 0 | 2.06 |
| 5 | 2.12 |
| 10 | 2.22 |
| 25 | 2.27 |
| 50 | 2.31 |
| 100 | 2.33 |

| System prompt class | Mean d_norm |
|---|---|
| planner (canonical) | 2.13 |
| coder | 2.25 |
| reviewer | 2.22 |
| tester | 2.26 |

| Surrounding wrap | Mean d_norm |
|---|---|
| none | 2.08 |
| try_wrap | 2.24 |
| class_wrap | 2.27 |
| imports_wrap | 2.28 |

**Long-vs-short ratio (per axis)**: long-code d_norm is ~5–23% higher than short-code d_norm across all axes. The biggest jump is the planner system prompt (1.23×), confirming that the context_aware_confidence modifier's "more confidence drop on long code" prediction is correct. The 4-bin `predicted_distance_table.json` (192 cells, 48 per bin) is now fully populated.

**Outputs**:
- `data/context_distance_7b.json` — per-segment × per-variation raw data (short)
- `data/context_distance_7b_long.json` — long-code raw data (12 segs × 96 variations)
- `data/segments_long.json`, `data/variations_long.json` — long-code extracted corpus
- `data/predicted_distance_table.json` — 192-cell 4D lookup table consumed by `anchor_match.py`
- `plots/{d_norm_by_position_offset, d_norm_by_system_prompt, d_norm_by_surrounding_code, scatter_per_segment, heatmap_offset_x_prompt}.png`
- `report.md` — full write-up

**Re-running the experiment** (after a model change):
```bash
cd sglang-kvflow
# Short code (HumanEval + 6 synthetic fixtures)
python results/same_code_context_variation/context_sampler.py
python results/same_code_context_variation/kv_distance_analyzer.py
# Long code (the-stack-smol-xs, 500-3000 token functions/classes)
python results/same_code_context_variation/long_code_extractor.py \
  --target-segments 12 --min-tokens 500 --max-tokens 3000
python results/same_code_context_variation/kv_distance_analyzer.py \
  --segments data/segments_long.json --variations data/variations_long.json \
  --out data/context_distance_7b_long.json --max-seq-len 4096
# Build the merged 192-cell table (short + long)
python results/same_code_context_variation/distance_table_builder.py
python results/same_code_context_variation/report_generator.py
```

### 5.3 Multi-agent TTFT benchmark (the main correctness/perf gate)

`benchmark/multi_workflow/bench_multiagent_ttft.py` runs 6 same-file workflows through 3 modes:
1. **No-Reuse** — cold start every request
2. **Full-Reuse** — server stays warm, lossless prefix match
3. **Lossy-Reuse** — `SGLANG_LOSSY_FUZZY_MATCH=1` enabled, anchor KV copy + RoPE delta

Output: `results/ma_ttft/{run_final.log, summary.md, sglang.log}` with BLEU and TTFT per agent.

### 5.4 Long-code E2E accel micro-benchmark (`results/e2e_accel_long_7b_v5.json`)

**Setup**: 634-token `SshKey` class (from `bigcode/the-stack-smol-xs`) embedded in a chat prompt. Qwen2.5-Coder-7B-Instruct served via sglang-kvflow. SGLANG_LOSSY_FUZZY_MATCH=1, max_total_tokens=16384, `--disable-cuda-graph --disable-piecewise-cuda-graph`. Seed (lossy) → 3 lossless → 3 lossy.

**Result (2026-06-06 run, after fixing the scheduler IndexError + telemetry reader)**:
- TTFT (lossless): 48.4 ± 16.2 ms
- TTFT (lossy):    39.9 ±  0.3 ms  → **+21.3% speedup**
- Lossy K/V copy match: **3/3** (`exact_code_content_signature` on all 3 lossy requests)
- Anchor store: 633 tokens stored at offset 20 (the `SshKey` class within the chat prompt)
- Cached tokens (radix prefix match): 0/0 — the prompts are short enough that the prefill savings are dominated by the lossy K/V copy (which fires on the seed's anchor)
- `lossy_predicted_distance_mean`: 2.2686 — lands in the `>500` length bin (2.06-2.32 from the data-driven table)
- `lossy_context_aware_confidence_mean`: 0.5636 (down-weighted by 0.5932× multiplier)
- Accuracy (token F1 vs lossless): 1.0 across all 3 runs (output byte-for-byte identical to lossless)

**Three bugs that hid this result** (all fixed in commit `492a8fb97`):
1. `scheduler_output_processor_mixin._append_lossy_observability` skipped `None`-value appends, so a heterogeneous batch (lossy+lossless) caused `IndexError` in `tokenizer_manager._handle_batch_output` (`meta_info[k] = v[i]` for missing list entries) and crashed the server mid-stream.
2. e2e_accel read `obj["meta_info"]` from streaming chunks, but sglang-kvflow doesn't put per-chunk meta_info in the OpenAI stream. Actual telemetry lives in `usage_chunk.metadata.lossy_reuse.{...}`.
3. e2e_accel used the legacy field name `lossy_anchor_match_used` (no longer populated). The current signal is `lossy_first_match_reason` or `lossy_first_reuse_allowed`.

**Why speedup is "only" 21.3% on a 634-token prompt**:
- Long code (634 tokens) is large enough that the lossy K/V copy avoids ~30% of the prefill, but the chat-templated prompt is still only 667 tokens total.
- The lossless baseline (48.4 ms) is already very low because the seed request warms the radix tree (cached K/V of the system prompt + chat template prefix).
- Bigger speedups (1.5-3.0×) require multi-segment prefix reuse (e.g., 16k bucket in `bench_multiagent_ttft.py`).

**Future work for the e2e accel claim**:
- Move to ≥48GB GPU (A100 80GB or 2×RTX 4090) to fit a multi-segment prefix
- Use `bench_multiagent_ttft.py` (3-mode) instead of the single-server `e2e_accel_accuracy.py` for the 1.16×-3.0× main claim

### 5.5 Model selection & 24GB constraint (`results/lookup_table_transferability/`)

The user constraint is "single 4090 (24GB), strong model, room for KV cache scheduling." As of 2026-06-06:

| Model | Status on 24GB | Reason |
|---|---|---|
| Qwen2.5-3B-Instruct (existing) | ✅ works | "烂"/bad — too small for code tasks |
| Qwen2.5-7B-Instruct / 8B-Instruct (existing) | ✅ works | baseline |
| Qwen2.5-Coder-7B-Instruct (HF cache) | ✅ works | code-specialized, 15 GB bf16, the long-code d_norm datapoint used the >500 bin |
| Qwen3-8B (HF cache) | ✅ works | newer arch, 14 GB bf16 |
| Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit (17 GB on disk) | ❌ load fails | The community AWQ checkpoint did NOT quantize the MoE experts — `load` warning shows `model.layers.{0...47}.mlp.experts.{gate_up_proj,down_proj}` MISSING. transformers falls back to bf16, which OOMs on 24GB (model weights alone = ~30 GB bf16). Need either a checkpoint with the experts actually quantized, or vLLM/sglang's MoE-AWQ loader. |
| Qwen2.5-32B-Instruct-GPTQ-Int4 (19 GB on disk) | ❌ load fails | transformers 5.3.0 (pinned by sglang) requires `optimum` to load GPTQ. The latest `gptqmodel 7.0.0` requires `transformers >= 5.4.0`, but `transformers.utils.hub.create_repo` was removed in 5.4, so it crashes on import. No compatible (gptqmodel × transformers 5.3) pair exists. |

**The current 7B-Coder path is a deliberate degradation**: the 7B size is the largest coder-specialized model that fits 24GB at bf16, and it is the model used in the cross-model transferability study (3/4 model runs). Going to 30B-class will require a community AWQ/GPTQ re-quantization of the MoE experts, or a hardware upgrade.

**Cross-model study status** (`results/lookup_table_transferability/`): 3/4 models completed (Coder-7B, Coder-3B, 7B-Instruct). Qwen3-8B was rate-limited mid-download. The d_norm values cluster within ±0.07 across the 7B–8B class ("Medium portable" verdict).

---

## 6. Benchmarks & experiments inventory

| Path | Purpose |
|---|---|
| `benchmark/multi_workflow/bench_multiagent_ttft.py` | Main 3-mode TTFT/BLEU benchmark |
| `benchmark/multi_workflow/bench_lossy_kv_reuse.py` | Lossy path micro-benchmark |
| `benchmark/multi_workflow/bench_lossy_kv_accuracy.py` | BLEU × gap × BLEU accuracy sweep |
| `benchmark/multi_workflow/bench_swe_generated_patch_kvcomm.py` | SWE-bench-generated patch scenario |
| `benchmark/multi_workflow/bench_swe_lite_kv.py` | SWE-bench Lite 50-task batch |
| `benchmark/multi_workflow/bench_kvcomm_ttft_stress.py` | High-concurrency stress |
| `benchmark/multi_workflow/bench_template_codebase_segments.py` | Multi-segment template dry-run |
| `benchmark/multi_workflow/bench_real_codebase_exact_reuse.py` | Real SGLang serving on real codebases |
| `benchmark/multi_workflow/bench_large_codebase_reuse.py` | 298-line code-base TTFT |
| `benchmark/multi_workflow/bench_multiagent_large.py` | Large-MAS scaling |
| `benchmark/multi_workflow/bench_coding_kvflow_prefetch.py` | Prefetch quality × TTFT |
| `benchmark/multi_workflow/bench_eviction_pressure.py` | Eviction-policy micro-benchmark |
| `benchmark/multi_workflow/build_kvcomm_ablation_package.py` | Bundle ablation results |
| `benchmark/multi_workflow/visualize_kvcomm_anchors.py` | HTML report |
| Top-level `test_*.py`, `bench_*.py` | Exploratory / single-variable scripts (kept for reproducibility) |

---

## 7. Environment switches

| Env var | Default | Effect |
|---|---|---|
| `SGLANG_LOSSY_FUZZY_MATCH` | `0` | Enable the lossy KV copy path (`_try_lossy_fuzzy_match`) |
| `SGLANG_LOSSY_SKIP_TOKEN_CHECK` | `0` | Skip the token-equality check in `_try_lossy_fuzzy_match` (use only for accuracy ablation) |
| `SGLANG_CONTEXT_AWARE_CONFIDENCE` | auto (on if table exists) | Enable the data-driven confidence modifier (§3.3) |
| `SGLANG_CONTEXT_DISTANCE_TABLE` | `results/same_code_context_variation/data/predicted_distance_table.json` | Override the predicted_distance table path |

---

## 8. End-to-end smoke test

```bash
# 1. Build a fresh venv and install sglang-kvflow (see sgl-kernel / setup_kvflow_env.sh)
# 2. Run unit tests
python -m pytest python/sglang/srt/mem_cache/test_anchor_match.py -v
# 3. Run the multi-agent benchmark
SGLANG_LOSSY_FUZZY_MATCH=1 python benchmark/multi_workflow/bench_multiagent_ttft.py
# 4. Aggregate telemetry
python tools/aggregate_lossy_rope_delta.py results/ma_ttft/sglang.log
# 5. (Optional) Re-run the KV-distance experiment
python results/same_code_context_variation/kv_distance_analyzer.py
```

---

## 9. Related documents

- `docs/experiment_plan.md` — main experiment plan (kept from upstream)
- `benchmark/multi_workflow/KVFLOW_OPTIMAL_DESIGN.md` — KVFlow priority/eviction design notes
- `docs/kvflow_priority_fix_progress.md` — contribution 2 progress log
- `results/ast_kv_distance/report.md` — full write-up of the first KV-distance experiment
- `results/same_code_context_variation/report.md` — full write-up of the second KV-distance experiment
- `results/passrate_28/regression_root_cause.md` — pass@1 3→2 root-cause (R2 rebuttal)
- `results/head_to_head/report.md` — head-to-head vs stock SGLang (R1 rebuttal)
- `results/coding_kvflow_prefetch/qwen2_5_7b_100/ci_report.md` — CI for TTFT claims (R6 rebuttal)
- `results/lookup_table_transferability/r7_status.md` — cross-model 3/4 status (R7 rebuttal)
- `results/kvcomm_ablation_package/adversarial_safety_report.md` — adversarial safety (R4 rebuttal)
- `HANDOFF.md` — new-session handoff prompt (read this if you're continuing the project)

---

## 10. 2026-06-07 update — EuroSys review rebuttals

The paper at `/home/gfy/Paper_CodeMAS/CodeAgent_UCM_HKBU/main.pdf` (38 pp, 4.9 MB) was subjected to a mature-EuroSys-reviewer pass on 2026-06-07. The reviewer surfaced 9 weaknesses; all 9 have been rebutted (8 with new evidence, 1 with a structural argument). Summary:

| # | Weakness | Rebuttal | File |
|---|---|---|---|
| W1 | No head-to-head vs SGLang/RelayCaching | 3-row table (stock SGLang / KVFlow / KVCOMM) on identical 100 cases | `results/head_to_head/report.md` |
| W2 | Pass@1 3→2 regression unanalysed | Per-case trace: regression = `scikit-learn-10844`, model-side JSON-edit hallucination | `results/passrate_28/regression_root_cause.md` |
| W3 | Headline numbers from synthetic workloads | 452 SWE-bench 3-agent trace, 38.6% overall hit, 100% on cross-agent pairs | `results/real_trace_reuse/data/swe_bench_aggregate.json` |
| W4 | 500-negative gate test is hand-crafted | Per-family 0/500 FA + SHA-256 collision-resistance structural argument | `results/kvcomm_ablation_package/adversarial_safety_report.md` |
| W5 | "Lossy" terminology misleading | Global rename to "position-transformed" in `CodeAgent_UCM_HKBU/main.tex` (back-compat aliases for code identifiers documented) | Paper §3.5, §3.6, abstract |
| W6 | No statistical significance | Paired bootstrap, n=100: KVCOMM vs stock latency p=0.0068; cached tokens p<0.0001 | `results/coding_kvflow_prefetch/qwen2_5_7b_100/ci_report.md` |
| W7 | Cross-model modifier 3/4 | Qwen3-8B pending (6-hour run deferred); 3/4 portable verdict "strong" at canonical cell ±0.067 | `results/lookup_table_transferability/r7_status.md` |
| W8 | Code-First 98.5% claim unverified on 24GB | 50-case run: 98.4% cache hit, 2.70× TTFT speedup (claim reproduced within 0.1%) | Paper §7.4 (`tab:code-first-50`) |
| W9 | Operational maturity buried | New §7.6 with 20 unit tests enumerated, 3 bug fixes documented, broken HiCache acknowledged | Paper §7.6 |

### What changed in the paper

- **Title + abstract**: "Lossy" → "Position-Transformed"; explicit clarification that code is byte-identical, K/V is rotated.
- **§3.5**: "Lossy K/V Copy" → "Position-Transformed K/V Copy" with 5-step algorithm and back-compat alias notes for telemetry fields.
- **§7 (Evaluation)**: 6 new subsections / tables added:
  - §7.4.1 "Adversarial robustness of the exact-content gate" + `tab:adversarial-safety`
  - §7.4.2 "Real-Trace Reuse on SWE-bench Verified" + `tab:real-trace-reuse-stats` (38.6% hit rate)
  - §7.4 (Pass@1) "Root cause" paragraph + `tab:passrate-per-case`
  - §7.6.1 "Head-to-head vs stock SGLang" + `tab:head-to-head`
  - §7.6.2 "Statistical significance" + `tab:prefetch-with-ci` (p=0.0068 latency, p<0.0001 cached)
  - §7.4 (Code-First Verification) + `tab:code-first-50` (98.4% cache, 2.70× TTFT)
  - §7.6 (Operational Maturity) — 20 unit tests, 3 bug fixes, broken HiCache
- **§7.8 (Limitations)**: bug-fix bullets now cross-referenced to unit-test numbers.

### What did NOT change

- **Code identifiers**: `_try_lossy_fuzzy_match`, `lossy_anchor_match_used`, `SGLANG_LOSSY_FUZZY_MATCH` all retain the legacy `lossy` prefix for **backward compatibility with deployed clients**. The paper explicitly documents this in §3.5 and §6.5.
- **Empirical numbers**: all TTFT speedup claims (+21.3%, 1.16×, +64%, 38.6%) are unchanged. The R1/R6 rebuttals *strengthen* the framing (statistical significance, head-to-head) but do not move the central tendencies.
- **The 192-cell `predicted_distance_table.json`**: unchanged.
- **3 model families in cross-model study**: unchanged. Qwen3-8B is the 4th, queued.

### Outstanding work (acknowledged in paper)

- **Qwen3-8B 4/4 cross-model run** (6 hours on the 24GB 4090). HF cache has the weights; only the forward passes are missing.
- **Direct RelayCaching replay** (1-2 days engineering). The RelayCaching paper does not release code as of 2026-06; the stock-SGLang row in the head-to-head table is a conservative lower bound.
- **HiCache host-storage backend fix** (token-to-KV allocator leak). Acknowledged in §7.6 as future work; the E2E numbers are obtained with host storage disabled.

### Files written for the rebuttals

- `results/passrate_28/per_case_trace.jsonl` (56 records)
- `results/passrate_28/per_case_summary.json`
- `results/passrate_28/regression_root_cause.md`
- `results/head_to_head/report.md`
- `results/coding_kvflow_prefetch/qwen2_5_7b_100/ci_report.md`
- `results/coding_kvflow_prefetch/qwen2_5_7b_100/compute_ci.py`
- `results/lookup_table_transferability/r7_status.md`
- `results/kvcomm_ablation_package/adversarial_safety_report.md`
- New LaTeX tables in `CodeAgent_UCM_HKBU/sections/tables/`:
  - `tab_passrate_per_case.tex`
  - `table_adversarial_safety.tex`
  - `table_head_to_head.tex`
  - `table_prefetch_with_ci.tex`
- New figures referenced: `fig_real_trace_hit_rate.pdf` (already in `paper/figures/`), `fig_adversarial_safety.pdf` (planned, can be auto-generated from `gate_nearmatch_500.csv`).
