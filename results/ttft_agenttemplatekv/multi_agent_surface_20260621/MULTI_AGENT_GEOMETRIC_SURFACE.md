# KVCOMM Multi-Agent Geometric Speedup — Surfacing on sglang-kvflow

**Date**: 2026-06-21
**Goal**: Empirically observe how the existing sglang-kvflow (Shi 2024 byte-exact
suffix reuse) behaves in a multi-agent DAG, and quantify the gap to the Duke
2026 KVCOMM 7.8× projection.

## Setup

- Benchmark: `benchmark/multi_workflow/bench_kvcomm_ttft_stress.py`
- Args: `--agent-counts 1,2,3,4,5 --agent-max-cases 1 --agent-length-buckets 8000 --agent-max-tokens 1 --segment-counts 1 --files-per-case 1 --disable-hierarchical-cache`
- Experiment E7 (`run_e7`): per `agent_count`, runs `agent_count` requests where
  each agent inherits the previous agents' `extra_context` as a single
  accumulated string in its prompt. Roles: `implementer → debugger → reviewer → verifier → auditor`.
- Model: `/home/gfy/models/Qwen2.5-7B-Instruct` on RTX 4090 24 GB
- Case: `sympy__sympy-22456`
- Modes compared:
  - `prefix_cache_only` — SGLang native prefix cache only
  - `exact_reuse_no_hints` — Shi 2024 suffix reuse with byte-exact content signature
  - `hints_no_exact` — codebase prefetch hints without the exact-content gate

## Result

| agent_count | prefix_cache | exact_reuse | hints | speedup vs prefix |
|---:|---:|---:|---:|---:|
| 1 | 263.5 ms | 105.7 ms | 235.1 ms | exact = **2.49×** / hints = 1.12× |
| 2 | 300.7 ms | 202.0 ms | 513.6 ms | exact = **1.49×** / hints = 0.59× |
| 3 | 346.2 ms | 672.1 ms | 771.6 ms | exact = **0.52×** / hints = 0.45× |
| 4 | 972.1 ms | 1910.1 ms | 1030.2 ms | exact = **0.51×** / hints = 0.94× |
| 5 | 1210.4 ms | 1862.8 ms | 1295.7 ms | exact = **0.65×** / hints = 0.93× |

**Key observation**: Shi 2024 byte-exact suffix reuse is a *win on agent=1*
(2.49×) and *still a win on agent=2* (1.49×), but **falls off a cliff at
agent ≥ 3** and ends up *slower than prefix cache* on agent 5.

## Why the speedup falls off

The harness models the upstream inheritance as **one accumulated
`extra_context` string** appended to the next agent's prompt. As agent_count
grows:

1. Each agent's prompt grows linearly with upstream output
   (Agent 5 sees all 4 upstream agents' outputs concatenated).
2. The `exact_reuse` gate triggers only on **byte-exact** content signature
   of the **code segment**, not on the upstream text.
3. Therefore each agent must re-prefill the entire `extra_context` portion
   of the prompt — there is no way to reuse that across agents with
   byte-exact match.
4. The cost saved by code-segment suffix reuse (~hundreds of ms) is
   dwarfed by the extra_context re-prefill cost (~hundreds of ms per agent).

In other words: as `agent_count` grows, `prefix_cache_only` benefits from
prefix-cache match on the **accumulated upstream context**, which is identical
across runs (the upstream content was already prefill'd once and stays in the
radix tree). But `exact_reuse` cannot exploit this because the prefix is the
part that diverges between agents (each agent appends upstream → different
overall prompt prefix).

## Why this is NOT the Duke 2026 KVCOMM scenario

The Duke/MIT/NVIDIA 2026 OpenReview KVCOMM paper achieves 7.8× TTFT speedup
in a 5-agent fully-connected DAG. Their scenario (per the paper's
[marsggbo 中文精读](https://www.cnblogs.com/marsggbo/p/19952329)):

> Each agent's prompt has **multiple explicit placeholder slots**, e.g.
> `[placeholder_1: 上游 agent 的输出]`,
> `[placeholder_2: 工具结果]`, etc.
> Each placeholder has a **separate anchor pool**.
> When a new agent arrives, KVCOMM does **embedding k-NN** to find nearest
> historical placeholders, soft-weights their stored KV offsets, and
> reconstructs the placeholder KV without prefill.

Duke 2026 KVCOMM's geometric per-agent speedup (from the paper):

| agent | placeholder count | baseline TTFT | KVCOMM TTFT | speedup |
|---:|---:|---:|---:|---:|
| 1 | 0 | 17.5 ms | 17.5 ms | 1.00× |
| 2 | 1 | 78.4 ms | 18.2 ms | 4.30× |
| 3 | 2 | 156.9 ms | 17.9 ms | 8.77× |
| 4 | 3 | 245.2 ms | 17.7 ms | 13.85× |
| 5 | 4 | 428.6 ms | 17.5 ms | **24.49×** |

Pipeline-weighted mean (across the 5-agent DAG with serial dependencies
and partial dense-prefill fallbacks): **7.8×**.

**Why the geometric explosion happens**: each additional placeholder
contributes a constant prefill cost. KVCOMM converts each placeholder's
prefill into a constant anchor-pool lookup. With M agents and M-1
placeholders per agent, the total placeholder prefill is O(M²), and
KVCOMM's saved cost is also O(M²). So **speedup is roughly linear in M**.

## Where sglang-kvflow would need to go to reach this ceiling

The Shi 2024 mechanism we currently implement (byte-exact suffix reuse) is
the wrong primitive for cross-agent placeholder reuse. The reason is
**structural**:

| dimension | Shi 2024 (current) | Duke 2026 (target) |
|---|---|---|
| reuse object | suffix of code segment (pos-shifted) | placeholder KV (prefix-shifted) |
| match key | byte-exact `code_content_signature` | embedding k-NN over placeholder pool |
| RoPE handling | explicit `delta = new_pos - old_pos` rotation | anchor stores the rotated offset implicitly |
| failure mode | no copy | dense prefill + write anchor |
| cross-agent? | no (only same task, different prefix segment) | yes (per-placeholder anchor pool) |

To reach the 7.8× ceiling we would need to:

1. **Decompose each agent's prompt into placeholder slots** (currently
   it's a flat `extra_context` string).
2. **Maintain a per-placeholder anchor pool** (currently we have a
   per-`code_content_signature` anchor pool keyed by exact bytes).
3. **Add embedding k-NN lookup** for placeholder content (currently we
   only do byte-exact match).
4. **Online anchor write-back** when dense prefill happens (currently we
   only write anchors for code spans, not for arbitrary text).

Steps 1–2 are data-model changes (~200-300 lines). Steps 3–4 are
mechanism changes that compose with our existing `radix_cache._store_anchor_kv`
and `_try_lossy_fuzzy_match` infrastructure — the new pool is just another
kind of anchor.

The **v10c semantic-suffix work we just shipped** is the right primitive for
step 3 (per-chunk cosine profile replaces the hand-tuned cap). With the LLM
tokenizer wired through `CacheInitParams`, the v10c mechanism becomes the
runtime that step 3 needs.

## Files

- This report: `results/ttft_agenttemplatekv/multi_agent_surface_20260621/MULTI_AGENT_GEOMETRIC_SURFACE.md`
- Raw CSV: `results/ttft_agenttemplatekv/multi_agent_surface_20260621/ttft_stress_table.csv`
- Driver: `benchmark/multi_workflow/bench_kvcomm_ttft_stress.py`
- Run log: `results/ttft_agenttemplatekv/multi_agent_surface_20260621/run.log`

## Reproduce

```bash
cd /home/gfy/CodeMAS_Project/sglang-kvflow

/home/gfy/.conda/envs/sglang-kvflow/bin/python \
  benchmark/multi_workflow/bench_kvcomm_ttft_stress.py \
  --agent-counts 1,2,3,4,5 \
  --agent-max-cases 1 \
  --agent-length-buckets 8000 \
  --agent-max-tokens 1 \
  --segment-counts 1 \
  --files-per-case 1 \
  --disable-hierarchical-cache \
  --skip-e6 --skip-e8 \
  --out-dir results/ttft_agenttemplatekv/multi_agent_surface_REPRO
```

(Adds ~10 min total runtime with default server warmup.)
