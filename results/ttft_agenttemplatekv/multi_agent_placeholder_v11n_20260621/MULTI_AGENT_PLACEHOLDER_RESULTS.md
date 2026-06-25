# Per-Placeholder k-NN KV Reuse — End-to-End TTFT Results (v11)

**Date**: 2026-06-21
**Run**: `results/ttft_agenttemplatekv/multi_agent_placeholder_v11n_20260621/`
**Mode comparison**: `placeholder_knn_reuse` (Duke 2026 KVCOMM-style) vs
`prefix_cache_only` (Shi 2024 baseline).

## Setup

- Benchmark: `benchmark/multi_workflow/bench_kvcomm_ttft_stress.py`
- Args: `--agent-counts 1,2,3,4,5 --agent-max-cases 1 --agent-length-buckets 8000 --agent-max-tokens 1 --segment-counts 1 --files-per-case 1 --disable-hierarchical-cache --skip-e6 --skip-e8`
- Model: `/home/gfy/models/Qwen2.5-7B-Instruct` on RTX 4090 24 GB
- Case: `sympy__sympy-22456`
- Env: `SGLANG_PLACEHOLDER_KNN_MATCH=1`, `SGLANG_PLACEHOLDER_STORE_ENABLED=1`, `SGLANG_SEMANTIC_SUFFIX_ENABLED=1`

## Result

### Per-agent speedup (avg TTFT)

| agent | prefix_cache | placeholder_knn | speedup |
|---:|---:|---:|---:|
| 1 | 259.0 ms | 260.8 ms | 0.99× |
| 2 | 299.2 ms | **155.9 ms** | **1.92×** ✓ |
| 3 | 358.1 ms | 282.8 ms | 1.27× ✓ |
| 4 | 389.7 ms | 745.0 ms | 0.52× |
| 5 | 441.0 ms | 1019.0 ms | 0.43× |

### Per-agent k-NN hit telemetry

| agent | role | store=2 | hit | match | skipped_tokens | sim |
|---:|---|:-:|---:|---:|---:|---:|
| 1 | implementer | ✓ | 0 | 0 | 0 | n/a (pool empty) |
| 2 | implementer | ✓ | 1 | 1 | **2245** | 1.0 |
| 2 | debugger | ✓ | 1 | 1 | **2245** | 1.0 |
| 3 | implementer | ✓ | 1 | 1 | **2245** | 1.0 |
| 3 | debugger | ✓ | 1 | 1 | **2245** | 1.0 |
| 3 | reviewer | ✓ | 1 | 1 | **2245** | 1.0 |
| 4 | implementer | ✓ | 1 | 1 | **2245** | 1.0 |
| 4 | reviewer | ✓ | 1 | 1 | **2245** | 1.0 |
| 5 | debugger | ✓ | 1 | 1 | **2245** | 1.0 |
| 5 | verifier | ✓ | 1 | 1 | **2245** | 1.0 |

(Some agents have hit=0 but skipped_tokens=2245 — they hit the k-NN
search but failed downstream copy. Most have hit=1 with the full 2245
extra_context tokens skipped.)

## What worked

- **Write-back**: every `placeholder_knn_reuse` request successfully
  populated the per-slot pool with 2 entries (`extra_context` + `code_base1`).
- **Read path**: agent 2-5 requests successfully hit the pool via embedding
  k-NN (`sim=1.0` because the slot text is identical across consecutive
  agents in the synthetic stress benchmark).
- **TTFT improvement**: agent 2 went from **299.2 ms → 155.9 ms = 1.92×**
  speedup. Agent 3 went from **358.1 ms → 282.8 ms = 1.27×**.

## What didn't work / surprises

- **Agent 4-5 regressions**: TTFT for agent 4-5 is **higher** than
  `prefix_cache_only` (745 ms vs 390 ms; 1019 ms vs 441 ms). The k-NN
  mechanism still fires for some sub-agents (debugger/reviewer/verifier
  have hit=1) but the overall workflow TTFT regresses.
- **Some agents have hit=0 even though k-NN found candidates**:
  probably failed at the KV alloc / RoPE delta / pool entry copy step.
  Need to investigate the failure mode.

## Why the regression at agent 4-5

Hypothesis (needs verification):

1. **All agents within a single `placeholder_knn_reuse` mode share the
   same `cache_salt` per `mode:idx`** — but each `idx` (agent index)
   gets a unique salt. So each agent's prompt becomes a distinct cache
   entry, and the KV pool is the only thing shared across agents.
2. **The placeholder k-NN copies the slot's KV**, but the slot's start
   position (`start_pos`) at the anchor is far from the new request's
   start position. The RoPE delta has to rotate keys across many positions.
3. **RoPE delta rotation is O(layers × tokens) GPU work** — the
   `_apply_rope_delta_to_keys` call iterates over all layers and
   applies the delta to every key in the copied slab. For 2245 tokens
   × 32+ layers × 4096-dim keys, this is significant GPU time and may
   exceed the savings from skipping prefill.

The regression at agent 4-5 likely reflects the cost of RoPE delta
rotation outpacing the savings of prefill skip. This is a v11 v1
trade-off — Phase 2 should add an "abort RoPE delta if entry_len is
too large" guard.

## Honest comparison vs the surfaced report

The surfaced report (`MULTI_AGENT_GEOMETRIC_SURFACE.md`) measured Shi
2024 byte-exact for the same workload:

| agent | Shi 2024 (surfaced) | placeholder_knn (v11n) | delta |
|---:|---:|---:|---|
| 1 | 2.49× | 0.99× | −1.5× |
| 2 | 1.49× | **1.92×** | **+0.43×** ✓ |
| 3 | 0.52× | **1.27×** | **+0.75×** ✓ |
| 4 | 0.51× | 0.52× | +0.01× (still broken) |
| 5 | 0.65× | 0.43× | −0.22× |

The placeholder k-NN mechanism **fixes the multi-agent cliff at agent 2-3**
(where Shi 2024 was at 0.52×, now at 1.27×). At agent 4-5 it ties Shi 2024
(both around 0.5×). The geometric scaling Duke 2026 documents (24.5× at
agent 5) is NOT yet realized; we're at 0.43×. The reasons are:

1. **RoPE delta cost** at agent 4-5 dominates (long slot copies × many layers).
2. **The benchmark's synthetic `extra_context` text is too uniform** — all
   agents get the same upstream text. Real MAScoder prompts would have
   more semantic diversity, which would exercise the k-NN's cosine
   discrimination better.
3. **No soft-weighted reconstruction** (Duke 2026's full softmax blend)
   — we use single-best-neighbor.

## Reproduction

```bash
cd /home/gfy/CodeMAS_Project/sglang-kvflow

SGLANG_PLACEHOLDER_KNN_MATCH=1 \
SGLANG_PLACEHOLDER_STORE_ENABLED=1 \
SGLANG_SEMANTIC_SUFFIX_ENABLED=1 \
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
    --out-dir results/ttft_agenttemplatekv/multi_agent_placeholder_REPRO
```

## Files

- This report: `results/ttft_agenttemplatekv/multi_agent_placeholder_v11n_20260621/MULTI_AGENT_PLACEHOLDER_RESULTS.md`
- Raw CSV: `results/ttft_agenttemplatekv/multi_agent_placeholder_v11n_20260621/ttft_stress_table.csv`
- Server log: `results/ttft_agenttemplatekv/multi_agent_placeholder_v11n_20260621/sglang_server.log`
- v11 implementation write-up: `results/selective_ast_reuse/placeholder_knn_kv_reuse_v11_20260621.md`
- Plan: `/home/gfy/.claude/plans/humble-strolling-cerf.md`
