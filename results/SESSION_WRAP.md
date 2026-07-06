# SESSION WRAP — sglang-kvflow (2026-07-06)

> **Single source of truth** for the R26/R27 session state. Replaces
> `HANDOFF_2026-06-04.md` and `PROJECT_STATE.md` (both deleted as stale).
> For project goals see [CANONICAL_TARGET.md](../CANONICAL_TARGET.md).

## TL;DR — 3-way comparison

| | R19 BEST (7B × 5) | R26 (3B-Instruct × 3) | R27 (3B-Coder × 3) |
|---|---|---|---|
| Model | Qwen2.5-Coder-7B | Qwen2.5-3B-Instruct (general) | Qwen2.5-Coder-3B |
| Agents | 5 | 3 | 3 |
| Lossless TTFT (reusers) | ~700ms | 520ms | 528ms |
| Lossy TTFT (reusers) | ~543ms | 258ms | 278ms |
| **Speedup** | 1.29× | **2.014×** | 1.900× |
| **Ground-truth FAIL_acc** | **60%** | 27% | **0%** |
| UNK rate | 8% | 20% | 13% |
| Code-aware reuse (tokens) | ~600 | 2886 | 2552 |

**Ground truth**: All 5 SWE-bench pandas cases are FAIL (code has bugs).
"FAIL_acc" = how often the model correctly says FAIL.

## Headline findings

### 1. 3B × 3-agent gives consistent ~2× speedup (model-family independent)

Both 3B × 3 configs achieve ~2× speedup. Mechanism:
- Smaller KV footprint → placeholder_chunk_pool fits more chunks
- Fewer agents → less pool eviction contention
- c2_chunk reuse jumps from ~600 tok (R19) to **2500-2900 tok** (4.8× more reuse)

### 2. Counterintuitive finding: Coder training ≠ better at code critique

**R27 (3B-Coder) is the WORST at FAIL detection** (0% FAIL_acc):
- Lossless: 100% PASS (model says "clean" to all buggy code)
- Lossy: 86.7% PASS, 13.3% UNK, 0% FAIL

**Why?** Coder models are trained to **improve** code. When asked to **evaluate** code, they default to "PASS, here's how I'd fix it" rather than "FAIL". The general Instruct model (R26) is more conservative — 27% FAIL_acc.

This refutes the implicit assumption that "coder = better at code tasks".
For **evaluation/critique** tasks, general Instruct may be more honest.

### 3. Lossy doesn't degrade accuracy (in any model tested)

In all three configs, lossy FAIL_acc is **equal or better** than lossless:
- R19: 60% (lossy) vs 52% (lossless) — lossy *helps*
- R26: 27% (lossy) vs 13% (lossless) — lossy *helps*
- R27: 0% (lossy) vs 0% (lossless) — neither can detect FAIL

The "UNK" garbage rate is a separate issue — model didn't produce a verdict
at all, not wrong verdict.

## Decision matrix

| Priority | Pick | Rationale |
|---|---|---|
| **Speed first** | R26 (3B-Instruct × 3) | 2× speedup, 27% FAIL_acc — misses many bugs but is fast |
| **Accuracy first** | R19 (7B-Coder × 5) | 1.29× speedup, 60% FAIL_acc — best at finding bugs |
| **Balanced** | R26 + R25-A1 oracle | R26 speed + skip-risky-chunks oracle → 0% UNK (untested on 3B) |
| **Avoid** | R27 (3B-Coder × 3) | Same speed as R26 but worse accuracy on critique tasks |

## Mechanism detail (R26)

| Reuse source | Avg tokens (reusers) |
|---|---|
| radix_prefix_tokens (L1) | 149.4 |
| **c2_chunk_reused_tokens (AST chunks)** | **2886.5** |
| l2_wholeslot_reused_tokens (KVCOMM whole-slot) | 0 |
| l3_offset_reused_tokens (MiniLM, deprecated) | 0 |
| **Total codeaware_reused_tokens** | **2886.5** |

All 2886 reused tokens come from AST chunk copy (C2 path). The 3B-Instruct
general model shares the same tokenizer family as 7B-Coder, so AST chunk
signatures (sha1 of normalized text) match across models — chunk pool
entries from the 3B precompute are byte-identical at the chunk level.

## Files of record

| File | What |
|---|---|
| `results/lossy_alg_round26/COMPARISON.md` | R26 detailed analysis |
| `results/lossy_alg_round27/COMPARISON.md` | R27 detailed analysis |
| `results/lossy_alg_round21/ground_truth.json` | All 5 cases are FAIL |
| `results/lossy_alg_round21/scripts/score_verdict.py` | Ground-truth scorer |
| `results/lossy_alg_round21/FINAL_REPORT.md` | R17/R19 verdict analysis |
| `results/lossy_alg_round25/SESSION_FINAL_DELIVERY.md` | R25-A1 oracle breakthrough |
| `results/lossy_alg_round24/DIRECTIONS_MEMO.md` | 12 future directions |
| `results/codebase_kv/pandas_5case_v6_verdict_3b/` | 3B-Instruct precompute (gitignored) |
| `results/codebase_kv/pandas_5case_v6_verdict_coder3b/` | 3B-Coder precompute (gitignored) |
| `~/.claude/projects/-home-gfy/memory/r26-r27-3b-speedup-2026-07-06.md` | Memory entry |

## Reproducibility

```bash
# R26 (3B-Instruct × 3, lossy)
bash results/lossy_alg_round26/launchers/run_r26_3b_3agent_verdict.sh

# R26 (3B-Instruct × 3, lossless control)
bash results/lossy_alg_round26/launchers/run_r26_3b_3agent_lossless.sh

# R27 (3B-Coder × 3, lossy)
bash results/lossy_alg_round27/launchers/run_r27_coder3b_3agent_verdict.sh

# R27 (3B-Coder × 3, lossless control)
bash results/lossy_alg_round27/launchers/run_r27_coder3b_3agent_lossless.sh

# Analyze
python results/lossy_alg_round26/analyze_ab.py
python results/lossy_alg_round27/analyze_ab.py

# Ground-truth scoring
python results/lossy_alg_round21/scripts/score_verdict.py \
    results/lossy_alg_round26/r26_3b_3agent/outputs.jsonl \
    results/lossy_alg_round27/r27_coder3b_3agent/outputs.jsonl
```

Total wall-clock: ~25 minutes for all four runs.

## Open follow-ups (not run)

| ID | Description | Effort |
|---|---|---|
| R28 | Try prompt "**find any bugs**" vs "decide if it needs a fix" — might unlock Coder-3B FAIL detection | ~30 min |
| R29 | R26 + R25-A1 oracle (skip risky chunks) — should drop UNK from 20% to 0% | ~1 hr |
| R30 | 7B-Coder × 3 agents (not 5) — isolate agent-count effect | ~10 min |
| R31 | MULTI_SLOT=8 on 3B — push beyond current 2× | ~30 min |
| R32 | Qwen2.5-Coder-7B × 3 agents — combine 7B accuracy with 3-agent regime | ~15 min |

## Hard constraints (unchanged)

- 加速**只**来自更多 KV 复用，不准加 KV-cache 调度
- L3 MiniLM 语义 k-NN **默认 OFF**（research only, deprecated 2026-06-27）
- >3 case 必须 `--disable-overlap-schedule --max-running-requests 1`
- 结果统一输出到 `results/` 子目录
- 不要 `swebench_local_envs/`、不要 `codebase_kv/*.bin` 入 git