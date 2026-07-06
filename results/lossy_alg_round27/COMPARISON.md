# R27 COMPARISON: 3-way R19 vs R26 vs R27

**Date**: 2026-07-06
**Configs**:
- **R19 BEST**: 7B-Coder × 5 agents, R19 BEST env vars
- **R26**: Qwen2.5-3B-Instruct (general) × 3 agents, R19 BEST env vars
- **R27**: Qwen2.5-Coder-3B-Instruct (coder-trained) × 3 agents, R19 BEST env vars

## Headline (3-way)

| Metric | R19 BEST (7B × 5) | R26 (3B-General × 3) | **R27 (3B-Coder × 3)** |
|---|---|---|---|
| **Lossless TTFT (reusers)** | ~700ms | 520ms | 528ms |
| Lossy TTFT (reusers) | ~543ms | 258ms | 278ms |
| **Speedup** | 1.29× | 2.014× | **1.900×** |
| **Ground truth FAIL_acc** | **60%** | 27% | 0% ← worst |
| UNK rate | 8% | 20% | 13% |
| Code-aware reuse (tokens) | ~600 | 2886 | 2552 |

## 🔑 反直觉发现 (Counterintuitive Finding)

**The Coder model is HEAVILY biased toward PASS on this verdict task.**

| | Lossless PASS-rate | Lossy PASS-rate | FAIL outputs |
|---|---|---|---|
| R19 7B-Coder | 48% | 32% | 52-60% of cases |
| R26 3B-Instruct (general) | 86.7% | 53.3% | 13-27% of cases |
| **R27 3B-Coder** | **100%** | **86.7%** | **0% of cases** |

**Why?** Coder models are trained on completion/fix-the-bug tasks — when asked to evaluate code, they tend toward "PASS, here's how I'd improve it" rather than "FAIL". The general Instruct model (R26) is more conservative about declaring code "clean".

## Speedup is Consistent (~2×) Across 3B Models

The 2× speedup on 3B × 3-agents is **model-family independent**:
- R26 (3B-Instruct): 2.014×
- R27 (3B-Coder): 1.900×

The mechanism is identical: smaller KV footprint → larger placeholder_chunk_pool → more reuse (2552-2886 tokens avg vs R19's ~600).

## 准确度排序 (FAIL_acc, higher = better)

| Rank | Config | FAIL_acc | Δ vs lossless |
|---|---|---|---|
| 🥇 | R19 7B-Coder × 5 | **60%** | +8pp (lossy *helps*) |
| 🥈 | R26 3B-Instruct × 3 | 27% | +13pp (lossy *helps*) |
| 🥉 | R27 3B-Coder × 3 | **0%** | 0pp (can't get worse) |

**Lossy doesn't degrade accuracy in any model** — it's actually slightly *better* than lossless for both 7B and 3B-General. The 3B-Coder's 0% FAIL is a property of the model itself, not lossy compression.

## 决策建议

### 想要 best speedup
**R26 (3B-Instruct general) × 3 agents**:
- 2.014× speedup
- 27% FAIL_acc — model says FAIL when warranted, but misses many bugs

### 想要 best accuracy  
**R19 (7B-Coder) × 5 agents**:
- 1.29× speedup
- 60% FAIL_acc — best at finding bugs
- 48% PASS-rate on FAIL ground truth — calibrated

### 想要 balanced
**R26 with oracle (R25-A1)**: skip lossy copy when (agent≥3 ∧ c2_reuse≥600) ∨ c2_reuse≥1800. Should drop UNK from 20% → 0% (based on R25's CV on 7B data; 3B transferability untested).

### Avoid
**R27 (3B-Coder) × 3 agents**: same speedup as R26 but worse accuracy. The "coder" training hurts on this specific verdict task.

## Files

- R27 lossy: `results/lossy_alg_round27/r27_coder3b_3agent/`
- R27 lossless: `results/lossy_alg_round27/r27_coder3b_3agent_lossless/`
- Coder-3B precompute: `results/codebase_kv/pandas_5case_v6_verdict_coder3b/` (72 chunks)
- Launchers: `results/lossy_alg_round27/launchers/`
- A/B analyzer: `results/lossy_alg_round27/analyze_ab.py`

## What R27 Did NOT Test

- **Different prompt instruction** for the verdict task — maybe "**find any bugs**" instead of "decide if it needs a fix" would coax more FAILs out of Coder-3B
- **Qwen2.5-Coder-7B-Instruct × 3 agents** (not 5) — would isolate whether Coder-7B at 3 agents matches R19 accuracy
- **R26 + oracle retrain on 3B telemetry** — should give R26's speedup + 0% UNK guarantee

## What R27 Did Confirm

- 2× speedup on 3B × 3 is robust across model families
- Lossy doesn't hurt accuracy (in any tested model)
- Smaller model + same task → faster TTFT, not necessarily better accuracy
- "Coder-trained" ≠ "better at code critique" — task-specific calibration matters