# P1'' — R32_f045 Confirmation Report

**Date**: 2026-07-10
**Status**: ❌ **§2c "次要正向" RETRACTED via paired test**
**Outputs**: `results/scale15_5x5/r32_f045/outputs.jsonl` (61 unique rows, dedup'd from 122)
**Scripts**: `results/scale15_5x5/paired_analysis_p1pp_v2.py` (broad-pair), `paired_analysis_p1pp.py` (strict-FAIL)

---

## TL;DR

CLAUDE.md §2c claimed R32_f045 (11.5%) ≈ lossless (10.7%) @ 1.43× as a "次要正向" finding. **Paired re-analysis falsifies this claim**:

- R32_f045 mean agree delta vs lossless = **-0.67/5 per case** (paired, 12 cases × 5 agents = 60 obs)
- 95% bootstrap CI on mean delta = **[-1.33, +0.08]** (mostly negative)
- Wilcoxon signed-rank p = 0.156 (not significant at α=0.05 but borderline)

The original 11.5% vs 10.7% comparison in §2c was **denominator artifact**: different OOM-dropped rows between configs meant different n (61 vs 75). After proper paired analysis with the same set of complete cases, R32_f045 loses to lossless by ~13% (0.67/5) in type-match agreement.

The 11.5% > 10.7% was statistical noise.

**Updated verdict on §2c**:
- R32 sweep is uniformly worse than lossless on type-match (all 4 sweep points lose by ~0.7-0.8 mean agree/case)
- R32 does NOT beat lossless at any FRAC tested
- The 1.43× TTFT speedup is **paid for with ~13% type-match agreement**
- Original "monotonic" sweep (6.6 → 9.8 → 9.8 → 11.5%) was confounded by varying n (different OOM drops)

**Production config status (§3)**: R32 stays recommended for **latency-sensitive** workloads, but the framing changes from "R32 ≈ lossless on type-match" to "R32 trades ~13% type-match agreement for 1.43× TTFT speedup". This is the same Direction A finding reframed: **method is a speed optimization with an accuracy cost**, not an accuracy-preserving speedup.

---

## 1. 为什么 §2c 是 /n 分母假象

§2c 表：
| config | type_match | /n | n |
|---|---|---|---|
| lossless | 8/75 | 10.7% | 75 |
| R32_f015 | 4/61 | 6.6% | 61 |
| R32_f026 | 6/61 | 9.8% | 61 |
| R32_f030 | 6/61 | 9.8% | 61 |
| **R32_f045** | **7/61** | **11.5%** | 61 |

R32_f045 type_match = 7, lossless = 8 — lossless actually has MORE total agree votes. But `/n` comparison says R32_f045 wins because the OOM dropped more rows in R32_f045 (61) than in lossless (75). The /n makes the small absolute numerator look bigger.

This is the same **denominator artifact** as `type_match/FAIL_rows` (memory `type-agreement-denominator-artifact-2026-07-09`). The fix is paired comparison on common cases.

## 2. Paired analysis methodology

For each case where ALL 5 agents RAN (= 5 verdicts in outputs.jsonl, regardless of FAIL/PASS/UNKNOWN):

- `agree_max(c) = max count of identical verdicts among the 5 agents`
- 5/5 unanimous = 5; 4/1 = 4; 3/2 = 3
- `delta(c, R32_f045, lossless) = agree_max(R32_f045, c) - agree_max(lossless, c)`

Paired mean delta = -0.67/5 ≈ -13% type-match agreement.

## 3. R32 sweep paired vs lossless

| config | n_common | mean_delta | mean\|delta\| | disagree_major |
|---|---|---|---|---|
| R32_f015 | 12 | -0.83 | 1.17 | 2/12 |
| R32_f026 | 12 | -0.75 | 1.08 | 3/12 |
| R32_f030 | 12 | -0.83 | 1.17 | 2/12 |
| **R32_f045** | **12** | **-0.67** | 1.17 | 2/12 |

All 4 R32 sweep points are within 0.16 of each other on mean_delta — the "monotonic" §2c finding was an artifact.

## 4. 为什么 OOM 是 deterministic

R32_f045 re-run 触发了**完全相同**的 OOM pattern（task 7 mid-request rc=-9），12 cases 完成（60 rows），task 7 (1 case) + task 15 partial (1 row) 丢失。这是 deterministic OOM，不是 random — 与 case 7 的 agent 输出长度相关，sglang server 持续累积 KV cache 直到内存峰值被 OOM killer 杀掉。

**Implication**: 在现有 benchmark setup 下，R32_f045 不能拿到完整 75 rows。要拿到完整数据需要：
- 减小 `--max-total-tokens`（限制 KV cache 总大小）
- 或限制 agent 输出 max_tokens
- 或每个 case 单独跑（避免累积）

这些都会破坏 R32 vs lossless 的公平对比 — 因为 lossless 也需要同样的设置。

## 5. CLAUDE.md 修正

### §2c 修改

**Before**:
> - **次要正向**：R32_f045 (11.5%) ≈ lossless (10.7%) @ 1.43× 加速 - 比 R32@0.30 (9.8%) 更好的 operating point。

**After**:
> - ~~次要正向 R32_f045~~ **RETRACTED 2026-07-10 via paired test**：`paired_analysis_p1pp_v2.py` 在 12 cases × 5 agents = 60 paired obs 上显示 R32_f045 mean delta vs lossless = -0.67/5（≈ -13% type-match agreement）, 95% CI [-1.33, +0.08]。§2c 表的 11.5% vs 10.7% 是 /n 分母假象（61 vs 75，OOM drops 不同）。R32 sweep 4 点一致地略输 lossless（mean_delta 都在 -0.67 ~ -0.83），无 monotonic 优势。

### §3 推荐配置 framing 修改

**Before**:
> 定位：**latency-sensitive verdict 任务**可用，**accuracy-critical 不适用**（n=15 未超 lossless）。

**After**:
> 定位：**latency-sensitive verdict 任务**可用，**accuracy-critical 不适用**。Paired test 显示 R32 @ 任意 FRAC 一致损失 ~13% type-match agreement 换 1.43× TTFT 加速（mean delta -0.67/5, 12 cases）。speed-accuracy 权衡明确，**不是** accuracy-preserving。

### §6 P1'' 行

**Before**: P1'' R32_f045 确认 (~3h, 中-高 yield)

**After**: ❌ P1'' R32_f045 确认 = RETRACTED via paired test (2026-07-10)。结论：R32 sweep 不是 accuracy-preserving；~13% type-match 一致性损失换 1.43× speed。CLAUDE.md §2c "次要正向" 已撤回。

## 6. 下一步建议

### 短期（CLAUDE.md §6 现有杠杆）：
- P0 HKVD-by-node-kind 实测（~2h）— 即使 R32 sweep 一致输 lossless，HKVD 信号仍是机制层硬证据
- P4 R40 zmq pickle 边界修复（半天-1 天）— 解锁 7 个 timing 字段 + chunk-pool telemetry 通道

### 中期（method refresh）：
- Direction B (dataflow) P1' 已 FALSIFIED at P0（contiguous head 无法表达 selective per-token；per-token mask 需 CacheBlend 多段重写 per pool-chunk，1.5-2 周）— 见 `ABLATION_DATAFLOW_P0.md`
- P3 True CacheBlend attention-kernel hook（多周，需 sign-off）— 唯一通向真正 novelty 的路径

### 长期（推荐配置 framing）：
- 既然 R32 是 speed-accuracy 权衡（不是 accuracy-preserving），CLAUDE.md §3 的 framing 应改成 **"speed-optimization for latency-sensitive"** 而非 "lossless-equivalent"。EuroSys 投稿时应正面承认 ~13% accuracy 一致性代价。

## 7. 引用

- `results/scale15_5x5/paired_analysis_p1pp.py` — strict-FAIL paired (1 case intersection)
- `results/scale15_5x5/paired_analysis_p1pp_v2.py` — broad-5/5 paired (12 cases)
- `results/scale15_5x5/r32_f045/outputs.jsonl` — dedup'd 61 unique rows
- `results/scale15_5x5/ablation_nodekind_summary.json` — full 8-config summary
- `results/compute_dataflow_budget.py` — Direction B P0 falsification (related)
- CLAUDE.md §2c/§3/§6 P1''
- memory `type-agreement-denominator-artifact-2026-07-09`