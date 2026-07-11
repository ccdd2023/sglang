# True CacheBlend — GO/NOGO Decision Memo (Phase T0)

Date: 2026-07-11
Author: Claude (sglang-kvflow)
Plan: /home/gfy/.claude/plans/abstract-waddling-sundae.md

## 决策: **GO**

走 Path A prototype (per-token 1-token chunked prefill)，验证 True CacheBlend 是否能填补 R32 与 CacheBlend-class 系统 (2.2-3.3× TTFT) 之间的 0.5-1× headroom。

## 上下文 reminder

**当前状态 (CLAUDE.md §2h, 2026-07-11 wrap-up)**：
- R32 (`FRAC=0.30`) 是 unique Pareto，1.43× speed 换 ~13% type-match 一致性损失
- 三重证伪：Direction A (algorithm) -3.3pp falsified；Direction B (dataflow P0) contiguous ≡ R32 falsified；HKVD-by-node-kind (interface ≤ body) p=0.9999 falsified
- Phase 4 多信号 HKVD：control_flow vs data_flow **POSITIVE +470% p=0.0000** @ mechanism layer
- Phase 5 control-flow-selective recompute **NEGATIVE at policy layer** (-1.6pp vs R32_f045 @ equal B)
- **code-structure-recompute 研究线** 在 policy 层 FULLY DEAD
- True CacheBlend (per-token mask, 不是 contiguous head) 是未被探索的最后一个 frontier

## 接受约束 (sunk-cost guard)

我明示接受以下 tripwires，**任一** trigger 立即收尾不再 iterate：

1. **Phase T1 overhead gate**: minipre p95 > `SGLANG_TRUE_CACHEBLEND_OVERHEAD_GATE_P95_MS` (default 8ms) → Path A infeasible → retire 或升级 Path B
2. **Phase T2 signal sanity**: HKVD-vs-control_flow overlap < 25% → signal 不适用 → 退回 T1 uniform-p%
3. **Phase T3 Pareto fail**: ≥ 3 of 5 criteria miss bar → NEGATIVE @ policy layer (4th falsification)
4. **Phase T3 statistical fail**: paired Wilcoxon p ≥ 0.10 vs R32_f030 → NOISE not win
5. **Wall-clock cap**: T3 未启动 by 2026-07-16 (今天 +5 天) → 整体 retire regardless of progress

## 防御性 design constraints

- **Default OFF**: 所有新增 ENVs (`SGLANG_TRUE_CACHEBLEND`, `SGLANG_TRUE_CACHEBLEND_PCT`, `SGLANG_TRUE_CACHEBLEND_USE_HKVD_LABELS`) default 0，不动 production 行为
- **Additive-only modifications**: 不修改任何既有 path，不删除任何 line；新字段 default 0/empty 保证 zero-cost
- **Telemetry required before measurement**: 4 emission sites (radix_cache counter + serving_chat × 2 + scheduler_output_processor_mixin × 2 + bench stress × 1) 必须 wired 否则等于 R32 baseline (无法识別 overhead)
- **NEGATIVE = paper contribution**: 即使 NEGATIVE，本方向也是 4th falsification paper-section 的强证据（沿用 Phase 5 doc pattern 写到 ABLATION_TRUE_CACHEBLEND.md）
- **No "再试一次" iteration**: T1 fail → 不微调参数尝试 T1'；直接跳 T5 Path B 或 retire
- **No sglang mainline coupling**: 本工作 not coupled to upstream, not blocking PR；失败后可 clean-revert

## 风险评估

如果 5 个 tripwires 全部 PASS 而 Pareto fail (即 overhead OK 但 signal doesn't transfer)：
- 与 Phase 5 同 pattern (mechanism POSITIVE → policy NEGATIVE) → 表明 per-token selective recompute 在 code-streaming setting 也失效
- 配合 Phase 5 共同验证 **所有 selective-recompute path 都 death**，paper claim 升级
- 此时只可能的 2nd 升级：Path B (masked query forward via Triton custom_mask) — 但需要 attention backend 改动，scope out of default 2 周 cap

如果 T3 POSITIVE (per-token mask beats R32 at equal B + overhead within gate)：
- 1.43× → ~1.7× speed headroom + type-match 改善
- 升级 R32 到 R-XX (R40+) 配置
- 同时写 paper-section "True CacheBlend in sglang"

## 计划 timeline 估算

| Phase | Days | Wall-clock cap |
|---|---|---|
| T1 pilot | 1 | by 2026-07-12 |
| T2 signal | 1.5 | by 2026-07-13 |
| T3 ablation | 1 | by 2026-07-14 |
| T4 decision | 0.5 | by 2026-07-15 |
| **Hard cap** | | **2026-07-16** |

每 phase end: write brief status memo (≤200 words) 记录 outcome + next decision。

## 资源承诺

- 每天只工作 ~1 phase，不并行 multi-task（避免分散 signal）
- 任一 tripwire 不晚于 30 分钟内决定 → 收尾 not iterate
- ABLATION_TRUE_CACHEBLEND.md 与 CLAUDE.md update 在 T4 完成后同 commit 推送

## 不做的清单 (scope guard)

- ❌ 改 attention backend (Path B) — 默认 not in scope
- ❌ 改 `ForwardBatch` schema — 默认 not in scope
- ❌ 改 `ScheduleBatch` 主流程 — 仅 extend `cacheblend_stage` 分支
- ❌ 加新的 prebuilt KV pool — 复用 `pandas_15case_v1` (120 chunks)
- ❌ 多 task 并行 ablation — 单 task serial
- ❌ 微调 optimum FRAC/PCT 一旦 NEGATIVE — 立即收尾
- ❌ 跳过 telemetry 直接 ablation — failure mode 不可识别

## 后续 (T4 完成)

无论 verdict 是 NEGATIVE / MARGINAL / POSITIVE：
- `results/ABLATION_TRUE_CACHEBLEND.md` — 完整 report (mirror Phase 5 pattern)
- `CLAUDE.md` §6 P3' 状态更新
- `CLAUDE.md` §3 / §4 / §9 conditional updates
- 1 new memory pointer
- Optional: deck §14 (Phase 6 slot)

---

**签字**: Claude，2026-07-11
**Plan 引用**: `/home/gfy/.claude/plans/abstract-waddling-sundae.md`
**下游 tasks**: T1 (#54) → T2 (#55) → T3 (#56) → T4 (#57)
