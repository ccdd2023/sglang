# CLAUDE.md — sglang-kvflow fork

> 接手文档：下一个 session 读本文即可在 15 分钟内了解项目状态、关键文件、硬约束、open work。
> Created: 2026-07-10 (post n=15 scale-up retest + deck novelty 不足② + slide 18 补救路线图).

---

## 1. 项目 (1 段)

**sglang-kvflow fork**：在 sglang `RadixCache` 之上加 **code-aware 有损 KV 缓存复用**，目标 = coding multi-agent system (MAS) 工作流的 TTFT 加速，accuracy 在同 prompt 下与通用复用算法持平。**EuroSys 2026 投稿目标**。**硬约束：加速只来自更多复用 — 禁止任何 KV-cache 调度技巧**（无 async scheduler / no preemption / no batching trick）。fork 在 `python/sglang/srt/mem_cache/`：`radix_cache.py` (L2/L4/C2/MULTI_SLOT) + `ast_chunker.py` (server-side AST chunker)。

四层路径（byte-exact 逐层开启）：
- **L2** — whole-slot byte-exact + RoPE rotation (cross-position)
- **L4** — AST-boundary chunk reuse (byte-exact per function/class)
- **C2 / MULTI_SLOT** — CacheBlend gap-prefill + multi-slot batched copy
- **L3 MiniLM 语义 k-NN** — **deprecated**, research only; byte-exact match is the reuse gate

## 2. 当前状态 (2026-07-10)

| 指标 | 数值 | 备注 |
|---|---|---|
| **TTFT 加速 (n=15)** | **1.38× solid** | R32: 745ms vs lossless 1032ms |
| **推荐配置** | **R32 (FRAC=0.30)** | n=15 上 R32 > R38b |
| **HKVD 机制** | **pos1 K_dev +7.2% > pos5** | 实测验证（`hkvd_by_position_20260709/`） |
| **Novelty 缺口** | code structure 只驱动速度 | 方法 = KVCOMM + CacheBlend-lite |

### 2a. n=15 推翻 n=5 (2026-07-09)

`SCALE15_HKVD_REPORT.md` 揭示：n=5 (全 combine_file 易 case) 上 R32 head_recompute "恢复到 lossless (2->5)" 是 **易 case 偏置**。n=15 多样化复测后：
- lossless 10.7% > **R32 9.8%** > R38b 6.7% (type_match)
- R32 仍 1.38× 速度优势（345 tok 真实 code-aware 复用）
- **方法本质 = 速度优化（Pareto 权衡 ~1pp 精度换 1.38× 加速），非精度保持**

### 2b. Novelty 缺口 + 补救 (slide 17 不足② + slide 18 补救路线图)

诊断：当前 **AST chunking 只驱动 chunking(速度)**；accuracy 杠杆 FRAC = "recompute 前 N token" = **位置驱动、代码无关** → 无 code-aware accuracy 杠杆。R34/R40-P3 想用 type annotation 补但挂在罕见特征上 → no-op retired。

补救（slide 18 三方向）：让 code structure 决定 **recompute 什么**：
- **A · node-kind FRAC**（今晚可做）：recompute 签名/控制流节点（def/class/if/return），copy docstring/boilerplate
- **B · dataflow**（真 novelty）：只 recompute 引用了"上游已变 symbol"的 token
- **C · task-cycle**：AST-diff 跨 agent 迭代，未变区域复用

**决定性实验**：固定总 recompute budget B，等预算比 uniform(R32) / position(R38b) / node-kind(A) / dataflow(B) 的 accuracy → 若 A/B > R32/R38b @ equal B 证明 code structure 买精度。

## 3. 推荐配置 (production-ready, 7B-Coder × 5 verdict 任务)

```bash
# R32 — constant-FRAC=0.30 head_recompute (推荐，n=15 上 R32>R38b)
export SGLANG_PLACEHOLDER_KNN_MATCH=0
export SGLANG_L3_AST_GATE=0
export SGLANG_CHUNKED_PLACEHOLDER_KNN=1
export SGLANG_CHUNKED_PLACEHOLDER_KNN_MATCH=1
export SGLANG_CHUNK_TOPLEVEL=1
export SGLANG_CACHEBLEND_CHUNK=1
export SGLANG_CACHEBLEND_MULTI_SLOT=1
export SGLANG_PRECOMPUTE_KV_DIR=results/codebase_kv/pandas_5case_v4
export SGLANG_PRECOMPUTE_CANONICAL_PREFIX=1
export SGLANG_PRECOMPUTE_PROMPT_ALIGN=1
export SGLANG_PRECOMPUTE_SELECTIVE_REFRESH_FRAC=0.25
# 不设 CHUNK_HEAD_RECOMPUTE_FRAC_EARLY/LATE = 默认 R32 行为
```

定位：**latency-sensitive verdict 任务**可用，**accuracy-critical 不适用**（n=15 未超 lossless）。code-gen 任务 R38b 反而亏（见 slide 28 R40-P2 git apply 0/5 vs 1/5）。

## 4. 关键文件 (5 个，按优先级)

1. `CLAUDE.md` ← **本文**
2. `results/CODE_AWARE_LOSSY_KV_PROGRESS_R28_R39.html` (35 pages) + `.pdf` — **deck** 全部当前叙述（含 slide 17 不足② + slide 18 补救）
3. `results/SCALE15_HKVD_REPORT.md` — **n=15 + HKVD 实测权威源**（2026-07-09）
4. `results/R40_COMBINED_REPORT.md` — R40 Phase 1/2/3 实现（顶部有 n=15 超越横幅）
5. `results/R34_R37_SUMMARY.md` — R33-R37 SWE-bench fix-mode 证据

参考附录（保留，含原始数字 + 复现命令）：
- `results/HARNESS_CHANGE_NOTES_20260707.md` — R36/R37 harness 变更
- `results/research_code_aware_kv_reuse_2026-07-06.md` — 5-agent 文献综述（14 arxiv）
- `results/r##_tech_name_index.md` — R## → 算法名映射
- `results/design_proposal_r40_20260708.md` — R40 设计提案（historical）

历史快照（pre-n=15，**不要从这取数**）：
- `reports/code_aware_lossy_kv_progress_20260707.pdf` (16 页 LaTeX) — pre-n=15 快照；按 HANDOFF §7 日期快照约定不覆盖

## 5. 硬约束 (bullet, 来自 memory — 不要凭印象)

**实验设置**
- 加速只来自复用；L3 MiniLM k-NN 默认 OFF；code-aware 复用命中必须真实（`codeaware_reused_tokens > 0`）
- 结果输出到 `results/`，不用 `/tmp`
- **>3 case 必加 `--disable-overlap-schedule --max-running-requests 1`**（避免并发干扰）
- `--force-evict` **不是**真实 flag（只是 load-bearing 思路命名）
- F1 读 `outputs.jsonl` 的 `output_text`，**不要信 `rows.csv`**

**评估**
- 用 `type_match/25`（固定分母）而非 `type_match / FAIL_rows`（分母假象 — 见 `type-agreement-denominator-artifact-2026-07-09`）
- type_match 全 5/5 = lossless-level accuracy recovery (n=5 上的；n=15 推翻)
- verdict 精度 ≠ 任务能力（R38b verdict 5/25 = lossless 但 code-gen git apply 0/5 vs lossless 1/5 — slide 28）

**gitignored 大件**（不要重新 track）
- `results/codebase_kv/` (~1.2 GB/run)
- `results/swebench_local_envs/repos/` (21G)
- `results/giant_codebase/pandas_src/` (462M)
- `results/ttft_agenttemplatekv/` (pre-R3 placeholder sweep)
- `reports/build/` (LaTeX 衍生产物)

## 6. Open work (按优先级)

| Pri | 项目 | 依据 | 估时 |
|---|---|---|---|
| **P0** | **HKVD-by-node-kind 实测**（slide 18 "先验证"）— 扩展 `measure_hkvd_by_position.py` 测签名节点 vs body 节点 deviation | slide 18 决定性实验前置 | ~2h |
| **P1** | code-structure-driven selective recompute 实施（A node-kind FRAC 修复 R34） | slide 18 三方向 | 多天 |
| **P2** | 等预算消融决定性实验（uniform vs position vs node-kind vs dataflow @ equal B） | slide 18 novelty 证明 | 1-2 天 |
| **P3** | True CacheBlend HKVD attention-kernel hook（多周，需 user sign-off） | slide 17 不足① | 多周 |
| **P4** | R40 Phase 1 架构 block 修复（scheduler↔tokenizer zmq pickle 边界丢失 7 个 timing 字段） | slide 26 | 半天-1 天 |
| **P5** | `results/codebase_kv/` 过期 pool 清理（gitignored，~9 GB reclaimable，列在 §10） | 见下 | 手动 |

## 7. 本次 cycle 变更 (2026-07-09/10)

1. **scale-15 n=15** 推翻 n=5 精度结论 → 推荐改为 R32
2. **HKVD-by-position 实测**验证机制假设（pos1 K_dev +7.2% > pos5）
3. **Deck 更新**：
   - 新增 **slide 17 不足②** (Novelty 缺口诊断)
   - 新增 **slide 18 补救路线图**（3 方向 + 决定性实验 + HKVD-by-node-kind 先验证）
   - 重排为算法-first（问题→算法→实验→结论）
   - TL;DR 折叠 novelty 提示
4. **归档**：过期 root handoffs → `_archive/`；过期 rounds → `results/_archive/2026-07-10_superseded_rounds/`；旧 deck 快照 → `results/_archive/2026-07-10_old_deck_and_backups/`
5. **新建 CLAUDE.md**（取代 4 个过期 root handoff）
6. **R40_COMBINED_REPORT.md** 顶部加 n=15 超越横幅

## 8. Memory 指针 (auto-persisted)

- `type-agreement-denominator-artifact-2026-07-09` — `type_match/FAIL` 分母假象；n=5 的 50%>41.7% 是分母缩水，非真实贡献
- `scale15-hkvd-2026-07-09` — n=15 推翻 n=5 精度；HKVD +7.2% 实测
- `r32-cacheblend-head-recompute` — R32 head_recompute FRAC=0.30 唯一 Pareto
- `direction-3-phase-c-d` — L4 chunk pool strict byte-exact
- `l4-contiguity-ceiling-2026-06-28` — safe L4 = 0 reuse in giant-codebase → CacheBlend 必要
- `c2-cacheblend-lossy-not-safe-2026-06-28` — byte-exact text ≠ KV-exact when prefix 不同

## 9. Git

- **Branch**: `fix/placeholder-pool-activation` (持续开发分支)
- **近期 commits** (2026-07-09/10):
  - `90a596f93` docs(deck): reorder to algorithm-first
  - `38077d2a4` feat(scale15+HKVD): validate mechanism, n=15 revises n=5
  - `8a91c3509` docs(R40): audit fixes
  - `bc3e7f3fb` docs(R40): update progress deck with 3 new R40 slides
  - `21900d017` feat(R40-Phase3): type-aware FRAC override
- **重生 deck PDF**: `python3 results/gen_progress_pdf_r28_r39.py`（Playwright，1280×720px）

## 10. 可回收 disk（手动）

`results/codebase_kv/` 过期 pool（gitignored，~9 GB reclaimable）：
```
pandas_5case/ pandas_5case_v2/ pandas_5case_v3/
pandas_5case_v5_role_impl/ pandas_5case_v6_filelevel_coarse/
pandas_5case_v6_verdict_3b/ pandas_5case_v6_verdict_coder3b/
pandas_5case_v7_50files/ pandas_5case_v8_role_impl/
pandas_5case_v9_role_impl/ smoke_5/
```
仅 `pandas_5case_v4/`, `pandas_5case_v6_verdict/`, `pandas_15case_v1/` 被当前 harness 引用。手动 `rm -rf` 其余即可。

---

**TL;DR for next session**: R32 1.38× speed (n=15 solid)；method = 速度优化非精度保持；novelty gap = code structure 没驱动 recompute；P0 是 HKVD-by-node-kind 实测。一切数字以 `results/CODE_AWARE_LOSSY_KV_PROGRESS_R28_R39.html` deck slide 20 (TL;DR) 为准。