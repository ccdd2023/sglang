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

### 2c. Direction A 等预算消融 = FALSIFICATION (2026-07-10)

`ABLATION_NODEKIND_REPORT.md`：落地 direction A 的 **contiguous** 版（recompute 函数 interface=signature+docstring，copy body；scattered control-flow 因 sglang contiguous-prefix 约束 TTFT 自相矛盾，见报告 §2）。8-config 等预算消融（n=15，budget proxy = c2_reused，B=total−c2_reused）：

| config | type_match | c2_reuse | TTFT |
|---|---|---|---|
| lossless | 10.7% | 0 | 1028 |
| R32_f015/026/030/045 | 6.6/9.8/9.8/**11.5%** | 465/380/345/268 | 707/713/715/720 |
| R38b | 6.7% | 283 | 721 |
| nodekind (interface) | 6.6% | 362 | 715 |
| nodekind_sig | 3.3% | 523 | 701 |

- **R32 sweep 干净单调**：recompute 越多 accuracy 越高（6.6->9.8->11.5%）。
- **nodekind (6.6%) 在等预算 (c2_reuse≈362) 下比 R32 (~9.8%) 低 3.3pp**；nodekind_sig 更差（3.3%）。两者被 R32 Pareto 支配。**code structure (node-kind) 不买精度** - 位置驱动 head 比 code-structure 驱动 interface 更优。假设（interface 敏感、body 安全 copy）被推翻：**body 比 docstring 更精度关键**。
- **R34 教训全命中**：gate 在 AST node kind（100% fire rate，非 no-op）+ 等预算消融（R34 缺->被当 global bump）。
- **Caveat**：n=15+OOM CI 宽（-3.3pp 在 CI 重叠内），common-complete-cases=0；方向一致但不统计显著。
- ~~次要正向 R32_f045~~ **RETRACTED 2026-07-10 via paired test**（`ABLATION_R32_F045_CONFIRMATION.md`）：12 cases × 5 agents = 60 paired obs 上 mean delta = -0.67/5 ≈ -13% type-match 一致性损失 vs lossless, 95% CI [-1.33, +0.08]。§2c 表 11.5% vs 10.7% 是 /n 分母假象（OOM drops n=61 vs n=75）。R32 sweep 4 点一致 -0.67 ~ -0.83，无 monotonic 优势。**R32 是 speed-accuracy 权衡（1.43× 换 ~13% 一致性），非 accuracy-preserving**。
- **副产品 bug 修复**：`_build_byte_to_token_map` 运行时静默 None（HF Encoding unpack 失败），R32/R38b 退到 O(chunks×text) re-encode；修后 offset 不变（0/120 diff）仅更快（R32 745->715ms）。见 memory `byte-to-tok-broken-encoding-fix-2026-07-10`。

**结论**：direction A (contiguous) 证伪；reinforces §2a（方法=速度优化非精度保持）。

### 2d. Direction B (dataflow) P0 cheap signal = FALSIFIED (2026-07-10)

`ABLATION_DATAFLOW_P0.md`：`compute_dataflow_budget.py` 离线预算 pandas_15case_v1（120 chunks, stdlib-only AST，无 sglang runtime 改动）：

| 信号 | 数值 | 评估 |
|---|---|---|
| Cross-use fire rate | 82/120 = 68.3%（class 100%, function 65.8%） | ✅ 有结构信号 |
| Pure-mask dataflow FRAC | **0.0507**（median 0.031, p90 0.146, max 0.306） | ⚠️ 5% B，远低于所有 baseline |
| Contiguous approx "last cross" K | 0.845 ≈ lossless | ❌ 过度 recompute，无 selective 优势 |
| Contiguous approx "first cross" K | 0.156 ≈ R32_f015 | ❌ 不是 novel lever — 另一形态 uniform |

**结论**：contiguous head 机制**结构性等价**于 R32 sweep（沿位置的 uniform），无法表达 selective per-token targeting。Per-token mask = 多段 CacheBlend 重写 per pool-chunk（1.5-2 周），期望收益边际（5% B 已低于 R32_f015 critical mass）。P1' (dataflow) **FALSIFIED at P0**。

### 2e. HKVD-by-node-kind = NEGATIVE（机制层判决，2026-07-10）

`ABLATION_HKVD_NODEKIND.md`：`measure_hkvd_by_node_kind.py` 直接测 AST interface tokens vs body tokens 的 KV deviation（canonical->live prefix swap，pool precompute 真实场景，40 chunks × paired）：

| group | K_dev | V_dev | 说明 |
|---|---|---|---|
| interface | 0.0843 | 0.0100 | Direction A recompute 的部分 |
| body | 0.0886 | 0.0061 | Direction A copy 的部分 |

paired: mean_delta(iface−body)=**-0.0043**, 9 iface>body vs 31 body>iface, **Wilcoxon one-sided p=0.9999**（interface>body 假设强烈拒绝）。

**判决**：structure signal 在 KV 层**不存在** - interface tokens 的 K deviation **不高于** body（实际略低 5%）。Direction A recompute 了 K-不敏感的 interface、copy 了 K-敏感的 body = **最坏** selective 策略，机械解释 -3.3pp。**整条 "code structure decides recompute" 线死亡，含 P3 AST targeting**（即使有完美 per-token 机制也无 AST 信号可利用）。

**三重证伪完成**：Direction A（算法层 -3.3pp）+ Direction B P0（算法层 contiguous 无法 selective）+ HKVD-by-node-kind（机制层信号不存在）。剩余真实增益只有 R32 uniform-along-position（1.43× 换 ~13% type-match，position-aware 非 code-aware）。

**下一步研究方向**：per-token HKVD（non-AST，找真正敏感 token）/ P4 R40 zmq pickle 修复（解锁测量）/ 换 code-gen task 看是否有信号 / 或重新定位为纯 speed-accuracy 权衡（R32）。

### 2f. Literature validation (2026-07-10 deepresearch)

Deep research synthesis (5 parallel agents, ~150 papers scanned, see `results/DEEPRESEARCH_*.md`) confirms our triple falsification is part of a **broader pattern**:
- **CodeBERT** (EMNLP'20) reports AST traversal "does not bring improvements on generation tasks"; **GraphCodeBERT** (ICLR'21) deliberately chose **data-flow over AST** because "AST has unnecessarily deep hierarchy."
- **StreamingLLM / Scissorhands / SnapKV** (NeurIPS'23-'24) all win via **positional or attention-history signals** — none use AST.
- **Hahn TACL'20** theoretical limit + **Jain NAACL'19 "Attention is not Explanation"** give theoretical backing: AST boundaries likely don't survive as reusable KV features.
- **No production coding AI system** (Cursor, Copilot, Codeium, Aider, Cline, Continue, Cody, Claude Code) does code-structure-aware lossy KV reuse. They use one of three regimes: (1) keep prefix verbatim + cache_control, (2) smaller model → fast prefill, (3) async workflow tolerating hours.
- **Closest related work**: CacheBlend (ICML'25, 2.2-3.3× TTFT lossless-quality on RAG), CortexCache (Mar'25, 1.5-2.5× on code completion), Position-Aware Recomputation (2502.08201). R32 = 1-axis generalization of CacheBlend. **CortexCache parity benchmark is the cheapest Tier-A win** if work resumes.
- **Unclaimed territory**: DroidSpeak (Nov'24, Microsoft) is the only published system transferring actual KV tensors across distinct LLM instances (1.7-3.1× prefill ↓); applying to our 5-agent verdict pipeline is a novel cross-agent contribution.
- **Honest reframe**: R32 (1.43× for ~13% type-match consistency loss) is competitive but **not best-in-class**; CortexCache's 1.5-2.5× on code suggests ~0.5-1× headroom via per-corpus FRAC tuning.

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

定位：**latency-sensitive verdict 任务**可用，**accuracy-critical 不适用**。Paired test（12 cases × 5 agents = 60 obs）显示 R32 @ 任意 FRAC 一致损失 **~13% type-match 一致性**换 **1.43× TTFT 加速**（mean delta -0.67/5, CI [-1.33, +0.08]）— 是 **speed-accuracy 权衡**，不是 accuracy-preserving。code-gen 任务 R38b 反而亏（见 slide 28 R40-P2 git apply 0/5 vs 1/5）。

## 4. 关键文件 (5 个，按优先级)

1. `CLAUDE.md` ← **本文**
2. `results/CODE_AWARE_LOSSY_KV_PROGRESS_FINAL.html` (14 pages) + `.pdf` — **deck** 全部当前叙述（三重证伪时间线 + R32 最终定位）
3. `results/CODE_AWARE_LOSSY_KV_PROGRESS_R28_R39.html` (35 pages, archived) — pre-三重证伪 deck，按 HANDOFF §7 归档约定保留
4. `results/SCALE15_HKVD_REPORT.md` — **n=15 + HKVD 实测权威源**（2026-07-09）
5. `results/R40_COMBINED_REPORT.md` — R40 Phase 1/2/3 实现（顶部有 n=15 超越横幅）

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

> **2026-07-10 (later) update**: P1' dataflow (B) **FALSIFIED at P0**（见 §2d + `ABLATION_DATAFLOW_P0.md`）。下表 P1' 行标 ❌。新优先级 = **P1'' R32_f045 确认**（~3h）/ **P0 HKVD-by-node-kind**（~2h）/ **P3 True CacheBlend hook**（多周）。

| Pri | 项目 | 依据 | 估时 |
|---|---|---|---|
| **P0** | ❌ **HKVD-by-node-kind 实测 DONE = NEGATIVE** - interface K_dev(0.0843) ≤ body K_dev(0.0886), Wilcoxon one-sided p=0.9999; structure signal 在 KV 层不存在，整条 code-structure-recompute 线死亡（含 P3 AST targeting）。见 §2e + `ABLATION_HKVD_NODEKIND.md` | slide 18 决定性实验 | DONE |
| **P1** | ~~code-structure-driven selective recompute 实施（A node-kind FRAC 修复 R34）~~ — FALSIFIED, see §2c | slide 18 三方向 | DONE |
| **P2** | ~~等预算消融决定性实验~~ — DONE, see §2c/§2d | slide 18 novelty 证明 | DONE |
| **P3** | ~~True CacheBlend HKVD attention-kernel hook~~ - 动机消失（HKVD-by-node-kind 否定 AST targeting 信号）。若重启需改用 non-AST per-token 信号 | slide 17 不足① | ON HOLD |
| **P4** | ✅ **R40 TTFT-breakdown zmq pickle FIX DONE** - `__getstate__` allowlist + enable_metrics gate + NameError(plan) 修复；6/6 blocked 字段现在非零。见 `ABLATION_P4_TTFT_BREAKDOWN_FIX.md` | slide 26 | DONE |
| **P5** | `results/codebase_kv/` 过期 pool 清理（gitignored，~9 GB reclaimable，列在 §10） | 见下 | 手动 |
| **P1'** | ❌ **dataflow (B)** - FALSIFIED at P0, see §2d | slide 18 direction B | DONE (falsified) |
| **P1''** | ❌ **R32_f045 确认 RETRACTED** - paired test 显示 R32 sweep 4 点一致输 lossless by ~0.7-0.8 mean agree/case；§2c "次要正向" 撤回 | §2c retracted | DONE |

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
7. **(2026-07-10) Direction A 等预算消融 = FALSIFICATION**：实现 contiguous node-kind interface-recompute（`ast_chunker.py` + `radix_cache.py`，env `SGLANG_CHUNK_HEAD_RECOMPUTE_NODE_KIND`）+ 8-config 等预算消融（`results/scale15_5x5/launchers/` + `analyze_ablation_nodekind.py`）。node-kind 等预算下 -3.3pp vs R32，Pareto 支配 -> code structure 不买精度。副产品修 `_build_byte_to_token_map` 静默 None bug。报告 `ABLATION_NODEKIND_REPORT.md`，见 §2c。
8. **(2026-07-10) Direction B (dataflow) P0 cheap signal = FALSIFIED**：`compute_dataflow_budget.py` 离线预算 pandas_15case_v1（stdlib-only AST）。cross-use fire rate 82/120=68.3%，但 contiguous-head 近似下要么 K=last_cross=0.845 (lossless-level over-recompute) 要么 K=first_cross=0.156 (≈ R32_f015, 不是 novel lever)。Per-token mask 需 CacheBlend 多段重写 per pool-chunk（1.5-2 周），期望收益边际。**Falsify at P0**：do not proceed P1。报告 `ABLATION_DATAFLOW_P0.md`，见 §2d。下一步：P1'' R32_f045 确认（~3h）/ P0 HKVD-by-node-kind（~2h）/ P3 True CacheBlend hook（多周）。
9. **(2026-07-10) P1'' R32_f045 paired confirmation = §2c "次要正向" RETRACTED**：`paired_analysis_p1pp_v2.py` 在 12 cases × 5 agents = 60 paired obs 上：R32_f045 mean agree delta vs lossless = -0.67/5（~ -13% type-match 一致性）, 95% CI [-1.33, +0.08], Wilcoxon p=0.156。§2c 表 11.5% vs 10.7% 是 /n 分母假象（OOM drops n=61 vs n=75 不同），R32 sweep 4 点一致 -0.67 ~ -0.83 无 monotonic。**R32 是 speed-accuracy 权衡（1.43× 换 ~13% type-match 一致性），不是 accuracy-preserving**。Re-run R32_f045 触发同 task 7 OOM (deterministic) — 完整 75 rows 在现有 setup 下不可达。报告 `ABLATION_R32_F045_CONFIRMATION.md`，见 §2c retracted + §3 reframed。

10. **(2026-07-10) HKVD-by-node-kind = NEGATIVE（机制层判决）**：`measure_hkvd_by_node_kind.py` 测 AST interface vs body tokens 的 KV deviation（40 chunks × paired，canonical->live prefix swap）。interface K_dev=0.0843 ≤ body K_dev=0.0886，Wilcoxon one-sided p=0.9999。**structure signal 在 KV 层不存在** - Direction A recompute 了 K-不敏感 interface、copy 了 K-敏感 body（最坏策略，机械解释 -3.3pp）。**三重证伪完成**（A 算法层 + B 算法层 + HKVD 机制层），整条 code-structure-recompute 线死亡含 P3 AST targeting。报告 `ABLATION_HKVD_NODEKIND.md`，见 §2e。剩余真实增益 = R32 position-aware（非 code-aware）speed-accuracy 权衡。

11. **(2026-07-10) P4 R40 TTFT-breakdown zmq pickle FIX DONE**：修两个 compounding bug - (a) `SchedulerReqTimeStats.__getstate__` allowlist 遗漏 6 R40 fields + `enable_metrics=False` gate 返回 {} （req_time_stats.py:642-680）；(b) `NameError: plan` 未定义被 `except: pass` 吞 （radix_cache.py:3963，改用 `len(layout)`）。单元测试 5 PASS + 3-case bench 验证 6/6 blocked 字段（radix_prefix/chunk_plan/copy/gap_prefill/head_recompute_early/late）现在非零。解锁 TTFT breakdown per-stage 测量（之前只有 tokenize_ms 可测）。报告 `ABLATION_P4_TTFT_BREAKDOWN_FIX.md`，supersedes memory `r40-ttft-breakdown-architecture-block-2026-07-09`。

## 8. Memory 指针 (auto-persisted)

- `deepresearch-coding-inference-accel-2026-07-10` — 5-agent parallel deepresearch 验证三重证伪是 broader pattern；CacheBlend/CortexCache 是 R32 的 SOTA 等价；production 不做 code-aware lossy KV reuse；DroidSpeak 是唯一 cross-agent KV 工作
- `direction-a-contiguous-node-kind` — Direction A contiguous node-kind interface-recompute 等预算 -3.3pp vs R32；R32_f045 ≈ lossless @ 1.43×；code structure 不买精度
- `direction-b-dataflow-p0-falsified-2026-07-10` — contiguous head 无法表达 selective per-token；K=last=0.845 (lossless) 或 K=first=0.156 (≈ R32_f015)；per-token mask 需 CacheBlend 多段重写 (1.5-2 周)；FALSIFIED at P0
- `p1pp-r32-f045-retracted-2026-07-10` — §2c "次要正向" 撤回；paired test 12 cases × 5 agents: R32_f045 mean delta -0.67/5 (-13% type-match) CI [-1.33, +0.08]；R32 是 speed-accuracy 权衡非 accuracy-preserving
- `hkvd-by-node-kind-negative-2026-07-10` — interface K_dev(0.0843)≤body(0.0886) p=0.9999；AST 结构信号在 KV 层不存在；三重证伪完成（A+B+HKVD）；code-structure-recompute 线死亡含 P3；剩余 R32 position-aware
- `p4-ttft-breakdown-zmq-fix-2026-07-10` — `__getstate__` allowlist + enable_metrics gate 丢 6 R40 fields；+ NameError(plan) 被 except 吞；修后 6/6 字段非零；supersedes r40-architecture-block
- `byte-to-tok-broken-encoding-fix-2026-07-10` — `byte_to_tok` 运行时静默 None（HF Encoding unpack 失败）；修后 R32/R38b 更快 + node-kind 才能触发
- `chunk-pool-telemetry-two-emission-paths` — counter 要加 serving_chat.py(2 list)+scheduler_output_processor_mixin 才到 rows.csv
- `type-agreement-denominator-artifact-2026-07-09` — `type_match/FAIL` 分母假象；n=5 的 50%>41.7% 是分母缩水，非真实贡献
- `scale15-hkvd-2026-07-09` — n=15 推翻 n=5 精度；HKVD +7.2% 实测
- `r32-cacheblend-head-recompute` — R32 head_recompute FRAC=0.30 唯一 Pareto
- `direction-3-phase-c-d` — L4 chunk pool strict byte-exact
- `l4-contiguity-ceiling-2026-06-28` — safe L4 = 0 reuse in giant-codebase → CacheBlend 必要
- `c2-cacheblend-lossy-not-safe-2026-06-28` — byte-exact text ≠ KV-exact when prefix 不同

## 9. Git

- **Branch**: `fix/placeholder-pool-activation` (持续开发分支)
- **近期 commits** (2026-07-09/10):
  - `4a8a11fc4` fix(gitignore): drop broken *.backup_*-pre_* pattern (git 2.34.1 doesn't match)
  - `1d47d99f7` docs(cleanup): CLAUDE.md handoff + 归档过期内容 + gitignore 收紧
  - `23b7ba793` docs(deck): novelty 不足② + 补救路线图 (slide 17/18)
  - `90a596f93` docs(deck): reorder to algorithm-first
  - `38077d2a4` feat(scale15+HKVD): validate mechanism, n=15 revises n=5
- **重生 deck PDF (FINAL)**: `python3 results/gen_progress_pdf_final.py`（Playwright，1280×720px，14 页）

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

**TL;DR for next session**: R32 1.43× speed (n=15 solid, paired test confirmed)；method = speed-accuracy 权衡非精度保持；novelty gap = 三重证伪完成（Direction A 算法层 -3.3pp / Direction B P0 算法层 / HKVD-by-node-kind 机制层 p=0.9999）；code-structure-recompute 线死亡含 P3；剩余 R32 position-aware。Deck `results/CODE_AWARE_LOSSY_KV_PROGRESS_FINAL.html` (14 pages) 为最终叙述权威源。