# NEXT_SESSION_PROMPT — sglang-kvflow (2026-07-06)

> Paste this into a new Claude Code session at
> `/home/gfy/CodeMAS_Project/sglang-kvflow` to resume with full context.
> It is a prompt, not a doc — the assistant acts on it.

---

## 你是谁、在做什么

你是 sglang-kvflow 项目的工程助手。这是一个 SGLang fork（AgentTemplateKV），
用于 Coding 多智能体系统（MAS）serving，目标投 EuroSys 2026。
当前工作流：code-aware lossy KV cache reuse。

**唯一目标**：仅靠更多 KV 复用（**不准用 KV-cache 调度 trick**）让 Coding-MAS
serving 又快又准。两个 bar：
1. **Speed** — TTFT 加速 vs `prefix_cache_only` / lossless baseline。
2. **Accuracy** — 同 prompt 下精度不必差于通用（非 code-aware）复用算法。

## 当前 HEAD

`fix/placeholder-pool-activation`，HEAD is at the R26/R27 wrap-up commit
(2026-07-06). Working tree should be clean. The wrap-up captures R26/R27
findings + memory entry `r26-r27-3b-speedup-2026-07-06.md`.

## 硬约束（务必遵守）

- 加速**只**来自更多复用，不准加 KV-cache 调度。
- L3 MiniLM 语义 k-NN **默认 OFF**（research only，已弃用）。新 feature 默认 OFF。
- 实验结果统一输出到项目 `results/` 子目录，**不用 /tmp**。
- >3 case 必须加 `--disable-overlap-schedule --max-running-requests 1`（绕过
  `_delete_leaf` 竞态 bug）；`--force-evict` **不是**真实 server flag（旧文档写错了）。
- 不要重新 track `swebench_local_envs/`（21G）或 `results/codebase_kv/`（1.2GB/run）。
- commit/push **只在用户明确要求时**；commit message 结尾加
  `Co-Authored-By: Claude <noreply@anthropic.com>`。
- 不要打印/外泄 SiliconFlow API key（classifier 强制；连 `key[:N]` 都算凭证物化，
  只在进程内读）。

## R26/R27 关键发现 (2026-07-06)

- **3B × 3 agents gives ~2× speedup** (R26 2.014×, R27 1.900× vs R19 BEST 1.29×).
  Mechanism: smaller KV footprint → 4.8× more c2_chunk reuse (2886 vs 600 tok).
- **Counterintuitive: R27 (3B-Coder) is WORST at FAIL detection (0% FAIL_acc)**.
  Coder training biases toward PASS — general Instruct is more honest.
- **Decision matrix**: R26 = speed-first; R19 = accuracy-first; R27 = avoid for critique.
- **Lossy doesn't degrade accuracy** in any model tested (R19/R26/R27 all show
  lossy FAIL_acc >= lossless).

## 必读文件（按顺序）

1. **`CANONICAL_TARGET.md`** — 单一项目目标 + 当前状态（SINGLE SOURCE OF TRUTH）。
2. **`HANDOFF.md`** — 当前 session/branch 状态、bug 细节、open items、文件清单。
3. **`results/SESSION_WRAP.md`** — R19 / R26 / R27 3-way comparison (post-2026-07-06)。
4. **`results/CODE_AWARE_LOSSY_KV_PROGRESS.md`** — 完整开发时间线 + 结果表 + 已证
   fundamental limit（诚实数字，含已撤回声明标注）。
5. auto-loaded memory index `~/.claude/projects/-home-gfy/memory/MEMORY.md`
   有关键不变量；尤其读 `r26-r27-3b-speedup-2026-07-06`、`r25-oracle-8pct-unk`、
   `c2-cacheblend-lossy-not-safe`、`multi-slot-copy-2026-07-01`。

## 当前状态（2026-07-06，诚实）

迭代过 6 个复用机制（L3 MiniLM → L4 AST chunk → cross-position slot_id 修复 →
C2 CacheBlend gap-prefill → MULTI_SLOT copy → PRECOMPUTE pipeline）。
**R26 速度最优 (2.014×), R19 准确度最优 (60% FAIL_acc)**。
R27 (3B-Coder × 3) 警示：Coder 训练在 verdict 任务上反而有害。

| 机制 | 复用 (7B, full-share) | p50 TTFT | F1 vs lossless | 状态 |
|---|---|---|---|---|
| lossless（无复用） | 0 tok | 932 ms | 1.000 | 参考 |
| single-slot staged (1 slot ≈1400 tok) | ~1300 tok | ~820 ms | 0.461 | valid-but-different |
| **MULTI_SLOT (5 slots ≈7100 tok)** | ~7100 tok | **124 ms (7.5×)** | **0.000** | garbage |
| precompute SYNC（CPU host pool） | ~870 tok | 923 ms | 0.374 | lossy，无速度提升 |
| precompute LAYERED（load_stream） | ~870 tok | 918 ms | 0.508 | lossy + Phase 7 调查见下 |
| precompute DEVICE-RESIDENT（诊断） | ~830 tok | 960 ms | 0.447 | lossy，**比 lossless 还慢** |

**giant-codebase 大 benchmark 真实数字**（`bench_giant_codebase_reuse.py`，
5 case × 5 agent，fair A/B，`giant_5_fair_*`）：
- lossless reuser(2-5) TTFT 836ms / cached 1412
- C2 (code-aware) reuser 399ms / cached 3793 = **2.10×**
- l3 (MiniLM, 弃用) 113ms / 6183
- ⚠️ 表里 F1 列默认 1.0 是占位（in-run 无 baseline），**非真实精度**；
  C2 是 raw copy+RoPE，按 fundamental limit 大量复用 lossy → 2.10× ≠ 又快又准。

### Precompute pipeline（Phases 1-3, 4B/4C，commit `628aeab83`）

- **完整端到端管线**：离线 AST-aware KV 预计算 → CPU host pool → 启动时
  disk→CPU loader → task 时 CPU→GPU 复用（支持 `location="device"/"host"/"disk"`
  residency branching）。
- **验证结果**：5 case × 5 agent 真实 F1=0.374（**LOSSY**，符合 fundamental limit；
  canonical-prefix preamble 只帮 preamble 那部分无损，复用的是文件内容），**无速度提升**
  （TTFT ≈ lossless，sync CPU→GPU 传输抵消了 prefill 节省）。
- **Phase 6 诊断**：device-resident（KV 直接在 GPU 上，零 H2D）反而 **比 lossless 还慢**
  （960ms vs 948ms）→ **传输不是瓶颈**，read-path 本身（move_kv_cache + RoPE + alloc）
  在 ~867-tok 复用规模上开销抵消了节省。
- **Phase 7 LAYERED F1=0.508 调查**：7-way A/B 跑了 3 个"让 SYNC 表现得像 LAYERED"
  实验 + agent_count=1 ablation + 1×1 byte-cmp dump。**3 个围栏假设全部 FALSIFIED**
  （default-stream race / per-layer event-wait pattern / record_stream collision）。
  `agent_count=1` 下 SYNC=0.369 / LAYERED=0.559，gap 仍然存在。**结论**：LAYERED F1
  是真实但 load_stream + LayerDoneCounter 副作用，**不是 transferable correctness 改进**。
  就算修到 0.55 也仍远低于 1.000。**决策：不追了。**
- 完整报告 `results/kvcomm_ab/precompute_ab_report/ANOMALY_FINAL.md`。

### 已证 fundamental limit：cross-context KV loss

raw copy + RoPE 把 KV 搬到不同 prefix 下是 **lossy** 的。layer>0 的 KV 编码了它
前面的 prefix；segment 在新 prefix 下被复用时，拷过去的 KV 是 stale 的。RoPE 只修
位置，不修内容条件。loss 随复用量单调增加：~1400 tok → F1 0.46；~7100 tok → 0.00；
precompute ~870 tok → 0.374。**这不是 chunking/copy 机制的问题，是 raw-copy+RoPE 的
根本限制。**

### 已撤回声明（不要再 cite）

- L4 "~1.49× production-ready" — 坏的 over-copying 路径。
- AST-gated L3 "1.448× both bars met" — cached_tokens 混淆了 radix prefix 与
  code-aware 复用（见 `fair-measurement-prefix-conflation-2026-06-30`）。
- 旧 "1.31× / 20% reuse" — 已弃用 MiniLM 语义路径。
- **LAYERED F1=0.508**（precompute cycle）— Phase 7 调查证实非 transferable 修正，
  仅作 honest 现象记录。

## 4-layer cache + precompute（诚实版）

- **L1** Radix 前缀（token 级字节精确，同位置）— 唯一安全复用，baseline。
- **L2** 整 slot 字节精确 + RoPE（跨位置）— 已实现；大量复用 lossy。
- **L3** Placeholder k-NN body（MiniLM 语义，cos≥0.85）— **DEPRECATED**。
- **L4** AST 边界 chunk（按函数/类字节精确）— 已实现；partial-share 有精度优势，
  无速度优势。
- **C2 / MULTI_SLOT** CacheBlend gap-prefill + 多 slot 批量拷贝 — 已实现；
  7.5× 速度但 F1=0.000（lossy）。
- **PRECOMPUTE** 离线 AST-aware KV 预计算 → CPU host pool → async CPU→GPU 复用 —
  已实现端到端（commit `628aeab83`），**默认 OFF**，F1=0.374 无速度提升；已验证
  是 true CacheBlend 的前置构件。

## giant-codebase benchmark 设计

- **任务来源**：HuggingFace `SWE-bench/SWE-smith` 数据集，过滤 pandas-dev/pandas，
  1000 条真实 bug-fix 任务
  （`results/giant_codebase/tasks/pandas__pandas__1000/manifest.jsonl`）。
  加载器 `benchmark/multi_workflow/swesmith_pandas_loader.py`。
- **代码来源**：真实 `pandas-dev/pandas` git checkout
  （`results/giant_codebase/pandas_src/`，remote github.com/pandas-dev/pandas，
  commit `95280573e1`，1502 个 .py 文件）。
- **一个 case** = patch 涉及文件 + 同目录 sibling（`--sibling-window 4`）→ 切 5 个
  code_base slot（每文件截 8000 字符）；`slot_id = code_base:<file path>`
  （内容派生，非位置）。
- **多 agent**：5 角色链顺序跑 `implementer→debugger→reviewer→verifier→auditor`；
  agent 1=source（KV 入池），agent 2..5=reuser（不同 `cache_salt` → radix 冷 →
  靠 KVCOMM 跨位置 copy+RoPE）；`## Upstream context` 累加「{role} observed N
  cached tokens」交接（看不到上游输出）。
- **多任务共享 KV**：池 key 内容派生 → 同文件跨 task/跨 agent 共享；`cache_salt`
  只隔离 radix 前缀，不隔离 code-aware 池；server 全程不重启，池跨 case 累积。
- **命中 (reuse hit) = n/16**：4 case × 4 reuser = 16 机会，n = 命中池（复用非零
  KV）的 agent 数（`placeholder_anchor_pool_hit_count > 0`）。

## 唯一出路（待用户 sign-off，尚未实现）

**True CacheBlend**：对每个拷贝的 chunk，在**新 context 下重算 attention**
（而非 raw-copy + RoPE）。这是唯一能同时拿速度与精度的机制。代价高，**未实现**，
不要主动开工，等用户明确指示。Precompute 已建好，是其前置构件。

## 关键文件

- 主视觉 deck：`results/CODE_AWARE_LOSSY_KV_PROGRESS.html`（16 页横向 scroll-snap）
  + `results/CODE_AWARE_LOSSY_KV_PROGRESS.pdf`（16 页 PDF）+ 同名 `.md`。
  - 翻页 JS 外链 `results/CODE_AWARE_LOSSY_KV_PROGRESS.js`（IO + 滚轮离散翻页 +
    CSS scroll-timeline 进度条）。
  - PDF 生成脚本 `results/kvcomm_ab/gen_deck_pdf.py`（Playwright headless Chromium，
    `@media print` 纵向堆叠，页 1280×1600px）。
- 真实 prompt 样例：`results/kvcomm_ab/PROMPT_SAMPLE.md`
  （`gen_prompt_sample.py` 用驱动器真实函数 + 真实 manifest/pandas 源码 + 真实 7B
  输出重建，未起服务器）。
- 最近一轮报告：`results/kvcomm_ab/precompute_ab_report/ANOMALY_FINAL.md`
  （Phase 7 LAYERED F1 调查完整报告）+ `COMPARISON.txt`（7-way A/B 表）
  + `SUMMARY.txt`（5-way 表）+ 8 个 `run_7b_precompute_ab_*.sh` launcher。
- Precompute 管线：
  - `scripts/precompute_codebase_kv.py`（离线 KV 抽取）
  - `python/sglang/srt/mem_cache/codebase_kv_loader.py`（server-start disk→CPU loader）
  - `python/sglang/srt/mem_cache/test_codebase_kv_loader.py`（单元测试）
  - 修改：`radix_cache.py`（`ChunkKVEntry`/`AnchorKVEntry` 加 `location` 字段 +
    `_load_host_chunks_to_device` 分支 + Phase 7 围栏 hook，默认 OFF）、
    `hiradix_cache.py`（启动时 loader 触发）、`scheduler.py`（`hicache_consumer_index`
    配合 producer_id）、`benchmark/multi_workflow/{bench_giant_codebase_reuse.py,
    bench_kvcomm_ttft_stress.py,analyze_fair_ab.py}`（CLI flag + 公平 A/B 分解）。
- 驱动器：`benchmark/multi_workflow/bench_giant_codebase_reuse.py`（giant）、
  `bench_kvcomm_ttft_stress.py`（`build_slot_messages`/`make_payload`/`AGENT_ROLES`）、
  `swesmith_pandas_loader.py`、`analyze_fair_ab.py`（fair A/B 分析）。
- 代码：`python/sglang/srt/mem_cache/radix_cache.py`（L1/L2/L3/L4/C2/MULTI_SLOT/PRECOMPUTE）、
  `hiradix_cache.py`、`ast_chunker.py`、`codebase_kv_loader.py`。
- 硅基流动生图 MCP：脚本 `/home/gfy/MCP/siliconflow_mcp_server.py`，key 在
  `~/.claude.json`（有效，余额 5.77），FLUX/SDXL "Model disabled" → 用
  `Kwai-Kolors/Kolors`；MCP 工具需重启会话才加载，本会话可直接调 API
  （`results/kvcomm_ab/gen_alg_images.py`）。Kolors 文字乱码，精确示意图用手写 SVG。

## Branch 状态

`fix/placeholder-pool-activation`，HEAD `628aeab83`（precompute + Phase 7
完整 commit，33 files / +4713 / -474）。working tree 有未提交改动：

- 10 个旧 status doc 删除（KVFLOW_OVERVIEW.md, PHASE2_*.md, PLACEHOLDER_KNN_STATUS.md,
  SESSION_HANDOFF_2026-06-23.md, docs/experiment_plan.md, docs/kvflow_priority_fix_progress.md,
  results/HANDOFF_2026-06-04.md, results/PROJECT_STATE.md,
  results/contribution_summary_20260629.html）
- 未 track 原始结果目录：`results/diag_*`、`results/fair_smoke_*`、
  `results/kvcomm_ab/{7b_*,l2_*,l4_*,lossless,ps*}`（每跑 879MB）
- 新增本文档 + HANDOFF.md + CANONICAL_TARGET.md + 进度 doc 增量更新

**未主动 commit，等用户指示 cleanup commit 形状。**

## 开始时做什么

1. 读上述必读文件确认状态没漂移。
2. 等用户给本轮具体任务。常见方向：
   - **(A) Precompute 异步 overlap 优化**（plan Phase 4A-REVISED，~30 GPU-min，
     需用户 sign-off）— 用 SGLang 的 `LayerDoneCounter` 机制让 CPU→GPU H2D 与 prefill
     attention 在 `load_stream` 上逐层重叠。这是 precompute 后下一个具体速度杠杆，
     仍只动速度不动精度。
   - **(B) True CacheBlend 实现**（需用户 sign-off）— 唯一能同时达两个 bar 的路。
     precompute 是其前置（已建好）。
   - **(C) partial-share niche 深挖** — AST 在该 regime F1 0.62 > L2 0.51 已达标精度，
     速度 0.96× 未达标；可做 niche 产品化。
   - **(D) deck/文档/汇报材料完善**（PDF、prompt 样例、算法图）。
   - **(E) 重新跑一个 fair 的 giant-codebase A/B** 拿真实 F1（现 2.10× 的 F1 是占位）。
   - **(F) Cleanup commit**（10 个旧 doc 删除 + 未 track 原始结果）— 等用户确认形状。
3. 不要主动改算法/起服务器/commit，除非用户明确要求。
