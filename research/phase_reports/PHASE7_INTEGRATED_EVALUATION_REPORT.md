# Phase 7 正式研究报告：集成评测、R0 Ceiling 与 Workflow Scheduler 描述性结果

> 报告类型：正式阶段研究报告（自包含、可审计）
> 覆盖阶段：Phase 7（V7 计划 → 22 个 primary starts + 1 个 evidence correction → 双模型 review → final disposition）
> 撰写时间：2026-07-28
> 报告状态：`最终权威`
> 最终判定：`engineering = VALID` / `r0_mechanism = NEGATIVE` / `w_system_behaviour = INCONCLUSIVE-DESCRIPTIVE` / `publication = READY WITH CAVEATS`
> 关联报告：[Phase4](PHASE4_RECOVERY_METHODS_REPORT.md)｜[Phase5](PHASE5_WORKFLOW_SCHEDULING_REPORT.md)｜[Phase6](PHASE6_CROSS_STORE_SUBSTRATE_REPORT.md)｜[跨阶段总报告](PHASE4_TO_PHASE7_SUMMARY.md)

---

## 0. 引用约定

| 前缀 | 含义 | 绝对根路径 |
| --- | --- | --- |
| `docs:` | 文档仓库（本报告所在仓库） | `/home/chris/Workspaces/code-agent-kvcache` |
| `impl:` | 实现/结果仓库（cross-store-substrate worktree） | `/home/chris/Workspaces/kvcache-research/worktrees/cross-store-substrate` |
| `p7:` | Phase7 结果目录 | `impl:benchmark/approx_kv/results/phase7/` |

状态标签：`最终权威` / `历史/已被替代` / `diagnostic/proxy`。

---

## 1. 文档定位、证据状态与 Executive Summary

### 1.1 文档定位

Phase 7 是本项目**第一个完整走完「预注册 → 授权 → 执行 → 双模型 review → 证据补正 → 重新合并 → final disposition」全流程**的阶段。它在 Phase6 底座之上，用 chunk4096（而非 Phase4 的 chunk1024）重新测量 R0 ceiling，并在同一底座上跑 workflow scheduler 的 S0/S4 对照。

本报告是 Phase7 的正式研究报告，同时是本轮研究的收官报告。

### 1.2 证据状态总览

| 证据源 | 状态 | 说明 |
| --- | --- | --- |
| `p7:PHASE7_FINAL_DISPOSITION.json` | **`最终权威`** | `disposition_sha256=4013f054751f44e222ab698d0d232bc6fe516feb29111fb4c64f53b75f797d66`；file SHA=`731ace74108bc6ec0b6beb160bb3a55fcdbaae2fd2959c1a3f3d0b7aff466af5` |
| `p7:phase7-consolidated-summary.json` | **`最终权威`** | `summary_sha256=9d0aafcd776305c7ac679b14845e4a5c79e44c04bb696ecf2d83644fca4c2c69`；file SHA=`05acb52d51ea68bcff95c392e13348f73292dad0921b154efaf1e02e150eb135` |
| `p7:RESULT_MANIFEST.json` | **`最终权威`** | `files` 共 **88** 条，`known_gaps=[]`，file SHA=`6b6b0af19b70b9958866dc9674102026b348b1d5a4254e7f2f9d732a77548a65` |
| `p7:phase7-primary-manifest.json`（rev12） | `最终权威` | `status=authorized`，self=`2d66a1bc…`，design=`50003145…` |
| `p7:raw/*.json`（22 份） | `最终权威`（不可变） | 原 22 个 primary run 的 raw；**correction 后仍保持字节不变** |
| `p7:compact/*.json`（22 份） | `最终权威` | correction 后 `--force` 全量重生成 |
| `p7:logs/*.log`（22 份） | `最终权威`（不可变） | 服务端日志 |
| `p7:phase7-runs.jsonl` | `最终权威` | 中央 run 日志，sha256=`3655a374a554161a1c52a2ac1ce23a3cc2fc321c9d92772b2193aae60697ea67` |
| `p7:capacity-correction/raw/p6delta-s0-rho2-chunk4096-r0-terminal-reason.json` | `最终权威`（post-hoc） | correction raw sha256=`59049306019351d2100565be3653d74e4ebe60a1fb49d1e03016e20c9cce7534` |
| `p7:reviews/*.txt`（8 份） | `最终权威` | Sol / Opus 的 original、cross-consolidation、targeted-delta、final-verify、publication-ready |
| `p7:phase7-final-opus-review.json` | `最终权威` | Entry review，artifact SHA=`2c83f7ddec41d65a937a6b71cab851230c0fb06ff725f09cb36dd596e81e8649` |
| `p7:evidence/*.json`（4 份） | `最终权威` | 版本化的 Docker CPU test 证据 |
| `docs:IMPLEMENTATION_PLAN_LATEST.md`（V7，byte-frozen） | `最终权威` | plan of record |
| `docs:IMPLEMENTATION_PLAN_V4/V5/V6_ARCHIVED.md` | `历史/已被替代` | 计划演进 |

### 1.3 Executive Summary

1. **执行规模**：`22` 个 primary server starts，`1.310141803888889` GPU-equivalent 小时（wall-clock span `1.528835326388889` h）；硬上限为 `36 starts / 6 GPUh`，实际用掉 `21.8%`。另有 `1` 个独立计账的 evidence correction start，`0.09833181611111111` h。

2. **R0 机制在 chunk4096 下是 `NEGATIVE`**。四个 A8 restart-0 setting 的 paired request-path median 全部低于 `1.0`：

   | body | rho | request-path median | N8 full-setup | N8 incremental |
   | ---: | ---: | ---: | ---: | ---: |
   | 1024 | 1.5 | `0.7723084788319753` | `0.6086457910880934` | `0.6838342404592734` |
   | 1024 | 2.0 | `0.7750652993475325` | `0.6100955039216343` | `0.6854804712027397` |
   | 2048 | 1.5 | `0.9333835627802327` | `0.6397908630435616` | `0.7607318873368634` |
   | 2048 | 2.0 | `0.9361732730155323` | `0.6418893640462963` | `0.7640829718628741` |

   全部未达预注册 MDE（`max(5%, 2×sample_sd)`，`mde_fraction=0.05`），触发停止规则 `ES-R0-MDE`，跳过 8 个 supplement starts。**不得发布任何 R0 speedup headline。**

3. **chunk1024 sensitivity 证实了 Phase4 的 chunk confound**：同一 body2048/rho2/S0 配置在 chunk1024 下 request-path median = `1.7370152775837997`、N8 full-setup median = `1.1889604660811057`。artifact 显式标注 `headline=false`，`interpretation="chunk-coupled sensitivity diagnostic; not a mechanism-intrinsic headline"`。

4. **W scheduler 矩阵（12 starts）结论为 `INCONCLUSIVE/DESCRIPTIVE`**：

   | rho | all-reusable mean（S0/S4） | workflow-only mean | miss delta（S4−S0，all/workflow） | peak ratio（S4/S0） |
   | ---: | ---: | ---: | ---: | ---: |
   | 1.5 | `1.0020817511131008` | `1.0275736602204162` | `-14` / `-10` | `1.01122824527478` |
   | 2.0 | `1.0250294118270489` | `1.044182842048327` | `+2` / `-10` | `0.9980643600290346` |

   比较为 **seed-matched 但非相邻 launch block**（`not_a_paired_launch_block=true`），且 R0 臂中 `45.9%–72.1%` 的请求走 dense fallback。

5. **R4-like 是 synthetic 5x footprint proxy，绝不执行 KVCOMM**：S0 臂 122 个请求中只有 `12` 次真实 recovery（rate `0.09836065573770492`），`110` 次 dense fallback（rate `0.9016393442622951`，terminal reason 全为 `unsupported`）；S4 臂 `diagnostic_unavailable`（第 5 个 representation 部分 registration 失败）。

6. **一次证据补正（correction）**：wave-0 的 S0 cell 缺失强制 terminal reason（`approximate_recovery_failed_dense=40` 但 `terminal_reason_counts={}`），无法离线恢复，因此单独跑了 1 个 supplementary start，得到 `40/40` 全部为 direct exclusive `unsupported <- store_miss`。**原 22 份 raw/log 保持字节不变。**

7. **治理与 provenance**：`RESULT_MANIFEST` 从 `48`（Phase6）→ `5/5`（rev12 授权时）→ `75/75` → `79/79`（重新合并后）→ **`88/88`**（publication package 封版），`known_gaps=[]`。

8. **最终 disposition**：`engineering=VALID`、`r0_mechanism=NEGATIVE`、`w_system_behaviour=INCONCLUSIVE/DESCRIPTIVE`、`publication=READY WITH CAVEATS`，open P0/P1 = `0/0`。

9. **自然压力下的 reservation-failure fallback 仍未证明**：Phase7 观察到 **0 次**自然 reservation failure，Phase6 的 fault-injected-only caveat 原样保留。

---

## 2. Phase 7 动机、研究问题、冻结假设与非目标

### 2.1 动机

Phase4 得到的 `1.5x–2.0x` target-only / request-path 收益已被 CL2 证明与 `chunk=max-prefill=1024` 强耦合，而 CL1 的冻结 promotion 规则又给出 `practical family = NONE`。Phase6 建立了 exact/approximate 共预算的底座并以 `PASS WITH CAVEATS` 通过。

Phase7 的任务因此收窄为两件事：

1. **在 primary chunk = 4096 下重新建立（或撤回）R0 ceiling 的收益**；
2. **在真实 cross-store 底座上，用 fixed sequential workflow 对照 S0 与 S4**。

### 2.2 研究问题

| 编号 | 研究问题 | 结论位置 |
| --- | --- | --- |
| RQ7-1 | 在 chunk4096 下，R0（speed-only ceiling）的 paired request-path speedup 是否超过预注册 MDE？ | §4.6、§7.1 |
| RQ7-2 | 摊销到 N=8 次复用后，R0 是否转正？ | §4.6 |
| RQ7-3 | Phase4 的收益有多少可归因于 chunk 配置？ | §4.7 |
| RQ7-4 | 在含 approximate 臂的真实 workflow 中，S4 相对 S0 是否仍有可观察的差异？ | §4.8 |
| RQ7-5 | 5x 表示多重性（R4-like footprint）在 chunk4096 下的可达性与 victim 行为如何？ | §4.9 |
| RQ7-6 | 整套证据链能否在字节级被独立复核？ | §5.5、§9.3 |

### 2.3 预注册与 Entry Gate（V7 §8.1、§15.1）

**Phase6 通过本身不授权 Phase7。** 五个 Entry blocker：

1. Ceiling runner 未实现 → `run_p7_ceiling.py`；
2. Scheduler runner 未实现 → `run_p7_scheduler.py`；
3. R2 strategy `pending` → 须选定 `adapter` 或 `disabled_not_comparable`；
4. Implementation pin 未生成 → 自 manifest rev6 起 pin；
5. User authorization 未获得 → 需用户明确授权。

Entry 全部条件：Phase6 technical Exit = `PASS WITH CAVEATS`；V7 plan 最终 Opus review 完成；`PROJECT.md`/`HANDOFF.md` 标记 Current/Latest；primary manifest 已 pin/提交/hash 验证；两个 runner 的 Docker CPU tests + targeted review 通过；R2 = `disabled_not_comparable`；用户条件性授权生效。

**P7-0b chunk4096 feasibility gate 明确不属于 Entry 前 Gate，是授权后的 wave-0 内容。**

### 2.4 冻结假设

| 项 | 值 |
| --- | --- |
| primary chunk | `4096` |
| sensitivity chunk | `1024`（非 headline） |
| MDE | `max(5%, 2×sample_sd)`，`mde_fraction=0.05`，**在执行前冻结** |
| 停止规则 | `ES-R0-MDE`：restart-0 未达 MDE 则跳过 supplement starts |
| 独立复制单元 | `server restart` |
| 预算硬上限 | `36 server starts / 6 GPUh` |
| host / prefetch / async 轨道预算 | `0` |
| 并行度 | 本机只有一张 SM75，全部 GPU 任务全局串行 |
| runtime 输出 | 只写 `/results/phase7` staging；implementation worktree 在 Docker 中只读 |

### 2.5 非目标

- **不做** practical recovery 的 promotion（`practical family = NONE` 已定稿，PR-S0/PR-S4 不生成）。
- **不做** R2 GPU cell（`disabled_not_comparable`）。
- **不做** host / prefetch / async / HiCache 轨道（H4、RH4 不生成）。
- **不做** semantic correctness / accuracy 评估。
- **不做** 与 Phase4 chunk1024 结果的合并统计或排名。
- **不做** Phase8 的自动触发。

---

## 3. 环境、实现范围、方法与测量口径

### 3.1 执行环境（Docker 内执行）

`phase7-consolidated-summary.json.environment`（逐字段）：

| 项目 | 值 |
| --- | --- |
| `image_digest` | `sha256:0be6e16e2eb288dfd5fa8b0b41015f61731a139fb961d3366ccedf834289d781` |
| `model` | `Qwen/Qwen3-0.6B` |
| `model_revision` | `c1899de289a04d12100db370d81485cdf75e47ca` |
| `tokenizer_revision` | `c1899de289a04d12100db370d81485cdf75e47ca` |
| `chat_template_revision` | `model-revision-bound` |
| `gpu` | `NVIDIA GeForce RTX 2080 SUPER, SM75, 8192 MiB` |
| `driver` | `580.173.02` |
| `container_flags` | `--runtime=nvidia --gpus all --ipc=host --shm-size=8g --user 1000:1000` |

### 3.2 实现范围与 code pin

| 项 | 值 |
| --- | --- |
| primary execution code pin | `81405f4278b034911bc613c4ee17c79d15ee8f35`（tree `4a8bf0f6cd72ec09859f8301dc92b6aabb25d2bf`） |
| correction code pin | `a950ab914e6b029f86d8ef666a3f770fc42980a7` |
| runners | `run_p7_ceiling.py`、`run_p7_scheduler.py`、`run_p6_4_capacity_pilot.py`（三者共享同一授权门） |
| consolidator | `impl:benchmark/approx_kv/consolidate_phase7_results.py` + `test/registered/unit/bench/test_consolidate_phase7_results.py` |
| Phase7 模块 | `impl:benchmark/approx_kv/phase7/{common,correction,correction_review,evidence,review,statistics}.py` |
| CPU evidence HEAD | `a01642fdd6cdc869cb2c991c23003ff600665f37` |
| summary-binding commit | `107f6bfd39f25506929327a146377dafd964db6b` |
| final publication branch HEAD | `0206f17b4255e4b248dafaaeb943be57428dae2f` |

Docker CPU 证据（版本化于 `p7:evidence/`）：

| 套件 | 结果 |
| --- | --- |
| full targeted | `269 passed + 22 subtests` |
| capacity | `24 passed` |
| ceiling | `55 passed` |
| scheduler | `12 passed` |
| correction capacity | `36 passed` |

### 3.3 arm 定义与 paired launch block

同一 server 内三臂（V7 §5.3）：

- **D0**：无 reuse metadata 的 dense arm；
- **E0**：exact-cache arm；
- **R0**：approximate recovery arm；
- 三臂共享 server argv / plugin env / filler manifest / capacity 与 rho 目标；每臂之间完整 reset，arm 顺序按 formal repeat 交替。

`paired launch block` 定义：同一 `(body, rho, restart)` 下，以相同 image/model/capacity 目标/server-seed 计划**连续启动**的一组相邻 server 进程。eviction policy、HiCache、chunked-prefill 或 capacity 不同均需独立 server 进程。

> **W 矩阵的 S0 与 S4 实际上不是相邻 launch block**（S0 的三次 restart 连续跑完后才跑 S4），因此只能称为 `seed-matched_non_adjacent_restart_comparison`（见 §4.8 与 §5.3）。

### 3.4 测量口径

| 指标 | 定义 |
| --- | --- |
| `request_path` | `seed_head_ms + target_only_ms`；**这是预注册的 MDE 指标**（`is_preregistered_mde_metric=true`） |
| `target_only` | 排除 `seed_head_ms`；非 MDE 指标 |
| `full_lifecycle` | 单位为 formal repeat；含 source preparation |
| `speedup_N`（full-setup） | `dense_total_N / (source_preparation + Σ request_path_i)`，N ∈ {1,2,4,8}，**全部来自实际累计，不插值不外推** |
| `speedup_incremental_N` | 用 `incremental_setup = recovery_source_preparation − dense_source_materialization` 替代 full setup |
| `break_even` | `speedup_N > 1` 的第一个实测 N；N≤8 未观察到时写 `>8/not_observed` |
| `ratio_of_marginal_p95s` | **不是配对统计量**；`p95_pairing="nonpaired"` |
| `arm_interval_peak_device_bytes` | 自上次完整 reset 起的本臂 high-water |

### 3.5 cache outcome 与 terminal reason（V7 §5.8）

每个请求只归入六类之一：`dense_no_reuse_baseline` / `exact_gpu_hit` / `ordinary_exact_cache_miss` / `approximate_gpu_recovery` / `host_demand_load` / `approximate_recovery_failed_dense`。

approximate 失败**有且只有一个** exclusive terminal reason：`cross_store_reservation_failed` / `device_allocation_failed` / `unsupported` / `registration_failed` / `prefix_gap`。

---

## 4. 全部实验：计划演化、矩阵、执行顺序、核心数值

### 4.1 计划演化 V4 → V5 → V6 → V7

| 版本 | 状态 / commit | 关键内容 | 被替代原因 |
| --- | --- | --- | --- |
| V4 | `docs:IMPLEMENTATION_PLAN_V4_ARCHIVED.md` | 以 Phase6 Exit = `PASS WITH CAVEATS` 为基线 | 转为 result-bound 的 V5 draft |
| V5 | commit `d314cc4143b6d9ceffa08240fc13139c382c4529`，SHA-256 `ba6aec34ed5f333fb…` | 冻结 R0 primary / 条件 R2 / R4-like diagnostic / 16 settings / 33 starts 上界；full review 两位 reviewer 均 FAIL → 8 个 P0 关闭 → MDE=5% 冻结 → primary manifest rev4 PASS | bounded feasibility 证明恢复 R2 须改核心 dispatch，故 V6 把 R2 解析为 `disabled_not_comparable` 并删除 R2 GPU cell |
| V6 | commit `14a573eb942742fddeba372fea03326b5d6c251a`，SHA-256 `86e25989e7b36bd02cc22749835be220062067ab…` | runner 实现、R2 disposition、source pin、13 committed / 30 starts（含条件 14/31）、final code pin `5d9a5793d73121f088890aa6c02cfebc31cd97be` | final Opus review = `FAIL`（1 P0 / 3 P1） |
| **V7（现行）** | plan commit `c80ec165713772c533e17ef4c50f083c36dc9d72`；final code pin `81405f4278b034911bc613c4ee17c79d15ee8f35` | runtime staging(`/results/phase7`)、capacity runner 授权绑定单一 setting、CPU/review evidence 版本化内容绑定、plan byte-frozen、activation 移至 `PROJECT.md`/`HANDOFF.md` | — |

### 4.2 R2 bounded feasibility → `disabled_not_comparable`

- 用户在 2026-07-28T00:07:40 选定「方案 B」：实现两个 runner，对 R2 执行**有界（bounded）feasibility**，非侵入时实现 adapter。
- 结论（`docs:TRACKING.md` 2026-07-28T04:31:07）：历史 CacheBlend package 与 core hooks 已删除；恢复至少需要修改 scheduler/runtime dispatch 与 store lifecycle → 触发预定停止条件（V7 §15.2 第 23 条），选择 `disabled_not_comparable`。
- V7 中 R2 状态：Phase7 GPU settings = `0`；historical evidence 仅 Phase4 chunk1024；**不得与 Phase7 chunk4096 结果合并统计或排名**（V7 §8.4）。
- `r2_like` 是 S0 wave-0 中的**合成 2x footprint profile**，summary 字段原文：`"r2_like_semantics": "2x synthetic representation multiplicity footprint only; not R2 execution"`。

### 4.3 Manifest revision 链与授权

早期 revision 的 design hash 随设计修订而变化；**只有完成 V7 review 的
rev11 → authorized rev12 保持**
`50003145f2e7f0e866613dbd420e73ba3983a6c182a360d6918098b1d1f7b987`
不变：

| revision | 状态 | 关键字段 |
| ---: | --- | --- |
| rev6 | superseded | 因 `build_result_manifest.py` 陈旧的「runner 未实现」文字而被替代 |
| rev7 | `pinned_blocked` | self=`3eebdbfb…`，design=`ec5aaf59…` |
| rev9 | final Opus review-of-record | self=`5daa95a3…`，design=`ae8d8e1e…` |
| rev11 | V7 review-of-record | self=`48c86bf0f4df6f5c8baa41fc6871e3bcdfba59ea46f9022875f8453f2d5a5236`，design=`50003145…`，code pin=`81405f42…`；final Opus=`PASS WITH CAVEATS`，open P0/P1=`0/0` |
| **rev12** | **`authorized`** | self=`2d66a1bcdb6dc92a72c59fefc581212fcd541accbc8ededa221495d30d039bef`，design=`50003145…`（与 rev11 一致），blockers=[]，primary commit=`759476c901b1ac30eeeb96f83b9257586a103c4e`，envelope HEAD=`d42a5d1546680de458122f605f8a486b8faf5564`，`RESULT_MANIFEST=5/5` |

Entry review artifact：`p7:phase7-final-opus-review.json`，SHA=`2c83f7ddec41d65a937a6b71cab851230c0fb06ff725f09cb36dd596e81e8649`，containing commit=`b0837505abc4efe7df914c3b504693956fe71df9`。summary 中记录 `design_preserving_rev11_to_rev12=true`，解释为 *"rev12 activates authorization and supersedes reviewed rev11 without changing the reviewed design payload"*。

### 4.4 执行清单：22 primary starts + 1 correction

`phase7-consolidated-summary.json.execution.counts`：

```text
wave0_required             = 2
a8_primary_restart0        = 4
a8_primary_supplements_skipped = 8
chunk1024_sensitivity      = 2
w_main                     = 12
r4_diagnostic              = 2
rho3_conditional_disabled  = 1
executed_starts            = 22   (committed_manifest_starts = 30, conditional = 1)
```

预算对照（`execution.budget_comparison`）：

| 项 | 值 |
| --- | ---: |
| `actual_elapsed_gpu_equivalent_hours` | `1.310141803888889` |
| `actual_elapsed_seconds` | `4716.510494` |
| `wall_clock_span_hours` | `1.528835326388889` |
| `expected_gpu_hours_total` | `3.8` |
| `hard_cap_gpu_hours` / `hard_cap_starts` | `6` / `36` |
| `actual_fraction_of_hard_cap` | `0.21835696731481483` |
| `executed_minus_committed_starts` | `-8` |
| `within_hard_cap` | `true` |

执行顺序（`execution.executed`，按实际启动顺序）：

```text
 1. p6delta-s4-rho2-chunk4096            r0   (wave-0)
 2. p6delta-s0-rho2-chunk4096            r0   (wave-0)
 3. p7-a8-r0-body1024-rho1.5             r0
 4. p7-a8-r0-body1024-rho2.0             r0
 5. p7-a8-r0-body2048-rho1.5             r0
 6. p7-a8-r0-body2048-rho2.0             r0
 7. p7-a8-r0-body2048-rho2-chunk1024-sensitivity  r0
 8. p7-a8-r0-body2048-rho2-chunk1024-sensitivity  r1
 9-11. p7-w-r0-lru-rho1.5                r0,r1,r2
12-14. p7-w-r0-hierarchical-rho1.5       r0,r1,r2
15-17. p7-w-r0-lru-rho2.0                r0,r1,r2
18-20. p7-w-r0-hierarchical-rho2.0       r0,r1,r2
21. p7-w-r4like-lru-rho2                 r0
22. p7-w-r4like-hierarchical-rho2        r0
--- 独立计账 ---
23. p6delta-s0-rho2-chunk4096-r0-terminal-reason  (evidence correction)
```

> 从这份顺序可直接看出 W 的 S0（lru）与 S4（hierarchical）**不是相邻 launch block**：每个 policy 的三次 restart 连续跑完后才切换 policy。这正是 review 要求把比较标注为 `seed-matched_non_adjacent_restart_comparison` 的事实依据。

### 4.5 Wave-0：chunk4096 feasibility（P7-0b）

两个 cell 均来自 `p7:compact/p6delta-s{0,4}-rho2-chunk4096-r0.compact.json`：

| cell | policy | requested / observed capacity | `capacity_relative_error` | 顶层 status |
| --- | --- | --- | ---: | --- |
| `p6delta-s4-rho2-chunk4096` | S4 | `11392` tokens / `11392` tokens（`1,306,525,696` bytes，`11392` pages） | `0.0` | `diagnostic_unavailable` |
| `p6delta-s0-rho2-chunk4096` | S0 | 同上 | `0.0` | `diagnostic_unavailable` |

profile 级可达性（本报告直接读取 `key_metrics.cells[].profiles[]` 核对）：

| profile | representation kinds | S4 | S0 |
| --- | --- | --- | --- |
| `exact_only` | — | `reachable` | `reachable` |
| `r0_like` | `canonical_base` | `reachable` | `reachable` |
| `r1_like_k32` | `canonical_base`,`repair_state` | `reachable` | `reachable` |
| `r2_like` | `canonical_base`,`precomputed_adapter` | `reachable` | `reachable` |
| `r4_like` | `canonical_base`,`anchor`,`delta`,`anchor`,`delta` | **`diagnostic_unavailable`** | `reachable` |

双向 pressure（两 cell 均 `passed=true`）：

| cell | exact→approx victim bytes | approx→exact victim bytes | approx evicted | exact evicted |
| --- | ---: | ---: | ---: | ---: |
| S4 | `14,562,623,488` | `20,303,904,768` | `44,509,954,048` | `26,198,409,216` |
| S0 | `33,676,066,816` | `17,929,404,416` | `47,299,166,208` | `25,212,092,416` |

`fallback_reachability` 两 cell 均为 `{"passed": false, "rounds": 0}` —— **wave-0 未观察到任何自然 fallback 轮次**。

`inactive_counter_assertion` 四个必需 counter（`host_load`、`prefetch_request`、`prefetch_loaded_tokens`、`async_load`）均为 `manifest_pinned_disabled=true` + `verification="indirectly_verified"`（**不是显式 0**），符合 FINDING-CL1-B 的证据分级规则。

**必带 caveat：wave-0 的 registration reachability 不等于 approximate-recovery success。**

### 4.6 A8：R0 ceiling（chunk4096）→ `NEGATIVE`

`a8_ceiling` 关键字段：

| 字段 | 值 |
| --- | --- |
| `mechanism_status` | `NEGATIVE` |
| `mde_fraction` | `0.05` |
| `n_per_setting` / `independent_restarts_per_setting` | `1` / `1` |
| `independent_replicate_unit` | `server_restart` |
| `formal_repeats_are_not_independent_replicates` | `true` |
| `targets_are_not_independent_replicates` | `true` |
| `headline_speedup_allowed` | **`false`** |
| `three_restart_range.available` | `false`（reason = `ES-R0-MDE`） |
| `primary_supplement_disposition` | rule=`ES-R0-MDE`，skipped_starts=`8`，reason=*"restart-0 request-path medians did not reach the 5% MDE"* |

四个 setting 的核心数值：

| body | rho | request-path median | N1 full | N2 full | N4 full | N8 full | N8 incremental | break-even |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1024 | 1.5 | `0.7723084788319753` | `0.2453152592379244` | `0.37227554253479145` | `0.5026737512297415` | `0.6086457910880934` | `0.6838342404592734` | `>8/not_observed` |
| 1024 | 2.0 | `0.7750652993475325` | — | — | — | `0.6100955039216343` | `0.6854804712027397` | `>8/not_observed` |
| 2048 | 1.5 | `0.9333835627802327` | — | — | — | `0.6397908630435616` | `0.7607318873368634` | `>8/not_observed` |
| 2048 | 2.0 | `0.9361732730155323` | — | — | — | `0.6418893640462963` | `0.7640829718628741` | `>8/not_observed` |

每个 setting 的 outcome 计数完全符合设计：`approximate_gpu_recovery=16`、`dense_no_reuse_baseline=16`、`exact_gpu_hit=16`、`approximate_recovery_failed_dense=0`、`host_demand_load=0`、`ordinary_exact_cache_miss=0`。

body1024/rho1.5 的辅助数值（示例）：

- `request_path.paired_dense_over_recovery_ratio`：min `0.7621386872318755`，p50 `0.7723084788319753`，max `0.7789576170555851`，n=16；
- `request_path.paired_recovery_minus_dense_ms`：min `51.63878601160832`，p50 `53.31127547833603`，max `55.80541302333586`；
- `primary_views.cached_tokens`：D0=`0`、E0=`64`、R0=`1088`（全部 16 个样本一致）；
- `full_lifecycle.dense_over_recovery_ratio`：p50 `0.7771812139821177`（n=2，单位是 formal repeat）；
- `cold_start.server_cold_start_ms = 49048.97877399344`，`shared_by_all_arms=true`，`included_in_arm_latency_ratios=false`。

**canary 限制（必须保留）**：body1024 的 same-context canary `complete_8_tokens=true`、`matched=true`，但 `distinct_output_tokens=1`，`discriminative_power="limited"` —— **判别力有限**。

### 4.7 chunk1024 sensitivity（非 headline）

2 个 starts（body2048 / rho2 / S0，restart 0 与 1）：

| restart | request-path median | N1 full | N2 full | N4 full | N8 full | N8 incremental |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `1.7388709604881618` | `0.3713619199000795` | `0.612654789907146` | `0.9065432506504975` | `1.1914499917406536` | `1.4176819547967658` |
| 1 | `1.7351595946794376` | `0.369097224681946` | `0.6089369542946363` | `0.9012555896400676` | `1.1864709404215579` | `1.410947853600695` |
| **median** | **`1.7370152775837997`** | `0.3702295722910127` | `0.6107958721008911` | `0.9038994201452826` | **`1.1889604660811057`** | `1.4143149041987304` |

artifact 字段：`headline=false`，`interpretation="chunk-coupled sensitivity diagnostic; not a mechanism-intrinsic headline"`。

**解读：这组数字与同一 body2048/rho2 在 chunk4096 下的 `0.9362x` 形成直接对照，定量地重现了 Phase4 的 chunk confound。它只证明 chunk 耦合，不证明机制固有收益。**

### 4.8 W scheduler 矩阵（12 starts）

设计：S0(lru) / S4(hierarchical) × rho{1.5, 2.0} × 3 restart，R0 臂 + E0 臂。

`w_scheduler.interpretation` 的核心结论（逐字段）：

| rho | all-reusable mean（`s0_over_s4`） | workflow-only mean | miss delta（S4−S0） all / workflow | peak ratio（S4/S0） | comparison design |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1.5 | `1.0020817511131008` | `1.0275736602204162` | `-14` / `-10` | `1.01122824527478` | `seed-matched_non_adjacent_restart_comparison` |
| 2.0 | `1.0250294118270489` | `1.044182842048327` | `+2` / `-10` | `0.9980643600290346` | 同上 |

rho1.5 的 median-across-restarts 明细（all-reusable）：

| 臂 | requests | partial/full miss | ttft mean | ttft p50 | ttft p95 | wall clock |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| S0 R0 | 122 | `70` | `198.6089777630815` ms | `150.19555800245143` | `553.4587066897075` | `24236.202257248806` ms |
| S4 R0 | 122 | `56` | `197.79241161605106` ms | `153.22209300938994` | `534.7563851741143` | `24136.48086419562` ms |

workflow-only（rho1.5）：S0 miss `10` / S4 miss `0`；S0 ttft mean `365.71614290587604` ms vs S4 `354.6214382047765` ms。

per-role hit fraction（rho1.5，restart 0）：S0 的 architect/coder/debugger clamped hit fraction 分别为 `0.0588 / 0.0303 / 0.0588`，S4 均为 `1.0`；live_filler S0 `0.6204`、S4 相应更低。**即：S4 确实把 workflow 对象保住了，S0 没有。**

`ratio_of_marginal_p95s`（方向 `s4_over_s0`）：rho1.5 all-reusable `0.9663636722902451`、workflow-only `0.9647370876542719`；`p95_pairing="nonpaired"`。

**dense fallback 混杂（决定性限制）**：`latency_ratio_fallback_mix` 原文 *"W latency ratios mix approximate recoveries with 45.9%-72.1% dense fallback"*，`minimum_dense_fallback_rate=0.45901639344262296`，`maximum_dense_fallback_rate=0.7213114754098361`。

`within_policy_r0_vs_e0`（同一 policy 内 R0 与 E0 的对照，rho1.5 hierarchical，all-reusable）：

- `r0_over_e0_mean_latency_ratio` p50 = `1.514742225673491`（n=3）；
- `r0_over_e0_p50_latency_ratio` p50 = `1.8121045025287077`；
- `ratio_of_marginal_p95s` p50 = `1.2164065236022525`；
- `paired_delta_median_ms` p50 = `-66.55775650870055` ms；
- `interpretation`：**"R0 is slower than request-paired E0"**。

`victim_footprint.primary_axis_after_a8_negative = true`，理由原文：*"A8 is NEGATIVE, so scheduler victim/accounting and memory footprint are primary; latency ratios are secondary descriptive evidence"*。

`claim_rule` 原文：*"the preregistered rules do not permit a practical benefit claim; latency is secondary after A8 NEGATIVE"*。

### 4.9 R4-like diagnostic（2 starts）

**`not_kvcomm = true`，`performance_ranking_enabled = false`，`ranking = disabled`，`proxy = "R4-like-5x synthetic footprint proxy"`。**

| policy | status | requests | recovery | dense fallback | terminal reasons | registration_failed | arm peak device bytes |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: |
| S0 | `valid`（两 round 均 `available`） | `122` | `12`（rate `0.09836065573770492`） | `110`（rate `0.9016393442622951`） | `unsupported=110`，其余全 0 | `false` | `1,490,370,560` |
| S4 | `inconclusive`（两 round 均 `diagnostic_unavailable`） | `0` | `0` | `0` | 全 0 | **`true`** | `1,373,388,800` |

S0 的 memory footprint（每 round）：`approx_device_bytes=645,922,816`、`nonfree_resident_bytes=1,023,934,464`、`nonfree_resident_tokens=8928`、`exact_only_estimated_bytes=378,011,648`、`arm_interval_peak_device_bytes=1,490,370,560`、`reserved_device_bytes=0`。

**记账规则（`overlap_note` 原文）**：*"nonfree_resident_bytes already contains approx_device_bytes; exact_only_estimated_bytes = max(nonfree - approx_device, 0) and must not be added back to approx_device_bytes"*。

S0 的 churn bytes（`evicted_bytes + demoted_bytes`，不含 wasted）：两 round 各 `15,166,111,744`，合计 `30,332,223,488`。

S4 的 victim 序列显示：在 `workflow-01` 的 setup 阶段，approximate requester 连续驱逐了自身的 `anchor`（`234,881,024` bytes）与 `delta`（`645,922,816` bytes）等对象 —— 即 **5 个 representation 的第 5 个无法完成 registration**，导致 `diagnostic_unavailable`。

`representation_metadata` 每条均含 `{"arm_label":"R4-like-5x","executes_kvcomm":false,"performance_ranking_enabled":false,"profile":"r4_like","resident_multiplicity":5}`。

### 4.10 Evidence correction（第 23 个 run，独立计账）

- **触发**：Sol round-1 review 的 P1-1 —— wave-0 S0 cell 中 `approximate_recovery_failed_dense=40` 但 `terminal_reason_counts={}`，缺失强制的 exclusive terminal reason，**无法离线恢复**。
- **决策**：只重跑这 1 个 setting；原 22 个 raw/log 不可变；S4/A8/W/R4 不重跑。
- **执行**：`run_id = p7-capacity-correction-p6delta-s0-rho2-chunk4096-r0-20260729T022541822920Z`，`started_at=2026-07-29T02:25:41.822979+00:00`，`completed_at=2026-07-29T02:31:35.817517+00:00`，`elapsed_seconds=353.994538`。
- **结果**：`40/40` approximate replay 全部为 direct exclusive **`unsupported <- store_miss`**；无 reservation failure、无 ambiguity、reset 与 inactive counter 均通过。
- **provenance**：correction raw sha256=`59049306019351d2100565be3653d74e4ebe60a1fb49d1e03016e20c9cce7534`；`original_raw_sha256=80e8e83d7b587b1ed566889e1603eead82eb2618b58a2f9a1816fb8eae741ff3`（原 raw 保持字节不变）；correction manifest rev2 self=`0c57df5b75c48d53a11c1355d37f24fd360f15d2f2fb89afa88dfa6923b68283`；execution head `db1c4843fa08ace9b08200fc67d22ad2a7138ff8`，`worktree_clean=true`，post-pin changed paths 仅 4 个且全在 allowlist 内。
- **计账**：`classification="post_hoc_evidence_correction"`，`counts_against_authorized_22_starts=false`，`counts_against_preregistered_gpu_hour_budget=false`，`excluded_from_executed_starts=true`。

### 4.11 成功 / 失败 / 被跳过项汇总

| 项 | 状态 | 说明 |
| --- | --- | --- |
| wave-0（2 starts） | 成功 | capacity error = 0；reset/inactive/orphan 通过 |
| A8 restart-0（4 starts） | 成功但结论 `NEGATIVE` | 工程有效，机制未达 MDE |
| A8 supplements（restart 1–2） | **被跳过（8 starts）** | 预注册停止规则 `ES-R0-MDE` |
| chunk1024 sensitivity（2 starts） | 成功 | 非 headline |
| W 矩阵（12 starts） | 成功但结论 `INCONCLUSIVE/DESCRIPTIVE` | 非相邻 launch block + 大量 dense fallback |
| R4-like S0（1 start） | 成功（诊断） | 12/122 recovery |
| R4-like S4（1 start） | **`diagnostic_unavailable`** | 第 5 个 representation registration 失败 |
| rho3 conditional（1 start） | **被禁用** | 未执行 |
| R2 GPU cells | **0 个** | `disabled_not_comparable` |
| host / prefetch / async / HiCache 轨道 | **未生成** | V7 预算为 0 |
| 自然 reservation failure | **0 次观察** | Phase6 caveat 保留 |
| evidence correction（1 start） | 成功 | 40/40 direct `unsupported <- store_miss` |

---

## 5. 发现并修复的问题

### 5.1 V6 → V7：final Opus review 的 1 P0 / 3 P1

| 级别 | 问题 | 根因 | 修复 | 验证 |
| --- | --- | --- | --- | --- |
| P0 | runtime artifact 写入 repo，导致第二次 run 无法保持 clean envelope | runner 直接写实现 worktree | 改为只写 `/results/phase7` staging；implementation worktree 在 Docker 中只读 | 执行期间 `worktree_clean=true` |
| P1 | ceiling 的 required CPU 证据陈旧 | 证据未与 runner blob 内容绑定 | 版本化 JSON，绑定 Docker digest / exact command / exit code / summary | `p7:evidence/ceiling-cpu.json` 等 4 份 |
| P1 | P7-0b capacity runner 未接授权门 | runner 未共享 Phase7 gate | 加入 Phase7 authorized / single-setting / blob gate | CPU tests `24 passed`；correction 时 `36 passed` |
| P1 | plan 自改 Current/Latest 与 design hash 保持互斥，形成 review 循环 | plan 内含自身激活状态 | plan byte-frozen；activation 移至 `PROJECT.md`/`HANDOFF.md` | V7 plan blob 提交后未再修改 |

### 5.2 Round-1 result review 的 findings

**Sol（`p7:reviews/sol-original.txt`）**：Verdict = `FAIL`（证据合同问题，核心数值 PASS），P0=`0` / P1=`1` / P2=`3`。

| 编号 | 问题 |
| --- | --- |
| P1-1 | wave-0 S0 fallback 缺失强制 terminal reason（`approximate_recovery_failed_dense=40` 但 `terminal_reason_counts={}`） |
| P2-1 | `p95_ratio_s4_over_s0` 命名违反冻结的统计合同，应为 `ratio_of_marginal_p95s` |
| P2-2 | S0 / S4 非相邻 launch block |
| P2-3 | wave-0 未分 cell 级 / profile 级状态 |

**Opus（`p7:reviews/opus-original.txt`）**：Verdict = `PASS_WITH_CAVEATS`，Publication = `NOT_READY`，P0=`0` / P1=`5` / P2=`12`。全部数值逐字节可复现（22 个 compact 自哈希、22 个 raw 内部自哈希、`summary_sha256=115f593e…` 匹配）；但缺 W primary 结论轴，reachability 标签易误读。

**Cross-consolidation**：核心数值 PASS；publication package = `NOT_READY`；accepted P0/P1/P2 = `0/6/9`。

**决策**：只运行 1 个 S0 supplementary correction；原 22 raw/log 不可变；S4/A8/W/R4 不重跑。

### 5.3 修复内容（映射到证据合同）

| 问题 | 修复 |
| --- | --- |
| wave-0 缺 exclusive terminal reason | 独立 correction run，40/40 direct `unsupported <- store_miss`（§4.10） |
| p95 命名 | 全部改为 `ratio_of_marginal_p95s`，并附 `p95_pairing="nonpaired"` 与 `ratio_of_marginal_p95s_direction` |
| launch block 相邻性 | 显式标注 `comparison_design="seed-matched_non_adjacent_restart_comparison"` 与 `not_a_paired_launch_block=true` |
| cell 级 vs profile 级 | wave-0 输出改为 cell 顶层 status + 每个 profile 的 `reachability`/`valid` 分述 |
| W primary 结论轴缺失 | 新增 `victim_footprint` 为 A8 NEGATIVE 后的 primary 轴，latency 降为 secondary descriptive |
| 记账双计风险 | 全部 footprint 字段附 `overlap_note`；`arm_interval_peak_device_bytes` 明确语义为 `arm_high_water_since_last_full_reset` |
| publication 层补充 | 补 W victim/footprint、R4 fallback/5x accounting、wave-0 cell/profile/outcome/reason、A8 n=1 与 request-path 定义、W fallback 比例与 non-adjacent protocol、p95 与完整 provenance |

### 5.4 Correction 治理链

| 步骤 | 结果 |
| --- | --- |
| correction code pin | `a950ab914e6b029f86d8ef666a3f770fc42980a7`；capacity runner blob=`c92225d4644c0960ff537fa2d8759a2745750beff04be2662085e118d549867f` |
| correction manifest rev1 | self=`2a907feec24f07d8601be0b48a21fc98cc6bdc08b6499031e2d284cf8b00a3f1`，status=`pinned_blocked`，allowed_setting 仅 S0/rho2/chunk4096 restart0，`original_raw_sha256=80e8e83d…` |
| correction Opus review | `PASS_WITH_CAVEATS`；两项治理 P1 闭合；`RESULT_MANIFEST=75/75` |
| 只读 targeted delta review（2026-07-28T19:19:44） | `PASS_WITH_CAVEATS`，open P0/P1=`0/0`，entry=`READY_AFTER_REVIEW_ARTIFACT_AND_AUTH_CORRECTION`；6 条 P2 不阻塞 |
| correction rev2 | `authorized` → 执行 → 40/40 direct `unsupported <- store_miss` |
| Reconsolidation | 22 compact + summary 全量 `--force` 重生成；summary SHA=`e2deec5afbbfffb7b22de84d7218e3ca1aac0ba7327abcad5b1ae402753e4134`；**`RESULT_MANIFEST=79/79`** |
| final targeted delta closure（Opus） | byte-reproducible payload 验证；`summary_sha256=9d0aafcd…` 匹配；`result_manifest: declared=79, verified=79, pending=0`；`total_p0_p1_closed=9`（CP1-1~7、SOL-P1-1、SOL-P2-1~3、NEW-P1-1 全部关闭）；剩余 5 条 P2 不阻断 |
| sol-publication-ready | `PASS_WITH_CAVEATS / READY_WITH_CAVEATS`，P0/P1/P2 = `0/0/0` |
| final disposition | `PHASE7_FINAL_DISPOSITION.json`，`disposition_sha256=4013f054…` |
| 最终 manifest | **`88/88`**，`known_gaps=[]`，file SHA=`6b6b0af1…`；文件数演进 `75 → 79`（reconsolidation 新增 4）`→ 88`（publication package 封版，8 份 review 报告等纳入） |

### 5.5 Consolidator（离线一次性，纯读 staging）

`impl:benchmark/approx_kv/consolidate_phase7_results.py` + `test/registered/unit/bench/test_consolidate_phase7_results.py`。验证项：

- authorized rev12 与 design hash；
- 22 个实际 start；
- 8 个 `ES-R0-MDE` 跳过；
- rho3 disabled；
- raw / log / central JSONL 的 hash；
- 工程不变量（reset、inactive counter、orphan）。

输出 22 个自哈希 compact + 1 个自哈希 consolidated summary；默认拒绝覆盖，`--force` 显式允许。首次实现验证（2026-07-28T11:30 / 12:00）新增了 execution envelope、source SHA、manifest file SHA、runner path/hash、central run_id/phase/output 的逐 run 绑定。

**注意**：consolidator 按设计在 correction 执行前不可端到端跑通，这**不是缺陷**。

---

## 6. Lessons Learned

### 6.1 机制层

1. **chunk 配置可以完全改变机制结论的符号。** 同一 body2048/rho2/S0：chunk1024 下 request-path `1.737x`，chunk4096 下 `0.936x`。
2. **R0 作为 speed-only ceiling 在 chunk4096 下仍为负**，意味着在这个模型/硬件/prompt 族下，raw KV 复制加位置修正的固定开销（seed_head + transfer + 最后 1 token forward）已经吃掉了省下的 prefill。
3. **摊销到 N=8 仍不转正**（`0.6086–0.6419`），说明问题不只是 setup 一次性成本。
4. **表示多重性是硬约束**：5x 的 `r4_like` 在 S4 下 registration 失败，在 S0 下虽可 registration，但 122 个请求里只有 12 次真实 recovery。
5. **在真实 workflow 中，approximate 臂相对 request-paired exact 臂更慢**（`r0_over_e0_mean_latency_ratio ≈ 1.51`）。

### 6.2 系统层

6. **runtime 结果不得污染 code worktree。** 三个 runner 只写 `/results/phase7` staging，否则第二次 run 无法保持 clean envelope。
7. **多个 runner 必须共享同一授权门**，capacity pilot 在 Phase7 模式中只能执行一个 manifest setting。
8. **A8 的 source 必须真实 pin 到 sequence 结束**：使用默认关闭、服务端门控、上限 16 条的 `pin_until_reset` registration lease；reset 必须释放全部 lease。
9. **process-lifetime peak 不能冒充 per-arm peak**：full reset 现在清零 cross-store budget high-water，字段明确为 `arm_interval_peak_device_bytes`。
10. **memory accounting 不得双计 approx store**：`nonfree_resident_bytes` 已含 approx slot。

### 6.3 测量层

11. **缺失的证据无法离线重建。** wave-0 S0 缺 exclusive terminal reason，只能通过 1 次独立 correction run 补证。
12. **停止规则必须在执行前冻结并被机器执行。** `ES-R0-MDE` 直接省下了 8 个 start（约 36% 的 committed starts）。
13. **必须同时报告 full-setup 与 incremental-setup 两个摊销口径**，不得只报较有利的一版。
14. **break-even 未观察到就写 `>8/not_observed`，禁止插值或公式外推**——这是相对 Phase4 的直接改进。

### 6.4 统计层

15. **p95 统计口径必须命名为 `ratio_of_marginal_p95s`**，不是 paired 统计量。
16. **A8 primary 每个 setting 只有 1 个独立 process-level restart（n=1）**；formal repeats 与 targets 都不是独立样本。
17. **launch block 相邻性必须明确披露。** S0/S4 实际非相邻，只能称 `seed-matched non-adjacent`。
18. **同一 trace 内的请求行不是独立 timing replicate。**
19. **canary 的判别力必须量化。** body1024 canary 只有 `1` 个 distinct output token。

### 6.5 治理 / provenance 层

20. **`result_git_sha=null` 不等于 provenance 缺失。** runner 无法知道将来容纳自己输出的 commit；`RESULT_MANIFEST.json` 才是权威的 file→commit 映射。
21. **不要建立 summary ↔ manifest 的双向 hash 循环。** summary 明确记录 `hash_omitted_to_avoid_summary_result_manifest_cycle=true`。
22. **code pin 与 execution envelope 必须分层。** V7 采用 code pin commit 为祖先，后续只允许 Phase7 result envelope 路径变化，并逐 blob 验证 runner 与 manifest。
23. **plan 必须 byte-frozen，activation 记录在别处**，否则 design hash 与激活状态互斥会形成 review 循环。
24. **CPU / review evidence 必须内容绑定**：记录固定 Docker digest、exact required command、exit code 与 summary。
25. **evidence correction 必须独立计账**，不改原执行集合，不改原 GPU 小时数，原 raw/log 保持字节不变。
26. **停止条件触发的删项不是失败。** R2 走 `disabled_not_comparable` 是按预定决策树的选择，不是实现失败后的临时删项。

---

## 7. 最终结论

### 7.1 最终 disposition（`PHASE7_FINAL_DISPOSITION.json`）

```text
phase7_execution          = complete
main_session_disposition  = PASS_WITH_CAVEATS
publication_disposition   = READY_WITH_CAVEATS
open_findings             = { P0: 0, P1: 0 }

engineering        = VALID
r0_mechanism       = NEGATIVE
w_system_behaviour = INCONCLUSIVE/DESCRIPTIVE
```

判定规则（`status_derivation`）：

| 状态 | 规则（原文摘要） |
| --- | --- |
| `engineering=VALID` | *"VALID is emitted only after all expected primary artifacts, central bindings, hashes, resets, inactive counters, and any required evidence correction pass"* |
| `mechanism=NEGATIVE` | *"NEGATIVE is derived from all four A8 restart-0 request-path medians remaining below the preregistered 1.05 MDE"* |
| `system_behaviour=INCONCLUSIVE/DESCRIPTIVE` | *"W has three independent restarts per policy/rho but is descriptive after A8 NEGATIVE and mixes dense fallback"* |

双模型最终 review：

| Reviewer | Verdict | Publication | open P0/P1 |
| --- | --- | --- | --- |
| Sol | `PASS_WITH_CAVEATS` | `READY_WITH_CAVEATS` | `0/0` |
| Opus | `PASS_WITH_CAVEATS` | `READY` | `0/0` |

### 7.2 当前仍成立的结论

| 结论 | 作用域 |
| --- | --- |
| 整套 22-start 执行在工程上有效（hash / provenance / reset / inactive counter 全通过 + 必需的证据补正） | Phase7 全部 primary artifact |
| chunk4096 下 R0 ceiling 的 request-path speedup 为 `0.7723–0.9362x`，未达 5% MDE | 本模型、合成 prompt、SM75、chunk4096 |
| chunk4096 下 R0 摊销到 N=8 的 full-setup speedup 为 `0.6086–0.6419x`，break-even `>8/not_observed` | 同上 |
| chunk1024 下同配置为 `1.7370x`（request-path）/ `1.1890x`（N8 full） | **仅证明 chunk 耦合** |
| S4 在 workflow 对象上的 clamped hit fraction 达 `1.0`，S0 仅 `0.03–0.06` | W 矩阵，描述性 |
| S4 相对 S0 的 miss 减少：rho1.5 all `-14` / workflow `-10`；rho2.0 all `+2` / workflow `-10` | 描述性 |
| S4 vs S0 的 latency 比值仅 `1.0021x–1.0442x` | 描述性；混有 `45.9%–72.1%` dense fallback |
| 同一 policy 内 R0 比 request-paired E0 慢约 `1.51x`（mean latency ratio） | 描述性 |
| `r4_like`（5x）在 chunk4096/S4 下 registration 失败；在 S0 下 122 请求仅 12 次 recovery | diagnostic/proxy |
| 自然压力下未观察到任何 reservation failure（0 次） | Phase7 全程 |
| 全部 88 个 artifact 的 file→commit 映射已验证，`known_gaps=[]` | provenance |

### 7.3 被收窄或推翻的结论

| 原结论 | 处置 |
| --- | --- |
| Phase4 chunk1024 下 R0/R1-k0 家族的 `1.5x–2.0x` 表观收益可迁移到 chunk4096 | **被 Phase7 R0 ceiling 否定**：R0 为 `0.77–0.94x`；该结果不外推到 Phase7 未执行的 R2、R4 或其它 repair 机制 |
| 「长 body 恢复更划算」 | 方向仍在（body2048 `0.936x` 高于 body1024 `0.772x`），但**两者都 `<1.0`**，不构成收益 |
| 「摊销若干次后恢复必然转正」 | **未观察到**：N≤8 全部 `>8/not_observed` |
| 「S4 相对 S0 有实用收益」 | **不允许声称**：预注册规则不允许 practical benefit claim；A8 NEGATIVE 后 latency 降为次要 |
| 「R4-like 结果可代表 KVCOMM」 | **明确禁止**：`not_kvcomm=true`，`ranking=disabled` |
| 「wave-0 五个 profile 可达 = 恢复可用」 | **收窄**：registration reachability ≠ approximate-recovery success |

### 7.4 明确**不能**声称的内容（`required_caveats` 逐条）

1. `practical=NONE` 严格限定于被测实现与 chunk1024 qualification 范围。
2. **R0 是 ceiling path，不是被 promote 的 practical candidate。**
3. R2 是 `disabled_not_comparable`，Phase7 未执行。
4. `r2_like` 与 `R4-like` 均为 synthetic footprint profile，**不是 R2 或 KVCOMM 执行**。
5. chunk1024 sensitivity 是 chunk-coupled，**不是机制固有 headline**。
6. **自然压力下的 reservation-failure fallback 可达性仍未证明。**
7. **wave-0 的 registration reachability 不是 approximate-recovery success。**
8. W 比较为 seed-matched 但**非相邻 launch block**，且含大量 dense fallback。
9. **独立 timing replicate 单元是 server restart。**
10. **A8 primary 每个 setting 只有 1 个独立 process-level restart（n=1）。**
11. **body1024 same-context canary 判别力有限**（仅 1 个 distinct output token）。
12. **host / prefetch / async / HiCache 轨道均未生成。**
13. （scope caveat）P6-4 的 rho1.1/1.5/3 feasibility 仍限定 chunk1024，除非单独重新验证。

附加禁止项：

14. **不得发布任何 R0 speedup headline**（`headline_speedup_allowed=false`）。
15. **不得把 Phase4 chunk1024 数值与 Phase7 chunk4096 结果合并统计或排名。**
16. **不得对 R4-like 做性能排名**（`performance_ranking_enabled=false`）。
17. **不得据 artifact 的 `result_git_sha` 声称 provenance 完整。**
18. **不得把 `nonfree_resident_bytes` 与 `approx_device_bytes` 相加。**

---

## 8. 该结论能预测什么（可证伪预测与预注册问题）

| 编号 | 预测 | 证伪条件 |
| --- | --- | --- |
| P7-1 | **若** coupled chunk/max-prefill effect 在 body≫4096 时仍占主要份额，则 dense 跨更多 chunk 后 R0 request-path speedup 应上升 | 若 speedup不升，说明 copy/setup 等成本抵消了 chunk 边界差异；现有数据不预言是否一定越过 1.0 |
| P7-2 | 将 device limit 提高到超过 measured live footprint，可能使 S4 的 R4-like registration 可达；显存容量、带宽和算力必须分开做 factorial | 预算明确足够仍失败才支持实现缺陷；更强算力对 R0 的方向没有现成先验，因为 dense 与 recovery 会被不同程度加速 |
| P7-3 | 降低 dense fallback 能减少 policy 估计的混杂，但不保证 S4−S0 latency 差异扩大 | 在 matched fallback/coverage 条件下比较策略，才能识别真实 policy effect |
| P7-4 | 补跑 A8 restart 1–2 用于估计跨进程稳定性；当前 n=1 不支持方向预测 | 若补跑后任一 setting 达到预注册门槛，应按新证据更新；否则 NEGATIVE 获得更强复制支持 |
| P7-5 | 现有 workload 下自然 reservation failure 可能继续稀少 | 非零事件只证明 reachability；关闭 caveat 还需关联 fallback 完成、输出匹配和 clean accounting |
| P7-6 | 真实 trace 的 artifact reuse-count 分布决定 full-setup 与 incremental-setup 哪个口径更具代表性，当前没有 `>8` 的方向先验 | 先采集真实 trace；reuse≤8 时 full-setup 仍应是主口径，显著高于8时才可提高 incremental 权重 |

### 8.1 Phase 8 触发条件

`PHASE7_FINAL_DISPOSITION.json.phase8`：

```text
automatically_triggered           = false
requires_new_plan_and_authorization = true
```

候选范围（`docs:PROJECT.md` / `HANDOFF.md` 记录）：RTX PRO 6000、更大模型 / context、并发 workflow、真实 repository、source/dependency invalidation、端到端 coding correctness。

Phase7 schema 已预留 forward-compatible 字段供未来触发条件判断：per-role / per-artifact latency、host transfer/overlap、concurrency 标识、source/revision/invalidation metadata。

---

## 9. 局限、未完成项与 artifact / provenance 索引

### 9.1 局限

1. 单 GPU（SM75，8GB）、单模型（Qwen3-0.6B）、合成 prompt 与合成 workflow trace。
2. A8 primary 每 setting 仅 `n=1` 个独立 restart；formal repeats 与 targets 不是独立样本。
3. W 的 S0/S4 非相邻 launch block。
4. W 的 latency 比值混有 `45.9%–72.1%` dense fallback。
5. body1024 canary 只有 1 个 distinct output token。
6. `r4_like` 与 `r2_like` 均为 synthetic footprint proxy。
7. host / prefetch / async / HiCache 轨道完全未生成。
8. 全程串行，无并发/多租户干扰。
9. 无 accuracy / semantic correctness 评估。

### 9.2 未完成项

| 未完成项 | 影响 |
| --- | --- |
| A8 restart 1–2（8 starts） | 被 `ES-R0-MDE` 跳过；`three_restart_range.available=false` |
| rho3 conditional start | 被禁用，未执行 |
| R2 GPU cells | `disabled_not_comparable` |
| 自然压力 reservation-failure 可达性 | 仍未证明（Phase6 遗留 + Phase7 0 次观察） |
| chunk4096 下 `practical` 重新 qualification | V7 scope 决策，非经验结论 |
| P6-4 rho1.1/1.5/3 在 chunk4096 下的重新验证 | 现有结论限定 chunk1024 |
| host / prefetch / async / HiCache 轨道 | 预算为 0 |
| 更大 body / 更大模型 / 并发 / 真实 repository | Phase8 候选 |
| 5 条 P2（final verify 后剩余） | 不阻断 publication |

### 9.3 Artifact / provenance 索引

**权威索引**：`p7:RESULT_MANIFEST.json`（`schema_version=2`，`files` 共 **88** 条，`known_gaps=[]`，file SHA=`6b6b0af19b70b9958866dc9674102026b348b1d5a4254e7f2f9d732a77548a65`）。结果目录共有 89 个 tracked physical files；manifest 为避免自引用哈希循环，不把 `RESULT_MANIFEST.json` 自身列入 `files`，因此验证条目为 88。

`authority` 原文：*"This manifest, not the `result_git_sha` field inside an artifact, is the authoritative mapping. Regenerate and re-check it in the same commit that adds or changes an artifact, otherwise entries rot into pending or stale-blob states."*

目录结构：

| 路径 | 内容 | 数量 |
| --- | --- | ---: |
| `p7:raw/` | 22 个 primary run 的 raw JSON（不可变） | 22 |
| `p7:compact/` | 自哈希 compact（correction 后 `--force` 重生成） | 22 |
| `p7:logs/` | 服务端日志（不可变） | 22 |
| `p7:reviews/` | `sol-original` / `opus-original` / `sol-cross-consolidation` / `opus-cross-consolidation` / `opus-targeted-delta` / `sol-final-verify-pre-authority` / `opus-final-verify` / `sol-publication-ready` | 8 |
| `p7:evidence/` | `ceiling-cpu.json` / `scheduler-cpu.json` / `capacity-pilot-cpu.json` / `capacity-correction-cpu.json` | 4 |
| `p7:capacity-correction/` | `raw/`、`logs/`、`phase7-runs.jsonl` | 3 |
| `p7:` 根 | `phase7-consolidated-summary.json`、`PHASE7_FINAL_DISPOSITION.json`、`RESULT_MANIFEST.json`、`phase7-primary-manifest.json`、`phase7-capacity-correction-manifest.json`、`phase7-final-opus-review.json`、`phase7-capacity-correction-opus-review.json`、`phase7-runs.jsonl` | 8 |

关键 hash 一览：

| 对象 | SHA-256 |
| --- | --- |
| consolidated summary（canonical） | `9d0aafcd776305c7ac679b14845e4a5c79e44c04bb696ecf2d83644fca4c2c69` |
| consolidated summary（file） | `05acb52d51ea68bcff95c392e13348f73292dad0921b154efaf1e02e150eb135` |
| RESULT_MANIFEST（file） | `6b6b0af19b70b9958866dc9674102026b348b1d5a4254e7f2f9d732a77548a65` |
| final disposition（self / file） | `4013f054751f44e222ab698d0d232bc6fe516feb29111fb4c64f53b75f797d66` / `731ace74108bc6ec0b6beb160bb3a55fcdbaae2fd2959c1a3f3d0b7aff466af5` |
| primary manifest rev12（self / design / file） | `2d66a1bc…` / `50003145…` / `c0a14e3a8127db4ad7fba20920a156b9b731e7ea36711c8c104a49bca3a8667d` |
| central JSONL | `3655a374a554161a1c52a2ac1ce23a3cc2fc321c9d92772b2193aae60697ea67` |
| entry review artifact | `2c83f7ddec41d65a937a6b71cab851230c0fb06ff725f09cb36dd596e81e8649` |
| correction raw / original raw | `59049306019351d2100565be3653d74e4ebe60a1fb49d1e03016e20c9cce7534` / `80e8e83d7b587b1ed566889e1603eead82eb2618b58a2f9a1816fb8eae741ff3` |
| correction manifest rev2（self） | `0c57df5b75c48d53a11c1355d37f24fd360f15d2f2fb89afa88dfa6923b68283` |
| Sol reports（original / cross / final） | `f3cb83b8…` / `fcccb4de…` / `ea65a6b1…` |
| Opus reports（original / cross / delta / final-verify） | `46d52e7d…` / `3cc01d25…` / `8d983ead…` / `20a03d99…` |

关键 commit / HEAD：

| 项 | 值 |
| --- | --- |
| V7 plan commit | `c80ec165713772c533e17ef4c50f083c36dc9d72` |
| primary execution code pin | `81405f4278b034911bc613c4ee17c79d15ee8f35` |
| primary manifest commit | `759476c901b1ac30eeeb96f83b9257586a103c4e` |
| execution envelope HEAD | `d42a5d1546680de458122f605f8a486b8faf5564` |
| correction code pin / execution head | `a950ab914e6b029f86d8ef666a3f770fc42980a7` / `db1c4843fa08ace9b08200fc67d22ad2a7138ff8` |
| CPU evidence HEAD | `a01642fdd6cdc869cb2c991c23003ff600665f37` |
| summary-binding commit | `107f6bfd39f25506929327a146377dafd964db6b` |
| final publication branch HEAD | `0206f17b4255e4b248dafaaeb943be57428dae2f` |
| entry review artifact commit | `b0837505abc4efe7df914c3b504693956fe71df9` |

**关于 `result_git_sha=null`**：这是 runner 的固有限制（无法知道将来容纳自己输出的 commit），**不等于 provenance 缺失**。`RESULT_MANIFEST.json` 是权威的 file→commit 映射；summary 显式记录 `hash_omitted_to_avoid_summary_result_manifest_cycle=true`，以避免 summary ↔ manifest 的双向 hash 循环。

---

## 10. 与其它阶段报告的关系

| 关系 | 说明 |
| --- | --- |
| ← [Phase4](PHASE4_RECOVERY_METHODS_REPORT.md) | Phase7 在 chunk4096 下重测 R0，定量确认了 Phase4 收益的 chunk 耦合；Phase4 的 R2 被解析为 `disabled_not_comparable` |
| ← [Phase5](PHASE5_WORKFLOW_SCHEDULING_REPORT.md) | Phase7 W 矩阵沿用 S0/S4（S1–S3 未过 revalidation gate），并首次在含 approximate 臂的场景下测量 |
| ← [Phase6](PHASE6_CROSS_STORE_SUBSTRATE_REPORT.md) | Phase7 全部实验运行在 Phase6 底座上；P6-F 的 fault-injected-only caveat 原样保留 |
| → [跨阶段总报告](PHASE4_TO_PHASE7_SUMMARY.md) | 汇总最终决策规则、可预测事项与仍未解决的研究问题 |
