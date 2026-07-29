# Phase 5 正式研究报告：Workflow-Aware Cache Scheduling 与 Prefetch 隔离

> 报告类型：正式阶段研究报告（自包含、可审计）
> 覆盖阶段：Phase 5（S0–S4 scheduler policy、P0–P3 prefetch）+ 直接修正 Phase 5 结论的 Closeout CL3
> 撰写时间：2026-07-28
> 报告状态：`最终权威`（叙事层）；数值权威性以本文逐条状态标签为准
> 关联报告：[Phase4](PHASE4_RECOVERY_METHODS_REPORT.md)｜[Phase6](PHASE6_CROSS_STORE_SUBSTRATE_REPORT.md)｜[Phase7](PHASE7_INTEGRATED_EVALUATION_REPORT.md)｜[跨阶段总报告](PHASE4_TO_PHASE7_SUMMARY.md)

---

## 0. 引用约定

| 前缀 | 含义 | 绝对根路径 |
| --- | --- | --- |
| `docs:` | 文档仓库（本报告所在仓库） | `/home/chris/Workspaces/code-agent-kvcache` |
| `impl:` | 实现/结果仓库（cross-store-substrate worktree） | `/home/chris/Workspaces/kvcache-research/worktrees/cross-store-substrate` |

状态标签：`最终权威` / `历史/已被替代` / `diagnostic/proxy`。

---

## 1. 文档定位、证据状态与 Executive Summary

### 1.1 文档定位

Phase 5 的定位是**在纯 exact cache 条件下单独验证 workflow-aware scheduling 与 prefetch 的价值**，与 Phase 4 的 lossy recovery 完全隔离。本报告固化 Phase 5 的原始实验、其原始结论、以及 Closeout CL3 零 GPU 重算对这些结论的分母级修正。

**本报告最重要的边界声明：Phase 5 只测 exact Radix scheduler，没有执行任何有损 KV 恢复。**

### 1.2 证据状态总览

| 证据源 | 状态 | 说明 |
| --- | --- | --- |
| `impl:benchmark/approx_kv/results/phase5-scheduler/sm75-scheduler-matrix.json` | `authoritative_historical` | S0–S4 × rho 原始矩阵；`workflow_summary.requests` 固定 = 20 |
| `impl:benchmark/approx_kv/results/phase5-scheduler/sm75-prefetch-matrix.json` | `diagnostic/proxy` | 12 cell prefetch 矩阵；host tier 饱和 + 同步 H2D，只能作功能/开销 canary |
| `impl:benchmark/approx_kv/results/phase5-scheduler/sm75-restart-validation.json` | `authoritative_historical` | 8 runs，仅覆盖 rho1.5/2.0 × {S0,S4} × restart{0,1} |
| `impl:benchmark/approx_kv/results/phase6/cl3-phase5-recalculation.json` | **`最终权威`（分母修正）** | 零 GPU 重算；`run_id=cl3-20260727T031459Z`，`raw_sha256=17f010b75e5f18dd38c675550ef041a90d12e93211f2384093e759b13bd3af41` |
| `impl:benchmark/approx_kv/run_phase5_scheduler_matrix.py`（810 行） | 实现证据 | S0–S4 / P0–P3 runner |
| `impl:benchmark/approx_kv/run_cl3_phase5_recompute.py`（431 行） | 实现证据 | CL3 零 GPU 重算 |
| `impl:benchmark/approx_kv/workloads.py` | 实现证据 | `CacheObjectKind` 枚举、`build_object_catalog` 轮转标签 |
| `docs:research/PHASE5_RECALCULATED_METRICS.json` | `diagnostic/proxy` | 更早的 closeout 重算尝试（含 `scipy.optimize.milp` 变尺寸离线上界探索）；分母命名与 CL3 不同，**不作为权威口径引用** |
| `docs:PROJECT.md:1372-1389` | `最终权威` | CL3 权威表与 review 修正后的精确表述 |

### 1.3 Executive Summary

1. **Phase 5 的全部 workflow cache hit 都是 exact Radix/HiCache hit，miss 走普通 dense prefill。** runner 未发送 `approx_kv` metadata，R0/R1 恢复路径并未执行（`docs:PROJECT.md:2588-2598`）。因此 Phase 5 结论的作用域是 **exact-cache scheduler policy isolation**，不是「五条 recovery workload × 五条 scheduler」的组合矩阵。

2. **原始结论：S4 hierarchical + P0 off 被定为默认配置**，S4 是唯一在 rho1.5/2.0/3.0 三个高压档稳定优于 S0 LRU 的策略。

3. **CL3 分母修正后，该结论必须按口径分列**：

   - `workflow-only`（只统计 20 个 workflow 请求）：S4 在 rho1.5/2.0/3.0 分别为 `1.321 / 1.148 / 1.147`，S1/S2/S3 在 rho2.0/3.0 掉到 `≈1.00`；**S4 相对其它策略的描述性数值分离只在此口径出现**。
   - `all-reusable`（全部 `expected_reusable_prefix_tokens > 0` 的请求）：S1–S4 相对 S0 的改善均在 `1.087–1.187` 之间，**四策略数值上几乎不可区分**，且现有独立 restart 数不足以支持策略排序。
   - 精确表述必须是：**「S4 相对 S1–S3 的独特性消失，相对 S0 仍有数值改善」**，不能写成「S4 优势消失」。

4. **现有数据不能排序 S2 与 S4。** workflow-only 下 S4 的均值更高，
   但 all-reusable 的部分 rho 点由 S2 略高，且多数 cell 只有 1 个 restart。
   S2 必须命名为 **Belady-style next-request-ordinal oracle**，**不是
   variable-size offline optimum**。

5. **Prefetch canary 未观察到稳定 mean 改善。** CL3 用同策略 P0
   正确配对后，9 个 paired cell 的 mean speedup 落在
   `0.9885–1.0038`；原始矩阵中 P2 的 p95 相对 P0 增加
   `2.91%–4.89%`，P3 增加 `3.75%–3.98%`。由于 host 大于工作集、
   H2D 同步且无独立 restart，这不是普遍性能否定。

6. **S4 的 object kind 标签是纯轮转合成的，不是真实 object DAG。** `workloads.py:298,314-315` 中 `kind = kinds[index % len(kinds)]`，与对象间真实依赖/恢复关系无关。

7. **rho sweep 混杂了两个变量**（对象数 15/20/27/40 与 capacity/pressure），因此不能单独证明或否定「固定 workload 下 speedup 随 rho 单调变化」。

---

## 2. Phase 5 动机、研究问题、冻结假设与非目标

### 2.1 动机

来源：KVFlow（arXiv 2507.07400）+ KVCOMM。历史结论（`docs:TRACKING.md:77`）：「cache pressure 决定 priority 的价值，sequential workflow 中强制 prefetch 可能造成 cache churn」。

`docs:PROJECT.md:13-14` 的业务目标是「比较近似 KV 恢复与 workflow-aware cache scheduling，降低 Coding Agent TTFT」。Phase 5 选择**先隔离 recovery 收益**（`docs:PROJECT.md:1914`）：在纯 exact-cache 条件下单独验证 scheduling/prefetch 价值，避免与 Phase 4 的 lossy recovery 混杂。这就是「prefetch isolation」动机。

职责划分（`docs:TRACKING.md:1554`）：**KVFlow 负责用固定 workflow 的未来执行距离做 cache priority / eviction / CPU→GPU 调度；KVCOMM 负责跨 role/prefix 复用。** Phase 5 实现的 S1–S4 是 KVFlow 思路的 policy 化。

### 2.2 研究问题

| 编号 | 研究问题 | 结论位置 |
| --- | --- | --- |
| RQ5-1 | 在 exact cache 下，workflow-aware priority 能否显著优于 LRU？ | §7.1 |
| RQ5-2 | 收益在什么 pressure 区间出现、什么区间消失？ | §4.3、§7.2 |
| RQ5-3 | 对象层级（object class）是否比单纯 next-use 距离更有价值？ | §7.2 |
| RQ5-4 | 强制 prefetch 在 sequential workflow 下是否有收益，还是造成 churn？ | §4.4、§7.1 |
| RQ5-5 | 结论对分母（哪些请求计入）是否敏感？ | §4.5、§7.2 |

### 2.3 冻结假设

- 服务器：`mem_fraction_static = 0.35`，usable KV capacity ≈ `13,130` tokens（JSON `gauges_after."sglang:max_total_num_tokens" = 13130.0` 实测验证）。
- 固定 workflow：`Architect → Coder → Debugger`，两轮 `Architect→Coder→Debugger→Coder→Debugger` + live filler replay。
- 5 个固定 workflow 对象：Architect ×1、Coder ×2、Debugger ×2。
- metadata 独立链：`workflow_steps` / `belady` / `recovery_value` / `hierarchical`，**不复用 `Req.priority`**（用户明确要求，`docs:PROJECT.md:2395`）。
- `protected_tokens` 切 reusable-prefix 边界。
- prefetch victim 为「对象边界 + 全部 dynamic suffix 后代」的原子子树。
- 工程严谨性：setting 逐个运行、独立 server、请求串行、顺序随机、repeat 间 flush（`docs:HANDOFF.md:633`）。

### 2.4 非目标

- **不做** lossy KV recovery（Phase 5 完全不执行 R0–R5 任何一条）。
- **不做** 恢复质量 / 输出一致性评估。
- **不做** 真实 object DAG 建模（kind 标签是合成的）。
- **不做** 并发/多租户调度。
- **不做** 跨 Phase 的 speedup 排名（Phase4 与 Phase5 分母定义不同）。

---

## 3. 环境、实现范围、方法与测量口径

### 3.1 执行环境（Docker 内执行）

| 项目 | 值 |
| --- | --- |
| 镜像 digest | `sha256:0be6e16e2eb288dfd5fa8b0b41015f61731a139fb961d3366ccedf834289d781` |
| 模型 | `Qwen/Qwen3-0.6B` |
| model revision | `c1899de289a04d12100db370d81485cdf75e47ca` |
| GPU | NVIDIA GeForce RTX 2080 SUPER，SM75，8192 MiB |
| CUDA / Torch / Transformers / Python | `12.9` / `2.9.1+cu129` / `5.12.1` / `3.12.3` |
| Phase5 matrix `source_git_sha` | `5a87166b436e00fa730aa7062e949516ca823a96` |
| CL3 `source_git_sha` | `0b5e4f7b59f05ae3cfaaf307dadbfe74910d8f25` |

证据：`impl:benchmark/approx_kv/results/phase5-scheduler/sm75-scheduler-matrix.json` 的 `image_digest`/`model`/`model_revision`/`machine`/`source_git_sha` 字段。

### 3.2 S0–S4 策略定义

权威处：`docs:PROJECT.md:2517-2524`；脚本 `impl:benchmark/approx_kv/run_phase5_scheduler_matrix.py:36-40`。

| 策略 | 脚本 label | 定义 |
| --- | --- | --- |
| S0 | `lru` | LRU baseline |
| S1 | `workflow_steps` | coarse workflow steps-to-execution |
| S2 | `belady` | **Belady-style next-request ordinal oracle**（在已记录 trace 上的下一次请求序号；**不是** variable-size offline optimum） |
| S3 | `recovery_value` | synthetic saved-cost / resident-byte value density |
| S4 | `hierarchical` | dead / recoverable / exact / repair / anchor / exact / canonical-base 层级 |

### 3.3 P0–P3 prefetch 定义

权威处：`docs:PROJECT.md:2395-2410`。

| Prefetch | 定义 |
| --- | --- |
| P0 | off |
| P1 | free-space-only（只在有空闲空间时载入） |
| P2 | known-dead-object-only eviction（只驱逐已知 dead 对象） |
| P3 | oracle-farther-use（驱逐更远使用者），且对象须可恢复 |

### 3.4 Workload 与 rho sweep

Trace family（`docs:PROJECT.md:2578-2588`；脚本 `run_phase5_scheduler_matrix.py:281-323`）：

- `active_workflow_objects` 按 role 选 5 个对象；
- `fillers` 按 `dead_count = len(fillers)//3` 切分 live / dead；
- phase 标签：`pressure_live_fill` / `pressure_live_backup` / `pressure_dead` / `pressure_replay` / `workflow`。

rho（commit-bound 实际值，`docs:PROJECT.md:2620`）：

| target rho | 实际 rho | 对象数 |
| ---: | ---: | ---: |
| 1.1 | `1.153` | 15 |
| 1.5 | `1.537` | 20 |
| 2.0 | `2.075` | 27 |
| 3.0 | `3.073` | 40 |

**关键 caveat（`docs:PROJECT.md:2830-2831`）：rho sweep 靠增加对象数扩大 working set，而不是固定对象集合只调 capacity。因此该 sweep 混杂了 working-set composition 与 capacity/pressure 两个变量，不能单独证明或否定「固定 workload 下 speedup 随 rho 单调变化」这一历史 claim。**

### 3.5 测量口径

原始 Phase 5 汇总使用单一口径：`workflow_summary` 的 20 个 workflow 请求。CL3 引入三种分母（`cl3-phase5-recalculation.json` 的 `definitions.denominators`）：

| 分母 | 定义（原文） |
| --- | --- |
| `workflow_only` | `phase == workflow` |
| `all_reusable` | `expected_reusable_prefix_tokens > 0`（含 pressure fill / replay） |
| `full_trace_wall_clock` | `sum of client elapsed_ms over every measured request in one formal repeat, including pressure fill and replay` |

hit fraction 口径（`run_cl3_phase5_recompute.py:75-79`）：

```python
def clamped_hit_fraction(record):
    expected = record.get("expected_reusable_prefix_tokens")
    if not expected:
        return None
    return min(1.0, max(0.0, cached_tokens / expected))
```

即 **per-request clamp 后再聚合**，修正了原始汇总「先聚合再 clamp / 未 clamp」的偏差。

样本独立性定义（`definitions.sample_independence`）：**"requests inside one trace are not independent experiments; per-restart and per-repeat values are reported separately"**。

---

## 4. 全部实验：矩阵、执行顺序、核心数值

### 4.1 执行顺序

```text
S0-S4 × rho{1.1,1.5,2.0,3.0} scheduler matrix（P0）
  → prefetch matrix：S4 × P0/P1/P2/P3 × rho{1.5,2.0,3.0}（12 cell）
  → restart validation：{S0,S4} × rho{1.5,2.0} × restart{0,1}（8 runs）
  → [FINDING-GAP-1] Closeout CL3 被发现从未执行，列为 Phase7 Entry 阻塞
  → CL3 零 GPU 重算（2026-07-26/27 补齐）
```

### 4.2 S0–S4 原始矩阵 — `authoritative_historical`

artifact：`impl:benchmark/approx_kv/results/phase5-scheduler/sm75-scheduler-matrix.json`；字段结构 `runs[i].settings.{policy,target_pressure,restart,formal_repeats}`、`runs[i].workflow_summary.{cache_hit_fraction,cached_tokens,expected_reusable_tokens,requests=20,ttft_mean_ms,ttft_p50_ms,ttft_p95_ms}`、`telemetry_delta.counters.*`。

**`workflow_summary.requests` 固定 = 20**（已实测验证）——这正是 CL3 要修正的分母问题的原始出处。

S4 vs S0（原始 workflow 口径，`docs:PROJECT.md:2461-2515`）：

| rho | S4 mean(ms) | S0 mean(ms) | speedup | S4 hit frac | S0 hit frac |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1.1 | ≈`148.50` | — | `1.456x` | `1.000` | `0.510` |
| 1.5 | `163.46` | `215.93` | `1.32x` | `0.903` | `0.510` |
| 2.0 | `188.96` | `216.56` | `1.15x` | `0.705` | `0.510` |
| 3.0 | `189.31` | `214.37` | `1.13x` | `0.705` | `0.511` |

S1/S2/S3/S4 相对 S0 的 mean speedup（`docs:PROJECT.md:2538,2826`）：

| 策略 | rho1.1 | rho1.5 | rho2.0 | rho3.0 |
| --- | ---: | ---: | ---: | ---: |
| S1 workflow-steps | `1.446` | `1.144` | `1.006` | `0.994` |
| S2 Belady-style | `1.428` | `1.148` | `0.999` | `0.996` |
| S3 recovery-value | `1.454` | `1.150` | `1.011` | `0.990` |
| **S4 hierarchical** | `1.456` | `1.321` | `1.146` | `1.132` |

### 4.3 Restart validation — `authoritative_historical`

artifact：`impl:benchmark/approx_kv/results/phase5-scheduler/sm75-restart-validation.json`（8 runs）。

```text
hierarchical rho1.5 restart0/1: mean 161.88 / 160.82 ms, hit 0.9026
lru          rho1.5 restart0/1: mean 214.88 / 215.65 ms, hit 0.5103
hierarchical rho2.0 restart0/1: mean 188.75 / 192.12 ms, hit 0.7047
lru          rho2.0 restart0/1: mean 215.17 / 213.60 ms, hit 0.5104
```

三次独立 server 进程验证的 speedup 区间（`docs:PROJECT.md:2461-2470`）：rho1.5 = `1.32–1.34x`；rho2.0 = `1.11–1.15x`。

**限制：restart validation 只覆盖 rho1.5/2.0，且只对比 S4 vs S0；S1/S2/S3、rho1.1、rho3.0、prefetch 矩阵均无独立 restart。**

### 4.4 Prefetch 矩阵（12 cell）— `diagnostic/proxy`

artifact：`impl:benchmark/approx_kv/results/phase5-scheduler/sm75-prefetch-matrix.json`。

| policy + mode | rho1.5 / 2.0 / 3.0 mean(ms) | p95(ms) | hit frac |
| --- | --- | --- | ---: |
| hierarchical p0 | `149.94 / 151.03 / 151.47` | `153.23 / 153.10 / 154.47` | `1.0` |
| hierarchical p1 | `149.66 / 148.72 / 151.94` | `152.82 / 152.56 / 155.02` | `1.0` |
| hierarchical p2 | `152.14 / 152.59 / 152.07` | `160.70 / 160.59 / 158.96` | `1.0` |
| hierarchical p3 | `150.24 / ≈148.72 / 152.27` | `158.98 / 159.02 / 160.62` | `1.0` |

机制观察：

- P1 无主动 load；
- P2 每档主动 load `2,016` token / admission eviction `2,088` token；
- P3 在 rho3 主动 load `4,032–5,040` token / eviction `4,104–5,112` token；
- **P2/P3 无稳定 mean 收益，且 p95 从 P0 的约 `152–154ms` 恶化到约 `159–161ms`**，与历史 KVFlow 结论「sequential workflow 中强制 prefetch 可能造成 cache churn」一致（`docs:TRACKING.md:77`）。

**为什么只能作 canary**（`docs:PROJECT.md:2885`、`docs:HANDOFF.md:616`）：Phase5 prefetch 的 host 容量大于工作集，P0 整条 trace 无 miss；同步 H2D 又可能把成本落在相邻请求间隙。因此现有结果更适合作**功能 / 安全 / 开销 canary**，而非正式性能结论。

### 4.5 CL3 零 GPU 重算 — `最终权威`（分母修正）

#### 4.5.1 触发与范围

- 触发：FINDING-GAP-1（`docs:PROJECT.md:934`）——Closeout CL3 从未执行，被列为 Phase7 Entry 的真实阻塞项；后于 2026-07-26/27 补齐（`docs:PROJECT.md:1368-1370`）。
- run_id = `cl3-20260727T031459Z`；`raw_sha256 = 17f010b75e5f18dd38c675550ef041a90d12e93211f2384093e759b13bd3af41`；`derivation = "zero_gpu_recalculation"`；`performance_claim = "phase5_exact_cache_recalculation_only"`。
- 覆盖 `40` 个已提交 cell、`18` 个 scheduler paired 行、`9` 个 prefetch paired 行、`16` 个 scheduler aggregate cell。
- **不重跑任何 GPU cell**，纯从已提交 raw `result.json` 重新聚合（`load_cells` 逐 request 读取 `payload["results"]`）。

#### 4.5.2 FINDING-CL3-A：分母敏感性（权威表）

数值直接来自 `cl3-phase5-recalculation.json` 的 `scheduler_aggregate.<policy>:p0:rho<X>.<denominator>.median_mean_speedup`（相对 S0 LRU）：

| 策略 | 分母 | rho1.1 | rho1.5 | rho2.0 | rho3.0 |
| --- | --- | ---: | ---: | ---: | ---: |
| S4 hierarchical | workflow-only | `1.4568` | `1.3210` | `1.1484` | `1.1468` |
| S1 workflow-steps | workflow-only | `1.4544` | `1.1349` | `0.9959` | `1.0066` |
| S2 Belady-style | workflow-only | `1.4512` | `1.1249` | `1.0041` | `1.0081` |
| S3 recovery-value | workflow-only | `1.4658` | `1.1385` | `1.0046` | `1.0029` |
| S4 hierarchical | all-reusable | `1.1799` | `1.0894` | `1.1510` | `1.0972` |
| S1 workflow-steps | all-reusable | `1.1774` | `1.0961` | `1.1487` | `1.1067` |
| S2 Belady-style | all-reusable | `1.1785` | `1.0879` | `1.1554` | `1.1086` |
| S3 recovery-value | all-reusable | `1.1869` | `1.1001` | `1.1577` | `1.1016` |

p95 ratio（all-reusable 口径，越小越好）实测落在
`0.9835–1.0094`，区间跨过 `1.0`；最大回归约 `0.94%`。因此只能说
**没有观察到一致或显著的 p95 分离**，不能说所有 cell 都未恶化。

restart 计数（JSON `restarts` 字段）：仅 `hierarchical:p0:rho1.5` 与 `hierarchical:p0:rho2.0` 为 `2`，其余 14 个 aggregate cell 均为 `1`。

#### 4.5.3 review 修正后的精确表述（必须原样保留）

> S4 相对 S1–S3 的**独特高 rho 优势只在 workflow-only 口径成立**；在 all-reusable 口径下 S1–S4 相对 S0 **均有相近的描述性改善**（约 `1.09x–1.19x`），**现有 restart 数不足以支持策略排序**。

**「S4 优势消失」是不准确表述**——消失的是它相对 S1–S3 的独特性，相对 S0 仍有数值改善（`docs:PROJECT.md:1385-1389`）。

#### 4.5.4 FINDING-CL3-B：prefetch 重新配对

- 原 Phase5 prefetch 矩阵**没有 LRU 臂**，因此与 LRU 比较不成立。
- CL3 改为**同策略 P0 配对**：`baseline = "hierarchical + p0"`，`pairing_axis = "prefetch"`（`definitions.prefetch_baseline` 原文：*"S4 hierarchical with prefetch mode p0; the Phase 5 prefetch matrix has no LRU arm, so P0 is the only valid control"*）。
- 9 个 paired cell 的 mean speedup（workflow-only / all-reusable）：

| mode | rho1.5 | rho2.0 | rho3.0 |
| --- | --- | --- | --- |
| P1 | `0.9885` / `0.9918` | `0.9964` / `0.9968` | `1.0009` / `0.9953` |
| P2 | `0.9968` / `1.0004` | `0.9906` / `0.9963` | `1.0022` / `1.0013` |
| P3 | `0.9898` / `0.9966` | `0.9978` / `0.9999` | `1.0038` / `1.0015` |

**全部落在 `0.9885–1.0038`，即本 canary 未观察到稳定 mean 改善。**
`S4 + P0` 仍是后续实验的保守默认对照，但这不是对 prefetch 的普遍性能否定。

#### 4.5.5 CL3 引入的措辞/统计限制

| 项 | 规则 |
| --- | --- |
| 「within noise」 | **禁止使用**。CL3 多数 cell 只有 1 个 restart，只能写「数值上几乎不可区分」 |
| 样本独立性 | 同一 trace 内的请求不是独立实验；per-restart / per-repeat 值分别报告 |
| S2 命名 | `"S2 is a Belady-style next-request-ordinal oracle over the recorded trace; it is not a variable-size offline optimum"` |
| variable-size offline optimum | `definitions.variable_size_offline_optimum = "declared a Phase 7 deliverable; not computed here"` |

> 补充说明：`docs:research/PHASE5_RECALCULATED_METRICS.json` 中确实存在一个用 `scipy.optimize.milp` 求解的 `offline_variable_size_upper_bound` 段（rho1.1/1.5/2.0/3.0 下 `optimal_hits = 29`，misses = `0/6/16/34`，`capacity_tokens = 13130`）。该文件是更早的 closeout 重算尝试，分母命名与 CL3 不同，**本报告不把它作为权威口径**，只记录其存在，避免与 CL3 的「未计算」声明冲突时被误读为矛盾。

### 4.6 Phase 5 明确不包含 lossy recovery（代码级证实）

`docs:PROJECT.md:2592-2598`（2026-07-24T14:54:28 条目）明确：

- Phase5 workflow **完全没有执行有损 KV 恢复**——无 `approx_kv register/reuse`，无 Raw+RoPE / EPIC / CacheBlend / KVCOMM / CacheTune target；
- 所有 workflow hit 均为 **exact Radix/HiCache hit**，miss 走普通 dense prefill；
- Phase5 虽从 R1 EPIC 分支创建，但 runner 未发送 `approx_kv` metadata，R0/R1 恢复路径并未执行（`docs:PROJECT.md:2588-2589`）。

### 4.7 Synthetic kind 标签 ≠ 真实 object DAG（代码级证实）

| 证据 | 内容 |
| --- | --- |
| `impl:benchmark/approx_kv/workloads.py:9-14` | `CacheObjectKind` 枚举 = `CANONICAL_BASE / STAGE_VARIANT / ANCHOR / REPAIR_METADATA` |
| `impl:benchmark/approx_kv/workloads.py:298,314-315` | `build_object_catalog()` 中 `kinds = tuple(CacheObjectKind)`，`kind = kinds[index % len(kinds)]` → **纯轮转赋值，与对象间真实依赖/恢复关系无关** |
| `impl:benchmark/approx_kv/run_phase5_scheduler_matrix.py:440` | `custom_params_factory` 把 `object_kind = cache_object.kind.value` 传给 S4 hierarchical policy 作为分层判据 |

权威 caveat 原文（`docs:PROJECT.md:2884`）：**"Phase5 S4 的 object kind 由轮转标签构造，不能直接外推为真实 approximate-object DAG 的验证"**。

必须严格区分：**exact Radix scheduler 结果（真实生效）** vs **"kind" 标签所暗示的近似对象层级语义（合成、非真实 DAG）**。

### 4.8 成功 / 失败 / 被跳过项汇总

| 项 | 状态 | 说明 |
| --- | --- | --- |
| S0–S4 × 4 rho 矩阵 | 成功 | 20 个 workflow 请求口径 |
| Restart validation | 部分完成 | 仅 rho1.5/2.0 × {S0,S4} |
| Prefetch 12 cell 矩阵 | 成功但降级 | 只能作功能/开销 canary |
| S1/S2/S3 独立 restart | **未做** | 无法支持策略排序 |
| rho1.1 / rho3.0 独立 restart | **未做** | 同上 |
| prefetch 独立 restart | **未做** | 同上 |
| 固定对象集合、只调 capacity 的 rho sweep | **未做** | 现有 sweep 混杂两个变量 |
| variable-size offline optimum | **未在 CL3 计算** | 声明为 Phase7 交付物 |
| 真实 object DAG 上的 S4 验证 | **未做** | kind 为合成轮转标签 |
| 与 lossy recovery 的组合矩阵 | **未做** | 属于 Phase6/7；后因 `practical=NONE` 大部分被跳过 |
| async H2D prefetch | **未做** | P1–P3 重新进入主结果的前置条件 |

---

## 5. 发现并修复的问题

### 5.1 FINDING-GAP-1：CL3 从未执行

- **症状**：Closeout 声称完成，但 CL3（Phase5 分母重算）从未真正跑过（`docs:PROJECT.md:934`）。
- **根因**：Closeout 完成判定依赖人工勾选，没有机器可读的 completion gate。
- **修复**：把 CL3 列为 Phase7 Entry 的真实阻塞项，并在 2026-07-26/27 用零 GPU 重算补齐。
- **验证**：`cl3-phase5-recalculation.json` 产出，覆盖 40 cell / 18 scheduler paired / 9 prefetch paired。
- **对旧结论影响**：直接导致 §4.5 的分母修正，S4 的「独特性」结论被收窄。

### 5.2 hit fraction 聚合口径偏差

- **症状**：`expected_reusable_prefix_tokens = 0` 的请求会污染聚合 hit fraction，且未 clamp 会产生 `>1` 的比值。
- **根因**：原始汇总先聚合再判断。
- **修复**：`run_cl3_phase5_recompute.py:75-79` 改为 **per-request clamp 后再聚合**，`expected` 为假值时返回 `None` 并排除。
- **验证**：CL3 输出的 `clamped_hit_fraction_mean` 恒在 `[0,1]`。
- **对旧结论影响**：修正了 hit fraction 的量级，未改变 S4 > S0 的方向。

### 5.3 Prefetch 对照臂错误

- **症状**：原始 prefetch 结论隐含与 LRU 比较。
- **根因**：prefetch 矩阵根本没有 LRU 臂（12 个 cell 全部是 `hierarchical`）。
- **修复**：CL3 改为同策略 P0 配对，并在 `definitions.prefetch_baseline` 中写明理由。
- **验证**：9 个 paired cell 全部落在 `0.9885–1.0038`。
- **对旧结论影响**：正确对照仍未观察到稳定 mean 改善；结论强度降为
  **功能/开销 canary**，不能扩展为一般的 prefetch 性能结论。

### 5.4 S2 命名过度声明

- **症状**：S2 被称为「Belady oracle 上界」，隐含「离线最优」。
- **根因**：Belady 最优性只在等尺寸页假设下成立；本 workload 是变尺寸 KV 对象。
- **修复**：正式命名为 **Belady-style next-request-ordinal oracle**，并把 variable-size offline optimum 明确列为未计算的 Phase7 交付物。
- **验证**：`definitions.s2_naming` / `definitions.variable_size_offline_optimum` 字段。
- **对旧结论影响**：「Belady 上界未优于 S4，说明对象层级比
  next-use 更关键」这一推断**失去上界依据**。只能分母逐项报告
  S2/S4 数值，不能作策略排序。

### 5.5 rho sweep 变量混杂

- **症状**：历史 claim「压力越大 priority 越有价值」被当作普适规律。
- **根因**：rho sweep 通过增加对象数（15/20/27/40）扩大 working set，同时改变了 working-set composition 与 pressure。
- **修复**：在 `docs:PROJECT.md:2830-2831` 中显式记录该混杂，并把历史 claim 重新表述为「从 working-set 可容纳进入 oversubscribe 区间」的局部现象。
- **验证**：commit-bound rho 与对象数一一对应可查。
- **对旧结论影响**：单调性 claim 被撤回。

### 5.6 mean 与 p50/p95 混用

- **症状**：只报 mean speedup 会掩盖分布形态。
- **根因**：TTFT 呈双峰分布（多数 fast hit + 少数约 `280ms` miss），mean 的下降主要由 miss 数减少驱动。
- **修复**：强制分开报告 mean / p50 / p95。
- **验证**：p50 speedup 在 rho1.5/2/3 稳定在约 `1.42–1.45x`。
- **对旧结论影响**：结论方向不变，但解释必须说明「mean 改善主要来自 miss 计数变化」。

---

## 6. Lessons Learned

### 6.1 机制层

1. **workflow-aware priority 的描述性效果随 pressure 区间变化。**
   rho1.1 的 workflow-only 数值约为 `1.45x`，高 rho 时策略间分离缩小；
   但现有 rho sweep 同时改变对象组成，不能把这种形状归因于 pressure
   本身，也不能断言极端压力下的因果机制。
2. **本同步 H2D、host 不受压的 sequential canary 未观察到 prefetch
   mean 改善。** P2/P3 有真实 load/eviction 活动
   （`2,016`–`5,040` token），同时出现上述 p95 增加；这与 churn
   解释相容，但未完成因果隔离。
3. **workflow-only 下 S4 与 S2 出现描述性数值分离；all-reusable 下
   顺序并不稳定。** S2 不是真正的变尺寸离线最优，kind 标签也是合成的，
   因此不能从当前数据推出对象层级优于 next-use。

### 6.2 系统层

4. **priority metadata 必须与既有 `Req.priority` 解耦**，否则会与 SGLang 原有调度语义纠缠。
5. **prefetch victim 必须按对象边界 + dynamic suffix 后代原子处理**，否则会留下悬挂的部分对象。
6. **host tier 若大于工作集，prefetch 实验就失去区分度**；要得到有意义的 prefetch 性能结论，必须先制造 host 侧压力并使用异步 H2D。

### 6.3 测量层

7. **分母决定结论。** 同一批原始数据，在 workflow-only 与 all-reusable 两种分母下给出方向相同但**区分度完全不同**的结论。任何 cache policy 结果都必须声明「哪些请求计入」。
8. **零 GPU 重算是极高性价比的审计手段。** CL3 不重跑任何 GPU cell，仅从已提交 raw 重新聚合，就修正了两个关键结论（分母敏感性、prefetch 对照臂）。这要求 raw artifact 必须保存逐 request 记录。
9. **hit fraction 必须 per-request clamp。**

### 6.4 统计层

10. **独立复制单元是 server restart。** CL3 中 16 个 aggregate cell 里只有 2 个有 2 次 restart，其余为 1 次；这不足以支持任何策略排序。
11. **禁止「within noise」。** 未做统计检验时只能写「数值上几乎不可区分」。
12. **同一 trace 内的请求不是独立重复**，只能用于描述性 p50/p95。
13. **跨 Phase 的 speedup 不可直接排名**：Phase4 分母是 `dense / recovery target-only`，Phase5 分母是 `S0 LRU workflow TTFT / policy workflow TTFT`。

### 6.5 治理 / provenance 层

14. **Closeout 项必须有机器可读的完成证据**，否则会出现 FINDING-GAP-1 这类「以为做过其实没做」的缺口。
15. **命名即声称。** 把一个 next-request-ordinal 启发式命名为 "Belady oracle" 会隐含地引入一个未被证明的最优性上界。
16. **合成标签必须在结论中显式标注**，否则读者会把 `canonical_base / anchor / repair_metadata` 误读为真实近似对象层级的验证。

---

## 7. 最终结论

### 7.1 当前仍成立的结论

| 结论 | 作用域限定 | 证据 |
| --- | --- | --- |
| Phase5 全部 workflow hit 是 exact Radix/HiCache hit，无任何 lossy recovery | 全局 | §4.6 |
| S4 hierarchical + P0 off 是 Phase5 的默认配置 | exact-cache、sequential workflow | §4.2 |
| S4 在 workflow-only 口径下于 rho1.5/2.0/3.0 稳定优于 S0（`1.15x–1.32x`） | workflow-only 分母 | §4.5.2 |
| S1–S4 在 all-reusable 口径下相对 S0 均有 `1.09x–1.19x` 描述性改善 | all-reusable 分母；无排序能力 | §4.5.2 |
| 四种策略的 all-reusable p95 ratio 为 `0.9835–1.0094`，未形成一致分离，最大回归约 `0.94%` | 同上 | §4.5.2 |
| P1/P2/P3 相对同策略 P0 的 mean 为 `0.9885–1.0038`，未观察到稳定改善 | 本 host 容量与同步 H2D canary | §4.5.4 |
| P2/P3 有真实 load/eviction 活动；P2 p95 增加 `2.91%–4.89%`，P3 增加 `3.75%–3.98%` | 原始矩阵；无独立 restart | §4.4 |
| Restart validation 下 S4 vs S0 = rho1.5 `1.32–1.34x`、rho2.0 `1.11–1.15x` | 3 次独立 server 进程 | §4.3 |

### 7.2 被收窄的结论

| 原结论 | 收窄后表述 |
| --- | --- |
| 「S4 是唯一稳定优于 S0 的策略」 | 只在 **workflow-only** 口径成立；all-reusable 下 S1–S4 数值上几乎不可区分 |
| 「Belady 上界未优于 S4，说明对象层级比 next-use 更关键」 | S2 不是真正的上界；只能按 workflow-only/all-reusable 分列报告 S2/S4 数值，现有数据不能排序 |
| 「压力越大 priority 越有价值」 | 局部现象（从可容纳进入 oversubscribe 区间），非单调规律；rho sweep 混杂对象数与 capacity |
| 「S4 的 canonical/anchor/repair 层级被验证有效」 | kind 是**轮转合成标签**，不能外推为真实 approximate-object DAG 的验证 |
| 「prefetch 无收益」 | 收窄为：本功能/开销 canary 未观察到稳定 mean 改善；host tier 大于工作集、同步 H2D 与 restart 不足阻止一般化 |

### 7.3 被推翻的结论

| 被推翻结论 | 替代结论 |
| --- | --- |
| prefetch 结果可与 LRU 臂比较 | 矩阵中不存在 LRU 臂；唯一有效对照是同策略 P0 |
| 「S4 优势在 all-reusable 下消失」（一度出现的表述） | 消失的是相对 S1–S3 的**独特性**；相对 S0 仍有 `1.09x–1.18x` 改善 |
| Phase5 结论可代表 lossy recovery 场景 | Phase5 完全未执行 lossy recovery |

### 7.4 明确**不能**声称的内容

1. **不能**声称 Phase5 验证了近似 KV 恢复的调度价值——它只测了 exact Radix。
2. **不能**在任何口径下对 S1/S2/S3/S4 排序（独立 restart 不足）。
3. **不能**把 S2 称为 variable-size offline optimum 或「理论上界」。
4. **不能**把 S4 的 kind 层级结果外推到真实 approximate object DAG。
5. **不能**用 Phase5 的 rho sweep 论证「speedup 随 rho 单调」。
6. **不能**把 prefetch 矩阵当作性能结论（只能是功能/开销 canary）。
7. **不能**把 Phase5 speedup 与 Phase4 speedup 并列排名（分母不同）。
8. **不能**使用「within noise」这类未经检验的统计表述。
9. **不能**把 CL3 的零 GPU 重算说成新增实验证据——它是同一批 raw 数据的重新聚合。

---

## 8. 该结论能预测什么（可证伪预测与预注册问题）

| 编号 | 预测 | 证伪条件 |
| --- | --- | --- |
| P5-1 | 真实 cross-store metadata 是一个尚未隔离的自变量；替换轮转 kind 后 victim 序列与效果量可能改变，现有数据不给方向先验 | 若真实与合成 metadata 在多 restart 下产生等价 victim 序列和效果量，则 kind 构造在该 workload 中不是重要变量 |
| P5-2 | 固定对象集合、只调 capacity 的 rho sweep 用于判别收益曲线是单峰、单调还是无规律；现有 confounded sweep 不预注册方向 | 以预注册模型比较或置信区间选择支持的曲线；任一形状都不能由当前数据提前宣布 |
| P5-3 | 当前 all-reusable 效果量接近，补到每策略至少 3 个独立 restart 后可能仍无法稳定排序 | 若出现跨 restart 稳定且不重叠的分离，则应更新为可排序结论 |
| P5-4 | host 压力 + async H2D factorial 可区分同步传输开销与 eviction churn | 若异步化后 p95 回归消失，支持同步传输解释；若在受控 transfer 下仍随 churn bytes 增长，支持 eviction 解释 |
| P5-5 | 提高 approximate recovery 覆盖率会减少 fallback 对 policy 估计的混杂，但 S4−S0 的方向与幅度没有现成先验 | 在 matched coverage/fallback 率下比较策略；若差异仍随 fallback 率系统变化，需继续建模混杂而非宣称策略效应 |

> Phase7 的 W 矩阵给出了高 fallback 混合场景
> （`1.0021x`–`1.0442x`，R0 臂 `45.9%–72.1%` 请求走 dense
> fallback），但没有做 matched-coverage factorial，因此只能作为 P5-5
> 的后续实验动机，不能确认稀释因果。

### 8.1 对下一 Phase 的直接推论

- Phase6 必须先建立真实的 cross-store 底座，S4 才能接入真实 approximate 对象元数据（`docs:PROJECT.md:2553,2884`）。
- S1–S3 重新进入主矩阵的 gate（`docs:PROJECT.md:2820-2832`）：相对 S0 mean 改善 `≥5%` 且 p95 恶化 `≤5%`；或在 fallback/footprint/victim correctness 上提供 S4 没有的明确收益。**当前均未达标**，因此 Phase7 主矩阵只保留 S0/S4。
- P1–P3 重新进入主结果的 gate：mean 相对 P0 改善 `≥3%`、p95 不恶化、wasted/churn bytes 受控、不驱逐更早使用的高价值对象。**当前均未达标。**
- Phase5 的工程严谨性（setting 逐个运行、独立 server、请求串行、顺序随机、repeat 间 flush）被直接继承到 Phase7 runner 设计。

---

## 9. 局限、未完成项与 artifact / provenance 索引

### 9.1 局限

1. 单 GPU、单模型、合成 workflow trace；5 个 workflow 对象 + filler，不是真实 agent 会话。
2. kind 标签为轮转合成，非真实对象依赖图。
3. rho sweep 同时改变对象数与 pressure。
4. 绝大多数 cell 只有 1 次独立 server restart。
5. prefetch 的 host tier 大于工作集，且 H2D 为同步。
6. 只测 sequential（串行）请求，无并发干扰。
7. 只测 TTFT 与 hit fraction，不测吞吐、不测输出质量。

### 9.2 未完成项

| 未完成项 | 影响 |
| --- | --- |
| S1/S2/S3、rho1.1、rho3.0、prefetch 的独立 restart | 无法做策略排序 |
| 固定对象集合、只调 capacity 的 rho sweep | 无法验证单调性 |
| variable-size offline optimum | S2 缺少真正的上界参照 |
| 真实 approximate object DAG 上的 S4 验证 | kind 语义无法外推 |
| async H2D + host 侧真实压力下的 prefetch | P1–P3 无法进入性能结论 |
| Phase5 scheduler × Phase4 recovery 的组合矩阵 | 属 Phase6/7；后因 `practical=NONE` 大部分被跳过 |
| 并发/多租户下的调度干扰 | 全流程未处理 |

### 9.3 Artifact / provenance 索引

| artifact | 位置 | 状态 |
| --- | --- | --- |
| `sm75-scheduler-matrix.json` | `impl:benchmark/approx_kv/results/phase5-scheduler/` | `authoritative_historical` |
| `sm75-prefetch-matrix.json` | 同上 | `diagnostic/proxy`（功能/开销 canary） |
| `sm75-restart-validation.json` | 同上 | `authoritative_historical`（仅 rho1.5/2.0 × S0/S4） |
| `cl3-phase5-recalculation.json` | `impl:benchmark/approx_kv/results/phase6/` | **`最终权威`（分母修正）**，`raw_sha256=17f010b7…` |
| `run_phase5_scheduler_matrix.py`（810 行） | `impl:benchmark/approx_kv/` | 实现证据 |
| `run_cl3_phase5_recompute.py`（431 行） | `impl:benchmark/approx_kv/` | 实现证据 |
| `workloads.py` | `impl:benchmark/approx_kv/` | `CacheObjectKind` 与轮转赋值证据 |
| `PHASE5_RECALCULATED_METRICS.json` | `docs:research/` | 更早的 closeout 重算尝试；`diagnostic/proxy`，不作权威口径 |
| `phase4_phase5_closeout.py` / `test_phase4_phase5_closeout.py` | `docs:research/` | closeout 计算脚本与测试 |

Phase6 的 `RESULT_MANIFEST.json`（`impl:benchmark/approx_kv/results/phase6/`）为 CL3 artifact 提供 file→commit 映射，`--check` 通过 `48/48`。

### 9.4 Phase5 结论在 Phase7 中的采纳情况

| 结论 | Phase7 处置 |
| --- | --- |
| S4 作为对照策略 | **采纳**：Phase7 主矩阵为 `R0 W × S0/S4`（`docs:PROJECT.md:583`） |
| S1–S3 | **排除**：未通过 revalidation gate |
| P1–P3 | **排除**：未通过 gate；Phase7 host/prefetch/async 轨道预算为 0 |
| S4 的 kind 语义 | 需在 Phase6/7 重新接入 cross-store 真实 metadata |
| exact-only prefetch 回归 | 因 `practical family = NONE` 定稿，若执行只能标为「Phase5 回归 canary」（`docs:PROJECT.md:974`） |

---

## 10. 与其它阶段报告的关系

| 关系 | 说明 |
| --- | --- |
| ← [Phase4](PHASE4_RECOVERY_METHODS_REPORT.md) | Phase5 刻意不使用任何 Phase4 恢复路径，以隔离 scheduler 效应 |
| → [Phase6](PHASE6_CROSS_STORE_SUBSTRATE_REPORT.md) | Phase6 建立 exact/approximate 共预算底座，使 S4 能接入真实对象元数据；CL3 是 Phase6 期间补齐的 |
| → [Phase7](PHASE7_INTEGRATED_EVALUATION_REPORT.md) | Phase7 W 矩阵沿用 S0/S4，结论为 `INCONCLUSIVE/DESCRIPTIVE` |
| → [跨阶段总报告](PHASE4_TO_PHASE7_SUMMARY.md) | 汇总 workflow-only vs all-reusable 这一关键方法论反转 |
