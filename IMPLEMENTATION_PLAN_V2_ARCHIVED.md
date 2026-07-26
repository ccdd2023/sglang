# 实施计划 V2（Archived）：Phase 6 Cross-Store Recovery × Scheduling

> 版本：V2
>
> 状态：Archived / Read-only
>
> 最后更新：2026-07-24T17:03:19-07:00
>
> 归档时间：2026-07-25T10:27:12-07:00
>
> 当前版本：[`IMPLEMENTATION_PLAN_LATEST.md`](IMPLEMENTATION_PLAN_LATEST.md)
>
> 注意：正文保留归档当时的“latest/current”等历史措辞，仅用于还原V2，不代表当前状态。
>
> 取代版本：[`IMPLEMENTATION_PLAN_V1_ARCHIVED.md`](IMPLEMENTATION_PLAN_V1_ARCHIVED.md)

## 1. 文档职责

- 本文件在V2归档前曾是当时最新、可执行的实施计划。
- `PROJECT.md` 是项目事实、决策和结果的最终事实来源。
- `HANDOFF.md` 是新会话所需的当前快照。
- `TRACKING.md` 是不可改写的时间线。
- 若本文件与 `PROJECT.md` 冲突，以 `PROJECT.md` 中较新的明确决策为准，并立即同步修正本文件。

后续发生重大计划变更时：

1. 将当前版本保存为带版本号的 archived 文件；
2. 提升本文件内部版本号；
3. 保持文件名 `IMPLEMENTATION_PLAN_LATEST.md` 不变，作为稳定入口。

## 2. 已完成前置阶段

- Phase 1–3：SM75、pressure harness、approximate KV common core 已完成。
- Phase 4：R0/R1/R2/R4/R5 已完成当前 SM75 recovery 筛选；R3 defer。
- Phase 5：S0–S4、P0–P3 exact-cache scheduler isolation 已完成。
- Phase 5 最终默认：`S4 hierarchical + P0 off`。

Phase 5 没有执行有损 KV 恢复，因此不能直接代表 Phase 4 recovery objects 与 scheduler 的组合效果。

## 3. Phase 6 为什么重写

旧版 Phase 6 计划直接组合：

```text
Phase 4前两条 recovery
× S0-S4
× P1-P3
```

该设计存在以下问题：

- Phase 5 的 S4 只管理 exact Radix，不管理 approximate store。
- exact 与 approximate objects 尚未在同一预算中竞争。
- Phase 5 rho sweep改变了对象集合，存在 composition 混杂。
- S1–S3 在 exact 高压下没有稳定收益，但尚未在 lossy store 上重新验证。
- P2/P3 当前为同步 H2D，Phase 5 没有稳定收益且 p95 更差。
- Phase 4 target-only、combined cost 与 production readiness 不一致。
- R2/R5 仍依赖 precomputed adapter。

因此 Phase 6 必须先完成 cross-store 数据面，再做组合实验。

## 4. Phase 6 核心目标

在以下条件全部一致时：

- 同一固定 sequential workflow；
- 同一逻辑对象集合；
- 同一对象顺序和类别；
- 同一 GPU/host memory budget；
- 同一 warm-up/formal protocol；

让 exact Radix 与 approximate source/adapter/anchor objects 真实竞争，比较：

```text
dense
vs exact S0/S4
vs lossy recovery S0/S4
vs S4 + HiCache demand load
```

当前仍只做性能和系统机制结论，不扩展 semantic quality claim。

## 5. 固定术语与 baseline

| ID | 配置 |
| --- | --- |
| D0 | dense，无 reuse |
| E0 | exact cache + S0 LRU + GPU-only + P0 |
| E4 | exact cache + S4 + GPU-only + P0 |
| R-S0 | 当前 lossy recovery + S0 + GPU-only + P0 |
| R-S4 | 当前 lossy recovery + S4 + GPU-only + P0 |
| H4 | exact cache + S4 + HiCache + P0 |
| RH4 | 当前 lossy recovery + S4 + HiCache + P0 |

Phase 4 speedup 与 Phase 6 scheduler speedup必须分开：

```text
Phase 4 recovery speedup = dense target prefill / recovery target
Phase 6 scheduler speedup = 同一 recovery 下的 S0 workflow / S4 workflow
```

## 6. Recovery 候选分类

| 路径 | Phase 6 定位 |
| --- | --- |
| R0 Raw+RoPE / k0 | speed-only ceiling，固定保留 |
| R1 EPIC k32 | genuine in-request practical candidate |
| R2 CacheBlend 1% | precomputed oracle；online lifecycle完成前不能成为 practical winner |
| R4 KVCOMM | canonical/anchor/delta object hierarchy 诊断 |
| R5 CacheTune | 当前被R2支配且combined margin较小，不进主矩阵 |
| R3 Cache-Craft | 继续 defer |

R0 与 R1-k0 是同一物理恢复路径，不得作为两个独立候选。

## 7. P6-0：冻结公平性合同

- workflow固定为：

```text
Architect -> Coder -> Debugger -> Coder -> Debugger
```

- 固定5个active workflow objects。
- 固定live/dead filler的ID、类别、顺序和逻辑内容。
- body主点：`1024/2048`。
- header：`64`。
- long source segment：`<=512 tokens`。
- 所有rho使用相同对象集合，只改变capacity。

主结果采用matched-state口径：

- approximate target不写回exact Radix；
- exact baseline使用变化的target suffix；
- 避免后续轮次退化成完整exact hit。

另保存native-system口径，展示真实写回行为。

## 8. P6-1：cross-store对象模型与原子分配

### 8.1 对象 DAG

定义并实现：

- exact stage variant/bundle；
- canonical raw segment chain；
- EPIC repair state；
- R2 fresh adapter；
- R4 canonical/anchor/delta；
- host copy。

### 8.2 统一 metadata

approximate store/handle必须具备：

- object kind；
- next-use；
- measured/synthetic recovery value；
- device/host residency；
- dependency edges；
- generation；
- lease/pin；
- eviction/demotion eligibility。

### 8.3 原子 allocation

统一流程：

1. reserve目标空间；
2. 同时枚举exact与approx victims；
3. 按S0/S4选择victims；
4. 原子evict/demote；
5. allocation成功后commit；
6. 失败则rollback或dense fallback。

硬约束：

- pin、lease、in-flight H2D、active recovery reservation不可驱逐；
- R0 segment chain按连续可恢复前缀计价；
- 禁止保留无法产生恢复收益的孤儿suffix；
- reset/fallback后generation、slot、lease、host ref归零。

## 9. P6-2：固定 workload 与 capacity feasibility pilot

- 保持40个逻辑对象和类别固定。
- 只允许一次性校准filler长度。
- 后续rho只通过GPU KV capacity变化产生。
- 优先使用`max_total_tokens`；接口不稳定时才使用`mem_fraction_static`。

pilot：

```text
R0 + S0
body = 2048
rho = 1.1, 1.5, 2.0, 3.0
```

必须记录：

- `rho_logical`；
- `rho_physical_exact`；
- `rho_physical_approx`；
- exact/approx device pages；
- exact/approx host pages；
- allocator headroom。

若固定40对象无法安全达到最低rho，则重新一次性校准对象长度，不能按setting改变对象组成。

## 10. P6-3：恢复路径重新筛选

候选：

```text
R0
R1-k32
R2-1%
R4 diagnostic
```

setting：

```text
body = 1024, 2048
rho = 1.5, 2.0
scheduler = S0
tier = GPU-only
prefetch = P0
warmup = 1
formal = 2 independent rounds
```

要求：

- R2必须把fresh preparation放入计时。
- R2未online化时标为precomputed oracle。
- R4报告setup与N=`1/2/4/8` amortized workflow cost。
- intent-to-treat target必须包含fallback。

筛选结果：

1. R0固定保留为speed ceiling；
2. 另选一个production-practical winner；
3. R2只有online lifecycle完成后才可成为practical winner。

## 11. P6-3.5：Scheduler Revalidation Gate

S1–S3不取消，但必须重新获得资格。

### 11.1 Practical recovery gate

对最终practical recovery运行：

```text
body = 2048
rho = 1.5, 3.0
scheduler = S0-S4
warmup = 1
formal = 2
server restart = 1
```

### 11.2 R4 object-hierarchy诊断

```text
R4
body = 2048
rho = 2.0
scheduler = S0-S4
```

检查canonical/anchor/delta真实victim sequence。

### 11.3 晋级条件

S1/S2/S3满足任一条件才进入主矩阵：

- mean相对S0改善`>=5%`，且p95恶化`<=5%`；
- fallback率、physical footprint或victim correctness明显优于S4。

未通过的策略保留诊断结果，但不进入大矩阵。

## 12. P6-4：主 Scheduler 组合矩阵

默认：

```text
两条入选 recovery
× S0/S4
× GPU-only
× P0
```

若P6-3.5中有策略晋级，只加入通过门槛的策略。

pressure：

- body1024：rho=`1.5/2.0`，作为control；
- body2048：rho=`1.1/1.5/2.0/3.0`。

共享baseline按body/rho只测一次：

- D0；
- E0；
- E4。

S2若未晋级，仍在body2048的rho1.1和rho3保留victim-sequence诊断。

setting顺序采用blocked random或Latin-square，一次只运行一个GPU server。

## 13. P6-5：HiCache Demand-Load 矩阵

只对最终practical winner执行：

```text
scheduler = S4
prefetch = P0
tier = GPU + HiCache
body = 2048
rho = 1.5, 2.0, 3.0
```

baseline：H4。

要求：

- exact与approx使用相同GPU/host budget；
- 使用相同write policy；
- approximate source真实demote到host；
- 后续成功demand load或明确dense fallback；
- 不允许pinned object或host ref泄漏。

## 14. P6-5.5：Prefetch Revalidation Gate

P1–P3不取消，但必须重新获得资格。

运行：

```text
final practical recovery
S4 + HiCache
body = 2048
rho = 2.0, 3.0
prefetch = P0-P3
warmup = 1
formal = 2
server restart = 1
```

若仍为同步等待H2D：

- P1–P3只作功能、安全和churn canary；
- 不发布正式prefetch性能claim。

只有实现真正async H2D，并可与当前stage执行或GPU idle窗口重叠时，才进入正式性能比较。

P1/P2/P3晋级条件：

- mean相对P0改善`>=3%`；
- p95不恶化；
- wasted/churn bytes受控；
- 不驱逐next-use更早的高价值对象。

未通过时，P0继续作为最终默认。

## 15. P6-6：Metrics 与统计

### 15.1 客户端

- workflow wall-clock；
- mean/p50/p95 TTFT；
- Architect/Coder/Debugger分角色TTFT；
- 每restart paired delta。

### 15.2 Cache分类

- exact GPU hit；
- approximate GPU hit；
- host demand load；
- dense fallback。

### 15.3 Recovery

- source/register preparation；
- intent-to-treat target TTFT；
- effective contiguous recovered tokens；
- repair tokens/layers；
- fallback reason；
- N=`1/2/4/8` amortized latency；
- break-even reuse count。

### 15.4 Memory

- exact/approx resident pages与bytes；
- object-kind victim count；
- eviction/demotion bytes；
- H2D bytes/time；
- wasted objects；
- orphan segments必须为0。

### 15.5 Warm-up与cold-start

每个setting：

1. 保存cold-start/warm-up原始结果；
2. warm-up不进入formal统计；
3. 清空exact、approx、HiCache、metadata、generation；
4. 运行formal；
5. formal repeat之间再次完整清空。

## 16. P6-7：Restart 与最终有效性门

- screening：1个server process，每格formal2。
- final primary cells：3个独立server restart，每次formal2。
- server restart是独立统计单位。
- 不把同一trace中的请求当成独立实验样本。

最低有效性：

- 请求完成率100%；
- 无OOM、allocator corruption、stale handle、double free；
- exact+approx+host pool accounting通过；
- 至少一次approx source真实evict/demote；
- 后续成功recovery、load或明确dense fallback；
- S4相对同一recovery的S0，在至少两个rho>=1.5点mean改善`>=5%`；
- p95恶化不得超过`5%`。

若不满足，最终结论必须写为：

```text
S4对exact cache有效，
但对lossy approximate store没有可证明收益。
```

并停止扩大矩阵。

## 17. 预期交付

- unified exact/approx policy实现；
- cross-store allocator与rollback测试；
- fixed-workload capacity runner；
- recovery re-screen JSON；
- scheduler revalidation JSON；
- S0/S4主矩阵；
- HiCache demand-load矩阵；
- prefetch revalidation JSON；
- 三restart compact manifests；
- cold-start与steady-state分离结果；
- speed ceiling、precomputed oracle、practical winner三类结论。

## 18. 当前状态

- V2计划已完成并冻结。
- Phase 6尚未创建实现分支。
- 尚未修改prototype。
- 尚未启动Phase 6 GPU实验。
