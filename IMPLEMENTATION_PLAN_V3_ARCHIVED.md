# 实施计划 V3（Archived）：Phase 6 Cross-Store Recovery × Scheduling

> 版本：V3
>
> 状态：Archived / Read-only
>
> 最后更新：2026-07-25T10:53:22-07:00
>
> 归档时间：2026-07-25T23:21:19-07:00
>
> 当前版本：[`IMPLEMENTATION_PLAN_LATEST.md`](IMPLEMENTATION_PLAN_LATEST.md)
>
> 注意：正文保留归档当时的current/latest措辞，仅用于还原V3。
>
> 取代版本：[`IMPLEMENTATION_PLAN_V2_ARCHIVED.md`](IMPLEMENTATION_PLAN_V2_ARCHIVED.md)

## 1. 文档职责与版本规则

- `PROJECT.md`：项目事实、结果和明确决策的最终事实来源。
- 本文件：当前最新、可执行的实施计划。
- `CONSOLIDATED_PHASE4_PHASE6_REVIEW.txt`：Phase 4/5审计、R2/R5 corrected rerun及双模型review的完整证据。
- `HANDOFF.md`：当前快照和下一步。
- `TRACKING.md`：不可改写的时间线。
- V1/V2 archive只用于历史追溯，不再作为执行依据。
- 若本文件与`PROJECT.md`中更新、明确的事实或决策冲突，以`PROJECT.md`为准，并立即同步本文件。

重大计划变更时：

1. 将当前latest保存为新的版本化archive；
2. 提升本文件内部版本号；
3. 保持文件名`IMPLEMENTATION_PLAN_LATEST.md`不变；
4. 通过双模型独立review与交叉consolidate后才标记为`Current / Latest`。

## 2. V3的证据基线

### 2.1 已完成阶段

- Phase 1–3：SM75、pressure harness、approximate KV common core已完成。
- Phase 4：R0/R1/R2/R4/R5已完成原始筛选；R3 defer。
- Phase 5：S0–S4、P0–P3 exact-cache scheduler isolation已完成。
- Corrected Phase 4 key rerun：
  - R2实现`c73c9c5ab`，结果`e36f1529b`；
  - R5实现`46d1f85c2`，结果`abcedd62b`；
  - 两条结果均已push并核对remote SHA。

### 2.2 Corrected R2/R5结果

| 路径 | body | target-only | adapter-combined | request-path | recovery-object lifecycle |
| --- | ---: | ---: | ---: | ---: | ---: |
| R2 | 1024 | `1.659x` | `0.441x` | `0.526x` | `0.324x` |
| R2 | 2048 | `2.044x` | `0.407x` | `0.434x` | `0.246x` |
| R5 | 1024 | `1.614x` | `0.449x` | `0.527x` | `0.327x` |
| R5 | 2048 | `1.978x` | `0.406x` | `0.433x` | `0.246x` |

固定结论：

- target-only恢复收益真实存在；
- 原R2 body2048 `1.14x`和R5 `1.04x` single-use combined正收益已被推翻；
- R2/R5仍是precomputed target oracle，不是production-practical candidate；
- R2修复1% tokens，R5修复约8.3%；repair ratio是target差异的高置信解释之一，但未做matched-ratio因果实验；
- 不再使用“R5被R2性能支配”的表述；
- 首token一致率1.0只是不发生粗暴损坏的guardrail，不是semantic correctness。
- corrected rerun仍有pressure残差：recovery setup额外留下evictable scratch branches，R5两臂的filler内容也未完全配对；这些不改变single-use负结论，但限制精细排序。

### 2.3 Phase 5结论边界

- Phase 5只验证exact Radix，不含任何Phase 4 lossy recovery。
- workflow-only与all-reusable/full-trace分母会改变S1–S4排序。
- `S4 > S0`可成立，但“S4唯一稳定优于S0”和“Belady未优于S4”必须限定分母。
- S4的kind标签为synthetic轮转标签，不能当作真实approximate-object DAG证据。
- Phase 5 rho sweep改变对象数量和dead/live组成，不能证明固定workload下的rho单调性。
- prefetch矩阵host tier大于working set且H2D同步，只能作为功能/开销canary。

## 3. Recovery候选的V3定位

| 路径 | V3定位 | 主矩阵资格 |
| --- | --- | --- |
| R0 Raw+RoPE / R1-k0 | speed-only ceiling；同一物理恢复机制 | 固定control |
| R1 EPIC | 唯一genuine in-request candidate family | 通过P6-3a后才可晋级practical |
| R2 CacheBlend | precomputed oracle family的默认代表 | oracle/diagnostic，不是practical |
| R5 CacheTune | 与R2高度冗余的hardware-controller oracle point | 默认不进主矩阵；不以性能支配为理由 |
| R4 KVCOMM | canonical/anchor/delta hierarchy diagnostic | 独立诊断，不参与practical winner排序 |
| R3 Cache-Craft | order-sensitive safety未覆盖，继续defer | 不阻塞Phase 6 |

R2作为precomputed family默认代表的原因是合同简单、repair ratio显式且已有1%历史轴，不是因为已证明机制优于R5。

## 4. 两级Entry与前置门

V3区分：

### Implementation Entry

允许创建Phase 6分支并执行P6-0/P6-1。必须完成：

- G0全部0-GPU收尾；
- 本V3双模型Plan Review及主会话disposition；
- 无主会话标记为`accepted-blocking-P0`的open finding。

Implementation Entry前不要求新的Phase4/5 GPU重跑。

### Experiment Entry

允许进入P6-3 recovery筛选、P6-4 scheduler及P6-5 host矩阵。必须完成：

- P6-2 fixed-workload/capacity pilot；
- P6-3a R0/R1 candidate-family qualification；
- chunk配置门执行或显式waive；
- 对应里程碑双模型review。

### G0：0-GPU结果与schema收尾（必须）

1. 给旧R2/R5 key-cell artifact加`superseded_by`指针。
2. 完成零GPU历史更正：
   - Phase4“全矩阵”改为OAT slices；
   - 记录R5 final-SHA `rho_sweep_points`为空；
   - R1旧in-request矩阵标注旧SHA、无压力和不同轴；
   - 建authoritative raw manifest并标记R1 rho0.9 stale raw。
3. 同时定义并记录：
   - `rho_logical_demand`：请求/逻辑working-set相对capacity的需求比；
   - `rho_resident`：`(kv_used + kv_evictable) / capacity`；
   - `rho_host`：host working set / host capacity。
4. 将`full_lifecycle`统一改称`recovery_object_lifecycle`，并列出排除项：
   - pressure filler构造；
   - server启动/停止；
   - namespace cleanup；
   - unrelated workload wall-clock。
5. 从现有raw生成：
   - paired ratio/delta；
   - per-restart表；
   - cold-start样本；
   - formula N=`1/2/4/8/16` amortization；
   - Phase 5 all-reusable/full-trace/per-role重算。
6. R2 fallback counter不可用时必须写`indirectly_verified`，不能把`None`当显式0。
7. artifact保存：
   - server plugin env；
   - raw SHA256；
   - server log路径；
   - CPU test命令、实现SHA和pass count。
8. 增加CPU测试：
   - 四类ledger恒等式；
   - pressure补偿公式；
   - unique extra-key scratch branch在无per-round flush时的生命周期/GC。
9. 为authority文档仓库定义并落实版本化备份：
   - 先核实目标remote与`ccdd2023`权限；
   - 未配置remote前，Implementation Entry保持blocked。

### P6-3a：R0/R1 candidate family确认（Experiment Entry前必须GPU）

目的：在P6-3主筛选前确认candidate family及其k值，而不是沿用旧短上下文/OAT结果。它不阻塞P6-0/P6-1。

固定合同：

```text
paths = R0/R1-k0, R1-k4, R1-k8, R1-k16, R1-k32
body = 1024, 2048
header = 64
rho_logical_demand = 2.0
tier = GPU-only
scheduler = S0
prefetch = P0
restart = 1 screening; k0和screening winner补至3
warmup = 1 per arm
formal = 2 per arm
```

要求：

- cumulative causal-prefix source registration；
- independent extra-key materialization namespace；
- same-server paired dense；
- same filler manifest；
- pressure预算扣除`setup_used + setup_evictable`；
- target-only、adapter/repair preparation、request-path、recovery-object lifecycle；
- N=`1/2/4/8`解析投影；
- 性能formal固定`max_new_tokens=1`；
- 独立quality canary使用`max_new_tokens>=8`，记录逐token一致率、decode eviction delta；可用时记录top-k/logprob差异；
- 显式fallback counter、eviction、pool reset、raw SHA和server env。

输出：

- R0/R1-k0机制等价检查；
- body2048 practical k选择；
- 若所有k在request-path上均无收益，R1仍可保留为机制diagnostic，但不得称practical winner。

promotion规则必须在执行前冻结：

1. explicit fallback、pool invariant和8-token guardrail全部通过；
2. body2048/rho2的paired request-path speedup在至少2/3 restart中`>1.0x`；
3. all-reusable p95恶化`<=5%`；
4. 从通过者中选择median paired request-path speedup最高的k；
5. 若差异`<=2%`，优先repair token更多且p95更低者；
6. 无通过者则`practical family = NONE`。

### P6-3b：chunk配置门（Experiment Entry前执行或显式waive）

目的：冻结Phase 6的`chunked_prefill_size`配置并披露它对recovery speedup的影响。复用P6-3a已有`body1024/chunk1024` raw，避免重复cell。

```text
arms = dense, R1-k0, selected R1-k
body = 768, 1024
chunked_prefill_size = 1024, 4096
header = 64
rho_logical_demand = 2.0
restart = 2 screening; 3 if配置选择改变primary claim
formal = 2
```

若显式waive，所有后续性能结论必须写：

> 在当前SM75、预注册chunked-prefill配置下测得，不外推到其他chunking。

不得写成与chunking无关的“长body天然更适合恢复”。waive必须由主会话记录理由和风险。

### G3：双模型Plan Review与Milestone Review（必须）

使用：

- GPT-5.6 Sol，Max Thinking，long context；
- Claude Opus 5，Max Thinking，long context。

流程：

1. 两模型独立阅读G0及适用的P6-3a/P6-3b代码、raw和manifest；
2. 各自产出atomic finding register；
3. 互换全文，分别交叉consolidate；
4. 主会话形成最终disposition；
5. reviewer自行标记P0不自动形成blocker；
6. 只有主会话disposition标记为`accepted-blocking-P0`的finding阻塞；
7. 两模型分歧默认按更严建议处理，主会话可显式override，但必须在TRACKING记录风险；
8. 模型不可用时不得静默换模型，必须记录fallback配置并重新确认review有效性。

### 不阻塞Phase 6的条件性确认

| 不确定项 | 是否必须 | 触发条件 |
| --- | --- | --- |
| R2 vs R5 matched repair ratio | 否 | 仅当仍要发布两机制性能排序 |
| corrected R2/R5 rho1.1/3 | 否 | 仅当要发布rho稳健性claim |
| R2显式fallback counter GPU补点 | 否 | 仅当cached-token间接证据不够 |
| Phase 5 fixed-40 exact-only重跑 | 否 | 可并入P6-2 fixed-workload pilot |
| prefetch性能重跑 | 否 | 必须等真实async与host-pressure设计 |
| R3深层实现 | 否 | 继续作为未覆盖轴记录 |

## 5. 双模型Review制度

双模型review不是只在V3定稿时使用，而是Phase 6的固定质量门。

### 5.1 Reviewer配置

- Reviewer A：GPT-5.6 Sol，Max Thinking，long context。
- Reviewer B：Claude Opus 5，Max Thinking，long context。

### 5.2 必须触发review的里程碑

1. 本V3计划定稿；
2. P6-1 cross-store allocator实现完成；
3. P6-2 capacity pilot完成；
4. P6-3 recovery re-screen完成；
5. P6-4 scheduler主矩阵完成；
6. P6-5 HiCache feasibility及矩阵完成；
7. 最终结论和论文表格冻结。

### 5.3 Review输出合同

每次review必须包含：

- evidence paths / commit SHA / raw SHA；
- fact、interpretation、inference分离；
- atomic findings；
- resolved / partially resolved / open状态；
- 是否阻塞下一阶段；
- 两模型分歧及消歧证据；
- 主会话最终disposition。

review blocker规则：

- 只有主会话标记为`accepted-blocking-P0`的finding阻塞；
- reviewer自行标P0只表示建议严重度；
- 主会话override必须记录理由、证据和承担风险；
- follow-up只需review新增delta，不重复读取无变化的全部raw；
- 两模型均不可用时里程碑暂停；单模型fallback必须显式记录且不得冒充双模型review。

## 6. 固定术语、baseline与成本ledger

### 6.1 Baseline

`paired launch block`定义为：

> 同一`(body, rho, restart)`下，以相同image/model/capacity目标/server-seed计划连续启动的一组相邻server进程。

不同eviction policy、HiCache开关、chunked-prefill或capacity需要独立server进程，不能声称在同一进程内配对。配对发生在同一launch block内。

| ID | 配置 |
| --- | --- |
| D0 | dense，无reuse |
| E0 | exact + S0 + GPU-only + P0 |
| E4 | exact + S4 + GPU-only + P0 |
| R0-S0 | speed ceiling + S0 + GPU-only + P0 |
| R0-S4 | speed ceiling + S4 + GPU-only + P0 |
| R1-S0 | 通过P6-3a的practical family + S0 + GPU-only + P0 |
| R1-S4 | 通过P6-3a的practical family + S4 + GPU-only + P0 |
| O2 | R2 representative precomputed oracle |
| H4 | exact + S4 + HiCache + P0 |
| RH4 | practical recovery + S4 + HiCache + P0 |

### 6.2 成本ledger

原子字段：

- `source_preparation_ms`
- `target_adapter_preparation_ms`
- `materialize_ms`
- `register_copy_ms`
- `transfer_ms`
- `seed_head_ms`
- `post_pressure_reseed_ms`
- `target_only_ms`
- `cold_start_ms`

逐路径定义：

- R0/R1：`source_preparation_ms = canonical source materialization + registration`；
- R2/R5：额外记录`target_adapter_preparation_ms = fresh materialization + fresh registration`；
- R4：`source_preparation_ms = canonical + anchor + delta setup`。

派生字段：

```text
request_path_ms =
  seed_head
  + target_adapter_preparation
  + post_pressure_reseed
  + transfer
  + target_only

recovery_object_lifecycle_ms =
  source_preparation
  + request_path

amortized_ms_N =
  (source_preparation + target_adapter_preparation) / N
  + seed_head
  + post_pressure_reseed
  + transfer
  + target_only
```

`protocol_overhead_ms`只有在同时测得：

- comparable one-token control；
- server-side KV-copy时间；
- register request elapsed；

时才可报告；否则必须为`not_measured`，不得把整个register耗时视为可删除开销。

必须输出`amortized_ms_N1/N2/N4/N8`和`break_even_reuse_count`。

禁止把target-only直接称为end-to-end。

### 6.3 Correctness guardrail

- exact cache优先；
- controlled reconstruction；
- 不满足能力/不变量时dense fallback；
- first-token guardrail；
- 至少8-token greedy一致率；
- 可用时记录top-k/logprob差异；
- 不将这些guardrail升级为semantic correctness或代码任务质量claim。

## 7. P6-0：公平性、schema与统计冻结

在任何Phase 6 GPU矩阵前完成。

### 7.1 固定workload

- workflow：`Architect -> Coder -> Debugger -> Coder -> Debugger`；
- 固定40个逻辑对象；
- 固定object ID、类别、顺序、dead/live身份；
- 至少两档对象长度；
- body主点`1024/2048`；
- header`64`；
- segment`<=512`；
- rho只通过capacity改变。

### 7.2 Matched-state

primary方案固定为：

1. 每个warmup/formal round开始前完整清空并重建同等source状态；
2. 每个round只发送一个measured target；
3. approximate target不写回exact；
4. exact baseline只使用本round预先构造的exact source，不依赖上一round target写回。

不再允许每条路径自行选择matched-state方案。只改变final suffix不能阻止body exact hit。

native-system口径必须给approximate写回加provenance/taint，后续不得计为exact hit。

### 7.3 Pressure对齐

- same filler manifest；
- 同一body/rho/restart block内配对；
- 预算同时扣除`setup_used`和`setup_evictable`；
- 记录`rho_logical_demand/rho_physical_demand/rho_resident/rho_host`；
- 记录resident和temporary peak tokens/pages/bytes。

rho固定拆分：

- `rho_logical_demand = logical reusable working set / configured capacity`；
- `rho_physical_demand = requested physical pages including representations / capacity`；
- `rho_resident = sampled (used + evictable) / capacity`；
- `rho_host = host working set / host capacity`。

不得只写`rho`。

### 7.4 Schema

必须保存：

- run ID；
- source/result commit；
- raw SHA256；
- server argv和plugin env；
- machine/GPU/image；
- requested/observed capacity；
- `crosses_chunk_boundary`；
- `segment_count`；
- warmup/formal/restart；
- test命令和结果；
- cold-start与steady-state；
- per-request、per-trace、per-restart统计。

### 7.5 Flush/reset协议

每个setting：

1. 保存cold-start样本；
2. 运行独立warmup；
3. warmup后清空exact、approx store、HiCache、metadata、generation、host refs和extra-key scratch；
4. 验证pool/accounting；
5. 运行formal repeat；
6. 每个formal repeat之间再次执行同样清空与验证；
7. 最后执行post-setting reset invariant。

### 7.6 Cache outcome分类

每请求只归入一个主类别：

- exact GPU hit；
- approximate GPU recovery；
- host demand load；
- dense fallback。

带approx provenance/taint的对象永远计为approximate，不得计为exact。

### 7.7 客户端与memory指标

- full-trace workflow wall-clock；
- all-reusable request TTFT；
- workflow-only SLA TTFT；
- Architect/Coder/Debugger per-role TTFT和miss；
- victim count/tokens/bytes by object kind；
- evict/demote bytes；
- H2D/D2H bytes/time；
- wasted/churn bytes；
- orphan count必须为0。

### 7.8 统计

- primary estimator预注册；
- 同时报告paired delta/ratio；
- 全局统一规则：primary p95使用all-reusable分母，任何晋级判定要求p95恶化`<=5%`；
- workflow-only作为SLA次级视图；
- 报告miss count；
- screening单restart，晋级后3 restart；
- server restart主要是时序replicate；若改变workload seed，策略间必须paired；
- 明确多重比较数量和区间估计。
- 同一trace内的请求不是独立实验样本。

## 8. P6-1：Cross-store对象模型与原子分配

### 8.1 对象DAG

定义：

- exact stage variant/bundle；
- R0 canonical segment chain；
- R1 repair state；
- R2/R5 precomputed adapter family；
- R4 canonical/anchor/delta；
- host copy；
- materialization scratch branch。

每条边必须定义：

- dependency；
- orphan条件；
- atomic victim closure；
- demotion/load-back语义；
- provenance。

### 8.2 Approx store预算与反向压力

现有`max_records`不足以代表物理预算。必须增加：

- token/page/byte budget；
- exact allocation可驱逐/demote approx；
- approx allocation可驱逐exact；
- host demotion触发器；
- temporary reservation accounting。

### 8.3 原子协议

1. reserve；
2. 枚举exact+approx victims；
3. 按S0/S4选victim closure；
4. 原子evict/demote；
5. allocation成功后commit；
6. 失败rollback或dense fallback。

不可驱逐：

- pin/lease；
- active request；
- in-flight H2D/D2H；
- recovery reservation；
- target正在使用的dependency closure。

统一policy语义：

- S0使用跨exact/approx/host统一`event_ordinal`，不得混用独立wall-clock；
- S4在实现前冻结class order、value定义、dependency closure和tie-break。

lock order固定为：

```text
global policy state
-> object metadata
-> host transfer state
-> device allocator
```

failure injection必须覆盖reserve后、victim选择后、evict后、allocation后、commit前。rollback必须说明是否恢复victim内容；无法恢复时该路径只能标记INVALID并完整reset。

### 8.4 Exit gate

GPU-free failure injection必须证明：

- generation/lease/slot/host-ref归零；
- no double free；
- no stale handle；
- orphan=0；
- rollback后exact/approx/host accounting恢复。
- 多对象、无per-round flush条件下extra-key scratch branch可GC；
- ledger恒等式与pressure补偿公式CPU测试通过。

通过双模型review后才能进入P6-2。

## 9. P6-2：固定workload与capacity feasibility

pilot至少覆盖：

```text
R0 footprint
R2-like ~2x footprint
R4-like ~5x footprint
body = 2048
rho_logical_demand = 1.1, 1.5, 2.0, 3.0
```

优先`max_total_tokens`，同时记录requested和observed capacity及页对齐容差。

协议：

```text
restart = 1
warmup = 1
formal validity rounds = 2
performance ranking = disabled
```

必须证明：

- 同一40对象集合；
- capacity变化不改变chunking/scheduler语义；
- 每类表示在最低capacity可运行；
- 至少一次approx source真实evict/demote；
- fallback/rollback路径可达；
- pool reset通过。

E0/E4因policy不同使用同一paired launch block中的独立server进程。

若R4-like footprint在最低capacity不可达：

1. 不按setting改变对象组成；
2. 只允许一次性缩短固定对象长度或降低最高representation multiplicity；
3. 重新冻结manifest；
4. 若仍不可达，R4标记diagnostic-unavailable，不阻塞R0/R1主线。

本pilot同时取代单独的Phase 5 fixed-40 exact-only重跑；E0/E4作为pilot baseline即可。

## 10. P6-3：Candidate Qualification、Chunk Gate与Recovery筛选

### 10.1 P6-3a：R0/R1 candidate-family qualification

执行§4 P6-3a合同。

screening：

```text
paths = R0/R1-k0, R1-k4, R1-k8, R1-k16, R1-k32
body = 1024, 2048
rho_logical_demand = 2.0
restart = 1
warmup = 1
formal = 2
```

k0与screening winner补至3 restart。

按§4中预注册promotion规则决定：

- `practical family = R1-kX`；或
- `practical family = NONE`。

P6-3a只产生数据，不得在看到结果后修改promotion规则。

### 10.2 P6-3b：Chunk configuration gate

复用P6-3a的`body1024/chunk1024` raw，新增：

```text
arms = dense, R1-k0, selected R1-k
body = 768, 1024
chunked_prefill_size = 1024, 4096
restart = 2 screening
formal = 2
```

输出：

- Phase 6固定`chunked_prefill_size`；
- 配置选择理由；
- speedup对chunking的敏感性；
- 结论披露模板。

可显式waive，但必须把全部后续结论限定在预注册chunk配置，不得发布跨chunking body-length claim。

### 10.3 P6-3c：Recovery重新筛选

候选：

- R0 speed ceiling；
- P6-3a通过的R1 practical configuration（若非NONE）；
- R2 precomputed family代表；
- R4 hierarchy diagnostic。

R4必须按V3 causal/paired/ledger/guardrail合同单独重测，但不参与practical winner排序。R5不因“更慢”被排除，而因与R2高度冗余且无practical-ledger收益不进默认矩阵。

setting：

```text
body = 1024, 2048
rho_logical_demand = 1.5, 2.0
scheduler = S0
tier = GPU-only
prefetch = P0
restart = 1 screening
formal = 2
```

primary cells补至3 restart。

每条候选必须报告：

- paired launch block内D0；
- 全部成本ledger；
- 一次setup后连续8个target，N=`1/2/4/8`取该序列前缀；
- cold-start；
- explicit fallback；
- effective recovered tokens；
- materialize/register/transfer分解；
- first-token及独立8-token guardrail；
- physical footprint和pressure残差。

规则：

1. R0固定保留为ceiling；
2. practical winner只能来自P6-3a通过的真实in-request路径；
3. precomputed family只作为oracle；
4. R4只作为DAG/victim diagnostic；
5. P6-4使用新round确认，不复用筛选数字；
6. practical family为NONE时，停止practical scheduler/HiCache claim，只保留ceiling/oracle/diagnostic。

## 11. P6-3.5：Scheduler Revalidation Gate

对practical recovery：

```text
body = 2048
rho_logical_demand = 1.5, 3.0
scheduler = S0-S4
restart = 1 screening
formal = 2
```

R4另在rho2运行S0-S4 victim-sequence诊断。

primary metric：

- all-reusable request TTFT；
- full-trace workflow wall-clock；
- workflow-only作为SLA次级视图；
- miss count同时报告。

S1/S2/S3晋级：

- all-reusable mean相对S0改善`>=5%`；
- all-reusable p95恶化`<=5%`；
- 或满足以下任一量化条件：
  - fallback率相对降低`>=50%`，且S0每trace至少有1次fallback；
  - physical resident peak降低`>=10%`；
  - victim correctness=`100%`且S4低于`100%`。

screening通过后必须3 restart确认。

S2只能称Belady-style；真正upper bound使用GPU-free variable-size offline optimum。
若S1/S2在冻结trace上产生同一victim顺序，结果必须标记为退化同序，不能作为两个独立机制证据。

## 12. P6-4：主Scheduler矩阵

默认primary：

```text
R0 ceiling × S0/S4
R1 practical × S0/S4（仅practical family非NONE）
GPU-only
P0
```

diagnostic：

- R2 oracle仅body2048、rho1.5/2；
- R4 hierarchy仅预注册结构点；
- S2保留eviction-onset和rho3 victim诊断。

pressure：

- body1024：rho1.5/2；
- body2048：rho1.1/1.5/2/3。

每个body/rho/restart block都必须配对测量：

- D0；
- E0；
- E4；
- 当前recovery的S0/S4。

禁止跨全矩阵只测一次共享baseline。

restart：

- 全cell先1 restart screening；
- 运行前manifest预声明primary cells；
- primary cells补至3 restart；
- diagnostic默认1 restart，除非影响主结论。

practical family为NONE时：

- 跳过R1 primary和P6-5 practical HiCache矩阵；
- 只保留R0 ceiling、R2 oracle、R4 diagnostic；
- 最终结论明确“未找到production-practical recovery”。

## 13. P6-5：HiCache Feasibility与Demand Load

### 13.1 Host feasibility gate

R5调试已证明：在pressure下错误启用host tier会增加H2D并OOM。

正式矩阵前必须单点证明：

```text
practical recovery
body = 2048
rho_logical_demand = 1.5
S4 + HiCache + P0
write policy = write-through
```

要求：

- source真实demote；
- host ref/accounting正确；
- demand load成功或显式fallback；
- temporary H2D peak可容纳；
- reset后host/device ref归零。

未通过则P6-5标记blocked，不运行大矩阵。

### 13.2 Demand-load矩阵

gate通过后：

```text
practical recovery
S4 + HiCache + P0
body = 2048
rho_logical_demand = 1.5, 2.0, 3.0
rho_host必须显式记录
```

H4/RH4在同block配对，固定host budget和write policy。

basic demand-load只要求真实demote+load，不强制`rho_host>=1`。

若发布host eviction/admission或prefetch管理claim：

- 至少一个primary stress cell必须`rho_host>=1`；
- 必须观察非零host miss/demotion/admission；
- 否则该矩阵标记为host-saturated canary。

## 14. P6-5.5：Prefetch Gate

### 功能/安全gate

setting：

```text
body = 2048
rho_logical_demand = 2.0, 3.0
prefetch = P0-P3
warmup = 1
formal = 2
restart = 1 screening
```

同步实现只验证：

- hint；
- load；
- admission；
- churn；
- lock release；
- no leak。

不发布性能claim。

### Async性能gate

只有真实async H2D存在时执行，并先明确：

- within-request demand-load overlap；或
- cross-request/concurrent prefetch。

记录CUDA-event overlap ratio与workflow wall-clock。

client模式必须预注册为：

- serial within-request pipeline；或
- pipelined/concurrent cross-request。

晋级：

- mean相对P0改善`>=3%`；
- all-reusable p95恶化`<=5%`；
- wasted bytes `<=10%` loaded bytes；
- admission churn bytes `<=20%` useful loaded bytes；
- 不驱逐更早next-use对象。

## 15. 最终统计与有效性

### 工程有效性

- 100%完成率；
- 无OOM、allocator corruption、stale handle、double free；
- exact+approx+host accounting通过；
- rollback/reset通过；
- explicit success/load/fallback；
- raw/commit/env/test provenance完整。
- orphan count=`0`；
- extra-key scratch在无flush生命周期测试中可GC。

### 机制结论

与工程有效性分开：

- `POSITIVE`
- `NEGATIVE`
- `INCONCLUSIVE`

工程失败统一为`INVALID`，不得写成机制负结果。

所有POSITIVE/NEGATIVE晋级结论统一要求：

- primary p95以all-reusable为分母；
- p95恶化`<=5%`；
- 同一trace中的请求不作为独立实验样本；
- paired launch block和per-restart估计同时报告。

### Primary cells

运行前manifest预声明哪些cell做3 restart。不得在看过screening结果后临时挑选。

## 16. Phase 6双模型里程碑门

每个里程碑：

1. A/B独立review；
2. 全文互换；
3. 两份cross-consolidated draft；
4. 主会话final disposition；
5. 更新本计划、PROJECT、HANDOFF、TRACKING；
6. 只有主会话接受为blocking的P0关闭后才能进入下一里程碑。

任何单模型“通过”都不能替代该门。
review流程、override和model fallback统一引用§5，不在本节另设第二套规则。

## 17. 交付物

- V3-reviewed fairness/schema contract；
- G0 0-GPU correction manifest；
- P6-3a R0/R1 candidate qualification；
- P6-3b chunk configuration decision/waiver；
- cross-store object DAG与allocator；
- unified event-clock和S4 class/value specification；
- failure-injection tests；
- extra-key GC、ledger恒等式、pressure补偿CPU tests；
- fixed-workload capacity pilot；
- variable-size offline optimum；
- recovery re-screen；
- scheduler revalidation；
- paired main matrix；
- host feasibility canary；
- HiCache demand-load matrix；
- prefetch functionality/async reports；
- per-restart compact manifests及raw SHA；
- logical cells、server startups、arm rounds和GPU-hour budget报告；
- 双模型review与disposition记录；
- speed ceiling、practical、precomputed oracle、diagnostic四类结论。

## 18. Review建议在V3中的处理

| Review项 | V3锚点 | 状态 |
| --- | --- | --- |
| C-02/C-06/C-07/C-08 | G0 | accepted |
| C-03/C-05/C-09/C-10/C-11/C-13/C-15/C-19/C-29 | §6–§7、P6-3 | accepted |
| C-16 | P6-3b chunk gate | accepted / waive allowed |
| C-33/C-34/C-35 | P6-2 fixed workload | accepted |
| C-39/C-42–C-50 | P6-0/P6-1 | accepted |
| C-53/C-54/C-55/C-56/C-61–C-64 | §7、§11、§12、§15 | accepted |
| PRC-01–PRC-05 | G0、§6 | accepted |
| PRC-07/PRC-08 | §7 pressure alignment | accepted |
| PRC-09–PRC-12 | G0/schema | accepted |
| PRC-13 | R2/R5 remote persistence | completed |
| PRC-14 | §6.3 guardrail | accepted |
| PRC-15 | G0/P6-1 extra-key GC | accepted |
| PRC-16 | §2/§3/P6-3 | accepted |
| PRC-17 | P6-3 N sequence | accepted |
| PRC-18 | P6-5 feasibility | accepted |
| PRC-19/PRC-20 | G0、§7、P6-1 tests | accepted |
| PRC-21 | P6-3a与R4独立diagnostic | accepted |
| PRC-22 | P6-3b | accepted / waive allowed |
| PRC-23 | 条件性表 | conditional |
| C-65 | G0 authority repository versioning | accepted / pending execution |

### 条件性

- `PRC-06 register_and_insert`：作为潜在新原型，不阻塞Phase6 entry；
- R2/R5 matched-ratio：只有保留机制排序claim时执行；
- rho1.1/3 corrected oracle补点：只有发布rho稳健性claim时执行；
- stronger semantic metrics：当前只作为guardrail，不扩展semantic claim。

### 保持defer

- R3深层实现；
- 正式prefetch性能claim，直到async路径存在；
- RTX PRO 6000 scale和并发矩阵，直到SM75机制与fairness gate通过。

## 19. 当前状态

- V2已归档为`IMPLEMENTATION_PLAN_V2_ARCHIVED.md`。
- V3已完成GPT-5.6 Sol Max Thinking与Claude Opus 5 Max Thinking独立review、全文互换和交叉consolidate。
- 双模型review的P0已在本版计划中处置；reviewer自行P0不再自动形成blocker。
- Phase 6实现分支尚未创建。
- G0尚未完成，因此Implementation Entry仍blocked。
- P6-3a与P6-3b属于Experiment Entry前门，不阻塞P6-0/P6-1。
- 未启动新的Phase 6 GPU实验。

## 20. 实验预算与Early-stop

V3执行前同时报告：

- logical cells；
- server startups；
- arm rounds；
- estimated GPU hours。

当前上限估算：

```text
logical cells ≈ 140+
server startups ≈ 45–90
arm rounds ≈ 450–600
GPU time ≈ 4.5–13 hours
```

缩减规则：

1. P6-3a先1 restart screening，只给k0和winner补至3；
2. P6-3b复用P6-3a已有raw；
3. P6-2只做validity，不做排名；
4. N=`1/2/4/8`来自一次setup后的8-target序列；
5. P6-4先1 restart screening，只补预声明primary cells；
6. 任一工程INVALID立即停止该路径；
7. practical family=NONE时跳过practical scheduler/HiCache矩阵。
