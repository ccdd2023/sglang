# 实施计划 V6（Candidate）：Pinned Result-Bound Integrated Evaluation

> 版本：V6
>
> 状态：Candidate / Pending Final Opus Review
>
> 最后更新：2026-07-28T04:45:16-07:00
>
> 当前阶段：Phase6 technical Exit=`PASS WITH CAVEATS`；Phase7 runners、
> Docker CPU tests与targeted reviews已完成；R2已解析为
> `disabled_not_comparable`；final code pin=`5d9a5793d73121f088890aa6c02cfebc31cd97be`；
> rev7 manifest已pin并通过自检，rev6已supersede；最终Opus 5 Max Thinking
> review尚未完成；未进入Phase7。
>
> 取代版本：[`IMPLEMENTATION_PLAN_V5_ARCHIVED.md`](IMPLEMENTATION_PLAN_V5_ARCHIVED.md)

## 1. 文档职责与版本规则

- `PROJECT.md`：项目事实、结果和明确决策的最终事实来源。
- 本文件：当前最新、可执行的phase计划。
- `CONSOLIDATED_PHASE4_PHASE6_REVIEW.txt`：Phase4/5审计、corrected rerun与双模型review证据。
- `HANDOFF.md`：当前快照和下一步。
- `TRACKING.md`：不可改写时间线。
- V1/V2/V3/V4/V5 archive只用于历史追溯。

若本文件与`PROJECT.md`中更新、明确的事实或决策冲突，以`PROJECT.md`为准并立即同步。

重大计划变更必须：

1. 归档当前latest；
2. 提升版本号；
3. GPT-5.6 Sol Max Thinking与Claude Opus 5 Max Thinking独立review；
4. 全文互换并交叉consolidate；
5. 主会话完成final disposition后才标记`Current / Latest`。

## 2. 已冻结的证据与结论

### 2.1 已完成阶段

- Phase1–3：SM75、pressure harness、approximate KV common core。
- Phase4：R0/R1/R2/R4/R5筛选；R3 defer。
- Phase5：S0–S4、P0–P3 exact-cache scheduler isolation。
- Corrected R2/R5 key rerun：
  - R2最终closeout head `ce55860a9`；
  - R5最终closeout head `71f15d5d1`。

### 2.2 Corrected R2/R5结果

| 路径 | body | chunk/max-prefill | target-only | adapter-combined | request-path | recovery-object lifecycle |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R2 | 1024 | `1024/1024` | `1.659x` | `0.441x` | `0.526x` | `0.324x` |
| R2 | 2048 | `1024/1024` | `2.044x` | `0.407x` | `0.434x` | `0.246x` |
| R5 | 1024 | `1024/1024` | `1.614x` | `0.449x` | `0.527x` | `0.327x` |
| R5 | 2048 | `1024/1024` | `1.978x` | `0.406x` | `0.433x` | `0.246x` |

固定结论：

- target-only recovery收益仅在`chunked_prefill_size=max_prefill_tokens=1024`
  下测得，标记为`chunk-confounded`；
- CL2显示body1024差异主要来自dense臂跨越额外prefill chunk；上述收益不能
  迁移到Phase7 primary chunk4096，须由P7-1重新建立或撤回；
- 旧R2 `1.14x`和R5 `1.04x` single-use combined正收益已被推翻；
- R2/R5是precomputed oracle，不是practical candidate；
- R2与R5的target差异高度受1% vs 8.3% repair ratio影响，不能写成机制优劣；
- first-token一致只是不发生粗暴损坏的guardrail，不是semantic correctness；
- 不需要再次强制重跑同一R2/R5矩阵。
- corrected rerun仍有pressure残差：setup evictable未完全补偿，recovery arm eviction多约10.6–22.4%；
- R5 dense/recovery两臂的filler token内容不同，因此不用于精细机制排序。

### 2.3 Phase5结论边界

- Phase5只测exact Radix，不代表lossy recovery。
- workflow-only与all-reusable/full-trace分母会改变策略排序。
- S4 synthetic kind不能外推为真实approx object hierarchy。
- 原rho sweep改变对象数及dead/live组成。
- prefetch host tier饱和且H2D同步，只能作为功能/开销canary。

## 3. V6 Pinned Result-Bound架构

V6不改Phase编号，只把已经完成的Closeout/Phase6固化为证据输入，并将
Phase7收窄为实际触发的`practical=NONE`分支。与V5相比，V6删除不可实现而
不改变core dispatch的R2 GPU cells，并冻结runner、source pin、segment与
post-pin execution envelope。

```text
Closeout + Phase6 evidence
           |
           v
P7-0 pinned result-bound manifest
           |
           +--> P7-1 R0 ceiling
           |
           +--> P7-2 R0 ceiling × S0/S4 + R4-like diagnostic
           |
           v
P7-3 final validation/review
```

资源约束：

- 本机只有一张SM75，全部Phase7 GPU任务全局串行；
- hard cap=`36 server starts / 6 GPUh`；
- host/prefetch/async轨道预算为0；
- 任何新增轨道必须升级计划版本。

V3 → V4 → V5编号映射（历史）：

| V3 | V4 |
| --- | --- |
| G0 | Closeout CL0 |
| P6-3a | Closeout CL1 |
| P6-3b | Closeout CL2 |
| P6-0/P6-1/P6-2 | Phase6 P6-0/P6-1/P6-2/P6-3/P6-4/P6-H |
| P6-3c/P6-3.5/P6-4 | Phase7 P7-1/P7-2 |
| P6-5/P6-5.5 | V4历史host/prefetch tracks；V6默认defer |

### Closeout Lane

- **已完成**，作为V6冻结输入，不再执行。

### Phase6：Cross-Store Substrate & Feasibility

只回答：

> exact与device-approximate对象能否在同一device budget中安全竞争，
> 同时host-resident approximate对象是否遵守独立host limit？

不选择winner，不发布scheduler或prefetch性能claim。

状态：`PASS WITH CAVEATS`。

### Phase7：Integrated Recovery × Scheduling Evaluation

只回答：

> R0性能上限与R4-like diagnostic在chunk边界已披露的条件下是什么；
> S4是否改变R0 ceiling的system behaviour？

R2只保留Phase4 chunk1024历史引用并标`disabled_not_comparable`。
V6不回答HiCache/prefetch，因为`practical=NONE`已触发停止分支。

### 可选Phase8

只有Phase7证明值得扩大时，才进入scale、concurrency、RTX PRO 6000和large-codebase评测。

## 4. Entry与依赖

### Phase6 Entry

**已完成，历史门禁。**

### Phase7 Entry

必须完成：

- 全部Closeout Lane；
- Phase6 technical Exit=`PASS WITH CAVEATS`；
- V6 final Opus 5 Max Thinking review与主会话disposition完成；
- practical family=`NONE`、chunk primary=`4096`、sensitivity=`1024`；
- Phase7 primary manifest预注册并hash/commit验证；
- primary manifest冻结唯一`phase7_pinned_implementation_sha`与runner blob；
- code pin commit之后只允许Phase7 result-envelope路径变化；
- authority docs在独立docs仓库同步更新；
- 用户条件性授权已记录；只有最终Opus review无开放P0/P1后才生效。

### 条件项

| 项 | 是否阻塞 | 触发条件 |
| --- | --- | --- |
| R2/R5 matched repair ratio | 否 | 仅当发布两机制性能排序 |
| corrected R2/R5 rho1.1/3 | 否 | 仅当发布rho稳健性 |
| R2 Phase7 GPU补点 | **禁用** | 需未来新版恢复机制并重新review |
| R3深层实现 | 否 | 仅当要覆盖order-sensitive轴 |
| register-and-insert原型 | 否 | 仅当估计理想setup下限 |
| async prefetch | 阻塞prefetch性能claim | 不阻塞Phase7其他矩阵 |

## 5. 跨阶段共享合同

### 5.1 Recovery分类

| 路径 | 定位 |
| --- | --- |
| R0 / R1-k0 | speed ceiling，同一物理恢复family |
| R1 EPIC | genuine in-request candidate family |
| R2 CacheBlend | precomputed oracle family默认代表 |
| R5 CacheTune | 与R2冗余的hardware-controller oracle point |
| R4-like synthetic footprint proxy | 5x resident multiplicity；不得归因于KVCOMM机制 |
| R3 Cache-Craft | defer；order-sensitive轴未覆盖 |

R5不因“被R2性能支配”而排除；默认不进入primary是因为与R2高度冗余且没有practical-ledger收益。

### 5.2 Baseline

| ID | 配置 |
| --- | --- |
| D0 | dense，无reuse |
| E0 | exact + S0 + GPU-only + P0 |
| E4 | exact + S4 + GPU-only + P0 |
| C0 | R0 ceiling + S0 |
| C4 | R0 ceiling + S4 |
| PR-S0 | practical recovery + S0；V6不生成 |
| PR-S4 | practical recovery + S4；V6不生成 |
| O2 | R2 historical oracle；V6不生成GPU cell |
| H4 | exact + S4 + HiCache + P0；V6不生成 |
| RH4 | practical recovery + S4 + HiCache + P0；V6不生成 |

### 5.3 Paired launch block

`paired launch block`：

> 同一`(body, rho, restart)`下，以相同image/model/capacity目标/server-seed计划连续启动的一组相邻server进程。

eviction policy、HiCache、chunked-prefill或capacity不同均需独立server进程，配对发生在launch block级，不虚构同进程比较。

同一server内的arm合同：

- D0：无reuse metadata的dense arm；
- E0：exact-cache arm；
- R0：approximate recovery arm；
- 三臂共享server argv/plugin env、filler manifest、capacity与rho目标；
- 每臂之间完整reset，arm顺序按formal repeat交替；
- exact-only是plugin-enabled server内的隔离arm，不要求独立exact-only server。

### 5.4 成本ledger

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

recovery_total_N =
  source_preparation
  + sum(request_path_i for i in measured recovery targets 1..N)

dense_total_N =
  sum(request_path_i for i in matched dense targets 1..N)

speedup_N = dense_total_N / recovery_total_N

incremental_setup =
  recovery_source_preparation
  - dense_source_materialization

recovery_incremental_total_N =
  incremental_setup
  + sum(request_path_i for i in measured recovery targets 1..N)

speedup_incremental_N =
  dense_total_N / recovery_incremental_total_N
```

`protocol_overhead_ms`无独立one-token control、KV-copy时间和register elapsed时必须写`not_measured`。

禁止把target-only称为end-to-end。

必须输出：

- `speedup_N1/N2/N4/N8`，全部来自实际累计；
- `full_setup_break_even_observed_N`：`speedup_N>1`的第一个实测N；
- `incremental_setup_break_even_observed_N`：`speedup_incremental_N>1`的第一个实测N；
- N<=8未观察到时写`>8/not_observed`，禁止插值或公式外推。

这是保守双口径披露：headline默认full-setup；incremental-setup用于分离两臂
共享source materialization，不得只报告较有利的一版。

`cold_start_ms`细分：

- server-first-use；
- plugin-first-use；
- shape-first-use；
- setting-warmup；
- steady-state。

### 5.5 Matched-state

single-target方案固定：

1. 每round完整清空并重建同等source状态；
2. 每round只发送一个measured target；
3. approximate target不写回exact；
4. exact baseline只使用本round预构造的exact source。

只改变final suffix不能防止body exact hit。

native-system结果另报，并给approx写回加provenance/taint。

#### 5.5.1 A8 amortization block

- 每个formal repeat包含两个隔离sequence：`dense-A8`与`recovery-A8`；
- 每个sequence含8个预注册target；sequence内不reset，第8个target后reset；
- dense-A8执行与recovery-A8相同的source materialization但`register=false`，
  使`dense_source_materialization`可测；不可测时incremental口径写
  `not_separable`；
- 8个target使用独立target ID与extra-key，禁止后续target exact-hit前一target；
- source对象pin到sequence结束；
- 每target记录exclusive outcome/reason、`rho_resident`以及
  `filler/prior-target/pinned-source`三段token组成轨迹；
- 只有前N个target全部为预期arm outcome时，`speedup_N`有效，否则N=`INVALID`；
- dense/recovery共享target列表、顺序、suffix与pressure manifest。

#### 5.5.2 W workflow block

- W是独立多对象、role-annotated fixed workflow trace；
- 用于S0/S4，不得复用A8结果；
- 冻结object size、next-use、role、dead/live、request order与filler manifest。

### 5.6 Pressure与rho

- same filler manifest；
- 预算扣除`setup_used + setup_evictable`；
- 记录resident与temporary peak tokens/pages/bytes。

rho固定为：

- `rho_logical_demand = logical reusable working set / configured capacity`；
- `rho_physical_demand = requested physical pages including all representations and scratch / capacity`；
- `rho_resident = sampled (used + evictable) / capacity`；
- `rho_host = host working set / host capacity`。

GPU侧统一以tokens/pages记录，跨tier比较必须同时记录bytes。setup footprint只能在physical demand或filler补偿中计入一次，不得重复计算。

不得只写`rho`。

### 5.7 Flush/reset

每个setting：

1. cold-start；
2. 独立warmup；
3. warmup后清空exact、approx、HiCache、metadata、generation、host ref、extra-key scratch；
4. 验证accounting；
5. formal；
6. formal repeat之间再次完整清空；
7. post-setting reset invariant。

### 5.8 Cache outcome

每请求只归入：

- `dense_no_reuse_baseline`；
- `exact_gpu_hit`；
- `ordinary_exact_cache_miss`；
- `approximate_gpu_recovery`；
- `host_demand_load`；
- `approximate_recovery_failed_dense`。

approximate失败另有且只有一个exclusive terminal reason：

- `cross_store_reservation_failed`；
- `device_allocation_failed`；
- `unsupported`；
- `registration_failed`；
- `prefix_gap`。

带approx taint的一律计为approximate。

### 5.9 Correctness与promotion

- exact first；
- controlled reconstruction；
- 不支持时dense fallback；
- 性能请求`max_new_tokens=1`；
- 独立quality canary `max_new_tokens>=8`；
- 记录逐token一致率、decode eviction delta；
- 可用时记录top-k/logprob差异；
- 不扩展semantic correctness claim。

#### 5.9.A same-context corruption canary

- `source header == target header`；
- 任一输出token失配=`INVALID`工程缺陷；
- 证据模板：P6-H、P6-F independent control；
- 仍不等价于bitwise KV或logit fidelity。

#### 5.9.B cross-context exact-output promotion gate

- `source header != target header`；
- 输出失配是设计内近似结果，不自动等于corruption、semantic failure或一般不可用；
- exact-output equality只是一项保守产品promotion策略；
- 必须记录逐位置一致率，不得把失配引用为数据损坏证据。

### 5.10 统计

- primary estimator预注册；
- target按ID配对；
- restart称为“进程级timing replicate”；
- primary=`每restart的paired-target median`，并列出全部restart值与范围；
- p95每restart分别计算，再取restart间中位数；
- pooled `p95(A)/p95(B)`命名为`ratio_of_marginal_p95s`，只作附录；
- 禁止请求级bootstrap制造独立样本；
- workflow-only为SLA视图；
- full-trace wall-clock单独报告；
- 同trace请求不是独立实验样本；
- screening执行完整restart-0矩阵；只在预注册checkpoint后补restart1–2；
- 报告miss count、MDE/noise model、seed与比较数量；
- n=3不强制伪CI，直接报告三点与范围。
- server restart主要是timing replicate；改变workload seed时所有策略必须paired。
- all-reusable p95只适用于workflow-trace实验；单target CL1使用paired target p95/target sample分布。

### 5.11 统一Manifest/Schema

Closeout、Phase6和Phase7统一保存：

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
- cold-start/steady-state；
- per-request/per-trace/per-restart；
- ledger及rho definitions；
- validity/negative/inconclusive状态。

Phase7 primary manifest还必须冻结：

- plan/implementation/runner/source-tree commit与hash；
- `preregistered_manifest_sha256`反向绑定；
- image/model/tokenizer/chat-template revision；
- A8/W/filler manifest hash；
- A8的8-target完整列表；
- W的object/role/size/next-use/dead-live/request-order完整列表；
- setting ID、arm order、reset boundary、依赖/reuse关系；
- 完整server argv/env模板：chunk/max-prefill、mem_fraction_static、
  max_total_tokens、attention/cuda-graph backend、max_running_requests、
  eviction policy；
- requested/observed capacity、seed、MDE、early-stop checkpoint；
- outcome/reason taxonomy；
- expected starts/GPUh/hard-cap计数；
- raw/log路径与停服后hash；
- skipped tracks与全部NONE/P6-H/fallback caveat。

### 5.12 共享客户端、Memory与Transfer指标

- workflow wall-clock；
- all-reusable、workflow-only、per-role TTFT/miss；
- cache outcome与exclusive terminal reason taxonomy；
- victim count/tokens/bytes by object kind；
- evict/demote bytes；
- H2D/D2H bytes和time；
- materialization scratch peak；
- allocator reservation peak；
- wasted/churn bytes；
- orphan count必须为0。

## 6. Phase4/5 Closeout Lane（不新增Phase）

Closeout可与Phase6并行，但必须在Phase7前完成。

### CL0：0-GPU治理与artifact收尾

1. 旧R2/R5 key-cell加`superseded_by`。
2. Phase4全矩阵改OAT slices。
3. 记录R5 final-SHA rho sweep为空。
4. 标注R1旧矩阵SHA/pressure/axis。
5. authoritative raw manifest与R1 stale raw。
6. 统一rho、ledger、excluded components。
7. 从raw生成paired/per-restart/cold-start/formula-N。
8. Phase5 all-reusable/full-trace/per-role重算。
9. R2 fallback缺counter时写`indirectly_verified`。
10. artifact保存server env、raw SHA、server log、测试命令。
11. CPU测试：
    - ledger恒等式；
    - pressure补偿；
    - extra-key无flush GC。
12. authority文档仓库完成版本化备份。

### CL1：R0/R1 Candidate Qualification

```text
paths = R0/R1-k0, R1-k4, R1-k8, R1-k16, R1-k32
body = 1024, 2048
header = 64
rho_logical_demand = 2.0
tier = GPU-only
scheduler = S0
prefetch = P0
restart = 1 screening
```

k0和screening winner补至3 restart。

要求：

- cumulative causal-prefix；
- isolated extra-key；
- paired dense/filler；
- pressure扣除used+evictable；
- 全ledger；
- explicit fallback；
- first-token及独立8-token canary；
- pool reset、raw SHA、server env。

promotion规则：

1. fallback/pool/guardrail通过；
2. body2048 request-path在至少2/3 restart中`>1.0x`；
3. p95恶化`<=5%`；
4. 选median paired request-path最高者；
5. 差异`<=2%`时优先repair更多且p95更低；
6. 无通过者则`practical family = NONE`。

promotion规则在CL1执行前冻结；CL1只产生数据，不得在看到结果后修改规则。

CL1的p95定义为全部formal measured target samples上的paired target p95；不使用workflow all-reusable分母。

### CL2：Chunk Configuration Gate

```text
arms = dense, R1-k0, selected R1-k
body = 768, 1024
chunked_prefill_size = 1024, 4096
rho_logical_demand = 2.0
restart = 2 screening
formal = 2
```

复用CL1 raw。

CL2是P6-4 capacity pilot的前置配置门。

输出：

- Phase7固定chunk配置；
- 敏感性；
- 披露模板。

可显式waive，但所有结论必须限定在预注册chunk配置。

waive时：

- P6-4使用预注册worst-case provisional chunk；
- 若Phase7最终chunk配置不同，必须重跑受影响的P6-4 feasibility cell。

### CL3：Phase5零GPU重算

- workflow-only；
- all-reusable；
- full-trace wall-clock；
- per-role TTFT/miss；
- paired/per-restart；
- corrected hit-fraction clamp；
- Belady-style与offline optimum。

### CL4：Closeout Dual Review

Sol/Opus独立review、全文互换、主会话disposition。CL4可与Phase6里程碑review并行，但Phase7 Entry前必须形成统一disposition并冻结candidate、chunk和stop manifest。

## 7. Phase6：Cross-Store Substrate & Feasibility

### 7.1 Phase6目标

- exact/approx/host统一对象语义；
- 跨store预算与双向pressure；
- 原子allocation与rollback；
- lifecycle/GC；
- fixed40 capacity feasibility。
- generic host roundtrip correctness。

### 7.2 Phase6非目标

- 不选practical winner；
- 不发布scheduler speedup；
- 不发布HiCache/prefetch性能；
- 不做R2/R5机制排名。

### 7.3 P6-0：Contract Freeze

- fixed40对象、ID、顺序、dead/live；
- 至少两档长度；
- header64、body1024/2048、segment<=512；
- `chunked_prefill_size`使用CL2结果；未完成时使用预注册worst-case provisional值；
- matched-state、pressure、rho、ledger、reset及§5.11统一schema全部冻结；
- `chunked_prefill_size`采用CL2结果；CL2未完成时只能使用预注册worst-case provisional值；
- logical cells/server starts/rounds/GPU-hour预算。

### 7.4 P6-1：Object DAG与Policy

对象：

- exact stage variant/bundle；
- R0 canonical chain；
- R1 repair state；
- R2/R5 adapter family；
- R4 canonical/anchor/delta；
- host copy；
- materialization scratch。

每条边定义dependency、orphan、atomic closure、demotion/load、provenance。

Policy：

- S0跨exact/approx/host统一`event_ordinal`；
- S4冻结class order、value、dependency和tie-break。

### 7.5 P6-2：Allocator与Rollback

预算：

- token/page/byte；
- exact可驱逐/demote approx；
- approx可驱逐exact；
- host demotion；
- temporary reservation。

协议：

1. reserve；
2. 枚举victims；
3. 选atomic closure；
4. evict/demote；
5. allocate；
6. commit；
7. 失败rollback或dense fallback。

lock order：

```text
global policy state
-> object metadata
-> host transfer state
-> device allocator
```

failure injection覆盖reserve后、victim后、evict后、allocation后、commit前。

### 7.6 P6-3：Lifecycle与GC

- generation/lease/slot/host-ref归零；
- no stale handle/double free；
- orphan=0；
- extra-key无flush GC；
- host transfer lock release；
- rollback后accounting恢复；
- ledger/pressure CPU tests。

### 7.7 P6-4：Fixed40 Capacity Pilot

P6-4开始前必须完成CL2，或使用预注册worst-case provisional chunk。CL2最终选择不同配置时，重跑受影响feasibility cell。

```text
R0 footprint
R1-like worst-case repair/temporary footprint (pre-registered k=32)
R2-like ~2x
R4-like ~5x
body = 2048
rho_logical_demand = 1.1, 1.5, 2.0, 3.0
restart = 1
warmup = 1
formal validity = 2
performance ranking = disabled
```

要求：

- 同一40对象；
- 只调capacity；
- exact baseline是在plugin-enabled server内、经完整reset隔离的`exact_only`
  footprint profile；P6-4没有独立E0/E4 server；
- capacity不改变chunking语义；
- exact/approx真实竞争；
- fallback/rollback可达；
- pool reset。

R4-like不可达时只允许一次性调整固定对象长度/representation multiplicity并重新冻结manifest；仍不可达则标`diagnostic-unavailable`。

本pilot不取代Phase5 fixed-40 exact-only重跑；其exact baseline仅用于
P6-4同server footprint/accounting对照。

### 7.8 P6-H：Generic Host Roundtrip Canary

与具体recovery winner无关，使用最小approx object证明：

- GPU object真实demote到host；
- host ref/accounting正确；
- demand load和H2D completion；
- transfer lock释放；
- temporary peak可容纳；
- reset后host/device ref归零。

```text
restart = 1
warmup = 1
formal validity = 2
performance claim = disabled
```

### 7.9 Phase6 Exit Gate

必须证明：

- exact与device-approximate共享device budget；host approximate遵守独立host limit；
- 双向pressure有效；
- allocation失败可回滚；
- fixed40四rho可运行或有明确不可达结论；
- R1-like worst-case footprint可运行或有明确不可达结论；
- generic host roundtrip canary通过；
- 无泄漏、无orphan；
- same-context、1 restart/2 round的8-token output canary与matched dense一致；
  不扩展为一般KV/logit fidelity；
- **dense fallback集成路径**：P6-F fault-injected canary验证通过
  （见§7.9.1）；自然压力可达性未证明；
- raw/commit/env/test provenance完整。

Phase6结果经双模型review后才允许Phase7。

#### 7.9.1 dense fallback可达性：**fault-injected canary验证通过**

历史说明：用户最初选择方案C治理性豁免；正式Exit review据此判FAIL。
随后用户明确以§15.1冻结的test-only路线取代该豁免。P6-F v3与两轮targeted
delta review现已关闭该blocker。

必须使用的表述：

> At source commit `9e6c2026e0fd68ed691bac072d05d6711d4c2b7c`,
> a test-only, one-shot `AFTER_RESERVE` fault caused one real approximate
> reuse request to record one non-reset reservation failure, one
> `reuse/dense_fallback` outcome and 1024 exclusively attributed
> `cross_store_reservation_failed` tokens. Dense and fallback namespaces
> each exposed exactly the isolated 64-token header; the remaining
> 1024-token suffix traversed ordinary dense prefill/decode, completed eight
> tokens and matched the dense control. Pre-flush and post-reset reserved,
> provisional, lease and orphan accounting was clean. A separate server
> launched without the injection flag independently re-registered and
> recovered normally.

证据：`p6-f-v3-fallback-canary.json`，两个versioned server log，
`RESULT_MANIFEST.json`（48/48）。

验证范围：

- `fault_injected=true`；
- `natural_pressure_reachability=false`；
- 验证的是integrated fallback功能，不是自然压力下的可达性或性能。

旧P6-4 raw仍不改写；其`fallback_reachability.passed=false`是自然压力实验的
真实记录。P6-F是独立的test-only canary，不覆盖或重写该字段。

Phase7中任何依赖自然reservation失败的claim，仍必须重新取得自然证据，
并保留`natural_pressure_reachability=false` caveat。

## 8. Phase7：Result-Bound Integrated Evaluation

### 8.1 Entry（全部必须满足）

- Phase6 technical Exit=`PASS WITH CAVEATS`；
- V6完成最终Opus 5 Max Thinking review并成为`Current / Latest`；
- latest `phase7-primary-manifest.json`已pin、提交并hash验证；
- 两个runner的Docker CPU tests与targeted reviews通过；
- R2=`disabled_not_comparable`；
- 用户条件性授权在上述Gate全部关闭后生效。

Phase6通过本身**不授权**Phase7。

### 8.2 P7-0：Result-Bound Freeze

| Track | V6冻结值 | 允许的claim |
| --- | --- | --- |
| Ceiling | R0 | 仅性能上限；不通过exact-output gate，不是practical |
| Practical | **NONE** | 不生成任何practical cell |
| Oracle | R2=`disabled_not_comparable` | 只保留Phase4 chunk1024历史引用，不生成GPU cell |
| Diagnostic | R4-like synthetic footprint proxy | 5x resident footprint/victim diagnostic；不得归因KVCOMM |
| R5 | 默认排除 | 与R2功能重叠；不得写“被性能支配” |

`NONE`的准确含义：

> 在本模型、合成prompt族、SM75、`chunk=max-prefill=1024`与冻结exact-output
> promotion规则下，没有candidate通过。已排除已修复的
> eviction-dependent prefix-overwrite缺陷，但未证明context差异是唯一原因，
> 也未排除header-dependent实现缺陷。V6将primary迁移到4096；`NONE`在4096下
> 未重新qualification，跳过practical是V6 scope决策，不是新的经验结论。

### 8.2.1 P7-0工程前置（0-GPU，阻塞任何GPU运行）

以下工程前置已经实现；latest pinned manifest必须绑定其最终blob与review状态：

1. `run_p7_ceiling.py`：
   - A8 workload；
   - D0/E0/R0三臂；
   - 真实N=1/2/4/8累计；
   - exclusive outcome/reason；
2. `run_p7_scheduler.py`：
   - 独立W workflow；
   - S0/S4；
   - all-reusable/full-trace/workflow-only/per-role；
3. `build_phase7_manifest.py --check`；
4. R2已冻结为`disabled_not_comparable`：
   - bounded feasibility确认历史CacheBlend package与core hooks已删除；
   - 恢复R2至少需要修改scheduler dispatch、runtime与store lifecycle；
   - 因而V6删除R2 GPU cells，只保留Phase4历史引用；
5. R4统一为`R4-like synthetic footprint proxy`，不得声称执行KVCOMM重建。

额外冻结：

- A8 `segment_tokens_max=512`；
- W `segment_tokens_max=512`；
- A8 source使用服务端门控、最多16条的`pin_until_reset` registration lease；
- reset必须释放persistent lease并清零reserved/provisional/orphan与arm peak；
- code pin之后只允许primary/result manifest envelope commits。
- final Opus review evidence固定写入
  `benchmark/approx_kv/results/phase7/phase7-final-opus-review.json`；
  review revision保持`pinned_blocked`，review通过后使用保持同一design hash的后续revision
  转为`authorized`。

### 8.2.2 P7-0b Chunk-migration feasibility gate

V6将primary chunk从Phase6 P6-4的`1024`迁移到`4096`，因此触发兼容性复核：

```text
required:
  body=2048, chunk=max-prefill=4096, restart=1, warmup=1, formal=2
  cell A: S4/rho2.0
  cell B: S0/rho2.0
conditional:
  cell C: S4/rho3.0（仅当最终报告需要任何chunk4096/rho3 claim时）
```

必须覆盖`exact_only`、`R0-like`、`R1-like-k32`、`R2-like`、
`R4-like-5x` footprint、accounting与死亡瞬间store gauge
（采样间隔`<=0.05s`）。

- A/B只关闭Phase7相关的`rho2/chunk4096`兼容性；
- 若不运行C，rho3 feasibility永久限定为chunk1024，Phase7不得发布任何
  chunk4096/rho3 claim；
- 若waive A/B或出现无法解释的死亡，整个Phase6 feasibility限定为chunk1024，
  Phase7不得引用其4096兼容性。

### 8.3 Chunk与共同运行合同

- **Phase7 primary chunk：`4096`**。
  原因：CL2 body1024显示`1024`会使dense未缓存部分为1025 token，额外跨越
  chunk边界；`4096`用作反chunk-boundary偏差的公平性配置，**不是性能结论**。
  CL2输出为`inconclusive`且未自动选择4096，因此该选择是result-bound、
  post-hoc但保守的配置决策，必须如实披露。
- `1024`仅作预注册sensitivity，不得进入headline：
  `body=2048, rho=2.0, S0, restart=2, formal=2`。
  body1024两chunk历史结果与body768 boundary-free对照直接复用CL2，不重跑。
- chunk与`max_prefill_tokens`必须同时记录；不得把coupled配置变化归因于单一参数。
- tier=`GPU-only`，prefetch=`P0`。
- warmup=`1`，formal=`2`。
- primary使用`3`个进程级timing replicate；diagnostic使用`1`个。
- primary estimator为**per-restart paired trace median**；pooled p95仅作描述。
- 同一trace内请求不是独立样本。
- N=`1/2/4/8`必须由一次setup后的**真实连续8 targets**取前缀计算，
  不再使用单请求外推。
- 每run必须持久化raw JSON、server log、环境、model revision、source/result
  commit与`RESULT_MANIFEST`映射。

### 8.4 P7-1：Recovery Ceiling与Oracle

#### R0 ceiling primary（A8）

```text
body = 1024, 2048
rho_logical_demand = 1.5, 2.0
chunked_prefill_size = 4096
scheduler = S0
restart = restart-0 screening；过预注册MDE后补restart1–2
warmup = 1
formal = 2
targets_per_setup = 8
```

每个setting的paired launch block必须包含：

- D0：dense/no reuse；
- E0：exact-only；
- R0：approximate recovery；
- 三臂共享filler/capacity/rho，完整reset隔离，arm顺序按repeat交替。

wave-1=`4 settings × restart-0 = 4 starts`；
通过预注册MDE/validity checkpoint后补`8` starts，总上限`12`。

必须报告：

- target-only、request-path、full lifecycle；
- 实测N=1/2/4/8与`break_even_observed_N`；
- cold start；
- recovered/cached token；
- first-token、8-token逐位置一致率；
- exact/approx byte footprint与peak；
- fallback按**exclusive terminal reason**分组；
- `fault_injected`与natural failure严格分列。
- A8高压sequence是当前最可能自然触发reservation failure的场景；
  将其预注册为secondary observation，但不为触发而改变capacity或压力。

R0的输出不一致只能说明它未通过本项目的保守promotion gate；不得扩展为
semantic质量或一般不可用性claim。

#### Historical oracle disposition

R2在V6中固定为：

```text
strategy = disabled_not_comparable
Phase7 GPU settings = 0
historical evidence = Phase4 chunk1024 only
```

不得把Phase4 R2数值与Phase7 chunk4096结果放入同一排名或合并统计。
未来若恢复R2，必须另起计划版本、恢复或重写机制、重新冻结design hash并review。

### 8.5 P7-2：Narrow Scheduler Matrix

P7-2使用独立W workload。`practical=NONE`，因此：

- **跳过**practical S0–S4 revalidation；
- **跳过**S1/S2/S3 promotion；
- 主矩阵只有`R0 ceiling × S0/S4`。

```text
body = 2048
rho_logical_demand = 1.5, 2.0
chunk = 4096
policies = S0, S4
restart = 3
warmup = 1
formal = 2
```

S0与S4各自运行：`2 rho × 2 policy × 3 restart = 12 server starts`。
**不得复用P7-1 A8的S0结果。**
每个launch block包含同server、完整reset隔离的exact-only paired baseline。

primary视图：

- all-reusable mean/median与miss count；
- full-trace wall-clock；
- workflow-only SLA；
- per-role TTFT；
- per-restart paired delta。
- physical peak；
- victim/evict accounting by object kind。

解释规则：

- S4相对S1–S3的独特性只在workflow-only历史结果中出现；
- all-reusable下S1–S4相对S0均有相近描述性改善，现有历史数据不足以排序；
- P7只回答S4是否改变**R0 ceiling**下的system behaviour，不宣称practical收益。
- 即使R0 A8 ceiling判`NEGATIVE`，W矩阵仍执行；其primary结论改为
  cross-store victim/footprint behaviour，延迟视图降为次要。

#### R4 victim diagnostic

```text
R4-like synthetic footprint proxy
body=2048, rho=2.0, chunk=4096, policies=S0/S4, restart=1
```

仅输出5x resident footprint、victim sequence/class/accounting；
不做性能排名，不归因KVCOMM。S1–S3未实现cross-store policy，不得生成。

### 8.6 明确跳过的轨道

因`practical=NONE`，V6默认**不创建**以下cell：

- practical scheduler revalidation；
- R2 Phase7 adapter/oracle cells；
- HiRadix/Unified cross-store adapter；
- practical HiCache demand-load矩阵；
- practical prefetch功能或性能矩阵；
- async H2D性能claim；
- exact-only prefetch回归canary。

P0下仍需做0成本inactive断言：prefetch/host/async相关counter在全部V6 run中
零增量；这不是轨道或性能实验。

若未来用户单独授权这些轨道，必须先提升计划版本并重新预注册manifest；
不能在本V6 Phase7运行中临时添加。

### 8.7 Early-stop

结果型early-stop只在完整restart-0矩阵后评估。

#### MDE冻结

执行前写入primary manifest：

```text
noise_model =
  CL2 boundary-free body768 @ chunk4096
  request_path_speedup =
    1.005757, 1.004354, 1.010055, 1.002774
  mean = 1.005735
  sample_sd = 0.003127
  2 * sample_sd = 0.006254

MDE = max(5%, 2 * sample_sd) = 5%
```

body768在两种chunk下均单chunk，因此仅用于noise floor，不是Phase7 result cell。
MDE冻结后才允许生成primary manifest；不得根据restart-0结果修改。

立即停止当前track并记录`NEGATIVE`/`INCONCLUSIVE`，若：

1. unexpected primary OOM、请求未完成、stale handle、double free、
   accounting/reset/orphan失败→`INVALID`；
2. R4/P6-4Δ有`<=0.05s`死亡快照且store已耗尽→`DIAGNOSTIC_UNAVAILABLE`；
3. R0改善未同时满足：
   - per-restart paired median改善`>=5%`；
   - 且超过上述MDE；
   则记`NEGATIVE/INCONCLUSIVE`，不发布ceiling speedup headline；
4. chunk规则作用于配对speedup比值`R=dense/approx(request-path)`。
   CL2 body1024的`R1024≈1.547`、`R4096≈1.025`已相差约51%，
   因此V6**预先声明不发布“机制固有speedup”headline**，只报告chunk-coupled结果；
5. S4在rho1.5与rho2.0均：
   - all-reusable mean改善`<5%`，且
   - `miss_S4>=miss_S0 AND peak_S4>=peak_S0`；
   则停止scheduler收益claim；
6. 观察到**已进入approximate recovery**的natural reservation failure时，
   若未完成dense fallback，则整个
   Phase7工程状态`INVALID`。

P6-F只证明fault-injected fallback功能。任何natural-pressure claim必须由
Phase7自然事件重新取证。

### 8.8 P7-3：Final Validation

工程状态：

- `VALID`
- `INVALID`

机制状态：

- `POSITIVE`
- `NEGATIVE`
- `INCONCLUSIVE`

必须汇总：

- R0 speed ceiling与chunk sensitivity；
- `practical=NONE`；
- R2=`disabled_not_comparable`历史引用；
- R4 diagnostic；
- S0/S4 ceiling差异；
- 明确跳过的host/prefetch轨道；
- fault-injected fallback与natural-pressure scope分界；
- per-restart、raw/log/hash/commit/test provenance。

最终结果必须经Sol/Opus独立review、报告互换、targeted delta closure与主会话
disposition后才可发布。

## 9. Phase8（Potential Scope — Not Yet Created）

V6不自动触发Phase8。以下均为**未来版本**的必要但不充分条件：

- R0 ceiling在chunk4096下显示稳定且超过MDE的系统空间；
- S4在W workload中显示可重复的miss/peak/system-behaviour改善；
- 用户接受`practical=NONE`后仍希望扩大ceiling/diagnostic验证；
- host/async轨道必须先在升级后的计划版本中重新创建与授权。

Phase8候选范围：

- RTX PRO 6000；
- 更大模型/context；
- 并发workflow；
- real repository/codebase artifact；
- source/dependency invalidation；
- end-to-end coding correctness。

Phase8必须另行版本化规划，不在V6中预先承诺矩阵。

Phase7 schema必须提前采集足以判断Phase8触发条件的forward-compatible字段：

- per-role/per-artifact latency与hit；
- host transfer/overlap；
- concurrency-ready identifiers；
- source/revision/invalidation metadata。

## 10. 双模型Review制度

Reviewer：

- GPT-5.6 Sol / Max Thinking / long context；
- Claude Opus 5 / Max Thinking / long context。

里程碑：

1. V6 plan定稿；
2. Phase7 primary manifest定稿；
3. P7-1 recovery ceiling/oracle；
4. P7-2 narrow scheduler matrix/R4 diagnostic；
5. P7-3 final validation与最终结论。

host/prefetch不在V6默认轨道中，不创建对应review里程碑。

流程：

1. 独立review；
2. atomic findings；
3. 全文互换；
4. cross-consolidate；
5. 主会话disposition。

只有主会话接受为`accepted-blocking-P0`的finding阻塞。override必须记录理由和风险。模型不可用时不得静默替换。

Phase7执行前新增最终门：

1. code pin与latest pinned manifest完成；
2. Claude Opus 5 / Max Thinking / long context独立review最终plan、
   manifest、runner/test evidence、R2 disposition与implementation binding；
3. accepted feedback全部闭合；
4. 无开放P0/P1后，用户条件性授权才生效。

## 11. Result-Bound预算

| Phase7 item | logical settings | server starts | 说明 |
| --- | ---: | ---: | --- |
| P7-0 manifest/contract | `1` | `0` | 0-GPU |
| P6-4Δ-4096 required | `2` | `2` | wave-0；S4/rho2、S0/rho2 |
| R0 A8 primary | `4` | `12` | wave-1=4，过checkpoint后补8 |
| chunk1024 sensitivity | `1` | `2` | 仅body2048，2 restart |
| R0 W × S0/S4 | `4` | `12` | 独立workflow，不复用A8 |
| R4-like W × S0/S4 | `2` | `2` | 1 restart |
| **committed合计** | **`13 GPU + 1行政`** | **`30`** | 不含条件项 |
| P6-4Δ S4/rho3（条件） | `1` | `1` | 仅需chunk4096/rho3 claim时 |
| **含全部条件项** | **`14 GPU + 1行政`** | **`31`** | hard cap内余5 |

- GPUh按wave结算；基于Phase5/P6/CL1历史server启动与请求时长，V6预注册
  `expected_gpu_hours_total=3.8h`（wave-0=`0.3h`、wave-1=`0.4h`、
  wave-2=`2.9h`、rho3条件项=`0.2h`）；
- hard cap：`36 server starts / 6 GPUh`；
- GPUh headroom=`2.2h`；manifest validator要求expected总量`<=85%` hard cap；
- 重试计入同一hard cap；任一上限先到即绑定；
- 若rho3条件项触发，31 starts后仍余5次重试；触发前重新结算余量；
- 基于CL2 chunk4096 body1024约`1.025x`的历史量级，R0 A8很可能未过5% MDE，
  预期不补8个primary starts；届时committed实际约`22` starts，但预算仍按30保留；
- 超出hard cap必须停止并升级计划版本，不能临时扩表；
- host/prefetch/async预算为`0`。

Early-stop以§8.7为唯一权威定义，不在本节重复或添加结果后规则。

## 12. 交付物

### Closeout

- correction manifest；
- R0/R1 qualification；
- chunk配置/waiver；
- Phase5重算；
- dual review。

### Phase6

- shared contract；
- unified manifest/schema；
- object DAG；
- event-clock/S4 spec；
- allocator/rollback；
- lifecycle/GC tests；
- R1-like worst-case footprint report；
- generic host roundtrip canary；
- fixed40 capacity pilot；
- Phase6 exit review。

### Phase7

- result-bound primary manifest；
- `run_p7_ceiling.py`、`run_p7_scheduler.py`、`build_phase7_manifest.py --check`
  与CPU回归；
- A8/W/filler workload manifests；
- P6-4Δ-4096 compatibility report；
- R0 ceiling与真实N=1/2/4/8 amortization；
- chunk4096 primary与chunk1024 sensitivity；
- R2 historical disposition report（固定`disabled_not_comparable`）；
- R4-like synthetic footprint/victim diagnostic；
- R0 ceiling × S0/S4 narrow matrix；
- exact-cache-miss与approximate fallback分离后的taxonomy；
- per-restart compact/raw/log SHA与result manifest；
- logical settings/server starts/GPU-hour budget；
- 明确的host/prefetch skipped manifest；
- final dual review与main-session disposition。

## 13. Review Disposition Mapping

| Review范围 | V6锚点 | 状态 |
| --- | --- | --- |
| C-01–C-16 Phase4 provenance/fairness | CL0、§5、CL1/CL2 | accepted |
| C-17–C-23 R1/R3/R4 | CL1、P7-1/P7-2、defer | accepted/conditional |
| C-24–C-38 Phase5 metrics/prefetch | CL3、P6-4、P7-2；prefetch在V6 defer | accepted/deferred |
| C-39–C-65 Phase6 architecture/statistics/governance | §5、Phase6、§10–§12 | accepted |
| PRC-01–PRC-05 artifact/ledger | CL0、§5.4 | accepted |
| PRC-06 register-and-insert | 条件项 | conditional |
| PRC-07–PRC-12 fairness/provenance | §5、CL0、P6-0 | accepted |
| PRC-13 remote persistence | 已完成 | completed |
| PRC-14–PRC-20 guardrail/GC/tests | §5.9、CL0、P6-3 | accepted |
| PRC-21 R0/R1/R4 backfill | CL1、P7-1/P7-2 | accepted |
| PRC-22 chunk factorial | CL2 | accepted/waive allowed |
| PRC-23 rho robustness | 条件项 | conditional |

## 14. 当前状态

V4已归档为`IMPLEMENTATION_PLAN_V4_ARCHIVED.md`。

Phase6最终状态：

```text
technical_exit = PASS WITH CAVEATS
```

- 主会话disposition：`PHASE6_EXIT_DISPOSITION.json`；
- formal Exit与P6-F targeted delta reviews全部完成，无开放P0/P1；
- evidence manifest：`48/48`；
- fallback仅在`fault_injected=true`的集成canary强度验证，
  `natural_pressure_reachability=false`；
- practical promotion结果为`NONE`，严格限定于被测实现与冻结规则；
- 未进入Phase7。

V6 candidate当前状态：

- V5已由不可变commit归档；
- Phase7矩阵已收窄为R0 ceiling、R4-like proxy与R0×S0/S4；
- R2 bounded feasibility结论为`disabled_not_comparable`，无Phase7 GPU cell；
- host、HiCache、prefetch、async轨道默认全部跳过；
- `run_p7_ceiling.py`、`run_p7_scheduler.py`与共享Phase7模块已实现；
- 固定Docker镜像targeted CPU suite=`152 passed + 10 subtests`；
- 三轮targeted review已闭合全部P0/P1；
- final code pin=`5d9a5793d73121f088890aa6c02cfebc31cd97be`；
- rev7 primary manifest已pin并通过provenance check；
- V6 final Opus review与final disposition待完成；
- 用户已给条件性授权，但尚未生效；未进入Phase7。

## 15. V6冻结约束与已吸收教训

### 15.1 V6生效条件

V6只有在以下条件全部满足后才可改为`Current / Latest`：

1. V5不可变归档完成；
2. final code pin与latest pinned manifest完成；
3. Opus 5 Max Thinking final review完成；
4. 主会话逐项disposition并闭合accepted feedback；
5. 无开放P0/P1；
6. manifest、runner blobs、plan commit/hash与execution envelope全部验证通过。

V6生效后，用户已给出的条件性授权才生效；在此之前不得启动Phase7 GPU。

### 15.2 已由执行结果确定并吸收的合同修订

以下修订已进入V6正文或Phase7运行合同；review时必须检查是否完整且无冲突：

1. **guardrail语义歧义已消解。** §5.9把8-token canary定义为“记录逐token
   一致率、不扩展semantic correctness claim”，但冻结的CL1 runner把8-token
   完全一致当作promotion硬门。V6区分same-context corruption canary与
   cross-context conservative promotion gate。
2. **fallback证据分级。** 带label的Prometheus counter在未发生事件时不会输出
   任何series，因此“counter缺失”只能记为`indirectly_verified`，不得记为显式
   `0`。该规则已在代码中强制。
3. **chunk披露强制化。** 任何recovery speedup claim必须同时声明
   `chunked_prefill_size`与`max_prefill_tokens`，并附带一个prompt可单chunk
   容纳的对照点。CL2的4-repeat median显示body1024在chunk`1024`下约
   `1.547x`、在chunk`4096`下约`1.025x`。
4. **Phase6 Exit Gate已新增数据保真条目。** §7.9原先只要求安全竞争、双向
   pressure、可回滚、无泄漏，没有任何一条要求“近似reuse在压力下必须与matched
   dense逐token一致”。正因为缺这一条，该底座通过了三轮review和全部CPU回归，
   却在GPU压力下返回损坏KV。V6把它列为独立的Exit条件。
5. **压力态保真回归已新增。** P6-H与P6-F均经过真实GPU server路径；
   后续不得退回只依赖CPU fake allocator。
6. **Phase5结论按分母分列。** S4相对S1–S3的独特高rho优势只在
   workflow-only出现；all-reusable下S1–S4相对S0均有相近描述性改善，
   现有restart数不足以排序。
7. **Phase7 primary manifest与runner binding。** Entry条件要求manifest、
   两个runner、CPU tests、runner blob hashes和`build_phase7_manifest.py
   --check`全部通过。
8. **recovery必须在请求自身prefix锁的保护下执行。** 这是本轮P0的直接教训：
   `init_next_round_input`阶段请求尚未加锁，而victim枚举条件恰为
   `lock_ref == 0`，两者叠加使请求可以驱逐并覆写自己的KV。任何新增的
   recovery/分配路径都必须复用`protect_request_prefix`。
9. **统计口径必须如实标注。** 所报“paired p95”实为pooled
   `p95(approx)/p95(dense)`，不是配对统计量；N=1/2/4/8摊销是外推值而非实测；
   CL1只有3个restart级独立单元、CL2为2、CL3多数为1，同一trace内的请求
   不得当作独立重复。禁止使用“within noise”这类未经检验的统计判断。
10. **`practical=NONE`必须限定作用域**：它是冻结promotion规则在本模型、
    合成prompt族、exact-output不变量、本GPU与`chunk=1024`下的结论，
    不是普遍不可行性claim。
11. **区分“上下文差异”与“实现缺陷”必须用2×2对照**：
    `same/different header × low/high pressure`。仅凭“修复前后计数相同”
    不足以归因，必须补上“无eviction条件下仍偏离”这一格。
12. **allocation失败必须优雅降级。** `alloc_token_slots`在cross-store无法
    腾出空间时抛`RuntimeError`杀死scheduler进程，应改为可记录的失败。
13. **“容量不可达”必须附死亡瞬间的store gauge快照。** 凡是要写
    `diagnostic-unavailable`且理由为容量的cell，都必须证明当时没有可回收
    资源残留，而不是仅凭server崩溃就下结论。
14. **遥测采样间隔必须短于workload的分配动态。** 本轮诊断C v1用`0.4s`轮询，
    而最后`1.3s`内`num_used_tokens`从`5376`涨到`10688`，导致“最后一个成功
    样本”早于致命请求，产生了一个**自信但完全错误**的“这是我方缺陷”结论；
    改用`0.05s`后结论反转。任何“死亡瞬间状态”类证据都必须声明采样间隔，
    并论证它足够细。
15. **不要把`num_used_tokens`与store gauge相加。** 前者已包含approximate
    store占用的slot，相加会凭空造出并不存在的“未归属token”。
16. **`dense_fallback`的标签已区分“近似恢复失败”与“普通exact-cache
    miss”。** `run_p6_4_capacity_pilot.py:413-419`在profile无approximate
    metadata时曾把普通miss标为`dense_fallback`，导致错误Exit证据；
    runner已改为`exact_cache_miss`，raw artifact由correction view解释。
17. **cell级状态与profile级状态必须分别陈述。** P6-4完整矩阵每个cell顶层
    都是`diagnostic-unavailable`；可达的是其中的非R4 profile。
    写成“三个cell可达”是过声明。
18. **artifact的`result_git_sha`天然为null**（runner无法知道将来容纳自己
    输出的commit）。因此必须另行维护`RESULT_MANIFEST.json`提供
    file→commit映射、内容哈希与验证命令，并且不得据artifact字段声称
    “provenance完整”。
19. **code pin与execution envelope必须分层。** manifest不能自包含它所在
    commit的SHA；V6采用code pin commit为祖先、后续只允许Phase7 result
    envelope路径变化，并逐blob验证runner与manifest。
20. **A8 source必须真实pin到sequence结束。** V6使用默认关闭、服务端门控、
    上限16条的`pin_until_reset` registration lease；reset必须释放全部lease。
21. **process-lifetime peak不能冒充per-arm peak。** full reset现在清零
    cross-store budget high-water；artifact字段明确为
    `arm_interval_peak_device_bytes`。
22. **memory accounting不得双计approx store。** `nonfree_resident_bytes`
    已包含approx slot；另报`approx_device_bytes`与
    `exact_only_estimated_bytes`，禁止相加重复计算。
23. **R2方案B的停止条件已触发。** bounded feasibility确认恢复R2需要重建
    已删除的CacheBlend package并修改冻结dispatch，因此V6按预定决策树选择
    `disabled_not_comparable`，不是实现失败后的临时删项。

### 15.3 已完成或作废的旧P0方向

本节原列出的stale-victim两个候选方向已经过时，不再是待办：

- prefix自我驱逐根因已由`protect_request_prefix`修复；
- stale victim已改为隔离、刷新并重试；
- provisional recovery slot已在admission拒绝、waiting abort、下一轮rematch与
  teardown路径回收，且仅在`allocator.free`成功后清除引用；
- SWA/Unified释放元数据已回传。

后续不得再按旧§15.3重复设计或重做这些修复。Phase6当前无开放blocker。
