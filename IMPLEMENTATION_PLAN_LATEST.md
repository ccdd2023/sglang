# 实施计划 V4（Latest）：Cross-Store Substrate -> Integrated Evaluation

> 版本：V4
>
> 状态：Current / Latest
>
> 最后更新：2026-07-27T16:10:00-07:00
>
> 当前阶段：CL0–CL3、P6-H、P6-4与P6-F全部执行完毕；Phase6 technical
> Exit为`PASS WITH CAVEATS`；未进入Phase7。
>
> 取代版本：[`IMPLEMENTATION_PLAN_V3_ARCHIVED.md`](IMPLEMENTATION_PLAN_V3_ARCHIVED.md)

## 1. 文档职责与版本规则

- `PROJECT.md`：项目事实、结果和明确决策的最终事实来源。
- 本文件：当前最新、可执行的phase计划。
- `CONSOLIDATED_PHASE4_PHASE6_REVIEW.txt`：Phase4/5审计、corrected rerun与双模型review证据。
- `HANDOFF.md`：当前快照和下一步。
- `TRACKING.md`：不可改写时间线。
- V1/V2/V3 archive只用于历史追溯。

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

| 路径 | body | target-only | adapter-combined | request-path | recovery-object lifecycle |
| --- | ---: | ---: | ---: | ---: | ---: |
| R2 | 1024 | `1.659x` | `0.441x` | `0.526x` | `0.324x` |
| R2 | 2048 | `2.044x` | `0.407x` | `0.434x` | `0.246x` |
| R5 | 1024 | `1.614x` | `0.449x` | `0.527x` | `0.327x` |
| R5 | 2048 | `1.978x` | `0.406x` | `0.433x` | `0.246x` |

固定结论：

- target-only recovery收益存在；
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

## 3. V4 Phase架构

V4不为Phase4/5收尾新增phase编号。

```text
Phase4/5 Closeout Lane ─┐
       CL1 -> CL2 ------+------> P6-4 Capacity Pilot
                \ provisional_worst_case chunk ---^
                       |
Phase6 Substrate ------+------> Phase7 Integrated Evaluation
```

资源约束：

- 0-GPU closeout与Phase6工程可以并行；
- 本机只有一张SM75，CL1、CL2和P6-4 GPU任务全局串行；
- CL2必须在P6-4前完成，或P6-4使用预注册worst-case provisional chunk并在CL2改变配置后重跑受影响pilot。

V3 -> V4编号映射：

| V3 | V4 |
| --- | --- |
| G0 | Closeout CL0 |
| P6-3a | Closeout CL1 |
| P6-3b | Closeout CL2 |
| P6-0/P6-1/P6-2 | Phase6 P6-0/P6-1/P6-2/P6-3/P6-4/P6-H |
| P6-3c/P6-3.5/P6-4 | Phase7 P7-1/P7-2 |
| P6-5/P6-5.5 | Phase7 P7-3/P7-4 |

### Closeout Lane

- 前阶段结果、schema、artifact和候选确认；
- 可与Phase6工程并行；
- 必须在Phase7 Entry前完成。

### Phase6：Cross-Store Substrate & Feasibility

只回答：

> exact、approx和host对象能否在同一budget中安全竞争？

不选择winner，不发布scheduler或prefetch性能claim。

### Phase7：Integrated Recovery × Scheduling Evaluation

只回答：

> 在正确的cross-store底座上，recovery、scheduler、HiCache和prefetch是否产生可证明收益？

### 可选Phase8

只有Phase7证明值得扩大时，才进入scale、concurrency、RTX PRO 6000和large-codebase评测。

## 4. Entry与依赖

### Phase6 Entry

允许创建Phase6分支并实现底座。必须完成：

- CL0 0-GPU收尾；
- V4双模型Plan Review及主会话disposition；
- 无`accepted-blocking-P0`。

Phase6 Entry前不要求新的Phase4/5 GPU重跑。

### Phase7 Entry

必须完成：

- 全部Closeout Lane；
- Phase6 exit gate；
- Phase6结果双模型review；
- practical family与chunk配置已冻结或显式NONE/waive。

### 条件项

| 项 | 是否阻塞 | 触发条件 |
| --- | --- | --- |
| R2/R5 matched repair ratio | 否 | 仅当发布两机制性能排序 |
| corrected R2/R5 rho1.1/3 | 否 | 仅当发布rho稳健性 |
| R2显式fallback GPU补点 | 否 | 仅当间接证据不足 |
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
| R4 KVCOMM | canonical/anchor/delta diagnostic |
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
| PR-S0 | practical recovery + S0 |
| PR-S4 | practical recovery + S4 |
| O2 | R2 precomputed oracle |
| H4 | exact + S4 + HiCache + P0 |
| RH4 | practical recovery + S4 + HiCache + P0；practical=NONE时不存在 |

### 5.3 Paired launch block

`paired launch block`：

> 同一`(body, rho, restart)`下，以相同image/model/capacity目标/server-seed计划连续启动的一组相邻server进程。

eviction policy、HiCache、chunked-prefill或capacity不同均需独立server进程，配对发生在launch block级，不虚构同进程比较。

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

amortized_ms_N =
  (source_preparation + target_adapter_preparation) / N
  + seed_head
  + post_pressure_reseed
  + transfer
  + target_only
```

`protocol_overhead_ms`无独立one-token control、KV-copy时间和register elapsed时必须写`not_measured`。

禁止把target-only称为end-to-end。

必须输出：

- `amortized_ms_N1/N2/N4/N8`
- `break_even_reuse_count`

`cold_start_ms`细分：

- server-first-use；
- plugin-first-use；
- shape-first-use；
- setting-warmup；
- steady-state。

### 5.5 Matched-state

primary方案固定：

1. 每round完整清空并重建同等source状态；
2. 每round只发送一个measured target；
3. approximate target不写回exact；
4. exact baseline只使用本round预构造的exact source。

只改变final suffix不能防止body exact hit。

native-system结果另报，并给approx写回加provenance/taint。

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

- exact GPU hit；
- approximate GPU recovery；
- host demand load；
- dense fallback。

带approx taint的一律计为approximate。

### 5.9 Correctness guardrail

- exact first；
- controlled reconstruction；
- 不支持时dense fallback；
- 性能请求`max_new_tokens=1`；
- 独立quality canary `max_new_tokens>=8`；
- 记录逐token一致率、decode eviction delta；
- 可用时记录top-k/logprob差异；
- 不扩展semantic correctness claim。

#### 5.9.1 guardrail语义冻结（2026-07-27，CL1重跑前冻结）

原§5.9只说“记录逐token一致率”，而冻结的CL1 runner把8-token完全一致当作
promotion硬门，两者在CL1首轮产生了歧义（FINDING-CL1-C）。现按如下方式冻结，
**在任何重跑数据产生之前生效**：

1. **保留8-token完全一致为promotion硬门。** 理由是P6-H已证明：当近似路径
   真的损坏KV时，机械证据（byte、token、lease、reset）会全部通过，唯一暴露
   问题的信号就是输出偏离matched dense。放弃这道门等于放弃唯一的数据保真
   探针。
2. **同时必须记录逐token一致率**，不得只记布尔值；一致率用于区分“完全损坏”
   与“单token发散”。
3. **该门的语义边界写死为**：它是“未发生数据损坏”的guardrail，
   **不是**semantic correctness或生成质量claim；不得据此声称近似恢复
   保持模型质量。
4. body1024与body2048分别报告，不合并。

该冻结适用于CL1、CL2、P6-H及Phase7全部recovery实验。

### 5.10 统计

- primary estimator预注册；
- paired delta/ratio和per-restart同时报告；
- all-reusable为primary p95分母；
- 任何晋级要求p95恶化`<=5%`；
- workflow-only为SLA视图；
- full-trace wall-clock单独报告；
- 同trace请求不是独立实验样本；
- screening 1 restart，primary补至3；
- 报告miss count、CI和比较数量。
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

### 5.12 共享客户端、Memory与Transfer指标

- workflow wall-clock；
- all-reusable、workflow-only、per-role TTFT/miss；
- cache outcome四分类；
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
- E0/E4独立server、同paired block；
- capacity不改变chunking语义；
- exact/approx真实竞争；
- fallback/rollback可达；
- pool reset。

R4-like不可达时只允许一次性调整固定对象长度/representation multiplicity并重新冻结manifest；仍不可达则标`diagnostic-unavailable`。

本pilot同时取代单独的Phase5 fixed-40 exact-only重跑；E0/E4作为pilot baseline。

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

- exact、approx、host同budget安全竞争；
- 双向pressure有效；
- allocation失败可回滚；
- fixed40四rho可运行或有明确不可达结论；
- R1-like worst-case footprint可运行或有明确不可达结论；
- generic host roundtrip canary通过；
- 无泄漏、无orphan；
- **近似reuse在压力下与matched dense逐token一致**（2026-07-27新增，
  由P6-H提供证据；缺这一条正是本轮P0躲过三轮review与全部CPU回归的原因）；
- **dense fallback可达性**：接受`indirectly_verified`强度结案
  （2026-07-27用户决定，见§7.9.1）；带label的counter无series只能记为
  `indirectly_verified`，不得写成显式`0`；
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

## 8. Phase7：Integrated Recovery × Scheduling Evaluation

### 8.1 Phase7 Entry

- Closeout CL0–CL4完成；
- Phase6 Exit通过；
- practical family已冻结或为NONE；
- chunk配置已执行或waive；
- Phase7 primary manifest已预注册。

### 8.1.1 practical family冻结结果（2026-07-27）

CL1已在修复后的底座上完成screening与3-restart确认，
`promotion.status=complete`、`passing=[]`、**`winner=NONE`**。

因此Phase7的`practical=NONE`分支**已确定触发**，不再是待定分支：

- §8.4 practical scheduler revalidation：**跳过**，不生成PR-S0/PR-S4；
- §8.5 P7-3 practical HiCache track与RH4：**跳过**，不实现专用
  HiRadix/Unified cross-store adapter；
- §8.6 P7-4 prefetch性能track：**跳过**；
- 保留R0 ceiling、R2 oracle、R4 diagnostic；
- exact-only prefetch若执行，只能标为Phase5回归canary。

该结论的**准确强度**（第三轮审计要求）：

> 在被测实现与配置下，没有candidate通过冻结的exact-output promotion规则。
> 已知的eviction-dependent prefix-overwrite缺陷**被排除**为该偏离的解释，
> 但**未证明context差异是因果原因**，**也未排除header-dependent实现缺陷**。

依据是P0修复前后CL1的guardrail失败计数与输出序列完全相同
（screening `17+6`/48，confirm `12+4`/48）。注意所谓2×2**并非真正的
factorial**（拼接了两套runner/policy/chunk/env/SHA不同的实验），
低压cell的eviction telemetry也未覆盖完整target allocation窗口。
机制上，CL1把在`source_header`（`32_000+`）下计算的body KV复制到
`target_header`（`36_000+`）之后使用，前缀不同导致attention上下文不同，
KV本来就是近似的；而P6-H的source与target header相同，修复后输出逐token一致。

Phase7因此**大幅收窄**：主矩阵只保留R0 ceiling与diagnostic轨道，
不存在practical recovery × scheduler的笛卡尔积。

### 8.2 P7-0：Candidate Freeze

| Track | 候选 |
| --- | --- |
| Ceiling | R0 |
| Practical | selected R1-k或NONE |
| Oracle | R2 |
| Diagnostic | R4 |

R5因与R2冗余不进默认矩阵，不使用“被R2性能支配”理由。

### 8.3 P7-1：Recovery Evaluation

```text
body = 1024, 2048
rho_logical_demand = 1.5, 2.0
scheduler = S0
tier = GPU-only
prefetch = P0
restart = 1 screening
formal = 2
```

primary补至3 restart。

每路径：

- paired D0；
- 全ledger；
- 一次setup后连续8 targets，N=1/2/4/8取前缀；
- cold-start；
- explicit fallback；
- recovered tokens；
- materialize/register/transfer；
- quality guardrail；
- physical footprint。

R4按相同合同重测，但不参与practical排序。

### 8.4 P7-2：Scheduler Evaluation

#### Revalidation

若`practical family=NONE`，跳过practical revalidation，仅保留exact/ceiling/oracle diagnostic。

```text
practical recovery
body = 2048
rho_logical_demand = 1.5, 3.0
scheduler = S0-S4
restart = 1
formal = 2
```

primary：

- all-reusable request TTFT；
- full-trace wall-clock；
- workflow-only SLA；
- miss count。

S1/S2/S3晋级：

- mean改善`>=5%`；
- p95恶化`<=5%`；
- 或fallback降低`>=50%`且S0每trace至少1次；
- 或physical peak降低`>=10%`；
- 或victim correctness=100%且S4<100%。

通过后补至3 restart。

S2只能称Belady-style。若S1/S2在冻结trace上产生同一victim顺序，必须标记为退化同序；真正upper bound使用GPU-free variable-size offline optimum。

R4独立diagnostic：

```text
R4
body = 2048
rho_logical_demand = 2.0
scheduler = S0-S4
restart = 1
formal = 2
output = canonical/anchor/delta victim sequence
```

#### Main Matrix

```text
R0 ceiling × S0/S4
selected R1 practical × S0/S4（若非NONE）
GPU-only
P0
```

diagnostic：

- R2 body2048 rho1.5/2；
- R4 `body2048 / rho2 / S0-S4` canonical/anchor/delta victim-sequence diagnostic；
- S2 eviction-onset/rho3。

body1024：rho1.5/2。
body2048：rho1.1/1.5/2/3。

每个paired launch block测D0/E0/E4和当前recovery arm。

practical=NONE时：

- 跳过practical revalidation；
- 不生成PR-S0/PR-S4；
- 跳过RH4和practical host矩阵；
- 跳过prefetch性能track；
- 保留R0 ceiling、R2 oracle、R4 diagnostic；
- exact-only prefetch若执行，只能标Phase5回归canary。

### 8.5 P7-3：HiCache Demand Load

practical=NONE时整个practical HiCache track跳过。

#### Feasibility

```text
practical recovery
body = 2048
rho_logical_demand = 1.5
S4 + HiCache + P0
write policy = write-through
```

证明：

- source demote；
- host ref/accounting；
- demand load或explicit fallback；
- H2D peak；
- reset。

失败则HiCache track blocked。

#### Matrix

```text
body = 2048
rho_logical_demand = 1.5, 2.0, 3.0
rho_host显式记录
```

H4/RH4同block。

basic demand-load不强制`rho_host>=1`；host eviction/admission claim必须至少一个`rho_host>=1`且出现非零host miss/demotion。

### 8.6 P7-4：Prefetch

#### 功能/安全

```text
body = 2048
rho_logical_demand = 2.0, 3.0
P0-P3
warmup = 1
formal = 2
restart = 1
```

只验证hint/load/admission/churn/lock/no leak。

practical=NONE或host feasibility失败时：

- practical prefetch功能/性能track停止；
- 可预注册选择执行E4+HiCache exact-only回归canary，或整体跳过；
- 不得把exact-only canary写成Phase7 practical结论。

#### Async性能

只有真实async H2D存在时执行，并预注册serial within-request或concurrent cross-request模式。

晋级：

- mean改善`>=3%`；
- p95恶化`<=5%`；
- wasted bytes`<=10%`loaded；
- churn bytes`<=20%`useful loaded；
- 不驱逐更早next-use对象。

### 8.7 P7-5：Final Validation

工程状态：

- `VALID`
- `INVALID`

机制状态：

- `POSITIVE`
- `NEGATIVE`
- `INCONCLUSIVE`

primary cells预注册并做3 restart。同trace请求不作独立样本。

工程有效性必须包含：

- 100% completion；
- no OOM/allocator corruption/stale handle/double free；
- exact/approx/host accounting；
- explicit success/load/fallback；
- rollback/reset；
- orphan=0；
- raw/commit/env/test provenance。

最终双模型review冻结：

- speed ceiling；
- practical或NONE；
- precomputed oracle；
- diagnostic；
- host/prefetch结论。

## 9. Phase8（Potential Scope — Not Yet Created）

只有Phase7满足至少一项时创建：

- practical recovery在至少两个rho上稳定为正；
- host demand-load显示可扩展收益；
- async prefetch有真实overlap收益；
- SM75结果值得扩大验证。

Phase8候选范围：

- RTX PRO 6000；
- 更大模型/context；
- 并发workflow；
- real repository/codebase artifact；
- source/dependency invalidation；
- end-to-end coding correctness。

Phase8必须另行版本化规划，不在V4中预先承诺矩阵。

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

1. V4定稿；
2. Closeout完成；
3. Phase6 allocator完成；
4. Phase6 capacity pilot完成；
5. Phase6 exit；
6. Phase7 recovery/scheduler；
7. Phase7 host/prefetch；
8. 最终结论。

CL4与Phase6里程碑review可并行进行；Phase7 Entry前必须将两条lane的finding合并为一份统一disposition。

流程：

1. 独立review；
2. atomic findings；
3. 全文互换；
4. cross-consolidate；
5. 主会话disposition。

只有主会话接受为`accepted-blocking-P0`的finding阻塞。override必须记录理由和风险。模型不可用时不得静默替换。

## 11. 预算与Early-stop

执行前同时报告：

- logical cells；
- server startups；
- arm rounds；
- estimated GPU hours。

预估：

| Lane/Phase | logical cells | server starts | GPU时间 |
| --- | ---: | ---: | ---: |
| Closeout | `24–30` | `14–16` | 约`1.5–2.5h` |
| Phase6 | `21–24` | `6–8` | 约`1.0–1.5h` |
| Phase7 practical=NONE | `55–70` | `25–35` | 约`3–5h` |
| Phase7 practical存在 | `70–92` | `35–50` | 约`5–8h` |
| primary补restart/async/重试上界 | — | 总计最高约`90` | `10–14h` |

关键路径是`max(Closeout, Phase6工程+pilot) + Phase7`，不是把Closeout与Phase6简单相加。

Early-stop：

1. Closeout screening后只给k0/winner补restart；
2. Phase6只做validity；
3. Phase7先1 restart screening；
4. 只补预声明primary；
5. 工程INVALID立即停止路径；
6. practical=NONE时跳过practical scheduler/HiCache；
7. host feasibility失败则停止host/prefetch；
8. async不存在则不做prefetch性能。

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

- candidate manifest；
- recovery N-amortization；
- scheduler revalidation/main matrix；
- host feasibility/demand-load；
- prefetch functionality/async；
- variable-size offline optimum；
- per-restart compact/raw SHA；
- logical cells/server starts/rounds/GPU-hour budget；
- final dual review。

## 13. Review Disposition Mapping

| Review范围 | V4锚点 | 状态 |
| --- | --- | --- |
| C-01–C-16 Phase4 provenance/fairness | CL0、§5、CL1/CL2 | accepted |
| C-17–C-23 R1/R3/R4 | CL1、P7-1/P7-2、defer | accepted/conditional |
| C-24–C-38 Phase5 metrics/prefetch | CL3、P6-4、P7-2/P7-4 | accepted |
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

- V3已归档为`IMPLEMENTATION_PLAN_V3_ARCHIVED.md`。
- V4已完成GPT-5.6 Sol Max Thinking与Claude Opus 5 Max Thinking独立review、全文互换、交叉consolidate与最终delta verification。
- 双模型确认V4定稿P0已闭合；generic host canary使用`P6-H`避免复用历史`P6-5`标签。

以下状态于2026-07-27T11:55:00-07:00更新（此前的2026-07-26版本已过时）：

- Closeout CL0已完成；R2/R5 final head为`ce55860a9`/`71f15d5d1`。
- Phase6实现分支`research/cross-store-substrate`已推送；head随本轮修订滚动，
  以远程分支为准（本地与远程SHA每次push后核对一致）。
- 全部门禁均已在Docker SM75镜像内执行。

| 门禁 | 状态 | 结论 |
| --- | --- | --- |
| CL0 | 完成 | authority manifest与supersession已冻结 |
| CL1 | 完成 | `practical family = NONE`（冻结规则下成立；因果归因见§14.1） |
| CL2 | 完成 | gate `inconclusive`，显式waive为provisional chunk `1024` |
| CL3 | 完成 | S4优势仅在workflow-only分母成立 |
| P6-H | **通过** | `status=valid`；1 restart/2 round的8-token输出canary与dense一致 |
| P6-4 | **完整跑通** | 三个S4 cell中的四个non-R4 profile可达；两个capacity-limit cell有独立死亡态证据；双向pressure通过 |
| P6-F | **通过** | fault-injected integrated fallback canary；自然压力可达性未证明 |
| CL4 | **完成** | formal Exit与targeted delta reviews全部完成，无开放P0/P1 |

- 本轮共修复6个缺陷：P0 prefix自我覆写、P6-H reseed断言、
  P1-1 SWA释放元数据、P1-3 provisional slot泄漏、P1-2 stale victim重试，
  以及P6-4 runner逐cell容错。
- `technical_exit = PASS WITH CAVEATS`（§7.9.1与
  `PHASE6_EXIT_DISPOSITION.json`）。诊断C v1曾声称该项受阻于我方回收缺陷，
  该结论**已撤回**；v2/v3以`0.05s`采样证明S0/rho2与S4/rho3为真实容量耗尽。
- Phase7 primary manifest仍未预注册，是Phase7 Entry的第二个缺口。
- 未进入Phase7。

## 15. Phase6 Exit剩余阻塞与V5待冻结修订（草案，尚未生效）

### 15.1 为什么现在仍不发布V5

状态更新 2026-07-27（本节随执行结果滚动更新，用户已明确指示本轮不升级版本）：

| V5前置条件 | 状态 |
| --- | --- |
| 1. P0修复完成并有专门回归 | **已完成**，GPU验证通过 |
| 2. CL1在修复后底座重跑并重新判定 | **已完成**；`NONE`仅是被测实现与冻结规则下的promotion结果，未排除header-dependent缺陷 |
| 3. P6-H通过 | **已完成**；`status=valid`仅代表1 restart/2 round的8-token输出canary通过，不是bitwise KV或logit保真证明 |
| 4. P6-4完整四rho矩阵valid或明确不可达 | **已完成**；三个S4 cell中的四个non-R4 profile可达，R4与两个capacity-limit cell均有明确结论；所有顶层full-matrix cell仍为`diagnostic-unavailable` |
| 5. CL4双模型review与disposition | **已完成**；formal Exit与targeted delta reviews最终均关闭P0-1/P0-3，无开放P0/P1，结论`PASS WITH CAVEATS` |

最终状态：

```text
technical_exit = PASS WITH CAVEATS
```

直接证据包括：双向pressure（exact→approx `47.5GB`、
approx→exact `58.8GB`）、R1-like worst-case（k32）profile可达、
P6-H有限范围的输出canary、两个容量不可达cell的独立死亡态遥测，以及
P6-F集成fault-injected fallback canary。

**2026-07-27诊断C（v2）结论**：以`0.05s`采样确证，S0/rho2的OOM是
**真实容量不可达**——死亡瞬间approximate store为`0`字节`0`记录，
可用`704` token而请求需`1024`；且exact压力此前已成功从approximate对象
回收`2.2GB`，**回收路径工作正常，不存在缺陷**。

（诊断C v1曾以`0.4s`粗采样得出“这是我方缺陷”的相反结论，已撤回。
采样间隔必须短于workload的分配动态，否则会得到自信但错误的判断。）

§15.1冻结的test-only验收合同已全部满足，两位targeted reviewer均关闭全部
P0/P1，最终判定为PASS（blocker强度）/PASS WITH CAVEATS（全Exit强度）。

本文件继续保持`Current / Latest`，不提升版本号、不归档
（用户已明确本轮不升级）。

上述步骤现已全部完成。`PHASE6_EXIT_DISPOSITION.json`记录最终主会话结论。

**版本决策**：Phase6结果已稳定、Phase7分支已由`practical=NONE`显著收窄，
现在**有必要创建result-bound V5**。但本轮不自动进入Phase7：

1. 归档V4；
2. 创建V5并写入实际candidate、chunk waiver、矩阵裁剪、预算与early-stop；
3. 按§1执行Sol/Opus独立review、互换、consolidate与主会话disposition；
4. 预注册Phase7 primary manifest；
5. 等待用户明确授权后才运行Phase7。

### 15.2 已由执行结果确定、必须写入V5的合同修订

以下修订与“最终选出哪个candidate”无关，只与证据合同有关，已可冻结为待纳入项：

1. **guardrail语义歧义必须先消解。** §5.9把8-token canary定义为“记录逐token
   一致率、不扩展semantic correctness claim”，但冻结的CL1 runner把8-token
   完全一致当作promotion硬门。两者必须二选一并在重跑前写死，否则同一歧义会
   在每次重跑重现。
2. **fallback证据分级。** 带label的Prometheus counter在未发生事件时不会输出
   任何series，因此“counter缺失”只能记为`indirectly_verified`，不得记为显式
   `0`。该规则已在代码中强制。
3. **chunk披露强制化。** 任何recovery speedup claim必须同时声明
   `chunked_prefill_size`与`max_prefill_tokens`，并附带一个prompt可单chunk
   容纳的对照点。CL2证明body1024在chunk`1024`下为`1.549x`、在chunk`4096`下
   仅为`1.025x`。
4. **Phase6 Exit Gate新增数据保真条目。** §7.9当前只要求安全竞争、双向
   pressure、可回滚、无泄漏，没有任何一条要求“近似reuse在压力下必须与matched
   dense逐token一致”。正因为缺这一条，该底座通过了三轮review和全部CPU回归，
   却在GPU压力下返回损坏KV。V5必须把它列为独立的Exit条件。
5. **新增压力态保真回归。** 需要一个在真实device压力下比较近似reuse与matched
   dense的回归，且不得只依赖CPU fake allocator。
6. **Phase5结论按分母分列。** CL3证明S4只在workflow-only分母保持优势，
   all-reusable分母下S1/S2/S3/S4彼此不可区分，必须按服务目标分别陈述。
7. **Phase7 primary manifest预注册模板。** Entry条件要求它存在，但目前既无
   manifest也无生成它的runner。
8. **recovery必须在请求自身prefix锁的保护下执行。** 这是本轮P0的直接教训：
   `init_next_round_input`阶段请求尚未加锁，而victim枚举条件恰为
   `lock_ref == 0`，两者叠加使请求可以驱逐并覆写自己的KV。任何新增的
   recovery/分配路径都必须复用`protect_request_prefix`。
9. **统计口径必须如实标注。** 所报“paired p95”实为pooled
   `p95(approx)/p95(dense)`，不是配对统计量；N=1/2/4/8摊销是外推值而非实测；
   CL1只有3个restart级独立单元、CL2为2、CL3多数为1，同一trace内的请求
   不得当作独立重复。禁止使用“within noise”这类未经检验的统计判断。
10. **`practical=NONE`必须限定作用域**：它是冻结promotion规则在本模型、
    合成prompt族、exact-output不变量、本GPU与chunk配置下的结论，
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
16. **`dense_fallback`的标签必须区分“近似恢复失败”与“普通exact-cache
    miss”。** `run_p6_4_capacity_pilot.py:413-419`在profile无approximate
    metadata时，仅凭`cached_tokens < expected`就标为`dense_fallback`，
    这直接导致本轮一条Exit证据被错误采信。Phase7任何fallback报告前必须修复。
17. **cell级状态与profile级状态必须分别陈述。** P6-4完整矩阵每个cell顶层
    都是`diagnostic-unavailable`；可达的是其中的非R4 profile。
    写成“三个cell可达”是过声明。
18. **artifact的`result_git_sha`天然为null**（runner无法知道将来容纳自己
    输出的commit）。因此必须另行维护`RESULT_MANIFEST.json`提供
    file→commit映射、内容哈希与验证命令，并且不得据artifact字段声称
    “provenance完整”。

### 15.3 已完成或作废的旧P0方向

本节原列出的stale-victim两个候选方向已经过时，不再是待办：

- prefix自我驱逐根因已由`protect_request_prefix`修复；
- stale victim已改为隔离、刷新并重试；
- provisional recovery slot已在admission拒绝、waiting abort、下一轮rematch与
  teardown路径回收，且仅在`allocator.free`成功后清除引用；
- SWA/Unified释放元数据已回传。

后续不得再按旧§15.3重复设计或重做这些修复。当前唯一Exit blocker以§15.1
冻结的test-only集成验证合同为准。
