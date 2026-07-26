# 实施计划 V1（归档）：SGLang 有损跨上下文 KV 恢复与高压力调度实验

> 状态：Archived / 已被 `IMPLEMENTATION_PLAN_LATEST.md`（V2）取代。
>
> 本文件保留 2026-07-21 时的原始 Phase 0–9 计划，不再作为当前执行依据。

## 1. 问题与目标

本轮实验不研究 AST、自动切分、label 或 indexing，也不把“有损”理解为量化、低比特或普通 KV pruning。

目标是在同一固定代码段出现在不同 `Architect`、`Coder`、`Debugger` prefix/context 后时，避免完整目标上下文 prefill，比较多种近似 KV 恢复路径，并研究 KVFlow-style priority、eviction、HiCache load-back 和 prefetch 如何组合，最大化客户端观测 TTFT 加速。

唯一性能主目标是客户端 TTFT（包含排队）；同时记录 server-side 分解用于定位瓶颈。正确率、输出一致性、代码质量和语义质量不作为优化指标，最低运行门槛仅为请求不崩溃并成功返回首 token。

## 2. 已确认范围

- 主 workflow 保持 `Architect -> Coder -> Debugger`，并包含 `Debugger -> Coder` 的顺序 retry trace。
- 第一阶段只做 sequential workflow；并发、跨 workflow 竞争和并发 prefetch 明确后置。
- workload 使用一个手工固定的大代码段，在不同 role/prefix/position 下重复出现；不研究如何选择或切分该代码段。
- 可以使用 synthetic code-like/token-calibrated 数据制造稳定、可重复的压力。
- 有损恢复路径允许：
  - raw canonical/other-context KV reuse + Key RoPE relocation；
  - KVCOMM-style base KV + context offset/anchor interpolation；
  - CacheBlend、Cache-Craft、EPIC、CacheTune 风格的局部 recompute/repair；
  - dense full prefill 作为 baseline 和机械失败 fallback。
- 不做 KV quantization、mixed precision、INT8/INT4、普通 token/head/layer pruning。
- 论文发现和论文机制事实只以已配置的 arXiv/alphaXiv MCP 为依据。
- Git fetch、branch、编辑、依赖下载、构建、测试、server 和 benchmark 都在 Docker 内执行；宿主机只启动容器、挂载目录和收集结果。
- `ccdd2023/main` fast-forward 到 upstream `main`，然后创建并推送 `latest-main`。
- 本地 RTX 2080 SUPER/SM75 只做 Docker 内小模型功能和压力实验；最终在 RTX PRO 6000/SM120 上 scale。

## 3. 当前代码与分支状态

- `ccdd2023/sglang:main` 当前为 `3343a79466aa714d34a14d08d3929f7953a47212`。
- upstream `sgl-project/sglang:main` 当前为 `c0ed009f5b566be023661bd4e93065b8b4b8b31f`。
- GitHub compare 显示旧 fork main 是 upstream main 的祖先，upstream ahead `4654` commits、behind `0`，因此可执行 fast-forward-only 同步。
- 远程当前不存在 `latest-main`。
- 本地 SM75 donor：
  - 工作区 `/home/chris/Workspaces/kvcache-research/sglang-running`；
  - 分支 `fix/qwen3-0.6b-docker-sm75`，HEAD `845a49088`；
  - 核心 patch 位于 `patches/sm75-native-fallback.patch`；
  - 只应迁移 `activation.py`、`layernorm.py` 的 capability `< 8` native fallback，旧 `qwen3.py`/backend 适配必须按最新 upstream 重审。
- KVFlow donor：
  - `feature/workflow-priority`，HEAD `5bb9afc9234aa9caa9df51e87f119e5bfaf186de`；
  - 有 priority eviction、HiCache benchmark 和 sequential round-robin harness；
  - 历史 prefetch 在 sequential pressure 下出现 churn，不能默认开启。
- 近似 KV 数据面 donor：
  - `integration/coding-aware-prefetch`，HEAD `d4a7ec132d80597c7b55a562beb8432e804ab127`；
  - 可继承 segment identity/store、generation/lease、full-key RoPE rotation、coverage validation、middle-KV handoff 和 transfer stats；
  - 仅完成接口骨架，没有真实 scheduler/request 自动接线，也没有 KVCOMM base/offset/anchor 算法。
- `fix/placeholder-pool-activation` 只作为 HKVD measurement、benchmark 和负结果档案 donor；其 AST、截断 signature、Unicode offset、gap、slot lifecycle 和 “True CacheBlend” 路径不能作为实现基线。
- 最新 upstream 的 `PriorityStrategy` 语义是“较小 `node.priority` 先淘汰，然后 LRU”；历史分支的语义反转不能直接 cherry-pick。实施时必须将 request scheduling priority 与 KV eviction protection score 分离。

## 4. 论文机制到实验路径的映射

| 路径 | arXiv 依据 | MVP 解释 |
| --- | --- | --- |
| Raw reuse + RoPE | KVCOMM `2510.12872` 的位置修正组成部分；历史分支已有 copy-and-rotate primitive | 不做 context offset 或 repair，作为最低恢复成本/最高损失的 TTFT 上界 |
| KVCOMM anchor | KVCOMM `2510.12872` | canonical base、placeholder/neighbor prefix `ΔK/ΔV`、single/multi-anchor interpolation；另设 always-share speed mode，不把它称为 faithful 结果 |
| EPIC fixed-k | EPIC `2410.15332` | 每个固定 chunk 重算前 `k` 个 token，扫描 `k=0/2/4/8/16/32` |
| Selective repair | CacheBlend `2405.16444`、Cache-Craft `2502.15734` | 固定比例、HKVD/deviation-guided 两种 token 选择；只比较 TTFT，不使用论文质量结论 |
| Hardware-aware repair budget | CacheTune `2605.24022` | 根据实测 H2D 与 recompute 开销选择 `k`/repair ratio，目标只取 TTFT 最小值 |
| Workflow eviction | KVFlow `2507.07400` | steps-to-execution、future-use priority、HiCache load-back |
| Cost/size priority | RAGCache `2404.12457` | 借鉴 `frequency * recompute_cost / size`，改成 sequential trace 的 future distance 与 recovery saving |
| Conservative prefetch | PBKV `2605.06472` | sequential 首版仅允许 free-space/retired-only，不为 prefetch 驱逐活跃高价值 cache |

## 5. 总体实施策略

采用“共享数据面 + 多恢复策略 + 多调度策略 + 同一 benchmark harness”的结构，不押注单一路径。

先建立 raw-reuse 上界和稳定的高压力 harness，再独立加入 EPIC-like、selective repair、KVCOMM anchor。恢复路径通过统一接口输出完整连续的 prefix KV；scheduler 只决定对象保留、tier、恢复策略和 prefetch admission，不把算法逻辑散落到 Radix tree。

实验采用分阶段筛选，避免直接跑完整笛卡尔积：

1. 无压力/轻压力 microbenchmark，测每条恢复路径本身的 TTFT 与开销。
2. 在 `rho >= 1` 的 GPU oversubscription 下筛选恢复路径。
3. 只将前两条恢复路径与全部 scheduler 组合。
4. 对最终候选做完整 pressure、prefix length、role position 和 retry-distance sweep。

## 6. 实施阶段

### Phase 0：Docker-only Git 同步与实验分支

1. 先把本计划的当前方向、范围和决策同步到仓库 `PROJECT.md`、`TRACKING.md`、`HANDOFF.md`；plan mode 当前禁止修改仓库，因此该项在退出 plan mode 后立即执行。
2. 在容器内显式验证 GitHub 写身份为 `ccdd2023`，不输出凭据。
3. 在容器内配置只读 upstream remote，核对远程 SHA。
4. 对 `ccdd2023/main` 执行 fast-forward-only 到 upstream `c0ed009f...`；若远程在执行前变化，则重新固定新 SHA 并记录 compare 结果。
5. 从同步后的 fork main 创建并推送 `latest-main`。
6. 固定 manifest：fork/upstream SHA、branch SHA、Docker image digest、CUDA/PyTorch/SGLang/model/tokenizer revision。

验收：fork main 与固定 upstream SHA 一致；`latest-main` 指向同一基线；工作树无意外修改。

### Phase 1：迁移 SM75 patch 并建立双硬件兼容基线

1. 仅迁移 `activation.py`、`layernorm.py` 的 SM75 native fallback。
2. 对最新 upstream 的 fused-op dispatch、kernel 包名、FlashInfer 和 attention backend 自动选择重新核对，不复用旧 API 假设。
3. 保证 capability `< 8` 才走 fallback，SM80+ 和 SM120 不受影响。
4. 在 Docker build 中完成完整 runtime image 构建；不在宿主机安装或构建依赖。
5. Docker 内运行 targeted activation/layernorm tests。
6. 使用 Qwen3-0.6B 启动 server，通过 health、model info、chat completion 首 token smoke。

验收：SM75 请求可返回首 token；无 SM75 patch 时的 SM80+/SM120 路径不被改写；完整镜像构建成功。

### Phase 2：Sequential TTFT harness 与压力校准

新增一个独立 benchmark，优先复用：

- `benchmark/priority/bench_priority.py` 的 token-calibrated sequential client；
- `benchmark/multi_workflow/bench_kvcomm_ttft_stress.py` 的 length bucket、HiCache 和 TTFT telemetry 思路。

必须先修复旧 stress harness 的跨 mode prompt 污染，且不继承 AST/codebase hints。

固定 trace：

```text
Architect
-> Coder
-> Debugger(fail)
-> Coder(retry)
-> Debugger(success)
```

每一步使用冻结的 role prefix、固定代码段和固定 synthetic suffix；`max_new_tokens=1`，使 TTFT 主导。另保留以下 sequential stress 变体：

- 三阶段 cycle；
- retry-heavy loop；
- 增加冷 filler step 的长 reuse-distance trace；
- 可选四阶段 `Architect -> Coder -> Debugger -> Tester`，只作 synthetic stress，不改变项目固定 workflow。

压力定义：

```text
rho = active reusable KV bytes / evictable GPU KV capacity
```

运行前从 SGLang 实际 allocator/cache metrics 读取可用 token/page capacity，不只用理论估算。通过以下旋钮独立控制压力：

- 固定代码段 token 长度；
- 同时保留的 role/context variant 数量；
- sequential task 数量；
- `--mem-fraction-static` / `--max-total-tokens`；
- HiCache ratio。

首轮 pressure 档位：

```text
rho = 0.5, 0.9, 1.1, 1.5, 2.0, 3.0
```

HiCache host capacity在首轮保持足以容纳全部 working set，先隔离 GPU eviction 与 H2D；host oversubscription 后置。

每个配置独立重启 server、清空 cache、执行固定 warmup，再做重复 measured rounds。配置顺序随机化，保存原始逐请求 JSON/CSV。

### Phase 3：Approximate KV 独立数据面

从 `integration/coding-aware-prefetch` 移植并按最新 upstream 调整：

- segment key/handle/store；
- generation、lease、pin/unpin、reset；
- reuse plan、copied spans、dense ranges；
- full coverage validation；
- raw KV copy + all-layer Key RoPE delta rotation；
- transfer stats 和 dense fallback。

近似 KV store 与 exact Radix 完全隔离：

- exact Radix 只记录真实完整 forward 得到的 exact prefix；
- approximate request 不把 reconstructed KV 写回 exact Radix；
- exact prefix 之后至倒数第二个 prompt token必须由 `copied_spans + dense_ranges` 完整覆盖；
- 最后一个 prompt token固定做真实 1-token forward，以产生首 token logits；
- stale handle、residency miss、token mismatch、gap 或 slot failure 立即走 dense fallback。

验收：raw reuse path 在不同正/负/零 position delta 下均可返回首 token；无未初始化 gap、stale slot、double free 或 allocator leak。

### Phase 4：并列实现多条有损恢复路径

#### R0：Raw reuse + RoPE

- canonical KV 直接复制到目标 slots；
- 只修正 Key 的目标位置；
- 不加 context offset、不局部重算；
- 作为速度上界，不声称 faithful KVCOMM 或正确恢复。

#### R1：EPIC-like fixed-k repair

- 每个固定 chunk 的前 `k` token做真实 dense repair，其余 body 复制并重定位；
- 扫描 `k=0/2/4/8/16/32`；
- 使用统一 transfer plan，不在 scheduler 中逐 token mini-prefill。

#### R2：Selective repair

- 固定 leading fraction 作为低开销基线；
- HKVD/deviation-guided selection 作为第二实现；
- 扫描 repair ratio `1%/5%/15%/30%`；
- 不沿用旧 “True CacheBlend” 逐 token scheduler prototype。

#### R3：KVCOMM anchor

按递增复杂度实现：

1. canonical base only；
2. single/self-anchor `ΔK/ΔV`，用于测量 reconstruction 开销下界；
3. per-role anchor；
4. multi-anchor soft interpolation；
5. neighboring-prefix offset。

提供两种明确分离的运行模式：

- `paper-mechanism`：保留 length/entropy/shareability gate 和 dense fallback；
- `speed-only`：机械条件满足即复用，不因质量风险 fallback。

任何 speed-only 结果不得描述为 faithful KVCOMM 质量结果。

#### R4：Hardware-aware repair controller

- 对每个 chunk length、tier 和硬件测量 `T_dense`、`T_H2D`、`T_copy_rope`、`T_anchor`、`T_repair(k/r)`；
- 在请求前选择预测 TTFT 最低的恢复路径或 repair budget；
- 第一版使用离线 profile table，不引入在线学习。

### Phase 5：并列实现多种 eviction/scheduler

所有策略使用专门的 cache protection metadata，不复用可能承担 request scheduling 语义的通用 `priority` 字段，除非最新 upstream 审计证明无冲突。

#### S0：LRU

作为无 workflow 信息基线。

#### S1：KVFlow steps-only

只使用下一次 stage 的 step distance；共享 base/anchor 对象取所有依赖 stage 中最紧急者。

#### S2：Belady oracle next-use

synthetic sequential trace 已知完整未来，因此直接驱逐 next-use 最远的对象，作为 eviction 上界。

#### S3：Recovery-aware value density

定义对象的保留价值：

```text
saved_ms = T_dense - T_selected_recovery
value_density = saved_ms / resident_bytes
score = value_density / (1 + next_use_distance)
```

分别对 exact bundle、canonical base、anchor delta 和 partial-repair metadata 计价。

#### S4：Hierarchical object policy

按对象依赖关系分层：

1. 先淘汰无未来使用对象；
2. 再淘汰可由更低 tier 恢复的 exact variant；
3. 再淘汰低 value-density anchor；
4. 最后淘汰被多个 future stages 共享的 canonical base。

#### Prefetch 变体

- P0：关闭；
- P1：只使用空闲 GPU space；
- P2：只允许驱逐无未来使用对象；
- P3：oracle next-stage prefetch，上界实验。

首轮 sequential 结果以 P0 为默认；历史上导致 active-cache churn 的强制 prefetch 不作为默认配置。

### Phase 6：实验矩阵与筛选

基线：

| ID | 配置 |
| --- | --- |
| B0 | 完整目标上下文 dense prefill |
| B1 | exact prefix + GPU LRU |
| B2 | exact prefix + LRU + HiCache |
| B3 | exact cache + KVFlow steps-only |
| B4 | 固定有损恢复路径 + LRU，无 workflow scheduler |

恢复路径：

```text
R0 raw+RoPE
R1 EPIC fixed-k
R2 selective repair
R3 KVCOMM anchor
R4 hardware-aware selector
```

调度路径：

```text
S0 LRU
S1 steps-only
S2 oracle next-use
S3 value-density
S4 hierarchical object policy
```

分层：

```text
T0 GPU-only
T1 GPU + HiCache demand load
```

筛选顺序：

1. `R0-R4` 在 S0、无 prefetch 下做 recovery microbenchmark。
2. 选择 TTFT 最低的两条恢复路径。
3. 将两条路径与 `S0-S4` 在 `rho=1.1/1.5/2.0/3.0` 组合。
4. 对前两名加入 P1-P3。
5. 最终候选再扫代码段长度、role prefix 长度、position delta、retry distance 和本地/远程硬件。

不直接运行全部笛卡尔积，避免把 GPU 时间消耗在明显劣势配置。

### Phase 7：指标与有效性判定

主指标：

- client-observed TTFT p50/p95；
- 相对 B0 dense 和当前最优 exact baseline 的 speedup。

解释指标：

- queue wait；
- exact lookup；
- host lookup；
- H2D time/bytes；
- raw copy + RoPE time；
- anchor interpolation time；
- repair token 数与 repair time；
- last-token forward；
- first-token decode；
- GPU/host hit；
- eviction/load-back 次数；
- dense fallback 次数；
- wasted prefetch 与 churn bytes；
- peak/steady GPU KV pages 和 host bytes。

最低有效性门槛：

- 请求完成率 100%，server 无 crash/OOM/allocator corruption；
- 在至少两个 `rho >= 1.5` 档位上，p50 TTFT 相对最优 exact baseline 有稳定正收益；
- p95 不因 eviction/load-back 抖动抵消 p50 收益；
- 三次独立 server restart 后趋势一致；
- 若所有实用路径都不优于 exact baseline，保留 R0/S2 作为速度上界并停止扩大复杂度。

不收集或不使用 semantic accuracy、代码正确率、输出一致性、KL、pass@1 作为筛选条件。

### Phase 8：RTX PRO 6000 scale

1. 先执行 SM120/container/server smoke。
2. 使用与本地相同 Git SHA、image digest、model/tokenizer revision 和 trace。
3. 运行 7B/8B、长代码段、更大 GPU KV working set 和 HiCache H2D。
4. 只复测本地筛出的前两条恢复路径与前两种 scheduler。
5. 保存完整 machine manifest、PCIe、RAM、disk 和 driver 信息。

RTX PRO 6000 只验证趋势和系统机制，不等同 H100 论文主实验环境。

### Phase 9：后置工作

只有当 sequential high-pressure 下已经找到有效方法后，才进入：

- 多 workflow 并发；
- dynamic branch predictor；
- status-aware skip scheduling；
- concurrent prefetch overlap；
- host cache oversubscription；
- 真实 code-agent closed-loop quality/patch 测试。

这些内容不属于首版实施。

## 7. 预计修改组件

具体文件名在同步最新 upstream 后复核，目标结构如下：

```text
python/sglang/srt/mem_cache/approx_kv/
  config.py
  types.py
  store.py
  transfer.py
  radix_backend.py
  recovery/
    raw_rope.py
    epic_fixed_k.py
    selective_repair.py
    kvcomm_anchor.py
    hardware_selector.py
  scheduling/
    policy.py
    oracle.py
    telemetry.py

benchmark/approx_kv/
  bench_ttft_pressure.py
  run_docker.sh
  workloads.py

test/registered/unit/mem_cache/
  test_approx_kv_transfer.py
  test_approx_kv_rope.py
  test_approx_kv_recovery.py
  test_approx_kv_eviction.py
```

需要重新核对的 upstream 接口：

- `memory_pool.py` 的 `move_kv_cache`、`get_key_buffer`、`set_kv_buffer`、CPU copy/load；
- `schedule_batch.py` / scheduler 的 `prefix_indices`、`extend_input_len`、last-token forward；
- `radix_cache.py`、`hiradix_cache.py`、HiCache storage state；
- `evict_policy.py::PriorityStrategy` 当前排序方向；
- rotary embedding helper 路径和 `is_neox_style` 参数；
- `CacheInitParams` 与 server args。

## 8. 风险与处理

| 风险 | 处理 |
| --- | --- |
| 历史 priority 语义与最新 upstream 冲突 | 不直接 cherry-pick 语义反转；独立 cache protection score |
| approximate KV 污染 exact Radix | 独立 store，近似 KV 禁止写回 exact tree |
| 非连续 KV/gap 导致未初始化读取 | `require_full_coverage` 强校验，最后 token真实 forward |
| SM75 patch 破坏 SM120 | capability guard 仅 `<8`，远程做 SM120 smoke |
| recovery 路径过多导致组合爆炸 | successive screening，只组合前两名 |
| sequential prefetch churn | P0 默认，P1/P2 保守，P3 只作 oracle 上界 |
| 旧 benchmark prompt 污染 | 每个 mode/配置独立构造 trace并重启 server |
| 论文机制被误标 | paper-mechanism、inspired、speed-only 三类结果明确分开 |
| 本地小模型结论不具规模性 | RTX PRO 6000 复测前两名，不把 SM75 数字当最终性能结论 |

## 9. Todo

| ID | 任务 | 依赖 |
| --- | --- | --- |
| sync-project-docs | 将本轮范围、决策和计划同步到 `PROJECT.md`、`TRACKING.md`、`HANDOFF.md` | 无 |
| sync-latest-main | Docker 内 fast-forward fork main 并创建 `latest-main` | sync-project-docs |
| port-sm75 | 在最新 main 迁移 SM75 native fallback | sync-latest-main |
| docker-baseline | 完整 Docker build、单测与 Qwen3 smoke | port-sm75 |
| pressure-harness | 建立 sequential TTFT/high-pressure harness 与 telemetry | docker-baseline |
| approx-data-plane | 移植独立 approximate KV store/transfer/RoPE/coverage core | docker-baseline |
| recovery-raw | 实现 R0 raw+RoPE | approx-data-plane |
| recovery-epic | 实现 R1 fixed-k repair | approx-data-plane |
| recovery-selective | 实现 R2 selective repair | approx-data-plane |
| recovery-kvcomm | 实现 R3 base/offset/anchor | approx-data-plane |
| recovery-selector | 实现 R4 hardware-aware selector | recovery-raw, recovery-epic, recovery-selective, recovery-kvcomm |
| scheduler-policies | 实现 S0-S4 与 P0-P3 | pressure-harness, approx-data-plane |
| local-screening | 本地 sequential pressure 筛选恢复和调度路径 | recovery-selector, scheduler-policies |
| pro6000-scale | RTX PRO 6000 复测最终候选 | local-screening |
| concurrency-deferred | 并发 workflow 和并发 prefetch | pro6000-scale，且仅在 sequential 有效时 |

## 10. 计划完成定义

本计划的首版完成条件不是实现所有论文机制，而是：

1. `latest-main` 基线和 SM75 Docker 环境稳定；
2. 至少三条不同的跨 context 有损 KV 恢复路径可返回首 token；
3. 至少四种 eviction/scheduler 策略在同一 sequential high-pressure harness 下可比较；
4. 找到一个在多个高压力档位上稳定优于最优 exact baseline 的 TTFT 组合，或以数据证明当前路径无收益；
5. 最终候选在 RTX PRO 6000 上完成同口径复测；
6. 并发明确留到后续阶段，不混入首版归因。
