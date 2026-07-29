# Phase 6 正式研究报告：Cross-Store Substrate、正确性与容量可行性

> 报告类型：正式阶段研究报告（自包含、可审计）
> 覆盖阶段：Phase 6（P6-0 / P6-4 / P6-F / P6-H + 诊断 C/D + Formal Exit Review）
> 撰写时间：2026-07-28
> 报告状态：`最终权威`（叙事层）；技术 Exit = **`PASS WITH CAVEATS`**
> 关联报告：[Phase4](PHASE4_RECOVERY_METHODS_REPORT.md)｜[Phase5](PHASE5_WORKFLOW_SCHEDULING_REPORT.md)｜[Phase7](PHASE7_INTEGRATED_EVALUATION_REPORT.md)｜[跨阶段总报告](PHASE4_TO_PHASE7_SUMMARY.md)

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

Phase 6 是本项目唯一一个**以系统正确性与容量可行性为主要交付物、明确禁止性能声称的阶段**。它的任务是让 exact 与 device-approximate 对象在同一 device 预算下安全竞争，同时用独立 host limit 管理 host 副本，并诚实地给出「哪些配置在本硬件上根本达不到」。exact–host / HiCache 统一不在本阶段范围内。

`impl:benchmark/approx_kv/results/phase6/p6-4.json` 与 `p6-h.json` 的 `performance_claim` 字段均显式为 `"disabled"`。

### 1.2 证据状态总览

| 证据源 | 状态 | 说明 |
| --- | --- | --- |
| `impl:benchmark/approx_kv/results/phase6/PHASE6_EXIT_DISPOSITION.json` | **`最终权威`** | 10 项 Exit gate 的最终判定，`technical_exit="pass_with_caveats"`，`phase7_authorized=false` |
| `impl:benchmark/approx_kv/results/phase6/RESULT_MANIFEST.json` | **`最终权威`** | file→commit 映射，`--check` 通过 `48/48` |
| `impl:benchmark/approx_kv/results/phase6/p6-0-contract.json` | `最终权威` | fixed40 workload 与 schema 合同，`contract_sha256=a498daa3…` |
| `impl:benchmark/approx_kv/results/phase6/p6-4.json` | `最终权威` | fixed40 capacity pilot 最终结果（`run_id=p6-4-20260727T104820Z`） |
| `impl:benchmark/approx_kv/results/phase6/p6-h.json` | `最终权威` | host roundtrip + same-context 输出 canary（`raw_sha256=842c3563…`） |
| `impl:benchmark/approx_kv/results/phase6/p6-f-v3-fallback-canary.json` | `最终权威`（fault-injected 强度） | reservation-failure → dense fallback canary |
| `impl:benchmark/approx_kv/results/phase6/diagnostic-C-store-state-at-oom.json`（schema_version=3） | `最终权威` | 0.05s 采样的死亡瞬间 store 状态 |
| `impl:benchmark/approx_kv/results/phase6/diagnostic-C-v1-SUPERSEDED-coarse-sampling.json` | **`历史/已被替代`** | 0.4s 采样，结论完全错误 |
| `impl:benchmark/approx_kv/results/phase6/p6-4-outcome-correction.json` | `最终权威`（更正） | 撤回「12 次 GPU dense fallback」与 r4_like fallback 归类 |
| `impl:benchmark/approx_kv/results/phase6/p6-4-reduced-profiles-4x-bytes-per-token-probe.json` | `最终权威`（更名后） | 原名 `p6-4-fallback-injection.json`，因「没有任何 fault injection」而误导，已更名 |
| `impl:benchmark/approx_kv/results/phase6/p6-h-attempt1-chunk1024-failed.json` | `历史/已被替代` | P6-H 首次尝试 OOM 失败记录 |
| `impl:benchmark/approx_kv/results/phase6/p6-4-exact-only.json` | `历史/已被替代` | P0 修复前的隔离运行 |
| `impl:python/sglang/srt/mem_cache/cross_store/*`（1073 行） | 实现证据 | 底座源码 |

### 1.3 Executive Summary

1. **原 Phase6 计划被作废并重定义。** 原因：Phase5 只验证 exact Radix、未管 approx store；S1–S3 高压下未稳定优于 S0；P2/P3 无稳定收益；Phase5 rho sweep 混杂对象组成；R2/R5 仍是 precomputed adapter（`docs:PROJECT.md:2617-2626`）。新目标是在**同一 fixed sequential workflow、同一逻辑对象集合、同一 GPU 预算**下，让 exact Radix 对象与 Phase4 approximate 对象**真实竞争**。

2. **底座实现完成**：`cross_store/` 共 1073 行，含 frozen dataclass 对象模型、DAG 反向依赖闭包驱逐、reserve/commit/release/demote/promote 预算、S0_LRU 与 S4_HIERARCHICAL 两种策略序、allocator 的原子回滚与 stale victim 处理、coordinator 的双 store 资源聚合。

3. **发现并修复了 11 类实现/治理缺陷，并纠正了 1 项测量方法错误**（正文 Bug 1–12）。其中最严重的是 **P0：请求自身的 exact prefix 未加锁导致自我驱逐与自我覆写**——恢复窗口发生在 `init_next_round_input`，早于 `add_one_req` 获取 prefix 锁，而 victim 枚举条件恰为 `lock_ref == 0`。修复为 `protect_request_prefix` 上下文管理器（提交 `af81934e4`）。

4. **另一类真实资源泄漏**（P1-3）：recovery 在 scheduler admission **之前**分配并挂载 device slot，若 `add_one_req()` 拒绝该请求则 recovered suffix 不被释放。修复为 provisional 所有权模型（提交 `40f09c1fe`，补漏 `scheduler.py:3045` / `:4090`）。

5. **一次重大的测量方法论翻转**：诊断 C v1 用 `0.4s` 轮询，得出「S0/rho2 OOM 是我方缺陷」的**自信但完全错误**结论；改为 `0.05s` 后结论反转为**真实容量不可达**。同时确认 **`num_used_tokens` 已包含 approximate store 占用的 slot，两者不可相加**。

6. **fallback taxonomy 误标被撤回**：所谓「12 次 GPU dense fallback」全部来自 `exact_only` profile 的普通 exact-cache miss；`r4_like` 的 4096 fallback token 实为 registration 容量失败。

7. **最终 Exit = `PASS WITH CAVEATS`**，10 项 gate 中 3 项 `satisfied`、4 项 `satisfied_with_scope`、1 项 `satisfied_for_completed_runs`、1 项 `verified_at_fault_injected_canary_strength`、1 项 `satisfied_for_formal_exit_package`。`RESULT_MANIFEST --check` 通过 `48/48`。

8. **永久 caveat：自然压力下的 reservation-failure fallback 可达性未被证明。** P6-F 只在 **fault-injected** 强度上验证了 integrated canary（`fault_injected=true; natural_pressure_reachability=false`）。

9. **Phase6 通过不自动授权 Phase7。** `phase7_authorized=false`。

---

## 2. Phase 6 动机、研究问题、冻结假设与非目标

### 2.1 为何原 Phase6 作废

`docs:PROJECT.md:2617-2626` 列出五条理由：

1. Phase5 只验证 exact Radix，完全没有管 approximate store；
2. S1–S3 在高压下未稳定优于 S0；
3. P2/P3 无稳定收益；
4. Phase5 的 rho sweep 混杂了对象组成；
5. R2/R5 仍是 precomputed adapter，不是 practical 路径。

在这些前提下，原计划的「5 恢复路径 × 5 scheduler 笛卡尔积」既不可执行也无意义。

### 2.2 新核心目标

在**同一 fixed sequential workflow、同一逻辑对象集合、同一 GPU 预算**下，让 exact Radix 对象与 Phase4 approximate 对象真实竞争，覆盖四条臂：

```text
dense  vs  exact S0/S4  vs  lossy recovery S0/S4  vs  S4 + HiCache demand load
```

（`docs:PROJECT.md:2628-2637`）

### 2.3 Phase6 / Phase7 再划分

`docs:TRACKING.md:1573-1587`：

- **Phase6**：只做 cross-store substrate / correctness / fixed40 feasibility；
- **Phase7**：承接 scheduler / prefetch / HiCache / practical winner 评测；
- 命名：`P6-H` 替代历史上有歧义的 `P6-5`。

### 2.4 研究问题

| 编号 | 研究问题 | 结论位置 |
| --- | --- | --- |
| RQ6-1 | exact 与 device-approximate 能否共享 device 预算，host 副本能否在独立 host limit 下安全参与 lifecycle？ | §7.1（gate 1、2） |
| RQ6-2 | 分配失败能否优雅回滚，不留泄漏与孤儿？ | §7.1（gate 3、7） |
| RQ6-3 | fixed40 workload 在四个 rho 下是否可行？不可行时能否给出可辩护的「容量不可达」证据？ | §4.4、§4.5 |
| RQ6-4 | 压力下的 approximate reuse 是否与 matched dense 逐 token 一致？ | §4.3（P6-H） |
| RQ6-5 | reservation 失败能否降级为 dense fallback 且输出仍正确？ | §4.6（P6-F） |
| RQ6-6 | R1-like 最坏情况（k32）的 footprint 是否可达？ | §4.5 |

### 2.5 冻结假设（P6-0 contract）

artifact：`impl:benchmark/approx_kv/results/phase6/p6-0-contract.json`，`contract_sha256 = a498daa36449993ff166dd70870005be22a1da0a7d09e97e8f779d72cbf3fb30`，`run_id = phase6-contract-20260726T213959Z`。

| 项 | 值 |
| --- | --- |
| workload | fixed40：40 个对象，`manifest_sha256=30c9ae8de429a6389e58bbdcdf096101cf6296ff14d4e6fcf5c2b87c6b1f0749` |
| body / header | `2048` / `64` |
| `chunked_prefill_size` | `1024`（`chunk_source = "provisional_worst_case"`） |
| warmup / formal / restarts | `1` / `2` / `1` |
| `performance_ranking_enabled` | `false` |
| matched-state | 每 round 重建、每 round 只发 1 个 measured target、approx target 不写回 exact、exact baseline 只用本 round 预构造 source |
| cache outcomes | `exact_gpu_hit` / `approximate_gpu_recovery` / `host_demand_load` / `dense_fallback` |
| status 取值 | `valid` / `negative` / `inconclusive` / `invalid` |
| rho 四口径 | `logical_demand` / `physical_demand` / `resident` / `host` |

representation profiles（决定物理占用倍数）：

| profile | representation kinds | resident multiplicity | temporary multiplicity |
| --- | --- | ---: | ---: |
| `exact_only` | — | 0 | 0 |
| `r0_like` | `canonical_base` | 1 | 1 |
| `r1_like_k32` | `canonical_base`, `repair_state` | 1 | 2 |
| `r2_like` | `canonical_base`, `precomputed_adapter` | **2** | 2 |
| `r4_like` | `canonical_base`, `anchor`, `delta`, `anchor`, `delta` | **5** | 1 |

> **`r2_like` 与 `r4_like` 都是 synthetic footprint profile，分别只表示 2x 与 5x 的表示多重性；它们不执行 CacheBlend，也不执行 KVCOMM。**

### 2.6 非目标

- **只做系统正确性、生命周期与容量可行性结论，不发布性能或 semantic quality claim**（`docs:PROJECT.md:2639-2660`）。
- **不做** 5 恢复路径 × 5 scheduler 笛卡尔积。
- **R2/R5/R3 不进主矩阵。**
- **prefetch / HiCache 仅 canary 级。**
- 不做 exact-host / HiCache 统一（gate 1 的 caveat 明确未实现）。
- 不做 bitwise KV / logit fidelity 声称。

---

## 3. 环境、实现范围、方法与术语

### 3.1 执行环境（Docker 内执行）

| 项目 | 值 |
| --- | --- |
| 镜像 digest | `sha256:0be6e16e2eb288dfd5fa8b0b41015f61731a139fb961d3366ccedf834289d781` |
| 模型 / revision | `Qwen/Qwen3-0.6B` / `c1899de289a04d12100db370d81485cdf75e47ca` |
| GPU | NVIDIA GeForce RTX 2080 SUPER，SM75，8192 MiB（`gpu_memory_bytes=8163426304`） |
| driver | `580.173.02` |
| CUDA / Torch / Transformers / Python | `12.9` / `2.9.1+cu129` / `5.12.1` / `3.12.3` |
| 容器参数 | `--runtime=nvidia --gpus all --ipc=host --shm-size=8g --user 1000:1000` |
| P6-4 `source_git_sha` | `fb284cad4cab774f3b77ae17811f2f7b21cb5ab7` |
| P6-4 plugin env | `SGLANG_APPROX_KV_BYTES_PER_TOKEN=114688`、`SGLANG_APPROX_KV_CORE=1`、`SGLANG_APPROX_KV_CROSS_STORE=1`、`SGLANG_APPROX_KV_HOST_BUDGET_BYTES=0`、`SGLANG_APPROX_KV_REGISTER_EVICTS_EXACT=1` |
| P6-H plugin env | 同上但 `SGLANG_APPROX_KV_HOST=1`、`SGLANG_APPROX_KV_HOST_BUDGET_BYTES=8589934592` |

### 3.2 Cross-store substrate 架构（实现层，只读确认）

代码位于 `impl:python/sglang/srt/mem_cache/cross_store/`，共 1073 行：

| 模块 | 行数 | 职责 |
| --- | ---: | --- |
| `types.py` | 81 | `CrossStoreObject`（frozen dataclass）：`tier`(DEVICE/HOST)、`provenance`(EXACT/APPROXIMATE)、`kind`(EXACT_VARIANT / CANONICAL_BASE / REPAIR_STATE / PRECOMPUTED_ADAPTER / ANCHOR / DELTA / HOST_COPY / MATERIALIZATION_SCRATCH / FILLER)、`pinned/leased/in_flight/reserved` → `protected` 属性、`saved_ms`、`value_density` |
| `object_graph.py` | 131 | `CrossStoreObjectGraph`：DAG register/touch/replace、`eviction_closure`（反向依赖闭包遍历）、`remove_closure`、`assert_no_orphans`、`_has_cycle` 环检测 |
| `budget.py` | 164 | `CrossStoreBudget`：`device/host_limit_bytes`、reserve/commit/release/restore/demote/promote、`peak_device_bytes`、`reset_accounting(force=)`、`seed_usage`/`reconcile_usage` |
| `policy.py` + `class_order.py` | 34 + 40 | `CrossStorePolicy`：S0_LRU 用 `(event_ordinal, object_id)`；S4_HIERARCHICAL 用 `(effective_class, value_density, next_use_key, event_ordinal, object_id)` |
| `allocator.py` | 355 | `CrossStoreAllocator.allocate()`：reserve → 迭代选 victim（demote 优先，否则 eviction closure）→ 执行 action → backend alloc → 逐个 commit → `commit_device`；失败时全部逆序 `undo()` + `release_device_reservation` 并标 `requires_reset`；stale victim（`KeyError`）计入 `stale_victims` 并刷新快照 |
| `coordinator.py` | 201 | `CrossStoreCoordinator._allocate_tokens`：从 `tree_cache`/`approx_store` 读容量算 `device_limit_bytes`，建 `CrossStoreBudget`，选 `PolicyKind`，聚合 `resources()`（exact + approx），注入 `fault_injector`（test-only），调用 allocator，回写 `manager.record_cross_store_eviction/result/reservation_failure` |
| `event_clock.py` | 30 | 单调事件时钟，供 LRU 排序 |

**Host copy 说明**：没有独立的 `host.py` 模块。host 是 `CrossStoreTier.HOST` 枚举值，由 `budget.py` 的 `host_used_bytes` / `host_limit_bytes` 及 `demote/promote` 管理；实际 host backend 是 `allocator_cpu_copy`（`p6-h.json` 的 `host_backend` 字段），且 `hicache_tier_exercised = false`。

### 3.3 Test-only fault injection 机制

代码：`config.py:78,107,160,196`；`coordinator.py:155-170`。

`ApproxKVConfig.cross_store_test_reservation_failure`（默认 `False`，env-gated）→ `coordinator._allocate_tokens` 中，仅当 `requester == "approximate"` 且未消费过时注入 `fault_injector`：在 `AllocationFailurePoint.AFTER_RESERVE` 抛 `RuntimeError("test-only injected cross-store reservation failure")`，one-shot（`_test_reservation_failure_consumed`）。

用于 P6-F（实现提交 `e59bb7a9c`）；artifact 显式标 `fault_injected=true`、`natural_pressure_reachability=false`，**不冒充自然可达**。

CPU 对应回归：`test_fault_injection_rolls_back_reversible_actions`（遍历全部 `AllocationFailurePoint`）、`test_reservation_failure_degrades_to_dense_fallback`（mutation 验证，`11bc9b3e4`）。

### 3.4 术语与测量口径

| 术语 | 定义 |
| --- | --- |
| `reachable` / `diagnostic-unavailable` | profile 级或 cell 级的**可达性**，指该配置能否在本容量下完成 registration 与 replay；**不是 approximate-recovery 成功率** |
| `bidirectional pressure` | exact requester 驱逐 approximate victim 的字节数，与 approximate requester 驱逐 exact victim 的字节数，二者均须非零 |
| `stale victim` | allocator 快照中的 victim 已在同一轮更早的驱逐中被摘出 radix 树 |
| `provisional recovery slot` | 在 scheduler admission 之前分配并挂载的 device slot，所有权尚未转移给请求 |
| `orphan` | object graph 中失去父引用但仍占用资源的对象 |
| `arm_interval_peak_device_bytes` | 自上次完整 reset 以来的本臂 high-water，**不是 process-lifetime peak** |
| `num_used_tokens` | **已包含 approximate store 占用的 slot**，禁止与 store gauge 相加 |

---

## 4. 全部实验：矩阵、执行顺序、核心数值

### 4.1 执行顺序（里程碑）

| 阶段 | 时间 | 内容 / 产出 |
| --- | --- | --- |
| P6-0 | 2026-07-26T13:06 / 14:41 | fixed40 / token hash / chunk / schema 合同，`p6-0-contract.json`，SHA256 固化 |
| CL1 / CL2 | 07-26 18:xx – 07-27 | `cl1-screening/confirm(-rerun)`、`cl2-chunk-gate.json`；结论 `practical family = NONE`（详见 [Phase4 报告](PHASE4_RECOVERY_METHODS_REPORT.md) §4.7–§4.8） |
| P6-H | 07-26 19:52 起 | attempt1 chunk1024 OOM 失败 → `p6-h-attempt1-chunk1024-failed.json`；定位 2 个缺陷；修复后 `p6-h.json` valid（`run_id=p6-h-20260727T071106Z`） |
| P6-4 | 07-26 20:03 起 | attempt1 崩溃（`_delete_leaf` 断言）→ 定位 P0 根因 → `p6-4-exact-only.json`（隔离）→ P0 修复后 `p6-4.json`（`run_id=p6-4-20260727T104820Z`） |
| 诊断 C | 07-27 11:55 / 12:15 | v1（0.4s，误判「我方缺陷」）→ SUPERSEDED → v2（0.05s，「真实容量不可达」） |
| 诊断 D | — | S4/rho3.0 同法直接测量（`diagD-rho3-tight.jsonl`） |
| P6-F | 07-27 15:20 / 16:10 | test-only fault-injection canary v1 → v2 → v3，最终 valid，关闭 P0-1 / P0-3 |
| Exit review | 07-27 13:20 / 16:10 | 首轮双模型 FAIL → P0-1/P0-2/P0-3 修复 → 最终 `PASS WITH CAVEATS` |

### 4.2 双向 pressure（Gate 2，`satisfied`）

`PHASE6_EXIT_DISPOSITION.json` gate 2：

| 方向 | 字节数 |
| --- | ---: |
| exact requester → approximate victim | `47,475,326,976` |
| approximate requester → exact victim | `58,778,517,504` |

死亡瞬间细分（`diagnostic-C-store-state-at-oom.json`）：

| 场景 | exact→approx | approx→approx | exact→exact | approx→exact |
| --- | ---: | ---: | ---: | ---: |
| S0 / rho2 死亡时 | `2,202,009,600` | `411,041,792` | `8,592,424,960` | `1,767,800,832` |
| S4 / rho3 死亡时 | `1,746,927,616` | `866,123,776` | — | — |

### 4.3 P6-H：host roundtrip + same-context 输出 canary（Gate 6/8，`satisfied_with_scope`）

artifact：`impl:benchmark/approx_kv/results/phase6/p6-h.json`，`raw_sha256 = 842c3563ad20caedc55d46f596127cc2fcacf63b56ef927c7b878635ed3a12be`。

| 项 | 值 |
| --- | --- |
| body / header | `1024` / `64` |
| `chunked_prefill_size` | `4096`（首次尝试用 1024 时 OOM 失败） |
| requested / observed capacity | `3400` tokens / `3400` tokens（`389,939,200` bytes / `3400` pages） |
| `observed_logical_demand` | `0.30117647058823527` |
| restarts / formal_repeats | `1` / `2` |
| host backend | `allocator_cpu_copy`；`hicache_tier_exercised = false` |
| host 预算 | `SGLANG_APPROX_KV_HOST_BUDGET_BYTES = 8589934592`（8 GiB） |
| demand H2D | 每 round `1024` token；`mean_h2d_bytes = 117,440,512` |
| 输出 canary | 8-token 输出，两 round 均与 matched dense 完全一致 |
| `performance_claim` | `"disabled"` |

**Exit caveat（原文）**：*"One restart, two formal rounds, eight-token output canary. Does not establish bitwise KV/logit fidelity or HiCache qualification."*

Review B 额外强调：**P6-H 不是「KV 数据保真证明」**（`docs:TRACKING.md:2534-2535`）。

### 4.4 诊断 C / D：容量不可达的直接证据

#### 4.4.1 v1（`历史/已被替代`）

artifact：`diagnostic-C-v1-SUPERSEDED-coarse-sampling.json`。0.4s 轮询下的观测：死亡瞬间 approx store 仍持有 `384` token（`leases=0`，可驱逐却未驱逐），`832` token「无法归属」，`reservation_failures_total` 从未递增 → **误判为「我方缺陷」**（cross-store 回收路径被跳过，或存在第二处泄漏）。

#### 4.4.2 v2（`最终权威`）

artifact：`diagnostic-C-store-state-at-oom.json`（`schema_version=3`）。0.05s 重采样下：

- 死亡瞬间 `approx_kv_store_device_bytes = 0`、`records = 0`、`leases = 0`；
- 累计回收 `exact → approximate victim = 2,202,009,600` bytes，**证明回收路径工作**；
- `704` token free vs `1024` token 请求 → 单纯不够，而非泄漏。

**结论逆转：S0/LRU rho2.0 属于真实容量不可达，不是我方缺陷。** 并且 `832` token「未归属」本身是采样伪影——**`num_used_tokens` 已包含 approximate store 占用的 slot，两者不可相加**。

#### 4.4.3 诊断 D（S4/rho3.0）

`diagD-rho3-tight.jsonl`：capacity `7595`，死亡瞬间 store 同样清零，`exact → approximate` 回收 `1,746,927,616` bytes，判定为同一签名 → **真实容量不可达**（直接测量，非外推）。

### 4.5 P6-4 fixed40 capacity pilot（Gate 4/5）

artifact：`impl:benchmark/approx_kv/results/phase6/p6-4.json`（`run_id = p6-4-20260727T104820Z`）。

| cell | 顶层 status | requested / observed capacity |
| --- | --- | --- |
| S4 rho1.1 | `diagnostic-unavailable` | `20713` / `20713`（双向 pressure，40 次 recovery） |
| S4 rho1.5 | `diagnostic-unavailable` | `15190` / `15190` |
| S0 rho2.0 | `diagnostic-unavailable` | `11392` / 不可达（device 耗尽） |
| S4 rho2.0 | `diagnostic-unavailable` | `11392` / `11392` |
| S4 rho3.0 | `diagnostic-unavailable` | `7595` / 不可达（device 耗尽） |

**profile 级实况（Review B 强制要求按 profile 而非 cell 陈述）**：在三个可达 cell（S4 rho1.1/1.5/2.0）内，`exact_only` / `r0_like` / `r1_like_k32` / `r2_like` 四个 profile 全部 `reachable` 且 `valid`；**`r4_like`（约 5x multiplicity）在所有 cell 均 `diagnostic-unavailable`**。

这一点已由本报告直接读取 `p6-4.json` 的 `cells[].profiles[]` 字段核对：三个 S4 cell 均显示前四个 profile `reachability=reachable, valid=true`，`r4_like` 为 `diagnostic-unavailable`。

结论：

- **R1-like worst-case（k32）footprint 可达**（Gate 5，`satisfied`）；
- `r4_like` 不可达是**计划预先允许的 R4 例外**，不代表底座缺陷；
- **每个 cell 的顶层字段都是 `diagnostic-unavailable`（因为 R4-like 不可达），写成「三个 cell 可达」是过声明**（`docs:TRACKING.md:2525-2528`）。

### 4.6 P6-F：reservation-failure → dense fallback canary（Gate 9）

artifact：`impl:benchmark/approx_kv/results/phase6/p6-f-v3-fallback-canary.json`。gate 9 判定为 `verified_at_fault_injected_canary_strength`。

| 字段 | 值 |
| --- | ---: |
| `reservation_failures` | `1` |
| `reuse_dense_fallback_requests` | `1` |
| `cross_store_reservation_failed_tokens` | `1024` |
| `device_allocation_failed_tokens` | `0` |
| `dense_and_fallback_cached_tokens` | `64` |
| `output_completed_and_matches_dense` | `true` |
| `independent_injection_disabled_control` | `true` |
| `pre_flush_accounting_clean` | `true` |
| `post_reset_accounting_clean` | `true` |

**永久 caveat（原文）**：*"fault_injected=true; natural_pressure_reachability=false. Natural reservation-failure reachability under pressure remains unproven and must not be claimed."*

演化链：

1. v1（`p6-f-fallback-canary.json`）首次证明功能；
2. targeted review 发现 2 个 P1（fallback token 双重归因；clean accounting 只在 flush 后测）；
3. v2 关闭两个 P1（新增 `cross_store_reserved_device_bytes` / `approx_kv_provisional_tokens` gauge、独立无注入 control）；
4. v2 review 发现 1 个新 P1（control log 在 server stop 前 hash）；
5. v3 修复（dense/fallback cache path 锁定 64-token header）；
6. v3 两位 targeted reviewer 均 PASS，无新 P0/P1。

### 4.7 成功 / 失败 / 被跳过项汇总

| 项 | 状态 | 说明 |
| --- | --- | --- |
| cross-store substrate 实现 | 成功 | 1073 行，含完整回滚与 orphan 断言 |
| 双向 pressure | 成功 | gate 2 `satisfied` |
| P6-H host roundtrip + 输出 canary | 成功（scope 内） | 1 restart / 2 round / 8-token canary |
| P6-H chunk1024 首次尝试 | **失败** | OOM，记录于 `p6-h-attempt1-chunk1024-failed.json`，改用 chunk4096 |
| P6-4 attempt1 | **失败** | `_delete_leaf` 断言崩溃，导出 P0 根因 |
| P6-4 三个 S4 cell 的四个非 R4 profile | 成功 | `reachable` + `valid` |
| P6-4 `r4_like` | **不可达** | 全部 cell `diagnostic-unavailable`（计划内例外） |
| P6-4 S0/rho2 与 S4/rho3 | **不可达** | 真实容量不可达，有 0.05s 死亡瞬间证据 |
| P6-F fault-injected fallback | 成功 | 仅 fault-injected 强度 |
| **自然压力下的 reservation failure** | **未观察到 / 未证明** | Phase7 亦为 0 次 |
| exact–host / HiCache 统一 | **未实现** | gate 1 caveat |
| bitwise KV / logit fidelity | **未做** | 明确非目标 |
| scheduler / prefetch / practical winner 评测 | **未做** | 划归 Phase7 |

---

## 5. 发现并修复的问题（逐条：症状 → 根因 → 修复 → 验证 → 影响）

### Bug 1 — `resolve_reuse_spans` 吞掉 `prefix_gap`（P6-H 第一批，已修复）

- **症状**：把整段 dense prefill 误记成「exact 命中且 0 fallback」。
- **根因**（`docs:TRACKING.md:1813-1821`）：当 exact prefix 短于第一个 segment 的 `target_start` 时，直接 `record_request("reuse","exact")` 返回，既不记 `prefix_gap` fallback 也不计 token。
- **修复**：区分「完全覆盖」与「存在 gap 无法挂载」，后者记 `prefix_gap` dense fallback。
- **验证**：新增 2 个回归测试。
- **影响**：修复前的 P6-H fallback 计数不可信。

### Bug 2 — Paired dense 驱逐 recovery header（P6-H 第一批，已修复）

- **症状**：reuse 永远无法挂载，demand H2D 不可能触发。
- **根因**（`docs:TRACKING.md:1821-1828`）：tight capacity 下 paired dense 请求驱逐了 recovery namespace 的 header。
- **修复**（提交 `5e47904ecba6b8d7b5d03693277360a1cecfa679`）：reuse 前重新 seed header，并断言 reuse 确实挂载了 registered body。
- **验证**：P6-H 最终 run 中 demand H2D 每 round `1024` token。
- **影响**：修复前无法证明 host roundtrip 真实发生。

### Bug 3 —【P0】请求自身 exact prefix 未加锁，导致自我驱逐 / 自我覆写（核心缺陷）

- **症状（FINDING-P6H-A）**：机械证据全部通过（host export `1024` token、`cross_store_demoted_bytes_total = 117,440,512`、demand H2D `1024` token、`leases=2`、0 reservation failure、0 orphan、reset 通过），**唯独 recovered 输出与 matched dense 不一致**。
- **根因**（`docs:TRACKING.md:1941-1988`，`runtime.py:83-119`）：`Req.init_next_round_input` 调用 `restore_request_prefix` 发生在 `schedule_policy.add_one_req` 获取 prefix 锁**之前**，此时 `req.last_node.lock_ref == 0`；而 `RadixCache.cross_store_resources()` 的过滤条件恰为 `node.lock_ref == 0`，使请求自身的 prefix 节点成为合法 victim。压力下 `allocate_recovery_slots` 驱逐该节点 → slot 回 free list → `allocate_backend()` 把同一批 slot 作为 recovery 目的地 → **请求自身即将 attend 的 KV 被覆写**。
- **隔离证据**：`p6-h-pressure-corruption-isolation.json` 的 5 组实验确认触发条件是「reuse 执行时存在真实 device 压力（竞争性近似 registration + 紧容量）」，与 residency tier、是否 demotion 无关。对照 `control-exact-cache-guardrail.json`：纯 exact radix 命中 16/16 逐 token 一致，排除 prefill 数值不确定性。
- **修复**（提交 `af81934e4`）：新增 `protect_request_prefix` 上下文管理器，在整个 recovery 窗口持有标准 prefix 锁（`inc_lock_ref` 一路 walk 到 root，保护整条 matched chain 并移出 `evictable_leaves`）；在 `schedule_batch.py` 的唯一调用点包裹，覆盖 EPIC 与普通路径；加固 exact victim guard，stale victim 改抛 `KeyError`（而非 `_delete_leaf` 断言杀死 scheduler）。
- **验证**：GPU 上先前必然损坏的配置（`max_total_tokens=3400` + 竞争性 registration + 真实 demotion + H2D）修复后逐 token 与 dense 完全一致；新增 5 个回归（`test_locked_prefix_is_never_offered_as_a_victim`、`test_stale_victim_raises_keyerror_instead_of_asserting` 等）；targeted 套件 `204 passed, 5 skipped`（唯一失败 `test_memory_allocated` 经 `git stash` 确认为改动前既有失败）。
- **影响**：
  1. 解除了 CL1「因果归因无效」的阻塞——但**修复后 CL1 guardrail 失败计数完全不变**，证明该缺陷**不是** CL1 输出偏离的成因；
  2. 附带修正：P6-H reseed 断言原本对满命中 N-token prompt 报告 `N-1` cached 判为异常，现更正为**预期**（最后一个 token 必须真实 forward）。

### Bug 4 — `_delete_leaf` 断言 / stale victim（与 Bug 3 同源）

- **症状**：P6-4 attempt1 scheduler 崩溃。
- **根因**（`docs:TRACKING.md:1865-1904`，`cross_store/allocator.py`）：allocator 的驱逐循环在一次迭代内可选中整个 eviction closure 并按快照顺序逐个 `action()`；快照只在「上一轮驱逐过 exact 资源」时于下一次循环开头刷新。因此同轮内先执行的驱逐可能已把后续 resource 对应的 radix 节点从父节点摘除，再对该 stale 节点调用 `evict` 就触发 `_delete_leaf` 断言。`excluded_roots` / `inactive_resources` 只记 identity，不检测树内可达性。
- **修复**：随 Bug 3 根因修复（锁保护 + stale victim 改 `KeyError`）一并解决。
- **验证**：P6-4 修复后完成。
- **影响**：说明「in-flight 请求引用的节点被驱逐」这一缺陷类有两个表现形式。

### Bug 5 — Review P1-1：SWA / Unified 窗口 release metadata 丢失（已修复，`db2d18ff0`）

- **症状**：在 SWA/Unified cache 下可能递减其它请求仍持有的祖先锁。
- **根因**（`docs:TRACKING.md:2125-2131`）：`protect_request_prefix` guard 丢弃了 `inc_lock_ref` 返回的 SWA 窗口与 skipped-node 元数据，直接 `dec_lock_ref(node)`。
- **修复**：回传 `result.to_dec_params()`（`runtime.py:100-119` 的 `params()` 分支）；标准 RadixCache 忽略该参数。
- **验证**：新增回归。
- **影响**：修复本身引入的次生缺陷，说明「新增保护路径必须完整传递原 API 的返回元数据」。

### Bug 6 — Review P1-2：stale victim 导致整个 allocation 放弃，detached 节点仍留在 `evictable_leaves`（已修复，`3379e6699`）

- **症状**：失败后 stale 节点仍被反复广告；若它排在有效 victim 之前，分配直接 `committed=False`。
- **根因**（`docs:TRACKING.md:2131-2139,2179-2189`）：加固后的 attachment 检查抛 `KeyError` 后，detached 节点仍留在 `evictable_leaves`；`CrossStoreAllocator.allocate()` 遇 `KeyError` 立即放弃而非刷新快照改选其它 victim。
- **修复**：跳过该 candidate、刷新快照继续选择，同时把 detached 节点移出 `evictable_leaves`（`allocator.py` 中 `except KeyError: stale_victims += 1; refresh_resources = True`）。
- **验证**：实测确认修复后不再重复广告。
- **影响**：提高了紧容量下的分配成功率。

### Bug 7 — Review P1-3：Provisional slot 在 admission 之前挂载，拒绝/abort 时不释放（**真实资源泄漏**，已修复，`40f09c1fe`）

- **症状**：容量单调枯竭，最终 `available_size = 0` 且 `evictable_size = 0`，只剩 64 个 locked token。
- **根因**（`docs:TRACKING.md:2139-2166`，`runtime.py:37-78`）：recovery 在 scheduler admission **之前**分配并挂载 device slot；若 `add_one_req()` 拒绝该请求，清理只释放 Mamba 状态，recovered suffix 不释放；下一次 rematch 覆写 `prefix_indices`，丢失 slot 的唯一引用。`approx_kv_restored_len`（`runtime.py:571`、`epic_runtime.py:678`）只被写入、全仓库无消费者，佐证缺少清理路径。
- **修复**：改为 provisional 所有权模型——`prepare_for_extend` 前属 provisional（`release_provisional_recovery_slots` / `commit_provisional_recovery_slots`），在 `init_next_round_input` 重新 match 前与请求 teardown 时回收；`prepare_for_extend` 拿到所有权后清除标记，杜绝 double free。
- **验证**：全目录回归对照，改动前后均为 `935 failed`（既有基线失败，与本次无关），本次净增 3 个 pass。
- **影响 / 重要更正**（`docs:TRACKING.md:2199-2202`）：先前把 P6-4 S0/rho2 OOM 归因于「修复后峰值 device 需求真实上升」被**降级为次要**；曾一度认为主因是本泄漏。但**两次修复后该 cell 仍确定性 OOM**，最终归类为**真实容量不可达而非实现缺陷**。旁证：CL1 reset invariant 48/48 全过（成功路径不泄漏，泄漏只在 admission 拒绝时发生）。

### Bug 8 — Sol 审计 P0-1：provisional 清理仍不完整（已修复）

- **症状**：被拒请求若在重试前 abort，slot 永久泄漏；不 abort 也在当前 batch 内错误占用容量。
- **根因**（`docs:TRACKING.md:2503-2508`）：`scheduler.py:3024-3050`（admission 拒绝）与 `scheduler.py:4069-4108`（waiting-request abort）两处不立即释放 provisional slot。
- **修复**：两处补上 `release_provisional_recovery_slots(self.tree_cache, req)`（`scheduler.py:3045`、`:4090`），并把 marker 清除移到 `allocator.free` **成功之后**（否则 free 失败会丢失唯一引用）——见 `runtime.py:59-65` 的注释 *"Clear only after the free succeeded"*。
- **验证**：新增 3 个回归。
- **影响**：关闭 Exit review 的一个 P0。

### Bug 9 — Sol 审计 P0-2：provenance 未真正关闭（已修复）

- **症状**：首版 `RESULT_MANIFEST.json` 的 32 项中只有 29 项可验证（2 项 `pending_this_commit`，1 项 blob hash 不匹配）。
- **根因**（`docs:TRACKING.md:2509-2513`）：manifest 在提交**之前**生成。
- **修复**：新增 `impl:benchmark/approx_kv/build_result_manifest.py`，既生成也验证（`--check`），缺失 / pending / hash 不符即失败；规定必须在同一 commit 内重新生成并复查。
- **验证**：最终 `48/48` 通过。
- **影响**：确立了本项目此后所有阶段的 provenance 模式。

### Bug 10 — Fallback taxonomy 误标（Review A 发现，已撤回并更正）

- **症状**：Exit 证据里声称有「12 次 GPU dense fallback」。
- **根因**（`docs:TRACKING.md:2483-2489`，`p6-4-outcome-correction.json`）：这 12 次全部来自 `exact_only` profile——该 profile 没有 approximate metadata，而 runner（`run_p6_4_capacity_pilot.py:413-419`）仅凭 `cached_tokens < expected` 就把普通 exact-cache miss 标为 `dense_fallback`；`r4_like` 的 `4096` fallback token 实为 registration 容量失败，replay outcome 实际是 `approximate_gpu_recovery`。
- **修复**：两条结论均撤回，disposition 改为 `governance_exemption_unverified`；runner 改为输出 `exact_cache_miss`；文件 `p6-4-fallback-injection.json` 因「没有任何 fault injection」而误导，更名为 `p6-4-reduced-profiles-4x-bytes-per-token-probe.json`。
- **验证**：`p6-4-outcome-correction.json` 记录更正。
- **影响**：这是 Exit 首轮 FAIL 的直接原因之一（P0-1）。

### Bug 11（治理/证据类）— `.gitignore:179 *.jsonl` 静默排除原始遥测

- **症状**：被引用为证据的原始 JSONL 未进入版本控制。
- **根因**（`docs:TRACKING.md:2498-2502`）：`.gitignore` 通配规则。
- **修复**：`git add -f` 纳入版本，并新增 `RESULT_MANIFEST.json`。
- **验证**：`--check` 48/48。
- **影响**：与 Bug 9 共同关闭 provenance P0。

### Bug 12（方法论）— 采样间隔过粗导致「死亡瞬间」结论完全错误

见 §4.4.1–§4.4.2。**这是本阶段代价最高的方法论教训**：一个 `0.4s` 的轮询间隔产生了一个**自信但完全错误**的「这是我方缺陷」结论；改为 `0.05s` 后结论完全反转。

---

## 6. Lessons Learned

### 6.1 机制层

1. **恢复路径必须维持 exact 路径已有的不变量**，尤其是 prefix 锁。在 `add_one_req` 之前分配资源时，必须自行取得等价 prefix protection；否则会落入「请求自身不受保护」的窗口。
2. **表示多重性（representation multiplicity）是容量的一等约束。** `r4_like` 的 5x footprint 在所有 Phase6 cell 都不可达，而 `r2_like` 的 2x 只在三个已完成的 S4 cell 中验证为可达；这是当前硬件与合同下的可行性边界。
3. **allocator 的 victim 快照必须与树结构变更同步。** 同一轮驱逐内的树结构变化会让后续 victim 变成 stale。

### 6.2 系统层

4. **provisional 所有权模型是必需的**：在 admission 之前分配的资源必须有明确的「谁负责释放」协议，并覆盖 reject、abort、rematch、teardown 四条路径。
5. **释放标记必须在 `free` 成功之后清除**，否则 free 失败会丢失唯一引用。
6. **分配失败必须优雅降级。** `alloc_token_slots` 在 cross-store 无法腾出空间时抛 `RuntimeError` 会杀死 scheduler 进程，应改为可记录的失败（`docs:IMPLEMENTATION_PLAN_LATEST.md` §15.2 第 12 条）。
7. **回滚只能保证可逆动作。** gate 3 的 caveat 明确：*"Reversible actions are rolled back. Irreversible victims are not restored; they are correctly accounted and may require reset."*

### 6.3 测量层

8. **遥测采样间隔必须短于 workload 的分配动态。** 临近死亡的 `1.3s` 内 `num_used_tokens` 从 `5376` 涨到 `10688`，`0.4s` 采样点必然早于致命请求。任何「死亡瞬间状态」类证据都必须声明采样间隔并论证其足够细。
9. **不要把 `num_used_tokens` 与 store gauge 相加。** 前者已包含 approximate store 占用的 slot，相加会凭空造出并不存在的「未归属 token」。
10. **「容量不可达」必须附死亡瞬间的 store gauge 快照**，证明当时没有可回收资源残留，而不是仅凭 server 崩溃就下结论。
11. **cell 级状态与 profile 级状态必须分别陈述。** 写成「三个 cell 可达」是过声明。
12. **`dense_fallback` 必须与 `ordinary_exact_cache_miss` 严格区分。**
13. **artifact 命名即声称。** `p6-4-fallback-injection.json` 里没有任何 fault injection。

### 6.4 统计 / 证据强度层

14. **fault-injected 与 natural-pressure 是两种不同的证据强度**，必须在 artifact 中显式标注，并原样携带到下游阶段的任何依赖性 claim 中。
15. **8-token 输出 canary 不是 KV 保真证明。** 它只是「不发生粗暴损坏」的 guardrail。
16. **单 restart / 双 formal round 的 canary 不能升级为性能或质量结论。**

### 6.5 治理 / provenance 层

17. **manifest 必须与 artifact 在同一 commit 内生成并 `--check`**，否则条目会腐化为 pending 或 stale-blob。
18. **artifact 的 `result_git_sha` 天然为 `null`**（runner 无法知道将来容纳自己输出的 commit），因此必须另行维护 `RESULT_MANIFEST.json` 提供 file→commit 映射，并且**不得据 artifact 字段声称 provenance 完整**。
19. **被撤回的结论必须同时从叙述性文档中移除**——Sol 复核发现 `PROJECT.md` 仍留有已撤回的旧表格与结论矛盾（P0-3）。
20. **阶段通过不自动授权下一阶段。**

---

## 7. 最终结论

### 7.1 Exit Gate 最终判定（`PHASE6_EXIT_DISPOSITION.json`）

| # | Gate | 判定 | 关键 caveat |
| ---: | --- | --- | --- |
| 1 | exact / approximate / host 对象安全竞争 | `satisfied_with_scope` | exact 与 device-approximate 共享 device 预算；host 用独立 host limit；**exact–host / HiCache 统一未实现** |
| 2 | 双向 pressure | `satisfied` | `47,475,326,976` / `58,778,517,504` bytes |
| 3 | 分配失败回滚 | `satisfied_with_scope` | 可逆动作回滚；**不可逆 victim 不恢复**，但被正确记账，可能需要 reset |
| 4 | fixed40 四 rho 结果或明确不可达结论 | `satisfied` | 每个顶层 cell 因 R4-like 不可达而为 `diagnostic-unavailable`；四个非 R4 profile 在三个 S4 cell 可达；S0/rho2 与 S4/rho3 有独立测得的容量极限死亡状态 |
| 5 | R1-like 最坏 footprint | `satisfied` | `r1_like_k32` 在 S4 rho1.1/1.5/2.0 的两个 formal round 均 reachable + valid |
| 6 | 通用 host roundtrip canary | `satisfied_with_scope` | 1 restart / 2 formal round / 8-token canary；**不建立 bitwise KV 或 logit 保真，也不构成 HiCache qualification** |
| 7 | 无泄漏、无孤儿 | `satisfied_for_completed_runs` | P6-H clean reset、P6-4 完成轮次、P6-F v3 gauge |
| 8 | same-context 压力 reuse 输出 canary | `satisfied_with_scope` | 仅输出 token canary，**不是通用 KV 保真 claim** |
| 9 | reservation-failure 关联的 dense fallback | `verified_at_fault_injected_canary_strength` | **`fault_injected=true; natural_pressure_reachability=false`；自然可达性未证明且不得声称** |
| 10 | raw / commit / environment / log / test provenance | `satisfied_for_formal_exit_package` | `RESULT_MANIFEST --check` `48/48`；历史失败尝试与可选诊断日志可能仍未版本化 |

**最终判定**：`technical_exit = "pass_with_caveats"`，`phase7_authorized = false`。两位 reviewer 均关闭 P0-1 与 P0-3，无新增 P0/P1。

### 7.2 Formal Exit Review 时间线

| 时间 | 事件 |
| --- | --- |
| 07-27 13:20 | 首轮双模型 review：**两方均判 FAIL**。10 项 gate 中 1/2/5/6 满足，3/4 部分满足，7 对已完成 run 满足，8 仅字面 canary 满足，**9（dense fallback 可达）与 10（provenance）不满足** |
| — | 3 个 P0：P0-1 fallback 未证明（12 次「GPU dense fallback」证据错误，已撤回）；P0-2 S4/rho3 未直接证明容量不可达（仅由 S0/rho2 外推）；P0-3 provenance 不完整 |
| — | Review B 另提 6 项措辞弱化：`practical=NONE` 非 factorial；chunk 只测 body768/1024 且双变量；S4 分母「消失的是相对 S1–S3 的独特性」而非绝对优势；P6-H 非 KV 保真证明；P6-4 应按 profile 级陈述；历史 v1 需机器可读的 `status=superseded` 标记 |
| 07-27 14:00 | Sol Max 复核：指出「九项直接证据 + 一项豁免」当时不可辩护；发现 P0-1（provisional 清理真实代码缺陷，见 Bug 8）、P0-2（provenance 实际 29/32 未闭合）、P0-3（`PROJECT.md` 仍留有已撤回的旧表格）；真实状态为 `technical_exit=FAIL` + `governance_disposition=已对一项未验证条件接受豁免` |
| 07-27 16:10 | P6-F v3 关闭最后 blocker；两位 reviewer 最终均判 `PASS WITH CAVEATS` |

关闭记录（`review_disposition.closed`）：

- `integrated_fallback`：P6-F v3；两位 targeted reviewer 在 fault-injected-canary 强度上 PASS；
- `s4_rho3`：直接的 0.05 秒死亡状态遥测；
- `provenance`：版本化的 raw 遥测与主日志 + 自动化 result-manifest 校验。

### 7.3 当前仍成立的结论

| 结论 | 作用域 |
| --- | --- |
| exact 与 device-approximate 对象可在同一 device 预算下安全竞争，双向 pressure 均被实测 | 本底座实现 |
| 分配失败可回滚可逆动作，并正确记账不可逆 victim | 同上 |
| `r1_like_k32`（R1 最坏情况 footprint）在三个 S4 cell 可达 | fixed40，chunk1024 |
| `r4_like`（5x multiplicity）在本硬件 fixed40 下全部不可达 | 同上 |
| S0/rho2 与 S4/rho3 是**真实容量不可达**，不是实现缺陷 | 有 0.05s 死亡瞬间证据 |
| 同 header（same-context）压力下的 approximate reuse 输出与 matched dense 一致 | 1 restart / 2 round / 8-token canary |
| reservation 失败可降级为 dense fallback 且输出仍与 dense 一致 | **仅 fault-injected 强度** |
| 底座在完成的 run 上无泄漏、无孤儿 | `satisfied_for_completed_runs` |

### 7.4 被收窄或推翻的结论

| 原结论 | 处置 |
| --- | --- |
| 「观察到 12 次 GPU dense fallback」 | **推翻并撤回**：全部是 `exact_only` profile 的普通 exact-cache miss |
| 「`r4_like` 的 4096 fallback token 是 dense fallback」 | **推翻**：实为 registration 容量失败，replay outcome 是 `approximate_gpu_recovery` |
| 「S0/rho2 OOM 是我方回收路径缺陷」（诊断 C v1） | **推翻**：0.05s 采样后反转为真实容量不可达 |
| 「S0/rho2 OOM 主因是 provisional slot 泄漏」 | **收窄**：两次修复后仍确定性 OOM，归类为真实容量不可达 |
| 「三个 cell 可达」 | **收窄**：cell 顶层全部 `diagnostic-unavailable`；可达的是其中的非 R4 profile |
| 「P6-H 证明了 KV 数据保真」 | **收窄**：只是 8-token 输出 canary |
| 「integrated fallback 已验证」 | **收窄**：只在 fault-injected 强度验证 |

### 7.5 明确**不能**声称的内容

1. **不能**声称自然压力下 reservation-failure fallback 可达——Phase6 只有 fault-injected 证据，Phase7 观察到 **0 次**自然 reservation failure。
2. **不能**把 P6-H 的 8-token canary 说成 bitwise KV 或 logit fidelity。
3. **不能**把 P6-H 说成 HiCache qualification（`hicache_tier_exercised = false`，host backend 是 `allocator_cpu_copy`）。
4. **不能**声称「三个 cell 可达」——只能说「三个 S4 cell 内的四个非 R4 profile 可达」。
5. **不能**把 `r2_like` / `r4_like` 的任何结果归因于 CacheBlend 或 KVCOMM 机制。
6. **不能**发布任何 Phase6 性能数字（`performance_claim = "disabled"`，`performance_ranking_enabled = false`）。
7. **不能**声称 exact–host / HiCache 统一已实现。
8. **不能**把 P6-4 的 rho1.1/1.5/3 可行性结论迁移出 chunk1024。
9. **不能**据 artifact 的 `result_git_sha` 字段声称 provenance 完整（它天然为 `null`）。
10. **不能**把 Phase6 的通过当作 Phase7 的授权。

---

## 8. 该结论能预测什么（可证伪预测与待验证假设）

| 编号 | 预测 | 证伪条件 |
| --- | --- | --- |
| P6-1 | 在 chunk4096 下重跑 wave-0（S0/S4 × rho2）时，四个非 R4 profile 仍可达，`r4_like` 在 S4 下仍不可达 | 若 `r4_like` 在 chunk4096 下可达，则「5x multiplicity 是主约束」被证伪 |
| P6-2 | 现有 workload 下自然 `cross_store_reservation_failed` 可能继续稀少，容量不足更常表现为 registration 失败或 `unsupported` | 非零自然 reservation failure 只证明 reachability；要关闭 caveat，还必须证明关联 dense fallback 完成、输出匹配且 pre/post-reset accounting 干净 |
| P6-3 | admission 前分配资源的新路径若缺少**等价的** prefix protection 与 provisional ownership，不变量破坏风险显著增加 | 使用不同 helper 但提供等价锁保护、所有权转移和 reject/abort/rematch/teardown 清理即可证伪「必须复用同名实现」 |
| P6-4 | 若在相同代码/config 下把 device limit 提高到超过已测 live footprint 与请求峰值需求，容量死亡 cell 应转为可达 | 若预算已明确超过需求仍不可达，才应重新怀疑实现缺陷；GPU 型号或算力本身不是充分条件 |
| P6-5 | 在 W（workflow）矩阵中引入 approximate 臂后，多数请求将走 dense fallback，terminal reason 以 `unsupported` 为主 | 若 terminal reason 以 `cross_store_reservation_failed` 或 `device_allocation_failed` 为主，则本预测被证伪 |

> P6-1、P6-2、P6-5 在 [Phase7 报告](PHASE7_INTEGRATED_EVALUATION_REPORT.md) §4 中均观察到与预测方向一致的结果（wave-0 S4 的 `r4_like` 不可达、自然 reservation failure 计数为 0、correction run 40/40 为 `unsupported <- store_miss`）。这些是**与预测一致的观察**，不构成对预测的证明。

### 8.1 Phase6 给下一阶段的四条边界（`next_boundary`）

1. 未获用户明确授权不得进入 Phase 7；
2. 必须创建并独立 review 一份 result-bound 的 V5 计划；
3. 必须预注册 Phase 7 primary manifest；
4. **必须把 fault-injected-only / natural-pressure-not-proven 的 fallback caveat 带入任何依赖 reservation failure 的 Phase 7 claim。**

### 8.2 Phase6 明确标注为「Closeout 输入而非 Phase6 claim」的三条

`PHASE6_EXIT_DISPOSITION.json` 的 `closeout_inputs_not_phase6_claims`：

- **CL1**：在被测实现/配置下没有 candidate 通过冻结的 exact-output promotion 规则。eviction-dependent prefix-overwrite 缺陷已排除，但 context 差异未被证明是因果，header-dependent 缺陷仍可能存在。
- **CL2**：在 body1024 上，耦合的 chunk / max-prefill 配置造成了 dense 的 chunk 边界惩罚。body2048 未测量，不得泛化。
- **CL3**：S4 相对 S1–S3 的描述性数值分离只出现在 workflow-only；all-reusable 下各策略相对 S0 的描述性收益相近，且独立 restart 不足以做策略排序。

---

## 9. 局限、未完成项与 artifact / provenance 索引

### 9.1 局限

1. 单 GPU（SM75，8GB）、单模型、合成 fixed40 workload。
2. 全部 cell 只有 1 次 restart、2 个 formal round。
3. host 层只是 `allocator_cpu_copy`，**未走 HiCache tier**。
4. 输出正确性只用 8-token canary 衡量。
5. reservation-failure fallback 只在 fault-injected 强度验证。
6. P6-0 contract 的 `chunked_prefill_size = 1024` 被标为 `provisional_worst_case`；P6-H 因 OOM 改用 `4096`——**这两个数字不同，引用时必须分别注明**。
7. 全部结论禁止性能声称。

### 9.2 未完成项

| 未完成项 | 影响 |
| --- | --- |
| 自然压力下的 reservation-failure 可达性 | 永久 caveat，需 Phase8+ 重新取证 |
| exact–host / HiCache 统一 | gate 1 scope 限制 |
| bitwise KV / logit fidelity | 输出 canary 无法替代 |
| `r4_like` 在任意配置下可达 | 需更大显存或降低 multiplicity |
| P6-4 rho1.1/1.5/3 在 chunk4096 下的重新验证 | 现有结论限定 chunk1024 |
| 多 restart / 多 formal round 的统计强度 | 现为 1 restart |
| 历史失败尝试与可选诊断日志的版本化 | `known_gaps` 明确记录 |

### 9.3 Artifact / provenance 索引

**权威索引**：`impl:benchmark/approx_kv/results/phase6/RESULT_MANIFEST.json`（`schema_version=2`，`files` 共 **48** 条，`--check` 通过 `48/48`）。

`purpose` 原文：*"Supply the file-to-commit mapping that the individual result artifacts cannot: a runner never knows the commit that will contain its own output, so every artifact carries `result_git_sha=null` and `result_commit_status=pending_result_commit`."*

`authority` 原文：*"This manifest, not the `result_git_sha` field inside an artifact, is the authoritative mapping."*

`known_gaps` 原文：*"The primary P6-H, P6-4 and P6-F server logs are versioned and content-addressed. Some historical failed attempts and optional diagnostic logs still exist only at absolute host paths and are not part of the Phase 6 Exit evidence package."*

主要 artifact（均位于 `impl:benchmark/approx_kv/results/phase6/`）：

| 文件 | 类别 | 状态 |
| --- | --- | --- |
| `PHASE6_EXIT_DISPOSITION.json` | Exit 判定 | `最终权威` |
| `RESULT_MANIFEST.json` | provenance | `最终权威`（48/48） |
| `p6-0-contract.json` | 合同冻结 | `最终权威`（`contract_sha256=a498daa3…`） |
| `p6-4.json` | capacity pilot | `最终权威` |
| `p6-4-exact-only.json` | 隔离运行 | `历史/已被替代` |
| `p6-4-outcome-correction.json` | 更正 | `最终权威` |
| `p6-4-reduced-profiles.json` / `p6-4-reduced-profiles-4x-bytes-per-token-probe.json` | 探针 | `diagnostic/proxy`（后者为更名后文件） |
| `p6-4-rho2-isolate.json` / `p6-4-rho3-diagnostic.json` / `p6-4-fallback-probe-rho2p5.json` | 诊断 | `diagnostic/proxy` |
| `p6-4-PREFIX-CONTROL-prefix-commit.json` | 对照 | `diagnostic/proxy` |
| `p6-4-*-server.log`（5 份） | 服务端日志 | 版本化 |
| `p6-h.json` | host roundtrip + canary | `最终权威`（`raw_sha256=842c3563…`） |
| `p6-h-attempt1-chunk1024-failed.json` | 失败记录 | `历史/已被替代` |
| `p6-h-pressure-corruption-isolation.json` | 隔离矩阵（5 组） | `最终权威`（诊断） |
| `p6-h-final-server.log` | 服务端日志 | 版本化 |
| `control-exact-cache-guardrail.json` | 纯 exact 对照（16/16 一致） | `最终权威` |
| `p6-f-fallback-canary.json` / `p6-f-v2-*.json` / `p6-f-v3-fallback-canary.json` | fallback canary v1/v2/v3 | v3 为 `最终权威` |
| `p6-f-*-server.log`（4 份） | 服务端日志 | 版本化 |
| `diagnostic-C-store-state-at-oom.json`（schema v3） | 死亡瞬间状态 | `最终权威` |
| `diagnostic-C-v1-SUPERSEDED-coarse-sampling.json` | 0.4s 采样 | **`历史/已被替代`** |
| `diagC2-labeled.jsonl` / `diagC3-tight.jsonl` / `diagC-store-gauges.jsonl` / `diagD-rho3-tight.jsonl` | 原始遥测 | 版本化（`git add -f`） |
| `cl1-*.json` / `cl2-chunk-gate.json` / `cl3-phase5-recalculation.json` | Closeout | 见 [Phase4](PHASE4_RECOVERY_METHODS_REPORT.md) / [Phase5](PHASE5_WORKFLOW_SCHEDULING_REPORT.md) 报告 |
| `context-vs-pressure-2x2.json` | 2×2（非真 factorial） | `diagnostic/proxy` |
| `phase6-exit-fallback-disposition.json` | fallback disposition | `最终权威` |

关键实现提交：

| 提交 | 内容 |
| --- | --- |
| `af81934e4` | `protect_request_prefix`（P0 修复） |
| `db2d18ff0` | SWA/Unified release metadata 回传 |
| `3379e6699` | stale victim 刷新重试 + 移出 `evictable_leaves` |
| `40f09c1fe` | provisional recovery slot 所有权模型 |
| `5e47904ecba6b8d7b5d03693277360a1cecfa679` | P6-H reseed header + reuse 断言 |
| `e59bb7a9c` | test-only fault injection |
| `11bc9b3e4` | `test_reservation_failure_degrades_to_dense_fallback` mutation 验证 |
| `fd63aa7f67032e1649e15443a144436f809af0da` | Exit disposition 前的 implementation HEAD |

验证命令（`RESULT_MANIFEST.json.verification_commands`）：

```text
targeted_regression : python3 -m pytest -q test/registered/unit/mem_cache/test_approx_kv_core.py \
    test/registered/unit/mem_cache/test_approx_kv_runtime.py \
    test/registered/unit/mem_cache/test_approx_kv_integration_source.py \
    test/registered/unit/mem_cache/test_approx_kv_hicache_backend.py \
    test/registered/unit/mem_cache/test_approx_kv_cuda.py \
    test/registered/unit/mem_cache/test_cross_store_substrate.py \
    test/registered/unit/mem_cache/test_epic_leadingk.py \
    test/registered/unit/bench/
manifest_self_check : python3 -m benchmark.approx_kv.build_result_manifest --check
known_baseline      : 全树 mem_cache + bench 有 935 个既有失败（有无本分支改动均相同）
```

---

## 10. 与其它阶段报告的关系

| 关系 | 说明 |
| --- | --- |
| ← [Phase4](PHASE4_RECOVERY_METHODS_REPORT.md) | Phase6 期间执行的 CL1/CL2 直接修正 Phase4 结论；P0 修复解除了 CL1 的因果归因阻塞 |
| ← [Phase5](PHASE5_WORKFLOW_SCHEDULING_REPORT.md) | Phase6 期间补齐 CL3；Phase5 的 S4 需在 Phase6 底座上接入真实 cross-store metadata |
| → [Phase7](PHASE7_INTEGRATED_EVALUATION_REPORT.md) | Phase6 的 `PASS WITH CAVEATS` 是 Phase7 的证据输入之一，但**不构成授权**；P6-F 的 fault-injected caveat 被原样带入 Phase7 |
| → [跨阶段总报告](PHASE4_TO_PHASE7_SUMMARY.md) | 汇总「exact/approx 共预算」「自然 vs fault-injected fallback」「provenance 治理」三条主线 |
