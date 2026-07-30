# 实施计划 Phase7.5：C40 Clean-Room Reproduction & Extended Evaluation

> **Version**: `V1-r13`
> **Status**: `Reviewed Candidate / PENDING USER AUTHORIZATION`
> **Document date**: 2026-07-30
> **Phase name（冻结）**: `Phase7.5 C40 Clean-Room Reproduction & Extended Evaluation`
> **Plan ID（冻结）**: `P7.5-C40-V1`

```text
plan_drafting_authorized        = true
plan_review_authorized          = true
branch_creation_authorized      = false
implementation_authorized       = false
docker_test_execution_authorized= false
gpu_execution_authorized        = false
quality_campaign_authorized     = false
budget_authorized               = false
```

**本文件的全部 Gate、Work Package、矩阵、样本量、预算、工时与命令均为
`estimate` / `proposed`，除"编制与审阅本计划"之外没有任何一项已获授权。**
下一会话在获得用户明确授权之前，**不得**创建 branch、不得写实现代码、不得
运行任何测试（含 CPU 测试）、不得启动 Docker 实验、不得占用 GPU。

---

## 目录

- [0. 文档职责、状态与授权边界](#0-文档职责状态与授权边界)
- [1. Executive Summary](#1-executive-summary)
- [2. Authority 与输入证据](#2-authority-与输入证据)
- [3. 方法定义与新颖性边界](#3-方法定义与新颖性边界)
- [4. Clean-Room 合规合同](#4-clean-room-合规合同)
- [5. Branch / Base / 结果目录冻结](#5-branch--base--结果目录冻结)
- [6. 能力分层：Parity Core / Mandatory Extensions / Conditional Lanes](#6-能力分层parity-core--mandatory-extensions--conditional-lanes)
- [7. 架构与组件设计](#7-架构与组件设计)
- [8. Middle-Span 状态机与请求生命周期（含 §8.6 chunk-splitting 执行协议）](#8-middle-span-状态机与请求生命周期含-86-chunk-splitting-执行协议)
- [9. 结构化 Provenance Schema 与路径模型](#9-结构化-provenance-schema-与路径模型)
- [10. Identity / Fingerprint / Approx Depth](#10-identity--fingerprint--approx-depth)
- [11. 配置、环境变量与 Feature Gate](#11-配置环境变量与-feature-gate)
- [12. Terminal Reason 与 Outcome Taxonomy](#12-terminal-reason-与-outcome-taxonomy)
- [13. Telemetry 与 Metrics](#13-telemetry-与-metrics)
- [14. Versioned Manifest Schema](#14-versioned-manifest-schema)
- [15. Candidate / Axis Taxonomy 与 Staged Gates](#15-candidate--axis-taxonomy-与-staged-gates)
- [16. Work Packages WP0a–WP12](#16-work-packages-wp0awp12)
- [17. 测试与实验设计](#17-测试与实验设计)
- [18. Workloads](#18-workloads)
- [19. 统计合同](#19-统计合同)
- [20. Gates P7.5-G0a – P7.5-G11](#20-gates-p75-g0a--p75-g11)
- [21. Stop Rules](#21-stop-rules)
- [22. 预算](#22-预算)
- [23. Governance](#23-governance)
- [24. 交付物与 Artifact 索引](#24-交付物与-artifact-索引)
- [25. 下一会话可直接执行的顺序](#25-下一会话可直接执行的顺序)
- [26. 允许与禁止的表述](#26-允许与禁止的表述)
- [27. 冻结默认与待用户决策项](#27-冻结默认与待用户决策项)
- [28. 版本与变更记录](#28-版本与变更记录)

---

## 0. 文档职责、状态与授权边界

### 0.1 本文件的职责

`IMPLEMENTATION_PLAN_PHASE7_5_C40.md` 是 **Phase7.5 的候选执行计划**。它：

1. 定义 Phase7.5 的范围、架构、工作包、测试、实验、统计口径、Gate、
   Stop rule、预算与治理；
2. 冻结 clean-room 边界与命名，使下一会话在获得授权后可以**直接执行**；
3. **不**改写、不取代 `IMPLEMENTATION_PLAN_LATEST.md`（Phase7 V7，
   byte-frozen plan of record）；
4. **不**改写 Phase4–7 的任何 artifact、manifest、disposition 或结论。

### 0.2 与既有计划文件的关系

| 文件 | 角色 | Phase7.5 对其的操作 |
| --- | --- | --- |
| `IMPLEMENTATION_PLAN_LATEST.md` | Phase7 V7 byte-frozen plan of record | **只读引用**，不修改、不取代、不合并 |
| `IMPLEMENTATION_PLAN_V1..V6_ARCHIVED.md` | 历史归档 | 只读 |
| `IMPLEMENTATION_PLAN_PHASE7_5_C40.md`（本文件） | Phase7.5 候选计划 | independent review已闭合为0/0/0，G0q artifact可用；当前等待用户授权，授权后才byte-freeze并pin |
| `PROJECT.md` / `HANDOFF.md` / `TRACKING.md` | 项目事实来源 | 按仓库协作指令在每轮有效讨论后更新；属 `plan drafting/review` 授权范围 |

### 0.3 证据分级（全文强制）

| 级别 | 含义 |
| --- | --- |
| `verified-local` | 本机可复核的 git / 文件 / 命令输出 |
| `verified-code` | 直接读取源码得到的事实（含行号） |
| `verified-docker` | 在固定 Docker image 内实测得到的结果 |
| `derived` | 由上述事实推导，推导链在文中给出 |
| `estimate` | 估计值，无实测支撑，**不得**当作结论 |
| `proposed` | 建议方案，未授权、未冻结 |
| `external-unverified` | 外部来源、本项目未能独立复核 |

### 0.4 授权边界（硬性）

```text
本轮授权范围 = 编制 + 只读审阅 + 发布本计划文档
             + 生成本计划的 review artifact（纯文档）
             + 由隔离 reviewer 执行 G0q quarantine signature 提取
               （只读 collaborator ref，只产出哈希/签名）
             + 同步更新 PROJECT.md / TRACKING.md / HANDOFF.md（仓库协作指令要求）
以上五类工作统称 `plan drafting/review`，不涉及任何实现代码、branch 或实验。

禁止（未获授权前）：
  - git branch / git checkout -b / git worktree add
  - 任何源码文件的创建或修改（含测试）
  - pytest / ruff / black / isort 等任何测试或 lint 执行
  - docker run / docker build
  - 任何 GPU 占用
  - 任何对 Phase4–7 结果目录的写操作
```

---

## 1. Executive Summary

### 1.1 一句话定义

**Phase7.5 是对合作者 V40 分支所体现的方法（`C40 = G40 structured grounded
selector × R0 Raw+RoPE executor`）的 clean-room 独立重实现与扩展评测：只采用
其研究思路与已审计的行为规范，不使用其任何源码，在本项目当前
`cross-store-substrate` 底座上从全新 branch 重建，并在其已实现能力之外补齐
未实现能力，最后与本项目既有可行性测试、固定 workflow 及外部质量基准统一
评测。**

### 1.2 核心判断（`derived`）

1. **恢复 primitive 无新颖性**。C40 的数据面与本项目 Phase4 定义的
   `R0 Raw+RoPE` 数学等价（复制 V、按 `rope_delta` 旋转 K、岛外 dense）。
   **不得**为它引入新的 recovery primitive 编号（不得叫 `R6` / `L0`）。
2. **新贡献在策略层与系统层**：structured repository-event provenance 驱动的
   **admission / selection / invalidation**，加上 strict-middle span 的
   **系统组合**（middle-span 状态机、consume/produce 双角色生命周期、
   approx provenance/depth、cross-store 记账复用）。
3. **性能不得默认转正**。Phase7 在同 image / 同模型 / `chunk=4096` 下已判
   R0 为 `NEGATIVE`（paired request-path median `0.7723 / 0.7751 / 0.9334 /
   0.9362`；N8 full-setup `0.6086–0.6419`）。C40 相对 R0 只多一个 selector，
   selector 不改变已进入复用路径的每 token 搬运成本。因此
   **不利先验成立，必须由 pilot 检验，禁止预先宣称收益**。
4. **最高价值在正确性与覆盖率**：冻结语料与声明collector范围内的
   `collector_observed_FN=0`、fail-closed完整性、time-weighted coverage
   `w` 的真实值、以及质量损伤上界，比speedup headline更重要。
5. **当前底座存在一个必须新写的 seam**：`resolve_reuse_spans()` 只支持
   "从 `exact_length` 起连续开始"的 span，strict-middle island 会被判
   `prefix_gap` 并整体退回 dense。若不新写 middle-span controller，
   **C40 将恒等于 dense，实验毫无意义**（`verified-code`）。

### 1.3 Phase7.5 的成功定义

| 层级 | 成功条件 | 失败条件 |
| --- | --- | --- |
| **Clean-room 合规** | 在声明的扫描范围与检查方法下**未检测到**禁止的 collaborator 源码血缘（必要不充分，§4.4） | 任何 import / 复制 / cherry-pick / patch-id / blob / AST 签名命中被检出 |
| **正确性** | 冻结语料上 `collector_observed_FN = 0`；same-context 张量与输出未超出冻结的 `baseline_envelope`；四计数族恒等式成立且不跨族混计；**在所执行的 soak 与测试范围内未观察到 orphan/leak** | `collector_observed_FN > 0`，或 same-context 超出 baseline envelope → 方法不可用 |
| **覆盖率** | `r`、`w`、`C_selector` 全部披露；conditional 结论必与 `w` 同现 | 只报 conditional 而缺 `w` / `mu_theta` → 判 `INVALID` |
| **速度** | confirmatory 阶段 `mu_theta` 判定为 `POSITIVE`（`L90 > delta0`，§19.2b） | `SMALL_POSITIVE_BELOW_MDE` / `INCONCLUSIVE` / `NEGATIVE` 均不发布 speedup headline；**CI 跨 0 或跨 `delta0` 一律 `INCONCLUSIVE`** |
| **质量** | 给出可信损伤上界 `X pp`（基于实测 discordant rate） | Dense 自身 run-to-run 翻转率 > 15% → 先修 harness |
| **治理** | 每 Gate 执行前有明确授权与 pinned manifest；最终 open P0/P1 = 0 | 无授权执行 → 结果不进入 disposition |

### 1.4 一页式路线图

```text
P7.5-G0a  document authority / plan review        host 只读, 0 GPU, 无 Docker   [AUTHORIZED]
   │
P7.5-G0q  quarantine signature extraction         host 只读, 0 GPU, 无 Docker   [AUTHORIZED]
   │        （隔离 reviewer；BLOCKED_QUARANTINE_INPUT ⇒ 链中断）
   │
P7.5-G1a  branch bootstrap                        host git, 无 Docker           [branch auth]
   │
P7.5-G0b  manifest / bootstrap builder            Docker CPU                    [impl + docker auth]
   │
P7.5-G1b  provenance / dependency / cleanroom     Docker CPU                    [manifest authorized]
   │
P7.5-G2   selector / identity / property          Docker CPU
P7.5-G3   middle-span runtime + lifecycle         Docker CPU
P7.5-G4   same-context canary                     Docker GPU, Track B
P7.5-G5   cross-context pilot                     Docker GPU, Track B
P7.5-G6   confirmatory primary C40-1R0            Docker GPU, Track C
P7.5-G7   conditional extensions (mR0/R1k/host)   Docker GPU, Track D
P7.5-G8   workflow scheduler S0/S4                Docker GPU, Track D
P7.5-G9   RepoBench-P / SWE-bench 质量            Docker GPU, Track E  [需 G3 + G4]
P7.5-G10  prefetch composition（正交，条件）      Docker GPU, Track F
P7.5-G11  consolidation / dual review / disposition Docker CPU, 0 GPU
```

每个 Gate 都是**独立授权单元**；后一个 Gate **不由**前一个 Gate 的结果自动触发。
`G0a → G1a → G0b → G1b` 的顺序是为消除"Gate 需要 manifest / manifest 需要
branch / branch 需要 Gate Exit"的循环依赖而冻结的，不可调换。

---

## 2. Authority 与输入证据

### 2.1 Authority 文档（只读）

| 编号 | 路径 | 用途 |
| --- | --- | --- |
| A1 | `/home/chris/Workspaces/code-agent-kvcache/PROJECT.md` | 项目事实、决策 D-xxx、约束、Phase7.5 决策记录（§`2026-07-29T17:42:31-07:00`） |
| A2 | `IMPLEMENTATION_PLAN_LATEST.md` | Phase7 V7 byte-frozen；**不得修改或取代** |
| A3 | `research/CODING_AWARE_V40_BRANCH_TECHNICAL_REPORT.md` | C40 方法定义、P0 阻塞项、middle-span 缺口、测试与统计设计（2266 行） |
| A4 | `research/phase_reports/PHASE4_RECOVERY_METHODS_REPORT.md` | R0–R5 恢复方法权威定义 |
| A5 | `research/phase_reports/PHASE5_WORKFLOW_SCHEDULING_REPORT.md` | S0–S4 调度、P0–P3 预取轴、固定 workflow |
| A6 | `research/phase_reports/PHASE6_CROSS_STORE_SUBSTRATE_REPORT.md` | cross-store 底座语义与非目标 |
| A7 | `research/phase_reports/PHASE7_INTEGRATED_EVALUATION_REPORT.md` | R0 `NEGATIVE` 结论、chunk 混淆、四本账 |
| A8 | `research/phase_reports/PHASE4_TO_PHASE7_SUMMARY.md` | 跨阶段总结与教训 |
| A9 | `research/RESEARCH_SYNTHESIS.md` | KVFlow / KVCOMM 职责边界、novelty 上限 |
| A10 | `HANDOFF.md` / `TRACKING.md` | 当前状态快照与 append-only 记录 |

### 2.2 代码底座 pin（`verified-local`）

```text
substrate worktree :
  /home/chris/Workspaces/kvcache-research/worktrees/cross-store-substrate

base commit (proposed base for Phase7.5 branch) :
  origin/research/cross-store-substrate@0206f17b4255e4b248dafaaeb943be57428dae2f
  tree = 3873d5683f98410524479c57c2068c6e1df98f7d
  date = 2026-07-28 20:37:05 -0700
  subject = "results: bind the complete Phase7 publication record"

primary pin (Phase7 implementation reference, 只读引用) :
  81405f4278b034911bc613c4ee17c79d15ee8f35
  subject = "fix: include nested Phase7 artifacts in provenance"
```

### 2.3 固定实验环境（继承 Phase7，`verified-docker` 于 Phase7）

```text
image  : ghcr.io/ccdd2023/sglang@sha256:0be6e16e2eb288dfd5fa8b0b41015f61731a139fb961d3366ccedf834289d781
model  : Qwen/Qwen3-0.6B @ c1899de289a04d12100db370d81485cdf75e47ca
device : SM75 / RTX 2080 SUPER 8 GiB
primary: chunked_prefill_size = max_prefill_tokens = 4096, page_size = 1,
         tp = pp = 1, enable_mixed_chunk = false
```

> **禁止使用 tag**，必须使用 digest。任何模型或 image 变更都构成新的
> fingerprint，必须重新 pin 并在 manifest 中披露。

### 2.4 Phase7 的不利先验（预注册基准线）

| 项 | 值 | 来源 |
| --- | --- | --- |
| R0 paired request-path median（4 setting） | `0.7723 / 0.7751 / 0.9334 / 0.9362` | A7 §1.3(2) |
| R0 N8 full-setup | `0.6086 – 0.6419` | A7 §1.3(2) |
| chunk1024 的假性 headline | `1.737x`（显式 `headline=false`） | A7 §1.3(3) |
| Phase7 实际用量 | `22 starts / 1.310142 h` + correction `1 start / 0.098332 h` | A10 |
| Phase7 RESULT_MANIFEST | `88/88`，`known_gaps=[]` | A10 |

**这条基准线必须写入 Phase7.5 的每一份 manifest 与每一份结果 summary。**

### 2.5 底座既有能力清单（`verified-code`，可直接复用，**不重写**）

| 能力 | 位置（`xs:` = substrate） | 复用方式 |
| --- | --- | --- |
| copy + RoPE 变换 | `xs:python/sglang/srt/mem_cache/approx_kv/radix_backend.py`（`copy_and_rotate`） | 直接调用 |
| plan 执行与全覆盖校验 | `xs:approx_kv/transfer.py`（`execute_reuse_plan`，`require_full_coverage=True`） | 直接调用 |
| plan / span / dense range 类型 | `xs:approx_kv/types.py`（`RecoveryMode.COPY`、`TransferSpan`、`DenseRange`、`KVReusePlan`、`KVTransferStats`） | 直接复用；C40 只增薄适配字段 |
| segment store 生命周期 | `xs:approx_kv/store.py`（`register/lookup/is_current/pin/unpin/gc_expired_leases/ensure_resident/load_resident/commit_residency/release/reset`） | 直接调用 |
| cross-store 分配 / 预算 / 对象图 / 策略 | `xs:cross_store/{coordinator,allocator,budget,object_graph,policy,class_order,event_clock,types}.py` | 直接调用；**禁止自建驱逐循环** |
| provisional slot 生命周期 | `xs:approx_kv/runtime.py`（release / commit）、`xs:schedule_batch.py` 既有 hook | 直接复用记账语义 |
| prefix ownership 保护 | `xs:approx_kv/runtime.py`（`protect_request_prefix`，含 `to_dec_params()` SWA metadata） | 必须保留，不得绕过 |
| fallback 记账 | `xs:approx_kv/manager.py`（`record_fallback` / `record_request`） | 必须复用，新增 reason 与 canonical inventory（§2.6）互斥 |
| EPIC leading-k 能力 | `xs:approx_kv/{epic_plugin,epic_runtime,epic_recompute,epic_capability}.py`，`SUPPORTED_EPIC_K_VALUES=(0,2,4,8,16,32)` | conditional lane 复用（`C40-R1-k`） |
| host residency / async transfer | `xs:approx_kv/{hicache_backend,async_transfer}.py` | conditional lane 复用 |
| 结果 manifest 构建与递归校验 | `xs:benchmark/approx_kv/build_result_manifest.py` | 新增 Phase7.5 builder，沿用 `--check` 模式 |
| 离线 consolidator 模式 | `xs:benchmark/approx_kv/consolidate_phase7_results.py` | 作为**模式参考**新写 Phase7.5 consolidator |

### 2.6 底座既有 fallback reason inventory（`verified-code`，**不得重名**）

**不得**在计划或代码中硬编码"既有 N 项"这一数字。冻结 base 的 reason 集合必须
由**自动生成的 canonical inventory** 给出，并绑定 sha256。

`0206f17b...` 上实测存在的固定 reason（`verified-code`，本清单本身也必须由
工具复算，不作为手写权威）：

```text
approx_kv_core_disabled            cross_store_error
cross_store_exact_pressure_error   cross_store_exact_pressure_failed
cross_store_reservation_failed     device_allocation_failed
epic_attention_sink_disabled       epic_forward_batch_construction_failed
epic_forward_batch_release_failed  epic_forward_batch_unavailable
epic_model_runner_unbound          epic_not_genuinely_layerwise
epic_plugin_missing                epic_plugin_wrong_type
epic_precomputed_repair_invalid    epic_recompute_failed
prefix_gap                         registration_dependency_missing
registration_store_capacity        residency_load_failed
residency_miss                     rope_config_unavailable
source_pin_stale                   source_slice_mismatch
stale_handle                       store_miss
```

**`DenseRange.reason` 常量族**（同样占用 reason 命名空间）：

```text
epic_leading_k_repair              (xs:approx_kv/epic_plugin.py, EPIC_LEADING_K_REPAIR_REASON)
```

> `stale_handle` / `residency_miss` / `source_slice_mismatch` 由
> `xs:approx_kv/transfer.py` 的 `fallback_chunks` 路径产生，**不经过**
> `record_fallback`，而是进入 `KVTransferStats.fallback_reasons` 并由上层
> 归因。扫描器必须覆盖这一路径，否则会重演"14 项"式的漏计。

**另有动态 prefix-family**（reason 字符串带 `:{detail}` 后缀）：

```text
epic_capability_unsupported:<detail>
epic_plan_invalid:<detail>
```

**规则**：

1. WP0b 必须实现 `build_p75_reason_inventory`，从冻结 base **自动扫描**至少以下
   四类来源，生成 `evidence/reason-inventory.json`（含 sha256）：
   - `manager.record_fallback(<literal|f-string>)` 的全部实参；
   - `KVTransferStats.fallback_reasons` 的 append/初始化字面量
     （含 `transfer.py` 的 `fallback_chunks[chunk] = "<reason>"`）；
   - `backend.dense_prefill(..., reason=<literal>)` 的 reason 实参；
   - `DenseRange.reason` 与各 plugin 的 reason 常量（如 EPIC 的 repair reason）。
   扫描必须基于 AST 而非纯正则，并对无法静态求值的 reason 表达式**报错而非跳过**；
2. C40 新增 reason 一律 `c40_` 前缀，冲突检查同时覆盖**固定 reason**与
   **prefix-family**（`c40_` 不得与任何既有 prefix 冲突）；
3. 若 base 的 inventory 与已版本化的 sha256 不一致 ⇒ Gate 不通过（防止底座漂移）；
4. C40 的 exclusive 记账断言以该 inventory 为分母定义域，**不引用固定数字**。

### 2.7 必须新写的 seam（`verified-code`，本计划的技术核心）

`xs:python/sglang/srt/mem_cache/approx_kv/runtime.py` 的 `resolve_reuse_spans()`：

```python
next_target = exact_length
for segment in ordered_segments:
    if segment.target_end <= exact_length:
        continue
    if segment.target_start > next_target:
        break                      # <-- 跨 gap 的 middle span 在此被丢弃
    ...
restore_length = restore_end - exact_length
if restore_length <= 0:
    ...
    manager.record_fallback("prefix_gap", pending_length)
    manager.record_request("reuse", "dense_fallback")
    return None
```

**语义**：底座只处理"从 `exact_length` 起连续开始"的恢复 span。C40 的岛
**严格位于中部**，`target_start > exact_length` 几乎总成立，必然存在
dense prefix gap，会被 `break` 丢弃并整体退回 dense。

**结论**：必须新写 middle-span staging controller（§7、§8）。
**禁止表述**："R0 runtime 零改动即可支撑 C40"。

---

## 3. 方法定义与新颖性边界

### 3.1 冻结的方法定义

```text
C40  =  G40 (structured grounded observation selector)  ×  R0 (Raw + RoPE executor)
```

- **`R0` 执行器（数据面，无新颖性）**：`V` 逐 token 原样复制；`K` 施加
  `rope_delta = target_start - source_start` 的旋转；岛外全部 dense。
  **无** context-dependent 修正、**无** selected-token 重算、**无** anchor 插值。
- **`G40` 选择器（策略面，新贡献所在）**：基于**结构化 repository 事件
  provenance** 的准入 / 选择 / 失效控制，决定"哪一段历史 tool observation
  允许被搬到下一个请求，以及何时必须放弃"。

**"有损"的来源**：token 序列相同，但该段 token 在**新的左上下文**下本应产生
不同的 K/V。RoPE 只修正位置坐标，不修正 causal context 差异。

### 3.2 新颖性边界（硬性）

| 层 | 成分 | 最近先例 | 新颖性判断 |
| --- | --- | --- | --- |
| 数据面 primitive | copy V + RoPE K | 本项目 R0 / EPIC `k=0` / PIC 族 | **无** |
| 复用几何 | 非 prefix、中部岛、唯一匹配 | Prompt Cache / PIC / MEPIC | 低 |
| identity / lifecycle | token hash + generation + lease | 通用 KV store 设计 | 无 |
| **准入 gate** | 结构化 repository read/write provenance + generation + content hash | 未见完全相同的公开系统 | **唯一可能有增量的一层** |
| **系统组合** | strict-middle 状态机 + consume/produce 双角色 + approx depth + cross-store 记账 | 无直接对应 | **第二个可主张的增量** |

**可主张的增量表述（唯一允许）**：

> "Repository-event-grounded admission control for non-prefix KV reuse in
> coding agents, together with a strict-middle span reuse system integrated
> into a cross-store KV substrate."

### 3.3 明确"不是什么"

| 说法 | 成立？ | 理由 |
| --- | --- | --- |
| C40 是 prefetch | **否** | 纯 KV 复用；coding-only lane 不依赖 prefetch |
| C40 是 KVCOMM `2510.12872` 重建 | **否** | 无 canonical base、无 ΔK/ΔV、无 anchor pool、无 multi-anchor 插值、无 entropy/length shareability gate |
| C40 是 CacheBlend selective repair | **否** | 无 HKVD / K-deviation 打分，无 selected-token 逐层重算 |
| C40 是 Cache-Craft / CacheTune | **否** | 无 CCI/CFO 判据，无 roofline repair-ratio controller |
| C40 是 KVFlow | **否** | 无 steps-to-execution、无 priority eviction、无 CPU 分层调度 |
| C40 的 source 是 synthetic replay | **否** | source 必须由**服务端**在前一个**真实** agent 请求完成时物化 |
| token 相同 ⇒ KV 相同 | **否** | 这正是"lossy"的来源 |
| "可变编码" 是 KVCOMM 原文术语 | **否** | 不得把 delta compression / AST index / HiCache 写成论文已有能力 |

### 3.4 正确的实现顺序（本项目既定原则，不可跳步）

```text
1. exact cache
   先证明 exact 路径完全正确：无自我驱逐、无污染、无 orphan、
   feature gate 关闭时 copied_tokens ≡ 0。
2. controlled C40 reconstruction
   在 exact 之上加受控的 G40 × R0，全程 fail-closed，全部 span 严格中部，
   全部决策可审计。
3. dense fallback
   任何不确定一律回落，且必须可归因到唯一 terminal reason。
```

**禁止**跳过第 1 步直接做第 2 步；**禁止**让第 2 步的近似 KV 写入 exact
Radix（除非经过 dense materialization）。

---

## 4. Clean-Room 合规合同

### 4.1 允许的输入（`ALLOWED`）

1. `research/CODING_AWARE_V40_BRANCH_TECHNICAL_REPORT.md` 中已审计的
   **行为规范**（behavioural specification）：六步选择算法的**语义描述**、
   fail-closed 语义、strict-middle 约束、min/cap 语义、唯一性要求。
2. 已记录的**失败模式**（failure modes）：B-01 同路径写漏检、B-02 mixed
   read/write group 被接受、B-03 文档与代码矛盾、B-04 dead feature flag、
   B-05 lease TTL 无 GC、B-06 依赖不自包含与硬编码路径。
3. **公开论文与公开研究机制**：R0 Raw+RoPE、EPIC / LegoLink leading-k、
   Prompt Cache / PIC / MEPIC / MiniPIC 的 non-prefix modular reuse 设定、
   CacheBlend 的 selective repair 概念（仅作对照，不实现）。
4. 本项目自有的 Phase4–7 代码、测试、报告与教训。

### 4.2 禁止的输入（`FORBIDDEN`）

```text
禁止 copy / cherry-pick / merge / rebase / import / vendored 引入：
  - review/coding-aware-v40-prefetch-20260729 及其任何 ref 的源码
  - 该分支的 tests、runner、campaign 脚本、audit 脚本
  - 该分支的 runtime（kvcomm_exact.py / kvcomm/ / coding_reuse_policy.py /
    bridge_reuse_litellm_model.py 等）
  - 该分支的 sidecar/manifest 格式实现（version-1/2/3 的实现代码）
  - 该分支的正则表达式常量表（_READONLY_EVIDENCE_COMMAND、
    _MUTATION_MARKERS、_SHELL_MUTATION、_INPLACE_MUTATION 等）
  - 该分支的旧 kvcomm store / manager / scheduler
```

> **关于正则常量**：C40 的根修方案本就是"不再依赖命令正则"，因此不存在
> "需要复制正则表"的场景。若 conditional lane 需要次级启发式信号，必须
> **独立重新设计**，且**只能加严不能放宽**（§9.6）。

### 4.3 独立命名要求（冻结）

| 类别 | 要求 | 反例（禁止） |
| --- | --- | --- |
| 模块 | 新包 `coding_c40/`，全部新文件名 | `coding_reuse_policy.py`、`kvcomm_exact.py` |
| 类型 | `C40*` / `ToolEvent*` / `GroundedIsland*` 前缀 | `ExactMiddleCanaryController`、`KVCommManager` |
| 环境变量 | `SGLANG_APPROX_KV_C40_*` | `SGLANG_KVCOMM_*`、`SGLANG_CODING_AWARE_*` |
| Terminal reason | `c40_*` 前缀 | 复用旧 reason 名 |
| manifest | `c40_manifest_version: 1`（**新格式，非 version-4**） | "在 version-3 基础上扩展" |
| 结果目录 | `phase7_5_c40/` | `c40/`、`phase7/` |
| runner | `run_p75_*.py` | `run_v40*.py`、`run_v4x_campaign.py` |
| tests | `test_c40_*.py` | 任何与分支同名的测试 |

### 4.4 自动化 clean-room compliance test（必须实现，WP1b）

`test/registered/unit/mem_cache/test_c40_cleanroom_compliance.py`（`proposed`）

| 检查 | 断言 |
| --- | --- |
| CR-1 禁止导入 | 全仓 `grep` 无 `import .*kvcomm`、`from .*kvcomm`、`coding_aware`、`coding_reuse_policy`、`bridge_reuse_litellm` |
| CR-2 禁止符号名 | 无 `ExactMiddleCanaryController`、`KVReuseSidecar`、`_READONLY_EVIDENCE_COMMAND`、`_INPLACE_MUTATION` 等 collaborator 私有符号 |
| CR-3 git 血缘（**只消费 quarantine 签名**） | 基于 §4.6 quarantine manifest：(a) `git merge-base --is-ancestor <collaborator_tip> HEAD` 成立则自动失败；(b) `<base>..HEAD` commit 与 `commit_ids` 有直接交集则自动失败；(c) 新增/修改后的精确 blob 命中 `exclusive_blob_hashes` 则自动失败；(d) 与 `patch_id_alerts` 的命中**只进入人工复核**，不能自动失败。前提是 `quarantine_status=="AVAILABLE"`；若为 `BLOCKED_QUARANTINE_INPUT`，CR-3未通过且不得进入G1a |
| CR-3b 结构签名比对 | 对整个new-branch diff涉及的**完整文件和函数**计算normalized AST signature，分别与`ast_signatures_file`/`ast_signatures_function`比对；不对孤立diff hunk做AST解析。命中只进入人工复核。先应用`allowed_shared_signatures`（shared-base/public algorithm allowlist），再记录未豁免命中的裁决 |
| CR-4 无硬编码用户路径 | 扫描范围**只含源码**，显式排除冻结的历史结果目录（其中 Phase7 的 raw/log 合法地包含宿主路径，`verified-local` @ `0206f17b`：`benchmark/approx_kv/results/` 下有 `115` 处 `/home/` 命中，均为非 `.py` 的 raw/log 文件）。冻结命令与实测基线：<br>`grep -rn "/home/" python/sglang/srt/mem_cache/ benchmark/approx_kv/ test/registered/ --include='*.py' \| grep -v '/results/' \| grep -v 'test_'`<br>基线输出 = **空**（`verified-local`，0 行）。Phase7.5 新增代码后必须仍为空 |
| CR-5 新命名空间（**带 allowlist**） | 新增的 **C40 源码 / 测试 / runner** 路径必须匹配 `coding_c40/` 或 `*_c40_*` 或 `run_p75_*`。以下**显式 allowlist** 不受命名约束，但必须逐项列举并在 manifest 中固定：<br>**(a) 支持性新增文件**：`Dockerfile.p75`、`requirements.lock`、`benchmark/approx_kv/build_p75_manifest.py`、`benchmark/approx_kv/build_p75_result_manifest.py`、`benchmark/approx_kv/results/phase7_5_c40/**`（含 evidence）。<br>**(b) 既有文件的最小修改**（由 §8.6.4 生成并绑定 sha256）：`managers/schedule_batch.py`、`managers/schedule_policy.py`、`managers/scheduler.py`、`mem_cache/allocation.py`、`mem_cache/common.py`、`mem_cache/approx_kv/config.py`。<br>allowlist 之外的任何新增或修改即判失败。**allowlist 只豁免命名规则，不豁免血缘扫描**（见 CR-9） |
| CR-9 全 diff 血缘扫描 | 扫描对象 = **整个 new-branch diff**（`git diff <base>..HEAD`），**包括** allowlisted 既有文件的 **modified hunks**。每个 hunk 都要过 CR-1/CR-2/CR-3 与 quarantine 签名比对（§4.6）。**禁止**以"该文件在 allowlist 内"为由跳过其 hunk 的血缘检查 |
| CR-6 manifest 独立 | Phase7.5 manifest schema 名为 `c40_manifest_version`，不含 `sidecar_version` 字段 |
| CR-7 写目标隔离 | **宿主持久化写挂载**恰好两处；容器内`/scratch`/`/tmp` tmpfs临时写允许但不得bind到宿主且`--rm`后销毁。mount静态断言拒results根rw、global log非file bind、phase目录缺失及任何额外host-rw |
| CR-8 Phase4–7 只读 | 采用**显式冻结目录清单**而非通配（`phase{2..7}*` 会误匹配 `phase7_5_c40`）。冻结清单：`results/phase2`、`results/phase3`、`results/phase4-r1`、`results/phase5-scheduler`、`results/phase6`、`results/phase7`。任何 Phase7.5 runner / consolidator 对清单内路径的写操作被断言拒绝；`results/phase7_5_c40` 是唯一允许写入的目录 |

**Exit 条件**：CR-1..CR-9（含 CR-3b）全绿，并把结果写入
`evidence/cleanroom-compliance.json`（自哈希，含 allowlist 全文与每项检查的
命令、输出摘要与判定）。

> **合规检查的能力边界（必须披露，逐字写入 disposition）**：
>
> 自动化检查**只能**支持这一条结论：
> **"在本次扫描范围（`<base>..HEAD` 全 diff，含 allowlisted 文件的 modified
> hunks）与本次检查方法（import / 符号 / merge-base / patch-id / blob hash /
> normalized AST signature）下，未检测到禁止的 collaborator 源码血缘。"**
>
> 它**不能**支持"无血缘"、"完全独立"、"零借鉴"等任何更强的断言。
> clean-room 的实质保证来自 §4.1/§4.2 的输入边界、§4.6 的隔离流程与执行纪律；
> **自动化检查是必要不充分条件**。

### 4.6 Quarantine 流程（隔离 reviewer 产出签名，实施者只消费）

**问题**：要检测 collaborator 血缘，就必须拥有 collaborator 源码的指纹；
但让实施者接触这些源码会直接破坏 clean-room。

**执行载体**：`P7.5-G0q` / `WP0q`（§20、§16），属 **G0a 的子项**，
在当前 `plan_drafting/review_authorized = true` 范围内即可执行
（只读 collaborator ref、只产出哈希与签名、不写任何源码、不建 branch）。

**冻结流程**：

```text
角色分离（硬性）：
  隔离 reviewer   = 不参与 Phase7.5 任何实现工作的独立执行者/会话
  实施者          = 编写 C40 代码的执行者/会话
  二者**不得**是同一会话，且实施者**不得**读取 collaborator ref 的任何文件内容

步骤 1（隔离 reviewer，单独授权）
  在隔离环境中对 collaborator-exclusive 范围
      git rev-list <merge_base>..<collaborator_tip>
  计算：
  **冻结的五类签名字段（全文唯一命名，不得使用别名）**：
      commit_ids[]                该范围内每个 commit 的 sha
      exclusive_blob_hashes[]     **collaborator-exclusive** blob 的 sha1 与 sha256
                                  （已在冻结 base 中出现的 blob **排除**，
                                    否则会把 shared base 误判为 collaborator 血缘）
      patch_id_alerts[]           git patch-id 输出（**alert 用途**，非自动 fail）
      ast_signatures_file[]       **file-level** normalized AST structural signature
      ast_signatures_function[]   **function-level** normalized AST structural signature
  辅助字段：
      allowed_shared_signatures[] 与冻结 base 相同的签名、以及公开算法常规形态
                                  （RoPE 旋转 / sha256 / dataclass 样板 /
                                    标准状态机等）的签名 allowlist
  **不得**输出任何源码、片段、标识符原文、注释、字符串常量或路径以外的内容。
      （路径本身也只保留 sha256，不写明文路径。）

步骤 2  产出 quarantine manifest（自哈希）
      文档 authority repo：
          evidence/review/c40-quarantine-manifest.json
      字段仅含上述五类哈希/签名 + 范围定义 + 生成方法 + self_sha256
      merge_base / collaborator_tip 以 sha 记录（tip 冻结为 13671eb708da...）

步骤 3（实施者）
      G0b builder 把该文件**逐字节复制**到新 branch 的
        benchmark/approx_kv/results/phase7_5_c40/evidence/review/
      并断言 source/destination sha256 相同；实施者只读消费 branch 内副本
      的哈希/签名做比对；
      产出 evidence/quarantine-consumed.json，记录消费的 manifest sha256、
      比对结果与任何命中的人工裁决。
```

#### 4.6.3 自动 fail vs 人工复核（**冻结分级**）

```text
自动 fail（命中即 SR-9，立即停止全部 lane）：
  A1  禁止 import / 禁止符号名（CR-1 / CR-2）
  A2  精确 forbidden blob hash 命中（新增或修改后的 blob 的 sha1/sha256
      出现在 quarantine.exclusive_blob_hashes 中）——这等价于逐字节复制
  A3  已确认的 cherry-pick：merge-base 祖先关系成立，或 commit sha 直接出现

人工复核（命中**不自动 fail**，须记录裁决）：
  M1  patch-id 命中：两个 commit 的 diff 规范化后相同。
      共同 seam（例如都在同一处插入一行 hook 调用）会自然产生相同 patch-id，
      **不足以**证明抄袭。裁决须记录 hunk 内容摘要与判定理由。
  M2  normalized AST structural signature 命中：
      公开算法的常规实现形态（RoPE 旋转、sha256、dataclass 样板、
      标准状态机 switch）会自然撞签名。
      先查 allowed_shared_signatures；未命中 allowlist 才进人工复核。

**硬性**：M1 / M2 **绝不**自动 fail —— 否则任何在同一 seam 上做正确接线的
clean-room 实现都会被误杀。但每一条命中都必须在
evidence/quarantine-consumed.json 中留下裁决记录（裁决人、理由、时间戳），
未裁决即视为 Gate 未通过。
```

**manifest 字段（§14.1 的 `cleanroom` 段据此改写）**：

```json
{
  "cleanroom": {
    "policy_version": 2,
    "implementation_consumed_collaborator_source": false,
    "reviewer_only_source_access": true,
    "quarantine_manifest_sha256": "<...>",
    "quarantine_scope": {
      "merge_base": "<sha>",
      "collaborator_tip": "13671eb708da...",
      "signature_kinds": ["commit_ids","exclusive_blob_hashes","patch_id_alerts",
                        "ast_signatures_file","ast_signatures_function"],
      "contains_source_text": false
    },
    "allowed_shared_signatures_sha256": "<...>",
    "auto_fail_checks": ["forbidden_import","forbidden_symbol",
                         "exact_forbidden_blob","confirmed_cherry_pick"],
    "manual_review_checks": ["patch_id_alerts","ast_signatures_file",
                             "ast_signatures_function"],
    "quarantine_status": "AVAILABLE",
    "manual_review_adjudications": [],
    "allowed_inputs": ["behavioural_spec", "failure_modes", "public_papers"],
    "scan_scope": "full_branch_diff_including_allowlisted_modified_hunks",
    "compliance_test": "test_c40_cleanroom_compliance.py",
    "compliance_evidence_sha256": "<...>",
    "conclusion": "no_forbidden_lineage_detected_under_declared_scope_and_methods",
    "conclusion_is_necessary_not_sufficient": true
  }
}
```

> **表述限定（硬性）**：`conclusion` 字段的取值只能是
> `no_forbidden_lineage_detected_under_declared_scope_and_methods`。
> **禁止**出现 `no_lineage` / `clean` / `independent` 等更强措辞。

### 4.5 声明模板

见 §4.6 的 `cleanroom` 段（`policy_version = 2`）。该段必须原样写入
每份 plan manifest 与每份结果 summary。

---

## 5. Branch / Base / 结果目录冻结

### 5.1 冻结项

```text
proposed_branch : research/phase7.5-c40-cleanroom
proposed_base   : origin/research/cross-store-substrate@0206f17b4255e4b248dafaaeb943be57428dae2f
worktree_root   : /home/chris/Workspaces/kvcache-research/worktrees/  (proposed)
worktree_name   : phase7.5-c40-cleanroom                              (proposed)
runtime_results : /results/phase7_5_c40                               (容器内 staging)
versioned_results: benchmark/approx_kv/results/phase7_5_c40/          (仓库内最终版本化)
```

### 5.2 Branch creation commands（**未来步骤，当前未授权**）

> 以下命令**当前不得执行**。仅在 `branch_creation_authorized = true` 之后，
> 由下一会话按顺序执行，并把每条命令的输出写入 `evidence/branch-creation.json`。

```bash
# 步骤 0：确认底座 clean 且 HEAD 正确（只读）
cd /home/chris/Workspaces/kvcache-research/worktrees/cross-store-substrate
git --no-pager status --short          # 期望：空
git --no-pager rev-parse HEAD          # 期望：0206f17b4255e4b248dafaaeb943be57428dae2f

# 步骤 1：只读复核 remote base（不更新本地 refs）
git --no-pager ls-remote origin research/cross-store-substrate

# 步骤 2：创建独立 worktree + 新 branch（不污染现有 worktree）
cd /home/chris/Workspaces/kvcache-research
git -C worktrees/cross-store-substrate worktree add \
    ../phase7.5-c40-cleanroom \
    -b research/phase7.5-c40-cleanroom \
    0206f17b4255e4b248dafaaeb943be57428dae2f

# 步骤 3：验证 base 与 clean-room 起点
cd worktrees/phase7.5-c40-cleanroom 2>/dev/null || cd ../phase7.5-c40-cleanroom
git --no-pager log --oneline -1
git --no-pager rev-parse HEAD^{tree}   # 期望：3873d5683f98410524479c57c2068c6e1df98f7d
git --no-pager branch --show-current   # 期望：research/phase7.5-c40-cleanroom

# 步骤 4：记录 clean-room 起点声明（不 push，**写到 worktree 之外**）
#   - 写 /home/chris/Workspaces/kvcache-research/results/phase7_5_c40/bootstrap/
#         branch-creation.json（含上述全部输出与 sha256）
#   - 不在新 worktree 内创建任何文件
#   - 不执行 git push（push 需单独授权）

# 步骤 5：验证新 worktree 仍然 clean
git --no-pager status --porcelain        # 期望：空输出
```

### 5.2b Bootstrap evidence 的落点（**必须在 worktree 之外**，NEW-01）

```text
问题：G1a 只有 branch 授权，没有 implementation 授权。
      若把 branch-creation.json 写进新 worktree，worktree 立刻变 dirty，
      HEAD 也不再等于 base 的干净状态，破坏 G1a 的 Exit 条件本身。

冻结落点（按优先级）：
  1. /home/chris/Workspaces/kvcache-research/results/phase7_5_c40/bootstrap/
         branch-creation.json
     （与全局 run log 同一 results 根，天然在任何 worktree 之外）
  2. 会话工作区（session workspace）下的等价路径
     ——仅当路径 1 不可写时使用，并在 evidence 中记录实际落点

G1a 结束后必须成立：
  git -C <new worktree> status --porcelain   ->  空输出
  git -C <new worktree> rev-parse HEAD       ->  0206f17b4255e4b248dafaaeb943be57428dae2f

导入时机：
  G0b 的 builder 负责把 <bootstrap_dir>/branch-creation.json
  读入、校验、复制进 benchmark/approx_kv/results/phase7_5_c40/evidence/
  并写入 p75-plan-manifest 的绑定（sha256 以导入后的仓库内文件为准，
  同时记录 bootstrap 原始路径与其 sha256 以供追溯）。
```

**约束**：

- **不 push**、**不创建 PR**、**不改动** `cross-store-substrate` worktree 的 HEAD；
- **G1a 结束时新 worktree 必须 clean 且 HEAD == base**；
- Phase7.5 的全部实现只在新 worktree 内进行；
- 任何对 `origin` 的写操作（push / tag）需**单独授权**，并须先按仓库协作指令
  核实 `ccdd2023` 身份与权限，使用账号级显式认证，不改全局默认账号。

### 5.3 Phase4–7 冻结保护

```text
禁止：
  - 修改 benchmark/approx_kv/results/phase{2,3,4-r1,5-scheduler,6,7}/ 下任何文件
  - 修改 Phase4–7 的 RESULT_MANIFEST / DISPOSITION / review artifact
  - 把 Phase7.5 结果回填进 Phase7 统计
  - 合并 Phase7 与 Phase7.5 的任何数值

允许：
  - 只读引用 Phase7 的 R0 NEGATIVE 基准线（作为预注册先验）
  - 只读引用 Phase4–7 的 CPU 测试与 runner 作为模式参考（同一仓库、非 collaborator 代码）
```

---

## 6. 能力分层：Parity Core / Mandatory Extensions / Conditional Lanes

### 6.1 Layer 1 — Parity Core（必须实现，9 项）

对应"合作者已实现的能力"，Phase7.5 必须**功能等价地独立重实现**。

| # | 能力 | 验收要点 |
| --- | --- | --- |
| PC-1 | **real prior-request source** | source 由服务端在**前一个真实 agent 请求**完成时物化（`cache_finished_req` 等价 hook）；**禁止** synthetic replay 作为 source |
| PC-2 | **successful read-only tool observation** | 只有"工具执行成功 + 纯读"的 observation 可成为候选；成功性由结构化 `exit_code` / `success` 字段判定，不由输出文本猜测 |
| PC-3 | **排除 assistant / tool-call reasoning tokens** | 候选 token 只来自 `role == "tool"` 的消息体；`assistant_tokens_selected == 0` 必须为断言 |
| PC-4 | **exact token identity + 唯一出现 + strict-middle + min/cap + 单一最大岛** | token 序列在 **target prompt** 中**恰好出现一次**；**strict-middle 以 target 为准**：`0 < target_start` 且 `target_start + length < len(target_prompt_ids)`；**source 侧边界另行检查**：`0 <= source_start` 且 `source_start + length <= len(source_prompt_ids)`，且该 span 在 source 中已完整物化；`length >= min_tokens`；`length <= copy_cap`；primary lane 选唯一最大岛（截断后 token 数最大，同分取更新的候选） |
| PC-5 | **later same-path write invalidation** | source 之后任何对同路径的写事件使候选失效（结构化判定，见 §9） |
| PC-6 | **V copy + K RoPE relocation** | 复用 `copy_and_rotate()`；`rope_delta = target_start - source_start`；`require_full_coverage=True` |
| PC-7 | **dense prefix / suffix** | `copied_spans ∪ dense_ranges == [0, len(target))`，无重叠无空洞 |
| PC-8 | **fail-closed + exclusive terminal reason** | 任何不确定一律 dense；每次 **attempted** fallback 恰好一个 `primary_reason`；`Σ_reason tokens == attempted_recovery_failed_dense_tokens`（族 2，§12.4） |
| PC-9 | **no synthetic replay；coding-only 不依赖 prefetch** | primary lane 在 `SGLANG_APPROX_KV_PREFETCH=0` 下完整可用 |

### 6.2 Layer 2 — Mandatory Extensions（必须超越原实现，9 组）

| # | 扩展 | 说明 |
| --- | --- | --- |
| ME-1 | **结构化 ToolEvent provenance** | 不以 command regex 为 authority。字段：`repo_id` / `worktree_id` / `commit` / `dirty` / `worktree_generation` / `read_paths` / `write_paths` / `rename_pairs` / `unknown_effect` / `path_content_sha256_before` / `path_content_sha256_after` / `tool_success` / `tool_timeout` / `tool_truncated`。**unknown → fail closed** |
| ME-2 | **路径规范化、结构建模与 repo/branch/worktree identity isolation** | repo-relative 规范化；rename 建模为"旧路径写 + 新路径写"；symlink 解析与 target 路径同时视为受影响；directory-level mutation 展开为其下全部已知路径。<br>**repo/branch/worktree identity isolation 是 mandatory**：`repo_id` / `worktree_id` / `branch` 进入 segment key，不同 identity 的 source **互不可见**（这是正确性要求，不是可选实验） |
| ME-3 | **完整 fingerprint** | `model_id@revision` / `tokenizer_revision` / `chat_template_hash` / `rope_config_hash` / `dtype` / `kv_layout` / `tp` / `pp` / `page_size` / `image_digest`；跨 fingerprint 复用**必须拒绝** |
| ME-4 | **真正的 feature gate** | 关闭 C40 时断言 `copied_tokens ≡ 0` 且 selector 不被调用；**禁止** dead flag |
| ME-5 | **middle-span 状态机** | `DENSE_PREFIX → COPY_READY → DENSE_SUFFIX → COMMIT`；copy 失败 `→ DENSE_ISLAND_FALLBACK`（请求正常完成）；abort / reset / timeout `→ ABORTED_TERMINAL`；**terminal reject `→ TERMINAL_REJECTED`**；capacity deferral `→ ADMISSION_DEFERRED`（非 terminal）（§8） |
| ME-6 | **consume/produce 双角色 + 生命周期分层** | request-lifetime 状态跨 chunk 保留；per-round transient 单独清理；scheduler lock handoff 可审计；copied slot 在 suffix forward **之前**写入 `req_to_token` 并完成所有权转正 |
| ME-7 | **approx provenance / depth** | primary lane 只允许从 `approx_depth == 0` 的 dense/exact 请求物化下一 source；**禁止**近似 KV 写入 exact Radix；chaining 只作独立 diagnostic 轴并冻结最大 depth |
| ME-8 | **lease 自动 GC + final cleanup + retraction reset** | 周期性 `gc_expired_leases()` 真正被调用；`finish / abort / reject / timeout / reset` 五条 final 路径全部清理 consume 与 produce；`retract/preempt` 作为第六条**非终局** reset 路径释放本轮 C40 状态并允许后续重新 staging |
| ME-9 | **底座全复用 + 自包含执行环境** | cross-store allocator / budget / object graph / stale victim 容忍 / SWA metadata / provisional / fallback 记账**全部复用，不重写**；Docker 依赖锁（`requirements.lock` + 专用 layer）、central JSONL log、manifest / consolidator 自包含 |

### 6.3 Layer 3 — Conditional Lanes（9 个，尽可能实现，**不得阻塞 core**）

| Lane | 内容 | 阻塞 core？ | 前置 |
| --- | --- | --- | --- |
| CL-A | **多岛 non-overlap selection + total copy budget + payoff optimizer** | 否 | core 通过 G3 |
| CL-B | **optional leading-k repair**，复用现有 EPIC 能力，定义 `C40-R1-k`（`k ∈ {2,8,32}`） | 否 | core 通过 G6 |
| CL-C | **AST span / path dependency** 作为**辅助** invalidator 与 gate。embedding distance 仍保留且为主信号；AST **不替代** embedding，且 `C40_AST_GATE=1` 必须要求 `C40_EMBED_GATE=1`；AST 只能进一步收紧、不得放宽 | 否 | core 通过 G3 |
| CL-D | **host demand-load**（`H1`）；**prefetch-neutral hint 接口**；async prefetch 作为**正交后续** | 否 | core 通过 G6 |
| CL-E | **concurrency / multi-tenant 并发行为**（identity isolation 本身由 ME-2 保证，本 lane 验证其在**并发**下仍成立并度量并发开销） | 否 | core 通过 G3 **且** decision manifest 已切换为 `dir` 模式（§14.2b 规则 8） |
| CL-F | **source chaining max-depth diagnostic** | 否 | ME-7 已实现 |
| CL-G | **quality-calibrated gate**（logit / KL / top-k / task 信号） | 否 | G9 pilot 有数据后 |
| CL-I | **Exact-boundary clipping**（optional，`SGLANG_APPROX_KV_C40_EXACT_OVERLAP_CLIP`，默认 `0`）：B-2 情形下把 island 裁剪到 `[exact_length, target_end)` 后复用。模块 `coding_c40/overlap_clip.py`；测试 `test_c40_exact_overlap_clip.py`（§17.23）；实验 Stage E-4。开启时 outcome 必须带 `geometry = clipped_at_exact_boundary`，**不进入** primary headline | 否 | core 通过 G3 |
| CL-H | **Cross-version / source-dependency evolution experiment**（optional）：在**同一** repo identity 内沿 commit 序列演进，度量 source 随版本演化的失效时机、`w` 衰减曲线与依赖传播行为。**不重复** ME-2 的 identity isolation（那是 mandatory 正确性要求），本 lane 只做**演化行为的实验刻画** | 否 | ME-2 已实现且 W7 可得 |

**硬性规则**：任一 conditional lane 的实现或失败**不得**阻塞 Layer 1/2 的
Gate 推进；conditional lane 的实验点必须在 staged gate 中显式声明，
**不进入** primary headline。

### 6.4 三层的关系图

```text
        ┌───────────────────────────────────────────────────────┐
        │  Layer 3  Conditional Lanes (CL-A .. CL-I)            │
        │  多岛 / repair / AST / host / prefetch-hint /         │
        │  concurrency / chaining / quality gate /              │
        │  cross-version evolution / exact-overlap clip         │
        └───────────────────┬───────────────────────────────────┘
                            │ 依赖，但不阻塞
        ┌───────────────────▼───────────────────────────────────┐
        │  Layer 2  Mandatory Extensions (ME-1 .. ME-9)         │
        │  结构化 provenance / fingerprint / 状态机 /           │
        │  双角色生命周期 / approx depth / lease GC / 自包含    │
        └───────────────────┬───────────────────────────────────┘
                            │ 必须先成立
        ┌───────────────────▼───────────────────────────────────┐
        │  Layer 1  Parity Core (PC-1 .. PC-9)                  │
        │  真实 source / 只读证据 / token identity /            │
        │  strict-middle / V copy + K RoPE / dense / fail-closed│
        └───────────────────┬───────────────────────────────────┘
                            │ 全部构建在
        ┌───────────────────▼───────────────────────────────────┐
        │  既有底座（零重写）approx_kv + cross_store            │
        └───────────────────────────────────────────────────────┘
```

---

## 7. 架构与组件设计

### 7.1 建议模块树（`proposed`）

```text
python/sglang/srt/mem_cache/approx_kv/coding_c40/
├── __init__.py            公共导出；不导出任何内部可变状态
├── provenance.py          ToolEvent 解析、路径规范化、rename/symlink 建模、
│                          worktree generation、content hash 绑定、unknown fail-closed
├── types.py               C40 专属数据类型（frozen dataclass）：
│                          ToolEvent / RepoState / PathEffect / GroundedIsland /
│                          C40Candidate / C40Plan / C40Decision / C40Fingerprint /
│                          ConsumeState / ProduceState / C40Outcome
├── selector.py            G40 选择器：候选枚举 → 失效过滤 → token 识别 →
│                          唯一性 → strict-middle → min/cap → 单岛/多岛选择
├── optimizer.py           [CL-A] 多岛 non-overlap + total copy budget + payoff 打分
├── overlap_clip.py        [CL-I] B-2 exact-boundary clipping（default off）
├── controller.py          middle-span staging controller：状态机驱动、
│                          与 protect_request_prefix / provisional slot 协作
├── state.py               request-lifetime 与 per-round transient 状态容器；
│                          consume/produce 双角色；scheduler lock handoff 记录
├── stats.py               C40 telemetry：counters / gauges / histograms、
│                          exclusive terminal reason 记账、selector overhead 计时
├── adapter.py             与 xs: 底座的薄适配：构造 KVReusePlan、调用
│                          execute_reuse_plan、注册/查询 ApproxKVSegmentStore、
│                          approx provenance/depth 标注
├── gates.py               [CL-C][CL-G] 辅助 gate：AST span 依赖、embedding
│                          distance、quality-calibrated 阈值（默认全部关闭）
└── reason_inventory.py    §2.6 的 AST reason 扫描器（WP0b）
```

**实验面（benchmark）**：

```text
benchmark/approx_kv/coding_c40/
├── __init__.py
├── collectors/            provenance 采集器（authority + 独立 oracle 实现）
│   ├── wrapper_declared.py    结构化封闭工具的 read/write event 声明
│   ├── syscall_trace.py       默认 authority：strace/ptrace collector
│   ├── preload_shim.py        supplemental oracle：LD_PRELOAD（非 authority）
│   └── merkle_snapshot.py     secondary check：content hash / generation
├── live_corpus.py         W4a live trajectory corpus 生成与冻结（§18.4）
├── trajectory.py          冻结 trajectory 的加载/校验/重放（不产生 KV 决策）
├── plan_freeze.py         离线运行 selector，冻结 Elig 与 span 清单 + sha256
├── workloads_c40.py       §18 的 7 类 workload 构造器
└── consolidate_p75_results.py   离线 consolidator（只读 raw + central JSONL）

benchmark/approx_kv/
├── run_p75_selector_offline.py   [CPU] selector 离线/差分/property 驱动
├── run_p75_canary.py             [GPU] same-context canary + 张量校验
├── run_p75_micro.py              [GPU] cross-context 三臂 micro workload
├── run_p75_workflow.py           [GPU] Architect→Coder→Debugger + S0/S4
├── run_p75_quality.py            [GPU] RepoBench-P / SWE-bench harness 驱动
├── build_p75_manifest.py         pinned manifest 构建与 --check
└── build_p75_result_manifest.py  结果目录递归自哈希 manifest
```

**测试面**：

```text
test/registered/unit/mem_cache/
├── test_c40_cleanroom_compliance.py     §4.4
├── test_c40_provenance.py               §9 对抗矩阵 + property
├── test_c40_selector.py                 §6.1 PC-2..PC-5、PC-9
├── test_c40_identity_fingerprint.py     §10
├── test_c40_plan_coverage.py            PC-7 全覆盖/无重叠/无空洞
├── test_c40_controller_state.py         §8 状态机、ADMISSION_DEFERRED 与 lifecycle
├── test_c40_terminal_reasons.py         §12 四计数族与互斥性
├── test_c40_lease_soak.py               ME-8 soak
├── test_c40_optimizer.py                [CL-A]
├── test_c40_chunk_continuation.py       §8.6.5 TC-1..TC-99
├── test_c40_exact_overlap_clip.py       [CL-I] §17.23
└── test_c40_cuda.py                     [GPU] K/V + RoPE 张量（需 GPU 标记）

test/registered/unit/bench/
├── test_c40_manifest.py                 §14
├── test_c40_plan_freeze.py              冻结清单可复现性
└── test_c40_consolidator.py      consolidator
```

### 7.2 Data flow（端到端）

```text
[Agent / tool wrapper 侧 — 容器内]
  每条工具调用：
    结构化封闭工具 → wrapper 直接声明 read/write set（authority，§9.2）
    任意 shell/bash → **event-level collector（唯一 authority）**
                      默认 strace/ptrace；等效 authority 仅 fanotify / eBPF
                      LD_PRELOAD 只可作补充差分 oracle，不能作 authority
                      collector 不可用或不完整 ⇒ unknown_effect=true ⇒ ineligible
    Merkle snapshot / git status → **仅** final-state / generation / integrity
                      的 secondary check，**不产生** read_paths，
                      **不能**证明"无瞬时写"，**绝不**充当 authority
    → 生成 ToolEvent（结构化，含 read_paths/write_paths/unknown_effect/...）
                    │
                    ▼
  trajectory (ordered ToolEvent + message stream)
                    │
                    ▼
[C40 selector — provenance.py + selector.py]
  1. 解析 ToolEvent → RepoState 序列 + PathEffect 时间线
  2. 枚举 "successful read-only observation" 候选（PC-2）
  3. 结构化失效过滤（PC-5 + ME-1 + ME-2）：
       later write ∩ source_paths ≠ ∅        → invalid
       worktree_generation 变化且无逐路径 hash 证明 → invalid
       unknown_effect / timeout / truncated  → invalid
  4. token 化（只取 role=="tool" 正文；PC-3）
  5. 唯一出现 + min/cap + strict-middle（PC-4）
  6. 单岛（primary）或多岛 non-overlap + budget（CL-A）
  7. 输出 C40Decision（完整可审计 dict）+ 至多 N 个 GroundedIsland
                    │  只声明 what，不做 residency / eviction / priority
                    ▼
[C40 metadata → server]
  由 CLI/env 指定路径的 C40 **decision** manifest（c40_decision_manifest_version=1，§14.2b）
  授权判定另读 plan manifest（c40_manifest_version=1，§14.1）
  禁止任何 HTTP 字段选择 KV span
════════════════════ 进程边界 ════════════════════
[SGLang server 侧 — controller.py + adapter.py]
  **真实调度顺序（verified-code，见 §7.4）**：

  ── 新请求组批轮 ──────────────────────────────────────────────
  scheduler.py: req.init_next_round_input(self.tree_cache)
        │  内部先 release_provisional_recovery_slots，再 match_prefix
        │  C40 第三分支：stage_middle_span(tree_cache, req)  ← **只读判定**
        │  exact_length = len(req.prefix_indices)          ← 仅此一次读取
        │  按 §8.6.2 分类 B-1..B-4（B-1 走 exact；B-2 fail-closed）
        │  产出 request-lifetime plan：
        │    middle_cursor = exact_length
        │    borrowed_exact_indices =
        │      req.prefix_indices.to(torch.int64).clone().contiguous()
        │    owned_materialized_indices =
        │      empty contiguous torch.int64 tensor on the same device
        │    middle_cursor = exact_length
        │    target_start / target_end / rope_delta / source_key
        │    state = STAGED
        │  **不分配 slot、不驱逐、不 pin 目标侧资源**
        ▼
  scheduler.py: adder.add_one_req(req, ...)
        │  成功 → 常规 request lock 接管 prefix ownership（写 handoff 记录）
        │  容量不足 → ADMISSION_DEFERRED（仅保留source plan；
        │             target mapping清空，下轮重match）
        │  controller 在此提供 next_extend_boundary = target_start，
        │  使本轮 extend_range.end 恰好停在 target_start（§8.6）
        │  即使整个 prompt 小于 chunk_tokens_limit，只要 boundary < prompt_end，
        │  也必须强制登记为 middle chunk（new_chunked_req + inflight 记账），
        │  禁止把 boundary 当成 prefill 已完成
        ▼
  prepare_for_extend → forward（dense prefix 的一个 chunk）
        │
        ▼
  下一轮 get_next_batch_to_run → stash_chunked_request（scheduler.py:2745-2756）
    → c40_commit_enqueued_prefill(req, req_to_token_pool)      ← **必须新增**
        │  使用 prepare_for_extend 冻结的 round snapshot
        │    (prefix_len, extend_len, extend_start/end, mapping indices)
        │  追加 owned indices；middle_cursor := extend_end
        │  这是 overlap/no-overlap 都会在下一次 add_chunked_req **之前**
        │  执行的同步 scheduling seam
        │  仅当state==DENSE_PREFIX才比较target_start：
        │    小于 → 继续DENSE_PREFIX；等于 → COPY_READY；大于 → boundary_overrun
        │  state∈{DENSE_SUFFIX,DENSE_ISLAND_FALLBACK}只推进cursor，不做该比较
        ▼
  scheduler.process_batch_result（可能晚一轮）
    → c40_verify_prefill_result(batch_snapshot, result)        ← verification only
        使用 batch.extend_lens/prefix_lens/out_cache_loc 快照校验 transaction；
        **不得**读取已被下一轮覆盖的 req.extend_range，**不得**推进 cursor
        ▼
  ── chunked 续算轮（若 dense prefix 需要多个 chunk）────────────
  scheduler.py: self.chunked_req.init_next_round_input()   ← **无 tree_cache**
        │  因此**不能**在此做需要 tree_cache 的 C40 判定；
        │  request-lifetime plan 必须自持，不依赖 match_prefix 或 prefix_indices
        │  req_to_token_pool.alloc 复用同一 req_pool_idx ⇒ mapping 不丢失
        ▼
  adder.add_chunked_req(self.chunked_req)
        │  controller 同样提供 next_extend_boundary = target_start
        ▼
  （回到 forward → 下一轮 stash seam 提交 enqueued range）
  ──────────────────────────────────────────────────────────────

  COPY_READY:
    adapter.build_plan() → C40ExecutionPlan(
        base_plan=KVReusePlan(TransferSpan + DenseRange[]),
        dense_rules=request-scoped reason+interval rules)
        │  rope_delta = target_start - source_start
        │  require_full_coverage = True；不修改 frozen DenseRange（§7.3.1）
        ▼
    xs: execute_reuse_plan() → RadixKVTransferBackend.copy_and_rotate()
        │  V 原样搬运；K 按 rope_delta 旋转；部分拷贝/部分旋转 = 硬不变量错误
        ▼
    copy-boundary commit（§8.6.3 步骤 5）：
        req_to_token_pool.req_to_token[req.req_pool_idx,
                                       target_start:target_end] = island_indices
        owned_materialized_indices = torch.cat(..., island_indices).contiguous()
        middle_cursor := target_end
        （**不修改** req.prefix_indices）；state := DENSE_SUFFIX
        ▼
  DENSE_SUFFIX：下一次 extend 起点取 middle_cursor（== target_end）
        suffix forward attention 到
          req_to_token[req.req_pool_idx, 0:target_end]
        ▼
  cache_finished_req 等价 hook：
     1) consume cleanup（release lease、写 outcome）
     2) produce（按 approx_depth 规则物化下一 source；ME-7）
```

### 7.3 Ownership 边界（硬性）

| 组件 | 拥有 | **禁止** |
| --- | --- | --- |
| `provenance.py` | ToolEvent 解析与路径事实 | 任何 KV / token / 调度决策 |
| `selector.py` | 候选选择与 decision dict | 调用 `ensure_resident`；触发 eviction / priority / deadline；分配任何 slot |
| `optimizer.py` | 多岛组合与 budget 分配 | 直接改写 plan 之外的任何状态 |
| `controller.py` | 状态机推进、slot 预留与转正、异常清理 | 重新做选择决策；重写底座 allocator |
| `adapter.py` | 构造 plan、调用底座、标注 provenance/depth | 位置修正（**只在 transfer backend 内发生一次**） |
| `stats.py` | 记账 | 影响控制流 |
| 底座 `xs:` | 分配、驱逐、lease、记账、transfer | 被 C40 绕过或替换 |

**接线原则**：selector 只产生 plan；位置修正只在 transfer backend 内发生一次；
任何缺失 / 过期 / 不匹配 / 迟到 → fail closed 到 plan 已声明的 dense range。

#### 7.3.1 Dense range 执行合同（**必须显式约定，否则会重复计算**）

`verified-code`：`xs:approx_kv/transfer.py` 的 `execute_reuse_plan()` 在执行时刻
会**立即**对 `plan.dense_ranges` 的每一段调用 `backend.dense_prefill(...)`，
并把它们计入 `stats.recomputed_tokens`。但 C40 的几何要求 dense prefix 在 copy
**之前**已由正常 extend 路径算完、dense suffix 在 copy **之后**才算。
若不加约定，会出现"prefix 被算两次、suffix 被提前算"的严重错误。

`verified-code` 补充：底座 `DenseRange` 是 frozen dataclass，字段只有
`target_start / length / reason`。Phase7.5 **不修改**该共享类型，也不把
disposition 偷塞进 `reason`。

**冻结合同**：`coding_c40/adapter.py` 构造 request-scoped
`C40ExecutionPlan`，其中包含：

```text
base_plan: KVReusePlan
dense_rules: immutable ordered list[
  (range_start, range_end, accepted_reasons, disposition)
]
```

`execute_c40_plan()` 在调用底座 `execute_reuse_plan(base_plan, backend)` 前，
把rules安装到request-scoped context。底座真实回调签名是：
`backend.dense_prefill(target_start=..., length=..., reason=...)`。
adapter按**reason + 回调区间被某条rule完整包含**解析，而不是精确三元组：
底座会用`_contiguous_ranges`重新切分dense range，精确长度不稳定。

底座copy fallback reason必须预登记并直通唯一C40归因：

| base reason | C40 reason | disposition |
| --- | --- | --- |
| `stale_handle` | `c40_source_generation_stale` | `fallback_to_controller` |
| `residency_miss` | `c40_source_not_resident` | `fallback_to_controller` |
| `source_slice_mismatch` | `c40_token_slice_mismatch` | `fallback_to_controller` |

primary冻结`TransferSpan.chunk_start == span.target_start`且
`chunk_length == span.length`（page_size=1）；因此底座fallback chunk与island
rule完全同区间。任何更宽chunk只允许在独立conditional lane中实现并新增
containment rule/test，primary不得依赖`chunk ⊇ span`宽松语义。

未知reason、区间不被唯一rule包含或context未清空才fail closed；已知底座
fallback不得吞成`c40_copy_exception`。由此无需修改
`mem_cache/approx_kv/types.py`。

允许的 disposition 与行为：

| `disposition` | 含义 | `dense_prefill` 行为 |
| --- | --- | --- |
| `already_materialized` | dense prefix，已在 `DENSE_PREFIX` 阶段由正常 extend 算完并写入 `req_to_token` | **只记账**（覆盖簿记 + telemetry），**不触发任何计算** |
| `deferred_to_suffix` | dense suffix，将在 `DENSE_SUFFIX` 阶段由正常 extend 计算 | **只记账**，**不触发任何计算**；由状态机保证其后必然被计算 |
| `fallback_to_controller` | 底座copy validation/freshness fallback | 记录唯一reason并转`DENSE_ISLAND_FALLBACK`；backend不自行计算，正常extend重算 |

coverage rules必须覆盖完整`target_token_ids`：
- `[0, exact_length)` borrowed exact：`already_materialized`，
  计`accounted_exact_tokens`，**不**计accounted_dense/forwarded_dense；
- `[exact_length, target_start)` owned dense prefix：`already_materialized`；
- island：copied span或底座fallback rule；
- `[target_end, prompt_end)`：`deferred_to_suffix`。
构造`DenseRange`/rule时任何`length==0`区间必须**直接省略**，不得实例化
（底座`DenseRange`拒绝非正长度）；B-3的owned-dense-prefix为空、冷cache时
borrowed-exact为空都属于合法省略。
否则`require_full_coverage/mechanically_valid`必然失败。

**断言（CPU 测试必须覆盖）**：

```text
A1  任一 c40_copied* 请求中，dense prefix 区间的实际 forward 次数 == 1
A2  dense suffix 区间在 copy 执行时刻的实际 forward 次数 == 0
A3  require_full_coverage=True 仍成立（覆盖簿记不因"只记账"而缺口）
A4  分开报告accounted_exact_tokens、accounted_dense_tokens与
    forwarded_dense_tokens；borrowed exact不得混入dense，禁止把底座
    stats.recomputed_tokens直接当真实计算量
```

### 7.4 Scheduler hook 位置（全部为**当前底座既有 seam**）

> **真实调用顺序（`verified-code` @ `0206f17b`，必须按此设计，不得凭直觉假设）**
>
> | 位置 | 事实 |
> | --- | --- |
> | `xs:managers/scheduler.py:3017` | 新请求：`req.init_next_round_input(self.tree_cache)` —— **带 tree_cache**，内部先 `release_provisional_recovery_slots` 再 `match_prefix` |
> | `xs:managers/scheduler.py:3018` | **随后**才 `adder.add_one_req(req, ...)` —— 即 `match_prefix` 在 `add_one_req` **之前** |
> | `xs:managers/scheduler.py:2967` | chunked 续算：`self.chunked_req.init_next_round_input()` —— **不带 tree_cache** |
> | `xs:managers/scheduler.py:2968` | `adder.add_chunked_req(self.chunked_req)` |
> | `xs:managers/schedule_policy.py:830-860` | `add_chunked_req` 内 `req.set_extend_range(len(prefix_indices), len(prefix_indices) + new_len)`，`new_len = min(cand_extend_input_len, _rem_tokens)` |
> | `xs:managers/schedule_batch.py:1184` | `Req.set_extend_range(start, end)` 是 chunk 边界的唯一写入点 |
> | `xs:managers/scheduler.py:2745-2756` | `stash_chunked_request` 只在 `extend_range.end > len(prefix_indices)` 时执行 |
> | `xs:managers/schedule_batch.py:1080` | `skip_radix_cache_insert = (... or self.approx_kv_metadata is not None)` |
>
> **由此得出的四条设计约束（冻结）**：
>
> 1. `stage_middle_span` 发生在 `init_next_round_input(tree_cache)` 内，此时
>    **尚未** admit，因此该阶段**只能做只读判定与簿记**，**不得**分配 target
>    slot、不得驱逐、不得 pin 目标侧资源；
> 2. `protect_request_prefix` 是**会退出的 context manager**
>    （`xs:approx_kv/runtime.py`），在 `init_next_round_input` 返回前必然退出，
>    **不能**跨调度轮持有。跨轮 prefix ownership 由 `add_one_req` 成功后
>    scheduler 的**常规 request lock** 承担（§8.5 handoff）；
> 3. **chunked 续算轮拿不到 `tree_cache`**，因此 C40 的 request-lifetime plan
>    必须自持；overlap scheduler 又会先调度下一轮、后处理上一轮 result，
>    所以 cursor 的权威推进必须接在 `stash_chunked_request` 的同步 seam，
>    不能依赖 `process_batch_result`；
> 4. 底座已有`skip_radix_cache_insert`消费seam，但C40不依赖HTTP
>    `approx_kv_metadata`自动置位；C40在**成功admission commit**时显式置True。
>    该请求中间chunk不会stash到tree_cache，但KV slot由
>    **`req_to_token_pool` 的 per-request mapping**
>    （`req_to_token[req.req_pool_idx, :]`）在整个请求生命周期持续持有，
>    且 `req_to_token_pool.alloc` 对已有 `req_pool_idx` 的请求**复用同一 slot**，
>    因此 chunked 续算不会丢失 mapping。
>    `[dense prefix + copied island]` 的可达性来自 **pool mapping**，
>    **不**来自 Radix，也**不**来自 `req.prefix_indices`（§8.6.1）。

| Hook | 类型 | `xs:` 位置 | C40 行为 |
| --- | --- | --- | --- |
| `init_next_round_input` 入口 | **既有** | `schedule_batch.py:1211`（已调 `release_provisional_recovery_slots`） | **只**释放上一轮未转正的 provisional / transient；**保留** request-lifetime consume/produce 状态、source lease、`middle_cursor`。若 rematch 使冻结 plan 失效，记录 exclusive reason 后才 `→ DENSE_ISLAND_FALLBACK` 并清理 |
| `match_prefix` 之后、恢复之前 | **既有**（新增分支） | `schedule_batch.py` approx恢复分支旁 | `elif config.c40_enabled: stage_middle_span(...)`；只读创建staging/source plan，**此时不改skip flag、不标lifecycle entered** |
| admission / 常规锁获取 | **既有** | `scheduler.py:3018` → `schedule_policy.add_one_req` | 成功admit完成lock handoff；capacity deferral只保留source-side plan，清空target borrowed/effective/cursor，下轮重新match/rebuild；无slot/lease，不转terminal |
| **chunk 边界钳制（新请求）** | **需修改既有** | `schedule_policy.add_one_req`（`set_extend_range` 调用点） | 若 `c40_state == DENSE_PREFIX` 且 `target_start > middle_cursor`（**用 cursor，不用 `len(prefix_indices)`**），则 `extend_end := min(extend_end, controller.next_extend_boundary(req))`，其中 `next_extend_boundary = target_start`（§8.6.3 步骤 3） |
| **forced-middle 标记（新请求）** | **需修改既有** | `schedule_policy.add_one_req` 非 chunked / chunked 分支汇合前 | 若 C40 边界使 `extend_end < len(full_untruncated_fill_ids)`，则本轮必须按 chunked-prefill 语义提交：`new_chunked_req = req`、预算中的 `max_new_tokens = 0`，并由 scheduler 使 `inflight_middle_chunks += 1`。该规则即使原始 `input_tokens <= chunk_tokens_limit` 也成立 |
| **单 chunked-owner 互斥** | **需修改既有** | `schedule_policy.add_one_req` forced-middle preflight | 新请求发现已有owner时保持底座`OTHER`/not-added路径并defer；当前add_chunked owner绕过。首个owner耗尽本轮chunk/input scheduler budget，后续请求不得覆盖 |
| **chunk 边界钳制（续算）** | **需修改既有** | `schedule_policy.add_chunked_req:830-868`（`set_extend_range` 调用点） | 同上；`truncated` 必须由**钳制后的** `extend_end < len(full_untruncated_fill_ids)` 计算，而不是在钳制前计算。钳制后仍有 suffix 时必须返回原 `req`，使 `self.chunked_req` 与 `inflight_middle_chunks` 生命周期保持 |
| **extend 起点改用 cursor** | **需修改既有** | 同上两处 | `DENSE_SUFFIX` / `DENSE_ISLAND_FALLBACK` 下 `set_extend_range` 的**起点取 `middle_cursor`**，不取 `len(req.prefix_indices)`（后者不反映本请求已物化的 dense prefix 与 copied island） |
| **round snapshot / enqueue mark / commit** | **必须新增** | `prepare_for_extend`冻结snapshot；`run_batch`成功dispatch后标`forward_enqueued`; `stash_chunked_request`提交 | 只有enqueued snapshot可推进cursor；enqueue前失败只清snapshot并由底座释放；nonfinal每transaction恰好stash-commit一次 |
| **`c40_verify_prefill_result(batch_snapshot,result)`** | **必须新增** | `scheduler.process_batch_result:3602`，委派processor前 | 非final snapshot只验证；若`is_final && !committed`，可一次性terminal-commit（已离开chunked调度，不影响下一轮range）；禁止读取共享`req.extend_range`。teardown另有幂等兜底 |
| **copy-boundary commit** | **必须新增** | B-3 `add_one_req` / B-4 `add_chunked_req` pre-range hook | copy前gate显式含island allocation；真实allocator分配后live available_size已自然下降，**不得再加offset charge**；失败free后自然回补，再以dense remaining重跑gate |
| **retract/preempt reset** | **需修改既有** | `schedule_batch.release_req(...)` 在 `release_kv_cache` 前 + `Req.reset_for_retract()` | 入口先判`req.to_finish/finished`：为真则归final abort，不计retract；否则仅当`lifecycle_live`时执行retract reset。reset清live state但保留sticky `lifecycle_ever_entered`到最终outcome审计 |
| 所有权转正 | **既有语义先例** | `commit_provisional_recovery_slots` | 由上一行的 copy-boundary commit 复用其记账语义 |
| 拒绝路径（terminal） | **既有** | `scheduler.py` reject | `→ TERMINAL_REJECTED`，释放 slot 与 lease |
| 中止路径 | **既有** | `scheduler.py` abort | `→ ABORTED_TERMINAL` |
| timeout 路径 | **既有** | `scheduler.py` 的 `_abort_on_waiting_timeout` / `_abort_on_running_timeout` | `→ ABORTED_TERMINAL` |
| reset 路径 | **既有** | flush / reset 入口 | `→ ABORTED_TERMINAL` + store reset |
| 通用 teardown | **既有** | `common.py` `release_kv_cache` | 兜底清理 |
| 请求结束 / source 物化 | **既有** | RadixCache `cache_finished_req` | 先闭合 consume，再按 provenance/approx_depth 规则处理 produce |

### 7.5 必须保留的既有不变量（新代码不得绕过）

| 不变量 | 对 C40 的要求 |
| --- | --- |
| prefix ownership | staging（`init_next_round_input` 内，在 `add_one_req` **之前**）只做只读判定，不分配 / 不驱逐 / 不 pin；admit 后由 scheduler 常规 request lock 覆盖 `DENSE_PREFIX → COPY_READY → DENSE_SUFFIX`。`protect_request_prefix` 是会退出的 context manager，**不得**用作跨轮保护。若存在任何 pre-admission 驱逐动作，须用显式 guard lease 保护并在常规 lock 获取后做**可审计 handoff**；释放参数必须保留 `to_dec_params()` 的 SWA metadata |
| provisional ownership | copy 目标 slot 必须走 provisional 分配并在 commit 时进入底座 `kv_committed_len/kv_allocated_len` ledger；五条final cleanup与retract reset（§8.1.2）统一调用底座release，使owned free once / borrowed unlock only |
| stale victim 容忍 | 腾空间必须复用现有 allocator，**不得自建驱逐循环** |
| exclusive fallback / 不双计 | `c40_*` reason 与 canonical inventory（§2.6）互斥；`Σ_reason tokens == attempted_recovery_failed_dense_tokens`（族 2，§12.4） |
| object graph 无孤儿 | source 段与其 host copy / 依赖必须注册进图，驱逐走 `remove_closure` |
| exact 纯净性 | 近似copy成功的请求禁止写exact Radix；从未copy或copy完整rollback后全dense完成的请求在prefill完成时恢复原始insert策略。C40关闭与B-0/B-1/ineligible路径逐位一致 |

### 7.6 exact / approx store 与 source depth

```text
                ┌──────────────────────────────────────┐
                │  exact Radix (tree_cache)            │
                │  只接受 dense/exact 计算得到的 KV     │
                │  approx_depth == 0                   │
                └───────────┬──────────────────────────┘
                            │ 只允许 depth-0 请求物化 primary source
                            ▼
                ┌──────────────────────────────────────┐
                │  ApproxKVSegmentStore (approx)       │
                │  record.provenance ∈ {EXACT, APPROX} │
                │  record.approx_depth ∈ {0,1,2,...}   │
                └───────────┬──────────────────────────┘
                            │
        primary lane        │        diagnostic lane (CL-F)
        source.depth == 0   │        source.depth >= 1
        ────────────────────┼────────────────────────────
        允许被 C40 消费     │  仅在显式开启 chaining 时允许，
                            │  冻结 max_depth，逐层报告质量衰减，
                            │  且**不进入** primary headline
```

**硬性规则**：

1. `provenance == APPROXIMATE` 的段**禁止**写入 exact Radix；
2. primary C40 qualification 只接受 `approx_depth == 0` 的 source；
3. 任何请求若使用过 approximate copy，其物化的下一 source 必须标
   `provenance=APPROXIMATE` 且 `approx_depth = source.approx_depth + 1`；
4. chaining 只作独立 diagnostic 轴，默认 `max_chain_depth = 0`（关闭）。

---

## 8. Middle-Span 状态机与请求生命周期

### 8.1 状态机

```text
        ┌────────────────────────────────────────────────────────────┐
        │  INIT                                                      │
        │  match_prefix 完成；C40 metadata 已解析并通过 fingerprint   │
        │  校验；尚未分配任何 slot                                    │
        └───────────────┬────────────────────────────────────────────┘
                        │ selector 判定 eligible 且 span 严格中部
                        │ 且 source handle current 且 generation 一致
                        ▼
        ┌────────────────────────────────────────────────────────────┐
        │  DENSE_PREFIX                                              │
        │  [exact_length, target_start) 交给正常 extend 路径计算       │
        │  期间：由 scheduler 常规 request lock 保护（**不是**已退出的  │
        │        protect_request_prefix context）；source handle 已 pin │
        │  可跨多个 chunk / 多个调度轮；request-lifetime 状态保留       │
        │  失败/rematch 失效 → DENSE_ISLAND_FALLBACK；               │
        │  scheduler 中止 → ABORTED_TERMINAL                          │
        └───────────────┬────────────────────────────────────────────┘
                        │ dense prefix 已写入 req_to_token 且位置对齐
                        ▼
        ┌────────────────────────────────────────────────────────────┐
        │  COPY_READY                                                │
        │  校验：handle current / generation 一致 / token slice 逐元素相等│
        │        / source device-resident / 目标槽 provisional 已分配   │
        │        / fingerprint 一致 / approx_depth == 0（primary）      │
        │  执行：adapter.build_plan() → execute_reuse_plan()            │
        │  任一校验或执行失败 → DENSE_ISLAND_FALLBACK                   │
        │       （唯一 c40_* terminal reason；请求**不失败**）           │
        └───────────────┬────────────────────────────────────────────┘
                        │ copied_k == copied_v == length
                        │ 且 stats.mechanically_valid
                        │ 且 copied slot 已写入 req_to_token 并所有权转正
                        ▼
        ┌────────────────────────────────────────────────────────────┐
        │  DENSE_SUFFIX                                              │
        │  [target_start+length, len(prompt)) 正常 extend             │
        │  suffix forward 可 attention 到 [dense prefix + copied island]│
        └───────────────┬────────────────────────────────────────────┘
                        ▼
        ┌────────────────────────────────────────────────────────────┐
        │  COMMIT                                                    │
        │  释放 consume lease；provisional 槽转正；写 outcome/stats；   │
        │  再按 approx_depth 规则处理 produce                          │
        └────────────────────────────────────────────────────────────┘

        ┌────────────────────────────────────────────────────────────┐
        │  DENSE_ISLAND_FALLBACK                                     │
        │  copy 校验/执行失败、或 pre-copy 阶段失效时的**受控回退**     │
        │  （不是请求中止）：                                          │
        │    1. release_provisional_recovery_slots（copy 目标槽）      │
        │    2. unpin(consume lease)                                  │
        │    3. 记唯一 c40_* terminal reason                          │
        │    4. **[middle_cursor, len(prompt)) 全部交给正常 extend**    │
        │       middle_cursor = 已真实写入 req_to_token 的最大连续位置：│
        │         INIT 时           = exact_length                     │
        │         DENSE_PREFIX 途中 = 已完成的 prefix 末端             │
        │         COPY_READY 失败时 = target_start                     │
        │       **禁止**固定从 target_start 重算 —— 那会在             │
        │       [middle_cursor, target_start) 留下 KV 空洞。            │
        │  → 与 DENSE_SUFFIX 汇合 → COMMIT（请求正常完成）             │
        └────────────────────────────────────────────────────────────┘

  STAGED --(add_one_req 容量不足)--> ADMISSION_DEFERRED   ← **非 terminal**
      ADMISSION_DEFERRED:
        只保留source-side plan（source_key/generation/target span/rope_delta）
        清空target-side exact_length/borrowed/owned/effective tensor/cursor/classification
        **无** allocation、**无** provisional slot、**无** consume lease
        不记 terminal reason；下一轮重新match_prefix并从头构造target-side staging
        （generation / content hash 可能已变 ⇒ 那时才可能转 DENSE_ISLAND_FALLBACK）

  任意状态 --(scheduler terminal reject)--> TERMINAL_REJECTED
  任意状态 --(abort / reset / timeout)--> ABORTED_TERMINAL
  COMMIT/任一活跃状态 --(retract / preempt)--> NONE
      c40_prepare_retract → 底座 release → c40_reset_after_retract
      **不产生 outcome**；请求回 waiting queue，下一轮从 staging 重建
  任意状态 --(rematch-invalid / C40 内部 exception)--> DENSE_ISLAND_FALLBACK
      ABORTED_TERMINAL / TERMINAL_REJECTED:
        release_provisional_recovery_slots + unpin(consume lease)
        + 丢弃 pending produce + 唯一 c40_* 归因
        + **该请求由 scheduler 终止**，C40 只负责资源清理与归因，
          **不得**声称请求会继续执行
```

#### 8.1.1 `ADMISSION_DEFERRED` vs `TERMINAL_REJECTED`（**必须区分**）

| 项 | `ADMISSION_DEFERRED` | `TERMINAL_REJECTED` |
| --- | --- | --- |
| 触发 | `add_one_req` 因 **capacity/budget** 未接纳（`AddReqResult.NO_TOKEN` 等），请求仍在 waiting queue | scheduler 对该请求作出**终局拒绝**（参数非法、超限、显式 reject 路径） |
| 是否 terminal | **否** | **是** |
| request-lifetime plan | 只保留source-side immutable plan（source key/generation/target span/rope delta）；target-side exact/borrowed/cursor每轮重建 | 清除 |
| allocation / provisional slot | **无**（本来就没分配） | 释放（若有） |
| consume lease | **无** | 释放 |
| 是否记 terminal reason | **否**（只记 `c40_admission_deferred_total` 计数器） | 是（`c40_terminal_rejected_requests_total`） |
| 请求 outcome | 不产生 outcome（请求尚未完成） | `terminal_rejected` |
| 下一步 | 下一轮重新match_prefix、重建exact/borrowed/effective tensor/B-0..B-4几何，再校验source | 请求结束 |

> **硬性规则**：`add_one_req` 的 capacity deferral **不得**转
> `ABORTED_TERMINAL`，也**不得**记任何 fallback token reason。把排队重试
> 当成失败会同时污染 fallback 恒等式与 outcome 计数。
> 任一**pre-admission**路径转`DENSE_ISLAND_FALLBACK`或`dense_ineligible`
> 时也必须调用`clear_target_staging()`，与ADMISSION_DEFERRED相同地清空
> exact/borrowed/owned/effective/cursor；helper在`req_pool_idx is None`时
> 强制退化为当前`req.prefix_indices`，禁止陈旧target tensor参与allocation。
> 若该请求**同轮仍要以dense被admit**，`clear_target_staging()`还必须设置
> `c40_state=NONE`（fallback reason/outcome留在独立audit容器），使后续
> allocation/extend完全走当前exact prefix；不得以空cursor继续C40状态机。

#### 8.1.2 五条 final cleanup + 一条 retraction reset 的真实 hook

| 路径 | hook 类型 | `xs:` 位置 | C40 目标状态 | 验收断言 |
| --- | --- | --- | --- | --- |
| **finish** | **既有** | RadixCache `cache_finished_req` | `COMMIT` | consume lease 已释放；produce 按 depth 规则处理；`c40_active_leases` 减 1；`c40_cleanup_total{path="finish"}` +1 |
| **abort** | **既有** | `scheduler.py` abort 入口 | `ABORTED_TERMINAL` | provisional/lease 归零；`c40_aborted_requests_total{path="abort"}` +1；`c40_cleanup_total{path="abort"}` +1 |
| **pending chunked abort** | **既有** | `scheduler.process_pending_chunked_abort:2646-2673` → direct `release_kv_cache` | `ABORTED_TERMINAL` | `prepare_abort`后即使`to_finish=None`，以`req.finished()==true`识别abort；必须走同一abort final cleanup，禁止误记retract或漏sticky audit |
| **terminal reject** | **既有** | `scheduler.py` reject 入口 | `TERMINAL_REJECTED` | provisional/lease 归零；`c40_terminal_rejected_requests_total` +1；`c40_cleanup_total{path="terminal_reject"}` +1 |
| **timeout** | **既有** | `scheduler.py` `_abort_on_waiting_timeout` / `_abort_on_running_timeout` | `ABORTED_TERMINAL` | provisional/lease 归零；`c40_aborted_requests_total{path="timeout"}` +1；`c40_cleanup_total{path="timeout"}` +1 |
| **reset / flush** | **既有** | flush/reset 入口 + `ApproxKVSegmentStore.reset()` | `ABORTED_TERMINAL` + store reset | `c40_aborted_requests_total{path="reset"}` +1；`c40_cleanup_total{path="reset"}` +1；全部 gauge 归零；`record_count`/`lease_count`/`orphan_count` == 0 |
| **retract / preempt** | **既有** | `schedule_batch.release_req`入口 | 若`req.to_finish is None && !req.finished()`：`NONE`并重排队；否则归上表abort final path | 纯retract才计`path="retract"`且无outcome；OOM“abort last request”必须在reset前计`path="abort"`，sticky audit保留到aborted outcome写出 |
| （兜底） | **既有** | `common.py` `release_kv_cache` | 依上下文 | 不产生重复计数（与上表任一路径互斥） |
| （**非清理路径**） | **需新增** | `schedule_policy.add_one_req` 容量不足分支（返回 `AddReqResult.NO_TOKEN` 等） | `ADMISSION_DEFERRED` | **不**触发任何 cleanup 计数；request-lifetime staging/plan **完整保留**；`provisional_slots == 0`；`consume_lease is None`；只递增 `c40_admission_deferred_total`；下轮重新验证 generation/fingerprint |

**统一计数规则**：

```text
c40_cleanup_total 的final path只对sticky
`req.c40_audit.lifecycle_ever_entered == true`计数；
retract path只对当时`lifecycle_live == true`计数。
C40关闭、B-0/B-1、dense_ineligible 等从未进入C40 lifecycle的请求不写该metric。

c40_cleanup_total{path} 的 path 域 ==
  {finish, abort, terminal_reject, timeout, reset, retract}
  final path 集合 = {finish, abort, terminal_reject, timeout, reset}
  retract 是可重复的非终局事件；同一请求可经历 0..N 次 retract，之后再走
  恰好一条 final path。
  final outcome与cleanup metric写出后才清
  `req.c40_audit.lifecycle_ever_entered`；pure retract不得清。
  若 cleanup 本身失败：
    c40_cleanup_total 仍按进入时冻结的 original path 计一次；
    只另记 c40_invalid_engineering_total{kind="cleanup_failed"}；
    `cleanup_failed` **不是** request-outcome path，整块 engineering invalid，
    不参与 outcome 恒等式。
Σ_{path ∈ final paths} c40_cleanup_total{path}
    == count(final outcomes where req.c40_audit.lifecycle_ever_entered == true)
       # 条件：cleanup_failed == 0
c40_cleanup_total{path="retract"} == c40_retraction_events_total
ADMISSION_DEFERRED **不**计入 c40_cleanup_total（它不是 cleanup 事件）
```

**两类失败必须严格区分（authority §10.3.1 的 `DENSE_FALLBACK` 语义）**：

| 事件 | 状态 | 请求结局 |
| --- | --- | --- |
| copy 校验失败 / 机械校验失败 / source 不可用 / plan 失效 | `DENSE_ISLAND_FALLBACK` | **请求正常完成**（全 dense 路径） |
| abort / reset / timeout | `ABORTED_TERMINAL` | 请求按 scheduler 既有语义终止 |
| scheduler **terminal reject** | `TERMINAL_REJECTED` | 同上（与 `ADMISSION_DEFERRED` 严格区分，§8.1.1） |
| C40 代码路径抛出未预期异常（含 copy） | 统一规则：先记 `c40_copy_exception` / `c40_internal_exception` 并完成资源清理 → `DENSE_ISLAND_FALLBACK`，请求以全 dense 正常完成。**只有**在"清理本身失败"或"底座不变量已被破坏（例如已发生部分拷贝且无法回滚）"时才 re-raise；此时记族4 `cleanup_failed` 并使整块engineering invalid，禁止把该无效块用于稳定的族3/outcome恒等式 | 前者正常完成；后者由 scheduler 终止且整块作废 |

### 8.2 状态转移矩阵（用于 property 测试）

| From \ Event | `admit_ok` | `admit_deferred` | `prefix_done` | `copy_ok` | `copy_fail` | `suffix_done` | `retract` | `abort`/`reset`/`timeout` | `terminal_reject` | `rematch_invalid` | `exception` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `STAGED` | `DENSE_PREFIX` | `ADMISSION_DEFERRED` | — | — | — | — | `NONE` | `ABORTED_TERMINAL` | `TERMINAL_REJECTED` | `NONE`（audit fallback/ineligible，纯dense） | `NONE`（pre-admission exception清target） |
| `ADMISSION_DEFERRED` | `DENSE_PREFIX` | `ADMISSION_DEFERRED` | — | — | — | — | `NONE` | `ABORTED_TERMINAL` | `TERMINAL_REJECTED` | `NONE`（清target后纯dense） | `NONE` |
| `DENSE_PREFIX` | — | — | `COPY_READY` | — | — | — | `NONE` | `ABORTED_TERMINAL` | `TERMINAL_REJECTED` | `DENSE_ISLAND_FALLBACK` | `DENSE_ISLAND_FALLBACK` |
| `COPY_READY` | — | — | — | `DENSE_SUFFIX` | `DENSE_ISLAND_FALLBACK` | — | `NONE` | `ABORTED_TERMINAL` | `TERMINAL_REJECTED` | `DENSE_ISLAND_FALLBACK` | `DENSE_ISLAND_FALLBACK` |
| `DENSE_SUFFIX` | — | — | — | — | — | `COMMIT` | `NONE` | `ABORTED_TERMINAL` | `TERMINAL_REJECTED` | — | `ABORTED_TERMINAL` |
| `DENSE_ISLAND_FALLBACK` | — | `ADMISSION_DEFERRED`（仅B-3 pre-admission regate失败，清target staging） | — | — | — | `COMMIT` | `NONE` | `ABORTED_TERMINAL` | `TERMINAL_REJECTED` | — | `ABORTED_TERMINAL` |
| `COMMIT` | — | — | — | — | — | — | `NONE` | `ABORTED_TERMINAL` | `TERMINAL_REJECTED` | — | — |
| `NONE` | staging 后进入 `STAGED` | — | — | — | — | — | — | — | — | — | — |
| `ABORTED_TERMINAL` | — | — | — | — | — | — | — | — | — | — | — |
| `TERMINAL_REJECTED` | — | — | — | — | — | — | — | — | — | — | — |

**不变量**：

- primary中既有chunked owner**每轮必须加入can_run_list**；不支持parked
  continuation。仅当`self.chunked_req.c40.lifecycle_live`时，
  scheduler在`inflight_middle_chunks += 1`前断言`adder.contains(owner)`；
  非C40 hybrid-SWA合法parked路径保持baseline；C40 prefill结束计数恰回0；

- `DENSE_PREFIX` / `COPY_READY` / `DENSE_SUFFIX`（**活跃复用状态**）必然持有
  provisional slot 或 consume lease；`STAGED` / `ADMISSION_DEFERRED` / `COMMIT` /
  `ABORTED_TERMINAL` / `TERMINAL_REJECTED` / `DENSE_ISLAND_FALLBACK` 则
  **必然不持有**（`STAGED` / `ADMISSION_DEFERRED` 是未分配状态，其余三者是清理后状态）；
- `ADMISSION_DEFERRED` 可连续停留任意轮次，期间 `provisional_slots == 0`、
  `consume_lease is None`；source-side plan保持，target-side borrowed/cursor
  必须为空并在每轮staging重建；
- `ABORTED_TERMINAL` 后 `provisional_slots == 0` 且 `consume_lease is None`；
- `DENSE_ISLAND_FALLBACK` 是**进入即完成清理**的状态：转入时同步释放 copy 目标槽
  与 consume lease，因此其后同样 `provisional_slots == 0` 且
  `consume_lease is None`；输出与final insertion eligibility在full-dense后
  与never-eligible一致；中间stash被抑制会改变Radix tree shape/时序，
  不要求逐字段一致，必须单独度量；
- `COMMIT` 后 `provisional_slots == 0` 且所有权已转正；
- `retract` 不是 terminal：hook 顺序必须为
  `c40_prepare_retract → release_kv_cache(is_insert=False) → reset_for_retract
  → c40_reset_after_retract`；完成后 state=`NONE`、所有 C40 tensor/lease/plan
  清空、`suppress_produce_once` 被消费，下一轮不得读取旧 cursor/index；
- 若`release_req`入口已存在`req.to_finish`或`req.finished()`，该事件不是
  pure retract：直接按abort final path记账，禁止先计retract再清sticky audit；
- 任一请求最多一次进入 `COPY_READY`（primary 单岛）；多岛 lane 下按岛计数并保持 non-overlap。

### 8.2.1 exact Radix 纯净性的实现机制（`verified-code` 的既有 seam）

底座已存在 `req.skip_radix_cache_insert`消费seam
（`xs:mem_cache/common.py:104,200`）。C40不依赖HTTP metadata自动值，而在
成功admission后显式控制：

```text
1. staging不改flag；成功admission commit时保存original并置True。
2. 中间chunk恒True，防`cache_unfinished_req`改写target ledger。
3. approximate copy成功commit ⇒ final release仍True，禁止写exact Radix。
4. 从未成功copy且全dense fallback完整prefill ⇒ 只在prefill完成后恢复original；
   final内容可正常insert，但中间stash缺失可能使tree shape不同。
5. pre-admission失败/state NONE立即恢复original；pure retract在release后恢复。
6. frozen arm policy：
   - D0/E0/C40-D：normal exact insertion；
   - C40-1R0与R0 span-matched：仅`copy_committed`请求抑制final insertion；
   - dense_ineligible与fully-dense fallback：恢复normal insertion。
   该差异是mechanism/system effect，进入end-to-end结果，不得隐藏。
```

### 8.3 request-lifetime vs per-round transient

```text
request-lifetime（跨 chunk / 跨调度轮保留，只在 terminal 或 rematch-invalid 时清理）：
  consume_state:
    source_handle, source_lease, source_key, source_generation,
    target_span(source_start, target_start, length, rope_delta),
    fsm_stage, middle_cursor, terminal_reason
  produce_state:
    candidate_span, output_key, source_provenance, approx_depth,
    pending_materialization

per-round transient（每轮 init_next_round_input 清理）：
  provisional_indices（本轮未转正）
  临时 allocator reservation
  一次性 copy transaction 句柄
```

**硬性规则**：`init_next_round_input` **只能**清理 per-round transient。
它**不能**清空 request-lifetime 状态或 source lease，否则多轮 `DENSE_PREFIX`
尚未到达 `COPY_READY` 就会丢失计划（会表现为"永远 dense"的静默失败）。
唯一额外清空入口是显式 `retract/preempt` 生命周期：
`reset_for_retract()` 必须调用 `c40_reset_after_retract`，因为底座已释放原 KV，
保留旧 cursor/index 反而会引用已 free slot。

### 8.4 consume / produce 双角色

```text
一个 rolling 请求通常同时是：
  consume : 消费上一请求注册的 source，执行本次 middle-span reuse
  produce : 本请求结束后，从当前 prompt/KV 物化下一请求可能使用的 source

因此 metadata 不能是互斥的 REGISTER / REUSE 单枚举。

请求结束顺序（冻结）：
  1. consume cleanup：release consume lease，写 consume outcome
  2. produce 判定：
       if 本请求使用过 approximate copy:
            new_source.provenance = APPROXIMATE
            new_source.approx_depth = consume.source.approx_depth + 1
            禁止写入 exact Radix
       if primary lane 且 new_source.approx_depth > 0:
            不物化（记 c40_produce_skipped_approx_depth）
  3. finish / abort / reject / timeout / reset 必须**同时**清理两套状态；
     retract/preempt 必须取消本轮 produce、释放 consume，并在底座释放后把
     两套状态重置为 NONE（但请求本身继续，后续可重新 staging）：
       不得因 source materialization 成功而漏掉 target lease
       不得因 target fallback 而留下 pending source
```

### 8.5 Scheduler lock handoff（可审计）

```text
阶段 1  pre-admission（发生在 init_next_round_input 内，match_prefix 之后）
        只解析 / 校验 metadata 与 fingerprint、只读地确定候选 span
        **不分配** target slot、不驱逐、不 pin 目标侧资源
        本阶段所在的 protect_request_prefix context 会在返回前退出，
        因此**不得**把它当作跨轮保护
阶段 2  add_one_req 成功（在阶段 1 之后）
        add_one_req 的 `_lock_node(req.last_node)` 临时 admission lock
        覆盖最终容量复核与 c40_on_admission_commit；随后
        `_req_inc_lock_ref(req)` 的常规 request lock 在临时 lock 退出前接管，
        不允许出现未保护窗口
        写入 handoff 记录：
          {"stage":"lock_handoff","req_id":...,"acquired":true,
           "guard_lease_used":false,
           "staged_span":{...},"staged_at_round":n,"ts":...}
        admit 失败（容量不足）⇒ ADMISSION_DEFERRED：
          保留source-side immutable plan；清空target-side
          exact_length/borrowed/owned/effective tensor/middle_cursor/classification
          无 slot、无 lease、不记 terminal reason
          下一轮重新match_prefix并重建target geometry，再验证source：
            通过 ⇒ 使用新borrowed mapping继续 admit
            不通过 ⇒ audit记录fallback/ineligible reason，清target staging并
                      `c40_state=NONE`，同轮/下轮按纯dense target继续
阶段 3  在常规锁已生效后，才执行 dense prefix 推进、slot 预留与 copy

例外：若未来确实需要在 admission 前做会触发驱逐的动作，必须引入
      **可跨调用的显式 prefix-guard lease**，并在 scheduler lock 获取成功后
      做一次可审计 handoff；**禁止**依赖一个已经退出的临时 context manager。
```

---

### 8.6 Strict-middle chunk-splitting 执行协议（**可执行规格，冻结**）

这是把"strict-middle island"落到**真实 chunked-prefill 调度器**上的完整协议。
`chunked_prefill_size = 4096` 意味着 dense prefix 通常会跨多个 forward 轮，
因此必须显式控制每轮的 extend 边界，否则 chunk 会越过 `target_start`，
copy 目标区间会被 dense 覆盖，C40 将永远无法触发。

#### 8.6.0 底座对象与 API（`verified-code` @ `0206f17b`）

```text
mapping 存储（**唯一真实存储**）：
  scheduler.req_to_token_pool                      # ReqToTokenPool 实例
  req_to_token_pool.req_to_token                   # torch.int32 [alloc_size, max_context_len]
  req_to_token_pool.req_to_token[req.req_pool_idx, a:b] = kv_indices
  req_to_token_pool.write(indices, values)         # memory_pool.py:273
  req.req_pool_idx                                 # schedule_batch.py:863
  req_to_token_pool.alloc(reqs)                    # 对已有 req_pool_idx 的请求**复用**同一 slot
                                                   #（chunked prefill 续算依赖此行为）

overlap-safe seams：
  prepare_for_extend:
      allocation完成后冻结 C40RoundSnapshot(transaction_id, prefix_len,
      extend_start/end, extend_len, mapped_indices)
  scheduler.stash_chunked_request(req)             # scheduler.py:2630
      在下一轮 add_chunked_req 之前提交 enqueued snapshot并推进cursor
  scheduler.process_batch_result(batch, result)    # scheduler.py:3602
      只做 snapshot/result verification；ScheduleBatch.copy 必须携带
      immutable transaction id，禁止读取共享 req.extend_range
```

**必须纠正的两个错误假设（本协议不依赖它们）**：

```text
✘ 不依赖 `req.prefix_indices` 的增长来表达"已物化到哪里"。
   prefix_indices 由 match_prefix 重建，语义是"来自 tree_cache 的 exact 前缀"，
   **不是**"本请求已物化的 token 数"。C40 的 dense prefix 与 copied island
   在中间chunk都不会进入tree_cache（lifecycle期间skip=True），
   因此执行期间不会反映在prefix_indices上；纯dense fallback只在完整prefill后
   恢复final insertion。

✘ 不存在 `req.req_to_token` 这个属性。
   mapping 只在 req_to_token_pool.req_to_token[req.req_pool_idx, :] 这一处。
```

#### 8.6.1 Request-lifetime 状态与**所有权分离**（F-01/F-02）

```text
挂在 req.c40 容器对象上（单一容器，便于 init_next_round_input 保留）：

c40_state ∈ {NONE, STAGED, ADMISSION_DEFERRED, DENSE_PREFIX, COPY_READY,
             DENSE_SUFFIX, COMMIT, DENSE_ISLAND_FALLBACK,
             ABORTED_TERMINAL, TERMINAL_REJECTED}

exact_length   : int   staging 时刻 len(req.prefix_indices) 的快照（之后不再依赖）
middle_cursor  : int   本请求 effective prefix 的长度（见下）
target_start / target_end / source_key / source_generation /
source_offset / rope_delta / consume_lease / produce_state
staging_case : enum{B-3,B-4,CL-I}
outcome_geometry : enum{contiguous_at_exact_boundary,strict_middle,
                        clipped_at_exact_boundary}
mapping: B-3 -> contiguous_at_exact_boundary
         B-4 -> strict_middle
         CL-I -> clipped_at_exact_boundary
suppress_produce_once : bool
original_skip_radix_cache_insert : bool
approx_copy_committed : bool
lifecycle_live : bool  # req.c40内；当前lifecycle是否活跃，retract后清false

req.c40_audit.lifecycle_ever_entered : bool
  # 独立sticky审计容器，不随reset_for_retract清空；
  # 直到最终outcome/final cleanup写出后才清
req.c40_audit.primary_reason : str | None
req.c40_audit.secondary_reasons : list[str]
req.c40_audit.pending_outcome : str | None
req.c40_audit.radix_insert_suppressed : bool
req.c40_audit.recovery_attempted : bool
  # 进入c40_on_admission_commit后置true；用于区分族1/族2
```

**两类 KV index 必须严格分离（这是 F-01 的核心）**：

| 集合 | 来源 | 所有权 | teardown 处置 |
| --- | --- | --- | --- |
| `borrowed_exact_indices` | staging 时对 `req.prefix_indices.to(torch.int64).clone().contiguous()` 得到的一维 index tensor（只复制**索引元数据**，不复制 KV） | 索引指向的 KV **不属于本请求**；由 tree_cache 持有 | **只做 lock release**（`dec_lock_ref` 等价路径），**绝不 `free`** |
| `owned_materialized_indices` | 一维 contiguous `torch.int64` tensor；从 `req_to_token` 读出的 int32 indices 与 copied indices 均先显式 `.to(device=borrowed.device, dtype=torch.int64)`，再按顺序 `torch.cat` | 索引指向的 KV **属于本请求**；该 tensor 是审计视图，不是第二套 allocator ledger | 由底座 `cache_finished_req/release_kv_cache` 按 `[cache_protected_len, kv_committed_len)` **统一 free 恰好一次**；C40 不逐项再次 free |

```text
派生量（冻结定义）：
  effective_prefix_indices(req)
      := req.c40.effective_prefix_tensor
  effective_prefix_tensor
      := 在staging/enqueued-commit/copy-commit时重建并绑定到req.c40的
         **长生存**torch.cat(...).contiguous() tensor；使用处禁止临时构造
         assert tensor.ndim == 1 and tensor.dtype == torch.int64
         assert tensor.device == req_to_token_pool.req_to_token.device
         assert tensor.is_contiguous()
         # req.c40持有强引用，覆盖prefix pointer H2D与Triton kernel生命周期
  effective_prefix_len(req)
      := len(borrowed_exact_indices) + len(owned_materialized_indices)
  middle_cursor
      := effective_prefix_len              # 二者恒等（INV-1）

不变式：
  INV-1  middle_cursor == effective_prefix_len
                       == len(borrowed) + len(owned)
  INV-2  effective_prefix_indices 的每个位置都已：
         (a) copy committed，或
         (b) allocation+mapping完成且对应forward已按GPU stream顺序enqueue；
         result hook随后只验证，不作为下一轮调度的cursor权威
  INV-3  **admission成功后**的单次active lifecycle内middle_cursor单调不减；
         ADMISSION_DEFERRED target-side reset、final teardown或retract可归零
  INV-4  单个admission attempt内，staging快照后只允许`_lock_node`内final
         equality guard再次读取prefix_indices；若deferred，下一轮必须重新match/
         snapshot，禁止复用上一轮borrowed tensor
  INV-5  borrowed ∩ owned == ∅（同一 KV index 不得同时出现在两个集合）
  INV-6  req.cache_protected_len == len(borrowed_exact_indices) 在 primary
         full-attention 路径恒成立且之后不变；不成立即 fail closed
  INV-7  **仅在prefill阶段的enqueued-range commit与copy-boundary commit内部**：
         req.kv_committed_len == req.kv.kv_allocated_len == middle_cursor
  INV-8  teardown 由底座区间 ledger 把 owned 全部 free 恰好一次；
         borrowed 全部 unlock 恰好一次、**零 free**；C40 不执行第二次逐项 free
```

> **为什么必须分离**：dense prefix 与 copied island 是本请求分配的 slot，
> 必须 free；而 `match_prefix` 借来的 exact 前缀属于 Radix，free 它会
> **破坏其他请求的 cache 并导致 double-free**。把两者混进一个
> 未区分所有权的单一 materialized-index 集合会在 teardown 时二选一皆错。
> 底座已经用 `cache_protected_len` 作为 borrowed/owned 分界，并在
> `cache_finished_req(is_insert=False)` 中 free
> `[cache_protected_len:key_len)`、随后 `dec_lock_ref(last_node)`。
> 因此 C40 的职责是**维护并断言该 ledger**，不是另建释放循环。

#### 8.6.1a Effective prefix helper（**必须新增，统一替换裸 `len(prefix_indices)`**）

`verified-code`：底座在多处直接用 `req.prefix_indices` 计算 extend 起点、
预算与切片。C40 的 dense prefix 与 copied island **不会**进入 `prefix_indices`
（执行中间chunk时`skip_radix_cache_insert=True`），因此这些位置在C40请求上会**低估**
已物化长度，导致重算 dense prefix、slice 错位、logprob 偏移。

**冻结方案**：新增单一 helper 并在下列位置统一改用它。

```python
# python/sglang/srt/managers/schedule_batch.py（新增 helper，proposed）
def effective_prefix_len(req) -> int:
    c40 = getattr(req, "c40", None)
    if c40 is None or c40.state is C40State.NONE or req.req_pool_idx is None:
        return len(req.prefix_indices)          # 非 C40 请求：行为逐字段不变
    return c40.middle_cursor                    # C40 请求：borrowed + owned

def effective_prefix_indices(req):
    c40 = getattr(req, "c40", None)
    if c40 is None or c40.state is C40State.NONE or req.req_pool_idx is None:
        return req.prefix_indices
    return c40.effective_prefix_indices          # borrowed ++ owned

def scheduling_prefix_len(req) -> int:
    # 仅B-3 transaction的pre-gate阶段使用尚未commit的projected prefix。
    c40 = getattr(req, "c40", None)
    if c40 is not None and c40.state is C40State.STAGED and c40.staging_case in {"B-3","CL-I"}:
        return c40.target_end
    return effective_prefix_len(req)
```

**必须改用 helper 的位置（`verified-code` 行号来自 `0206f17b`）**：

| 位置 | 原用法 | 改为 |
| --- | --- | --- |
| `schedule_batch.py:2213` | `r.get_fill_ids()[len(r.prefix_indices):]`（input 切片） | `[effective_prefix_len(r):]` |
| `schedule_batch.py:2217` | `prefix_lens = [len(r.prefix_indices) for r in reqs]` | `[effective_prefix_len(r) for r in reqs]` |
| `schedule_batch.py:2368` | `len(req.prefix_indices)`（logprob 相关） | `effective_prefix_len(req)` |
| `schedule_batch.py:2511/2516/2526/2548/2551` | mamba seqlen 相关 | `effective_prefix_len(req)`（C40 primary 不启用 mamba/SSM，此处为一致性防御） |
| `schedule_policy.add_one_req` / `add_chunked_req` 的 `set_extend_range` 调用点 | `len(req.prefix_indices)` 作 extend 起点 | `effective_prefix_len(req)` |
| `schedule_policy` 的预算计算（`cand_extend_input_len`） | `len(full_ids)-len(prefix_indices)` | 常规用effective；B-3 transaction pre-gate用`scheduling_prefix_len=req.target_end`，成功copy后effective自然等于target_end |
| `schedule_policy.py:1101` `input_tokens`重算 | host-loadback后再次按prefix计算 | primary虽fail-close host-loadback，C40分支仍统一使用`scheduling_prefix_len`以防未来seam漂移 |
| `_update_prefill_budget(prefix_len=...)` cache-hit参数 | `add_one_req`首轮 | 首轮传`len(borrowed_exact_indices)`；`add_chunked_req`及后续轮继续传底座字面`0`，禁止重复累加exact hit |
| `schedule_policy.py:1160-1163` page-alignment重算 | 再次使用裸prefix长度 | 常规用effective；B-3 pre-gate用`scheduling_prefix_len`；primary page_size=1但仍保持语义统一 |
| `mem_cache/allocation.py:316` | `prefix_tensors = [r.prefix_indices for r in batch.reqs]` | `[effective_prefix_indices(r) for r in batch.reqs]`；在kernel前断言长度匹配、dtype=int64、contiguous、device==`req_to_token_pool.req_to_token.device` |
| `schedule_batch.py:2337-2339` `cached_tokens_device` | 不能用effective prefix | 只按borrowed exact device portion计算；copied island仅进入`c40_copied_tokens_total`，禁止出现在用户可见`cached_tokens` |
| `schedule_batch.py:2314-2318` `new_cached = pre_len - already_computed` | copy后effective pre_len会跳过island | copy commit执行`req.already_computed += island_len`；B-3首轮仍得到borrowed exact增量，B-4不重复计prefix |
| `mem_cache/common.py:167-248` | 底座 `cache_finished_req` + overallocated tail 释放 | 保留单一释放权威；调用前断言INV-6、`middle_cursor <= req.effective_kv_committed_len()`及pending snapshot已幂等处理；**不得**在decode后断言INV-7 |
| `scheduler.py` reject/abort/timeout/reset 四路 | 通用 teardown | 全部汇入同一 `release_kv_cache` / C40 cleanup helper；禁止各路径自建 owned free 循环 |

> **非 C40 请求零行为变化**：helper 在 `c40 is None` 时逐字段退化为原表达式；
> `TC-12` 与 `T-GATE-3` 断言该等价性。

#### 8.6.1b `req_pool_idx` 的生命周期（F-02）

```text
pre-admission（staging）：
  req.req_pool_idx **可能为 None**（尚未 alloc request slot）
  => **禁止**在 staging 阶段写 req_to_token_pool
  => staging 只记录 borrowed_exact_indices（= prefix_indices 快照的引用）
     与 plan 元组，不触碰 pool

admit 成功后：
  B-4：首个正常 prepare_for_extend 中的 req_to_token_pool.alloc(reqs)
       分配 req_pool_idx。
  B-3：admission-commit seam 已用同一个底座 API
       req_to_token_pool.alloc([req]) 提前分配，以便零 dense-prefix 情形
       在 suffix forward 前完成 copy；返回 None 时保持 req.kv=None 并
       ADMISSION_DEFERRED；仅成功后才以 `exact_length` 初始化
       `ReqKvInfo/kv_committed_len`，保持 req_pool/kv 成对不变量；
       后续 alloc(reqs) **复用**同一 slot。
  （memory_pool.py 的 alloc 对已有 req_pool_idx 的请求 reuse 同一 slot，
    chunked 续算与 B-3 都依赖此行为）
  此后 req.req_pool_idx 非 None，pool mapping 可写

dense prefix 每个 chunk forward 完成后：
  该 chunk 的 owned slot **已由底座正常路径**写入
      req_to_token_pool.req_to_token[req.req_pool_idx, a:b]
  `prepare_for_extend` 先冻结 round snapshot；forward enqueue 后，下一轮
  `stash_chunked_request` 的 commit hook（§8.6.3 步骤4）只做三件事：
      1. 把 [a:b] 的实际 indices 追加进 owned_materialized_indices
      2. middle_cursor += (b - a)
      3. 断言 INV-1
  底座 `prepare_for_extend` 同轮已把
      req.kv_committed_len = req.kv.kv_allocated_len = extend_range.end
  commit hook 额外断言二者等于新的 middle_cursor（INV-7）
  **不重写 pool、不重算已完成的 token**

suffix 阶段：
  extend 起点 = effective_prefix_len(req) = middle_cursor
  => 已完成的 dense prefix 与 copied island **不会**被重算
```

#### 8.6.2 前置边界分类（**primary 行为必须无歧义**）

staging 时刻按 `exact_length`（= `len(req.prefix_indices)`）与 island 区间的关系分类：

```text
先决所有权检查（primary，冻结）：
  req.cache_protected_len 必须等于 exact_length。
  若 SWA/Mamba/特殊 cache layout 使二者不等：
      outcome = dense_ineligible
      selector_reason = c40_unsupported_cache_protection_geometry
      不创建 C40 request-lifetime plan。
  另外，若存在 host load-back、Hybrid/Mamba pool、dLLM、session、
  non-tree/ChunkCache、disaggregation、HiSparse/speculative，或走
  add_one_req_ignore_eos 的 disabled-tree-cache 专用路径：
      outcome = dense_ineligible
      selector_reason = c40_unsupported_scheduler_mode
      不创建 C40 request-lifetime plan。
理由：primary 的 borrowed_exact_indices 必须与底座 release ledger 的
      borrowed 边界完全相同；任何不等都不能靠猜测释放语义。
```

| 情形 | 条件 | primary（`C40-1R0`）行为 | outcome / reason |
| --- | --- | --- | --- |
| **B-0 island 位于序列起点** | `target_start == 0` | 不属于strict-middle，staging即fail closed；不得进入B-3 transaction | `outcome=dense_ineligible`；`selector_reason=c40_span_not_strictly_middle` |
| **B-1 island 完全在 exact 前缀内** | `target_end <= exact_length` | **不启用 C40**：该区间已由 exact cache 命中，复用无收益且无必要 | `outcome = exact`；不计 attempted fallback；计 `c40_skipped_covered_by_exact_total`（诊断计数器，**非** terminal reason） |
| **B-2 island 与 exact 边界交叠** | `target_start < exact_length < target_end` | **primary fail-closed**（不 clip） | `outcome = dense_ineligible`；`selector_reason = c40_exact_overlap_unsupported` |
| **B-3 island 紧接 exact 边界** | `target_start == exact_length > 0` 且 `target_end < len(prompt)` | controller零dense-prefix退化路径；不走底座连续restore | `c40_copied / contiguous_at_exact_boundary` |
| **B-4 严格中部（primary 主路径）** | `target_start > exact_length` 且 `target_end < len(prompt)` | 完整执行 §8.6.3 七步协议 | `outcome = c40_copied`，`geometry = strict_middle` |
| **B-5 几何兜底** | 以上均不匹配（含`target_end >= len(prompt)`） | staging fail closed；不得依赖selector已过滤 | `outcome=dense_ineligible`；`selector_reason=c40_span_not_strictly_middle` |

```text
冻结说明：
1. B-2 在 **primary lane 一律 fail-closed**，不做 clip。
   "把 island 裁剪到 [exact_length, target_end)" 属 **conditional lane CL-I**
   （新增，见 §6.3），默认关闭；开启时其 outcome 必须带
   geometry = clipped_at_exact_boundary，且**不进入 primary headline**。
   CL-I clip后把execution target_start重写为exact_length、同步增加source_offset、
   staging_case=CL-I；其admission/copy生命周期完全复用B-3零dense-prefix路径。
2. B-3 虽然几何上等价于底座的连续恢复，但**仍走 controller**，
   以保证 outcome taxonomy 单一、telemetry 口径一致、terminal reason 归属明确。
3. B-0..B-5 的判定发生在 staging（只读），B-5保证穷尽。
```

#### 8.6.2a Admission 判据：identity membership signal（F-03，**冻结**）

**问题（`verified-code` @ `0206f17b`）**：`AddReqResult` 只有三个取值
（`schedule_policy.py:435-438`）：

```python
class AddReqResult(Enum):
    CONTINUE = auto()   # Continue to add requests
    NO_TOKEN = auto()   # No token left
    OTHER    = auto()   # Other reasons to stop adding requests
```

这三个值描述的是 **"是否继续向本批添加更多请求"**，**不是**
"当前这个请求是否被加入了"。`add_one_req` 在多条路径上会
**先把 req 追加进 `can_run_list`**（或设 `new_chunked_req`），
**再**返回 `NO_TOKEN` / `OTHER` 表示"批已满，别再加了"。
因此 **`status == NO_TOKEN` 并不蕴含 "本请求未被加入"**。
若 C40 据此判 `ADMISSION_DEFERRED`，会在请求**已经进入本批并即将 forward**
时错误地丢弃 plan 或跳过 pin，产生状态机与实际执行不一致。

**冻结方案**：**不修改** `add_one_req` 或 `add_chunked_req` 的既有返回类型；
由 `PrefillAdder` 暴露 identity membership 查询，作为唯一 admission signal。

```python
# python/sglang/srt/managers/schedule_policy.py（proposed）
def contains(self, req: Req) -> bool:
    return any(candidate is req for candidate in self.can_run_list)
```

```text
实现约束（冻结）：
  1. add_one_req 仍返回 AddReqResult；add_chunked_req 仍返回 Req | None。
     不得把 add_chunked_req 的返回值改成 decision 对象：
       return req 可能表示 hybrid-SWA 下"本轮未加入但仍需保留 chunked req"；
       return None 可能表示"本轮已加入且这是最后一块"。
  2. 调用方必须保存当前对象 identity，并在调用返回后立即执行
       added = adder.contains(req)
     `added` 当且仅当该对象已进入 `can_run_list`。
     chunked caller 的冻结写法：
       current = self.chunked_req
       next_chunked = adder.add_chunked_req(current)
       added = adder.contains(current)
       self.chunked_req = next_chunked if added else current
     因此"未加入本轮"不会把既有 chunked request 丢失。
  3. `new_chunked_req` / `self.chunked_req` 只表达后续是否仍有 middle chunk，
     **不**替代 membership 判据。
  4. manifest 的 `admission_signal` 冻结为
       `prefill_adder_identity_membership_v1`。
  5. **禁止**任何形式的"从 AddReqResult 或 Req|None 猜 added"。
```

**C40 的使用规则（冻结）**：

| `added` | 既有返回值 | C40 行为 |
| --- | --- | --- |
| `True` | `add_one_req` 可为 `CONTINUE` / `NO_TOKEN` / `OTHER`；`add_chunked_req` 可为 `Req` / `None` | **一律视为已加入**；`add_one_req` 内部在最终 append 前的 admission-commit seam 已原子执行 revalidate+pin，并进入 `DENSE_PREFIX`/`COPY_READY` 或 fail closed。B-3 若需立即 copy，使用底座 `req_to_token_pool.alloc([req])` 提前分配 request slot；B-4 由首个 dense chunk 的正常 allocation 获得 slot。**不得** `ADMISSION_DEFERRED` |
| `False` | 任意 | 新请求进入 `ADMISSION_DEFERRED`；既有 chunked request 保留原 `self.chunked_req` 并等待下一轮，不执行 C40 copy、不写 pool、不新建 lease |

**C40内部deferral返回值（冻结，不得混用）**：

| cause | return | scheduler行为 |
| --- | --- | --- |
| single-owner conflict（已有chunked owner） | 保持底座`AddReqResult.OTHER` | transaction未开始；本轮停止继续加request，membership=false后defer |
| B-3 request-pool slot耗尽 | `AddReqResult.NO_TOKEN` | rollback后停批；与真实req-pool capacity exhaustion一致 |
| 其它底座capacity/alignment拒绝 | 保持底座原返回值 | transaction尚未开始，或统一rollback后返回 |

任一返回后都执行identity membership postcondition；`added=false`时资源必须全零。
`scheduler.py`把现有not-added清理抽成`cleanup_not_added_req(req)`并在
membership=false时统一调用；所有C40 deferral保持底座原status，不创造
`CONTINUE+not-added`新语义。

```text
原子性要求：
  `add_one_req` 在所有capacity/compatibility/alignment拒绝检查（含B-3
  projected-prefix计算）通过后、最终
  `can_run_list.append(req)`/`new_chunked_req=req` 之前，且仍处于底座
  `with self._lock_node(req.last_node)` 临时 admission lock 内，调用
  `c40_on_admission_commit(req)`；其中 "revalidate + pin source"
  必须在同一临界区内完成，
  中间不得让出调度（否则 generation 可能在两步之间变化）。
  revalidate 失败时**不得**留下已 pin 的 lease。
  调用返回后的 `adder.contains(req)` 是对真实 membership 的唯一外部确认，
  不是把 revalidate 推迟到返回之后。
  在`_lock_node`内、任何allocation/pin前必须最终守卫：
    req.cache_protected_len == len(borrowed_exact_indices)
    torch.equal(borrowed_exact_indices,
                req.prefix_indices.to(torch.int64))
  不成立即fail closed并清target-side staging；该正确性同步属于C40 admission
  overhead，必须计入端到端时间。
```

**必须新增的测试**：

| ID | 断言 |
| --- | --- |
| `TC-21` | `add_one_req` 后 `added=True` 且 `status=NO_TOKEN` ⇒ C40 进入 `DENSE_PREFIX`（**不**进 `ADMISSION_DEFERRED`） |
| `TC-22` | `add_one_req` 后 `added=True` 且 `status=OTHER` ⇒ 同上 |
| `TC-23` | `added=False` ⇒ `ADMISSION_DEFERRED`，且 `provisional_slots==0`、`consume_lease is None`、plan 保持 |
| `TC-24` | `added=True` 但 revalidate 失败 ⇒ `DENSE_ISLAND_FALLBACK` 且**无**残留 lease |
| `TC-25` | `add_chunked_req` 的四种组合被正确区分：`return req/None × added true/false`；原返回契约与非 C40 行为逐字段不变 |

#### 8.6.3 协议七步

```text
步骤 1  staging（pre-admission，只读）
        位置：init_next_round_input(tree_cache) 内、match_prefix 之后
        动作：
          exact_length := len(req.prefix_indices)          # 快照，仅此一次
          assert req.cache_protected_len == exact_length
              不成立 -> dense_ineligible(c40_unsupported_cache_protection_geometry)
          以下 scheduler/layout 模式 primary 一律 fail closed，不进入 plan：
            req.needs_host_load_back() == true
            req.is_dllm() == true
            req.sampling_params.ignore_eos 且 tree_cache.disable == true
            req_to_token_pool 是 HybridReqToTokenPool / tree_cache.supports_mamba()
            token_to_kv_pool_allocator 属任一SWA/PureSWA/Hybrid-SWA布局
            req.session is not None
            not tree_cache.is_tree_cache() / tree_cache.disable == true
            get_server_args().disaggregation_mode != NULL
            get_server_args().enable_hisparse == true
            get_server_args().speculative_algorithm != None
            get_server_args().chunked_prefill_size is None
            get_server_args().page_size != 1
            get_server_args().enable_mixed_chunk == true
            get_server_args().enable_dynamic_chunking == true
            get_server_args().enable_deterministic_inference == true
            req.multimodal_inputs is not None / encoder-decoder request
          -> dense_ineligible(c40_unsupported_scheduler_mode，附 subreason)
          取值来源冻结：Req字段只从req读取；cache/layout只从tree_cache及
          tree_cache.req_to_token_pool读取；server模式只用本模块既有
          `get_server_args()`，staging不得假设持有Scheduler对象。
          按 §8.6.2 分类 B-0..B-5
          B-0 -> dense_ineligible(c40_span_not_strictly_middle)
          B-1 -> outcome=exact，不进入 C40
          B-2 -> clip flag=0: dense_ineligible(c40_exact_overlap_unsupported)
                 clip flag=1: 重写为CL-I staging_case并复用B-3生命周期
          B-5 -> dense_ineligible(c40_span_not_strictly_middle)
          若req.return_logprob且`0 <= req.logprob_start_len < target_end`：
            dense_ineligible(c40_logprob_range_unsupported)
          `logprob_start_len == -1`表示无prompt-logprob区间，不触发该拒绝；
          B-3/B-4 -> 写 request-lifetime plan
                     borrowed_exact_indices :=
                       req.prefix_indices.to(torch.int64).clone().contiguous()
                     owned_materialized_indices :=
                       同 device 的空 contiguous torch.int64 tensor
                     effective_prefix_tensor := borrowed_exact_indices
                     middle_cursor := exact_length             # == len(borrowed)
                     c40_state := STAGED
        禁止：分配 slot、驱逐、pin 目标侧资源、**写 req_to_token_pool**
              （此时 req.req_pool_idx 可能为 None，§8.6.1b）

步骤 2  admission（**判据见 §8.6.2a，禁止从既有返回值反推**）
        位置：scheduler.py:3018 -> add_one_req
        PRE-FLIGHT（零资源、仍为STAGED）：
        needs_forced_middle := staging_case==B-4 and target_start>exact_length
        若needs_forced_middle且已有chunked owner：
          保持底座原`AddReqResult.OTHER`/not-added路径；
          membership=false后转ADMISSION_DEFERRED并清target staging。
          **不**自造CONTINUE返回，不进入commit。
        随后用projected_prefix_len（B-3/CL-I为target_end，B-4为exact_length）
        完成全部capacity/chunk/alignment gate，并预计算不可变
        `AdmissionPlan(extend_start,end,forced_middle,truncated,budget_delta)`；
        B-3/CL-I 的 total-token gate必须额外计入`island_len` copy allocation，
        不能只按suffix extend计算；
        任何NO_TOKEN/OTHER return均发生在C40 transaction之前。
        随后在最终append前执行
          （仍在底座 `_lock_node(req.last_node)` 内）
          c40_on_admission_commit(req):
                         req.c40_audit.recovery_attempted := true
                         B-3/CL-I先在唯一commit block内用底座
                                      slot_result := req_to_token_pool.alloc([req])
                                      若返回 None 或 req.req_pool_idx 仍为 None：
                                        不设置req.kv、不pin、不append；
                                        返回 NO_TOKEN，added=false -> ADMISSION_DEFERRED
                                      成功后才初始化：
                                        req.kv = ReqKvInfo(kv_allocated_len=exact_length,
                                                           swa_evicted_seqlen=0)
                                        req.kv_committed_len = exact_length
                                      且断言 exact_length == target_start >= 1。
                         **原子** revalidate source generation/content hash/fingerprint + pin
                         通过 -> 原子进入lifecycle：
                                   original_skip_radix_cache_insert :=
                                     req.skip_radix_cache_insert
                                   req.skip_radix_cache_insert := true
                                   lifecycle_live := true
                                   req.c40_audit.lifecycle_ever_entered := true
                                   approx_copy_committed := false
                                 B-4: c40_state := DENSE_PREFIX
                                 B-3/CL-I: c40_state := COPY_READY，执行copy/ledger commit
                         commit成功后立即按预计算AdmissionPlan设置range/budget/
                         new_chunked_req并append；之间**禁止普通return**。
                         任何异常或最终`added=false`必须按目的分支rollback：
                         - rollback_for_deferral：释放copied/
                         provisional/lease与B-3 request slot，恢复kv_committed_len，
                           恢复skip=original、lifecycle_live=false、
                           lifecycle_ever_entered/recovery_attempted/
                           radix_insert_suppressed恢复transaction前值、清pending reason，
                           令req.kv=None；
                         - fallback_to_dense：释放copy/lease并恢复skip=original、
                           lifecycle_live=false，但保留recovery_attempted=true与
                           audit reason，state=NONE后同轮纯dense。
                         postcondition（deferral）：
                           not adder.contains(req) =>
                             req_pool_idx is None and req.kv is None and
                             provisional_slots==0 and consume_lease is None and
                             lifecycle_live==false and
                             req.skip_radix_cache_insert==transaction_original
                         primary 已在步骤1拒绝Hybrid/Mamba/dLLM，因此此处不绕过
                         `alloc_req_slots`的Mamba headroom语义
                         不通过 -> rollback lease/slot，audit记录attempted fallback，
                                   clear_target_staging + state=NONE，
                                   重新按当前exact prefix做dense gate/range
        status = adder.add_one_req(...)
        added  = adder.contains(req)                           # identity membership
        if added:
            常规 request lock 接管；写 handoff 记录
            B-3 的 request slot 已提前分配；B-4 的 request slot 将由
            首个正常 prepare_for_extend/alloc_for_extend 分配
        else:
            c40_state := ADMISSION_DEFERRED（非 terminal）
            保留source-side plan，清空target-side borrowed/owned/effective/cursor
            无 slot、无 lease、不写 pool
            下一轮回到步骤1重新match/rebuild/revalidate

步骤 3  chunk 边界钳制（**每一轮都执行**）
        位置：add_one_req 与 add_chunked_req 的 set_extend_range 调用点
        规则：
          if path == add_one_req:
              # 只执行步骤2已冻结且已通过owner/gate的AdmissionPlan
              extend_start/end/forced_middle := admission_plan.*
          elif c40_state == DENSE_PREFIX and target_start > middle_cursor:
              boundary   = controller.next_extend_boundary(req)   # == target_start
              extend_end = min(extend_end, boundary)
          req.set_extend_range(<本轮 extend 起点>, extend_end)
          forced_middle = (extend_end < len(req.full_untruncated_fill_ids))
          if forced_middle:
              max_new_tokens := 0
              if path == add_one_req:
                  assert adder.new_chunked_req is None
                  adder.new_chunked_req := req
              else:  # add_chunked_req current owner
                  truncated := True
              actual_len := req.extend_range.length
              normal _update_prefill_budget(..., extend_input_len=actual_len, ...)
              reserve_exclusive_chunked_owner()
                # 将 rem_chunk_tokens 置0并同步扣尽对应rem_input_tokens；
                # 不改rem_total/cur_rem offsets，不污染log_input_tokens
        scheduler 在 can_run_list 提交后对仍为 chunked 的 req 执行
          inflight_middle_chunks += 1
        **无论原始 input_tokens 是否 <= chunk_tokens_limit，上述规则均成立**。
        `add_chunked_req` 的 `truncated` 必须在钳制后重算；
        禁止沿用 `cand_extend_input_len > _rem_tokens` 的钳制前值。
        注意：钳制条件用 **middle_cursor**，不用 len(prefix_indices)
        硬断言：若 `extend_range.end < prompt_end`，
                本轮结束前 `inflight_middle_chunks > 0`；
                否则会被 batch_result_processor 当作 prefill 完成并采样。

步骤 4  overlap-safe enqueued-range commit + result verification（**新增**）
        A. prepare_for_extend 在 allocation/mapping 完成后冻结：
             C40RoundSnapshot(transaction_id, prefix_len, extend_start,
                              extend_end, extend_len, mapped_indices,
                              forward_enqueued=false, is_final)
           snapshot 必须复制值，不引用之后会变化的 req.extend_range。
           `run_batch`成功enqueue后把同一transaction标`forward_enqueued=true`；
           enqueue前失败不得commit，只由底座ledger释放allocation。
        B. 下一轮 get_next_batch_to_run 的 stash_chunked_request seam
           （scheduler.py:2630 / :2745-2756），在任何 add_chunked_req 前调用：
             c40_commit_enqueued_prefill(req, req_to_token_pool)
           completed = snapshot.extend_len
           assert snapshot.forward_enqueued
           assert snapshot.extend_start == middle_cursor
           new_indices = snapshot.mapped_indices
          new_indices := new_indices.to(device=borrowed.device, dtype=torch.int64)
          owned_materialized_indices :=
              torch.cat((owned_materialized_indices, new_indices)).contiguous()
          effective_prefix_tensor :=
              torch.cat((borrowed_exact_indices,
                         owned_materialized_indices)).contiguous()
          middle_cursor := snapshot.extend_end
          snapshot.committed := true（同一transaction禁止二次commit）
          assert middle_cursor == len(borrowed) + len(owned)          # INV-1
          if config.c40_validation:
              assert not torch.isin(new_indices, borrowed).any().item()  # INV-5
          if c40_state == DENSE_PREFIX:
              if   middle_cursor == target_start: c40_state := COPY_READY
              elif middle_cursor <  target_start: 保持 DENSE_PREFIX
              else: c40_boundary_overrun -> DENSE_ISLAND_FALLBACK
                                            + invalid_engineering{boundary_overrun}
          elif c40_state in {DENSE_SUFFIX, DENSE_ISLAND_FALLBACK}:
              assert middle_cursor <= len(prompt)
              # 只推进continuation cursor；不得再与target_start比较
          else:
              fail loud: unexpected snapshot state
        C. process_batch_result 只调用
             c40_verify_prefill_result(batch_snapshot, result)
           completed 必须取 `batch.extend_lens[i]` / `batch.prefix_lens[i]`
           与immutable transaction id，禁止读取共享 `req.extend_range`；
           非final chunk：只校验已commit snapshot，不推进cursor。
           final chunk（`snapshot.is_final && snapshot.forward_enqueued
           && !snapshot.committed`）：
             允许在verification hook内调用一次
             `commit_pending_snapshot(final_only=true)`；
             此时请求已离开chunked调度，cursor不再影响任何下一轮range，
             因此不破坏overlap-safe调度权威。
           teardown前再调用幂等`commit_pending_snapshot()`兜底；
           `forward_enqueued=false`时只清snapshot，不追加owned/cursor；
           已commit transaction必须no-op，禁止重复追加owned。
        语义限定：INV-2中的"物化"对chunked continuation表示
          allocation完成 + mapping写入 + forward按GPU stream顺序enqueue。
        该定义与底座stash/cache unfinished seam一致，并适用于默认overlap scheduler。

步骤 5  copy 与 copy-boundary commit
        位置：
          B-3/CL-I -> add_one_req 的 C40 pre-range hook
          B-4 -> 下一轮 add_chunked_req 的 C40 pre-range hook
        （二者都在 suffix 的 set_extend_range / allocation / forward 之前）
        B-4在copy前也必须用`island_len + planned_suffix_len`做当前adder的
        capacity gate；不得只按suffix预算。若该gate拒绝，不尝试copy，
        记唯一runtime reason `c40_copy_budget_insufficient`，释放consume lease，
        转DENSE_ISLAND_FALLBACK并在本轮admit dense continuation。
        COPY_READY 校验通过 => execute_reuse_plan()
        随后（**DENSE_SUFFIX 的第一次 forward 之前**）必须完成：
          a. island 的 KV indices 写入 pool mapping：
               req_to_token_pool.req_to_token[
                   req.req_pool_idx, target_start:target_end] = island_indices
             （等价写法 req_to_token_pool.write((req.req_pool_idx, slice), values)）
          b. island_indices_i64 :=
                 island_indices.to(device=borrowed.device, dtype=torch.int64)
             owned_materialized_indices :=
                 torch.cat((owned_materialized_indices,
                            island_indices_i64)).contiguous()
             effective_prefix_tensor :=
                 torch.cat((borrowed_exact_indices,
                            owned_materialized_indices)).contiguous()
          c. provisional -> 正式所有权转正（复用
             commit_provisional_recovery_slots 的 release/commit 记账语义）
             allocator live `available_size()` 已反映island占用；
             禁止额外修改rem_total/cur_rem offsets（避免双计），
             copy token只记`c40_copied_tokens_total`，不记log_input
          d. middle_cursor := target_end
          e. 设置：
             req.kv_committed_len := target_end
             req.kv.kv_allocated_len := target_end
             req.already_computed += island_len
             approx_copy_committed := true
             req.c40_audit.radix_insert_suppressed := true
               # B-3: 0+island，使首轮new_cached==borrowed exact
               # B-4: target_start+island==target_end，避免重复计island
             assert middle_cursor == len(borrowed) + len(owned)   # INV-1
             if config.c40_validation:
                 assert not torch.isin(island_indices_i64,
                                       borrowed).any().item()      # INV-5
             assert req.kv_committed_len == req.kv.kv_allocated_len
                                             == middle_cursor      # INV-7
          f. c40_state := DENSE_SUFFIX
        **不**修改 req.prefix_indices（它属 tree_cache 语义，与本请求物化无关）

步骤 6  DENSE_SUFFIX
        下一轮 set_extend_range 的**起点取 effective_prefix_len(req)**
        （== middle_cursor == target_end），**不取** len(req.prefix_indices)
        步骤 3 的钳制条件已不满足，不再钳制
        suffix forward 可 attention 到
          req_to_token[req.req_pool_idx, 0:target_end]
        即 [borrowed exact | owned dense prefix | owned copied island]
        **已完成部分不会被重算**（起点即 cursor）
        prefill完成时：
          if approx_copy_committed:
              skip_radix_cache_insert保持true至final release
          else:  # 全dense fallback、从未成功commit approximate copy
              req.skip_radix_cache_insert :=
                  original_skip_radix_cache_insert
              # 只在完整prefill完成后恢复，禁止中间chunk stash改写prefix ledger

步骤 6b  teardown（五条 final path 或 retract reset，§8.1.2）
        只调用一次底座 release_kv_cache/cache_finished_req：
          cache_finished_req(is_insert=False) free
              [req.cache_protected_len, req.effective_kv_committed_len())
          release_kv_cache 另行处理 overallocated tail
          dec_lock_ref(req.last_node) 释放 borrowed lock
        C40 **不得**再逐项 free owned_materialized_indices；
        owned/borrowed 列表只用于调用前后审计与 INV-6/INV-8 断言。
        copy 后、suffix 前发生 abort/reject/timeout/reset 时，
        INV-7 保证 copied island 已落入底座 committed/allocated 区间，
        因而同样被释放恰好一次。
        retract 额外要求：
          仅对 `lifecycle_live=true` 的C40请求，release前
          `suppress_produce_once=true`，`register_request_segments`
          与 C40 produce hook 都必须跳过本次注册；
          release 后清空 `req.c40`，下次 staging 只能读取新 mapping。
          reset hook恢复`req.skip_radix_cache_insert :=
          original_skip_radix_cache_insert`，因为本次release本就`is_insert=false`；
        任一final teardown前若存在uncommitted final snapshot，先幂等commit；
        随后只断言`middle_cursor <= effective_kv_committed_len()`，因为decode
        已推进kv_committed_len而不会推进prefill cursor。

步骤 7  fallback
        任一阶段失败 => DENSE_ISLAND_FALLBACK
        若失败发生在B-3/B-4 copy transaction内：
          先rollback并free copy allocation（live available_size自然回补）；
          再以`effective_prefix_len(req)==middle_cursor`重跑底座
          total-token/chunk/alignment gate，实际dense remaining包含island；
          B-3新请求不通过：返回对应NO_TOKEN/OTHER，完整rollback到零资源，
            added=false→ADMISSION_DEFERRED。
          B-4既有chunked owner：**禁止parked return**。copy失败后立即转
            DENSE_ISLAND_FALLBACK，并按底座`add_chunked_req`语义在本轮
            admit至少一个positive-length dense chunk；owner必须进入
            can_run_list，确保inflight有增有减。若allocator最终仍失败，
            走底座fail-loud engineering error，不得保留未入batch的owner。
        从 **middle_cursor** 继续 dense（解除钳制），
        后续 extend 覆盖 [middle_cursor, len(prompt))
        请求正常完成；无 KV 空洞（INV-2 保证）
```

#### 8.6.4 需要修改 / 新增的 current substrate 位置（进 clean-room allowlist）

| 文件 | 性质 | 具体内容 |
| --- | --- | --- |
| `python/sglang/srt/managers/schedule_batch.py` | 既有最小修改 | Req/C40状态；effective helpers；staging只读，admission成功后显式skip控制；round snapshot/copy；retract hooks；cache telemetry |
| `python/sglang/srt/mem_cache/allocation.py` | 既有最小修改 | `prefix_tensors` 改用 `effective_prefix_indices(r)`；`prefix_lens` 与每个 tensor 长度必须相等，否则在进入 Triton kernel 前 fail loud；正常 allocation 继续作为 dense chunk 的唯一 slot 分配权威 |
| `python/sglang/srt/managers/schedule_batch.py` 的 logprob 路径 | 既有最小修改 | logprob 起点偏移改用 `effective_prefix_len`，避免 C40 请求 logprob 错位 |
| `python/sglang/srt/managers/schedule_policy.py` | 既有最小修改 | identity membership；B-3 projected-gate后单commit-block transaction；effective range；forced-middle仅新请求做owner exclusion，owner continuation不defer；forced owner独占剩余chunk budget |
| `python/sglang/srt/managers/scheduler.py` | 既有最小修改 | not-added cleanup；snapshot enqueue/stash/result；inflight++前membership assert；普通abort与`process_pending_chunked_abort`统一final cleanup；retract接线 |
| `python/sglang/srt/mem_cache/common.py` | 既有最小修改 | 不改变主释放；仅当`lifecycle_live && suppress_produce_once`时跳过registration/C40 produce；C40 off的R0/R1 REGISTER retract行为逐字段不变 |
| `python/sglang/srt/mem_cache/approx_kv/config.py` | 既有最小修改 | 新增 `SGLANG_APPROX_KV_C40_*` 配置项 |
| `python/sglang/srt/mem_cache/approx_kv/coding_c40/**` | 全新 | controller含lifecycle hooks；adapter含`C40ExecutionPlan`及reason+interval dense rules与底座fallback reason直通映射 |

> 以上六个唯一的"既有文件最小修改"必须逐一列入 §4.4 CR-5 的 allowlist，
> **且其 modified hunks 仍要接受 CR-1/CR-2/CR-3/CR-9 的血缘扫描**（§4.6）。

#### 8.6.5 必须新增的测试（chunk continuation / multi-round / 边界分类）

| ID | 断言 |
| --- | --- |
| `TC-1` | `chunk = 4096`、dense prefix 长度 `> 4096` ⇒ `DENSE_PREFIX` 跨 `>= 2` 轮，且每轮 `extend_range.end <= target_start` |
| `TC-2` | 存在某一轮 `middle_cursor` 恰好等于 `target_start`（边界命中，非越过） |
| `TC-3` | 续算轮走 `add_chunked_req` 路径时 request-lifetime plan 未丢失（`target_start` / `source_key` / `middle_cursor` / `borrowed_exact_indices` / `owned_materialized_indices` 逐字段保持） |
| `TC-4` | `c40_commit_enqueued_prefill` 在下一轮add之前、`middle_cursor==target_start`时恰好一次触发COPY_READY；late result verification不重复推进 |
| `TC-5` | copy 后 `req_to_token_pool.req_to_token[req.req_pool_idx, target_start:target_end]` 与 island indices 逐元素一致；`middle_cursor == target_end` |
| `TC-6` | `DENSE_SUFFIX` 首轮 `extend_range.start == middle_cursor == target_end`（**不等于** `len(req.prefix_indices)`） |
| `TC-7` | 注入越界故障被 `c40_boundary_overrun` 检出并 fallback，同时记族 4 |
| `TC-8` | `ADMISSION_DEFERRED`连续N轮：source-side plan保持，target-side borrowed/effective/cursor每轮清空并由新match重建；模拟旧KV index被驱逐复用后仍只写新mapping；最终可admit |
| `TC-9` | fallback 从 `middle_cursor` 继续；`[exact_length, len(prompt))` 逐 token 覆盖无空洞 |
| `TC-10` | lifecycle中间chunk skip=true且pool mapping持续；copy成功final仍skip；full-dense fallback prefill完成后恢复original并可final insert exact |
| `TC-11` | 钳制在 `c40_state != DENSE_PREFIX` 时不生效（不影响非 C40 请求与 suffix 阶段） |
| `TC-12` | C40 关闭时 `set_extend_range` 与 `req_to_token` 写入行为与 baseline 逐字段一致 |
| `TC-13` | **INV-1**：任意时刻 `middle_cursor == len(borrowed_exact_indices) + len(owned_materialized_indices)` |
| `TC-14` | **INV-3**：admission成功后cursor单调；ADMISSION_DEFERRED target reset、final teardown、retract为允许归零事件 |
| `TC-15` | **INV-4**：单轮staging后除admission final guard外不再读取prefix_indices；deferred下一轮允许重新match/rebuild，guard必须与新tensor相等 |
| `TC-16` | **B-1**：`target_end <= exact_length` ⇒ `outcome == exact`，`c40_skipped_covered_by_exact_total` +1，**不**计 attempted fallback |
| `TC-17` | **B-2**：`target_start < exact_length < target_end` ⇒ `outcome == dense_ineligible`，`selector_reason == c40_exact_overlap_unsupported`，且**未**发生 clip |
| `TC-18` | **B-3**：`target_start == exact_length > 0` ⇒ controller退化路径、零dense prefix、`c40_copied/contiguous_at_exact_boundary`；未走底座连续restore |
| `TC-19` | **B-4**：严格中部路径 `geometry == strict_middle` |
| `TC-20` | teardown（五条final path任一）：只调用一次底座release；owned全部free恰好一次，borrowed全部unlock且零free；无C40第二套逐项free（INV-8） |
| `TC-26` | **double-free 检测**：注入"borrowed 被 free"或"C40 再次逐项 free owned"的故障必须被断言捕获；正常路径下 allocator 的 owned free 计数 == `len(owned)` |
| `TC-27` | **不重算**：`DENSE_SUFFIX` 首轮的 `extend_range.start == effective_prefix_len(req)`；`[0, middle_cursor)` 区间在整个请求内被 forward 的次数**恰好一次** |
| `TC-28` | **attention 可见性**：suffix forward 的 attention mask / seq_len 覆盖 `[0, target_end)`；对 borrowed 段与 owned 段均可见（用小模型逐层探针断言） |
| `TC-29` | **logprob合同**：`logprob_start_len < target_end`时fail closed；`>=target_end`时suffix logprob与dense逐位置对齐，偏移用effective prefix |
| `TC-30` | **helper 等价性**：`c40 is None` 时 `effective_prefix_len(req) == len(req.prefix_indices)`；非 C40 请求的 `prefix_lens` / input 切片 / 预算与 baseline 逐字段一致 |
| `TC-31` | **INV-5 validation-on**：用`torch.isin`捕获borrowed/owned重叠；仅debug/property运行，measured validation=0路径无`.item()`同步 |
| `TC-32` | **pre-admission 不写 pool**：staging 阶段 `req.req_pool_idx is None` 时，对 `req_to_token_pool` 的写调用计数为 0（mock/spy 断言） |
| `TC-33` | **短 prompt forced-middle**：`len(prompt) <= chunk_tokens_limit` 且 `target_start < len(prompt)` 时，首轮仍设置 `new_chunked_req`、`inflight_middle_chunks > 0`、`max_new_tokens=0`；不得在 `target_start` 采样；最终完整经历 `DENSE_PREFIX→COPY_READY→DENSE_SUFFIX`，`[0,len(prompt))` 每 token 恰好物化一次 |
| `TC-34` | **prefix tensor 成对性**：每个 C40 req 在 `write_cache_indices` 前满足 `len(effective_prefix_indices(req)) == effective_prefix_len(req)`；Triton 与非 Triton 路径写入的 `[0:prefix_len)` mapping 与 helper 逐元素一致 |
| `TC-35` | **copy ledger + 中途终止**：B-3 提前分配后始终满足 `(req_pool_idx is None) == (req.kv is None)`；copy commit 后立刻满足 `kv_committed_len == kv_allocated_len == middle_cursor == target_end`；在 copy failure 或 suffix 前分别注入 abort/reject/timeout/reset，copied island全部free一次、borrowed零free、overallocated assert不触发 |
| `TC-36` | **cache protection 几何**：`cache_protected_len != len(prefix_indices)` 时 primary fail closed 为 `c40_unsupported_cache_protection_geometry`，不创建 lease/slot/C40 plan |
| `TC-37` | **单 chunked-owner**：跨轮已有owner时第二个forced request保持OTHER且deferred；同轮首个owner耗尽budget使后续请求走底座stop；owner continuation不被自身guard挡住 |
| `TC-38` | **retract/preempt reset**：纯retract清live但保留sticky audit、重排队后重staging且旧index零复用；OOM abort-last子用例只计abort final、不计retract，aborted outcome与final cleanup identity一致 |
| `TC-39` | **B-3 slot/transaction failure**：alloc None时返回NO_TOKEN并保持零资源；成功后任一exception或non-append exit都rollback slot/kv/copied/provisional/lease；postcondition `not contains(req) => no resources` |
| `TC-40` | **unsupported scheduler modes**：覆盖SWA allocator、dynamic chunking、deterministic alignment、multimodal/encoder等完整subreason域 |
| `TC-41` | **overlap scheduler**：默认overlap开启时，round k的snapshot在round k+1 add前由stash seam commit；result可晚一轮但只verification；每个token forward恰好一次，无stale `req.extend_range`读取 |
| `TC-42` | **owner不可被非C40覆盖**：1个forced-middle C40后跟长/短非C40请求，后者均不得写`new_chunked_req`或入batch；C40 owner的inflight>0且不提前采样 |
| `TC-43` | **B-3普通early-return rollback**：覆盖`trunc_len<=0`、truncation alignment、OTHER/NO_TOKEN等非异常return；transaction要么未开始，要么完整rollback |
| `TC-44` | **B-0**：`target_start==0`在staging判`c40_span_not_strictly_middle`，不得进入B-3或触发`target_start>=1`断言 |
| `TC-45` | **deferral返回语义**：owner conflict保持底座OTHER，req-pool exhaustion为NO_TOKEN；transaction均未泄漏且added=false |
| `TC-46` | **cleanup metric domain**：C40 off、B-0/B-1/ineligible不写cleanup；final identity按`req.c40_audit.lifecycle_ever_entered`，retract按live |
| `TC-47` | **retract suppression作用域**：只有live C40 lifecycle在retract时抑制produce；C40 off的R0/R1 REGISTER请求仍按baseline注册，逐字段无变化 |
| `TC-48` | **prefix device contract**：effective prefix始终`torch.int64` contiguous且device等于`req_to_token_pool.req_to_token.device`；CPU/ChunkCache输入被primary fail closed |
| `TC-49` | **final snapshot commit**：最后prefill chunk令`add_chunked_req`返回None时不再stash；result verification执行一次terminal commit，teardown幂等兜底不重复owned/cursor |
| `TC-50` | **decode后teardown**：至少decode 1 token后允许`kv_committed_len > middle_cursor`；teardown只断言`middle_cursor <= effective_kv_committed_len`，不误触INV-7 |
| `TC-51` | **validation开销隔离**：validation=1才运行ownership`torch.isin/.item`；validation=0仍允许且必须运行admission`torch.equal`guard，其耗时单列并计入end-to-end |
| `TC-52` | **retract-abort sticky audit**：OOM abort-last保留`lifecycle_ever_entered`直至aborted outcome/final cleanup写出，`lifecycle_live`可清；不产生retract计数 |
| `TC-53` | **copy预算与fallback regate**：B-3/B-4 pre-gate计island；成功后live rem_total恰好因allocator减少island_len且无额外offset双计；failure free后自然回补并以dense remaining重跑gate |
| `TC-54` | **not-added底座清理**：所有OTHER/NO_TOKEN deferral统一清provisional/Mamba transient但保留source-side plan |
| `TC-55` | **staging config来源**：unsupported判定只读req/tree_cache/get_server_args；chunked_prefill=None或page_size!=1 fail closed；不引用Scheduler实例 |
| `TC-56` | **cached prefix tensor lifetime**：helper返回req.c40持有的同一长生存tensor；只在commit时重建；Triton pointer使用期间强引用存在，使用处无临时torch.cat |
| `TC-57` | **B-4 fallback必须admit**：copy失败后本轮仍把owner加入can_run_list并调度positive-length dense chunk；`adder.contains(owner)`在inflight++前为true，计数最终回0；禁止parked continuation |
| `TC-58` | **deferred rematch**：跨轮驱逐并复用旧KV index；source plan保留但target borrowed/effective/cursor重建，admission guard与新prefix逐元素相等，旧index零次写回 |
| `TC-59` | **dense callback契约**：callback按kwargs签名；`_contiguous_ranges`拆分后用reason+interval containment命中；stale/residency/slice三reason直通唯一C40归因，不吞成copy_exception |
| `TC-60` | **exact/copy telemetry分离**：B-3/B-4 copy均`already_computed += island_len`；结束时cached_tokens==len(borrowed)，device breakdown闭合，island只计C40 metric |
| `TC-61` | **exclusive owner disclosure**：forced-owner轮后续请求均不入batch；记录exclusive_owner_round与batch size，summary/ledger与实际逐轮一致 |
| `TC-62` | **skip Radix策略**：中间chunk恒true；copy成功final抑制；无copy的full-dense完成后恢复original；pre-admission失败/纯retract正确复位 |
| `TC-63` | **pending chunk abort**：`process_pending_chunked_abort`中`to_finish=None`但`finished()==true`，仍计单一abort final cleanup、释放lease/snapshot、sticky audit与aborted outcome一致 |
| `TC-64` | **zero-length coverage省略**：B-3空dense-prefix与cold-cache空borrowed不实例化DenseRange，full coverage仍成立 |
| `TC-65` | **SWA fail-close**：所有SWA/PureSWA/Hybrid-SWA allocator在staging拒绝，非C40 hybrid parked baseline不受membership assert影响 |
| `TC-66` | **provenance manifest**：每个Gate/run的selected collector profile与实际Docker capability/security逐字段一致；不得拿plan-level max allowlist当实际值 |
| `TC-67` | **ephemeral scratch**：只允许未bind宿主的`/scratch`/`/tmp`写；任何额外host-rw或results根rw拒绝，容器退出无持久残留 |
| `TC-68` | **fallback chunk==island**：primary TransferSpan的chunk与span边界完全相同，三种base fallback均命中唯一interval rule |
| `TC-69` | **B-5 geometry兜底**：`target_end>=prompt_len`或任一未匹配几何均dense_ineligible，不进入copy transaction |
| `TC-70` | **suffix commit状态守卫**：DENSE_SUFFIX/FALLBACK snapshot只推进cursor到prompt_end，不与target_start比较；boundary_overrun只可能来自DENSE_PREFIX |
| `TC-71` | **B-3 projected prefix**：pre-gate cand/range均从target_end计算，commit后effective==target_end，extend_end不超过prompt_end |
| `TC-72` | **collector manifest字段**：policy字段、profiles、selected_profile_by_gate与allowed-max齐备；authority/differential两类命令分别匹配对应profile |
| `TC-73` | **mount schema双目标**：manifest恰有phase-dir rw、global-log file rw、worktree ro；文本与静态校验均使用“两处” |
| `TC-74` | **exact hit单次记账**：add_one首轮prefix_len=borrowed exact，所有add_chunked后续轮prefix_len=0；log_hit_tokens不重复累加 |
| `TC-75` | **pre-admission fallback reset**：source重验失败转fallback/ineligible且未admit时清全部target staging；helper因req_pool_idx=None只返回新prefix，旧index零写回 |
| `TC-76` | **provenance层级**：plan manifest含policy/oracle/capability对象；ToolEvent只含该次实际`collector_impl`与capability列表，不嵌入plan policy dict；类型逐字段校验 |
| `TC-77` | **pre-admission矩阵**：STAGED owner-conflict与B-3 fallback regate失败均转ADMISSION_DEFERRED并清target staging |
| `TC-78` | **projected alignment全seam**：cand、1101 input_tokens、1160-63 alignment在B-3 pre-gate均用scheduling_prefix_len，范围不越prompt |
| `TC-79` | **B-4 copy gate reason**：island+suffix capacity拒绝产生唯一c40_copy_budget_insufficient并本轮dense fallback，不记copy_exception |
| `TC-80` | **logprob sentinel**：-1允许C40；`0<=start<target_end`拒绝；`start>=target_end`后缀logprob对齐 |
| `TC-81` | **pre-admission fallback同轮admit**：source重验失败后clear target+state NONE，随后同轮dense admission使用当前exact prefix，snapshot/owned C40 hook不触发 |
| `TC-82` | **explicit skip-Radix seam**：staging不改flag；无HTTP metadata时成功admission原子保存original并置skip=true；pre-admission拒绝保持baseline |
| `TC-83` | **state schema完整**：staging_case/outcome_geometry/original_skip/approx_copy/suppress/live与sticky audit序列化、reset和property矩阵一致 |
| `TC-84` | **capability profile exactness**：authority runtime只用PTRACE profile，G1b differential用PTRACE+SYS_ADMIN；实际值不得等于未选择的max allowlist |
| `TC-85` | **Radix insertion arm policy**：D0/E0/C40-D normal；copy-committed R0/C40抑制；fully-dense fallback在prefill完成恢复original并final insert |
| `TC-86` | **skip复位生命周期**：pre-admission失败不改skip；pure retract release后恢复；copy成功保持到final；无copy fallback只在完整prefill后恢复 |
| `TC-87` | **Radix system metrics**：每臂exact-prefix-hit与suppressed-request计数、summary policy逐项匹配§15.1；策略漂移令engineering invalid |
| `TC-88` | **audit schema持久性**：state置NONE后primary/secondary/pending outcome、recovery_attempted与radix suppression仍在audit容器直至final写出 |
| `TC-89` | **geometry映射**：staging_case与outcome_geometry按B-3/B-4/CL-I唯一映射，manifest/telemetry不用内部case名 |
| `TC-90` | **族1/族2边界**：按audit.recovery_attempted判定；deferred后admission前失效为ineligible，进入commit后失败为attempted fallback，即使live state已NONE也不改族 |
| `TC-91` | **admission顺序**：owner/gate/precomputed AdmissionPlan全部在零资源STAGED完成；commit后到append无普通return，B-4首轮必钳制且无lease deferral |
| `TC-92` | **transaction rollback状态**：deferral恢复skip/live/lifecycle-ever/recovery-attempted/radix-suppression/pending到事务前；dense fallback仅保留attempted audit |
| `TC-93` | **中间stash系统效应**：每臂stash-suppressed chunk、exact hit、final insert policy完整记录；full-dense final可insert但tree shape差异不误称逐字段相同 |
| `TC-94` | **G0q/G1b alert时序**：G0q只产reference signatures无需裁决；G1b diff match生成后全部人工裁决才Exit |
| `TC-95` | **CL-I复用B-3生命周期**：clip后target_start=exact、source_offset同步、staging_case=CL-I、projected prefix=target_end、零dense-prefixcopy成功 |
| `TC-96` | **强制报告字段**：§19.5模板含exact-hit、stash-suppression、insert-suppression与policy；缺任一按§19.6 INVALID |
| `TC-97` | **decision CI method**：小n同时计算bootstrap/t，模板记录`ci_method=conservative_of_both`及最终L90/U90；缺失即INVALID |
| `TC-98` | **collector profile替代**：ebpf profile的BPF/PERFMON与security opts分字段，选择顺序第三步可达；run实际profile精确绑定 |
| `TC-99` | **quality coverage派生量**：effective-task-rate与per-task copy_coverage公式固定，分布与w_quality分列且不互换 |

---

## 9. 结构化 Provenance Schema 与路径模型

### 9.1 ToolEvent schema v1（`proposed`，冻结后进 manifest）

```json
{
  "tool_provenance_schema_version": 1,
  "event_index": 17,
  "tool": "bash",
  "tool_success": true,
  "tool_exit_code": 0,
  "tool_timeout": false,
  "tool_truncated": false,
  "unknown_effect": false,

  "repo_id": "sha256:<canonical repo identity>",
  "worktree_id": "sha256:<worktree path + git dir identity>",
  "branch": "main",
  "commit": "0206f17b4255e4b248dafaaeb943be57428dae2f",
  "dirty": true,
  "worktree_generation": 17,

  "fs_events": [
    {"seq": 0, "op": "open",   "path": "pkg/a.py", "mode": "read"},
    {"seq": 1, "op": "open",   "path": "pkg/b.py", "mode": "read"}
  ],
  "read_paths":  ["pkg/a.py", "pkg/b.py"],
  "write_paths": [],
  "write_events_observed": 0,
  "write_then_restore_observed": false,
  "rename_pairs": [],
  "dir_write_roots": [],
  "symlink_resolutions": {"pkg/link.py": "pkg/a.py"},

  "path_content_sha256_before": {"pkg/a.py": "…", "pkg/b.py": "…"},
  "path_content_sha256_after":  {"pkg/a.py": "…", "pkg/b.py": "…"},

  "observation_role": "tool",
  "observation_char_len": 4213,

  "collector_authority_kind": "syscall_trace",
  "collector_impl": "strace_ptrace_v1",
  "collector_capabilities": ["SYS_PTRACE"],
  "collector_complete": true,
  "collector_version": 1,
  "secondary_checks": [
    {"method": "merkle_snapshot_v1", "agrees": true},
    {"method": "git_status_v2",      "agrees": true}
  ]
}
```

**字段约束（冻结）**：

| 字段 | 约束 |
| --- | --- |
| `collector_authority_kind` | 只能是 `wrapper_declared`（封闭工具）或 `syscall_trace`（任意 shell）。**不得**为 `merkle_snapshot` 或 `git_status` |
| `collector_complete` | trace 中途失败/丢事件/进程逃逸追踪 ⇒ `false` ⇒ 强制 `unknown_effect = true` |
| `fs_events` | event-level 原始序列（`seq` 单调）。`read_paths` / `write_paths` 由它派生，**不得**独立填写 |
| `write_events_observed` | 观察到的写事件计数；`> 0` 即进入 `effective_write_set`，**与内容哈希无关** |
| `write_then_restore_observed` | `write_events_observed > 0` 且 `sha256_before == sha256_after` 时为 `true`；该 group 仍然失效（§9.2.2） |
| `secondary_checks` | Merkle 与 git status 只能出现在这里；任一 `agrees == false` ⇒ 报警并 fail closed |

### 9.2 采集方式与 authority 层级（**冻结**）

#### 9.2.1 为什么 Merkle snapshot 不能作 authority

```text
Merkle / 内容快照的两个根本缺陷（决定它只能做 secondary check）：

(1) **不产生 read_paths**。
    读操作不改变文件系统状态，前后快照完全相同，因此
    "这次 tool observation 读了哪些文件"在快照里**根本不存在**。
    而 read_paths 是 C40 失效判定的核心输入（source_paths 由它得出）。

(2) **无法发现 write-then-restore**。
    命令写了文件又恢复成同样字节（例如 sed -i 后回滚、临时 patch 后 revert、
    build 产物覆盖后还原），前后快照一致，Merkle 判定为"无写"。
    但对 KV 复用而言，**发生过写事件本身**就意味着该文件在这段时间内
    存在过不同内容，中间任何被采集的 observation 都不可信。

因此：**Merkle 只能证明 final state，不能证明 event history。**
C40 需要的是 event history。
```

#### 9.2.2 冻结的 authority 层级

| 工具类别 | `read_paths` / `write_paths` 的 **authority** | 不满足时 |
| --- | --- | --- |
| **结构化封闭工具**（harness 自实现的 `read_file` / `grep` / `list_dir` / `apply_patch` / `write_file` 等） | **wrapper 声明的 read/write event 列表 + execution status**（工具语义封闭且已知，声明即事实） | wrapper 未声明或 status 缺失 ⇒ `unknown_effect = true` |
| **任意 shell / bash**（开放集合） | **event-level collector 是唯一 authority**。默认实现：Docker 内 `strace`/`ptrace` collector，追踪 `open/openat/openat2/creat/rename/renameat2/unlink/unlinkat/truncate/ftruncate/link/symlink/mkdir/rmdir/copy_file_range/chmod/chown` 及其写标志（`O_WRONLY`/`O_RDWR`/`O_CREAT`/`O_TRUNC`/`O_APPEND`），读事件由不带写标志的 `open*` 得出。**明确等价的 authority 替代实现仅有**：`fanotify`（`FAN_OPEN_PERM`/`FAN_MODIFY`，需 `SYS_ADMIN`）或 `eBPF`（`tracepoint/syscalls/sys_enter_open*` 等，需 `BPF`+`PERFMON`）。`LD_PRELOAD` shim 因漏静态链接/直接 syscall，**不得**作为 authority。实际 authority 必须在 manifest 声明 `collector_impl` 与 capability 并通过自检 | **collector 不可用、中途失败或 `collector_complete == false` ⇒ `unknown_effect = true` ⇒ 该 group ineligible 且其后候选一律 fail closed**。**没有**任何降级路径可以让 shell 类 group 在缺少 event-level collector 的情况下继续可用 |
| 其他外部工具 | 无结构化声明 | `unknown_effect = true` |

**Merkle snapshot 的角色（secondary only，冻结）**：

```text
Merkle snapshot **只**用于三件事，**绝不**作为 read/write authority：
  1. final-state content hash（path_content_sha256_before/after）
  2. worktree_generation 的生成与单调性
  3. integrity secondary check（与 event collector 结果不一致时报警并 fail closed）

明确的能力否定（必须逐字写入 evidence 与 disposition）：
  - Merkle **不提供** read_paths（读不改变状态）
  - Merkle **不能证明**"无瞬时写"（write-then-restore 前后快照一致）
  - 因此 Merkle **不能**单独判定任何 group 为 eligible
```

**write-then-restore 规则（冻结）**：

```text
若 trace 观察到对路径 p 的**任何写事件**（即使随后内容被恢复为同样字节，
即 sha256_before == sha256_after），p 仍然进入 effective_write_set，
该 group 及其后所有以 p 为 source_path 的候选一律失效。
理由：C40 的失效判定基于 **event history** 而非 final state。
内容哈希相同**不能**豁免写事件。
```

#### 9.2.3 Docker capabilities、默认命令与替代实现（必须在授权 manifest 中列明）

**默认 collector 的 Docker 运行形态（冻结）**：

```bash
# proposed，未授权执行
docker run --rm --user "$(id -u):$(id -g)" \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v "$WORKTREE":/w:ro \
  -v /home/chris/Workspaces/kvcache-research/results/phase7_5_c40:/results/phase7_5_c40:rw \
  -v /home/chris/Workspaces/kvcache-research/results/BENCHMARK_RUN_LOG.jsonl:/global_results/BENCHMARK_RUN_LOG.jsonl:rw \
  --tmpfs /scratch:rw,size=4g -w /scratch -e HOME=/tmp \
  ghcr.io/ccdd2023/sglang@sha256:0be6e16e2eb288dfd5fa8b0b41015f61731a139fb961d3366ccedf834289d781 \
  bash -c 'python -m benchmark.approx_kv.run_p75_selector_offline --collector strace_ptrace_v1 ...'
```

**capability 对照（`collector_impl` → 必需 capability）**：

| `collector_impl` | 必需 Docker 参数 | 备注 |
| --- | --- | --- |
| `strace_ptrace_v1`（**默认**） | `--cap-add=SYS_PTRACE --security-opt seccomp=unconfined` | 覆盖全部子进程（`-f`） |
| `fanotify_v1` | `--cap-add=SYS_ADMIN` | 需内核 `FAN_REPORT_FID` 支持 |
| `ebpf_v1` | `--cap-add=BPF --cap-add=PERFMON`（旧内核回退 `--privileged`，**须显式授权**） | 内核版本相关 |
| `preload_shim_v1` | 无额外 capability | **仅覆盖动态链接进程**；静态链接/直接 syscall 的进程会漏，必须在 evidence 标注该局限，且**不得**作为 primary authority，只能作差分 oracle |

```text
授权 manifest 必须显式列出实际使用的 collector_impl 与 capability 集合；
未列明的 capability 一律不得使用（§14.1 的
`provenance.collector_profiles + selected_profile_by_gate`）。

选择顺序（**这不是"降级到非 event 方法"的 fallback**，
          而是在**等价的 event-level 实现之间**选择）：
  1. strace_ptrace_v1   （默认）
  2. fanotify_v1        （若 ptrace 被安全策略禁用）
  3. ebpf_v1            （若前两者不可用且内核支持）
  4. 以上 event-level 实现**全部**不可用
     ⇒ **该 workload 的 shell 类 group 一律 unknown_effect = true ⇒ ineligible**
     ⇒ eligible 率会大幅下降，这是**正确的 fail-closed 行为**；
        **禁止**改用 Merkle / git status 顶替后继续声称 group 可用。
```

#### 9.2.4 层级汇总

| 层级 | 方法 | 状态 |
| --- | --- | --- |
| L1（core，必需） | 结构化封闭工具的 wrapper read/write event 声明 + execution status | authority |
| L2（core，必需） | 任意 shell 的 **event-level syscall trace** | authority |
| L3（core，必需） | Merkle snapshot → content hash / generation / integrity | **secondary only**；不提供 read_paths；不能证明无瞬时写 |
| L4（次级校验） | `git status --porcelain=v2 -z` 交叉检查 | **secondary check only** |

**差分测试 oracle 的独立性（硬性，防循环验证）**：

```text
被测采集器与 oracle 必须是**不同的 event collector 实现**：
  被测 = strace/ptrace collector   ⇒  primary oracle = fanotify 或 eBPF
  被测 = fanotify/eBPF collector   ⇒  primary oracle = strace/ptrace
  LD_PRELOAD shim 只可作为**第三个 supplemental oracle**，不得成为唯一 oracle，
  也不得用于计算 authority-level FN
**禁止**用 Merkle snapshot 或 git status 充当 oracle
  （它们不产生 read_paths，也漏 write-then-restore，无法作为 event 层 ground truth）
必须输出 oracle_agreement（两个 event collector 的一致率与分歧清单）；
分歧逐条裁决并记录。
```

### 9.3 失效判定（结构化，权威）

```text
候选岛 I 由 group g 产生，其 source_paths = read_paths(g)。
I 在 target 请求处仍然有效  ⟺  以下**全部**成立：

  (1) g.tool_success == true 且 g.tool_timeout == false
      且 g.tool_truncated == false 且 g.unknown_effect == false
  (2) g.write_events_observed == 0  且 g.write_paths == []
      且 g.rename_pairs == []  且 g.dir_write_roots == []
      （注意：内容哈希未变**不能**豁免写事件，§9.2.2）
  (3) source_paths != []                                  （空 ⇒ fail closed）
  (4) ∀ later group h (h.index > g.index):
        h.unknown_effect == false                          （否则 fail closed）
        effective_write_set(h) ∩ normalized(source_paths) == ∅
  (5) repo_id / worktree_id / branch 与 target 一致（ME-2 / CL-H）
  (6) ∀ p ∈ source_paths:
        sha256(p) at target time == g.path_content_sha256_after[p]
        （若任一路径缺 hash ⇒ fail closed）
  (7) worktree_generation(target) == g.worktree_generation
        或 (6) 已对全部 source_paths 逐路径证明未变
```

其中：

```text
effective_write_set(h) =
      normalized(paths of h.fs_events where mode == "write")   # event-level authority
    ∪ normalized(h.write_paths)                                 # 派生结果，应与上式相等
    ∪ { old, new  for (old,new) ∈ h.rename_pairs }
    ∪ expand(h.dir_write_roots)          # 目录级写展开到其下全部已知路径
    ∪ symlink_closure(h.write_paths)     # 符号链接双向闭包
```

### 9.4 路径规范化（`normalize`）

| 规则 | 说明 |
| --- | --- |
| N-1 | 全部归一到 **repo-relative** 规范路径；剥离 `/<repo_root>/` 前缀 |
| N-2 | `./x`、`x`、`/testbed/x`、`../repo/x` 归一到同一键 |
| N-3 | `..` / `.` 解析；不允许逃逸 repo root（逃逸 ⇒ `unknown_effect=true`） |
| N-4 | 符号链接解析到真实 target，并把 link 与 target **同时**加入受影响集合 |
| N-5 | Unicode NFC 规范化；大小写按文件系统语义处理（Linux 区分大小写） |
| N-6 | 幂等性：`normalize(normalize(p)) == normalize(p)`（property 测试 P4） |
| N-7 | 无法规范化 ⇒ `unknown_effect = true` ⇒ fail closed |

### 9.5 Rename / symlink / directory 建模

```text
rename (old → new)  ⇒  effective_write_set 同时包含 old 与 new
                       且 old 的 content hash 绑定立即失效
symlink L → T       ⇒  写 L 视为写 T；写 T 视为写 L
                       读 L 的 source_paths 同时记录 L 与 T
directory write D   ⇒  展开为 D 下**全部已知路径**；若 D 下存在未知路径，
                       该 group 标记 unknown_effect = true
git 操作            ⇒  checkout / restore / stash pop / revert / apply / am
                       一律通过 worktree generation + status diff 捕获，
                       **不依赖命令名判断**
```

### 9.6 命令正则的地位（降级为次级信号）

```text
命令字符串正则 **不是 authority**。

允许：作为 **只能加严、不能放宽** 的次级信号
      （例如：结构化判定为"纯读"，但正则怀疑有写 ⇒ 仍判无效）
禁止：用正则把结构化判定为"有写/未知"的 group 放宽为可用
禁止：把"未识别的命令"默认视为无写（当前 collaborator 实现正是如此，是 P0 根因）
禁止：用 Merkle / git status 的"内容未变"结论覆盖 trace 的写事件判定
禁止：继续向 deny-list 追加 perl/dd/truncate/rsync/install/ln/... —
      命令行写文件方式是**开放集合**，枚举必然继续漏
```

### 9.7 Fail-closed 默认表

| 情况 | 判定 |
| --- | --- |
| `unknown_effect == true` | **不可**成为 source；其后所有候选也失效 |
| shell 类 group 无 event-level collector（`collector_complete == false` 或无 collector） | `unknown_effect = true` ⇒ ineligible。**不存在**用 Merkle / git status 顶替的路径 |
| `collector_authority_kind` 为 `merkle_snapshot` / `git_status` | schema 非法 ⇒ 整请求禁用 C40 |
| `write_events_observed > 0`（即使 `sha256_before == sha256_after`） | 该路径进入 `effective_write_set`，相关候选失效 |
| 任一 `secondary_checks[].agrees == false` | 报警并 fail closed |
| 非 bash 工具且无结构化 provenance | `unknown_effect = true` |
| 工具超时 / 输出被截断 | `unknown_effect = true` |
| 缺 `path_content_sha256_after` | 该候选失效 |
| 缺 `worktree_generation` | 该候选失效 |
| 路径无法规范化 | `unknown_effect = true` |
| `repo_id` / `worktree_id` / `branch` 不匹配 | 该候选失效 |
| fingerprint 任一字段缺失 | 整个请求禁用 C40 |

---

## 10. Identity / Fingerprint / Approx Depth

### 10.1 Segment key（复用底座 `KVSegmentKey`，新增 C40 字段）

```text
C40 segment identity =
    base(KVSegmentKey)                       # 底座既有
  + c40_fingerprint_sha256                   # §10.2
  + repo_id / worktree_id / branch
  + worktree_generation
  + source_paths_content_sha256 (canonical, 排序后哈希)
  + segment_token_sha256
  + source_prefix_token_sha256               # 左上下文标识（用于 cross-context 断言）
  + provenance ∈ {EXACT, APPROXIMATE}
  + approx_depth ∈ {0,1,2,...}
```

**注册与查询规则**：

- 同 key 重注册 ⇒ 产生**新的 `generation`**；stale handle 不能 `pin` / `load` / `release`；
- `token_hash` / `token_count` 不匹配 ⇒ 拒绝；
- 缺 `source_paths_content_sha256` 或 `worktree_generation` ⇒ **拒绝（fail closed）**。

### 10.2 Fingerprint（ME-3，缺一即禁用 C40）

```json
{
  "c40_fingerprint_version": 1,
  "model_id": "Qwen/Qwen3-0.6B",
  "model_revision": "c1899de289a04d12100db370d81485cdf75e47ca",
  "tokenizer_revision": "<...>",
  "chat_template_sha256": "<...>",
  "rope_config_sha256": "<...>",
  "kv_dtype": "float16",
  "kv_layout": "<...>",
  "head_dim": 128,
  "rotary_dim": 128,
  "tp_size": 1,
  "pp_size": 1,
  "page_size": 1,
  "image_digest": "sha256:0be6e16e2eb288dfd5fa8b0b41015f61731a139fb961d3366ccedf834289d781",
  "code_pin_commit": "<phase7.5 branch commit>",
  "code_pin_tree": "<tree sha>"
}
```

**规则**：

1. `c40_fingerprint_sha256 = sha256(canonical_json(above))`；
2. 跨 fingerprint 复用**必须拒绝**，记 `c40_fingerprint_mismatch`；
3. **禁止 fingerprint bypass**：任何允许"跳过 fingerprint 校验"的代码路径、
   环境变量或测试钩子都视为 P0（§21 Stop rule）。

### 10.3 Approx depth 规则（ME-7）

| 场景 | `provenance` | `approx_depth` | primary lane 允许消费？ |
| --- | --- | --- | --- |
| dense/exact 请求物化的 source | `EXACT` | `0` | **是** |
| 使用过 C40 copy 的请求物化的 source | `APPROXIMATE` | `parent_depth + 1` | **否**（记 `c40_source_depth_exceeded`） |
| chaining diagnostic lane 开启 | `APPROXIMATE` | `<= max_chain_depth` | 仅 diagnostic，不进 headline |

---

## 11. 配置、环境变量与 Feature Gate

### 11.1 环境变量（`proposed`，全部新命名）

| 变量 | 默认 | 说明 | 层 |
| --- | --- | --- | --- |
| `SGLANG_APPROX_KV_C40` | `0` | C40 总开关；`0` 时 selector 不被调用且 `copied_tokens ≡ 0` | Core |
| `SGLANG_APPROX_KV_C40_MANIFEST` | `""` | C40 **decision** manifest 路径（§14.2b；必须来自 CLI/env，**禁止硬编码**） | Core |
| `SGLANG_APPROX_KV_C40_PLAN_MANIFEST` | `""` | C40 **授权/plan** manifest 路径（§14.1；runtime 授权门读取） | Core |
| `SGLANG_APPROX_KV_C40_MANIFEST_MODE` | `file` | decision manifest 模式：`file`（单文件 + 全局 sequence，串行 harness）/ `dir`（每请求一份，`<dir>/<request_uid>.json`，per-uid sequence）。`dir` 是 CL-E concurrency / repo isolation 的**硬前置**（§14.2b 规则 8） | Core / CL-E |
| `SGLANG_APPROX_KV_C40_MIN_TOKENS` | `128` | 岛最小 token 数（`>= 32`） | Core |
| `SGLANG_APPROX_KV_C40_COPY_CAP` | `4096` | 单岛 copy 上限（`>= 128`） | Core |
| `SGLANG_APPROX_KV_C40_ROLLING_GROUPS` | `6` | rolling 窗口 group 数（`>= 4`） | Core |
| `SGLANG_APPROX_KV_C40_STRICT_MIDDLE` | `1` | 强制严格中部；`0` 仅供 negative test | Core |
| `SGLANG_APPROX_KV_C40_VALIDATION` | `0` | 调试/测试专用高成本ownership断言；measured run必须为0，禁止在scheduler关键路径做`.item()`/D2H sync | Core debug |
| `SGLANG_APPROX_KV_C40_MAX_ISLANDS` | `1` | `1` = primary 单岛；`>1` 启用 CL-A | CL-A |
| `SGLANG_APPROX_KV_C40_TOTAL_COPY_BUDGET` | `0`（= 用 `COPY_CAP`） | 多岛总 copy 预算 | CL-A |
| `SGLANG_APPROX_KV_C40_REPAIR_K` | `0` | `k>0` 启用 `C40-R1-k`（复用 EPIC，`k ∈ {0,2,4,8,16,32}`） | CL-B |
| `SGLANG_APPROX_KV_C40_AST_GATE` | `0` | AST span/path dependency 辅助 gate | CL-C |
| `SGLANG_APPROX_KV_C40_EMBED_GATE` | `0` | embedding distance gate（**保留，AST 不替代**） | CL-C |
| `SGLANG_APPROX_KV_C40_HOST_DEMAND` | `0` | host demand-load（`H1`） | CL-D |
| `SGLANG_APPROX_KV_C40_PREFETCH_HINT` | `0` | prefetch-neutral hint 接口（**不得改变 selected span**） | CL-D |
| `SGLANG_APPROX_KV_C40_EXACT_OVERLAP_CLIP` | `0` | **CL-I**：B-2（`target_start < exact_length < target_end`）时是否把 island 裁剪到 `[exact_length, target_end)` 后复用。`0` = primary 的 fail-closed 行为（`c40_exact_overlap_unsupported`）；`1` = 启用裁剪，outcome 带 `geometry = clipped_at_exact_boundary`，**不进入** primary headline | CL-I |
| `SGLANG_APPROX_KV_C40_MAX_CHAIN_DEPTH` | `0` | source chaining 诊断最大深度 | CL-F |
| `SGLANG_APPROX_KV_C40_QUALITY_GATE` | `0` | quality-calibrated gate | CL-G |
| `SGLANG_APPROX_KV_C40_REPO_ISOLATION` | `1` | repo/worktree/branch 隔离（默认开） | CL-E/H |

**硬性**：plan manifest 与runner对任何latency/mechanism/system setting要求
`SGLANG_APPROX_KV_C40_VALIDATION == 0`；值为1只允许CPU/property/debug canary，
不得进入`theta_j`、`E_work`或headline。

**依赖约束（构造期校验，失败即 `ValueError`）**：

```text
C40=1                     requires SGLANG_APPROX_KV_CORE=1
C40_HOST_DEMAND=1         requires SGLANG_APPROX_KV_HOST=1
C40_REPAIR_K>0            requires SGLANG_APPROX_KV_EPIC=1 且 k ∈ SUPPORTED_EPIC_K_VALUES
C40_MAX_ISLANDS>1         requires C40=1
并发/多租户实验         requires C40_MANIFEST_MODE=dir
C40_EXACT_OVERLAP_CLIP=1  requires C40=1；且该请求的 outcome 必须带
                          geometry=clipped_at_exact_boundary，
                          并被排除在 primary headline 之外
C40_AST_GATE=1            requires C40_EMBED_GATE=1
                          （AST 是**辅助**信号，不得在 embedding gate 关闭时单独生效；
                            AST 只能进一步**拒绝**，不得放宽 embedding gate 的判定）
C40_PREFETCH_HINT=1       不得要求 SGLANG_APPROX_KV_PREFETCH=1（hint 必须 prefetch-neutral）
```

### 11.2 真正的 Feature Gate（ME-4）

```text
断言测试（CPU，必须存在）：
  T-GATE-1  C40=0 时：selector 未被调用（调用计数器 == 0）
  T-GATE-2  C40=0 时：copied_tokens == 0 且无任何 c40_* metric 被写
  T-GATE-3  C40=0 时：请求路径与baseline完全一致；特别覆盖approx-KV
            REGISTER请求的retract，确认source registration未被C40抑制
  T-GATE-4  C40=1 但 manifest 为空：安全降级为 dense，记 c40_manifest_absent
  T-GATE-5  不存在任何"名义存在但未接线"的 flag（dead flag 静态检查）
```

### 11.3 Runtime 写路径约束

```text
宿主持久化写目标**恰好两处**：

  (a) /results/phase7_5_c40/**            普通 artifact（raw / logs / central /
                                          compact / frozen / summary / evidence）
                                          读写不限模式
  (b) /global_results/BENCHMARK_RUN_LOG.jsonl
                                          **仅允许 append**，且**仅** run-level 事件
                                          （§13.5.0）；禁止 truncate / seek 重写 /
                                          删除；禁止写入 request-level 记录；
                                          禁止在该挂载点写任何其他文件

容器内允许`/scratch/**`与`/tmp/**`的tmpfs/ephemeral写，用于复制只读worktree、
临时venv/cache与对抗语料mutation；它们不得bind到宿主、不得进入结果claim，
容器必须`--rm`，退出后全部销毁。除此之外的host-rw挂载一律拒绝。

对 CR-8 冻结清单（phase2 / phase3 / phase4-r1 / phase5-scheduler / phase6 / phase7）
内任何路径的写操作必须被断言拒绝；注意 phase7_5_c40 **不在**该清单内
所有输入/输出路径来自 CLI 或环境变量；禁止 /home/<user>/ 硬编码（CR-4）
```

---

## 12. Terminal Reason 与 Outcome Taxonomy

### 12.1 设计规则

1. 每次 **attempted** dense fallback **恰好**一个 `primary_reason`（exclusive）；ineligible 请求另用 `selector_reason`，二者**不合并**；
2. `Σ c40_* + Σ inventory_reason == attempted_recovery_failed_dense_tokens`，
   **无双计**；其中 `inventory_reason` 的定义域来自 §2.6 自动生成的 canonical
   inventory。**该恒等式只对族 2（attempted fallback）成立**（§12.4）；
3. 所有 `c40_*` reason 与 canonical inventory 及其 prefix-family**互斥且不重名**；
4. **direct** 归因优先：能在触发点直接确定的 reason 不得被上游泛化 reason 吞掉
   （Phase7 的 `unsupported <- store_miss` correction 教训）；
5. **多因并发时按冻结 precedence 表取唯一 `primary_reason`**，其余写入
   **非计数**字段 `secondary_reasons[]`（只用于诊断，不进任何求和）。

#### 12.1.1 冻结的 precedence 表（数字小者优先）

| 优先级 | 类别 | 说明 |
| ---: | --- | --- |
| `10` | fingerprint / manifest（B 类） | 环境不一致时其他判定无意义，必须最先归因 |
| `20` | provenance fail-closed（`unknown_effect` / timeout / truncated / hash missing） | 安全性最高，先于任何几何判定 |
| `30` | invalidation（later write / generation / content hash mismatch / repo mismatch） | |
| `40` | 证据资格（tool 不成功 / mixed group / 长度不足 / assistant 污染） | |
| `50` | 几何与预算（唯一性 / strict-middle / min-cap / overlap / budget） | |
| `60` | 辅助 gate（AST / embedding / quality） | 只在前面全部通过后才可能触发 |
| `70` | identity / depth（generation stale / depth exceeded / provenance rejected） | |
| `80` | runtime 状态机（slot / token slice / residency / mechanical / ownership / lease） | |
| `90` | scheduler 终止（`c40_aborted_by_scheduler`） | 最后归因，避免吞掉更具体的原因 |

同一优先级内出现多因 ⇒ 按 §12.2 表内**自上而下**的书写顺序取第一个。
precedence 表本身必须冻结进 manifest（`reason_precedence_sha256`）。

#### 12.1.2 计量单位（必须分开，禁止混计）

```text
族 1  c40_ineligible_requests_total{selector_reason}          单位 = 请求数
族 2  c40_attempted_fallback_requests_total{reason,phase}     单位 = 请求数
      c40_attempted_fallback_tokens_total{reason,phase}       单位 = token 数
族 3  c40_aborted_requests_total{path}
      c40_terminal_rejected_requests_total
      c40_admission_deferred_total                            单位 = 事件数
      c40_retraction_events_total                             单位 = 事件数
族 4  c40_invalid_engineering_total{kind}                     单位 = 事件数

恒等式（各族独立封闭，见 §12.4）：
  族 1: Σ == count(dense_ineligible)
  族 2: Σ requests == count(dense_attempted_fallback)
        Σ tokens   == attempted_recovery_failed_dense_tokens   ← 唯一 token 恒等式
  族 3: 见 §12.4 族 3

禁止：把 request-count 与 token-count 相加或互相替代
禁止：把族 1 / 族 3 计入族 2 的 token 恒等式
```

### 12.2 C40 terminal reason 表（`proposed`）

#### A. Selector / 准入阶段（请求进入 server 前即可确定）

| Reason | 触发 |
| --- | --- |
| `c40_no_candidate_group` | rolling 窗口内无成功只读候选 |
| `c40_insufficient_rolling_history` | 窗口 group 数不足 |
| `c40_unknown_effect_fail_closed` | 存在 `unknown_effect=true` 的 group |
| `c40_tool_not_successful` | `tool_success=false` / 非零 exit code |
| `c40_tool_timeout_or_truncated` | 超时或输出截断 |
| `c40_mixed_read_write_group` | 同 group 内既读又写 |
| `c40_later_same_path_write` | 后续 group 写了 source 路径 |
| `c40_worktree_generation_changed` | generation 变化且未逐路径证明未变 |
| `c40_content_hash_mismatch` | source 路径内容哈希与注册时不一致 |
| `c40_content_hash_missing` | 缺 before/after 哈希 |
| `c40_repo_or_worktree_mismatch` | repo/worktree/branch 不一致 |
| `c40_path_normalization_failed` | 路径无法规范化 |
| `c40_empty_source_paths` | 未提取到任何 repository 路径 |
| `c40_observation_too_short` | observation 长度低于阈值 |
| `c40_below_min_tokens` | token 数 `< min_tokens` |
| `c40_not_unique_in_target` | 在 target prompt 中出现 0 次或 >1 次 |
| `c40_span_not_strictly_middle` | 触及 prompt 首或尾 |
| `c40_exact_overlap_unsupported` | **B-2**：`target_start < exact_length < target_end`；primary lane fail-closed，不 clip（clip 属 CL-I） |
| `c40_logprob_range_unsupported` | `0 <= logprob_start_len < target_end`，prompt-logprob区间覆盖copied island；sentinel -1不触发 |
| `c40_unsupported_scheduler_mode{subreason}` | subreason冻结为`host_load_back` / `hybrid_or_mamba` / `swa_allocator` / `dllm` / `ignore_eos_disabled_tree` / `session` / `non_tree_cache` / `disaggregation` / `hisparse_or_speculative` / `chunked_prefill_disabled` / `page_size_not_one` / `mixed_chunk_enabled` / `dynamic_chunking_enabled` / `deterministic_alignment` / `multimodal_or_encoder`；primary不进入C40 |
| `c40_assistant_token_contamination` | 候选包含 assistant / tool_call token |
| `c40_island_overlap_rejected` | [CL-A] 多岛重叠 |
| `c40_copy_budget_exhausted` | [CL-A] 超总 copy 预算 |
| `c40_ast_gate_rejected` | [CL-C] AST 依赖 gate 拒绝 |
| `c40_embed_gate_rejected` | [CL-C] embedding distance gate 拒绝 |
| `c40_quality_gate_rejected` | [CL-G] quality-calibrated gate 拒绝 |

#### B. Identity / fingerprint 阶段

| Reason | 触发 |
| --- | --- |
| `c40_manifest_absent` | 开关开但 manifest 缺失 |
| `c40_manifest_schema_invalid` | manifest 版本 / 字段不合法 |
| `c40_fingerprint_mismatch` | fingerprint 不一致 |
| `c40_fingerprint_incomplete` | fingerprint 字段缺失 |
| `c40_source_generation_stale` | source generation 已过期 |
| `c40_source_depth_exceeded` | `approx_depth > 0`（primary lane） |
| `c40_source_provenance_rejected` | provenance 不被 primary lane 接受 |

#### C. Runtime / 状态机阶段

| Reason | 触发 |
| --- | --- |
| `c40_missing_request_pool_slot` | **非capacity-deferral场景**中，已确认`added=true`或copy transaction已commit后`req_pool_idx`意外丢失；slot耗尽仍走ADMISSION_DEFERRED且不记此reason |
| `c40_dense_prefix_failed` | dense prefix 阶段失败 |
| `c40_rematch_invalidated_plan` | rematch 使冻结 plan 失效 |
| `c40_token_slice_mismatch` | source/target token slice 不相等 |
| `c40_source_not_resident` | source 未 device-resident 且加载失败 |
| `c40_provisional_alloc_failed` | provisional 槽分配失败（且 cross-store 未记账时才记） |
| `c40_copy_budget_insufficient` | B-4 pre-copy capacity gate无法同时容纳island+planned suffix；未执行copy，直接dense fallback |
| `c40_mechanical_validation_failed` | `copied_k != len` 或 `mechanically_valid == false` |
| `c40_copy_exception` | copy 过程异常（清理后 `→ DENSE_ISLAND_FALLBACK`；仅在清理失败或不变量已破坏时才 re-raise） |
| `c40_ownership_commit_failed` | copied slot 未能在 suffix 前转正 |
| `c40_unsupported_cache_protection_geometry` | staging 后 `cache_protected_len != len(prefix_indices)`；primary 无法证明 borrowed/owned 分界，fail closed |
| `c40_lease_expired_midflight` | 执行途中 lease 过期 |
| `c40_internal_exception` | C40 非 copy 路径的未预期异常（同上处置） |
| `c40_aborted_by_scheduler` | abort / reset / timeout（`→ ABORTED_TERMINAL`） |
| `c40_terminal_rejected` | scheduler 终局拒绝（`→ TERMINAL_REJECTED`） |
| `c40_boundary_overrun` | dense prefix 越过 `target_start`（§8.6.3 步骤 4）；同时记族 4 `c40_invalid_engineering_total{kind="boundary_overrun"}` |
| `c40_prefill_completion_misclassified` | `extend_range.end < prompt_end` 但未登记 middle chunk / `inflight_middle_chunks <= 0`；fail loud，并记族 4 `kind="prefill_completion_misclassified"` |
| `c40_retraction_reset` | 非terminal诊断事件：pressure retract/preempt触发C40 state reset；只计`c40_retraction_events_total`，不产生request outcome或fallback token |
| `c40_cleanup_failed_unrecoverable` | 清理本身失败或底座不变量已破坏，异常上抛。这是**唯一**允许 re-raise 的路径；只记族4 `cleanup_failed` 并使整块engineering invalid，request outcome/族3恒等式不再用于该无效块（SR-3） |

#### D. Produce 阶段（不计入 consume fallback，单独计数族）

| Reason | 触发 |
| --- | --- |
| `c40_produce_skipped_approx_depth` | 因 depth 规则不物化下一 source |
| `c40_produce_store_capacity` | store 容量不足 |
| `c40_produce_no_valid_span` | 无合格候选 span |

### 12.3 Outcome taxonomy（每请求恰好一个）

```text
C40Outcome ∈ {
  "exact",                 # 完全 exact 命中，未进入 C40 判定路径
  "c40_copied",            # C40 成功：dense_prefix + copied island + dense_suffix
                           #   附 geometry ∈ {strict_middle,
                           #                  contiguous_at_exact_boundary,
                           #                  clipped_at_exact_boundary[CL-I]}
  "c40_copied_multi",      # [CL-A] 多岛成功
  "c40_copied_repaired",   # [CL-B] 单岛 + leading-k repair
  "dense_ineligible",      # selector 判定不合格，**从未尝试** recovery
  "dense_attempted_fallback",  # 曾进入 attempted recovery 但回落 dense
                               #   （DENSE_ISLAND_FALLBACK）
  "dense_disabled",        # C40 关闭
  "aborted",               # ABORTED_TERMINAL（abort / reset / timeout）
  "terminal_rejected"      # TERMINAL_REJECTED（scheduler 终局拒绝）
}

注 1：ADMISSION_DEFERRED 与 retract/preempt **不产生** outcome
      （请求尚未完成，会重试）；同一请求最终只能产生一个 outcome。
注 2：§8.6.2 的 B-1（island 完全落在 exact 前缀内）产生 outcome == "exact"，
      并递增诊断计数器 c40_skipped_covered_by_exact_total，
      **不**计入 attempted fallback，也**不**计入 ineligible。
```

### 12.4 四个独立计数族（**互不混合，恒等式各自封闭**）

此前把"ineligible"与"fallback"混在同一恒等式里会同时污染两者。冻结为四族：

#### 族 1 — Ineligible dense（**不是 fallback**）

```text
定义：在进入`c40_on_admission_commit`之前判定不合格的请求，
      即`recovery_attempted=false`。它们可能曾持有source-side staging plan，
      但从未分配target slot、从未pin source、从未进入identity/runtime attempt。
计数：c40_ineligible_requests_total{selector_reason}
恒等式：
  Σ_reason c40_ineligible_requests_total == count(outcome == "dense_ineligible")
**不进入**任何 token 口径的 fallback 恒等式。
理由：它们没有"本可复用而未复用的 token"这一量 —— 根本没有选出 span。
      把它们计入 fallback token 会凭空制造分母。
```

#### 族 2 — Attempted C40 fallback（**唯一的 token 恒等式所在**）

```text
定义：`recovery_attempted=true`，已进入admission identity校验或runtime状态机，
      且audit pending/final outcome为`dense_attempted_fallback`的请求。
      live state可为DENSE_ISLAND_FALLBACK，也可在pre-admission rollback后为NONE；
      归属只看sticky audit，不看当前state。
计数：
  c40_attempted_fallback_requests_total{terminal_reason, phase}
  c40_attempted_fallback_tokens_total{terminal_reason, phase}
      （token = 该请求本可复用的 island 长度）
恒等式（**唯一的 token 恒等式**）：
  Σ_reason c40_attempted_fallback_requests_total
      == count(outcome == "dense_attempted_fallback")
  Σ_reason c40_attempted_fallback_tokens_total
      == attempted_recovery_failed_dense_tokens
每个 attempted fallback 恰好一个 primary_reason（§12.1 precedence）；
secondary_reasons[] 不参与求和。
```

#### 族 3 — Request-outcome 终止族（**不进入任何 fallback 恒等式**）

```text
定义：请求未正常完成的终局分类；retract/preempt 不在本族。
计数：
  c40_aborted_requests_total{path}          path ∈ {abort, reset, timeout}
  c40_terminal_rejected_requests_total
  c40_admission_deferred_total              （**事件计数，非请求 outcome**，
                                              同一请求可多次递增）
  c40_retraction_events_total               （**事件计数，非请求 outcome**，
                                              同一请求可多次递增）
恒等式：
  count(outcome == "aborted")            == Σ_path c40_aborted_requests_total
  count(outcome == "terminal_rejected")  == c40_terminal_rejected_requests_total
  c40_retraction_events_total             == c40_cleanup_total{path="retract"}
**不**计入族 1 与族 2 的任何恒等式；
**不**贡献 attempted fallback token。
```

#### 族 4 — Invalid engineering（**实验有效性标记，不是请求分类**）

```text
定义：使该 run/block 的**测量**失效的工程事件。
计数：
  c40_invalid_engineering_total{kind}
      kind ∈ {capacity_error, accounting_mismatch, orphan_detected,
              lease_leak, telemetry_gap, fingerprint_bypass_detected,
              boundary_overrun, prefill_completion_misclassified,
              cleanup_failed}
规则：任一 kind > 0 ⇒ 该 restart / launch block 标记 engineering_valid = false，
      **整块**不进入统计（而不是剔除个别请求）。
```

#### 12.4.1 总恒等式

```text
count(exact) + count(c40_copied*) + count(dense_ineligible)
  + count(dense_attempted_fallback) + count(dense_disabled)
  + count(aborted) + count(terminal_rejected)
  == n_requests_observed

`ADMISSION_DEFERRED` 与 `retract` 是事件而非 outcome，不出现在上式；
经历任意次 retract 的同一 request 最终仍只贡献一个 outcome。
```

#### 12.4.2 Arm-specific abort 的处理（**禁止静默排除**）

```text
问题：若只有 C40 臂发生 abort/terminal_reject，而 Dense 臂没有，
      直接从两臂中"同时剔除"该请求会破坏配对；
      只从 C40 臂剔除则制造 survivorship。

冻结规则（二选一，执行前在 manifest 中冻结，不得事后选择）：

规则 A（默认，latency 实验）：
  任一臂在某个 paired restart / launch block 内发生 arm-specific abort
  或 terminal_reject ⇒ **该 paired block 整体标记 invalid**，
  不进入 theta_j 的计算；block 数与原因写入 summary。
  若 invalid block 比例 > 20%，该 cell 判 INVALID（不进 disposition）。

规则 B（quality 实验）：
  arm-specific abort 按**预注册的 task failure** 计入该臂
  （即算作该 task 在该臂上 not resolved），而不是丢弃该 task pair。
  这保持 McNemar 的配对完整性。

retract/preempt 不是 abort：请求只要最终完成就保留在 pair 内，其全部重算/
排队延迟计入端到端时间；只有最终 abort/terminal_reject 才触发规则 A/B。

**禁止**：
  - 静默从 All 中排除 aborted 请求（无论单臂还是双臂）
  - 事后根据哪种处理方式结果更好来选择规则
  - 把 aborted 计入 attempted fallback token
```

#### 12.4.3 对 `mu_theta` / `E_cond` 的影响

```text
theta_j / mu_theta / E_cond 的 All 集合定义（冻结）：
  All_j = 该 restart 内**两臂都正常完成**的 formal 请求
  若某请求在任一臂 aborted / terminal_rejected
      ⇒ 按 §12.4.2 规则 A 使**整个 paired block invalid**
      ⇒ **不是**把该请求从 All 中悄悄删掉
  ADMISSION_DEFERRED 不影响 All（请求最终仍会完成）
  retract/preempt 不从 All 删除请求；其重算与等待成本自然计入该臂
      end_to_end_ms，并单独报告 retraction event 数
  dense_ineligible 请求**仍在 All 中**（它们是 workload 的真实组成部分，
      且正是 w < 1 的来源）
```

## 13. Telemetry 与 Metrics

### 13.1 Counters（`proposed`）

| Metric | 说明 |
| --- | --- |
| `c40_requests_total{outcome}` | 按 outcome 分类的请求数 |
| `c40_ineligible_requests_total{selector_reason}` | 族 1：从未尝试 recovery |
| `c40_attempted_fallback_requests_total{reason,phase}` | 族 2 请求口径；`phase ∈ {identity,runtime}` |
| `c40_attempted_fallback_tokens_total{reason,phase}` | 族 2 token 口径（**不得与请求数相加**） |
| `c40_aborted_requests_total{path}` | 族 3；`path ∈ {abort,reset,timeout}` |
| `c40_terminal_rejected_requests_total` | 族 3 |
| `c40_admission_deferred_total` | 族 3 事件计数（同一请求可多次递增） |
| `c40_retraction_events_total` | 族 3 非终局事件计数（同一请求可多次递增，不产生 outcome） |
| `c40_invalid_engineering_total{kind}` | 族 4：使该 block 测量失效 |
| `c40_cleanup_total{path}` | final按`req.c40_audit.lifecycle_ever_entered`、retract按live；前五final互斥，retract可重复 |
| `c40_copied_tokens_total` | 成功复制的 token 数 |
| `c40_copied_islands_total` | 成功复制的岛数 |
| `c40_dense_prefix_tokens_total` / `c40_dense_suffix_tokens_total` | dense 覆盖 token |
| `c40_source_registered_total{provenance,depth}` | 物化的 source 数 |
| `c40_source_rejected_total{reason}` | produce 阶段拒绝 |
| `c40_lease_gc_total` | 自动 GC 回收的 lease 数 |
| `c40_state_transitions_total{from,to}` | 状态机转移 |
| `c40_skipped_covered_by_exact_total` | §8.6.2 B-1：island 完全被 exact 前缀覆盖（诊断，**非** terminal reason） |
| `c40_copied_total{geometry}` | `geometry ∈ {strict_middle, contiguous_at_exact_boundary, clipped_at_exact_boundary}` |
| `c40_exclusive_owner_rounds_total` | forced-middle为保持底座single-owner而独占chunk budget的轮数；同时记录该轮batch size |
| `p75_radix_insert_suppressed_requests_total{arm,outcome}` | harness级因approx copy committed而抑制final exact Radix insertion的请求数 |
| `p75_radix_stash_suppressed_chunks_total{arm,outcome}` | 因C40执行中间chunk skip而未调用cache_unfinished_req的chunk数 |
| `p75_exact_prefix_hit_tokens_total{arm}` | harness级每臂exact prefix hit tokens；非`c40_*`，C40关闭时也可记录 |
| `c40_aborted_requests_total{path}` 的边界 | `rematch` 与普通 `exception` 走 `DENSE_ISLAND_FALLBACK`，**不计入** abort |

### 13.2 Gauges

| Metric | 说明 |
| --- | --- |
| `c40_active_leases` | 当前consume lease数（terminal、fallback或retract reset后必须归零） |
| `c40_pending_produce` | 待物化 source 数（terminal 后归零） |
| `c40_provisional_slots` | 未转正 provisional 槽（每轮末归零或明确归属） |
| `c40_store_records{provenance}` | store 记录数 |
| `c40_orphan_count` | 孤儿记录数（**必须恒为 0**） |

### 13.3 Histograms / Timers（selector overhead 必测）

| Metric | 说明 |
| --- | --- |
| `c40_selector_total_ms` | selector 端到端耗时，**在 ineligible 请求上也必须测量** |
| `c40_selector_provenance_ms` | provenance 解析与失效判定 |
| `c40_selector_tokenize_ms` | token 化 |
| `c40_selector_unique_search_ms` | 唯一性搜索 |
| `c40_selector_manifest_io_ms` | manifest I/O |
| `c40_admission_guard_ms` | `_lock_node`内borrowed/current-prefix equality guard耗时；必须计入request path/end-to-end |
| `c40_controller_stage_ms{stage}` | 各状态阶段耗时 |
| `c40_copy_execute_ms` | `execute_reuse_plan` 耗时 |
| `c40_island_length_tokens` | 岛长度分布 |
| `c40_rope_delta_abs` | `|rope_delta|` 分布（**必须存在非零值**，否则不是 cross-context） |

### 13.4 四本账 latency ledger（继承 Phase7）

```text
target_only_ms          目标请求本身（server 侧）
request_path_ms         = seed_head_ms + target_only_ms      （server 侧，单独报告）
selector_total_ms       selector 端到端开销（客户端/harness 侧，**全部请求**都发生）
end_to_end_ms           = selector_total_ms + request_path_ms   ← **estimand 使用的时间**
full_lifecycle_ms       含 source 物化与清理
speedup_N (N ∈ {1,2,4,8})   必须实测，禁止插值外推
speedup_N = dense_total_N / (source_preparation + Σ_{i<=N} end_to_end_i)
break_even_N            若 N<=8 未观察到，写 ">8 / not_observed"
```

> **冻结口径**：`E_cond` / `theta_j` 的 `T_C40(i)` **必须**取 `end_to_end_ms`
> （即已包含 `C_selector`）；`T_dense(i)` 取对应臂的 `end_to_end_ms`（Dense 臂
> 的 `selector_total_ms = 0`）。server-only 的 `request_path_ms` 比值必须**另行
> 单独报告**，并显式标注 `excludes_selector_overhead=true`，**不得**充当 headline。

### 13.5 双层日志合同（**全局 run-level + phase request-level**）

#### 13.5.0 全局 authority run log（**每个 setting 必须追加**）

```text
路径（authority 固定，不可更改）：
  /home/chris/Workspaces/kvcache-research/results/BENCHMARK_RUN_LOG.jsonl

粒度：**run-level**（每个 setting / 每个 server start 一条或数条事件），
      **不写** request-level 记录（那属于 phase JSONL）。

每条事件至少包含：
  ts (ISO 8601), phase="phase7_5_c40", gate, stage, cell_id,
  run_id, setting_id, arm, restart_index, chunk, body, rho,
  image_digest, model_revision, code_pin_commit, code_pin_tree,
  plan_manifest_revision, c40_fingerprint_sha256,
  event ∈ {run_started, run_completed, run_failed},
  elapsed_s, gpu_equivalent_h, engineering_valid,
  phase_jsonl_path, phase_jsonl_sha256

写入规则：
  - **只追加**，绝不重写或删除既有行；
  - 每个 server start 至少一对 run_started / run_completed（或 run_failed）；
  - 与 phase JSONL 的 run_id 必须一一对应。

Docker 挂载与写权限（**必须在授权 manifest 中列明，未列明则不得挂载**）：
  -v /home/chris/Workspaces/kvcache-research/results/phase7_5_c40:
     /results/phase7_5_c40:rw
  -v /home/chris/Workspaces/kvcache-research/results/BENCHMARK_RUN_LOG.jsonl:
     /global_results/BENCHMARK_RUN_LOG.jsonl:rw
  **禁止**把整个宿主 `results/` 根目录以 rw 挂载
  上述**两处**是唯一允许的宿主可写挂载；其余宿主路径一律:ro或不挂载。
  manifest 字段：
    "mounts": [
      {"host": "/home/chris/Workspaces/kvcache-research/results/phase7_5_c40",
       "container": "/results/phase7_5_c40", "mode": "rw",
       "bind_type": "directory", "purpose": "phase_artifacts"},
      {"host": "/home/chris/Workspaces/kvcache-research/results/BENCHMARK_RUN_LOG.jsonl",
       "container": "/global_results/BENCHMARK_RUN_LOG.jsonl", "mode": "rw",
       "bind_type": "file", "purpose": "global_run_log_append_only"},
      {"host": "<worktree>", "container": "/w", "mode": "ro"}
    ]
  静态断言：任何mount若host等于宿主results根且mode=rw，manifest校验失败。
```

#### 13.5.1 Phase-local central JSONL（request-level）

```text
所有 runner 追加写入单一 phase central JSONL（append-only，不重写）。
每行一个事件，含：
  run_id / phase="phase7_5_c40" / arm / restart / request_index / ts / event / payload

每个 server start 有唯一 run_id，包含：
  image digest、model+tokenizer revision、code pin(commit+tree)、
  全部 SGLANG_* 环境变量快照、chunk / max-prefill / page_size / tp / pp /
  eviction policy、c40_fingerprint_sha256、plan manifest revision

每个请求记录：
  is_warmup / is_formal / repeat_index / eligible / skip_reason /
  四本账时间 / outcome / exclusive terminal reason /
  selector overhead 各分项 / island 元组(source_start,target_start,length,rope_delta)
  exact_prefix_hit_tokens / radix_stash_suppressed_chunks /
  radix_insert_suppressed / final_radix_insert_policy

文件级 sha256 进 manifest。
离线 consolidator 只读该 JSONL 与 raw 输出，产出自哈希 compact/summary。
```

#### 13.5.2 两层日志的绑定（**manifest 强制**）

```text
RESULT_MANIFEST / plan manifest 必须同时绑定：
  1. phase central JSONL 的文件级 sha256；
  2. 全局 BENCHMARK_RUN_LOG.jsonl 的
       - global_log_line_range: [first_line_no, last_line_no]（本 phase 写入的范围）
       - global_log_range_sha256: 该 line range 内容的 sha256
       - global_log_snapshot_sha256: 写入完成时刻整个文件的 sha256
     （全局日志是共享 append-only 文件，其他 phase 也会追加，
       因此绑定 line range + range hash + snapshot hash 三者，
       既能定位本 phase 的贡献，又能证明未改写他人记录。）
  3. 一致性断言：phase JSONL 中每个 run_id 都能在 global log 的
     line range 内找到对应的 run_started / run_completed 事件对。
```

---

## 14. Versioned Manifest Schema

### 14.1 C40 plan manifest（`c40_manifest_version = 1`）

```json
{
  "c40_manifest_version": 1,
  "plan_id": "P7.5-C40-V1",
  "revision": 1,
  "status": "pinned_blocked",
  "authorization": {
    "plan_authorized": true,
    "branch_creation_authorized": false,
    "implementation_authorized": false,
    "docker_test_execution_authorized": false,
    "gpu_execution_authorized": false,
    "authorized_gates": []
  },
  "cleanroom": { "...": "见 §4.6，policy_version=2" },
  "code_pin": { "branch": "research/phase7.5-c40-cleanroom",
                "commit": "<...>", "tree": "<...>",
                "base_commit": "0206f17b4255e4b248dafaaeb943be57428dae2f" },
  "environment": {
    "image_digest": "sha256:0be6e16e...",
    "model": "Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca",
    "chunked_prefill_size": 4096, "max_prefill_tokens": 4096,
    "page_size": 1, "tp": 1, "pp": 1, "enable_mixed_chunk": false,
    "eviction_policy": "<lru|hierarchical>", "hicache": false
  },
  "c40_fingerprint": { "...": "见 §10.2" },
  "tool_provenance_schema_version": 1,
  "provenance": {
    "collector_impl": "strace_ptrace_v1",
    "primary_oracle_impl": "fanotify_v1",
    "supplemental_oracle_impl": "preload_shim_v1",
    "collector_capabilities": {
      "strace_ptrace_v1": ["SYS_PTRACE"],
      "fanotify_v1": ["SYS_ADMIN"],
      "ebpf_v1": ["BPF", "PERFMON"],
      "preload_shim_v1": []
    },
    "collector_security_opts": {
      "strace_ptrace_v1": ["seccomp=unconfined"],
      "fanotify_v1": [],
      "ebpf_v1": [],
      "preload_shim_v1": []
    },
    "collector_profiles": {
      "authority_default": {
        "docker_capabilities": ["SYS_PTRACE"],
        "docker_security_opts": ["seccomp=unconfined"]
      },
      "differential_default": {
        "docker_capabilities": ["SYS_PTRACE", "SYS_ADMIN"],
        "docker_security_opts": ["seccomp=unconfined"]
      },
      "ebpf_authority": {
        "docker_capabilities": ["BPF", "PERFMON"],
        "docker_security_opts": []
      }
    },
    "selected_profile_by_gate": {
      "P7.5-G1b": "differential_default",
      "runtime_default": "authority_default"
    }
  },
  "selector_version": 1,
  "config": { "min_tokens": 128, "copy_cap": 4096, "rolling_groups": 6,
              "max_islands": 1, "repair_k": 0, "max_chain_depth": 0 },
  "prior": {
    "phase7_r0_reference": {
      "request_path_median": [0.7723, 0.7751, 0.9334, 0.9362],
      "n8_full_setup": [0.6086, 0.6419],
      "status": "NEGATIVE",
      "same_image": true, "same_model": true, "chunk": 4096
    }
  },
  "matrix": { "...": "§15 的 staged cells" },
  "budget": { "...": "§22" },
  "runners": [ {"path": "benchmark/approx_kv/run_p75_micro.py", "sha256": "<...>"} ],
  "mounts": [
    {"host": "/home/chris/Workspaces/kvcache-research/results/phase7_5_c40",
     "container": "/results/phase7_5_c40", "mode": "rw",
     "bind_type": "directory", "purpose": "phase_artifacts"},
    {"host": "/home/chris/Workspaces/kvcache-research/results/BENCHMARK_RUN_LOG.jsonl",
     "container": "/global_results/BENCHMARK_RUN_LOG.jsonl", "mode": "rw",
     "bind_type": "file", "purpose": "global_run_log_append_only"},
    {"host": "<phase7.5 worktree>", "container": "/w", "mode": "ro"}
  ],
  "docker_capabilities_allowed_max": ["SYS_PTRACE", "SYS_ADMIN", "BPF", "PERFMON"],
  "docker_security_opts_allowed": ["seccomp=unconfined"],
  "logs": {
    "phase_central_jsonl": {"path": "central/p75-central.jsonl", "sha256": "<...>"},
    "global_run_log": {
      "path": "/home/chris/Workspaces/kvcache-research/results/BENCHMARK_RUN_LOG.jsonl",
      "line_range": [0, 0],
      "range_sha256": "<...>",
      "snapshot_sha256": "<...>"
    }
  },
  "statistics_freeze": {
    "s_pilot": null, "n_pilot": null, "method": null,
    "delta0_log": 0.048790, "delta1_log": 0.095310,
    "alpha_one_sided": 0.05, "power": 0.80, "n_min": 4,
    "n_confirmatory": null, "starts_needed": null,
    "computation_trace": null,
    "pre_data_frozen": true
  },
  "tolerance_freeze": {
    "baseline_envelope": {"dK_max": null, "dV_max": null,
                          "dlogit_max": null, "greedy_identical_rate": null},
    "noninferiority_margin_pp": 5, "ni_alpha_one_sided": 0.05, "ni_power": 0.80,
    "w6a_w6b_disjoint": true, "w6a_task_ids_sha256": null, "w6b_task_ids_sha256": null,
    "tau_task_formula": "max(10, ceil(0.25*n_tasks))",
    "tau_coverage": 0.10,
    "tau_copied_tokens_per_effective_task": 128,
    "pre_data_frozen": true
  },
  "design_sha256": "<canonical hash of design-relevant subset>",
  "self_sha256": "<canonical hash of whole document excluding this field>",
  "blockers": ["user_authorization_pending"]
}
```

**规则**：

- `design_sha256` 只覆盖**设计相关子集**（matrix / config / statistics / gates），
  使"授权状态变更"不改变 design hash；
- 任何 design 变更 ⇒ 归档旧 revision、新建 revision、重新 review；
- `status ∈ {draft, pinned_blocked, authorized, superseded}`；
- runtime 必须校验 `status == "authorized"` 且当前 Gate 在 `authorized_gates`
  中，否则拒绝启动（授权门）。

### 14.2 结果目录与 RESULT_MANIFEST

```text
容器内 staging   : /results/phase7_5_c40/
版本化最终位置   : benchmark/approx_kv/results/phase7_5_c40/

benchmark/approx_kv/results/phase7_5_c40/
├── RESULT_MANIFEST.json                递归自哈希，--check 可只读重放
├── p75-plan-manifest.json              §14.1
├── C40_DISPOSITION.json                最终 disposition（§23.5）
├── evidence/
│   ├── cleanroom-compliance.json
│   ├── branch-creation.json
│   ├── cpu-tests.json
│   ├── docker-deps.json                pip freeze + pip check baseline diff
│   └── review/                         双模型 review artifacts
├── raw/                                每 start 一个 raw JSON（不可变）
├── logs/                               每 start 一个 server log（不可变）
├── central/                            central JSONL（append-only）
├── compact/                            每 start 一个自哈希 compact JSON
├── frozen/                             冻结 trajectory 与 span 清单 + sha256
└── summary/                            自哈希 consolidated summary
```

**RESULT_MANIFEST 必须绑定**：image digest、model+tokenizer revision、
code pin（commit + tree）、runner path + sha256、plan manifest revision +
self/design hash、每个 raw/log/central 文件的 sha256、
**全局 `BENCHMARK_RUN_LOG.jsonl` 的 line range + range sha256 + snapshot sha256**、
Docker mounts 与 capabilities、`statistics_freeze`、`tolerance_freeze`、
`known_gaps` 列表。
`--check` 必须能在只读模式下重放校验（Phase7 的 `88/88` 模式）。

### 14.2b C40 decision manifest（`c40_decision_manifest_version = 1`）

**必须与 §14.1 的静态授权 manifest 分离**：前者是"这个 Gate 被授权跑什么"，
后者是"这个请求允许复用哪一段 KV"。二者混在一个文件会同时破坏授权不可变性与
请求级动态性。

```json
{
  "c40_decision_manifest_version": 1,
  "sequence": 4217,
  "c40_fingerprint_sha256": "<必须与 server 侧一致，否则拒绝>",
  "selector_version": 1,
  "tool_provenance_schema_version": 1,
  "request": {
    "request_uid": "<harness 生成的唯一 id>",
    "prompt_token_sha256": "<target prompt 全序列哈希>",
    "prompt_token_count": 6144
  },
  "decision": {
    "eligible": true,
    "primary_reason": null,
    "secondary_reasons": [],
    "islands": [
      {"source_key": "<...>", "source_generation": 3,
       "source_start": 811, "target_start": 1544, "length": 1024,
       "rope_delta": 733,
       "segment_token_sha256": "<...>",
       "source_prefix_token_sha256": "<...>",
       "approx_depth": 0}
    ]
  },
  "expiry": {"issued_at_ns": 0, "valid_for_ms": 60000,
             "worktree_generation": 17},
  "self_sha256": "<...>"
}
```

**规则（硬性）**：

1. **原子更新**：写入必须 `write temp → fsync → atomic rename`；server 侧读到
   部分写入的文件即拒绝（`c40_manifest_schema_invalid`）；
2. **单调 `sequence`**：server 侧只接受严格递增的 `sequence`，重放旧序号即拒绝；
3. **请求绑定**：server 侧必须校验 `prompt_token_sha256` 与实际 prompt 一致，
   否则拒绝——这防止 decision 被错配到另一个请求；
4. **过期**：超过 `valid_for_ms` 或 `worktree_generation` 变化即失效；
5. **禁止任何 HTTP 字段选择 KV span**：span 只能来自本 manifest；
6. 路径由 `SGLANG_APPROX_KV_C40_MANIFEST` 指定，**禁止硬编码**；
7. 该 manifest **不参与**授权判定；授权只看 §14.1 的 `status` 与 `authorized_gates`；
8. **并发协议（CL-E 硬前置）**：单一路径 + 全局 `sequence` 只适用于**串行** harness。
   开启 concurrency / multi-tenant lane 时必须切换为**每请求一份 decision 文件**：
   `<manifest_dir>/<request_uid>.json`，`SGLANG_APPROX_KV_C40_MANIFEST` 指向
   **目录**；`sequence` 降级为**每 `request_uid` 单调**而非全局；server 侧按
   `request_uid` 查找，找不到即 `c40_manifest_absent`。模式由
   `SGLANG_APPROX_KV_C40_MANIFEST_MODE ∈ {file, dir}` 显式选择，
   `dir` 是 CL-E 与 `C40_REPO_ISOLATION` 实验的**硬前置**。

### 14.3 冻结清单（frozen plan）schema

```json
{
  "frozen_plan_version": 1,
  "trajectory_sha256": "<...>",
  "selector_version": 1,
  "c40_fingerprint_sha256": "<...>",
  "n_requests_total": 0,
  "n_requests_eligible": 0,
  "eligible": [
    {"request_index": 12,
     "source_key": "<...>", "source_generation": 3,
     "source_start": 811, "target_start": 1544, "length": 1024,
     "rope_delta": 733,
     "source_paths": ["pkg/a.py"],
     "source_paths_content_sha256": {"pkg/a.py": "<...>"},
     "worktree_generation": 17}
  ],
  "skip_reason_histogram": {"c40_later_same_path_write": 41, "...": 0},
  "self_sha256": "<...>"
}
```

---

## 15. Candidate / Axis Taxonomy 与 Staged Gates

### 15.1 Candidate taxonomy（冻结命名）

| Candidate ID | 全称 | 定义 | 层 |
| --- | --- | --- | --- |
| `D0` | Dense | 全 dense；不使用任何近似恢复。仍执行完整 source lifecycle 事件以保持状态对齐 | baseline |
| `E0` | Exact | 只用 exact prefix cache，无近似恢复 | baseline |
| `R0` | Raw+RoPE span-matched ceiling | 回放**冻结 span 清单**执行 R0；不运行 selector。**速度上界与 span-matched 控制** | control |
| `C40-D` | C40 selector-only, all dense | selector 完整运行并产出决策，但**强制全部 dense**（`copy` 被禁用）。用于隔离 **selector 开销与控制路径成本** | control |
| `C40-1R0` | C40 单岛 primary | `G40 × R0`，`max_islands=1`。**Phase7.5 的 primary 被评系统** | primary |
| `C40-mR0` | C40 多岛 | `max_islands>1` + non-overlap + total copy budget + payoff optimizer | conditional CL-A |
| `C40-1R1k` | C40 单岛 + leading-k repair | 复用既有 EPIC 能力，`k ∈ {2,8,32}`。命名为 `C40-R1-k`，**不冒充 CacheBlend / KVCOMM** | conditional CL-B |

**每臂 exact Radix final insertion 策略（冻结）**：

| Arm | final insertion |
| --- | --- |
| `D0` / `E0` / `C40-D` | baseline normal |
| span-matched `R0` | copy committed请求抑制；未copy/全dense请求normal |
| `C40-1R0`及copy extensions | copy committed请求抑制；dense_ineligible与fully-dense fallback在prefill完成后恢复normal |

所有arm同时报告`p75_exact_prefix_hit_tokens_total{arm}`；
C40/R0 copy臂另报`p75_radix_insert_suppressed_requests_total{arm,outcome}`。
所有C40 lifecycle臂报告`p75_radix_stash_suppressed_chunks_total{arm,outcome}`。
primary `theta_j`包含该stateful系统效应；臂策略与上表不符则engineering invalid。

**命名硬性规则**：

- **禁止**引入新的 recovery primitive 编号（不得叫 `R6` / `L0`）；
- `C40-1R1k` 只能描述为"复用本项目既有 EPIC leading-k 能力的可选修复"，
  **禁止**写成 CacheBlend selective repair 或 KVCOMM reconstruction；
- `R0` 在 Phase7.5 中特指 **span-matched** 回放（臂 C），与 Phase7 的
  unconditional R0 **不是同一个东西**，报告中必须显式区分。

### 15.2 正交轴

| 轴 | 取值 | 说明 |
| --- | --- | --- |
| **G** 准入 | `G0`（无准入）/ `G40`（grounded coding observation） | `G40` 为本阶段贡献 |
| **R** 恢复 | `R0`（Raw+RoPE）/ `R1-k`（EPIC leading-k） | 既有权威定义 |
| **S** 调度 | `S0`（LRU）/ `S4`（hierarchical） | 底座 `PolicyKind` |
| **H** 驻留 | `H0`（device only）/ `H1`（host demand-load） | CL-D |
| **PF** 预取 | `PF0`（off）/ `PF1`（hint / async，条件） | CL-D；**不得改变 selected span** |
| **chunk** | `4096`（primary）/ `2048`、`1024`（sensitivity） | Phase7 已证 chunk 是强混淆源；三水平来自 authority §9.10 |
| **body** | `256` / `512` / `1024` / `2048` / `4096` | 岛长度分布；五水平来自 authority §9.10 |
| **rho** | `1.5` / `2.0` | 压力 |
| **restart** | `0..3` | **唯一独立复制单元** |

### 15.3 反笛卡尔积原则（硬性）

```text
禁止：{7 candidates} × {2 S} × {2 H} × {2 PF} × {3 chunk} × {5 body} × {2 rho} × {4 restart}
      = 7 × 2 × 2 × 2 × 3 × 5 × 2 × 4 = 13440 cells
      —— 不可执行、不可授权、无统计意义

采用：staged gates。每个 stage 只打开**一个**新轴，且必须先通过前一 stage 的 Exit。
      Stage D-3（chunk × body × rho × arm 的完整因子）是唯一的例外：
      它是 authority §9.10 的 mandatory 隔离实验，但只在 candidate 轴上取
      {D0, E0, C40-1R0} 三个点，不与 S/H/PF 轴做笛卡尔积。
```

### 15.4 Staged gate 矩阵（`proposed`，每个 stage 独立授权）

#### Stage A — CPU only（0 GPU）

| Cell | Candidates | 轴 | 数量 |
| --- | --- | --- | --- |
| A-1 clean-room + 静态合规 | — | — | 0 start |
| A-2 selector 对抗 + property + 差分 | `C40-D`（离线） | — | 0 start |
| A-3 状态机 / lifecycle / 记账（fake backend） | `C40-1R0`（模拟） | — | 0 start |

#### Stage B — GPU same-context canary（Track B）

| Cell | Candidates | 固定轴 | starts |
| --- | --- | --- | --- |
| B-1 张量 + 输出 canary | `E0`, `C40-1R0` | `S0,H0,PF0,chunk4096,rho1.5` | `<= 2` |

#### Stage C — GPU cross-context pilot（Track B）

| Cell | Candidates（同 start 内交替） | 轴 | starts |
| --- | --- | --- | --- |
| C-0 calibration | `D0`, `C40-1R0` | `chunk4096, body2048, rho2.0` | `1`（**先跑，用于冻结剩余分配**；**不计入** pilot 的 restart 数） |
| C-1 pilot | `D0`, `C40-1R0`, `R0`(span-matched), `C40-D` | `chunk4096, body{512,2048}, rho2.0` | **`>= 3` 独立 restart**（每 restart 1 start ⇒ `3` starts） |

**Track B start 预算核对（NEW-02）**：

```text
  G4  same-context canary        <= 2 starts
  C-0 calibration                 = 1 start
  C-1 pilot                      >= 3 starts（统计要求 n_pilot >= 3，§19.2）
  ------------------------------------------------
  合计（下限配置）                = 2 + 1 + 3 = 6 starts   <=  8 starts cap  ✓
  余量                            = 2 starts（用于 pilot 失败重跑或 G4 复测）

若 pilot 需要超过 3 个 restart（例如首轮 restart 工程无效），
在 <= 8 starts 内可再补 2 个；超过 8 starts 必须停止并重新申请（SR-10）。
```

#### Stage D — Confirmatory primary（Track C，需二次授权）

| Cell | Candidates | 轴 | restarts |
| --- | --- | --- | --- |
| D-1 primary（**非纯投影**：D-3 chunk4096 子集 + 内联 primary controls） | `D0`, `E0`, `C40-1R0`（来自 D-3）+ `R0` span-matched, `C40-D`（额外内联 8 cells） | `chunk4096 × body{512,2048} × rho{1.5,2.0}` | `n_confirmatory`（默认 `4`），**不额外占 start** |
| D-2 chunk 分离（**纯投影**） | `D0`, `E0`, `C40-1R0` | `chunk{1024,2048} × body{512,2048} × rho2.0`（**只作 sensitivity，永不 headline**） | 同 D-3，不另加 start |
| D-3 完整因子（**mandatory**，预算须二次授权） | `D0`, `E0`, `C40-1R0` | `chunk{1024,2048,4096} × body{256,512,1024,2048,4096} × rho{1.5,2.0}`；90 arm-cells / **3 starts per restart** | authority §9.10 必做项；`n=4` ⇒ **12 starts**（cap 12 + contingency 4 = 16）。D-1/D-2 为其投影，不另加 start |

#### Stage E — Conditional extensions（Track D，需二次授权）

| Cell | Candidates | 轴 | 触发条件 |
| --- | --- | --- | --- |
| E-1 多岛 | `C40-1R0` vs `C40-mR0` | `max_islands ∈ {1,3}` | Stage D 完成且 CL-A 通过 CPU gate |
| E-2 repair | `C40-1R0` vs `C40-1R1k` | `k ∈ {8}`（先单点） | Stage D 完成且 CL-B 通过 CPU gate |
| E-3 host | `C40-1R0 @H0` vs `@H1` | `H ∈ {H0,H1}` | Stage D 完成且 CL-D 通过 CPU gate |
| E-4 exact-overlap clip | `C40-1R0`（clip off）vs `C40-1R0 + CL-I`（clip on） | `C40_EXACT_OVERLAP_CLIP ∈ {0,1}`，primary body/rho | Stage D 完成且 CL-I 通过 CPU gate（§17.23） |

#### Stage F — Workflow scheduler（Track D）

| Cell | Candidates | 轴 | 约束 |
| --- | --- | --- | --- |
| F-1 | `D0`, `C40-1R0` | `S ∈ {S0,S4} × rho ∈ {1.5,2.0}` | **S0/S4 必须相邻交替启动**（`S0,S4,S0,S4,...`），并报告 matched coverage |

#### Stage G — Quality（Track E，task-runs 单列计量；**独立 lane，不依赖 Stage C/D**）

| Cell | 基准 | 单位 |
| --- | --- | --- |
| G-1 | RepoBench-P（`>= 1000` 例） | 例 |
| G-2a | SWE-bench Verified **W6a calibration**（40 tasks × 3 配对 seed，不进入检验） | task-runs |
| G-2b | SWE-bench Verified **W6b confirmatory**（与 W6a disjoint 的 `n_confirm` tasks，二次授权） | task-runs |

#### Stage H — Prefetch composition（Track F，正交、条件、可能 BLOCKED_EXTERNAL）

| Cell | Candidates | 硬性验收 |
| --- | --- | --- |
| H-1 | `D0` / `C40-1R0 @PF0` / `PF1-only` / `Combined` | Combined 的 selected span 与 `C40-1R0 @PF0` **逐 token 相同**；关闭 prefetch 精确恢复；无 lease/worker/CUDA event 泄漏 |

### 15.4b Stage D-3 完整因子的 start 打包与闭合条件（**冻结**）

#### 15.4b.1 为什么"90 cells"不等于"90 starts"

```text
完整因子的 arm-cell 数：
  chunk {1024, 2048, 4096}          =  3
  body  {256, 512, 1024, 2048, 4096} =  5
  rho   {1.5, 2.0}                   =  2
  arm   {D0, E0, C40-1R0}            =  3
  ------------------------------------------
  3 × 5 × 2 × 3 = **90 arm-cells**

但 **server start 只由 chunk 决定**：chunked_prefill_size / max_prefill_tokens
是**启动参数**，必须重启 server 才能改；而 body / rho / arm 都可以在
**同一个 server 进程内**按顺序切换（body 是请求构造参数，rho 是压力参数，
arm 是执行分支）。

因此打包规则（冻结）：
  每个 restart 按 chunk 打包 **3 个 server starts**（chunk = 1024 / 2048 / 4096）
  每个 start 内**顺序执行** 5 body × 2 rho × 3 arms = 30 arm-cells
    - arm 之间按 formal repeat 交替（A,B,C,A,B,C,...，§17.21）
    - body / rho 切换之间做完整 reset
  chunk=4096 的 start **额外**内联 8 个 primary-control arm-cells（§15.4b.2）
  => 每 restart 覆盖 90 个 D-3 arm-cells + 8 个 primary controls，占 3 starts
  => n = 4 restarts  ⇒  **4 × 3 = 12 server starts**
```

#### 15.4b.2 D-1 不是纯投影：它额外含 primary controls

```text
D-2 sensitivity = **纯投影**
  = D-3 中 chunk in {1024, 2048} 的子集
  不增加任何执行，不增加 start。

D-1 primary **不是纯投影**：
  D-3 的 arm 轴只有 {D0, E0, C40-1R0}，**不含**两个必需的 control 臂：
     臂 C  R0 span-matched（度量 overhead_selector_control）
     臂 D  C40-D selector-only（度量 selector_only_overhead）
  这两个 control 是 §19.4 四臂 counterfactual 的必需组成，
  缺了就无法把"selector 判定与控制路径开销"从机制效应中分离。

冻结做法（**不增加 start**）：
  在 **D-3 的 chunk=4096 那个 start 内**，除 30 个 arm-cells 之外，
  **额外顺序执行** 臂 C 与臂 D，但**仅限 primary body / rho**：
      body in {512, 2048} x rho in {1.5, 2.0} x arm in {R0-span-matched, C40-D}
      = 2 x 2 x 2 = 8 个额外 arm-cells
  这 8 个 cell 与该 start 内其余 cell 共享同一 server 进程，
  按 formal repeat 交替执行，**不需要**新的 server start。

因此：
  chunk=4096 的 start 内 arm-cells = 30（D-3 部分）+ 8（primary controls）= 38
  chunk=1024 / 2048 的 start 内各 30（纯 D-3）
  每 restart 合计 38 + 30 + 30 = 98 arm-cells / 3 starts
```

#### 15.4b.3 预算与授权

| 项 | 值 |
| --- | --- |
| 建议 start cap | **`12` starts**（`4 restarts × 3 chunk`；primary controls 内联在 chunk4096 的 start 内，不额外占 start）**+ contingency `4`** = **`16`** |
| GPUh | **必须由 1-start calibration 冻结**；calibration 前不得给出 GPUh 数字 |
| 授权 | **二次授权**（Track C）；`docker_test` + `gpu` + `budget` 三项齐备 |
| `n` 的来源 | `n_confirmatory` 由 §19.2a 计算；若 `n > 4`，start 数按 `n × 3` 线性增长，**必须重新申请** |

#### 15.4b.4 闭合条件（**`known_gap` 不能替代完成**）

```text
G6 完整 Exit 需**同时**满足：
  (i)  D-3 完整因子（90 arm-cells x n restarts）已完成；
  (ii) primary controls（臂 C R0 span-matched、臂 D C40-D，
       primary body/rho 共 8 arm-cells x n restarts）已完成。

两者齐备 ⇒ mechanism disposition 可取
             POSITIVE / SMALL_POSITIVE_BELOW_MDE / NEGATIVE / INCONCLUSIVE

(i) 或 (ii) 任一未完成 ⇒ **G6 不能视为完整通过**：
             1. G6 记为 `PARTIAL_EXIT`，不得写成 Exit；
             2. mechanism disposition **只能**取 `PARTIAL` 或 `INCONCLUSIVE`，
                **不得**取 POSITIVE / NEGATIVE / SMALL_POSITIVE_BELOW_MDE；
             3. 把未完成项写入 known_gaps 是**披露义务**，
                **不是**豁免 —— known_gap 不替代完成；
             4. 若日后补做，须重新走 G6 授权与 Exit 判定。
```

### 15.5 Stage 之间的推进规则

```text
Stage A Exit  → 可申请 Stage B
Stage B Exit  → 可申请 Stage C
Stage C Exit  → 可申请 Stage D（**必须先按 §19.2a 冻结 s_pilot / n_confirmatory / 判定阈值**）
Stage D Exit  → 可申请 Stage E / F（并行，互不阻塞）
Stage B Exit  → 可申请 Stage G（**独立 lane、独立预算、task-runs 计量**；
                 **不依赖** Stage C/D，因此 speed lane 停止不影响质量评测）
Stage F Exit  → 可申请 Stage H（若 Track F 可获得）

禁止：任何 Stage 由前一 Stage 的结果自动触发
禁止：把 conditional stage 的结果写入 primary headline
```

---

## 16. Work Packages WP0a–WP12

> 全部工时为 `estimate`。WP0a/WP0q 在当前 plan-drafting/review 授权内为
> `AUTHORIZED`；WP1a 及其后全部为 `PENDING USER AUTHORIZATION` 或
> `NOT REQUESTED`，以各 WP 表内状态为准。
> "文件面"列出的路径为 `proposed`，实现时可微调但必须保持 §4.3 的命名冻结。

### WP0a — Document Authority / 计划冻结（**纯文档**）

| 项 | 内容 |
| --- | --- |
| **文件面** | 本文件；`evidence/review/plan-review-*.json`（review artifact，纯文档，可暂存于文档仓库） |
| **内容** | 冻结方法定义、branch/base/命名/结果目录、能力三层、架构、状态机、provenance schema、统计合同、Gate 顺序、Stop rules、预算结构；定义 CR-1..CR-9、manifest schema 与 quarantine 流程；把 Phase7 R0 先验写入计划；执行 independent review；同步三份 authority 文档 |
| **验收** | independent review 的 closure artifact 已版本化并绑定文档 sha256，其中 `open P0/P1 = 0/0` |
| **依赖** | 无 |
| **工时** | `0.15–0.25 人周` |
| **资源** | host 只读，0 GPU，**无 Docker** |
| **授权** | `AUTHORIZED` |

### WP0q — Quarantine Signature Extraction（**隔离 reviewer 执行**）

| 项 | 内容 |
| --- | --- |
| **文件面** | `evidence/review/c40-quarantine-manifest.json`（纯哈希/签名，自哈希，versioned） |
| **内容** | §4.6步骤1–2：提取`commit_ids`/`exclusive_blob_hashes`/`patch_id_alerts`/`ast_signatures_file`/`ast_signatures_function`/`allowed_shared_signatures`；不输出任何源码文本 |
| **验收** | 自哈希可复算；`contains_source_text == false`；范围定义完整；artifact 已版本化 |
| **依赖** | WP0a 的 §4.6 合同已冻结；由**隔离 reviewer** 执行 |
| **工时** | `0.15–0.25 人周` |
| **资源** | host 只读，0 GPU，**无 Docker** |
| **授权** | `AUTHORIZED`（`plan drafting/review` 范围内） |

### WP1a — Branch Bootstrap（**只需 branch 授权，无 Docker**）

| 项 | 内容 |
| --- | --- |
| **文件面** | 新 worktree `phase7.5-c40-cleanroom`（**保持 clean**）；`<bootstrap_dir>/branch-creation.json`（worktree **之外**，手写内容，非生成） |
| **内容** | 执行 §5.2 的 branch/worktree 创建命令；逐条记录命令与输出；bootstrap evidence 落在 worktree 之外（§5.2b）。**不**在此步创建仓库内结果目录 |
| **验收** | branch HEAD == base commit 且 tree 匹配；**`git status --porcelain` 为空**；底座 worktree HEAD 未变；未 push |
| **依赖** | WP0a；`branch_creation_authorized = true` |
| **工时** | `0.1–0.15 人周` |
| **资源** | host git，0 GPU，**无 Docker、无测试执行** |
| **授权** | `PENDING USER AUTHORIZATION` |

### WP0b — Manifest / Bootstrap Builder（branch 存在后）

| 项 | 内容 |
| --- | --- |
| **文件面** | `benchmark/approx_kv/build_p75_manifest.py`、`build_p75_result_manifest.py`、`benchmark/approx_kv/coding_c40/reason_inventory.py`；产出 `p75-plan-manifest.json` rev1、`evidence/reason-inventory.json` |
| **内容** | 实现 plan manifest 构建与 `--check`；实现 §2.6 的 AST reason 扫描器（四类来源，无法静态求值即报错）；把 WP1a 的手写 `branch-creation.json` 重新哈希绑定 |
| **验收** | `--check` 通过；`design_sha256` / `self_sha256` 可复算；`status == pinned_blocked`；`blockers` 显式列出未授权项；reason inventory sha256 冻结 |
| **依赖** | WP1a；`implementation_authorized = true` **且** `docker_test_execution_authorized = true`；授权凭据记入 `evidence/g0b-authorization.json` |
| **工时** | `0.1–0.2 人周` |
| **资源** | Docker CPU，0 GPU |
| **授权** | `PENDING USER AUTHORIZATION` |

### WP1b — Clean-room 合规测试（消费 quarantine 签名）

| 项 | 内容 |
| --- | --- |
| **文件面** | `test/registered/unit/mem_cache/test_c40_cleanroom_compliance.py`；`evidence/{cleanroom-compliance,quarantine-consumed}.json` |
| **内容** | 实现 CR-1..CR-9（§4.4）；消费**隔离 reviewer** 产出的 quarantine manifest（§4.6），实施者只读签名、不接触 collaborator 源码；扫描范围覆盖整个 new-branch diff（含 allowlisted 既有文件的 modified hunks） |
| **验收** | CR-1..CR-9 全绿；结论表述限定为"**在本次扫描范围与检查方法下未检测到禁止血缘**"，**不得**写成"无血缘" |
| **依赖** | WP0b；manifest `status == authorized` 且 `"P7.5-G1b" ∈ authorized_gates` |
| **工时** | `0.25–0.4 人周` |
| **资源** | Docker CPU，0 GPU |
| **授权** | `PENDING USER AUTHORIZATION` |

### WP2 — ToolEvent provenance 采集与解析

| 项 | 内容 |
| --- | --- |
| **文件面** | `coding_c40/provenance.py`、`coding_c40/types.py`（ToolEvent/RepoState/PathEffect）；`test_c40_provenance.py` |
| **内容** | ToolEvent schema v1（event-level `fs_events`）；**authority 采集器**：(a) 结构化封闭工具的 wrapper read/write event 声明、(b) 任意 shell 的 event-level syscall trace（默认 `strace_ptrace_v1`；authority 替代仅 `fanotify_v1`/`ebpf_v1`，须声明 capability）；`preload_shim_v1` 仅 supplemental differential oracle；**secondary check**：Merkle snapshot（content hash / generation / integrity）与 `git status`。<br>差分测试的 primary oracle 必须是与被测 authority 不同的 full event collector。<br>write-then-restore 规则；`collector_complete == false` 即 `unknown_effect`；路径规范化 N-1..N-7；rename/symlink/directory 建模 |
| **验收** | §17.1 对抗矩阵在**冻结语料**上全部通过（结论限定于该语料）；property `P1–P8`（含 `P3b`）各 1000 例无反例；差分测试在冻结语料上 **collector-observed FN = 0**（被测与 oracle 为两套**不同的 event collector 实现**）；`oracle_agreement` 报告已输出；write-then-restore 用例被检出；`normalize` 幂等；Docker capability 集合已在 manifest 中列明 |
| **依赖** | WP1b |
| **工时** | `1.5–2 人周` |
| **资源** | Docker CPU（差分测试用 `--tmpfs /scratch` 内副本，**绝不**写宿主 worktree） |

### WP3 — G40 selector 与决策审计

| 项 | 内容 |
| --- | --- |
| **文件面** | `coding_c40/selector.py`；`coding_c40/types.py`（GroundedIsland/C40Candidate/C40Decision）；`test_c40_selector.py`；`benchmark/approx_kv/run_p75_selector_offline.py` |
| **内容** | PC-2..PC-5、PC-9；rolling 窗口；只读证据分类（结构化）；token 化（只取 `role=="tool"`）；唯一出现；min/cap；strict-middle；单一最大岛；完整可审计 `C40Decision` dict；selector overhead 计时分项 |
| **验收** | PC-2..PC-5 单测全绿；`assistant_tokens_selected == 0` 断言；每次拒绝有唯一 selector reason；离线 runner 可对冻结 trajectory 产出 `frozen/` 清单 |
| **依赖** | WP2 |
| **工时** | `1–1.5 人周` |
| **资源** | Docker CPU |

### WP4 — Identity / Fingerprint / Approx depth

| 项 | 内容 |
| --- | --- |
| **文件面** | `coding_c40/adapter.py`（key/fingerprint 构造）；`test_c40_identity_fingerprint.py` |
| **内容** | ME-3 完整 fingerprint；segment key 扩展（content hash / worktree generation / repo / branch）；provenance 与 approx_depth 标注；stale handle 拒绝；重注册产生新 generation |
| **验收** | 跨 fingerprint 复用被拒；缺字段 fail closed；**在已扫描代码路径中未发现 fingerprint bypass**（静态扫描 + 动态注入双重检查；扫描范围须在 evidence 中列明）；depth 规则单测通过 |
| **依赖** | WP3 |
| **工时** | `0.5–0.75 人周` |
| **资源** | Docker CPU |

### WP5 — Middle-span controller 与状态机

| 项 | 内容 |
| --- | --- |
| **文件面** | `coding_c40/controller.py`、`coding_c40/state.py`；`xs:schedule_batch.py` 第三分支接线；`test_c40_controller_state.py` |
| **内容** | §8 全部：状态机（含 `ADMISSION_DEFERRED` / `TERMINAL_REJECTED` / `DENSE_ISLAND_FALLBACK` / retract→NONE）、**§8.6 chunk-splitting 执行协议**（所有权分离、effective-prefix/allocation成对替换、forced-middle + 单chunked-owner互斥、admission/copy/progress hooks、copy ledger、identity membership、B-3 transactional slot allocation）、request-lifetime vs transient、consume/produce双角色、scheduler lock handoff、五条final cleanup + retraction reset |
| **验收** | 状态转移矩阵与`TC-1..TC-99`全绿；新增CI method、eBPF profile和quality coverage派生量 |
| **依赖** | WP4 |
| **工时** | `2–2.5 人周` |
| **资源** | Docker CPU（fake/mock backend） |

### WP6 — Adapter、plan 构造与底座复用

| 项 | 内容 |
| --- | --- |
| **文件面** | `coding_c40/adapter.py`；`test_c40_plan_coverage.py` |
| **内容** | 构造 `KVReusePlan`（`TransferSpan` + `DenseRange[]`，`require_full_coverage=True`）；`rope_delta = target_start - source_start`；调用 `execute_reuse_plan`；接入 cross-store allocator/budget/object graph/stale victim/SWA/provisional/fallback 记账；**零重写底座** |
| **验收** | `copied_spans ∪ dense_ranges == [0, len(target))`，无重叠无空洞；位置修正只在 transfer backend 内发生一次；object graph 无孤儿；allocator 未被绕过（静态检查） |
| **依赖** | WP5 |
| **工时** | `0.5–0.75 人周` |
| **资源** | Docker CPU |

### WP7 — Terminal reason、telemetry、lease GC、feature gate

| 项 | 内容 |
| --- | --- |
| **文件面** | `coding_c40/stats.py`；`test_c40_terminal_reasons.py`、`test_c40_lease_soak.py`；config 变更 |
| **内容** | §12全部reason；§13全部metric；ME-4真feature gate；ME-8周期性`gc_expired_leases()` + 五条final cleanup + retraction reset；central JSONL事件写入 |
| **验收** | 族 1–4 四个恒等式（§12.4）全部成立且无跨族混计；`c40_*` 与 canonical inventory（含 prefix-family）互斥；`evidence/reason-inventory.json` sha256 与冻结值一致；T-GATE-1..5 全绿；在 `10k` 请求 soak 范围内 lease/record/orphan/provisional 全部归零 |
| **依赖** | WP6 |
| **工时** | `0.5–1 人周` |
| **资源** | Docker CPU |

### WP8 — Docker 依赖锁、central log、manifest、consolidator

| 项 | 内容 |
| --- | --- |
| **文件面** | `Dockerfile.p75`（派生 layer）、`requirements.lock`；`benchmark/approx_kv/build_p75_manifest.py`、`benchmark/approx_kv/build_p75_result_manifest.py`、`benchmark/approx_kv/coding_c40/consolidate_p75_results.py`；`test_c40_manifest.py`、`test_c40_consolidator.py` |
| **内容** | ME-9：基于 digest 的专用 layer（**禁止运行时 `pip install`**；必要时 `pip install --target` + `PYTHONPATH`）；`pip-compile`/`uv pip compile` 产出含 hash 的 lock；central JSONL 合同；结果 manifest 递归自哈希与 `--check`；离线 consolidator 产出自哈希 compact/summary |
| **验收** | `pip install --require-hashes -r requirements.lock` 后 `pip check` **相对已版本化 baseline 无新增冲突**（若另建 clean base 并固定新 digest，则要求零输出）；`RESULT_MANIFEST --check` 只读重放通过；consolidator 默认拒绝覆盖、`--force` 显式允许 |
| **依赖** | WP0b（可与 WP2–WP7 并行） |
| **工时** | `0.75–1 人周` |
| **资源** | Docker CPU |

### WP9 — Workload、冻结 trajectory 与 runner

| 项 | 内容 |
| --- | --- |
| **文件面** | `benchmark/approx_kv/coding_c40/{trajectory,plan_freeze,workloads_c40}.py`；`run_p75_canary.py`、`run_p75_micro.py`、`run_p75_workflow.py`；`test_c40_plan_freeze.py` |
| **内容** | §18 的 W1 / W2 / W3 / **W4a live corpus**（`>= 24` 条 live trajectory，含 fixture repos 与 seed 冻结）；冻结 trajectory + 冻结 span 清单 + sha256 绑定；四臂 runner；warm-up 丢弃与 formal repeats 合同；**全局 run log 追加**（§13.5.0）；共享授权门（校验 manifest `status=="authorized"` 与当前 Gate） |
| **验收** | 冻结清单可逐字节复现；W4a 的 `>= 24` 条 trajectory 均实际发出 source-producing 与 consume 请求（非 replay）；runner 在未授权 manifest 下拒绝启动（正负用例）；warm-up 请求写入 phase central log 且不进统计；全局 run log 的 line range + range/snapshot sha256 已绑定 |
| **依赖** | WP7、WP8 |
| **工时** | `0.5–1 人周` |
| **资源** | Docker CPU（GPU 执行属后续 Gate） |

### WP10 — GPU 正确性与 pilot 执行

| 项 | 内容 |
| --- | --- |
| **文件面** | `test_c40_cuda.py`；`raw/`、`logs/`、`central/` 产出 |
| **内容** | Stage B（**先** Control-1/Control-2 baseline envelope 冻结，**再** same-context canary、K/V+RoPE 张量、corruption canary）；Stage C（1-start calibration + `>= 3` restart pilot）；四本账与实测 `speedup_{1,2,4,8}`；按 §19.2a 计算并冻结 `s_pilot` / `n_confirmatory` |
| **验收** | `baseline_envelope` 已冻结并写入 manifest；`max|ΔK|`/`max|ΔV|`/`max|Δlogit|` 未超出 `max(dtype/frozen tol, baseline_envelope)`；贪心输出一致性按 §17.5.2 条件门判定；`rotated_k_tokens == copied_k_tokens == span_len` 逐层成立；注入 `±1 rope_delta` 被检出；pilot 产出 `>= 3` 个 `theta_j` 与 `s_pilot`，并冻结 `n_confirmatory` |
| **依赖** | WP9；`gpu_execution_authorized = true`（Track B） |
| **工时** | `0.25–0.5 人周` |
| **资源** | Docker GPU，`<= 8 starts / <= 2 GPUh`（先 1-start calibration） |

### WP11 — Conditional lanes 实现

| 项 | 内容 |
| --- | --- |
| **文件面** | `coding_c40/optimizer.py`（CL-A）、`coding_c40/gates.py`（CL-C/CL-G）；repair 接线（CL-B）；host demand-load 与 prefetch-neutral hint（CL-D）；concurrency/isolation（CL-E/CL-H）；chaining diagnostic（CL-F） |
| **内容** | §6.3 全部 **9** 个 lane（CL-A..CL-I）；每个 lane 独立 feature flag、独立测试、独立 terminal reason |
| **验收** | 每 lane 关闭时行为与 primary 逐字段一致；多岛 non-overlap 与 budget property 测试通过；`C40-R1-k` 复用既有 EPIC 能力且 `k=0` 退化为 `C40-1R0`；prefetch hint **不改变 selected span**（逐 token 断言）；AST gate 只作辅助，embedding gate 仍可独立开启 |
| **依赖** | WP7（core 完成后并行；**不得阻塞 WP9/WP10**） |
| **工时** | `4–8 人周`（全部 **9** 个 lane），按 lane 拆分交付：CL-A `0.70–1.25`、CL-B `0.50–0.95`、CL-C `0.70–1.25`、CL-D `0.70–1.25`、CL-E `0.55–1.10`、CL-F `0.25–0.45`、CL-G `0.25–0.45`、CL-H `0.25–0.95`、CL-I `0.10–0.35`；下界合计 `4.00`，上界合计 `8.00` |
| **资源** | Docker CPU（GPU 执行属 Stage E） |

### WP12 — 质量 harness、consolidation、双模型 review 与 disposition

| 项 | 内容 |
| --- | --- |
| **文件面** | `run_p75_quality.py`；`summary/`、`C40_DISPOSITION.json`、`evidence/review/` |
| **内容** | §18 的 W5 / W6（+ 可选 W7），含显式的 source-producing → consume 两阶段协议与覆盖率报告；§18.6b `NO_COVERAGE` 判定；McNemar + 层次/task-cluster 分析；离线 consolidation；双模型 review；最终 disposition 分列 engineering/mechanism/system/quality/publication |
| **验收** | W6a / W6b 的 task 划分 disjoint 且已 pre-data 冻结（清单 sha256 入 manifest）；`effective_paired_tasks` / `w_quality` / `mean_copied_tokens_per_effective_task` 分段报告；若低于 §18.6b 预冻结阈值则判 `NO_COVERAGE` 且**不得**表述为质量无损；否则质量结论以"该样本量与实测 discordant rate 下无法排除 `<= X pp` 损伤"表述并给出 X 推导；`RESULT_MANIFEST` 递归校验通过（含全局 run log 绑定）；open P0/P1 = 0；disposition 五类分别给出 |
| **依赖** | 质量部分：**WP9**（harness/runner）+ **WP10 的 Stage B（G4 same-context canary）** + `quality_campaign_authorized = true`。G4 属于 WP10 的 Stage B，因此质量 lane 依赖 **WP10 的 Stage B**，而**不依赖** Stage C（G5 pilot）与 G6 confirmatory —— 这正是 SR-6/SR-7 只停 speed lane 的技术前提。consolidation/review 部分：依赖已实际执行的各 Gate 产出 |
| **工时** | `1.5–2 人周`（不含 task-runs 机时） |
| **资源** | Docker GPU（task-runs 单列计量）+ 0-GPU consolidation |

### 16.1 依赖图

```text
WP0a  (纯文档, 已授权)
 └─ WP1a  (branch bootstrap, 无 Docker)
     └─ WP0b  (manifest builder, Docker CPU)
         ├─ WP1b  (cleanroom compliance)
         │   └─ WP2 ─ WP3 ─ WP4 ─ WP5 ─ WP6 ─ WP7 ─┬─ WP9 ─ WP10[StageB=G4] ─┬─ WP10[StageC=G5] ─ WP12(consolidation)
         │                                          │                          │
         └─ WP8 ───────────────────────────────────┘                          └─ WP12(质量部分=G9)
                                                                                  ↑ 需 G3(WP7 Exit) + G4,
     WP11（conditional lanes）──────────────────────────────────────────────────   不经过 StageC / G6
     （不阻塞 WP9 / WP10 / WP12）

无循环性质：WP0a 不产出 manifest；WP1a 不需要 manifest 也不需要 Docker；
            WP0b 在 branch 上生成 manifest；WP1b 起才消费 manifest 授权门。
```

### 16.2 工时汇总（`estimate`，自下而上求和，**不做截断**）

| WP | 下界 | 上界 |
| --- | ---: | ---: |
| WP0a（文档，已授权） | `0.15` | `0.25` |
| WP0q（quarantine 提取，已授权） | `0.15` | `0.25` |
| WP1a（branch bootstrap） | `0.10` | `0.15` |
| WP0b（manifest builder） | `0.10` | `0.20` |
| WP1b（cleanroom compliance） | `0.25` | `0.40` |
| WP2 | `1.50` | `2.00` |
| WP3 | `1.00` | `1.50` |
| WP4 | `0.50` | `0.75` |
| WP5 | `2.00` | `2.50` |
| WP6 | `0.50` | `0.75` |
| WP7 | `0.50` | `1.00` |
| WP8 | `0.75` | `1.00` |
| WP9 | `0.50` | `1.00` |
| WP10 | `0.25` | `0.50` |
| **WP0a–WP10 合计（14 项）** | **`8.25`** | **`12.25`** |

| 分组 | 工时 | 校验 |
| --- | --- | --- |
| **core parity + mandatory extensions**（WP0a–WP10，含 pilot 执行） | **`8.25–12.25 人周`** | 与上表 14 个分项求和一致；按 Track 拆分为 A（WP0a..WP9）`8.00–11.75` + B（WP10）`0.25–0.5` |
| **conditional extensions**（WP11，全部 lane） | **额外 `4–8 人周`** | **九**个 lane 分项下界合计 `4.00`、上界合计 `8.00`，一致 |
| **quality campaign**（WP12 工程部分） | **`1.5–2 人周`，另算** | task-runs 机时与 Track E 的 starts/GPUh **单独计量** |

> **一致性规则**：若任何 WP 的分项 estimate 被修订，本表与 §22.3 必须同步重算；
> **禁止**在汇总处直接截断或改写区间使其"看起来"符合目标值。

---

## 17. 测试与实验设计

> 所有测试与实验**必须在 Docker 内执行**。host 只允许：git 只读操作、
> 文档读取/编辑、G0a review artifact 与 G0q quarantine signature 产出、
> G1a branch bootstrap。**G0b 起的 manifest builder/`--check` 属 Docker CPU
> Gate，禁止在 host 以"非实验"名义绕过授权执行。**

### 17.1 T1 — Selector 对抗矩阵（CPU，最高优先级）

**目的**：把 collaborator 分支的 B-01 / B-02 两个 P0 变成永久回归测试，
并把"写检测"从命令正则升级为结构化判定。

#### 17.1.1 写工具对抗矩阵（每格必须产生 `invalidated = 1`）

| 类别 | 用例（每类 `>= 5` 变体） |
| --- | --- |
| shell 重定向 | `cat > f`、`cat >> f`、`echo x > f`、`printf ... > f`、`tee f`、`tee -a f`、`cat <<EOF > f` |
| in-place 编辑 | `sed -i`、`sed --in-place`、`perl -pi -e`、`perl -i.bak -pe`、`ruby -i -pe`、`ed`、`ex -sc` |
| 截断 / 写块 | `truncate -s 0 f`、`dd of=f`、`dd of=f conv=notrunc`、`: > f`、`> f` |
| 复制 / 移动 / 链接 | `cp`、`cp -a`、`install -m`、`mv`、`git mv`、`rsync`、`ln -sf`、`ln -f` |
| 打补丁 | `patch -p1`、`git apply`、`git am`、`git checkout --`、`git restore`、`git stash pop`、`git revert` |
| 语言内写 | `python -c "open(...,'w')"`、`python - <<PY`、`pathlib.write_text`、`json.dump(fp)`、`shutil.copy`、`os.rename`、`np.save` |
| 构建副作用 | `make`、`python setup.py build_ext --inplace`、`pip install -e .`、`pytest --lf` 写 cache |
| 归档解包 | `tar -xf`、`unzip -o`、`git clone` 覆盖 |
| 编码 / 引用绕过 | 反引号、`$( )`、`eval`、变量拼路径、base64 解码后写、多行续行 `\` |
| 路径变体 | `/testbed/pkg/a.py`、`./pkg/a.py`、`pkg/a.py`、`../repo/pkg/a.py`、软链接指向、大小写、Unicode 路径 |
| **目录级写** | `rm -rf dir/`、`mkdir -p dir/sub`、`cp -r src/ dst/`、`find . -name '*.pyc' -delete` |
| **rename 组合** | `git mv a b`、`mv a b && mv b a`、`mv dir1 dir2` |
| **symlink** | `ln -s a link; echo x > link`、`ln -s dir ldir; echo x > ldir/f` |
| **write-then-restore**（Merkle 盲区，只能由 event-level collector 检出） | `cp a a.bak; sed -i s/x/y/ a; mv a.bak a`；`echo y > a; echo x > a`（还原为同字节）；`patch -p1 < d; patch -R -p1 < d`；`git apply d; git apply -R d`；仅以 `O_APPEND` 打开但不写入 |

> **注**：这些用例用于**验证结构化 provenance 采集器能捕获其效果**，
> **不是**用来构建命令正则表。断言对象是 `write_paths` / `worktree_generation`
> / `content hash`，而不是命令字符串匹配结果。

#### 17.1.2 混合 group 矩阵（每格必须判 `ineligible`）

至少覆盖：`read; write`、`write; read`、`read && write`、`read || write`、
`read | tee f`、`read; (write)`、`read; bash -c "write"`、
`for f in *; do write; done`、以及超时被截断后仍有写副作用的场景。

#### 17.1.3 Property 测试（Hypothesis 或等价，每条 `>= 1000` 例）

> `P3` 与 `P3b` 的拆分理由见下表：把"失效单调性"与"几何唯一性"混为一条会得到
> 一个**为假**的性质（rolling 窗口滚动与 token 重复变化都会造成反例）。

| ID | 性质 |
| --- | --- |
| `P1` | 若结构化 `effective_write_set ∩ normalized(source_paths) != ∅`，则 `invalidated == 1` |
| `P2` | 若 `unknown_effect == True`，则该 group **不可**成为 source |
| `P3` | **失效单调性（在冻结窗口与冻结 target 下定义）**：固定 rolling 窗口内容与 target prompt 不变，只在 trajectory 尾部追加 group；则对窗口内**每一个既有 candidate**，其 `invalidated` 标志**只能由 false 变 true，不能由 true 变 false**。<br>**必须排除的两个混淆**（否则该性质为假）：<br>(a) 追加会使 rolling 窗口滚动、把旧 group 滚出，从而改变候选集合与 target prompt —— 测试中必须**冻结窗口**；<br>(b) 追加会改变 target prompt 的 token 重复情况，使原本“出现多次”的 span 变为唯一 —— 该唯一性判定属**几何层**，与失效层分开断言，**不纳入** `P3` |
| `P3b` | **几何层纯函数性**：在 target prompt 与候选 token 序列固定时，“唯一出现 / strict-middle / min-cap”判定是纯函数（同输入同输出，无隐藏状态） |
| `P4` | 路径规范化幂等：`normalize(normalize(p)) == normalize(p)`；`/testbed/x`、`./x`、`x` 归一到同一键 |
| `P5` | rename 对称性：`(old,new) ∈ rename_pairs ⇒ old ∈ EWS 且 new ∈ EWS` |
| `P6` | symlink 闭包：写 link ⇔ 写 target（双向） |
| `P7` | generation 单调：`worktree_generation` 非递减；变化必使旧候选失效除非逐路径 hash 证明 |
| `P8` | fingerprint 完备：任一 fingerprint 字段缺失 ⇒ 整请求禁用 C40 |

#### 17.1.4 差分测试（**Docker 内临时可写副本**，FN 必须为 0）

用真实 SWE-bench trajectory 或合成 agent trace **真正执行**命令，采集
ground-truth `write_paths` 并与 selector 判定做差分。**FN（漏检写）必须为 0**；
FP 可以有（fail-closed 方向安全）。

**oracle 独立性（硬性，防循环验证）**：

```text
被测与 oracle 必须是**两套不同的 event collector 实现**：
  被测 = strace/ptrace collector  ->  primary oracle = fanotify 或 eBPF
  被测 = fanotify/eBPF collector  ->  primary oracle = strace/ptrace
  LD_PRELOAD shim 只能作为第三个 supplemental oracle；其结果单独报告，
  不进入 authority-level FN 分母/分子
**禁止**用 Merkle snapshot 或 git status 充当 oracle：
  它们不产生 read_paths，也漏 write-then-restore，无法作为 event 层 ground truth。
必须同时输出 oracle_agreement（两个 event collector 的一致率与分歧清单）；
分歧即视为 FN 风险，必须逐条裁决并记录。
```

**结论表述限定（硬性）**：本项只能得出
**"在冻结对抗语料 + 冻结 workload 语料上、由所声明的 event collector 观察到的
FN = 0"**（记作 `collector_observed_FN = 0`）。
**禁止**写成"selector 无漏检"、"FN 恒为 0"或任何普遍性断言 ——
命令行写文件方式是开放集合，任何有限语料都不能支撑普遍结论。

**执行约束（硬性）**：

```text
- 被测仓库以 :ro 挂载
- 测试开始时在**容器内**复制到临时可写目录（--tmpfs /scratch 或容器可写层）
- 所有 mutation 命令只作用于该副本
- 绝不在宿主 worktree 上执行任何写操作，也不向宿主目录挂载可写卷
- 容器 --rm 退出即销毁，差分结果经 stdout 或显式 artifact 卷输出
```

```bash
# proposed，未授权执行
# 默认被测 authority = strace_ptrace_v1；primary oracle = fanotify_v1
docker run --rm --user "$(id -u):$(id -g)" \
  --cap-add SYS_PTRACE --cap-add SYS_ADMIN --security-opt seccomp=unconfined \
  -v "$PWD":/w:ro \
  -v /home/chris/Workspaces/kvcache-research/results/phase7_5_c40:/results/phase7_5_c40:rw \
  -v /home/chris/Workspaces/kvcache-research/results/BENCHMARK_RUN_LOG.jsonl:/global_results/BENCHMARK_RUN_LOG.jsonl:rw \
  --tmpfs /scratch:rw,size=2g -w /scratch -e HOME=/tmp \
  ghcr.io/ccdd2023/sglang@sha256:0be6e16e2eb288dfd5fa8b0b41015f61731a139fb961d3366ccedf834289d781 \
  bash -c 'cp -a /w /scratch/repo && cd /scratch/repo && \
           python -m benchmark.approx_kv.run_p75_selector_offline \
             --mode differential \
             --collector strace_ptrace_v1 \
             --oracle-collector fanotify_v1 \
             --supplemental-oracle-collector preload_shim_v1 \
             --out -'
```

**collector 参数与 capability 的对应（冻结）**：

| `--collector` | 必需 Docker 参数 | 可用作 |
| --- | --- | --- |
| `strace_ptrace_v1`（**默认 authority**） | `--cap-add SYS_PTRACE --security-opt seccomp=unconfined` | authority / primary oracle |
| `fanotify_v1` | `--cap-add SYS_ADMIN` | authority / oracle |
| `ebpf_v1` | `--cap-add BPF --cap-add PERFMON` | authority / oracle |
| `preload_shim_v1` | 无额外 capability | **只能作 supplemental oracle**（不覆盖静态链接/直接 syscall，不进入 authority-level FN） |

```text
硬性：
  - `--collector` 与 primary `--oracle-collector` **必须不同**，且二者都必须
    属 `{strace_ptrace_v1, fanotify_v1, ebpf_v1}`。
  - 若使用非默认collector，必须更新
    `provenance.collector_impl/collector_profiles/selected_profile_by_gate`；
    run记录实际profile，不能只引用allowed-max。
  - **没有可用的 event-level collector 时，本测试不得运行**
    （runner 必须在启动时检测 collector 可用性并直接失败退出），
    **禁止**改用 Merkle / git status 顶替后继续跑差分。
```

**成功标准**：对抗矩阵在冻结语料上全部通过；冻结语料上
`collector_observed_FN=0`；property `P1–P8` 各1000例无反例。
**失败标准**：任一 `collector_observed_FN > 0` → Gate 不通过（§21 Stop rule SR-1）。

### 17.2 T2 — Identity / Fingerprint（CPU）

覆盖：`token_hash` / `token_count` 不匹配拒绝；同 key 重注册产生新 `generation`；
stale handle 不能 `pin` / `load` / `release`；`model_id` / `tokenizer_revision` /
`chat_template` / `rope_config` / `dtype` / `layout` / `page_size` / `tp` / `pp` /
`image_digest` 全部进入 fingerprint；跨 fingerprint 复用必须拒绝。

**新增（相对 collaborator 实现）**：`source_paths_content_sha256`、
`worktree_generation`、`repo_id`、`branch` 必须进入 key；缺失即拒绝。
**并且**：静态扫描 + 动态注入双重检查，结论限定为
**"在已扫描代码路径中未发现 fingerprint bypass"**（`no_fingerprint_bypass_detected_in_scanned_paths`）；
扫描范围（文件清单 + 检查方法）必须在 evidence 中列明。
**禁止**写成"不存在 bypass"。

### 17.3 T3 — Dense / copy 覆盖（CPU）

对每个 plan 断言：`copied_spans ∪ dense_ranges == [0, len(target))`，
无重叠、无空洞（`require_full_coverage=True`）；多岛 lane 下额外断言岛间
non-overlap 与 `Σ island_len <= total_copy_budget`。

### 17.4 T4 — K/V + RoPE 张量测试（GPU）

覆盖：正 / 负 `rope_delta`；全层全 head；`rotary_dim < head_dim`；
`rotated_k_tokens == copied_k_tokens == span_len` 逐层成立；
V 逐元素与 source 相等（不旋转）；注入 `±1 rope_delta` 的 corruption canary 被检出。

### 17.5 T5 — Same-context corruption canary（GPU，常驻回归）

#### 17.5.1 必须先跑 baseline 稳定性 control（**冻结前置**）

直接断言"逐字符一致"会把**后端本身的非确定性**（atomics 归约顺序、
cuBLAS/cuDNN 算法选择、chunk 切分差异、TF32/FP16 累加顺序）误判为 C40 缺陷。
因此必须先测出 backend 的**baseline envelope**：

```text
Control-1  Dense-vs-Dense 重复稳定性
  同一 prompt、同一 server、连续两次 dense 执行
  记录 max|ΔK|, max|ΔV|, max|Δlogit|, 输出是否逐字符一致
  重复 >= 20 次，取分布

Control-2  E0-vs-Dense（exact cache 命中 vs 全 dense）
  同一 prompt，一次走 exact prefix cache，一次强制全 dense
  记录同样四个量
  重复 >= 20 次

baseline_envelope := {
  dK_max   : Control-1/2 中 max|ΔK| 的最大值
  dV_max   : 同上
  dlogit_max : 同上
  greedy_identical_rate : 输出逐字符一致的比例
}
该 envelope 在执行前**冻结**并写入 manifest（`tolerance_freeze` 段，sha256 绑定）。
```

#### 17.5.2 C40 same-context 判据（**相对 baseline，不是绝对**）

```text
C40 在 rope_delta == 0 的同上下文条件下执行 copy，断言：

  max|ΔK|     <= max(dtype_epsilon_bound, baseline_envelope.dK_max)
  max|ΔV|     <= max(dtype_epsilon_bound, baseline_envelope.dV_max)
  max|Δlogit| <= max(frozen_logit_tol,    baseline_envelope.dlogit_max)

**硬 fail 条件**：只有当上述任一量**超过 baseline envelope**时才判失败。
落在 envelope 内的差异归因于 backend 非确定性，**不构成** C40 缺陷。

**输出逐字符一致**的门：
  仅当 baseline_envelope.greedy_identical_rate == 1.0（即 Control-1/2 本身
  在 20 次重复中全部逐字符一致）时，才把"C40 输出逐字符一致"作为硬 gate。
  若 baseline 本身就不稳定（rate < 1.0），则改为：
    C40 的 greedy_identical_rate 不得**显著低于** baseline
    （单侧比例检验，alpha=0.05），并在 evidence 中披露两个 rate。
```

#### 17.5.3 corruption 检出能力

```text
注入 ±1 rope_delta 的 corruption canary 必须被检出。
若 baseline envelope 宽到连 ±1 rope_delta 都无法区分
  ⇒ 说明该配置下 same-context canary 不具备检出能力
  ⇒ 必须收紧配置（如固定 cuBLAS 算法、关闭 TF32）或换更敏感的探针，
     **不得**以"envelope 内"为由放行。
```

### 17.6 T6 — Cross-context logit / 输出 / 质量（GPU）

- 必须断言 `source_prefix_token_sha256 != target_prefix_token_sha256` 且
  `rope_delta != 0`，以证明**确实是跨上下文**；
- 报告 `Δlogit` 分布、top-1 一致率、KL 散度；
- **不对** cross-context top-1 一致率做方向性预测（不由 top-1 反推"是否跨上下文"）。

### 17.7 T7 — Exclusive fallback 分类学与四计数族（CPU + GPU）

族 2 的 `Σ_reason tokens == attempted_recovery_failed_dense_tokens`；族 1/3/4 各自封闭（§12.4）；无跨族混计；
每个 reason 至少有一个**直接触发**的测试用例（避免 Phase7 的
`unsupported <- store_miss` 间接归因缺陷）；`ADMISSION_DEFERRED` 不产生 terminal reason 的负向用例；arm-specific abort 按 §12.4.2 规则 A/B 处理的用例；`c40_*` 与 canonical inventory（含 prefix-family）的互斥性静态检查。

### 17.8 T8 — Lease / abort / reject / timeout / reset soak

`10k` 请求 soak，随机注入 reject / abort / timeout / reset / exception，
断言结束后：

```text
c40_active_leases == 0
c40_pending_produce == 0
c40_provisional_slots == 0
c40_orphan_count == 0
store.record_count / lease_count / orphan_count 与预期一致
gc_expired_leases 被周期性调用且有非零回收计数
```

### 17.9 T9 — 双向压力（bidirectional pressure）

在 `rho ∈ {1.5, 2.0}` 下交替施加"exact 侧压力"与"approx 侧压力"，
断言 allocator 的 stale victim 容忍生效、无 capacity error、无 double free、
无 stale handle、exact 纯净性保持（近似 KV 未污染 exact Radix）。

### 17.10 T10 — Chunk × body 全因子（隔离已知混淆源）

**完整因子定义（对齐 authority 报告 §9.10）**：

| 因子 | 水平 |
| --- | --- |
| `chunked_prefill_size` / `max_prefill_tokens` | `1024`、`2048`、`4096`（primary = `4096`） |
| island body 长度 | `256`、`512`、`1024`、`2048`、`4096` |
| `rho`（KV 压力） | `1.5`、`2.0` |
| arm | `D0` dense / `E0` exact / `C40-1R0` |

**分期执行（避免全笛卡尔积）**：

| 期 | 覆盖 | 说明 |
| --- | --- | --- |
| pilot（Stage C-1） | `chunk ∈ {4096}` × `body ∈ {512, 2048}` × `rho ∈ {2.0}`，**workload = W4a live corpus** | **不做全因子**；**`>= 3` restarts**，只估 `s_pilot` 并按 §19.2a 冻结 `n_confirmatory`。**禁止**用 W1 数据估 `s_pilot` |
| confirmatory（Stage D-1） | `chunk = 4096` × `body ∈ {512, 2048}` × `rho ∈ {1.5, 2.0}` | primary headline 唯一来源 |
| sensitivity（Stage D-2） | `chunk ∈ {1024, 2048}` × `body ∈ {512, 2048}` × `rho ∈ {2.0}` | 全部标 `headline=false` |
| 完整因子（Stage D-3，**mandatory**，预算须二次授权） | `chunk ∈ {1024,2048,4096}` × `body ∈ {256,512,1024,2048,4096}` × `rho ∈ {1.5,2.0}` × arm ∈ {`D0`,`E0`,`C40-1R0`} = **90 arm-cells**，按 chunk 打包为 **3 starts/restart**（§15.4b） | authority 报告 §9.10 定为**必须做的隔离实验**。未完成 ⇒ G6 记 `PARTIAL_EXIT`，mechanism disposition 只能取 `PARTIAL`/`INCONCLUSIVE`；`known_gap` **不替代完成**（§15.4b.4） |

**报告要求**：任何速度数字必须同时给出其 chunk 水平；只有 `chunk = 4096` 的结果
可作 headline；`chunk = 1024` / `2048` 只能作 sensitivity diagnostic
（`headline=false`）。Phase7 已证 chunk1024 可产生 `1.737x` 的假性 headline。

### 17.11 T11 — 四本账 + 实测 N（1/2/4/8）

对每个 cell 输出：

```text
target_only_ms, request_path_ms(server 侧), end_to_end_ms(estimand 用), full_lifecycle_ms
speedup_1, speedup_2, speedup_4, speedup_8          (full-setup, 实测)
speedup_incremental_1..8
break_even_N   (若 N<=8 未观察到，写 ">8 / not_observed")
```

**禁止插值外推**。

### 17.12 T12 — Selector overhead 单独测量

`C_selector` 必须在**全部请求（含 ineligible）**上测量，并分解为
`provenance_ms` / `tokenize_ms` / `unique_search_ms` / `manifest_io_ms`。
`C40-D` 臂（selector 运行但全 dense）用于独立验证控制路径成本。

### 17.13 T13 — Full-workload estimand（`theta_j` / `mu_theta` / `E_cond` / `w` / `r`）

见 §19.3。三臂（`D0` / `C40-1R0` / `R0` span-matched）+ 控制臂 `C40-D`
必须各自**完整执行同一冻结请求流**。

### 17.14 T14 — S0 / S4 matched coverage

`S0 = PolicyKind.S0_LRU`，`S4 = PolicyKind.S4_HIERARCHICAL`。
**必须相邻交替启动**（`S0,S4,S0,S4,...`），否则只能报
`seed_matched_non_adjacent_restart_comparison`（Phase7 的已知缺陷）。
同时报告 **matched coverage**：两臂 `expected_reusable_prefix_tokens > 0`
的请求集合必须一致。

### 17.15 T15 — Host residency（`H0` vs `H1`）

`H1` demand-load 下断言：加载失败 fail-closed 到 dense；
`residency_load_failed` 与 `c40_source_not_resident` 归因不重叠；
host 预算记账正确；无 host 侧 orphan。

### 17.16 T16 — 多岛（CL-A）

non-overlap property；`Σ island_len <= total_copy_budget`；
payoff optimizer 的确定性（同输入同输出）；岛数为 1 时与 `C40-1R0` 逐字段一致。

### 17.17 T17 — Optional repair（CL-B）

`C40-R1-k` 在 `k=0` 时**必须**退化为 `C40-1R0`（逐字段一致）；
`k>0` 时 leading-k 被划为 `DenseRange` 并逐层重算；
**禁止**把它描述为 CacheBlend / KVCOMM。

### 17.18 T18 — 四模式 prefetch composition（CL-D，条件）

`Dense` / `Coding-only` / `Prefetch-only` / `Combined` 四模式。
**硬性验收**：Combined 的 selected span 与 Coding-only **逐 token 相同**；
关闭 prefetch 精确恢复 Coding-only；无 lease / worker / CUDA event 泄漏；
额外覆盖 late / cancel / stale prefetch 三类时序。

### 17.19 T19 — Concurrency / multi-tenant / repo isolation（CL-E）

并发请求下：不同 repo/worktree/branch 的 source **互不可见**；
同 repo 并发 consume 同一 source 的 lease 计数正确；
并发 abort 不产生 orphan；决策可重放。

### 17.20 T20 — Manifest / provenance 测试

结果目录含自哈希 `RESULT_MANIFEST.json`，绑定 image digest、model+tokenizer
revision、code pin（commit + tree）、runner path + sha256、plan manifest
revision、每个 raw/log/central 文件 sha256、`known_gaps`。
`--check` 只读重放通过。CI lint：CI lint 使用 CR-4 的冻结命令，基线与目标均为空输出。

### 17.21 T21 — 执行环境与测量合同

| 项 | 规则 |
| --- | --- |
| 基础镜像 | digest 固定，**禁止 tag** |
| 依赖安装 | 专用 Docker layer；必须运行时安装时用 `pip install --target` + `PYTHONPATH` |
| 锁文件 | `requirements.lock`（含全部传递依赖与 hash） |
| 验收 | 相对已版本化 `pip check` baseline 无新增冲突；新 clean base 才要求零输出 |
| 记录 | 每次运行把 `pip freeze` 全量写入 artifact 并计入 manifest sha256 |
| warm-up | 每 arm 每 start 先跑 `W` 个 warm-up 请求全部丢弃；`W >= 2`，执行前冻结；仍写入 central log（`is_warmup=true`） |
| formal repeats | `M` 执行前冻结；arm 之间按 formal repeat **交替**（`A,B,C,A,B,C,...`） |
| reset | arm 之间完整 reset；`arm_interval_peak_device_bytes` 自上次完整 reset 起计 |
| 禁止 | 事后按结果决定丢弃哪些请求；任何 post-hoc 丢弃必须单独记录并在 disposition 中披露 |

### 17.22 T22 — Same-engine 对照（可选，Track D）

若要做跨方法比较，必须在**同一 engine、同一 model+revision、同一 prompt、
同一任务顺序、同一 generation limits、同一 chunk 配置**下重跑：
`R0`（底座既有）、`EPIC k ∈ {0,2,8,32}`（底座既有）、
`CacheBlend`（本项目 R2）、`KVCOMM`（本项目 R4，`authoritative_historical_diagnostic`）。

**禁止**引用外部论文的 `164/225`、`169/225`、`8.55×`、`4.77×` 与 C40 并列。

### 17.23 T23 — Exact-overlap clipping（CL-I，条件）

| ID | 断言 |
| --- | --- |
| `T23-1` | **disabled parity**：`SGLANG_APPROX_KV_C40_EXACT_OVERLAP_CLIP=0` 时，B-2 请求 outcome == `dense_ineligible` 且 `selector_reason == c40_exact_overlap_unsupported`，与 primary 逐字段一致；`overlap_clip.py` 的入口调用计数为 0 |
| `T23-2` | 开启时，B-2 请求的 island 被裁剪为 `[exact_length, target_end)`，且裁剪后仍满足 `length >= min_tokens`、唯一性、`target_end < len(prompt)`；否则 fail-closed |
| `T23-3` | 裁剪后 `rope_delta` 按新的 `target_start = exact_length` 重算，`source_offset` 同步平移 |
| `T23-4` | 裁剪成功的 outcome == `c40_copied` 且 `geometry == clipped_at_exact_boundary`；`c40_copied_total{geometry="clipped_at_exact_boundary"}` +1 |
| `T23-5` | **headline 隔离**：consolidator 在计算 primary `theta_j` 时排除 `geometry == clipped_at_exact_boundary` 的请求（或该 arm 整体标 conditional），断言 primary summary 中该 geometry 计数为 0 |
| `T23-6` | 裁剪路径同样遵守所有权分离（INV-5/INV-6/INV-8）与 chunk 钳制（`next_extend_boundary = exact_length`，即本轮 dense prefix 长度为 0） |

---

## 18. Workloads

### 18.1 W1 — Synthetic deterministic micro workload

| 项 | 内容 |
| --- | --- |
| 目的 | **仅** engineering / calibration diagnostic：可达性、张量正确性、时序仪表校准 |
| 构造 | 确定性生成 `[dense prefix][island][dense suffix]` 的 prompt；island 长度 ∈ `{128, 512, 1024, 2048, 4096}`；`rope_delta` 覆盖正负与零 |
| provenance | 合成 ToolEvent（结构化字段直接给定，**不经过命令解析**） |
| 用途 | T3 / T4 / T5 / T11 / T12；**Stage B**（G4 canary）与 **Stage C-0 calibration** |
| **硬性限制（P1-08）** | W1 的 source 不是由真实前序请求物化的（属构造式），因此**禁止**用 W1 数据计算 `theta_j` / `mu_theta` / `E_cond` / `w` / `r`，**禁止**进入 mechanism disposition 或任何 headline。W1 结果一律标 `workload=W1, role=engineering_diagnostic_only` |
| 依赖 | 无外部数据 |

### 18.2 W2 — Mutation / adversarial repository trace

| 项 | 内容 |
| --- | --- |
| 目的 | selector 正确性上限测试；差分测试 ground truth |
| 构造 | 在容器内临时可写副本上**真正执行** §17.1.1 的全部 mutation 用例（含 write-then-restore 类）；ground truth 由 §17.1.4 规定的**第二套 event collector**（与被测实现不同）给出；Merkle 与 `git status` 只作第三方报警，**不充当 oracle** |
| 容器要求 | 需 `--cap-add=SYS_PTRACE`（或等效实现所需 capability），并在授权 manifest 中列明（§9.2.3） |
| 用途 | T1（对抗矩阵 + 差分 + property）；Stage A |
| 依赖 | Docker `--tmpfs /scratch`；**绝不**写宿主 worktree |

### 18.3 W3 — 固定 workflow（`Architect → Coder → Debugger`）adapted with tool observations

| 项 | 内容 |
| --- | --- |
| 目的 | 与本项目 Phase5/Phase7 的既有 workflow 评测口径对齐 |
| 构造 | 沿用固定 workflow：`Architect → Coder → Debugger`，两轮 `Architect→Coder→Debugger→Coder→Debugger`，Debugger 失败后条件返回 Coder；5 个固定 workflow 对象（Architect ×1、Coder ×2、Debugger ×2）+ live filler replay |
| **本阶段新增** | 在每个 stage 之间插入**真实结构化 tool observation**（读文件、grep、以及部分写事件），使 C40 有真实 source 可消费，同时保留 Phase5 的对象结构与 rho 压力语义 |
| 用途 | T14（S0/S4 matched coverage）、T9（双向压力）；Stage F |
| 约束 | workflow 定义不得改动（`D-008`）；新增的只是 observation 层 |

### 18.4 W4a — C40 Live Trajectory Corpus v1（**默认 workload，冻结**）

> 取代此前"真实 trajectory 不可得就阻塞"的设计。W4a 是**默认**可执行的
> live corpus，不是 synthetic replay。

| 项 | 内容 |
| --- | --- |
| **名称（冻结）** | `W4a C40 Live Trajectory Corpus v1` |
| **性质** | **live**：每条 trajectory 都在 Docker 内对 versioned fixture repo **实时执行**工具调用并**实际发出**请求。<br>**source-producing 与 consume 的具体含义（冻结）**：<br>1. 一个**真实的** agent 请求（含该次工具 observation 的完整 prompt）被发送到 server 并正常完成；<br>2. 服务端在 `cache_finished_req` 时刻从**该请求自身已计算的 KV** 中物化 source（`approx_depth = 0`）；<br>3. 下一个**真实的** agent 请求（新的左上下文 + 相同 observation token）作为 target，尝试消费该 source。<br>三步全部在线发生，**不存在**离线构造或 replay 环节 —— 满足 PC-1（real prior-request source）与 PC-9（no synthetic replay） |
| **规模（下限）** | **`>= 24` 条 trajectory**，构成：<br>• `8` 条 **纯 read**（只读工具序列，预期高 eligibility）<br>• `8` 条 **read-then-write**（读后写同路径，预期被 invalidation 正确排除）<br>• `8` 条 **mixed / unknown**（混合读写、unknown_effect、timeout、truncated、write-then-restore） |
| **两类 agent 形态** | 每类 8 条中一半为 `Architect / Coder / Debugger` 固定 workflow 形态，一半为通用 coding-agent 形态（自由工具序列） |
| **fixture repos** | versioned fixture 仓库（固定 commit，随计划一同 pin），**不使用**宿主任何真实仓库 |
| **每条 trajectory 的硬性要求** | 至少发出 **1 个 source-producing request** 与 **1 个 target consume request**；两者之间插入真实工具调用；provenance 由 §9.2 的 authority collector 采集 |
| **冻结项** | `seed`、任务脚本、工具协议、`max_new_tokens` / 温度等 generation limits、fixture repo commit、trajectory 的 sha256 清单 |
| **可支撑的结论** | 机制层（`theta_j` / `mu_theta` / `E_cond` / `w` / `r` / `C_selector`）、正确性、覆盖率。**Stage C-1 pilot 与 Stage D confirmatory 的 primary workload 必须是 W4a** |
| **外部有效性限制（必须披露）** | fixture repos 与脚本化任务**不能**代表真实生产 coding-agent 的路径分布、文件规模分布与工具使用分布。因此 W4a 支撑的 `w` / `r` 只能表述为"**在 W4a corpus 上**"，**不得**外推为一般 coding agent 的覆盖率 |

### 18.4b W4b — 真实外部 trajectory（**conditional，Gate probe**）

| 项 | 内容 |
| --- | --- |
| 目的 | 提升外部有效性：验证 W4a 上的 `w` / `r` 是否在真实轨迹上仍然成立 |
| 可得性 | 由 **Gate probe** 判定（§27.1）：在 G1b 阶段执行一次只读探测，判定真实轨迹语料是否可在受控环境中获得且许可允许 |
| probe 通过 | 作为 W4a 的**补充轴**执行；结论标注 `trajectory_source=external_real` |
| probe 不通过 | 该轴标 `BLOCKED_EXTERNAL`，**不占预算、不阻塞任何 lane**；机制结论仍由 W4a 支撑，但外部有效性限制必须在 disposition 中显式保留 |
| **硬性** | W4b 缺席**不**降级为"用 synthetic 顶替"；W4a 本身就是 live 的，不存在 synthetic 替代问题 |

### 18.5 W5 — RepoBench-P

| 项 | 内容 |
| --- | --- |
| 目的 | 静态、便宜、可大样本的质量评测 |
| 指标 | `exact-line agreement`、`edit similarity`、`cache-ready TTFT` |
| 样本 | `>= 1000` 例 |
| 统计 | paired bootstrap 报告差值 95% CI；按 task cluster 重采样 |
| 用途 | Stage G-1 |
| **source-producing / consume 流程（必须明确）** | RepoBench-P 原生是单轮补全，**没有**天然的前序请求。因此必须显式构造两阶段协议：<br>**阶段 1（source-producing）**：对该 task 的 cross-file context，以一次**真实请求**读取并让服务端物化 source（`approx_depth = 0`）；<br>**阶段 2（consume）**：正式补全请求作为 target，尝试消费该 source。<br>两阶段的 prompt、顺序与 generation limits 全部冻结 |
| **覆盖率报告（强制）** | 每个 arm 必须报告：`effective_paired_tasks`、`w_quality`（time-weighted coverage）、`mean_copied_tokens_per_effective_task`、`eligible_task_rate`、`copy_coverage` 分布 |
| **NO_COVERAGE 规则** | 若低于 §18.6b 的**预先冻结**阈值（`tau_task` / `tau_coverage` / `tau_copied_tokens_per_effective_task`），质量比较**只能**报 `NO_COVERAGE`，**不得**冒充"C40 的质量" |

### 18.6 W6 — SWE-bench Verified（**calibration / confirmatory 严格 disjoint**，F-09）

#### 18.6.1 冻结的非劣性设计参数（**pre-data**）

```text
practical non-inferiority margin  M = **5 percentage points**（绝对差）
    即：C40 的 task 解决率不低于 Dense 减 5 pp
one-sided alpha                    = 0.05
power (1 - beta)                   = 0.80
primary task-level outcome         = majority_resolved（3 个配对 seed 中 >= 2 次 resolved）

**硬性**：M / alpha / power 在本计划即冻结。
若无法达到所需样本量，结论只能是 INCONCLUSIVE 或 BLOCKED，
**禁止**事后放宽 M（例如改成 10 pp）来"让结论成立"。
任何改动须归档旧 plan、创建新 revision、更新 design_sha256 并重新授权。
```

#### 18.6.2 两段式 disjoint split（**calibration 不进入最终检验**）

| 段 | 名称 | 任务集合 | 用途 | 是否进入最终检验 |
| --- | --- | --- | --- | --- |
| **W6a** | calibration | 从 SWE-bench Verified 中**随机抽取且冻结**的 `40` 个 task × 3 配对 seed | 估计 discordance rate `p_d`、run-to-run 方差、`effective_paired_tasks` 与 coverage；据此按 §18.6.3 计算 `n_confirm` | **否** |
| **W6b** | confirmatory | 与 W6a **完全不相交**（`W6a ∩ W6b = ∅`）的 task 集合，`n_confirm` 个 task × 3 配对 seed | 唯一的非劣性检验数据源 | **是** |

```text
硬性规则：
  1. W6a 与 W6b 的 task id 划分在**看到任何结果之前**冻结并写入 manifest
     （随机种子 + 划分清单 sha256）。
  2. W6a 的任何数据**禁止**进入最终 McNemar / 非劣性检验，
     也**禁止**与 W6b 合并（否则功效计算失效，等价于用同一批数据既定样本量又做检验）。
  3. 若 SWE-bench Verified 中可用且满足前置条件的 task 数
     **不足以**在 W6a 之外划出 n_confirm 个 disjoint task：
        quality disposition := INCONCLUSIVE_INSUFFICIENT_DISJOINT_TASKS
        （若连 W6a 都无法构成，则 BLOCKED_EXTERNAL）
     **禁止**复用 W6a task、**禁止**放宽 M、**禁止**降低 power 后仍称 confirmatory。
```

#### 18.6.3 非劣性样本量与检验算法（冻结）

```text
输入（来自 W6a）：
  p_d      = discordant pair rate（Dense 通过而 C40 失败，或反之，
             按聚合后的 majority_resolved 计）
  p_10     = P(Dense=1, C40=0)     # C40 劣于 Dense 的不一致对比例
  p_01     = P(Dense=0, C40=1)     # C40 优于 Dense 的不一致对比例
             p_d = p_10 + p_01

样本量（配对二分类非劣性，McNemar 型）：
  H0: (p_10 - p_01) >= M          # C40 劣化达到或超过 margin
  H1: (p_10 - p_01) <  M          # 非劣

  n_confirm = ceil( ( (z_{0.95} + z_{0.80})^2 * p_d ) / M^2 )
            = ceil( ( 6.182558 * p_d ) / 0.05^2 )
            = ceil( 2473.0 * p_d )

  示例（示意，非预测）：
    p_d = 0.05  ->  ceil(123.7)  = 124 tasks
    p_d = 0.10  ->  ceil(247.3)  = 248 tasks
    p_d = 0.20  ->  ceil(494.6)  = 495 tasks
  （若 n_confirm 超出可用 disjoint task 数或预算 ⇒ 见 §18.6.2 规则 3）

检验（在 W6b 上执行）：
  b = #{Dense=1, C40=0}      c = #{Dense=0, C40=1}      n_d = b + c
  非劣性检验统计量（连续性校正的配对差 z 检验）：
      d_hat  = (b - c) / n_task
      se_hat = sqrt( (b + c) / n_task^2 )
      z      = (M - d_hat) / se_hat
      单侧 p = 1 - Phi(z)
  判定：
      p < 0.05                 ->  NON_INFERIOR(margin = 5 pp)
      d_hat > 0 且 CI 下界 > 0 ->  DAMAGING（C40 确实劣化）
      其余                      ->  INCONCLUSIVE(<= X pp)，
                                  X 由 d_hat 的单侧 95% 上界给出并显式报告
  同时报告标准 McNemar 双侧检验作为 descriptive。
  bootstrap 按 **task** 聚簇重采样（同一 task 的 3 次 repeat 整体进出）。
```

#### 18.6.4 其他执行要求

| 项 | 内容 |
| --- | --- |
| 计量 | 以 **task-runs** 单列计量（`n_tasks × n_repeats × n_arms`）。质量评测**仍会启动 GPU server**，其 server starts 与 GPUh **必须独立记录**在 Track E 账下；**禁止**折算成 server starts，也**禁止**并入 Track A–D 的 cap |
| **source-producing / consume 流程** | SWE-bench 的 agent 轨迹天然多轮：**前序**工具 observation 请求作为 source-producing，**后续**推理/补丁请求作为 consume target。必须显式记录每个 task 的 `(source_request_id, target_request_id)` 对并写入 central log |
| **覆盖率报告（强制）** | 每 arm 报告 `effective_paired_tasks`、`w_quality`、`mean_copied_tokens_per_effective_task`、`eligible_task_rate`、`copy_coverage` 分布；W6a 与 W6b **分别**报告 |
| **NO_COVERAGE 规则** | 见 §18.6b（阈值 pre-data 冻结） |
| Early stop | 若 W6a 的 40 task × 3 repeats 中 **Dense 自身**的 run-to-run 翻转率 > 15%，先修 harness 稳定性，不进入 W6b |
| 授权 | W6a 与 W6b **分别**授权：W6a 属 Track E 初始申请；W6b 的 `n_confirm` 须在 W6a 完成后**二次授权** |

### 18.6b 质量评测的 `NO_COVERAGE` 规则（**冻结，适用于 W5 与 W6**）

```text
问题：若 C40 在质量 workload 上几乎从不生效（eligible 或 copy coverage 极低），
      那么"C40 arm"与"Dense arm"实际上执行的是**几乎相同的计算**，
      两者的质量差异只反映噪声，却会被误读为"C40 质量无损"。
      这是一种严重的误导。

**冻结阈值（PRE-DATA，本计划即冻结；观察后不得调整）**：
  tau_task
      = max(10, ceil(0.25 * n_tasks))
        单位 = **有效 paired task 数**（两臂都完成、且 C40 臂至少发生过
        一次成功 copy 的 task 数），不是比率
        例：n_tasks = 40  -> tau_task = max(10, 10) = 10
            n_tasks = 130 -> tau_task = max(10, 33) = 33
            n_tasks = 500 -> tau_task = max(10, 125) = 125
  tau_coverage
      = 0.10        （time-weighted coverage w_quality 的下限）
  tau_copied_tokens_per_effective_task
      = 128         （有效 task 上平均实际复制 token 数的下限）

派生量（冻结）：
  c40_effective_task_rate = effective_paired_tasks / n_tasks
  copy_coverage(task) = C40臂该task实际copied tokens /
                        该task全部target prefill tokens
  报告copy_coverage的median/p25/p75/min/max；它与time-weighted
  `w_quality`是不同量，不得互换。

**硬性（NEW-03）**：以上三个阈值在**看到任何 quality 数据之前**即已冻结
（写入本计划与 manifest 的 tolerance_freeze）。
**禁止**"由 quality pilot 观测后冻结" —— 那等于按结果选阈值。
任何改变必须归档旧 plan、创建新 plan revision、更新 design_sha256
并重新 review 与授权。

判定：
  if effective_paired_tasks < tau_task
     or w_quality < tau_coverage
     or mean_copied_tokens_per_effective_task < tau_copied_tokens_per_effective_task:
      quality disposition := NO_COVERAGE
      表述限定为：
        "在该 workload 上 C40 的有效作用面过小
         （effective_paired_tasks = X / n_tasks，w_quality = Y，
           mean_copied_tokens_per_effective_task = Z），
          本次评测**不能**支持任何关于 C40 质量的结论。"
      **禁止**表述为"未观察到质量损伤"、"质量无损"或"非劣"。
  else:
      正常进入 §19.8 的 McNemar / 非劣性分析，
      且结论必须与 c40_effective_task_rate 和 copy_coverage 同时呈现。
```

### 18.7 W7 — 可选：真实仓库 / source-version workload

| 项 | 内容 |
| --- | --- |
| 目的 | 验证 branch/worktree/source dependency invalidation（CL-H）在真实版本演化下的行为 |
| 构造 | 在一个真实仓库上按 commit 序列演进，观察 source 失效时机与 `w` 的变化 |
| 状态 | `optional`，不阻塞任何 core gate；可得性由 Gate probe 判定（§27.1） |

### 18.8 Workload × Test 对照表

| Workload | T1 | T2–T3 | T4–T7 | T8–T9 | T11–T13 | T14 | T15–T19 | T20–T21 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| W1 synthetic（**仅 engineering/calibration**） | | ✔ | ✔ | ✔ | — | | — | ✔ |
| W2 adversarial | ✔ | ✔ | | | | | | ✔ |
| W3 workflow | | | | ✔ | ✔ | ✔ | ✔ | ✔ |
| W4a live corpus | ✔ | ✔ | ✔ | ✔ | ✔ | | ✔ | ✔ |
| W4b external（conditional） | ✔ | | | | ✔ | | | ✔ |
| W5 RepoBench-P | | | | | | | | ✔ |
| W6 SWE-bench | | | | | | | | ✔ |
| W7 real repo | ✔ | | | | | | ✔ | ✔ |

---

## 19. 统计合同

> **本节的每一条都是硬性约束。违反任一条 ⇒ 该实验判 `INVALID`，不进入 disposition。**

### 19.1 独立复制单元

```text
latency 的独立复制单元 = server restart
  同一 server 内的 formal request **不是**独立样本
  primary estimand 为 restart-level 的 theta_j（§19.2a），
    每个 restart 贡献**恰好一个** theta_j
  所有 latency CI 用 restart-level cluster bootstrap
    （对 restart 重采样，restart 内保留全部请求）

accuracy 的独立复制单元 = task
  一个 task 的多个请求只贡献一个 pass/fail
  accuracy CI 按 task 聚簇重采样（同一 task 的多次 repeat 整体进出）

禁止：对 formal request 做朴素 bootstrap
```

### 19.2 三阶段执行计划

| 阶段 | 内容 | 允许的结论 |
| --- | --- | --- |
| **restart-0（engineering screening）** | 单个 restart；只验证配置可达、无 capacity error、遥测字段完整、reset/orphan/lease 归零、fallback 分类学自洽 | **只能**得出"工程可执行 / 不可执行"。**不得**用其未知方差做任何效应判定，也不得据此 early stop |
| **pilot（`>= 3` 个独立 restart）** | 估计 restart 间方差 `s_pilot`、请求内自相关；对 accuracy 估计 **discordant pair rate** | 产出方差与 discordance 估计；据此**冻结 confirmatory n 与判定阈值**（§19.2a）。**不得**给出 `POSITIVE` / `NEGATIVE` |
| **confirmatory** | 按冻结计划执行；`n` 由 §19.2a 计算，**默认下限 4 restarts/cell** | 允许给出 `POSITIVE` / `NEGATIVE` / `SMALL_POSITIVE_BELOW_MDE` / `INCONCLUSIVE`（§19.2b） |

> pilot 从 `>= 2` 提升到 **`>= 3`**：`n = 2` 时样本标准差只有 1 个自由度，
> 其抽样分布极宽（`sd` 的相对标准误约 `52%`），无法支撑任何样本量计算。
> `n = 3` 是能给出有意义 `s_pilot` 的最小值，仍需在 disposition 中标注其精度局限。

### 19.2a Primary performance estimand 与样本量计算（**冻结**）

#### 19.2a.1 estimand 定义

**独立复制单元 = server restart。** 对第 `j` 个 restart（`j = 1..n`），
Dense-full 与 C40-full 在**同一 restart 配对块**内交替执行完整请求流，定义

```text
S_D,j = Σ_{i∈All_j} T_dense-full(i)        # 第 j 个 restart 的 Dense 臂总时间
S_C,j = Σ_{i∈All_j} T_c40-full(i)          # 第 j 个 restart 的 C40  臂总时间

theta_j = log( S_D,j / S_C,j )             # restart-level 配对 log-ratio

其中：
  All_j = 第 j 个 restart 内两臂都正常完成的 formal 请求（§12.4.3）
  T(i)  = end_to_end_ms(i) = selector_total_ms(i) + request_path_ms(i)
          （C_selector 已内生计入，§13.4）
```

**primary estimand（冻结）**：

```text
mu_theta = mean_j(theta_j)          # restart-level 配对均值，**这是 primary**
报告时同时给出：
  exp(mu_theta)  = **geometric mean speedup**（restart 的几何平均加速比）
```

**descriptive 辅助量（**不得**与 primary 等同）**：

```text
E_work_pooled = ( Σ_j S_D,j ) / ( Σ_j S_C,j )        # pooled ratio-of-sums

关系与禁止：
  E_work_pooled 是**按 restart 时长加权**的比值，
  exp(mu_theta) 是**等权**的几何平均比值。
  两者在 restart 时长不等或 theta_j 有离散度时**不相等**，
  且 pooled 量会被最长的 restart 主导。
  **禁止**把 E_work_pooled 当作 primary、
  **禁止**对它做 restart-level CI 后按 primary 解读、
  **禁止**在同一句话中把二者混用或互相替代。
  E_work_pooled 只作 **descriptive**，必须显式标注
  `estimand_role = "descriptive_pooled"`。
```

**尺度约定（冻结）**：

```text
theta 的单位是**自然对数比**（log-ratio），无量纲。
  theta = 0                    ⇒ 两臂等速
  theta > 0                    ⇒ C40 更快
  delta0 = log(1.05) = 0.048790  ⇒ 5%  加速（practical threshold）
  delta1 = log(1.10) = 0.095310  ⇒ 10% 加速（design alternative）
使用 log 尺度：比值分布右偏，log 后近似对称，配对差的正态近似更可靠，
CI 可指数化回比值尺度报告。
```

#### 19.2a.2 `s_pilot` 的 estimand 与尺度

```text
s_pilot = sample standard deviation of { theta_j : j = 1..n_pilot }
        = sqrt( Σ_j (theta_j - mean_pilot)^2 / (n_pilot - 1) )

estimand : restart 间 theta 的总体标准差 sigma_theta
scale    : 与 theta 同尺度，即 **log-ratio 单位**，无量纲
自由度   : n_pilot - 1（n_pilot >= 3 ⇒ df >= 2）

s_pilot **不是**请求级时间的标准差，**不是**比值尺度的标准差。
任何把 s_pilot 与毫秒或百分比直接比较的写法都是错误的。
```

#### 19.2a.3 冻结的设计参数（**pre-data，改动须新 plan revision**）

```text
delta0 (practical threshold)   = log(1.05) = 0.048790
delta1 (design alternative)    = log(1.10) = 0.095310
alpha  (one-sided)             = 0.05
power  (1 - beta)              = 0.80
n_min                          = 4

**硬性**：delta0 / delta1 / alpha / power 必须在**看到任何 confirmatory 数据前**
冻结。若要改变其中任一项，必须：
  1. 归档当前 plan 为 `IMPLEMENTATION_PLAN_PHASE7_5_C40_V<n>_ARCHIVED.md`；
  2. 创建新 plan revision 并更新 design_sha256；
  3. 重新走 independent review 与用户授权。
**禁止**在观察数据后调整 delta1（那会使功效计算失去意义）。
```

> `delta1 > delta0` 的设计含义：以"检出 10% 加速"为设计目标来定样本量，
> 而判定阈值仍用 5%。这样在真实效应约为 10% 时有 80% 功效把 CI 下界推过 5%。

#### 19.2a.4 样本量计算

**主公式（正态近似）**：

```text
  n = ceil( ( (z_{0.95} + z_{0.80}) * s_pilot / (delta1 - delta0) )^2 )

  z_{0.95} = 1.644854      （one-sided alpha = 0.05）
  z_{0.80} = 0.841621      （power = 0.80）
  z_sum    = 2.486475
  delta1 - delta0 = 0.095310 - 0.048790 = 0.046520

  => n = ceil( (2.486475 * s_pilot / 0.046520)^2 )
       = ceil( 2856.9 * s_pilot^2 )

示例（示意，非预测）：
  s_pilot = 0.01  =>  2856.9 * 0.0001 = 0.29   =>  n_min 生效  =>  n = 4
  s_pilot = 0.02  =>  2856.9 * 0.0004 = 1.14   =>  n_min 生效  =>  n = 4
  s_pilot = 0.04  =>  2856.9 * 0.0016 = 4.57   =>  n = 5（t 校正后可能 6）
  s_pilot = 0.08  =>  2856.9 * 0.0064 = 18.28  =>  n = 19（t 校正后约 21）
```

**校正步骤（必做）**：

```text
Step 1  n0 = max( ceil(2856.9 * s_pilot^2), n_min )

Step 2  t 分布迭代校正（sigma 由 n_pilot 估计，正态近似偏乐观）：
          repeat:
              n_new = ceil( ( (t_{0.95, n-1} + t_{0.80, n-1}) * s_pilot
                              / (delta1 - delta0) )^2 )
              n = max(n_new, n_min)
          until 收敛（通常 1–2 次）

Step 3  cluster bootstrap simulation 校正（当 pilot 的 theta_j 明显非正态）：
          for n_candidate in n(Step 2), n+1, ..., n_cap:
              hits = 0
              for b in 1..B (B >= 10000):
                  # 在 delta1 备择下重采样
                  sample = bootstrap_resample(
                               {theta_j - mean_pilot + delta1},
                               size = n_candidate, replace = True)
                  ci90 = restart_cluster_bootstrap_ci(sample, level=0.90,
                                                      two_sided=True)
                  if ci90.lower > delta0: hits += 1     # L90 == 单侧 95% 下界
              if hits / B >= 0.80: n_confirmatory = n_candidate; break

Step 4  n_confirmatory = max(Step 2/3 结果, n_min = 4)

Step 5  预算检查：
          starts_needed = n_confirmatory * 3        # D-3 按 chunk 打包，§15.4b
          if starts_needed > 已授权 start cap:
              **不执行**；提交二次授权申请（含 s_pilot、n、starts、GPUh 估算）
              在获得授权前**不得**以更小的 n 执行并宣称 confirmatory
```

**冻结要求**：pilot 结束后把 `s_pilot`、`n_pilot`、所选校正方法、
`n_confirmatory`、`starts_needed` 与完整计算过程写入
`p75-plan-manifest` 的 `statistics_freeze` 段并做 sha256 绑定；
**冻结之后不得因结果不利而修改**。

#### 19.2b 四态判定规则（**冻结**）

**决策用 CI 的口径（F-05，冻结）**：

```text
决策使用 **90% two-sided CI**：[L90, U90]
  = restart-cluster bootstrap 的 5% 与 95% 分位（n 小时同时报 t 分布 90% CI，
    取两者中更保守者）

为什么是 90% 而不是 95%：
  本计划的全部判定都是**单侧**命题
    POSITIVE  : theta > delta0        （单侧 alpha = 0.05）
    NEGATIVE  : theta < 0             （单侧 alpha = 0.05）
  90% two-sided CI 的**每一个端点**恰好等价于一个 **单侧 95% 界**：
    L90 是单侧 95% 下界，U90 是单侧 95% 上界。
  因此用 [L90, U90] 做上述判定，其单侧 type-I error 恰为 0.05，
  与 §19.2a.3 冻结的 alpha = 0.05 **一致**，
  也与样本量公式中使用的 z_{0.95} = 1.644854 **一致**。
  若改用 95% two-sided CI（端点为单侧 97.5% 界），
  则实际单侧 alpha = 0.025，与样本量公式不匹配，会系统性欠功效。

另须报告（**descriptive only，不用于四态判定**）：
  95% two-sided CI [L95, U95]，标注 `ci_role = "descriptive_95"`
```

设 `[L90, U90]` 为 `mu_theta` 的 **restart-cluster bootstrap 90% two-sided CI**：

| 判定 | 条件（**必须同时满足所列全部**） | 含义 |
| --- | --- | --- |
| **`POSITIVE`** | `L90 > delta0` | 有统计证据表明加速**超过** 5% 的实用阈值 |
| **`NEGATIVE`** | `U90 < 0` | 有统计证据表明 C40 **确实变慢**（整个 CI 在"变慢"一侧） |
| **`SMALL_POSITIVE_BELOW_MDE`** | `L90 > 0` **且** `U90 < delta0` | 有统计证据表明**确实变快**，但幅度**确定地小于** 5%（CI 完全落在 `(0, delta0)` 内） |
| **`INCONCLUSIVE`** | 其余全部，**包括**：CI 跨 `0`（`L90 <= 0 <= U90`）、CI 跨 `delta0`（`L90 <= delta0 <= U90`）、CI 同时跨两者 | 数据不足以区分 |

```text
硬性规则（写入 disposition 与所有 summary）：
  0. 四态判定**只用** [L90, U90]；95% CI 只作 descriptive，**禁止**用于判定。
  1. CI 跨 0（即比值 CI 覆盖 1）⇒ **INCONCLUSIVE**，绝不判 NEGATIVE。
  2. CI 跨 delta0 ⇒ **INCONCLUSIVE**，绝不判 SMALL_POSITIVE_BELOW_MDE。
     （旧的 "U < delta0 即 BELOW_MDE" 规则会把"跨 0 的 CI"错误吞进
       BELOW_MDE，本版已删除该规则。）
  3. SMALL_POSITIVE_BELOW_MDE 是一个**正向但不足**的结论，
     必须同时报告 exp(L)、exp(U) 让读者看到区间落在 1.0x 与 1.05x 之间。
  4. 四态判定只允许在 confirmatory 阶段给出；pilot 只能给
     PILOT_UNDERPOWERED / PILOT_BELOW_MDE 之类的过程性标签。
```

**判定示例（log-ratio）**：

| `[L90, U90]` | 判定 |
| --- | --- |
| `[0.060, 0.120]` | `POSITIVE`（`L90 > 0.048790`） |
| `[-0.080, -0.010]` | `NEGATIVE`（`U90 < 0`） |
| `[0.010, 0.040]` | `SMALL_POSITIVE_BELOW_MDE`（`L90 > 0` 且 `U90 < 0.048790`） |
| `[-0.010, 0.040]` | `INCONCLUSIVE`（跨 0） |
| `[0.020, 0.070]` | `INCONCLUSIVE`（跨 `delta0`） |
| `[-0.020, 0.090]` | `INCONCLUSIVE`（同时跨 0 与 `delta0`） |

**报告要求**：四态判定必须与 `n`、`n_pilot`、`s_pilot`、`delta0`、`delta1`、
CI 方法、`[L90, U90]`、`exp(L90)` / `exp(U90)`、descriptive `[L95, U95]`、
`E_work_pooled`（标 descriptive）、`w`、`r`、`C_selector` 同时出现；
缺任一项判 `INVALID`。

### 19.3 合法 estimand（ratio-of-sums，禁止 median 合成）

| 记号 | 定义（**同一批请求上的配对总时间比**） |
| --- | --- |
| **`E_cond`** | `Σ_{i∈Elig} T_dense(i) / Σ_{i∈Elig} T_C40(i)`；`Elig` 为**预先冻结**的 eligible 集合；`T` 取 `end_to_end_ms`（**含 `C_selector`**）；`target_only` 与 server-only `request_path` 另行分别报告。这是 ratio-of-sums，**不是** median 的比 |
| **`E_work_pooled`**（**descriptive only**） | `(Σ_j S_D,j) / (Σ_j S_C,j)`，`T` 同取 `end_to_end_ms`；Dense 与 C40 都必须各自**真实执行完整、有状态的请求流**，保留相同顺序、cache pressure、source 生产/消费和 ineligible dense 路径。**这不是 primary** —— primary 是 `mu_theta = mean_j(theta_j)`（§19.2a），`E_work_pooled` 必须标 `estimand_role="descriptive_pooled"` 且**不得**与 `exp(mu_theta)` 混用 |
| **`w`** time-weighted coverage | `Σ_{i∈Elig} T_dense(i) / Σ_{i∈All} T_dense(i)`（按 restart 计算后取均值，同时报 pooled 值）；分子分母**都取 dense 基线时间**，因此 `w` 与 C40 是否变快无关 |
| **`r`** eligibility rate（计数） | `|Elig| / |All|`，附 Wilson 95% CI 与 skip_reason 直方图。**仅作描述**，不参与任何加速换算 |
| **`C_selector`** | selector 自身开销（路径提取、tokenize、唯一性搜索、manifest I/O），**在 ineligible 请求上也会发生**。它通过 `end_to_end_ms = selector_total_ms + request_path_ms` **已经内生地计入** `E_cond` / `theta_j` 的分母，同时必须单独报告其绝对值与分项 |

**禁止**：

```text
禁止 median-of-ratios 合成
禁止 1/((1-f)+f/s) 这类 Amdahl 式 request-fraction × median-speedup 公式作为硬约束
禁止 survivorship 差分（"eligible 子集 vs 全体"的差值无法归因，取消该差分）
禁止 把 eligible-only 结果与另一条 Dense trace 事后拼接
```

### 19.4 Counterfactual 设计（四臂，完整执行）

```text
Step 1  冻结一条可重放 trajectory：每个请求的完整 prompt、tool observation、
        顺序、source-producing event 与预期 repository generation。
Step 2  对该冻结 trajectory 离线运行 selector，冻结：
          Elig 集合 + 每个 i 的 span 元组
          (source_start, target_start, length, rope_delta, source_key, generation)
        冻结清单写入 manifest 并做 sha256 绑定（§14.3）。
Step 3  四臂分别从 clean reset 开始，**完整执行 All 请求**：
          臂 A  D0 dense-full       : 所有请求 dense，仍执行完整 source lifecycle
          臂 B  C40-1R0 full        : eligible 执行 C40；ineligible 在**同臂** dense
          臂 C  R0 span-matched full: eligible 回放同 span 的 R0；ineligible 同臂 dense
          臂 D  C40-D selector-only : selector 完整运行但强制全 dense
        四臂都保留请求顺序、cache pressure、source register/release 与下一请求状态。
Step 4  每个 restart 严格按 §19.2a 对完整 All 集合求和：
          S_D,j = Σ_{i∈All_j} end_to_end_ms(Dense-full, i)
          S_C,j = Σ_{i∈All_j} end_to_end_ms(C40-full, i)
        不含请求间 idle gap、server startup、reset 或臂间 orchestration 时间。
        由该唯一口径得到 theta_j；primary = mu_theta = mean_j(theta_j)。
        整条 trace 的 wall-clock 可另报 `trace_wall_clock_ms`，仅 diagnostic，
        不得代入 primary estimand。
        E_cond 只是从**同一条**完整 trace 中切出冻结的 Elig 请求做次级分析，
        不产生合成 workload。
```

**臂差分的正确含义**：

| 差分 | 允许的命名 | **禁止**的命名 |
| --- | --- | --- |
| `E_cond(C) / E_cond(B)` | `overhead_selector_control`（selector 判定与控制路径开销） | "selection gain"、"selection 收益" |
| 臂 D 与臂 A 的 restart-level log-ratio 均值 | `selector_only_overhead`（selector 全开但零复制的纯开销） | 任何形式的收益 |
| `E_cond(B)` vs 全体 | — | `Δ_survivorship`（**已取消**，请求集合不同无法归因） |

**survivorship 的正确处理是结构性的**：用 `w` 显式披露"C40 只能作用于 dense
基线时间的百分之几"；所有 conditional 结论**必须**与 `w` 同时出现；
全 workload 结论**只**看由完整 C40-full trace 实测的 `mu_theta`（primary）；`E_work_pooled` 只作 descriptive。

### 19.5 汇报模板（强制）

```text
C40 = G40 × R0 | chunk=<...> | model=<id@rev> | image=<digest> | restarts=n=<n>
  stage                            : <restart0 | pilot | confirmatory>
  s_pilot (log-ratio scale)        : ___    (n_pilot=___, df=___)
  delta0 / delta1                  : log(1.05)=0.048790 / log(1.10)=0.095310
  alpha / power                    : 0.05 (one-sided) / 0.80
  n_confirmatory (frozen)          : ___    method=<normal+t | bootstrap>
  starts_needed = n * 3            : ___
  mu_theta (log-ratio, PRIMARY)    : ___    CI90 [L90=___, U90=___] (decision CI)
  ci_method                        : <bootstrap | t | conservative_of_both>
                                            CI95 [___, ___] ci_role="descriptive_95"
  exp(mu_theta) geometric speedup  : ___    [exp(L90)=___, exp(U90)=___]
  E_work_pooled (DESCRIPTIVE only) : ___    estimand_role="descriptive_pooled"
  DECISION : <POSITIVE | SMALL_POSITIVE_BELOW_MDE | NEGATIVE | INCONCLUSIVE>
  r  eligibility rate (count)      : e/N = ___   Wilson95 [___, ___]
  w  time-weighted coverage        : ___         (dense-baseline time share)
  C_selector overhead              : ___ ms/req  (measured on ALL requests)
  E_cond (ratio-of-sums, eligible) : ___         cluster-bootstrap 90% CI [___, ___]
  E_work_pooled (DESCRIPTIVE)      : ___         estimand_role="descriptive_pooled"
  arm C span-matched R0            : ___   -> overhead_selector_control = ___
  arm D selector-only (C40-D)      : ___   -> selector_only_overhead    = ___
  exact_prefix_hit_tokens by arm   : D0=___ E0=___ R0=___ C40-D=___ C40=___
  radix insert suppressed by arm   : R0=___ C40=___ (copy_committed only)
  radix stash suppressed chunks    : R0=___ C40=___
  final_radix_insert_policy        : <matches frozen §15.1 table>
  Phase7 R0 reference (same image/model, chunk4096) : 0.772-0.936  (NEGATIVE)
```

### 19.6 Headline 判据

| 结论级别 | 条件（全部以 §19.2b 的 `theta` 四态判定为准） |
| --- | --- |
| 允许 workload-level headline | `mu_theta` 判定为 **`POSITIVE`**（`L > log(1.05)`），且 `w` / `r` / `C_selector` / `n` / `s_pilot` 同时披露，且 chunk 为 primary（`4096`），且处于 confirmatory 阶段 |
| 只允许 conditional 表述 | `E_cond` 对应的 log-ratio `L90 > delta0`，但 workload 级 `mu_theta` 为 `SMALL_POSITIVE_BELOW_MDE` 或 `INCONCLUSIVE` ⇒ 必须写成"仅在占 dense 基线时间 `w` 的 eligible 子集上观察到条件加速；全 workload 判定为 `<SMALL_POSITIVE_BELOW_MDE\|INCONCLUSIVE>`" |
| 判 `SMALL_POSITIVE_BELOW_MDE` | `L90 > 0` 且 `U90 < delta0`：确实变快但幅度确定小于 5% |
| 判 `NEGATIVE` | **仅当** `U90 < 0`（confirmatory 阶段）。CI 跨 0 **不得**判 `NEGATIVE` |
| 判 `INCONCLUSIVE` | 其余全部，含 CI 跨 0 与 CI 跨 `delta0` |
| 判 `INVALID`（不进 disposition） | 未完整四臂、缺核心统计/臂C、abort处理错误，或缺per-arm exact-hit/stash-suppression/insert-suppression字段，或实际final insertion policy与§15.1冻结表不符 |

### 19.7 配对与 p95

```text
配对方式：同一 (body, rho, restart) 下的**相邻 launch block**
          不相邻只能称 seed_matched_non_adjacent_restart_comparison
p95     ：ratio_of_marginal_p95s **不是**配对统计量，必须标 p95_pairing="nonpaired"
```

### 19.8 质量统计

```text
每 task/arm 使用 3 个固定且配对的 seed
primary task-level binary outcome = majority_resolved（3 次中 >= 2 次 official resolved）
McNemar 只对 Dense 与 C40 的 paired majority_resolved 做检验
n_discordant = 聚合后 task pair 中一臂通过、另一臂失败的 task 数
McNemar 功效计算基于 task-level discordant rate，**不是** Dense pass rate 或总 repeat 数
secondary：task random intercept 层次二项模型 或 task-cluster bootstrap
质量结论表述：
  "在该样本量与实测 discordant rate 下，无法排除 <= X pp 的损伤"（给出 X 及推导）
  **禁止**写成"无损伤" / "质量无损"
非劣性margin已在任何质量数据前冻结为`5 pp`；W6a只估discordance/方差和
confirmatory样本量，**不得**改变margin，也不得预先宣称无损
```

### 19.9 必须披露的配置

```text
chunked_prefill_size, max_prefill_tokens, page_size, tp/pp,
eviction policy, HiCache on/off, model+tokenizer revision, image digest,
c40 config（min_tokens / copy_cap / rolling_groups / max_islands / repair_k /
max_chain_depth / validation）、selector_version、tool_provenance_schema_version

所有latency/mechanism/system measured run必须：
  SGLANG_APPROX_KV_C40_VALIDATION=0
高成本`torch.isin(...).any().item()` ownership检查只在validation-on的
CPU/property/debug canary运行，禁止把其同步开销混入theta_j。
唯一豁免是正确性必需的admission final equality guard
`torch.equal(borrowed, current_prefix)`；其耗时必须计入
`c40_admission_guard_ms`并自然进入end_to_end_ms。

还必须报告`c40_exclusive_owner_rounds_total`及这些轮的batch-size分布；
forced-middle通过耗尽chunk budget形成单owner轮属于系统机制成本，必须自然计入
end_to_end_ms并在summary披露。

还必须按arm报告：
  p75_exact_prefix_hit_tokens_total
  p75_radix_insert_suppressed_requests_total（适用arm）
  p75_radix_stash_suppressed_chunks_total（C40 lifecycle arm）
  final_radix_insert_policy
并明确：copy-committed suppression是primary系统效应；fully-dense fallback在
prefill完成后恢复normal insertion。任何arm策略漂移判engineering invalid。
```

### 19.10 三个可证伪预测（预注册）

| ID | 预测 | 证伪条件 |
| --- | --- | --- |
| `PR-C40-1` | 在冻结语料上，结构化 event-level selector（相对命令正则 baseline）会使 eligible 比例（`r` 与 `w` 同时）**显著下降**，因为大量 mixed read-write group、unknown-effect group 与 write-then-restore group 被正确排除 | 两种判定方式的 eligible 率差异不显著（配对 bootstrap，α = 0.05） |
| `PR-C40-2` | 在固定 `chunk4096` + Qwen3-0.6B 下，confirmatory 的 `mu_theta` 判定**不会**是 `POSITIVE` | `mu_theta` 的 `L90 > delta0`（即判定为 `POSITIVE`）。**注**：这是把不利先验形式化为可证伪命题，**不是**断言 C40 必负；`SMALL_POSITIVE_BELOW_MDE` / `INCONCLUSIVE` 都不构成对该预测的证伪 |
| `PR-C40-3` | same-context canary 下，`max\|ΔK\|` / `max\|ΔV\|` / `max\|Δlogit\|` 全部落在 `max(dtype/frozen tol, baseline_envelope)` 内 | 超出 baseline envelope ⇒ 数据面实现缺陷。**不对** cross-context top-1 一致率做方向性预测 |

---

## 20. Gates P7.5-G0a – P7.5-G11

> 每个 Gate 都是**独立授权单元**；后一个 Gate **不由**前一个 Gate 的结果自动触发。
>
> **当前授权状态**：`P7.5-G0a` 与 `P7.5-G0q` = `AUTHORIZED`
> （二者都在 `plan_drafting/review_authorized = true` 范围内：纯文档 / 只读提取，
> 不建 branch、不写代码、不跑 Docker）。
> **其余全部 Gate（G1a 起）= `PENDING USER AUTHORIZATION`**。
>
> **冻结的 Gate 顺序**：
> `G0a → G0q → G1a → G0b → G1b → G2 → G3 → G4 → G5 → G6 → G7/G8 → G9 → G10 → G11`。
> 前五个的顺序是为消除循环依赖并保证 clean-room 输入齐备而冻结的，**不可调换**（§20.1）。

### P7.5-G0a — Document Authority / Plan Review（**纯文档，当前授权**）

| 项 | 内容 |
| --- | --- |
| **Entry** | 用户授权编制与审阅计划（已满足）；authority 文档已读取 |
| **Actions** | 冻结方法定义、branch/base/命名/结果目录、能力三层、架构、状态机、provenance schema、统计合同、Gate 顺序、Stop rules、预算结构；定义 CR-1..CR-9、manifest schema 与 quarantine 流程（§4.6）；把 Phase7 R0 先验写入计划；执行 independent review 并生成 **versioned review artifact**；同步 `PROJECT.md` / `TRACKING.md` / `HANDOFF.md` |
| **Exit** | 计划经 independent review 且 closure artifact 已版本化（自哈希 + 绑定文档 sha256），其中记载的 `open P0/P1` 计数为 `0/0`；三份 authority 文档已同步 |
| **Stop** | review 出现无法闭合的 P0；或用户不授权后续 Gate |
| **Artifacts** | 本文件；`evidence/review/plan-review-*.json`（可暂存于文档仓库，branch 创建后迁入结果目录） |
| **Resource** | host 只读，0 GPU，**无 Docker** |
| **Authorization** | `AUTHORIZED`（`plan_drafting_authorized` + `plan_review_authorized`） |

> **G0a 不产出任何 manifest 文件**：manifest builder 是代码，必须在新 branch 上
> 实现（G0b）。这是消除"G0 需要 manifest → manifest 需要 branch → branch 需要
> G0 Exit"循环的关键。

---

### P7.5-G0q — Quarantine Signature Extraction（**G0a 子项，当前授权范围内**）

| 项 | 内容 |
| --- | --- |
| **Entry** | G0a 的计划正文已冻结 clean-room 合同（§4.6）；`plan_drafting/review_authorized = true`。**不需要** branch、**不需要** implementation 授权、**不需要** Docker |
| **执行者** | **隔离 reviewer**（不参与 Phase7.5 任何实现的独立会话/执行者）。实施者**不得**执行本 Gate，也**不得**读取 collaborator ref 的任何文件内容 |
| **Actions** | 对`git rev-list <merge_base>..<collaborator_tip>`范围提取：`commit_ids[]`、`exclusive_blob_hashes[]`（排除冻结base已有blob）、`patch_id_alerts[]`、`ast_signatures_file[]`、`ast_signatures_function[]`、`allowed_shared_signatures[]`。产出自哈希quarantine manifest，只含哈希/签名，不含源码文本、标识符原文、注释、字符串常量或明文路径 |
| **Exit（全部满足才算 Exit）** | manifest生成/自哈希/无源码文本；范围与五类签名齐备；`quarantine_status=AVAILABLE`；artifact版本化。**此Gate只产出reference signatures，不存在new-branch match，故不要求adjudication** |
| **Stop / BLOCKED** | collaborator ref 不可获得（fetch 受限、ref 已删除、无访问权限）⇒ `quarantine_status := "BLOCKED_QUARANTINE_INPUT"`，**G0q 判为未通过**。<br>**此时不得进入 G1a**（§20.1 的 bootstrap 链在此中断），因为 CR-3/CR-3b 将缺失全部比对输入，clean-room 结论无法达到本计划要求的强度。<br>**禁止**把缺输入的 CR-3 记为"全绿"或"skipped 但通过"。<br>解除方式：取得 ref 访问后重跑 G0q；或由用户显式授权降级路径（须写入新 plan revision，并在 disposition 中把 clean-room 结论强度显式下调）。<br>另：输出中检出源码文本 ⇒ 须重做 |
| **Artifacts** | `evidence/review/c40-quarantine-manifest.json`（versioned，自哈希） |
| **Resource** | host 只读，0 GPU，**无 Docker** |
| **Authorization** | `AUTHORIZED`（属 `plan drafting/review` 范围：只读提取 + 生成纯哈希 artifact） |

> **两份 artifact 的版本化要求（冻结）**：
> `plan-review-*.json`（G0a）与 `c40-quarantine-manifest.json`（G0q）
> **都必须**版本化并自哈希。G1a 之前可暂存于文档仓库，
> G0b 之后由 builder 迁入 `benchmark/approx_kv/results/phase7_5_c40/evidence/review/`
> 并重新绑定 sha256。**只有**两者齐备且 plan review artifact 中
> `open P0/P1 = 0/0` 时，主会话才可把本计划状态升级为
> `Reviewed Candidate / PENDING USER AUTHORIZATION`。
> 实际`patch_id_alerts`/AST match只在G1b拿new-branch diff比对后产生；
> 所有match的人工裁决是G1b Exit条件，不得前移到G0q/G1a。

### P7.5-G1a — Branch Bootstrap（**只需 branch 授权**）

| 项 | 内容 |
| --- | --- |
| **Entry** | G0a Exit且G0q artifact `AVAILABLE`（无需branch-match adjudication）且`branch_creation_authorized=true`；不要求manifest/implementation/Docker |
| **Actions** | 执行 §5.2 的 branch/worktree 创建命令；逐条记录命令与输出；把 `branch-creation.json` **写到新 worktree 之外**（§5.2b），使新 worktree 保持 **clean 且 HEAD == base**；**不**在此步建立仓库内结果目录（那会弄脏 worktree） |
| **Exit** | 新 worktree 存在；`git rev-parse HEAD` == `0206f17b4255e4b248dafaaeb943be57428dae2f`；`HEAD^{tree}` == `3873d5683f98410524479c57c2068c6e1df98f7d`；`git branch --show-current` == `research/phase7.5-c40-cleanroom`；**`git status --porcelain` 输出为空（worktree clean）**；`cross-store-substrate` worktree 的 HEAD 未变；**未 push**；bootstrap evidence 已落在 worktree 之外的路径 |
| **Stop** | base commit/tree 不匹配；底座 worktree 非 clean；用户未授权 |
| **Artifacts** | 新 branch + worktree；`<bootstrap_dir>/branch-creation.json`（**worktree 之外**，手写内容；G0b 后由 builder 迁入仓库并重新绑定 sha256） |
| **Resource** | host git，0 GPU，**无 Docker**、**无测试执行** |
| **Authorization** | `PENDING USER AUTHORIZATION`（只需 `branch_creation_authorized`） |

---

### P7.5-G0b — Manifest / Bootstrap Builder（branch 存在后）

| 项 | 内容 |
| --- | --- |
| **Entry** | G1a Exit **且** `implementation_authorized = true` **且** `docker_test_execution_authorized = true` **且** `"P7.5-G0b" ∈ authorized_gates`（首个 authorized_gates 条目由用户在授权时直接指定，不依赖任何已存在的 manifest） |
| **Actions** | 在新 branch 上实现 WP0b：`build_p75_manifest.py`（plan manifest 构建与 `--check`）、`build_p75_reason_inventory`（§2.6 的 AST 扫描器）、`build_p75_result_manifest.py`；生成 `p75-plan-manifest.json` rev1（`status = pinned_blocked`）与 `evidence/reason-inventory.json`；把 G1a 的手写 `branch-creation.json` 重新哈希并绑定进 manifest；把文档仓库已版本化的 plan-review 与 quarantine manifest **逐字节导入** branch 的 evidence/review，断言导入前后 sha256 相同 |
| **Exit** | `p75-plan-manifest.json` rev1 存在且 `--check` 通过；`design_sha256` / `self_sha256` 可复算；`evidence/reason-inventory.json` sha256 已冻结；manifest 的 `blockers` 显式列出全部未授权项；`status == pinned_blocked`（**不是** `authorized`） |
| **Stop** | reason inventory 扫描出现无法静态求值的 reason 表达式；manifest 自哈希不可复算 |
| **Artifacts** | `p75-plan-manifest.json` rev1；`evidence/reason-inventory.json` |
| **Resource** | Docker CPU，0 GPU |
| **Authorization** | `PENDING USER AUTHORIZATION` |

> **授权门的自举规则（冻结）**：`P7.5-G0b` 是**唯一**允许在没有 authorized
> manifest 的情况下执行的代码 Gate。其授权凭据是**用户的显式授权声明本身**，
> 由执行者在 `evidence/g0b-authorization.json` 中记录（含用户原话引用与时间戳）。
> 从 `P7.5-G1b` 起，所有 Gate 都必须通过 manifest 的
> `status == authorized` 且 `"<gate>" ∈ authorized_gates` 校验。

---

### P7.5-G1b — Provenance / Dependency / Clean-room CPU

| 项 | 内容 |
| --- | --- |
| **Entry** | G0b Exit **且** manifest `status == authorized` 且 `"P7.5-G1b" ∈ authorized_gates` **且** `implementation_authorized = true` **且** `docker_test_execution_authorized = true` |
| **Actions** | WP1b（clean-room 合规测试 CR-1..CR-9，消费隔离 reviewer 产出的 quarantine manifest，§4.6）、WP2（ToolEvent provenance：wrapper 声明 + event-level collector 为唯一 shell authority + Merkle/git status 仅 secondary）、WP8（Docker 依赖锁与 central log 合同） |
| **Exit** | CR-1..CR-9全绿；new-branch patch-id/AST match全部在`quarantine-consumed.json`人工裁决；CR-7/CR-4/provenance/FN/pip checks通过；evidence自哈希。结论只限声明scope/method |
| **Stop** | CR-1..CR-9 任一检出禁止血缘（SR-9）；全部 event-level collector 实现在固定 image 内均不可用（此时 shell 类 group 一律 ineligible，须重新评估 workload 可行性）；依赖锁无法成立 |
| **Artifacts** | `evidence/{cleanroom-compliance,quarantine-consumed,docker-deps,provenance-collector}.json` |
| **Resource** | Docker CPU，0 GPU |
| **Authorization** | `PENDING USER AUTHORIZATION` |

---

### P7.5-G2 — Selector / Identity / Property（Docker CPU）

| 项 | 内容 |
| --- | --- |
| **Entry** | G1b Exit **且** manifest `status == authorized` 且 `"P7.5-G2" ∈ authorized_gates` **且** `implementation_authorized = true` **且** `docker_test_execution_authorized = true` |
| **Actions** | WP3、WP4；T1（对抗矩阵 + 混合 group + property P1–P8（含 P3b）+ 差分）、T2（identity/fingerprint）。<br>**T3（dense/copy 覆盖）在此只做 selector 输出层的覆盖断言**（候选岛与 dense 区间的集合覆盖、无重叠无空洞）；涉及 `KVReusePlan` 构造与底座 `require_full_coverage` 的完整 T3 属 **WP6/G3**，不在本 Gate |
| **Exit** | 对抗矩阵在冻结语料上全部通过；冻结语料上`collector_observed_FN=0`（含`oracle_agreement`报告）；property `P1–P8`（含`P3b`）各1000例无反例；跨fingerprint复用被拒；在已扫描代码路径中未发现fingerprint bypass；每次拒绝有唯一selector reason；selector层覆盖断言通过 |
| **Stop** | 任一observed FN > 0（SR-1）；发现fingerprint bypass（SR-5） |
| **Artifacts** | `evidence/cpu-tests.json`（含各测试计数与 sha256）；`frozen/` 目录的首批冻结清单 |
| **Resource** | Docker CPU，0 GPU |
| **Authorization** | `PENDING USER AUTHORIZATION` |

### P7.5-G3 — Middle-Span Runtime / Lifecycle（Docker CPU）

| 项 | 内容 |
| --- | --- |
| **Entry** | G2 Exit **且** manifest `status == authorized` 且 `"P7.5-G3" ∈ authorized_gates` **且** `implementation_authorized = true` **且** `docker_test_execution_authorized = true` |
| **Actions** | WP5、WP6、WP7；**T3 完整版**（`KVReusePlan` 覆盖 + `require_full_coverage` + §7.3.1 的 A1–A4）、T7（exclusive fallback）、T8（soak）、T9（双向压力，CPU 可模拟部分）、T20/T21（manifest 与执行合同） |
| **Exit** | 状态转移矩阵与`TC-1..TC-99`全绿；所有lifecycle/coverage/provenance/write/telemetry/Radix-arm合同闭合；资源、四族、10k soak、T-GATE、RESULT_MANIFEST通过 |
| **Stop** | accounting/orphan/leak（SR-3）；approx 污染 exact Radix（SR-4） |
| **Artifacts** | `evidence/cpu-tests.json` 更新；`RESULT_MANIFEST.json` 首版 |
| **Resource** | Docker CPU，0 GPU |
| **Authorization** | `PENDING USER AUTHORIZATION` |

### P7.5-G4 — Docker GPU Same-Context（Track B）

| 项 | 内容 |
| --- | --- |
| **Entry** | G3 Exit **且** manifest `status == authorized` 且 `"P7.5-G4" ∈ authorized_gates` **且** `docker_test_execution_authorized = true` **且** `gpu_execution_authorized = true` **且** `budget_authorized = true`（Track B） |
| **Setting** | `Qwen/Qwen3-0.6B@c1899de2...`、image `sha256:0be6e16e...`、`tp=pp=1`、`page_size=1`、`chunked_prefill=4096`、mixed-chunk/spec-decode off、`SGLANG_APPROX_KV_CORE=1`、`CROSS_STORE=1`、`HOST=0`、`PREFETCH=0`、`EPIC=0`、`C40=1` |
| **Actions** | Stage B-1；**先** T5 的 Control-1（Dense-vs-Dense）与 Control-2（E0-vs-Dense）稳定性基线并冻结 envelope，**再** T4（K/V+RoPE 张量）、T5 主体与 corruption canary |
| **Exit** | **先完成 Control-1/Control-2 并冻结 `baseline_envelope`**（§17.5.1）；随后 `max\|ΔK\|` / `max\|ΔV\|` / `max\|Δlogit\|` 均未超出 `max(dtype/frozen tol, baseline_envelope)`；贪心输出一致性按 §17.5.2 的条件门判定；`rotated_k_tokens == copied_k_tokens == span_len` 逐层成立；注入 `±1 rope_delta` 被检出（若不可检出则判配置不具备检出能力，须收紧） |
| **Stop** | 任一量**超出 baseline envelope** ⇒ 停止 GPU lane，回到 CPU lane（SR-2）。落在 envelope 内的差异不构成 stop |
| **Artifacts** | `raw/`、`logs/`、`central/` 的 B-1 记录；compact |
| **Resource** | `<= 2 starts`（含在 Track B cap 内） |
| **Authorization** | `PENDING USER AUTHORIZATION` |

### P7.5-G5 — Cross-Context Pilot（Track B）

| 项 | 内容 |
| --- | --- |
| **Entry** | G4 Exit **且** manifest `status == authorized` 且 `"P7.5-G5" ∈ authorized_gates` **且** `docker_test_execution_authorized = true` **且** `gpu_execution_authorized = true` **且** `budget_authorized = true`（Track B） |
| **Actions** | Stage C-0（**1-start calibration，先做**，不计入 pilot restart 数）→ Stage C-1 pilot（**`>= 3` 独立 restart**）；**workload 必须为 W4a live corpus**（§18.4）；四臂完整执行；T6、T11、T12、T13 |
| **Exit** | `>= 3` 独立 restart 完成，且**全部使用 W4a live corpus**（W1 只用于 C-0 calibration，不进入 `s_pilot`）；四臂完整 trace 齐备；每 restart 产出一个 `theta_j`；`s_pilot`（log-ratio 尺度）已估计；按 §19.2a 计算并冻结 `n_confirmatory` 与判定阈值写入 manifest `statistics_freeze`；`r` / `w` / `C_selector` / `E_cond` / `E_work_pooled`(descriptive) 全部报告；CI 由 restart-cluster bootstrap 给出（决策用 90% two-sided，descriptive 另报 95%）；`overhead_selector_control` 与 `selector_only_overhead` 已量化。**pilot 不得给出终局四态判定** |
| **Stop** | `w` 或 `r` 过低（SR-6）；性能明显为负且 pilot 方差不支持继续（SR-7，只停 speed lane）；预算 cap 触及（SR-10） |
| **Artifacts** | pilot raw/log/central/compact；`frozen/` 冻结清单；`statistics_freeze`（`s_pilot` / 方法 / `n_confirmatory` / 计算过程，sha256 绑定） |
| **Resource** | `1`（calibration）`+ >= 3`（pilot）`= >= 4 starts`；与 G4 的 `<= 2` 合计 `>= 6`，**必须** `<= 8 starts / <= 2 GPUh`（Track B cap，§22.2） |
| **Authorization** | `PENDING USER AUTHORIZATION` |

### P7.5-G6 — Confirmatory Primary `C40-1R0`（Track C）

| 项 | 内容 |
| --- | --- |
| **Entry** | G5 Exit **且** 基于 pilot 结果的**二次授权**：manifest `status == authorized` 且 `"P7.5-G6" ∈ authorized_gates` **且** `docker_test_execution_authorized = true` **且** `gpu_execution_authorized = true` **且** `budget_authorized = true`（Track C） |
| **Actions** | 执行 **Stage D-3 完整因子**（`n_confirmatory` restarts × 3 chunk-packed starts），**并在 chunk4096 的 start 内内联执行 primary controls 臂 C / 臂 D**（§15.4b.2）；D-1 primary = D-3 chunk4096 子集 + primary controls；D-2 sensitivity 为纯投影 |
| **Exit** | **D-3 完整因子已完成 且 primary controls（臂 C / 臂 D）已完成**（§15.4b.2；D-2 为纯投影）。任一未完成 ⇒ 只能记 `PARTIAL_EXIT`，mechanism disposition 只能取 `PARTIAL`/`INCONCLUSIVE`（§15.4b.4），`known_gap` 不替代完成；四本账 + 实测 `speedup_{1,2,4,8}` + `break_even_N` 齐备；`r`/`w`/`C_selector`/`E_cond`/`mu_theta`/`E_work_pooled`(descriptive) 全报告；CI 为 restart-cluster bootstrap 的 90% two-sided（决策用）+ 95%（descriptive）；臂 C 与臂 D 已执行；chunk 因子显式分离；headline 仅允许来自 primary chunk 且需 `mu_theta` 判定为 `POSITIVE`（`L90 > delta0`，§19.2b）；四态判定与 `n` / `s_pilot` 同时披露 |
| **Stop** | 同 G5；另加：出现 same-context mismatch（SR-2）或 approx 污染（SR-4）立即全停 |
| **Artifacts** | confirmatory raw/log/central/compact/summary |
| **Resource** | `n_confirmatory × 3` starts（`n=4` ⇒ `12`），建议 cap `12 + contingency 4 = 16`；GPUh 须由 1-start calibration 冻结 |
| **Authorization** | `NOT REQUESTED`（须 pilot 后重新申请 `docker_test` + `gpu` + `budget`） |

### P7.5-G7 — Conditional Extensions（多岛 / repair / host，Track D）

| 项 | 内容 |
| --- | --- |
| **Entry** | G6 Exit **且** 对应 lane 已通过 CPU gate **且** 单独授权：manifest `status == authorized` 且 `"P7.5-G7" ∈ authorized_gates` **且** `docker_test_execution_authorized = true` **且** `gpu_execution_authorized = true` **且** `budget_authorized = true`（Track D） |
| **Actions** | Stage E-1（`C40-mR0`）、E-2（`C40-1R1k`，先 `k=8` 单点）、E-3（`H1` host demand-load）、**E-4（CL-I exact-overlap clip）**；T15、T16、T17、**T23** |
| **Exit** | **每 lane 关闭时与 primary 逐字段一致（disabled parity）**：`C40_MAX_ISLANDS=1`、`C40_REPAIR_K=0`、`C40_HOST_DEMAND=0`、**`C40_EXACT_OVERLAP_CLIP=0`** 时，outcome / terminal reason / telemetry / 时序均与 primary 逐字段相同；多岛 non-overlap 与 budget property 通过；`k=0` 退化为 `C40-1R0`；host 失败 fail-closed 且归因不重叠；**CL-I 开启时 B-2 请求 outcome 带 `geometry=clipped_at_exact_boundary` 且未进入 primary headline** |
| **Stop** | 任一 lane 破坏 core 不变量 ⇒ 关闭该 lane，**不阻塞** core 结论 |
| **Artifacts** | 每 lane 独立 compact + summary 分节 |
| **Resource** | 待估计并单独授权 |
| **Authorization** | `NOT REQUESTED` |

### P7.5-G8 — Workflow Scheduler（`S0` / `S4`，Track D）

| 项 | 内容 |
| --- | --- |
| **Entry** | G6 Exit **且** 单独授权：manifest `status == authorized` 且 `"P7.5-G8" ∈ authorized_gates` **且** `docker_test_execution_authorized = true` **且** `gpu_execution_authorized = true` **且** `budget_authorized = true`（Track D） |
| **Actions** | Stage F-1；W3 workload；**S0/S4 相邻交替启动**；报告 all-reusable 与 workflow-only 两个口径与 matched coverage |
| **Exit** | 两臂 matched coverage 一致；配对为相邻 launch block；若 C40 臂 dense fallback 率 > 40%，结论只能写 `DESCRIPTIVE` |
| **Stop** | matched coverage 无法对齐 ⇒ 只报 `DESCRIPTIVE`，不做机制归因 |
| **Artifacts** | scheduler 矩阵 raw/compact/summary |
| **Resource** | 待估计并单独授权 |
| **Authorization** | `NOT REQUESTED` |

### P7.5-G9 — RepoBench-P / SWE-bench 质量（Track E）

| 项 | 内容 |
| --- | --- |
| **Entry** | **硬前置（缺一不可）**：<br>1. **G3 Exit**（C40 已真正实现并通过 CPU 正确性面 —— 没有可用的 C40 实现就无所谓"C40 的质量"）；<br>2. **G4 Exit**（same-context canary 通过 —— 未证明数据面正确前，任何质量差异都无法归因）；<br>3. manifest `status == authorized` 且 `"P7.5-G9" ∈ authorized_gates`；<br>4. `implementation_authorized = true`、`docker_test_execution_authorized = true`、`gpu_execution_authorized = true`、`budget_authorized = true`（Track E，独立预算）、`quality_campaign_authorized = true`。<br>**明确不依赖 G5/G6**：`SR-6` / `SR-7` 停 speed lane 时，只要 G3+G4 已通过，quality lane 仍可独立授权执行。**但 speed lane 停止不等于可以跳过 G3/G4** |
| **Actions** | Stage G-1（RepoBench-P `>= 1000` 例）、**G-2a（W6a calibration：40 tasks × 3 配对 seed，只估 `p_d`/方差/coverage）**、**G-2b（W6b confirmatory：与 W6a 完全 disjoint 的 `n_confirm` tasks，需二次授权）**；三者均按显式 source-producing → consume 两阶段协议执行并报告覆盖率；§18.6b 覆盖率门；§18.6.3 + §19.8 统计 |
| **Exit** | 先判 §18.6b 的**预冻结**阈值（`tau_task = max(10, ceil(0.25*n_tasks))`、`tau_coverage = 0.10`、`tau_copied_tokens_per_effective_task = 128`）：任一不满足 ⇒ 判 `NO_COVERAGE` 并停止（**不得**称质量无损）；否则按 §18.6.3 在 **W6b** 上执行非劣性检验（margin = 5 pp，pre-data 冻结），给出 `NON_INFERIOR(5 pp)` / `DAMAGING` / `INCONCLUSIVE(<= X pp)` 之一并附 X 的推导；若无法划出足够的 disjoint task ⇒ `INCONCLUSIVE_INSUFFICIENT_DISJOINT_TASKS`。**禁止**事后放宽 margin |
| **Stop** | Dense 自身 run-to-run 翻转率 > 15% ⇒ 先修 harness（SR-8）；外部基准不可获得（SR-9b） |
| **Artifacts** | 质量 raw（task-run 级）、compact、summary；discordant 计数表 |
| **Resource** | **task-runs 单列计量**，`n_tasks × n_repeats × n_arms`；其 server starts / GPUh 独立记录在 Track E 账下，**不得**混入 Track A–D 的 starts cap |
| **Authorization** | `NOT REQUESTED` |

### P7.5-G10 — Prefetch Composition（正交，Track F）

| 项 | 内容 |
| --- | --- |
| **Entry** | G8 Exit **且** prefetch 能力可在受控环境中获得 **且** 单独授权：manifest `status == authorized` 且 `"P7.5-G10" ∈ authorized_gates` **且** `docker_test_execution_authorized = true` **且** `gpu_execution_authorized = true` **且** `budget_authorized = true`（Track F） |
| **Actions** | Stage H-1 四模式；T18（含 late / cancel / stale prefetch） |
| **Exit** | Combined 的 selected span 与 Coding-only **逐 token 相同**；关闭 prefetch 精确恢复 Coding-only；无 lease/worker/CUDA event 泄漏 |
| **Stop** | span 不一致 ⇒ 停止合并，prefetch 仅作正交后续 |
| **Artifacts** | 四模式 compact + span 逐 token 比对证据 |
| **Resource** | 待估计；若外部依赖不可得，保持 `BLOCKED_EXTERNAL`，**不占用**任何预算 |
| **Authorization** | `NOT REQUESTED` / 可能 `BLOCKED_EXTERNAL` |

### P7.5-G11 — Consolidation / Dual Review / Disposition

| 项 | 内容 |
| --- | --- |
| **Entry** | 至少 G3 Exit（correctness lane）；speed/quality lane 视授权而定；**且** manifest `status == authorized` 且 `"P7.5-G11" ∈ authorized_gates` **且** `implementation_authorized = true` **且** `docker_test_execution_authorized = true`（consolidation 在 Docker 内执行并写入结果目录与 manifest，属受控写操作） |
| **Actions** | 离线 consolidator 产出自哈希 compact/summary；构建 `RESULT_MANIFEST` 并递归 `--check`；双模型 review（执笔 / 审阅分离）；生成 `C40_DISPOSITION.json` |
| **Exit** | `RESULT_MANIFEST` 递归校验通过、`known_gaps` 明确；open P0/P1 = 0；五类 disposition 分别给出（engineering / mechanism / system / quality / publication） |
| **Stop** | review 出现无法闭合的 P0 ⇒ 不发布 |
| **Artifacts** | `summary/`、`RESULT_MANIFEST.json`、`C40_DISPOSITION.json`、`evidence/review/` |
| **Resource** | 0 GPU |
| **Authorization** | `PENDING USER AUTHORIZATION` |

### 20.1 Gate 依赖与 lane 分离

```text
bootstrap 链（严格顺序，无循环）:
  G0a  纯文档 / plan review     [AUTHORIZED]       无 Docker、无 branch、无 manifest
   └─ G0q  quarantine 提取      [AUTHORIZED]       只读、无 Docker、无 branch
        │   Exit = quarantine reference artifact AVAILABLE
        │   BLOCKED_QUARANTINE_INPUT ⇒ **链在此中断，不得进入 G1a**
        └─ G1a  branch bootstrap [branch auth]     无 Docker、无 manifest
            └─ G0b  manifest builder [impl+docker auth] 产出 pinned_blocked manifest
                └─ G1b  provenance / cleanroom CPU  [manifest authorized]
                    └─ G2 → G3 → G4 → ...

correctness lane :  G0a → G0q → G1a → G0b → G1b → G2 → G3 → G4 → G11
speed lane       :  (G4) → G5 → G6 → G7/G8 → G11
quality lane     :  (G3 + G4) → G9 → G11      ← 不经过 G5/G6，但**必须**有 G3+G4
composition lane :  (G8) → G10 → G11

speed lane 停止**不**停止 correctness lane 与 quality lane：
  SR-6 / SR-7 触发时跳过 G5→G6 的扩张，但 G9 仍可在独立授权下执行
  （前提是 G3 与 G4 均已 Exit：必须存在已实现且 same-context 已验证的 C40）。
quality lane 停止**不**停止 correctness lane。
correctness lane（G0a–G4）停止 ⇒ 全部停止。
G0q BLOCKED ⇒ G1a 起全部不得启动。

当前授权：G0a / G0q = AUTHORIZED（plan review 范围）；G1a 起全部 PENDING。
授权门自举：G0b 是唯一凭"用户显式授权声明"执行的代码 Gate；
            G1b 起一律凭 manifest 的 status/authorized_gates。
```

---

## 21. Stop Rules

| ID | 触发条件 | 影响范围 | 动作 |
| --- | --- | --- | --- |
| **SR-1** | selector 差分出现 **FN > 0**，或任一 property 反例 | **全部 lane** | 立即停止；回到 WP2/WP3 修复；修复后重跑 T1 全量 |
| **SR-2** | same-context canary **超出冻结的 `baseline_envelope`**（§17.5.1），或在 baseline 稳定的前提下输出不一致 | **全部 lane** | 立即停止 GPU；判定为数据面实现缺陷；回到 WP5/WP6。**落在 envelope 内的差异不触发本规则** |
| **SR-3** | accounting 不平 / orphan > 0 / lease 或 slot 泄漏 | **全部 lane** | 立即停止；修复后重跑 T7/T8 |
| **SR-4** | approx KV 污染 exact Radix，或 `approx_depth` 规则被绕过 | **全部 lane** | 立即停止；这是正确性 P0 |
| **SR-5** | 在扫描或动态注入中**检出** fingerprint bypass 路径（代码、env 或测试钩子） | **全部 lane** | 立即停止；移除 bypass 并扩大静态扫描范围 |
| **SR-6** | eligibility 过低：`r < 5%` **或** time-weighted coverage `w < 5%` | **speed lane 停止**；correctness / quality lane 继续 | 记录为覆盖率结论；不再申请 confirmatory speed 预算；`w` 必须写入 disposition |
| **SR-7** | 性能先验未被推翻：pilot（`>= 3` restart）的 `mu_theta` `U90 < delta0`，或按 §19.2a Step 5 计算出的 `starts_needed` 超出已授权 cap | **speed lane 停止**；correctness / quality lane 继续 | pilot **只能**记 `PILOT_BELOW_MDE` 或 `PILOT_UNDERPOWERED`（pilot 阶段不具备给出 `POSITIVE`/`NEGATIVE`/`SMALL_POSITIVE_BELOW_MDE` 终局判定的资格，§19.2）；跳过 Stage D 扩张；**不得**发布任何 speedup headline；**不得**据此判 `NEGATIVE`。若日后要正式判定，必须重新申请并执行 confirmatory |
| **SR-8** | 质量受损：C40 相对 Dense 的 `majority_resolved` 显著下降；或 Dense 自身 run-to-run 翻转率 > 15%；或 `effective_paired_tasks` / `w_quality` / `mean_copied_tokens_per_effective_task` 低于 §18.6b 的**预冻结**阈值 | **quality lane 停止**（第二种先修 harness） | 先修 harness；若确为 C40 损伤，记 `DAMAGING` 并停止扩样；若为覆盖不足，记 `NO_COVERAGE`（§18.6b），**不得**表述为质量无损 |
| **SR-9** | clean-room 违规：**CR-1..CR-9** 中任一 *自动 fail* 类检查命中（禁止 import / 精确 forbidden blob / cherry-pick 确认） | **全部 lane** | 立即停止；回滚相关变更；重新执行 **CR-1..CR-9** 全套。<br>**注**：patch-id 与结构相似度命中**不自动 fail**，只触发人工复核（§4.6.3） |
| **SR-9b** | Gate probe 判定外部资源不可获得（W4b 真实外部 trajectory / RepoBench-P / SWE-bench harness / prefetch 能力） | **对应轴标 `BLOCKED_EXTERNAL`** | 不占预算；不阻塞其他 lane。**注**：W4a 是 live corpus 且默认可执行，因此 speed lane **不**因外部资源缺席而阻塞；W4b 缺席只降低外部有效性，须在 disposition 中保留该限制 |
| **SR-10** | 预算 cap 触及（Track B `<= 8 starts` 或 `<= 2 GPUh`） | **该 Track 停止** | 停止并重新申请；**不得**自行追加 |
| **SR-11** | 无授权执行（任何 Gate 缺 pinned manifest 或 `status != authorized`） | **该结果不进入 disposition** | 结果作废；重新走"预注册 → pin → 授权 → 执行" |

### 21.1 Lane 影响速查

```text
停止 correctness/全部 lane :  SR-1  SR-2  SR-3  SR-4  SR-5  SR-9
只停 speed lane            :  SR-6  SR-7
只停 quality lane          :  SR-8
只停对应 Track / lane      :  SR-9b  SR-10
结果作废但不停 lane        :  SR-11
```

**关键判断**：`SR-6` 与 `SR-7` 是**最可能触发**的两条（Phase7 已给出不利先验）。
它们**不**否定 Phase7.5 的价值——正确性、覆盖率与质量损伤上界本身就是可发表的结论。

---

## 22. 预算

> **全部为 `estimate`，全部 `PENDING USER AUTHORIZATION`。**
> 在完成 1-start calibration 之前，任何总量上界都缺乏依据。

### 22.1 Track 划分

| Track | 内容 | 人力（`estimate`） | GPU starts | GPUh | task-runs | 授权状态 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| **A** | Zero-GPU clean-room + core（WP0a / **WP0q** / WP1a / WP0b / WP1b / WP2–WP9） | `8.00–11.75 人周` | **0** | **0** | 0 | `PENDING`（**推荐先做**；其中 WP0a/WP0q 已在文档授权内） |
| **B** | 小规模 GPU pilot（G4 + G5 = WP10） | `0.25–0.5 人周` | **hard cap `<= 8 starts`**（下限配置 `2 + 1 + 3 = 6`，余量 `2`，§15.4 Stage C） | **建议 hard cap `<= 2 GPUh`** | 0 | `PENDING`（须单独申请） |
| **C** | Confirmatory core（G6：D-3 + 内联 primary controls） | 待定 | `n_confirmatory × 3`（`n=4` ⇒ `12`），建议 cap `12 + 4 = 16` | 待定（须 1-start calibration 后冻结） | 0 | `NOT REQUESTED`（须 pilot 后二次授权） |
| **D** | Extensions（WP11 九个 conditional lane + G7 + G8） | `4–8 人周` | 待定 | 待定 | 0 | `NOT REQUESTED` |
| **E** | Quality task-runs（G9：G-1 + G-2a calibration + G-2b confirmatory） | `1.5–2 人周`（工程部分） | 独立记录（**不并入 A–D 的 starts 预算**） | 独立记录 | G-2a：`40 × 3 × 2`；G-2b：`n_confirm × 3 × 2`（`n_confirm = ceil(2473.0 × p_d)`，须二次授权） | `NOT REQUESTED`（独立预算） |
| **F** | Prefetch external / composition（G10） | `0.5 人周` | 待定 | 待定 | 0 | `NOT REQUESTED` / 可能 `BLOCKED_EXTERNAL`（**不占当前预算**） |

### 22.2 Track B 的 cap 使用规则（硬性）

```text
1. 先执行 **1 个 start 的校准**（Stage C-0 single-start calibration），
   实测该配置下每 start 的 wall-clock 与 GPU-equivalent 小时；
2. **以校准结果冻结**剩余 starts 的分配；
   上表的 `<= 8 starts / <= 2 GPUh` 只是申请时的初始建议上界，
   **不是**已核实的容量；
3. 一旦达到 cap，**停止并重新申请**，不得自行追加；
4. Track B 的任何 start 都不得在 Track A Exit 前执行。
```

**参考量级**：Phase7 实际使用 `22 starts / 1.310142 GPUh`（硬上限
`36 starts / 6 GPUh`，实际占 `21.8%`）；correction 额外 `1 start / 0.098332 GPUh`。

> **人力口径说明（避免重复计数）**：Track A = WP0a/WP0q/WP1a/WP0b/WP1b/WP2–WP9 = `8.00–11.75 人周`；
> Track B = WP10 = `0.25–0.5 人周`；两者相加 = **`8.25–12.25 人周`**，与 §16.2
> 的 WP0a–WP10 合计一致。§22.3 的"core parity + mandatory extensions
> `8.25–12.25 人周`"指的正是 **Track A + Track B** 的合计，不是 Track A 单独的值。

### 22.3 人周汇总（`estimate`）

| 分组 | 建议区间 | 说明 |
| --- | --- | --- |
| **core parity + mandatory extensions** | **`8.25–12.25 人周`** | WP0a–WP10 分项求和（§16.2 表）；含 middle-span controller（`2–2.5`）、provenance（`1.5–2`）、selector（`1–1.5`）、adapter/telemetry/manifest/runner 与两轮独立 review |
| **conditional extensions** | **额外 `4–8 人周`** | WP11 **九个** lane（CL-A..CL-I）分项求和；可按 lane 拆分交付，任一 lane 延期不阻塞 core |
| **quality campaign** | **另算** | WP12 工程部分 `1.5–2 人周` + task-runs 机时 + Track E 的 starts/GPUh（三者单独计量、单独授权） |

### 22.4 计量规则（硬性）

```text
禁止：把 task-runs 混进 server starts 预算（两者是不同计量单位）
禁止：把 quality campaign 的 starts / GPUh 计入 Track B/C 的 cap
要求：Track E 同时记录 (task-runs, server starts, GPUh) 三个量，但只在 Track E 账下汇总
禁止：在未完成 1-start calibration 前给出"已核实"的总量上界
要求：后续任何扩样（restart 数、SWE-bench n、conditional lane 扩展）
      都需要**二次授权**，不由前一 Gate 结果自动触发
```

### 22.5 预算申请模板

```text
Track       : <A|B|C|D|E|F>
Gate        : P7.5-G<n>
Cells       : <staged cells>
Starts      : <n>            （或 task-runs: <n_tasks × n_repeats × n_arms>）
GPUh cap    : <h>
Calibration : <是否已完成 1-start calibration，实测每 start GPUh = ___>
Manifest    : revision <n>, status=authorized, design_sha256=<...>
Stop rules  : <适用的 SR-x 列表>
Justification: <为何该 Gate 现在值得投入；预期产出的具体结论类型>
```

---

## 23. Governance

### 23.1 Plan review 制度

```text
执笔 / 审阅分离：
  - 执笔模型与审阅模型必须不同（建议 Opus 级执笔、Sol 级审阅，或互换）
  - 至少两轮：initial review → closure review
  - findings 分级 P0 / P1 / P2；**最终 open P0/P1 必须 = 0**
  - P2 可保留，但必须登记进 PROJECT.md 并在 disposition 中披露

计划 design 变更 ⇒ 归档旧 revision、创建新 revision、更新 design hash、
                   重新 review
授权状态变更（不改 design）⇒ 只递增 revision，design hash 不变
```

### 23.2 Pin 与 evidence 链

| 项 | 要求 |
| --- | --- |
| branch / code pin | 每个 Gate 执行前 pin `commit + tree`，写入 manifest 并在 raw 中回绑 |
| Docker CPU evidence | 每个 CPU Gate 产出 `evidence/cpu-tests.json`，含测试计数、runner sha256、image digest |
| GPU manifest | 每个 GPU Gate 前 manifest 转 `authorized` 并列出 `authorized_gates` |
| phase central JSONL | append-only；文件级 sha256 进 manifest |
| 全局 run log | 每 setting 追加 run-level 事件到 `/home/chris/Workspaces/kvcache-research/results/BENCHMARK_RUN_LOG.jsonl`；绑定 line range + range sha256 + snapshot sha256；**只追加，绝不改写他人记录** |
| raw / log | **不可变**；任何 correction 只能**新增** supplementary run，不改写原文件 |
| compact / summary | 自哈希；默认拒绝覆盖，`--force` 显式允许 |
| `RESULT_MANIFEST` | 递归自哈希，`--check` 只读重放；`known_gaps` 显式列出 |

### 23.3 Runtime staging 不污染 worktree

```text
runtime 写目标恰好两处（§11.3）：容器内 /results/phase7_5_c40/**（普通 artifact）
与 /global_results/BENCHMARK_RUN_LOG.jsonl（仅 append run-level 事件）
GPU wave 之间**不**提交 raw/compact/log
全部 wave 结束后**一次性**版本化到 benchmark/approx_kv/results/phase7_5_c40/
Docker 挂载：被测仓库 :ro；可写只用 --tmpfs 或显式 artifact 卷
绝不向宿主 worktree 挂载可写卷
```

### 23.4 Correction governance

```text
若执行后发现口径缺陷（如 Phase7 的 `unsupported <- store_miss`）：
  1. 原 raw/log **不可变**；
  2. 只运行**最小必要**的 supplementary correction run；
  3. correction 需独立 manifest（revision + self hash）与独立 review；
  4. correction 后**全量重生成** compact 与 summary，并在同一 commit
     重建 RESULT_MANIFEST；
  5. correction 机时**单独计量**，不并入原 starts 统计；
  6. TRACKING.md 只追加不改写；纠正以新记录形式追加。
```

### 23.5 Final dispositions（五类分列，互不合并）

| 类别 | 判定对象 | 可取值 |
| --- | --- | --- |
| **engineering** | 实现是否工程有效（可达、无 capacity error、遥测完整、记账自洽） | `VALID` / `INVALID` |
| **mechanism** | `C40-1R0` 相对 Dense 的机制效应（primary = `mu_theta`，§19.2a） | `POSITIVE`（`L90 > delta0`）/ `SMALL_POSITIVE_BELOW_MDE`（`L90 > 0` 且 `U90 < delta0`）/ `NEGATIVE`（`U90 < 0`）/ `INCONCLUSIVE`（其余，含跨 0 或跨 `delta0`）/ `PARTIAL`（D-3 或 primary controls 未完成，§15.4b.4） |
| **system** | 系统行为（S0/S4、压力、fallback 率、覆盖率） | `DESCRIPTIVE` / `INCONCLUSIVE` / `CHARACTERIZED` |
| **quality** | 任务准确率非劣性（margin = 5 pp，pre-data 冻结） | `NO_COVERAGE`（§18.6b）/ `NON_INFERIOR(5 pp)` / `DAMAGING` / `INCONCLUSIVE(<= X pp)` / `INCONCLUSIVE_INSUFFICIENT_DISJOINT_TASKS` / `BLOCKED_EXTERNAL` |
| **publication** | 是否可发布及 caveat | `READY` / `READY WITH CAVEATS` / `NOT READY` |

**规则**：五类**分别**给出，**禁止**用其中一类的正面结论掩盖另一类的负面结论。

### 23.6 Phase8 不自动触发

```text
Phase7.5 的任何结论**不自动**触发 Phase8。
若要进入 Phase8，必须：
  1. Phase7.5 完成 G11 并给出五类 disposition；
  2. 用户明确授权创建 Phase8 计划；
  3. Phase8 走独立的 plan / manifest / 结果目录。
```

### 23.7 GitHub 协作约束

```text
- 项目交流与 prototype 实现仓库：https://github.com/ccdd2023/sglang
- 指定操作账号：ccdd2023
- 执行读取以外的 GitHub 操作前必须核实身份与权限；
  不得假设 CLI 当前默认账号即为 ccdd2023
- 使用账号级显式认证，避免为单次操作改变全局默认账号
- 不得输出、记录或提交 token、认证头或其他凭据
- Phase7.5 的 push / PR / tag 均需**单独授权**
```

---

## 24. 交付物与 Artifact 索引

### 24.1 文档交付物

| 交付物 | 路径 | 阶段 |
| --- | --- | --- |
| Phase7.5 执行计划（本文件） | `IMPLEMENTATION_PLAN_PHASE7_5_C40.md` | G0 |
| Plan review artifacts | `benchmark/approx_kv/results/phase7_5_c40/evidence/review/`（G0a 阶段可暂存于文档仓库，G1a 后迁入） | G0a / G11 |
| Phase7.5 阶段报告 | `research/phase_reports/PHASE7_5_C40_REPORT.md`（冻结默认 `FD-4`） | G11 |
| PROJECT / HANDOFF / TRACKING 更新 | 仓库根 | 每轮有效讨论后 |

### 24.2 代码交付物（`proposed`，见 §7.1）

```text
python/sglang/srt/mem_cache/approx_kv/coding_c40/
    __init__.py provenance.py types.py selector.py optimizer.py
    controller.py state.py stats.py adapter.py gates.py
benchmark/approx_kv/coding_c40/
    __init__.py trajectory.py plan_freeze.py workloads_c40.py
    consolidate_p75_results.py
benchmark/approx_kv/
    run_p75_selector_offline.py run_p75_canary.py run_p75_micro.py
    run_p75_workflow.py run_p75_quality.py
    build_p75_manifest.py build_p75_result_manifest.py
test/registered/unit/mem_cache/  test_c40_*.py
test/registered/unit/bench/      test_c40_manifest.py test_c40_plan_freeze.py
                                 test_c40_consolidator.py
Dockerfile.p75  requirements.lock
```

### 24.3 结果交付物

见 §14.2 的目录树。

### 24.4 Authority artifact 索引（只读引用）

```text
docs:PROJECT.md
docs:IMPLEMENTATION_PLAN_LATEST.md                       (Phase7 V7, byte-frozen)
docs:research/CODING_AWARE_V40_BRANCH_TECHNICAL_REPORT.md
docs:research/phase_reports/PHASE4_RECOVERY_METHODS_REPORT.md
docs:research/phase_reports/PHASE5_WORKFLOW_SCHEDULING_REPORT.md
docs:research/phase_reports/PHASE6_CROSS_STORE_SUBSTRATE_REPORT.md
docs:research/phase_reports/PHASE7_INTEGRATED_EVALUATION_REPORT.md
docs:research/phase_reports/PHASE4_TO_PHASE7_SUMMARY.md
docs:research/RESEARCH_SYNTHESIS.md
xs:python/sglang/srt/mem_cache/approx_kv/{types,runtime,transfer,radix_backend,store,manager,config}.py
xs:python/sglang/srt/mem_cache/cross_store/{coordinator,allocator,budget,object_graph,policy}.py
xs:benchmark/approx_kv/{build_result_manifest,consolidate_phase7_results}.py
```

---

## 25. 下一会话可直接执行的顺序

> **前置**：本节每一步都需要对应授权。当前**只有第 0–3 步**（阅读、只读 plan
> review、文档同步）在 `plan_drafting/review_authorized = true` 下已授权；
> 第 4 步起全部 `PENDING USER AUTHORIZATION`。
>
> **顺序不可调换**：`G0a → G0q → G1a → G0b → G1b` 是冻结的 bootstrap 链。
> `G0q` 若为 `BLOCKED_QUARANTINE_INPUT`，**不得**进入第 4 步。

```text
[已授权：plan drafting / review + 文档同步]（= WP0a / P7.5-G0a，无 Docker）
 0. 阅读 HANDOFF.md → PROJECT.md → TRACKING.md 最新记录 → 本计划
    确认 §0.4 的授权边界仍然有效
 1. 对本计划做 independent review（执笔/审阅分离，只读；不修改代码仓库、
    不建 branch、不跑测试）
 2. 生成 versioned review artifact：evidence/review/plan-review-<n>.json
    （自哈希 + 绑定本文件 sha256；记录 open P0/P1 计数）
 2b. **P7.5-G0q / WP0q（隔离 reviewer 执行，AUTHORIZED）**：产出 versioned
    evidence/review/c40-quarantine-manifest.json
    （只含 commit ids / exclusive blob hashes / patch-id alerts /
      file+function AST signatures / allowed_shared_signatures；无源码文本）
    —— 只有 **plan-review artifact 与 quarantine manifest 都已版本化**
       且 plan review 的 open P0/P1 = 0/0 时，
       主会话才可把状态升级为 `Reviewed Candidate / PENDING USER AUTHORIZATION`
 3. 更新 PROJECT.md（新增 Phase7.5 计划记录）
    追加 TRACKING.md（ISO 8601 时间戳）
    更新 HANDOFF.md 快照（阶段切换 + 下一步）

[需要 branch_creation_authorized = true]（= WP1a / P7.5-G1a，**无 Docker、无测试**）
 4. 执行 §5.2 的 branch/worktree 创建命令（逐条记录输出）
 5. 手写 branch-creation.json 到 **worktree 之外**（§5.2b）：
        /home/chris/Workspaces/kvcache-research/results/phase7_5_c40/bootstrap/
    —— 本步**不**在新 worktree 内创建任何文件（保持 clean 且 HEAD == base）
    —— 本步**不**生成 manifest（manifest 是代码产物，属第 6 步）
    —— 验证：git -C <new worktree> status --porcelain 输出为空

[需要 implementation_authorized = true 且 docker_test_execution_authorized = true]
（= WP0b / P7.5-G0b；授权凭据记入 evidence/g0b-authorization.json）
 6. WP0b build_p75_manifest / build_p75_result_manifest / reason_inventory
    → 建立仓库内结果目录树（§14.2）
    → 从 bootstrap 目录导入 branch-creation.json 并重新绑定 sha256
    → 导入 G0a/G0q 的两份 review artifact 并重新绑定 sha256
    → p75-plan-manifest.json rev1（status = pinned_blocked）
    → evidence/reason-inventory.json
    → 独立 review → 用户授权 → status = authorized，
      authorized_gates 至少含 "P7.5-G1b"

[需要 manifest authorized 且 implementation + docker_test 授权]
 7. WP1b clean-room compliance（CR-1..CR-9，消费第 2b 步的 quarantine 签名）
        —— auto-fail 与人工复核分级见 §4.6.3
        —— 结论表述限定为"本次扫描范围与方法下未检测到禁止血缘"（= P7.5-G1b）
 8. WP2  ToolEvent provenance（wrapper 声明 + event-level syscall trace
        + Merkle/git status 仅 secondary；对抗矩阵、property、differential）（= P7.5-G1b）
 9. WP8  Docker 依赖锁 + central log 合同（可与 WP2 并行）            （= P7.5-G1b）
10. WP3  G40 selector                                                （= P7.5-G2）
11. WP4  identity / fingerprint / approx depth                       （= P7.5-G2）
12. WP5  middle-span controller、状态机、所有权分离（borrowed/owned）、
         effective_prefix helper 与 allocation 成对替换、identity membership、
         forced-middle chunk 语义与底座 KV ledger，
         chunk-splitting 执行协议                                     （= P7.5-G3）
13. WP6  adapter 与 plan 构造                                        （= P7.5-G3）
14. WP7  terminal reason / telemetry / lease GC / feature gate        （= P7.5-G3）
15. WP9  workload、W4a live corpus、冻结 trajectory 与 runner
    → 生成 p75-plan-manifest rev2（pinned_blocked）
    → 独立 review → authorized rev3（authorized_gates 增列 GPU Gate）

[需要 docker_test_execution_authorized = true 且 gpu_execution_authorized = true
 且 budget_authorized = true（Track B）]
16. WP10 Stage B-1 same-context canary（<= 2 starts）                （= P7.5-G4）
17. WP10 Stage C-0 1-start calibration → 冻结剩余分配                （= P7.5-G5）
18. WP10 Stage C-1 pilot（**>= 3 独立 restart**，W4a live corpus，估 s_pilot）（= P7.5-G5）
    Track B 合计 starts = G4(<=2) + calibration(1) + pilot(>=3) <= 8
    → 按 §19.2a 冻结 confirmatory n 与判定阈值

[需要 quality_campaign_authorized = true + docker_test + gpu + budget（Track E）]
（独立 lane；硬前置 = 第 14 步 Exit(G3) 与第 16 步 Exit(G4)）
19. WP12 质量部分（G9）：G-1 RepoBench-P + G-2a W6a calibration(40 tasks)
    → 估 p_d → 计算 n_confirm → **二次授权** → G-2b W6b confirmatory
      （task 集合与 W6a 完全 disjoint；margin=5pp 已 pre-data 冻结）
    task-runs / starts / GPUh 均在 Track E 账下单独计量
    **不依赖** 17/18/20；SR-6/SR-7 停 speed lane 时本步仍可执行，
    但**不得**在缺少 G3/G4 的情况下执行

[需要二次授权：docker_test + gpu + budget]
20. Stage D confirmatory（G6）：D-3 完整因子（n × 3 chunk-packed starts）
    **并在 chunk4096 的 start 内内联 primary controls 臂 C / 臂 D**（§15.4b.2）
21. WP11 conditional lanes + Stage E/F（G7/G8）
22. Stage H prefetch composition（G10，若 Gate probe 判定可获得）

[需要 docker_test_execution_authorized = true，0 GPU]
23. G11 consolidation → RESULT_MANIFEST --check → 双模型 review
    → C40_DISPOSITION.json（五类分列）
24. 更新 PROJECT.md / TRACKING.md / HANDOFF.md
```

### 25.1 每步都必须遵守的操作约束

```text
- 所有实验（含 CPU 测试）在 Docker 内执行；host 只做**授权范围内的 git 操作**
  （只读复核；以及在 `branch_creation_authorized=true` 后的 worktree/branch 创建）、
  文档读取/编辑、G0a review artifact 与 G0q quarantine signature；
  **G0b 起的 manifest builder / `--check` 一律在 Docker 内执行**
- 被测仓库 :ro 挂载；可写只用 --tmpfs 或显式 artifact 卷
- runtime 写目标恰好两处：/results/phase7_5_c40/**（普通）与
  /global_results/BENCHMARK_RUN_LOG.jsonl（仅 append run-level）
- Phase4–7 结果目录只读
- 不 push、不建 PR、不改 origin，除非单独授权
- 每轮有效工作后更新 PROJECT.md 并追加 TRACKING.md（ISO 8601）
- 阶段切换 / 架构决策 / 重大阻塞 / 风险变化 / 下一步改变时更新 HANDOFF.md
```

---

## 26. 允许与禁止的表述

### 26.1 允许（在标注证据级别的前提下）

```text
✔ "C40 = G40 structured grounded selector × R0 Raw+RoPE executor"
✔ "恢复 primitive 沿用本项目已有的 R0，没有新的恢复公式"
✔ "新贡献在 structured provenance 驱动的 selection / invalidation 与系统组合"
✔ "Repository-event-grounded admission control for non-prefix KV reuse in coding agents"
✔ "在占 dense 基线时间 w 的 eligible 子集上观察到 E_cond = X；
    全 workload 的 mu_theta CI 跨 0，判定为 INCONCLUSIVE"
✔ "在该样本量与实测 discordant rate 下，无法排除 <= X pp 的质量损伤"
✔ "在本次扫描范围与检查方法下，未检测到禁止的 collaborator 源码血缘（必要不充分）"
✔ "在冻结语料上、由所声明的 event collector 观察到 collector_observed_FN = 0"
✔ "在已扫描代码路径中未发现 fingerprint bypass"
✔ "在 10k 请求 soak 范围内未观察到 lease/orphan 泄漏"
✔ "在 W4a live corpus 上 w = X、r = Y；外部有效性受 fixture 语料限制"
✔ "mu_theta CI 覆盖 0，判定为 INCONCLUSIVE（不是 NEGATIVE）"
✔ "Phase7 对同 image / 同模型 / chunk4096 下 R0 的判定为 NEGATIVE，
    这是 C40 的不利先验，需由 pilot 检验"
✔ "chunk1024 的数字标注 headline=false，仅作 sensitivity"
```

### 26.2 禁止

```text
✘ 把 C40 命名为新的 recovery primitive（R6 / L0）
✘ 把 C40 称为 CacheBlend-style selective repair
✘ 把 C40 的 dense prefix/suffix 称为 "repair"
✘ 把 C40 称为 KVCOMM reconstruction / prefetch / KVFlow
✘ 把 "可变编码" 写成 KVCOMM 原文术语
✘ 把 delta compression / AST index / SGLang HiCache 写成论文已有能力
✘ 把超大 codebase 描述为单一连续 KV Cache
✘ 说 AST 替代 embedding distance
✘ 说 "R0 runtime 零改动即可支撑 C40"
✘ 用 median-of-ratios 合成 workload speedup
✘ 用 Amdahl 式 1/((1-f)+f/s) 作为可核对硬约束
✘ 用 survivorship 差分估计 selection 收益
✘ 把臂 C/B 差分称为 "selection gain"
✘ 只报 conditional speedup 而不报 w 与 mu_theta
✘ 在 pilot 冻结 `s_pilot` / `n_confirmatory` 之前引用任何四态判定
✘ 把“CI 跨 0”写成 NEGATIVE，或把“CI 跨 delta0”写成 SMALL_POSITIVE_BELOW_MDE（两者都只能是 INCONCLUSIVE）
✘ 用 pilot 结果给出 POSITIVE / NEGATIVE / SMALL_POSITIVE_BELOW_MDE 终局判定
✘ 把 E_work_pooled 当作 primary，或与 exp(mu_theta) 混用/互替
✘ 在看到 confirmatory 数据后调整 delta0 / delta1 / alpha / power
✘ 把 theta（log-ratio）与毫秒或百分比直接比较
✘ 声称 "质量无损" / "无损伤"
✘ 事后放宽非劣性 margin（M = 5 pp 已 pre-data 冻结）
✘ 把 W6a calibration 数据并入 W6b 的最终非劣性检验
✘ 在 disjoint task 不足时复用 calibration task 后仍称 confirmatory
✘ 在 NO_COVERAGE 情形下表述为"未观察到质量损伤"
✘ 声称 clean-room "无血缘" / "完全独立"（只能说"未检测到禁止血缘"）
✘ 声称 selector "无漏检" / "FN 恒为 0"（只能说冻结语料上 collector_observed_FN=0）
✘ 声称"不存在 fingerprint bypass"（只能说已扫描路径中未发现）
✘ 声称"无泄漏"而不限定 soak 范围
✘ 用 Merkle snapshot 或 git status 充当 write/read authority 或差分 oracle
✘ 说 Merkle 能提供 read_paths 或能证明"无瞬时写"
✘ 在 event-level collector 不可用时改用 Merkle 顶替后仍声称 group 可用
✘ 把 W4a 的 w / r 外推为一般 coding agent 的覆盖率
✘ 把 ADMISSION_DEFERRED 当作失败或 fallback
✘ 用 known_gap 替代 D-3 的完成
✘ 把 3 次 repeat 当成 3 个独立 task
✘ 把 formal request 当成独立样本做 bootstrap
✘ 把 task-runs 折算成 server starts
✘ 引用外部 164/225、169/225、8.55×、4.77× 与 C40 并列
✘ 引用 collaborator 的 357.6→327.5ms、295.5→258.3ms、1.089x 作为已验证结论
   （这些是 external-unverified claim，来自不同 cohort/protocol，raw 不在 Git）
✘ 把 Phase7.5 结果回填或合并进 Phase4–7 统计
```

---

## 27. 冻结默认与待用户决策项

### 27.1 已冻结的默认（**不再是 TBC**）

以下项目此前列为待确认，现全部**冻结为默认值**。执行者按默认值实施，
无需再次询问；若将来要偏离，须走 §23.1 的 design 变更流程。

| ID | 项目 | **冻结默认** | 依据 |
| --- | --- | --- | --- |
| `FD-1` | tool wrapper 归属 | **benchmark harness 内自建 collector**（`benchmark/approx_kv/coding_c40/` 下的 wrapper + event collector），**不改动**任何外部 agent | 保持 clean-room 边界；避免外部依赖 |
| `FD-2` | runtime package 位置 | **`python/sglang/srt/mem_cache/approx_kv/coding_c40/`** | middle-span controller 必须接 scheduler seam（§8.6.3），无法留在 benchmark 侧 |
| `FD-3` | AST gate 范围 | **Python-only，且仅作 auxiliary**；`C40_AST_GATE=1` 必须 `C40_EMBED_GATE=1`；AST 只能加严不能放宽 | §6.3 CL-C；embedding distance 仍为主信号 |
| `FD-4` | 阶段报告路径 | **`research/phase_reports/PHASE7_5_C40_REPORT.md`** | 与 Phase4–7 报告集一致 |
| `FD-5` | W4 workload | **`W4a C40 Live Trajectory Corpus v1`**（§18.4），`>= 24` 条 live trajectory | 取代"真实轨迹不可得就阻塞" |
| `FD-6` | 模型 | **`Qwen/Qwen3-0.6B @ c1899de2...`**，与 Phase7 一致 | 保持与 R0 不利先验的可比性 |
| `FD-7` | provenance collector 默认实现 | **Docker 内 `strace`/`ptrace` collector**（需 `--cap-add=SYS_PTRACE`），等效实现须声明 `collector_impl` | §9.2.2 |
| `FD-8` | conditional lane 默认优先级 | `CL-A 多岛 > CL-D host/prefetch-hint > CL-B repair > CL-E 并发 > CL-I exact-boundary clipping > CL-C AST > CL-H 版本演化 > CL-F chaining > CL-G quality gate` | 按对 primary 结论的信息增益排序；**用户可覆盖** |

### 27.2 Gate probe（取代"数据集/外部资源可得性"TBC）

外部资源的可得性**不再作为待确认项**，改为在对应 Gate 内执行一次
**只读 probe**，probe 结果写入 evidence 并直接决定该轴状态：

| Probe ID | 在哪个 Gate 执行 | 探测对象 | 通过 ⇒ | 不通过 ⇒ |
| --- | --- | --- | --- | --- |
| `PB-1` | G1b | W4b 真实外部 trajectory 语料的可得性与许可 | W4b 作为补充轴执行 | W4b 标 `BLOCKED_EXTERNAL`；W4a 仍支撑机制结论，外部有效性限制保留在 disposition |
| `PB-2` | G1b | RepoBench-P 数据集在受控环境中的可得性与许可 | Stage G-1 可执行 | G-1 标 `BLOCKED_EXTERNAL`，不占预算 |
| `PB-3` | G1b | SWE-bench Verified harness 与镜像的可得性 | Stage G-2 可执行 | G-2 标 `BLOCKED_EXTERNAL`，不占预算 |
| `PB-4` | G8 Exit 后 | prefetch 能力在受控环境中的可得性 | G10 可申请 | G10 保持 `BLOCKED_EXTERNAL`，不占预算、不阻塞任何 lane |
| `PB-5` | G1b | `--cap-add=SYS_PTRACE` 在固定 image 内是否可用（否则试等效实现） | 使用默认 collector | 按 §9.2.3 fallback 顺序降级；全部不可用 ⇒ shell 类 group 一律 `unknown_effect=true` |

> probe 是**只读探测**，不占 GPU、不产生实验数据，属对应 Gate 的常规 Action。

### 27.3 真正需要用户决策的项（**仅此 4 类**）

| ID | 决策项 | 说明 | 默认建议 |
| --- | --- | --- | --- |
| `UD-1` | **逐 Track 的授权与预算** | 每个 Track / Gate 的 `implementation` / `docker_test` / `gpu` / `budget` 授权，以及 start cap 与 GPUh cap | 按 §22 的分 Track 申请；Track A 先行 |
| `UD-2` | **更大模型是否作为独立轴** | 是否在 Qwen3-0.6B 之外增加更大模型（如 Qwen2.5-Coder-3B）。这会破坏与 Phase7 先验的可比性，且本机 8 GiB 显存可能不足 | **默认不加**；如需加，作为**独立轴**并单独授权，不替换 primary |
| `UD-3` | **是否 push 到 origin** | Phase7.5 branch 的 push / PR / tag 涉及 GitHub 写操作与 `ccdd2023` 身份核实 | **默认不 push**；如需 push 须单独授权并核实权限 |
| `UD-4` | **conditional lane 优先级是否覆盖默认** | `FD-8` 已给出默认排序 | 采用 `FD-8` 默认，除非用户指定 |

## 28. 版本与变更记录

### 28.0 Independent-review finding traceability

| Finding | Closure anchor |
| --- | --- |
| `F-01` / `F-02` | §8.6.1–§8.6.5：borrowed/owned、effective prefix、底座KV ledger、allocation/retract tests |
| `F-03` | §8.6.2a：`prefill_adder_identity_membership_v1`，不修改既有返回类型 |
| `F-04` | §4.4/§4.6/G0q：五类quarantine字段、blocked-ref语义 |
| `F-05` | §19.2a–§19.6：90% two-sided decision CI |
| `F-06` | §6.3 CL-I、§17.23、Stage E-4、G7 |
| `F-07` | §20.1及G0a→G0q→G1a→G0b→G1b |
| `F-08` | §9.2/§17.1.4：collector capability与primary/supplemental oracle分层 |
| `F-09` | §18.6/§19.8/G9：5pp pre-data margin、W6a/W6b disjoint |
| `F-10` | §8.1/§12/§13/WP5：terminal taxonomy、四计数族、final cleanup + retract |
| `F-11` | CR-7/§11.3/§23.3：恰好两处写目标与最小rw mount |

| Version | 日期 | Status | 变更 |
| --- | --- | --- | --- |
| `V1`（初稿） | 2026-07-29 | `Draft Candidate` | 初稿。冻结 Phase7.5 定义、clean-room 合同、branch/base、能力三层、架构与模块树、middle-span 状态机、provenance schema、identity/fingerprint、config、terminal reason、telemetry、manifest schema、candidate/axis taxonomy 与 staged gates、Work Packages、测试与实验设计、workloads、统计合同、Gate、stop rules、Track A–F 预算、governance、执行顺序、表述边界 |
| `V1-r1`（中间修订，已被当前稿取代） | 2026-07-29 | `Superseded Draft` | 闭合首轮独立审阅中的 Gate 循环、provenance authority、统计 estimand、W4a workload、质量阈值、conditional lane 与授权边界问题；该修订未获得 review closure，不是可执行 plan of record |
| `V1-r2`（中间修订，已被当前稿取代） | 2026-07-29 | `Superseded Draft` | 对冻结底座做首轮逐API closure：forced-middle、effective-prefix allocation、单一释放ledger、identity membership、adapter-local dense disposition、collector/mount/预算/estimand；第二轮review仍发现并发chunked与retraction blocker |
| `V1-r3`（中间修订，已被当前稿取代） | 2026-07-29 | `Superseded Draft` | 引入overlap-safe snapshot、single-owner、retract、B-3 transaction与initial deferred/remount修订；第四轮review仍发现telemetry/mount及若干契约残留 |
| `V1-r4`（中间修订，已被当前稿取代） | 2026-07-29 | `Superseded Draft` | 删除parked、逐轮rematch、live-budget/dense-callback、初版cached telemetry与最小mount；第五轮review发现B-3 cached公式和若干P2 |
| `V1-r5`（中间修订，已被当前稿取代） | 2026-07-29 | `Superseded Draft` | 修正cached telemetry、mount、SWA/coverage/collector/tmpfs等；第六轮review仍发现suffix状态与projected-prefix等残留 |
| `V1-r6`（中间修订，已被当前稿取代） | 2026-07-29 | `Superseded Draft` | suffix状态、projected prefix、collector/mount与exact-hit修订；第七轮review发现pre-admission reset及若干P2 |
| `V1-r7`（中间修订，已被当前稿取代） | 2026-07-30 | `Superseded Draft` | helper/target reset与manifest provenance初版；第八轮review发现同轮fallback state及skip-Radix seam残留 |
| `V1-r8`（中间修订，已被当前稿取代） | 2026-07-30 | `Superseded Draft` | pre-admission fallback/skip-Radix初版；第九轮review发现final insertion臂策略与audit分类残留 |
| `V1-r9`（中间修订，已被当前稿取代） | 2026-07-30 | `Superseded Draft` | explicit skip-Radix初版；第十轮review发现final insertion臂策略/复位与audit分类缺口 |
| `V1-r10`（中间修订，已被当前稿取代） | 2026-07-30 | `Superseded Draft` | admission后skip与每臂Radix策略初版；第十轮review发现preflight/commit顺序与rollback状态缺口 |
| `V1-r11`（中间修订，已被当前稿取代） | 2026-07-30 | `Superseded Draft` | preflight/rollback/Radix系统指标初版；第十一轮review发现旧CONTINUE、CL-I与sticky rollback残留 |
| `V1-r12`（中间修订，已被当前稿取代） | 2026-07-30 | `Superseded Draft` | 清除旧返回/CL-I/sticky残留；第十二轮review仅余CI method P2与三项观察 |
| `V1-r13`（当前） | 2026-07-30 | `Reviewed Candidate / PENDING USER AUTHORIZATION` | §19.5增加ci_method；quality coverage派生量；eBPF capability/security profile；补充TC-97..99。最终独立closure=`PASS / READY_FOR_PLAN_FREEZE`，open P0/P1/P2=`0/0/0`；G0q=`AVAILABLE` |

> **版本史规则**：本表记录初稿、内部修订与当前reviewed-candidate状态。
> 任何 "open P0/P1 = 0" 或 "review 通过" 的记载**必须**有对应的
> versioned review artifact（`evidence/review/plan-review-*.json`，自哈希并绑定
> 文档 sha256）才能写入；**禁止**在没有 artifact 的情况下声称 review 已闭合。
>
> 状态升级到 `Reviewed Candidate / PENDING USER AUTHORIZATION` 需**同时**具备：
> (a) versioned `plan-review-*.json` 且其中 `open P0/P1 = 0/0`；
> (b) versioned `c40-quarantine-manifest.json`（G0q 产出）。
> 二者齐备后由**主会话**执行升级。

### 28.1 版本规则

```text
- 本文件当前为 `Reviewed Candidate / PENDING USER AUTHORIZATION`；
- closure review与G0q已完成；对应artifact与本状态升级在同一发布commit版本化；
- 用户授权后才 byte-freeze 并 pin 进 p75-plan-manifest；
- 任何 design 变更 ⇒ 归档为 IMPLEMENTATION_PLAN_PHASE7_5_C40_V<n>_ARCHIVED.md，
  创建新版本，更新 design hash，重新 review；
- 授权状态变更（不改 design）只递增 manifest revision，不改本文件版本号；
- 本文件**不**取代 IMPLEMENTATION_PLAN_LATEST.md（Phase7 V7 byte-frozen）。
```

### 28.2 最终状态声明

```text
Plan            : IMPLEMENTATION_PLAN_PHASE7_5_C40.md
Version         : V1-r13
Status          : Reviewed Candidate / PENDING USER AUTHORIZATION
Review          : PASS / READY_FOR_PLAN_FREEZE — open P0/P1/P2 = 0/0/0
Quarantine      : AVAILABLE — evidence/review/c40-quarantine-manifest.json
Phase           : Phase7.5 C40 Clean-Room Reproduction & Extended Evaluation
Method          : C40 = G40 structured grounded selector × R0 Raw+RoPE executor
Branch (frozen) : research/phase7.5-c40-cleanroom              [NOT CREATED]
Base   (frozen) : origin/research/cross-store-substrate@0206f17b4255e4b248dafaaeb943be57428dae2f
Results (frozen): /results/phase7_5_c40  →  benchmark/approx_kv/results/phase7_5_c40/
Order   (frozen): exact cache → controlled C40 reconstruction → dense fallback

plan drafting/review authorized          = true   (incl. review artifact + doc sync)
branch creation authorized               = false
implementation authorized                = false
docker test execution authorized         = false
GPU execution authorized                 = false
budget authorized                        = false
quality campaign authorized               = false

Primary estimand : theta_j = log( S_D,j / S_C,j )  per restart
                   mu_theta = mean_j(theta_j)      <- PRIMARY
                   exp(mu_theta) = geometric mean speedup
                   E_work_pooled = (sum_j S_D,j)/(sum_j S_C,j)  <- DESCRIPTIVE only
Design params    : delta0=log(1.05)=0.048790  delta1=log(1.10)=0.095310
                   alpha=0.05 (one-sided)  power=0.80  n_min=4   [PRE-DATA FROZEN]
Sample size      : n = ceil(((z.95+z.80)*s_pilot/(delta1-delta0))^2) = ceil(2856.9*s_pilot^2)
                   then t-iteration / cluster-bootstrap correction, n>=4
Decision CI      : 90% two-sided [L90,U90]  (endpoints == one-sided 95% bounds)
                   95% two-sided reported as descriptive only
Decision states  : POSITIVE (L90>delta0) / NEGATIVE (U90<0)
                   SMALL_POSITIVE_BELOW_MDE (L90>0 and U90<delta0)
                   INCONCLUSIVE (all else, incl. CI crossing 0 or delta0)

Gate bootstrap   : G0a(doc) -> G0q(quarantine) -> G1a(branch) -> G0b(manifest)
                   -> G1b(provenance/cleanroom) -> G2..G11
                   G0a/G0q = AUTHORIZED; G1a onwards = PENDING
                   G0q BLOCKED_QUARANTINE_INPUT => chain halts before G1a
Track B starts   : G4(<=2) + calibration(1) + pilot(>=3) = 6  <= cap 8
Track C starts   : n_confirmatory * 3 (n=4 => 12), cap 12 + contingency 4 = 16

Phase4–7 artifacts                       = FROZEN, read-only
IMPLEMENTATION_PLAN_LATEST.md (V7)       = byte-frozen, not superseded
Phase8                                   = NOT auto-triggered
```
