# Coding-aware lossy KV handoff — 2026-07-17

> 新 session 的 coding-aware 权威入口。先读本文，再读
> `docs/kvflow/ARCHITECTURE.md`。旧 checkout 中的
> `HANDOFF_2026-07-14.md` 只代表 7 月 14 日快照，不能覆盖本文的 V11
> 状态。

## 0. 新 session 从这里开始

工作目录和分支已经隔离：

```bash
cd /home/gfy/CodeMAS_Project/sglang-kvflow-worktrees/coding-aware
git branch --show-current
# research/coding-aware-lossy

git status --short --branch
# 应为 clean，并跟踪 origin/research/coding-aware-lossy
```

功能开关：

```bash
export SGLANG_KVCOMM_CORE=1
export SGLANG_CODING_AWARE_LOSSY=1
export SGLANG_KV_PREFETCH=0
```

快速回归：

```bash
PYTHONPATH=python:tools \
  /home/gfy/.conda/envs/sglang-kvflow/bin/python -m pytest -q \
  python/sglang/srt/mem_cache/kvcomm/test_core.py \
  python/sglang/srt/mem_cache/kvcomm/test_radix_backend.py \
  python/sglang/srt/mem_cache/coding_aware/test_policy.py \
  tools/test_check_kvflow_branch_scope.py

/home/gfy/.conda/envs/sglang-kvflow/bin/python \
  tools/check_kvflow_branch_scope.py \
  --role coding --base kvflow/shared-core
```

2026-07-17 迁移后实测结果：**36 passed；coding branch scope: OK**。
RTX 4090 reference atlas 测量已完成；这不是 model-server TTFT canary。

## 1. 项目目标：我们现在达成的一致理解

我们研究的是：

> 在没有 KV 淘汰压力的 coding 工作流中，利用已经构建好的、位于请求
> 中部的 KV，选择性复制安全区域并重算风险区域，从而降低 TTFT，同时
> 保留客观 coding accuracy。

允许：

- 代码和其他稳定 prompt 模块的 KV 提前构建；
- prefix 之外的 middle-of-request KV 复用；
- AST、依赖图、文件版本、任务阶段和真实工具轨迹作为风险信号；
- 在报告中把离线 KV 构建成本与在线 TTFT 分开呈现。

必须公平：

- Dense、coding-aware 和 matched controls 使用完全相同的 target prompt；
- 所有 KV 模式使用同一个预构建 pool、eligible set 和机械 copy path；
- 预构建是否计入系统边界必须显式报告，不能只给 steady-state TTFT；
- accuracy 使用官方功能测试或真实工作流测试，不使用模型自评 verdict；
- speedup 必须来自更多 KV 复用，不借助 batching、preemption、异步调度
  或并发顺序技巧。

不属于本分支：

- prefix/middle KV 的预测、prefetch、eviction 和 residency scheduling；
- HiCache CPU/storage→GPU 调度；
- 并发请求下的预取优先级。

这些由 `research/prefetch` 负责。两条路线只通过 policy-neutral KVCOMM
接口组合，组合测试只进入 `integration/coding-aware-prefetch`。

## 2. 干净分支目前有什么

代码基线：

- coding 分支基线：`3d2b15055`
- shared core：`c16bfbb8e`，tag `kvcomm-core-v0.1-rc3`
- prefetch 当前交接：`fa86f8f16`
- integration 当前交接：`d4a7ec132`

核心文件：

- `python/sglang/srt/mem_cache/coding_aware/policy.py`
  - `CodingSegment`
  - `CodingRisk.CRITICAL/STABLE`
  - `build_coding_reuse_plan`
- `python/sglang/srt/mem_cache/kvcomm/`
  - segment identity、generation、lease/resource lifecycle；
  - token-slice 验证；
  - complete-copy/full-RoPE transfer；
  - mismatch/stale/non-resident 时 fail closed 到 Dense。
- `benchmark/multi_workflow/`
  - canonical raw-tool provenance 和 FileVersion selection；
  - V11 labels、只读 resume atlas runner；
  - 严格 4,960-row aggregate 和注册统计 gate。

已经验证：

- critical segment 全量重算；
- stable segment 可按整数 `head_tokens` 重算头部、复制 body；
- source/target token mismatch、missing source、越界和 overlap 安全拒绝；
- RoPE delta 由 source/target 逻辑位置计算；
- coding policy 不导入 scheduler/prefetch，也不调用 `ensure_resident`。

有意不迁入干净分支：

- 旧 checkout 中大规模 benchmark launcher；
- `radix_cache.py` 中历史混合实现。

不要把旧 `radix_cache.py` 整文件复制回来。只迁移信号生成和薄 adapter，
真实搬运必须使用 KVCOMM shared core。

## 3. 已经排除的路线

以下结果不能再被包装成当前成功方案：

| 路线 | 客观结果 | 结论 |
|---|---|---|
| 历史 Uniform FRAC 30% | 曾观察到 31.2% TTFT 改善，但属于通用 prefill-skip + 重复 coding 内容 | 不是 coding-specific；修复后必须重新建立公平 baseline |
| ASTSpanKV | 32-case calibration 有 1 个 Dense→wrong；TTFT 比 Dense 慢 74.29% | accuracy 和 speed gate 都失败 |
| AST-IslandKV | 8/8 功能保留；最快 B8 仍慢 5.04% | 减少 fragmentation 仍没有 Pareto |
| TaskCone L2 follow-up | HumanEval 30/30，TTFT 快 82.94% | matched controls 的 preservation CI 下界为 0，不能声称 coding signal |
| WorkflowModuleKV V9 | 真实 prompt 中稳定非代码容量中位数仅 0.33% | 容量 gate 前停止 |
| SessionGraphKV V10 | schema 修正后合法非 prefix 容量 9.12% | 低于 20% gate，V10 已 falsified |
| FileVersion SessionGraphKV V11 | 4,960-row atlas 有效；delta-R² 0.02467，safe-harm gate 为负 | P0 falsified，P1 保持关闭 |

尤其注意：

- “AST 分类失败”不等于“coding-aware 全部失败”；
- “预构建 KV 可用”不等于历史 31.2% 自动有效；
- 多轮 prefix-staged 小 island copy 的启动开销可以超过节省的 prefill；
- 只有优于 exact-budget Uniform 和 Shuffled，才能证明定位信号有效。

## 4. 已完成并排除的研究路径：FileVersion SessionGraphKV V11

V11 不再只看 AST。它把真实 coding session 拆成：

- task/system instruction；
- agent message；
- tool output/test output；
- workspace edit trace；
- source view；
- current target。

策略只复制同一 session 中较早出现、token-identical、成本为正且仍然
合法的模块。源码视图只有在 canonical raw tool provenance 证明对应文件
之后没有被写入时才能继续复用。

始终 Dense：

- 当前 instruction/observation/target；
- graph distance 0 或 1；
- workspace 变化后的 test output 和 edit trace；
- 路径未解析或之后发生写入的 source view；
- missing、duplicate、length/token-hash mismatch label。

### 4.1 最终可以引用的事实

固定 public cohort：**64 sessions / 192 later-turn requests**。

容量和合法性：

- canonical raw event→file provenance：PASS；
- global fail-closed unresolved write：0；
- median file-version reusable fraction：**21.43%**；
- median cost-positive fraction：**21.43%**；
- 每个 session 至少两个 later turns 有复用：**100%**；
- median copy islands：**4**；
- stable source-view tokens：**206,378**。

P0 已完成：

- negative controls：32 sessions / 1,280 rows，PASS；
- identity max JS：`3.48e-05`；
- change-after max JS：`7.62e-04`；
- 注册门槛：`≤1e-3`；
- upstream edit：128 modules / 640 rows；
- 从 0% 到 50% recompute 的 session-cluster median harm reduction：
  **88.0%**；
- 10k bootstrap 95% CI：**[81.2%, 90.5%]**；
- 119/128 modules 在 50% recompute 时改善。
- formal development atlas：32 sessions / 8 disturbances /
  **4,960 rows**；
- duplicate、missing、extra design keys：**0**；
- lookup p95：**0.04795 ms**，通过 `<2ms` 门槛；
- workflow-feature delta-R²：**0.02467**，10k bootstrap 95% CI
  **[0.01062, 0.04697]**，低于 `0.05` 门槛；
- distance≥2 safe-vs-unsafe harm reduction：**-119.711**，
  CI low **-211.419**，低于 `30%` 门槛；
- formal P0 verdict：**FALSIFIED**。

因此 V11 只证明机械负对照、容量和局部方向性机制，**没有**证明：

- workflow accuracy preservation；
- SGLang end-to-end speedup；
- coding-specific superiority；
- holdout/generalization。

### 4.2 当前精确断点

原始实验资产仍在只读研究 checkout：

```text
/home/gfy/CodeMAS_Project/sglang-kvflow
```

新完成的 delta、formal aggregate、manifest 和最终 gate 在：

```text
/home/gfy/CodeMAS_Project/kvflow-artifacts/impactkv_sessiongraph_v11_20260717/
```

关键文件：

```text
P0_FINAL_VERDICT.md
P0_CAUSAL_ATLAS_GATE.json
ARTIFACT_MANIFEST.json
causal_atlas/development_remaining4_delta_chunk512.jsonl
causal_atlas/FORMAL_DEVELOPMENT.jsonl
```

`ARTIFACT_MANIFEST.json` 记录所有只读输入 SHA。旧 790-row unchunked
partial 已显式列入 forbidden inputs，没有混入 formal aggregate。

## 5. 新 session 的开发顺序

### A. 最小迁移已完成

从旧 checkout 只迁移以下 V11 逻辑到本分支：

```text
benchmark/multi_workflow/sessiongraph_raw_provenance.py
benchmark/multi_workflow/build_sessiongraph_provenance_manifest.py
benchmark/multi_workflow/audit_fileversion_session_capacity.py
benchmark/multi_workflow/build_sessiongraph_v11_labels.py
benchmark/multi_workflow/measure_sessiongraph_atlas.py
benchmark/multi_workflow/analyze_sessiongraph_atlas.py
benchmark/multi_workflow/analyze_sessiongraph_v11_negative_controls.py
benchmark/multi_workflow/analyze_sessiongraph_v11_upstream.py
benchmark/multi_workflow/validate_sessiongraph_v11_artifacts.py
```

迁移原则：

- 逐文件审计，不 merge `fix/placeholder-pool-activation`；
- 不迁 paper、旧 reports 或大量历史 launcher；
- 不迁 scheduler/prefetch；
- 把绝对结果依赖改成显式 CLI 输入；
- 新增 test fixtures，在线路径拒绝 gold/test hidden fields；
- provenance、labels、prompt hash 和统计独立测试。

原始结果目录保持外部只读证据；没有复制 GB 级 KV pool 到 Git。

### B. P0 已完成并 falsified

剩余 180 rows、严格多输入 aggregate 和 10k bootstrap gate 均已完成。
artifact validation、negative controls 和 lookup 通过；delta-R² 与
safe-vs-unsafe gate 失败。按照注册，冻结 V11 falsification，不修改
threshold、disturbance、cohort 或 bootstrap 规则。

### C. P1 保持关闭

以下 P1 模式原本固定，但由于 P0 falsified，**不得运行**：

1. Dense；
2. exact native prefix；
3. FileVersion SessionGraphKV；
4. exact-budget Uniform；
5. Shuffled seed 1729；
6. type-only exact-budget/island control。

P1 accuracy 必须来自客观 workflow 测试。先跑一次 Dense 固定分母并冻结
Dense-correct anchors；之后：

- coding-aware 对 anchors 零 regression；
- paired median end-to-end/TTFT speedup `≥5%`，case/session bootstrap CI
  low `>0`；
- prompt hash、token IDs、eligible set 和整数预算逐请求一致；
- mismatch、zero-gap、partial-RoPE、policy-invalid 均为 0；
- coding-specific claim 还要求优于 Uniform、Shuffled 和 type-only。

如果三种 matched policy 都零退化，只能声称“预构建 middle KV 可安全
加速”，不能声称 SessionGraph 定位更好。

## 6. 关键资产索引

旧 checkout 中的权威证据：

```text
results/impactkv_sessiongraph_v11_20260717/FINAL_VERDICT.md
results/impactkv_sessiongraph_v11_20260717/R0_FILE_VERSION_CAPACITY_REPORT.md
results/impactkv_sessiongraph_v11_20260717/P0_MECHANISM_PARTIAL_REPORT.md
results/impactkv_sessiongraph_v11_20260717/EXPERIMENT_REGISTRATION.json
results/impactkv_sessiongraph_v11_20260717/STAGE_STATUS.json
results/impactkv_sessiongraph_v11_20260717/causal_atlas/design.jsonl
```

历史失败证据：

```text
results/impactkv_astspan_retest_20260716/FINAL_VERDICT.md
results/impactkv_astisland_v1_20260716/FINAL_VERDICT.md
results/impactkv_workflowmodule_v9_20260716/FINAL_VERDICT.md
results/impactkv_sessiongraph_v10_20260717/FINAL_VERDICT.md
```

分支和架构：

```text
KVFLOW.md
docs/kvflow/ARCHITECTURE.md
docs/kvflow/STATUS.md
docs/kvflow/HANDOFF.md
```

## 7. Git 和工作区纪律

- 所有新 coding-aware 代码写在当前 clean worktree。
- 原始 `/home/gfy/CodeMAS_Project/sglang-kvflow` 是脏研究快照，不 reset、
  不 cleanup、不作为合作者 base。
- 不把 prefetch branch merge 到 coding branch。
- shared core 更新通过 `git merge kvflow/shared-core`。
- 两路线组合只 merge 到 `integration/coding-aware-prefetch`。
- 不提交大结果、模型、KV pool 或 paper。
- 每次提交前运行 coding scope checker 和上述 27-test suite。

## 8. 给新 session 的一句话

> FileVersion SessionGraphKV V11 的最小迁移和 4,960-row P0 已完成；
> formal verdict 是 **FALSIFIED**。不要运行 P1，不要修改 paper 或门槛；
> 下一条 coding-aware 假设必须作为新的、独立注册路线提出。
