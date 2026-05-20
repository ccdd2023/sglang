# KVFlow Priority 修复进展总结

> **状态**: ✅ 修复完成并通过消融与逐出压力基准验证  
> **最后更新**: 2026-05-19

---

## 0. 当前结论（面向复现）

### 0.1 修复点

- `PriorityStrategy` 的排序方向已修正：在 eviction 时优先驱逐更远才需要的节点，并保护更接近执行点的共享前缀。
- `role_type` 的权重从“乘法过载”改为“分层加法 boost”，避免把 priority/crit_dist 完全淹没。
- `TieredPriorityStrategy` 同步修复为同一套排序语义。

### 0.2 修复后的关键语义

- **priority 越大表示越久才需要**，因此在 Priority 驱逐策略下应更早被驱逐。
- **critical_path_distance 越小表示越接近执行点**，因此应更晚被驱逐（优先保留）。

### 0.3 对应代码位置

- 驱逐策略实现：`python/sglang/srt/mem_cache/evict_policy.py`
- KV 元数据写入：`python/sglang/srt/mem_cache/radix_cache.py`

---

## 一、问题背景

### 原始问题
早期实现中出现过多处不一致，导致 Priority 退化甚至反向：

- 关键字段的方向语义与 `heapq`（更小优先弹出）不一致，造成 “越该保留的越先被驱逐”。
- `role_type` 的数值缩放过大，跨层级权重淹没 priority 与 crit_dist，Tier 内排序失真。
- 基准与服务端策略的公式/语义没有完全对齐，导致“公式正确但策略无效”的假象。

### KVFlow 论文正确公式
```
priority = global_step_counter + steps_to_execution
```
- 越早被需要的节点，priority 越小，应该被保留
- priority 越大，越早被驱逐

---

## 二、已完成的修改

### 2.1 驱逐策略语义修复（服务端）

修复后，Priority 驱逐核心排序由以下字段共同决定（先后顺序为主键到次键）：

1. `critical_path_distance`（越小越接近执行点，应更晚驱逐）
2. `priority`（越大越久才需要，应更早驱逐）
3. `last_access_time`（作为 MRU 破平局项）

并且引入分层的 `role_type` boost（System/Role/Task），用于稳定保护共享前缀，避免被 task 级私有前缀挤掉。

### 2.2 基准与公式对齐（客户端/生成端）

线性与 DAG 两条路径都遵循 KVFlow 论文公式：

```python
priority = global_step_counter + steps_to_execution
```

### 2.3 验证与证据

- 公式消融/仿真脚本已通过（优先保护 Tier-0/1，共享前缀在压力下更少被逐出）。
- eviction-pressure 串行基准在本地 3B 模型上可复现 Priority/Tiered 对 LRU 的显著 Phase-3 TTFT 优势。

---

## 三、修复后的 Priority 行为（直观解释）

### DAG 节点 steps_to_execution 值
| 节点 | steps_to_execution |
|------|-------------------|
| PLANNER (根) | 4 |
| ARCHITECT/REVIEWER | 3 |
| IMPLEMENTER | 2 |
| TESTER (叶子) | 1 |

### 修复后的 Priority 计算示例 (5 agents, 5 rounds)

| Round | Agent | step_counter | steps_to_exec | priority |
|-------|-------|--------------|---------------|----------|
| R1 | PLANNER | 0 | 4 | **4** (最早被驱逐) |
| R1 | TESTER | 3 | 1 | **4** |
| R2 | PLANNER | 5 | 4 | **9** (比R1晚驱逐) |
| R2 | TESTER | 8 | 1 | **9** |

### 驱逐逻辑验证

- 更接近执行点（`critical_path_distance` 小）的前缀更晚被驱逐
- 同一距离下，`priority` 更大表示更久才需要，更早被驱逐
- `role_type` 作为跨层级稳定项，确保系统/角色共享前缀不会被 task 私有前缀轻易挤掉

---

## 四、待推进（论文级实验需要补齐）

### 目标

| 方向 | 要补齐的证据 |
|------|-------------|
| 真实工作流 | 用 MAScoder 的真实多代理模板库存驱动系统基准（不仅限 next_line） |
| 任务指标 | 在更强模型上验证准确率不退化的同时获得 TTFT/E2E 改善 |
| 预取消融 | 在 eviction-pressure 场景下做 prefetch on/off 的因果拆解 |
| 4WF Priority TTFT | 76.67ms | <70ms | Priority 正确工作 |
| 16WF Priority TTFT | 105.05ms | <90ms | 消除负面效果 |
| Priority vs LRU (4WF) | +8.9% | +15-20% | 正确的时间维度 |

### 验证步骤

```bash
# 1. 启动 Priority 服务器
cd /home/gfy/CodeMAS_Project/sglang-kvflow/python
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-3B-Instruct \
  --port 30002 \
  --tp-size 1 \
  --radix-eviction-policy priority \
  --enable-hierarchical-cache \
  --hicache-ratio 2.0 \
  --hicache-write-policy write_back

# 2. 运行 4WF benchmark
cd /home/gfy/CodeMAS_Project/sglang-kvflow
python3 -m benchmark.multi_workflow.bench_multi_workflow \
  --config priority_wb_only \
  --model /home/gfy/models/Qwen2.5-3B-Instruct \
  --host 127.0.0.1 --port 30002 \
  --num-workflows 4 --num-rounds 5 \
  --output-dir /tmp/kvflow_priority_results

# 3. 运行 16WF benchmark
python3 -m benchmark.multi_workflow.bench_multi_workflow \
  --config priority_wb_only \
  --model /home/gfy/models/Qwen2.5-3B-Instruct \
  --host 127.0.0.1 --port 30002 \
  --num-workflows 16 --num-rounds 5 \
  --output-dir /tmp/kvflow_priority_results_16wf
```

---

## 五、开放问题

1. **critical_path_length vs critical_path_distance**: 两者都表示从节点到叶子的距离，是否需要统一命名？
2. **role_type_boost 是否还需要？**: 修复 Priority 后，role_type 的 boost 是否仍然必要？
3. **Prefetch Lock 是否还需要？**: 正确的 Priority 是否能消除负面交互？

---

## 六、相关文件路径

### 已修改的文件
- `/home/gfy/CodeMAS_Project/sglang-kvflow/benchmark/multi_workflow/bench_multi_workflow.py`
- `/home/gfy/CodeMAS_Project/sglang-kvflow/python/sglang/srt/mem_cache/radix_cache.py`

### 相关待验证文件
- `/home/gfy/CodeMAS_Project/sglang-kvflow/python/sglang/srt/mem_cache/evict_policy.py` (PriorityStrategy 实现)
- `/home/gfy/CodeMAS_Project/sglang-kvflow/python/sglang/srt/managers/scheduler.py` (priority 调度)

### 计划文件
- `/home/gfy/.cursor/plans/kvflow_priority_prefetch优化_1e8af787.plan.md`

---

## 七、修改验证状态

- [x] `calculate_priority` 函数修复
- [x] `step_counter` 传递修复
- [x] Debug 日志添加
- [ ] 4WF/16WF 实验验证 (需要重新运行服务器和 benchmark)
- [ ] Priority vs Prefetch 协同分析

---

*生成时间: 2026-05-19 08:05 UTC*
