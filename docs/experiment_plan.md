# KVFlow 实验设计文档

> **目标**: 找到让 Priority + HiCache + Prefetch 稳定超过 LRU + HiCache 的配置和场景
> **状态**: 第一轮机制验证完成；已形成可复现的串行实验套件
> **最后更新**: 2026-05-19

---

## 0.5. 2026-06 更新（Context-Aware Confidence Modifier）

### 0.5.1 新增实验

- `results/ast_kv_distance/` — 121 段代码 × all-pairs L2 距离，Qwen2.5-Coder-7B 最后 4 层。结论：AST 类型单独不是好信号（within/cross ratio = 1.21，within-type 实际上比 cross-type **更远** 21%）。
- `results/same_code_context_variation/` — 24 段代码 × 96 个 prompt 变体（6 position_offset × 4 system_prompt × 4 surrounding_wrap）= 2,304 forward pass。输出 144-cell 4D `predicted_distance_table.json`，被 `context_aware_confidence` modifier 在 runtime 实时查询。

### 0.5.2 改造的 gate

- **移除** `structural_distance_gate` tier（4 sites + `_try_structural_distance_gate` helper）。AST 单独放行复用是错的——实验数据不支持。
- **新增** `context_aware_confidence` 修饰器：在 `exact_code_content_signature` 命中**之后**运行，按查表得的 `predicted_d_norm` 把 base 0.95 置信度乘以 `multiplier = 0.5 + 0.5 * (1 - d/d_max)`。当 `predicted_d = d_max` 时 multiplier=0.5 → confidence=0.475 → 拒绝复用。详见 `KVFLOW_OVERVIEW.md` §3.3。

### 0.5.3 新增 prompt-context 字段（plumbing）

- `nesting_depth` / `prompt_position_offset` / `system_prompt_class` / `surrounding_code_hash` 在 5 个文件中已 plumbing：MAScoder `code_anchor.py` + `kvflow_integration.py`；sglang-kvflow `protocol.py` + `schedule_batch.py` + `scheduler.py` + `radix_cache.py`。
- 新增 telemetry：`lossy_predicted_distance` / `lossy_context_aware_confidence` / `lossy_context_aware_multiplier`。

### 0.5.4 关键数据点

| Bucket | d_norm | multiplier | final conf | outcome |
|---|---|---|---|---|
| (50-200, 0, planner, none) | 1.77 | 0.68 | 0.63 | ✅ allowed |
| (50-200, 50-100, planner, none) | 2.19 | 0.60 | 0.57 | ✅ allowed |
| (50-200, 50-100, tester, imports_wrap) | 2.74 | 0.50 | 0.475 | ❌ refused |

### 0.5.5 Bug 修复

- `_split_node` 现在传播 8 个 anchor / context-anchor 字段到 prefix 节点（之前 prefix 变 "anchor-blind"）
- `ref_count` 在 TreeNode 驱逐时 GC（`_decrement_anchor_refs`），归零的 entry 从 `anchor_kv_store` 移除
- `_store_anchor_kv` 在缺 `code_anchor_token_spans` 时打 `logger.warning`（不再静默 return）

### 0.5.6 测试状态

`python -m pytest python/sglang/srt/mem_cache/test_anchor_match.py` —— 20/20 通过（6 个 regression test + 14 个原有）。

### 0.5.7 相关分支

- sglang-kvflow: `feature/context-aware-kv-reuse`
- MAScoder: `feature/code-anchor-integration`

---

## 0. 2026-05 更新摘要

### 0.1 已确认的关键前提

- 双 server（两个 scheduler）同机同时跑会争用 GPU 资源，TTFT 对比会被污染；需要采用单 GPU 串行 A/B 套件。
- KVFlow 的收益强依赖 “是否存在逐出压力”。在 cache 不触发 eviction 的场景下，Priority/Tiered 可能仅引入开销而不带来收益。

### 0.2 最新实验结论（本地 Qwen2.5-3B-Instruct）

- steady-state 线性共享 workload：LRU 更快（此类场景难以体现 eviction 质量差异）。
- eviction-pressure workload（shared prefix 被 flood 请求挤压）：Priority/Tiered 显著优于 LRU，Phase-3 TTFT 提升稳定可复现。

### 0.3 当前推荐实验入口

- 系统级串行套件：`benchmark/multi_workflow/run_serial_policy_suite.py`
- 逐出压力串行套件：`benchmark/multi_workflow/run_serial_eviction_suite.py`

---

## 1. 核心假设

### 1.1 为什么 Priority 应该超过 LRU

在多 Agent 代码生成系统中，典型的工作流是：

```
Round 1: [Agent A] → [Agent B] → [Agent C] → [Agent D]
Round 2: [Agent A] → [Agent B] → [Agent C] → [Agent D]
Round 3: [Agent A] → [Agent B] → [Agent C] → [Agent D]
```

**LRU 的问题**：当 Round 1 的 D 完成后，GPU cache 开始驱逐。A→D 的 prefix 被 LRU 认为是"最近最少使用"，在 Round 2 需要 A 时会被重新 prefill。

**Priority 的优势**：我们知道 Round 2 第一个访问的是 A，所以 A 的 priority 最低（最近需要），D 的 priority 最高（最远需要）。驱逐时会优先清 D 保留 A。

### 1.2 关键条件

Priority 超越 LRU 需要满足以下条件：

| 条件 | 说明 | 如果不满足 |
|------|------|----------|
| **重复访问模式** | 同一组 Agent 多次执行相同 workflow | Priority 无法利用未来信息 |
| **Shared prefix 存在** | 多 Agent 共享 system prompt 等公共前缀 | 驱逐一个和驱逐多个效果一样 |
| **Cache 压力足够大** | KV 总需求 > 可用 cache 容量 | 两种策略都不需要驱逐 |
| **Priority 正确传播** | `req.priority` 正确写入 `TreeNode.priority` | Priority 策略退化为 LRU |
| **Prefetch 生效** | CPU→GPU 预取足够快 | 需要等待 GPU eviction 后再 load |

---

## 2. 实验设计

### 2.1 三个实验场景

| 实验 | Workflows | Agents/WF | KV 压力/round | Cache 容量 | 压力比 | 预期 |
|------|-----------|------------|--------------|-----------|--------|------|
| **exp1** | 1 | 4 | ~10k tokens | 60k/90k | ~15% | 两者相近（cache 够大） |
| **exp2** | 4 | 5 | ~51k tokens | 60k/90k | ~85% | kvflow 略优（shared prefix 保留更好） |
| **exp3** | 8 | 8 | ~197k tokens | 60k/90k | ~220% | **kvflow 明显优于 hicache** |

**预期**: exp3 是关键分水岭。如果 exp3 中 kvflow 没有明显优势，说明需要调优。

### 2.2 评测指标

| 指标 | 说明 | 测量方式 |
|------|------|---------|
| **TTFT** | Time to First Token（首 token 延迟） | 每次请求计时 |
| **Round E2E** | 一轮完整执行的端到端延迟 | Agent × workflow 一轮总耗时 |
| **cached_tokens** | 命中的 KV cache token 数 | `usage.prompt_tokens_details.cached_tokens` |
| **unique_cached** | unique prefix（不含 shared）命中数 | 估算：`cached - min(cached, shared_p_len)` |
| **cache_hit_rate** | 缓存命中率 | `cached_tokens / total_prefix_tokens` |
| **unique_hit_rate** | unique prefix 缓存命中率 | `unique_cached / unique_p_len` |

**核心判定标准**:

- `speedup.ttft > 1.05` 且 `unique_hit_rate_pct` 提升 > 5% → kvflow 有效
- `speedup.ttft < 1.0` 且差距持续 → 需要调优或问题排查

---

## 3. 配置参数调优

### 3.1 HiCache 参数

当前默认配置：

| 参数 | hicache (baseline) | kvflow |
|------|---------------------|--------|
| `--hicache-ratio` | 2.0 | 2.5 |
| `--hicache-write-policy` | write_through | write_back |
| `--enable-hicache-prefetch` | ❌ | ✅ |
| `--max-total-tokens` | 60k | 90k |

**可能需要调优的参数**:

```bash
# HiCache ratio：CPU/GPU 比例
--hicache-ratio 1.0    # 小: CPU 和 GPU 容量相近
--hicache-ratio 3.0    # 大: CPU 有大量空间存 evicted blocks

# Write policy
write_through   # 每次 GPU 写入同时写 CPU（延迟高但数据安全）
write_back      # GPU eviction 时异步写 CPU（延迟低但需要 prefetch 配合）

# Prefetch threshold：最小预取 token 数
# 当前 hardcoded: load_back_threshold = 10
# 太小: 频繁预取小 block，overhead 高
# 太大: 错过预取时机

# Write-through threshold
write_through_threshold = 1 (write_through) / 2 (write_back)
# 影响: 每 N 个 token 才触发一次 write
```

### 3.2 Priority 参数

```python
# Priority 计算遵循 KVFlow 公式（线性/阶段并行）:
# priority = global_step_counter + steps_to_execution
#
# 其中 steps_to_execution 对线性 workflow 可用 “剩余步数” 近似:
# steps_to_execution = max(1, total_steps - current_step)
#
# 语义对齐服务端 PriorityStrategy：
# - priority 越大表示越久才需要，因此越早可被驱逐
# - priority 越小表示越快会复用，因此应尽量保留
```

### 3.3 驱逐阈值

```python
# 驱逐策略以 “更应该被驱逐” 的排序键为主导。
# 当前 PriorityStrategy 会联合 critical_path_distance / priority / last_access_time 做排序，
# 在高逐出压力下优先驱逐更远才需要的节点，并优先保护更接近执行点的共享前缀。

# 可能问题：
# - evictable_leaves 不包含 pinned/locked 节点
# - 如果 pinned 节点过多，GPU 实际可用空间 < max_total_tokens
# - write_back 时 locked 节点会先 demote 到 CPU，再 evict
```

---

## 4. 排查清单

如果 kvflow 无法超过 hicache，按以下顺序排查：

### 4.1 基础检查

- [ ] **Server 日志确认使用了正确的 eviction policy**
  ```
  grep "eviction_policy\|PriorityStrategy\|LRUStrategy" server_*.log
  ```

- [ ] **Priority 值是否被正确发送**
  ```
  grep "priority" server_*.log
  # 应该看到类似: "priority=15, priority=16, ..."
  ```

- [ ] **HiCache 是否真正启用**
  ```
  grep "hicache\|Hierarchical" server_*.log
  # 应该看到: "HiCache enabled, ratio=2.0"
  ```

- [ ] **Prefetch 是否触发**
  ```
  grep "prefetch\|load_back" server_*.log
  # 应该看到: "[prefetch] SUCCESS" 或 "queued N tokens"
  ```

### 4.2 缓存命中率检查

```bash
# 检查 cached_tokens 是否 > 0
grep "cached_tokens\|cached" bench_*.log

# 如果 cached_tokens = 0 或很小：
# 1. 可能是 --enable-cache-report 没启用
# 2. 可能是 cache 被完全驱逐
# 3. 可能是 priority 没传播
```

### 4.3 常见问题

| 症状 | 可能原因 | 解决方案 |
|------|----------|----------|
| kvflow TTFT 比 hicache 慢 | Priority 传播链断了 | 检查 `req.priority` 是否在 `cache_finished_req` 时传入 |
| cached_tokens 几乎为 0 | 所有 prefix 都被驱逐 | 增大 cache 或减小 KV 压力 |
| prefetch 没触发 | `ongoing_load_back` 队列满 | 检查 `load_back_threshold` 是否过大 |
| write_back 死锁 | locked 节点无法 evict | 检查 `pin_expiry` 是否合理 |
| hicache 和 kvflow 几乎一样 | cache 够大，无驱逐 | 减小 `max-total-tokens` 或增加 KV 压力 |

---

## 5. 迭代计划

### Round 1（已完成）- 机制与实验基建验证

- [x] 修复 PriorityStrategy 方向与层级权重问题
- [x] 打通 KVFlow hint 端到端透传
- [x] 构建单 GPU 串行 suite，避免双 server 争用污染结果
- [x] 增加 eviction-pressure 基准，用于稳定观测 eviction 质量差异

### Round 2（进行中）- 用真实工作流与更强模型验证任务级收益

**目标**: 在真实 MAScoder workload 下验证 “共享前缀保护 + 预取” 是否能稳定带来收益

- [ ] 导出并扩充真实模板库存（不仅限 next_line）
- [ ] 在更强模型上复跑 RepoBench / DS1000 子集，观察准确率与时延共同变化
- [ ] 分离 “hint 开销” 与 “eviction 收益”，给出适用边界结论

### Round 3（待定）- 预取与锁策略（仅在证据充足时推进）

**目标**: 若在高压场景下仍观察到 prefetch 与 eviction 的负交互，再考虑引入锁/窗口机制

- [ ] 增加可观测性：prefetch 队列长度、load_back 成功率、evict 触发次数
- [ ] 在 eviction-pressure 基准中做 prefetch 消融

---

## 6. 代码关键位置

| 功能 | 文件 | 关键函数/行 |
|------|------|-------------|
| PriorityStrategy | `mem_cache/evict_policy.py` | `PriorityStrategy.get_priority()` |
| Priority 写入 TreeNode | `mem_cache/radix_cache.py` | `_insert_helper()` 中 `node.priority = max(node.priority, priority)` |
| Priority 传播 | `mem_cache/radix_cache.py` | `cache_finished_req()` 中 `priority = getattr(req, "priority", 0)` |
| Priority-aware evict | `mem_cache/hiradix_cache.py` | `evict()` 中 `eviction_heap` 构建 |
| Prefetch 选择节点 | `mem_cache/hiradix_cache.py` | `prefetch_next_agent()` 选择 `lowest priority` 节点 |
| Prefetch 调度 | `managers/scheduler.py` | `check_hicache_events()` → `prefetch_next_agent()` |
| Priority scheduling | `managers/schedule_policy.py` | `calc_priority()` 中 priority 队列排序 |
| 请求 priority 设置 | `managers/scheduler.py` | `_set_or_validate_priority()` |

---

## 7. 调优方向总结

```
可能的优化方向（按优先级）:

1. [调参] hicache-ratio: 2.0 → 3.0（给 write_back 更多 CPU 空间）
2. [调参] max-total-tokens: 90k → 70k（增加驱逐压力）
3. [调参] priority 公式: 考虑 workflow 维度
4. [代码] prefetch 阈值: load_back_threshold = 10 → 32
5. [代码] write_through_threshold: 2 → 1（更频繁写 CPU）
6. [代码] add priority-aware eviction for host-side leaves（当前 evict_host 用同样策略）
```
