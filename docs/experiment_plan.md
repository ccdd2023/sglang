# KVFlow 实验设计文档

> **目标**: 找到让 Priority + HiCache + Prefetch 稳定超过 LRU + HiCache 的配置和场景
> **状态**: 实验设计中
> **最后更新**: 2026-03-26

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
# Priority 计算（bench_multi_workflow.py）
priority = step_counter + num_agents  # 当前公式

# 问题 1: step_counter 是全局递增的，但 round-robin 下每个 agent 每 num_agents 步被访问一次
# 所以 priority = N 的节点，下次被访问在 N + num_agents 步后

# 问题 2: 如果多个 workflow 并发，step_counter 是全局的
# 但不同 workflow 的 agent 访问顺序可能不同
# 这可能导致 priority 在跨 workflow 时不准确

# 可能更好的公式
priority = total_steps_in_workflow - steps_until_next_use
# 即: 越快被访问的节点，priority 越小
```

### 3.3 驱逐阈值

```python
# hiradix_cache.py: evict()
# 当前驱逐逻辑：
# 1. 从 evictable_leaves 构建 min-heap（按 priority）
# 2. pop() 获取最小 priority = 最远需要的节点
# 3. 驱逐

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

### Round 1（当前）- 基础对比

- [x] 修复 pipeline bug（`--max-total-tokens` 硬编码）
- [x] 简化配置：只保留 hicache vs kvflow
- [ ] 运行 `quick` 冒烟测试
- [ ] 运行 `exp3` 验证是否在高压下产生差异

### Round 2 - 如果 Round 1 无明显差异

**目标**: 找到让差异显现的参数

- [ ] 增大 `hicache-ratio`（如 3.0）给 kvflow 更多 CPU 空间
- [ ] 减小 `max-total-tokens`（如 40k）增加 GPU 侧驱逐压力
- [ ] 调整 priority 公式：考虑 workflow 并发时的跨 workflow priority 干扰
- [ ] 检查 `prefetch_next_agent()` 是否真的在调度循环中被调用

### Round 3 - 如果 Round 2 仍无差异

**目标**: 确认是代码问题还是场景问题

- [ ] 添加 `SGLANG_DEBUG=priority` 日志打印 priority 传播链路
- [ ] 在 `evict()` 中添加日志，打印每次驱逐的 priority 值
- [ ] 添加 metrics 统计 evict 被跳过的 locked 节点数
- [ ] 对比 evict 发生次数：hicache vs kvflow 是否接近

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
