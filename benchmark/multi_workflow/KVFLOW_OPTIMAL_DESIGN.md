# KVFlow 最优场景测试方案

## 背景：为什么原始测试场景无法体现 KVFlow 优势

根据之前的实验结果 (README.md)，KVFlow 在当前基准测试中**没有显著收益**（差异 <3%），原因是：

### 问题 1：各 Workflow Agent 前缀完全独立

原始 `bench_multi_workflow.py` 中，每个 workflow 的 agent 前缀是**独立生成**的：
- `shared_p_len=2048` 是相同文本但会被 dedup
- `unique_p_len=1024` 每个 agent 独立生成，内容不同

这导致**没有跨 workflow 共享前缀**，Priority 策略无法发挥优势。

### 问题 2：测试规模不足

| 配置 | KV压力/轮 | Cache容量 | 压力比 |
|------|-----------|-----------|--------|
| 8wf × 8ag | 196.6k tokens | 60k / 90k | 2-3x |

虽然有一定压力，但无法产生足够的缓存抖动来区分 LRU 和 Priority。

### 问题 3：测试轮数较少

原始测试只有 5 轮（包括 1 轮 warmup），无法充分体现长期缓存复用效果。

---

## 改进方案：`bench_kvflow_optimal.py`

### 核心改进：三层前缀结构

```
┌─────────────────────────────────────────────────────────────┐
│ System Prompt (4096 tokens) - ALL agents share this!        │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Group Prefix (2048 tokens) - Workflows in same group    │ │
│ │ ┌─────────────────────────────────────────────────────┐ │ │
│ │ │ Unique Prefix (1024 tokens) - Per-agent unique     │ │ │
│ │ └─────────────────────────────────────────────────────┘ │ │
│ └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 缓存压力计算

以 8 workflows × 8 agents 为例：

| 层级 | Token数 | 副本数 | 总计 |
|------|--------|--------|------|
| System Prompt | 4096 | 1 (共享) | 4,096 |
| Group Prefix | 2048 | 2 (2组) | 4,096 |
| Unique Prefix | 1024 | 64 (8×8) | 65,536 |
| **总计** | | | **~73,728 tokens** |

- 60k cache: **1.2x 压力**
- 90k cache: **0.8x 压力** (基本无压力)

为了产生足够压力，我们调整参数使 KV 压力达到 200k+ tokens。

### Priority 策略的优势场景

KVFlow 的 Priority 策略在以下场景应表现出优势：

1. **System Prompt 被保护**：所有 agent 都使用相同 system prompt，Priority=0（最高优先级）会被保留
2. **跨 workflow 复用**：Round N 的 system prompt = Round N+1 的 cache hit
3. **LRU 的劣势**：LRU 会驱逐最近未访问的 prefix，可能错误驱逐 system prompt

### 测试场景设计

| 实验 | 规模 | KV压力 | 目标 |
|------|------|--------|------|
| **Exp 1: 小规模** | 4wf × 4ag | ~20k | 验证功能正确性 |
| **Exp 2: 中规模** | 8wf × 6ag | ~60k | 60k cache 压力边界 |
| **Exp 3: 大规模** | 8wf × 8ag | ~80k | 90k cache 对比 |

### 预期结果

| 场景 | LRU (hicache) | Priority (kvflow) | 预期差异 |
|------|---------------|-------------------|----------|
| 小规模 | 正常 | 正常 | 无差异 (cache 够用) |
| 中规模 | TTFT 略高 | TTFT 稳定 | Priority 略优 |
| 大规模 | Round 5+ 开始抖动 | Round 5+ 稳定 | **Priority 明显优** |

---

## 测试参数对比

### 原始 vs 改进

| 参数 | 原始测试 | 改进测试 |
|------|----------|----------|
| System Prompt | 无 (或每个 workflow 独立) | 4096 tokens, 跨 workflow 共享 |
| Group Prefix | 无 | 2048 tokens, 4 workflows 共享 |
| Unique Prefix | 1024 tokens | 1024 tokens |
| 轮数 | 5 | 10 |
| 重点测试 | 无 | System Prompt 缓存命中率 |

### 服务器配置

```bash
# hicache (LRU baseline)
--radix-eviction-policy lru
--enable-hierarchical-cache
--hicache-ratio 2.0
--hicache-write-policy write_through
--max-total-tokens 60000

# kvflow (Priority)
--radix-eviction-policy priority
--enable-hierarchical-cache
--hicache-ratio 2.5
--hicache-write-policy write_back
--enable-hicache-prefetch
--max-total-tokens 90000
```

---

## 运行测试

### 1. 下载 Qwen3-8B 模型

```bash
# 查看下载进度
squeue --me
cat /home/comp/25480812/logs/model-download-*.out

# 如果需要手动下载
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-8B', local_dir='/home/comp/25480812/models/hub/models--Qwen--Qwen3-8B')"
```

### 2. 运行完整测试

```bash
cd /home/comp/25480812/CodeMAS_Project/sglang-kvflow/benchmark/multi_workflow

# 提交 SLURM 作业
sbatch --partition=short run_kvflow_optimal_8b.sh

# 或在 GPU 节点上直接运行
bash run_kvflow_optimal_8b.sh
```

### 3. 分析结果

```bash
python analyze_kvflow_optimal.py \
    --result-dir /home/comp/25480812/logs/kvflow-8b/results
```

---

## 关键指标

### TTFT (Time To First Token)

- 衡量 prefill 阶段性能
- 缓存命中 = TTFT 降低
- **LRU 可能问题**：System prompt 被错误驱逐，导致 Round 3+ TTFT 升高

### Round E2E

- 衡量完整 workflow 性能
- 包含 prefill + decode + KV 管理开销
- **Priority 优势**：减少 CPU-GPU 拷贝，改善 decode 流畅度

### 缓存命中率

- 通过 `--enable-cache-report` 获取
- **期望**：KVFlow 的 system prompt 命中率 > LRU

---

## 已知限制

1. **GPU 内存限制**：8B 模型 + 4x A100 可能需要调整 TP size
2. **下载时间**：Qwen3-8B 模型约 16GB，需要稳定网络
3. **实验时间**：完整测试约 2-3 小时
