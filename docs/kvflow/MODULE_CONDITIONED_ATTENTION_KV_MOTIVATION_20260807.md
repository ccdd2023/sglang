# 为什么同样的 KV 偏移，在不同 Coding 模块中后果不同

日期：2026-08-07  
状态：探索、task-disjoint 局部机制、未开启单岛 outcome 与 64-token action 验证均完成；局部机制通过，但两个最终行为 selector gate 均失败，未进入 SGLang canary。

## 一页结论

这轮实验回答了一个比“KV 距离越小是否越好”更准确的问题：**同样发生 stale KV 偏移时，如果后续 coding 模块确实大量读取这段 observation，局部计算会不会受到更大影响？**

答案是“会”，并且在新任务上可以复现：

| 主要结果 | 数值 | 含义 |
|---|---:|---|
| 留一任务：`KV drift + 模块` 的风险排序 | `0.791` | 只知道偏移和模块身份已经有较强解释力 |
| 留一任务：加入模块条件 Attention×KV | **`0.942`** | 新信号提高 `+0.151` |
| task-bootstrap 改善 95% 区间 | **`[+0.0749,+0.1568]`** | 提升不是由单个任务造成 |
| 高 drift 内：高 Attention / 低 Attention | **`2.421×`** | Attention 决定偏移进入计算的程度 |
| task-module 配对方向一致率 | **`93.75%`** | 大多数可配对任务方向一致 |
| 局部 output change → 最终 logit JS | `0.152` | 局部机制不能替代最终 accuracy |
| 冻结 16-token probe → 完整 128-token drift | **`0.810`** | 便宜 probe 的确能近似完整 KV 漂移 |
| 完整 drift → 未开启候选的 final JS | **`0.030`** | 真正断裂在“内部偏移 → 最终输出” |
| 64-token continuation 分叉 | `18/36` | 长 action 比 immediate top-1 有分辨率 |
| 不同候选产生不同 action | `7/19 cases` | 低于冻结 8-case gate；不能为此降门槛 |

因此，本轮支持把“模块条件 Attention/KV”作为 **局部机制解释**；它没有证明一个新的 online selector 已经可部署，更没有用内部量冒充 coding accuracy。真实三岛策略只有 1 个 request 满足等预算容量，按预注册门槛停止，未打开三岛 outcome。随后完成的单岛迁移实验进一步否定了“先把这个局部信号直接实现为 online guard”：它能解释局部 attention output，却不能稳定选择最终输出更接近 Dense 的候选。

## 先说明这次要回答什么

当前方法把历史里成功、只读、路径仍然有效的 repository observation 作为中间段 KV 复用对象。可见 token 没有变化，但这些 KV 来自旧 prefix，因此仍然是有损复用。

过去我们常问“旧 KV 与 Dense KV 相差多远”。这不够，因为一段 KV 即使偏移很大，只要后续计算几乎不读取它，局部影响仍可能很小。反过来，一段经常被当前代码路径、测试反馈或下一步 action 读取的 observation，即使偏移中等，也可能产生可见扰动。

本实验把问题改成：

```text
这段 observation 的 K/V 改了多少？
              ×
后续哪个 prompt 模块在多大程度上读取它？
              ↓
该模块的 attention output 实际改变多少？
```

最终代码是否通过测试仍由官方 execution 判断。本报告中的 Attention、KV deviation、row TV 和 final-logit JS 都不是 accuracy 的替代指标。

## 已完成的探索实验

现有 M48 数据包含 50 个 RepoBench-P case、294 个相同长度的单岛物理 splice。我们在不使用 outcome 的情况下，按 Attention 和 KV deviation 各自的中位数切成四格。

![现有 294 个候选的四象限结果](assets/module_conditioned_attention_kv_20260807/01_exploratory_factorial.png)

| Dense Attention | KV deviation | 候选数 | Final-logit JS 中位数 |
|---|---|---:|---:|
| 低 | 低 | 71 | `3.46e-4` |
| 高 | 低 | 76 | `3.59e-4` |
| 低 | 高 | 76 | `7.27e-4` |
| **高** | **高** | **71** | **`1.99e-3`** |

在同为高 KV deviation 时，高 Attention 组的 JS 中位数是低 Attention 组的 `2.735×`。这说明 Attention 不是装饰性特征：它改变了同样的 KV 偏移能否进入当前计算。

但探索结果也给出限制：

| 排序信号 | 与 final-logit JS 的 Spearman |
|---|---:|
| Attention only | `0.158` |
| Raw KV drift | **`0.526`** |
| Attention × drift | `0.513` |

乘积没有超过 raw drift。结合此前模块报告中“模块内乘积更好、把模块混在一起反而更差”的结果，正确的下一步不是继续寻找一个全局乘积，而是显式保留 `assistant action`、repository evidence、tool feedback 和 `next action` 的模块身份。

## 独立验证如何避免重复使用旧证据

新的 cohort 从本地 SWE-bench Verified-500 中 outcome-blind 选择 20 题：

- 排除已有 trajectory-backed motivation 使用过的 49 题；
- 每个 repository 最多两题；
- 7 题为 15–60 分钟、7 题为 1–4 小时、6 题为少于 15 分钟；
- 使用相同 30B Dense coding-agent 后端、相同 prompt 和最多 32 次请求；
- 不运行官方 evaluator，也不根据任务是否解决筛选 prompt；
- 不启用 reuse、Radix prefix reuse 或 prefetch。

排除旧任务后只剩 11 个 repository，其中 Flask 只有 1 个候选，因此 repository cap 下的最大可扩展规模是 `Σ min(2, repo_count)=21`，而不是最初设想的 32。该修正在任何 Attention、KV deviation 或 splice outcome 打开前登记；初始 20 题和统计门槛没有变化。

实际采集共完成 `487` 次 30B Dense agent 请求。两道 Matplotlib 题在官方 SWE 容器启动阶段失败，没有产生可测 prompt；其余得到 `18` 条轨迹，其中 `16` 个任务产生合格候选。失败发生在任何 treatment 之前，也没有补选更有利的题。

最终冻结 `64` 个 request、`137` 个候选。每个候选都是完全相同的 128 个 source/target 可见 token，并且 FileVersion 在 target 时仍有效。这批轨迹只负责提供真实 multi-turn coding prompt。Attention 和 K/V 机制由本地 Qwen2.5-Coder-3B BF16 代理测量，因为当前 30B AWQ SGLang 路径不能导出完整 attention tensor。

## 密封实验协议

每个候选必须同时满足：

1. 来自成功的只读 repository observation；
2. source 和 target 中 128 个 token 完全一致；
3. FileVersion 在 target 时仍然有效；
4. 位于 prompt 中间；
5. 不读取未来 action、reference patch 或 evaluator outcome。

实验分三次打开数据：

1. **容量阶段**：只看候选是否存在，至少覆盖 12 个任务、48 个 request 和 128 个候选；为了同时满足 request 与 candidate 两个门槛，最多可按冻结的任务均衡顺序取 80 个 request；
2. **内部量阶段**：只测 Dense module Attention 和 K/V deviation，然后冻结每个模块的四象限与待 splice 候选；
3. **因果阶段**：才执行 K-only、V-only、K+V，以及真实三岛组合。

模块必须覆盖至少 8 个任务和 48 个 candidate-module 点；每个四象限至少 12 点、6 个任务。最终至少三个模块满足容量，其中必须包含 `next action` 和一种 repository evidence/tool feedback。门槛失败就停止，不降低阈值补救。

### “物理 splice”到底做了什么

例如 agent 先前读过 `src/parser.py`，几轮以后 prompt 中仍出现完全相同的 128 个 observation token：

```text
source 时刻：旧 prefix + [parser.py 的 128 token]
target 时刻：新 prefix + [完全相同的 128 token] + 后续 action/tool history
```

Dense 会在 target 的新 prefix 下重新计算这 128 个 token 的 `K_target, V_target`。有损复用则拿 source 时刻的 `K_source, V_source` 放到 target 的相同 token 区间；K 会先按 source/target 位置差进行 RoPE 平移，V 不做旋转。可见文本没有改，改变的只有该区间内部携带的模型状态。

本轮分别构造三种干预：

| 干预 | target 区间实际使用的状态 | 回答的问题 |
|---|---|---|
| K-only | 旧 K + Dense V | 位置/匹配关系改变本身有多大影响 |
| V-only | Dense K + 旧 V | 被读取出来的内容表征改变有多大影响 |
| K+V | 旧 K + 旧 V | 当前 lossy reuse 的合成影响 |

对每个后续 prompt 模块，我们比较 Dense 与 splice 后的 attention row 以及 attention output；对完整 prompt 末端再比较 next-token logits。前者是局部因果链，后者仍只是模型内部结果，不能冒充 coding accuracy。

## Task-disjoint 结果：Attention 与 KV 偏移确实是交互关系

内部量阶段覆盖 `64/64` 个 request，无错误。四象限冻结后，以下四类模块满足每格至少 12 点、6 个任务的门槛：

- assistant action：agent 后续给出的分析或工具调用；
- path-relevant repository evidence：后续再次读取与候选相同路径的 repository evidence；
- other tool result：测试、搜索或其他工具反馈；
- generation marker：模型即将生成下一步 action 的位置。

路径无关 repository evidence 只有 30 个 candidate-module 点，四格中最多 9 点，因此按原门槛自然排除，没有为了得到结论而降阈值。

![四类后续模块中的四象限结果](assets/module_conditioned_attention_kv_20260807/02_confirmatory_module_factorial.png)

![四象限结果热力图](assets/module_conditioned_attention_kv_20260807/02b_confirmatory_module_heatmap.png)

最重要的模式不是“Attention 大就一定坏”或“KV drift 大就一定坏”，而是两者同时高时局部改变最大：

| 后续模块 | 低 A / 低 D | 高 A / 低 D | 低 A / 高 D | **高 A / 高 D** |
|---|---:|---:|---:|---:|
| assistant action | `3.58e-3` | `5.25e-3` | `5.28e-3` | **`1.12e-2`** |
| path-relevant evidence | `1.95e-3` | `4.88e-3` | `4.93e-3` | **`1.21e-2`** |
| other tool result | `1.30e-3` | `3.61e-3` | `3.14e-3` | **`1.15e-2`** |
| next action | `7.89e-4` | `1.21e-3` | `8.58e-4` | **`2.05e-3`** |

这里 A 是 Dense 模型对候选 observation 的注意力质量，D 是 source/target K/V deviation。表中是物理 K+V splice 后的局部 attention-output 相对变化中位数。

## 不是同一批任务上的拟合提升

为了避免“多加特征当然会拟合得更好”，每次都把一个完整任务留出：模型只在其他任务上拟合，再给被留出的任务排序。

![留一任务的局部风险排序](assets/module_conditioned_attention_kv_20260807/03_held_out_risk_prediction.png)

`KV drift + 模块身份` 的 held-out Spearman 是 `0.791`；加入模块条件 Attention/KV 交互后是 `0.942`。按任务等权计算，平均改善 `0.1158`，task-bootstrap 95% 区间为 `[0.0749, 0.1568]`。四个预先登记的单岛门槛全部通过。

这比探索实验更重要：探索实验里简单的全局 `Attention × drift` 还略逊于 raw drift，而新实验保留模块身份并在未参与训练的任务上改善排序。被支持的不是一个全局乘积，而是“**哪个 coding 模块正在读取这段 stale KV**”这一条件变量。

## K 和 V 分开以后看到什么

![K-only、V-only 与 K+V](assets/module_conditioned_attention_kv_20260807/04_kv_component_ablation.png)

| 物理干预 | Final-logit JS 中位数 |
|---|---:|
| K-only | `1.96e-4` |
| V-only | **`2.44e-4`** |
| K+V | `2.34e-4` |

V-only 在这批冻结候选上略高于 K-only，说明不能只用 RoPE/位置变化解释 lossy reuse；旧 V 携带的 prefix-conditioned 内容表征同样重要。K+V 也不是 K-only 与 V-only 的简单相加，表明两部分存在抵消或非线性组合。因此当前算法不应把“只修 K”当作充分方案。

## 为什么这仍然不是 accuracy 证明

局部 attention-output change 与最终 next-token logit JS 的 Spearman 只有 `0.152`。这与之前 KV deviation 降低却不保证最终 coding accuracy 提升的观察一致：局部扰动经过很多层和后续 token 后可能被放大、抵消，最终任务还要经过多轮 action、工具反馈和执行器。

所以这轮只证明了一个局部因果机制：

```text
旧 prefix 造成 K/V 偏移
        ×
后续 coding 模块是否读取该 observation
        ↓
该模块的 attention/output 局部变化
```

官方 task completion、accuracy 和 TTFT 仍必须在真正实现 online guard 后，用 Dense、CacheBlend、KVCOMM 的 same-prompt 公平实验另测。

## 三岛策略为什么没有继续跑

单岛门槛通过后，协议要求三个 arm 都复制 `3 × 128 = 384` token：当前近期优先、模块风险后路径效用、seeded random。候选还必须由留一任务训练得到的风险阈值判为 safe。

实际只有 `1` 个 request、`1` 个任务同时拥有至少 4 个候选和至少 3 个 safe 候选，远低于 `24 request / 8 task`。因此登记状态为 `STOPPED_BEFORE_MULTI_OUTCOMES`，没有执行三岛 splice。

这个失败对开发方向很关键：当时的结果只允许继续检验单岛 guard，不允许一次挑三个“安全岛”。下一节记录了这项后续检验；它没有通过，所以现在也不应直接实现单岛 online guard。

## 后续单岛迁移实验：探针有效，但最终 selector 无效

为了避免三岛组合误差掩盖信号，我们从同一 task-disjoint cohort 中拿出此前物理 outcome 从未打开的 `82` 个候选，覆盖 `57 cases / 16 tasks`。每个候选仍是完全相同的 128-token、FileVersion 有效 observation。实验在读取新 outcome 前固定了四个等预算单岛方法；每个 arm 都复制恰好 128 token：

| 方法 | 如何选一个 observation | 是否可在线 |
|---|---|---|
| Current recency | 选最近的合法 observation | 是，当前基线 |
| Fixed probe | 使用旧 M49 已冻结的 zero-based layer 17、前 16 token，取 shifted-K/V cosine drift 最大值最小者 | 是，但需付 16-token probe 成本 |
| Module oracle | 用先前 55 个物理候选训练的留一任务模块 Attention×KV 模型选最小风险者 | 否；Dense Attention 只作机制上界 |
| Seeded random | 固定 seed 随机选 | 是，负控制 |

候选、arm 和门槛先冻结，之后才一次性执行全部 `82` 个物理 K+V splice。注册门槛不仅看全体中位数，还要求在 selector 与 recency 真正选到不同 observation 的配对 case 中，至少 60% 获胜。这样可以避免不同 arm 大量选到同一候选、或少数极端值把 aggregate median 拉低。

![冻结 probe 的单岛迁移失败归因](assets/module_conditioned_attention_kv_20260807/06_single_island_probe_transfer_failure.png)

| 等 128-token arm | Final-JS 中位数 | 对 recency 中位数比 | disagreement case | 配对胜率 | 60% gate |
|---|---:|---:|---:|---:|---|
| Current recency | `3.430e-4` | `1.000` | — | — | — |
| Fixed probe | `2.489e-4` | `0.726` | 9 / 8 tasks | **`44.44%`** | fail |
| Module oracle | `2.290e-4` | `0.668` | 10 / 8 tasks | **`50.00%`** | fail |
| Seeded random | `2.290e-4` | `0.668` | 6 / 6 tasks | `33.33%` | diagnostic |

只看 aggregate median，会误以为 probe 和 oracle 都大幅改善；但 random 得到了与 oracle 完全相同的中位数，而配对胜率揭示 probe/oracle 都没有稳定赢 recency。冻结 probe 对 final JS 的全局 Spearman 只有 `0.0187`，平均 case 内 Spearman 只有 `0.0421`，也远低于两个 `0.30` 门槛。因此正式结论是 `NOT_SUPPORTED_FOR_RUNTIME_CANARY`。

### 失败到底发生在哪一层

后验分析没有调 layer、token 数或门槛，只把已经打开的变量接成因果链：

| 关系 | Spearman | task-bootstrap 95% 区间 | 解释 |
|---|---:|---:|---|
| 16-token probe → 完整 128-token KV drift | **`0.810`** | **`[0.675,0.887]`** | 便宜 probe 对完整漂移的近似很好 |
| 完整 KV drift → final-logit JS | `0.030` | `[-0.165,0.189]` | 漂移大小几乎不能排序最终输出变化 |
| 最大模块 Attention×drift → final JS | `0.034` | `[-0.129,0.170]` | 把局部机制直接压成 candidate scalar 仍不行 |
| 16-token probe → final JS | `0.0187` | `[-0.113,0.154]` | 失败不是因为只测了 16 token |

![KV drift 四分位与最终输出变化](assets/module_conditioned_attention_kv_20260807/07_drift_quartile_final_js.png)

完整 drift 从 Q1 到 Q4 增大约 48 倍，但 final JS 的中位数依次是 `1.96e-4 / 3.41e-4 / 1.59e-4 / 3.03e-4`，不是单调关系。82 个候选也只有 1 个改变 immediate top-1 token；这个二元标签分辨率太低，不能拿来训练 guard。

这项反证修正了上一阶段的开发建议：**不再调 probe layer/H，不实现基于 KV distance 的 SGLang runtime guard，也不把局部 Attention×KV 当作 accuracy surrogate。** 下一项实验必须先建立有分辨率的多-token action 或 execution-level 行为标签，再讨论能否用在线可见信息预测；否则只是继续优化一个与最终 coding 任务脱节的内部量。

## 64-token action 实验：比单 token 更敏感，但 selector 仍失败

上一实验的 82 个 splice 只有 1 个改变 immediate top-1 token。为避免因单 token 标签太粗而误杀可行方向，我们在打开 continuation 前冻结了多候选子集：`19 cases / 11 tasks / 36` 个唯一 arm-selected splice；四种方法仍复制相同的 128 token。Dense 和每个 physical K+V splice 都从同一个完整 prompt 贪心生成最多 64 token，遇 EOS 停止。

这个实验只问“选不同 observation 是否会改变 agent 的下一段 action”，不把逐 token 一致率写成 accuracy。任务是否完成仍需执行 patch/tests。

![64-token action 标签的分辨率与配对结果](assets/module_conditioned_attention_kv_20260807/08_action_divergence_resolution.png)

| Arm | 与 Dense 完全相同 | 平均 normalized edit distance | disagreement 中胜 / 平 / 负 | 60% win gate |
|---|---:|---:|---:|---|
| Current recency | `52.63%` | `0.1412` | — | — |
| Fixed probe | `63.16%` | `0.1345` | `44.44% / 33.33% / 22.22%` | fail |
| Module oracle | `63.16%` | `0.1345` | `40.00% / 40.00% / 20.00%` | fail |
| Seeded random | **`63.16%`** | `0.1568` | `50.00% / 33.33% / 16.67%` | diagnostic |

总体 exact-match 看起来 probe/oracle 比 recency 高 10.5 个百分点，但 random 完全相同；而同请求配对后，probe/oracle 胜率又只有 `44.4% / 40.0%`。这与 final-JS 实验的教训一致：aggregate 指标容易被大量“选到同一候选”和少数 case 混淆，不能取代 disagreement pair。

64-token 标签的分辨率确实提高了：`18/36` splice 与 Dense 分叉；但同一 case 内，不同候选真正生成不同 continuation 的只有 `7 cases / 7 tasks`。task 数通过 6-task gate，case 数低于预先冻结的 8，正式 decision 为 `ACTION_TARGET_TOO_SPARSE`。我们不会把门槛从 8 降到 7，也不会因为差一个 case 临时把长度改成 128 token 重跑。

### 为什么 18 个分叉仍不能当 accuracy

完整文本审计显示 token divergence 混合了三种情况：

1. **接近同义改写**：Requests case 中 Dense 写“always a string”，splice 写“always treated as a string”，计划修改仍是 `str(method).upper()`；
2. **解释路径发生变化**：Xarray case 的两个候选分别归因于 MultiIndex 与单维变量处理；必须执行后才知道哪一个正确；
3. **真正的编辑对象变化**：Pylint case 的候选分别计划改 `_splitstrip` 与 `_regexp_csv_transfomer`，可能影响最终 patch。

因此逐 token edit distance 只是发现“有行为变化”的筛子，既不判断同义，也不判断修改是否正确。probe 与 action edit distance 的 `ρ=0.352`、module oracle 的 `ρ=0.265` 也不能补救配对 selector gate。下一次质量实验应直接解析并执行完整 tool action，或者进入 official task completion；不能继续延长 continuation，再把更多措辞差异当作更好的 accuracy proxy。

## 与现有算法的关系

实验不会立刻改 SGLang。它分别审视现有算法的三层：

| 层 | 当前做法 | 本实验能否支持 |
|---|---|---|
| Validity | FileVersion、mutation provenance、fail closed | 只检查候选是否合法，不把它当风险分数 |
| Utility | 路径相交、同目录、interaction distance | 解释哪段 coding evidence 更可能有用 |
| Local mechanism | 当前主要按长度和近期程度选择 | 支持“谁读取 stale KV”这一解释；**不支持直接实现 guard** |

单岛 held-out 模块模型已经稳定优于 drift baseline，但这只是 oracle Attention 下的局部机制代理。三岛比较已注册后因容量不足停止；后续等预算单岛 final-JS 比较也没有通过配对胜率和相关性门槛。因此目前保留 validity 与 coding utility 候选定义，不把 Attention/KV scalar 晋级为线上 risk policy。

## 当前 artifact

```text
/home/gfy/CodeMAS_Project/kvflow-artifacts/
  impactkv_attention_kv_factorial_20260807/exploratory_m48/
  impactkv_attention_kv_task_disjoint_20260807_r1/
  impactkv_module_conditioned_attention_kv_20260807/task_disjoint20/
  impactkv_single_island_probe_transfer_20260807/unopened82/
  impactkv_single_island_action_divergence_20260807/frozen19/
```

代码入口：

```text
benchmark/multi_workflow/analyze_attention_kv_factorial.py
benchmark/multi_workflow/run_attention_kv_task_disjoint_campaign.py
benchmark/multi_workflow/motivate_module_conditioned_attention_kv.py
benchmark/multi_workflow/build_module_conditioned_attention_kv_figures.py
benchmark/multi_workflow/build_module_conditioned_attention_kv_prompt_appendix.py
benchmark/multi_workflow/validate_single_island_probe_transfer.py
benchmark/multi_workflow/analyze_single_island_probe_transfer_failure.py
benchmark/multi_workflow/validate_single_island_action_divergence.py
benchmark/multi_workflow/analyze_single_island_action_divergence.py
```

完整 prompt、模块切分和实际 lossy 复用文本见：

```text
docs/kvflow/MODULE_CONDITIONED_ATTENTION_KV_FULL_PROMPTS_20260807.md
docs/kvflow/assets/module_conditioned_attention_kv_20260807/PROMPT_INDEX.csv
docs/kvflow/assets/module_conditioned_attention_kv_20260807/FULL_PROMPTS.jsonl
```

本轮没有修改旧脏 checkout、paper、prefetch 或已有预注册门槛。
