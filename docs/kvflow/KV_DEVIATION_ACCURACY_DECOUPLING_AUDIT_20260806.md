# 从通用 Lossy KV Reuse 到 Coding-aware：方法、实验链与 KV Proxy 审计

日期：2026-08-06  
用途：面向对项目已有大致了解、但没有跟随每轮实验的研究讨论  
范围：先解释相关工作的算法和评价方式，再引出当前 SGLang 方法与冻结实验；没有补跑 GPU，没有修改旧结果或预注册门槛。

## 结论摘要

通用 lossy KV 工作已经形成一条共同路线：CacheBlend 和 Cache-Craft 在新上下文中保留大部分旧 chunk KV，只重算最受 prefix 影响的 token；KVComm 用相似 anchor 的 K/V offset 近似新 agent 上下文；CacheGen 对 KV 本身做有损编码和传输。这些工作都会用 KV、attention、embedding distance 或 PPL 指导系统动作，但最终仍用 F1、ROUGE、Accuracy、Pass@1 或人工正确性证明质量。

它们没有直接解决 coding-agent 的两个特殊问题：历史 observation 可能因为 repository mutation 而在语义上过期；即使内容仍有效，不同源码、测试和文档对当前 action 的价值也不相同。因此，本项目把通用方法中的“选择哪些 token 重算”进一步拆成三层：文件版本是否有效、当前任务是否需要、旧 contextual KV 是否安全。

当前 SGLang 实现已经具备完整的物理复用路径：只从成功、只读、路径可定位的工具结果建立 cache source；文件发生相关修改时让 source 失效；在后续相同文本再次出现时，把 prompt 分成 Dense gap 和 copied island 交替执行。K 做 RoPE 位置修正、V 直接复制。旧 K/V 来自旧 prefix，因此即使 island 文本相同，它仍然是 **contextually lossy reuse**，不是 exact-prefix cache。

目前最成熟的保守策略一次只复制一个 grounded observation；多 observation 执行器可以复制最多三个 island，但“按长度和新近度多复制”尚未通过 accuracy 晋级。真正未解决的是：在版本合法的 observation 中，怎样区分“对当前任务有用”与“旧 contextual KV 足够安全”。

这份报告聚焦后一个问题：**KV deviation 变小，能否作为最终 coding accuracy 变好的依据？** 当前答案是：不能把它当作单调替代指标或已校准的失败概率。

最关键的证据是：同一批 50 道 DS-1000 题、同 prompt/token、同 selector，把 repair 从 75% 提高到 90% 后，平均 stale fraction 从 `23.00%` 降到 `9.28%`，与 Dense 完全相同的输出从 `35/50` 增到 `38/50`，官方执行通过却从 `11/50` 变为 `10/50`。这足以否定“更接近 Dense 必然提高任务正确率”，但不等于“KV deviation 完全无用”。单 island KV drift 与局部 causal logit JS 仍有 Spearman `0.526`，所以它适合做机制诊断和风险约束。

报告的主线是：

```text
相关 lossy KV 工作怎样复用、修复和评价
        ↓
这些方法在 coding-agent 场景还缺少什么
        ↓
我们的 observation / FileVersion / SGLang 方法怎样补足
        ↓
为什么这种复用有损，以及 Validity / Utility / Risk 的区别
        ↓
实验先确认 KV drift 能解释局部扰动
        ↓
再检验它能否推广到 output、NLL 和官方 execution accuracy
        ↓
得到下一代两阶段 selector 与正式评价方案
```

## 1. 相关工作怎样做 Lossy KV Reuse

### 1.1 它们共同面对什么问题

Exact prefix cache 只在旧文本仍位于 prompt 前缀时完全安全；prefix 后面新增内容不会反向改变它的 K/V。更困难的情况是：一个文本块在新请求中换了前置上下文，或者 KV 本身经过了有损压缩。此时旧 state 与当前 Dense/full-prefill state 不再相同，系统必须在“少算一些”和“答案别变坏太多”之间取舍。

下面四项工作分别从 selective recomputation、cross-agent approximation、context-aware chunk repair 和 KV compression 解决这个问题。先理解它们的算法动作，才能看出我们为什么没有直接照搬其中一个 selector。

### 1.2 CacheBlend：复用 chunk，大偏差 token 分层重算

以 RAG 为例，文档块 `A` 和 `B` 可以分别提前建立 KV；新请求把它们拼成 `A → B → question`。`B` 的旧 KV 没见过前面的 `A`，直接复制会遗漏 `A→B` cross-attention。

CacheBlend 的处理流程是：

1. 复用每个 chunk 已经预计算的 KV，并修正位置；
2. 在每层只选一小部分 token 重新计算 current Q/K/V；
3. 未选 token 保留旧 K/V，选中 token 与全部 current/cached K/V 做 attention；
4. 优先修复相对 full prefill KV deviation 较大的 token，以降低 forward-attention deviation；
5. 将层间重算与下一层 KV 读取流水化，减少 TTFT。

因此 CacheBlend 的 KV/attention deviation 是 **选择 token 和解释机制** 的工具。论文最终在 2WikiMQA、MuSiQue 上用 F1，在 MultiNews、SAMSum 上用 ROUGE-L，并同时报告 TTFT；“质量没有明显下降”不是由 deviation 本身证明的。

### 1.3 KVComm：用相似 anchor 的 offset 近似不同 agent 上下文

KVComm 面向 multi-agent system。不同 agent 的角色 prompt 不同，但其中经常出现相同或近似的 placeholder 内容。直接复制某个 agent 的 KV 会忽略角色 prefix 差异。

它不是挑一段 token 做局部 repair，而是：

1. 找到长度结构匹配的 placeholder；
2. 再用 embedding distance 筛选相似 anchor；
3. 保存这些 anchor 在不同 agent prefix 下产生的 K/V offset；
4. 对新 placeholder 按距离对多个 anchor offset 做 soft aggregation；
5. 用聚合 offset 近似目标 agent 的 K/V，跳过完整 prefill。

这里 embedding/L2 distance 负责 matching 和加权，最终质量则用 MMLU/GSM8K Accuracy 与 HumanEval Pass@1 判断。KVComm 还分别报告 agent-level reuse rate 与 latency。

### 1.4 Cache-Craft：先估计 chunk 的上下文依赖，再修复关键 token

Cache-Craft 同样复用非前缀 RAG chunk，但它重点估计一个 chunk 有多依赖旧 prefix：

1. 用跨 chunk attention 与 chunk 内 attention 的比值构造 Cache Context Impact（CCI）；
2. 将 CCI 与新旧 prefix mismatch 结合，得到 Cache Fix Overhead（CFO）；
3. CFO 决定需要重算多少 token；
4. 优先重算对前序 chunk 有高 inter-attention 的 token；
5. 保存同一 chunk 的多个 prefix variant，优先选择 CFO 较低的 cache。

CCI/CFO 负责决定修复动作，但超参数 `α` 的选择被写成一个质量约束问题：在 validation F1 不低于目标值时，最小化预期 recomputation。最终实验继续报告 ROUGE-L、Jaccard/Accuracy、TTFT，并用 250 人 user study 检查回答正确性。

### 1.5 CacheGen：压缩 KV，而不是跨 prefix 搬运 contextual state

CacheGen 是相邻但不同的路线。它假设上下文 KV 本来就适合复用，主要瓶颈是从远端存储或网络读取完整 tensor 太慢。它利用 KV 在 token、layer 和 channel 上的分布特征，把 cache 编码成更小的有损 bitstream，再在 GPU 附近解码和流式传输。

因此它的 loss 来源主要是 **表示压缩**，不是 CacheBlend/Cache-Craft 那样的 prefix-context mismatch。评价时，CacheGen 在 WikiText 上用 PPL，在 LongChat 上用 Accuracy，在 TriviaQA/NarrativeQA 上用 F1，并报告 TTFT 与压缩率。论文明确说明 PPL 不等同于文本生成质量，只是 proxy。

### 1.6 相关工作的共同证据结构

为避免把不同层级的分数混在一起：F1/ROUGE 主要衡量生成文本与参考答案的重合；Accuracy/Pass@1 判断一道题是否答对或一次代码生成是否通过；PPL 衡量参考 next token 的概率，是语言模型 proxy；TTFT 衡量用户等到第一个生成 token 的时间，是系统指标。

下图表示论文原生实验覆盖了哪一层证据，不表示跨论文分数高低。各论文的数据集、模型、prompt 和 reuse-rate 定义不同，不能直接横向排名。

![相关 lossy-KV 工作的证据层级](assets/kv_deviation_accuracy_20260806/10_related_work_evidence_matrix.png)

| 工作 | Lossy 动作 | 内部 proxy | 最终质量证据 |
|---|---|---|---|
| [CacheBlend](https://arxiv.org/html/2405.16444) | 非前缀 chunk 大部复用，少量 token 分层重算 | KV / attention deviation | F1、ROUGE-L |
| [KVComm](https://arxiv.org/html/2510.12872) | 用多个相似 anchor 的 offset 近似目标 agent KV | length、embedding/L2 distance、KV offset error | Accuracy、HumanEval Pass@1 |
| [Cache-Craft](https://arxiv.org/html/2502.15734) | 根据 prefix 影响修复 chunk 中关键 token | CCI、CFO、inter-attention | ROUGE-L、Jaccard/Accuracy、人工评价 |
| [CacheGen](https://arxiv.org/html/2310.07240) | KV tensor 有损编码、传输和解码 | PPL 等压缩 proxy | Accuracy、F1、PPL |

共同规律是：**内部 proxy 指导动作，下游指标批准动作。** 没有一项工作仅凭 KV/attention distance 下降就完成最终质量证明。

### 1.7 KVComm 的距离消融为什么特别重要

KVComm 附录把不同距离策略真正跑到了 HumanEval 和 MMLU 的任务指标上：

![KVComm 距离与匹配消融](assets/kv_deviation_accuracy_20260806/11_kvcomm_distance_ablations.png)

| Offset 近似 | HumanEval Pass@1 | Reuse rate |
|---|---:|---:|
| 只取 L2 最近 anchor | 47.20% | 78.9% |
| 按 cosine 对多个 anchor 加权 | 83.23% | 82.5% |
| 按 L2 对多个 anchor 加权 | 83.23% | 81.1% |
| Dense / Original | 84.45% | 0% |

“Nearest” 已经使用距离，却只有 `47.20%` Pass@1；多 anchor soft aggregation 达到 `83.23%`。这不说明 L2 无用，而是说明“局部最近”只是一项候选条件，组合方式和跨层误差仍需任务结果验证。

另一项 MMLU matching 消融中，只要求长度一致时 Accuracy `62.1%`、reuse `93.3%`；增加 embedding-distance gate 后 Accuracy `68.0%`、reuse `70.1%`。更严格的 distance gate 可以改善质量，也会放弃复用机会；trade-off 仍由 Accuracy 判断。

### 1.8 为什么还需要我们的 coding-aware 路线

上述方法分别知道 RAG chunk、agent placeholder、prefix attention 或 tensor compression，却不维护 coding session 中的 repository lifecycle。例如：

- 一段 `parser.py` 输出可能仍逐字出现在 prompt 中，但文件随后已被修改；
- 两段 KV 都很稳定，其中测试文件与当前 failure path 相关，配置文档却无关；
- F1 或代码相似度很高，生成代码仍可能因一个 dtype、索引或副作用无法通过测试。

这引出我们的增量：先利用 tool type、repository path、FileVersion 和 mutation provenance 确定哪些 observation 合法；再研究 path/action utility 与 contextual KV risk；最终用官方 execution 而不是词面相似度批准方法。

## 2. 当前工作究竟是怎样做的

### 2.1 复用机会来自真实 coding-agent 历史

假设 agent 正在修复 `parse_config`，历史中出现以下事件：

| 时刻 | Agent 历史 | 后续状态 |
|---|---|---|
| T1 | `sed` 读取 `src/parser.py`，得到 900-token 源码 | T3 后同文件被修改 |
| T2 | `cat` 读取 `tests/test_parser.py`，得到 600-token 测试 | 文件未修改 |
| T3 | `apply_patch` 修改 `src/parser.py` | T1 observation 失效 |
| T4 | `cat` 读取 `docs/config.md`，得到 700-token 文档 | 文件未修改，但未必与当前失败最相关 |
| T5 | 下一次模型请求包含上述历史 | 需要决定复制哪些旧 K/V |

Dense 会在 T5 重新计算整段历史。我们的系统则看到 T2 和 T4 的文本仍然原样存在，可考虑复用；T1 虽然文本也可能还留在 prompt 中，但它描述的是修改前的 `parser.py`，必须拒绝。

这里的复用单位不是“函数签名”或任意 token，而是 agent 已经自然产生的一段 **tool observation**。当这段 observation 在后续 prompt 中形成一个连续 token 区间时，报告称它为一个 **island**。

### 2.2 当前 SGLang 路径分成五步

![当前 coding-aware lossy KV reuse 的完整路径](assets/kv_deviation_accuracy_20260806/00_current_method_pipeline.png)

| 步骤 | 已实现动作 | Coding 信息在哪里 |
|---|---|---|
| 1. 候选准入 | 只接受成功的 `rg/grep/find/sed/cat/head/tail` 等只读工具结果；排除 reasoning、测试、diff、写操作和状态查询 | 区分 repository evidence 与 agent 自己的推理文本 |
| 2. Source 建立 | 对 tool-result 原文做 chat-template 渲染和精确 token 定位，保存 source handle、token hash、路径、内容 hash 和 cache dtype | source 必须能追溯到实际读取的 repository path |
| 3. 版本检查 | Source 注册时和 target 使用前都检查后续 mutation；同路径写入、repository-scoped search 后的任意写入、路径歧义都会 fail closed | FileVersion 和 mutation provenance 防止复制已经过期的代码事实 |
| 4. Pool 与选择 | 保守基线一次保留一个 observation；扩展执行器维护最多三个 source，并按长度、新近度和非重叠约束选择 island | 当前选择仍偏系统启发式，尚未真正使用 path utility |
| 5. 物理执行 | `Dense prefix → copy island → Dense gap → ... → Dense suffix`；K 做 source/target 位置差的 RoPE 旋转，V 直接复制 | SGLang 中真实减少 middle-span prefill，不删除 prompt，不使用 prefetch |

多 source 的“persistent”只表示它可以跨后续请求继续使用，不表示永久存在。文件变更、pool eviction、session reset 或 ledger 不一致都会释放 source；任何 token hash、位置、模型、dtype 或 lease 校验失败都会回退 Dense。

### 2.3 为什么文本完全相同，KV 仍然是有损的

设重复文本为 `x`，它第一次出现时的 prefix 为 `C_s`，当前 target 的新 prefix 为 `C_t`。Dense target 会计算：

```text
KV_dense = TransformerKV(C_t, x)
```

复用路径使用：

```text
KV_reuse = TransformerKV(C_s, x)
```

RoPE 可以把 K 从 source 位置旋转到 target 位置，但不能补回 `C_t` 带来的新 cross-attention。因此通常有 `KV_reuse ≠ KV_dense`。这正是本报告所说的 stale K/V：token 内容没变，contextual hidden state 来自旧上下文。

在上面的例子里，T2 的测试文件可能既合法又与当前 parser failure 有关，但它的旧 K/V 仍然是在 T2 的旧 prefix 下形成的。FileVersion 只能证明“文件内容仍有效”，不能证明“旧隐藏状态在 T5 安全”。

### 2.4 当前算法状态与未完成部分

当前代码里要区分“已经工作的基础设施”和“已经晋级的策略”：

| 组件 | 当前状态 | 证据含义 |
|---|---|---|
| 单 observation grounded reuse | 保守研究 baseline | 已有 exact-same-prompt、真实 copy、零 fallback 的速度证据 |
| Target-time FileVersion guard | 保留 | 解决内容是否合法，不承诺预测模型伤害 |
| 最多三 island 的 pool/executor | 基础设施可用 | 增加复制机会和速度，但无约束的长度/新近度策略未通过 accuracy 晋级 |
| Path/action dependency | 已有 motivation，尚未成为最终线上 selector | 应回答哪段 observation 对当前 action 有用，即 Utility |
| K/V/attention probe | 已有机制证据，便宜 request-level proxy 未通过 | 应作为 Risk 约束，不能直接冒充 accuracy |

所以“我们的当前方法”不是一个已经完成的统一分数。现有系统负责合法 source 与物理 copy；下一代研究目标才是：**先排除 contextual risk 过高的候选，再按 coding path utility 分配固定 copy budget。**

## 3. 为什么要审计 KV deviation

### 3.1 Validity、Utility、Risk 是三个不同问题

| 决策层 | 问题 | `parse_config` 例子 | 可用信号 |
|---|---|---|---|
| Validity | 这段 observation 是否仍代表当前文件版本？ | T1 的 `parser.py` 已被修改，所以无效；T2 测试仍有效 | path、content hash、mutation、token identity |
| Utility | 当前 action 是否需要这段信息？ | 当前是 parser test failure，T2 测试通常比 T4 配置文档更有用 | command/path dependency、当前文件、action 类型、recency |
| Risk | 内容虽然有效且有用，旧 K/V 搬到新 prefix 后会不会扰动模型？ | T2 的旧 K/V 可能缺失 T3 patch 对当前上下文的影响 | K/V drift、attention、causal splice、短 probe |

过去若把三个问题压成一个分数，就容易发生概念错位：同路径并不表示旧 KV 一定危险；KV 很接近也不表示这段内容对当前任务有用。KV deviation 主要属于 Risk 层，本报告只审计它能否继续承担更强的“最终 accuracy 代理”角色。

### 3.2 本报告会反复出现的术语

| 术语 | 通俗解释 | 它不代表什么 |
|---|---|---|
| Dense | 在当前完整 prompt 下重新计算所有 K/V | Dense 输出不一定是正确答案 |
| Source / target | Source 是旧请求中建立 cache 的位置；target 是后续尝试使用它的请求 | 不是训练集/测试集含义 |
| Island | Target prompt 中准备复制的一段连续旧 K/V | 不等于完整函数或 AST 节点 |
| Stale token / stale fraction | 仍保留旧 contextual K/V 的 token；stale fraction 是其占目标区间比例 | 不表示文本内容过期 |
| Repair / recompute ratio | 在 reusable 区间内重新计算 current K/V 的比例；其余才继续复用旧 K/V | `repair 75%` 不是“复制 75%” |
| Value-difference selector | 比较 current V 与 cached V，优先重算差异最大的 token | 只能给出内部差异排序 |
| Residual V-mass | Selector 重算以后，未覆盖位置仍剩下多少 V-difference 能量 | 不是失败概率，除非经过校准 |
| Causal logit JS | 只替换一个 island 后，next-token 概率分布相对 Dense 改变多少；0 表示分布相同 | 不是任务正确率 |
| Exact next-line | 生成行与参考行逐字一致 | 不是运行测试后的功能正确性 |
| NLL | 参考 token 在模型分布下的平均负对数概率，越低通常越好 | 不运行代码，不知道程序是否正确 |
| Official execution | 在数据集官方容器或测试器中运行生成代码 | 是本报告最高等级的 coding quality 证据 |
| Rescue / damage | 同一道题从 fail→pass / pass→fail | 需要成对统计，不能只比较总通过数 |
| AUROC | 用一个分数区分失败和通过的排序能力；0.5 约等于随机，1 最好 | 小样本或数据混杂时不能解释因果 |

### 3.3 从局部机制到最终任务的假设链

实验不是同一件事反复测试，而是逐层检验下面四个假设：

| 假设 | 为什么需要它 | 对应实验 | 当前结果 |
|---|---|---|---|
| H1：KV drift 能定位局部模型扰动 | 否则它连 Risk 特征都不应保留 | 单 island causal splice | 支持：Spearman `0.526` |
| H2：减少 residual drift 会单调提高任务 accuracy | 若成立，可直接优化 deviation | DS-1000 同题提高 repair | 否定单调性：stale↓，execution `11/50→10/50` |
| H3：生成前 residual V-mass 能校准失败风险 | 若成立，可直接设在线阈值 | 72-request frozen audit | 不支持：failure AUROC `0.276`，且有 task-mix 混杂 |
| H4：更接近输出层的 NLL 可以替代 execution | 若成立，可低成本大规模选策略 | Function-capsule 独立集 | 不支持：development 优势在 independent split 反转 |

后文严格按 H1→H4 展开。这样可以同时保留“KV drift 有局部机制价值”和“它不能证明最终 accuracy”这两个不矛盾的结论。

## 4. 实验一：先确认 KV drift 确实反映局部扰动

### 4.1 为什么做

在讨论 accuracy 之前，先要证明内部 distance 不是任意数字。这个实验每次只把一个旧 observation island 的 K/V splice 到 target，其他部分重新计算，然后比较 Dense 与 reuse 的 next-token 概率分布。

使用 causal logit JS 的原因是它紧邻干预：只改一个 island，立即观察模型分布改变多少。JS 越大，说明这个 island 的旧 contextual state 对当前输出分布扰动越大。

### 4.2 结果和含义

| 信号与结果 | Spearman | 能支持的解释 |
|---|---:|---|
| 单 island KV drift → causal logit JS | 0.526 | KV drift 能定位局部表示扰动 |
| Request KV drift → 代码相似度变化 | 0.230 | 完整生成后关系明显变弱 |
| Request KV drift → composed NLL | 0.193 | 对请求级 token 质量预测较弱 |
| 16-token probe → composed NLL | 0.169 | 很短 probe 不能代表完整请求 |
| Dense-target drift → NLL repair utility | 0.059 | 偏差最大的位置不等于最值得修的位置 |

![KV drift 从局部干预到请求级结果的关联变化](assets/kv_deviation_accuracy_20260806/07_proxy_scope_correlation_decay.png)

这些点来自不同冻结 cohort，不能当作同一总体上的严格“衰减曲线”。但第一行回答了 H1：KV drift 不是无效信号，它与局部因果扰动存在中等相关性。后几行则提示，从局部 hidden state 到完整生成会经历组合、解码分支和任务判分三次信息损失。

因此本轮决策不是删除 KV drift，而是把它降级为 **local risk diagnostic / constraint candidate**，不直接优化成最终 accuracy。

## 5. 实验二：让 KV 明显更接近 Dense，execution accuracy 会提高吗

### 5.1 为什么这是主实验

H2 要检验的是单调命题：如果我们保留更少旧 K/V、重算更多 current K/V，最终代码是否至少不变差。最直接的办法是在同一题上增加 repair budget，而不是更换数据集、prompt 或 selector。

这一实验使用 50 道 DS-1000 development 题。DS-1000 的官方 evaluator 会实际运行代码和测试，因此比 exact-line、NLL 或代码相似度更接近真正的任务完成率。

### 5.2 控制了什么，只改变了什么

两臂完全共享：

- 同 50 道题、同 prompt/token hash；
- 同模型、推理引擎和生成参数；
- 同一个第 24 层 value-difference ranking；
- 两臂都有正数 stale K/V，没有 Dense fallback。

唯一核心变化是：

- `repair 75%`：重算 V-difference 最大的 75% token；
- `repair 90%`：重算同一排名的 top-90% token。

Top-90% 是 top-75% 的超集，因此第 24 层未覆盖的 V-difference mass 按构造不会增加。这不是“平均上应该更近”，而是 selector 的集合关系保证更强 repair 留下更少检查层差异。

### 5.3 结果

| 动作 | Repair ratio | 平均 stale fraction | 与 Dense 输出完全相同 | 官方执行通过 |
|---|---:|---:|---:|---:|
| 第 24 层 value-difference repair | 75% | 23.00% | 35/50 | 11/50 |
| 第 24 层 value-difference repair | 90% | 9.28% | 38/50 | 10/50 |
| Dense | 100% | 0 | 50/50 | 12/50 |

![更低 stale fraction 没有带来更高官方通过数](assets/kv_deviation_accuracy_20260806/03_functional_stale_fraction_counterexample.png)

重算从 75% 提高到 90% 后：

- stale fraction 相对减少 `59.6%`；
- 与 Dense 逐字相同的输出增加 3 道；
- 官方 execution pass 反而减少 1 道。

逐题配对能看出变化来自哪里：

![同一批 DS-1000 题目的配对结果转移](assets/kv_deviation_accuracy_20260806/06_functional_paired_transition.png)

| 变化 | 题目 | 类型 |
|---|---|---|
| Rescue | `ds1000/661` | Matplotlib / Seaborn stripplot |
| Damage | `ds1000/644` | Matplotlib shaded error region |
| Damage | `ds1000/163` | Pandas MultiIndex columns |

50 题中 47 题没有翻转；3 个翻转为 1 rescue、2 damage，McNemar `p=1.0`。样本不足以宣称“更多 repair 在总体上更差”。但 H2 声称的是单调保证；一个受控的 `staleness↓、Dense identity↑、execution↓` 反例已经足以否定该保证。

Dense 本身只有 `12/50` pass，也解释了为什么“更像 Dense”与“更正确”不是同一目标。一次小 hidden-state 改变可能破坏 Dense 原本正确的轨迹，也可能偶然修正 Dense 原本错误的轨迹。

### 5.4 证据边界

实验直接记录 stale-token fraction，并保证检查层 residual V-difference 不增；它没有保存所有层的最终完整 KV tensor norm。因此严谨结论是：

> 降低检查层未覆盖 V-difference、减少 stale token，并不能保证提高最终执行准确率。

不能扩大为“所有 KV norm 与 accuracy 完全无关”。

## 6. 实验三：固定相同 repair 预算，distance-aware selector 会更好吗

### 6.1 为什么还需要这一组

上一组通过多重算来降低 residual difference。为了排除“只是 repair ratio 改变了解码轨迹”，这里固定所有 lossy 方法都重算 75%，只改变 **哪些 token 被重算**。

这是一项受控的 CacheBlend-derived selector probe，不是当前 SGLang observation-pool 的线上算法。它的用途是隔离“distance-aware token selection”是否有效，不能把它写成当前方法的端到端成绩。

### 6.2 两个 selector 怎样工作

通用 value-difference selector：

1. 在检查层计算 current V 与 cached V 的逐 token 差异；
2. 重算差异最大的 75%；
3. 剩余 25% 继续使用旧 K/V。

Semantic + distance-consensus selector：

1. 先由非空代码、非注释代码和结构边界提出 coding-token mask；
2. 比较该 mask 与通用 K/V-difference top-k 的重合度；
3. 重合度至少 75% 才允许 coding mask 改变选择，否则退回通用 selector；
4. 总 repair ratio 仍为 75%，所以计算预算不增加。

它不是“从多个 cache source 中挑几何最近者”，而是让 coding mask 必须获得当前 K/V distance 的同意。

### 6.3 独立 200 题结果

RepoBench-P 测试 next-line completion；它比 NLL 更接近生成结果，但没有运行程序，所以证据等级低于 DS-1000 execution。

| 方法 | Exact next-line | 相对通用 selector | Cache-ready speedup |
|---|---:|---:|---:|
| Dense | 54/200 | −3 | 1.000x |
| 通用 V-difference selector | 57/200 | 0 | 1.181x |
| Semantic + V-distance consensus | 55/200 | −2 | 1.178x |

![固定预算下 distance-aware selector 没有提高 exact accuracy](assets/kv_deviation_accuracy_20260806/09_distance_selector_counterexample.png)

Distance-consensus 在 169/200 题上实际激活，平均 coding/KV agreement 为 `86.87%`。相对通用 selector，它产生 3 个 rescue、5 个 damage，净少 2 个 exact，速度也没有补偿性提升。

历史翻转样本中的 K/V agreement 也没有把 rescue 与 damage 分开：

| Agreement 信号 | Rescue 均值 | Damage 均值 | 高 agreement 识别 rescue 的 AUROC |
|---|---:|---:|---:|
| K-difference top-k | 86.28% | 87.58% | 0.417 |
| V-difference top-k | 87.31% | 87.92% | 0.458 |

翻转样本只有 10 个，不能声称“agreement 导致错误”。可以得出的决策是：在相同预算下，简单的 coding mask + K/V consensus 没有提供可泛化的安全判别力，因此没有进入当前 selector。

## 7. 实验四：Residual V-mass 能在生成前预测失败吗

### 7.1 从机制分数到在线 gate

即使 deviation 不能单调决定 accuracy，它仍可能作为“高风险报警器”。这组受控支线在 72 个会触发 layer-1/top-60% 判断的 DS-1000 validation 请求上，计算：

```text
Residual V-mass
= top-60% repair 之后未覆盖的 squared V-difference
  / repair 之前总 squared V-difference
```

原假设是：数值越大，保留下来的旧 V 差异越多，任务越容易失败。实验在生成前冻结该分数，再与官方 execution outcome 对齐。

### 7.2 分桶结果

| 分桶 | Residual V-mass | 固定路线通过 | Guard 路线通过 | Dense 通过 |
|---|---:|---:|---:|---:|
| Q1：最低 | 0.00297–0.00418 | 1/18 | 1/18 | 2/18 |
| Q2 | 0.00419–0.00498 | 4/18 | 4/18 | 4/18 |
| Q3 | 0.00512–0.00652 | 7/18 | 9/18 | 9/18 |
| Q4：最高 | 0.00679–0.01012 | 7/18 | 6/18 | 5/18 |

![Residual V-mass 分桶与官方 accuracy](assets/kv_deviation_accuracy_20260806/01_v_stale_mass_accuracy_quartiles.png)

固定路线不根据分数改变执行，最适合做纯预测审计。用“更高 residual V-mass 预测失败”的 AUROC 只有 `0.276`，bootstrap 95% CI `[0.158, 0.409]`。

![Residual V-mass 预测任务失败的 AUROC](assets/kv_deviation_accuracy_20260806/05_failure_auc_with_ci.png)

不能把低于 0.5 解释成“高 deviation 提升 accuracy”。Dense 根本不用 lossy KV，却也出现同方向 AUROC `0.340`。Q1 中有 4 道 PyTorch 且全部失败，Q3 中有 11 道 NumPy；分数同时携带 prompt 长度、库类型和题目难度信息，task mix 构成了混杂。

所以 H3 当前不成立：Residual V-mass 还不是 treatment-induced failure probability，不能跨任务直接设一个固定门槛。

## 8. 实验五：NLL 更接近输出，能否替代 execution

### 8.1 NLL 怎样计算

给定参考答案 token `y_1...y_T`，平均 NLL 为：

```text
NLL = -(1/T) × Σ log p(y_t | prompt, y_<t)
```

NLL 越低，说明模型给参考 token 的平均概率越高。它比 KV distance 更接近输出层，但仍然没有自由生成并运行代码；一个参数、shape 或副作用错误就足以让 execution fail，而平均 NLL 可能几乎不变。

### 8.2 为什么检查 function capsule 实验

这组实验不是当前 observation-pool 方法。它曾尝试从 repository 选择多个完整函数，组成可复用的 function capsule，再对 capsule 尾部做 Dense repair。把它放在这里，是为了检验更靠近输出层的 proxy 是否能承担方法晋级，而不是重新主张 function capsule 路线。

Development 8 题上曾得到：

- stale-KV NLL loss `0.00333`；
- pipeline 相对 full-tail 的 NLL 优势 `+0.02393`；
- 7/8 wins，1 个 severe loss。

独立 17 题上：

- stale-KV NLL loss 仍很小，为 `0.00417`；
- pipeline 优势反转为 `−0.00707`；
- 只有 5/17 wins，并出现 3 个 severe losses。

![NLL 从开发集到独立集发生反转](assets/kv_deviation_accuracy_20260806/08_nll_generalization_reversal.png)

这里主要失败的是“所选函数是否足以完成任务”，不是 stale-KV kernel 突然损坏。实验说明 NLL 可以排除明显的语言模型退化，却不能替代任务信息充分性和 execution accuracy。H4 因此不成立。

## 9. 五组实验合起来说明什么

### 9.1 证据不是互相矛盾，而是作用范围不同

| 现在可以说 | 依据 |
|---|---|
| KV drift 能定位单个 copied island 对局部 logit distribution 的扰动 | 单 island Spearman `0.526` |
| 更强 repair 能减少 stale token 和检查层 residual V-difference | 75%→90% 的嵌套 top-k 构造 |
| 更接近 Dense 不保证更高 coding execution accuracy | DS-1000 `11/50→10/50` |
| 简单 distance consensus 没有改善固定预算 exact accuracy | RepoBench-P `57/200→55/200` |
| 当前 residual V-mass 没有校准成在线失败概率 | AUROC `0.276` 且受 task mix 混杂 |
| NLL 不能替代独立集任务判分 | Development 优势在 independent split 反转 |

核心原因有三层：

1. **Island 组合：** 多个旧 hidden state 按顺序进入模型，后一个 island 已受到前一个 island 影响，单岛 distance 不能简单相加或取最大值。
2. **自回归分支：** Greedy decoding 对 logit margin 敏感，一次很小的 token 分支变化会改变后续完整轨迹。
3. **程序语义：** 文本连续变化，执行结果却离散；一个 index、dtype 或状态副作用即可翻转 pass/fail。

### 9.2 辅助证据及其边界

| 实验 | 观察 | 可以支持什么 | 不能支持什么 |
|---|---|---|---|
| 固定 selector 的 75/80/85/90% sweep | 80%→85% 时 stale token 142→107，exact 38→36 | 文本保真度也非单调 | 不是功能 execution accuracy |
| Online stronger-repair guard，25 请求 | 3 rescue、2 damage、20 unchanged | Repair 对最终结果是双向干预 | 样本不足以证明净收益 |
| ProbeHead 低-deviation gate | 4,639 配置中 0 个同时满足容量和 JS-harm 门槛 | 单一 head-distance threshold 没有容量—安全共同点 | 未进入 task accuracy，不能声称 accuracy 降低 |

对应图表：

- [重算密度与 next-line 非单调曲线](assets/kv_deviation_accuracy_20260806/02_nested_value_diff_density_sweep.png)
- [Stronger repair 的配对结果](assets/kv_deviation_accuracy_20260806/04_stronger_repair_paired_outcomes.png)

## 10. 审计以后，当前路线怎样调整

### 10.1 保留与停止

保留：

- Grounded read-only observation 作为自然 resident 的 source 类型；
- FileVersion、token identity 和 mutation provenance 作为硬 Validity gate；
- 多 source lease、eviction 和 Dense/copy 交替执行器；
- K/V drift、attention、NLL 作为机制诊断与候选 Risk 特征；
- 真实 positive-staleness copy，不退化成 exact reuse 或用大面积 fallback 虚构 accuracy。

停止：

- 不再把平均 KV deviation 最小化写成最终优化目标；
- 不由单一 distance threshold 直接声称任务失败概率；
- 不把 NLL、JS、exact-line 或 Dense-output identity 写成 coding accuracy；
- 不通过堆叠关键词或 AST 规则去补偿一个未校准的风险分数。

### 10.2 下一代 selector 应是分层决策

```text
FileVersion / provenance validity
        ↓ 只保留合法 observation
K/V contextual-risk constraint
        ↓ 剔除明显高风险候选，不宣称概率校准
Path/action utility ranking
        ↓ 固定预算内选择当前任务真正需要的 observation
SGLang middle-span lossy KV copy
        ↓
Official execution accuracy + TTFT 决定是否晋级
```

Risk 与 Utility 不应直接相乘成一个统一分数。高 utility、高 risk 的 observation 很重要，但旧 KV 危险，应 Dense 重算；低 utility、低 risk 的 observation 即使安全，也不一定值得占用 copy budget。

### 10.3 正式评价链

```text
KV / attention / NLL
  └─ 解释机制、筛选明显高风险候选、做早期淘汰

Exact / F1 / ROUGE
  └─ 检查输出保真度，不冒充 coding functional accuracy

Official execution / Pass@1
  └─ 决定 coding quality 是否晋级

TTFT + copied token-layer cost + reuse rate
  └─ 决定质量通过以后是否真的更快
```

最终主图应该是 **execution accuracy–TTFT Pareto frontier**；KV/attention/NLL 应放在机制图和消融图里解释 selector，而不是放在主结论里替代 accuracy。

## 11. 下一项能够闭合问题的实验

### 11.1 工作负载与控制

- 选择 Dense 有非零 pass rate 的独立 coding-agent 任务；
- 同时包含 Dense-known-pass preservation cohort 和 outcome-independent representative cohort；
- 所有方法使用相同 MAS、消息、prompt/token hash、模型、生成参数和官方 evaluator；
- 当前 observation pool 自然 materialize source，不加入 prefetch，也不额外改写 prompt。

### 11.2 等预算方法臂

| Arm | 固定 copy/recompute 预算下怎样选择 | 回答的问题 |
|---|---|---|
| Dense | 全量 current KV | Accuracy 与 TTFT 基线 |
| General value-difference | 重算差异最大 token，复用其余 token | 通用 distance baseline |
| Conservative grounded observation | 一次只复用一个合法只读 observation | 当前 SGLang 研究 baseline |
| Lowest-risk reuse | 在合法 observation 中优先选择 probe distance 最低者 | 距离近是否真的更安全 |
| Utility-first | 按 path/action dependency 选择，不读取 distance | Coding utility 是否优于 risk-only |
| Risk-constrained utility | 先过滤高 risk，再按 utility 排序 | 分层策略是否优于单一 distance |
| Seeded random matched | 匹配 island 数、长度和 token-layer 成本 | 排除预算、位置和碎片数混杂 |

### 11.3 同时记录与晋级标准

同时记录：每层 K/V drift、attention JS、selected/copied/recomputed token、真实 positive-staleness、cache-ready TTFT、source build、N=4/N=16 amortized latency、输出 hash、NLL、官方 execution、逐题 rescue/damage 和 bootstrap confidence interval。

Risk-constrained utility 只有同时满足以下条件才晋级：

1. 对通用 value-difference 的 paired execution delta 为正；
2. 对 Lowest-risk reuse 的 paired execution delta 为正；
3. 相对自身 Dense 保持正 cache-ready speedup；
4. 真实 copied K/V 大于 0，不能依靠大面积 fallback；
5. 结论在独立 split 保持同方向。

这组实验会直接回答最终研究问题：**在相同 prompt、相同引擎和相同计算预算下，coding path utility 能否比单纯选择 KV distance 更近的复用带来更好的 execution accuracy–TTFT trade-off。**

## 12. 可用于论文或汇报的严谨表述

推荐表述：

> 当前 SGLang 方法从真实 coding-agent 历史中缓存成功、只读且路径可定位的 repository observation，经 target-time FileVersion 校验后，在后续相同文本位置执行 middle-span lossy KV copy。RoPE 仅修正 K 的位置，旧 K/V 仍来自不同 prefix。机制实验显示单 island KV drift 与局部 causal logit JS 的 Spearman 为 0.526；但在相同 DS-1000 请求、prompt/token、引擎和 value-difference selector 下，将 repair 从 75% 提高到 90%，使平均 stale fraction 从 23.00% 降到 9.28%，官方执行通过数却从 11/50 变为 10/50。独立 RepoBench-P 的同预算 distance-consensus selector 也从 57/200 降至 55/200 exact；另一批 72-request 审计中，residual V-mass 预测失败的 AUROC 为 0.276。由此，KV deviation 可作为局部风险诊断和约束候选，但不是 coding-task accuracy 的充分、单调或已校准替代指标。

不应使用：

- “KV deviation 与 accuracy 完全无关”；
- “降低 KV deviation 会导致 accuracy 降低”；
- “AUROC 小于 0.5 说明高 deviation 因果提升 accuracy”；
- “已经直接测量所有层最终 KV tensor norm”；
- “RepoBench-P exact-line 等同于功能 accuracy”；
- “多 observation pool 已经是 accuracy-ready 的最终算法”。

## 附录 A：历史标签与本文名称

本文尽量使用算法含义而不是版本号。查阅旧 artifact 时可按下表对应：

| 本文名称 | 历史标签 |
|---|---|
| Conservative grounded single-observation reuse | V40 |
| Target-time FileVersion guard | V45 |
| Bounded three-observation pool/executor | V46 |
| Function-capsule NLL experiment | P27C / P27E |
| Controlled coding-conditioned repair route | V88 |
| Online residual V-mass guard | V90 |

## 附录 B：可复核数据

机器可读汇总：

- [`audit_data.json`](assets/kv_deviation_accuracy_20260806/audit_data.json)
- [`functional_stale_fraction_counterexample.csv`](assets/kv_deviation_accuracy_20260806/functional_stale_fraction_counterexample.csv)
- [`functional_accuracy_transition.csv`](assets/kv_deviation_accuracy_20260806/functional_accuracy_transition.csv)
- [`v_stale_mass_accuracy_quartiles.csv`](assets/kv_deviation_accuracy_20260806/v_stale_mass_accuracy_quartiles.csv)
- [`distance_consensus_accuracy.csv`](assets/kv_deviation_accuracy_20260806/distance_consensus_accuracy.csv)
- [`kv_agreement_transition_probe.csv`](assets/kv_deviation_accuracy_20260806/kv_agreement_transition_probe.csv)
- [`nested_value_diff_density_sweep.csv`](assets/kv_deviation_accuracy_20260806/nested_value_diff_density_sweep.csv)
- [`triggered_request_outcomes.csv`](assets/kv_deviation_accuracy_20260806/triggered_request_outcomes.csv)
- [`proxy_scope_correlations.csv`](assets/kv_deviation_accuracy_20260806/proxy_scope_correlations.csv)
- [`nll_generalization.csv`](assets/kv_deviation_accuracy_20260806/nll_generalization.csv)
- [`related_work_evidence_coverage.csv`](assets/kv_deviation_accuracy_20260806/related_work_evidence_coverage.csv)
- [`kvcomm_distance_ablations.csv`](assets/kv_deviation_accuracy_20260806/kvcomm_distance_ablations.csv)

重建命令：

```bash
python3 benchmark/multi_workflow/analyze_kv_deviation_accuracy.py
```

脚本校验关键 cohort、case identity，并在 `audit_data.json` 中记录冻结输入的路径和 SHA-256。分析入口为 [`analyze_kv_deviation_accuracy.py`](../../benchmark/multi_workflow/analyze_kv_deviation_accuracy.py)。
