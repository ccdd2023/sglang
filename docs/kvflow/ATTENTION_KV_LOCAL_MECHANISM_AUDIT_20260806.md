# Attention 与 K/V 偏移能为当前 Coding-aware Reuse 证明什么

日期：2026-08-06  
定位：当前 SGLang 中间段有损 KV 复用的机制实验；不是新的 accuracy 结果，也不是线上 selector 晋级报告。

## 结论先行：不是“没办法做了”，而是要缩小理论命题

`KV deviation 下降 ⇒ coding accuracy 提高` 已被我们自己的实验否定，不能再作为论文主张。但这不等于 Attention 和 KV deviation 都失去价值。新的冻结实验支持一个更准确的命题：

> 对固定层、固定 query 和一个实际复制的 observation island，K/V 偏移造成的 attention-output 扰动可以严格拆成 K 与 V 两项；island 获得的 attention mass 与 K/V 偏移结合后，比裸 KV distance 更能解释本层扰动。

它不能推出：

> 本层扰动更小，所以最终代码一定通过测试。

完整证据链现在应写成：

```text
FileVersion / mutation provenance
        │  证明 observation 内容仍合法（Validity）
        ▼
coding path dependency
        │  解释当前任务更可能使用哪段信息（Utility）
        ▼
attention mass × contextual K/V perturbation
        │  上界并解释一次局部读取受到多大影响（Local Risk）
        ▼
真实 SGLang TTFT + 官方容器执行
           最终批准速度与任务质量（System / Accuracy）
```

因此，理论没有消失，而是被放回它真正能证明的层级。最终 accuracy 继续由 execution 决定。

## 1. 这个实验与当前方法有什么关系

实验没有构造随机 token，也没有改用 CacheBlend 风格的 repair。它直接复用了当前保守方法的 26 个真实 source/target island，来自 13 个 coding-agent 任务：

1. source 是历史中成功、只读、路径可定位的 repository observation；
2. target prompt 中再次出现完全相同的 token span；
3. source K 按 source/target 位置差做 RoPE 旋转；
4. source V 原样复制；
5. island 后面的 target suffix 继续 Dense 计算；
6. 分别只复制 K、只复制 V、同时复制 K/V，观察因果差异。

模型是 `Qwen2.5-Coder-3B-Instruct` BF16 机制代理，采样 Transformer 第 1、9、18、27、36 层。局部界限使用 prompt 最后 32 个 query token，因为它们直接形成下一步 action；另一个已经完成的 global-block 实验覆盖了完整 prompt 的所有结构块和 Dense suffix，而不是只看几十个 token。两项实验回答不同问题：

| 实验 | Query 覆盖 | 回答的问题 |
|---|---|---|
| Global block attention | 完整 prompt 的结构分块、Dense suffix、generation query | 复用后全局 attention 路由是否整体改变 |
| 本次 perturbation bound | 最后 32 个 action-facing query，5 层、所有 heads | 某个旧 island 的 K/V 偏移如何传到本层输出 |

所有结果都属于 3B 离线机制证据，不冒充原生 SGLang 30B accuracy 或 TTFT。

## 2. 为什么必须同时看 Attention 与 K/V

### 2.1 一个直观例子

假设历史里有两段同样长的 observation：

- A 是当前失败测试对应的 `test_parser.py`；
- B 是较早读取、当前不再使用的配置文档。

即使 A、B 的旧 KV distance 相同，它们也不一定同样危险。如果当前 query 给 A 20% attention、给 B 0.2%，A 的偏移更可能立刻进入当前计算。反过来，A 的 attention 很高但 source/target K/V 几乎相同，也未必造成大扰动。

这就是为什么只看 attention 或只看 KV distance 都不完整：

```text
attention mass = 当前 query 读这段内容的强度
K/V deviation  = 读到的地址或内容相对 Dense 改了多少
二者结合       = 这次读取实际可能受到多大影响
```

### 2.2 K 和 V 分别改变什么

对一个固定 query `q`，Dense attention 权重和输出记为：

```text
a_i = softmax(q · k_i / √d)
o   = Σ_i a_i v_i
```

复制旧 island `S` 后，K、V 和 attention 变为 `k'_i`、`v'_i`、`a'_i`、`o'`。局部输出差可精确重写为：

```text
o' - o = Σ_{i∈S} a'_i (v'_i - v_i)       ← V 项：读到的内容改变
       + Σ_i (a'_i - a_i) v_i             ← K 项：读取位置重新分配
```

所以：

- V 偏移即使不改变 attention，也会改变读取到的向量；
- K 偏移即使 V 完全不变，也会改变各位置的权重；
- K/V 同时替换时，两项会叠加，也可能在下游部分抵消。

## 3. 一个适用于当前 island reuse 的有限扰动界

对上式取二范数并使用三角不等式，可得：

```text
||o' - o||₂
≤ Σ_{i∈S} a'_i ||Δv_i||₂
  + ||a' - a||₁ · max_i ||v_i||₂
```

其中第一项是 V-content error，第二项是 K 引起的 attention-redistribution error。再定义：

```text
A_S  = Dense 对 island S 的 attention mass
A'_S = reuse 后对 island S 的 attention mass
ε    = max_{i∈S} |q · Δk_i| / √d
```

当只有 island `S` 的 logits 改变时：

```text
||a' - a||₁ ≤ min(
    2 tanh(ε),
    2 A_S [exp(2ε) - 1]
)
```

于是得到 mass-aware 有限界：

```text
||o' - o||₂
≤ A'_S · max_{i∈S} ||Δv_i||₂
  + min(2 tanh ε, 2 A_S[exp(2ε)-1]) · max_i ||v_i||₂
```

这个式子给出了我们需要的理论解释：

- `A_S` 很小时，K 扰动对权重分配的影响受到 attention mass 限制；
- `A'_S` 很小时，V 扰动几乎没有机会被当前 query 读出；
- attention 高但 `ΔK/ΔV` 小，或 deviation 大但 attention 低，都可能保持较小的局部影响；
- K 与 V 必须分开考虑，不能只用一个平均 cosine distance 代替全部机制。

它仍然是单层、固定 query、output projection 之前的界限；跨层 residual、MLP 和 autoregressive decoding 不在这个定理中。

## 4. 冻结 26 例的数值检查

### 4.1 有限界是否真的覆盖实测扰动

| 检查项 | 结果 |
|---|---:|
| 当前方法真实 island | 26 例 / 13 个任务 |
| Case-layer 聚合点 | 130 |
| Head-query-layer 有限界检查点 | 66,560 |
| 精确有限界违反 | **0** |
| 原始解析界违反 | **0** |
| Mass-aware 解析界违反 | **0** |
| Joint coverage | **100%** |
| 精确界最大超界量 | `6.78e-7`，低于冻结数值容差 |
| Mass-aware 界最大超界量 | `-4.87e-21` |

Mass-aware 界在 130/130 个 case-layer 点上都比不含 mass 的解析界更紧；两者平均值分别为 `19.33` 和 `27.50`，中位比值为 `0.717`。它仍明显比实测误差宽，因此适合解释“哪些量控制局部误差”，不适合直接拿阈值预测失败。

### 4.2 Attention 加权是否比裸 KV distance 更能解释局部误差

![局部机制与端到端传播](assets/attention_kv_theory_20260806/01_local_mechanism_vs_endpoint.png)

| 局部信号 | 与本层 attention-output 相对变化的 Spearman |
|---|---:|
| Attention mass only | 0.170 |
| Raw K/V cosine drift | 0.671 |
| **Attention mass × drift** | **0.833** |
| K/V first-order score | 0.805 |
| 不含 mass 的解析界 | 0.532 |
| Mass-aware 解析界 | 0.725 |

冻结门槛要求 `attention × drift ≥ 0.30`，并且至少比 raw drift 高 `0.05`。实际分别为 `0.833` 和 `+0.162`，两项都通过。

这个高相关性并不是独立 accuracy 预测，而是一次 mechanism-consistency test：理论说 attention 应调制 deviation 的局部影响，真实 current-method island 的层内计算确实呈现这一关系。

### 4.3 结果不是由单独一层偶然造成

![逐层局部相关性](assets/attention_kv_theory_20260806/02_layerwise_local_correlations.png)

| Transformer 层 | Raw drift | Attention × drift | First-order score |
|---:|---:|---:|---:|
| 1 | -0.185 | **0.745** | 0.865 |
| 9 | 0.843 | **0.921** | 0.642 |
| 18 | 0.824 | **0.910** | 0.688 |
| 27 | 0.546 | **0.876** | 0.793 |
| 36 | 0.389 | **0.811** | 0.810 |

Attention 加权在五个采样层都高于 raw drift。特别是第一层，裸 drift 与局部误差呈负相关，而加入当前 query 对 island 的实际 attention 后变为 `0.745`。这说明同样的 K/V 几何差异只有在“模型是否读取它”的条件下才有稳定解释。

### 4.4 K-only 与 V-only 都不是可以忽略的项

![K 与 V 的物理 splice 分解](assets/attention_kv_theory_20260806/03_key_value_component_js.png)

| 物理干预 | Final-logit JS 中位数 | Top-1 改变 |
|---|---:|---:|
| 只复制旧 K，V 用 Dense target | `3.41e-4` | 0/26 |
| 只复制旧 V，K 用 Dense target | `4.56e-4` | 1/26 |
| 同时复制旧 K/V | `4.98e-4` | 1/26 |

V-only JS 在 16/26 例高于 K-only，K-only 在另外 10/26 例更高。不存在一个始终占优的分量。因此下一步若做 repair，不应默认“只修 K”或“只修 V”总是正确，而应保留分量消融。

## 5. 完整 prompt 的 attention 是否被复用破坏

局部界只看 action-facing query，不能回答完整 prompt 的路由。此前的 global-block 实验已经对同一批 26 个 runtime-faithful island 做了全局结构分块：system、user task、assistant action、相关/不相关 observation、copied island、suffix 和 generation query 都被纳入，而不是保存一个不可读的逐 token 方阵。

![全局 block attention 保持情况](assets/attention_kv_theory_20260806/04_global_block_attention_preservation.png)

| 全局指标 | 结果 |
|---|---:|
| Generation query block-attention TV 中位数 | 0.00492 |
| Dense suffix query block-attention TV 中位数 | 0.00264 |
| Generation 最高注意结构块一致率 | 99.23% |
| Dense suffix 最高注意结构块一致率 | 99.23% |
| Final-logit JS 中位数 | 0.000498 |
| Final top-1 改变 | 1/26 |

这支持一个有限结论：当前保守单-island 复用在这批样例上通常保留全局 attention 路由。它不证明生成代码正确，也不能自动推广到最多三 island 的激进策略。

## 6. 为什么局部理论仍然不能推出 Accuracy

同一批数据给出了清楚的断点：

| 关系 | Spearman | 冻结判断 |
|---|---:|---|
| Attention × drift → 本层 attention-output change | **0.833** | 通过 |
| First-order score → 本层 change | **0.805** | 通过 |
| 本层 change → physical splice final-logit JS | **0.220** | 未达 0.30 |
| Attention × drift → final-logit JS | 0.262 | 弱 |
| First-order score → final-logit JS | 0.329 | 仅弱到中等 |

因此总决策是 `PARTIAL_OR_FALSIFIED`，而不是把三个通过项包装成整条链都成立。

局部误差到官方 execution 中间还隔着：

1. output projection、residual 和后续层可能放大或抵消扰动；
2. next-token 概率的小变化可能不改变 argmax，也可能在接近决策边界时改变 token；
3. autoregressive decoding 一旦分叉，后续 prompt 状态完全不同；
4. 代码通过/失败是离散、非平滑的执行判定；
5. Dense 自己也会失败，所以“更像 Dense”不等于“更接近正确程序”。

这也解释了此前提高 repair、降低 stale fraction 却没有提高 DS-1000 execution accuracy 的结果。没有矛盾：repair 确实可能降低局部表示误差，但任务正确性不是这个误差的单调函数。

## 7. 现在可以怎样表述当前算法的理论依据

| 当前动作 | 合理依据 | 不能声称什么 |
|---|---|---|
| 只复用成功、只读、路径定位的 observation | 减少无依据 reasoning 和不可追踪 state；形成连续可复制 island | grounded observation 一定具有更小 KV error |
| FileVersion / mutation fail-closed | 保证 repository fact 在语义上没有明显过期 | file mutation 一定造成更大 logit JS |
| Coding path dependency | 两批 paired attention 实验均显示 path-relevant observation 更被当前 query 使用 | path relevant 一定更安全 |
| RoPE-shifted K + copied V | 保留 token identity 和位置语义，并实际省去 middle-span prefill | 位置旋转会恢复新 prefix 下的 Dense contextual state |
| Attention × K/V 机制分析 | 有有限界和 `0.833` 局部相关性支持 | 能直接预测最终 accuracy |
| 官方 execution + TTFT | 最终判断方法是否值得晋级 | 可以被 NLL、JS 或 KV distance 替代 |

最合适的论文主张不是“我们用 KV deviation 优化 accuracy”，而是：

> 我们把 coding-agent lossy KV reuse 分成语义合法性、任务依赖性和局部 contextual perturbation 三层。Coding provenance 约束前两层；attention-weighted K/V 分解解释第三层；最终系统 trade-off 由相同 prompt 下的 TTFT 与官方 execution 验证。

## 8. 下一步 motivation 实验

当前最值得继续的不是再找一个单标量假装 accuracy，而是检验二维机制是否有因果交互：

1. 在新的 task-disjoint coding-agent 请求上，每个 target 冻结多个 version-valid observation；
2. 按 Dense oracle attention mass 和 K/V drift 构造四格：高 attention/高 drift、高/低、低/高、低/低；
3. 每格复制相同 token budget，做单-island 物理 splice；
4. 主指标先用本层 output change 与 final-logit JS，确认 `attention × drift` 的交互；
5. 只有覆盖足够的官方可执行任务后，才比较 paired execution rescue/damage；
6. oracle 机制成立后，再单独研究不依赖完整 Dense prefill 的廉价在线估计器。

预期判别应是：高 attention/高 drift 局部影响最大，低 attention/高 drift 明显低于它，从而证明 attention 不是装饰性特征；但是否通过测试仍作为独立终点报告。

## 9. 复现入口与边界

代码：

```text
benchmark/multi_workflow/motivate_attention_kv_perturbation_bound.py
benchmark/multi_workflow/test_motivate_attention_kv_perturbation_bound.py
benchmark/multi_workflow/build_attention_kv_perturbation_figures.py
```

冻结 artifact：

```text
/home/gfy/CodeMAS_Project/kvflow-artifacts/
  impactkv_attention_kv_bound_20260806/frozen26_mass_aware/
  impactkv_global_block_attention_20260806/frozen26_r2/
```

本轮没有修改旧脏 checkout、paper、prefetch 分支或既有预注册门槛，也没有把 3B mechanism proxy 写成 30B 原生 accuracy 结果。
