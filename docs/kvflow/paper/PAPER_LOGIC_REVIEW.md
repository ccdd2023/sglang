# ImpactKV 论文逻辑（审查稿）

投稿形态：ASPLOS 2027，`sigplan,anonymous,review,nonacm`，正文 11 页。  
题目：*ImpactKV: Coding-Aware Lossy KV Reuse for Shifted File-Island Prefill*  
方法名：ImpactKV。评测范围写死为 **sequential one-token prefill**，不是 serving 皇冠。

这份文档按**论证顺序**写，不按实验时间线。数字只引用论文已发表、且 `RESULT.status == COMPLETE` 的战役。  
主表 `tab:eval-summary` **只**来自作业 96092。旁路表不得绑进 1.375×。

---

## 0. 一句话

编码 agent 会在**不同 prompt 位置**重读同一份仓库文件：token ID 相同，RoPE 相位和左右上下文不同。前缀缓存因此 miss。ImpactKV 的贡献不是「再拷多一点 token」，而是：

1. 从 agent 协议里**决定哪些文件岛可以近似**（admit policy）；
2. 在引擎里把这些岛做成 **fail-closed 的 true-lossy 拷贝**（source 侧预旋转 \(K\)，\(V\) 原样）；
3. 用 **cache-ready vs 含 source-build** 两套账把速度说清楚。

Headline 只证明第 2+3 点在 30B 上成立。第 1 点靠编译器定义（文件模块 whitelist），不是靠 SWE-bench resolved。Unconstrained LCS 不是比较对象。

---

## 1. 问题：要复用的东西不在前缀上

### 1.1 工作负载

SWE-bench 式多角色编码（planner / implementer / tester / reviewer）。  
后一个角色重读前一个角色已经编码过的 `repository_code` 文件。中间插了 tool observation，文件在 prompt 里的起点从 \(s\) 变成 \(t\)。

定义 \(\Delta = t - s\)。

- \(\Delta = 0\) 且左边字节也相同 → 普通 radix / prefix cache，不是本文。
- \(\Delta \neq 0\) 且 token ID 相同 → **true-lossy**：token 精确，激活近似。这是本文。

数据集卡（`tab:dataset`，派生自 96092，不是新 GPU）：

| 项 | 数 |
|---|---|
| 轨迹来源 | 24 个 Verified 任务的 live-agent（convenience sample，不是 500-task） |
| 有资格的任务 | 20 / 24 |
| target groups | 235（rolling-6 **轮次**，不是 235 个任务） |
| true-lossy 岛 | 421 |
| 丢掉的 \(\Delta=0\) 岛 | 48（防止实验退化成前缀缓存） |

### 1.2 现有系统答的是另一个问题

| 系统 | 答的问题 | 为什么不够 |
|---|---|---|
| PagedAttention / SGLang radix / KVFlow | 精确前缀共享与预取 | 要求 \(\Delta=0\) |
| CacheBlend / KVCOMM / RelayCaching | 尽量多拷、再修一部分 | 不知道哪些 coding span 可以近似 |
| Continuum / Tokencake | KV 生命周期 / 调度 | 保留 ≠ 文件级 admit |

论文立场：**policy（哪些文件可以进 lossy 岛）** 和 **copier（能搬多少 token）** 是两件事。后者通常更快；前者才是本文 novelty。

### 1.3 损失落在哪（3B probe，不是 30B 注意力）

30B 速度战役**不记 attention**。另开 Qwen2.5-Coder-3B 探针（`tab:attn-proxy`）：

- suffix attention TV **0.00264**（拷完后后缀注意力几乎不塌）
- 岛在 **source 时刻形成** 的 TV **0.0462**（大约高一个数量级）
- Dense 上 top-10% key 吃掉 80.1% 质量（只说明注意力稀疏，不当 admit 门）

**推理方向：** 默认修后缀（CacheBlend 式）是错的 repair；该修的是 island 在 source 前缀下形成的 \(K\) 相位 → 所以 M2 做 source 侧 \(K\) 预旋转，而不是 suffix recompute。  
**硬边界：** template **does not estimate Attention**。TV 不进 `tab:eval-summary`，不绑 1.375×。

---

## 2. 方法：四个模块，Headline 只用两个

`tab:module-io`：

| 模块 | 输入 | 输出 | Headline 96092 |
|---|---|---|---|
| **M0** template compiler | 冻结轨迹 + tokenizer | PLAN：hash、\((s,t,L)\)、\(\Delta\neq 0\) | **开** |
| **M1** prefix reuse | source/target IDs，cap 在 target_start | radix LCP，不拷岛 | **关** |
| **M2** lossy copy | 已 admit 的岛 + source \(K/V\) + \(\Delta\) | 整岛拷贝，或机械失败则 Dense | **开** |
| **M3** prefetch | later-roles、驻留、next-use | 给 M1/M2 的 device hint；miss → M2 | **关** |

关 M1/M3 的原因：主表不能被 radix hit 或 prefetch 领功。

### 2.1 M0：文件模块 admit（贡献 1）

可拷贝当且仅当四条同时成立：

1. 单文件 `repository_code`（一个 path）
2. 该 path 在轨迹里的 content version 仍有效（编辑后的旧文件不拷）
3. source/target 在该 span 上 token ID 逐 token 相同
4. \(\Delta \neq 0\)（零位移丢掉）

不允许的门：path 名 alone、AST 标签、Attention 分数、在线 KV 距离。  
Path/version 只用来**定位** span；**admit 规则只有 token-ID 相等 + 版本有效 + 非零位移**。

编译结果（96092 PLAN）：235 groups、421 岛；每组 1/2/3 岛 = 89/106/40；平均每组拷 1528 / 目标 prompt 4403 token（约 1/3 可拷，其余 Dense）。

「Template」在本文里 = 从冻结轨迹编译出的文件岛 PLAN，**不是** LLM 合成的任务 DAG。

### 2.2 M2：source 侧 \(K\) 预旋转 + fail-closed（贡献 2）

材料化：

\[
K^{\mathrm{pool}} = R_{\Delta}\, K^{\mathrm{source}}, \qquad
V^{\mathrm{pool}} = V^{\mathrm{source}}
\]

拷贝时剩余旋转 \(\Delta_{\mathrm{res}} = (t-s) - \Delta_{\mathrm{source}}\)。  
Admit 路径令 \(\Delta_{\mathrm{source}} = t-s\)，故 \(\Delta_{\mathrm{res}}=0\)，拷贝临界路径不再转一次。  
这是**速度优化**，不是 Accuracy 机制。\(V\) 永不旋转（RoPE 不作用于 V）。

Fail-closed（机械无效才 Dense，不是「\(\Delta\neq 0\) 就丢掉」）：

- token-hash 不匹配
- 覆盖不全
- alloc 失败

整岛丢弃，Dense 重算该 span；**从不部分拼接**。  
剩余非零 \(\Delta_{\mathrm{res}}\) **不是** admitted fast path。96092：421 次 source 预旋转、0 fallback。零 fallback 不是拆掉检查的理由。

目标计算路径：Dense 前缀 → 拷每个已 admit 岛 → Dense 剩余 → decode。

### 2.3 M1 / M3 存在但不进主表

- M1：radix \(\Delta=0\) 前缀，cap 在 shifted island 之前。
- M3：对**已经 admit 的** PREFIX/MIDDLE 做驻留提示；hint 来自 coding protocol 的 **later-roles / next-island**，不是 remaining-uses。DEVICE + 立刻下一个 use 没有重叠窗口，顺序 1-token 下不是 prefetch 赢面。Miss 必须退化到仍持有的 M2 拷贝，不得把有效岛改成 Dense。

---

## 3. 三条贡献如何被实验托住

| 贡献 | 论文声称 | 证据 | 明确不是 |
|---|---|---|---|
| 文件模块 admit | 资格 = token 相等 + 版本有效 + \(\Delta\neq 0\) | M0 编译器；48 个零位移被丢掉；LCS 对照说明「无 whitelist 的 copier」是另一条臂 | 不是 Attention admit；不是 AST |
| \(K\) 预旋转 + fail-closed | residual 0；机械失败 Dense | 96092：1684/1684 copy、0 fallback、421 预旋转；ledger 记事件，不靠延迟反推 | 不是 leftover rotate 当 fast path |
| 两套账 | cache-ready 1.375×；N=4 含一次 source-build 0.841× | 同一 705 pairs；N=1/2/8 由同一 frozen 派生 | cache-ready ≠ 一次 serving 墙钟 |

第三条是**会计贡献**：不把 source 材料化藏起来。N=1 是 0.389×（更慢），必须留在论文里。盈亏平衡在 4 次和 8 次消费之间（N=8 = 1.044×）。

---

## 4. 评测设计：测拷贝核，不测 resolved

### 4.1 速度战役怎么跑（作业 96092）

- 模型：Qwen3-Coder-30B-A3B-Instruct AWQ-4bit，SGLang
- 不重采样 agent：用同一 chat template + rolling-6 重建 **exact token IDs**
- decode **1 token**（TTFT = prefill）
- 每组 1 warmup + 3 measured → 235×3 = **705 pairs**
- reuse 臂每组材料化 source 一次；Dense 臂不材料化
- **cache-ready TTFT** = source 已在时的 target generate
- **N=4** = 一次 source-build + 四次后来使用（含 warmup 会计口径与主表一致：0.841×）
- 每 20 组重启 server（两边都重启，对比仍配对）
- prefetch off，ordinary prefix off，两边都关

### 4.2 主表数字（只许引用 96092）

`tab:eval-summary`：

| 指标 | Dense | ImpactKV |
|---|---|---|
| groups / islands / pairs | 235 / — / 705 | 235 / 421 / 705 |
| copy / fallback | — | **1684/1684，0** |
| source \(K\) 预旋转 | — | 421 |
| cache-ready | 1× | **1.375×** |
| 配对 TTFT 节省（中位） | — | 19.2% |
| 配对赢率 | — | 96.5% |
| N=4 + 一次 source-build | 1× | **0.841×** |
| 平均 source-build | — | 879 ms/组 |
| one-token 一致率 | — | **94.8%（不是 Accuracy）** |
| prefetch / prefix | off | off |

读法：

- **1.375×** 回答：「岛已经在的时候，target prefill 快多少？」
- **0.841×** 回答：「把这次材料化算进去、只用四次，还快吗？」答案是慢。部署前提是**同一份 source KV 被后续角色/重试再消费**。
- 机制闭环：计划拷了就拷了，0 fallback。

### 4.3 从同一 96092 派生的切片（不是新 GPU）

用来回答「快从哪来」，全部 caption 写 *not a new GPU arm / not tab:eval-summary*。

- 更长 prompt、更多岛、更高 copied-fraction → 更快（7K+ 达 2.00×；3 岛 1.685×；Q4 1.917×；Q1 仅 1.131×）
- \(|\Delta|\) **不单调**（大位移 \(\ge 3000\) 仍是 1.588×）→ 不是「位移小才快」的前缀故事
- unique source 115 个，其中 101 个出现在多个 group。若 unique source 只付一次材料化，N=4 反事实是 **1.175×**；96092 **不是**这么记账的，**不是 headline**
- repo 切片（附录 `tab:repo-slice`）：不排名；pydata 最弱 1.119×，django 最强 1.551×

### 4.4 质量：one-token ≠ resolved

94.8% 只是和 Dense 的 **第一个 decode token** 是否相同，字段标记 `not_accuracy`。

官方 SWE-bench resolved 仍是 live-agent、full-decode、hidden test。  
轨迹生产者 expanded24：Dense 3/24 vs policy 5/24，McNemar \(p=0.625\)，**无统计功效，不是 headline**。

---

## 5. 旁路战役：各答一个问题，全部不得进主表

### 5.1 Unconstrained LCS copy is **not** a comparison target

作业 132385 / 7B LCS 臂测过「最长精确公共串」贪心拷贝。它**不是**比较对象。

原因：没有文件边界和版本 whitelist，one-token 从 coding 的 94.8%（30B）/ 80.4%（7B）掉到 87.1% / 74.5%。精度损失太大，不能拿来和 ImpactKV 比速度。

Headline 比较仍是 **coding-aware file-module vs Dense**（96092 主表；7B 是 `tab:7b-swebench` 的 Dense vs Coding）。LCS 只作为被拒绝的做法写一句质量掉点，不进速度对照表。CacheBlend/KVCOMM 仍然没跑、也不欠。

### 5.2 Template prefetch（作业 119795）— `tab:template-prefetch`

相对**本战役自己的 Dense**（不是 96092）：

- coding (copy) 1.390×
- prefetch-only 0.996×（不是 copy 赢）
- combined 1.392× ≈ coding → **overhead，不是 prefetch 加速**

原因：顺序 1-token、岛已在 device 上，没有 H2D 重叠窗口。  
**Dual prefix+copy vs Dense 不是 prefetch 数字。** 表停在 119795。

### 5.3 7B dual-island（124825–124829）— `tab:7b-dual-island`

Qwen2.5-Coder-7B-Instruct，7 组 **重新 tokenize** 的双岛（禁止拿 30B ids 在 7B 词表上重放）。相对 **7B Dense**：

- prefix-only 1.155×（radix \(\Delta=0\)，shifted 岛 Dense）
- lossy-only 4.526×
- dual 8.490×

用途：在能装下双岛的小模型上，**拆开** M1 和 M2，证明两个算法各自相对 Dense 加速。  
**禁止**混进 30B 主表。  
该战役上 combined vs dual = 0.528×（7 hints、4 miss 后 Dense 重算）——论文承认这是模块 bug：M3 miss 必须退回 M2，而不是关掉 copy。30B prefetch 表仍用 119795，不把这次 bug 当 prefetch 结论。

### 5.4 明确没做、论文写明不欠的

- 500-task Verified 重放
- 并发 / P99 / session-completion / C=4
- 同 token 30B CacheBlend/KVCOMM 臂
- 7B/3B Accuracy 或 TV 混进 30B TTFT
- 把 1.375× 说成 e2e serving 或 SOTA
- 用 0 fallback 当借口拆 fail-closed
- 重跑或改 96092 数字

---

## 6. Related work 里的定位句

- **精确 KV：** 前缀系统在 \(\Delta=0\) 时是对的；它们不拷非零位移的文件岛。
- **Lossy / 跨 agent：** CacheBlend 等「拷更多，通常更快」；不是 coding-file whitelist；**不声称 30B raw TTFT 超过它们**。3B 八条 prompt 的局部 TV（0.00214 vs CacheBlend 0.0454 / KVCOMM 0.0745）只是探针，不是 30B 排名，也不是 matched-coverage Accuracy。
- **调度 / TTL：** 正交。Prefetch 是驻留旁路，不是 headline 拷贝核。
- **RoPE 相位校正：** 先前 code-aware serving 已有「字节相同、位置不同」的 \(K\) 校正。本文把它接到 rolling-6 **文件模块岛** + 两套 TTFT 账 + headline 关掉 prefetch。

---

## 7. 结论句（论文自己的诚实陈述）

> Coding-aware true-lossy file-module reuse 是长多文件 agent prompt 上的 **cache-ready 赢**；只有同一份 source KV 被消费不止一次时，才是 **systems 赢**。它不是并发 serving 结果。

展开成三条可检查的句子：

1. 前缀缓存 miss 移位文件；ImpactKV 只拷版本有效、token 相同、\(\Delta\neq 0\) 的文件岛。
2. 96092：1684/1684、0 fallback、cache-ready 1.375×、N=4 为 0.841×；prefetch 在该战役关闭。
3. one-token 不是 resolved；prefetch 顺序下不卖加速；LCS 是 copier baseline 不是新 headline。

---

## 8. 章节如何承载这条链

| 节 | 逻辑职责 |
|---|---|
| Intro | 记忆系统问题（资格 / 地址翻译 / 翻译失败）；三条贡献；headline 数字；3B probe 降权 |
| Background | SWE-bench 只是 substrate；RoPE 为何让前缀失效；3B 损失位置；为何 generic copier 是错的 default |
| Problem | 系统模型；三条 design goals；什么会**作废**一个声称；M0–M3 I/O |
| Template | admit 四条；算法；235/421；template 不许估 Attention |
| KV management | 岛对象；预旋转公式；fail-closed；主表关掉的旋钮 |
| Implementation | SGLang pool + manifest；exact-prompt 重放协议；30B ids 不得在 7B 上重放 |
| Evaluation | 主表 → 形状 → N-use → 切片 → 非 Accuracy → LCS → 不声称 → prefetch → 7B |
| Related | 前缀 / lossy / 调度 / RoPE 四档 |
| Discussion | 1-token 是对的拷贝核实验、错的 resolved 实验；威胁；不欠清单 |
| Appendix | 24 个 instance 名单；repo 切片不排名 |

「什么会作废声称」（problem 节，审查时应对着看）：

- 把 7B 短文本 Accuracy 和 30B 长 prompt TTFT 合成一个方法
- 把 1.375× 叫成一次 serving 墙钟赢
- 把 94.8% 叫成仓库级 Accuracy
- 打开 prefetch/radix 却把增益算给 coding-aware copy
- 看见结果后再改 residual-\(\Delta\) 门

---

## 9. 请你拍板的逻辑风险

1. **Unconstrained LCS 不是比较对象。**  
   精度损失（30B 94.8%→87.1%；7B 80.4%→74.5%）是拒绝它的理由。不要把它放进速度对照表。COMPLETE 事实留在制品里，正文只写质量掉点和「not a comparison target」。

2. **没有 30B CacheBlend**  
   论文已写不欠。LCS 只替代「naive exact copier」，不能假装替代 CacheBlend。related work 里 CacheBlend TV 仍是 3B 八条 prompt。

3. **7B prefetch miss → Dense**  
   正文承认是 bug。审稿人可能问 30B 上是否已修。30B 119795 的 miss 是 0，但那是「没发 hint」而不一定是「miss 路径已修」。

4. **24-task convenience sample**  
   已经写明。主声称是拷贝核，不是 Verified 排行。不要在 intro 里把 SWE-bench 当 Accuracy 舞台（题目也禁止出现 SWE-bench）。

5. **N=4 = 0.841×**  
   这是诚实，也是 serving 论文的常规杀伤。全文必须反复：cache-ready ≠ deployed one-use。盈亏靠多次消费，论文**没有**并发实验证明多次消费在线上发生的速率。

---

## 10. 数字来源（防混表）

| 标签 | 作业 | 能否进主表 |
|---|---|---|
| `tab:eval-summary` | 96092 COMPLETE | **唯一 headline** |
| `tab:nuse` / 长度 / 岛数 / \(\Delta\) / fraction / repo | 由 96092 派生 | 否 |
| `tab:attn-proxy` | 3B COMPLETE 探针 | 否 |
| `tab:template-prefetch` | 119795 COMPLETE | 否 |
| `tab:7b-dual-island` | 124825–124829 COMPLETE | 否 |
| LCS 132385 / 7B LCS 臂 | COMPLETE 但**不是比较对象** | 否（质量掉点太大，不进速度对照表） |

检查脚本：`paper_swebench_ucm/scripts/check_asplos_claims.py`（必须在该目录跑）。当前应为 PASS，且主表数字锁定 1.375 / 0.841 / 1684/1684。
