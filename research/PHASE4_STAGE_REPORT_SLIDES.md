## 1. 主要问题：为什么要做跨上下文 KV 恢复

- Architect、Coder、Debugger 会反复读取同一段代码，但 role、header 和 causal context 一直在变。
- Exact prefix cache 无法直接复用这类“内容相同、前置上下文不同”的 body KV，通常只能重新做 dense prefill。
- body 越长，重复计算越贵；working set 超出 cache 后，真实 eviction 还会继续推高 TTFT。
- 我们想回答的是：能否用受控 KV 恢复替代一部分 dense prefill，以及每条路径真正需要付出多少额外成本。
- 这里只看请求能否跑通和 TTFT，不讨论 accuracy 或输出等价。

---

## 2. 我们复现的 research

| 路径 | 核心机制 |
| --- | --- |
| R0 Raw+RoPE | 复制 body K/V，只做 RoPE 位置修正 |
| R1 EPIC | 逐层重算 leading-k tokens，其余 body 继续复用 |
| R2 CacheBlend | 按 KV deviation 选择少量 tokens 修复 |
| R3 Cache-Craft | 根据 context/profile 选择 direct、partial 或 full recompute；本轮 deferred |
| R4 KVCOMM | 用 canonical base、context delta 和 multi-anchor interpolation 重建目标 KV |

---

## 3. 结果总览

| 路径 | 代表性结果 / 状态 |
| --- | --- |
| R0 Raw+RoPE | 长 body 约 `2.07x`，是 speed-only 上限 |
| R1 EPIC | 长 body 约 `1.98x`，有修复路径中最好 |
| R2 CacheBlend | `repair ratio = 1%`，即按 KV deviation 修复约 1% 的 body tokens；1% 不是 speedup |
| R4 KVCOMM | 长 body target-only 约 `1.76x`；约 6 次 reuse 后摊销 setup |
| R3 Cache-Craft | Deferred |

- 最明显的分界是 body 长度：crossover 位于 **768 与 1024 tokens 之间**，只测小 body 会得出错误结论。

---

## 4. 当前结果的比较口径

- body 覆盖 512–2048 tokens，header 覆盖 0–256 tokens。
- `header` 指目标请求中位于 body 之前、可以 exact match 的 prefix/context 长度，不是 attention head 数量，也不是 head dimension。
- CacheBlend 的 `repair ratio = 1%` 指修复约 1% 的 body tokens：body1024 约 10 tokens，body2048 约 20 tokens。
- working-set pressure 从无压力增加到约 3x；高压点都在真实 eviction 下验证。
- Target-only TTFT 只看目标请求本身。有 setup 或 preparation 时，还要看 combined cost 或 break-even reuse 次数。
- 这些只是初步性能结果，不代表 production-ready，也不包含 accuracy 验证。

---

## 5. 跨路径观察

- 最重要的变量是 body 长度。小 body 会放大固定恢复开销，让 repair 看起来没有收益。
- 真实 eviction 没有吃掉长 body 的收益；压力升高后，EPIC 和 KVCOMM 的 speedup 基本稳定，header 增大时相对收益也略有改善。
- repair budget 并非越大越好。CacheBlend 的 1% 比 5% / 15% / 30% 更快。
- 有额外成本的路径不能只看 target-only：CacheBlend 要看 combined，KVCOMM 要看 break-even。
- 纯速度上限由 R0 给出；需要 repair 时，R1 EPIC 的结果最好。

---

## 6. R0 Raw+RoPE 与 R1 EPIC

| 路径 | body1024 | body2048 | 定位 |
| --- | ---: | ---: | --- |
| R0 Raw+RoPE / k0 | `1.73x` | `2.07x` | 纯速度上限，不做 context repair |
| R1 EPIC k32 | `1.53x` | `1.98x` | 有修复路径中最好 |

- EPIC 不需要单独的 dense preparation，也没有一次性 setup。
- body1024 的 EPIC k32 在不同 pressure 下保持约 `1.5x`。
- body 变长后，逐层 repair 的开销被摊薄，同时保留了接近 raw reuse 的收益。

---

## 7. R2 CacheBlend 与 R4 KVCOMM

| 路径 | body1024 | body2048 | 成本边界 |
| --- | --- | --- | --- |
| CacheBlend（约修复 1% body tokens） | target-only `1.64x`；combined `0.82x` | target-only `2.02x`；combined `1.14x` | 依赖 fresh preparation；repair ratio 越高越慢 |
| KVCOMM | target-only `1.37x` | target-only `1.76x` | setup 约 1.1s / 2.2s |

- body2048 是 CacheBlend 首次出现 single-use combined 正收益的点，fresh preparation 仍是主要限制。
- KVCOMM 的 break-even 估算：body1024 约 14 次 reuse，body2048 约 6 次 reuse。
- CacheBlend 更接近 single-use repair；KVCOMM 更适合同一内容的高频跨上下文复用。

---

## 8. Next

- 把较优的恢复路径与 HiCache（High Cache）结合，验证分层缓存、eviction 和 load-back 能否协同。
- 在更大硬件上运行，重点补充 `RTX 6000` 结果。
- 继续扩展长 context，完成各路径的统一对比。
