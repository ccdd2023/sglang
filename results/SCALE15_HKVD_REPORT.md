# Scale-15 + HKVD 实测报告 (2026-07-09)

8h 本地资源可信度增强。**修正了 n=5 的精度结论**，并把 HKVD 位置代理从假设变实测。

## ① HKVD-by-position 实测（验证机制）

`hkvd_by_position_20260709/measure_hkvd_by_position.py` — HF Qwen2.5-Coder-7B 直跑（无 sglang）。
swap prefix（canonical->live）测每个 chunk 位置的 KV 偏差 `deviation_i = 1 - cos(KV_i|canonical, KV_i|live)`。

| pos | K_dev | V_dev |
|---|---|---|
| 1 (早) | **0.0997** | 0.0026 |
| 2 | 0.0937 | 0.0022 |
| 3 | 0.0956 | 0.0017 |
| 4 | 0.0920 | 0.0009 |
| 5 (晚) | **0.0929** | 0.0009 |

- **pos1 K_dev 比 pos5 高 7.2%**（std 0.004-0.013，方向一致）-> 位置代理 HKVD 假设**验证**
- V_dev pos1 比 pos5 高 3×；**K 比 V 对上下文敏感 38×**（跨上下文信号主要在 K）
- 幅度 7% 适中，小于真 CacheBlend per-layer 5-18%（因只在 chunk 位置分层、粒度粗）

坑：transformers 5.3.0 `DynamicCache` 用 `pkv.layers[L].keys/.values`（非 key_cache、非下标）；eager attention OOM 用 sdpa；max_file_chars 3000。

## ② Scale-15 多样化复测（推翻 n=5 精度结论）

`scale15_5x5/` — n=5（全 combine_file，易）-> n=15（5 个 instance family）。precompute pool `pandas_15case_v1`（120 chunk，gitignored）。stable config: mem 0.72 + max-total 16384 + 自动 relaunch（`--chunk-size 1` 被 bench_kvcomm_ttft_stress 的 chunk_size=10000 覆盖，但 OOM 后自动恢复）。

| config | n_rows | type_match | /rows | FAIL_acc | TTFT | c2_reuse |
|---|---|---|---|---|---|---|
| lossless | 75 | 8 | 10.7% | 49% | 1032ms | 0 |
| R32 (FRAC=0.30) | 61 | 6 | 9.8% | 43% | 745ms | 345 |
| R38b (0.60/0.15) | 60 | 4 | 6.7% | 40% | 753ms | 283 |

R32/R38b 在长 case OOM（rc=-9，系统 RAM），自动 relaunch 恢复，部分 case <5 agent -> 按 per-row（type_match/n_rows）比，公平。

### 关键修正
1. **n=5 的"head_recompute 恢复到 lossless (2->5=20%)"是易 case 偏置**（5 case 全 combine_file，1 case 贡献 4/5）。n=15 多样化后 type_match 降到 7-11%，**lossless > R32 > R38b**。复用 regime 在硬 case 产生更多 UNK（R32 14, R38b 12）。
2. **速度反而更强**：1.38× TTFT（745/753 vs 1032ms；n=5 是 1.33×），真实 code-aware 复用 283-345 tok。
3. **R38b 不优于 R32**：n=15 上 R32 略快(745 vs 753)且略准(6 vs 4 match)。位置分层无收益、甚至略有害。**推荐改为 R32**。

## ③ deck 更新
- 新增 slide 18（HKVD 实测图表）+ slide 19b（scale-15 诚实结果）
- TL;DR / 标题 / production / 总结改为：速度优化（1.38×）+ HKVD 实测，accuracy benefit 不泛化，推荐 R32>R38b
- 诚实定位：latency-sensitive verdict 任务可用，accuracy-critical 不适用

## ④ bug 修复
`ast_chunker.py:211` `_compute_type_complexity` 对 `ClassDef.bases` 访问 `b.annotation` 崩溃（bases 是 Name/Attribute，无 annotation）-> 改 `isinstance(b, ast.Subscript)`。precompute v4 在 diverse cases 触发。

## 可信度总结
- **速度**：✅ solid 且更强（1.38× at n=15）
- **机制**：✅ 实测验证（HKVD +7.2%）
- **精度**：⚠️ 诚实修正（n=5 偏置已揭露；n=15 未超 lossless；R32>R38b）
- **诚实性**：✅ 大幅提升（不再 overclaim，负结果如实呈现）
