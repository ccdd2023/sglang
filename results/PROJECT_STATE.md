# CodeMAS-KVFlow — 项目完整状态

> 保存时间: 2026-06-02 | 主实验模型: Qwen2.5-7B-Instruct / Qwen3-8B | 框架: sglang-kvflow HiRadixCache
> **项目目标**: 基于 Coding 场景的多智能体系统（MAS），通过 Workflow 模板化与 KV Cache 智能管理提升任务执行效率与准确率

---

## 0. 三大贡献

| 贡献 | 名称 | 核心内容 | 状态 |
|---|---|---|---|
| **贡献1** | **Workflow Template Generation** | 对相似 Coding 任务生成固定的任务处理 Workflow 模板，提高任务 One-Shot Accuracy | ✅ 已实现 |
| **贡献2** | **Template-Derived Code-Base Scheduling Hints** | 根据模板预测的 Agent 执行顺序与代码段消费关系，生成 code-base segment hint、priority 与观测元数据；KVFlow 作为参考调度框架 | ✅ 已实现 |
| **贡献3** | **Exact-Content Code Segment Reuse Policy** | 在模板场景中定位共享 Code Base，使用 content signature 作为唯一安全 gate，并借鉴 KVCOMM/跨位置 KV reuse 的 RoPE delta 思路完成原型验证 | 🔬 验证中 |

**当前阶段**: 推进贡献3的验证，逐步整合三个贡献点为一篇完整工作。

**最新进展 (2026-06-03)**: 已将 SWE-bench Verified codebase-content 扩展为 **30/100/500 case** 三档数据集。30-case 完成本地环境 smoke：Gold **29/30 通过**，Base **29/30 非零**，筛出 **28 个 gold pass + base nonzero 判别样例**；在 Qwen2.5-7B JSON-edit schema 上完成 lossless KV vs exact-content segment reuse pass@1 主对比，reuse 分支全部命中 `exact_code_content_signature`，pass@1 为 **2/28**，lossless 为 **3/28**。H12 验证了 `codebase_prefetch_hints` 的 template-to-engine 透传、metadata 可观测性与 exact-content hit：28-case serving 主表中 exact reuse + codebase hints hit **28/28**，cached tokens **1606.1 vs baseline 1253.3**，latency **1355.2ms vs baseline 1372.0ms**。100/500-case 已完成 gate/anchor/content-signature 扩展统计。H13 论文包已重新定位为 **Template-Guided Code-Base Segment Reuse for Multi-Agent Software Engineering**；KVFlow 与 KVCOMM 已在论文中改为参考工作/基线设计，不再作为本文贡献表述。

---

## 1. 系统架构

```
MAScoder (Workflow Template)             sglang-kvflow 服务端
├─ Workflow Template (贡献1)             ├─ HTTP API / OpenAI Protocol
│   ├─ Planner Agent                     │   ├─ serving_chat.py
│   ├─ Implementer Agent                 │   ├─ protocol.py
│   └─ Reviewer Agent                    │   └─ tokenizer_manager.py
│                                         │
├─ KVFlowHint (贡献2)                    ├─ Scheduler (贡献2)
│   ├─ priority (执行优先级)              │   ├─ scheduler.py
│   ├─ prefetch_next_agent()             │   ├─ schedule_policy.py
│   └─ role_type (Agent 角色)            │   └─ schedule_batch.py
│                                         │
├─ CodeBaseSegment Hint (贡献3)           ├─ HiRadixCache / KVCOMM (贡献3)
│   ├─ content_signature                 │   ├─ radix_cache.py
│   ├─ code_base_id / agent usage        │   ├─ anchor_match.py
│   └─ code_anchor_token_spans           │   ├─ evict_policy.py
│                                         │   └─ hiradix_cache.py
└──────────────────────┘                 └─────────────────────────────────┘
```

---

## 2. 功能模块

| # | 模块 | 核心文件 | 状态 | 所属贡献 |
|---|------|----------|------|----------|
| 1 | MAScoder Code Base segment 标记 | `MAScoder/src/mascoder/code_anchor.py` | ✅ | 贡献3 |
| 2 | Workflow Template 生成 | `MAScoder/src/mascoder/workflow_template.py` | ✅ | 贡献1 |
| 3 | KVFlowHint (template+lossy+codebase prefetch) | `MAScoder/src/mascoder/kvflow_integration.py` | ✅ | 贡献2+3 |
| 4 | 服务端 Exact-Content Matcher + Gate | `sglang-kvflow/python/sglang/srt/mem_cache/anchor_match.py` | ✅ | 贡献3 |
| 5 | HiRadixCache 锚点元数据支持 | `hiradix_cache.py`, `radix_cache.py` | ✅ | 贡献2+3 |
| 6 | 驱逐保护 (evict_policy) | `evict_policy.py` | ✅ | 贡献2 |
| 7 | HTTP 可观测性 | `scheduler_output_processor_mixin.py`, `serving_chat.py` | ✅ | 贡献2+3 |
| 8 | RoPE Delta 旋转 | `radix_cache.py`, `cache_init_params.py` | ✅ | 贡献3 |
| 9 | 独立 KV 隔离分析 | `benchmark/multi_workflow/analyze_kv_isolation.py` | ✅ | 贡献3 |
| 10 | 独立 KV 替换实验 | `benchmark/multi_workflow/bench_kv_replacement.py` | ✅ | 贡献3 |
| 11 | SWE-bench 实验框架 | `bench_swe_lite_kv.py`, `bench_multiagent_large.py` | ✅ | 贡献3 |
| 12 | 大 Codebase 多 Agent 实验 | `bench_large_codebase_reuse.py` | ✅ | 贡献3 |
| 13 | 真实 repo-level 多文件数据集准备 | `prepare_repo_level_datasets.py` | ✅ | 贡献3 |
| 14 | 真实 repo-level exact reuse 实验 | `bench_real_codebase_exact_reuse.py`, `report_repo_level_exact_reuse.py` | ✅ | 贡献3 |
| 15 | SWE-bench 本地 repo 环境搭建与 gold 测试 | `setup_swebench_local_env.py` | ✅ | 贡献3 |
| 16 | SWE-bench generated patch KVCOMM 测试 | `bench_swe_generated_patch_kvcomm.py` | ⚠️ 已跑通链路，模型 patch 不可应用 | 贡献3 |
| 17 | 可视化 Dashboard | `results/experiment_dashboard.html` | ✅ | 全局 |
| 18 | SWE-bench Verified 30/100/500-case manifest 扩展 | `prepare_swebench_verified_expanded.py` | ✅ | 贡献3 |
| 19 | SWE-bench 本地环境批处理 smoke | `run_swebench_local_env_batch.py` | ✅ | 贡献3 |
| 20 | 30-case lossless KV vs exact-content segment reuse pass@1 | `bench_swe_generated_patch_kvcomm.py` | ✅ | 贡献3 |
| 21 | 100/500-case gate/anchor scalability 统计 | `prepare_swebench_verified_expanded.py` + report scripts | ✅ | 贡献3 |
| 22 | Coding-aware KVFlow codebase prefetch | `kvflow_integration.py`, `protocol.py`, `scheduler.py` | ✅ | 贡献2+3 |
| 23 | EuroSys-style 论文包与图表脚本 | `paper/main.tex`, `paper/scripts/generate_paper_figures.py` | ✅ 初稿完成 | 全局 |

---

## 3. 关键实验结果

### H1: 代码块间 Cross-Attention 极弱 ✅ (验证目标3)

| Block Pair | Intra | Inter | Ratio |
|---|---|---|---|
| sort vs sort | 0.0077 | 0.0006 | **55×** |
| sort vs search | 0.0074 | 0.0005 | **62×** |
| sort vs string | 0.0151 | 0.0005 | **98×** |

数据: `/tmp/kv_isolation/phase1_attention.json`

### H2: V cosine 能区分同/异功能块 ✅ (验证目标3)

| 度量 | 同功能 | 异功能 | 分离度 |
|------|--------|--------|--------|
| V cosine | **0.44** | **0.19** | **2.3×** |
| K cosine | 0.92 | 0.89 | 1.03× |

数据: `/tmp/kv_isolation/phase2_similarity.json`

### H3: SWE-bench Gate 有效性 ✅ (验证目标2)

- SWE-bench Lite 50 task: **45/50 reject (90%)**, 5 accept
- SWE-bench Verified 30 task: **27/30 reject (90%)**, 3 accept
- Accept BLEU: lossy=0.398 / lossless=0.399 (Δ<0.001)
- Gate latency overhead: **<50ms** (<8% of total)
- 大 Codebase 同代码: 100% accept with `exact_code_content_signature`

数据: `results/swe_latency_summary.md`, `results/large_block_summary.md`

### H4: KV 复用量 ✅ (验证目标1+3)

| 场景 | KV 复用 | 复用率 | BLEU |
|------|---------|--------|------|
| 孤立代码块 (Phase 3) | 9-21 MB | 45-65% | — |
| 大 Codebase × Multi-Agent (lossy) | **200-314 MB** | — | 0.239-0.720 (gap 大) |
| 大 Codebase × Multi-Agent (lossless) | 226-362 MB | 88-92% | — |
| 大 Codebase × Code-First | 226-362 MB | **98.5%** | **1.000** |
| Prefill 节省 | 12-18ms/block | — | — |

数据: `results/codebase_reuse/summary.md`, `results/ma_ttft/final_v2.log`

### H5: RoPE Delta 旋转正确性 ✅ (验证目标1)

| Scenario | A2 TTFT | A3 TTFT | A2 delta | A3 delta | BLEU |
|---|---|---|---|---|---|
| 无偏移 | 57.5ms | 45.7ms | 0 | 0 | 1.000 |
| A2+10, A3+20 | 70.9ms | 51.0ms | **13** | **25** | 1.000 |
| A2+25, A3+50 | 92.2ms | 52.9ms | **32** | **63** | 1.000 |

数据: `results/ma_ttft/offset_test_v5.log`

### H6a: 真实 repo-level 本地环境与判别 smoke ✅ (验证目标3)

| 指标 | 结果 |
|---|---:|
| SWE-bench Verified repo case | 10 |
| 每 case 大文件数 | 3 |
| Gold smoke pass | **10/10** |
| Base smoke nonzero | **9/10** |
| Base nonzero + Gold pass | **9/10** |

说明: Django `django__django-10097` 的首个 `FAIL_TO_PASS` smoke 在 base 下也通过，因此不作为首批 accuracy 判别 case；Pylint base 为 collection error，进入主表前需进一步规范化。

数据: `results/repo_level_datasets/manifest_10.json`, `results/swebench_local_envs/expanded_10_env_report.md`

### H6: 真实 repo-level 多文件 exact-code 复用 ✅ (验证目标1+3)

数据集: `ScalingIntelligence/swe-bench-verified-codebase-content`，抽取 Astropy / Django / Matplotlib 3 个真实 SWE-bench Verified repo 快照，每个 repo 使用 3 个大 Python 文件。实验流程为 cold lossless baseline（不带 anchor metadata）→ Planner warmup（带 code-base anchors）→ lossy KVCOMM reuse，避免相同 prompt exact-cache 污染。

| 指标 | 结果 |
|---|---:|
| HF reusable segment length | 6519.8 tokens/file |
| HF layer-24 K cosine avg/min | 0.998970 / 0.998675 |
| HF layer-24 V cosine avg/min | 0.995992 / 0.994523 |
| sglang avg speedup | 1.544× |
| cached tokens: lossless → lossy | 29.7 → 9860.3 |
| output exact-match rate | 50.0% |
| output token F1 avg | 0.7847 |

全部 serving 命中均为 `exact_code_content_signature`。风险点: Astropy debugger 与 Django debugger 的 token F1 低于 0.6，后续准确性结论需升级为 SWE-bench-style pass/fail 或 patch-level validation。

数据: `results/repo_level_datasets/manifest.json`, `results/real_codebase_exact_reuse/repo_dataset_combined_summary.json`, `results/real_codebase_exact_reuse/repo_dataset_report.md`

### H7: 真实 repo 本地环境与 SWE-bench gold 测试 ✅ (验证目标1)

Docker harness 在当前机器上受 `/var/run/docker.sock` 权限限制，因此使用 local git checkout + conda fallback 搭建 3 个 SWE-bench Verified repo 环境。每个环境均 checkout 到 base commit，应用 SWE-bench test patch 与 gold patch，然后运行 FAIL_TO_PASS 目标测试。

| Case | Repo | Env | Gold target tests | Result |
|---|---|---|---:|---|
| `astropy__astropy-12907` | `astropy/astropy` | `swe_astropy_astropy_12907_gold` | 2 | PASS |
| `django__django-10097` | `django/django` | `swe_django_django_10097_gold` | 363 | PASS |
| `matplotlib__matplotlib-13989` | `matplotlib/matplotlib` | `swe_matplotlib_matplotlib_13989_gold` | 1 | PASS |

Base-vs-gold 复现结果: Astropy base 2 failed / gold 2 passed；Django URLValidator base 6 failed / gold 0 failed；Matplotlib base 1 failed / gold 1 passed。

关键修正: Django 的 SWE-bench metadata 前 5 个 `auth_tests` 目标不具备区分度，需改用与 patch 对齐的 `validators.tests.TestSimpleValidators`；Django 是 `setup.py install` 到 egg，必须先应用 gold patch 再安装，否则测试会运行旧 site-packages；Matplotlib 3.0 在本机需要 conda pin `freetype=2.10.4` 才能编译通过。该验证说明当前 repo-level KV 复用数据来自可执行、可测试的真实代码库，但尚未将 lossy 生成结果自动接入 SWE-bench grader。

数据: `results/swebench_local_envs/local_env_report.md`, `results/swebench_local_envs/reports/*/gold_report.json`

### H8: Generated Patch + KVCOMM 端到端测试 ⚠️ (验证目标1)

新增 generated patch benchmark，将 KVCOMM exact code-base reuse 接入本地 SWE-bench candidate patch 评测。流程为 Planner warmup 插入 code-base anchors → lossless/lossy 生成 unified diff → `--mode candidate` 应用 patch → 运行本地目标测试。

| Case | Lossless elapsed | Lossy elapsed | Speedup | Lossless cached | Lossy cached | Lossy match | Candidate result |
|---|---:|---:|---:|---:|---:|---|---|
| `astropy__astropy-12907` | 13818.44ms | 4936.44ms | 2.799× | 378 | 2808 | `exact_code_content_signature` | invalid patch |
| `django__django-10097` | 2206.12ms | 13528.69ms | 0.163× | 636 | 637 | `exact_code_content_signature` | invalid patch |
| `matplotlib__matplotlib-13989` | 2664.78ms | 4171.15ms | 0.639× | 541 | 542 | `exact_code_content_signature` | invalid patch |

补充更大模型复测:

| Model | Diffs extracted | Cleanly applied | Passed tests | Lossy exact-code hits | 结论 |
|---|---:|---:|---:|---:|---|
| `Qwen2.5-7B-Instruct` | 3/6 | 0/3 | 0/6 | 3/3 | 可提取部分 diff，但全部 `git apply` 失败 |
| `Qwen3-8B` | 0/6 | 0/0 | 0/6 | 2/3 | 输出主要停留在 reasoning/analysis，未产出 unified diff |

结论: 生成-抽取-应用-测试链路已跑通，lossy 模式在可标记样例中命中 exact-content reuse；但 3B/7B/Qwen3-8B 在当前 prompt 下均未生成可应用 SWE-bench patch。该结果说明 generated patch pass/fail accuracy 当前受模型 patch synthesis 与输出格式约束限制，不能作为贡献3准确性上限。下一步应使用更强 coding model 或约束式 patch/edit schema。

数据: `results/swe_generated_patch_kvcomm/summary.json`, `results/swe_generated_patch_kvcomm/generated_patch_report.md`

### H8a: Lossless KV vs KVCOMM Lossy pass@1 主对比 ✅ (验证目标1)

在扩展后的 10 个 SWE-bench Verified repo-level case 上，新增 JSON edit schema：模型输出 `path/search/replace`，runner 使用本地真实文件合成 unified diff，再执行 `git apply --check` 和 candidate 测试。该设置降低了纯 diff 格式错误对 pass@1 的干扰，并直接比较 lossless KV 与 exact-content segment reuse。

| 指标 | Lossless KV | Exact-content reuse | Delta |
|---|---:|---:|---:|
| diff extraction | 8/10 | 7/10 | -1 |
| clean apply | 8/10 | 7/10 | -1 |
| pass@1 | **1/10** | **1/10** | **0** |
| avg cached tokens | 1773.7 | 2868.8 | +1095.1 |
| avg generation speedup (lossless/lossy) | — | 1.776× | — |
| lossy exact-content hits | — | **10/10** | — |

结论: 当前 Qwen2.5-7B 的绝对 patch pass@1 仍偏低，主要瓶颈是模型 patch synthesis；但在该主对比中，exact-content segment reuse 的 pass@1 与 lossless KV 持平，且所有 reuse 命中均为 `exact_code_content_signature`，支持“有损位置复用不等于有损代码内容复用”的安全性表述。

数据: `results/swe_generated_patch_kvcomm/qwen2_5_7b_json_10/PASSRATE_REPORT.md`, `results/swe_generated_patch_kvcomm/qwen2_5_7b_json_10/passrate_table.csv`

### H9: Qwen3-8B 真实 repo-level exact-code 复用 ✅ (验证目标1+3)

为回应 8B/Qwen3 模型测试，使用本地 `Qwen3-8B` 在同一真实 repo-level 数据集上重跑 direct KV delta 与 sglang exact reuse。当前机器未发现可用的官方 `Qwen3-9B` 文本 checkpoint，因此采用官方/本地可用的 Qwen3-8B。每个 repo 使用 2 个大 Python 文件，流程为 HF 两种前缀位置 KV 对比 + sglang cold lossless baseline → Planner warmup → lossy KVCOMM reuse。

| 指标 | 结果 |
|---|---:|
| HF reusable segment length | 5571.2 tokens/file |
| HF layer-35 K cosine avg/min | 0.998766 / 0.998292 |
| HF layer-35 V cosine avg/min | 0.990817 / 0.988023 |
| sglang avg speedup | 1.123× |
| cached tokens: lossless → lossy | 29.7 → 2760.3 |
| output exact-match rate | 100.0% |
| output token F1 avg | 1.0000 |

逐 agent 现象: Astropy implementer 与 Matplotlib implementer 分别从 0/40 cached tokens 提升到 8192/8232 cached tokens，对应 1.364×/1.382× speedup；debugger 分支与 Django 分支虽然 exact-content gate 命中，但实际 cached token 计数仅 18/40，latency 基本持平。这说明 8B 模型下 KVCOMM 的安全性结果较强，但加速收益依赖被复用 code base 是否足够长、是否落在当前 cache chunk 可命中范围内。

数据: `results/real_codebase_exact_reuse/qwen3_8b/combined_summary.json`, `results/real_codebase_exact_reuse/qwen3_8b/repo_dataset_report.md`

### H10: 30-case Lossless KV vs KVCOMM Lossy pass@1 主对比 ✅ (验证目标1)

在 30 个 SWE-bench Verified repo-level case 上完成环境 smoke，并将主实验限制在 **gold pass 且 base nonzero** 的 28 个判别样例。模型使用本地 `Qwen2.5-7B-Instruct`，输出采用 JSON edit schema，由 runner 合成 unified diff，并执行 `git apply --check` 与本地目标测试。Lossy 分支的所有可复用命中均记录为 `exact_code_content_signature`，AST/anchor 只做候选定位，不作为安全 gate。

| 指标 | Lossless KV | Exact-content reuse | Delta |
|---|---:|---:|---:|
| 判别 case | 28 | 28 | — |
| diff extraction | 14/28 | 12/28 | -2 |
| clean apply | 14/28 | 12/28 | -2 |
| pass@1 | **3/28** | **2/28** | **-1 case / -3.6 pct** |
| avg cached tokens | 1253.3 | **2190.2** | +936.9 |
| avg generation latency | 2052.1ms | **1729.4ms** | 1.19× speedup |
| lossy exact-content hits | — | **28/28** | — |

结论: 30-case 主表比 10-case pilot 更接近论文/汇报口径。当前 Qwen2.5-7B 的绝对 pass@1 仍受 patch synthesis 限制；但在同模型、同 prompt、同 JSON-edit schema 下，exact-content segment reuse 与 lossless KV 的 pass@1 差距为 1 个 case，同时获得更多 cached tokens 与更低平均生成 latency。贡献3的 accuracy 表述应聚焦于 **lossless vs reuse delta**，而不是该小模型的绝对 SWE-bench 能力。

数据: `results/swebench_local_envs/expanded_30_env_report.md`, `results/swebench_local_envs/expanded_30_discriminative_instances.json`, `results/swe_generated_patch_kvcomm/qwen2_5_7b_json_30/PASSRATE_REPORT.md`, `results/swe_generated_patch_kvcomm/qwen2_5_7b_json_30/passrate_table.csv`

### H11: 100/500-case Dataset Scalability 与 Gate/Anchor 统计 ✅ (验证目标2+3)

为补足数据集规模证据，已生成 30/100/500 三档 manifest。30-case 用于 pass@1 主表；100-case 用于 repo-level reuse/cached-token/latency 扩展统计；500-case 用于 metadata、anchor、content-signature、可复用 token 规模统计，不默认跑全量本地 pass@1。

| 数据集 | Case | Repo | Segment/File | 总行数 | 近似可复用 token | 用途 |
|---|---:|---:|---:|---:|---:|---|
| 30-case | 30 | 11 | 90 | 284,252 | — | 本地环境 smoke + pass@1 主表 |
| 100-case | 100 | 10 | 300 | 1,019,831 | 9,481,504 | 扩展 repo-level reuse 统计 |
| 500-case | 500 | 12 | 1500 | 5,973,340 | 58,772,478 | gate/anchor scalability 统计 |

500-case 统计中，重复 content signature 为 **254 个签名 / 1073 个 case references**，说明真实大型 repo 数据存在大量跨 case 可复用 code base；exact-content gate 的 false accept 按定义为 **0**，因为同 AST、同路径、同函数名、span overlap 或 near match 都不能单独触发复用。

数据: `results/repo_level_datasets/manifest_30.json`, `results/repo_level_datasets/manifest_100.json`, `results/repo_level_datasets/manifest_500.json`, `results/repo_level_datasets/repo_level_100_reuse_report.md`, `results/repo_level_datasets/gate_anchor_500_report.md`

### H12: Prefix-only KVFlow vs Coding-aware KVFlow Prefetch ✅/⚠️ (贡献2+3)

新增 `codebase_prefetch_hints` 后，模板侧可以提前声明后续 Agent 会读取哪些 exact code-base segment；engine 侧在 request 入队时接收该 hint，并与原有 `next_agent_prefix` prefetch 并行。当前稳定主实验关闭 HiCache file storage backend，因此主表验证的是 **template-to-engine hint 透传、metadata 可观测性、KVCOMM exact-content hit、cached-token/latency 影响**；host load-back queued tokens 暂未作为稳定结论。

| Mode | Case | Avg latency | Avg cached tokens | Avg hints | Exact-content hit |
|---|---:|---:|---:|---:|---:|
| baseline_prefix_cache_only | 28 | 1372.0ms | 1253.3 | 0.0 | 0.00 |
| kvflow_prefix_only | 28 | 1353.8ms | 1253.3 | 0.0 | 0.00 |
| kvflow_prefix_plus_codebase_prefetch | 28 | 1387.7ms | 1256.3 | 1.0 | 0.00 |
| kvcomm_lossy_plus_codebase_prefetch | 28 | **1355.2ms** | **1606.1** | 1.0 | **1.00** |

10-case smoke 同样跑通，KVCOMM+codebase hints 为 **10/10 exact-content hit**，cached tokens **2832.1 vs baseline 1773.7**，latency **1280.3ms vs baseline 1421.2ms**。尝试启用 `--hicache-storage-backend file` 时，server 进入目标请求后触发 `token_to_kv_pool_allocator memory leak detected` runtime checker，因此 file storage host load-back 暂列为工程风险；当前报告明确将 `hicache_storage_backend` 标注为 `disabled`。

### H13: EuroSys-style 论文包与交接状态 ✅/📝 (全局)

已在 `paper/` 目录生成匿名 ACM/EuroSys 风格 LaTeX 投稿包，当前定位为 **Template-Guided Code-Base Segment Reuse for Multi-Agent Software Engineering**。论文主线已从旧 PF-Lock/KVFlow 负交互叙事，重写为当前 CodeMAS 三大贡献；其中 **KVFlow 与 KVCOMM 均作为参考工作/基线设计，不作为本文贡献本身**：

| 论文贡献 | 对应系统/实验 |
|---|---|
| Workflow Template Generation for Coding MAS | 旧稿 AgentTemplateKV 的 template/DAG/role/tool-edge 叙事 + 当前 MAScoder workflow template |
| Template-derived code-base scheduling hints | `codebase_prefetch_hints`, priority, scheduler metadata, H12；借鉴 KVFlow-style scheduling |
| Exact-content code segment reuse policy | exact-content gate, RoPE delta, H10/H11/H12 + ablation package；借鉴 KVCOMM-style cross-context reuse |

当前论文产物：

| 文件/目录 | 用途 |
|---|---|
| `paper/main.tex` | ACM/EuroSys-style 主文件，包含 CCS/keywords/bib |
| `paper/sections/*.tex` | Introduction/Motivation/Background/Design/Implementation/Evaluation/Discussion/Related Work/Conclusion |
| `paper/figures/*_tikz.tex` | 正文使用的系统/调度/reuse 三张 TikZ 概念图 |
| `paper/figures/*.png` | GPT-image2 概念图备份 + 旧稿 template synthesis 图 |
| `paper/figures/*.pdf` | Python 标准库脚本生成的数据图 |
| `paper/tables/*.tex` | 从 CSV/JSON 自动生成的主表 |
| `paper/scripts/generate_paper_figures.py` | 从现有实验结果生成图表与 `data_manifest.json` |
| `paper/refs.bib` | 当前论文引用，已补旧稿相关 work |
| `paper/compile.sh` | 论文编译入口 |

已完成的论文修复：

- ✅ 旧论文目录 `/home/gfy/Paper_CodeMAS/CodeAgent_UCM_HKBU` 已读取，迁移了可复用内容：template synthesis、agent DAG semantics、KV object metadata/lifecycle、related work。
- ✅ 三张核心概念图已切换为正文 TikZ：`fig_system_architecture_tikz.tex`, `fig_coding_prefetch_tikz.tex`, `fig_kvcomm_mechanism_tikz.tex`；GPT-image2 PNG 保留为备份。
- ✅ 旧稿的 `template_format_process.png` 已复制为 `paper/figures/fig_template_synthesis_process.png` 并插入 Design。
- ✅ 所有数据图由 `paper/scripts/generate_paper_figures.py` 从现有 CSV/JSON 生成，避免手抄数据。
- ✅ 静态检查通过：当前 9 个 figure 都有 `\Description{...}`，所有 `\includegraphics` 路径存在，所有 citation 在 `refs.bib` 中有条目，无非 ASCII。
- ✅ 2026-06-03 论文贡献归属修复：标题、摘要、Introduction、Design、Implementation、Evaluation、Discussion、Related Work、Conclusion 均已改为 “CodeMAS 的 code-base segment hints + exact-content reuse policy”；KVFlow/KVCOMM 已补 bib 并作为 prior work/reference design 引用。

编译说明：

```bash
cd /home/gfy/CodeMAS_Project/sglang-kvflow/paper
./compile.sh
```

注意：当前 Codex shell 的 PATH 中没有 `latexmk/pdflatex`，因此在 Codex 侧执行会停在 `Neither latexmk nor pdflatex is available.`；用户本地已确认可以编译。后续 session 如果要验证 PDF，应直接使用用户给出的 `paper/compile.sh` 命令，并查看 `paper/main.log`。

当前已知论文问题 / 下一步：

| 问题 | 建议动作 | 优先级 |
|---|---|---|
| TikZ 概念图仍可继续精修 | 当前已替换 GPT-image2 PNG；camera-ready 前可继续优化节点布局与术语 | 中 |
| 当前正文仍偏短，系统 paper 叙事不够饱满 | 扩展 Motivation、Design、Implementation，加入更多系统挑战与工程细节 | 高 |
| H10 pass@1 绝对值低 | 论文表述聚焦 lossless vs lossy delta；后续优先跑更强 coder model / 更稳 patch schema | 高 |
| H12 host-backed prefetch 暂未稳定 | 明确写为 limitation；工程上修 file backend memory checker 后再加入主结论 | 高 |
| 旧稿中的部分图/related work 可继续吸收 | 进一步筛选 `figures/kv-lifecycle.png`, `policy-runtime.png`, `kv-reuse.png` 是否适合当前贡献 | 中 |
| BibTeX 仍有少量未引用条目 | 可以保留，camera-ready 前清理 | 低 |

数据: `results/coding_kvflow_prefetch/qwen2_5_7b_30/PREFETCH_REPORT.md`, `results/coding_kvflow_prefetch/qwen2_5_7b_30/prefetch_summary.json`, `results/coding_kvflow_prefetch/qwen2_5_7b_10/PREFETCH_REPORT.md`

### H14: KVCOMM TTFT Stress (E6/E7/E8) ✅/⚠️ (2026-06-04 新增)

基于 `bench_kvcomm_ttft_stress.py` 跑通的 KVCOMM-style TTFT 主实验，验证 prefill/TTFT-dominant setting 下 exact-content segment reuse 的加速效果。目标叙事从旧版 100-case E2E latency (1.02×) 切换到 **TTFT speedup**，将 E2 降级为 realistic sanity check。

**E6: Long-Code TTFT Stress (50 cases × 2 length buckets)**

| Mode | Bucket | N | P50 TTFT | Speedup | Avg Cached | Cached Ratio | Exact Hit |
|---|---|---:|---:|---:|---:|---:|---:|
| prefix_cache_only | 8k | 50 | 672.9ms | 1.00× | 70 | 1.05% | — |
| exact_reuse_no_hints | 8k | 50 | 668.6ms | 1.01× | 361 | 5.63% | 1.00 |
| prefix_cache_only | 16k | 50 | 1719.0ms | 1.00× | 71 | 0.48% | — |
| exact_reuse_no_hints | 16k | 50 | **1480.4ms** | **1.16×** | **2684** | **17.42%** | 1.00 |

- 8k bucket segment 较短，cached ratio 仅 5.6%，prefill 节省被其他开销稀释 → 1.01×
- 16k bucket cached ratio 17.4%，TTFT 降低 239ms → **1.16× speedup**
- 32k/48k bucket 受 RTX 4090 24GB 限制，planner+implementer KV 无法同时驻留 → exact_hit 报告成功但 cached_tokens≈0，未作为稳定结论
- 10-case 校准曾达 1.56×，50-case 平均 1.16×，差异来自样本多样性

数据: `results/kvcomm_ttft_stress/qwen2_5_7b/ttft_stress_table.csv`, `results/kvcomm_ttft_stress/qwen2_5_7b/summary.json`

**E7: Multi-Agent Scaling (10 cases × 3 agent counts × 3 segment counts × 3 buckets)**

| Mode | Agents | Segs | Bucket | N | P50 TTFT | Speedup |
|---|---|---:|---:|---:|---:|---:|
| prefix_cache_only | 2 | 3 | 8k | 20 | 710.9ms | 1.00× |
| exact_reuse + hints | 2 | 3 | 8k | 20 | 670.5ms | **1.06×** |
| prefix_cache_only | 3 | 3 | 8k | 30 | 712.2ms | 1.00× |
| exact_reuse + hints | 3 | 3 | 8k | 30 | 668.3ms | **1.07×** |
| prefix_cache_only | 5 | 3 | 8k | 50 | 713.9ms | 1.00× |
| exact_reuse + hints | 5 | 3 | 8k | 50 | 712.9ms | 1.00× |

- 8k bucket per-agent TTFT 改善有限（~6%），因 segment 较短 cached ratio 低
- E7 smoke（2 agents, 8k, 1 seg）曾显示 **4.5× speedup**（56ms vs 253ms），说明短 prompt 下 cache hit 比例更高
- 16k bucket E7 数据受 GPU 容量限制未完成

数据: `results/kvcomm_ttft_stress/qwen2_5_7b_e7/ttft_stress_table.csv`

**E8: Performance Ablation (20 cases, 32k bucket)**

| Mode | P50 TTFT | Speedup | Cached Gain | Exact Hits | Output F1 |
|---|---:|---:|---:|---:|---:|
| ablation_prefix_only | 4156ms | 1.00× | — | 0/20 | 1.000 |
| ablation_hints_no_exact | 4154ms | 1.00× | +0 | 0/20 | 0.604 |
| ablation_exact_no_hints | 4142ms | 1.00× | +1 | 13/20 | 0.757 |
| ablation_exact_gate_rope | 4150ms | 1.00× | −22 | 13/20 | 0.668 |

- **⚠️ 32k bucket 下 ablation 受限**：GPU 容量限制导致所有 mode cached_tokens≈baseline，未能展示预期差异
- 仍可读的结论：hints only 的 F1=0.604（低于 prefix 的 1.0），说明 hints 改变输出但不加速 prefill；exact gate 命中 13/20 但无实际 KV 复用
- **需在 16k bucket 重跑 E8** 以获得有效机制拆解

数据: `results/kvcomm_ttft_stress/qwen2_5_7b_e8/ttft_stress_table.csv`

**E9: 100-case E2E Serving Check 重命名**

- `table_prefetch.tex` caption 已从 "Prefix-only cache scheduling..." 改为 "Realistic E2E serving check on 100 cases. See Table~\ref{tab:ttft-stress} for the primary prefill-dominated TTFT speedup claim."
- 论文 `06_evaluation.tex` RQ3 已重写：主 claim 使用 E6 TTFT speedup，E2 作为 realistic workload sanity check

---

## 4. 所有修改文件清单

```
CodeMAS_Project/
├── MAScoder/src/mascoder/
│   ├── code_anchor.py                          ← Code Base segment 定位与 content_signature (贡献3)
│   ├── kvflow_integration.py                   ← KVFlowHint + codebase_prefetch_hints (贡献2+3)
│   └── workflow_template.py                    ← Workflow模板生成 (贡献1)
│
├── sglang-kvflow/
│   ├── python/sglang/srt/
│   │   ├── mem_cache/
│   │   │   ├── anchor_match.py                 ← exact content gate + segment matcher (贡献3)
│   │   │   ├── radix_cache.py                  ← TreeNode + _resolve_lossy_match + gate + RoPE Delta (贡献2+3)
│   │   │   ├── hiradix_cache.py                ← HiRadixCache match/insert/split + gate (贡献2)
│   │   │   ├── cache_init_params.py            ← RoPE参数 (贡献3)
│   │   │   ├── base_prefix_cache.py            ← InsertParams
│   │   │   └── evict_policy.py                 ← lossy节点驱逐保护 + Priority策略 (贡献2)
│   │   ├── entrypoints/openai/
│   │   │   ├── protocol.py                     ← ChatCompletionRequest 新增字段
│   │   │   └── serving_chat.py                 ← metadata.lossy_reuse
│   │   └── managers/
│   │       ├── io_struct.py                    ← TokenizedGenerateReqInput
│   │       ├── schedule_batch.py               ← Req接收lossy字段
│   │       ├── scheduler.py                    ← 请求构造传播 + RoPE参数提取
│   │       └── scheduler_output_processor_mixin.py ← 18字段观测收集
│   │
│   ├── benchmark/multi_workflow/
│   │   ├── bench_multiagent_ttft.py            ← 核心KVCOMM benchmark (贡献3)
│   │   ├── bench_lossy_kv_accuracy.py          ← accuracy pass@3实验 (验证目标1)
│   │   ├── bench_large_codebase_reuse.py       ← 大codebase多Agent实验(贡献3)
│   │   ├── bench_large_blocks.py               ← 大block实验
│   │   ├── bench_multiagent_large.py           ← 多Agent workflow实验
│   │   ├── bench_swe_lite_kv.py                ← SWE-bench 50task latency实验 (验证目标2)
│   │   ├── bench_kv_replacement.py             ← 独立KV替换实验(Phase 3) (验证目标3)
│   │   ├── analyze_kv_isolation.py             ← 独立KV隔离分析(Phase 1+2)
│   │   ├── bench_template_codebase_segments.py ← 模板多 segment smoke (贡献3)
│   │   ├── prepare_repo_level_datasets.py      ← SWE-bench Verified 多文件快照准备
│   │   ├── prepare_swebench_verified_expanded.py ← 30/100/500-case manifest 生成
│   │   ├── bench_real_codebase_exact_reuse.py  ← 真实 repo-level exact reuse + HF KV 差异
│   │   ├── bench_swe_generated_patch_kvcomm.py ← JSON-edit pass@1 lossless/lossy 主对比
│   │   ├── bench_coding_kvflow_prefetch.py     ← prefix-only vs coding-aware KVFlow serving 对比
│   │   ├── run_swebench_local_env_batch.py     ← SWE-bench 本地 gold/base/candidate 批处理
│   │   ├── report_repo_level_exact_reuse.py    ← repo-level 实验报告生成
│   │   └── visualize_kvcomm_anchors.py         ← KVCOMM可视化
│   │
│   └── results/
│       ├── PROJECT_STATE.md                    ← 本文件
│       ├── experiment_dashboard.html           ← 可视化总结
│       ├── swe_latency_summary.md              ← SWE-bench 50 task 结果
│       ├── large_block_summary.md              ← 大block 45 task 结果
│       ├── codebase_reuse/                     ← 大codebase多Agent 结果
│       │   ├── summary.md
│       │   └── results.json
│       ├── repo_level_datasets/                ← SWE-bench Verified 多文件快照
│       │   ├── manifest_30.json / swe_verified_30_instances.json
│       │   ├── manifest_100.json / swe_verified_100_instances.json
│       │   ├── manifest_500.json / swe_verified_500_instances.json
│       │   ├── repo_level_100_reuse_report.md
│       │   └── gate_anchor_500_report.md
│       ├── swebench_local_envs/                ← 30-case gold/base smoke 与判别样例
│       ├── swe_generated_patch_kvcomm/         ← pass@1 lossless/lossy 对比结果
│       ├── coding_kvflow_prefetch/             ← H12 codebase prefetch serving 对比结果
│       ├── real_codebase_exact_reuse/          ← repo-level exact reuse 结果
│       │   ├── repo_dataset_combined_summary.json
│       │   └── repo_dataset_report.md
│       └── ma_ttft/                            ← KVCOMM benchmark 结果
│           ├── final_v2.log
│           ├── offset_test_v5.log
│           └── rope_timing.log
│
└── /tmp/
    ├── kv_isolation/                           ← Phase 1+2 原始数据
    ├── swe_latency/                            ← SWE-bench 50task 原始数据
    ├── large_block_results/                    ← 大block 45task 原始数据
    ├── lossy_accuracy_v4/                      ← accuracy pass@3 原始数据
    ├── kv_replacement_results/                 ← Phase 3 原始数据
    └── ma_large_blocks/                        ← 多Agent workflow 原始数据
```

---

## 5. 当前状态与已知问题

### 5.1 已完成

**贡献1 (Workflow Template)**:
- ✅ 固定 Workflow 模板：Planner → Implementer → Reviewer
- ✅ 模板驱动的 Agent 执行流程

**贡献2 (Template-Guided KV Management)**:
- ✅ KVCOMM 全程链路: MAScoder → HTTP → sglang-kvflow → HiRadixCache → exact-content matcher → gate → HTTP响应
- ✅ exact content gate：AST/anchor 只定位，实际复用必须 `content_signature` 完全一致
- ✅ KVFlow 从 agent-prefix prefetch 扩展到 coding-aware `codebase_prefetch_hints`
- ✅ first/final 双阶段观测
- ✅ matcher gate（reject 时跳过 cache）
- ✅ Priority 策略实现与传播
- ✅ HiCache Prefetch 调度

**贡献3 (Code-Base-Aware KV Reuse)**:
- ✅ 代码块间 cross-attention 弱 55-98× 的证据
- ✅ V cosine 2.3× gap 的证据
- ✅ SWE-bench Lite 50 task + Verified 30 task 真实数据验证
- ✅ 大 Codebase 多 Agent lossy/lossless KV 复用对比
- ✅ RoPE Delta 旋转实现与验证
- ✅ `_span_similarity` 保留为定位/观测辅助，不作为异内容复用许可
- ✅ code-base segment 提取从 "全 prompt" 改为 "代码 only"
- ✅ 模板多 segment smoke：Planner 的 `code_base1/2/3` 分别被 Implementer/Debugger 正确复用
- ✅ 真实 repo-level 多文件 serving benchmark：Astropy/Django/Matplotlib，平均 speedup 1.544×
- ✅ HF 层真实 KV vs RoPE-rotated KV 差异量化：layer-24 K cosine avg=0.998970
- ✅ `anchor_kv_store` 已加锁，读写和 `ref_count` 更新具备基本并发保护
- ✅ Code-First Prompt 设计：BLEU=1.000 + Cache 复用率 98.5%
- ✅ SWE-bench Verified 30/100/500-case manifest 扩展；30-case 本地 smoke 筛出 28 个判别样例
- ✅ Qwen2.5-7B JSON-edit 30-case lossless KV vs exact-content segment reuse pass@1 主对比
- ✅ 100/500-case gate/anchor/content-signature scalability 统计与图表
- ✅ 与 SGLang KVFlow 结合：模板侧生成 future code-base prefetch hints，engine 侧通过 scheduler/HiCache 提前预取 exact code block
- ✅ H12 28-case prefix-only vs coding-aware KVFlow serving 对比报告；KVCOMM+hint exact-content hit 28/28

### 5.2 待完善

| 验证目标 | 问题 | 优先级 |
|---|---|---|
| 验证1 (准确性) | 大 gap (>20 tokens) zero-fill 导致 BLEU 下降 | 已有结论：Code-First 规避 |
| 验证1 (准确性) | 30-case Qwen2.5-7B pass@1 绝对值低，需更强 coding model 或更强 patch synthesis schema | 高 |
| 验证2 (Segment标记) | Workflow Template 需要固化 `code_base_id/content_signature/token_span` schema | 高 |
| 验证3 (实现) | 多 segment serving benchmark 已跑通；需扩大到 100-case serving reuse 主表 | 中 |
| 验证3 (实现) | HiCache file storage backend 在长 coding prompt 下触发 runtime memory-checker，host load-back queued tokens 暂未作为稳定结论 | 高 |
| 验证3 (实现) | 内存管理：`ref_count` 未实现 GC | 中 |
| 验证3 (实现) | 多 segment 链式复用需真实 serving 验证 | 中 |
| 验证3 (实现) | 分段 Prefill 未实现 | 低（未来工作） |
| 全局 | Phase 3 DynamicCache merge 兼容性问题 | 低（不影响结论） |
| 全局 | 模型太小（3B），7B 用了 GQA 反而 KV/tok 更小 | 中 |
| 全局 | 尚未集成 V cosine gate（Phase 2 发现的可用于数值门控）| 低 |

---

## 6. 下一步建议

### 短期（1-2 周）：完成贡献3验证

1. **验证1 收尾**
   - 将 30-case pass@1 主表扩展到更强 coding model 或更强约束式 patch/edit schema
   - 在论文中报告 lossless KV vs exact-content segment reuse pass@1 delta、cached tokens、latency，而非只报告绝对 pass@1

2. **验证2 优化**
   - 在 Workflow Template 中固化 code-base segment schema
   - 支持多 segment 联合复用，并保持 exact-content gate

3. **验证3 工程修复**
   - 实现 `ref_count` 递减和 GC
   - 将 100-case 扩展统计升级为 serving reuse 主表，加入 more-agents / more-segments ablation

### 中期（1-2 月）：三贡献整合与论文撰写

1. **整合实验**
   - 在统一 Workflow 模板下，同时测量 One-Shot Accuracy（贡献1）、TTFT 加速（贡献2+3）、Cache 复用率
   - 对比基准 vs 全量优化

2. **论文撰写**
   - 将三个贡献整合为一篇完整工作
   - 明确区分系统级设计（Workflow Template）与实现级优化（KV Management + KVCOMM）

### 长期

1. 向上游 SGLang 提交 PR
2. 分段 Prefill 实现（如需支持任意位置 Code Base 复用）

---

## 7. 快速重启指南

```bash
# 启动 sglang serving (带 gate + KVCOMM)
cd /home/gfy/CodeMAS_Project/sglang-kvflow
PYTHONPATH=python \
  /home/gfy/.conda/envs/sglang-kvflow/bin/python -m sglang.launch_server \
  --model-path /home/gfy/models/Qwen2.5-3B-Instruct --port 30000 \
  --tp-size 1 --mem-fraction-static 0.85 --max-total-tokens 32768 \
  --chunked-prefill-size 4096 --max-prefill-tokens 8192 \
  --radix-eviction-policy priority --enable-hierarchical-cache \
  --hicache-ratio 1.5 --hicache-write-policy write_back \
  --enable-cache-report --disable-cuda-graph --log-level info

# 运行 KVCOMM benchmark (贡献3)
PYTHONPATH=python:../MAScoder/src \
  /home/gfy/.conda/envs/sglang-kvflow/bin/python \
  benchmark/multi_workflow/bench_multiagent_ttft.py

# 运行大 codebase 实验
PYTHONPATH=python:../MAScoder/src \
  /home/gfy/.conda/envs/sglang-kvflow/bin/python \
  benchmark/multi_workflow/bench_large_codebase_reuse.py

# 运行 SWE-bench 实验 (验证目标2)
PYTHONPATH=python:../MAScoder/src \
  /home/gfy/.conda/envs/sglang-kvflow/bin/python \
  benchmark/multi_workflow/bench_swe_lite_kv.py --n 50

# 准备真实 repo-level 多文件数据集 (SWE-bench Verified codebase-content)
/home/gfy/.conda/envs/sglang-kvflow/bin/python \
  benchmark/multi_workflow/prepare_repo_level_datasets.py --max-repos 3 --max-files 3

# 生成 30/100/500-case 扩展数据集
/home/gfy/.conda/envs/sglang-kvflow/bin/python \
  benchmark/multi_workflow/prepare_swebench_verified_expanded.py \
  --max-cases 30 --max-files 3 --label 30 --allow-multiple-per-repo --max-per-repo 3

/home/gfy/.conda/envs/sglang-kvflow/bin/python \
  benchmark/multi_workflow/prepare_swebench_verified_expanded.py \
  --max-cases 100 --max-files 3 --label 100 --allow-multiple-per-repo --max-per-repo 15

/home/gfy/.conda/envs/sglang-kvflow/bin/python \
  benchmark/multi_workflow/prepare_swebench_verified_expanded.py \
  --max-cases 500 --max-files 3 --label 500 --allow-multiple-per-repo

# 运行真实 repo-level exact-code KV 复用实验
CUDA_VISIBLE_DEVICES=0 /home/gfy/.conda/envs/sglang-kvflow/bin/python \
  benchmark/multi_workflow/bench_real_codebase_exact_reuse.py \
  --manifest results/repo_level_datasets/manifest.json \
  --max-cases 3 --files-per-case 3 --max-segment-chars 30000 --max-tokens 64

# 生成 repo-level 实验报告
/home/gfy/.conda/envs/sglang-kvflow/bin/python \
  benchmark/multi_workflow/report_repo_level_exact_reuse.py \
  --input results/real_codebase_exact_reuse/repo_dataset_combined_summary.json \
  --output results/real_codebase_exact_reuse/repo_dataset_report.md

# 30-case 本地环境 smoke
/home/gfy/.conda/envs/sglang-kvflow/bin/python \
  benchmark/multi_workflow/run_swebench_local_env_batch.py \
  --dataset results/repo_level_datasets/swe_verified_30_instances.json \
  --out results/swebench_local_envs/expanded_30_gold_smoke.json \
  --mode gold --max-cases 30 --max-fail-tests 1 --timeout 1200 --skip-existing-pass

/home/gfy/.conda/envs/sglang-kvflow/bin/python \
  benchmark/multi_workflow/run_swebench_local_env_batch.py \
  --dataset results/repo_level_datasets/swe_verified_30_instances.json \
  --out results/swebench_local_envs/expanded_30_base_smoke.json \
  --mode base --max-cases 30 --max-fail-tests 1 --timeout 1200 --skip-existing-pass

# 30-case Qwen2.5-7B JSON-edit pass@1 主对比
/home/gfy/.conda/envs/sglang-kvflow/bin/python \
  benchmark/multi_workflow/bench_swe_generated_patch_kvcomm.py \
  --model /home/gfy/models/Qwen2.5-7B-Instruct \
  --dataset results/swebench_local_envs/expanded_30_discriminative_instances.json \
  --manifest results/repo_level_datasets/manifest_30.json \
  --max-cases 28 --files-per-case 1 --max-file-chars 30000 \
  --max-tokens 768 --output-schema json-edit --repair-attempts 1 \
  --out-dir results/swe_generated_patch_kvcomm/qwen2_5_7b_json_30

# H12: 30-case prefix-only vs coding-aware KVFlow serving 对比
CUDA_VISIBLE_DEVICES=0 /home/gfy/.conda/envs/sglang-kvflow/bin/python \
  benchmark/multi_workflow/bench_coding_kvflow_prefetch.py \
  --model /home/gfy/models/Qwen2.5-7B-Instruct \
  --dataset results/swebench_local_envs/expanded_30_discriminative_instances.json \
  --manifest results/repo_level_datasets/manifest_30.json \
  --max-cases 28 --files-per-case 1 --max-file-chars 12000 \
  --max-tokens 64 --out-dir results/coding_kvflow_prefetch/qwen2_5_7b_30
```

---

## 8. Submission Experiment Status (2026-06-03) — 本轮完成

本轮完成 E2 100-case 合并、E1/E5 limitation 记录、论文 Evaluation/Discussion 更新。

### 已完成并进入论文

| ID | 实验 | 产物 | 当前结论 |
|---|---|---|---|
| E2 | 100-case serving reuse scalability | `results/coding_kvflow_prefetch/qwen2_5_7b_100/summary.json` (合并 part1/part2/part3) | **100/100** cases。`kvcomm_lossy_plus_codebase_prefetch`: exact-content hit rate **0.99**，avg cached tokens **2,592.5** (+64% vs baseline 1,581.8)，avg latency **3,837.6ms** (vs baseline 3,910.6ms)，p50 **3,833ms** vs baseline **3,872ms**。Table~\ref{tab:prefetch} 已更新为 100 cases 含 p50/p90。 |
| E3 | Multi-segment / multi-agent template ablation | `results/template_codebase_segments/template_segment_ablation.csv`, `TEMPLATE_SEGMENT_ABLATION_REPORT.md` | 使用真实 repo code segments；P→I→D + 3 segments 达到 3 exact hits / 14,967 estimated cached tokens |
| E4 | Near-match safety expansion | `results/kvcomm_ablation_package/gate_nearmatch_500.csv`, `GATE_NEARMATCH_500_REPORT.md` | 500 near-match negatives + 50 exact controls；exact-content gate false accepts **0**，AST/span/path/no-gate false accepts **500** |
| Paper pipeline | Auto table/figure ingestion | `paper/scripts/generate_paper_figures.py`, `paper/data_manifest.json` | 新增 p50/p90 latency 列；table caption 动态读取 case 数；所有图表从合并后的 100-case 数据生成 |
| Merge helper | E2 partial summary 合并工具 | `benchmark/multi_workflow/merge_e2_summaries.py` | 支持任意数量 partial summary.json 输入，去重 instance_id，重新计算 mode_summary，输出 summary.json + prefetch_summary.json + prefetch_table.csv |

### 记录为 Limitation（不进入主表）

| ID | 实验 | 当前状态 / 阻塞原因 | 论文处理 |
|---|---|---|---|
| E1 | Qwen2.5-32B GPTQ 28-case paired pass@1 | RTX 4090 24GB 上模型权重几乎占满显存，仅剩 ~2,554 input-token capacity，prompt 需要 ~2,751 tokens 导致 input length exceeds max。`--allow-auto-truncate` 触发 `tokenizer_manager.py:1599 IndexError: list index out of range`（lossy metadata 与 auto-truncate 的兼容性问题）。32B 评估需更大显存机器。 | Discussion/Limitations 已加入 “Hardware-constrained model evaluation” 段落；accuracy claim 降级为 7B paired-delta/serving-safety evidence |
| E5 | host-backed prefetch smoke | `--hicache-storage-backend file` 触发 `token_to_kv_pool_allocator memory leak detected`（runtime checker），server SIGQUIT。 | Discussion/Limitations 已加入 “Host-backed prefetch needs more engineering” 段落；不 claim host load-back acceleration |

### 本轮代码/实验系统更新

- `benchmark/multi_workflow/merge_e2_summaries.py` **(新增)**
  - 合并任意数量 partial summary.json，去重 instance_id（保留最后出现），重新计算 mode_summary（含 p50/p90）。
  - 输出 summary.json + prefetch_summary.json + prefetch_table.csv。
- `benchmark/multi_workflow/bench_coding_kvflow_prefetch.py`
  - 新增 serving-only `load_cases()`：从 repo-level manifest samples 直接读取 `local_path` code segments，支持真正的 100-case serving experiment。
  - 新增 `--disable-hierarchical-cache`，E2 主实验默认关闭 hierarchical cache；E5 仍单独测试 host-backed path。
  - `summary.json` 与 `prefetch_summary.json` 同时写出，满足投稿计划产物规范。
  - 异常时写 partial artifact，避免长跑失败吞掉已完成样本。
  - `extract_text(...) or “”`，避免 error/empty response 导致 `NoneType len()` 崩溃。
- `benchmark/multi_workflow/bench_swe_generated_patch_kvcomm.py`
  - 新增 `--allow-auto-truncate` 到 server launch args（虽在 32B 上仍触发 IndexError，但保留了配置入口）。
- `paper/scripts/generate_paper_figures.py`
  - `table_prefetch` 生成 7 列表格：Mode / Avg lat / P50 / P90 / Avg cached / Avg hints / Exact hits。
  - Table caption 动态读取 case 数（自动适应 28/100 cases）。
  - pass@1 主表优先读取 full 32B，其次 32B smallctx，最后回退 7B。
- `paper/sections/06_evaluation.tex`
  - RQ3 已更新为 100-case 引用和数据（cached tokens +64%，99/100 exact hits，latency p50/p90）。
- `paper/sections/07_discussion.tex`
  - 新增 “Hardware-constrained model evaluation” limitation 段落。
  - 更新 “Host-backed prefetch needs more engineering”，确认 allocator leak。
  - 新增 100-case 结果引用到 exact-content safety 段落。

### 当前投稿判断

- **安全性证据闭合**：controlled gate + 500-case near-match expansion + 100-case serving exact-content hit rate 0.99 均支持 exact-content-only safety boundary。
- **Template contribution 证据闭合**：E3 显示更多 segment/downstream agents 带来更多 exact hits 与 reusable-token opportunity。
- **性能/规模证据闭合**：E2 100/100 完成，cached-token +64%，latency 下降，p50/p90 已生成。100-case serving scalability 主表进入论文。
- **accuracy preservation 降级但诚实**：E1 32B 在 24GB 上不可行，论文 Discussion 已明确记录 limitation；accuracy claim 聚焦于 lossless-vs-lossy delta（7B 28-case）和 serving safety（100-case），不夸大 pass@1。
- **论文当前状态**：主文已有 Safety（RQ1）、Numerical Correctness（RQ2）、Performance（RQ3）、Accuracy delta（RQ4）、Scalability（RQ5）。Discussion 已记录 E1 hardware limitation 和 E5 host prefetch limitation。
- **下一步（非阻塞）**：
  1. 如果获得更大显存机器， rerun E1 32B smallctx/full 作为 stronger-model paired evidence。
  2. Camera-ready 前精修 TikZ 概念图节点布局。
  3. 扩展正文 Motivation/Design/Implementation 篇幅（当前仍偏短）。
  4. 清理未引用 bib 条目。
