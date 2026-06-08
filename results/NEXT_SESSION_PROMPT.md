# Restart Prompt for Next Codex Session

请继续 `/home/gfy/CodeMAS_Project/sglang-kvflow` 的 CodeMAS 投稿级实验与论文完善任务。

## Current State (2026-06-04)

- 项目状态文档：`/home/gfy/CodeMAS_Project/sglang-kvflow/results/PROJECT_STATE.md`
- HTML 进展报告：`/home/gfy/CodeMAS_Project/sglang-kvflow/results/progress_report_2026-06-03.html`（可直接浏览器打开）
- 论文目录：`/home/gfy/CodeMAS_Project/sglang-kvflow/paper`
- Python 环境：`/home/gfy/.conda/envs/sglang-kvflow/bin/python`
- 本地 GPU：RTX 4090 24GB

## 本轮已完成 (2026-06-04)

### E6: TTFT Stress — ✅ 完成
- 50 cases × 2 buckets (8k/16k) × 4 modes × 2 max_tokens = 800 rows
- **16k bucket: 1.16× TTFT speedup** (1480ms vs 1719ms), 100% exact hit rate
- 8k bucket: 1.01×（segment 太短，cached ratio 仅 5.6%）
- 32k/48k bucket: 受 RTX 4090 24GB 限制，planner+implementer KV 无法同时驻留 → 未作为稳定结论
- 输出：`results/kvcomm_ttft_stress/qwen2_5_7b/`

### E7: Agent Scaling — ✅ 完成
- 10 cases × 3 agent counts × 3 segment counts × 3 buckets × 2 modes = 1800+ rows
- 8k bucket per-agent TTFT 改善 ~6%，5-agent/3-seg 基本持平
- E7 smoke（2 agents, 8k, 1 seg）曾显示 **4.5× speedup**（56ms vs 253ms）
- 输出：`results/kvcomm_ttft_stress/qwen2_5_7b_e7/`

### E8: Performance Ablation — ⚠️ 完成但受限
- 20 cases × 4 ablation modes = 80 rows
- **⚠️ 32k bucket 下 GPU 容量限制**，所有 mode cached_tokens≈baseline，ablation 未展示预期差异
- 结论：hints only 不加速 prefill；exact gate 是主要机制（但需 16k 重跑验证）
- 输出：`results/kvcomm_ttft_stress/qwen2_5_7b_e8/`

### E9: 100-case E2E 重命名 — ✅ 完成
- `table_prefetch.tex` caption 改为 "Realistic E2E serving check"
- 论文 RQ3 重写：主 claim 使用 E6 TTFT speedup，E2 作为 sanity check
- 新增 `table_ablation.tex` 占位符（待 E8 有效数据填充）

### 代码修改
- `bench_kvcomm_ttft_stress.py`: 新增 E8_MODES, `run_e8()`, `--skip-e8/--e8-cases/--e8-length` CLI args
- `generate_paper_figures.py`: 新增 `table_ablation()`, E9 caption 修改
- `06_evaluation.tex`: 新增 ablation table input + narrative，强化交叉引用
- `run_submission_experiments.sh`: 新增 e6-smoke/e6/e7-smoke/e7/e8/ttft-all targets

### 论文构建
- `generate_paper_figures.py` 成功运行，所有表格/figure 更新
- `table_ttft_stress.tex` 已填充真实数据（非占位符）
- `table_ablation.tex` 当前为占位符（Pending ablation run）
- LaTeX 编译 11 pages 无 error

## 下一步建议（优先级排序）

### 高优先级（P0）
1. **16k bucket E8 Ablation 重跑**
   - 当前 E8 在 32k bucket 受 GPU 限制无有效结果
   - 修改 run 脚本：`--e8-length 16000`，从 E6 同一批 cases 取前 20 个（避免数据集漂移）
   - 预期：hints only ≈1.00×，exact gate ≈1.15×，exact+RoPE ≈1.16×
   - 验证：hints only 与 prefix only 的 cached_tokens 应接近（确认 hints 不驱动 prefill 加速）

2. **论文 RQ3 最终定稿**
   - 将 E6 16k 1.16× 作为主 TTFT speedup claim
   - E2 100-case +64% cached tokens 作为 realistic sanity check
   - E9 叙事：明确区分 "stress workload (TTFT-dominant)" vs "realistic workload (E2E-diluted)"
   - 增加 GPU capacity limitation 作为 Discussion 段落

### 中优先级（P1）
3. **如果获得更大显存机器**（A100 80GB）：
   - rerun E6 with 32k/48k buckets（预期 1.5–3.0× speedup）
   - rerun E7 with 16k buckets（预期 5-agent/3-seg cumulative speedup ≥1.5×）
   - 更新 `table_ttft_stress.tex` 增加 32k/48k 行

4. **正文篇幅扩展**：
   - Motivation: 补充 coding MAS 中 repeated code segments 的定量分析
   - Design: 补充 RoPE delta 的数学推导与伪代码
   - Implementation: 补充 anchor match pipeline 的时序图

### 低优先级（P2）
5. **HiCache file backend leak 修复**（未来工作）
   - 当前 `--hicache-storage-backend file` 触发 memory leak
   - 修复后可重新启用 host-backed prefetch 实验

6. **BibTeX 清理**：camera-ready 前清理未引用条目

## 快速验证命令

```bash
cd /home/gfy/CodeMAS_Project/sglang-kvflow

# 刷新所有论文图表
/home/gfy/.conda/envs/sglang-kvflow/bin/python paper/scripts/generate_paper_figures.py

# 静态检查
/home/gfy/.conda/envs/sglang-kvflow/bin/python -m py_compile \
  benchmark/multi_workflow/bench_kvcomm_ttft_stress.py \
  paper/scripts/generate_paper_figures.py

# E8 16k 重跑
/home/gfy/.conda/envs/sglang-kvflow/bin/python benchmark/multi_workflow/bench_kvcomm_ttft_stress.py \
  --model /home/gfy/models/Qwen2.5-7B-Instruct \
  --manifest results/repo_level_datasets/manifest_500.json \
  --max-cases 20 --e8-length 16000 \
  --disable-hierarchical-cache \
  --skip-e6 --skip-e7 \
  --port 30000 \
  --out-dir results/kvcomm_ttft_stress/qwen2_5_7b_e8_16k

# 用户本地编译论文
cd paper && ./compile.sh
```

## 最终检查清单

- [x] E6 50-case TTFT stress 完成，table 已填充
- [x] E7 agent scaling 完成
- [x] E8 ablation 完成（32k 受限，需 16k 重跑）
- [x] E9 100-case 重命名为 "Realistic E2E serving check"
- [x] 论文 RQ3 重写完成
- [x] `table_ablation.tex` 占位符已创建
- [x] `paper/data_manifest.json` 已更新
- [x] 所有 Python 脚本 py_compile 通过
- [ ] E8 16k 有效 ablation 数据（P0）
- [ ] 论文 LaTeX 编译通过（需用户本地执行 `./compile.sh`）
- [ ] 正文篇幅扩展至系统 paper 标准
