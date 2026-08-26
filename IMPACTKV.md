# ImpactKV — 命令速查

**接手请先读 [`HANDOFF.md`](HANDOFF.md)**（冻结数字、红线、第一天、论文/代码改哪里）。  
本页只放 clone / 单测 / 解压 / GPU 命令。

当前工作：在 **本 SGLang fork** 上做 coding-aware true-lossy **文件岛 KV 拷贝**，投稿 ASPLOS 2027。

**只 clone 这一个仓库。** 没有第二份 repo，也没有原集群。论文源、checker、冻结 PLAN、战役脚本都在本 branch。

| | |
|---|---|
| 仓库 | [`ccdd2023/sglang`](https://github.com/ccdd2023/sglang) |
| 分支 | **`integration/template-prefetch-swebench`** |
| 论文 + checker | [`docs/kvflow/paper/`](docs/kvflow/paper/) |
| 冻结 JSON / PLAN | 仓库内 [`benchmark/multi_workflow/offcluster/`](benchmark/multi_workflow/offcluster/)（解压到本机） |
| Headline | 7B job **137185**：cache-ready **1.492×** / 1684/1684 / 99.3% / 93.6% |
| 评测范围 | sequential one-token prefill vs 同一引擎 Dense；**不是** serving 皇冠 |

中文论证+图：[`docs/kvflow/paper/PAPER_LOGIC_CN.md`](docs/kvflow/paper/PAPER_LOGIC_CN.md)。  
历史 V40–V46 / RepoBench **不是** headline。从本文件开始。

---

## 0. 你机器上要具备什么

- **GPU**：Headline 7B 需要约 **24GB**（论文战役是 RTX 4090 + bf16，context 32k）。30B AWQ 附录另要一张能跑 `Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit` 的卡。
- **CUDA + Python ≥ 3.10**，按本仓库 [`docs/get_started/install.md`](docs/get_started/install.md) **Method 2（from source）** 装 **这个 fork**，不要 `pip install sglang` 装上游包。
- **磁盘**：7B 权重约 15GB；claim pack 解压后约 150MB。
- 不需要 Slurm、不需要 Enroot、不需要 mini-SWE-agent（Headline 是 exact-prompt 回放，不是 agent 评测）。

---

## 1. 方法（Headline 只用 M0+M2）

1. **M0 admit**：单文件 `repository_code`、版本有效、token-ID 相同、\(\Delta \neq 0\)。离线 oracle，不估 Attention。
2. **M2 copy**：source 侧预旋转 \(K\)，\(V\) 原样；页键 `(source_prefix_hash, content_hash, Δ)`。
3. **Fail-closed**：hash / 覆盖 / alloc 失败 → 整岛 Dense，从不半页拼接。
4. **M1 prefix / M3 prefetch**：代码里有，Headline 关掉。

引擎：`python/sglang/srt/mem_cache/kvcomm_exact.py` 与 `kvcomm/`。`kvcomm_prefetch/` 是 M3。

---

## 2. 克隆、环境、单测（无 GPU）

```bash
git clone git@github.com:ccdd2023/sglang.git sglang-kvflow
cd sglang-kvflow
git checkout integration/template-prefetch-swebench

# 按 docs/get_started/install.md Method 2 安装这个 checkout
pip install -e "python"
export PYTHONPATH="$PWD/python:$PWD"

python -m pytest -q \
  python/sglang/srt/mem_cache/test_kvcomm_exact.py \
  python/sglang/srt/mem_cache/kvcomm/test_core.py \
  python/sglang/srt/mem_cache/kvcomm/test_radix_backend.py \
  python/sglang/srt/mem_cache/kvcomm_prefetch/test_*.py
```

解压冻结产物（无 GPU，约十几秒）：

```bash
python benchmark/multi_workflow/fetch_impactkv_artifacts.py
export IMPACTKV_ARTIFACTS="$PWD/impactkv-artifacts"

cd docs/kvflow/paper
python3 scripts/check_asplos_claims.py          # 必须 PASS
python3 -m pytest -q scripts/
```

不要 `git add impactkv-artifacts/`。

---

## 3. Headline 数字（禁止手改冻结 RESULT）

| 战役 | 作业 | 引用 | 数字 |
|---|---|---|---|
| **7B 主表** | 137185 | `tab:eval-summary` | **1.492×** / **1684/1684** / 赢率 **99.3%** / agree **93.6%**（不是 Accuracy）。prefetch off，prefix off |
| 30B 附录 | 96092 | `tab:eval-30b` only | **1.375×** / 1684/1684 / 94.8%。**96.5% 禁止出现在 7B 正文** |
| 7B prefix-on | 139839 | `tab:7b-prefix-on` | prefix-only 1.526× / lossy-only 1.408× / dual 2.120× / 增量 1.390× |
| 同引擎 copier | 137400 | `tab:admit-ablation` | 文件岛 1.492×/93.6%；KVCOMM-style 2.100×/89.4%；CacheBlend-style 1.883×/91.9%。**不是原生栈** |

不要报 N=4。不要写 SOTA。不要把 \(1.492\times\) 和 \(1.375\times\) 写成一个 official method。

你自己 GPU 上重跑的 TTFT **不会**和 137185 bit 一致（卡型、驱动、负载都不同）。机械检查：`copy_events == 1684`、`fallback_events == 0`、`prefetch is False`。速度比对照冻结 `RESULT.json`，**禁止覆盖**冻结文件。

---

## 4. 下载模型（HuggingFace，公开权重）

Headline 7B：

```bash
huggingface-cli download Qwen/Qwen2.5-Coder-7B-Instruct \
  --local-dir "$HOME/models/Qwen2.5-Coder-7B-Instruct"
export IMPACTKV_MODEL="$HOME/models/Qwen2.5-Coder-7B-Instruct"
```

30B 附录（可选）：

```bash
huggingface-cli download cyankiwi/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit \
  --local-dir "$HOME/models/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit"
```

`IMPACTKV_MODEL` **必须是本地目录**（含 `config.json`）。默认 `IMPACTKV_HF_OFFLINE=1`，服务器不联网拉权重。

Chat template 用仓库里的 `benchmark/multi_workflow/qwen3_coder_tool_chat_template.jinja`（7B 战役也用它，不要换）。

---

## 5. 在你的 GPU 上复现 7B Headline

数字已冻结。重跑是为了验证引擎，不是为了改论文表。

冒烟（2 个 group，约数分钟）：

```bash
source benchmark/multi_workflow/impactkv_local_env.sh
export IMPACTKV_MAX_GROUPS=2
bash benchmark/multi_workflow/run_impactkv_headline.sh
```

冒烟写入 `impactkv-artifacts/runs/headline_7b_*`，`RESULT.json` 的 `status` 是 **`SMOKE`**，不能当论文数字。

全量 235 group（Dense + reuse，一张 4090 要很多小时）：

```bash
unset IMPACTKV_MAX_GROUPS
bash benchmark/multi_workflow/run_impactkv_headline.sh
```

脚本会：解压 claim pack（若尚未解压）→ **拷贝** PLAN 到新目录 → 跑 `run_swebench_prerotated_file_modules.py`。冻结的 `prefixkey_20260824/RESULT.json` 不会被动到。

类模板（offline 一类任务一份 prior，online 微调）不是 137185：

```bash
python benchmark/multi_workflow/compile_class_template.py \
  --manifest "$IMPACTKV_ARTIFACTS/.../DYNAMIC_MANIFEST.json" \
  --output "$IMPACTKV_ARTIFACTS/runs/coding_agent.template.json"
export SGLANG_KVCOMM_CLASS_TEMPLATE="$IMPACTKV_ARTIFACTS/runs/coding_agent.template.json"
```

线上 admit（source 不看 target；`SGLANG_KVCOMM_ONLINE_ADMIT` 只在 reuse 臂打开）是另一场战役，**不是** 137185：

```bash
unset IMPACTKV_MAX_GROUPS
bash benchmark/multi_workflow/run_impactkv_online_admit.sh
```

写出 `impactkv-artifacts/runs/online_admit_7b_*`。禁止覆盖 `prefixkey_20260824`。

有 Slurm 的机器可以用 `benchmark/multi_workflow/slurm/swebench_7b_file_modules.sbatch`（已改为相对本仓库路径）。没有 Slurm 就用上面的 `run_impactkv_headline.sh`。

Prefix-on / copier / 30B：同样先 `fetch`，再把对应 PLAN 拷到 `impactkv-artifacts/runs/...`，调用

- `run_swebench_7b_prefix_on.py`
- `run_swebench_7b_sota_copiers.py`
- `run_swebench_prerotated_file_modules.py` + 30B 模型

或提交同目录下的 sbatch。

---

## 6. 改代码 / 改论文

| 你要改 | 去哪 |
|---|---|
| 拷贝内核 | `python/sglang/srt/mem_cache/kvcomm_exact.py`、`kvcomm/` |
| Prefetch | `kvcomm_prefetch/` |
| 战役 | `benchmark/multi_workflow/run_swebench_*.py` |
| 论文 | `docs/kvflow/paper/sections/*.tex` |
| 声称 | `docs/kvflow/paper/scripts/check_asplos_claims.py` 必须 PASS |

脏 worktree：**禁止** `git add -A` / `git clean` / `git reset --hard`。推送 `origin integration/template-prefetch-swebench`。

---

## 7. 仓库地图

```
sglang/   branch integration/template-prefetch-swebench
  IMPACTKV.md
  python/sglang/srt/mem_cache/kvcomm*
  benchmark/multi_workflow/
    fetch_impactkv_artifacts.py
    run_impactkv_headline.sh
    impactkv_local_env.sh
    offcluster/impactkv-claim-pack.tar.gz
    run_swebench_*.py
  docs/kvflow/paper/
```

解压后的 JSON 在 `./impactkv-artifacts/`（gitignore）。权重从 HuggingFace 下到你自己的盘。
