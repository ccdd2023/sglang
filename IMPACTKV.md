# ImpactKV — 合作者入口（本仓库即可）

当前工作：在 **本 SGLang fork** 上做 coding-aware true-lossy **文件岛 KV 拷贝**，投稿 ASPLOS 2027。

**合作者只需要 clone 这一个仓库。** 不要找 CodeMAS / `kvflow-reports`：论文源、checker、战役脚本都在本 branch 里。

| | |
|---|---|
| 仓库 | [`ccdd2023/sglang`](https://github.com/ccdd2023/sglang) |
| 分支 | **`integration/template-prefetch-swebench`** |
| 论文 + checker | [`docs/kvflow/paper/`](docs/kvflow/paper/) |
| 冻结战役产物 | 集群目录 `kvflow-artifacts/`（约 4GB，**不进 git**） |
| Headline | 7B job **137185**：cache-ready **1.492×** / 1684/1684 / 99.3% / 93.6% |
| 评测范围 | sequential one-token prefill vs 同一引擎 Dense；**不是** serving 皇冠 |

中文论证+图：[`docs/kvflow/paper/PAPER_LOGIC_CN.md`](docs/kvflow/paper/PAPER_LOGIC_CN.md)（同目录 PDF）。  
这不是 KVFlow (Pan 2025) 的前缀调度论文；前缀缓存要求 \(\Delta=0\)。本文研究 **token ID 相同、位置不同**（\(\Delta \neq 0\)）的文件模块拷贝。

历史 V40–V46 / RepoBench 笔记（`docs/kvflow/COLLABORATOR_QUICKSTART_20260729.md`、`KVFLOW.md`）**不是** ASPLOS headline。从本文件开始。

---

## 1. 方法（Headline 只用 M0+M2）

1. **M0 admit**：单文件 `repository_code`、版本有效、token-ID 相同、\(\Delta \neq 0\)。离线 oracle，不估 Attention。
2. **M2 copy**：source 侧预旋转 \(K\)，\(V\) 原样；页键 `(source_prefix_hash, content_hash, Δ)`。
3. **Fail-closed**：hash / 覆盖 / alloc 失败 → 整岛 Dense，从不半页拼接。
4. **M1 prefix / M3 prefetch**：代码里有，Headline 战役关掉，主表不能被 radix 或 prefetch 领功。

引擎入口：

```
python/sglang/srt/mem_cache/kvcomm_exact.py
python/sglang/srt/mem_cache/kvcomm/{types,store,radix_backend,transfer}.py
python/sglang/srt/mem_cache/kvcomm_prefetch/   # M3，Headline 关闭
```

---

## 2. 克隆与单测（无 GPU）

```bash
git clone git@github.com:ccdd2023/sglang.git sglang-kvflow
cd sglang-kvflow
git checkout integration/template-prefetch-swebench
export PYTHONPATH="$PWD/python"

python -m pytest -q \
  python/sglang/srt/mem_cache/test_kvcomm_exact.py \
  python/sglang/srt/mem_cache/kvcomm/test_core.py \
  python/sglang/srt/mem_cache/kvcomm/test_radix_backend.py \
  python/sglang/srt/mem_cache/kvcomm_prefetch/test_*.py
```

论文数字门（需要冻结产物，见 §4）：

```bash
cd docs/kvflow/paper
python3 scripts/check_asplos_claims.py          # 必须 PASS
python3 -m pytest -q scripts/
# 可选：bash compile.sh   # 需 TeX Live + acmart
```

checker 会沿目录向上找到本仓库的 `kvcomm_exact.py`。换引擎树时：

```bash
export IMPACTKV_ENGINE_ROOT=/path/to/this/repo
```

---

## 3. Headline 数字（禁止手改 RESULT）

| 战役 | 作业 | 引用 | 数字 |
|---|---|---|---|
| **7B 主表** | 137185 | `tab:eval-summary` | **1.492×** / **1684/1684** / 赢率 **99.3%** / agree **93.6%**（不是 Accuracy）。prefetch off，prefix off |
| 30B 附录 | 96092 | `tab:eval-30b` only | **1.375×** / 1684/1684 / 94.8%。**96.5% 禁止出现在 7B 正文** |
| 7B prefix-on | 139839 | `tab:7b-prefix-on` | prefix-only 1.526× / lossy-only 1.408× / dual 2.120× / 增量 1.390×。不混进主表 |
| 同引擎 copier | 137400 | `tab:admit-ablation` | 文件岛 1.492×/93.6%；KVCOMM-style 2.100×/89.4%；CacheBlend-style 1.883×/91.9%。**不是原生栈** |

不要报 N=4（0.905× / 0.841× / `tab:nuse`）。不要写 SOTA。不要把 \(1.492\times\) 和 \(1.375\times\) 写成一个 official method。

---

## 4. 冻结产物（复现 checker / 画图）

产物 **不在 git 里**（数 GB）。本集群默认：

```bash
export IMPACTKV_ARTIFACTS=/home/gfy/CodeMAS_Project/kvflow-artifacts
```

换机器：把该目录拷过来，或 `export IMPACTKV_ARTIFACTS=/path/to/kvflow-artifacts`。

至少需要这些子目录（PLAN.json 约 40MB，含 token ids；dense/reuse json 更大，重跑 GPU 才要）：

| 目录 | 用途 |
|---|---|
| `impactkv_swebench_7b_file_modules_prefixkey_20260824/` | 7B 主表 + MOTIVATION/SLICES/PLAN |
| `impactkv_swebench_prerotated_file_modules_20260818/` | 30B 附录 |
| `impactkv_swebench_7b_sota_copiers_20260824/` | admit 克隆 |
| `impactkv_swebench_7b_prefix_on_20260825/` | prefix-on |
| `impactkv_swebench_template_prefetch_nextisland_20260821/` | prefetch 附录 |
| `impactkv_global_block_attention_20260806/frozen26_r2/` | 3B TV |
| `impactkv_attention_sparsity_20260806/frozen20/` | 3B 稀疏 |
| `impactkv_common_prompt_attention_kv_mechanism_20260813/` | 四臂热力图 |

**不要 `git add` 这些产物。** 缺目录时 checker 会失败并指出路径。

---

## 5. 复现 7B Headline（GPU，不要随手重跑）

数字已冻结。只有 checker 失败或引擎被改坏时才重跑，且**禁止覆盖** 137185/96092 的 `RESULT.json`。

```bash
# 本仓库根目录
python benchmark/multi_workflow/prepare_7b_swebench_file_modules_plan.py

# 提交（exclude gpu[10-13,15,17,23-24]；不要 login/debug GPU）
sbatch benchmark/multi_workflow/slurm/swebench_7b_file_modules.sbatch
```

GPU 禁止 `gpu[11-13]`。不要前台等 Slurm。

跑完后对照冻结 `RESULT.json` 的 `cache_ready_speedup_ratio_of_means` ≈ 1.4919，`copy_events == 1684`，`prefetch is False`。

Prefix-on / copier / prefetch 的 sbatch 在同一 `slurm/` 目录。

---

## 6. 改代码 / 改论文

| 你要改 | 去哪 |
|---|---|
| 拷贝内核、页键、fail-closed | `python/sglang/srt/mem_cache/kvcomm_exact.py`、`kvcomm/` |
| Prefetch（非 Headline） | `kvcomm_prefetch/` |
| SWE-bench 回放战役 | `benchmark/multi_workflow/run_swebench_*.py` |
| 论文正文 | `docs/kvflow/paper/sections/*.tex` |
| 评测图 | `docs/kvflow/paper/scripts/build_7b_eval_figures.py` |
| Motivation 折线 | `docs/kvflow/paper/scripts/build_motivation_heatmaps.py` |
| 声称是否合法 | `docs/kvflow/paper/scripts/check_asplos_claims.py` 必须 PASS |

脏 worktree：**禁止** `git add -A` / `git clean` / `git reset --hard`。只 add 点名的 ImpactKV 文件。  
推送目标：`origin integration/template-prefetch-swebench`。

---

## 7. 仓库地图

```
sglang/                                      # 本仓库，本 branch
  IMPACTKV.md                                # 本文件（合作者从这里开始）
  python/sglang/srt/mem_cache/kvcomm_exact.py
  python/sglang/srt/mem_cache/kvcomm/
  python/sglang/srt/mem_cache/kvcomm_prefetch/
  benchmark/multi_workflow/run_swebench_* / prepare_* / slurm/
  docs/kvflow/paper/                         # ASPLOS 论文源 + checker
  docs/kvflow/                               # 历史开发笔记（V40–V46，不是 headline）
```

冻结战役 JSON 在集群 `kvflow-artifacts/`，用 `IMPACTKV_ARTIFACTS` 指向它。
