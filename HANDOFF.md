# ImpactKV 合作者接手文档（ASPLOS 2027）

**读这一份就能开工。** 你只需要 GitHub 上的这一个仓库，在你自己的机器上工作。  
没有原集群账号，也不需要任何第二份 repo。

| | |
|---|---|
| 仓库 | [`ccdd2023/sglang`](https://github.com/ccdd2023/sglang) |
| 分支 | **`integration/template-prefetch-swebench`** |
| 命令速查 | [`IMPACTKV.md`](IMPACTKV.md) |
| 论文源 + checker | [`docs/kvflow/paper/`](docs/kvflow/paper/) |
| 中文论证（改稿用，不是投稿稿） | [`docs/kvflow/paper/PAPER_LOGIC_CN.md`](docs/kvflow/paper/PAPER_LOGIC_CN.md) |
| 冻结 PLAN / RESULT | [`benchmark/multi_workflow/offcluster/`](benchmark/multi_workflow/offcluster/)（解压到本机） |
| 投稿 | ASPLOS 2027，`acmart` `sigplan,anonymous,review,nonacm`，**正文 ≤ 11 页** |

历史笔记 `KVFLOW.md`、`docs/kvflow/COLLABORATOR_QUICKSTART_20260729.md`、V40–V46 / RepoBench **不是**当前 headline，不要从那里开始。

---

## 0. 你接手的是什么

编码 agent 会在**不同 prompt 位置**重读同一份仓库文件：token ID 相同，RoPE 相位和左右上下文不同。前缀缓存（要求 \(\Delta=0\)）因此 miss。

ImpactKV 做的是 **coding-aware true-lossy 文件岛 KV 拷贝**：

1. **Admit（M0）**：只拷单文件 `repository_code`、版本仍有效、token-ID 相同、\(\Delta \neq 0\)。离线 oracle，**不估 Attention**。
2. **Copy（M2）**：source 侧预旋转 \(K\)，\(V\) 原样。页键 `(source_prefix_hash, content_hash, Δ)`。
3. **Fail-closed**：hash / 覆盖 / alloc 失败 → **整岛 Dense**，从不半页拼接。

评测范围写死：**sequential one-token prefill vs 同一引擎 Dense**（source KV 已在，叫 cache-ready TTFT）。  
**不是** e2e serving、吞吐皇冠、SWE-bench resolved / Accuracy。

Headline **只用 M0+M2**。代码里的 radix 前缀（M1）和 prefetch（M3）在 7B 主表战役里是关掉的，主表不能被它们领功。

---

## 1. 第一天（约 2 小时，可以没有 GPU）

```bash
git clone git@github.com:ccdd2023/sglang.git sglang-kvflow
cd sglang-kvflow
git checkout integration/template-prefetch-swebench

# 按 docs/get_started/install.md Method 2 安装「这个 checkout」
# 不要 pip install sglang 去装上游包
pip install -e "python"
export PYTHONPATH="$PWD/python:$PWD"

python -m pytest -q \
  python/sglang/srt/mem_cache/test_kvcomm_exact.py \
  python/sglang/srt/mem_cache/kvcomm/test_core.py \
  python/sglang/srt/mem_cache/kvcomm/test_radix_backend.py \
  python/sglang/srt/mem_cache/kvcomm_prefetch/test_*.py

python benchmark/multi_workflow/fetch_impactkv_artifacts.py
export IMPACTKV_ARTIFACTS="$PWD/impactkv-artifacts"

cd docs/kvflow/paper
python3 scripts/check_asplos_claims.py    # 必须 PASS
python3 -m pytest -q scripts/             # 29 passed
```

过了这四步，环境就算接上了。  
`impactkv-artifacts/` 在 gitignore 里，**不要** `git add` 它。

有 GPU 时先冒烟，不要一上来全量 235 组：

```bash
huggingface-cli download Qwen/Qwen2.5-Coder-7B-Instruct \
  --local-dir "$HOME/models/Qwen2.5-Coder-7B-Instruct"
export IMPACTKV_MODEL="$HOME/models/Qwen2.5-Coder-7B-Instruct"
source benchmark/multi_workflow/impactkv_local_env.sh
export IMPACTKV_MAX_GROUPS=2
bash benchmark/multi_workflow/run_impactkv_headline.sh
```

冒烟目录里的 `RESULT.json` 状态是 **`SMOKE`**，不能写进论文。

---

## 2. 冻结合同（改任何东西之前先读）

这些数字已经进论文和 checker。**禁止手改对应 `RESULT.json`。**

| 战役 | 作业 | 写进论文的位置 | 冻结数字 |
|---|---|---|---|
| **7B 主表** | **137185** | `tab:eval-summary` | cache-ready **1.492×** / **1684/1684** / 赢率 **99.3%** / one-token **93.6%**。prefetch **off**，prefix **off** |
| 30B 附录 | **96092** | **只** `tab:eval-30b` | **1.375×** / 1684/1684 / agree **94.8%** |
| 7B prefix-on | **139839** | `tab:7b-prefix-on` | prefix-only 1.526× / lossy-only 1.408× / dual 2.120× / 增量 1.390× |
| 同引擎 copier | **137400** | `tab:admit-ablation` | 文件岛 1.492×/93.6%；KVCOMM-style 2.100×/89.4%；CacheBlend-style 1.883×/91.9%。**不是原生栈** |

7B PLAN 动机（无新 GPU，来自 `MOTIVATION.json`）：copied 均值 **1537** / prompt **4433**。不要写成 30B 的 1528/4403。

3B probe（不是 7B/30B 速度）：suffix TV **0.00264**，formation **0.0462**，top-10% **80.1%**。TV **不绑** 1.492× 或 1.375×。

### 禁止

1. 改 137185 / 96092 的 `RESULT.json`，或把你自己 GPU 上的 TTFT 写回冻结文件。
2. 把 **1.375×** 放进 `tab:eval-summary`；把 **96.5%** 写进 7B 正文（含两个摘要）。
3. 把 1.492× 和 1.375× 写成「一个 official method」（两套 checkpoint）。
4. 报 N=4（0.905× / 0.841× / `tab:nuse`）。RESULT 里仍有 n4 字段给 checker 校验，论文不报。
5. 写 **SOTA**。写 serving 吞吐皇冠。把 one-token agreement 叫 Accuracy / SWE-bench resolved。
6. 把 7B DS-1000 Accuracy 混进 TTFT 主表。
7. 引用作业 **135877 的 1.525×**（7B 上用了错误 RoPE）。
8. 给原生 CacheBlend / KVCOMM serving 栈排绝对 TTFT（拓扑不同）。对照只许 **同引擎政策克隆**（137400）。
9. 用 `\vspace` 硬挤页数（checker 会抓）。
10. `git add -A` / `git clean` / `git reset --hard`。只 add 点名的 ImpactKV 文件。
11. 把 `status != COMPLETE` 的 RESULT 写进论文。Slurm COMPLETED ≠ RESULT COMPLETE。冒烟 `SMOKE` 更不行。

改论文或数字相关脚本后，必须：

```bash
export IMPACTKV_ARTIFACTS="$PWD/impactkv-artifacts"
cd docs/kvflow/paper && python3 scripts/check_asplos_claims.py
```

**PASS 才能认为声称合法。**

---

## 3. 四个模块（Headline 只用两个）

在线路径：Dense prefix → M2 拷已 admit 的岛 → Dense remainder → 1-token decode。

| 模块 | 作用 | 作业 137185 |
|---|---|---|
| **M0** 编译器 | 冻结轨迹 → PLAN：hash、\((s,t,L)\)、\(\Delta\neq 0\) | **开**（headline 仍是离线 oracle） |
| **线上 admit** | source 只看协议（单文件 / 版本有效 / later-roles）；target 用 token 身份绑定，K 不预旋转 | **关**（`SGLANG_KVCOMM_ONLINE_ADMIT=1` 才开；不改 137185） |
| **M1** 前缀复用 | radix LCP，不拷岛 | **关** |
| **M2** lossy 拷贝 | \(K \leftarrow R_\Delta K\)，\(V\) 原样；失败则 Dense | **开** |
| **M3** prefetch | later-roles 驻留；miss 必须退化到仍持有的 M2 拷贝 | **关** |

PLAN 阶段丢掉 **48** 个 \(\Delta=0\) 岛，避免实验退化成前缀缓存。  
235 组、421 岛；组内 1/2/3 岛 = 89 / 106 / 40。

---

## 4. 仓库地图（只动这些）

```
sglang/                                    # 本 branch
  HANDOFF.md                               # 本文件
  IMPACTKV.md                              # 命令速查
  python/sglang/srt/mem_cache/
    kvcomm_exact.py                        # fail-closed 拷贝；source 侧 K 预旋转
    kvcomm/types.py                        # 页键含 source_prefix_hash + Δ
    kvcomm/store.py                        # leased 同 key 复用
    kvcomm/radix_backend.py                # lookup-before-alloc
    kvcomm/transfer.py
    kvcomm_prefetch/                       # M3，Headline 关闭
  benchmark/multi_workflow/
    fetch_impactkv_artifacts.py            # 解压冻结 JSON
    run_impactkv_headline.sh               # 本机 7B 战役（新目录）
    impactkv_local_env.sh                  # 无集群路径
    offcluster/impactkv-claim-pack.tar.gz  # ~15MB，进 git
    run_swebench_prerotated_file_modules.py
    run_swebench_7b_prefix_on.py
    run_swebench_7b_sota_copiers.py
    run_swebench_template_prefetch.py
    qwen3_coder_tool_chat_template.jinja   # 7B 战役也用这个，不要换
    slurm/swebench_*.sbatch                # 可选；已相对本仓库
  docs/kvflow/paper/
    main.tex / main_article.tex
    sections/*.tex
    figures/
    scripts/check_asplos_claims.py
    PAPER_LOGIC_CN.md / PAPER_LOGIC_CN.pdf
    compile.sh
```

解压后（`IMPACTKV_ARTIFACTS`）：

| 目录 | 角色 |
|---|---|
| `impactkv_swebench_7b_file_modules_prefixkey_20260824/` | 7B 主表 137185 |
| `impactkv_swebench_prerotated_file_modules_20260818/` | 30B 附录 96092 |
| `impactkv_swebench_7b_sota_copiers_20260824/` | 同引擎 copier 137400 |
| `impactkv_swebench_7b_prefix_on_20260825/` | prefix-on 139839 |
| `impactkv_swebench_template_prefetch_nextisland_20260821/` | prefetch 附录 |
| `impactkv_global_block_attention_20260806/frozen26_r2/` | 3B TV |
| `impactkv_attention_sparsity_20260806/frozen20/` | 3B 稀疏 |
| `impactkv_common_prompt_attention_kv_mechanism_20260813/` | 四臂热力图 |

GPU 重跑必须把 `PLAN.json` **拷到** `impactkv-artifacts/runs/...` 再跑。冻结目录只读。

---

## 5. 论文怎么改

叙事顺序（`main.tex` 的 `\input` 必须保持）：

1. Introduction  
2. Background（含 Limitation）  
3. Motivation（折线/热力图，**不要表格**）  
4. Design（`problem.tex`）  
5. M0 编译器（`template.tex`）  
6. M2 拷贝（`kv-management.tex`）  
7. Implementation  
8. Evaluation：Setup → overall → ablation → sensitivity  
9. Related work / Conclusion / Appendix  

中文论证按图走：`docs/kvflow/paper/PAPER_LOGIC_CN.md`。

编译：

```bash
cd docs/kvflow/paper
bash compile.sh          # 需 TeX Live + acmart；正文必须 ≤ 11 页
```

Checker 还会查：Motivation 里不能 `\begin{table`；必须有 `fig:motivation-coverage` / `fig:attn-proxy` 等。  
改图用 `scripts/build_motivation_heatmaps.py`、`scripts/build_7b_eval_figures.py`（只读冻结 JSON，无 GPU）。

---

## 6. 在你自己的 GPU 上复现

机器：Headline 7B 约 **24GB**（论文是 RTX 4090、bf16、context 32k）。  
不需要 Slurm / Enroot / mini-SWE-agent（这是 exact-prompt 回放，不是 agent 评测）。

```bash
export IMPACTKV_MODEL="$HOME/models/Qwen2.5-Coder-7B-Instruct"  # 必须含 config.json
unset IMPACTKV_MAX_GROUPS
bash benchmark/multi_workflow/run_impactkv_headline.sh
```

全量 235 组 Dense + reuse，一张 4090 要很多小时。  
你机器上的 TTFT **不会**和 1.492× bit 一致。机械检查：

- `copy_events == 1684`
- `fallback_events == 0`
- `prefetch is False`
- one-token agreement 对照冻结 93.6%（允许因采样/内核略偏，但不要改冻结文件）

30B 附录（可选）：

```bash
huggingface-cli download cyankiwi/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit \
  --local-dir "$HOME/models/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit"
export IMPACTKV_MODEL="$HOME/models/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit"
# 把 30B PLAN 拷到新 runs/ 目录后调用 run_swebench_prerotated_file_modules.py
```

Prefix-on / copier：`run_swebench_7b_prefix_on.py`、`run_swebench_7b_sota_copiers.py`，同样拷 PLAN 到新目录。

有自己的 Slurm 时，从**仓库根目录** `sbatch`，或先 `export IMPACTKV_PROJECT=/path/to/this/clone`。脚本已不再写死原集群路径。

默认 `IMPACTKV_HF_OFFLINE=1`：服务器不联网拉权重，模型必须是本地目录。

---

## 7. 日常工作流

1. 开新改动前：`check_asplos_claims.py` 已是 PASS。  
2. 改引擎 → 跑第四节的 kvcomm pytest。有 GPU 再 `IMPACTKV_MAX_GROUPS=2` 冒烟。  
3. 改论文 → 再跑 checker；正文仍 ≤ 11 页。  
4. 提交：

```bash
git add -- <点名的文件>
git commit -m "..."
git push origin integration/template-prefetch-swebench
```

不要 `git add -A`。这个 fork 里有大量历史实验脚本和上游 SGLang 文件，全加会把无关改动推进去。

---

## 8. 现在已经落地、你可以接着做的

已经在这个 branch 里：

- 文件岛拷贝内核 + fail-closed  
- 7B / 30B / prefix-on / 同引擎 copier 冻结数字与 PLAN  
- ASPLOS 论文源、图、声称 checker  
- 离线 claim pack + 本机运行脚本  

合理的后续（都不要破坏冻结合同）：

- 论文叙事、图表、Related work 打磨；checker 保持 PASS  
- 从冻结 JSON 加 ablation / sensitivity 图（无新 GPU 也可以）  
- 引擎可读性、测试、fail-closed 边角  
- 在你的 GPU 上独立复现，把 **新目录** 的 RESULT 当交叉验证，不当论文替换值  
- M1/M3 可以做附录或讨论，**不能**领 1.492× 的功  

不必再做、也不是欠债：N=4 账单、500-task、并发 P99、原生 CacheBlend 绝对 TTFT、C=4。

---

## 9. 常见坑

| 坑 | 正确做法 |
|---|---|
| `pip install sglang` | 从这个 checkout `pip install -e python` |
| 模型目录只有 `tokenizer.json` | HF 下完整 snapshot，要有 `config.json` |
| 在冻结目录里跑战役 | 拷 PLAN 到 `impactkv-artifacts/runs/` |
| 把冒烟 RESULT 写进论文 | `status` 必须是 COMPLETE，且作业对得上 |
| Chat template 换成 Qwen2.5 默认 | 7B 战役冻结的是 `qwen3_coder_tool_chat_template.jinja` |
| 从 `COLLABORATOR_QUICKSTART_20260729.md` 开工 | 那是 V46，忽略 |
| 页键去掉 prefix hash | 会混不同左上下文的页（旧作业 137091 的教训） |
| Prefetch miss 就 Dense 整岛 | miss 必须落到仍持有的 M2 copy |

有疑问时以 **checker PASS + 本文件的冻结合同** 为准，不以口头速度数字为准。
