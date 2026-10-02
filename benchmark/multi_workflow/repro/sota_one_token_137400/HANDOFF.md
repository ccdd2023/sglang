# 接手：7B 同一引擎 one-token 对照（作业 137400）

合作者拉下分支 `repro/sota-one-token-137400` 后，按本文可以重跑那一次能和有损方法比精度的实验。仓库是 `git@github.com:ccdd2023/sglang.git`。

这一次比的是下一 token 是否和 Dense 相同，外加 cache-ready TTFT。不是 SWE-bench official resolved，也不是原生 vLLM CacheBlend / 原生 KVCOMM。原生栈用另一套执行器，不在这个分支里。

## 冻结结果（RTX 4090，Qwen2.5-Coder-7B-Instruct）

作业 137400，235 组，705 对，温度 0，每次生成 1 个 token。1 轮预热不计入，后面 3 轮计入。前缀复用关闭，prefetch 关闭。源页用单独一次生成写入，再测目标。`RESULT.json` 的 sha256 是 `babd749f4754dd935ced09d7cc4dabe2f7eeb0798e64973f858b3cddbfc61620`。

| 方法 | 相对 Dense 的 cache-ready | 与 Dense 的下一 token 一致率 | 拷贝 |
|---|---:|---:|---:|
| Dense | 1.000× | — | 0 |
| 文件岛 `coding_natural_code_cost` | 1.492× | 93.6%（660/705） | 1684/1684 |
| KVCOMM 风格，无约束 LCS | 2.100× | 89.4% | 948/948 |
| CacheBlend 风格，每岛前 15% 重算、其余拷贝 | 1.883× | 91.9% | 948/948 |

速度会随机器变化。复现是否对齐，看 `status=COMPLETE`、三臂一致率、拷贝次数。不要把这张表写成官方 resolved，也不要写成原生 SOTA。

## 三套方法分别是什么

三臂走同一个旋转拷贝内核：K 按位移旋转，V 原样，失败则整岛重算。差别只在计划里允许拷哪些 span。

- 文件岛：`PLAN.coding.json`，`policy_label=coding_natural_code_cost`。只收协议切出的单文件岛。421 个岛。
- KVCOMM 风格：`prepare_7b_sota_copier_plans.py` 用 `prepare_swebench_general_lcs_plan.build_groups` 在同一批 token 上做无约束移位 LCS，`policy_label=general_shifted_lcs`。237 个岛。工具输出、命令、助手回合也会进来。
- CacheBlend 风格：同一批 LCS 岛，每岛丢掉前 15%（`shrink_island_for_blend`，`BLEND_RATIO=0.15`），剩下的仍走同一拷贝内核。`policy_label=cacheblend_style_blend15`。这不是 CacheBlend 论文里的选择性重算融合。

入口是 `benchmark/multi_workflow/run_swebench_7b_sota_copiers.py`。它按 dense、文件岛、KVCOMM 风格、CacheBlend 风格顺序各起一次服务。

## 拉代码

```bash
git clone git@github.com:ccdd2023/sglang.git
cd sglang
git fetch origin repro/sota-one-token-137400
git checkout repro/sota-one-token-137400
```

计划文件：

`benchmark/multi_workflow/repro/sota_one_token_137400/PLAN.coding.json`

sha256 `2cf6f1fd7324d18bd859ec9e312f4b240bd8e05a754c18d4236dc51f9d9f1dce`。235 组，模型名必须是 `Qwen2.5-Coder-7B-Instruct`。另外两份计划由准备脚本从这份生成，不要手改。

## 环境

一块能放下 7B bf16 的 NVIDIA GPU。原测量是 RTX 4090，静态显存比例 0.82。Python 需要能 `import sglang`，并装有本仓库的运行依赖（`pip install -e ".[all]"` 或你们现成的 sglang 环境）。

模型放到默认路径，或导出 `IMPACTKV_MODEL`：

```bash
huggingface-cli download Qwen/Qwen2.5-Coder-7B-Instruct \
  --local-dir "$HOME/models/Qwen2.5-Coder-7B-Instruct"
```

请求直接送 `input_ids`，温度 0。不要换 tokenizer，也不要改计划里的 token。

## 跑

在仓库根目录：

```bash
bash benchmark/multi_workflow/repro/sota_one_token_137400/run.sh
```

脚本会：

1. 把四个质量旋钮显式写成 0，从而关掉现在默认的 center-8 服务包。不写的话，未设置旋钮会走 `shift_dense_limit=1500`、`center_copy=1`、`right_seam_anchor=8`，拷贝次数对不上 1684/948。
2. 关掉 prefetch、producer-extract、扩展模块、L1、online admit。
3. 从 `PLAN.coding.json` 生成 `PLAN.kvcomm.json` 和 `PLAN.cacheblend.json`。
4. 调用 `run_swebench_7b_sota_copiers.py`。

产物默认在 `impactkv-artifacts/runs/sota_one_token_137400/`。可把别的目录作为第一个参数。整次大约数小时，取决于 GPU。已有 `dense.json` 或 `RESULT.coding.json` 时，对应臂会跳过。

跑完后对照：

```bash
python - << 'PY'
import json
from pathlib import Path
root = Path("impactkv-artifacts/runs/sota_one_token_137400/RESULT.json")
got = json.loads(root.read_text())
exp = json.loads(Path("benchmark/multi_workflow/repro/sota_one_token_137400/RESULT.json").read_text())
assert got["status"] == "COMPLETE"
for key in ("coding", "kvcomm_style", "cacheblend_style"):
    g = got[key]["one_token_output_agreement"]["fraction"]
    e = exp[key]["one_token_output_agreement"]["fraction"]
    copies = got[key]["mechanism"]["copy_events"]
    print(key, "agree", round(g, 4), "expected", round(e, 4), "copies", copies)
PY
```

## 不要做的事

- 不要打开 `SGLANG_KV_PRODUCER_EXTRACT`。那是后来的记账，会把这次标成不是 137185/137400。
- 不要把工作区里未提交的 `run_swebench_prerotated_file_modules.py` 改动混进这次复现。这个分支用的是已提交的运行器。
- 不要把 1.492× 和含额外 source 预填的 0.872× 写成同一个数。
- 不要把 50 题 official resolved（四臂都是 0/50）写进这张表。
