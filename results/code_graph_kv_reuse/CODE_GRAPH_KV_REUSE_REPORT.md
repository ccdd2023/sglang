# Code Graph-Aware Lossy Reuse Precision Study

> 自动生成：`results/code_graph_kv_reuse/code_graph_bundle_analyzer.py`

## 1. 这项实验回答什么

已有 AST 粒度实验回答的是“代码块切多大更稳定”。本实验进一步问：在真实 SWE-style 代码修改中，调用、导入和测试触达关系能否帮助选择 **lossy reuse 精度更高** 的 exact code bundle。

这里的 token 数不是优化目标，只作为 scope covariate 记录。我们后续会靠调度和预取处理执行成本；本贡献主要关心：哪类 code graph bundle 在跨 agent、跨 prompt 位置时 KV 更稳定，输出漂移更小，pass@1 损失更可控。

换句话说，AST 是 span boundary，code graph 是 precision-oriented bundle selection signal，安全 gate 仍然是 exact normalized content signature。

## 2. 工具链

- 解析器：Python 标准库 `ast`，用于函数、方法、类、import 和 call expression 抽取。
- 图构建：轻量静态 call graph。`ast.Call` 的 `Name`/`Attribute` 会解析到同文件优先、再 repo-local 同名 symbol。
- Import resolver：将 `a.b` 映射到 repo 内 `a/b.py` 或 `a/b/__init__.py`，只保留可静态定位的本地依赖。
- Test bundle：从 `test_patch` 和 FAIL_TO_PASS 对应的 Python 测试文件中抽取 `test*` 函数。
- 可选重工具：PyCG、CodeQL、Jedi 目前只作为后续 robustness，不进入第一版必需依赖。

参考关系：PyCG 说明 Python 静态 call graph 的可行性；CodeQL 的 data-flow/call graph 说明 AST 之外的程序关系可以作为代码理解对象；Tree-sitter/Jedi 可在后续扩展到多语言或更强 definition/reference resolution。本实验第一版刻意不用这些重依赖，以保证单机可复现。

## 3. 数据设置

- Manifest：`results/selective_ast_reuse/data/swe_selective_wholefile_68k_1file_instances.json`
- 分析 case：28
- 目标 symbol：51
- 派生 bundle 记录：255
- Precision manifest 行数：765，即每个 bundle 生成 planner/coder/reviewer 三角色 prompt，用于 paired KV distance 和 output-drift 实验。
- Bundle 类型：`ast_function_only`、`call_neighborhood_1hop`、`reverse_callers_1hop`、`import_dependency_bundle`、`test_target_bundle`

## 4. Bundle 定义

- `ast_function_only`：patch 命中的最小函数/方法/类 span。
- `call_neighborhood_1hop`：target 加上静态解析到的直接 callee。
- `reverse_callers_1hop`：target 加上 repo 内直接 caller。
- `import_dependency_bundle`：target 加上本文件 import front matter 和可定位本地 import 文件的 import front matter。
- `test_target_bundle`：target 加上变更测试文件中的 `test*` 函数。

## 5. Precision-first 静态结果

下表中的 token/symbol 不是“越小越好”的结论，而是后续解释精度差异时的控制变量。真正要比较的是下一阶段的 cross-role KV distance、output F1/drift 和 paired pass@1 non-degradation。

| bundle | n | mean scope tokens | p90 scope tokens | scope expansion | mean symbols | exact signature hit | precision priority |
|---|---:|---:|---:|---:|---:|---:|---|
| `ast_function_only` | 51 | 495.6 | 1182.0 | 1.00 | 1.00 | 1.00 | `medium` |
| `call_neighborhood_1hop` | 51 | 1720.6 | 3698.0 | 3.12 | 8.25 | 1.00 | `high` |
| `reverse_callers_1hop` | 51 | 3337.0 | 9748.0 | 20.95 | 6.69 | 1.00 | `diagnostic` |
| `import_dependency_bundle` | 51 | 599.7 | 1324.0 | 2.10 | 1.00 | 1.00 | `high` |
| `test_target_bundle` | 51 | 785.1 | 1422.0 | 4.56 | 4.29 | 1.00 | `medium` |

## 6. Bundle 示例

| bundle | case | target file | target symbol | scope tokens | included symbols |
|---|---|---|---|---:|---|
| `ast_function_only` | `pallets__flask-5014` | `src/flask/blueprints.py` | `Blueprint.__init__` | 247 | `Blueprint.__init__` |
| `call_neighborhood_1hop` | `pallets__flask-5014` | `src/flask/blueprints.py` | `Blueprint.__init__` | 583 | `Blueprint.__init__; BlueprintSetupState.__init__` |
| `reverse_callers_1hop` | `pallets__flask-5014` | `src/flask/blueprints.py` | `Blueprint.__init__` | 3995 | `Blueprint.__init__; Blueprint` |
| `import_dependency_bundle` | `pallets__flask-5014` | `src/flask/blueprints.py` | `Blueprint.__init__` | 287 | `Blueprint.__init__` |
| `test_target_bundle` | `pallets__flask-5014` | `src/flask/blueprints.py` | `Blueprint.__init__` | 830 | `Blueprint.__init__; test_blueprint_specific_error_handling; test_blueprint_specific_user_error_handling; test_blueprint_app_error_handling` |

![Bundle scope covariate](/home/gfy/CodeMAS_Project/sglang-kvflow/results/code_graph_kv_reuse/figures/fig_code_graph_bundle_scope.png)

![Precision design space](/home/gfy/CodeMAS_Project/sglang-kvflow/results/code_graph_kv_reuse/figures/fig_code_graph_precision_design_space.png)

![Bundle diagnostics](/home/gfy/CodeMAS_Project/sglang-kvflow/results/code_graph_kv_reuse/figures/fig_code_graph_bundle_diagnostics.png)

## 7. 论文可用解释

Code graph-aware bundling is not a safety mechanism. It decides which exact code spans should be compared, retained, or prefetched together for lossy reuse. The actual reuse gate remains the normalized content signature and token-level exact match. This distinction lets AgentTemplateKV use code structure to improve precision-oriented candidate selection while preserving exact-content safety.

在论文中可以把它写成一个 design validation：当 function/method 缺少局部上下文时，调用邻域、导入依赖和测试触达关系提供了比盲目扩展到 file prefix 更精细的 lossy-reuse precision 策略。file prefix 可以继续作为高复用稳定前缀候选，但不是本节关注点。

## 8. 下一步实验接口

本报告已经输出 `data/code_graph_bundle_table.csv`，其中每行都有 `bundle_type`、`target_file`、`target_symbol`、`token_count`、`content_signature`、`precision_priority`、`files` 和 `symbols`。同时输出 `data/code_graph_precision_manifest.jsonl`，其中包含 planner/coder/reviewer 三角色 prompt 和完整 bundle 文本。

当前完成的是 P-G0 precision scaffold。这里的 scope tokens、bundle expansion 和 exact signature 只能证明“这些 code graph bundle 可以被定义和追踪”，不能直接证明 KV distance、TTFT 或 accuracy。P-G1/P-G2/P-G3 必须继续用同一 manifest 跑 paired KV distance、output drift 和 pass@1 non-degradation，才能进入精度结果表。

建议优先顺序：先比较 `ast_function_only`、`import_dependency_bundle`、`call_neighborhood_1hop` 三类；`test_target_bundle` 用于 SWE-style output drift；`reverse_callers_1hop` 只作为诊断上界，不作为默认策略。

## 9. 边界

- Python-only；动态 dispatch、monkey patch、反射调用不会被完整解析。
- 当前 call graph 是 conservative locator，不保证语义完整性。
- `exact_signature_hit_rate=1.0` 表示派生 bundle 自身有 exact signature，并不表示线上 cache 一定 device-hit。
