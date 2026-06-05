# Cross-Model Transferability of predicted_distance_table

> Per-model re-run of `results/same_code_context_variation/` on 4 models.

## 1. Models studied

- Qwen/Qwen2.5-Coder-7B-Instruct  (loaded)
- Qwen/Qwen2.5-Coder-3B-Instruct  (loaded)
- Qwen/Qwen2.5-7B-Instruct  (loaded)
- Qwen/Qwen3-8B  (missing)

## 2. Per-axis d_norm per model

### 2.1 by_position_offset

- **Qwen2.5-Coder-7B**: 0=2.176, 5-25=2.328, 50-100=2.538
- **Qwen2.5-Coder-3B**: 0=2.005, 5-25=2.150, 50-100=2.390
- **Qwen2.5-7B**: 0=2.121, 5-25=2.262, 50-100=2.459

### 2.2 by_system_prompt_class

- **Qwen2.5-Coder-7B**: coder=2.425, planner=2.067, reviewer=2.428, tester=2.470
- **Qwen2.5-Coder-3B**: coder=2.254, planner=1.948, reviewer=2.242, tester=2.283
- **Qwen2.5-7B**: coder=2.374, planner=2.007, reviewer=2.344, tester=2.397

### 2.3 by_surrounding_code_class

- **Qwen2.5-Coder-7B**: class_wrap=2.363, imports_wrap=2.411, none=2.270, try_wrap=2.346
- **Qwen2.5-Coder-3B**: class_wrap=2.194, imports_wrap=2.240, none=2.109, try_wrap=2.184
- **Qwen2.5-7B**: class_wrap=2.298, imports_wrap=2.332, none=2.209, try_wrap=2.284

## 3. Pairwise mean |Δd_norm|

See `plots/cross_model_d_norm_heatmap.png` for the 4×4 matrix.

| pair | mean |Δd_norm| |
|---|---|
| Qwen2.5-Coder-7B-Instruct  vs  Qwen2.5-Coder-3B-Instruct | 0.0000 |
| Qwen2.5-Coder-7B-Instruct  vs  Qwen2.5-7B-Instruct | 0.0000 |
| Qwen2.5-Coder-3B-Instruct  vs  Qwen2.5-7B-Instruct | 0.0000 |

## 4. Verdict

**Strong portable**: tables agree within ±0.15 d_norm on average → 7-8B model-agnostic

## 5. Plots

- ![](cross_model_d_norm_heatmap.png)
- ![](d_norm_by_position_offset.png)
- ![](d_norm_by_surrounding_code_class.png)
- ![](d_norm_by_system_prompt_class.png)
