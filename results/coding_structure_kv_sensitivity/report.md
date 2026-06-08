# Coding-Structure KV Sensitivity

This experiment keeps target code bytes identical and varies only the coding-agent prompt structure around that code. It measures K/V distance on the target code span, not on the whole prompt.

## Setup

- Model: `Qwen/Qwen2.5-Coder-7B-Instruct`
- Canonical cell: `planner / code_first`
- Selected layers: `[-1, -2, -3, -4]`
- Variations: `96`

## Overall

- n = 96
- mean d_norm = 0.626
- p90 d_norm = 0.918
- max d_norm = 1.021

## By Coding Structure

| Bucket | n | mean | p50 | p90 | max |
|---|---:|---:|---:|---:|---:|
| code_first | 16 | 0.352 | 0.443 | 0.679 | 0.685 |
| issue_first | 16 | 0.677 | 0.635 | 0.921 | 0.945 |
| neighbor_file_before_code | 16 | 0.752 | 0.698 | 1.007 | 1.021 |
| planner_trace_before_code | 16 | 0.659 | 0.595 | 0.896 | 0.918 |
| previous_output_before_code | 16 | 0.657 | 0.587 | 0.906 | 0.930 |
| review_trace_before_code | 16 | 0.658 | 0.656 | 0.922 | 0.939 |

## By Agent Role

| Bucket | n | mean | p50 | p90 | max |
|---|---:|---:|---:|---:|---:|
| coder | 24 | 0.638 | 0.616 | 0.902 | 0.960 |
| planner | 24 | 0.555 | 0.591 | 0.881 | 0.969 |
| reviewer | 24 | 0.650 | 0.654 | 0.939 | 1.007 |
| tester | 24 | 0.660 | 0.659 | 0.922 | 1.021 |

## By AST Type

| Bucket | n | mean | p50 | p90 | max |
|---|---:|---:|---:|---:|---:|
| FunctionDef | 96 | 0.626 | 0.608 | 0.918 | 1.021 |

## Worst Cases

| seg_id | role | structure | d_norm | span_tokens | target_start |
|---|---|---|---:|---:|---:|
| he__HumanEval_138_is_equal_to_sum_even | tester | neighbor_file_before_code | 1.021 | 79 | 101 |
| he__HumanEval_138_is_equal_to_sum_even | reviewer | neighbor_file_before_code | 1.007 | 79 | 101 |
| he__HumanEval_138_is_equal_to_sum_even | planner | neighbor_file_before_code | 0.969 | 79 | 107 |
| he__HumanEval_138_is_equal_to_sum_even | coder | neighbor_file_before_code | 0.960 | 79 | 102 |
| he__HumanEval_138_is_equal_to_sum_even | reviewer | issue_first | 0.945 | 79 | 90 |
| he__HumanEval_138_is_equal_to_sum_even | reviewer | review_trace_before_code | 0.939 | 79 | 100 |
| he__HumanEval_138_is_equal_to_sum_even | tester | previous_output_before_code | 0.930 | 79 | 96 |
| he__HumanEval_138_is_equal_to_sum_even | tester | review_trace_before_code | 0.922 | 79 | 100 |
| he__HumanEval_138_is_equal_to_sum_even | tester | issue_first | 0.921 | 79 | 90 |
| he__HumanEval_138_is_equal_to_sum_even | tester | planner_trace_before_code | 0.918 | 79 | 105 |

## Interpretation Hook

Use this report to justify AgentTemplateKV policy choices: exact-content remains the safety gate, while coding structure controls whether reuse is low-risk, should be prefetched/protected, or should be refused because the target code span is structurally far from the canonical code-first template.
