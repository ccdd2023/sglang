# AST-Granularity KV Sensitivity

This experiment keeps every code object byte-identical and varies the AST granularity used as the reuse object. The canonical cache source is the planner view of the same exact span; coder and reviewer prompts measure whether that span remains a stable and useful K/V anchor.

## Setup

- Model: `Qwen/Qwen2.5-Coder-7B-Instruct`
- Canonical cell: `planner` on the same exact code object
- Selected layers: `[-1, -2, -3, -4]`
- Spans: `180`
- Variations: `540`

## Overall

- n = 540
- mean d_norm = 0.240
- p90 d_norm = 0.458
- max d_norm = 0.770

## By AST Granularity

| Bucket | spans | n | mean toks | retention toks | mean d_norm | p90 d_norm | max d_norm | weighted d_norm | reuse score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| file_prefix | 30 | 90 | 1901.300 | 57039.000 | 0.246 | 0.461 | 0.573 | 0.248 | 1301.733 |
| class | 30 | 90 | 164.333 | 4930.000 | 0.280 | 0.536 | 0.770 | 0.287 | 106.990 |
| function | 30 | 90 | 178.567 | 5357.000 | 0.222 | 0.415 | 0.457 | 0.223 | 126.218 |
| method | 30 | 90 | 172.500 | 5175.000 | 0.228 | 0.408 | 0.576 | 0.232 | 122.508 |
| control_block | 30 | 90 | 179.300 | 5379.000 | 0.245 | 0.463 | 0.585 | 0.246 | 122.598 |
| statement_window | 30 | 90 | 168.567 | 5057.000 | 0.220 | 0.486 | 0.750 | 0.217 | 113.424 |

## By Token Bin

| Bucket | spans | n | mean toks | retention toks | mean d_norm | p90 d_norm | max d_norm | weighted d_norm | reuse score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 160-383 | 71 | 213 | 188.972 | 13417.000 | 0.246 | 0.468 | 0.770 | 0.248 | 128.742 |
| 64-159 | 79 | 237 | 157.987 | 12481.000 | 0.233 | 0.452 | 0.750 | 0.233 | 108.825 |
| >=384 | 30 | 90 | 1901.300 | 57039.000 | 0.246 | 0.461 | 0.573 | 0.248 | 1301.733 |

## By Agent Role

| Bucket | spans | n | mean toks | retention toks | mean d_norm | p90 d_norm | max d_norm | weighted d_norm | reuse score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| coder | 180 | 180 | 460.761 | 82937.000 | 0.371 | 0.536 | 0.770 | 0.374 | 299.982 |
| planner | 180 | 180 | 460.761 | 82937.000 | 0.000 | 0.000 | 0.000 | 0.000 | 460.761 |
| reviewer | 180 | 180 | 460.761 | 82937.000 | 0.349 | 0.486 | 0.716 | 0.362 | 309.979 |

## Worst Cases

| span_id | granularity | role | path | lines | d_norm | span_tokens | target_start |
|---|---|---|---|---|---:|---:|---:|
| 4b0cbfa6e6540baa | class | coder | lib/matplotlib/patches.py | 3560-3591 | 0.770 | 244 | 75 |
| 99c8ea2a44b7d28d | statement_window | coder | lib/matplotlib/axes/_base.py | 521-532 | 0.750 | 137 | 78 |
| 4b0cbfa6e6540baa | class | reviewer | lib/matplotlib/patches.py | 3560-3591 | 0.716 | 244 | 74 |
| 99c8ea2a44b7d28d | statement_window | reviewer | lib/matplotlib/axes/_base.py | 521-532 | 0.688 | 137 | 77 |
| f0154fa5e6a8a678 | class | coder | pylint/checkers/base.py | 84-111 | 0.658 | 265 | 72 |
| 08d292feb489f018 | statement_window | coder | testing/python/metafunc.py | 161-172 | 0.650 | 154 | 72 |
| f0154fa5e6a8a678 | class | reviewer | pylint/checkers/base.py | 84-111 | 0.631 | 265 | 71 |
| e3e3a9c8b61e2f2c | statement_window | coder | lib/matplotlib/patches.py | 2281-2292 | 0.627 | 220 | 76 |
| 08d292feb489f018 | statement_window | reviewer | testing/python/metafunc.py | 161-172 | 0.594 | 154 | 71 |
| 6aff273154ea8946 | control_block | coder | lib/matplotlib/axes/_base.py | 1694-1710 | 0.585 | 174 | 79 |

## Regularities

- Exact content remains the non-negotiable reuse gate; AST granularity only chooses which exact byte span becomes the reusable object.
- Function and method spans form the best default policy unit: low mean/p90 distance, useful token payload, bounded retention cost, and natural alignment with coding-agent edits.
- Statement windows can be stable on average, but their semantic boundary is weak and their tail risk is high; use them as fallback exact spans, not as the primary template object.
- Class spans are useful when downstream agents repeatedly inspect related methods, but the higher p90 distance means they should require DAG evidence and TTL protection.
- File prefixes offer the largest theoretical saving, but retention cost is an order of magnitude larger; protect them only for stable codebase-front blocks with strong future-use evidence.
