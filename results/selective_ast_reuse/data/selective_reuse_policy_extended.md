# Selective AST reuse policy

- Source: `results/ast_granularity_kv_sensitivity/data/ast_granularity_distance_7b.json`
- p90 threshold: `0.5`
- max tail rate: `0.1`

| granularity | decision | p90 | max | tail>0.5 | retention tokens | reason |
|---|---|---:|---:|---:|---:|---|
| `class` | `recompute` | 0.562 | 0.770 | 0.200 | 4930 | `granularity_risk` |
| `control_block` | `reuse` | 0.468 | 0.585 | 0.083 | 5379 | `extended_safe_granularities` |
| `file_prefix` | `reuse` | 0.461 | 0.573 | 0.067 | 57039 | `extended_safe_granularities` |
| `function` | `reuse` | 0.424 | 0.457 | 0.000 | 5357 | `extended_safe_granularities` |
| `method` | `reuse` | 0.421 | 0.576 | 0.083 | 5175 | `extended_safe_granularities` |
| `statement_window` | `recompute` | 0.544 | 0.750 | 0.133 | 5057 | `granularity_risk` |
