# Near-Match Gate Safety Expansion

## Summary

- Negative near-match pairs: 500
- Positive exact controls: 50
- CSV: `gate_nearmatch_500.csv`
- Avg candidate tokens: 1992.8

## Policy Results

| policy | pairs | allowed | false accepts | false rejects |
|---|---:|---:|---:|---:|
| exact_content_gate | 550 | 50 | 0 | 0 |
| ast_only | 550 | 550 | 500 | 0 |
| span_overlap_only | 550 | 550 | 500 | 0 |
| path_function_name | 550 | 550 | 500 | 0 |
| content_signature | 550 | 50 | 0 | 0 |
| token_text_exact | 550 | 50 | 0 | 0 |
| no_gate | 550 | 550 | 500 | 0 |

## Interpretation

Exact-content policies reject all same-path/same-locator near matches whose code text changed.
AST/path/span/no-gate policies intentionally over-accept these pairs, demonstrating why locator metadata is not a safety gate.
