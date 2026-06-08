# Cross-Model Transferability Status (R7)

## Status: 4/4 complete (Qwen3-8B done on 2026-06-08)

The 4th model in the cross-model study is **Qwen3-8B**. The HF cache had
only 1.3 GB of the 16 GB model (3 of 5 shards were incomplete due to
HF rate limits). The 2026-06-08 run populated the cache by symlinking
the complete local model at `/home/gfy/models/Qwen3-8B/` into
`~/.cache/huggingface/hub/models--Qwen--Qwen3-8B/`. The 3,456 forward
passes (24 segments × 2304 variations) completed in ~5 min, producing
`context_distance_qwen-qwen3-8b.json` (108 KB) and
`predicted_distance_table_qwen-qwen3-8b.json` (66 KB).

## 4-model results (Qwen2.5-Coder-7B, Qwen2.5-Coder-3B, Qwen2.5-7B, Qwen3-8B)

### Canonical cell d_norm (50-200 tok, offset 0, planner, no wrap)

| Model | d_norm |
|---|---:|
| Qwen2.5-Coder-7B-Instruct  | 1.7703 |
| Qwen2.5-Coder-3B-Instruct  | 1.6362 |
| Qwen2.5-7B-Instruct        | 1.7195 |
| Qwen3-8B                   | 2.8982 |

The 4th model (Qwen3-8B) has a substantially higher baseline d_norm
(2.90 vs 1.64-1.77 for the Qwen2.5 family) — ~1.1 units higher.

### Pairwise mean |Δd_norm| across the 144 short-code cells

| pair | mean |Δd_norm| |
|---|---:|
| Qwen2.5-Coder-7B-Instruct  vs  Qwen2.5-7B-Instruct        | 0.0667 |
| Qwen2.5-Coder-3B-Instruct  vs  Qwen2.5-7B-Instruct        | 0.0989 |
| Qwen2.5-Coder-7B-Instruct  vs  Qwen2.5-Coder-3B-Instruct  | 0.1655 |
| Qwen2.5-Coder-7B-Instruct  vs  Qwen3-8B                   | **1.4753** |
| Qwen2.5-7B-Instruct        vs  Qwen3-8B                   | **1.5415** |
| Qwen2.5-Coder-3B-Instruct  vs  Qwen3-8B                   | **1.6408** |

**Mean off-diag |Δd_norm| = 0.832 across 12 pairs.**

## Verdict (revised 2026-06-08, 4/4 models)

- **Within the Qwen2.5 family (3 models)**: tables are portable, max
  mean Δd_norm = 0.166. A single Qwen2.5-anchored lookup table transfers
  across the 3 Qwen2.5 models without re-running the 3,456-pass
  experiment.
- **Qwen3-8B vs Qwen2.5 family (3 pairs)**: tables diverge by 1.47-1.64
  d_norm on average, well above the 0.30 "family-specific" threshold.
  Qwen3-8B is in a *different family* with substantially different
  internal representations, so the Qwen2.5-anchored table does not
  transfer to Qwen3-8B.
- **Combined verdict (all 4 models)**: **weak portable** — the 192-cell
  table is Qwen2.5-family-anchored. For Qwen3-8B and future Qwen3+
  models, per-model bias correction or a separate Qwen3-anchored table
  is required.

This revises the earlier "3/4 strong portable" claim to "3/4 (Qwen2.5)
strong portable, 1/4 (Qwen3) weak portable." The paper's
§7.7 should reflect this: the table is *Qwen2.5-family-portable* and
needs a per-family note for Qwen3.

## Bug fix during 4/4 re-run

The `cross_model_report.py:_slug_for()` function used
`replace("/", "--")` (double-dash), but the table filenames use
single-dash (slug is `qwen-qwen2.5-coder-7b-instruct`, not
`qwen--qwen2.5-coder-7b-instruct`). The double-dash produced a slug
that did not match any file, so `_load_table()` returned `None`, the
`if not ta or not tb: continue` check skipped every pair, and the
pairwise matrix stayed at 0.0000. The fix is one character:
`.replace("/", "-")` in `_slug_for()` at line 95.

## Files

- `data/predicted_distance_table_qwen-qwen2.5-coder-7b-instruct.json`
- `data/predicted_distance_table_qwen-qwen2.5-coder-3b-instruct.json`
- `data/predicted_distance_table_qwen-qwen2.5-7b-instruct.json`
- `data/predicted_distance_table_qwen-qwen3-8b.json` (new, 192 cells)
- `data/context_distance_qwen-qwen3-8b.json` (new, 108 KB, 2304 variations)
- `data/cross_model_comparison.json` (machine-readable summary)
- `report.md` (human-readable, regenerated after fix)
- `plots/cross_model_d_norm_heatmap.png` (4×4 pairwise Δd_norm heatmap)

## Reproducibility

```bash
bash results/lookup_table_transferability/run_all.sh   # all 4 models; 3 are skipped
/home/gfy/.conda/envs/sglang-kvflow/bin/python \
  results/lookup_table_transferability/cross_model_report.py
```
