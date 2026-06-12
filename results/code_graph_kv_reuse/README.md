# Code Graph-Aware Lossy Reuse Precision

This directory contains derived artifacts for the code-specific lossy KV reuse
precision line. It extends AST granularity analysis with lightweight
call/import/test-neighborhood bundles. Token counts are recorded as scope
covariates only; the research target is KV/output precision under lossy reuse.

Run:

```bash
python3 results/code_graph_kv_reuse/code_graph_bundle_analyzer.py --limit 30
```

Primary output:

- `data/code_graph_bundle_census.json`
- `data/code_graph_bundle_table.csv`
- `data/code_graph_precision_manifest.jsonl` (large derived manifest; not
  committed by default)
- `figures/*.png`
- `CODE_GRAPH_KV_REUSE_REPORT.md`

## Retention policy

Commit the scripts, aggregate reports, selected-case manifests, and small
summary files used by the paper:

- `data/graph_pass1_8_applyok.json`
- `pass1_graph_aware_8_with_tests_envfix_20260612/summary.json`
- `pass1_graph_aware_8_with_tests_envfix_20260612/pass1_8_with_tests_summary.{json,md}`
- `pass1_graph_aware_8_with_tests_envfix_20260612/pass1_8_with_tests_diagnostics.csv`

Do not commit raw server logs, `__pycache__`, or the full
`data/code_graph_precision_manifest.jsonl` when it grows large. The manifest is
derived from the local SWE-bench/RepoBench source snapshots and can be
regenerated with:

```bash
python3 results/code_graph_kv_reuse/code_graph_bundle_analyzer.py --limit 30
```

The 8-case graph-aware pass@1/apply-ok bundle uses SWE-bench instances selected
from the local `results/repo_level_datasets` manifests. Recreate the public
dataset inputs with the Princeton NLP SWE-bench Hugging Face releases, then run
the companion harness documented in `HANDOFF_2026_06_12.md`.

Safety boundary: graph relations choose candidate exact spans; actual KV reuse
must still be gated by normalized content signature and token-level exact match.
