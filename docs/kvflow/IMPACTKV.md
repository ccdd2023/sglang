# ImpactKV on this SGLang fork

This branch (`integration/template-prefetch-swebench`) is the **ASPLOS 2027**
ImpactKV engine: coding-aware true-lossy **file-island** KV copy.

Full collaborator notes (frozen numbers, paper, artifacts, GPU reproduce):

https://github.com/flaminyu/CodeMAS_Project/blob/master/IMPACTKV.md

(or `CodeMAS_Project/IMPACTKV.md` on the cluster).

## What landed here

| Path | Role |
|---|---|
| `python/sglang/srt/mem_cache/kvcomm_exact.py` | Fail-closed copy; source-side K pre-rotate |
| `python/sglang/srt/mem_cache/kvcomm/` | Page key `(prefix hash, content hash, Δ)`, store, radix backend |
| `python/sglang/srt/mem_cache/kvcomm_prefetch/` | M3; **off** in the 7B headline job 137185 |
| `benchmark/multi_workflow/run_swebench_*.py` | Exact-prompt SWE-bench replay campaigns |
| `benchmark/multi_workflow/slurm/swebench_*.sbatch` | Slurm; exclude `gpu[10-13,15,17,23-24]` |

Headline campaign (do not overwrite frozen RESULT): Qwen2.5-Coder-7B-Instruct,
job 137185, cache-ready **1.492×**, copies **1684/1684**, prefetch off, prefix off.

## Tests (no GPU)

```bash
export PYTHONPATH="$PWD/python"
python -m pytest -q \
  python/sglang/srt/mem_cache/test_kvcomm_exact.py \
  python/sglang/srt/mem_cache/kvcomm/test_core.py \
  python/sglang/srt/mem_cache/kvcomm/test_radix_backend.py \
  python/sglang/srt/mem_cache/kvcomm_prefetch/test_*.py
```

Historical V40–V46 / RepoBench notes in this folder are **not** the ASPLOS
headline. Start from the CodeMAS `IMPACTKV.md` instead of
`COLLABORATOR_QUICKSTART_20260729.md`.
