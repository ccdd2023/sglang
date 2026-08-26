# ImpactKV on this SGLang fork

This branch (`integration/template-prefetch-swebench`) is the **ASPLOS 2027**
ImpactKV engine: coding-aware true-lossy **file-island** KV copy.

**Takeover:** [`../../HANDOFF.md`](../../HANDOFF.md).  
Commands: [`../../IMPACTKV.md`](../../IMPACTKV.md) (own GPU; unpack `offcluster/`).

That file is the only collaborator landing page. Paper sources and the claim
checker live in [`paper/`](paper/). You do **not** need a CodeMAS clone.

## What landed here

| Path | Role |
|---|---|
| `python/sglang/srt/mem_cache/kvcomm_exact.py` | Fail-closed copy; source-side K pre-rotate |
| `python/sglang/srt/mem_cache/kvcomm/` | Page key `(prefix hash, content hash, Δ)`, store, radix backend |
| `python/sglang/srt/mem_cache/kvcomm_prefetch/` | M3; **off** in the 7B headline job 137185 |
| `benchmark/multi_workflow/run_swebench_*.py` | Exact-prompt SWE-bench replay campaigns |
| `benchmark/multi_workflow/slurm/swebench_*.sbatch` | Slurm; exclude `gpu[10-13,15,17,23-24]` |
| `docs/kvflow/paper/` | ASPLOS TeX + `scripts/check_asplos_claims.py` |

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

cd docs/kvflow/paper
python3 scripts/check_asplos_claims.py   # needs IMPACTKV_ARTIFACTS
```

Historical V40–V46 / RepoBench notes in this folder are **not** the ASPLOS
headline. Ignore `COLLABORATOR_QUICKSTART_20260729.md` unless you are reading
that older campaign.
