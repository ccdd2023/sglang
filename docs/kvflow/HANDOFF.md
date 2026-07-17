# KVFlow current handoff

## Branch entry points

```text
kvflow/shared-core
research/coding-aware-lossy
research/prefetch
integration/coding-aware-prefetch
```

Clean worktrees are under:

```text
/home/gfy/CodeMAS_Project/sglang-kvflow-worktrees/
```

The original dirty checkout remains untouched at:

```text
/home/gfy/CodeMAS_Project/sglang-kvflow
```

## Owner workflow

Coding-aware work:

```bash
cd /home/gfy/CodeMAS_Project/sglang-kvflow-worktrees/coding-aware
git merge kvflow/shared-core
export SGLANG_KVCOMM_CORE=1
export SGLANG_CODING_AWARE_LOSSY=1
export SGLANG_KV_PREFETCH=0
```

Prefetch work:

```bash
cd /home/gfy/CodeMAS_Project/sglang-kvflow-worktrees/prefetch
git merge kvflow/shared-core
export SGLANG_KVCOMM_CORE=1
export SGLANG_CODING_AWARE_LOSSY=0
export SGLANG_KV_PREFETCH=1
```

Do not cherry-pick commits directly between the two research branches. Test
their combination only in `integration/coding-aware-prefetch`.

## Tests

```bash
PYTHONPATH=python /home/gfy/.conda/envs/sglang-kvflow/bin/python -m pytest -q \
  python/sglang/srt/mem_cache/kvcomm/test_core.py
```

Branch-specific suites add:

```text
python/sglang/srt/mem_cache/coding_aware/test_policy.py
python/sglang/srt/mem_cache/kvcomm_prefetch/test_coordinator.py
python/sglang/srt/mem_cache/kvflow_integration/test_composition.py
test/registered/unit/mem_cache/test_radix_cache_unit.py
```

## Migration order

1. Shared owner connects the existing full-RoPE/slice-verified GPU backend.
2. Prefetch owner connects `ResidencyLoader` to HiCache CPU/storage loading.
3. Coding owner migrates only the active signal/label builder into the coding
   branch.
4. Integration reruns the four-mode compatibility matrix.

The active paper and large experiment directories remain in the original
checkout and are not part of these branch dependencies.
