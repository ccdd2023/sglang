# KVFlow current handoff

## Branch entry points

| Branch | Responsibility |
|---|---|
| `kvflow/shared-core` | Segment identity/lifecycle and policy-neutral transfer |
| `research/coding-aware-lossy` | Coding signals and recompute/copy planning |
| `research/prefetch` | Prefix/middle-KV residency and prefetch scheduling |
| `integration/coding-aware-prefetch` | Composition tests and thin adapters |

The shared interface candidate is tagged `kvcomm-core-v0.1-rc3`. Use
`git rev-parse <branch>` when an exact current commit is needed; this document
does not embed branch-head hashes that become stale after every handoff change.

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

## Middle-KV interface for the prefetch owner

The high-level entry point is:

```python
from sglang.srt.mem_cache.kvcomm_prefetch import MiddleKVPrefetchAPI
```

Its ownership flow is:

1. The producer calls `export_middle_kv(...)` after the source request computed
   an exact token/KV slice. The request retains ownership of its original
   device slots.
2. The scheduler keeps the returned `KVSegmentKey` and calls
   `prefetch(key, deadline_s=..., priority=...)`.
3. `PrefetchTicket.wait()` returns a device-resident handle. That handle is
   directly valid as a `TransferSpan.source` in the shared `KVReusePlan`.
4. The scheduler releases the ticket lease after request admission/finish and
   calls `drop(...)` when the prefetched copy is no longer cacheable.

The complete CPU-only example, including plan consumption and cleanup, is:

```bash
PYTHONPATH=python python examples/kvflow/middle_kv_prefetch.py
```

See `examples/kvflow/README.md` for production allocator mapping, identity,
failure and resource-ownership contracts. The ticket is synchronous in v1;
real background transfer/stream scheduling remains prefetch-branch work and
does not require a caller API change.

## Collaborator migration

The former collaborator branch is preserved remotely as:

```text
archive/context-aware-kv-reuse-20260717 @ 015d58c969cb
```

It includes the four local commits that had not reached
`origin/feature/context-aware-kv-reuse`. Start new prefetch work from:

```bash
git fetch origin
git switch --create research/prefetch \
  --track origin/research/prefetch
```

Do not merge the archived branch wholesale. Port any later unpublished
prefetch change as a small PR against `research/prefetch`; coding selectors,
experiment results and paper edits stay behind.

## Tests

```bash
PYTHONPATH=python /home/gfy/.conda/envs/sglang-kvflow/bin/python -m pytest -q \
  python/sglang/srt/mem_cache/kvcomm/test_core.py
```

Branch-specific suites add:

```text
python/sglang/srt/mem_cache/coding_aware/test_policy.py
python/sglang/srt/mem_cache/kvcomm_prefetch/test_coordinator.py
python/sglang/srt/mem_cache/kvcomm_prefetch/test_middle_kv.py
python/sglang/srt/mem_cache/kvflow_integration/test_composition.py
test/registered/unit/mem_cache/test_radix_cache_unit.py
```

## Migration order

1. Shared owner runs the existing full-RoPE/slice-verified backend in a real
   model-server canary.
2. Prefetch owner connects the high-level ticket API to scheduler prediction
   and optionally adds a HiCache storage-tier loader.
3. Coding owner migrates only the active signal/label builder into the coding
   branch.
4. Integration reruns the four-mode compatibility matrix.

The active paper and large experiment directories remain in the original
checkout and are not part of these branch dependencies.
