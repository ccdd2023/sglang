# KVFlow verified status — 2026-07-17

## Headline

The legacy implementation contains working KV reuse mechanisms, but it is not
a complete independent shared component. The new branch layout establishes a
stable interface; GPU and HiCache adapters still have to be migrated before
`KVCOMM_CORE_COMPLETE` can be claimed.

## Evidence

| Component | Status | Evidence | Remaining work |
|---|---|---|---|
| Legacy anchor/chunk implementation | VERIFIED locally, architecture mixed | 111/111 existing anchor and placeholder tests pass | Keep only as migration source |
| Shared segment identity/store | VERIFIED | Token/hash/generation, pinned-replacement guard, resource disposer and 10,000 lease-cycle tests | Add production metrics adapter |
| Shared transfer planner | VERIFIED on recording and Radix backends | Offset, full-RoPE accounting, stale/mismatch fallback tests | Run end-to-end GPU server canary |
| Coding policy isolation | VERIFIED; FileVersion SessionGraphKV V11 P0 FALSIFIED | V11 migration has 36 passing tests; formal 4,960-row atlas failed delta-R² and safe-harm gates | Keep P1 closed; register a new hypothesis before further objective experiments |
| Prefix/middle prefetch coordinator | VERIFIED on loader contract | Host-to-device loader call and lease tests | Connect scheduler and validate HiCache storage payloads |
| Combined composition | VERIFIED | Coding plan + middle-KV prefetch integration test | GPU end-to-end benchmark |

The all-layer Radix adapter now uses SGLang's physical `move_kv_cache`,
`load_cpu_copy`, and rotary implementation. It is covered on a deterministic
tensor cache for positive, negative and zero position deltas, including
byte-identical V. It has not yet run an end-to-end model-server GPU canary.

Current classification: **INTERFACE_COMPLETE / SERVER_CANARY_PENDING**.

## Legacy audit findings

- `feature/context-aware-kv-reuse` is an ancestor of the former active branch,
  so it is not an isolated collaborator branch.
- Legacy `AgentTemplateKVCache.prefetch_codebases()` primarily finds and pins
  an already device-resident anchor. It does not by itself implement
  host/storage-to-device middle-KV prefetch.
- Legacy `SGLANG_LOSSY_ENABLED` couples reuse and prefetch and defaults on.
- Context confidence may auto-enable merely because an experiment JSON exists.
- Coding selectors, KV movement, scheduler hooks and eviction accumulated in
  the same large cache implementation.

The new gates default off and never inspect a results directory.

## Acceptance gates for `kvcomm-core-v0.1`

Before tagging the runtime as complete:

- run the shared Radix full-RoPE adapter in a real model-server request;
- validate `ensure_resident` against a real HiCache storage payload;
- demonstrate exact-transfer completion identity against Dense;
- report copied K = rotated K, zeroed gaps = 0 and source mismatches = 0;
- run feature-off, coding-only, prefetch-only and combined server smoke tests;
- show no allocator/ref/lease growth in a sustained concurrency run.

Until those pass, collaborators should depend on the interfaces and tests, not
on a claim that the end-to-end runtime is complete.
