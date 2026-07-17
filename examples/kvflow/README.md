# KVFlow middle-KV prefetch handoff

`middle_kv_prefetch.py` is a CPU-only executable example of the interface
between the reuse producer and a prefetch scheduler:

```text
computed request KV
  -> export_middle_kv(...)
  -> host-resident KVSegmentHandle
  -> prefetch(handle.key)
  -> PrefetchTicket.wait()
  -> device-resident KVSegmentHandle / physical device indices
  -> reuse planner consumes the handle
  -> ticket.release(); api.drop(...)
```

Run it from the repository root:

```bash
PYTHONPATH=python python examples/kvflow/middle_kv_prefetch.py
```

## Production mapping

Construct `MiddleKVPrefetchAPI` once per cache/allocator:

```python
api = MiddleKVPrefetchAPI(
    manager=kvcomm_manager,
    allocator=tree_cache.token_to_kv_pool_allocator,
    model_id=model_config.model_path,
    cache_dtype=str(server_args.kv_cache_dtype),
)
```

After a request has computed a reusable middle segment, the producer supplies:

- `token_ids`: the exact logical tokens in the segment;
- `kv_indices`: their physical slots from the request token-to-KV mapping;
- `source_start`: the segment's original logical token position;
- `content_hash`: a stable content/version identity chosen by the caller.

`export_middle_kv` copies the all-layer K/V payload to host and registers a
`SegmentKind.MIDDLE` handle. It does **not** free, pin, or otherwise take
ownership of the producer request's original device slots.

The scheduler calls `prefetch(key, deadline_s=..., priority=...)`. In v1 this
operation is synchronous, but it returns a `PrefetchTicket` so the internals can
later use a CUDA event or transfer stream without changing the caller contract.
`ticket.wait()` returns the current device-resident handle, and
`ticket.device_indices()` exposes the physical slots when needed by the reuse
planner. The example also puts that handle in a policy-neutral
`TransferSpan`/`KVReusePlan` and calls `KVCommManager.execute`. In production,
use `RadixKVTransferBackend` for the actual all-layer K/V copy and full-RoPE
correction.

The ticket owns only a lease. `ticket.release()` or the context manager releases
that lease. `api.drop(key)` removes the cached segment and invokes the
allocator-backed device-slot releaser. Releasing the source request remains the
normal responsibility of RadixCache/request lifecycle code.

## Feature gates

The interface is disabled by default. A server integration must enable:

```bash
export SGLANG_KVCOMM_CORE=1
export SGLANG_KV_PREFETCH=1
```

The coding-aware policy flag is independent. A prefetch-only experiment does
not need `SGLANG_CODING_AWARE_LOSSY=1`.

## Error contract

- Missing segments and failed loads produce a ticket whose `wait()` raises
  `MiddleKVPrefetchError`; they are never reported as successful prefetches.
- Token count, model ID, cache dtype, segment kind, and token hash are part of
  segment identity.
- Allocation or host-to-device load failure frees any newly allocated slots.
- Duplicate/stale generations are rejected by the shared store.

The low-level `KVPrefetchCoordinator` remains available for schedulers that
need to prefetch mixed prefix and middle-segment batches. This high-level API
intentionally accepts only middle segments to make the handoff boundary
explicit.
