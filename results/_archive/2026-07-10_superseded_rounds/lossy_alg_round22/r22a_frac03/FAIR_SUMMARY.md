# Fair Measurement Summary (single run)

- mode: `placeholder_knn_reuse`
- agent_count: 5
- vary_code: False
- warm_anchor_pool: False
- exclude_source_agent: False (source role = `implementer`)
- total rows: 25 (reusers counted: 25)

## TTFT (reusers only, unless source included)
- avg TTFT = 713.4 ms, p50 = 748.8 ms, p90 = 880.8 ms
- source agent avg TTFT = 707.9 ms (prefill-bound; not a reuse beneficiary)

## cached_tokens decomposition (reusers only)
- avg cached_tokens = 466.4
- avg radix_prefix_tokens = 113.6  (radix L1 prefix — the ONLY part the prefix_cache_only baseline also sees; must cancel in a fair A/B)
- avg codeaware_reused_tokens = 352.7  (L2 whole-slot + L3 offset-gate body copy + C2 chunk copy — the code-aware contribution)
  - l2_wholeslot_reused_tokens = 0.0  (general KVCOMM baseline)
  - l3_offset_reused_tokens = 0.0  (MiniLM — deprecated regime)
  - c2_chunk_reused_tokens = 352.7  (AST chunk path)

## Interpretation
- If `radix_prefix_tokens` ≈ matches the prefix_cache_only run's `cached_tokens`,
  the radix prefix cancels and the speedup is honestly code-aware.
- `codeaware_reused_tokens` is the code-aware algorithm's own reuse, isolated.
- Source-agent (implementer) should show ~0 codeaware_reused (it is the source).
- Cross-config speedup + parity gate: run analyze_fair_ab.py over this and the
  prefix_cache_only run's rows.csv.

## Precomputed codebase KV
- precompute_kv_dir: `results/codebase_kv/pandas_5case_v4`
- canonical_prefix: True
- host_pool_size_gb: 2.0
- With precompute ON, the pool is warm at server start, so agent 1
  ALSO hits the pool (it is a reuse beneficiary, not the source).
- source_agent_included: True
- Reuse from precomputed (host-resident) chunks flows through the same
  c2_chunk_reused_tokens counter as on-demand L4 reuse; the difference
  vs a no-precompute run is agent 1's nonzero reuse + warm-from-start TTFT.
- Honest accuracy bound: only the canonical preamble is losslessly
  reusable; file content at shifted positions stays lossy (proven
  cross-context KV loss).
