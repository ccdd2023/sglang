# Fair Measurement Summary (single run)

- mode: `placeholder_slot_lossless`
- agent_count: 3
- vary_code: False
- warm_anchor_pool: False
- exclude_source_agent: True (source role = `implementer`)
- total rows: 15 (reusers counted: 10)

## TTFT (reusers only, unless source included)
- avg TTFT = 520.5 ms, p50 = 404.7 ms, p90 = 792.6 ms
- source agent avg TTFT = 426.4 ms (prefill-bound; not a reuse beneficiary)

## cached_tokens decomposition (reusers only)
- avg cached_tokens = 128.6
- avg radix_prefix_tokens = 128.6  (radix L1 prefix — the ONLY part the prefix_cache_only baseline also sees; must cancel in a fair A/B)
- avg codeaware_reused_tokens = 0.0  (L2 whole-slot + L3 offset-gate body copy + C2 chunk copy — the code-aware contribution)
  - l2_wholeslot_reused_tokens = 0.0  (general KVCOMM baseline)
  - l3_offset_reused_tokens = 0.0  (MiniLM — deprecated regime)
  - c2_chunk_reused_tokens = 0.0  (AST chunk path)

## Interpretation
- If `radix_prefix_tokens` ≈ matches the prefix_cache_only run's `cached_tokens`,
  the radix prefix cancels and the speedup is honestly code-aware.
- `codeaware_reused_tokens` is the code-aware algorithm's own reuse, isolated.
- Source-agent (implementer) should show ~0 codeaware_reused (it is the source).
- Cross-config speedup + parity gate: run analyze_fair_ab.py over this and the
  prefix_cache_only run's rows.csv.
