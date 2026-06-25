# Combined LMCache 100-case replay

- cases: 100
- rows: 400

## Mode summary

| mode | n | avg elapsed ms | avg cached tokens | exact match rate | avg token F1 | avg protected tokens |
|---|---:|---:|---:|---:|---:|---:|
| baseline_prefix_cache_only | 100 | 1350.5 | 12316.23 | 1.0 | 1.0 | 0.0 |
| kvflow_prefix_only | 100 | 1302.16 | 12316.23 | 0.55 | 0.7 | 0.0 |
| kvflow_prefix_plus_codebase_prefetch | 100 | 1385.44 | 12319.23 | 0.36 | 0.62 | 0.0 |
| kvcomm_lossy_plus_codebase_prefetch | 100 | 1437.49 | 12317.23 | 0.3 | 0.6 | 10698.11 |

## Shards

- `results/coding_kvflow_prefetch/qwen2_5_7b_100_lmcache_20260613`: 48 cases, astropy__astropy-12907 -> pallets__flask-5014, flush=False, args=`--enable-lmcache`
- `results/coding_kvflow_prefetch/qwen2_5_7b_100_lmcache_20260613_shard48_20_flush_noidlecheck`: 20 cases, psf__requests-1142 -> pydata__xarray-4695, flush=True, args=`--disable-overlap-schedule --max-running-requests 1 --enable-lmcache`
- `results/coding_kvflow_prefetch/qwen2_5_7b_100_lmcache_20260613_shard68_20_flush_noidlecheck`: 20 cases, pydata__xarray-4966 -> pytest-dev__pytest-5809, flush=True, args=`--disable-overlap-schedule --max-running-requests 1 --enable-lmcache`
- `results/coding_kvflow_prefetch/qwen2_5_7b_100_lmcache_20260613_shard88_12_flush_noidlecheck`: 12 cases, pytest-dev__pytest-5840 -> scikit-learn__scikit-learn-11310, flush=True, args=`--disable-overlap-schedule --max-running-requests 1 --enable-lmcache`

## Caveats

- This is an external-baseline replay artifact, not yet a paper headline.
- Cases 48-99 require cache flush and idle strict-check disable to avoid LMCache/SGLang radix/protected-KV failures on the 24GB GPU.
- The machine also had an unrelated GPU process during the run, so absolute latency needs a clean rerun before publication.
