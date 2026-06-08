# Lossy vs Lossless KV Reuse — Large Codebase x Multi-Agent

Model: Qwen2.5-3B (288 KB/tok) | 5 files

| File | Lines | Agent | mode | cached_tok | KV (MB) | Reuse% | ms |
|---|---|---|---:|---:|---:|---:|
| django_query | 81 | A1 Analyzer | - | 0 | 0.0 | 0.0% | 1686 |
| django_query | 81 | A2 lossy | - | 728 | 204.8 | 87.4% | 3498 |
| django_query | 81 | A2 lossless | - | 832 | 234.0 | 99.9% | 3476 |
| django_query | 81 | A3 lossy | - | 728 | 204.8 | 80.7% | 707 |
| django_query | 81 | A3 lossless | - | 901 | 253.4 | 99.9% | 701 |
| requests_session | 76 | A1 Analyzer | - | 0 | 0.0 | 0.0% | 1797 |
| requests_session | 76 | A2 lossy | - | 711 | 200.0 | 88.5% | 3444 |
| requests_session | 76 | A2 lossless | - | 802 | 225.6 | 99.9% | 3408 |
| requests_session | 76 | A3 lossy | - | 711 | 200.0 | 81.4% | 3408 |
| requests_session | 76 | A3 lossless | - | 872 | 245.2 | 99.9% | 3416 |
| ml_pipeline | 111 | A1 Analyzer | - | 0 | 0.0 | 0.0% | 1993 |
| ml_pipeline | 111 | A2 lossy | - | 992 | 279.0 | 91.3% | 3493 |
| ml_pipeline | 111 | A2 lossless | - | 1086 | 305.4 | 99.9% | 3467 |
| ml_pipeline | 111 | A3 lossy | - | 992 | 279.0 | 85.9% | 1123 |
| ml_pipeline | 111 | A3 lossless | - | 1154 | 324.6 | 99.9% | 1119 |
| dist_lock | 111 | A1 Analyzer | - | 0 | 0.0 | 0.0% | 2473 |
| dist_lock | 111 | A2 lossy | - | 878 | 246.9 | 89.5% | 3477 |
| dist_lock | 111 | A2 lossless | - | 980 | 275.6 | 99.9% | 3440 |
| dist_lock | 111 | A3 lossy | - | 878 | 246.9 | 83.5% | 1326 |
| dist_lock | 111 | A3 lossless | - | 1051 | 295.6 | 99.9% | 1322 |
| auth_handler | 120 | A1 Analyzer | - | 0 | 0.0 | 0.0% | 1638 |
| auth_handler | 120 | A2 lossy | - | 1115 | 313.6 | 91.6% | 3486 |
| auth_handler | 120 | A2 lossless | - | 1216 | 342.0 | 99.9% | 3444 |
| auth_handler | 120 | A3 lossy | - | 1115 | 313.6 | 86.6% | 994 |
| auth_handler | 120 | A3 lossless | - | 1286 | 361.7 | 99.9% | 2003 |

## Summary
| Metric | lossy | lossless | Delta |
|---|---|---|---|
| A2 avg KV | 249 MB | 277 MB | -28 MB |
| A3 avg KV | 249 MB | 296 MB | -47 MB |
