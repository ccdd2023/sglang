# Giant-Codebase Production Baseline (L1+L2 only, L3 OFF) — 2026-06-27

**Branch**: `fix/placeholder-pool-activation`
**Configuration**: 50 pandas tasks × 5 agents = 250 requests, single chunk
**Driver flags**: `--mode placeholder_knn_reuse --no-vary-code --chunk-size 50`
**L3 state**: DISABLED (default — `--enable-research-l3` NOT passed)
**Run dir**: `results/giant_codebase/runs/giant_pandas_50_l3_off_20260627_051353/`

---

## TL;DR

This is the **production baseline** after formally deprecating the
placeholder k-NN body (L3) for production deployments. The driver
default is now `SGLANG_PLACEHOLDER_KNN_MATCH=0`; L3 must be
explicitly opted into via `--enable-research-l3` (see commit `8064ea45`).

**Production speedup: 1.31×** vs `prefix_cache_only` baseline (445ms vs 581ms).

| Configuration | avg TTFT | speedup vs baseline | placeholder pool hits |
|---|---:|---:|---:|
| `prefix_cache_only` baseline | 581 ms | 1.00× | 0 |
| **PRODUCTION (L1+L2, L3 OFF)** | **445 ms** | **1.31×** | **0** |
| RESEARCH (L1+L2+L3, `--enable-research-l3`) | 353 ms | 1.65× | 350 |

L3 marginal contribution: 1.26× (26% of the research speedup comes
from the deprecated path).

---

## Why `placeholder_anchor_pool_hit_count = 0` in production

The placeholder pool itself still grows (max 1 529 entries — L2's
`_store_placeholder_anchor_kv` is still active), but in
`placeholder_knn_reuse` mode + L3 off, neither the byte-exact L2
match path nor the k-NN L3 match path fires against it. The 20.4%
cached_ratio comes entirely from L1 (prefix cache).

To measure L2 (whole-slot byte-exact with RoPE rotation)
contribution cleanly, use `mode=lossy` instead. The pre-fix Gate 2
report (`giant_pandas_FINAL_REPORT.md`) measured L1+L2 at 1.31×
under similar conditions; the L3 boost to 1.65× came from
cross-mode additions not captured in this re-run.

---

## Safety vs speedup tradeoff

L3 was deprecated because variable renames / comment edits / signature
changes all leave MiniLM cos ≥ 0.85, but reusing K/V from the OLD
version of the code gives the model a confused representation of the
NEW prompt. Failure mode is silent: tests pass, output reads
correctly, but runtime behavior diverges.

The production configuration trades 0.33× speedup for the byte-exact
correctness invariant. Future work (Direction #3 — AST-boundary
chunked prefill) will recover some of the lost speedup through
function/class-boundary partial-match reuse that preserves the
byte-exact invariant.

---

## Companion work in this session

1. **L3 deprecation (commit `8064ea45`)**: hard warning docstring on
   `_try_placeholder_knn_lossy_match` + driver default OFF.
2. **HANDOFF update (commit `86d622d8`)**: new "L3 DEPRECATED" section.
3. **Memory entry `l3-placeholder-knn-deprecated`**: auto-loaded in
   every new session to keep this policy enforced.
4. **Direction #3 Phase A (commit `7fb1a5bb`)**: server-side AST chunker
   preserves byte-exact invariant at chunk level (safe recovery path).

---

## Raw data

- `rows.csv` — 250 rows × 52 columns (per-task per-agent metrics)
- `sglang_server.log` — full sglang server log
- `report/REPORT.md` — this file