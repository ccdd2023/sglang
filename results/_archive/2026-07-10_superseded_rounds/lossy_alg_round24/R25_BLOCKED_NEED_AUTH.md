# R25 BLOCKED — needs user explicit authorization

## Status 2026-07-06

User said "继续" after session was at awaiting-state. Auto-mode correctly
interpreted this as a generic continuation (not specific authorization for
`pip install lmcache`).

## What assistant needs to do R25 (LMCache integration v0)

1. `pip install lmcache` (currently blocked by auto-mode)
2. `python -m sglang.launch_server --enable-lmcache --model Qwen/Qwen2.5-Coder-7B-Instruct ...`
3. Run R19 BEST config with --enable-lmcache
4. Compare vs R19 baseline on verdict task-completion metric

## Alternatives that do NOT need pip install

A. **A1 (per-chunk F1 oracle)** — uses existing outputs.jsonl data
B. **A3 (5-pool per-role extraction)** — re-extracts precompute pool, no install
C. **S0 (fix broken scripts)** — repo hygiene
D. **C2 (CI regression test)** — runs existing code
E. **A4 (adaptive chunk size)** — env var tuning, no install

## User has 3 options

1. **Authorize `pip install lmcache` specifically** (so I can do R25)
2. **Choose one of A1 / A3 / A4 / S0 / C2 instead** (no install needed)
3. **Stop and commit current state** (R19 BEST as final delivery)
