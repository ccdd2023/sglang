# v44 placeholder_knn_reuse Code Correctness Validation Report

**Date**: 2026-06-24
**Author**: fy, Claude-assisted
**Branch**: phase-2.7-prerot (commit 16d6fc681)
**Plan reference**: `/home/gfy/.claude/plans/session-handoff-2026-06-23-md-code-awar-drifting-aurora.md`

---

## TL;DR

[fill after runs complete]

- v44 placeholder_knn_reuse **passes / fails** the handoff §6.5 SWE-bench pass@1 ≤ 2pp regression gate on 10 cases
- [if fails] Root cause: [...]

---

## What was tested

| Item | Detail |
|---|---|
| Dataset | `results/repo_level_datasets/swe_verified_10_instances.json` (10 cases across astropy/django/matplotlib/mwaskom/pallets/psf/pydata/pylint-dev) |
| Model | Qwen2.5-3B-Instruct (default for `bench_swe_generated_patch_kvcomm`) |
| Modes | `lossless`, `lossy`, `lossy_prefetch` (baseline) + `placeholder_knn_lossy` (v44) |
| Eval | `setup_swebench_local_env.py --mode candidate --candidate-patch <patch>` → returncode 0 = pass |
| Server flags | `--disable-overlap-schedule --max-running-requests 1 --force-evict` (mandatory for > 3 cases; see memory `_delete-leaf-bug-2026-06-24`) |
| Compute | RTX 4090 24GB |
| Wall time | ~25-40 min for both baseline + v44 |

## What was NOT tested

- Codebase context selection / selective AST reuse accuracy (covered separately in §3 of handoff)
- Multi-agent (agent_count > 1) — these runs use agent_count = 1 implicit
- sympy — `swebench_local_envs/repos/` does NOT contain sympy (per handoff §6.5)
- HumanEval pass@1 — Phase 2.2 not run in this iteration; will be follow-up

## Results

### Baseline (lossless / lossy / lossy_prefetch)

```
[fill from aggregate_swe_pass_at_1.py output]
```

### v44 placeholder_knn_lossy

```
[fill from aggregate_swe_pass_at_1.py output]
```

### Pass@1 comparison

| mode | baseline pass@1 | v44 pass@1 | Δ pp | gate (-2pp) |
|---|---|---|---|---|
| lossless | X / 10 | (same — not run by v44) | — | — |
| lossy | Y / 10 | (same — not run by v44) | — | — |
| placeholder_knn_lossy | (not in baseline) | Z / 10 | — | — |

Cross-mode comparison (lossy baseline vs placeholder_knn_lossy v44):
- Δ = (Z - Y) / 10
- Required: Δ ≥ -2pp
- Actual: Δ = [...]

## Per-case delta (v44 vs lossy baseline)

| case_id | lossy pass | v44 pass | Δ |
|---|---|---|---|
| astropy__astropy-12907 | ... | ... | ... |
| ... | | | |

## Verdict

[ ] **v44 PASSES §6.5 gate** (regression ≤ 2pp on 10 SWE cases)
[ ] **v44 FAILS §6.5 gate** — root cause: [...]

## Companion gate: §6.8 F1-skip rate (PASS, telemetry only)

Already validated 2026-06-24 against v44 telemetry data:
- `placeholder_anchor_store_skipped_low_f1_count` = 0 / 30 anchor entries = **0.00%** (gate < 5%)
- sim_mean = 0.9851-0.9989 in v44 placeholder_knn_reuse mode

This is anchor-store hygiene, NOT code correctness. See memory `v44-f1-skip-gate-pass.md` for the explicit caveat.

## Next steps

- [ ] If PASS: scale to Phase 5 (50 cases including sympy env)
- [ ] If FAIL: enter Phase 4 (tune v44 implementation: topk_sim ≥ 0.99, K=1, O5-lite pre-rot)
- [ ] Phase 2.2 HumanEval 20-case pass@1
- [ ] Phase 3 threshold sweep (if v44 degradation is monotonic in threshold)

## Risks / Caveats

- 10 cases is too small for production claims; treat as smoke validation
- Qwen2.5-3B-Instruct (not 7B) was the model used — bench_swe default; rerun on 7B if results matter
- Single concurrent request (`--max-running-requests 1`); multi-agent scenarios untested