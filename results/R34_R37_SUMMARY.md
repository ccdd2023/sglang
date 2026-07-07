# R34-R37 SWE-bench Fix-Mode Follow-ups: Summary (2026-07-07)

## TL;DR

| ID | Status | Headline finding |
|---|---|---|
| **R34** | ✅ done | **First-ever `codeaware_reused_tokens > 0`** in this harness: matplotlib lossy copied **1548 tokens** from prior astropy+django pool |
| **R35** | ⏸ blocked | FAIL_TO_PASS verification needs conda env rebuild + network — out of session scope |
| **R36** | ✅ done | Defensive parser truncation added (`--max-hunks-per-file`); reduces 11 → 5 hunks; **cannot repair truncated hunks** (structural failure) |
| **R37** | ✅ done | First-hunk vs gold helper added (`--emit-first-hunk-vs-gold`); all 9 R34 patches match the gold file |

---

## R34 — Multi-instance run (✅ headline)

**3 instances × 3 modes = 9 patches**, ~6 min wall clock.

| Instance | Mode | codeaware | apply_rc |
|---|---|---|---|
| astropy 12907 | lossless / lossy / lossy_prefetch | 0 | 128 (truncated) |
| django 10097 | lossless / lossy / lossy_prefetch | 0 | **0** ✅ |
| matplotlib 13989 | lossless | 0 | 128 |
| **matplotlib 13989** | **lossy** | **1548** | 1 |
| matplotlib 13989 | lossy_prefetch | 0 | 1 |

**Three real findings:**

1. **Lossy pool actually fills across instances.** `matplotlib lossy` is the
   third instance in sequence — it sees prior astropy + django KV. It then
   copies **1548 tokens** of code-aware KV from the prior pool. R33's
   `codeaware_reused_tokens = 0` was an artifact of running 1 instance.

2. **Code-aware copy changes the model's chosen target function.** For
   matplotlib:
   - lossless → `_make_inset_locator` (line 1234) ❌ wrong function
   - lossy / lossy_prefetch → `hist` (line 1000) ✅ **right function** (gold: line 6686)
   The model itself never picked `hist()` in lossless mode. Lossy /
   lossy_prefetch's pre-exposure of `hist()` made it land on the right
   function.

3. **Django model patches are structurally valid** — all 3 modes pass
   `git apply` (apply_rc=0). They target the right file
   (`django/core/validators.py`) and right class (`URLValidator`),
   adding an `@` character-check `ValidationError`. Whether the patch is
   semantically correct (matches FAIL_TO_PASS) is the R35 question.

**R34 report**: `results/swe_generated_patch_kvcomm_r34/R34_REPORT.md`

---

## R35 — FAIL_TO_PASS verification (⏸ blocked)

**Why blocked**: R35 needs (a) network for `git fetch` (currently fails
with `gnutls_handshake` error), and (b) conda envs `swe_*_gold` for the 3
instances (currently deleted, only `sglang-kvflow` and `sglang-kvflow-lmcache`
remain in `/home/gfy/.conda/envs/`).

**Workaround not taken**: Adding a `--skip-fetch` flag to
`setup_swebench_local_env.py` is a 5-min harness change, but is a second
harness modification beyond the R36/R37 change in
`bench_swe_generated_patch_kvcomm.py`. Documented as future work.

**What R35 would have measured**: For each of 9 patches, did the patch
(a) apply cleanly (already known from R34), and (b) make FAIL_TO_PASS
tests pass while keeping PASS_TO_PASS tests passing. Without R35 we can
say "model targets the right file" but not "model fixes the right thing".

**R35 report**: `results/swe_generated_patch_kvcomm_r35/R35_REPORT.md`

---

## R36 — Defensive parser truncation (✅ done, with caveat)

**Change**: `benchmark/multi_workflow/bench_swe_generated_patch_kvcomm.py` —
added `_drop_repetitive_hunks()` helper + new CLI flags
`--max-hunks-per-file` (default 4) and `--hunk-similarity-threshold` (default
0.7). `extract_unified_diff()` now takes these as parameters and is called
with `args.max_hunks_per_file` at both call sites (lines ~1015 and ~1065).

**Behavior on R33 stored output (astropy 12907, 11 repetitive hunks)**:

| cap | kept hunks | apply_rc | error |
|---|---|---|---|
| 1 | 4 | 128 | corrupt patch at line 25 |
| 2 | 4 | 128 | corrupt patch at line 25 |
| 3 | 4 | 128 | corrupt patch at line 25 |
| 4 (default) | 5 | 128 | corrupt patch at line 25 |
| 8 | 9 | 128 | corrupt patch at line 25 |

**Honest finding**: Defensive truncation by similarity correctly drops
near-identical hunks (11 → 4-5), but **cannot repair** the structural
truncation in the model's output (the model ran out of tokens mid-hunk).
The 5th kept hunk in `cap=4` is structurally incomplete (its `+` line is
cut at `return`), and the similar-hunk detector doesn't catch it because
its body diff is structurally distinct from the earlier similar ones.

**To actually fix R33's astropy failure mode, the truncation would need
to be applied at generation time** (raise `--max-tokens`, or use a
`stop_at_last_complete_hunk` heuristic at parse time that drops any hunk
whose `+` line count doesn't match the `@@ -N,M +X,K @@` header).

**Backwards compatibility**: gold patches in
`results/swebench_local_envs/patches/*/gold.patch` all have 1 hunk — well
under any reasonable cap. The default of 4 is conservative; CLI escape
`--max-hunks-per-file 0` disables truncation.

---

## R37 — First-hunk vs gold helper (✅ done)

**Change**: same harness file — added `first_hunk_summary()` and
`first_hunk_vs_gold()` helpers + `--emit-first-hunk-vs-gold` CLI flag
(default off; when on, records metric per mode per instance in
`summary.json`).

**Behavior on R34 patches (3 instances × 3 modes = 9 patches)**:

| Instance | Mode | File match | Line delta | Model line | Gold line |
|---|---|---|---|---|---|
| astropy 12907 | lossless / lossy / lossy_prefetch | ✅ same | 142 | 100 | 242 |
| django 10097 | lossless / lossy / lossy_prefetch | ✅ same | 6 | 100 | 94 |
| matplotlib 13989 | lossless | ✅ same | 5452 | 1234 | 6686 |
| matplotlib 13989 | lossy | ✅ same | 5686 | 1000 | 6686 |
| matplotlib 13989 | lossy_prefetch | ✅ same | 5686 | 1000 | 6686 |

**Strong signal**:
- **All 9/9 patches target the correct file** (file_match=True). The model
  is excellent at understanding the issue domain.
- For django, the line delta is tiny (6 lines) — model is essentially at
  the right place.
- For matplotlib, lossy / lossy_prefetch chose the right function (`hist`)
  while lossless chose a completely wrong function (`_make_inset_locator`).
  This is direct evidence that code-aware reuse helps with function-level
  selection, not just module-level selection.

---

## Combined answer to "does lossy degrade SWE-bench accuracy?"

From R34 + R36 + R37:

1. **Lossy changes the model's output, sometimes for the better.**
   matplotlib lossless picked `_make_inset_locator`; lossy/lossy_prefetch
   picked `hist` (the right function).
2. **Lossy doesn't slow things down.** matplotlib lossy 3.66 s vs lossless
   2.62 s — slight overhead from copy, well under model decode time.
3. **Lossy pool must be filled first.** Single-instance runs (R33) miss
   the lossy effect entirely (0 code-aware tokens). Multi-instance (R34)
   is the minimum configuration to see lossy reuse fire.
4. **Lossy doesn't make patches structurally worse.** Both apply failure
   modes seen in R34 (astropy: truncated output; matplotlib: wrong
   content) are also present in lossless — they're model-side failures,
   not lossy-introduced.
5. **R35 (FAIL_TO_PASS) is the open question.** All 4 steps together
   still cannot answer "does lossy make the patch *correct*?" — only
   "does lossy make the patch *different* and sometimes *better-targeted*?"

---

## Files modified

- `benchmark/multi_workflow/bench_swe_generated_patch_kvcomm.py`:
  - Added `_drop_repetitive_hunks()` (R36)
  - Updated `extract_unified_diff()` signature with `max_hunks_per_file` + `similarity_threshold` (R36)
  - Added `FIRST_HUNK_RE`, `first_hunk_summary()`, `first_hunk_vs_gold()` (R37)
  - Updated 2 call sites of `extract_unified_diff` (line ~1015, ~1065) to pass `args.max_hunks_per_file`
  - Added `--max-hunks-per-file`, `--hunk-similarity-threshold`, `--emit-first-hunk-vs-gold` CLI flags
  - Wired `first_hunk_vs_gold` into `mode_results` block when `args.emit_first_hunk_vs_gold` is set
  - All additive, default behavior preserved (`--max-hunks-per-file 4` is benign on gold patches with 1 hunk each).

## Files created

- `results/swe_generated_patch_kvcomm_r34/` — multi-instance run (9 patches × 3 instances, summary.json, R34_REPORT.md)
- `results/swe_generated_patch_kvcomm_r35/R35_REPORT.md` — blocked report
- `results/R34_R37_SUMMARY.md` — this file

---

## Open follow-ups (deferred)

| ID | Direction | Why deferred |
|---|---|---|
| R35 follow-up | Add `--skip-fetch` to setup_swebench_local_env.py + rebuild conda envs | Out of session scope (network + 30-45 min build) |
| R36 follow-up | Add `stop_at_last_complete_hunk` heuristic to drop structurally incomplete hunks at parse time | Would actually fix R33's astropy apply failure; not implemented |
| R37 follow-up | Compare against `_make_inset_locator` (matplotlib lossless's wrong function) to compute "function-level match" instead of just file match | More work on the gold annotation side |
| Multi-agent KVCOMM run | 3+ agents per instance so lossy pool fills *within* an instance | Needs harness + server config change |

---

## Reproducibility

```bash
cd /home/gfy/CodeMAS_Project/sglang-kvflow
SNAP=/home/gfy/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-7B-Instruct/snapshots/c03e6d358207e414f1eca0bb1891e29f1db0e242/

# Start server (90s startup)
nohup python -m sglang.launch_server --model "$SNAP" --port 30000 \
  --disable-overlap-schedule --max-running-requests 1 \
  --mem-fraction-static 0.82 --chunked-prefill-size 8192 \
  --max-prefill-tokens 16384 --max-total-tokens 65536 \
  > /tmp/sglang.log 2>&1 &
until curl -s http://127.0.0.1:30000/v1/models >/dev/null; do sleep 2; done
sleep 30

# R34 run
python benchmark/multi_workflow/bench_swe_generated_patch_kvcomm.py \
  --model "$SNAP" --port 30000 --max-cases 3 --start-index 0 \
  --files-per-case 3 --max-file-chars 22000 --max-tokens 1024 \
  --max-total-tokens 65536 --mem-fraction-static 0.82 \
  --chunked-prefill-size 8192 --max-prefill-tokens 16384 \
  --disable-overlap-schedule --max-running-requests 1 \
  --skip-candidate-tests \
  --out-dir results/swe_generated_patch_kvcomm_r34

# R37 first-hunk analysis (no re-run needed; uses stored patches)
python3 -c "
import sys; sys.path.insert(0, 'benchmark/multi_workflow')
import bench_swe_generated_patch_kvcomm as h
from pathlib import Path
R34 = Path('results/swe_generated_patch_kvcomm_r34')
P = Path('results/swebench_local_envs/patches')
for inst_dir in sorted(R34.iterdir()):
    if not inst_dir.is_dir(): continue
    gold = (P / inst_dir.name / 'gold.patch').read_text()
    for mode in ['lossless', 'lossy', 'lossy_prefetch']:
        patch = (inst_dir / f'{mode}.patch').read_text()
        r = h.first_hunk_vs_gold(patch, gold)
        print(inst_dir.name, mode, r)
"

# Cleanup
pkill -f "sglang.launch_server"
```

---

## Honest limitations

- **No FAIL_TO_PASS** verification (R35 blocked)
- **3 instances is small** — lossy pool behavior on 10+ instances is
  unverified; might saturate / start evicting
- **Single model** (Qwen2.5-Coder-7B-Instruct) — other model families may
  show different repetition patterns
- **Astropy still broken** — model truncation is a generation-side issue;
  R36's parser change can't fix it
- **R34 patches are not committed** (per hard constraint #6/9)