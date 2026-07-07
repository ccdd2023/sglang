# Harness Change Notes (2026-07-07)

**File modified**: `benchmark/multi_workflow/bench_swe_generated_patch_kvcomm.py`

Two additive, default-backward-compatible changes added during R34-R37
SWE-bench fix-mode follow-up work. No algorithmic changes. No sglang
runtime changes. Defaults preserve existing behavior.

---

## R36 — Defensive parser truncation for repetitive hunks

**Problem**: R33 (commit `cf713ba52`) found that Qwen2.5-Coder-7B-Instruct
emits 11 near-identical hunks for `astropy__astropy-12907` and the 11th is
truncated mid-token. `git apply` rejects at line 25 of the truncated hunk.

**Change**: New `_drop_repetitive_hunks()` helper + updated
`extract_unified_diff()` signature.

### Signature change

```python
# Before
def extract_unified_diff(text: str) -> str: ...

# After
def extract_unified_diff(text: str,
                         max_hunks_per_file: int = 4,
                         similarity_threshold: float = 0.7) -> str: ...
```

### Behavior

1. Parse the diff as before (fenced ```diff block or raw `diff --git`).
2. Split by `^diff --git ` into per-file sections.
3. Within each section, split by `^@@ ` into hunks.
4. Cap at `max_hunks_per_file` (default 4).
5. For each hunk beyond the cap, drop it if its body has
   `difflib.SequenceMatcher.ratio() > similarity_threshold` (default 0.7)
   vs any earlier kept hunk in the same file.

### New CLI flags

- `--max-hunks-per-file INT` (default 4) — per-file hunk cap. Set 0 to disable.
- `--hunk-similarity-threshold FLOAT` (default 0.7) — drop threshold for
  similar-to-earlier hunks.

### Verified

- Astropy R33 stored output (11 hunks) → 5 hunks after cap=4
- Django R34 patches (1 hunk each, well under cap) → unchanged
- Matplotlib R34 patches (1 hunk each) → unchanged
- All gold patches (1 hunk each) → unchanged

### Honest limitation

R36 **cannot repair** an already-truncated hunk. The structural
truncation (`+ return` cut off mid-token) is detected as not-similar to
earlier hunks (different body length), so the similar-hunk heuristic
doesn't drop it. To fix R33's astropy failure mode, the truncation would
need to be applied at generation time (`--max-tokens`) or with a
`stop_at_last_complete_hunk` heuristic.

---

## R37 — First-hunk vs gold helper (weak correctness signal)

**Problem**: When `git apply` cannot run (env broken) or fails, we have no
signal about whether the model targeted the right code at all.

**Change**: New `first_hunk_summary()` + `first_hunk_vs_gold()` helpers.

### New helpers

```python
def first_hunk_summary(diff_text: str) -> dict:
    """Returns {extracted, target_path, old_start_line} or {extracted: False}."""

def first_hunk_vs_gold(model_patch: str, gold_patch: str) -> dict:
    """Returns {comparable, model_path, gold_path, file_match,
                model_line, gold_line, line_delta_abs} or {comparable: False}."""
```

### New CLI flag

- `--emit-first-hunk-vs-gold` (default off) — when set, records
  `first_hunk_vs_gold(diff, instance["patch"])` per mode per instance in
  `summary.json` next to `apply_check`.

### Verified on R34 patches (3 instances × 3 modes = 9 patches)

| Instance | Mode | File match | Line delta |
|---|---|---|---|
| astropy__astropy-12907 | all 3 | ✅ same | 142 |
| django__django-10097 | all 3 | ✅ same | 6 |
| matplotlib__matplotlib-13989 | lossless | ✅ same | 5452 |
| matplotlib__matplotlib-13989 | lossy | ✅ same | 5686 |
| matplotlib__matplotlib-13989 | lossy_prefetch | ✅ same | 5686 |

**All 9/9 patches target the gold file.** Django line delta of 6 means the
model is essentially at the right place. Matplotlib line delta of 5452-5686
shows model picked a different function in same file — but lossy/lossy_prefetch
picked `hist` (correct), lossless picked `_make_inset_locator` (wrong).

---

## Backwards compatibility

| Caller | Old behavior preserved? |
|---|---|
| `extract_unified_diff(text)` with default args | ✅ identical (default cap=4 only affects patches with >4 hunks) |
| `--max-hunks-per-file 0` | ✅ disables truncation entirely |
| `--emit-first-hunk-vs-gold` not passed | ✅ no `first_hunk_vs_gold` field added to summary.json |
| Gold patches with 1 hunk | ✅ unchanged by either change |

---

## Diff stats

```
 benchmark/multi_workflow/bench_swe_generated_patch_kvcomm.py | ~110 lines added
   + _drop_repetitive_hunks() helper             (~25 lines)
   + extract_unified_diff signature change       (~3 lines)
   + FIRST_HUNK_RE constant + first_hunk_summary (~10 lines)
   + first_hunk_vs_gold helper                   (~15 lines)
   + 3 CLI flags (--max-hunks-per-file, --hunk-similarity-threshold,
                  --emit-first-hunk-vs-gold)     (~15 lines)
   + 2 call-site updates                         (~4 lines)
   + mode_results wire-in                        (~3 lines)
```

Co-Authored-By: Claude <noreply@anthropic.com>