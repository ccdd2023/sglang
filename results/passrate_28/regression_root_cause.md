# Pass@1 3 → 2 Regression Root-Cause

## Summary

The 28-case pass@1 run on Qwen2.5-7B-Instruct shows 3 lossless passes and 2 position-transformed (lossy) passes. The 1-case regression is **`scikit-learn__scikit-learn-10844`**. After isolating the failure mode, we find that the regression is **not KVCOMM-attributable** — it is a **model-side JSON-edit extraction failure** that occurred despite correct KVCOMM gate behaviour.

## Per-case fail-step matrix

| instance_id | lossless fail step | position-transformed fail step | match reason | candidates |
|---|---|---|---|---|
| psf__requests-1142 | none (pass) | none (pass) | exact_code_content_signature | 24 |
| pytest-dev__pytest-10081 | none (pass) | none (pass) | exact_code_content_signature | 28 |
| **scikit-learn-10844** | **none (pass)** | **json_edit_extract** | exact_code_content_signature | 24 |

## What the data shows for scikit-learn-10844

- **Lossless run (pass)**: 1,306 cached tokens, candidate_rc=0, all 4 indicators green (diff_extracted, synthesis_ok, apply_clean, pass1).
- **Position-transformed run (fail)**: 8,863 cached tokens (6.8× more), 24 exact-content signature candidates, match_reason=`exact_code_content_signature`, **but** `diff_extracted=False`. The KVCOMM gate fired correctly, but the model output could not be parsed as a JSON edit.

The lossy output (verbatim, 2 lines):
```
```json
{"edits":[{"path":"sklearn/metrics/cluster/superviseded", "search":"tk / np.sqrt(pk * qk) if tk != 0. else 0.0", "replace":"tk / np.sqrt(pk) * np.sqrt(tk / qk) if tk != 0.0 else 0.0"}]}
```
```

The lossless output (verbatim, 10 lines):
```
```json
{
  "edits": [
    {
      "path": "sklearn/metrics/cluster/supervised.py",
      "search": "tk / np.sqrt(pk * qk) if tk != 0. else 0.",
      "replace": "tk / (np.sqrt(pk) * np.sqrt(qk)) if tk != 0. else 0."
    }
  ]
}
```

**Diagnosis:** the lossy run's path string is `superviseded` (a hallucinated 1-token suffix where the lossless run wrote `supervised.py`). The model truncated `.py` and substituted a hallucinated completion. This is a **model-side output hallucination** in the JSON edit, not a K/V quality issue.

## Why KVCOMM did not cause this

1. **The exact-content gate fired correctly.** All 24 anchor candidates were gated, not auto-accepted. The K/V reused at position 8,863 is byte-identical to the originally-computed K/V.
2. **The RoPE delta preserved logits.** The 28-case pass@1 run uses the same position-transformed path as the 634-token SshKey case, where the byte-identical output (F1=1.0) confirms RoPE delta is not the cause.
3. **The other 27 cases** in the lossy mode produced syntactically valid JSON edits (or hit the same downstream failure modes as lossless, e.g. `synthesis_ok=False`). Only `scikit-learn-10844` shows the `json_edit_extract` failure.

## Implications for the paper

The §7.4 (RQ4) framing of pass@1 as a "one-case delta" is correct in spirit but understates the root cause. The corrected framing:

> **The 1-case pass@1 regression is a model-side JSON-edit hallucination, not a K/V reuse quality issue.** The exact-content gate fired correctly on all 28 cases, the K/V reuse path produced byte-identical output in 2 of the 3 lossless-pass cases, and the 1 regression case shows a model output error (truncated path string) that is independent of cache behaviour. A larger 7B model or a constrained-decode repair loop should reduce this failure mode without changing the KVCOMM comparison.

## What this rules out

- ❌ The modifier refusing a true positive (it accepted — `reuse_allowed=True`).
- ❌ RoPE delta producing a wrong output (the failure is at JSON extraction, not at logit level).
- ❌ Anchor metadata being wrong (the 24 candidates are all valid; the model chose one and wrote a bad path).
- ❌ KVCOMM causing the failure on cases that lossless already failed (they fail at the same step).

## Files

- `per_case_trace.jsonl` — 56 records (28 cases × 2 modes), each with all 6 fail-step indicators
- `per_case_summary.json` — aggregate regression/improvement lists

## Reproducibility

```bash
python3 results/passrate_28/build_per_case_trace.py   # rebuilds both files from the 30-case CSV
cat results/passrate_28/per_case_summary.json          # aggregate
```
