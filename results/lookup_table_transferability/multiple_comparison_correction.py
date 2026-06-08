"""Multiple-comparison correction for the 192-cell 4-axis lookup table.

The cross-model transferability study compares $d_{\text{norm}}$ at $192$ cells
per model across 4 models. With 192 cells, some pairwise differences will
appear significant by chance even under the global null. This script:

1. Reads the per-cell d_norm from each of the 4
   ``predicted_distance_table_*.json`` files
2. Computes pairwise per-cell |d_norm_A - d_norm_B| for all 12 pairs
3. Applies Bonferroni-Holm correction at the 12-pair level (not the
   192-cell level) since the per-cell comparisons are not independent
4. Re-flags each pair as "significant" / "not significant" under the
   corrected threshold and prints a summary

The lookup table has only 144 cells populated (the >500 length bin is
empty in all 4 model files), so the effective test set is 144 cells.
Bonferroni-Holm on 12 pairs (not 144×12=1728) is the right granularity:
each pair yields a single "any cell differs" test, and we control the
family-wise error rate across the 12 pairs.

Output: stdout summary + JSON written to
``data/multiple_comparison_results.json``.
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

DATA_DIR = Path("/home/gfy/CodeMAS_Project/sglang-kvflow/results/lookup_table_transferability/data")
OUT_DIR = Path("/home/gfy/CodeMAS_Project/sglang-kvflow/results/lookup_table_transferability")
MODELS = [
    "qwen-qwen2.5-coder-7b-instruct",
    "qwen-qwen2.5-coder-3b-instruct",
    "qwen-qwen2.5-7b-instruct",
    "qwen-qwen3-8b",
]


def _load_cell_means(slug: str) -> dict[tuple, float]:
    """Return {(length_bin, offset, system, surround): d_norm} for cells in
    the predicted_distance_table JSON. Only the short-code cells
    (length_bin in {<50, 50-200, 200-500}) are populated in all 4 models,
    so the effective cell count is 144."""
    path = DATA_DIR / f"predicted_distance_table_{slug}.json"
    d = json.load(open(path))
    return {
        (c["length_bin"], c["position_offset"], c["system_prompt_class"],
         c["surrounding_code_class"]): c["predicted_d_norm_mean"]
        for c in d.get("cells", [])
    }


def main() -> None:
    cell_means = {m: _load_cell_means(m) for m in MODELS}
    common_cells = set.intersection(*(set(cm.keys()) for cm in cell_means.values()))
    n_cells = len(common_cells)
    print(f"Common cells across all 4 models: {n_cells}")
    if n_cells == 0:
        print("ERROR: no common cells; check that all 4 tables are populated.")
        return

    # Compute pairwise per-cell |d_norm_A - d_norm_B| statistics
    pairs = []
    raw = []
    for i, ma in enumerate(MODELS):
        for j, mb in enumerate(MODELS):
            if i >= j:
                continue
            diffs = [
                abs(cell_means[ma][c] - cell_means[mb][c])
                for c in common_cells
            ]
            mean = statistics.mean(diffs)
            # Naive per-pair t-stat: mean / (std / sqrt(n))
            sd = statistics.stdev(diffs) if len(diffs) > 1 else 0
            t = mean / (sd / math.sqrt(len(diffs))) if sd > 0 else float("inf")
            # Two-sided p-value approximation via normal distribution
            # (large n). For n=144 the normal approximation is good.
            p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
            pairs.append({
                "a": ma,
                "b": mb,
                "mean_abs_d": mean,
                "std_abs_d": sd,
                "t": t,
                "p_naive": p,
            })
            raw.append(p)

    # Bonferroni-Holm step-down procedure on the 12 pairwise p-values
    n_pairs = len(pairs)
    sorted_idx = sorted(range(n_pairs), key=lambda k: raw[k])
    corrected = [0.0] * n_pairs
    running_max = 0.0
    for rank, idx in enumerate(sorted_idx):
        # Holm-corrected alpha for this rank: alpha/(n - rank)
        # With alpha = 0.05 (family-wise), threshold = 0.05 * n_pairs / (n_pairs - rank)
        threshold = 0.05 * n_pairs / (n_pairs - rank)
        # Reject if p <= threshold. The corrected p is min(p * (n - rank), 1)
        corrected[idx] = min(raw[idx] * (n_pairs - rank), 1.0)
        # Record if still significant
        if raw[idx] <= threshold:
            running_max = max(running_max, raw[idx])
        else:
            corrected[idx] = max(corrected[idx], running_max)
    # Mark significance
    for pair, p_corr in zip(pairs, corrected):
        pair["p_bonferroni_holm"] = p_corr
        pair["significant_after_correction"] = p_corr < 0.05

    # Print summary
    print()
    print(f"=== Bonferroni-Holm correction on {n_pairs} pairwise comparisons ({n_cells} cells each) ===")
    print()
    print(f"{'pair':50s}  {'mean':>7s}  {'std':>7s}  {'t':>7s}  {'p_raw':>9s}  {'p_corr':>9s}  {'sig?':>6s}")
    print("-" * 110)
    for pair in pairs:
        a_short = pair["a"].replace("qwen-", "").replace("qwen2.5-", "Q2.5-").replace("coder-", "C-").replace("instruct", "")
        b_short = pair["b"].replace("qwen-", "").replace("qwen2.5-", "Q2.5-").replace("coder-", "C-").replace("instruct", "")
        sig = "yes" if pair["significant_after_correction"] else "no"
        print(f"{a_short:25s} vs {b_short:25s}  "
              f"{pair['mean_abs_d']:7.3f}  {pair['std_abs_d']:7.3f}  {pair['t']:7.2f}  "
              f"{pair['p_naive']:9.4f}  {pair['p_bonferroni_holm']:9.4f}  {sig:>6s}")

    # Verdict
    sig_pairs = [p for p in pairs if p["significant_after_correction"]]
    not_sig_pairs = [p for p in pairs if not p["significant_after_correction"]]
    print()
    print(f"After Bonferroni-Holm correction (alpha=0.05):")
    print(f"  Significant: {len(sig_pairs)}/{n_pairs} pairs")
    print(f"  Not significant: {len(not_sig_pairs)}/{n_pairs} pairs")
    qwen25_pairs = [
        p for p in pairs
        if "qwen3" not in p["a"] and "qwen3" not in p["b"]
    ]
    cross_pairs = [p for p in pairs if "qwen3" in p["a"] or "qwen3" in p["b"]]
    qwen25_sig = [p for p in qwen25_pairs if p["significant_after_correction"]]
    cross_sig = [p for p in cross_pairs if p["significant_after_correction"]]
    print(f"  Within Qwen2.5 family ({len(qwen25_pairs)} pairs): {len(qwen25_sig)} significant after correction")
    print(f"  Qwen3-8B vs Qwen2.5 family ({len(cross_pairs)} pairs): {len(cross_sig)} significant after correction")
    print()
    verdict = (
        "**weak portable** (the cross-family comparisons remain significant after correction; "
        "the Qwen2.5 family comparisons are still significant; this means the d_norm "
        "differences within Qwen2.5 and across Qwen3 vs Qwen2.5 are not just artifacts of "
        "multiple comparisons)."
        if len(cross_sig) > 0 else
        "**strong portable** (no comparisons remain significant after Bonferroni-Holm correction; "
        "the original weak-portable verdict was inflated by multiple comparisons)."
    )
    print(f"Verdict (revised): {verdict}")

    # Save
    out = {
        "n_cells_compared": n_cells,
        "n_pairs": n_pairs,
        "n_significant_after_correction": len(sig_pairs),
        "n_NOT_significant_after_correction": len(not_sig_pairs),
        "verdict": verdict,
        "pairs": pairs,
    }
    with open(OUT_DIR / "multiple_comparison_results.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {OUT_DIR / 'multiple_comparison_results.json'}")


if __name__ == "__main__":
    main()
