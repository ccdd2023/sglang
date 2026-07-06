#!/usr/bin/env python3
"""R25-A1: Per-chunk F1 oracle — learn which chunk features predict garbage output.

Approach:
1. Load R21-R23 verdict-mode runs (6 configs × 25 rows = 150 examples)
2. For each (case, agent) row, extract chunk-pool telemetry features
3. Label = 1 if output is UNK (broken/garbage), 0 if proper verdict
4. Fit a simple classifier (logistic regression with regularization)
5. Identify which features most predict garbage
6. Propose a per-decision skip policy

Output:
- Trained model coefficients
- Suggested SGLANG_ORACLE_SKIP_THRESHOLD env var value
- Cross-validation accuracy
"""
from __future__ import annotations
import csv, json, re
from collections import defaultdict
from pathlib import Path
import math

P = re.compile(r"\bVERDICT:\s*(PASS|FAIL)\b", re.I)
CONFIG_RUNS = [
    "lossy_alg_round21/lossless_verdict",
    "lossy_alg_round21/r17_verdict",
    "lossy_alg_round21/r19_verdict",
    "lossy_alg_round22/r22a_frac03",
    "lossy_alg_round22/r22b_verdict_pool",
    "lossy_alg_round23/r23_per_role",
]
FEATURE_COLS = [
    "c2_chunk_reused_tokens",
    "codeaware_reused_tokens",
    "lossy_anchor_match_gap_len",
    "lossy_anchor_rope_delta",
    "placeholder_chunk_pool_hit_count",
    "placeholder_chunk_pool_miss_count",
    "placeholder_chunk_pool_total_chunks_stored",
    "placeholder_chunk_pool_skip_byte_drift_count",
    "placeholder_chunk_pool_skip_no_entry_count",
    "placeholder_chunk_pool_skip_size_mismatch_count",
    "placeholder_chunk_pool_skip_gap_count",
    "placeholder_chunk_pool_skip_selective_refresh_count",
    "placeholder_chunk_pool_total_tokens_reused",
    "placeholder_chunk_pool_total_tokens_dense",
    "placeholder_chunk_pool_blend_stage_count",
    "placeholder_chunk_pool_blend_gap_tokens",
    "placeholder_chunk_pool_blend_run_tokens",
    "placeholder_anchor_pool_hit_count",
    "placeholder_knn_pre_rotated_hit_count",
    "placeholder_knn_pre_rotated_miss_count",
    "radix_prefix_tokens",
    "radix_only_prefix_len",
    "l2_wholeslot_reused_tokens",
    "l3_offset_reused_tokens",
    "agent_idx",
]


def load_features_and_labels():
    """Each row becomes a (feature_dict, label) pair. Label=1 if output UNK."""
    base = Path("/home/gfy/CodeMAS_Project/sglang-kvflow/results")
    X = []
    y = []
    meta = []  # for debugging
    for run in CONFIG_RUNS:
        out_p = base / run / "outputs.jsonl"
        csv_p = base / run / "rows.csv"
        if not out_p.exists() or not csv_p.exists():
            continue
        # outputs: map (case_id, agent_idx) -> text
        out_rows = [json.loads(l) for l in Path(out_p).read_text().splitlines() if l.strip()]
        verdicts = {(r['case_id'], r['agent_idx']): r.get('output_text', '') for r in out_rows}
        # csv: same keys, with feature columns
        csv_rows = list(csv.DictReader(open(csv_p)))
        for r in csv_rows:
            cid = r['case_id']
            try:
                aid = int(r['agent_idx'])
            except (KeyError, ValueError):
                continue
            text = verdicts.get((cid, aid), '')
            m = P.search(text or '')
            label = 0 if m else 1  # 1 = UNK garbage
            feats = {}
            for c in FEATURE_COLS:
                try:
                    feats[c] = float(r.get(c, 0) or 0)
                except (KeyError, ValueError):
                    feats[c] = 0.0
            X.append(feats)
            y.append(label)
            meta.append((run, cid, aid))
    return X, y, meta


def standardize(X, mean=None, std=None):
    if mean is None:
        mean = {k: sum(x[k] for x in X) / len(X) for k in FEATURE_COLS}
    if std is None:
        std = {}
        for k in FEATURE_COLS:
            m = mean[k]
            v = sum((x[k] - m) ** 2 for x in X) / len(X)
            std[k] = math.sqrt(v) or 1.0
    Xn = [{k: (x[k] - mean[k]) / std[k] for k in FEATURE_COLS} for x in X]
    return Xn, mean, std


def sigmoid(z):
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def fit_logreg(X, y, lr=0.05, n_iter=400, l2=0.01):
    """Plain logistic regression with L2."""
    n = len(X)
    w = {k: 0.0 for k in FEATURE_COLS}
    b = 0.0
    for it in range(n_iter):
        gw = {k: 0.0 for k in FEATURE_COLS}
        gb = 0.0
        for x, yi in zip(X, y):
            z = b + sum(w[k] * x[k] for k in FEATURE_COLS)
            p = sigmoid(z)
            err = p - yi
            for k in FEATURE_COLS:
                gw[k] += err * x[k]
            gb += err
        for k in FEATURE_COLS:
            w[k] -= lr * (gw[k] / n + l2 * w[k])
        b -= lr * (gb / n)
    return w, b


def predict(X, w, b, threshold=0.5):
    p = [sigmoid(b + sum(w[k] * x[k] for k in FEATURE_COLS)) for x in X]
    return [int(pi >= threshold) for pi in p], p


def main():
    X, y, meta = load_features_and_labels()
    print(f"Loaded {len(X)} rows from {len(CONFIG_RUNS)} runs")
    print(f"Label distribution: UNK={sum(y)}, OK={len(y) - sum(y)}")

    # 5-fold cross-validation
    n = len(X)
    k = 5
    k_int = k
    fold_size = n // k
    accs = []
    aucs = []
    for fold in range(k):
        start = fold * fold_size
        end = (fold + 1) * fold_size if fold < k - 1 else n
        Xtr = X[:start] + X[end:]
        ytr = y[:start] + y[end:]
        Xte = X[start:end]
        yte = y[start:end]
        Xtr_n, mean, std = standardize(Xtr)
        w, b = fit_logreg(Xtr_n, ytr, lr=0.1, n_iter=500, l2=0.05)
        Xte_n, _, _ = standardize(Xte, mean, std)
        yhat, probs = predict(Xte_n, w, b, threshold=0.5)
        acc = sum(1 for a, b_ in zip(yhat, yte) if a == b_) / len(yte)
        # Compute simple AUC (rank correlation)
        pos = sorted([(p, l) for p, l in zip(probs, yte)], reverse=True)
        tp = sum(l for _, l in pos[:sum(yte)])
        fp = sum(1 - l for _, l in pos[:sum(yte)])
        auc_approx = tp / max(1, sum(yte)) - fp / max(1, len(yte) - sum(yte))
        accs.append(acc)
        aucs.append(auc_approx)

    print(f"\n5-fold CV: accuracy={sum(accs)/k_int:.3f} ± {max(accs)-min(accs):.3f}, AUC-approx={sum(aucs)/k_int:.3f}")

    # Final model on all data
    Xn, mean, std = standardize(X)
    w, b = fit_logreg(Xn, y, lr=0.1, n_iter=800, l2=0.05)
    print("\n=== Top features by |coefficient| (after standardization) ===")
    ranked = sorted(w.items(), key=lambda kv: -abs(kv[1]))
    for k, v in ranked[:10]:
        direction = "↑ more → more UNK" if v > 0 else "↓ more → less UNK"
        print(f"  {k:50s} coef={v:+.4f}  ({direction})")

    # Predict prob on R19 BEST rows (the session's final config) for diagnostics
    print("\n=== Predicted UNK probability on R19 BEST reusers (agent != 1) ===")
    csv_p = Path("/home/gfy/CodeMAS_Project/sglang-kvflow/results/lossy_alg_round21/r19_verdict/rows.csv")
    if csv_p.exists():
        r19 = list(csv.DictReader(open(csv_p)))
        r19_reuser = [r for r in r19 if int(r.get('agent_idx', 0) or 0) != 1]
        for r in r19_reuser:
            feats = {k: float(r.get(k, 0) or 0) for k in FEATURE_COLS}
            feats_n = {k: (feats[k] - mean[k]) / std[k] for k in FEATURE_COLS}
            p = sigmoid(b + sum(w[k] * feats_n[k] for k in FEATURE_COLS))
            cid = r['case_id'][-20:]
            aid = r['agent_idx']
            print(f"  case={cid} agent={aid}: P(UNK)={p:.3f} {'⚠ HIGH-RISK' if p > 0.4 else ''}")

    # Save artifacts
    out = Path("/home/gfy/CodeMAS_Project/sglang-kvflow/results/lossy_alg_round25/oracle_model.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "feature_cols": FEATURE_COLS,
        "weights": w,
        "bias": b,
        "mean": mean,
        "std": std,
        "cv_accuracy": sum(accs)/k_int,
        "cv_auc_approx": sum(aucs)/k_int,
        "n_train": n,
    }, indent=2))
    print(f"\nSaved oracle model to {out}")


if __name__ == "__main__":
    main()
