#!/usr/bin/env python3
"""Multi-signal HKVD measurement (2026-07-11).

Extends the interface/body HKVD (2026-07-10) to **8 distinct code-structure
signals** using precomputed per-token labels from
``compute_per_token_signal_labels.py``.

For each signal axis (e.g., first_use vs reuse), for each sampled chunk:
  1. Build canonical + live prompts (same preamble as the pool precompute)
  2. Forward both through Qwen2.5-Coder-7B-Instruct (HF direct)
  3. Project the chunk's signal-token indices into prompt-token coordinates
  4. Compute per-layer K/V cosine between canonical and live KV
  5. Aggregate deviation per signal group
  6. Paired Wilcoxon signed-rank + bootstrap CI per signal axis

**Critical fix**: forward_kv applies the Encoding.unwrap defensive logic
(same bug pattern as radix_cache._build_byte_to_token_map) to avoid
silent wrong-token selection on transformers 5.x + Qwen2Tokenizer.

Reads:
  - <labels>          results/hkvd_multi_signal_20260711/signal_labels_per_chunk.jsonl
  - <manifest>        results/codebase_kv/pandas_15case_v1/manifest.jsonl
  - <repo>            results/giant_codebase/pandas_src/

Writes:
  - <out-dir>/hkvd_by_signal_per_chunk.jsonl
  - <out-dir>/hkvd_by_signal_summary.json
"""
from __future__ import annotations
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path("/home/gfy/CodeMAS_Project/sglang-kvflow")
MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"

# Mirror measure_hkvd_by_node_kind.py preamble (the one the pool precomputed
# with, sha1 e6fc21efe...).
DIRECTION_A_V3_PREAMBLE = (
    "# Repository: pandas-dev/pandas\n"
    "# Working set context for coding-agent serving. "
    "The following file is part of this repository.\n"
    "\n"
    "You are a senior software engineering agent.\n"
    "\n"
    "## Agent role\n"
    "ROLE\n"
    "\n"
    "## Case\n"
    "CASE\n"
    "\n"
    "## Instruction\n"
    "Inspect the repeated repository code and answer with one concise implementation risk.\n"
    "\n"
    "## Upstream context\n"
    "UPSTREAM\n"
)

# Signal axes with paired (a, b) groups. Order matters: a is "novel/structural"
# hypothesis side, b is "control/plain" side.
BINARY_AXES = [
    ("first_use", "reuse"),
    ("def", "ref"),
    ("control_flow", "data_flow"),
    ("import_dist_1", "import_dist_0"),
    ("rare_id", "common_id"),
]
MULTI_AXES = [
    ("cyc_high", "cyc_low"),
    ("cyc_med", "cyc_low"),
    ("cyc_high", "cyc_med"),
]
UNARY_AXES = [
    "type_complexity_toks",   # no paired control; vs uniform-zero baseline
    "linter_risky_toks",      # ditto
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--labels", required=True, type=Path,
                   help="signal_labels_per_chunk.jsonl from labeler")
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--repo-root", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--max-chunks", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-tokens", type=int, default=5,
                   help="skip signal pair when either side has <N tokens (default 5)")
    p.add_argument("--live-role", default="implementer")
    p.add_argument("--live-upstream", default="(none)")
    p.add_argument("--case-id", default="pandas-dev__pandas.95280573.combine_file__11s6papj")
    return p.parse_args()


def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="sdpa",
    ).cuda().eval()
    return tok, model


@torch.no_grad()
def forward_kv(tok, model, text: str):
    """Forward + per-layer K/V extraction. Defensive Encoding unwrap added.

    Same bug pattern as ``radix_cache._build_byte_to_token_map``: on
    transformers 5.x + Qwen2Tokenizer, ``enc["offset_mapping"][0].tolist()``
    returns ids (not char spans). Unwrap ``Encoding.offsets`` first.
    """
    enc = tok(text, return_tensors="pt", return_offsets_mapping=True,
              add_special_tokens=False)
    input_ids = enc["input_ids"].cuda()
    # Defensive unwrap (radix_cache.py:2115-2146 pattern)
    om = enc["offset_mapping"]
    first = om[0] if om is not None and len(om) > 0 else None
    if first is not None and hasattr(first, "offsets"):
        offsets = list(first.offsets)
    elif first is not None:
        try:
            offsets = list(first)
        except TypeError:
            offsets = []
    else:
        offsets = []
    # Validate offsets are (int, int) tuples
    if offsets and not isinstance(offsets[0], tuple):
        offsets = [(int(o[0]), int(o[1])) for o in offsets
                   if o is not None and len(o) >= 2]
    out = model(input_ids, use_cache=True, output_hidden_states=False)
    pkv = out.past_key_values
    layers = {}
    n_layers = len(pkv.layers)
    for L in range(n_layers):
        K = pkv.layers[L].keys      # [1, h, seq, d]
        V = pkv.layers[L].values
        layers[L] = (K[0].float().cpu(), V[0].float().cpu())  # [h, seq, d]
    return offsets, layers


def cos_per_layer(layersA, layersB, tokA, tokB):
    """Mean cosine over tokens+heads for K and V, per layer."""
    out = []
    Ls = sorted(layersA.keys())
    for L in Ls:
        KA, VA = layersA[L]
        KB, VB = layersB[L]
        ka = KA[:, tokA, :].reshape(-1, KA.shape[-1])
        kb = KB[:, tokB, :].reshape(-1, KB.shape[-1])
        va = VA[:, tokA, :].reshape(-1, VA.shape[-1])
        vb = VB[:, tokB, :].reshape(-1, VB.shape[-1])
        n = min(ka.shape[0], kb.shape[0])
        ka, kb, va, vb = ka[:n], kb[:n], va[:n], vb[:n]

        def cos(a, b):
            a = F.normalize(a, dim=-1)
            b = F.normalize(b, dim=-1)
            return (a * b).sum(-1).mean().item()
        out.append({"layer": L, "K_cos": cos(ka, kb), "V_cos": cos(va, vb)})
    return out


def deviation(perL):
    if not perL:
        return (float("nan"), float("nan"))
    k = float(np.mean([1 - d["K_cos"] for d in perL]))
    v = float(np.mean([1 - d["V_cos"] for d in perL]))
    return (k, v)


def build_prompt(prefix: str, chunk_text: str, name: str) -> tuple[str, int, int]:
    header = f"\n## code_base: {name}\n```python\n"
    footer = "\n```\n"
    chunk_start = len(prefix) + len(header)
    chunk_end = chunk_start + len(chunk_text)
    text = prefix + header + chunk_text + footer
    return text, chunk_start, chunk_end


def map_chunk_tok_to_prompt(chunk_tok_idxs: list[int], chunk_start_char: int,
                             offsets: list[tuple[int, int]]) -> list[int]:
    """Translate chunk-text-relative token indices to prompt-token indices.

    The labeler returned indices within chunk_text; here we find the prompt
    token whose char span overlaps [chunk_start_char + tok_char_start,
    chunk_start_char + tok_char_end). For each chunk token idx, look up its
    char span in ``offsets[chunk_start..chunk_end]`` and find the matching
    prompt index.
    """
    out = []
    # Build a map: chunk_text char span → prompt token index
    # Since the chunk sits at the end of the prompt (chunk_start..chunk_end
    # is monotonic in offsets), we can scan prompt offsets linearly.
    # Simpler: for each chunk_tok_idx, find the prompt token at the same
    # offset position. Use offsets[chunk_start_char] as anchor.
    for cti in chunk_tok_idxs:
        if cti < 0 or cti >= len(offsets):
            continue
        # Find prompt token whose offset matches offsets[cti]
        # (since chunk is contiguous in prompt, this is offsets[chunk_start_in_prompt + cti])
        target = offsets[cti]
        # Linear search from chunk_start region
        for pi in range(cti, len(offsets)):
            if offsets[pi] == target:
                out.append(pi)
                break
        else:
            # fallback: nearest
            for pi in range(len(offsets) - 1, -1, -1):
                if offsets[pi][0] >= target[0]:
                    out.append(pi)
                    break
    return sorted(set(out))


def bootstrap_ci(deltas: list[float], n_iter: int = 10000, seed: int = 42) -> tuple[float, float]:
    if not deltas:
        return (float("nan"), float("nan"))
    rnd = random.Random(seed)
    samples = sorted(sum(rnd.choice(deltas) for _ in deltas) / len(deltas)
                     for _ in range(n_iter))
    return (samples[int(0.025 * n_iter)], samples[int(0.975 * n_iter)])


def agg(lst):
    if not lst:
        return {"n": 0, "K_dev": float("nan"), "V_dev": float("nan"),
                "K_std": float("nan"), "V_std": float("nan")}
    ks = [x[0] for x in lst]
    vs = [x[1] for x in lst]
    return {"n": len(lst), "K_dev": float(np.mean(ks)),
            "V_dev": float(np.mean(vs)),
            "K_std": float(np.std(ks)), "V_std": float(np.std(vs))}


def main():
    args = parse_args()
    print(f"[hkvd_sig] loading {MODEL} ...", flush=True)
    tok, model = load_model()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load labels
    labels_by_slot: dict[str, dict] = {}
    for line in args.labels.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        labels_by_slot[rec["slot_id"]] = rec

    # Load manifest + reconstruct chunk_text
    recs = [json.loads(l) for l in args.manifest.read_text().splitlines() if l.strip()]
    print(f"[hkvd_sig] {len(recs)} chunks in manifest, "
          f"{len(labels_by_slot)} with signal labels", flush=True)

    by_file = defaultdict(list)
    for r in recs:
        by_file[r["slot_id"]].append(r)

    # Build candidates: chunks that have non-empty first_use OR reuse labels
    # (so we have at least one paired binary axis populated)
    file_text_cache: dict[str, str] = {}
    candidates: list[tuple[dict, str]] = []  # (rec, chunk_text)
    for slot_id, recs_in_file in by_file.items():
        relpath = slot_id.split(":", 1)[1]
        fpath = args.repo_root / relpath
        if not fpath.exists():
            continue
        if slot_id not in file_text_cache:
            file_text_cache[slot_id] = fpath.read_text()
        ftext = file_text_cache[slot_id]
        for r in recs_in_file:
            lab = labels_by_slot.get(slot_id)
            if not lab or not lab.get("signals"):
                continue
            sigs = lab["signals"]
            # Need at least one binary axis populated
            has_pair = any(sigs.get(f"{a}_toks") and sigs.get(f"{b}_toks")
                           for a, b in BINARY_AXES)
            if not has_pair:
                continue
            ct = ftext[r["byte_start"]:r["byte_end"]]
            if not ct.strip():
                continue
            candidates.append((r, ct))

    print(f"[hkvd_sig] {len(candidates)} candidates with paired labels", flush=True)

    # Sample deterministically (same logic as node-kind: classes first, then funcs)
    rnd = random.Random(args.seed)
    rnd.shuffle(candidates)
    classes = [c for c in candidates if c[0]["anchor_type"] == "class"]
    funcs = [c for c in candidates if c[0]["anchor_type"] == "function"]
    n_classes = min(len(classes), max(6, args.max_chunks // 5))
    n_funcs = args.max_chunks - n_classes
    sample = classes[:n_classes] + funcs[:n_funcs]
    print(f"[hkvd_sig] sampling {len(sample)} chunks "
          f"({n_classes} class + {n_funcs} function)", flush=True)

    live_prefix = (DIRECTION_A_V3_PREAMBLE
                   .replace("ROLE", args.live_role)
                   .replace("CASE", args.case_id)
                   .replace("UPSTREAM", args.live_upstream))
    canon_prefix = DIRECTION_A_V3_PREAMBLE

    # Per-axis paired deltas (K_dev side primarily)
    axis_pairs = BINARY_AXES + MULTI_AXES
    deltas_per_axis: dict[tuple[str, str], list[float]] = defaultdict(list)
    per_chunk_records: list[dict] = []

    for ci, (r, ct) in enumerate(sample, 1):
        name = r["name"]
        lab = labels_by_slot[r["slot_id"]]
        sigs = lab.get("signals", {})

        textA, csA, ceA = build_prompt(canon_prefix, ct, name)
        textB, csB, ceB = build_prompt(live_prefix, ct, name)
        offA, kvA = forward_kv(tok, model, textA)
        offB, kvB = forward_kv(tok, model, textB)

        # Validate offsets came back as proper (start, end) tuples
        if not offA or not isinstance(offA[0], tuple) or len(offA[0]) != 2:
            print(f"[hkvd_sig] WARNING: chunk {r['slot_id']}/{name} produced "
                  f"invalid offsets (type {type(offA[0]) if offA else 'empty'}); skipping",
                  flush=True)
            del kvA, kvB
            torch.cuda.empty_cache()
            continue

        chunk_rec: dict = {
            "slot_id": r["slot_id"], "name": name,
            "anchor_type": r["anchor_type"], "n_tokens": r["n_tokens"],
        }

        # Compute per-axis deviations
        for a, b in axis_pairs:
            toks_a_in_chunk = sigs.get(f"{a}_toks", [])
            toks_b_in_chunk = sigs.get(f"{b}_toks", [])
            if len(toks_a_in_chunk) < args.min_tokens or len(toks_b_in_chunk) < args.min_tokens:
                continue
            toks_a_A = map_chunk_tok_to_prompt(toks_a_in_chunk, csA, offA)
            toks_a_B = map_chunk_tok_to_prompt(toks_a_in_chunk, csB, offB)
            toks_b_A = map_chunk_tok_to_prompt(toks_b_in_chunk, csA, offA)
            toks_b_B = map_chunk_tok_to_prompt(toks_b_in_chunk, csB, offB)
            if not toks_a_A or not toks_a_B or not toks_b_A or not toks_b_B:
                continue

            perL_a = cos_per_layer(kvA, kvB, toks_a_A, toks_a_B)
            perL_b = cos_per_layer(kvA, kvB, toks_b_A, toks_b_B)
            ka, va = deviation(perL_a)
            kb, vb = deviation(perL_b)
            chunk_rec[f"{a}_k_dev"] = ka
            chunk_rec[f"{a}_v_dev"] = va
            chunk_rec[f"{b}_k_dev"] = kb
            chunk_rec[f"{b}_v_dev"] = vb
            chunk_rec[f"n_{a}_toks"] = len(toks_a_in_chunk)
            chunk_rec[f"n_{b}_toks"] = len(toks_b_in_chunk)
            if not (np.isnan(ka) or np.isnan(kb)):
                deltas_per_axis[(a, b)].append(ka - kb)

        per_chunk_records.append(chunk_rec)

        if ci % 5 == 0 or ci == len(sample):
            n_axes = sum(1 for ax in axis_pairs if deltas_per_axis[ax])
            print(f"[hkvd_sig] {ci}/{len(sample)}  "
                  f"axes_with_data={n_axes}/{len(axis_pairs)}", flush=True)
        del kvA, kvB
        torch.cuda.empty_cache()

    # ---- Aggregate + statistical tests ------------------------------------
    summary_axes: dict[str, dict] = {}
    for a, b in axis_pairs:
        deltas = deltas_per_axis.get((a, b), [])
        if not deltas:
            summary_axes[f"{a}_vs_{b}"] = {"n": 0}
            continue
        mean_a = float(np.mean([r[f"{a}_k_dev"] for r in per_chunk_records
                               if f"{a}_k_dev" in r]))
        mean_b = float(np.mean([r[f"{b}_k_dev"] for r in per_chunk_records
                               if f"{b}_k_dev" in r]))
        mean_delta = float(np.mean(deltas))
        ci_lo, ci_hi = bootstrap_ci(deltas)
        entry = {
            "n": len(deltas),
            "mean_a_K_dev": mean_a,
            "mean_b_K_dev": mean_b,
            "mean_delta_a_minus_b_K": mean_delta,
            "delta_std": float(np.std(deltas)),
            "n_a_gt_b": int(sum(1 for d in deltas if d > 0)),
            "n_b_gt_a": int(sum(1 for d in deltas if d < 0)),
            "n_tie": int(sum(1 for d in deltas if d == 0)),
            "ci95_lo": ci_lo,
            "ci95_hi": ci_hi,
            "rel_effect_pct": ((mean_a / mean_b - 1) * 100 if mean_b > 0 else float("inf")),
        }
        # Wilcoxon one-sided: a > b?
        try:
            from scipy.stats import wilcoxon
            if len(set(deltas)) > 1:
                stat, p = wilcoxon(deltas, alternative="greater")
                entry["wilcoxon_p_a_gt_b"] = float(p)
        except Exception as e:
            entry["wilcoxon_error"] = str(e)

        # Verdict
        p = entry.get("wilcoxon_p_a_gt_b", 1.0)
        ci_excludes_zero = (ci_lo > 0 or ci_hi < 0)
        rel = entry["rel_effect_pct"]
        if p < 0.05 and ci_excludes_zero and abs(rel) >= 5:
            entry["verdict"] = "POSITIVE" if rel > 0 else "NEGATIVE_CONTROL_BETTER"
        elif p < 0.10:
            entry["verdict"] = "MARGINAL"
        else:
            entry["verdict"] = "NULL"

        summary_axes[f"{a}_vs_{b}"] = entry

    # Print summary
    print("\n=== HKVD by signal axis (paired Wilcoxon, bootstrap 95% CI) ===")
    print(f"{'axis':<36} {'n':>4} {'mean_a':>9} {'mean_b':>9} "
          f"{'delta':>8} {'CI_lo':>8} {'CI_hi':>8} {'rel%':>7} {'p>a>b':>8} {'verdict':<14}")
    for k, v in summary_axes.items():
        if v.get("n", 0) == 0:
            print(f"{k:<36} {'(empty)':>8}")
            continue
        print(f"{k:<36} {v['n']:>4} "
              f"{v['mean_a_K_dev']:>9.4f} {v['mean_b_K_dev']:>9.4f} "
              f"{v['mean_delta_a_minus_b_K']:>+8.4f} "
              f"{v['ci95_lo']:>+8.4f} {v['ci95_hi']:>+8.4f} "
              f"{v['rel_effect_pct']:>+7.1f} "
              f"{v.get('wilcoxon_p_a_gt_b', 1.0):>8.4f} "
              f"{v.get('verdict', '?'):<14}")

    summary = {
        "model": MODEL,
        "n_sampled": len(sample),
        "n_candidates": len(candidates),
        "axes": summary_axes,
        "interpretation": (
            "Each axis is a paired per-chunk Wilcoxon (a vs b). "
            "Verdict: POSITIVE if p<0.05, CI excludes 0, |rel|>=5% AND mean_a>mean_b. "
            "NEGATIVE_CONTROL_BETTER if same thresholds but mean_b>mean_a. "
            "NULL otherwise. "
            "POSITIVE = novel code-structure signal real at KV layer."
        ),
    }

    out_json = args.out_dir / "hkvd_by_signal_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    out_per_chunk = args.out_dir / "hkvd_by_signal_per_chunk.jsonl"
    out_per_chunk.write_text("\n".join(json.dumps(r) for r in per_chunk_records) + "\n")
    print(f"\n[hkvd_sig] wrote {out_json}")
    print(f"[hkvd_sig] wrote {out_per_chunk}")


if __name__ == "__main__":
    main()