#!/usr/bin/env python3
"""HKVD-by-node-kind measurement (2026-07-10, P0 decisive experiment).

Extends ``measure_hkvd_by_position.py`` (2026-07-09) from "deviation by slot
position" to "deviation by AST node kind WITHIN a chunk".

Direction A (contiguous node-kind interface-recompute) was FALSIFIED at equal
budget (``ABLATION_NODEKIND_REPORT.md``: node-kind -3.3pp vs R32). But that
falsification only shows the CONTIGUOUS-HEAD mechanism cannot exploit structure
signal - it does NOT tell us whether the structure signal EXISTS at the KV
level. This script answers that question directly:

    For each pool chunk C with an AST interface boundary, split C's tokens into:
      interface_tokens = tokens in [chunk_start, interface_end_byte)
                         (signature + leading docstring - the part Direction A
                          recomputes)
      body_tokens      = tokens in [interface_end_byte, chunk_end)
                         (function/class body - the part Direction A copies)

    Measure, per token group g in {interface, body}:

        deviation_g = 1 - cosine( KV(g | canonical_prefix),
                                  KV(g | live_prefix)      )

    averaged over layers (+ over chunks). canonical_prefix = the literal
    ROLE/CASE/UPSTREAM placeholder preamble the pool precomputed with;
    live_prefix = the same preamble with real role/case/upstream filled in.

DECISIVE VERDICT:
  - interface_dev > body_dev  -> structure signal IS real at the KV level.
    Contiguous head was the wrong primitive (Direction A/B falsified), but
    P3 (True CacheBlend per-token mask) has a strong motive: we KNOW which
    tokens drift, we just need a mechanism that can target them per-token.
  - interface_dev <= body_dev -> structure signal is NOT real. The entire
    "code structure decides what to recompute" line is dead, including P3.
    Move to a different research axis entirely.

No sglang dependency - HuggingFace model direct forward (same pattern as
``measure_hkvd_by_position.py``). Reuses ``ASTChunker`` to get the interface
boundary in the chunk's own byte coordinate system.
"""
from __future__ import annotations
import json
import math
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path("/home/gfy/CodeMAS_Project/sglang-kvflow")
MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
MANIFEST = ROOT / "results/codebase_kv/pandas_15case_v1/manifest.jsonl"
REPO = ROOT / "results/giant_codebase/pandas_src"
OUT = ROOT / "results/hkvd_by_node_kind_20260710"
OUT.mkdir(parents=True, exist_ok=True)

# Same canonical preamble the pool precomputed with (preamble.txt, 69 tokens).
# ROLE/CASE/UPSTREAM are literal placeholders in the canonical version.
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

# ASTChunker lives in the sglang mem_cache package.
sys.path.insert(0, str(ROOT / "python/sglang/srt/mem_cache"))
from ast_chunker import ASTChunker  # noqa: E402

CHUNKER = ASTChunker()


def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="sdpa",
    ).cuda().eval()
    return tok, model


def get_interface_boundary(chunk_text: str, name: str, anchor_type: str) -> int:
    """Return interface_end_byte in chunk_text coordinates, or 0 if none.

    Mirrors ``compute_nodekind_budget.py``: re-chunk the standalone chunk_text
    and find the ChunkSpan whose name+anchor_type match the manifest record.
    """
    if not chunk_text.strip():
        return 0
    try:
        spans = CHUNKER.chunk_text(chunk_text)
    except Exception:
        return 0
    match = next((s for s in spans
                  if s.name == name and s.anchor_type == anchor_type), None)
    if match is None or match.interface_end_byte <= 0:
        return 0
    return match.interface_end_byte


def build_prompt(prefix: str, chunk_text: str, name: str) -> tuple[str, int, int]:
    """Build prefix + single code_base block.

    Returns (text, chunk_char_start, chunk_char_end) so callers can map the
    chunk's byte offsets (in chunk_text coordinates) into prompt coordinates.
    """
    header = f"\n## code_base: {name}\n```python\n"
    footer = "\n```\n"
    chunk_start = len(prefix) + len(header)
    chunk_end = chunk_start + len(chunk_text)
    text = prefix + header + chunk_text + footer
    return text, chunk_start, chunk_end


@torch.no_grad()
def forward_kv(tok, model, text: str):
    enc = tok(text, return_tensors="pt", return_offsets_mapping=True,
              add_special_tokens=False)
    input_ids = enc["input_ids"].cuda()
    offsets = enc["offset_mapping"][0].tolist()  # list of (start, end) char
    out = model(input_ids, use_cache=True, output_hidden_states=False)
    pkv = out.past_key_values
    layers = {}
    n_layers = len(pkv.layers)
    for L in range(n_layers):
        K = pkv.layers[L].keys      # [1, h, seq, d]
        V = pkv.layers[L].values
        layers[L] = (K[0].float().cpu(), V[0].float().cpu())  # [h, seq, d]
    return offsets, layers


def char_span_to_toks(offsets, cstart, cend):
    """Token indices whose char span overlaps [cstart, cend)."""
    return [i for i, (s, e) in enumerate(offsets) if s < cend and e > cstart]


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
    """Average (1 - cos) over layers. Returns (k_dev, v_dev)."""
    if not perL:
        return (float("nan"), float("nan"))
    k = float(np.mean([1 - d["K_cos"] for d in perL]))
    v = float(np.mean([1 - d["V_cos"] for d in perL]))
    return (k, v)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--max-chunks", type=int, default=40,
                   help="sample N chunks with interface boundary (default 40)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--live-role", default="implementer")
    p.add_argument("--live-upstream", default="(none)")
    args = p.parse_args()

    print(f"[hkvd_nk] loading {MODEL} ...", flush=True)
    tok, model = load_model()

    recs = [json.loads(l) for l in MANIFEST.read_text().splitlines() if l.strip()]
    print(f"[hkvd_nk] {len(recs)} chunks in manifest", flush=True)

    # Read each source file once, attach interface_end_byte to each chunk.
    by_file = defaultdict(list)
    for r in recs:
        by_file[r["slot_id"]].append(r)
    file_text = {}
    candidates = []  # (rec, chunk_text, iface_byte)
    for slot_id, recs_in_file in by_file.items():
        relpath = slot_id.split(":", 1)[1]
        fpath = REPO / relpath
        if not fpath.exists():
            continue
        ftext = fpath.read_text()
        for r in recs_in_file:
            ct = ftext[r["byte_start"]:r["byte_end"]]
            if not ct.strip():
                continue
            ib = get_interface_boundary(ct, r["name"], r["anchor_type"])
            if ib <= 0:
                continue
            candidates.append((r, ct, ib))
    print(f"[hkvd_nk] {len(candidates)} chunks with interface boundary "
          f"(fire rate {len(candidates)/len(recs)*100:.1f}%)", flush=True)

    # Sample (deterministic) for a balanced, bounded run.
    import random
    rnd = random.Random(args.seed)
    rnd.shuffle(candidates)
    # Prefer a class/function mix: take all classes first, then fill functions.
    classes = [c for c in candidates if c[0]["anchor_type"] == "class"]
    funcs = [c for c in candidates if c[0]["anchor_type"] == "function"]
    n_classes = min(len(classes), max(6, args.max_chunks // 5))
    n_funcs = args.max_chunks - n_classes
    sample = classes[:n_classes] + funcs[:n_funcs]
    print(f"[hkvd_nk] sampling {len(sample)} chunks "
          f"({n_classes} class + {n_funcs} function)", flush=True)

    # Per-group aggregation.
    dev = {"interface": [], "body": []}  # list of (k_dev, v_dev)
    per_anchor = defaultdict(lambda: {"interface": [], "body": []})
    per_layer_grp = {"interface": defaultdict(lambda: {"K": [], "V": []}),
                     "body": defaultdict(lambda: {"K": [], "V": []})}
    per_chunk_records = []

    # live prefix: fill ROLE/CASE/UPSTREAM
    live_prefix = (DIRECTION_A_V3_PREAMBLE
                   .replace("ROLE", args.live_role)
                   .replace("CASE", "pandas-dev__pandas.95280573.combine_file__11s6papj")
                   .replace("UPSTREAM", args.live_upstream))
    canon_prefix = DIRECTION_A_V3_PREAMBLE

    for ci, (r, ct, ib) in enumerate(sample, 1):
        name = r["name"]
        at = r["anchor_type"]
        textA, csA, ceA = build_prompt(canon_prefix, ct, name)
        textB, csB, ceB = build_prompt(live_prefix, ct, name)
        offA, kvA = forward_kv(tok, model, textA)
        offB, kvB = forward_kv(tok, model, textB)
        # chunk char span in prompt coords (A and B have identical chunk placement)
        # interface char range = [chunk_start, chunk_start + ib)
        # body char range      = [chunk_start + ib, chunk_end)
        iface_toksA = char_span_to_toks(offA, csA, csA + ib)
        iface_toksB = char_span_to_toks(offB, csB, csB + ib)
        body_toksA = char_span_to_toks(offA, csA + ib, ceA)
        body_toksB = char_span_to_toks(offB, csB + ib, ceB)

        rec = {"slot_id": r["slot_id"], "name": name, "anchor_type": at,
               "n_tokens": r["n_tokens"], "iface_byte": ib,
               "n_iface_toks": len(iface_toksA), "n_body_toks": len(body_toksA)}

        if iface_toksA and iface_toksB:
            perL = cos_per_layer(kvA, kvB, iface_toksA, iface_toksB)
            k, v = deviation(perL)
            dev["interface"].append((k, v))
            per_anchor[at]["interface"].append((k, v))
            for d in perL:
                per_layer_grp["interface"][d["layer"]]["K"].append(1 - d["K_cos"])
                per_layer_grp["interface"][d["layer"]]["V"].append(1 - d["V_cos"])
            rec["iface_k_dev"] = k
            rec["iface_v_dev"] = v
        if body_toksA and body_toksB:
            perL = cos_per_layer(kvA, kvB, body_toksA, body_toksB)
            k, v = deviation(perL)
            dev["body"].append((k, v))
            per_anchor[at]["body"].append((k, v))
            for d in perL:
                per_layer_grp["body"][d["layer"]]["K"].append(1 - d["K_cos"])
                per_layer_grp["body"][d["layer"]]["V"].append(1 - d["V_cos"])
            rec["body_k_dev"] = k
            rec["body_v_dev"] = v

        per_chunk_records.append(rec)
        if ci % 5 == 0 or ci == len(sample):
            ik = np.mean([x[0] for x in dev["interface"]]) if dev["interface"] else 0
            bk = np.mean([x[0] for x in dev["body"]]) if dev["body"] else 0
            print(f"[hkvd_nk] {ci}/{len(sample)}  iface_K_dev={ik:.4f}  "
                  f"body_K_dev={bk:.4f}  (running mean)", flush=True)
        del kvA, kvB
        torch.cuda.empty_cache()

    # ---- Aggregate ----------------------------------------------------------
    def agg(lst):
        if not lst:
            return {"n": 0, "K_dev": float("nan"), "V_dev": float("nan"),
                    "K_std": float("nan"), "V_std": float("nan")}
        ks = [x[0] for x in lst]
        vs = [x[1] for x in lst]
        return {"n": len(lst), "K_dev": float(np.mean(ks)),
                "V_dev": float(np.mean(vs)),
                "K_std": float(np.std(ks)), "V_std": float(np.std(vs))}

    summary = {g: agg(dev[g]) for g in ("interface", "body")}
    summary["per_anchor_type"] = {
        at: {g: agg(per_anchor[at][g]) for g in ("interface", "body")}
        for at in sorted(per_anchor)
    }

    # Paired per-chunk delta (only chunks with BOTH groups measured)
    paired = []
    for rec in per_chunk_records:
        if "iface_k_dev" in rec and "body_k_dev" in rec:
            paired.append((rec["iface_k_dev"], rec["body_k_dev"],
                           rec["iface_k_dev"] - rec["body_k_dev"]))
    if paired:
        deltas = [p[2] for p in paired]
        summary["paired"] = {
            "n": len(paired),
            "mean_delta_iface_minus_body_K": float(np.mean(deltas)),
            "std_delta": float(np.std(deltas)),
            "n_iface_gt_body": int(sum(1 for d in deltas if d > 0)),
            "n_body_gt_iface": int(sum(1 for d in deltas if d < 0)),
            "n_tie": int(sum(1 for d in deltas if d == 0)),
        }
        # Wilcoxon signed-rank (paired)
        try:
            from scipy.stats import wilcoxon
            if len(set(deltas)) > 1:
                stat, p = wilcoxon(deltas, alternative="greater")
                summary["paired"]["wilcoxon_p_one_sided_iface_gt_body"] = float(p)
        except Exception as e:
            summary["paired"]["wilcoxon_error"] = str(e)

    # per-layer
    per_layer_out = {}
    for g in ("interface", "body"):
        per_layer_out[g] = {
            str(L): {"K_mean": float(np.mean(v["K"])), "V_mean": float(np.mean(v["V"]))}
            for L, v in sorted(per_layer_grp[g].items())
        }

    # ---- Print --------------------------------------------------------------
    print("\n=== HKVD by node-kind (1 - cosine, averaged over layers + chunks) ===")
    print(f"{'group':>12} {'n':>5} {'K_dev':>10} {'V_dev':>10} {'K_std':>8}")
    for g in ("interface", "body"):
        s = summary[g]
        print(f"{g:>12} {s['n']:>5} {s['K_dev']:>10.4f} {s['V_dev']:>10.4f} "
              f"{s['K_std']:>8.4f}")

    print("\nper anchor_type:")
    for at, d in summary["per_anchor_type"].items():
        for g in ("interface", "body"):
            s = d[g]
            print(f"  {at:>10} {g:>10}: n={s['n']:>3}  K_dev={s['K_dev']:.4f}  "
                  f"V_dev={s['V_dev']:.4f}")

    if "paired" in summary:
        pr = summary["paired"]
        print(f"\n=== paired (per-chunk iface_dev - body_dev) ===")
        print(f"  n={pr['n']}  mean_delta={pr['mean_delta_iface_minus_body_K']:+.4f}  "
              f"std={pr['std_delta']:.4f}")
        print(f"  iface>body: {pr['n_iface_gt_body']}  body>iface: "
              f"{pr['n_body_gt_iface']}  tie: {pr['n_tie']}")
        if "wilcoxon_p_one_sided_iface_gt_body" in pr:
            print(f"  Wilcoxon one-sided (iface>body) p = "
                  f"{pr['wilcoxon_p_one_sided_iface_gt_body']:.4f}")

    # ---- Verdict ------------------------------------------------------------
    ik = summary["interface"]["K_dev"]
    bk = summary["body"]["K_dev"]
    print(f"\n>>> interface K_dev={ik:.4f}  vs  body K_dev={bk:.4f}")
    if ik > bk:
        rel = (ik / bk - 1) * 100 if bk > 0 else float("inf")
        print(f">>> STRUCTURE SIGNAL REAL: interface tokens drift {rel:.1f}% MORE "
              f"than body tokens under prefix swap.")
        print(">>> -> Contiguous head was the wrong primitive (Direction A/B "
              "falsified), but P3 (True CacheBlend per-token mask) has STRONG "
              "motive: we know which tokens drift.")
    else:
        rel = (bk / ik - 1) * 100 if ik > 0 else float("inf")
        print(f">>> STRUCTURE SIGNAL NOT REAL: body tokens drift {rel:.1f}% MORE "
              f"(or equal) than interface tokens.")
        print(">>> -> Entire 'code structure decides recompute' line is dead, "
              "INCLUDING P3. Move to a different research axis.")

    out_json = {
        "model": MODEL,
        "n_sampled": len(sample),
        "n_candidates_with_iface": len(candidates),
        "fire_rate": len(candidates) / len(recs),
        "summary": summary,
        "per_layer": per_layer_out,
        "per_chunk": per_chunk_records,
        "interpretation": (
            "deviation_g = 1 - cos(KV(g|canonical_prefix), KV(g|live_prefix)). "
            "interface = [chunk_start, interface_end_byte); body = rest. "
            "Verdict: interface_dev > body_dev => structure signal real => "
            "P3 motive strong; else entire code-structure-recompute line dead."),
    }
    (OUT / "hkvd_by_node_kind.json").write_text(json.dumps(out_json, indent=2))
    (OUT / "hkvd_by_node_kind_per_chunk.jsonl").write_text(
        "\n".join(json.dumps(r) for r in per_chunk_records) + "\n"
    )
    print(f"\n[hkvd_nk] wrote {OUT/'hkvd_by_node_kind.json'}")
    print(f"[hkvd_nk] wrote {OUT/'hkvd_by_node_kind_per_chunk.jsonl'}")


if __name__ == "__main__":
    main()