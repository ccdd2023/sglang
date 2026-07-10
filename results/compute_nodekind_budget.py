#!/usr/bin/env python3
"""Offline recompute-budget estimator for node-kind interface recompute (P1.3).

Direction A (contiguous interface-recompute): per byte-exact chunk hit, the
head K is set to the chunk's AST interface token count (signature, or
signature + docstring) instead of ``frac * chunk_len``. To run a fair
equal-budget ablation against R32 (uniform ``frac * chunk_len``), we need the
uniform frac* that matches node-kind's total recompute budget B:

    B_nodekind      = sum over hits of interface_tokens(chunk)
    B_R32(frac)     = frac * sum over hits of chunk_len
    frac*           = B_nodekind / sum(chunk_len)   (over the same hit set)

The runtime hit set is identical for R32 and node-kind (same pool, same
byte-exact matching; only K differs), so frac* = mean interface-token ratio.
This script estimates that ratio over ALL chunks in the precompute pool
(max-reuse case). The runtime ratio over actual hits will be close; the
ablation also does a Pareto sweep that brackets frac*, and B is measured
directly via ``placeholder_chunk_pool_total_tokens_dense``.

No sglang dependency - reads the pool manifest + source files, uses the
ASTChunker (same one the server uses) and a HuggingFace tokenizer.

Usage:
    python3 results/compute_nodekind_budget.py \\
        --manifest results/codebase_kv/pandas_15case_v1/manifest.jsonl \\
        --repo-root results/giant_codebase/pandas_src \\
        --model Qwen/Qwen2.5-Coder-7B-Instruct
"""
from __future__ import annotations
import argparse
import bisect
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("/home/gfy/CodeMAS_Project/sglang-kvflow")
sys.path.insert(0, str(ROOT / "python/sglang/srt/mem_cache"))
from ast_chunker import ASTChunker  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--repo-root", required=True, type=Path)
    p.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    p.add_argument("--out", default=None, type=Path,
                   help="write JSON summary here (default: <manifest_dir>/nodekind_budget.json)")
    return p.parse_args()


def load_tokenizer(model: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model, trust_remote_code=True)


def token_count_to_byte(text: str, tok, byte_pos: int) -> int:
    """Mirror RadixCache._lookup_byte_offset: number of tokens whose END byte
    <= byte_pos = bisect_right on sorted token end offsets."""
    out = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = out["offset_mapping"] if isinstance(out, dict) else out
    # transformers 5.x returns a BatchEncoding of tokenizers.Encoding objects
    # when called on a single string; unwrap to the (start, end) tuple list.
    if hasattr(offsets, "data") and "offset_mapping" in getattr(offsets, "data", {}):
        offsets = offsets["offset_mapping"]
    if offsets and hasattr(offsets[0], "offsets"):
        offsets = offsets[0].offsets
    elif offsets and not isinstance(offsets[0], tuple):
        offsets = list(offsets[0])
    ends = sorted(en for (_st, en) in offsets if en is not None and en >= 0)
    return bisect.bisect_right(ends, byte_pos)


def main():
    args = parse_args()
    print(f"[budget] loading tokenizer {args.model} ...", flush=True)
    tok = load_tokenizer(args.model)
    chunker = ASTChunker()

    recs = [json.loads(l) for l in args.manifest.read_text().splitlines() if l.strip()]
    print(f"[budget] {len(recs)} chunks in manifest", flush=True)

    # Group by source file so each file is read once.
    by_file: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        by_file[r["slot_id"]].append(r)

    per_chunk = []          # per-chunk record
    agg = defaultdict(float)
    per_type = defaultdict(lambda: {"n": 0, "iface_tok": 0, "sig_tok": 0,
                                    "chunk_len_manifest": 0, "chunk_len_standalone": 0})

    for slot_id, recs_in_file in by_file.items():
        relpath = slot_id.split(":", 1)[1]
        fpath = args.repo_root / relpath
        if not fpath.exists():
            print(f"[budget] WARN missing file {fpath}; skipping {len(recs_in_file)} chunks",
                  flush=True)
            continue
        ftext = fpath.read_text()
        for r in recs_in_file:
            chunk_text = ftext[r["byte_start"]:r["byte_end"]]
            if not chunk_text.strip():
                continue
            try:
                spans = chunker.chunk_text(chunk_text)
            except Exception:
                spans = []
            match = next((s for s in spans
                          if s.name == r["name"] and s.anchor_type == r["anchor_type"]), None)
            if match is None or match.interface_end_byte <= 0:
                # No interface boundary (module/control-flow or unparseable) -> K=0
                per_chunk.append({**{k: r[k] for k in ("slot_id", "name", "anchor_type",
                                                       "n_tokens")},
                                  "iface_tok": 0, "sig_tok": 0,
                                  "chunk_len_standalone": r["n_tokens"]})
                continue
            # K = tokens in [chunk.byte_start, boundary). For a standalone
            # chunk_text the chunk starts at byte 0, so K = token_count_to_byte(boundary).
            iface_tok = token_count_to_byte(chunk_text, tok, match.interface_end_byte)
            sig_tok = (token_count_to_byte(chunk_text, tok, match.signature_end_byte)
                       if match.signature_end_byte > 0 else 0)
            chunk_len_standalone = token_count_to_byte(chunk_text, tok, len(chunk_text))
            per_chunk.append({**{k: r[k] for k in ("slot_id", "name", "anchor_type", "n_tokens")},
                              "iface_tok": iface_tok, "sig_tok": sig_tok,
                              "chunk_len_standalone": chunk_len_standalone})
            t = r["anchor_type"]
            per_type[t]["n"] += 1
            per_type[t]["iface_tok"] += iface_tok
            per_type[t]["sig_tok"] += sig_tok
            per_type[t]["chunk_len_manifest"] += r["n_tokens"]
            per_type[t]["chunk_len_standalone"] += chunk_len_standalone

    total_iface = sum(c["iface_tok"] for c in per_chunk)
    total_sig = sum(c["sig_tok"] for c in per_chunk)
    total_len_manifest = sum(c["n_tokens"] for c in per_chunk)
    total_len_standalone = sum(c["chunk_len_standalone"] for c in per_chunk)
    n_chunks = len(per_chunk)
    n_with_iface = sum(1 for c in per_chunk if c["iface_tok"] > 0)

    frac_iface_manifest = total_iface / total_len_manifest if total_len_manifest else 0
    frac_sig_manifest = total_sig / total_len_manifest if total_len_manifest else 0
    frac_iface_standalone = total_iface / total_len_standalone if total_len_standalone else 0

    summary = {
        "manifest": str(args.manifest),
        "model": args.model,
        "n_chunks": n_chunks,
        "n_chunks_with_interface": n_with_iface,
        "interface_fire_rate": n_with_iface / n_chunks if n_chunks else 0,
        "B_nodekind_interface": total_iface,
        "B_nodekind_signature": total_sig,
        "total_chunk_len_manifest": total_len_manifest,
        "total_chunk_len_standalone": total_len_standalone,
        "frac_star_interface_manifest": frac_iface_manifest,
        "frac_star_signature_manifest": frac_sig_manifest,
        "frac_star_interface_standalone": frac_iface_standalone,
        "per_anchor_type": {
            t: {k: v for k, v in d.items()} for t, d in per_type.items()
        },
    }

    print("\n=== node-kind recompute budget (direction A) ===")
    print(f"chunks: {n_chunks}  (with interface boundary: {n_with_iface}, "
          f"fire rate {summary['interface_fire_rate']*100:.1f}%)")
    print(f"B_nodekind interface = {total_iface}   signature = {total_sig}")
    print(f"total chunk_len      = {total_len_manifest} (manifest) / "
          f"{total_len_standalone} (standalone retok)")
    print(f"frac* INTERFACE (manifest)   = {frac_iface_manifest:.4f}  "
          f"-> R32 frac to equalize B (sig+docstring recompute, body copy)")
    print(f"frac* SIGNATURE (manifest)   = {frac_sig_manifest:.4f}  "
          f"-> R32 frac to equalize B (signature-only recompute)")
    print(f"frac* INTERFACE (standalone) = {frac_iface_standalone:.4f}  "
          f"(cross-check, retokenized chunk text)")
    print("\nper anchor_type:")
    print(f"  {'type':>10} {'n':>5} {'iface_tok':>10} {'sig_tok':>9} "
          f"{'len_manifest':>13} {'iface_ratio':>11}")
    for t, d in sorted(per_type.items()):
        ratio = d["iface_tok"] / d["chunk_len_manifest"] if d["chunk_len_manifest"] else 0
        print(f"  {t:>10} {d['n']:>5} {d['iface_tok']:>10.0f} {d['sig_tok']:>9.0f} "
              f"{d['chunk_len_manifest']:>13.0f} {ratio:>11.4f}")

    out_path = args.out or (args.manifest.parent / "nodekind_budget.json")
    out_path.write_text(json.dumps(summary, indent=2))
    # Also dump per-chunk detail for auditing R34-style no-op checks.
    (args.manifest.parent / "nodekind_budget_per_chunk.jsonl").write_text(
        "\n".join(json.dumps(c) for c in per_chunk) + "\n"
    )
    print(f"\n[budget] wrote {out_path}")
    print(f"[budget] wrote {args.manifest.parent / 'nodekind_budget_per_chunk.jsonl'}")


if __name__ == "__main__":
    main()
