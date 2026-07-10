#!/usr/bin/env python3
"""HKVD-by-position measurement (2026-07-09).

Validates (or falsifies) the deck's unverified mechanism claim:
  "请求序列中的早期 chunk 离跨上下文前缀边界最近 -> stale KV 风险最高 -> 多算一点"

The pool precomputes each code_base chunk's KV with a CANONICAL preamble
(literal ROLE/CASE/UPSTREAM placeholders). At runtime the LIVE prompt fills
those with real role/case/instruction. The cross-context KV staleness is the
KV error this prefix swap induces on each chunk.

Because attention is causal, a chunk's KV depends only on its preceding
context. We measure, per chunk position i=1..5:

    deviation_i = 1 - cosine( KV(chunk_i | canonical_prefix),
                              KV(chunk_i | live_prefix)      )

averaged over layers (and over the chunk's tokens). The hypothesis predicts
deviation_1 > deviation_5 (early chunks more sensitive to prefix mismatch).

No sglang dependency - uses HuggingFace model directly. Reuses the kv_isolation
pattern from analyze_kv_isolation.py.
"""
from __future__ import annotations
import json, math, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path("/home/gfy/CodeMAS_Project/sglang-kvflow")
MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
MANIFEST = ROOT / "results/giant_codebase/tasks/pandas__pandas__1000/manifest.jsonl"
REPO = ROOT / "results/giant_codebase/pandas_src"
OUT = ROOT / "results/hkvd_by_position_20260709"
OUT.mkdir(parents=True, exist_ok=True)

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

# build segments via the bench's own helper so we use the REAL cases
sys.path.insert(0, str(ROOT / "benchmark/multi_workflow"))
from bench_giant_codebase_reuse import build_segments_for_task  # noqa: E402


def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="sdpa",
    ).cuda().eval()
    return tok, model


def build_prompt(prefix: str, segments) -> tuple[str, list[tuple[int, int]]]:
    """Build prefix + 5 code_base blocks. Return (text, per-seg char spans)."""
    parts = [prefix]
    spans = []
    cursor = len(prefix)
    for idx, seg in enumerate(segments, 1):
        header = f"\n## code_base{idx}: {seg.name}\n```python\n"
        body = seg.text
        footer = "\n```\n"
        parts.append(header)
        cursor += len(header)
        start = cursor
        parts.append(body)
        spans.append((start, start + len(body)))
        cursor = start + len(body)
        parts.append(footer)
        cursor += len(footer)
    return "".join(parts), spans


@torch.no_grad()
def forward_kv(tok, model, text: str):
    enc = tok(text, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False)
    input_ids = enc["input_ids"].cuda()
    offsets = enc["offset_mapping"][0].tolist()  # list of (start,end)
    out = model(input_ids, use_cache=True, output_hidden_states=False)
    pkv = out.past_key_values
    # transformers 5.x DynamicCache: pkv.layers[L].keys / .values -> [1, h, seq, d]
    layers = {}
    n_layers = len(pkv.layers)
    for L in range(n_layers):
        K = pkv.layers[L].keys      # [1, h, seq, d]
        V = pkv.layers[L].values
        layers[L] = (K[0].float().cpu(), V[0].float().cpu())  # [h, seq, d]
    return offsets, layers


def char_span_to_tok_span(offsets, cstart, cend):
    """Map a char span to token indices (inclusive range)."""
    toks = [i for i, (s, e) in enumerate(offsets) if s < cend and e > cstart]
    return toks


def cos_per_layer(layersA, layersB, tokA, tokB):
    """Mean cosine over tokens+heads for K and V, per layer."""
    out = []
    Ls = sorted(layersA.keys())
    for L in Ls:
        KA, VA = layersA[L]
        KB, VB = layersB[L]
        # slice to the chunk's tokens
        ka = KA[:, tokA, :].reshape(-1, KA.shape[-1])  # [h*tA, d]
        kb = KB[:, tokB, :].reshape(-1, KB.shape[-1])
        va = VA[:, tokA, :].reshape(-1, VA.shape[-1])
        vb = VB[:, tokB, :].reshape(-1, VB.shape[-1])
        # align token counts (BPE boundary may differ by 1)
        n = min(ka.shape[0], kb.shape[0])
        ka, kb, va, vb = ka[:n], kb[:n], va[:n], vb[:n]
        def cos(a, b):
            a = F.normalize(a, dim=-1); b = F.normalize(b, dim=-1)
            return (a * b).sum(-1).mean().item()
        out.append({"layer": L, "K_cos": cos(ka, kb), "V_cos": cos(va, vb)})
    return out


def main():
    n_cases = 5
    print(f"[hkvd] loading {MODEL} ...", flush=True)
    tok, model = load_model()
    cases = [json.loads(l) for l in MANIFEST.read_text().splitlines() if l.strip()][:n_cases]

    # per-position aggregation
    pos_dev = {i: [] for i in range(1, 6)}   # deviation (1-cos) per position
    per_layer_pos = {i: {} for i in range(1, 6)}  # position -> layer -> [K_dev, V_dev]

    for ci, c in enumerate(cases):
        segs = build_segments_for_task(c, REPO, segment_count=5, max_file_chars=3000, sibling_window=4)
        if len(segs) < 2:
            print(f"[hkvd] case {c['instance_id'][:40]}: only {len(segs)} segs, skip"); continue
        segs = segs[:5]
        cid = c["instance_id"]
        # canonical prefix (literal placeholders, as the pool precomputed)
        canon = DIRECTION_A_V3_PREAMBLE
        # live prefix: fill ROLE/CASE/UPSTREAM with real values
        live = (DIRECTION_A_V3_PREAMBLE
                .replace("ROLE", "implementer")
                .replace("CASE", cid)
                .replace("UPSTREAM", "(none)"))
        textA, spansA = build_prompt(canon, segs)
        textB, spansB = build_prompt(live, segs)
        print(f"[hkvd] case {ci+1}/{n_cases} {cid[:48]}: {len(segs)} segs, "
              f"tokA~{len(tok(textA)['input_ids'])} tokB~{len(tok(textB)['input_ids'])}", flush=True)
        offA, kvA = forward_kv(tok, model, textA)
        offB, kvB = forward_kv(tok, model, textB)
        for idx, (sa, sb) in enumerate(zip(spansA, spansB), 1):
            tA = char_span_to_tok_span(offA, sa[0], sa[1])
            tB = char_span_to_tok_span(offB, sb[0], sb[1])
            if not tA or not tB:
                continue
            perL = cos_per_layer(kvA, kvB, tA, tB)
            # average K_dev over layers
            k_dev = np.mean([1 - d["K_cos"] for d in perL])
            v_dev = np.mean([1 - d["V_cos"] for d in perL])
            pos_dev[idx].append((k_dev, v_dev))
            for d in perL:
                L = d["layer"]
                if L not in per_layer_pos[idx]:
                    per_layer_pos[idx][L] = {"K": [], "V": []}
                per_layer_pos[idx][L]["K"].append(1 - d["K_cos"])
                per_layer_pos[idx][L]["V"].append(1 - d["V_cos"])
        del kvA, kvB
        torch.cuda.empty_cache()

    # aggregate
    summary = {}
    print("\n=== HKVD by position (1 - cosine, averaged over layers + cases) ===")
    print(f"{'pos':>4} {'K_dev':>10} {'V_dev':>10} {'n':>4}")
    for i in range(1, 6):
        rows = pos_dev[i]
        if not rows:
            print(f"{i:>4}  (no data)"); continue
        k = np.mean([r[0] for r in rows]); v = np.mean([r[1] for r in rows])
        summary[i] = {"K_dev": k, "V_dev": v, "n": len(rows),
                      "K_std": float(np.std([r[0] for r in rows])),
                      "V_std": float(np.std([r[1] for r in rows]))}
        print(f"{i:>4} {k:>10.4f} {v:>10.4f} {len(rows):>4}")

    # per-layer (K_dev) for chart
    per_layer_out = {}
    for i in range(1, 6):
        per_layer_out[i] = {L: {"K_mean": float(np.mean(v["K"])), "V_mean": float(np.mean(v["V"]))}
                            for L, v in per_layer_pos[i].items()}

    (OUT / "hkvd_by_position.json").write_text(json.dumps({
        "model": MODEL, "n_cases": n_cases,
        "summary": summary, "per_layer": per_layer_out,
        "interpretation": "deviation_i = 1 - cos(KV(chunk_i|canonical_prefix), KV(chunk_i|live_prefix)). "
                          "Hypothesis: deviation decreases with position (early chunks more stale).",
    }, indent=2))
    print(f"\n[hkvd] wrote {OUT/'hkvd_by_position.json'}")

    # verdict
    if 1 in summary and 5 in summary:
        d1 = summary[1]["K_dev"]; d5 = summary[5]["K_dev"]
        print(f"\n>>> position 1 K_dev={d1:.4f}  vs  position 5 K_dev={d5:.4f}")
        if d1 > d5:
            print(f">>> HYPOTHESIS VALIDATED: early chunk (pos 1) has HIGHER KV deviation "
                  f"({(d1/d5-1)*100:.1f}% more than pos 5) -> position-proxy is evidence-backed.")
        else:
            print(f">>> HYPOTHESIS FALSIFIED: late chunk has higher deviation -> position-proxy NOT supported.")


if __name__ == "__main__":
    main()
