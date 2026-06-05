"""Same-Code × Different-Context KV distance analyzer.

Loads the segments + variations from context_sampler.py, runs Qwen2.5-Coder-7B-
Instruct for each (code, prompt_variation), captures K/V at the last 4 layers
(same as `ast_kv_distance/kv_distance_analyzer.py`), and computes the L2
distance between each variation and the **canonical** (offset=0, planner, none)
variation of the same code.

Output schema (data/context_distance_7b.json):
  config: model + grid size
  per_segment: [
    {seg_id, ast_type, length_bin, nesting_depth, n_variations,
     overall: {mean, std, p50, p90, max},
     by_position_offset:   {<offset>: {mean, std, p50, p90, max, n}},
     by_system_prompt_class: {...},
     by_surrounding_code_class: {...},
     max_distance_at: {position_offset, system_prompt_class, surrounding_code_class, d_norm}
    }, ...
  ]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import defaultdict
from typing import Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Match ast_kv_distance settings
SELECTED_LAYERS = (-1, -2, -3, -4)


# ---------- Model loading (replicated from ast_kv_distance) ----------------

def load_model(model_name: str, dtype: torch.dtype = torch.bfloat16, device: str = "cuda"):
    print(f"[ctx_distance] loading {model_name} (dtype={dtype}) on {device} ...", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # Avoid device_map=... (requires `accelerate` which may not be
    # installed in all envs). Load on CPU then move to device explicitly.
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, low_cpu_mem_usage=True,
        trust_remote_code=True, attn_implementation="eager",
    )
    model = model.to(device)
    model.eval()
    print(f"[ctx_distance] loaded in {time.time() - t0:.1f}s", flush=True)
    return model, tokenizer


# ---------- K/V capture (one chat-templated prompt per variation) --------

@torch.no_grad()
def capture_all(model, tokenizer, variations: list[dict], device: str = "cuda", max_seq_len: int = 1024) -> dict:
    """For each variation, run forward with chat template and capture last-4-layer K/V."""
    raw: dict = {}
    selected_set = sorted({li % getattr(model.config, "num_hidden_layers", 28) for li in SELECTED_LAYERS})
    for i, var in enumerate(variations):
        sys = var["system_prompt"]
        usr = var["user_prompt"]
        # Apply chat template manually (qwen2.5 uses ChatML <|im_start|>...<|im_end|>)
        prompt = (
            f"<|im_start|>system\n{sys}<|im_end|>\n"
            f"<|im_start|>user\n{usr}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_seq_len)
        enc = {k: v.to(device) for k, v in enc.items()}
        try:
            out = model(**enc, use_cache=True, return_dict=True)
        except Exception as e:
            print(f"[ctx_distance] forward failed for var {i}: {e}", flush=True)
            continue
        pkv = out.past_key_values
        if pkv is None or not hasattr(pkv, "layers") or len(pkv.layers) == 0:
            continue
        ks, vs = [], []
        for li in selected_set:
            layer = pkv.layers[li]
            ks.append(layer.keys.detach().to(torch.float32).cpu())
            vs.append(layer.values.detach().to(torch.float32).cpu())
        k_last = torch.cat(ks, dim=0).squeeze(1)
        v_last = torch.cat(vs, dim=0).squeeze(1)
        key = (var["seg_id"], var["position_offset"], var["system_prompt_class"], var["surrounding_code_class"])
        raw[key] = (k_last, v_last)
        if (i + 1) % 50 == 0:
            print(f"[ctx_distance] captured {i + 1}/{len(variations)} variations", flush=True)
    return raw


# ---------- L2 + d_norm computation (from ast_kv_distance) ---------------

def _l2_pair(real: torch.Tensor, anchor: torch.Tensor) -> float:
    S = min(real.shape[2], anchor.shape[2])
    r = real[..., :S, :]
    a = anchor[..., :S, :]
    diff = (r - a)
    norm = diff.norm(2, dim=-2)
    return norm.mean().item()


def _d_norm(d: float, seq_len: int) -> float:
    return d / max(1.0, math.sqrt(seq_len))


def compute_distances(raw: dict, variations: list[dict]) -> list[dict]:
    """For each variation, compute d_norm against the canonical (offset=0,
    planner, none) variation of the same seg_id."""
    # Index raw by (seg_id, position_offset, system, surrounding)
    canonical_keys: dict[str, tuple] = {}
    for (sid, off, sys_, surr), (k, v) in raw.items():
        if off == 0 and sys_ == "planner" and surr == "none":
            canonical_keys[sid] = (k, v)

    out: list[dict] = []
    for var in variations:
        key = (var["seg_id"], var["position_offset"], var["system_prompt_class"], var["surrounding_code_class"])
        if key not in raw:
            continue
        if var["seg_id"] not in canonical_keys:
            continue
        canon_k, canon_v = canonical_keys[var["seg_id"]]
        k, v = raw[key]
        dk = _l2_pair(k, canon_k)
        dv = _l2_pair(v, canon_v)
        d = (dk + dv) / 2.0
        S = min(k.shape[2], canon_k.shape[2])
        dn = _d_norm(d, S)
        out.append({**var, "d_key": dk, "d_value": dv, "d_mean": d, "d_norm": dn, "seq_len": S})
    return out


# ---------- Per-segment aggregation ---------------------------------------

def _stats(vals: list[float]) -> dict:
    if not vals:
        return {"count": 0}
    vs = sorted(vals)
    return {
        "count": len(vs),
        "mean": sum(vs) / len(vs),
        "std": (sum((x - sum(vs) / len(vs)) ** 2 for x in vs) / max(1, len(vs) - 1)) ** 0.5,
        "min": vs[0],
        "p50": vs[len(vs) // 2],
        "p90": vs[int(len(vs) * 0.9)],
        "max": vs[-1],
    }


def aggregate(records: list[dict]) -> dict:
    by_seg: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_seg[r["seg_id"]].append(r)

    per_segment = []
    for seg_id, recs in by_seg.items():
        seg = recs[0]
        all_d = [r["d_norm"] for r in recs]
        by_off: dict = defaultdict(list)
        by_sys: dict = defaultdict(list)
        by_sur: dict = defaultdict(list)
        for r in recs:
            by_off[r["position_offset"]].append(r["d_norm"])
            by_sys[r["system_prompt_class"]].append(r["d_norm"])
            by_sur[r["surrounding_code_class"]].append(r["d_norm"])
        # Find the worst (offset, system, surrounding) triple
        max_d = max(all_d)
        max_rec = next((r for r in recs if r["d_norm"] == max_d), None)
        per_segment.append({
            "seg_id": seg_id,
            "ast_type": seg["ast_type"],
            "length_bin": seg["length_bin"],
            "token_count": seg["token_count"],
            "n_variations": len(recs),
            "overall": _stats(all_d),
            "by_position_offset": {str(k): _stats(v) for k, v in by_off.items()},
            "by_system_prompt_class": {k: _stats(v) for k, v in by_sys.items()},
            "by_surrounding_code_class": {k: _stats(v) for k, v in by_sur.items()},
            "max_distance_at": {
                "position_offset": max_rec["position_offset"] if max_rec else None,
                "system_prompt_class": max_rec["system_prompt_class"] if max_rec else None,
                "surrounding_code_class": max_rec["surrounding_code_class"] if max_rec else None,
                "d_norm": max_d,
            },
        })
    return per_segment


# ---------- Main ---------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    p.add_argument("--segments", default="/home/gfy/CodeMAS_Project/sglang-kvflow/results/same_code_context_variation/data/segments.json")
    p.add_argument("--variations", default="/home/gfy/CodeMAS_Project/sglang-kvflow/results/same_code_context_variation/data/variations.json")
    p.add_argument("--out", default="/home/gfy/CodeMAS_Project/sglang-kvflow/results/same_code_context_variation/data/context_distance_7b.json")
    p.add_argument("--max-variations", type=int, default=-1, help="cap for runtime; -1 = all")
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    with open(args.segments) as f:
        segments = json.load(f)
    with open(args.variations) as f:
        variations = json.load(f)
    if args.max_variations > 0:
        variations = variations[: args.max_variations]
    print(f"[ctx_distance] {len(segments)} segments × {len(variations)} variations", flush=True)

    model, tokenizer = load_model(args.model, device=args.device)
    raw = capture_all(model, tokenizer, variations, device=args.device, max_seq_len=args.max_seq_len)
    print(f"[ctx_distance] captured {len(raw)} variation K/V tensors", flush=True)

    records = compute_distances(raw, variations)
    print(f"[ctx_distance] computed {len(records)} distance records", flush=True)

    per_segment = aggregate(records)

    summary = {
        "config": {
            "model": args.model,
            "n_segments": len(segments),
            "n_variations": len(variations),
            "n_variations_per_segment": len(variations) // max(1, len(segments)),
            "selected_layers": SELECTED_LAYERS,
            "max_seq_len": args.max_seq_len,
        },
        "per_segment": per_segment,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[ctx_distance] wrote {args.out}", flush=True)

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
