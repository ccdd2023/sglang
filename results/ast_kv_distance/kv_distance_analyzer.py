"""KVCOMM-style KV distance analyzer.

Loads Qwen2.5-Coder-7B-Instruct, captures K/V activations for every segment
collected by ast_sampler, then computes pairwise distances using the same
formulas as KVCOMM (kvcomm_engine.py:827-879, 1002-1072):

    d = (real.unsqueeze(0) - anchor).norm(2, dim=-2)         # L2 along seq
    w = softmax(-d / T, dim=0)                                # attention-like
    entropy = -sum(w * log2(w))                              # uncertainty
    threshold_gamma = 0.3                                    # KVCOMM default

The output is one record per pair (a, b) summarising:
    - d_key_mean, d_value_mean (lower == more reusable)
    - entropy
    - cross-entropy against the perfect-target distribution
    - bucketing: "low_entropy" (entropy < gamma*log2(A)) vs "high_entropy"

Pairs are also aggregated by (ast_type_a, ast_type_b), (template_a, template_b),
(length_bin_a, length_bin_b) for high-level analysis.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Match KVCOMM defaults
KVCOMM_THRESHOLD = 0.3
TEMPERATURE = 1.0
EPS = 1e-40

# Sampling controls — we don't need every layer for a coarse structural signal.
# Last 4 layers (KVCOMM's "preserved layer 0" idea inverted: deeper layers carry
# more task-specific signal and are where code-structure differences should
# manifest most strongly).
SELECTED_LAYERS = (-1, -2, -3, -4)

PROMPT_TEMPLATE = (
    "Below is a Python code snippet. Summarise its purpose in one sentence.\n\n"
    "```python\n"
    "{code}\n"
    "```\n\n"
    "Summary:"
)


@dataclass
class SegmentKV:
    seg_id: str
    ast_type: str
    template: str
    length_bin: str
    token_count: int
    k_last: torch.Tensor        # [layer, head, seq, dim]
    v_last: torch.Tensor        # [layer, head, seq, dim]


# ---------- Model loading -----------------------------------------------

def load_model(model_name: str, dtype: torch.dtype = torch.bfloat16, device: str = "cuda"):
    print(f"[kv_distance] loading {model_name} (dtype={dtype}) on {device} ...", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="eager",   # eager is friendlier for K/V hooks
    )
    model.eval()
    print(f"[kv_distance] loaded in {time.time() - t0:.1f}s", flush=True)
    return model, tokenizer


# ---------- KV capture --------------------------------------------------

@torch.no_grad()
def capture_segments(
    model, tokenizer, segments: list[dict], device: str = "cuda", max_seq_len: int = 1024
) -> list[SegmentKV]:
    """Use use_cache=True so the model returns its own K/V tensors."""
    out: list[SegmentKV] = []
    num_layers = getattr(model.config, "num_hidden_layers", 28)
    selected = sorted({li % num_layers for li in SELECTED_LAYERS})
    for i, seg in enumerate(segments):
        code = seg["source"]
        prompt = PROMPT_TEMPLATE.format(code=code)
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_seq_len)
        enc = {k: v.to(device) for k, v in enc.items()}
        try:
            model_out = model(**enc, use_cache=True, return_dict=True)
        except Exception as e:
            print(f"[kv_distance] forward failed for {seg['seg_id']}: {e}", flush=True)
            continue
        pkv = model_out.past_key_values
        if pkv is None or not hasattr(pkv, "layers") or len(pkv.layers) == 0:
            continue
        # pkv.layers[li].keys/.values -> [B, H, S, d]
        ks, vs = [], []
        for li in selected:
            layer = pkv.layers[li]
            ks.append(layer.keys.detach().to(torch.float32).cpu())
            vs.append(layer.values.detach().to(torch.float32).cpu())
        k_last = torch.cat(ks, dim=0).squeeze(1)   # [L, H, S, d]
        v_last = torch.cat(vs, dim=0).squeeze(1)
        out.append(
            SegmentKV(
                seg_id=seg["seg_id"],
                ast_type=seg["ast_type"],
                template=seg["template"],
                length_bin=seg["length_bin"],
                token_count=seg["token_count"],
                k_last=k_last,
                v_last=v_last,
            )
        )
        if (i + 1) % 5 == 0:
            print(f"[kv_distance] captured {i + 1}/{len(segments)} segments", flush=True)
    return out


# ---------- Distance metrics -------------------------------------------

def _l2_pair(real: torch.Tensor, anchor: torch.Tensor) -> float:
    """L2 norm of (real - anchor) along the sequence dim, averaged over
    (layer, head, dim). Mirrors KVCOMM's prefix-K branch in
    _compute_anchor_weight_entry (kvcomm_engine.py:850).

    real / anchor: [layer, head, seq, dim] (we squeeze the batch dim).
    """
    S = min(real.shape[2], anchor.shape[2])
    r = real[..., :S, :]
    a = anchor[..., :S, :]
    diff = (r - a)
    # L2 along seq, mean over the rest
    norm_per_layer_head_dim = diff.norm(2, dim=-2)  # [layer, head, dim]
    return norm_per_layer_head_dim.mean().item()


def _entropy_over_pool(query: torch.Tensor, pool: list[torch.Tensor], T: float = TEMPERATURE) -> tuple[float, float]:
    """KVCOMM's predict_as_anchor entropy (kvcomm_engine.py:1030-1038).

    Computes the softmax over a pool of candidates and returns its Shannon
    entropy in bits. Low entropy => pool contains a close match; high entropy
    => query is unlike every candidate.

    Returns: (entropy_bits, max_entropy_bits)
    """
    if not pool:
        return 0.0, 0.0
    dists = torch.tensor([_l2_pair(query, p) for p in pool])
    w = torch.softmax(-dists.float() / T, dim=0)
    eps = torch.tensor(EPS)
    entropy = -(w * (w + eps).log2()).sum().item()
    max_e = math.log2(len(pool))
    return entropy, max_e


def pairwise_distances(records: list[SegmentKV]) -> list[dict]:
    """All-pairs L2 distance; per-pair entropy is intentionally not used
    (single candidate is degenerate). Gate stats come from the pool-based
    entropy below.

    d_mean is the raw KVCOMM L2 mean over (layer, head, dim) at the truncated
    sequence length. d_norm is d / sqrt(seq_len) — this normalises the natural
    sqrt(N) growth of L2 norms over random vectors and lets us compare across
    length bins fairly.
    """
    out: list[dict] = []
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            ri, rj = records[i], records[j]
            dk = _l2_pair(ri.k_last, rj.k_last)
            dv = _l2_pair(ri.v_last, rj.v_last)
            d = (dk + dv) / 2.0
            S = min(ri.k_last.shape[2], rj.k_last.shape[2])
            d_norm = d / max(1.0, math.sqrt(S))
            out.append(
                {
                    "i": ri.seg_id,
                    "j": rj.seg_id,
                    "ast_type_i": ri.ast_type,
                    "ast_type_j": rj.ast_type,
                    "template_i": ri.template,
                    "template_j": rj.template,
                    "length_bin_i": ri.length_bin,
                    "length_bin_j": rj.length_bin,
                    "token_count_i": ri.token_count,
                    "token_count_j": rj.token_count,
                    "seq_len": S,
                    "d_key": dk,
                    "d_value": dv,
                    "d_mean": d,
                    "d_norm": d_norm,
                }
            )
    return out


def pool_entropy_stats(records: list[SegmentKV]) -> dict:
    """For each record, compute its entropy against a pool defined by a key
    function over (ast_type, template, length_bin).

    A query has high entropy against its own pool => the pool is internally
    diverse; low entropy => the pool is internally coherent (good for reuse).
    """
    from collections import defaultdict

    def _pool_index(key_fn) -> dict:
        idx: dict = defaultdict(list)
        for r in records:
            idx[key_fn(r)].append(r)
        return idx

    out: dict = {}
    for dim_name, key_fn in [
        ("ast_type", lambda r: r.ast_type),
        ("template", lambda r: r.template),
        ("length_bin", lambda r: r.length_bin),
    ]:
        idx = _pool_index(key_fn)
        per_pool: dict = {}
        for k, members in idx.items():
            if len(members) < 2:
                per_pool[k] = {"count": len(members), "skipped": True}
                continue
            for query in members:
                pool_k = [m.k_last for m in members if m.seg_id != query.seg_id]
                pool_v = [m.v_last for m in members if m.seg_id != query.seg_id]
                ent_k, max_k = _entropy_over_pool(query.k_last, pool_k)
                ent_v, max_v = _entropy_over_pool(query.v_last, pool_v)
                ent = (ent_k + ent_v) / 2.0
                max_e = (max_k + max_v) / 2.0
                gate_pass = ent <= KVCOMM_THRESHOLD * max_e
                per_pool.setdefault(k, {"count": len(members), "per_query": []})
                per_pool[k]["per_query"].append(
                    {"seg_id": query.seg_id, "entropy": ent, "max_entropy": max_e, "gate_pass": gate_pass}
                )
        rolled: dict = {}
        for k, info in per_pool.items():
            if "per_query" not in info:
                rolled[k] = info
                continue
            ents = [q["entropy"] for q in info["per_query"]]
            gates = [q["gate_pass"] for q in info["per_query"]]
            rolled[k] = {
                "count": info["count"],
                "entropy_avg": sum(ents) / len(ents),
                "entropy_min": min(ents),
                "entropy_max": max(ents),
                "gate_pass_rate": sum(1 for g in gates if g) / len(gates),
            }
        out[f"by_{dim_name}"] = rolled
    return out


# ---------- Aggregation -------------------------------------------------

def aggregate(pairs: list[dict]) -> dict:
    """Aggregate pair distances by (ast_type, ast_type), (template, template), (length_bin, length_bin).
    Reports both raw d_mean and length-normalised d_norm so length-bucket
    comparisons stay fair."""

    def _group(key_fn) -> dict:
        agg: dict = defaultdict(list)
        for p in pairs:
            k = key_fn(p)
            agg[k].append(p)
        out = {}
        for k, ps in agg.items():
            ds = sorted([p["d_mean"] for p in ps])
            dns = sorted([p["d_norm"] for p in ps])
            out[str(k)] = {
                "count": len(ps),
                "d_mean_avg": sum(ds) / len(ds),
                "d_mean_min": ds[0],
                "d_mean_p25": ds[len(ds) // 4],
                "d_mean_p50": ds[len(ds) // 2],
                "d_mean_p75": ds[3 * len(ds) // 4],
                "d_mean_max": ds[-1],
                "d_norm_avg": sum(dns) / len(dns),
                "d_norm_p50": dns[len(dns) // 2],
            }
        return out

    return {
        "by_ast_type_pair": _group(lambda p: (p["ast_type_i"], p["ast_type_j"])),
        "by_template_pair": _group(lambda p: (p["template_i"], p["template_j"])),
        "by_length_pair": _group(lambda p: (p["length_bin_i"], p["length_bin_j"])),
    }


def structural_gate_stats(pairs: list[dict]) -> dict:
    """For the user's research question: given AST-type-equal pairs vs cross-type
    pairs, what's the L2 distance distribution? Uses d_norm (length-normalised)
    for the within-vs-cross comparison to keep length effects out of the
    verdict. This is the candidate signal for a new gate tier in anchor_match.py."""
    within = [p for p in pairs if p["ast_type_i"] == p["ast_type_j"]]
    cross = [p for p in pairs if p["ast_type_i"] != p["ast_type_j"]]

    def _stats(ps):
        if not ps:
            return {"count": 0}
        ds = sorted([p["d_mean"] for p in ps])
        dns = sorted([p["d_norm"] for p in ps])
        return {
            "count": len(ps),
            "d_mean_avg": sum(ds) / len(ds),
            "d_mean_p50": ds[len(ds) // 2],
            "d_norm_avg": sum(dns) / len(dns),
            "d_norm_p50": dns[len(dns) // 2],
            "d_norm_min": dns[0],
            "d_norm_max": dns[-1],
        }

    ratio_norm = (
        _stats(within)["d_norm_avg"] / _stats(cross)["d_norm_avg"]
        if within and cross else None
    )
    return {
        "within_ast_type": _stats(within),
        "cross_ast_type": _stats(cross),
        "ratio_within_to_cross_d_norm": ratio_norm,
        "interpretation": (
            "ratio_norm < 1: within-type pairs are CLOSER per-token (more reusable); "
            "ratio_norm > 1: cross-type pairs are closer (structure irrelevant)"
        ),
    }


# ---------- Main -------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    p.add_argument("--segments", default="/home/gfy/CodeMAS_Project/sglang-kvflow/results/ast_kv_distance/data/segments.json")
    p.add_argument("--out", default="/home/gfy/CodeMAS_Project/sglang-kvflow/results/ast_kv_distance/data/distance_results.json")
    p.add_argument("--max-segments", type=int, default=80, help="cap for runtime; -1 = all")
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    with open(args.segments) as f:
        segments = json.load(f)
    if args.max_segments > 0:
        segments = segments[: args.max_segments]
    print(f"[kv_distance] processing {len(segments)} segments", flush=True)

    model, tokenizer = load_model(args.model, device=args.device)
    records = capture_segments(model, tokenizer, segments, device=args.device, max_seq_len=args.max_seq_len)
    print(f"[kv_distance] captured {len(records)} records", flush=True)

    pairs = pairwise_distances(records)
    print(f"[kv_distance] computed {len(pairs)} pairs", flush=True)

    summary = {
        "config": {
            "model": args.model,
            "max_segments": args.max_segments,
            "max_seq_len": args.max_seq_len,
            "kvcomm_threshold": KVCOMM_THRESHOLD,
            "selected_layers": SELECTED_LAYERS,
        },
        "segment_summary": {
            "n_segments": len(records),
            "by_ast_type": _counter(records, "ast_type"),
            "by_template": _counter(records, "template"),
            "by_length_bin": _counter(records, "length_bin"),
        },
        "pair_aggregations": aggregate(pairs),
        "pool_entropy_stats": pool_entropy_stats(records),
        "structural_gate_stats": structural_gate_stats(pairs),
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[kv_distance] wrote summary to {args.out}", flush=True)

    # Free GPU memory
    del model
    torch.cuda.empty_cache()


def _counter(records, field):
    from collections import Counter
    return dict(Counter(getattr(r, field) for r in records))


if __name__ == "__main__":
    main()
