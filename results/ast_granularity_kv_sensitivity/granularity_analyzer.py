#!/usr/bin/env python3
"""Measure KV sensitivity for exact code objects at different AST granularities."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path("/home/gfy/CodeMAS_Project/sglang-kvflow")
DATA = ROOT / "results" / "ast_granularity_kv_sensitivity" / "data"
SELECTED_LAYERS = (-1, -2, -3, -4)


def load_model(model_name: str, device: str):
    dtype = torch.bfloat16 if device.startswith("cuda") and torch.cuda.is_bf16_supported() else torch.float16
    if device == "cpu":
        dtype = torch.float32
    print(f"[ast_granularity] loading {model_name} dtype={dtype} device={device}", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="eager",
    ).to(device)
    model.eval()
    print(f"[ast_granularity] loaded in {time.time() - t0:.1f}s", flush=True)
    return model, tok


def render_chat(tokenizer: Any, system_prompt: str, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"


def token_bounds_for_text(tokenizer: Any, full_text: str, target_text: str) -> tuple[int, int]:
    char_pos = full_text.find(target_text)
    if char_pos < 0:
        raise ValueError("target code object not found in rendered prompt")
    start = len(tokenizer.encode(full_text[:char_pos], add_special_tokens=False))
    end = len(tokenizer.encode(full_text[: char_pos + len(target_text)], add_special_tokens=False))
    return start, end


def get_layer_kv(past_key_values: Any, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        return past_key_values.key_cache[layer_idx], past_key_values.value_cache[layer_idx]
    if hasattr(past_key_values, "layers"):
        layer = past_key_values.layers[layer_idx]
        return layer.keys, layer.values
    entry = past_key_values[layer_idx]
    if isinstance(entry, tuple):
        return entry[0], entry[1]
    return entry.key_cache, entry.value_cache


@torch.no_grad()
def capture_span_kv(model, tokenizer, variation: dict, device: str, max_seq_len: int):
    prompt = render_chat(tokenizer, variation["system_prompt"], variation["user_prompt"])
    start, end = token_bounds_for_text(tokenizer, prompt, variation["target_code"])
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_seq_len)
    if end > int(enc["input_ids"].shape[1]):
        raise ValueError(f"target code object truncated: end={end} seq={enc['input_ids'].shape[1]}")
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(**enc, use_cache=True, return_dict=True)
    layers = sorted({li % getattr(model.config, "num_hidden_layers", 28) for li in SELECTED_LAYERS})
    ks, vs = [], []
    for layer_idx in layers:
        k, v = get_layer_kv(out.past_key_values, layer_idx)
        ks.append(k[:, :, start:end, :].detach().to(torch.float32).cpu())
        vs.append(v[:, :, start:end, :].detach().to(torch.float32).cpu())
    return {
        "k": torch.cat(ks, dim=0),
        "v": torch.cat(vs, dim=0),
        "start": start,
        "end": end,
        "seq_len": int(enc["input_ids"].shape[1]),
    }


def d_norm(a: dict, b: dict) -> dict:
    length = min(a["k"].shape[-2], b["k"].shape[-2])
    ak, bk = a["k"][..., :length, :], b["k"][..., :length, :]
    av, bv = a["v"][..., :length, :], b["v"][..., :length, :]
    dk = (ak - bk).norm(2, dim=-2).mean().item()
    dv = (av - bv).norm(2, dim=-2).mean().item()
    dm = (dk + dv) / 2.0
    return {
        "d_key": dk,
        "d_value": dv,
        "d_mean": dm,
        "d_norm": dm / max(1.0, math.sqrt(length)),
        "span_tokens": length,
    }


def stats(vals: list[float]) -> dict:
    if not vals:
        return {"count": 0}
    vals = sorted(vals)
    mean = sum(vals) / len(vals)
    return {
        "count": len(vals),
        "mean": mean,
        "std": (sum((x - mean) ** 2 for x in vals) / max(1, len(vals) - 1)) ** 0.5,
        "min": vals[0],
        "p50": vals[len(vals) // 2],
        "p90": vals[min(len(vals) - 1, int(len(vals) * 0.9))],
        "max": vals[-1],
    }


def token_bin(tokens: int) -> str:
    if tokens < 64:
        return "<64"
    if tokens < 160:
        return "64-159"
    if tokens < 384:
        return "160-383"
    return ">=384"


def aggregate(records: list[dict]) -> dict:
    out = {"overall": stats([r["d_norm"] for r in records])}
    for axis in ("granularity", "agent_role", "ast_type", "token_bin"):
        buckets: dict[str, list[dict]] = defaultdict(list)
        for rec in records:
            buckets[str(rec[axis])].append(rec)
        axis_out = {}
        for name, rows in sorted(buckets.items()):
            vals = [r["d_norm"] for r in rows]
            toks = [r["span_tokens"] for r in rows]
            weighted = sum(r["d_norm"] * r["span_tokens"] for r in rows) / max(1, sum(toks))
            axis_out[name] = {
                **stats(vals),
                "mean_span_tokens": sum(toks) / max(1, len(toks)),
                "token_weighted_d_norm": weighted,
                "device_retention_cost_tokens": sum({r["span_id"]: r["span_tokens"] for r in rows}.values()),
                "reuse_score": (sum(toks) / max(1, len(toks))) / (1.0 + stats(vals).get("p90", 0.0)),
                "unique_spans": len({r["span_id"] for r in rows}),
            }
        out[f"by_{axis}"] = axis_out
    out["worst_cases"] = [
        {
            "span_id": r["span_id"],
            "granularity": r["granularity"],
            "agent_role": r["agent_role"],
            "path": r["path"],
            "lines": f"{r['start_line']}-{r['end_line']}",
            "d_norm": r["d_norm"],
            "span_tokens": r["span_tokens"],
            "target_start": r["target_start"],
        }
        for r in sorted(records, key=lambda row: row["d_norm"], reverse=True)[:10]
    ]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--spans", type=Path, default=DATA / "spans.json")
    parser.add_argument("--variations", type=Path, default=DATA / "variations.json")
    parser.add_argument("--out", type=Path, default=DATA / "ast_granularity_distance_7b.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--max-variations", type=int, default=-1)
    args = parser.parse_args()

    spans = {row["span_id"]: row for row in json.loads(args.spans.read_text(encoding="utf-8"))}
    variations = json.loads(args.variations.read_text(encoding="utf-8"))
    if args.max_variations > 0:
        variations = variations[: args.max_variations]
    model, tokenizer = load_model(args.model, args.device)

    by_span: dict[str, dict[str, dict]] = defaultdict(dict)
    for var in variations:
        by_span[var["span_id"]][var["agent_role"]] = var

    records = []
    captured = 0
    for span_id, role_vars in by_span.items():
        if "planner" not in role_vars:
            continue
        span = spans[span_id]
        canonical = capture_span_kv(model, tokenizer, role_vars["planner"], args.device, args.max_seq_len)
        for role in ("planner", "coder", "reviewer"):
            if role not in role_vars:
                continue
            current = canonical if role == "planner" else capture_span_kv(model, tokenizer, role_vars[role], args.device, args.max_seq_len)
            dist = d_norm(current, canonical)
            records.append(
                {
                    "span_id": span_id,
                    "agent_role": role,
                    "granularity": span["granularity"],
                    "ast_type": span["ast_type"],
                    "path": span["path"],
                    "start_line": span["start_line"],
                    "end_line": span["end_line"],
                    "approx_tokens": span["approx_tokens"],
                    "token_bin": token_bin(span["approx_tokens"]),
                    "content_signature": span["content_signature"],
                    **dist,
                    "target_start": current["start"],
                    "target_end": current["end"],
                    "seq_len": current["seq_len"],
                }
            )
            captured += 1
            if role != "planner":
                del current
            if captured % 30 == 0:
                print(f"[ast_granularity] captured {captured}/{len(variations)}", flush=True)
        del canonical
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    payload = {
        "config": {
            "model": args.model,
            "selected_layers": SELECTED_LAYERS,
            "canonical": {"agent_role": "planner", "same_exact_code_object": True},
            "n_spans": len(spans),
            "n_variations": len(variations),
            "max_seq_len": args.max_seq_len,
        },
        "summary": aggregate(records),
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[ast_granularity] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
