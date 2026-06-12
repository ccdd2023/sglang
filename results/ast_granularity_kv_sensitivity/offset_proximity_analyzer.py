#!/usr/bin/env python3
"""KVCOMM-style offset-proximity diagnostic for AST-selected exact code spans."""

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
    print(f"[offset_proximity] loading {model_name} dtype={dtype} device={device}", flush=True)
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
    print(f"[offset_proximity] loaded in {time.time() - t0:.1f}s", flush=True)
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


def rotate_half_neox(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope_delta_neox(keys: torch.Tensor, delta: int, rope_theta: float) -> torch.Tensor:
    dim = keys.shape[-1]
    inv_freq = 1.0 / (
        rope_theta
        ** (torch.arange(0, dim, 2, device=keys.device, dtype=torch.float32) / dim)
    )
    positions = torch.full((keys.shape[-2],), float(delta), device=keys.device)
    freqs = torch.einsum("i,j->ij", positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1).to(dtype=torch.float32)
    cos = emb.cos()[None, None, :, :]
    sin = emb.sin()[None, None, :, :]
    keys_f = keys.to(torch.float32)
    return keys_f * cos + rotate_half_neox(keys_f) * sin


def neutral_variation(span: dict[str, Any]) -> dict[str, str]:
    target_code = span["text"]
    user_prompt = (
        f"## Code object\n"
        f"granularity={span['granularity']} ast={span['ast_type']} "
        f"path={span['path']}:{span['start_line']}-{span['end_line']}\n"
        "```python\n"
        f"{target_code}\n"
        "```\n\n"
        "## Task\nRead this exact repository code object."
    )
    return {
        "agent_role": "base",
        "system_prompt": "You are a neutral code reader.",
        "user_prompt": user_prompt,
        "target_code": target_code,
    }


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


def offset_distance(role: dict, planner: dict, base: dict, rope_theta: float) -> dict[str, float]:
    length = min(role["k"].shape[-2], planner["k"].shape[-2], base["k"].shape[-2])
    role_k = role["k"][..., :length, :]
    planner_k = planner["k"][..., :length, :]
    base_k = base["k"][..., :length, :]
    role_v = role["v"][..., :length, :]
    planner_v = planner["v"][..., :length, :]
    base_v = base["v"][..., :length, :]

    role_k_aligned = apply_rope_delta_neox(role_k, base["start"] - role["start"], rope_theta)
    planner_k_aligned = apply_rope_delta_neox(planner_k, base["start"] - planner["start"], rope_theta)
    base_k_aligned = base_k

    delta_role_k = role_k_aligned - base_k_aligned
    delta_planner_k = planner_k_aligned - base_k_aligned
    delta_role_v = role_v - base_v
    delta_planner_v = planner_v - base_v

    dk = (delta_role_k - delta_planner_k).norm(2, dim=-2).mean().item()
    dv = (delta_role_v - delta_planner_v).norm(2, dim=-2).mean().item()
    dm = (dk + dv) / 2.0
    return {
        "offset_d_key": dk,
        "offset_d_value": dv,
        "offset_d_mean": dm,
        "offset_d_norm": dm / max(1.0, math.sqrt(length)),
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
        "tail_rate_050": sum(1 for x in vals if x > 0.5) / len(vals),
    }


def aggregate(records: list[dict]) -> dict:
    out = {"overall": stats([r["offset_d_norm"] for r in records])}
    buckets: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        buckets[rec["granularity"]].append(rec)
    out["by_granularity"] = {
        name: {
            **stats([r["offset_d_norm"] for r in rows]),
            "unique_spans": len({r["span_id"] for r in rows}),
            "retention_tokens": sum({r["span_id"]: r["span_tokens"] for r in rows}.values()),
        }
        for name, rows in sorted(buckets.items())
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--spans", type=Path, default=DATA / "spans.json")
    parser.add_argument("--variations", type=Path, default=DATA / "variations.json")
    parser.add_argument("--out", type=Path, default=DATA / "ast_granularity_offset_proximity_7b.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--max-spans", type=int, default=-1)
    args = parser.parse_args()

    spans = {row["span_id"]: row for row in json.loads(args.spans.read_text(encoding="utf-8"))}
    variations = json.loads(args.variations.read_text(encoding="utf-8"))
    by_span: dict[str, dict[str, dict]] = defaultdict(dict)
    for var in variations:
        by_span[var["span_id"]][var["agent_role"]] = var
    span_ids = list(by_span)
    if args.max_spans > 0:
        span_ids = span_ids[: args.max_spans]

    model, tokenizer = load_model(args.model, args.device)
    rope_theta = float(getattr(model.config, "rope_theta", 1000000.0))

    records = []
    for idx, span_id in enumerate(span_ids, start=1):
        role_vars = by_span[span_id]
        if not {"planner", "coder", "reviewer"}.issubset(role_vars):
            continue
        span = spans[span_id]
        try:
            base = capture_span_kv(model, tokenizer, neutral_variation(span), args.device, args.max_seq_len)
            planner = capture_span_kv(model, tokenizer, role_vars["planner"], args.device, args.max_seq_len)
            for role in ("coder", "reviewer"):
                current = capture_span_kv(model, tokenizer, role_vars[role], args.device, args.max_seq_len)
                dist = offset_distance(current, planner, base, rope_theta)
                records.append(
                    {
                        "span_id": span_id,
                        "agent_role": role,
                        "granularity": span["granularity"],
                        "ast_type": span["ast_type"],
                        "repo": span["repo"],
                        "path": span["path"],
                        "start_line": span["start_line"],
                        "end_line": span["end_line"],
                        "content_signature": span["content_signature"],
                        "base_start": base["start"],
                        "planner_start": planner["start"],
                        "target_start": current["start"],
                        "planner_to_base_delta": base["start"] - planner["start"],
                        "target_to_base_delta": base["start"] - current["start"],
                        **dist,
                    }
                )
                del current
            del base, planner
        except Exception as exc:
            print(f"[offset_proximity] skip {span_id}: {exc}", flush=True)
        if idx % 10 == 0:
            print(f"[offset_proximity] processed {idx}/{len(span_ids)} spans; records={len(records)}", flush=True)
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    payload = {
        "config": {
            "model": args.model,
            "selected_layers": SELECTED_LAYERS,
            "n_spans": len(span_ids),
            "n_records": len(records),
            "max_seq_len": args.max_seq_len,
            "base_role": "neutral code reader",
            "canonical": {"agent_role": "planner", "same_exact_code_object": True},
            "rope_aligned_to": "base_start",
        },
        "summary": aggregate(records),
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[offset_proximity] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
