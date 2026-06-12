#!/usr/bin/env python3
"""Layer-wise RoPE-aligned KV comparison across AST granularities."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path("/home/gfy/CodeMAS_Project/sglang-kvflow")
BASE = ROOT / "results" / "ast_granularity_kv_sensitivity"
DATA = BASE / "data"
FIG = BASE / "figures"
ORDER = ["file_prefix", "class", "function", "method", "control_block", "statement_window"]
COLORS = {
    "file_prefix": "#5b6c8f",
    "class": "#b54d35",
    "function": "#26814d",
    "method": "#50a366",
    "control_block": "#8172b3",
    "statement_window": "#c17f2f",
}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def load_model(model_name: str, device: str):
    dtype = torch.bfloat16 if device.startswith("cuda") and torch.cuda.is_bf16_supported() else torch.float16
    if device == "cpu":
        dtype = torch.float32
    print(f"[layerwise_ast] loading {model_name} dtype={dtype} device={device}", flush=True)
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="eager",
    ).to(device)
    model.eval()
    return model, tok


def forward_prompt(model, tokenizer, variation: dict[str, str], device: str, max_seq_len: int):
    prompt = render_chat(tokenizer, variation["system_prompt"], variation["user_prompt"])
    start, end = token_bounds_for_text(tokenizer, prompt, variation["target_code"])
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_seq_len)
    if end > int(enc["input_ids"].shape[1]):
        raise ValueError(f"target code object truncated: end={end} seq={enc['input_ids'].shape[1]}")
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(**enc, use_cache=True, return_dict=True)
    return out.past_key_values, start, end, int(enc["input_ids"].shape[1])


def choose_span_ids(spans: list[dict[str, Any]], by_span: dict[str, dict[str, dict]], n: int) -> list[str]:
    chosen: list[str] = []
    for gran in ORDER:
        candidates = [
            span for span in spans
            if span["granularity"] == gran and {"planner", "coder"}.issubset(by_span.get(span["span_id"], {}))
        ]
        candidates = sorted(candidates, key=lambda s: (abs(int(s["approx_tokens"]) - 220), s["path"], s["start_line"]))
        chosen.extend([span["span_id"] for span in candidates[:n]])
    return chosen


@torch.no_grad()
def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    spans_list = json.loads(args.spans.read_text(encoding="utf-8"))
    spans = {s["span_id"]: s for s in spans_list}
    variations = json.loads(args.variations.read_text(encoding="utf-8"))
    by_span: dict[str, dict[str, dict]] = defaultdict(dict)
    for var in variations:
        by_span[var["span_id"]][var["agent_role"]] = var
    span_ids = choose_span_ids(spans_list, by_span, args.spans_per_granularity)

    model, tokenizer = load_model(args.model, args.device)
    rope_theta = float(getattr(model.config, "rope_theta", 1000000.0))
    num_layers = int(getattr(model.config, "num_hidden_layers", 1))

    rows: list[dict[str, Any]] = []
    for idx, span_id in enumerate(span_ids, start=1):
        span = spans[span_id]
        role_vars = by_span[span_id]
        try:
            planner_kv, p_start, p_end, p_seq = forward_prompt(
                model, tokenizer, role_vars["planner"], args.device, args.max_seq_len
            )
            coder_kv, c_start, c_end, c_seq = forward_prompt(
                model, tokenizer, role_vars["coder"], args.device, args.max_seq_len
            )
            length = min(p_end - p_start, c_end - c_start)
            true_delta = c_start - p_start
            for layer in range(num_layers):
                p_k, p_v = get_layer_kv(planner_kv, layer)
                c_k, c_v = get_layer_kv(coder_kv, layer)
                p_k = p_k[:, :, p_start:p_start + length, :].detach().to(torch.float32).cpu()
                c_k = c_k[:, :, c_start:c_start + length, :].detach().to(torch.float32).cpu()
                p_v = p_v[:, :, p_start:p_start + length, :].detach().to(torch.float32).cpu()
                c_v = c_v[:, :, c_start:c_start + length, :].detach().to(torch.float32).cpu()
                p_k_rot = apply_rope_delta_neox(p_k, true_delta, rope_theta)
                for variant, candidate_k in (("no_rotation", p_k), ("correct_delta", p_k_rot)):
                    k_diff = candidate_k.float() - c_k.float()
                    v_diff = p_v.float() - c_v.float()
                    rows.append(
                        {
                            "model": args.model,
                            "span_id": span_id,
                            "repo": span["repo"],
                            "path": span["path"],
                            "granularity": span["granularity"],
                            "ast_type": span["ast_type"],
                            "start_line": span["start_line"],
                            "end_line": span["end_line"],
                            "approx_tokens": span["approx_tokens"],
                            "span_tokens": length,
                            "layer": layer,
                            "variant": variant,
                            "true_delta": true_delta,
                            "planner_start": p_start,
                            "coder_start": c_start,
                            "planner_seq_len": p_seq,
                            "coder_seq_len": c_seq,
                            "k_cosine": round(cosine(candidate_k, c_k), 8),
                            "k_l2_norm": round(torch.linalg.vector_norm(k_diff).item() / math.sqrt(k_diff.numel()), 8),
                            "v_cosine": round(cosine(p_v, c_v), 8),
                            "v_l2_norm": round(torch.linalg.vector_norm(v_diff).item() / math.sqrt(v_diff.numel()), 8),
                        }
                    )
            del planner_kv, coder_kv
        except Exception as exc:
            print(f"[layerwise_ast] skip {span_id}: {exc}", flush=True)
        if idx % 5 == 0:
            print(f"[layerwise_ast] processed {idx}/{len(span_ids)} spans rows={len(rows)}", flush=True)
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    return rows


def avg(rows: list[dict[str, str]], gran: str, layer: int, variant: str, field: str) -> float:
    vals = [
        float(r[field]) for r in rows
        if r["granularity"] == gran and int(r["layer"]) == layer and r["variant"] == variant
    ]
    return sum(vals) / len(vals) if vals else 0.0


def plot(rows: list[dict[str, str]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    layers = sorted({int(r["layer"]) for r in rows})
    for field, ylabel, title, filename in [
        ("k_cosine", "K cosine vs fresh coder KV", "Layer-wise RoPE-aligned K cosine by AST granularity", "fig_layerwise_ast_k_cosine.png"),
        ("k_l2_norm", "Normalized K L2 distance", "Layer-wise RoPE-aligned K distance by AST granularity", "fig_layerwise_ast_k_distance.png"),
    ]:
        plt.figure(figsize=(9.2, 5.2))
        for gran in ORDER:
            if not any(r["granularity"] == gran for r in rows):
                continue
            ys = [avg(rows, gran, layer, "correct_delta", field) for layer in layers]
            plt.plot(layers, ys, marker="o", markersize=2.5, linewidth=1.5, label=gran.replace("_", " "), color=COLORS[gran])
        plt.xlabel("Layer")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xticks(layers[:: max(1, len(layers) // 8)])
        plt.grid(alpha=0.25)
        plt.legend(ncol=2, frameon=False)
        plt.tight_layout()
        plt.savefig(out_dir / filename, dpi=180)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--spans", type=Path, default=DATA / "spans.json")
    parser.add_argument("--variations", type=Path, default=DATA / "variations.json")
    parser.add_argument("--out", type=Path, default=DATA / "layerwise_ast_granularity_comparison.csv")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--spans-per-granularity", type=int, default=5)
    args = parser.parse_args()

    rows = run(args)
    write_csv(args.out, rows)
    plot([{k: str(v) for k, v in row.items()} for row in rows], FIG)
    print(f"[layerwise_ast] wrote {args.out}")


if __name__ == "__main__":
    main()
