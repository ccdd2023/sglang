#!/usr/bin/env python3
"""Run code-graph bundle precision diagnostics with HF KV captures.

The experiment compares planner/coder/reviewer prompts for the same exact
code-graph bundle. It is precision-first: token counts are retained only as
scope covariates, while the main measurements are cross-role KV distance and
tail-risk by code task family and bundle type.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path("/home/gfy/CodeMAS_Project/sglang-kvflow")
BASE = ROOT / "results" / "code_graph_kv_reuse"
DATA = BASE / "data"
DEFAULT_MANIFEST = DATA / "code_graph_precision_manifest.jsonl"
SELECTED_LAYERS = (-1, -2, -3, -4)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def group_rows(rows: list[dict]) -> dict[tuple, dict[str, dict]]:
    grouped: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        key = (
            row["instance_id"],
            row["target_file"],
            row["target_symbol"],
            row["bundle_type"],
            row["content_signature"],
        )
        grouped[key][row["agent_role"]] = row
    return grouped


def task_family(row: dict) -> str:
    path = row["target_file"].lower()
    repo = row["repo"].lower()
    symbol = row["target_symbol"].lower()
    if "test" in path or row["bundle_type"] == "test_target_bundle":
        return "test_aligned"
    if "validator" in path or "validate" in symbol:
        return "validation"
    if "model" in path or "query" in path or "field" in path:
        return "model_or_query_logic"
    if "plot" in path or "axes" in path or "figure" in path or "matplotlib" in repo:
        return "plotting_rendering"
    if "io" in path or "fits" in path or "ascii" in path or "file" in path:
        return "io_parsing"
    if "request" in repo or "flask" in repo or "response" in path or "http" in path:
        return "web_http"
    return "general_library_logic"


def select_balanced_groups(
    grouped: dict[tuple, dict[str, dict]],
    *,
    max_targets: int,
    bundle_types: set[str],
    max_scope_tokens: int,
) -> list[tuple[tuple, dict[str, dict]]]:
    target_seen: set[tuple[str, str, str]] = set()
    selected_targets: list[tuple[str, str, str]] = []
    by_repo: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for key, roles in grouped.items():
        if not {"planner", "coder", "reviewer"}.issubset(roles):
            continue
        sample = roles["planner"]
        if sample["bundle_type"] not in bundle_types:
            continue
        if int(sample.get("token_count", 0)) > max_scope_tokens:
            continue
        target = (sample["instance_id"], sample["target_file"], sample["target_symbol"])
        if target in target_seen:
            continue
        target_seen.add(target)
        by_repo[sample["repo"]].append(target)

    # Round-robin by repo to cover varied code tasks.
    repos = sorted(by_repo, key=lambda r: (-len(by_repo[r]), r))
    while len(selected_targets) < max_targets:
        progressed = False
        for repo in repos:
            if by_repo[repo]:
                selected_targets.append(by_repo[repo].pop(0))
                progressed = True
                if len(selected_targets) >= max_targets:
                    break
        if not progressed:
            break
    selected_target_set = set(selected_targets)

    selected = []
    for key, roles in grouped.items():
        sample = roles.get("planner")
        if not sample or sample["bundle_type"] not in bundle_types:
            continue
        target = (sample["instance_id"], sample["target_file"], sample["target_symbol"])
        if target in selected_target_set and int(sample.get("token_count", 0)) <= max_scope_tokens:
            selected.append((key, roles))
    return selected


def load_model(model_name: str, device: str):
    dtype = torch.bfloat16 if device.startswith("cuda") and torch.cuda.is_bf16_supported() else torch.float16
    if device == "cpu":
        dtype = torch.float32
    print(f"[code_graph_precision] loading {model_name} dtype={dtype} device={device}", flush=True)
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
    print(f"[code_graph_precision] loaded in {time.time() - t0:.1f}s", flush=True)
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
        raise ValueError("bundle text not found in rendered prompt")
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
def capture_bundle_kv(model, tokenizer, row: dict, device: str, max_seq_len: int) -> dict:
    prompt = render_chat(tokenizer, row["system_prompt"], row["user_prompt"])
    start, end = token_bounds_for_text(tokenizer, prompt, row["bundle_text"])
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_seq_len)
    seq_len = int(enc["input_ids"].shape[1])
    if end > seq_len:
        raise ValueError(f"bundle truncated: end={end} seq={seq_len}")
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
        "seq_len": seq_len,
    }


def kv_distance(a: dict, b: dict) -> dict:
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
        "p50": vals[len(vals) // 2],
        "p90": vals[min(len(vals) - 1, int(len(vals) * 0.9))],
        "max": vals[-1],
        "tail_rate_050": sum(1 for x in vals if x > 0.5) / len(vals),
        "tail_rate_075": sum(1 for x in vals if x > 0.75) / len(vals),
    }


def aggregate(records: list[dict]) -> dict:
    out = {"overall": stats([r["d_norm"] for r in records])}
    for axis in ("bundle_type", "repo", "task_family", "agent_role", "precision_priority"):
        buckets: dict[str, list[float]] = defaultdict(list)
        for rec in records:
            buckets[str(rec[axis])].append(rec["d_norm"])
        out[f"by_{axis}"] = {k: stats(v) for k, v in sorted(buckets.items())}
    worst = sorted(records, key=lambda r: r["d_norm"], reverse=True)[:12]
    out["worst_cases"] = [
        {
            "instance_id": r["instance_id"],
            "repo": r["repo"],
            "task_family": r["task_family"],
            "bundle_type": r["bundle_type"],
            "agent_role": r["agent_role"],
            "target_file": r["target_file"],
            "target_symbol": r["target_symbol"],
            "d_norm": r["d_norm"],
            "scope_tokens": r["scope_tokens"],
        }
        for r in worst
    ]
    return out


def write_csv(records: list[dict], path: Path) -> None:
    if not records:
        return
    fieldnames = [
        "instance_id",
        "repo",
        "task_family",
        "target_file",
        "target_symbol",
        "bundle_type",
        "agent_role",
        "precision_priority",
        "content_signature",
        "scope_tokens",
        "span_tokens",
        "seq_len",
        "target_start",
        "d_key",
        "d_value",
        "d_norm",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow({k: rec.get(k, "") for k in fieldnames})


def write_report(payload: dict, out_dir: Path) -> None:
    summary = payload["summary"]
    rows = []
    for bundle, stat in summary["by_bundle_type"].items():
        rows.append(
            f"| `{bundle}` | {stat['count']} | {stat['mean']:.3f} | {stat['p50']:.3f} | {stat['p90']:.3f} | {stat['tail_rate_050']:.2f} |"
        )
    task_rows = []
    for task, stat in summary["by_task_family"].items():
        task_rows.append(
            f"| `{task}` | {stat['count']} | {stat['mean']:.3f} | {stat['p90']:.3f} | {stat['tail_rate_050']:.2f} |"
        )
    worst_rows = []
    for row in summary["worst_cases"][:8]:
        worst_rows.append(
            f"| `{row['instance_id']}` | `{row['task_family']}` | `{row['bundle_type']}` | `{row['agent_role']}` | {row['d_norm']:.3f} | `{row['target_symbol']}` |"
        )
    md = f"""# Code Graph Lossy-Reuse Precision KV Diagnostic

## 1. What Was Run

- Model: `{payload['config']['model']}`
- Selected layers: `{payload['config']['selected_layers']}`
- Sampled exact bundle groups: {payload['config']['sampled_groups']}
- Records: {payload['config']['n_records']} = sampled groups × coder/reviewer comparisons
- Canonical comparison: same exact bundle under `coder`/`reviewer` prompt vs `planner` prompt
- Scope tokens are recorded only as covariates, not as the optimization target.

## 2. By Bundle Type

| bundle | n | mean d_norm | p50 | p90 | tail d_norm>0.5 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## 3. By Code Task Family

| task family | n | mean d_norm | p90 | tail d_norm>0.5 |
|---|---:|---:|---:|---:|
{chr(10).join(task_rows)}

## 4. Worst Precision-Risk Cases

| case | task | bundle | role | d_norm | symbol |
|---|---|---|---|---:|---|
{chr(10).join(worst_rows)}

## 5. Interpretation Boundary

This is a KV precision diagnostic, not a TTFT or pass@1 result. A lower cross-role `d_norm` suggests the exact code bundle is more stable under lossy reuse across agent prompts. The next confirmation step is output drift and paired pass@1 non-degradation on the same sampled groups.
"""
    (out_dir / "CODE_GRAPH_PRECISION_KV_REPORT.md").write_text(md, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/gfy/models/Qwen2.5-3B-Instruct")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=BASE / "qwen2_5_3b_precision_kv")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--max-targets", type=int, default=12)
    parser.add_argument("--max-scope-tokens", type=int, default=6000)
    parser.add_argument(
        "--bundle-types",
        default="ast_function_only,import_dependency_bundle,call_neighborhood_1hop,test_target_bundle",
    )
    args = parser.parse_args()

    rows = load_jsonl(args.manifest)
    grouped = group_rows(rows)
    bundle_types = {x.strip() for x in args.bundle_types.split(",") if x.strip()}
    selected = select_balanced_groups(
        grouped,
        max_targets=args.max_targets,
        bundle_types=bundle_types,
        max_scope_tokens=args.max_scope_tokens,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "selected_groups.json").write_text(
        json.dumps(
            [
                {
                    "key": list(key),
                    "repo": roles["planner"]["repo"],
                    "task_family": task_family(roles["planner"]),
                    "scope_tokens": roles["planner"]["token_count"],
                }
                for key, roles in selected
            ],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    model, tokenizer = load_model(args.model, args.device)
    records = []
    failures = []
    for idx, (key, roles) in enumerate(selected, start=1):
        sample = roles["planner"]
        try:
            planner_kv = capture_bundle_kv(model, tokenizer, sample, args.device, args.max_seq_len)
            for role in ("coder", "reviewer"):
                role_kv = capture_bundle_kv(model, tokenizer, roles[role], args.device, args.max_seq_len)
                dist = kv_distance(role_kv, planner_kv)
                records.append(
                    {
                        "instance_id": sample["instance_id"],
                        "repo": sample["repo"],
                        "task_family": task_family(sample),
                        "target_file": sample["target_file"],
                        "target_symbol": sample["target_symbol"],
                        "bundle_type": sample["bundle_type"],
                        "agent_role": role,
                        "precision_priority": sample["precision_priority"],
                        "content_signature": sample["content_signature"],
                        "scope_tokens": sample["token_count"],
                        "seq_len": role_kv["seq_len"],
                        "target_start": role_kv["start"],
                        **dist,
                    }
                )
                del role_kv
            del planner_kv
        except Exception as exc:
            failures.append({"key": list(key), "error": str(exc)})
            print(f"[code_graph_precision] skip {key}: {exc}", flush=True)
        if idx % 10 == 0:
            print(f"[code_graph_precision] processed {idx}/{len(selected)} groups; records={len(records)}", flush=True)
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    payload = {
        "config": {
            "model": args.model,
            "selected_layers": SELECTED_LAYERS,
            "manifest": str(args.manifest),
            "sampled_groups": len(selected),
            "n_records": len(records),
            "max_seq_len": args.max_seq_len,
            "max_targets": args.max_targets,
            "max_scope_tokens": args.max_scope_tokens,
            "bundle_types": sorted(bundle_types),
        },
        "summary": aggregate(records),
        "records": records,
        "failures": failures,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(records, args.out_dir / "precision_kv_table.csv")
    write_report(payload, args.out_dir)
    print(json.dumps(payload["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
