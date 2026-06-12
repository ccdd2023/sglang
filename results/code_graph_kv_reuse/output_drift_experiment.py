#!/usr/bin/env python3
"""Output-drift diagnostic for code-graph lossy reuse candidates.

This is a precision experiment, not a serving/TTFT experiment. For each sampled
target and agent role, it compares the deterministic JSON output produced from
the minimal exact target span (`ast_function_only`) against outputs produced
from graph-aware bundles (`import_dependency_bundle`, `call_neighborhood_1hop`,
`test_target_bundle`).

The intent is to measure whether graph-aware exact bundles preserve the same
reuse-risk judgment and relevant-symbol reasoning before spending GPU time on
paired SWE pass@1 runs.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import re
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from code_graph_precision_kv_experiment import (
    DEFAULT_MANIFEST,
    group_rows,
    load_jsonl,
    render_chat,
    select_balanced_groups,
    task_family,
)


ROOT = Path("/home/gfy/CodeMAS_Project/sglang-kvflow")
BASE = ROOT / "results" / "code_graph_kv_reuse"
DEFAULT_SELECTED = BASE / "qwen2_5_3b_precision_kv_12targets" / "selected_groups.json"
BUNDLE_TYPES = (
    "ast_function_only",
    "import_dependency_bundle",
    "call_neighborhood_1hop",
    "test_target_bundle",
)


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT),
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def load_model(model_name: str, device: str):
    dtype = torch.bfloat16 if device.startswith("cuda") and torch.cuda.is_bf16_supported() else torch.float16
    if device == "cpu":
        dtype = torch.float32
    print(f"[output_drift] loading {model_name} dtype={dtype} device={device}", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    print(f"[output_drift] loaded in {time.time() - t0:.1f}s", flush=True)
    return model, tok


@torch.no_grad()
def generate_text(model, tokenizer, row: dict, device: str, max_seq_len: int, max_new_tokens: int) -> dict:
    row = dict(row)
    row["user_prompt"] = strict_user_prompt(row)
    prompt = render_chat(tokenizer, row["system_prompt"], row["user_prompt"])
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_seq_len)
    enc = {k: v.to(device) for k, v in enc.items()}
    input_len = int(enc["input_ids"].shape[1])
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen_ids = out[0, input_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return {"text": text, "prompt_tokens": input_len, "output_tokens": int(gen_ids.shape[0])}


def strict_user_prompt(row: dict) -> str:
    return (
        f"## SWE Instance\n{row['instance_id']}\n\n"
        f"## Target\nfile: {row['target_file']}\nsymbol: {row['target_symbol']}\n"
        f"bundle_type: {row['bundle_type']}\n\n"
        "## Exact Code Bundle\n"
        "```python\n"
        f"{row['bundle_text'].rstrip()}\n"
        "\n```\n\n"
        "## Required JSON Output\n"
        "Return ONLY minified JSON with exactly these keys:\n"
        "{\"relevant_symbols\":[\"symbol\"],\"missing_context\":[\"item\"],\"reuse_risk\":\"low|medium|high\"}\n"
        "Rules: relevant_symbols max 5; missing_context max 3; reuse_risk must be one lowercase label."
    )


def extract_json(text: str) -> tuple[dict | None, str]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    candidates = [cleaned]
    match = re.search(r"\{.*\}", cleaned, flags=re.S)
    if match:
        candidates.insert(0, match.group(0))
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
            return obj if isinstance(obj, dict) else None, ""
        except Exception as exc:
            last = str(exc)
    return None, last if "last" in locals() else "no_json_candidate"


def normalize_text(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text.lower())


def token_f1(a: str, b: str) -> float:
    aa, bb = normalize_text(a), normalize_text(b)
    if not aa and not bb:
        return 1.0
    if not aa or not bb:
        return 0.0
    counts: dict[str, int] = defaultdict(int)
    for tok in aa:
        counts[tok] += 1
    overlap = 0
    for tok in bb:
        if counts[tok] > 0:
            overlap += 1
            counts[tok] -= 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(bb)
    recall = overlap / len(aa)
    return 2 * precision * recall / (precision + recall)


def list_field(obj: dict | None, key: str) -> list[str]:
    if not obj:
        return []
    val = obj.get(key, [])
    if isinstance(val, list):
        return [str(x).strip().lower() for x in val if str(x).strip()]
    if isinstance(val, str):
        return [x.strip().lower() for x in re.split(r"[,;\n]", val) if x.strip()]
    return []


def scalar_field(obj: dict | None, key: str) -> str:
    if not obj:
        return ""
    return str(obj.get(key, "")).strip().lower()


def normalize_risk(value: str) -> str:
    value = value.strip().lower()
    if "high" in value:
        return "high"
    if "medium" in value or "moderate" in value:
        return "medium"
    if "low" in value:
        return "low"
    return value


def jaccard(a: list[str], b: list[str]) -> float:
    aa, bb = set(a), set(b)
    if not aa and not bb:
        return 1.0
    if not aa or not bb:
        return 0.0
    return len(aa & bb) / len(aa | bb)


def coverage(needed: list[str], candidate: list[str]) -> float:
    aa, bb = set(needed), set(candidate)
    if not aa:
        return 1.0
    return len(aa & bb) / len(aa)


def classify_failure(row: dict) -> str:
    if not row["candidate_json_valid"]:
        return "format_failure"
    if row["reuse_risk_match"] == 0:
        return "reuse_risk_drift"
    if row["baseline_symbol_coverage"] < 0.8:
        return "wrong_or_missing_symbol"
    if row["missing_context_jaccard"] < 0.5:
        return "missing_context_drift"
    if row["token_f1"] < 0.7:
        return "large_text_drift"
    return "ok"


def stats(vals: list[float]) -> dict:
    if not vals:
        return {"count": 0}
    vals = sorted(vals)
    return {
        "count": len(vals),
        "mean": sum(vals) / len(vals),
        "p50": vals[len(vals) // 2],
        "p90": vals[min(len(vals) - 1, int(len(vals) * 0.9))],
        "min": vals[0],
    }


def aggregate(records: list[dict]) -> dict:
    out = {
        "overall": {
            "pairs": len(records),
            "mean_token_f1": stats([r["token_f1"] for r in records]),
            "json_valid_rate": sum(1 for r in records if r["candidate_json_valid"]) / len(records) if records else 0.0,
            "reuse_risk_match_rate": sum(r["reuse_risk_match"] for r in records) / len(records) if records else 0.0,
            "high_risk_drift_rate": sum(1 for r in records if r["high_risk_drift"]) / len(records) if records else 0.0,
        }
    }
    for axis in ("candidate_bundle_type", "repo", "task_family", "agent_role"):
        buckets: dict[str, list[dict]] = defaultdict(list)
        for rec in records:
            buckets[str(rec[axis])].append(rec)
        out[f"by_{axis}"] = {}
        for key, rows in sorted(buckets.items()):
            out[f"by_{axis}"][key] = {
                "n": len(rows),
                "mean_token_f1": stats([r["token_f1"] for r in rows])["mean"],
                "p50_token_f1": stats([r["token_f1"] for r in rows])["p50"],
                "json_valid_rate": sum(1 for r in rows if r["candidate_json_valid"]) / len(rows),
                "reuse_risk_match_rate": sum(r["reuse_risk_match"] for r in rows) / len(rows),
                "high_risk_drift_rate": sum(1 for r in rows if r["high_risk_drift"]) / len(rows),
            }
    failures = defaultdict(int)
    for rec in records:
        failures[rec["failure_mode"]] += 1
    out["failure_modes"] = dict(sorted(failures.items()))
    return out


def selected_targets_from_file(path: Path) -> set[tuple[str, str, str]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    out = set()
    for row in rows:
        key = row["key"]
        out.add((key[0], key[1], key[2]))
    return out


def choose_rows(args: argparse.Namespace) -> list[dict]:
    rows = load_jsonl(args.manifest)
    if args.selected_groups.exists():
        selected_targets = selected_targets_from_file(args.selected_groups)
        rows = [
            r
            for r in rows
            if (r["instance_id"], r["target_file"], r["target_symbol"]) in selected_targets
            and r["bundle_type"] in set(BUNDLE_TYPES)
            and int(r.get("token_count", 0)) <= args.max_scope_tokens
        ]
        return rows

    grouped = group_rows(rows)
    selected = select_balanced_groups(
        grouped,
        max_targets=args.max_targets,
        bundle_types=set(BUNDLE_TYPES),
        max_scope_tokens=args.max_scope_tokens,
    )
    selected_keys = {key for key, _ in selected}
    return [
        r
        for r in rows
        if (
            r["instance_id"],
            r["target_file"],
            r["target_symbol"],
            r["bundle_type"],
            r["content_signature"],
        )
        in selected_keys
    ]


def write_csv(path: Path, records: list[dict]) -> None:
    if not records:
        return
    fields = [
        "instance_id",
        "repo",
        "task_family",
        "target_file",
        "target_symbol",
        "agent_role",
        "baseline_bundle_type",
        "candidate_bundle_type",
        "scope_tokens",
        "baseline_json_valid",
        "candidate_json_valid",
        "exact_match",
        "token_f1",
        "char_similarity",
        "relevant_symbols_jaccard",
        "baseline_symbol_coverage",
        "missing_context_jaccard",
        "reuse_risk_match",
        "high_risk_drift",
        "failure_mode",
        "baseline_output",
        "candidate_output",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rec in records:
            writer.writerow({k: rec.get(k, "") for k in fields})


def write_report(path: Path, payload: dict) -> None:
    summary = payload["summary"]
    rows = []
    for bundle, stat in summary.get("by_candidate_bundle_type", {}).items():
        rows.append(
            f"| `{bundle}` | {stat['n']} | {stat['mean_token_f1']:.3f} | {stat['json_valid_rate']:.2f} | {stat['reuse_risk_match_rate']:.2f} | {stat['high_risk_drift_rate']:.2f} |"
        )
    failures = "\n".join(f"| `{k}` | {v} |" for k, v in summary.get("failure_modes", {}).items())
    md = f"""# Code Graph Output Drift Diagnostic

## 1. Setup

- Model: `{payload['config']['model']}`
- Baseline: `ast_function_only` output for the same target and role
- Candidates: `import_dependency_bundle`, `call_neighborhood_1hop`, `test_target_bundle`
- Targets: {payload['config']['targets']}
- Pairs: {payload['summary']['overall']['pairs']}
- Generation: deterministic, max_new_tokens={payload['config']['max_new_tokens']}

## 2. By Candidate Bundle

| candidate bundle | n | mean token F1 | JSON valid | reuse-risk match | high-risk drift |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## 3. Failure Modes

| failure mode | n |
|---|---:|
{failures}

## 4. Interpretation

This diagnostic measures whether graph-aware exact bundles change the model's JSON risk judgment relative to the minimal exact target-span baseline. It is not a runtime KV-reuse pass@1 result. Use it to choose which bundle policies deserve paired SWE pass@1 evaluation.
"""
    path.write_text(md, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/gfy/models/Qwen2.5-3B-Instruct")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--selected-groups", type=Path, default=DEFAULT_SELECTED)
    parser.add_argument("--out-dir", type=Path, default=BASE / "qwen2_5_3b_output_drift_12targets")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-targets", type=int, default=12)
    parser.add_argument("--max-scope-tokens", type=int, default=3000)
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = choose_rows(args)
    grouped: dict[tuple[str, str, str, str], dict[str, dict]] = defaultdict(dict)
    for row in rows:
        key = (row["instance_id"], row["target_file"], row["target_symbol"], row["agent_role"])
        grouped[key][row["bundle_type"]] = row

    model, tokenizer = load_model(args.model, args.device)
    outputs: dict[tuple[str, str, str, str, str], dict] = {}
    for idx, row in enumerate(rows, start=1):
        key = (row["instance_id"], row["target_file"], row["target_symbol"], row["agent_role"], row["bundle_type"])
        try:
            outputs[key] = generate_text(model, tokenizer, row, args.device, args.max_seq_len, args.max_new_tokens)
        except Exception as exc:
            outputs[key] = {"text": "", "error": str(exc), "prompt_tokens": 0, "output_tokens": 0}
        if idx % 25 == 0:
            print(f"[output_drift] generated {idx}/{len(rows)}", flush=True)
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

    records = []
    for key, bundles in grouped.items():
        baseline = bundles.get("ast_function_only")
        if not baseline:
            continue
        base_key = (*key, "ast_function_only")
        base_out = outputs.get(base_key, {"text": ""})
        base_obj, base_err = extract_json(base_out.get("text", ""))
        for bundle_type, candidate in bundles.items():
            if bundle_type == "ast_function_only":
                continue
            cand_key = (*key, bundle_type)
            cand_out = outputs.get(cand_key, {"text": ""})
            cand_obj, cand_err = extract_json(cand_out.get("text", ""))
            rec = {
                "instance_id": candidate["instance_id"],
                "repo": candidate["repo"],
                "task_family": task_family(candidate),
                "target_file": candidate["target_file"],
                "target_symbol": candidate["target_symbol"],
                "agent_role": candidate["agent_role"],
                "baseline_bundle_type": "ast_function_only",
                "candidate_bundle_type": bundle_type,
                "scope_tokens": candidate["token_count"],
                "baseline_json_valid": base_obj is not None,
                "candidate_json_valid": cand_obj is not None,
                "exact_match": int(base_out.get("text", "") == cand_out.get("text", "")),
                "token_f1": token_f1(base_out.get("text", ""), cand_out.get("text", "")),
                "char_similarity": difflib.SequenceMatcher(None, base_out.get("text", ""), cand_out.get("text", "")).ratio(),
                "relevant_symbols_jaccard": jaccard(list_field(base_obj, "relevant_symbols"), list_field(cand_obj, "relevant_symbols")),
                "baseline_symbol_coverage": coverage(list_field(base_obj, "relevant_symbols"), list_field(cand_obj, "relevant_symbols")),
                "missing_context_jaccard": jaccard(list_field(base_obj, "missing_context"), list_field(cand_obj, "missing_context")),
                "reuse_risk_match": int(
                    normalize_risk(scalar_field(base_obj, "reuse_risk"))
                    == normalize_risk(scalar_field(cand_obj, "reuse_risk"))
                    and normalize_risk(scalar_field(base_obj, "reuse_risk")) != ""
                ),
                "baseline_output": base_out.get("text", ""),
                "candidate_output": cand_out.get("text", ""),
                "baseline_parse_error": base_err,
                "candidate_parse_error": cand_err,
            }
            rec["high_risk_drift"] = int(
                (not rec["candidate_json_valid"])
                or rec["reuse_risk_match"] == 0
                or rec["token_f1"] < 0.7
                or rec["baseline_symbol_coverage"] < 0.8
            )
            rec["failure_mode"] = classify_failure(rec)
            records.append(rec)

    payload = {
        "config": {
            "model": args.model,
            "manifest": str(args.manifest),
            "selected_groups": str(args.selected_groups),
            "targets": len({(r["instance_id"], r["target_file"], r["target_symbol"]) for r in rows}),
            "rows_generated": len(rows),
            "max_scope_tokens": args.max_scope_tokens,
            "max_seq_len": args.max_seq_len,
            "max_new_tokens": args.max_new_tokens,
            "git_commit": git_commit(),
            "command": " ".join(str(x) for x in [
                "output_drift_experiment.py",
                "--model", args.model,
                "--max-targets", args.max_targets,
                "--max-scope-tokens", args.max_scope_tokens,
            ]),
        },
        "summary": aggregate(records),
        "records": records,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(args.out_dir / "output_drift_table.csv", records)
    write_report(args.out_dir / "OUTPUT_DRIFT_REPORT.md", payload)
    print(json.dumps(payload["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
