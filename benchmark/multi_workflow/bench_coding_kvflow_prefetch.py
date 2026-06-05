#!/usr/bin/env python3
"""Benchmark KVFlow prefix prefetch vs coding-aware codebase prefetch."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import aiohttp
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

PROJECT = Path(__file__).resolve().parents[2]
MAS_SRC = PROJECT.parent / "MAScoder" / "src"
for entry in (str(MAS_SRC), str(PROJECT), str(PROJECT / "python")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from benchmark.multi_workflow.bench_swe_generated_patch_kvcomm import (  # noqa: E402
    DEFAULT_PYTHON,
    CodeSegment,
    build_anchor_fields,
    build_messages,
    extract_cached_tokens,
    extract_lossy_meta,
    extract_text,
    kill_port,
    load_cases as load_patch_cases,
    now_ms,
    post_chat,
    reset_repo_to_base,
    sha1_short,
    wait_ready,
)

DEFAULT_MODEL = "/home/gfy/models/Qwen2.5-7B-Instruct"
DEFAULT_DATASET = PROJECT / "results" / "repo_level_datasets" / "swe_verified_10_instances.json"
DEFAULT_MANIFEST = PROJECT / "results" / "repo_level_datasets" / "manifest_10.json"
OUT_DIR = PROJECT / "results" / "coding_kvflow_prefetch"

MODES = [
    "baseline_prefix_cache_only",
    "kvflow_prefix_only",
    "kvflow_prefix_plus_codebase_prefetch",
    "kvcomm_lossy_plus_codebase_prefetch",
]


def launch_server(args: argparse.Namespace) -> subprocess.Popen:
    env = dict(**os.environ)
    env["PYTHONPATH"] = str(PROJECT / "python")
    env["SGLANG_LOSSY_FUZZY_MATCH"] = "1"
    if args.hicache_storage_backend:
        env["SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR"] = str(args.out_dir / "hicache_file_storage")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / "sglang_server.log"
    cmd = [
        args.python,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model,
        "--port",
        str(args.port),
        "--tp-size",
        "1",
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--max-total-tokens",
        str(args.max_total_tokens),
        "--chunked-prefill-size",
        "8192",
        "--max-prefill-tokens",
        "16384",
        "--enable-cache-report",
        "--disable-cuda-graph",
        "--log-level",
        "error",
    ]
    if not args.disable_hierarchical_cache:
        cmd.extend(
            [
                "--radix-eviction-policy",
                "priority",
                "--enable-hierarchical-cache",
                "--hicache-ratio",
                str(args.hicache_ratio),
                "--hicache-write-policy",
                "write_back",
                "--enable-hicache-prefetch",
            ]
        )
    if args.hicache_storage_backend:
        cmd.extend(
            [
                "--hicache-storage-backend",
                args.hicache_storage_backend,
                "--hicache-storage-prefetch-policy",
                "best_effort",
            ]
        )
    return subprocess.Popen(
        cmd,
        cwd=str(PROJECT),
        env=env,
        stdout=open(log_path, "w"),
        stderr=subprocess.STDOUT,
    )


def make_codebase_hints(segments: list[CodeSegment], target_agent: str = "implementer") -> list[dict[str, Any]]:
    hints = []
    for idx, segment in enumerate(segments, 1):
        hints.append(
            {
                "code_base_id": f"code_base{idx}:{segment.name}",
                "content_signature": segment.signature,
                "target_agent": target_agent,
                "steps_to_use": 1,
                "priority": 1,
                "match_required": "exact_code_content_signature",
                "text": segment.text,
            }
        )
    return hints


def make_payload(
    args: argparse.Namespace,
    tokenizer: Any,
    messages: list[dict[str, str]],
    segments: list[CodeSegment],
    mode: str,
    salt: str,
) -> dict[str, Any]:
    include_anchor = mode == "kvcomm_lossy_plus_codebase_prefetch"
    payload: dict[str, Any] = {
        "model": args.model,
        "messages": messages,
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "return_cached_tokens_details": True,
        "reuse_mode": "lossy" if include_anchor else "lossless",
        "lossy_alignment_method": "kvcomm",
        "template_task_family": "coding_mas_kvflow_prefetch",
        "cache_salt": salt,
        "priority": 1,
    }
    if mode in {
        "kvflow_prefix_only",
        "kvflow_prefix_plus_codebase_prefetch",
        "kvcomm_lossy_plus_codebase_prefetch",
    }:
        payload["next_agent_prefix"] = "You are the implementer. Reuse the planner code context."
    if mode in {
        "kvflow_prefix_plus_codebase_prefetch",
        "kvcomm_lossy_plus_codebase_prefetch",
    }:
        payload["codebase_prefetch_hints"] = make_codebase_hints(segments)
    if include_anchor:
        payload.update(build_anchor_fields(tokenizer, messages, segments))
    return payload


def build_codebase_warmup_messages(segment: CodeSegment) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "Cache this exact code base for later coding agents."},
        {
            "role": "user",
            "content": f"## code_base: {segment.name}\n```python\n{segment.text}\n```\nReturn OK.",
        },
    ]


def token_f1(a: str, b: str) -> float:
    aa = a.split()
    bb = b.split()
    if not aa and not bb:
        return 1.0
    if not aa or not bb:
        return 0.0
    from collections import Counter

    ca, cb = Counter(aa), Counter(bb)
    overlap = sum((ca & cb).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(aa)
    recall = overlap / len(bb)
    return 2 * precision * recall / (precision + recall)


def load_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Load serving-only cases from the repo-level manifest.

    The generated-patch benchmark loader intentionally filters through local
    SWE-bench repos and target-test metadata. This serving experiment only
    needs stable code segments, so it should use the manifest-local codebase
    snapshots directly and cover all requested manifest samples.
    """

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = json.loads(args.dataset.read_text(encoding="utf-8"))
    row_by_id = {row["instance_id"]: row for row in rows}
    samples = manifest.get("samples") or []
    if not samples:
        return load_patch_cases(args)

    cases = []
    selected = samples[args.start_index : args.start_index + args.max_cases]
    for sample in selected:
        instance_id = sample["instance_id"]
        instance = {**sample, **row_by_id.get(instance_id, {})}
        instance.setdefault("instance_id", instance_id)
        instance.setdefault("repo", sample.get("repo", ""))
        segments = []
        for file_info in sample.get("files", [])[: args.files_per_case]:
            local_path = Path(file_info.get("local_path", ""))
            if not local_path.exists():
                continue
            text = local_path.read_text(encoding="utf-8", errors="replace")
            if args.max_file_chars and len(text) > args.max_file_chars:
                text = text[: args.max_file_chars]
            segments.append(CodeSegment(file_info["path"], text.rstrip()))
        if segments:
            cases.append({"instance": instance, "segments": segments, "target_paths": []})
    return cases


async def warm_codebase(
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    tokenizer: Any,
    instance_id: str,
    segments: list[CodeSegment],
) -> None:
    for idx, segment in enumerate(segments):
        messages = build_codebase_warmup_messages(segment)
        payload = {
            "model": args.model,
            "messages": messages,
            "max_tokens": 8,
            "temperature": 0.0,
            "top_p": 1.0,
            "return_cached_tokens_details": True,
            "reuse_mode": "lossless",
            "lossy_alignment_method": "kvcomm",
            "priority": 1,
        }
        payload.update(build_anchor_fields(tokenizer, messages, [segment]))
        await post_chat(session, args.port, payload)


async def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    args.out_dir = args.out_dir if args.out_dir.is_absolute() else PROJECT / args.out_dir
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    cases = load_cases(args)
    kill_port(args.port)
    await asyncio.sleep(1)
    proc = launch_server(args)
    results: list[dict[str, Any]] = []
    failed_reason = ""
    try:
        if not await wait_ready(args.port, timeout_s=args.server_timeout):
            raise RuntimeError(f"server did not become ready; see {args.out_dir / 'sglang_server.log'}")
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=900)) as session:
            for case in cases:
                instance = case["instance"]
                instance_id = instance["instance_id"]
                segments = case["segments"]
                reset_repo_to_base(instance, PROJECT / "results" / "swebench_local_envs" / "repos" / instance_id)
                await warm_codebase(session, args, tokenizer, instance_id, segments)

                planner_messages = build_messages(
                    instance,
                    segments,
                    "planner warmup; cache code-base anchors",
                    "json-edit",
                )
                planner_payload = make_payload(
                    args,
                    tokenizer,
                    planner_messages,
                    segments,
                    "kvcomm_lossy_plus_codebase_prefetch",
                    f"planner:{instance_id}",
                )
                planner_payload["max_tokens"] = 32
                await post_chat(session, args.port, planner_payload)

                mode_rows = []
                outputs: dict[str, str] = {}
                for mode in MODES:
                    messages = build_messages(instance, segments, mode, "json-edit")
                    payload = make_payload(args, tokenizer, messages, segments, mode, f"target:{instance_id}:{mode}")
                    start = now_ms()
                    response = await post_chat(session, args.port, payload)
                    elapsed_ms = response["elapsed_ms"]
                    output = extract_text(response["body"]) or ""
                    outputs[mode] = output
                    meta = extract_lossy_meta(response["body"])
                    mode_rows.append(
                        {
                            "mode": mode,
                            "elapsed_ms": round(elapsed_ms, 2),
                            "cached_tokens": extract_cached_tokens(response["body"]),
                            "output_chars": len(output),
                            "lossy_match_reason": meta.get("lossy_first_match_reason")
                            or meta.get("lossy_final_match_reason")
                            or meta.get("lossy_anchor_match_used"),
                            "matched_content_signature": meta.get("lossy_first_matched_content_signature")
                            or meta.get("lossy_final_matched_content_signature")
                            or meta.get("lossy_anchor_match_content_signature"),
                            "codebase_prefetch_hint_count": meta.get("codebase_prefetch_hint_count", 0),
                            "codebase_prefetch_text_count": meta.get("codebase_prefetch_text_count", 0),
                            "codebase_prefetch_queued_tokens": meta.get("codebase_prefetch_queued_tokens", 0),
                            "codebase_prefetch_matched_tokens": meta.get("codebase_prefetch_matched_tokens", 0),
                            "codebase_prefetch_success_count": meta.get("codebase_prefetch_success_count", 0),
                            "codebase_prefetch_device_hit_count": meta.get("codebase_prefetch_device_hit_count", 0),
                            "raw_metadata": meta,
                            "request_start_ms": round(start, 2),
                        }
                    )

                baseline_output = outputs.get("baseline_prefix_cache_only", "")
                for row in mode_rows:
                    output = outputs.get(row["mode"], "")
                    row["output_exact_match_vs_baseline"] = output == baseline_output
                    row["output_token_f1_vs_baseline"] = round(token_f1(output, baseline_output), 4)

                results.append(
                    {
                        "instance_id": instance_id,
                        "repo": instance["repo"],
                        "segments": [
                            {
                                "name": segment.name,
                                "chars": len(segment.text),
                                "signature": segment.signature,
                            }
                            for segment in segments
                        ],
                        "modes": mode_rows,
                    }
                )
                print(f"[case] {instance_id} done")
    except Exception as exc:
        failed_reason = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
        kill_port(args.port)
        if results:
            summary = {
                "model": args.model,
                "dataset": str(args.dataset),
                "manifest": str(args.manifest),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "hicache_storage_backend": args.hicache_storage_backend or "disabled",
                "hierarchical_cache": not args.disable_hierarchical_cache,
                "modes": MODES,
                "results": results,
                "failed_reason": failed_reason,
            }
            write_artifacts(args.out_dir, summary)

    summary = {
        "model": args.model,
        "dataset": str(args.dataset),
        "manifest": str(args.manifest),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hicache_storage_backend": args.hicache_storage_backend or "disabled",
        "hierarchical_cache": not args.disable_hierarchical_cache,
        "modes": MODES,
        "results": results,
    }
    write_artifacts(args.out_dir, summary)
    return summary


def summarize_mode(summary: dict[str, Any]) -> dict[str, dict[str, float]]:
    by_mode: dict[str, list[dict[str, Any]]] = {mode: [] for mode in MODES}
    for case in summary["results"]:
        for row in case["modes"]:
            by_mode[row["mode"]].append(row)
    stats = {}
    for mode, rows in by_mode.items():
        if not rows:
            continue
        stats[mode] = {
            "n": len(rows),
            "avg_latency_ms": statistics.mean(float(r["elapsed_ms"]) for r in rows),
            "median_latency_ms": statistics.median(float(r["elapsed_ms"]) for r in rows),
            "avg_cached_tokens": statistics.mean(float(r["cached_tokens"]) for r in rows),
            "avg_prefetch_queued_tokens": statistics.mean(float(r["codebase_prefetch_queued_tokens"]) for r in rows),
            "avg_prefetch_matched_tokens": statistics.mean(float(r["codebase_prefetch_matched_tokens"]) for r in rows),
            "avg_prefetch_hints": statistics.mean(float(r["codebase_prefetch_hint_count"]) for r in rows),
            "prefetch_success_rate": statistics.mean(
                1.0 if int(r["codebase_prefetch_success_count"] or 0) > 0 else 0.0
                for r in rows
            ),
            "prefetch_device_hit_rate": statistics.mean(
                1.0 if int(r["codebase_prefetch_device_hit_count"] or 0) > 0 else 0.0
                for r in rows
            ),
            "exact_content_hit_rate": statistics.mean(
                1.0 if r.get("lossy_match_reason") == "exact_code_content_signature" else 0.0
                for r in rows
            ),
            "avg_token_f1_vs_baseline": statistics.mean(float(r["output_token_f1_vs_baseline"]) for r in rows),
        }
    return stats


def write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = summarize_mode(summary)
    summary["mode_summary"] = stats
    summary_text = json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    (out_dir / "summary.json").write_text(summary_text, encoding="utf-8")
    (out_dir / "prefetch_summary.json").write_text(summary_text, encoding="utf-8")

    rows = []
    for case in summary["results"]:
        for row in case["modes"]:
            rows.append(
                {
                    "instance_id": case["instance_id"],
                    "repo": case["repo"],
                    **{k: v for k, v in row.items() if k != "raw_metadata"},
                }
            )
    with (out_dir / "prefetch_table.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["instance_id"])
        writer.writeheader()
        writer.writerows(rows)

    labels = list(stats)
    if labels:
        plt.figure(figsize=(9, 4.5))
        plt.bar(labels, [stats[m]["avg_latency_ms"] for m in labels])
        plt.xticks(rotation=20, ha="right")
        plt.ylabel("Avg latency (ms)")
        plt.tight_layout()
        plt.savefig(out_dir / "fig_latency.png", dpi=180)
        plt.close()

        plt.figure(figsize=(9, 4.5))
        plt.bar(labels, [stats[m]["avg_cached_tokens"] for m in labels])
        plt.xticks(rotation=20, ha="right")
        plt.ylabel("Avg cached tokens")
        plt.tight_layout()
        plt.savefig(out_dir / "fig_cached_tokens.png", dpi=180)
        plt.close()

        plt.figure(figsize=(9, 4.5))
        x = range(len(labels))
        plt.bar(x, [stats[m]["avg_prefetch_queued_tokens"] for m in labels], label="queued")
        plt.bar(x, [stats[m]["avg_prefetch_matched_tokens"] for m in labels], bottom=[stats[m]["avg_prefetch_queued_tokens"] for m in labels], label="matched")
        plt.xticks(list(x), labels, rotation=20, ha="right")
        plt.ylabel("Avg prefetch tokens")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "fig_prefetch_tokens.png", dpi=180)
        plt.close()

    report_lines = [
        "# Coding KVFlow Prefetch Report",
        "",
        "## Summary",
        "",
        f"- Model: `{summary['model']}`",
        f"- Dataset: `{summary['dataset']}`",
        f"- Cases: {len(summary['results'])}",
        f"- HiCache storage backend: `{summary.get('hicache_storage_backend', 'disabled')}`",
        f"- Hierarchical cache: `{summary.get('hierarchical_cache', True)}`",
        "- Safety rule: codebase prefetch may predict future code blocks, but KVCOMM reuse still requires `exact_code_content_signature`.",
        "",
        "## Main Table",
        "",
        "| mode | cases | avg latency ms | avg cached tokens | avg hints | avg prefetch queued | prefetch success | exact-content hit | avg token F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in labels:
        item = stats[mode]
        report_lines.append(
            f"| {mode} | {int(item['n'])} | {item['avg_latency_ms']:.1f} | "
            f"{item['avg_cached_tokens']:.1f} | {item['avg_prefetch_hints']:.1f} | "
            f"{item['avg_prefetch_queued_tokens']:.1f} | "
            f"{item['prefetch_success_rate']:.2f} | {item['exact_content_hit_rate']:.2f} | "
            f"{item['avg_token_f1_vs_baseline']:.4f} |"
        )
    report_lines.extend(
        [
            "",
            "## Figures",
            "",
            f"![Latency]({(out_dir / 'fig_latency.png').resolve()})",
            "",
            f"![Cached tokens]({(out_dir / 'fig_cached_tokens.png').resolve()})",
            "",
            f"![Prefetch tokens]({(out_dir / 'fig_prefetch_tokens.png').resolve()})",
            "",
            "## Per-Case Table",
            "",
            "| instance_id | mode | latency ms | cached | prefetch queued | match reason | token F1 |",
            "|---|---|---:|---:|---:|---|---:|",
        ]
    )
    for row in rows:
        report_lines.append(
            f"| {row['instance_id']} | {row['mode']} | {row['elapsed_ms']} | "
            f"{row['cached_tokens']} | {row['codebase_prefetch_queued_tokens']} | "
            f"{row.get('lossy_match_reason') or ''} | {row['output_token_f1_vs_baseline']} |"
        )
    report_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This benchmark isolates serving-side KVFlow behavior. It does not replace the SWE-bench pass@1 table; use it to show whether coding-aware prefetch improves cached-token/latency behavior before running expensive candidate tests.",
            "",
            "When `hicache_storage_backend` is disabled, `codebase_prefetch_hints` still verifies template-to-engine guidance and exact-content KVCOMM hits, but host load-back counters remain zero. Enable `--hicache-storage-backend file` only for storage-specific debugging; the current local file backend can trip SGLang's runtime memory checker on long coding prompts.",
        ]
    )
    (out_dir / "PREFETCH_REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--max-cases", type=int, default=10)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--files-per-case", type=int, default=2)
    parser.add_argument("--max-file-chars", type=int, default=22000)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--max-total-tokens", type=int, default=65536)
    parser.add_argument("--mem-fraction-static", type=float, default=0.78)
    parser.add_argument("--hicache-ratio", type=float, default=1.5)
    parser.add_argument("--hicache-storage-backend", default="")
    parser.add_argument("--disable-hierarchical-cache", action="store_true")
    parser.add_argument("--server-timeout", type=int, default=180)
    parser.add_argument("--eval-timeout", type=int, default=1200)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run_benchmark(parse_args()))
