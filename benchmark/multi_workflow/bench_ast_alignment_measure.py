"""AST-alignment partial-match hit-rate measurement.

Runs the existing placeholder k-NN pipeline against a small workload and
captures structured `[AST_ALIGN]` log lines emitted by the modified
`radix_cache.py`. Outputs:

  - rows_ast_alignment.csv : one row per match (slot_id, cos sim, slot
    text sha1 + first 40 chars, match text sha1 + first 40 chars)
  - sglang_server.log      : full server log (incl. AST_ALIGN lines)

This is a measurement-only driver: no new cache mode, no new prompt
format. The instrumentation added in `radix_cache.py` (gated by
`SGLANG_AST_ALIGNMENT_LOG=1`) does the heavy lifting.

Usage:

    PY=/home/gfy/.conda/envs/sglang-kvflow/bin/python
    SGLANG_AST_ALIGNMENT_LOG=1 $PY benchmark/multi_workflow/bench_ast_alignment_measure.py \\
        --manifest results/repo_level_datasets/manifest_500.json \\
        --out-dir results/ast_alignment_measurement_20260626/ \\
        --max-cases 5 --agent-count 5 --mode placeholder_knn_reuse

Post-process:

    $PY benchmark/multi_workflow/aggregate_ast_alignment.py \\
        --in-dir results/ast_alignment_measurement_20260626/ \\
        --out results/ast_alignment_measurement_20260626/REPORT.md
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT.parent / "MAScoder" / "src"))

from benchmark.multi_workflow.bench_kvcomm_ttft_stress import (  # noqa: E402
    AGENT_ROLES,
    CodeSegment,
    build_anchor_fields,
    build_placeholder_anchor_fields,
    build_slot_messages,
    build_stress_messages,
    launch_server,
    make_payload,
    post_chat_stream,
    row_from_response,
    wait_ready,
)


# ---------------------------------------------------------------------------
# Manifest loader — reads manifest_500.json (used by the 60-case stratified
# sweep) and emits the same `case_id / repo / segments` shape the rest of
# the pipeline expects.
# ---------------------------------------------------------------------------


def load_manifest_cases(manifest_path: Path, segment_count: int, max_file_chars: int,
                        max_cases: int) -> list[dict[str, Any]]:
    """Load cases from `manifest_500.json` (summary manifest).

    The summary manifest only stores a count of files per sample, with the
    actual file paths living under `sample["sample_dir"]`. We walk the
    sample_dir to find `.py` files and pick the largest ones (matching
    the load_long_cases ordering).
    """
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidates = []
    for sample in manifest.get("samples", []):
        sample_dir = Path(sample.get("sample_dir", ""))
        if not sample_dir.is_dir():
            continue
        py_files = sorted(sample_dir.rglob("*.py"), key=lambda p: p.stat().st_size, reverse=True)
        if len(py_files) < segment_count:
            continue
        selected = py_files[:segment_count]
        score = sum(p.stat().st_size for p in selected)
        candidates.append((score, sample, selected))
    candidates.sort(reverse=True, key=lambda item: item[0])
    cases = []
    for _, sample, files in candidates[:max_cases]:
        segments = []
        for path in files:
            text = path.read_text(encoding="utf-8", errors="replace").rstrip()
            if max_file_chars and len(text) > max_file_chars:
                text = text[:max_file_chars].rstrip()
            if text:
                rel = str(path.relative_to(Path(sample.get("sample_dir", ".")))) if sample.get("sample_dir") else path.name
                segments.append(CodeSegment(rel, text))
        if len(segments) >= segment_count:
            cases.append(
                {
                    "case_id": sample["instance_id"],
                    "repo": sample.get("repo", sample.get("repo_key", "")),
                    "segments": segments[:segment_count],
                }
            )
    return cases


# ---------------------------------------------------------------------------
# Log parser — extracts the structured [AST_ALIGN] lines that radix_cache.py
# emits when SGLANG_AST_ALIGNMENT_LOG=1 is set. The lines look like:
#
#   [AST_ALIGN] rid=42 slot_id=code_base1 cos=0.9914 slot_start=54 slot_end=287
#     match_start=54 match_end=287 slot_chars=143 match_chars=143
#     slot_sha1=abc123def456 match_sha1=abc123def456
#     slot_first="def cut(..." match_first="def cut(..."
#
# (real lines are single-line; wrapped here for readability)
# ---------------------------------------------------------------------------


_LOG_LINE_RE = re.compile(
    r"\[AST_ALIGN\]\s+"
    r"rid=(?P<rid>\S+)\s+"
    r"slot_id=(?P<slot_id>\S+)\s+"
    r"cos=(?P<cos>[\d.]+)\s+"
    r"slot_start=(?P<slot_start>-?\d+)\s+"
    r"slot_end=(?P<slot_end>-?\d+)\s+"
    r"match_start=(?P<match_start>-?\d+)\s+"
    r"match_end=(?P<match_end>-?\d+)\s+"
    r"slot_chars=(?P<slot_chars>-?\d+)\s+"
    r"match_chars=(?P<match_chars>-?\d+)\s+"
    r"slot_sha1=(?P<slot_sha1>\S*)\s+"
    r"match_sha1=(?P<match_sha1>\S*)\s+"
    r"slot_first=(?P<slot_first>.*?)\s+"
    r"match_first=(?P<match_first>.*?)\s*$"
)


def parse_log_file(log_path: Path) -> list[dict[str, Any]]:
    """Read a sglang server log and extract every `[AST_ALIGN]` line."""
    if not log_path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = _LOG_LINE_RE.search(line)
            if not m:
                continue
            rows.append({k: v for k, v in m.groupdict().items()})
    return rows


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------


@dataclass
class _ServerHandle:
    proc: subprocess.Popen
    port: int


def launch_with_ast_alignment_log(args: argparse.Namespace, out_dir: Path) -> _ServerHandle:
    """Launch sglang with all placeholder env vars set.

    Critical env vars (must be set BEFORE launch_server spawns the subprocess):
      - SGLANG_AST_ALIGNMENT_LOG=1     (gates the structured log emit)
      - SGLANG_PLACEHOLDER_KNN_MATCH=1 (turns on the placeholder k-NN body)
      - SGLANG_PLACEHOLDER_KNN_TOPK=5
      - SGLANG_PLACEHOLDER_KNN_MIN_COSINE=0.85
      - SGLANG_PLACEHOLDER_KNN_MAX_OVERLAP_RATIO=1.0
      - SGLANG_PLACEHOLDER_KNN_MAX_NEW_TOKEN_RATIO=1.0
      - SGLANG_PLACEHOLDER_KNN_MAX_SPAN_OVERLAP_RATIO=1.0
      - SGLANG_PLACEHOLDER_KNN_MIN_NEW_TOKENS=0
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ["SGLANG_AST_ALIGNMENT_LOG"] = "1"
    os.environ["SGLANG_PLACEHOLDER_KNN_MATCH"] = "1"
    os.environ["SGLANG_PLACEHOLDER_KNN_TOPK"] = str(args.placeholder_knn_topk)
    os.environ["SGLANG_PLACEHOLDER_KNN_MIN_COSINE"] = str(args.placeholder_knn_min_cosine)
    os.environ["SGLANG_PLACEHOLDER_KNN_MAX_OVERLAP_RATIO"] = "1.0"
    os.environ["SGLANG_PLACEHOLDER_KNN_MAX_NEW_TOKEN_RATIO"] = "1.0"
    os.environ["SGLANG_PLACEHOLDER_KNN_MAX_SPAN_OVERLAP_RATIO"] = "1.0"
    os.environ["SGLANG_PLACEHOLDER_KNN_MIN_NEW_TOKENS"] = "0"
    proc = launch_server(args)
    return _ServerHandle(proc=proc, port=args.port)


async def run_one_case(session: aiohttp.ClientSession, args: argparse.Namespace,
                       tokenizer: Any, case: dict[str, Any]) -> list[dict[str, Any]]:
    """Run `--agent-count` agents over one case; return the per-agent rows."""
    rows: list[dict[str, Any]] = []
    segments = case["segments"][: args.segment_count]
    upstream = "Planner cached exact repository code objects for downstream agents."
    for idx, role in enumerate(AGENT_ROLES[: args.agent_count], 1):
        salt = f"ast_measure:{case['case_id']}:{args.segment_count}:{args.agent_count}:{args.mode}:{idx}"
        payload = make_payload(
            args,
            tokenizer,
            case,
            segments,
            args.mode,
            max_tokens=args.agent_max_tokens,
            salt=salt,
            role=role,
            agent_idx=idx,
            extra_context=upstream + f" Previous agent index: {idx - 1}.",
        )
        try:
            response = await post_chat_stream(session, args.port, payload)
        except Exception as exc:
            print(f"[ast_measure] {case['case_id']} agent {idx} request error: {exc!r}", flush=True)
            return rows
        row = row_from_response(
            case,
            args.mode,
            response,
            args.agent_max_tokens,
            args.max_file_chars,
            args.segment_count,
            "agent_scaling",
            agent_id=role,
            agent_count=args.agent_count,
        )
        rows.append(row)
    return rows


async def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """Top-level coroutine."""
    args.out_dir.mkdir(parents=True, exist_ok=True)
    handle = launch_with_ast_alignment_log(args, args.out_dir)
    if not await wait_ready(handle.port, timeout_s=args.server_timeout):
        print("[ast_measure] server not ready, aborting", flush=True)
        handle.proc.kill()
        handle.proc.wait(timeout=10)
        return {"status": "server_not_ready"}

    cases = load_manifest_cases(args.manifest, args.segment_count, args.max_file_chars, args.max_cases)
    print(f"[ast_measure] loaded {len(cases)} cases from {args.manifest}", flush=True)
    if not cases:
        handle.proc.kill()
        handle.proc.wait(timeout=10)
        return {"status": "no_cases"}

    tokenizer: Any = None
    if args.mode.startswith("placeholder") or args.mode.startswith("exact_reuse"):
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        except Exception as exc:
            print(f"[ast_measure] tokenizer load failed: {exc!r}", flush=True)

    rows_csv_path = args.out_dir / "rows.csv"
    with rows_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer: csv.DictWriter | None = None
        async with aiohttp.ClientSession() as session:
            for case_idx, case in enumerate(cases):
                t0 = time.time()
                rows = await run_one_case(session, args, tokenizer, case)
                for row in rows:
                    row["task_index"] = case_idx
                    if writer is None:
                        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                        writer.writeheader()
                    writer.writerow(row)
                f.flush()
                print(
                    f"[ast_measure] task {case_idx + 1}/{len(cases)} "
                    f"({case['case_id']}) agents={len(rows)} elapsed={time.time() - t0:.1f}s",
                    flush=True,
                )

    # Graceful shutdown so the log file flushes.
    try:
        handle.proc.kill()
        handle.proc.wait(timeout=15)
    except Exception:
        pass
    return {"status": "ok", "n_cases": len(cases), "csv": str(rows_csv_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", type=str, default="/home/gfy/models/Qwen2.5-3B-Instruct")
    parser.add_argument("--port", type=int, default=30110)
    parser.add_argument("--python", type=str, default="/home/gfy/.conda/envs/sglang-kvflow/bin/python")
    parser.add_argument("--mem-fraction-static", type=float, default=0.85)
    parser.add_argument("--max-total-tokens", type=int, default=65536)
    parser.add_argument("--hicache-ratio", type=float, default=1.5)
    parser.add_argument("--disable-hierarchical-cache", action="store_true")
    parser.add_argument("--hicache-storage-backend", type=str, default="")
    parser.add_argument("--server-timeout", type=int, default=300)

    parser.add_argument("--max-cases", type=int, default=5)
    parser.add_argument("--agent-count", type=int, default=5)
    parser.add_argument("--mode", type=str, default="placeholder_knn_reuse")
    parser.add_argument("--segment-count", type=int, default=5)
    parser.add_argument("--max-file-chars", type=int, default=8000)
    parser.add_argument("--agent-max-tokens", type=int, default=64)
    parser.add_argument("--lossy-max-zero-gap", type=int, default=4)

    parser.add_argument("--placeholder-knn-topk", type=int, default=5)
    parser.add_argument("--placeholder-knn-min-cosine", type=float, default=0.85)
    return parser.parse_args()


import json  # placed at the bottom to keep parser-related imports tidy

def main() -> int:
    args = parse_args()
    args.out_dir = args.out_dir.expanduser().resolve()
    args.manifest = args.manifest.expanduser().resolve()
    if not args.manifest.is_file():
        print(f"[ast_measure] manifest not found: {args.manifest}", flush=True)
        return 2
    print(
        f"[ast_measure] starting: max_cases={args.max_cases} agent_count={args.agent_count} "
        f"mode={args.mode} segments={args.segment_count}",
        flush=True,
    )
    result = asyncio.run(run_benchmark(args))

    # Parse the structured AST_ALIGN log lines.
    log_path = args.out_dir / "sglang_server.log"
    matches = parse_log_file(log_path)
    matches_path = args.out_dir / "rows_ast_alignment.csv"
    if matches:
        with matches_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(matches[0].keys()))
            w.writeheader()
            w.writerows(matches)
        print(f"[ast_measure] wrote {len(matches)} AST_ALIGN rows -> {matches_path}", flush=True)
    else:
        print(f"[ast_measure] NO [AST_ALIGN] lines in {log_path} — placeholder pool did not match", flush=True)
        matches_path.write_text("rid,slot_id,cos,slot_start,slot_end,match_start,match_end\n", encoding="utf-8")

    print(f"[ast_measure] done: {result}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
