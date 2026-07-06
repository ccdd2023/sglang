"""Persistent-server multi-agent benchmark over a giant codebase.

Runs the sglang-kvflow placeholder k-NN pipeline against MANY SWE-Smith
tasks that all share the same giant Python repository (default: pandas).
Reuses helpers from ``bench_kvcomm_ttft_stress.py`` unchanged — this driver
is a thin orchestrator on top.

Key design points:
    - ONE sglang server runs for the entire ``--max-tasks`` run so the
      placeholder anchor pool accumulates across tasks (this is the whole
      point — per-case fresh-server drivers hide the cross-task gain).
    - Tasks come from a SWE-Smith manifest.jsonl (built by
      ``swesmith_pandas_loader.py``) plus the local checkout of the target
      repository (default: ``results/giant_codebase/pandas_src``).
    - The same 5-agent role chain as ``bench_kvcomm_ttft_stress.E7``:
      implementer / debugger / reviewer / verifier / auditor.
    - For >3-case runs, ``--disable-overlap-schedule --max-running-requests 1``
      are passed to dodge the ``_delete_leaf`` race bug (single-step the
      scheduler). NOTE: ``--force-evict`` is NOT a real server flag (no such
      arg in server_args.py) — older docstrings that mention it are wrong.
      Chunks of ``--chunk-size`` tasks auto-relaunch the server on death.
    - Output CSV has one row per (task × agent); the aggregator reads it
      to produce pool-growth curves and cumulative TTFT speedup.

Usage example:

    PY=/home/gfy/.conda/envs/sglang-kvflow/bin/python
    $PY benchmark/multi_workflow/bench_giant_codebase_reuse.py \\
        --manifest results/giant_codebase/tasks/pandas__pandas__1000/manifest.jsonl \\
        --repo-root results/giant_codebase/pandas_src \\
        --max-tasks 50 --agent-count 5 \\
        --mode placeholder_knn_reuse --segment-count 5 \\
        --max-file-chars 8000 \\
        --out-dir results/ttft_agenttemplatekv/giant_pandas_50_20260626/
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp

# Reuse all helpers from the existing bench driver without modifying it.
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from benchmark.multi_workflow.bench_kvcomm_ttft_stress import (  # noqa: E402
    AGENT_ROLES,
    CodeSegment,
    build_anchor_fields,
    build_placeholder_anchor_fields,
    build_slot_messages,
    build_stress_messages,
    extract_lossy_meta,
    launch_server,
    make_payload,
    now_ms,
    post_chat,
    post_chat_stream,
    percentile,
    row_from_response,
    safe_mean,
    wait_ready,
    warm_planner,
)
from benchmark.multi_workflow.swesmith_pandas_loader import load_manifest  # noqa: E402


# ---------------------------------------------------------------------------
# Task loading — turn a SWE-Smith manifest + a local repo checkout into the
# ``{case_id, repo, segments[]}`` shape that the rest of the pipeline expects.
# ---------------------------------------------------------------------------


def load_repo_text(repo_root: Path, rel_path: str) -> str | None:
    """Read ``repo_root / rel_path``; return None if the file is missing."""
    # SWE-Smith file paths are repo-relative (e.g. "pandas/core/reshape/tile.py").
    full = repo_root / rel_path
    if not full.is_file():
        return None
    try:
        return full.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def build_segments_for_task(
    task: dict[str, Any],
    repo_root: Path,
    segment_count: int,
    max_file_chars: int,
    sibling_window: int = 0,
) -> list[CodeSegment]:
    """Pick up to ``segment_count`` files for the task and load them.

    Order of preference:
        1. Files explicitly listed in the task's patch (in patch order).
        2. Sibling files from the same directory as (1) — only used when
           ``sibling_window > 0`` and we still need more segments to reach
           ``segment_count``.

    Selection priority: files appear in the same order as the patch lists them.
    We pick the FIRST ``segment_count`` files that exist on disk AND have
    non-empty text. If fewer than ``segment_count`` files are usable we return
    whatever we have (the caller will skip tasks with too few segments).

    The ``sibling_window`` knob is critical for AST anchor overlap: SWE-Smith
    tasks typically touch 1 file, but agents reading siblings of the same
    module will share FunctionDef/ClassDef anchors across tasks.
    """
    seen: set[str] = set()
    ordered: list[str] = []

    # (1) Patched files first.
    for rel in task.get("files", []):
        if rel not in seen:
            seen.add(rel)
            ordered.append(rel)

    # (2) Sibling files from the same directory as each patched file.
    if sibling_window > 0 and len(ordered) < segment_count:
        for rel in list(task.get("files", [])):
            anchor_dir = (repo_root / rel).parent
            if not anchor_dir.is_dir():
                continue
            for sibling in sorted(anchor_dir.iterdir()):
                if not sibling.is_file() or not sibling.name.endswith(".py"):
                    continue
                sib_rel = str(sibling.relative_to(repo_root))
                if sib_rel in seen:
                    continue
                seen.add(sib_rel)
                ordered.append(sib_rel)
                if len(ordered) >= segment_count + sibling_window:
                    break
            if len(ordered) >= segment_count + sibling_window:
                break

    segments: list[CodeSegment] = []
    for rel in ordered:
        if len(segments) >= segment_count:
            break
        text = load_repo_text(repo_root, rel)
        if not text:
            continue
        if max_file_chars and len(text) > max_file_chars:
            text = text[:max_file_chars].rstrip()
        if not text.strip():
            continue
        segments.append(CodeSegment(name=rel, text=text))
    return segments


def load_giant_codebase_cases(
    manifest_path: Path,
    repo_root: Path,
    max_tasks: int,
    segment_count: int,
    max_file_chars: int,
    sibling_window: int = 0,
    skip_indices: set[int] | None = None,
) -> list[dict[str, Any]]:
    """Build the case list compatible with the existing helper pipeline."""
    skip = skip_indices or set()
    records = load_manifest(manifest_path)
    cases: list[dict[str, Any]] = []
    skipped_short = 0
    skipped_index = 0
    for idx, task in enumerate(records):
        if idx in skip:
            skipped_index += 1
            continue
        if len(cases) >= max_tasks:
            break
        segments = build_segments_for_task(task, repo_root, segment_count, max_file_chars, sibling_window)
        if len(segments) < segment_count:
            skipped_short += 1
            continue
        cases.append(
            {
                "case_id": task["instance_id"],
                "repo": task.get("repo", ""),
                "segments": segments,
                "problem_statement": task.get("problem_statement", "")[:500],
            }
        )
    print(
        f"[giant_driver] loaded {len(cases)} cases from {len(records)} manifest records "
        f"(skipped {skipped_short} short, {skipped_index} by-index, sibling_window={sibling_window})",
        flush=True,
    )
    return cases


# ---------------------------------------------------------------------------
# Server lifecycle — wrap ``launch_server`` + chunked health-check + auto-relaunch.
# ---------------------------------------------------------------------------


@dataclass
class ServerHandle:
    proc: subprocess.Popen
    port: int
    out_dir: Path
    chunk_id: int

    def is_alive(self) -> bool:
        return self.proc.poll() is None


def launch_or_relaunch(handle: ServerHandle | None, args: argparse.Namespace, chunk_id: int) -> ServerHandle:
    """Start a fresh server, or relaunch if a previous handle died.

    In addition to whatever ``launch_server`` sets, we force-set the
    ``SGLANG_PLACEHOLDER_KNN_MATCH=1`` family of env vars so the
    placeholder k-NN pool is wired up server-side. Without these the
    placeholder_knn_reuse mode silently degrades to lossy with zero
    pool growth (matching the bench_kvcomm_ttft_stress default which
    does not opt in).
    """
    if handle is not None:
        if handle.is_alive():
            print(f"[giant_driver] chunk {chunk_id}: server still alive, reusing", flush=True)
            return handle
        print(f"[giant_driver] chunk {chunk_id}: previous server died (rc={handle.proc.returncode}), relaunching", flush=True)
        try:
            handle.proc.kill()
            handle.proc.wait(timeout=10)
        except Exception:
            pass
    # Force placeholder k-NN env vars BEFORE launch_server spawns the subprocess.
    # L3 (placeholder k-NN body) is deprecated for production (2026-06-27);
    # it reuses K/V from byte-different code which silently produces
    # wrong-but-plausible outputs when the variable name or comment
    # changes. Default OFF — require explicit --enable-research-l3 to opt in.
    #
    # SUPPORTED REGIME (2026-06-30, KVCOMM): the supported lossy reuse is
    # byte-exact code copied + RoPE-rotated to a new position (KVCOMM), with
    # AST chunking as the coding-specific optimization. This is the L2
    # whole-slot path (general baseline) and the L4 chunk + C2 path (AST). The
    # MiniLM L3 path AND the offset-aligned AST gate (which rides inside L3 →
    # semantic selection) are the WRONG regime and must stay OFF. They are off
    # by default here; do NOT export SGLANG_L3_AST_GATE / SGLANG_L3_AST_GATE_OFFSET
    # in the launcher scripts for the L2 / L4+C2 configs.
    if args.enable_research_l3:
        os.environ["SGLANG_PLACEHOLDER_KNN_MATCH"] = "1"
        os.environ["SGLANG_PLACEHOLDER_KNN_TOPK"] = str(args.placeholder_knn_topk)
        os.environ["SGLANG_PLACEHOLDER_KNN_MIN_COSINE"] = str(args.placeholder_knn_min_cosine)
        print(
            f"[giant_driver] chunk {chunk_id}: L3 placeholder k-NN ENABLED "
            f"(topk={args.placeholder_knn_topk}, min_cosine={args.placeholder_knn_min_cosine}). "
            f"RESEARCH ONLY — DO NOT USE IN PRODUCTION.",
            flush=True,
        )
    else:
        os.environ["SGLANG_PLACEHOLDER_KNN_MATCH"] = "0"
        print(
            f"[giant_driver] chunk {chunk_id}: L3 placeholder k-NN DISABLED (default). "
            f"Production path uses byte-exact match (L2) only.",
            flush=True,
        )
    # Phase 2.5+ optimization: skip the k-NN body when the prefix cache
    # already covers most of a slot. Default is 0.5. For the giant-codebase
    # benchmark we want the k-NN body to run even when the prefix cache
    # hits, so we set this to 1.0 (effectively disabled). Otherwise the
    # `cached_tokens = 7244` warmup hits block the k-NN body from ever
    # doing useful work.
    os.environ["SGLANG_PLACEHOLDER_KNN_MAX_OVERLAP_RATIO"] = "1.0"
    # O10: skip the k-NN body when the overall request is mostly new
    # tokens. Default 1.0 (disabled). For our persistent-server use case
    # we want the body to fire whenever spans + pool entries are present.
    os.environ["SGLANG_PLACEHOLDER_KNN_MAX_NEW_TOKEN_RATIO"] = "1.0"
    # O8: skip the k-NN body when span overlap > ratio. Default 1.0.
    os.environ["SGLANG_PLACEHOLDER_KNN_MAX_SPAN_OVERLAP_RATIO"] = "1.0"
    # O7: minimum new tokens to fire the k-NN body. Default 0 (any).
    os.environ["SGLANG_PLACEHOLDER_KNN_MIN_NEW_TOKENS"] = "0"
    proc = launch_server(args)
    return ServerHandle(proc=proc, port=args.port, out_dir=args.out_dir, chunk_id=chunk_id)


async def ensure_server_ready(handle: ServerHandle, args: argparse.Namespace) -> bool:
    """Wait for /health_generate; return True if ready, False if timed out."""
    return await wait_ready(handle.port, timeout_s=args.server_timeout)


# ---------------------------------------------------------------------------
# Main async loop.
# ---------------------------------------------------------------------------


async def run_one_task(
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    tokenizer: Any,
    case: dict[str, Any],
    chunk_id: int,
    case_idx: int = 0,
) -> list[dict[str, Any]]:
    """Run ``--agent-count`` agents over one case; return per-agent rows.

    Optionally calls ``warm_planner`` (the byte-exact + placeholder-kNN
    warmup pair) before the first agent so the placeholder anchor pool has
    something to match — without it, agent 1 of the very first task in a
    fresh server sees ``placeholder_anchor_store_entry_count = 0``.

    Per-agent variation: when ``--vary-code`` is set, each agent gets a
    `# Agent {N} variant\\n` prefix prepended to each segment's text. This
    breaks the byte-exact prefix cache match (so the placeholder k-NN
    body is consulted instead of short-circuiting on byte-exact match)
    while keeping the slot text semantically similar enough that the
    k-NN search succeeds. This mirrors the v44 placeholder k-NN
    validation cycle (see plan §3.1).
    """
    rows: list[dict[str, Any]] = []
    segments = case["segments"][: args.segment_count]

    # Per-agent segments (deep-copied so we can mutate text per agent).
    import copy

    agent_segments: list[list[CodeSegment]] = []
    for idx in range(1, args.agent_count + 1):
        if args.vary_code:
            salt_text = f"# Agent {idx} variant\n"
            segs = [
                CodeSegment(
                    name=s.name,
                    text=(salt_text + s.text) if not s.text.startswith(salt_text) else s.text,
                )
                for s in segments
            ]
        else:
            segs = [CodeSegment(name=s.name, text=s.text) for s in segments]
        # FAIR-MEASUREMENT / KVCOMM regime (Step 1): --position-shift cyclically
        # rotates the segment ORDER per agent (agent idx -> rotate by idx-1).
        # The segment CONTENT is byte-identical across agents; only the absolute
        # position differs. Combined with the per-agent distinct cache_salt
        # (below), agents 2..N have a cold radix tree → the shared code can ONLY
        # be recovered via the KVCOMM rotation-copy path (byte-exact selection +
        # RoPE delta), NOT radix prefix. This is the SUPPORTED regime (same
        # prompt at a different position, rotated). Must be used with
        # --no-vary-code (vary-code is content-shift, the wrong regime).
        if getattr(args, "position_shift", False) and len(segs) > 1:
            shift = (idx - 1) % len(segs)
            segs = segs[shift:] + segs[:shift]
        # PARTIAL-SHARING + ROTATION (2026-07-01): each agent DROPS one
        # different top-level function (FunctionDef/ClassDef) from each
        # segment, so the segment TEXT differs per agent (whole-slot byte-
        # exact match FAILS) while the remaining functions are byte-identical
        # across agents (AST chunking reuses them cross-position via the
        # content-derived slot_id + per-chunk signature). This is the regime
        # where AST granularity beats whole-slot: the realistic coding
        # scenario where agents share most of a file but each focuses on /
        # elides a different function. Combine with --position-shift so the
        # shared functions also sit at a different absolute position (KVCOMM
        # rotation). Gated by --partial-share (default OFF).
        if getattr(args, "partial_share", False):
            try:
                from sglang.srt.mem_cache.ast_chunker import ASTChunker

                _chunker = ASTChunker()
            except Exception:
                _chunker = None
            if _chunker is not None:
                new_segs = []
                for seg in segs:
                    chunks = _chunker.chunk_text(seg.text)
                    defs = [
                        c for c in chunks
                        if c.anchor_type in ("function", "class")
                        and c.byte_end > c.byte_start
                    ]
                    if len(defs) >= 2:
                        drop = defs[(idx - 1) % len(defs)]
                        new_text = (
                            seg.text[: drop.byte_start] + seg.text[drop.byte_end:]
                        )
                        new_segs.append(CodeSegment(name=seg.name, text=new_text))
                    else:
                        new_segs.append(seg)
                segs = new_segs
        agent_segments.append(segs)

    if args.warm_planner and tokenizer is not None:
        try:
            await warm_planner(
                session,
                args,
                tokenizer,
                case,
                segments,
                args.max_file_chars,
                args.segment_count,
            )
        except Exception as exc:
            print(
                f"[giant_driver] {case['case_id']} warm_planner error: {exc!r}",
                flush=True,
            )

    upstream = "Planner cached exact repository code objects for downstream agents."
    role_list = AGENT_ROLES[: args.agent_count]
    for idx, (role, segs) in enumerate(zip(role_list, agent_segments), 1):
        salt = f"giant:{case['case_id']}:{args.segment_count}:{args.max_file_chars}:{args.agent_count}:{args.mode}:{idx}"
        payload = make_payload(
            args,
            tokenizer,
            case,
            segs,
            args.mode,
            max_tokens=args.agent_max_tokens,
            salt=salt,
            role=role,
            agent_idx=idx,
            extra_context=upstream + f" Previous agent index: {idx - 1}.",
        )
        # DEBUG: verify placeholder spans are in payload
        if case_idx == 0 and idx == 1 and args.debug_first_task:
            print(
                f"[giant_driver] DEBUG task0 agent1 payload: "
                f"spans_count={len(payload.get('placeholder_anchor_token_spans', []))} "
                f"reuse_mode={payload.get('reuse_mode')} "
                f"cache_salt={payload.get('cache_salt')[:60]!r}",
                flush=True,
            )
        try:
            response = await post_chat_stream(session, args.port, payload)
        except Exception as exc:  # network / server crash mid-stream
            print(
                f"[giant_driver] {case['case_id']} agent {idx} ({role}) request error: {exc!r}",
                flush=True,
            )
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
        row["chunk_id"] = chunk_id
        row["agent_idx"] = idx  # for offline F1 alignment vs lossless outputs.jsonl
        rows.append(row)
        # Sidecar dump of the generated text for offline A/B correctness
        # comparison (e.g. C2 CacheBlend vs no-CacheBlend baseline). The CSV
        # only stores output_chars + a bag-of-tokens F1 that defaults to 1.0
        # when no in-run baseline exists, so the raw text is needed for a real
        # byte/sequence comparison.
        try:
            _out_text = response.get("text") or ""
            with (args.out_dir / "outputs.jsonl").open("a", encoding="utf-8") as _of:
                _of.write(json.dumps({
                    "case_id": case.get("case_id", ""),
                    "task_index": case_idx,
                    "agent_idx": idx,
                    "role": role,
                    "output_text": _out_text,
                }) + "\n")
        except Exception:
            pass
        cached = row.get("cached_tokens", 0)
        upstream += f" {role} observed {cached} cached tokens."
    return rows


def write_fair_summary(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    """FAIR-MEASUREMENT (B3): write a per-run summary that decomposes
    cached_tokens into radix-prefix vs code-aware-reused, and reports TTFT
    over the reuser agents only (agent 1 / "implementer" is the source and
    is excluded by default — see --exclude-source-agent).

    This single-run summary makes the radix-vs-code-aware split visible per
    config. The cross-config A/B (baseline vs experimental speedup + the
    warmup-parity gate) is computed by analyze_fair_ab.py over multiple
    runs' rows.csv files.
    """
    if not rows:
        return
    source_role = AGENT_ROLES[0]  # "implementer" == agent 1 (the source)
    exclude_source = bool(getattr(args, "exclude_source_agent", True))
    reuser_rows = [r for r in rows if not (exclude_source and r.get("agent_id") == source_role)]
    if not reuser_rows:
        reuser_rows = rows

    def fnum(key: str, src: list[dict[str, Any]] = reuser_rows) -> float:
        vals = [float(r.get(key) or 0) for r in src]
        return safe_mean(vals)

    ttft = [float(r.get("ttft_ms") or 0) for r in reuser_rows]
    src_ttft = [float(r.get("ttft_ms") or 0) for r in rows if r.get("agent_id") == source_role]
    lines = [
        "# Fair Measurement Summary (single run)",
        "",
        f"- mode: `{args.mode}`",
        f"- agent_count: {args.agent_count}",
        f"- vary_code: {args.vary_code}",
        f"- warm_anchor_pool: {getattr(args, 'warm_anchor_pool', False)}",
        f"- exclude_source_agent: {exclude_source} (source role = `{source_role}`)",
        f"- total rows: {len(rows)} (reusers counted: {len(reuser_rows)})",
        "",
        "## TTFT (reusers only, unless source included)",
        f"- avg TTFT = {safe_mean(ttft):.1f} ms, p50 = {percentile(ttft, 0.5):.1f} ms, "
        f"p90 = {percentile(ttft, 0.9):.1f} ms",
        f"- source agent avg TTFT = {safe_mean(src_ttft):.1f} ms (prefill-bound; not a reuse beneficiary)",
        "",
        "## cached_tokens decomposition (reusers only)",
        f"- avg cached_tokens = {fnum('cached_tokens'):.1f}",
        f"- avg radix_prefix_tokens = {fnum('radix_prefix_tokens'):.1f}  "
        "(radix L1 prefix — the ONLY part the prefix_cache_only baseline also sees; must cancel in a fair A/B)",
        f"- avg codeaware_reused_tokens = {fnum('codeaware_reused_tokens'):.1f}  "
        "(L2 whole-slot + L3 offset-gate body copy + C2 chunk copy — the code-aware contribution)",
        f"  - l2_wholeslot_reused_tokens = {fnum('l2_wholeslot_reused_tokens'):.1f}  (general KVCOMM baseline)",
        f"  - l3_offset_reused_tokens = {fnum('l3_offset_reused_tokens'):.1f}  (MiniLM — deprecated regime)",
        f"  - c2_chunk_reused_tokens = {fnum('c2_chunk_reused_tokens'):.1f}  (AST chunk path)",
        "",
        "## Interpretation",
        "- If `radix_prefix_tokens` ≈ matches the prefix_cache_only run's `cached_tokens`,",
        "  the radix prefix cancels and the speedup is honestly code-aware.",
        "- `codeaware_reused_tokens` is the code-aware algorithm's own reuse, isolated.",
        "- Source-agent (implementer) should show ~0 codeaware_reused (it is the source).",
        "- Cross-config speedup + parity gate: run analyze_fair_ab.py over this and the",
        "  prefix_cache_only run's rows.csv.",
    ]
    if getattr(args, "precompute_kv_dir", None):
        lines += [
            "",
            "## Precomputed codebase KV",
            f"- precompute_kv_dir: `{args.precompute_kv_dir}`",
            f"- canonical_prefix: {getattr(args, 'precompute_canonical_prefix', False)}",
            f"- host_pool_size_gb: {getattr(args, 'precompute_host_size_gb', 4.0)}",
            "- With precompute ON, the pool is warm at server start, so agent 1",
            "  ALSO hits the pool (it is a reuse beneficiary, not the source).",
            f"- source_agent_included: {not exclude_source}",
            "- Reuse from precomputed (host-resident) chunks flows through the same",
            "  c2_chunk_reused_tokens counter as on-demand L4 reuse; the difference",
            "  vs a no-precompute run is agent 1's nonzero reuse + warm-from-start TTFT.",
            "- Honest accuracy bound: only the canonical preamble is losslessly",
            "  reusable; file content at shifted positions stays lossy (proven",
            "  cross-context KV loss).",
            "",
        ]
    else:
        lines += [""]
    (args.out_dir / "FAIR_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


async def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """Top-level coroutine: launch server, iterate cases, dump CSV."""
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "rows.csv"
    csv_file = csv_path.open("w", newline="", encoding="utf-8")

    cases = load_giant_codebase_cases(
        args.manifest,
        args.repo_root,
        max_tasks=args.max_tasks,
        segment_count=args.segment_count,
        max_file_chars=args.max_file_chars,
        sibling_window=args.sibling_window,
    )
    if not cases:
        print("[giant_driver] ERROR: no usable cases — aborting", flush=True)
        csv_file.close()
        return {"rows": 0, "csv": str(csv_path)}

    # Tokenizer import is lazy: only needed if we ever enter a placeholder mode.
    tokenizer: Any = None
    if args.mode.startswith("placeholder") or args.mode.startswith("exact_reuse"):
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        except Exception as exc:
            print(
                f"[giant_driver] WARN: tokenizer load failed ({exc!r}); "
                "anchor spans will be skipped — placeholder modes will degrade to lossy.",
                flush=True,
            )

    fieldnames: list[str] | None = None
    writer: csv.DictWriter | None = None
    total_rows = 0
    all_rows: list[dict[str, Any]] = []

    handle: ServerHandle | None = None
    chunk_id = 0
    chunk_idx_within = 0
    async with aiohttp.ClientSession() as session:
        for case_idx, case in enumerate(cases):
            # Chunk boundary: relaunch if we've done --chunk-size tasks in this chunk
            # OR if the previous chunk's server died.
            if chunk_idx_within == 0 or chunk_idx_within >= args.chunk_size:
                chunk_id += 1
                chunk_idx_within = 0
                handle = launch_or_relaunch(handle, args, chunk_id)
                if not await ensure_server_ready(handle, args):
                    print(
                        f"[giant_driver] chunk {chunk_id}: server not ready, relaunching",
                        flush=True,
                    )
                    handle.proc.kill()
                    handle.proc.wait(timeout=10)
                    handle = launch_or_relaunch(None, args, chunk_id)
                    if not await ensure_server_ready(handle, args):
                        print(
                            f"[giant_driver] chunk {chunk_id}: server still not ready — aborting run",
                            flush=True,
                        )
                        break

            t0 = time.time()
            rows = await run_one_task(session, args, tokenizer, case, chunk_id, case_idx=case_idx)
            dt = time.time() - t0
            for row in rows:
                row["task_index"] = case_idx
                all_rows.append(row)
                if writer is None:
                    fieldnames = list(row.keys())
                    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                    writer.writeheader()
                writer.writerow(row)
                csv_file.flush()
                total_rows += 1
            print(
                f"[giant_driver] task {case_idx + 1}/{len(cases)} "
                f"({case['case_id']}) chunk={chunk_id} agents={len(rows)} "
                f"elapsed={dt:.1f}s total_rows={total_rows}",
                flush=True,
            )
            chunk_idx_within += 1

            # Detect a dead server mid-stream and break out so the outer
            # relaunch picks up on the next chunk boundary.
            if handle is not None and not handle.is_alive():
                print(
                    f"[giant_driver] chunk {chunk_id}: server died mid-loop, will relaunch next chunk",
                    flush=True,
                )
                chunk_idx_within = args.chunk_size  # force boundary on next iter

    csv_file.close()
    if handle is not None and handle.is_alive():
        try:
            handle.proc.kill()
            handle.proc.wait(timeout=15)
        except Exception:
            pass
    # FAIR-MEASUREMENT (B3): write a per-run decomposed summary so the
    # radix-vs-code-aware split is visible without a separate analyzer.
    write_fair_summary(args, all_rows)
    return {"rows": total_rows, "csv": str(csv_path), "chunks": chunk_id}


# ---------------------------------------------------------------------------
# Argparse.
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--manifest", type=Path, required=True, help="SWE-Smith manifest.jsonl from swesmith_pandas_loader.py")
    parser.add_argument("--repo-root", type=Path, required=True, help="Local checkout of the target repo (e.g. results/giant_codebase/pandas_src)")
    parser.add_argument("--out-dir", type=Path, required=True)

    # Reuse the launch args that ``bench_kvcomm_ttft_stress.parse_args`` builds.
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model id (e.g. Qwen/Qwen2.5-3B-Instruct)")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--python", type=str, default="/home/gfy/.conda/envs/sglang-kvflow/bin/python")
    parser.add_argument("--mem-fraction-static", type=float, default=0.85)
    parser.add_argument("--max-total-tokens", type=int, default=65536)
    parser.add_argument("--hicache-ratio", type=float, default=1.5)
    parser.add_argument("--disable-hierarchical-cache", action="store_true")
    parser.add_argument("--hicache-storage-backend", type=str, default="")
    parser.add_argument("--server-timeout", type=int, default=300)

    # Stress knobs.
    parser.add_argument("--max-tasks", type=int, default=50)
    parser.add_argument("--agent-count", type=int, default=5)
    parser.add_argument("--mode", type=str, default="placeholder_knn_reuse")
    parser.add_argument("--segment-count", type=int, default=5)
    parser.add_argument("--max-file-chars", type=int, default=8000)
    parser.add_argument("--agent-max-tokens", type=int, default=64)
    parser.add_argument("--lossy-max-zero-gap", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=3, help="Tasks per server lifetime before auto-relaunch (workaround for _delete_leaf race)")
    parser.add_argument("--sibling-window", type=int, default=4, help="Add up to N sibling .py files from the same directory as each patched file (default 4)")
    parser.add_argument("--warm-planner", action="store_true", default=True, help="Warm placeholder anchor pool with planner requests before each task (default on)")
    parser.add_argument("--no-warm-planner", dest="warm_planner", action="store_false", help="Skip warm_planner requests (faster but pool starts empty)")
    parser.add_argument(
        "--warm-anchor-pool",
        action="store_true",
        default=False,
        help=(
            "FAIR-MEASUREMENT (B1). Pre-populate the placeholder anchor pool "
            "with a warmup request before the timed agents. DEFAULT OFF: the "
            "warmup let agent 1 (the source) reuse code it should be "
            "generating, inflating the code-aware speedup. With it OFF, agent "
            "1's own timed request is the natural source; agents 2..N reuse "
            "from it. Set ON only to reproduce the pre-fix (inflated) numbers."
        ),
    )
    parser.add_argument(
        "--exclude-source-agent",
        action="store_true",
        default=True,
        help=(
            "FAIR-MEASUREMENT (B1). Exclude agent 1 (the implementer/source) "
            "from the speedup average. Agent 1 generates the code first and "
            "has nothing to reuse, so its TTFT is prefill-bound and unrelated "
            "to code-aware reuse; including it dilutes the measured code-aware "
            "speedup. Default ON; the reported speedup reflects reusers (2..N)."
        ),
    )
    parser.add_argument(
        "--include-source-agent",
        dest="exclude_source_agent",
        action="store_false",
        help="Include agent 1 in the speedup average (pre-fix behavior).",
    )
    parser.add_argument(
        "--allow-parity-failure",
        action="store_true",
        default=False,
        help=(
            "FAIR-MEASUREMENT (B2). Do not raise when the warmup-parity gate "
            "detects a radix-prefix mismatch between baseline and experimental "
            "configs. By default a mismatch (radix_delta > 15%%) raises "
            "RuntimeError and writes PARITY_VIOLATIONS.txt, because it means "
            "the measured speedup is confounded with radix prefix warmth."
        ),
    )
    parser.add_argument("--placeholder-knn-topk", type=int, default=5)
    parser.add_argument("--placeholder-knn-min-cosine", type=float, default=0.85)
    # --- Offline-precomputed codebase KV (Phase 4) ---
    parser.add_argument(
        "--precompute-kv-dir",
        default=None,
        help=(
            "Offline-precomputed codebase KV dir (results/codebase_kv/<run_tag>). "
            "When set, the server loads it into a CPU host pool at start "
            "(SGLANG_PRECOMPUTE_KV_DIR / SGLANG_PRECOMPUTE_HOST_SIZE_GB) so the "
            "pool is warm from the first request INCLUDING agent 1 — eliminating "
            "the source-agent warmup tax and surviving the 3-task server relaunch."
        ),
    )
    parser.add_argument(
        "--precompute-host-size-gb",
        type=float,
        default=4.0,
        help="CPU host pool size (GB) for precomputed KV (default 4).",
    )
    parser.add_argument(
        "--precompute-canonical-prefix",
        action="store_true",
        default=False,
        help=(
            "Prepend the canonical shared prefix (preamble.txt) to the system "
            "message. Only the preamble is losslessly reusable; file content at "
            "shifted positions stays lossy (proven fundamental limit)."
        ),
    )
    parser.add_argument(
        "--include-source-with-precompute",
        action="store_true",
        default=False,
        help=(
            "FAIR-MEASUREMENT for precompute. When --precompute-kv-dir is set the "
            "pool is warm at start, so agent 1 ALSO hits the pool and is a reuse "
            "beneficiary. Excluding it would understate the speedup. Sets "
            "exclude_source_agent=False (auto unless --exclude-source-agent given)."
        ),
    )
    parser.add_argument("--debug-first-task", action="store_true", help="Print placeholder pool status after first task's first agent")
    parser.add_argument("--task-mode", choices=["critique", "verdict"], default="critique",
                        help="critique = single-sentence risk (R10-R19 default). "
                             "verdict = single-line PASS/FAIL (R21 task-completion accuracy).")
    parser.add_argument("--vary-code", action="store_true", default=True, help="Per-agent byte-level variation to force placeholder k-NN path (default on)")
    parser.add_argument("--no-vary-code", dest="vary_code", action="store_false", help="Disable per-agent byte-level variation")
    parser.add_argument(
        "--position-shift",
        action="store_true",
        default=False,
        help=(
            "KVCOMM regime (Step 1). Cyclically rotate the segment ORDER per "
            "agent (same byte-identical code content, different absolute "
            "position). With the per-agent distinct cache_salt, agents 2..N "
            "have a cold radix tree → the shared code is recoverable ONLY via "
            "the KVCOMM rotation-copy path (byte-exact + RoPE delta), not radix "
            "prefix. This is the SUPPORTED regime (same prompt, different "
            "position, rotated). Use with --no-vary-code (vary-code is "
            "content-shift, the wrong regime)."
        ),
    )
    parser.add_argument(
        "--partial-share",
        action="store_true",
        default=False,
        help=(
            "PARTIAL-SHARING + ROTATION regime. Each agent DROPS one different "
            "top-level function (FunctionDef/ClassDef) from each segment, so the "
            "segment text differs per agent (whole-slot byte-exact match FAILS) "
            "while the remaining functions are byte-identical (AST chunking "
            "reuses them cross-position). The regime where AST granularity beats "
            "whole-slot: realistic coding scenario where agents share most of a "
            "file but each elides a different function. Use with "
            "--position-shift --no-vary-code."
        ),
    )
    parser.add_argument(
        "--enable-research-l3",
        action="store_true",
        default=False,
        help=(
            "RESEARCH ONLY. Enables the placeholder k-NN body "
            "(SGLANG_PLACEHOLDER_KNN_MATCH=1) which reuses K/V across "
            "byte-different code via MiniLM semantic similarity. "
            "DEPRECATED for production because variable renames / "
            "comment edits silently produce wrong-but-plausible "
            "outputs (cos=0.92 doesn't mean semantically equivalent). "
            "Default OFF — production must use byte-exact match (L2) "
            "only. See HANDOFF.md §L3-deprecation."
        ),
    )

    args = parser.parse_args()

    # Offline-precomputed KV: when the pool is warm at start (precompute on),
    # agent 1 is a reuse beneficiary, so include it in the speedup average
    # unless the caller explicitly passed --exclude-source-agent. Also raise
    # the 3-task server-relaunch cap: the pool reloads from disk at each
    # relaunch, so the relaunch no longer loses the pool (it only dodged the
    # _delete_leaf race, which the --disable-overlap-schedule invariant still
    # handles). KEEP --disable-overlap-schedule --max-running-requests 1 for >3.
    if getattr(args, "precompute_kv_dir", None):
        if getattr(args, "include_source_with_precompute", False):
            args.exclude_source_agent = False
        if args.chunk_size <= 3:
            args.chunk_size = 10000

    return args


def main() -> int:
    args = parse_args()
    args.out_dir = args.out_dir.expanduser().resolve()
    args.manifest = args.manifest.expanduser().resolve()
    args.repo_root = args.repo_root.expanduser().resolve()
    if not args.manifest.is_file():
        print(f"[giant_driver] ERROR: manifest not found: {args.manifest}", flush=True)
        return 2
    if not args.repo_root.is_dir():
        print(f"[giant_driver] ERROR: repo-root not found: {args.repo_root}", flush=True)
        return 2

    print(
        f"[giant_driver] starting: max_tasks={args.max_tasks} agent_count={args.agent_count} "
        f"mode={args.mode} segments={args.segment_count} chunk_size={args.chunk_size}",
        flush=True,
    )
    result = asyncio.run(run_benchmark(args))
    print(f"[giant_driver] done: {result}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
