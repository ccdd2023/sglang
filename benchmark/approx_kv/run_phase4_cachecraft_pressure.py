"""Phase 4 unified high-pressure runner scaffold for Cache-Craft (R3).

This mirrors the CLI/settings/warmup/repeats/central-log contract used by
R1 EPIC's ``run_phase4_epic_pressure.py`` so all six research paths report
under one unified schema:

- exact header sweep:  0, 32, 64, 128, 256 tokens
- lossy body sweep:    512, 768, 1024, 2048 tokens (canonical source
  registered in <=512-token segments per chunk, see
  ``cachecraft_workloads.py``)
- ``mem_fraction_static=0.35``
- actual reusable rho targets approx 0.9 / 1.1 / 1.5 / 2.0 / 3.0
- fixed S0 LRU / GPU-only / prefetch-off scheduler configuration
- one discarded warmup pass, >=2 (default 4) formal repeats per setting
- append-only central JSONL run log
- client-observed streaming TTFT (first non-empty ``data:`` frame)

Unlike R1's runner, this script's *first* action is a capability check
(``cachecraft_capability.inspect_scheduler_dispatch_capability``). As
documented there, the real scheduler request path
(``schedule_batch.py``) has no dispatch to
``cachecraft_runtime.restore_request_via_cachecraft`` yet: any request
tagged ``plugin: "cachecraft"`` would silently be served by the *generic*
raw-copy reuse path instead of Cache-Craft's CCI-driven decision. Running
the pressure sweep against such a server would produce numbers that look
like a successful Cache-Craft benchmark while actually measuring an
unrelated code path.

So, when the capability check reports unsupported (the current, honest
state of this worktree), this script:

- makes **no** HTTP requests and starts **no** GPU work at all;
- still builds the full settings dict for the requested
  header/body/rho/repeats point (so the log and any downstream tooling see
  exactly the same shape a completed run would have produced);
- appends a single ``status: "blocked"`` entry to the central JSONL log
  with the capability reason;
- exits with a distinct, non-zero, non-exception status
  (``BLOCKED_EXIT_CODE``) so calling scripts/CI can tell "capability
  missing" apart from "ran and failed".

Only once real scheduler dispatch (and a real attention-profile capture /
selected-token recompute hook) exist should ``--allow-real-run`` ever be
passed; even then, this script performs the same capability check again at
that time and will not silently start treating a still-missing hook as
available.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from benchmark.approx_kv.cachecraft_workloads import (
    UNIFIED_CANONICAL_SEGMENT_TOKENS,
    UNIFIED_EXACT_HEADER_TOKENS,
    UNIFIED_LOSSY_BODY_TOKENS,
    UNIFIED_MEM_FRACTION_STATIC,
    UNIFIED_MIN_FORMAL_REPEATS,
    UNIFIED_WARMUP_PASSES,
    build_non_prefix_segmented_workload,
)
from benchmark.approx_kv.metrics import (
    metric_subset,
    parse_prometheus_text,
    telemetry_delta,
    usable_kv_capacity_tokens,
)
from sglang.srt.mem_cache.approx_kv.cachecraft_capability import (
    inspect_scheduler_dispatch_capability,
)

# Distinct from a normal failure (which raises/propagates a traceback): a
# clean, expected "we refuse to fabricate a result" exit.
BLOCKED_EXIT_CODE = 3


def _repeat_count(value: str) -> int:
    repeats = int(value)
    if repeats < UNIFIED_MIN_FORMAL_REPEATS:
        raise argparse.ArgumentTypeError(
            f"repeats must be at least {UNIFIED_MIN_FORMAL_REPEATS}"
        )
    return repeats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30011")
    parser.add_argument(
        "--header-tokens",
        type=int,
        default=64,
        choices=UNIFIED_EXACT_HEADER_TOKENS,
    )
    parser.add_argument(
        "--body-tokens",
        type=int,
        default=1024,
        choices=UNIFIED_LOSSY_BODY_TOKENS,
    )
    parser.add_argument("--target-rho", type=float, required=True)
    parser.add_argument("--filler-tokens", type=int, default=736)
    parser.add_argument("--num-chunks", type=int, default=3)
    parser.add_argument(
        "--segment-tokens",
        type=int,
        default=UNIFIED_CANONICAL_SEGMENT_TOKENS,
    )
    parser.add_argument("--repeats", type=_repeat_count, default=4)
    parser.add_argument("--runner-git-sha", required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--central-log", required=True)
    parser.add_argument(
        "--allow-real-run",
        action="store_true",
        help=(
            "Acknowledge that a real server dispatch + recompute hook may "
            "now exist. Still re-checked via "
            "inspect_scheduler_dispatch_capability() before any network "
            "call is made; passing this flag alone never bypasses the "
            "check."
        ),
    )
    return parser.parse_args()


def append_log(path: str, entry: dict) -> None:
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(entry, sort_keys=True))
        file.write("\n")


def build_settings(args: argparse.Namespace) -> dict:
    prompt_tokens = args.header_tokens + args.body_tokens + 1
    return {
        "mode": "cachecraft",
        "plugin": "cachecraft",
        "target_rho": args.target_rho,
        "header_tokens": args.header_tokens,
        "body_tokens": args.body_tokens,
        "filler_tokens": args.filler_tokens,
        "num_chunks": args.num_chunks,
        "segment_tokens": args.segment_tokens,
        "target_prompt_tokens": prompt_tokens,
        "crosses_1024_token_chunk_boundary": prompt_tokens > 1024,
        "global_warmup_passes": UNIFIED_WARMUP_PASSES,
        "per_setting_warmup_passes": UNIFIED_WARMUP_PASSES,
        "formal_repeats": args.repeats,
        "mem_fraction_static": UNIFIED_MEM_FRACTION_STATIC,
        "scheduler": "S0 LRU",
        "tier": "GPU-only",
        "prefetch": False,
        "runner_git_sha": args.runner_git_sha,
        "image_digest": args.image_digest,
    }


def run_blocked(args: argparse.Namespace, capability) -> int:
    """Record a ``blocked`` central-log entry and exit without touching the
    network or the GPU. Never writes ``args.output``."""
    settings = build_settings(args)
    run_id = (
        "phase4-cachecraft-pressure-blocked-"
        f"rho{args.target_rho:.3f}-"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    append_log(
        args.central_log,
        {
            "run_id": run_id,
            "status": "blocked",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "settings": settings,
            "output": str(Path(args.output).resolve()),
            "reason": capability.reason,
        },
    )
    sys.stderr.write(
        "cachecraft pressure runner: BLOCKED, no server request was made.\n"
        f"reason: {capability.reason}\n"
    )
    return BLOCKED_EXIT_CODE


# ---------------------------------------------------------------------------
# The real-run path below is intentionally written to the exact same
# settings/warmup/repeats/log contract as R1's completed
# ``run_phase4_epic_pressure.py`` so it needs no redesign once real
# scheduler dispatch + a real recompute hook exist. It is unreachable from
# ``main()`` unless ``inspect_scheduler_dispatch_capability()`` reports
# ``supported=True`` -- which it does not in this worktree today -- so it
# is exercised by unit tests with a fake server transport, never against a
# real GPU/server in this session.
# ---------------------------------------------------------------------------


def metric_snapshot(session, base_url: str) -> dict[str, float]:
    response = session.get(f"{base_url}/metrics", timeout=30)
    response.raise_for_status()
    return parse_prometheus_text(response.text)


def request(
    session, base_url: str, input_ids: list[int], metadata: dict | None
) -> dict:
    sampling_params = {"max_new_tokens": 1, "temperature": 0}
    if metadata is not None:
        sampling_params["custom_params"] = {"approx_kv": metadata}
    start = time.perf_counter()
    first = None
    payload = None
    saw_done = False
    with session.post(
        f"{base_url}/generate",
        json={
            "input_ids": input_ids,
            "sampling_params": sampling_params,
            "stream": True,
        },
        stream=True,
        timeout=180,
    ) as response:
        response.raise_for_status()
        for raw_line in response.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8", errors="replace")
            if line == "data: [DONE]":
                saw_done = True
            elif line.startswith("data: ") and first is None:
                first = time.perf_counter()
                payload = json.loads(line[6:])
    if first is None or payload is None or not saw_done:
        raise RuntimeError("incomplete streaming response")
    return {
        "ttft_ms": (first - start) * 1000,
        "cached_tokens": payload["meta_info"]["cached_tokens"],
        "output_ids": payload["output_ids"],
    }


def flush(session, base_url: str, sentinel_salt: int) -> None:
    session.post(f"{base_url}/flush_cache", json={}, timeout=60).raise_for_status()
    time.sleep(0.1)
    request(session, base_url, [80_000 + sentinel_salt, 80_100 + sentinel_salt], None)
    time.sleep(0.1)


def filler_prompt(index: int, length: int) -> list[int]:
    first = 20_000 + index
    return [first] + [
        30_000 + ((index * 977 + offset * 37) % 20_000) for offset in range(length - 1)
    ]


def build_metadata(
    *, operation: str, segments: list[dict], plugin: str | None = None
) -> dict:
    metadata = {
        "operation": operation,
        "model_fingerprint": "qwen3-0.6b-sm75",
        "cache_dtype": "float16",
        "segments": segments,
    }
    if plugin is not None:
        metadata["plugin"] = plugin
    return metadata


def run_round(session, args: argparse.Namespace, round_index: int) -> dict:
    flush(session, args.base_url, round_index)
    baseline = metric_snapshot(session, args.base_url)
    capacity = usable_kv_capacity_tokens(baseline)
    persistent_tokens = args.body_tokens + args.header_tokens + 2
    target_working_tokens = int(math.ceil(args.target_rho * capacity))
    filler_count = max(
        0,
        math.ceil((target_working_tokens - persistent_tokens) / args.filler_tokens),
    )

    workload = build_non_prefix_segmented_workload(
        body_tokens=args.body_tokens,
        header_tokens=args.header_tokens,
        num_chunks=args.num_chunks,
        reorder_seed=round_index if round_index >= 0 else 0,
        max_segment_tokens=args.segment_tokens,
    )

    for chunk in workload.chunks:
        for segment in chunk.segments:
            request(
                session,
                args.base_url,
                list(segment.token_ids) + [900],
                build_metadata(
                    operation="register",
                    segments=[
                        {
                            "content_hash": segment.content_hash,
                            "target_start": 0,
                            "length": segment.length,
                        }
                    ],
                ),
            )

    for filler_index in range(filler_count):
        request(
            session,
            args.base_url,
            filler_prompt(filler_index, args.filler_tokens) + [950],
            None,
        )

    if workload.header_token_ids:
        request(session, args.base_url, list(workload.header_token_ids), None)
    before_target = metric_snapshot(session, args.base_url)

    target_segments = []
    cursor = args.header_tokens
    for chunk_id in workload.target_chunk_order:
        chunk = workload.chunk(chunk_id)
        for segment in chunk.segments:
            target_segments.append(
                {
                    "content_hash": segment.content_hash,
                    "target_start": cursor,
                    "length": segment.length,
                }
            )
            cursor += segment.length

    metadata = build_metadata(
        operation="reuse",
        segments=target_segments,
        plugin="cachecraft",
    )
    target = request(session, args.base_url, list(workload.target_token_ids), metadata)
    after_target = metric_snapshot(session, args.base_url)

    declared_tokens = persistent_tokens + filler_count * args.filler_tokens
    return {
        "round_index": round_index,
        "capacity_tokens": capacity,
        "target_rho": args.target_rho,
        "actual_declared_rho": declared_tokens / capacity,
        "peak_rho_with_target": (declared_tokens + args.body_tokens) / capacity,
        "filler_count": filler_count,
        "declared_working_tokens": declared_tokens,
        "is_reordered_workload": workload.is_reordered,
        "target": target,
        "baseline_metrics": metric_subset(baseline),
        "before_target_metrics": metric_subset(before_target),
        "after_target_metrics": metric_subset(after_target),
        "pressure_delta": telemetry_delta(baseline, before_target),
        "target_delta": telemetry_delta(before_target, after_target),
    }


def run_real(args: argparse.Namespace) -> int:
    import requests

    run_id = (
        f"phase4-cachecraft-pressure-rho{args.target_rho:.3f}-"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    settings = build_settings(args)
    append_log(
        args.central_log,
        {
            "run_id": run_id,
            "status": "running",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "settings": settings,
            "output": str(Path(args.output).resolve()),
        },
    )
    session = requests.Session()
    try:
        warmup = run_round(session, args, -1)
        rows = [run_round(session, args, index) for index in range(args.repeats)]
        values = [row["target"]["ttft_ms"] for row in rows]
        result = {
            "schema_version": 1,
            "run_id": run_id,
            "settings": settings,
            "warmup": warmup,
            "rows": rows,
            "target_p50_ms": statistics.median(values),
            "passed": True,
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        append_log(
            args.central_log,
            {
                "run_id": run_id,
                "status": "completed",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "settings": settings,
                "output": str(output_path.resolve()),
                "result_summary": {"target_p50_ms": result["target_p50_ms"]},
            },
        )
    except Exception as exc:
        append_log(
            args.central_log,
            {
                "run_id": run_id,
                "status": "failed",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "settings": settings,
                "output": str(Path(args.output).resolve()),
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise
    return 0


def main() -> int:
    args = parse_args()
    if args.target_rho <= 0:
        raise ValueError("target_rho must be positive")
    if args.segment_tokens <= 0:
        raise ValueError("segment_tokens must be positive")

    capability = inspect_scheduler_dispatch_capability()
    if not capability:
        return run_blocked(args, capability)
    return run_real(args)


if __name__ == "__main__":
    sys.exit(main())
