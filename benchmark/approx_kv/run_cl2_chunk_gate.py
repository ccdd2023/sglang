#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
from datetime import datetime, timezone
from pathlib import Path

import requests

from benchmark.approx_kv.phase6.runner import (
    append_jsonl,
    execution_status,
    launch_server,
    machine_manifest,
    source_provenance,
    stop_server,
    wait_ready,
    write_json,
)
from benchmark.approx_kv.phase6.schema import file_sha256, payload_sha256
from benchmark.approx_kv.run_cl1_qualification import (
    VALID_CANDIDATES,
    candidate_k,
    percentile,
    run_paired_setting,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--source-git-sha", required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument(
        "--selected-candidate",
        choices=(*VALID_CANDIDATES, "NONE"),
    )
    parser.add_argument("--selected-k", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--central-log", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=30011)
    parser.add_argument("--mem-fraction-static", type=float, default=0.35)
    parser.add_argument("--server-start-timeout-s", type=float, default=600)
    parser.add_argument("--header-tokens", type=int, default=64)
    parser.add_argument("--segment-tokens", type=int, default=512)
    parser.add_argument("--filler-tokens", type=int, default=512)
    parser.add_argument("--target-rho", type=float, default=2.0)
    parser.add_argument("--formal-repeats", type=int, default=2)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--kv-bytes-per-token", type=int, default=114688)
    return parser.parse_args()


def summarize(rows: list[dict]) -> dict:
    speedups = [
        repeat["request_path_speedup"] for row in rows for repeat in row["formal"]
    ]
    dense = [
        repeat["dense"]["target"]["ttft_ms"] for row in rows for repeat in row["formal"]
    ]
    approx = [
        repeat["approx"]["target"]["ttft_ms"]
        for row in rows
        for repeat in row["formal"]
    ]
    return {
        "median_request_path_speedup": statistics.median(speedups),
        "minimum_request_path_speedup": min(speedups),
        "paired_target_p95_ratio": percentile(approx, 0.95) / percentile(dense, 0.95),
        "all_guardrails_passed": all(
            repeat["passed"] for row in rows for repeat in row["formal"]
        ),
    }


def execute(args: argparse.Namespace, run_id: str) -> dict:
    if args.selected_candidate is not None and args.selected_k is not None:
        raise ValueError("provide selected-candidate or selected-k, not both")
    if args.selected_candidate is not None:
        selected_candidate = args.selected_candidate
    elif args.selected_k is not None:
        if args.selected_k not in {0, 4, 8, 16, 32}:
            raise ValueError("selected-k must be one of 0, 4, 8, 16, 32")
        selected_candidate = f"r1_k{args.selected_k}"
    else:
        raise ValueError("selected-candidate or selected-k is required")
    if args.formal_repeats < 2 or args.restarts < 2:
        raise ValueError("CL2 requires formal=2 and restarts=2 or greater")
    provenance = source_provenance(args.source_git_sha)
    observed_sha = provenance["source_git_sha"]

    candidates = tuple(
        dict.fromkeys(
            (
                "r1_k0",
                *(() if selected_candidate == "NONE" else (selected_candidate,)),
            )
        )
    )
    body_values = (768, 1024)
    chunk_values = (1024, 4096)
    results = []
    servers = []
    for chunk_size in chunk_values:
        for candidate in candidates:
            k = candidate_k(candidate)
            for restart_index in range(args.restarts):
                env = {"SGLANG_APPROX_KV_CORE": "1"}
                if candidate != "r0":
                    env.update(
                        {
                            "SGLANG_APPROX_KV_EPIC": "1",
                            "SGLANG_APPROX_KV_EPIC_K": str(k),
                        }
                    )
                log_path = args.log_dir / (
                    f"cl2-chunk{chunk_size}-{candidate}-" f"restart{restart_index}.log"
                )
                server = launch_server(
                    model=args.model,
                    model_revision=args.model_revision,
                    port=args.port,
                    mem_fraction_static=args.mem_fraction_static,
                    chunked_prefill_size=chunk_size,
                    policy="lru",
                    log_path=log_path,
                    plugin_env=env,
                    server_seed=101 + restart_index,
                )
                try:
                    wait_ready(
                        server,
                        port=args.port,
                        timeout_s=args.server_start_timeout_s,
                    )
                    for body_tokens in body_values:
                        run_paired_setting(
                            args,
                            candidate=candidate,
                            body_tokens=body_tokens,
                            restart_index=restart_index,
                            repeat_index=-1,
                        )
                        formal = [
                            run_paired_setting(
                                args,
                                candidate=candidate,
                                body_tokens=body_tokens,
                                restart_index=restart_index,
                                repeat_index=repeat_index,
                            )
                            for repeat_index in range(args.formal_repeats)
                        ]
                        results.append(
                            {
                                "chunked_prefill_size": chunk_size,
                                "candidate": candidate,
                                "body_tokens": body_tokens,
                                "restart_index": restart_index,
                                "formal": formal,
                            }
                        )
                    servers.append(
                        {
                            "chunked_prefill_size": chunk_size,
                            "candidate": candidate,
                            "restart_index": restart_index,
                            "server_argv": list(server.command),
                            "plugin_env": server.plugin_env,
                            "log_path": str(log_path),
                        }
                    )
                finally:
                    stop_server(server)

    summaries = {}
    for chunk_size in chunk_values:
        summaries[str(chunk_size)] = {}
        for candidate in candidates:
            summaries[str(chunk_size)][candidate] = {}
            for body_tokens in body_values:
                summaries[str(chunk_size)][candidate][str(body_tokens)] = summarize(
                    [
                        row
                        for row in results
                        if row["chunked_prefill_size"] == chunk_size
                        and row["candidate"] == candidate
                        and row["body_tokens"] == body_tokens
                    ]
                )

    gate_candidate = selected_candidate if selected_candidate != "NONE" else "r1_k0"
    eligible = []
    for chunk_size in chunk_values:
        rows = summaries[str(chunk_size)][gate_candidate]
        if all(
            row["all_guardrails_passed"] and row["paired_target_p95_ratio"] <= 1.05
            for row in rows.values()
        ):
            eligible.append(
                (
                    min(row["median_request_path_speedup"] for row in rows.values()),
                    chunk_size,
                )
            )
    selected_chunk = max(eligible)[1] if eligible else None
    payload = {
        "schema_version": 1,
        "run_id": run_id,
        "phase": "CL2",
        "source_git_sha": observed_sha,
        "source_tree_sha": provenance["source_tree_sha"],
        "result_git_sha": None,
        "result_commit_status": "pending_result_commit",
        "model": args.model,
        "model_revision": args.model_revision,
        "image_digest": args.image_digest,
        "machine": machine_manifest(),
        "settings": {
            "arms": ["dense", *candidates],
            "body_tokens": list(body_values),
            "chunked_prefill_size": list(chunk_values),
            "rho_logical_demand": args.target_rho,
            "restarts": args.restarts,
            "formal_repeats": args.formal_repeats,
        },
        "servers": servers,
        "results": results,
        "summaries": summaries,
        "selected_chunked_prefill_size": selected_chunk,
        "selected_candidate": selected_candidate,
        "status": "valid" if selected_chunk is not None else "inconclusive",
        "disclosure": (
            "P6-4 must use the selected chunk size. If a provisional chunk "
            "was used earlier and differs, affected feasibility cells must rerun."
        ),
    }
    payload["raw_sha256"] = payload_sha256(payload)
    return payload


def main() -> int:
    args = parse_args()
    run_id = datetime.now(timezone.utc).strftime("cl2-%Y%m%dT%H%M%SZ")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    append_jsonl(
        args.central_log,
        {
            "run_id": run_id,
            "phase": "CL2",
            "status": "running",
            "output": str(args.output.resolve()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )
    try:
        payload = execute(args, run_id)
        for manifest in payload["servers"]:
            manifest["log_sha256"] = file_sha256(Path(manifest["log_path"]))
        payload.pop("raw_sha256", None)
        payload["raw_sha256"] = payload_sha256(payload)
        write_json(args.output, payload)
        append_jsonl(
            args.central_log,
            {
                "run_id": run_id,
                "phase": "CL2",
                "status": "completed",
                "raw_sha256": payload["raw_sha256"],
                "output": str(args.output.resolve()),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
        return 0
    except (
        KeyError,
        MemoryError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        requests.RequestException,
    ) as exc:
        status = execution_status(exc)
        failure = {
            "schema_version": 1,
            "run_id": run_id,
            "phase": "CL2",
            "source_git_sha": args.source_git_sha,
            "image_digest": args.image_digest,
            "status": "invalid",
            "execution_status": status,
            "error": f"{type(exc).__name__}: {exc}",
        }
        failure["raw_sha256"] = payload_sha256(failure)
        write_json(args.output, failure)
        append_jsonl(
            args.central_log,
            {
                "run_id": run_id,
                "phase": "CL2",
                "status": status,
                "error": failure["error"],
                "raw_sha256": failure["raw_sha256"],
                "output": str(args.output.resolve()),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
