#!/usr/bin/env python3
"""Run one native lossy-KV backend under the common rolling SWE agent."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import statistics
from datetime import datetime, timezone
from pathlib import Path

import requests

from benchmark.multi_workflow import run_bridge_reuse_agent_experiment as bridge
from benchmark.multi_workflow.runtime_paths import RuntimePaths


PROJECT = Path(__file__).resolve().parents[2]
PATHS = RuntimePaths.from_project(PROJECT)
CONFIG = PROJECT / "benchmark/multi_workflow/swebench_common_qwen25_agent.yaml"
DEFAULT_MODEL = Path(
    os.environ.get(
        "IMPACTKV_COMMON_MODEL",
        str(Path.home() / "models/Qwen2.5-Coder-7B-Instruct"),
    )
).expanduser().resolve()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def health(url: str, expected_backend: str) -> dict:
    response = requests.get(url.rstrip("/") + "/health", timeout=5)
    response.raise_for_status()
    value = response.json()
    if value.get("backend") != expected_backend:
        raise RuntimeError(
            f"backend mismatch: expected {expected_backend}, got {value}"
        )
    return value


def mini_command(args: argparse.Namespace, run_dir: Path) -> list[str]:
    if args.container_backend == "enroot":
        prefix = [
            str(PATHS.mini_python),
            str(PROJECT / "benchmark/multi_workflow/run_minisweagent_enroot.py"),
        ]
    else:
        prefix = [str(PATHS.mini_executable), "swebench"]
    command = [
        *prefix,
        "--subset",
        str(args.dataset),
        "--split",
        "test",
        "--output",
        str(run_dir),
        "--workers",
        "1",
        "--config",
        "swebench.yaml",
        "--config",
        str(CONFIG),
        "--config",
        f"model.model_name=common_native/{args.backend}-{args.mode}",
        "--config",
        f"model.native_backend_url={args.backend_url}",
        "--config",
        f"model.native_backend_name={args.backend}-{args.mode}",
        "--config",
        f"model.tokenizer_json_path={args.model / 'tokenizer.json'}",
        "--config",
        f"model.reuse_client_ledger_path={run_dir / 'CLIENT_LEDGER.jsonl'}",
        "--config",
        f"agent.step_limit={args.step_limit}",
    ]
    if args.instance:
        command.extend(["--filter", f"^{args.instance}$"])
    return command


def summarize_native(run_dir: Path, arm: str) -> dict:
    rows = bridge.load_jsonl(run_dir / "CLIENT_LEDGER.jsonl")
    requests_ = [row for row in rows if row.get("event") == "request_complete"]
    metrics = [row.get("native_backend_metrics") or {} for row in requests_]
    ttfts = [
        float(row["ttft_seconds"]) * 1000
        for row in requests_
        if row.get("ttft_seconds") is not None
    ]
    elapsed = [float(row["request_elapsed_seconds"]) * 1000 for row in requests_]
    summary = {
        "arm": arm,
        "requests": len(requests_),
        "median_ttft_ms": statistics.median(ttfts) if ttfts else None,
        "p95_ttft_ms": (
            sorted(ttfts)[max(0, int(0.95 * len(ttfts)) - 1)] if ttfts else None
        ),
        "median_request_elapsed_ms": statistics.median(elapsed) if elapsed else None,
        "physical_reuse_requests": sum(bool(row.get("physical_reuse")) for row in metrics),
        "reused_k_tokens": sum(int(row.get("reused_k_tokens") or 0) for row in metrics),
        "reused_v_tokens": sum(int(row.get("reused_v_tokens") or 0) for row in metrics),
        "recomputed_tokens": sum(int(row.get("recomputed_tokens") or 0) for row in metrics),
        "cache_build_ms": sum(float(row.get("cache_build_ms") or 0) for row in metrics),
        "fallback_requests": sum(bool(row.get("fallback_reason")) for row in metrics),
        "input_identity_rows": sum(bool(row.get("input_ids_sha256")) for row in requests_),
    }
    write_json(run_dir / "RUNTIME_SUMMARY.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cacheblend", "kvcomm"), required=True)
    parser.add_argument("--mode", choices=("dense", "reuse"), required=True)
    parser.add_argument("--backend-url", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--step-limit", type=int, default=32)
    parser.add_argument("--instance")
    parser.add_argument("--official", action="store_true")
    parser.add_argument("--container-backend", choices=("docker", "enroot"), default="enroot")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(args.output)
    if not (args.model / "tokenizer.json").is_file():
        raise FileNotFoundError(args.model / "tokenizer.json")
    backend_health = health(args.backend_url, args.backend)
    args.output.mkdir(parents=True)
    run_registration = {
        "schema_version": 1,
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend": args.backend,
        "mode": args.mode,
        "backend_health": backend_health,
        "backend_url": args.backend_url,
        "model": str(args.model),
        "dataset": str(args.dataset),
        "snapshot": str(args.snapshot),
        "registration": str(args.registration),
        "prompt_contract": (
            "common rolling6 messages and Qwen2.5 tool template; backend receives "
            "pre-rendered token IDs and must echo the identical token hash"
        ),
        "protocol": {
            "model_dtype": os.environ.get("KVFLOW_REPRO_DTYPE", "native"),
            "temperature": 0,
            "step_limit": args.step_limit,
            "prompt_token_limit": 28000,
            "context_length": 32768,
            "max_new_tokens": 2048,
            "repetition_penalty": 1.05,
            "workers": 1,
            "prefetch": False,
        },
        "source_sha256": {
            str(CONFIG.relative_to(PROJECT)): sha256(CONFIG),
            str(Path(__file__).resolve().relative_to(PROJECT)): sha256(Path(__file__).resolve()),
            "benchmark/multi_workflow/bridge_reuse_litellm_model.py": sha256(
                PROJECT / "benchmark/multi_workflow/bridge_reuse_litellm_model.py"
            ),
        },
    }
    write_json(args.output / "RUN_REGISTRATION.json", run_registration)
    command = mini_command(args, args.output)
    write_json(args.output / "AGENT_COMMAND.json", command)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT)
    subprocess.run(command, cwd=PROJECT, env=env, check=True)
    summarize_native(args.output, f"{args.backend}-{args.mode}")
    if args.official:
        instance_ids = [args.instance] if args.instance else None
        bridge.run_official_evaluation(
            output=args.output.parent,
            run_dir=args.output,
            arm=f"{args.backend}-{args.mode}",
            instance_ids=instance_ids,
            registration=args.registration,
            snapshot=args.snapshot,
            container_backend=args.container_backend,
        )


if __name__ == "__main__":
    main()
