#!/usr/bin/env python3
"""Batch local SWE-bench environment setup/testing."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = PROJECT / "results" / "repo_level_datasets" / "swe_verified_10_instances.json"
DEFAULT_OUT = PROJECT / "results" / "swebench_local_envs" / "expanded_10_report.json"
RUNNER = PROJECT / "benchmark" / "multi_workflow" / "setup_swebench_local_env.py"
PYTHON = Path("/home/gfy/.conda/envs/sglang-kvflow/bin/python")


def run(cmd: list[str], timeout: int) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(shlex.quote(x) for x in cmd)}")
    return subprocess.run(cmd, cwd=str(PROJECT), text=True, capture_output=True, timeout=timeout)


def load_rows(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--mode", choices=["base", "gold"], default="gold")
    parser.add_argument("--max-cases", type=int, default=10)
    parser.add_argument("--max-fail-tests", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--skip-existing-pass", action="store_true")
    parser.add_argument("--recreate-env", action="store_true")
    parser.add_argument("--start-index", type=int, default=0)
    args = parser.parse_args()

    rows = load_rows(args.dataset)[args.start_index : args.start_index + args.max_cases]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, Any] = {}
    if args.out.exists():
        try:
            prev = json.loads(args.out.read_text(encoding="utf-8"))
            existing = {item["instance_id"]: item for item in prev.get("results", [])}
        except Exception:
            existing = {}

    results = []
    for row in rows:
        instance_id = row["instance_id"]
        if args.skip_existing_pass and existing.get(instance_id, {}).get("returncode") == 0:
            results.append(existing[instance_id])
            continue
        started = time.time()
        cmd = [
            str(PYTHON),
            str(RUNNER),
            "--dataset",
            str(args.dataset),
            "--instance-id",
            instance_id,
            "--mode",
            args.mode,
            "--max-fail-tests",
            str(args.max_fail_tests),
            "--timeout",
            str(args.timeout),
        ]
        if args.recreate_env:
            cmd.append("--recreate-env")
        try:
            proc = run(cmd, timeout=args.timeout + 600)
            result = {
                "instance_id": instance_id,
                "repo": row.get("repo", ""),
                "version": row.get("version", ""),
                "mode": args.mode,
                "returncode": proc.returncode,
                "elapsed_sec": round(time.time() - started, 2),
                "stdout_tail": proc.stdout[-5000:],
                "stderr_tail": proc.stderr[-3000:],
            }
        except subprocess.TimeoutExpired as exc:
            result = {
                "instance_id": instance_id,
                "repo": row.get("repo", ""),
                "version": row.get("version", ""),
                "mode": args.mode,
                "returncode": None,
                "elapsed_sec": round(time.time() - started, 2),
                "stdout_tail": (exc.stdout or "")[-5000:] if isinstance(exc.stdout, str) else "",
                "stderr_tail": "timeout",
            }
        results.append(result)
        summary = {
            "dataset": str(args.dataset),
            "mode": args.mode,
            "max_fail_tests": args.max_fail_tests,
            "timeout": args.timeout,
            "results": results,
        }
        args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2, ensure_ascii=False))

    passes = sum(1 for item in results if item.get("returncode") == 0)
    print(f"Batch complete: {passes}/{len(results)} passed. Saved: {args.out}")
    return 0 if passes == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
