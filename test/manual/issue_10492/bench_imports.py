#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

BENCHMARKS = [
    ("import_sglang", "import sglang"),
    (
        "import_model_config",
        "from sglang.srt.configs.model_config import ModelConfig",
    ),
    (
        "import_reasoning_parser",
        "from sglang.srt.parser.reasoning_parser import ReasoningParser",
    ),
    (
        "import_moe_utils",
        "from sglang.srt.layers.moe.utils import initialize_moe_config",
    ),
    (
        "import_engine",
        "from sglang.srt.entrypoints.engine import Engine",
    ),
    (
        "import_scheduler",
        "from sglang.srt.managers.scheduler import Scheduler",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--label", required=True)
    return parser.parse_args()


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "max": max(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
    }


def run_one(
    python_exe: str, repo_root: Path, stmt: str, sample_dir: Path, sample_name: str
) -> dict[str, object]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "python")
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    importtime_file = sample_dir / f"{sample_name}.importtime"

    cmd = [python_exe, "-X", "importtime", "-c", stmt]
    with importtime_file.open("w") as stderr_handle:
        started = time.perf_counter()
        completed = subprocess.Popen(
            cmd,
            env=env,
            cwd=repo_root,
            stdout=subprocess.DEVNULL,
            stderr=stderr_handle,
            text=True,
        )
        _, status, rusage = os.wait4(completed.pid, 0)
        finished = time.perf_counter()

    result = {
        "real": finished - started,
        "user": rusage.ru_utime,
        "sys": rusage.ru_stime,
    }
    result["returncode"] = os.waitstatus_to_exitcode(status)
    result["importtime_file"] = str(importtime_file)
    return result


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sample_root = output_path.parent / f"{args.label}-samples"
    sample_root.mkdir(parents=True, exist_ok=True)

    python_exe = os.environ.get("ISSUE10492_BENCH_PYTHON", "python3")
    report: dict[str, object] = {
        "label": args.label,
        "repo_root": str(repo_root),
        "python_executable": python_exe,
        "warmups": args.warmups,
        "runs": args.runs,
        "benchmarks": [],
    }

    for bench_name, stmt in BENCHMARKS:
        bench_dir = sample_root / bench_name
        bench_dir.mkdir(parents=True, exist_ok=True)

        for idx in range(args.warmups):
            warmup = run_one(
                python_exe,
                repo_root,
                stmt,
                bench_dir,
                f"warmup-{idx + 1}",
            )
            if warmup["returncode"] != 0:
                raise RuntimeError(
                    f"Warm-up failed for {bench_name}: {warmup['returncode']}"
                )

        samples = []
        for idx in range(args.runs):
            sample = run_one(
                python_exe,
                repo_root,
                stmt,
                bench_dir,
                f"run-{idx + 1}",
            )
            if sample["returncode"] != 0:
                raise RuntimeError(
                    f"Measured run failed for {bench_name}: {sample['returncode']}"
                )
            samples.append(sample)

        report["benchmarks"].append(
            {
                "name": bench_name,
                "statement": stmt,
                "samples": samples,
                "stats": {
                    metric: summarize([sample[metric] for sample in samples])
                    for metric in ("real", "user", "sys")
                },
                "importtime_sample": samples[0]["importtime_file"],
            }
        )

    output_path.write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
