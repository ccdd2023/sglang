#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

CAPACITY_PATTERN = re.compile(r"#tokens:\s*(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--mem-fraction-static", type=float, default=0.7)
    parser.add_argument("--chunked-prefill-size", type=int, default=8192)
    parser.add_argument("--attention-backend")
    parser.add_argument("--code-blocks", default="80,160,240")
    parser.add_argument("--role-prefix-blocks", type=int, default=16)
    parser.add_argument("--resident-variants", type=int, default=5)
    parser.add_argument("--warmup-repeats", type=int, default=1)
    parser.add_argument("--measured-repeats", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def wait_ready(port: int, process: subprocess.Popen, timeout_s: int = 600) -> None:
    deadline = time.monotonic() + timeout_s
    url = f"http://127.0.0.1:{port}/health_generate"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"server exited during startup with code {process.returncode}"
            )
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    return
        except Exception:
            pass
        time.sleep(5)
    raise TimeoutError("server did not become healthy")


def stop_server(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def read_capacity(log_path: Path) -> int:
    text = log_path.read_text(errors="replace")
    matches = CAPACITY_PATTERN.findall(text)
    if not matches:
        raise RuntimeError("unable to find GPU KV token capacity in server log")
    return int(matches[-1])


def server_command(args: argparse.Namespace, mode: str) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--chunked-prefill-size",
        str(args.chunked_prefill_size),
        "--max-prefill-tokens",
        str(args.chunked_prefill_size),
        "--enable-cache-report",
        "--cuda-graph-backend-decode",
        "disabled",
        "--cuda-graph-backend-prefill",
        "disabled",
    ]
    if args.attention_backend:
        command.extend(["--attention-backend", args.attention_backend])
    if mode == "dense":
        command.append("--disable-radix-cache")
    return command


def benchmark_command(
    args: argparse.Namespace,
    *,
    mode: str,
    blocks: int,
    capacity: int,
    output: Path,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "benchmark.approx_kv.bench_ttft_pressure",
        "--base-url",
        f"http://127.0.0.1:{args.port}",
        "--model",
        "default",
        "--trace",
        "long_distance",
        "--rounds",
        "1",
        "--code-blocks",
        str(blocks),
        "--role-prefix-blocks",
        str(args.role_prefix_blocks),
        "--code-tokens",
        str(blocks * 26),
        "--role-prefix-tokens",
        str(args.role_prefix_blocks * 32),
        "--resident-variants",
        str(args.resident_variants),
        "--gpu-kv-capacity-tokens",
        str(capacity),
        "--warmup-repeats",
        str(args.warmup_repeats),
        "--measured-repeats",
        str(args.measured_repeats),
        "--max-new-tokens",
        "1",
        "--output",
        str(output),
    ]
    if mode == "raw":
        command.extend(
            [
                "--tokenizer-path",
                args.model,
                "--approx-mode",
                "register_then_raw_speed",
            ]
        )
    return command


def machine_manifest() -> dict:
    torch_info = subprocess.check_output(
        [
            sys.executable,
            "-c",
            (
                "import json, torch; "
                "print(json.dumps({"
                "'torch': torch.__version__, "
                "'cuda': torch.version.cuda, "
                "'gpu': torch.cuda.get_device_name(), "
                "'capability': torch.cuda.get_device_capability(), "
                "'memory': torch.cuda.get_device_properties(0).total_memory"
                "}))"
            ),
        ],
        text=True,
    )
    git_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()
    return {
        "git_sha": git_sha,
        "torch": json.loads(torch_info),
        "python": sys.version,
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    blocks = [int(value) for value in args.code_blocks.split(",")]
    results = []

    for mode in ("dense", "exact", "raw"):
        log_path = args.output_dir / f"server-{mode}.log"
        environment = os.environ.copy()
        if mode == "raw":
            environment.update(
                {
                    "SGLANG_APPROX_KV_CORE": "1",
                    "SGLANG_APPROX_KV_LOSSY": "1",
                    "SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE": "0",
                }
            )
        with log_path.open("w") as log_file:
            process = subprocess.Popen(
                server_command(args, mode),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=environment,
            )
        try:
            wait_ready(args.port, process)
            capacity = read_capacity(log_path)
            for block_count in blocks:
                output = args.output_dir / f"{mode}-blocks{block_count}.json"
                subprocess.run(
                    benchmark_command(
                        args,
                        mode=mode,
                        blocks=block_count,
                        capacity=capacity,
                        output=output,
                    ),
                    check=True,
                )
                results.append(
                    {
                        "mode": mode,
                        "blocks": block_count,
                        "capacity": capacity,
                        "output": str(output),
                    }
                )
        finally:
            stop_server(process)

    manifest = {
        "machine": machine_manifest(),
        "config": vars(args) | {"output_dir": str(args.output_dir)},
        "runs": results,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
