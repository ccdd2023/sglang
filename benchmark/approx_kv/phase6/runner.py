from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import requests

from benchmark.approx_kv.metrics import parse_prometheus_text

REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class ServerProcess:
    process: subprocess.Popen
    log_file: Any
    log_path: Path
    command: tuple[str, ...]
    plugin_env: dict[str, str]


def current_git_sha() -> str:
    return subprocess.check_output(
        ("git", "rev-parse", "HEAD"),
        cwd=REPO_ROOT,
        text=True,
    ).strip()


def source_provenance(expected_git_sha: str) -> dict[str, str]:
    observed_git_sha = current_git_sha()
    if observed_git_sha != expected_git_sha:
        raise RuntimeError(
            f"source SHA mismatch: expected {expected_git_sha}, "
            f"got {observed_git_sha}"
        )
    status = subprocess.check_output(
        (
            "git",
            "status",
            "--porcelain",
            "--untracked-files=all",
        ),
        cwd=REPO_ROOT,
        text=True,
    )
    if status.strip():
        raise RuntimeError("benchmark source tree must be clean before execution")
    tree_sha = subprocess.check_output(
        ("git", "rev-parse", "HEAD^{tree}"),
        cwd=REPO_ROOT,
        text=True,
    ).strip()
    return {
        "source_git_sha": observed_git_sha,
        "source_tree_sha": tree_sha,
    }


def launch_server(
    *,
    model: str,
    model_revision: str,
    port: int,
    mem_fraction_static: float,
    chunked_prefill_size: int,
    policy: str,
    log_path: Path,
    plugin_env: Mapping[str, str],
    max_total_tokens: int | None = None,
    server_seed: int = 17,
    attention_backend: str = "torch_native",
    sampling_backend: str = "pytorch",
    extra_args: Sequence[str] = (),
) -> ServerProcess:
    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        model,
        "--revision",
        model_revision,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--tp-size",
        "1",
        "--mem-fraction-static",
        str(mem_fraction_static),
        "--chunked-prefill-size",
        str(chunked_prefill_size),
        "--max-prefill-tokens",
        str(chunked_prefill_size),
        "--max-running-requests",
        "2",
        "--attention-backend",
        attention_backend,
        "--sampling-backend",
        sampling_backend,
        "--cuda-graph-backend-decode",
        "disabled",
        "--cuda-graph-backend-prefill",
        "disabled",
        "--radix-eviction-policy",
        policy,
        "--enable-cache-report",
        "--enable-metrics",
        "--random-seed",
        str(server_seed),
        "--log-level",
        "warning",
    ]
    if max_total_tokens is not None:
        command.extend(("--max-total-tokens", str(max_total_tokens)))
    command.extend(extra_args)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w")
    environment = os.environ.copy()
    environment.update(plugin_env)
    environment.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(REPO_ROOT / "python")
        if not python_path
        else f"{REPO_ROOT / 'python'}{os.pathsep}{python_path}"
    )
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=environment,
        start_new_session=True,
    )
    return ServerProcess(
        process=process,
        log_file=log_file,
        log_path=log_path,
        command=tuple(command),
        plugin_env=dict(plugin_env),
    )


def tail_text(path: Path, lines: int = 160) -> str:
    if not path.exists():
        return ""
    return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])


def wait_ready(
    server: ServerProcess,
    *,
    port: int,
    timeout_s: float,
) -> None:
    deadline = time.monotonic() + timeout_s
    url = f"http://127.0.0.1:{port}/health_generate"
    while time.monotonic() < deadline:
        if server.process.poll() is not None:
            server.log_file.flush()
            raise RuntimeError(
                f"server exited with code {server.process.returncode}\n"
                f"{tail_text(server.log_path)}"
            )
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise TimeoutError(f"server did not become healthy\n{tail_text(server.log_path)}")


def stop_server(server: ServerProcess) -> None:
    process = server.process
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
    server.log_file.close()


def base_url(port: int) -> str:
    return f"http://127.0.0.1:{port}"


def metric_text(port: int) -> str:
    response = requests.get(f"{base_url(port)}/metrics", timeout=30)
    response.raise_for_status()
    return response.text


def metric_snapshot(port: int) -> dict[str, float]:
    return parse_prometheus_text(metric_text(port))


def flush_cache(port: int) -> None:
    response = requests.post(
        f"{base_url(port)}/flush_cache?timeout=30",
        json={},
        timeout=60,
    )
    response.raise_for_status()
    requests.get(f"{base_url(port)}/health_generate", timeout=30).raise_for_status()
    time.sleep(0.25)


def generate(
    *,
    port: int,
    input_ids: list[int],
    max_new_tokens: int,
    custom_params: Mapping[str, Any] | None = None,
    extra_key: str | None = None,
    timeout_s: float = 300,
) -> dict[str, Any]:
    sampling_params: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "temperature": 0,
    }
    if custom_params:
        sampling_params["custom_params"] = dict(custom_params)
    payload: dict[str, Any] = {
        "input_ids": input_ids,
        "sampling_params": sampling_params,
        "stream": False,
    }
    if extra_key is not None:
        payload["extra_key"] = extra_key
    started = time.perf_counter()
    response = requests.post(
        f"{base_url(port)}/generate",
        json=payload,
        timeout=timeout_s,
    )
    response.raise_for_status()
    body = response.json()
    return {
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        "cached_tokens": int(body["meta_info"]["cached_tokens"]),
        "output_ids": list(body["output_ids"]),
        "finish_reason": body["meta_info"].get("finish_reason"),
    }


def stream_generate(
    *,
    port: int,
    input_ids: list[int],
    max_new_tokens: int = 1,
    custom_params: Mapping[str, Any] | None = None,
    extra_key: str | None = None,
    timeout_s: float = 300,
) -> dict[str, Any]:
    sampling_params: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "temperature": 0,
    }
    if custom_params:
        sampling_params["custom_params"] = dict(custom_params)
    payload: dict[str, Any] = {
        "input_ids": input_ids,
        "sampling_params": sampling_params,
        "stream": True,
    }
    if extra_key is not None:
        payload["extra_key"] = extra_key
    started = time.perf_counter()
    first_at = None
    first_payload = None
    last_payload = None
    saw_done = False
    with requests.post(
        f"{base_url(port)}/generate",
        json=payload,
        stream=True,
        timeout=timeout_s,
    ) as response:
        response.raise_for_status()
        for raw_line in response.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8", errors="replace")
            if line == "data: [DONE]":
                saw_done = True
                continue
            if not line.startswith("data: "):
                continue
            body = json.loads(line[6:])
            last_payload = body
            if first_payload is None:
                first_payload = body
                first_at = time.perf_counter()
    if first_payload is None or first_at is None or not saw_done:
        raise RuntimeError("incomplete streaming generation response")
    output_ids = (
        last_payload["output_ids"]
        if last_payload is not None
        else first_payload["output_ids"]
    )
    return {
        "ttft_ms": (first_at - started) * 1000.0,
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        "cached_tokens": int(first_payload["meta_info"]["cached_tokens"]),
        "output_ids": list(output_ids),
        "finish_reason": first_payload["meta_info"].get("finish_reason"),
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True) + "\n")


def execution_status(error: BaseException) -> str:
    message = str(error).lower()
    if (
        "driver/library version mismatch" in message
        or "failed to initialize nvml" in message
    ):
        return "blocked"
    return "failed"


def machine_manifest() -> dict[str, Any]:
    import torch
    import transformers

    properties = torch.cuda.get_device_properties(0)
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": torch.cuda.get_device_capability(0),
        "gpu_memory_bytes": properties.total_memory,
        "transformers": transformers.__version__,
    }
