#!/usr/bin/env python3
"""Run the eviction-pressure benchmark serially across cache policies.

This suite complements the steady-state multi-workflow benchmark by stressing
cache eviction quality. It launches one server at a time, runs the adversarial
shared-prefix eviction test, and writes JSON/CSV/Markdown summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[2]
BENCH_SCRIPT = ROOT / "benchmark" / "multi_workflow" / "bench_eviction_pressure.py"
DEFAULT_MODEL = "/home/gfy/models/Qwen2.5-3B-Instruct"
DEFAULT_SERVER_PYTHON = "/home/gfy/.conda/envs/sglang-kvflow/bin/python"


POLICY_SERVER_ARGS: Dict[str, Dict[str, object]] = {
    "lru": {"eviction": "lru", "hicache": True, "write_policy": "write_back"},
    "priority": {"eviction": "priority", "hicache": True, "write_policy": "write_back"},
    "tiered": {"eviction": "tiered", "hicache": True, "write_policy": "write_back"},
}


def wait_http_ready(url: str, timeout_s: int = 300) -> None:
    deadline = time.time() + timeout_s
    last_error = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return
        except urllib.error.URLError as exc:
            last_error = str(exc)
        except Exception as exc:  # pragma: no cover
            last_error = str(exc)
        time.sleep(2)
    raise TimeoutError(f"Server not ready after {timeout_s}s: {last_error}")


def wait_server_ready(base_url: str, timeout_s: int = 300) -> None:
    wait_http_ready(f"{base_url}/v1/models", timeout_s=timeout_s)
    wait_http_ready(f"{base_url}/health_generate", timeout_s=timeout_s)


def start_server(args: argparse.Namespace, policy: str, port: int, log_path: Path) -> subprocess.Popen:
    cfg = POLICY_SERVER_ARGS[policy]
    cmd = [
        args.server_python,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model_path,
        "--port",
        str(port),
        "--tp-size",
        "1",
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--max-total-tokens",
        str(args.max_total_tokens),
        "--chunked-prefill-size",
        str(args.chunked_prefill_size),
        "--max-prefill-tokens",
        str(args.max_prefill_tokens),
        "--radix-eviction-policy",
        str(cfg["eviction"]),
        "--enable-cache-report",
        "--disable-cuda-graph",
        "--log-level",
        "info",
    ]
    if args.dtype:
        cmd.extend(["--dtype", str(args.dtype)])
    if args.quantization:
        cmd.extend(["--quantization", str(args.quantization)])

    if cfg["hicache"]:
        cmd.extend(
            [
                "--enable-hierarchical-cache",
                "--hicache-ratio",
                str(args.hicache_ratio),
                "--hicache-write-policy",
                str(cfg["write_policy"]),
                "--hicache-io-backend",
                "direct",
            ]
        )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fp = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=log_fp,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    proc._solo_log_fp = log_fp  # type: ignore[attr-defined]
    return proc


def stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=15)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
            try:
                proc.wait(timeout=5)
            except Exception:
                pass
    log_fp = getattr(proc, "_solo_log_fp", None)
    if log_fp:
        log_fp.close()


def maybe_export_real_templates(args: argparse.Namespace) -> Tuple[Optional[Path], Optional[Dict[str, object]]]:
    if args.real_templates:
        return Path(args.real_templates), None
    if not args.template_cache_input:
        return None, None

    export_script = Path(args.export_script)
    output_path = Path(args.output_dir) / "exports" / "real_templates.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(export_script),
        "--input",
        args.template_cache_input,
        "--output",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, cwd=str(ROOT))
    try:
        export_data = json.loads(output_path.read_text(encoding="utf-8"))
    except Exception:
        export_data = None
    export_stats = None
    if isinstance(export_data, dict):
        export_stats = {
            "total_templates": export_data.get("total_templates"),
            "by_task_type": export_data.get("by_task_type"),
            "by_task_family": export_data.get("by_task_family"),
        }
    return output_path, export_stats


def run_one_policy(
    args: argparse.Namespace,
    policy: str,
    port: int,
    real_templates: Optional[Path],
) -> Dict[str, object]:
    policy_output_dir = Path(args.output_dir) / policy
    policy_output_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.output_dir) / "logs" / f"{policy}.server.log"
    server = start_server(args, policy, port, log_path)
    try:
        base_url = f"http://127.0.0.1:{port}"
        wait_server_ready(base_url, timeout_s=args.server_timeout)
        cmd = [
            sys.executable,
            str(BENCH_SCRIPT),
            "--base-url",
            base_url,
            "--server-name",
            policy,
            "--num-unique",
            str(args.num_unique),
        ]
        if real_templates:
            cmd.extend(["--real-templates", str(real_templates)])
            cmd.extend(["--real-template-role", args.real_template_role])
        if args.verbose:
            cmd.append("--verbose")
        proc = subprocess.run(cmd, check=True, cwd=str(ROOT), capture_output=True, text=True)
        result = parse_single_server_summary(proc.stdout)
        result.update(
            {
                "policy": policy,
                "server_log": str(log_path),
                "raw_output_path": str(write_text(policy_output_dir / "bench_stdout.txt", proc.stdout)),
            }
        )
        write_text(policy_output_dir / "result.json", json.dumps(result, indent=2))
        return result
    finally:
        stop_server(server)


def write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def parse_single_server_summary(stdout: str) -> Dict[str, object]:
    result: Dict[str, object] = {}
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("Server:"):
            result["server"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("Phase-1:"):
            result["phase1_ms"] = float(stripped.split(":", 1)[1].strip().replace("ms", ""))
        elif stripped.startswith("Phase-2:"):
            result["phase2_avg_ms"] = float(stripped.split(":", 1)[1].strip().replace("ms", ""))
        elif stripped.startswith("Phase-3:"):
            result["phase3_ms"] = float(stripped.split(":", 1)[1].strip().replace("ms", ""))
        elif stripped.startswith("Delta:"):
            payload = stripped.split(":", 1)[1].strip()
            ms_part, pct_part = payload.split("(")
            result["delta_ms"] = float(ms_part.strip().replace("ms", ""))
            result["delta_pct"] = float(pct_part.strip().rstrip(")%"))
    if not result:
        raise RuntimeError("Failed to parse bench_eviction_pressure output")
    return result


def build_summary(results: List[Dict[str, object]], baseline_policy: str) -> Dict[str, object]:
    baseline = next((r for r in results if r["policy"] == baseline_policy), None)
    comparisons = []
    baseline_phase3 = baseline.get("phase3_ms") if baseline else None
    for result in results:
        if result["policy"] == baseline_policy:
            continue
        phase3 = result.get("phase3_ms")
        comparisons.append(
            {
                "policy": result["policy"],
                "phase3_speedup_vs_baseline": (
                    baseline_phase3 / phase3 if baseline_phase3 and phase3 else None
                ),
                "phase3_delta_ms_vs_baseline": (
                    baseline_phase3 - phase3 if baseline_phase3 and phase3 else None
                ),
            }
        )
    return {
        "baseline_policy": baseline_policy,
        "results": results,
        "comparisons": comparisons,
    }


def _fmt_float(value: object, digits: int = 2) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return "n/a"


def write_csv_summary(summary: Dict[str, object], output_dir: Path) -> Path:
    csv_path = output_dir / "suite_summary.csv"
    comparisons = {
        item["policy"]: item
        for item in summary.get("comparisons", [])
        if isinstance(item, dict) and item.get("policy")
    }
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "policy",
                "phase1_ms",
                "phase2_avg_ms",
                "phase3_ms",
                "delta_ms",
                "delta_pct",
                "phase3_speedup_vs_baseline",
                "phase3_delta_ms_vs_baseline",
                "server_log",
                "raw_output_path",
            ]
        )
        for result in summary.get("results", []):
            cmp_row = comparisons.get(result.get("policy"), {}) if isinstance(result, dict) else {}
            writer.writerow(
                [
                    result.get("policy"),
                    result.get("phase1_ms"),
                    result.get("phase2_avg_ms"),
                    result.get("phase3_ms"),
                    result.get("delta_ms"),
                    result.get("delta_pct"),
                    cmp_row.get("phase3_speedup_vs_baseline"),
                    cmp_row.get("phase3_delta_ms_vs_baseline"),
                    result.get("server_log"),
                    result.get("raw_output_path"),
                ]
            )
    return csv_path


def write_markdown_summary(summary: Dict[str, object], output_dir: Path) -> Path:
    md_path = output_dir / "suite_summary.md"
    baseline = summary.get("baseline_policy", "n/a")
    comparisons = {
        item["policy"]: item
        for item in summary.get("comparisons", [])
        if isinstance(item, dict) and item.get("policy")
    }
    lines = [
        "# KVFlow Eviction Pressure Suite",
        "",
        f"- Baseline policy: `{baseline}`",
        "",
        "| Policy | Phase-1 ms | Phase-2 avg ms | Phase-3 ms | Delta ms | Delta % | Phase-3 speedup |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in summary.get("results", []):
        cmp_row = comparisons.get(result.get("policy"), {}) if isinstance(result, dict) else {}
        lines.append(
            "| {policy} | {p1} | {p2} | {p3} | {delta_ms} | {delta_pct} | {speedup} |".format(
                policy=result.get("policy", "n/a"),
                p1=_fmt_float(result.get("phase1_ms"), 1),
                p2=_fmt_float(result.get("phase2_avg_ms"), 1),
                p3=_fmt_float(result.get("phase3_ms"), 1),
                delta_ms=_fmt_float(result.get("delta_ms"), 1),
                delta_pct=_fmt_float(result.get("delta_pct"), 1),
                speedup=_fmt_float(cmp_row.get("phase3_speedup_vs_baseline"), 3),
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serial single-GPU eviction-pressure policy suite")
    parser.add_argument("--policies", nargs="+", default=["lru", "priority", "tiered"], choices=sorted(POLICY_SERVER_ARGS))
    parser.add_argument("--baseline-policy", default="lru", choices=sorted(POLICY_SERVER_ARGS))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-unique", type=int, default=60)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--server-python", default=DEFAULT_SERVER_PYTHON)
    parser.add_argument("--base-port", type=int, default=32600)
    parser.add_argument("--server-timeout", type=int, default=300)
    parser.add_argument("--mem-fraction-static", type=float, default=0.75)
    parser.add_argument("--max-total-tokens", type=int, default=40000)
    parser.add_argument("--chunked-prefill-size", type=int, default=2048)
    parser.add_argument("--max-prefill-tokens", type=int, default=4096)
    parser.add_argument("--hicache-ratio", type=float, default=2.0)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--quantization", default=None)
    parser.add_argument("--real-templates", default=None)
    parser.add_argument("--template-cache-input", default=None)
    parser.add_argument("--real-template-role", default="planner")
    parser.add_argument("--real-templates-mode", choices=["mix", "dominant"], default="mix")
    parser.add_argument(
        "--export-script",
        default="/home/gfy/CodeMAS_Project/MAScoder/scripts/export_kv_templates.py",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    real_templates, export_stats = maybe_export_real_templates(args)
    results: List[Dict[str, object]] = []
    for idx, policy in enumerate(args.policies):
        port = args.base_port + idx
        print(f"\n=== Eviction suite: {policy} (port {port}) ===", flush=True)
        results.append(run_one_policy(args, policy, port, real_templates))

    summary = build_summary(results, args.baseline_policy)
    if real_templates:
        summary["real_templates_path"] = str(real_templates)
        summary["real_template_role"] = args.real_template_role
        summary["real_templates_mode"] = args.real_templates_mode
    if export_stats:
        summary["template_export_stats"] = export_stats
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "suite_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    csv_path = write_csv_summary(summary, output_dir)
    md_path = write_markdown_summary(summary, output_dir)
    print(f"\nSuite summary written to {json_path}")
    print(f"CSV summary written to {csv_path}")
    print(f"Markdown summary written to {md_path}")


if __name__ == "__main__":
    main()
