#!/usr/bin/env python3
"""Run multi-workflow KVFlow benchmarks serially on a single GPU.

This script removes the common benchmarking pitfall where multiple schedulers
compete for the same GPU. It launches exactly one sglang server at a time,
runs the benchmark, collects the newest JSON result, then tears the server
down before moving to the next policy.
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
BENCH_SCRIPT = ROOT / "benchmark" / "multi_workflow" / "bench_multi_workflow.py"
DEFAULT_MODEL = "/home/gfy/models/Qwen2.5-3B-Instruct"
DEFAULT_SERVER_PYTHON = "/home/gfy/.conda/envs/sglang-kvflow/bin/python"


POLICY_SERVER_ARGS: Dict[str, Dict[str, object]] = {
    "lru_nocache": {
        "eviction": "lru",
        "hicache": False,
        "write_policy": None,
        "benchmark_config": "lru_nocache",
    },
    "lru_wb_only": {
        "eviction": "lru",
        "hicache": True,
        "write_policy": "write_back",
        "benchmark_config": "lru_wb_only",
    },
    "lru_wb_pf": {
        "eviction": "lru",
        "hicache": True,
        "write_policy": "write_back",
        "benchmark_config": "lru_wb_pf",
    },
    "priority_wb_only": {
        "eviction": "priority",
        "hicache": True,
        "write_policy": "write_back",
        "benchmark_config": "priority_wb_only",
    },
    "priority_dag": {
        "eviction": "priority",
        "hicache": True,
        "write_policy": "write_back",
        "benchmark_config": "priority_dag",
    },
    "priority_pf_lock": {
        "eviction": "priority",
        "hicache": True,
        "write_policy": "write_back",
        "benchmark_config": "priority_pf_lock",
    },
    "kvflow": {
        "eviction": "priority",
        "hicache": True,
        "write_policy": "write_back",
        "benchmark_config": "kvflow",
    },
    "hicache": {
        "eviction": "lru",
        "hicache": True,
        "write_policy": "write_through",
        "benchmark_config": "hicache",
    },
    "hicache90k": {
        "eviction": "lru",
        "hicache": True,
        "write_policy": "write_back",
        "benchmark_config": "hicache90k",
    },
    "tiered_wb_only": {
        "eviction": "tiered",
        "hicache": True,
        "write_policy": "write_back",
        "benchmark_config": "priority_wb_only",
    },
}


def wait_http_ready(url: str, timeout_s: int = 300) -> None:
    deadline = time.time() + timeout_s
    last_error: Optional[str] = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return
        except urllib.error.URLError as exc:
            last_error = str(exc)
        except Exception as exc:  # pragma: no cover - defensive
            last_error = str(exc)
        time.sleep(2)
    raise TimeoutError(f"Server not ready after {timeout_s}s: {last_error}")


def wait_server_ready(base_url: str, timeout_s: int = 300) -> None:
    # /v1/models becomes ready earlier than generation; bench_multi_workflow
    # performs a /health_generate check, so we gate on both endpoints here.
    wait_http_ready(f"{base_url}/v1/models", timeout_s=timeout_s)
    wait_http_ready(f"{base_url}/health_generate", timeout_s=timeout_s)


def start_server(
    policy: str,
    server_python: str,
    model_path: str,
    port: int,
    hicache_ratio: float,
    mem_fraction_static: float,
    max_total_tokens: int,
    chunked_prefill_size: int,
    max_prefill_tokens: int,
    log_path: Path,
    dtype: Optional[str] = None,
    quantization: Optional[str] = None,
) -> subprocess.Popen:
    cfg = POLICY_SERVER_ARGS[policy]
    cmd = [
        server_python,
        "-m",
        "sglang.launch_server",
        "--model-path",
        model_path,
        "--port",
        str(port),
        "--tp-size",
        "1",
        "--mem-fraction-static",
        str(mem_fraction_static),
        "--max-total-tokens",
        str(max_total_tokens),
        "--chunked-prefill-size",
        str(chunked_prefill_size),
        "--max-prefill-tokens",
        str(max_prefill_tokens),
        "--radix-eviction-policy",
        str(cfg["eviction"]),
        "--enable-cache-report",
        "--disable-cuda-graph",
        "--log-level",
        "info",
    ]
    if dtype:
        cmd.extend(["--dtype", str(dtype)])
    if quantization:
        cmd.extend(["--quantization", str(quantization)])
    if cfg["hicache"]:
        cmd.extend(
            [
                "--enable-hierarchical-cache",
                "--hicache-ratio",
                str(hicache_ratio),
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


def maybe_export_real_templates(
    args: argparse.Namespace,
) -> Tuple[Optional[Path], Optional[Dict[str, object]]]:
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
    subprocess.run(cmd, check=True)
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


def newest_json(output_dir: Path, before: set[Path]) -> Path:
    candidates = {p for p in output_dir.glob("*.json")} - before
    if not candidates:
        candidates = set(output_dir.glob("*.json"))
    if not candidates:
        raise FileNotFoundError(f"No JSON results found in {output_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def run_one_policy(
    args: argparse.Namespace,
    policy: str,
    port: int,
    real_templates: Optional[Path],
    baseline_json: Optional[Path],
) -> Dict[str, object]:
    attempt_specs = build_attempt_specs(args)
    policy_output_dir = Path(args.output_dir) / policy
    policy_output_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.output_dir) / "logs" / f"{policy}.server.log"
    failures: List[Dict[str, object]] = []
    for attempt_idx, attempt in enumerate(attempt_specs, start=1):
        benchmark_config = str(POLICY_SERVER_ARGS[policy].get("benchmark_config", policy))
        print(
            "Attempt {idx}/{total} for {policy} (bench={bench}): wf={wf}, tier0={t0}, tier1={t1}, tier2={t2}, suffix={sfx}, out={out}".format(
                idx=attempt_idx,
                total=len(attempt_specs),
                policy=policy,
                bench=benchmark_config,
                wf=attempt["num_workflows"],
                t0=attempt["tier0_len"],
                t1=attempt["tier1_len"],
                t2=attempt["tier2_len"],
                sfx=attempt["suffix_len"],
                out=attempt["output_len"],
            ),
            flush=True,
        )
        server = start_server(
            policy=policy,
            server_python=args.server_python,
            model_path=args.model_path,
            port=port,
            hicache_ratio=args.hicache_ratio,
            mem_fraction_static=args.mem_fraction_static,
            max_total_tokens=args.max_total_tokens,
            chunked_prefill_size=args.chunked_prefill_size,
            max_prefill_tokens=args.max_prefill_tokens,
            dtype=args.dtype,
            quantization=args.quantization,
            log_path=log_path,
        )
        before = set(policy_output_dir.glob("*.json"))
        try:
            wait_server_ready(f"http://127.0.0.1:{port}", timeout_s=args.server_timeout)
            cmd = [
                sys.executable,
                str(BENCH_SCRIPT),
                "--config",
                benchmark_config,
                "--workflow-type",
                args.workflow_type,
                "--num-workflows",
                str(attempt["num_workflows"]),
                "--agents-per-workflow",
                str(args.agents_per_workflow),
                "--tier0-len",
                str(attempt["tier0_len"]),
                "--tier1-len",
                str(attempt["tier1_len"]),
                "--tier2-len",
                str(attempt["tier2_len"]),
                "--suffix-len",
                str(attempt["suffix_len"]),
                "--output-len",
                str(attempt["output_len"]),
                "--num-rounds",
                str(args.num_rounds),
                "--warmup-rounds",
                str(args.warmup_rounds),
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--model",
                args.model_path,
                "--output-dir",
                str(policy_output_dir),
                "--seed",
                str(args.seed),
                "--agents-seed",
                str(args.agents_seed),
            ]
            if args.workflow_type == "dag":
                cmd.extend(["--dag-config", args.dag_config])
            if real_templates:
                cmd.extend(["--real-templates", str(real_templates)])
                cmd.extend(["--real-templates-mode", args.real_templates_mode])
            if baseline_json:
                cmd.extend(["--baseline-json", str(baseline_json)])
            subprocess.run(cmd, check=True, cwd=str(ROOT))
            result_json = newest_json(policy_output_dir, before)
            with result_json.open("r", encoding="utf-8") as fh:
                result = json.load(fh)
            total_output_tokens = sum(
                step.get("output_tokens", 0)
                for workflow in result.get("results", [])
                for round_steps in workflow.get("rounds", [])
                for step in round_steps
            )
            if total_output_tokens == 0:
                raise RuntimeError(
                    f"Benchmark produced zero output tokens for policy={policy}. "
                    f"Check server log: {log_path}"
                )
            aggregate = result.get("aggregate", {})
            return {
                "policy": policy,
                "benchmark_config": benchmark_config,
                "result_json": str(result_json),
                "server_log": str(log_path),
                "aggregate": aggregate,
                "run_config": attempt,
                "failed_attempts": failures,
            }
        except Exception as exc:
            failures.append(
                {
                    "attempt": attempt,
                    "error": str(exc),
                }
            )
            print(f"Attempt failed for {policy}: {exc}", flush=True)
        finally:
            stop_server(server)

    raise RuntimeError(f"All attempts failed for policy={policy}: {json.dumps(failures, indent=2)}")


def build_attempt_specs(args: argparse.Namespace) -> List[Dict[str, int]]:
    specs: List[Dict[str, int]] = []
    seen: set[Tuple[int, int, int, int, int, int]] = set()
    for wf in parse_int_sequence(args.workflow_candidates, args.num_workflows):
        for scale in parse_float_sequence(args.tier_scale_candidates, 1.0):
            spec = {
                "num_workflows": max(1, wf),
                "tier0_len": max(16, int(args.tier0_len * scale)),
                "tier1_len": max(16, int(args.tier1_len * scale)),
                "tier2_len": max(16, int(args.tier2_len * scale)),
                "suffix_len": max(8, int(args.suffix_len * scale)),
                "output_len": max(8, int(args.output_len * scale)),
            }
            key = (
                spec["num_workflows"],
                spec["tier0_len"],
                spec["tier1_len"],
                spec["tier2_len"],
                spec["suffix_len"],
                spec["output_len"],
            )
            if key not in seen:
                seen.add(key)
                specs.append(spec)
    return specs


def parse_int_sequence(raw: str, fallback: int) -> List[int]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        return [fallback]
    return [int(item) for item in values]


def parse_float_sequence(raw: str, fallback: float) -> List[float]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        return [fallback]
    return [float(item) for item in values]


def build_summary(results: List[Dict[str, object]], baseline_policy: str) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "baseline_policy": baseline_policy,
        "policies": results,
        "comparisons": [],
    }
    baseline = next((r for r in results if r["policy"] == baseline_policy), None)
    if not baseline:
        return summary
    baseline_agg = baseline.get("aggregate", {})
    baseline_ttft = baseline_agg.get("stable_ttft_avg_ms")
    baseline_e2e = baseline_agg.get("stable_e2e_avg_ms")
    comparisons = []
    for result in results:
        if result["policy"] == baseline_policy:
            continue
        agg = result.get("aggregate", {})
        entry = {
            "policy": result["policy"],
            "ttft_speedup_vs_baseline": (
                baseline_ttft / agg["stable_ttft_avg_ms"]
                if baseline_ttft and agg.get("stable_ttft_avg_ms")
                else None
            ),
            "e2e_speedup_vs_baseline": (
                baseline_e2e / agg["stable_e2e_avg_ms"]
                if baseline_e2e and agg.get("stable_e2e_avg_ms")
                else None
            ),
        }
        comparisons.append(entry)
    summary["comparisons"] = comparisons
    return summary


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
                "benchmark_config",
                "result_json",
                "stable_ttft_avg_ms",
                "stable_e2e_avg_ms",
                "avg_output_throughput_tok_s",
                "ttft_speedup_vs_baseline",
                "e2e_speedup_vs_baseline",
                "run_num_workflows",
                "run_tier0_len",
                "run_tier1_len",
                "run_tier2_len",
                "run_suffix_len",
                "run_output_len",
                "server_log",
            ]
        )
        for result in summary.get("policies", []):
            agg = result.get("aggregate", {}) if isinstance(result, dict) else {}
            cmp_row = comparisons.get(result.get("policy"), {}) if isinstance(result, dict) else {}
            run_cfg = result.get("run_config", {}) if isinstance(result, dict) else {}
            writer.writerow(
                [
                    result.get("policy"),
                    result.get("benchmark_config"),
                    result.get("result_json"),
                    agg.get("stable_ttft_avg_ms"),
                    agg.get("stable_e2e_avg_ms"),
                    agg.get("avg_output_throughput_tok_s"),
                    cmp_row.get("ttft_speedup_vs_baseline"),
                    cmp_row.get("e2e_speedup_vs_baseline"),
                    run_cfg.get("num_workflows"),
                    run_cfg.get("tier0_len"),
                    run_cfg.get("tier1_len"),
                    run_cfg.get("tier2_len"),
                    run_cfg.get("suffix_len"),
                    run_cfg.get("output_len"),
                    result.get("server_log"),
                ]
            )
    return csv_path


def _fmt_float(value: object, digits: int = 3) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return "n/a"


def write_markdown_summary(summary: Dict[str, object], output_dir: Path) -> Path:
    md_path = output_dir / "suite_summary.md"
    baseline_policy = summary.get("baseline_policy", "n/a")
    comparisons = {
        item["policy"]: item
        for item in summary.get("comparisons", [])
        if isinstance(item, dict) and item.get("policy")
    }
    lines = [
        "# KVFlow Serial Policy Suite",
        "",
        f"- Baseline policy: `{baseline_policy}`",
        "",
        "| Policy | Bench Config | Workflows | T0/T1/T2 | Suffix | TTFT (ms) | E2E (ms) | Output tok/s | TTFT speedup | E2E speedup |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in summary.get("policies", []):
        agg = result.get("aggregate", {}) if isinstance(result, dict) else {}
        cmp_row = comparisons.get(result.get("policy"), {}) if isinstance(result, dict) else {}
        run_cfg = result.get("run_config", {}) if isinstance(result, dict) else {}
        lines.append(
            "| {policy} | {bench} | {wf} | {tiers} | {suffix} | {ttft} | {e2e} | {throughput} | {ttft_sp} | {e2e_sp} |".format(
                policy=result.get("policy", "n/a"),
                bench=result.get("benchmark_config", "n/a"),
                wf=run_cfg.get("num_workflows", "n/a"),
                tiers="{}/{}/{}".format(
                    run_cfg.get("tier0_len", "n/a"),
                    run_cfg.get("tier1_len", "n/a"),
                    run_cfg.get("tier2_len", "n/a"),
                ),
                suffix=run_cfg.get("suffix_len", "n/a"),
                ttft=_fmt_float(agg.get("stable_ttft_avg_ms"), 1),
                e2e=_fmt_float(agg.get("stable_e2e_avg_ms"), 1),
                throughput=_fmt_float(agg.get("avg_output_throughput_tok_s"), 2),
                ttft_sp=_fmt_float(cmp_row.get("ttft_speedup_vs_baseline"), 3),
                e2e_sp=_fmt_float(cmp_row.get("e2e_speedup_vs_baseline"), 3),
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serial single-GPU KVFlow policy suite")
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["lru_nocache", "priority_wb_only", "kvflow", "tiered_wb_only"],
        choices=sorted(POLICY_SERVER_ARGS.keys()),
        help="Policies to evaluate serially",
    )
    parser.add_argument("--baseline-policy", default="lru_nocache")
    parser.add_argument("--workflow-type", choices=["linear", "dag"], default="linear")
    parser.add_argument("--dag-config", default=None)
    parser.add_argument("--num-workflows", type=int, default=4)
    parser.add_argument("--agents-per-workflow", type=int, default=5)
    parser.add_argument("--tier0-len", type=int, default=512)
    parser.add_argument("--tier1-len", type=int, default=1024)
    parser.add_argument("--tier2-len", type=int, default=512)
    parser.add_argument("--suffix-len", type=int, default=64)
    parser.add_argument("--output-len", type=int, default=64)
    parser.add_argument("--num-rounds", type=int, default=5)
    parser.add_argument("--warmup-rounds", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--agents-seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--server-python", default=DEFAULT_SERVER_PYTHON)
    parser.add_argument("--base-port", type=int, default=31000)
    parser.add_argument("--server-timeout", type=int, default=300)
    parser.add_argument("--mem-fraction-static", type=float, default=0.75)
    parser.add_argument("--max-total-tokens", type=int, default=40000)
    parser.add_argument("--chunked-prefill-size", type=int, default=2048)
    parser.add_argument("--max-prefill-tokens", type=int, default=4096)
    parser.add_argument("--hicache-ratio", type=float, default=2.0)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--quantization", default=None)
    parser.add_argument(
        "--workflow-candidates",
        default="",
        help="Comma-separated workflow counts to try in order on failure; defaults to --num-workflows only",
    )
    parser.add_argument(
        "--tier-scale-candidates",
        default="",
        help="Comma-separated prompt scaling factors to try in order on failure; defaults to 1.0 only",
    )
    parser.add_argument("--real-templates", default=None)
    parser.add_argument("--template-cache-input", default=None)
    parser.add_argument("--real-templates-mode", choices=["mix", "dominant"], default="mix")
    parser.add_argument(
        "--export-script",
        default="/home/gfy/CodeMAS_Project/MAScoder/scripts/export_kv_templates.py",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workflow_type == "dag" and not args.dag_config:
        raise SystemExit("--dag-config is required for DAG mode")

    real_templates, export_stats = maybe_export_real_templates(args)
    results: List[Dict[str, object]] = []
    baseline_json: Optional[Path] = None
    for idx, policy in enumerate(args.policies):
        port = args.base_port + idx
        print(f"\n=== Running {policy} on port {port} ===", flush=True)
        result = run_one_policy(args, policy, port, real_templates, baseline_json)
        if policy == args.baseline_policy:
            baseline_json = Path(str(result["result_json"]))
        results.append(result)

    summary = build_summary(results, args.baseline_policy)
    if real_templates:
        summary["real_templates_path"] = str(real_templates)
        summary["real_templates_mode"] = args.real_templates_mode
    if export_stats:
        summary["template_export_stats"] = export_stats
    summary_path = Path(args.output_dir) / "suite_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    csv_path = write_csv_summary(summary, Path(args.output_dir))
    md_path = write_markdown_summary(summary, Path(args.output_dir))
    print(f"\nSuite summary written to {summary_path}")
    print(f"CSV summary written to {csv_path}")
    print(f"Markdown summary written to {md_path}")


if __name__ == "__main__":
    main()
