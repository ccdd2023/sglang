#!/usr/bin/env python3
"""Run the frozen SWE-bench bridge accuracy/speed experiment.

The runner launches one SGLang arm at a time, executes mini-SWE-agent with a
shared rolling-history policy, normalizes predictions, optionally invokes the
official evaluator, and records native KV-copy telemetry.  It never enables
prefetch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from benchmark.multi_workflow.prepare_minisweagent_swebench import (
    normalize_predictions,
)


PROJECT = Path(__file__).resolve().parents[2]
ARTIFACTS = Path("/home/gfy/CodeMAS_Project/kvflow-artifacts")
DEFAULT_OUTPUT = ARTIFACTS / "impactkv_bridge_agent_accuracy_speed_20260726"
DATASET = (
    ARTIFACTS
    / "swebench_verified_bridge_v1_20260724/minisweagent_dataset"
)
SNAPSHOT = (
    ARTIFACTS / "swebench_verified_bridge_v1_20260724/frozen_subset.json"
)
REGISTRATION = (
    PROJECT / "benchmark/multi_workflow/swebench_verified_bridge_v1.json"
)
CONFIG = PROJECT / "benchmark/multi_workflow/swebench_bridge_agent_reuse_v1.yaml"
CHAT_TEMPLATE = (
    PROJECT / "benchmark/multi_workflow/qwen3_coder_tool_chat_template.jinja"
)
MODEL = "/home/gfy/models/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit"
MINI = Path("/home/gfy/.venvs/mini-swe-agent-v2.3.0/bin/mini-extra")
EVAL_PYTHON = Path("/home/gfy/.conda/envs/sglang-kvflow/bin/python")
SERVER_PYTHON = EVAL_PYTHON
ARMS = (
    "dense",
    "general",
    "general_8k",
    "coding_aware",
    "coding_failure_v1",
    "coding_phase_v1",
    "coding_adaptive_v2",
    "coding_adaptive_v3",
    "coding_budget_v4",
    "coding_memory_dense_v5",
    "coding_memory_v5",
    "coding_source_guard_v6",
    "coding_evidence_payoff_v7",
    "general_dual_4k",
    "coding_dual_v8",
    "coding_version_graph_v17",
    "coding_post_mutation_v19",
    "coding_post_mutation_dual_v20",
    "coding_post_mutation_seam32_v22",
    "coding_post_mutation_target_prefix_v23",
)
DENSE_ARMS = ("dense", "coding_memory_dense_v5")
HOST_OVERFLOW_ARMS = (
    "general_8k",
    "coding_adaptive_v3",
    "coding_budget_v4",
    "coding_memory_v5",
    "coding_evidence_payoff_v7",
    "coding_dual_v8",
    "coding_version_graph_v17",
    "coding_post_mutation_v19",
    "coding_post_mutation_dual_v20",
    "coding_post_mutation_seam32_v22",
    "coding_post_mutation_target_prefix_v23",
)
DUAL_ISLAND_ARMS = (
    "general_dual_4k",
    "coding_dual_v8",
    "coding_post_mutation_dual_v20",
    "coding_post_mutation_seam32_v22",
    "coding_post_mutation_target_prefix_v23",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def prepare(output: Path) -> dict[str, Any]:
    if (output / "RUN_REGISTRATION.json").exists():
        return read_json(output / "RUN_REGISTRATION.json")
    frozen = read_json(REGISTRATION)
    value = {
        "registration_id": output.name,
        "registered_at_utc": utc_now(),
        "objective": (
            "Measure final official SWE-bench accuracy and native KV-reuse "
            "speed under one shared rolling-history agent protocol."
        ),
        "arms": {
            "dense": "same rolling history; all prompt tokens dense",
            "general": (
                "copy the retained shifted history block, capped at 4096 tokens"
            ),
            "general_8k": (
                "matched non-coding ablation: copy the retained shifted "
                "history block, capped at 8192 tokens, with host overflow"
            ),
            "coding_aware": (
                "copy older completed coding interactions, capped at 4096; "
                "the current latest and future newest interactions remain dense"
            ),
            "coding_failure_v1": (
                "use General reuse by default; keep the latest completed "
                "interaction dense after a failed tool or failure diagnostic"
            ),
            "coding_phase_v1": (
                "use General reuse by default; keep the latest completed "
                "interaction dense after repository mutation, diff, failed "
                "tool, or failure diagnostic"
            ),
            "coding_adaptive_v2": (
                "phase-gated coding reuse with an 8192-token safe-phase cap; "
                "risk phases protect the latest interaction and use 4096"
            ),
            "coding_adaptive_v3": (
                "V2 risk-gated reuse, but failed read-only searches remain "
                "General-safe; allocator overflow is stored on host and loaded "
                "only when the registered target request arrives"
            ),
            "coding_budget_v4": (
                "keep General's complete retained block; use an 8192-token "
                "cap in low-risk coding phases and a 4096-token cap after "
                "mutation, diff, or executable failure; failed read-only "
                "searches remain low-risk; host overflow is on-demand"
            ),
            "coding_memory_dense_v5": (
                "accuracy ceiling for V5: retain the recent six interactions "
                "plus the newest older executable failure, with no KV reuse"
            ),
            "coding_memory_v5": (
                "retain the recent six interactions plus the newest older "
                "executable failure; reuse the guaranteed recent-five "
                "overlap with an 8192-token cap and on-demand host overflow"
            ),
            "coding_source_guard_v6": (
                "use General-4K unless a substantial concrete Python source "
                "read remains in the reusable window; then copy only history "
                "before that read until a repository mutation resets the guard"
            ),
            "coding_evidence_payoff_v7": (
                "use General-4K by default; widen the same contiguous overlap "
                "to 6144 tokens only after a successful substantial read-only "
                "search/source observation and only when at least 1024 "
                "additional reusable tokens are available; host overflow is "
                "loaded on demand"
            ),
            "general_dual_4k": (
                "matched V8 control: lossless ordinary Radix prefix plus a "
                "shifted middle island capped at 4096 tokens"
            ),
            "coding_dual_v8": (
                "the same lossless ordinary prefix as general_dual_4k; widen "
                "the shifted middle island to 6144 tokens only after a "
                "successful substantial read-only coding observation and at "
                "least 1024 useful marginal tokens"
            ),
            "coding_version_graph_v17": (
                "bind completed file observations to online repository "
                "versions; exclude observations invalidated by later edits, "
                "keep the latest risky event dense, and copy the largest "
                "remaining contiguous valid island with a 4096-token cap"
            ),
            "coding_post_mutation_v19": (
                "use General-4K when there is no online file-version boundary; "
                "after a retained file observation is followed by a mutation "
                "of that file, copy the largest contiguous post-boundary "
                "island; never apply V17's blanket latest-risk guard"
            ),
            "coding_post_mutation_dual_v20": (
                "V19 post-mutation shifted island plus the exact ordinary "
                "Radix prefix populated by the preceding real request; no "
                "synthetic request or prefetch"
            ),
            "coding_post_mutation_seam32_v22": (
                "V20 dual reuse with the final 32 tokens before the shifted "
                "middle island recomputed densely to stabilize the numerical "
                "prefix-to-middle seam"
            ),
            "coding_post_mutation_target_prefix_v23": (
                "reuse the ordinary Radix prefix only on a registered target; "
                "keep unregistered requests and source-building requests "
                "dense, then copy the V19 post-mutation shifted island"
            ),
        },
        "dataset": {
            "registration_id": frozen["registration_id"],
            "instances": [
                row["instance_id"] for row in frozen["instances"]
            ],
            "count": len(frozen["instances"]),
            "snapshot_sha256": sha256(SNAPSHOT),
        },
        "frozen_protocol": {
            "model": MODEL,
            "agent_step_limit": 20,
            "prompt_token_limit": 28_000,
            "server_context_length": 32_768,
            "rolling_history_groups": 6,
            "copy_cap_tokens": 4_096,
            "coding_adaptive_v2_safe_copy_cap_tokens": 8_192,
            "coding_adaptive_v3_safe_copy_cap_tokens": 8_192,
            "coding_adaptive_v3_host_overflow": True,
            "general_8k_copy_cap_tokens": 8_192,
            "general_8k_host_overflow": True,
            "coding_budget_v4_safe_copy_cap_tokens": 8_192,
            "coding_budget_v4_risk_copy_cap_tokens": 4_096,
            "coding_budget_v4_excludes_latest_group": False,
            "coding_budget_v4_host_overflow": True,
            "coding_memory_v5_recent_groups": 6,
            "coding_memory_v5_extra_failure_memories": 1,
            "coding_memory_v5_copy_cap_tokens": 8_192,
            "coding_memory_v5_host_overflow": True,
            "coding_source_guard_v6_copy_cap_tokens": 4_096,
            "coding_source_guard_v6_host_overflow": False,
            "coding_evidence_payoff_v7_default_copy_cap_tokens": 4_096,
            "coding_evidence_payoff_v7_wide_copy_cap_tokens": 6_144,
            "coding_evidence_payoff_v7_min_marginal_tokens": 1_024,
            "coding_evidence_payoff_v7_host_overflow": True,
            "dual_island_ordinary_radix_prefix_reuse": True,
            "general_dual_4k_copy_cap_tokens": 4_096,
            "coding_dual_v8_default_copy_cap_tokens": 4_096,
            "coding_dual_v8_wide_copy_cap_tokens": 6_144,
            "coding_dual_v8_min_marginal_tokens": 1_024,
            "coding_dual_v8_host_overflow": True,
            "coding_version_graph_v17_copy_cap_tokens": 4_096,
            "coding_version_graph_v17_host_overflow": True,
            "coding_post_mutation_v19_copy_cap_tokens": 4_096,
            "coding_post_mutation_v19_host_overflow": True,
            "coding_post_mutation_dual_v20_copy_cap_tokens": 4_096,
            "coding_post_mutation_dual_v20_host_overflow": True,
            "coding_post_mutation_dual_v20_ordinary_radix_prefix_reuse": True,
            "coding_post_mutation_seam32_v22_copy_cap_tokens": 4_096,
            "coding_post_mutation_seam32_v22_host_overflow": True,
            "coding_post_mutation_seam32_v22_ordinary_radix_prefix_reuse": True,
            "coding_post_mutation_seam32_v22_prefix_repair_tokens": 32,
            "coding_post_mutation_target_prefix_v23_copy_cap_tokens": 4_096,
            "coding_post_mutation_target_prefix_v23_host_overflow": True,
            "coding_post_mutation_target_prefix_v23_target_only_prefix": True,
            "min_copy_tokens": 128,
            "temperature": 0,
            "workers": 1,
            "prefetch": False,
            "ordinary_radix_prefix_reuse": False,
        },
        "metrics": {
            "primary_accuracy": "official SWE-bench resolved / 18",
            "primary_speed": (
                "streaming TTFT and request wall time relative to rolling Dense"
            ),
            "physical_validity": (
                "source_materialized and target_copied ledger events; "
                "fallbacks reported separately"
            ),
            "excluded_as_accuracy": [
                "exact output agreement",
                "character agreement",
                "next-command agreement",
                "NLL",
            ],
        },
        "canary_gate": {
            "agent_terminal_output": True,
            "source_materialized_events_min": 1,
            "target_copy_events_min": 1,
            "fallback_events_max": 0,
        },
        "iteration_protocol": {
            "motivation_slice": [
                "astropy__astropy-14995",
                "psf__requests-1142",
                "sphinx-doc__sphinx-9230",
            ],
            "motivation_slice_is_posthoc": True,
            "promotion_gate": (
                "coding-evidence-payoff V7 must not lose official resolved "
                "tasks versus same-campaign General-4K and must reduce p95 "
                "TTFT by at least 5% while keeping median TTFT within 2%. "
                "At least three physical copies above 4096 tokens, zero "
                "target fallbacks, and zero prefetch are mandatory."
            ),
            "nondeterminism_audit": (
                "report command agreement before the first physical copy; "
                "never attribute pre-copy divergence to the reuse policy"
            ),
        },
        "reporting_rule": (
            "No SOTA claim until all three 18-task arms have official final "
            "accuracy and directly comparable speed telemetry."
        ),
        "source_sha256": {
            str(path.relative_to(PROJECT)): sha256(path)
            for path in (
                PROJECT
                / "benchmark/multi_workflow/bridge_reuse_litellm_model.py",
                PROJECT
                / "benchmark/multi_workflow/coding_reuse_policy.py",
                PROJECT
                / "benchmark/multi_workflow/context_bounded_litellm_model.py",
                CONFIG,
                CHAT_TEMPLATE,
                PROJECT / "python/sglang/srt/mem_cache/kvcomm_exact.py",
                PROJECT
                / "python/sglang/srt/mem_cache/kvcomm/radix_backend.py",
                PROJECT / "python/sglang/srt/mem_cache/kvcomm/transfer.py",
                PROJECT
                / "benchmark/multi_workflow/run_bridge_reuse_agent_experiment.py",
                PROJECT
                / "benchmark/multi_workflow/audit_coding_evidence_payoff.py",
                PROJECT
                / "benchmark/multi_workflow/test_coding_reuse_policy.py",
                PROJECT / "python/sglang/srt/mem_cache/test_kvcomm_exact.py",
                PROJECT
                / "python/sglang/srt/mem_cache/kvcomm/test_radix_backend.py",
            )
        },
        "protected": {
            "old_dirty_checkout_modified": False,
            "paper_modified": False,
            "preregistration_thresholds_modified": False,
        },
        "status": "REGISTERED_BEFORE_TREATMENT_GPU_RUN",
    }
    write_json(output / "RUN_REGISTRATION.json", value)
    write_json(
        output / "CAMPAIGN_STATUS.json",
        {
            "state": "registered",
            "updated_at_utc": utc_now(),
            "arms": {arm: "pending" for arm in ARMS},
        },
    )
    return value


def init_manifest(run_dir: Path, arm: str) -> Path:
    path = run_dir / "DYNAMIC_MANIFEST.json"
    write_json(
        path,
        {
            "version": 3,
            "model_id": MODEL,
            "cache_dtype": "bfloat16",
            "lease_ttl_s": 900,
            "ledger_path": str(run_dir / "SERVER_LEDGER.jsonl"),
            "rope": {
                "rotary_dim": 128,
                "base": 10_000_000,
                "is_neox_style": True,
            },
            "sources": [],
            "cases": [],
            "release_source_ids": [],
            "arm": arm,
            "host_overflow_enabled": arm in HOST_OVERFLOW_ARMS,
            "ordinary_prefix_reuse_enabled": arm in DUAL_ISLAND_ARMS,
            "ordinary_prefix_repair_tokens": (
                32 if arm == "coding_post_mutation_seam32_v22" else 0
            ),
            "ordinary_prefix_target_only": (
                arm == "coding_post_mutation_target_prefix_v23"
            ),
        },
    )
    return path


def launch_server(
    *,
    run_dir: Path,
    arm: str,
    manifest: Path,
    port: int,
    mem_fraction_static: float = 0.90,
) -> tuple[subprocess.Popen[str], Any]:
    log = (run_dir / "sglang_server.log").open("w", encoding="utf-8")
    env = os.environ.copy()
    env.update(
        {
            "HF_HUB_OFFLINE": "1",
            "PYTHONPATH": f"{PROJECT / 'python'}:{PROJECT}",
            "SGLANG_KVCOMM_CORE": "0" if arm in DENSE_ARMS else "1",
        }
    )
    if arm not in DENSE_ARMS:
        env["SGLANG_KVCOMM_EXACT_CANARY_MANIFEST"] = str(manifest)
    command = [
        str(SERVER_PYTHON),
        "-m",
        "sglang.launch_server",
        "--model-path",
        MODEL,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--context-length",
        "32768",
        "--chat-template",
        str(CHAT_TEMPLATE),
        "--attention-backend",
        "triton",
        "--mem-fraction-static",
        str(mem_fraction_static),
        "--chunked-prefill-size",
        "8192",
        "--max-prefill-tokens",
        "16384",
        "--page-size",
        "1",
        "--disable-cuda-graph",
        "--disable-overlap-schedule",
        "--enable-deterministic-inference",
        "--enable-request-time-stats-logging",
        "--random-seed",
        "709609581",
    ]
    if arm in DENSE_ARMS:
        command.append("--disable-radix-cache")
    write_json(run_dir / "SERVER_COMMAND.json", command)
    process = subprocess.Popen(
        command,
        cwd=PROJECT,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    deadline = time.monotonic() + 240
    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(
                    f"server exited {process.returncode}; inspect {log.name}"
                )
            try:
                response = requests.get(
                    f"http://127.0.0.1:{port}/model_info", timeout=2
                )
                if response.ok:
                    return process, log
            except requests.RequestException:
                pass
            time.sleep(2)
        raise TimeoutError("SGLang server did not become ready")
    except BaseException:
        stop_server(process, log)
        raise


def stop_server(process: subprocess.Popen[str], log: Any) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=30)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
    finally:
        log.close()


def mini_command(
    *,
    run_dir: Path,
    arm: str,
    manifest: Path,
    port: int,
    instance_filter: str | None,
) -> list[str]:
    command = [
        str(MINI),
        "swebench",
        "--subset",
        str(DATASET),
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
        f"model.reuse_arm={arm}",
        "--config",
        f"model.model_kwargs.api_base=http://127.0.0.1:{port}/v1",
        "--config",
        (
            "model.reuse_client_ledger_path="
            f"{run_dir / 'CLIENT_LEDGER.jsonl'}"
        ),
    ]
    if arm not in DENSE_ARMS:
        command.extend(
            [
                "--config",
                f"model.reuse_manifest_path={manifest}",
            ]
        )
    if instance_filter:
        command.extend(["--filter", f"^{instance_filter}$"])
    return command


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def summarize_runtime(run_dir: Path, arm: str) -> dict[str, Any]:
    client = load_jsonl(run_dir / "CLIENT_LEDGER.jsonl")
    server = load_jsonl(run_dir / "SERVER_LEDGER.jsonl")
    request_rows = [
        row for row in client if row.get("event") == "request_complete"
    ]
    ttfts = [
        1000 * float(row["ttft_seconds"])
        for row in request_rows
        if row.get("ttft_seconds") is not None
    ]
    elapsed = [
        1000 * float(row["request_elapsed_seconds"])
        for row in request_rows
    ]
    summary = {
        "arm": arm,
        "requests": len(request_rows),
        "target_registered_requests": sum(
            bool(row.get("target_registered")) for row in request_rows
        ),
        "source_registered_requests": sum(
            bool(row.get("source_registered")) for row in request_rows
        ),
        "median_ttft_ms": statistics.median(ttfts) if ttfts else None,
        "p95_ttft_ms": (
            sorted(ttfts)[max(0, int(0.95 * len(ttfts)) - 1)]
            if ttfts
            else None
        ),
        "median_request_elapsed_ms": (
            statistics.median(elapsed) if elapsed else None
        ),
        "source_materialized_events": sum(
            row.get("event")
            in ("source_materialized", "source_materialized_host")
            for row in server
        ),
        "source_materialized_device_events": sum(
            row.get("event") == "source_materialized" for row in server
        ),
        "source_materialized_host_events": sum(
            row.get("event") == "source_materialized_host" for row in server
        ),
        "source_materialization_skipped_events": sum(
            row.get("event") == "source_materialization_skipped"
            for row in server
        ),
        "target_copy_events": sum(
            row.get("event") == "target_copied" for row in server
        ),
        "target_fallback_events": sum(
            row.get("event") == "target_fallback" for row in server
        ),
        "copied_tokens": sum(
            int(row.get("copied_k_tokens", 0))
            for row in server
            if row.get("event") == "target_copied"
        ),
        "rotated_k_tokens": sum(
            int(row.get("rotated_k_tokens", 0))
            for row in server
            if row.get("event") == "target_copied"
        ),
        "host_source_copy_events": sum(
            row.get("event") == "target_copied"
            and row.get("source_residency") == "host"
            for row in server
        ),
        "readonly_evidence_decisions": sum(
            bool(
                row.get("reuse_policy_decision", {}).get(
                    "readonly_evidence"
                )
            )
            for row in request_rows
        ),
        "wide_6144_budget_decisions": sum(
            row.get("reuse_policy_decision", {}).get(
                "effective_copy_cap"
            )
            == 6144
            for row in request_rows
        ),
        "client_rows": len(client),
        "server_rows": len(server),
    }
    write_json(run_dir / "RUNTIME_SUMMARY.json", summary)
    return summary


def run_official_evaluation(
    *,
    output: Path,
    run_dir: Path,
    arm: str,
    instance_ids: list[str] | None = None,
) -> dict[str, Any]:
    predictions = run_dir / "predictions.jsonl"
    telemetry = run_dir / "TELEMETRY.json"
    normalize_predictions(
        run_dir,
        REGISTRATION,
        predictions,
        telemetry,
        f"impactkv__bridge-agent-{arm}-rolling6",
        allow_partial=instance_ids is not None,
    )
    frozen = read_json(REGISTRATION)
    ids = instance_ids or [
        row["instance_id"] for row in frozen["instances"]
    ]
    run_id = f"bridge-agent-{arm}-rolling6-20260726"
    if instance_ids is not None:
        subset_hash = hashlib.sha256(
            "\n".join(instance_ids).encode()
        ).hexdigest()[:10]
        run_id += f"-canary-{subset_hash}"
    command = [
        str(EVAL_PYTHON),
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        str(SNAPSHOT),
        "--split",
        frozen["dataset"]["split"],
        "--predictions_path",
        str(predictions),
        "--instance_ids",
        *ids,
        "--max_workers",
        "3",
        "--timeout",
        "3600",
        "--cache_level",
        "instance",
        "--clean",
        "true",
        "--run_id",
        run_id,
        "--namespace",
        "swebench",
        "--report_dir",
        str(run_dir / "reports"),
    ]
    write_json(run_dir / "OFFICIAL_EVALUATION_COMMAND.json", command)
    evaluation_env = os.environ.copy()
    rootless_docker_socket = Path(
        os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    ) / "docker.sock"
    if rootless_docker_socket.exists():
        evaluation_env["DOCKER_HOST"] = (
            f"unix://{rootless_docker_socket}"
        )
    write_json(
        run_dir / "OFFICIAL_EVALUATION_ENV.json",
        {"DOCKER_HOST": evaluation_env.get("DOCKER_HOST")},
    )
    with (run_dir / "official_evaluation.stdout.log").open(
        "w", encoding="utf-8"
    ) as log:
        result = subprocess.run(
            command,
            cwd=run_dir,
            env=evaluation_env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    reports = sorted(run_dir.glob("impactkv__*.json"))
    report = read_json(reports[-1]) if reports else None
    value = {
        "returncode": result.returncode,
        "run_id": run_id,
        "report_path": str(reports[-1]) if reports else None,
        "report": report,
    }
    write_json(run_dir / "OFFICIAL_RESULT.json", value)
    if result.returncode != 0 or report is None:
        raise RuntimeError("official SWE-bench evaluation failed")
    return value


def evaluate_existing(
    *, output: Path, arm: str, scope: str
) -> dict[str, Any]:
    suffix = "full_18" if scope == "full" else scope
    run_dir = output / arm / suffix
    if not (run_dir / "RUNTIME_SUMMARY.json").exists():
        raise FileNotFoundError(
            f"missing completed runtime summary: {run_dir}"
        )
    status = read_json(run_dir / "PIPELINE_STATUS.json")
    status["state"] = "official_evaluation"
    write_json(run_dir / "PIPELINE_STATUS.json", status)
    try:
        official_result = run_official_evaluation(
            output=output, run_dir=run_dir, arm=arm
        )
    except Exception:
        status["state"] = "official_failed"
        status["official_failed_at_utc"] = utc_now()
        write_json(run_dir / "PIPELINE_STATUS.json", status)
        raise
    status["state"] = "complete"
    status["finished_at_utc"] = utc_now()
    status["official"] = official_result["report"]
    write_json(run_dir / "PIPELINE_STATUS.json", status)
    return status


def run_arm(
    *,
    output: Path,
    arm: str,
    scope: str,
    port: int,
    instance_filter: str | None,
    official: bool,
) -> dict[str, Any]:
    prepare(output)
    if arm not in ARMS:
        raise ValueError(f"unsupported arm: {arm}")
    suffix = "full_18" if scope == "full" else f"canary_{instance_filter}"
    run_dir = output / arm / suffix
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty run: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = init_manifest(run_dir, arm)
    command = mini_command(
        run_dir=run_dir,
        arm=arm,
        manifest=manifest,
        port=port,
        instance_filter=instance_filter,
    )
    write_json(run_dir / "MINI_COMMAND.json", command)
    status = {
        "state": "starting_server",
        "arm": arm,
        "scope": scope,
        "started_at_utc": utc_now(),
        "prefetch": False,
    }
    write_json(run_dir / "PIPELINE_STATUS.json", status)
    process, server_log = launch_server(
        run_dir=run_dir, arm=arm, manifest=manifest, port=port
    )
    try:
        status["state"] = "agent_running"
        status["server_pid"] = process.pid
        write_json(run_dir / "PIPELINE_STATUS.json", status)
        env = os.environ.copy()
        env.update(
            {
                "HF_HUB_OFFLINE": "1",
                "PYTHONPATH": str(PROJECT),
                "NO_PROXY": "localhost,127.0.0.1,.local",
                "no_proxy": "localhost,127.0.0.1,.local",
            }
        )
        # LiteLLM/httpx may still honor ALL_PROXY for localhost even with
        # NO_PROXY.  This campaign is entirely local and the model is offline.
        for proxy_name in (
            "ALL_PROXY",
            "all_proxy",
            "HTTP_PROXY",
            "http_proxy",
            "HTTPS_PROXY",
            "https_proxy",
        ):
            env.pop(proxy_name, None)
        with (run_dir / "minisweagent.log").open(
            "w", encoding="utf-8"
        ) as log:
            result = subprocess.run(
                command,
                cwd=PROJECT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        status["agent_returncode"] = result.returncode
        status["agent_finished_at_utc"] = utc_now()
        if result.returncode != 0:
            status["state"] = "agent_failed"
            write_json(run_dir / "PIPELINE_STATUS.json", status)
            raise RuntimeError("mini-SWE-agent failed")
    finally:
        stop_server(process, server_log)

    runtime = summarize_runtime(run_dir, arm)
    status["state"] = "runtime_complete"
    status["runtime"] = runtime
    write_json(run_dir / "PIPELINE_STATUS.json", status)
    official_result = None
    if official:
        evaluation_ids = None
        if scope == "canary":
            pattern = re.compile(f"^(?:{instance_filter})$")
            frozen = read_json(REGISTRATION)
            evaluation_ids = [
                row["instance_id"]
                for row in frozen["instances"]
                if pattern.fullmatch(row["instance_id"])
            ]
            if not evaluation_ids:
                raise ValueError(
                    f"instance filter selected no registered tasks: "
                    f"{instance_filter}"
                )
        status["state"] = "official_evaluation"
        write_json(run_dir / "PIPELINE_STATUS.json", status)
        official_result = run_official_evaluation(
            output=output,
            run_dir=run_dir,
            arm=arm,
            instance_ids=evaluation_ids,
        )
    status["state"] = "complete"
    status["finished_at_utc"] = utc_now()
    if official_result is not None:
        status["official"] = official_result["report"]
    write_json(run_dir / "PIPELINE_STATUS.json", status)
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    run = sub.add_parser("run-arm")
    run.add_argument("--arm", choices=ARMS, required=True)
    run.add_argument("--scope", choices=("canary", "full"), required=True)
    run.add_argument("--port", type=int, default=30000)
    run.add_argument("--instance-filter")
    run.add_argument("--official", action="store_true")
    evaluate = sub.add_parser("evaluate-existing")
    evaluate.add_argument("--arm", choices=ARMS, required=True)
    evaluate.add_argument("--scope", choices=("full",), default="full")
    args = parser.parse_args()

    output = args.output.resolve()
    if args.command == "prepare":
        value = prepare(output)
    elif args.command == "evaluate-existing":
        value = evaluate_existing(
            output=output,
            arm=args.arm,
            scope=args.scope,
        )
    else:
        if args.scope == "canary" and not args.instance_filter:
            parser.error("--instance-filter is required for a canary")
        if args.scope == "full" and args.instance_filter:
            parser.error("--instance-filter is not allowed for full scope")
        value = run_arm(
            output=output,
            arm=args.arm,
            scope=args.scope,
            port=args.port,
            instance_filter=args.instance_filter,
            official=args.official,
        )
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
