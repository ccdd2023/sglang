#!/usr/bin/env python3
"""Run a bounded same-task bridge-v1 native KV-reuse pilot.

The workload is derived only from frozen trajectories of three officially
resolved SWE-bench tasks.  A target prompt emulates audited history compaction:
the first completed assistant/tool turn is replaced by a compaction notice,
while later turns remain byte-for-byte unchanged.

Arms:
* dense: recompute the compacted target prompt with Radix disabled.
* general: copy the largest shifted common suffix (capped at 4096 tokens).
* coding_aware: copy only the largest complete tool-observation block, keeping
  task instructions and assistant reasoning dense.

This is a next-action agreement and TTFT pilot, not an official SWE-bench
task-level accuracy evaluation.
"""

from __future__ import annotations

import argparse
import copy
import difflib
import hashlib
import json
import math
import os
import re
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import requests
from jinja2 import StrictUndefined, Template
from tokenizers import Tokenizer


PROJECT = Path(__file__).resolve().parents[2]
ARTIFACTS = Path("/home/gfy/CodeMAS_Project/kvflow-artifacts")
DEFAULT_OUTPUT = ARTIFACTS / "impactkv_bridge_reuse_pilot_20260726"
MODEL = "/home/gfy/models/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit"
TOKENIZER = Path(MODEL) / "tokenizer.json"
CHAT_TEMPLATE = PROJECT / "benchmark/multi_workflow/qwen3_coder_tool_chat_template.jinja"
TRAJECTORY_ROOT = (
    ARTIFACTS
    / "swebench_verified_bridge_v1_20260724"
    / "agent_dense_contextbound_v1"
    / "full_18"
)
TRAJECTORIES = {
    "astropy__astropy-7336": (
        TRAJECTORY_ROOT
        / "astropy__astropy-7336"
        / "astropy__astropy-7336.traj.json"
    ),
    "pytest-dev__pytest-7982": (
        TRAJECTORY_ROOT
        / "pytest-dev__pytest-7982"
        / "pytest-dev__pytest-7982.traj.json"
    ),
    "sympy__sympy-24539": (
        TRAJECTORY_ROOT
        / "sympy__sympy-24539"
        / "sympy__sympy-24539.traj.json"
    ),
}
LONG_TRAJECTORIES = {
    "astropy__astropy-7336": TRAJECTORIES["astropy__astropy-7336"],
    "pytest-dev__pytest-7982": TRAJECTORIES["pytest-dev__pytest-7982"],
    "scikit-learn__scikit-learn-12585": (
        TRAJECTORY_ROOT
        / "scikit-learn__scikit-learn-12585"
        / "scikit-learn__scikit-learn-12585.traj.json"
    ),
}
ARMS = ("dense", "general", "coding_aware")
BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute",
                }
            },
            "required": ["command"],
        },
    },
}
COMPACTION_NOTICE = {
    "role": "user",
    "content": (
        '<history_compaction dropped_turn_groups="1">'
        "Earlier interaction details were omitted to stay within the hardware "
        "context budget. Repository state persists; the most recent complete "
        "interactions follow.</history_compaction>"
    ),
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def token_ids_hash(token_ids: list[int]) -> str:
    digest = hashlib.sha256()
    for token_id in token_ids:
        digest.update(
            int(token_id).to_bytes(8, byteorder="little", signed=True)
        )
    return digest.hexdigest()


def template_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prepared = []
    for source in messages:
        message = {
            key: copy.deepcopy(value)
            for key, value in source.items()
            if key != "extra"
        }
        if message.get("content") is None:
            message["content"] = ""
        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function") or {}
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                function["arguments"] = json.loads(arguments)
        prepared.append(message)
    return prepared


def render_ids(
    *,
    messages: list[dict[str, Any]],
    template: Template,
    tokenizer: Tokenizer,
) -> list[int]:
    prompt = template.render(
        messages=template_messages(messages),
        tools=[BASH_TOOL],
        add_generation_prompt=True,
    )
    return tokenizer.encode(prompt, add_special_tokens=False).ids


def find_sublist(haystack: list[int], needle: list[int]) -> list[int]:
    if not needle or len(needle) > len(haystack):
        return []
    first = needle[0]
    return [
        index
        for index, value in enumerate(haystack[: len(haystack) - len(needle) + 1])
        if value == first and haystack[index : index + len(needle)] == needle
    ]


def common_blocks(source: list[int], target: list[int]) -> list[dict[str, int]]:
    matcher = difflib.SequenceMatcher(None, source, target, autojunk=False)
    rows = []
    for block in matcher.get_matching_blocks():
        if block.size <= 0 or block.a <= 0 or block.b <= 0:
            continue
        if block.a + block.size >= len(source):
            block = difflib.Match(block.a, block.b, block.size - 32)
        if block.b + block.size >= len(target):
            block = difflib.Match(block.a, block.b, block.size - 32)
        if block.size >= 128:
            rows.append(
                {
                    "source_start": block.a,
                    "target_start": block.b,
                    "length": block.size,
                }
            )
    return rows


def capped_tail(block: dict[str, int], cap: int) -> dict[str, int]:
    length = min(block["length"], cap)
    offset = block["length"] - length
    return {
        "source_start": block["source_start"] + offset,
        "target_start": block["target_start"] + offset,
        "length": length,
    }


def coding_tool_block(
    *,
    messages: list[dict[str, Any]],
    source_ids: list[int],
    target_ids: list[int],
    tokenizer: Tokenizer,
) -> dict[str, int]:
    candidates = []
    for message in messages[4:]:
        if message.get("role") != "tool":
            continue
        literal = (
            "<|im_start|>user\n<tool_response>\n"
            + str(message.get("content") or "")
            + "\n</tool_response><|im_end|>\n"
        )
        block_ids = tokenizer.encode(literal, add_special_tokens=False).ids
        source_positions = find_sublist(source_ids, block_ids)
        target_positions = find_sublist(target_ids, block_ids)
        if (
            len(block_ids) >= 128
            and len(source_positions) == 1
            and len(target_positions) == 1
        ):
            candidates.append(
                {
                    "source_start": source_positions[0],
                    "target_start": target_positions[0],
                    "length": len(block_ids),
                }
            )
    if not candidates:
        raise ValueError("no unique reusable tool-observation block")
    return capped_tail(max(candidates, key=lambda row: row["length"]), 4096)


def render_message_literal(message: dict[str, Any]) -> str:
    message = template_messages([message])[0]
    role = message["role"]
    if role == "assistant" and message.get("tool_calls"):
        value = "<|im_start|>assistant\n"
        if message.get("content"):
            value += str(message["content"]).strip() + "\n"
        for wrapped_call in message["tool_calls"]:
            call = wrapped_call.get("function", wrapped_call)
            value += f"<tool_call>\n<function={call['name']}>\n"
            for name, argument in call.get("arguments", {}).items():
                value += f"<parameter={name}>{argument}</parameter>\n"
            value += "</function>\n</tool_call>\n"
        return value + "<|im_end|>\n"
    if role == "tool":
        return (
            "<|im_start|>user\n<tool_response>\n"
            + str(message.get("content") or "")
            + "\n</tool_response><|im_end|>\n"
        )
    return (
        f"<|im_start|>{role}\n"
        + str(message.get("content") or "")
        + "<|im_end|>\n"
    )


def coding_stable_history_block(
    *,
    messages: list[dict[str, Any]],
    source_ids: list[int],
    target_ids: list[int],
    tokenizer: Tokenizer,
) -> dict[str, int]:
    # Keep the two newest completed coding interactions dense.  They are the
    # current decision frontier; only older completed assistant/tool pairs are
    # eligible for reuse.
    stable_messages = messages[4:-4]
    if len(stable_messages) < 2:
        raise ValueError("not enough completed history before dense frontier")
    literal = "".join(render_message_literal(row) for row in stable_messages)
    block_ids = tokenizer.encode(literal, add_special_tokens=False).ids
    source_positions = find_sublist(source_ids, block_ids)
    target_positions = find_sublist(target_ids, block_ids)
    if (
        len(block_ids) < 128
        or len(source_positions) != 1
        or len(target_positions) != 1
    ):
        raise ValueError("stable coding-history block is not uniquely reusable")
    return capped_tail(
        {
            "source_start": source_positions[0],
            "target_start": target_positions[0],
            "length": len(block_ids),
        },
        4096,
    )


def manifest_case(
    *,
    case_id: str,
    policy_label: str,
    source_ids: list[int],
    target_ids: list[int],
    span: dict[str, int],
) -> dict[str, Any]:
    source_start = span["source_start"]
    target_start = span["target_start"]
    length = span["length"]
    source_segment = source_ids[source_start : source_start + length]
    target_segment = target_ids[target_start : target_start + length]
    if source_segment != target_segment:
        raise ValueError(f"{case_id}: selected segment is not token-identical")
    return {
        "case_id": case_id,
        "content_hash": hashlib.sha256(
            (policy_label + ":" + token_ids_hash(source_segment)).encode()
        ).hexdigest(),
        "length": length,
        "policy_label": policy_label,
        "segment_token_hash": token_ids_hash(source_segment),
        "source_prefix_token_hash": token_ids_hash(source_ids[:source_start]),
        "source_prompt_hash": token_ids_hash(source_ids),
        "source_start": source_start,
        "target_prefix_token_hash": token_ids_hash(target_ids[:target_start]),
        "target_prompt_hash": token_ids_hash(target_ids),
        "target_start": target_start,
        "target_uses": 3,
    }


def prepare(output: Path, profile: str = "short") -> dict[str, Any]:
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    template = Template(
        CHAT_TEMPLATE.read_text(encoding="utf-8"),
        undefined=StrictUndefined,
    )
    cases = []
    manifests = {"general": [], "coding_aware": []}
    trajectories = LONG_TRAJECTORIES if profile == "long" else TRAJECTORIES
    for instance_id, path in trajectories.items():
        trajectory = read_json(path)
        messages = trajectory["messages"]
        if profile == "long":
            end = len(messages)
            while end > 2 and messages[end - 1].get("role") != "tool":
                end -= 1
        else:
            # Four completed assistant/tool turns provide a realistic, bounded
            # bridge while keeping all registered source segments resident.
            end = min(10, len(messages))
            if end % 2:
                end -= 1
        selected = messages[:end]
        if selected[-1].get("role") != "tool":
            raise ValueError(f"{instance_id}: selected prompt does not end in tool")
        source_messages = selected
        target_messages = [
            selected[0],
            selected[1],
            COMPACTION_NOTICE,
            *selected[4:],
        ]
        source_ids = render_ids(
            messages=source_messages,
            template=template,
            tokenizer=tokenizer,
        )
        target_ids = render_ids(
            messages=target_messages,
            template=template,
            tokenizer=tokenizer,
        )
        if max(len(source_ids), len(target_ids)) > 28_000:
            raise ValueError(f"{instance_id}: prompt exceeds frozen 28K limit")
        blocks = common_blocks(source_ids, target_ids)
        if not blocks:
            raise ValueError(f"{instance_id}: no shifted common block")
        general_span = capped_tail(
            max(blocks, key=lambda row: row["length"]),
            4096,
        )
        if profile == "long":
            coding_span = coding_stable_history_block(
                messages=selected,
                source_ids=source_ids,
                target_ids=target_ids,
                tokenizer=tokenizer,
            )
        else:
            coding_span = coding_tool_block(
                messages=selected,
                source_ids=source_ids,
                target_ids=target_ids,
                tokenizer=tokenizer,
            )
        cases.append(
            {
                "case_id": instance_id,
                "source_input_ids": source_ids,
                "source_prompt_tokens": len(source_ids),
                "target_input_ids": target_ids,
                "target_prompt_tokens": len(target_ids),
            }
        )
        manifests["general"].append(
            manifest_case(
                case_id=instance_id,
                policy_label="general_common_suffix",
                source_ids=source_ids,
                target_ids=target_ids,
                span=general_span,
            )
        )
        manifests["coding_aware"].append(
            manifest_case(
                case_id=instance_id,
                policy_label="coding_tool_observation",
                source_ids=source_ids,
                target_ids=target_ids,
                span=coding_span,
            )
        )

    cases_path = output / "PILOT_CASES.json"
    write_json(cases_path, {"cases": cases})
    manifest_hashes = {}
    for arm, rows in manifests.items():
        path = output / "manifests" / f"{arm}.json"
        write_json(
            path,
            {
                "cache_dtype": "bfloat16",
                "cases": rows,
                "lease_ttl_s": 900,
                "ledger_path": str(
                    output / "server" / arm / "EXACT_LEDGER.jsonl"
                ),
                "model_id": MODEL,
                "rope": {
                    "base": 10_000_000,
                    "is_neox_style": True,
                    "rotary_dim": 128,
                },
                "version": 2,
            },
        )
        manifest_hashes[arm] = sha256_file(path)
    registration = {
        "date": "2026-07-26",
        "decode": {
            "max_new_tokens": 128,
            "repetitions": 3,
            "temperature": 0,
        },
        "gates": {
            "capacity_or_context_failures": 0,
            "coding_next_command_agreement_not_below_general": True,
            "coding_ttft_below_dense": True,
            "native_copy_event_per_reuse_target": True,
        },
        "inputs": {
            "cases_sha256": sha256_file(cases_path),
            "chat_template_sha256": sha256_file(CHAT_TEMPLATE),
            "manifest_sha256": manifest_hashes,
            "trajectory_sha256": {
                key: sha256_file(path) for key, path in trajectories.items()
            },
        },
        "method": {
            "coding_aware": (
                (
                    "up to 4096 tokens of older completed assistant/tool "
                    "interactions; task instruction, compaction notice, and "
                    "two-turn coding frontier stay dense"
                )
                if profile == "long"
                else (
                    "largest complete unchanged tool-observation block; task "
                    "instruction, compaction notice, and assistant reasoning "
                    "stay dense"
                )
            ),
            "general": "largest shifted token-identical common suffix, capped at 4096",
            "history_change": (
                "replace first completed assistant/tool turn with the frozen "
                "bridge-v1 compaction notice"
            ),
            "prefetch": False,
        },
        "model": MODEL,
        "profile": profile,
        "runtime": {
            "attention_backend": "triton",
            "context_length": 32768,
            "deterministic_inference": True,
            "mem_fraction_static": 0.90,
        },
        "protected": {
            "old_dirty_checkout_modified": False,
            "paper_modified": False,
            "preregistration_thresholds_modified": False,
        },
        "scope": (
            "Three frozen officially-resolved tasks; next-action agreement "
            "pilot only, not official task-level SWE-bench accuracy"
        ),
        "status": "REGISTERED_BEFORE_PILOT_GPU_RUN",
    }
    write_json(output / "PILOT_REGISTRATION.json", registration)
    return registration


def launch_server(
    *,
    output: Path,
    arm: str,
    port: int,
) -> tuple[subprocess.Popen[str], Any, str]:
    base_url = f"http://127.0.0.1:{port}"
    log_path = output / "server" / arm / "server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stream = log_path.open("w", encoding="utf-8")
    env = os.environ.copy()
    env.update(
        {
            "HF_HUB_OFFLINE": "1",
            "PYTHONPATH": f"{PROJECT / 'python'}:{PROJECT}",
            "SGLANG_KVCOMM_CORE": "0" if arm == "dense" else "1",
        }
    )
    if arm != "dense":
        env["SGLANG_KVCOMM_EXACT_CANARY_MANIFEST"] = str(
            output / "manifests" / f"{arm}.json"
        )
    command = [
        "/home/gfy/.conda/envs/sglang-kvflow/bin/python",
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
        "--attention-backend",
        "triton",
        "--mem-fraction-static",
        "0.90",
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
    if arm == "dense":
        command.append("--disable-radix-cache")
    process = subprocess.Popen(
        command,
        cwd=PROJECT,
        env=env,
        stdout=stream,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if process.poll() is not None:
                stream.flush()
                raise RuntimeError(
                    f"{arm} server exited {process.returncode}; inspect {log_path}"
                )
            try:
                response = requests.get(base_url + "/model_info", timeout=2)
                if response.ok:
                    return process, stream, base_url
            except requests.RequestException:
                pass
            time.sleep(2)
        raise TimeoutError(f"{arm} server did not become ready")
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=30)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        stream.close()
        raise


def stop_server(process: subprocess.Popen[str], stream: Any) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=30)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    finally:
        stream.close()


def generate(
    *,
    base_url: str,
    input_ids: list[int],
    key: str,
    max_new_tokens: int,
    stream: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    response = requests.post(
        base_url + "/generate",
        json={
            "extra_key": key,
            "input_ids": input_ids,
            "return_logprob": False,
            "sampling_params": {
                "ignore_eos": False,
                "max_new_tokens": max_new_tokens,
                "temperature": 0,
            },
            "stream": stream,
        },
        stream=stream,
        timeout=900,
    )
    response.raise_for_status()
    if not stream:
        value = response.json()
        return {
            "elapsed_ms": 1000 * (time.perf_counter() - started),
            "output_text": str(value.get("text") or ""),
        }
    value = None
    ttft_ms = math.inf
    for chunk in response.iter_lines(decode_unicode=True):
        if not chunk or not chunk.startswith("data:"):
            continue
        payload = chunk[5:].strip()
        if payload == "[DONE]":
            break
        value = json.loads(payload)
        if "error" in value:
            raise RuntimeError(value["error"])
        completion_tokens = int(
            value.get("meta_info", {}).get("completion_tokens", 0)
        )
        if math.isinf(ttft_ms) and completion_tokens:
            ttft_ms = 1000 * (time.perf_counter() - started)
    if value is None or math.isinf(ttft_ms):
        raise RuntimeError("empty generation stream")
    return {
        "completion_tokens": int(
            value.get("meta_info", {}).get("completion_tokens", 0)
        ),
        "elapsed_ms": 1000 * (time.perf_counter() - started),
        "finish_reason": value.get("meta_info", {}).get("finish_reason"),
        "output_text": str(value.get("text") or ""),
        "ttft_ms": ttft_ms,
    }


def run_arm(
    output: Path, arm: str, port: int, profile: str = "short"
) -> dict[str, Any]:
    if not (output / "PILOT_REGISTRATION.json").exists():
        prepare(output, profile)
    cases = read_json(output / "PILOT_CASES.json")["cases"]
    ledger = output / "server" / arm / "EXACT_LEDGER.jsonl"
    result_path = output / "generations" / f"{arm}.json"
    log_path = output / "server" / arm / "server.log"
    if not result_path.exists() and (ledger.exists() or log_path.exists()):
        archive = output / "failed_canaries" / f"{arm}_{int(time.time())}"
        archive.mkdir(parents=True, exist_ok=True)
        if ledger.exists():
            ledger.replace(archive / ledger.name)
        if log_path.exists():
            log_path.replace(archive / log_path.name)
    if ledger.exists():
        ledger.unlink()
    process, stream, base_url = launch_server(output=output, arm=arm, port=port)
    rows = []
    source_rows = []
    try:
        for case in cases:
            if arm != "dense":
                source_rows.append(
                    {
                        **generate(
                            base_url=base_url,
                            input_ids=case["source_input_ids"],
                            key=f"pilot-source-{arm}-{case['case_id']}",
                            max_new_tokens=1,
                            stream=False,
                        ),
                        "case_id": case["case_id"],
                    }
                )
            for repetition in range(3):
                rows.append(
                    {
                        **generate(
                            base_url=base_url,
                            input_ids=case["target_input_ids"],
                            key=(
                                f"pilot-target-{arm}-{case['case_id']}-"
                                f"r{repetition}"
                            ),
                            max_new_tokens=128,
                            stream=True,
                        ),
                        "arm": arm,
                        "case_id": case["case_id"],
                        "repetition": repetition,
                        "target_prompt_tokens": case["target_prompt_tokens"],
                    }
                )
    finally:
        stop_server(process, stream)
    ledger_rows = []
    if ledger.exists():
        ledger_rows = [
            json.loads(line)
            for line in ledger.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    value = {
        "arm": arm,
        "ledger_rows": ledger_rows,
        "rows": rows,
        "source_rows": source_rows,
        "status": "complete",
    }
    write_json(result_path, value)
    return {
        "arm": arm,
        "copy_events": sum(
            row.get("event") == "target_copied" for row in ledger_rows
        ),
        "rows": len(rows),
        "status": "complete",
    }


def command_from_output(value: str) -> str | None:
    match = re.search(
        r"<function=bash>\s*<parameter=command>(.*?)</parameter>",
        value,
        flags=re.DOTALL,
    )
    if match is None:
        return None
    return " ".join(match.group(1).split())


def prefix_agreement(left: str, right: str) -> float:
    size = min(len(left), len(right))
    matched = 0
    while matched < size and left[matched] == right[matched]:
        matched += 1
    return matched / max(1, max(len(left), len(right)))


def summarize(output: Path) -> dict[str, Any]:
    values = {
        arm: read_json(output / "generations" / f"{arm}.json")
        for arm in ARMS
    }
    dense_rows = {
        (row["case_id"], row["repetition"]): row
        for row in values["dense"]["rows"]
    }
    arms = {}
    comparisons = {}
    for arm, value in values.items():
        rows = value["rows"]
        copy_rows = [
            row
            for row in value["ledger_rows"]
            if row.get("event") == "target_copied"
        ]
        fallback_rows = [
            row
            for row in value["ledger_rows"]
            if row.get("event") == "target_fallback"
        ]
        arms[arm] = {
            "cases": len({row["case_id"] for row in rows}),
            "copied_tokens": sum(
                int(row.get("copied_k_tokens", 0)) for row in copy_rows
            ),
            "copy_events": len(copy_rows),
            "fallback_events": len(fallback_rows),
            "median_ttft_ms": statistics.median(
                row["ttft_ms"] for row in rows
            ),
            "p95_ttft_ms": sorted(row["ttft_ms"] for row in rows)[
                math.ceil(0.95 * len(rows)) - 1
            ],
            "rows": len(rows),
            "source_release_events": sum(
                row.get("event") == "target_complete"
                and bool(row.get("source_released"))
                for row in value["ledger_rows"]
            ),
            "source_build_total_ms": sum(
                row["elapsed_ms"] for row in value["source_rows"]
            ),
            "ttft_by_case_ms": {
                case_id: statistics.median(
                    row["ttft_ms"]
                    for row in rows
                    if row["case_id"] == case_id
                )
                for case_id in sorted({row["case_id"] for row in rows})
            },
        }
        if arm == "dense":
            continue
        agreements = []
        dense_command_eligible = 0
        prefixes = []
        exact = []
        for row in rows:
            dense = dense_rows[(row["case_id"], row["repetition"])]
            dense_command = command_from_output(dense["output_text"])
            reuse_command = command_from_output(row["output_text"])
            if dense_command is not None:
                dense_command_eligible += 1
                agreements.append(reuse_command == dense_command)
            exact.append(row["output_text"] == dense["output_text"])
            prefixes.append(
                prefix_agreement(row["output_text"], dense["output_text"])
            )
        comparisons[arm] = {
            "command_agreement": (
                statistics.mean(agreements) if agreements else None
            ),
            "command_agreement_denominator": dense_command_eligible,
            "dense_no_command_rows": len(rows) - dense_command_eligible,
            "exact_output_agreement": statistics.mean(exact),
            "mean_character_prefix_agreement": statistics.mean(prefixes),
            "ttft_reduction_vs_dense_fraction": (
                1
                - arms[arm]["median_ttft_ms"]
                / arms["dense"]["median_ttft_ms"]
            ),
        }
        amortized_build = (
            arms[arm]["source_build_total_ms"] / len(rows)
        )
        conservative_ttft = arms[arm]["median_ttft_ms"] + amortized_build
        comparisons[arm].update(
            amortized_source_build_ms_per_target=amortized_build,
            median_ttft_plus_amortized_source_build_ms=conservative_ttft,
            ttft_reduction_vs_dense_including_amortized_build_fraction=(
                1
                - conservative_ttft
                / arms["dense"]["median_ttft_ms"]
            ),
        )
    expected_copies = 9
    capacity_failures = 0
    for arm in ARMS:
        log = output / "server" / arm / "server.log"
        text = log.read_text(encoding="utf-8", errors="replace")
        capacity_failures += len(
            re.findall(
                r"KV cache pool is full|exceeds the available KV cache|"
                r"maximum context length|ContextWindowExceeded",
                text,
                flags=re.IGNORECASE,
            )
        )
    gates = {
        "capacity_or_context_failures": capacity_failures == 0,
        "coding_next_command_agreement_not_below_general": (
            comparisons["coding_aware"]["command_agreement"]
            >= comparisons["general"]["command_agreement"]
        ),
        "coding_ttft_below_dense": (
            arms["coding_aware"]["median_ttft_ms"]
            < arms["dense"]["median_ttft_ms"]
        ),
        "native_copy_event_per_reuse_target": (
            arms["general"]["copy_events"] == expected_copies
            and arms["coding_aware"]["copy_events"] == expected_copies
            and arms["general"]["fallback_events"] == 0
            and arms["coding_aware"]["fallback_events"] == 0
        ),
    }
    result = {
        "arms": arms,
        "comparisons": comparisons,
        "gates": gates,
        "status": "PILOT_PASS" if all(gates.values()) else "PILOT_FAIL",
    }
    write_json(output / "PILOT_RESULT.json", result)
    if all(
        (output / "generations" / f"{arm}.json").exists() for arm in ARMS
    ):
        write_json(
            output / "PILOT_STATUS.json",
            {
                "active_arm": None,
                "completed": list(ARMS),
                "result_status": result["status"],
                "status": "complete",
            },
        )
    return result


def campaign(output: Path, port: int, profile: str = "short") -> dict[str, Any]:
    prepare(output, profile)
    status_path = output / "PILOT_STATUS.json"
    completed = []
    for arm in ARMS:
        write_json(
            status_path,
            {"active_arm": arm, "completed": completed, "status": "running"},
        )
        run_arm(output, arm, port, profile)
        completed.append(arm)
    result = summarize(output)
    write_json(
        status_path,
        {
            "active_arm": None,
            "completed": completed,
            "result_status": result["status"],
            "status": "complete",
        },
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--profile", choices=("short", "long"), default="short")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    run_parser = subparsers.add_parser("run-arm")
    run_parser.add_argument("--arm", choices=ARMS, required=True)
    run_parser.add_argument("--port", type=int, default=33300)
    subparsers.add_parser("summarize")
    campaign_parser = subparsers.add_parser("campaign")
    campaign_parser.add_argument("--port", type=int, default=33300)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.command == "prepare":
        value = prepare(output, args.profile)
    elif args.command == "run-arm":
        value = run_arm(output, args.arm, args.port, args.profile)
    elif args.command == "summarize":
        value = summarize(output)
    else:
        value = campaign(output, args.port, args.profile)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
