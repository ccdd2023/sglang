#!/usr/bin/env python3
"""Generate frozen SWE-bench predictions through an OpenAI-compatible server."""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_REGISTRATION = HERE / "swebench_verified_complex_v1.json"
DEFAULT_ORACLE_ROOT = Path(
    "/home/gfy/CodeMAS_Project/kvflow-artifacts/"
    "swebench_verified_complex_v1_20260724"
)
DEFAULT_OUTPUT = DEFAULT_ORACLE_ROOT / "dense_qwen25_7b_oracle_ast_v1"


def run(command: list[str], timeout: int = 300) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        command,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def patch_targets(patch: str) -> dict[str, list[int]]:
    targets: dict[str, list[int]] = {}
    current = ""
    for line in patch.splitlines():
        match = re.match(r"diff --git a/(.+?) b/(.+)$", line)
        if match:
            current = match.group(2)
            targets.setdefault(current, [])
            continue
        match = re.match(r"@@ -(?P<start>\d+)(?:,\d+)? \+\d+(?:,\d+)? @@", line)
        if match and current:
            targets[current].append(int(match.group("start")))
    return targets


def read_image_file(image: str, path: str) -> str | None:
    proc = run(
        [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "cat",
            image,
            f"/testbed/{path}",
        ],
        timeout=300,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout.decode("utf-8", errors="replace")


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1] + 4:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def ast_intervals(
    source: str,
    hunk_lines: list[int],
    *,
    window_lines: int,
    max_node_lines: int,
) -> list[tuple[int, int]]:
    lines = source.splitlines()
    line_count = len(lines)
    try:
        tree = ast.parse(source)
    except SyntaxError:
        tree = None
    nodes: list[ast.AST] = []
    if tree is not None:
        nodes = [
            node
            for node in ast.walk(tree)
            if isinstance(
                node,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
            )
            and getattr(node, "end_lineno", None) is not None
        ]

    intervals = []
    for line in hunk_lines or [1]:
        candidates = [
            node
            for node in nodes
            if node.lineno <= line <= int(node.end_lineno)
        ]
        candidates.sort(key=lambda node: int(node.end_lineno) - node.lineno)
        node = candidates[0] if candidates else None
        if node is not None and int(node.end_lineno) - node.lineno + 1 <= max_node_lines:
            decorator_lines = [
                decorator.lineno for decorator in getattr(node, "decorator_list", [])
            ]
            start = min([node.lineno, *decorator_lines])
            end = int(node.end_lineno)
        else:
            start = max(1, line - window_lines)
            end = min(line_count, line + window_lines)
        intervals.append((start, end))
    return merge_intervals(intervals)


def context_blocks(
    *,
    image: str,
    patch: str,
    max_context_chars: int,
    window_lines: int,
    max_node_lines: int,
) -> tuple[str, dict[str, Any]]:
    targets = patch_targets(patch)
    blocks = []
    metadata: dict[str, Any] = {"files": []}
    remaining = max_context_chars
    for path, hunk_lines in targets.items():
        source = read_image_file(image, path)
        if source is None:
            block = (
                f"===== {path} (absent at the base commit; create if needed) =====\n"
            )
            intervals: list[tuple[int, int]] = []
        else:
            lines = source.splitlines()
            intervals = ast_intervals(
                source,
                hunk_lines,
                window_lines=window_lines,
                max_node_lines=max_node_lines,
            )
            excerpts = []
            for start, end in intervals:
                excerpt = "\n".join(lines[start - 1 : end])
                excerpts.append(
                    f"----- exact base lines {start}-{end} -----\n{excerpt}"
                )
            block = f"===== {path} =====\n" + "\n\n".join(excerpts) + "\n"
        if len(block) > remaining:
            block = block[:remaining]
            block += "\n[context truncated by the frozen character budget]\n"
        blocks.append(block)
        remaining -= len(block)
        metadata["files"].append(
            {
                "path": path,
                "hunk_lines": hunk_lines,
                "intervals": intervals,
                "source_chars": len(source) if source is not None else 0,
                "context_chars": len(block),
            }
        )
        if remaining <= 0:
            break
    joined = "\n".join(blocks)
    metadata["context_chars"] = len(joined)
    metadata["target_file_count"] = len(targets)
    return joined, metadata


def build_prompt(instance: dict[str, Any], context: str) -> str:
    return (
        "Fix the following real repository issue using the exact base-code "
        "excerpts supplied below. File and hunk localization is provided, but "
        "the reference solution is not. Return ONLY a unified git diff "
        "beginning with `diff --git`. Keep the change minimal, preserve "
        "unrelated behavior, and do not edit tests. Diff context must be copied "
        "exactly from the supplied base code.\n\n"
        f"Repository: {instance['repo']}\n"
        f"Base commit: {instance['base_commit']}\n\n"
        f"Issue:\n{instance['problem_statement']}\n\n"
        f"Oracle-localized base code:\n{context}"
    )


def stream_chat(
    *,
    endpoint: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a repository maintenance agent. Output only an "
                    "applicable unified git diff and no prose or Markdown."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    first_content_at: float | None = None
    text_parts = []
    usage: dict[str, Any] = {}
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                usage = chunk.get("usage") or usage
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                content = str(choices[0].get("delta", {}).get("content") or "")
                if content:
                    if first_content_at is None:
                        first_content_at = time.perf_counter()
                    text_parts.append(content)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail[-2000:]}") from exc
    finished = time.perf_counter()
    return {
        "text": "".join(text_parts),
        "ttft_ms": (
            round((first_content_at - started) * 1000, 3)
            if first_content_at is not None
            else None
        ),
        "elapsed_ms": round((finished - started) * 1000, 3),
        "usage": usage,
    }


def extract_diff(text: str) -> str:
    fenced = re.search(r"```(?:diff)?\s*\n(.*?)```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1)
    start = text.find("diff --git ")
    if start < 0:
        return ""
    return text[start:].strip() + "\n"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, default=DEFAULT_REGISTRATION)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_ORACLE_ROOT / "frozen_subset.json",
    )
    parser.add_argument(
        "--oracle-result",
        type=Path,
        default=DEFAULT_ORACLE_ROOT / "ORACLE_RESULT.json",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--endpoint", default="http://127.0.0.1:30000")
    parser.add_argument(
        "--model",
        default="/home/gfy/models/Qwen2.5-7B-Instruct",
    )
    parser.add_argument(
        "--run-id",
        default="dense_qwen25_7b_oracle_ast_v1_20260724",
        help="Stable identifier written to the generation registration.",
    )
    parser.add_argument(
        "--model-label",
        default="impactkv/dense-qwen25-7b-oracle-ast-v1",
        help="Method label written to SWE-bench predictions.",
    )
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--request-timeout", type=int, default=1800)
    parser.add_argument("--max-context-chars", type=int, default=72000)
    parser.add_argument("--window-lines", type=int, default=60)
    parser.add_argument("--max-node-lines", type=int, default=320)
    args = parser.parse_args()

    registration = json.loads(args.registration.read_text(encoding="utf-8"))
    rows = json.loads(args.dataset.read_text(encoding="utf-8"))
    oracle = json.loads(args.oracle_result.read_text(encoding="utf-8"))
    eligible = set(oracle["oracle_valid_instance_ids"])
    image_by_id = {
        row["instance_id"]: row["image"] for row in registration["instances"]
    }
    selected = [row for row in rows if row["instance_id"] in eligible]
    args.output.mkdir(parents=True, exist_ok=True)

    generation_registration = {
        "registration_id": args.run_id,
        "source_registration": registration["registration_id"],
        "oracle_result": str(args.oracle_result),
        "eligible_instance_ids": [row["instance_id"] for row in selected],
        "model": args.model,
        "endpoint": args.endpoint,
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "context_policy": {
            "name": "oracle-hunk AST context",
            "gold_information_used": "target file paths and old hunk line numbers only",
            "gold_patch_content_exposed": False,
            "max_context_chars": args.max_context_chars,
            "window_lines": args.window_lines,
            "max_node_lines": args.max_node_lines,
        },
        "output_protocol": "unified git diff with deterministic fence stripping",
        "functional_metric": "official SWE-bench resolved",
        "prefetch": False,
    }
    write_json(args.output / "REGISTRATION.json", generation_registration)

    predictions = []
    telemetry = []
    for index, instance in enumerate(selected, start=1):
        instance_id = instance["instance_id"]
        print(f"[{index}/{len(selected)}] {instance_id}", flush=True)
        context, context_meta = context_blocks(
            image=image_by_id[instance_id],
            patch=instance["patch"],
            max_context_chars=args.max_context_chars,
            window_lines=args.window_lines,
            max_node_lines=args.max_node_lines,
        )
        prompt = build_prompt(instance, context)
        instance_dir = args.output / "instances" / instance_id
        instance_dir.mkdir(parents=True, exist_ok=True)
        (instance_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
        write_json(instance_dir / "context.json", context_meta)
        response = stream_chat(
            endpoint=args.endpoint,
            model=args.model,
            prompt=prompt,
            max_tokens=args.max_tokens,
            timeout=args.request_timeout,
        )
        raw = response["text"]
        patch = extract_diff(raw)
        (instance_dir / "model_output.txt").write_text(raw, encoding="utf-8")
        (instance_dir / "candidate.patch").write_text(patch, encoding="utf-8")
        predictions.append(
            {
                "instance_id": instance_id,
                "model_name_or_path": args.model_label,
                "model_patch": patch,
            }
        )
        telemetry.append(
            {
                "instance_id": instance_id,
                "ttft_ms": response["ttft_ms"],
                "elapsed_ms": response["elapsed_ms"],
                "usage": response["usage"],
                "prompt_chars": len(prompt),
                "context": context_meta,
                "raw_output_chars": len(raw),
                "diff_chars": len(patch),
                "diff_extracted": bool(patch),
            }
        )
        write_json(args.output / "TELEMETRY.json", telemetry)
        (args.output / "predictions.jsonl").write_text(
            "".join(
                json.dumps(row, ensure_ascii=False) + "\n" for row in predictions
            ),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
