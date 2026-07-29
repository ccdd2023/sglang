#!/usr/bin/env python3
"""Execute one or more immutable fair-comparison canary commands."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from benchmark.multi_workflow.fair_sota_comparison_v2 import ARTIFACT_ROOT


DEFAULT_PLAN = ARTIFACT_ROOT / "CANARY_COMMAND_PLAN.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def expected_output(command: dict[str, Any]) -> Path:
    argv = list(command["argv"])
    if "--metrics" in argv:
        return Path(argv[argv.index("--metrics") + 1])
    if "--output" not in argv:
        raise ValueError(f"{command['command_id']}: cannot locate output")
    output = Path(argv[argv.index("--output") + 1])
    if command["mode"] == "prepare":
        return output / "REGISTRATION.json"
    return output / f"{command['mode']}.json"


def clone_for_retry(
    command: dict[str, Any],
    retry_tag: str,
) -> dict[str, Any]:
    """Create a non-overwriting retry while preserving the original command."""

    if not retry_tag or any(character in retry_tag for character in "/\\ "):
        raise ValueError("retry_tag must be one path-safe word")
    cloned = json.loads(json.dumps(command))
    cloned["command_id"] = f"{command['command_id']}-{retry_tag}"
    argv = list(cloned["argv"])
    for flag in ("--metrics", "--output-dir", "--output"):
        if flag not in argv:
            continue
        index = argv.index(flag) + 1
        path = Path(argv[index])
        if flag == "--metrics":
            path = path.with_name(f"{path.stem}-{retry_tag}{path.suffix}")
        else:
            path = path.with_name(f"{path.name}-{retry_tag}")
        argv[index] = str(path)
    if "--run-id" in argv:
        index = argv.index("--run-id") + 1
        argv[index] = f"{argv[index]}-{retry_tag}"
    cloned["argv"] = argv
    return cloned


def execute_command(
    command: dict[str, Any],
    *,
    status_root: Path,
) -> dict[str, Any]:
    command_id = str(command["command_id"])
    status_path = status_root / f"{command_id}.status.json"
    stdout_path = status_root / f"{command_id}.stdout.log"
    stderr_path = status_root / f"{command_id}.stderr.log"
    output = expected_output(command)
    for path in (status_path, stdout_path, stderr_path, output):
        if path.exists():
            raise FileExistsError(
                f"{command_id}: refusing to overwrite existing {path}"
            )
    env = os.environ.copy()
    env.update({str(key): str(value) for key, value in command["env"].items()})
    started = _utc_now()
    status_root.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        completed = subprocess.run(
            list(command["argv"]),
            cwd=command["workdir"],
            env=env,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    status = {
        "command_id": command_id,
        "argv": command["argv"],
        "comparison_layer": command["comparison_layer"],
        "method": command["method"],
        "mode": command["mode"],
        "started_at_utc": started,
        "finished_at_utc": _utc_now(),
        "exit_code": completed.returncode,
        "expected_output": str(output),
        "expected_output_exists": output.exists(),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }
    _write_json(status_path, status)
    if completed.returncode != 0 or not output.exists():
        raise RuntimeError(
            f"{command_id} failed with exit={completed.returncode}; "
            f"see {stderr_path}"
        )
    return status


def select_commands(
    plan: dict[str, Any],
    command_ids: Sequence[str],
    *,
    run_all: bool,
) -> list[dict[str, Any]]:
    commands = list(plan["commands"])
    if run_all:
        return commands
    requested = list(command_ids)
    if not requested:
        raise ValueError("select --command-id or --all")
    by_id = {str(command["command_id"]): command for command in commands}
    missing = set(requested).difference(by_id)
    if missing:
        raise ValueError(f"unknown command IDs: {sorted(missing)}")
    return [by_id[command_id] for command_id in requested]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--command-id", action="append", default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--retry-tag")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    selected = select_commands(
        plan,
        args.command_id,
        run_all=args.all,
    )
    if args.retry_tag:
        selected = [
            clone_for_retry(command, args.retry_tag) for command in selected
        ]
    status_root = args.plan.parent / "canary/execution"
    results = [
        execute_command(command, status_root=status_root)
        for command in selected
    ]
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
