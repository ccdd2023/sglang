#!/usr/bin/env python3
"""Compare per-instance SWE-bench outcomes across Docker and Enroot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_report(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value.get("report", value)


def outcome(report: dict[str, Any], instance_id: str) -> str:
    for key, label in (
        ("resolved_ids", "resolved"),
        ("unresolved_ids", "unresolved"),
        ("empty_patch_ids", "empty_patch"),
        ("error_ids", "error"),
        ("incomplete_ids", "incomplete"),
    ):
        if instance_id in report.get(key, []):
            return label
    return "missing"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docker", type=Path, required=True)
    parser.add_argument("--enroot", type=Path, required=True)
    parser.add_argument("--instances", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    docker = read_report(args.docker)
    enroot = read_report(args.enroot)
    rows = []
    for instance_id in filter(None, args.instances.split(",")):
        docker_outcome = outcome(docker, instance_id)
        enroot_outcome = outcome(enroot, instance_id)
        rows.append(
            {
                "instance_id": instance_id,
                "docker": docker_outcome,
                "enroot": enroot_outcome,
                "match": docker_outcome == enroot_outcome,
            }
        )
    value = {
        "schema_version": 1,
        "matched": sum(row["match"] for row in rows),
        "total": len(rows),
        "all_match": all(row["match"] for row in rows),
        "instances": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(value, ensure_ascii=False, indent=2))
    raise SystemExit(0 if value["all_match"] else 1)


if __name__ == "__main__":
    main()
