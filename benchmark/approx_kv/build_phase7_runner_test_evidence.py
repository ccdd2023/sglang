#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from benchmark.approx_kv.build_phase7_manifest import RUNNER_SPECS
from benchmark.approx_kv.phase6.schema import file_sha256
from benchmark.approx_kv.phase7.evidence import (
    build_runner_test_evidence,
    write_runner_test_evidence,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", choices=tuple(RUNNER_SPECS), required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument(
        "--command",
        help=(
            "defaults to the frozen required_cpu_test of the runner; a "
            "different command is rejected"
        ),
    )
    parser.add_argument("--exit-code", type=int, required=True)
    parser.add_argument("--summary-line", required=True)
    parser.add_argument("--passed-count", type=int, required=True)
    parser.add_argument("--subtests-passed-count", type=int, required=True)
    parser.add_argument("--subtest", action="append", default=[])
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    spec = RUNNER_SPECS[args.runner]
    required_command = spec["required_cpu_test"]
    command = args.command or required_command
    if command != required_command:
        raise ValueError(
            f"{args.runner} CPU evidence command must be {required_command!r}"
        )
    runner_path = Path(spec["path"])
    payload = build_runner_test_evidence(
        runner_key=args.runner,
        runner_module=spec["module"],
        runner_path=str(runner_path),
        runner_sha256=file_sha256(REPO_ROOT / runner_path),
        image_digest=args.image_digest,
        command=command,
        exit_code=args.exit_code,
        summary_line=args.summary_line,
        passed_count=args.passed_count,
        subtests_passed_count=args.subtests_passed_count,
        subtests=args.subtest,
        timestamp=args.timestamp,
    )
    write_runner_test_evidence(args.output, payload)
    print(f"wrote {args.runner} CPU test evidence to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
