#!/usr/bin/env python3
"""Freeze a canonical V11 raw event-to-file provenance manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark.multi_workflow.sessiongraph_raw_provenance import (
    build_canonical_manifest,
)
from benchmark.multi_workflow.sessiongraph_v11 import write_jsonl


def build(raw_root: Path, manifest_path: Path, events_path: Path, gate_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    events, report = build_canonical_manifest(raw_root, manifest)
    write_jsonl(events_path, (event.row() for event in events))
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    gate_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--events-output", type=Path, required=True)
    parser.add_argument("--gate-output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                args.raw_root,
                args.manifest,
                args.events_output,
                args.gate_output,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
