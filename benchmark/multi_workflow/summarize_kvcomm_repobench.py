#!/usr/bin/env python3
"""Summarize the narrowed native KVCOMM RepoBench-P control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark.multi_workflow.cacheblend_coding_matrix import (
    _read_jsonl,
    summarize,
)


def summarize_kvcomm(
    workload: dict,
    dense_records: list[dict],
    reuse_records: list[dict],
    threshold: float,
) -> dict:
    value = summarize(
        workload,
        dense_records,
        reuse_records,
        recompute_ratio=0.0,
    )
    value["method"] = "KVCOMM"
    value["engine"] = "native Transformers multi-agent graph"
    value["config"] = {
        "threshold": threshold,
        "max_anchor_num": 20,
        "window_size": 5,
        "max_new_tokens": 64,
    }
    value.pop("recompute_ratio", None)
    value["claim_scope"] = (
        "KVCOMM reuse versus its native Dense graph on the same frozen "
        "RepoBench-P case IDs and public output instruction. Absolute latency "
        "includes the native three-agent plus FinalRefer topology."
    )
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--dense", type=Path, required=True)
    parser.add_argument("--reuse", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    value = summarize_kvcomm(
        json.loads(args.workload.read_text(encoding="utf-8")),
        _read_jsonl(args.dense),
        _read_jsonl(args.reuse),
        args.threshold,
    )
    args.output.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
