#!/usr/bin/env python3
"""Validate and summarize the RepoBench-P fair-comparison canary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmark.multi_workflow.cacheblend_coding_matrix import summarize
from benchmark.multi_workflow.fair_sota_comparison_v2 import (
    ARTIFACT_ROOT,
    _read_jsonl,
    _write_json,
    canonical_sha256,
    token_identity_audit,
    validate_ledger,
)
from benchmark.multi_workflow.run_v40_repobench_control import (
    summarize as summarize_v40,
)
from benchmark.multi_workflow.summarize_kvcomm_repobench import (
    summarize_kvcomm,
)


def _target_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in records
        if not row.get("metadata", {}).get("warmup")
        and not row.get("metadata", {}).get("source_observation")
    ]


def _v40_token_records(output: Path) -> list[dict[str, Any]]:
    cases = json.loads((output / "CASES.json").read_text(encoding="utf-8"))[
        "cases"
    ]
    return [
        {
            "case_id": case["case_id"],
            "token_ids_sha256": canonical_sha256(case["target_input_ids"]),
            "metadata": {},
        }
        for case in cases
    ]


def _compact(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "quality": result["quality"],
        "latency": result["latency"],
        "physical_reuse": result["physical_reuse"],
    }


def summarize_canary(output: Path = ARTIFACT_ROOT) -> dict[str, Any]:
    workload_path = output / "static/repobench-p/CANARY_WORKLOAD.json"
    workload = json.loads(workload_path.read_text(encoding="utf-8"))
    cacheblend_root = output / "canary/repobench-p/cacheblend"
    kvcomm_root = output / "canary/repobench-p/kvcomm"
    v40_root = output / "canary/repobench-p/v40/cap-4096"
    cacheblend_dense = _read_jsonl(cacheblend_root / "dense.jsonl")
    cacheblend_reuse = _read_jsonl(cacheblend_root / "reuse.jsonl")
    kvcomm_dense = _read_jsonl(kvcomm_root / "dense-budget64.jsonl")
    kvcomm_reuse = _read_jsonl(kvcomm_root / "reuse-budget64.jsonl")

    validation = {
        "cacheblend_dense": validate_ledger(
            workload,
            cacheblend_dense,
            expected_method="cacheblend",
            expected_mode="dense",
        ),
        "cacheblend_reuse": validate_ledger(
            workload,
            cacheblend_reuse,
            expected_method="cacheblend",
            expected_mode="reuse",
        ),
        "kvcomm_dense": validate_ledger(
            workload,
            kvcomm_dense,
            expected_method="kvcomm",
            expected_mode="dense",
        ),
        "kvcomm_reuse": validate_ledger(
            workload,
            kvcomm_reuse,
            expected_method="kvcomm",
            expected_mode="reuse",
        ),
    }
    cacheblend_result = summarize(
        workload,
        cacheblend_dense,
        cacheblend_reuse,
        recompute_ratio=0.5,
    )
    kvcomm_result = summarize_kvcomm(
        workload,
        kvcomm_dense,
        kvcomm_reuse,
        threshold=0.5,
    )
    v40_result = summarize_v40(v40_root)
    controlled_identity = token_identity_audit(
        {
            "cacheblend": _target_records(cacheblend_dense),
            "v40": _v40_token_records(v40_root),
        }
    )
    native_identity = token_identity_audit(
        {
            "cacheblend": _target_records(cacheblend_dense),
            "kvcomm": _target_records(kvcomm_dense),
            "v40": _v40_token_records(v40_root),
        }
    )
    passed = (
        controlled_identity["controlled_rankable"]
        and native_identity["classification"] == "native_only"
        and v40_result["status"] == "COMPLETE"
        and validation["cacheblend_reuse"]["physical_reuse_records"] == 3
        and validation["kvcomm_reuse"]["physical_reuse_records"] == 3
    )
    value = {
        "schema_version": 1,
        "status": "PASS" if passed else "FAIL",
        "scope": (
            "three-case protocol/mechanism canary; not an accuracy or "
            "superiority result"
        ),
        "validation": validation,
        "token_identity": {
            "v40_vs_cacheblend": controlled_identity,
            "including_native_kvcomm": native_identity,
        },
        "methods": {
            "v40": _compact(v40_result),
            "cacheblend": _compact(cacheblend_result),
            "kvcomm": _compact(kvcomm_result),
        },
        "kvcomm_adapter": {
            "algorithm_base": "3bf7410ca3fd63930241f9332e0c396c91fc05ed",
            "adapter_commit": "66f89fb6b5f64e3d7eff2511d8c5922ab641acde",
            "fixes": [
                "materialize source before canary target",
                "reset native cache between cases",
                "honor workload max_new_tokens=64 for every graph agent",
            ],
        },
        "excluded_failed_diagnostics": [
            "kvcomm/reuse.jsonl: no source materialization and accumulated OOM",
            "kvcomm/reuse-retry1.jsonl: 512-token agent budget caused OOM",
        ],
    }
    _write_json(output / "CANARY_AUDIT.json", value)
    return value


def render_markdown(value: dict[str, Any]) -> str:
    lines = [
        "# Fair SOTA comparison: protocol canary",
        "",
        f"Status: **{value['status']}**",
        "",
        (
            "This is a three-case protocol/mechanism check. It is not an "
            "accuracy or superiority result."
        ),
        "",
        "| Method | Dense→reuse exact | Cache-ready speedup | N=4 incl. build | Physical reuse |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in ("v40", "cacheblend", "kvcomm"):
        result = value["methods"][method]
        quality = result["quality"]
        latency = result["latency"]
        physical = result["physical_reuse"]
        if method == "v40":
            dense_exact = quality["dense_exact_line"]
            reuse_exact = quality["reuse_exact_line"]
            speed = latency["cache_ready_speedup"]
            n4 = latency["n4_including_build_speedup"]
            copies = physical["copy_events"]
        else:
            dense_exact = quality["dense_exact_line"]
            reuse_exact = quality["reuse_exact_line"]
            speed = latency["cache_ready_speedup_vs_native_dense"]
            n4 = latency["build_amortized"]["4"][
                "speedup_vs_native_dense"
            ]
            copies = value["validation"][f"{method}_reuse"][
                "physical_reuse_records"
            ]
        lines.append(
            f"| {method} | {dense_exact}/3→{reuse_exact}/3 | "
            f"{speed:.3f}× | {n4:.3f}× | {copies}/3 |"
        )
    identity = value["token_identity"]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "- V40 and CacheBlend consumed identical target token IDs: "
                f"{identity['v40_vs_cacheblend']['controlled_rankable']}."
            ),
            (
                "- KVCOMM remains native-only because its three-agent graph "
                "rewrites the model prompt."
            ),
            (
                "- Failed pre-fix KVCOMM ledgers remain preserved and are "
                "excluded explicitly rather than overwritten."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ARTIFACT_ROOT)
    args = parser.parse_args()
    value = summarize_canary(args.output)
    markdown = render_markdown(value)
    (args.output / "CANARY_AUDIT.md").write_text(markdown, encoding="utf-8")
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
