#!/usr/bin/env python3
"""Zero-GPU derivation of the plan section 5.9 per-token consistency rate.

The frozen CL1 runner stores only boolean first_token_match and
quality_8_token_match. Section 5.9 additionally requires a per-token
consistency rate and the decode eviction delta, so this script derives both
from the committed CL1 raw artifact without re-running any GPU experiment.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def token_consistency(dense: list[int], approx: list[int]) -> dict:
    width = min(len(dense), len(approx))
    matches = [dense[index] == approx[index] for index in range(width)]
    prefix = 0
    for matched in matches:
        if not matched:
            break
        prefix += 1
    return {
        "compared_tokens": width,
        "matched_tokens": sum(matches),
        "token_consistency_rate": (sum(matches) / width) if width else None,
        "longest_matching_prefix": prefix,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text())
    rows = []
    for row in payload["results"]:
        for repeat in row["formal"]:
            quality = token_consistency(
                repeat["dense"]["quality"]["output_ids"],
                repeat["approx"]["quality"]["output_ids"],
            )
            rows.append(
                {
                    "candidate": row["candidate"],
                    "body_tokens": row["body_tokens"],
                    "restart_index": row["restart_index"],
                    "repeat_index": repeat["repeat_index"],
                    "first_token_match": repeat["first_token_match"],
                    "quality_8_token_match": repeat["quality_8_token_match"],
                    "quality_token_consistency": quality,
                    "approx_fallback_tokens": repeat["approx"]["fallback_tokens"],
                    "approx_fallback_evidence": (
                        "indirectly_verified"
                        if repeat["approx"]["fallback_tokens"] is None
                        else "explicit_counter"
                    ),
                    "dense_decode_eviction_tokens": repeat["dense"][
                        "decode_eviction_tokens"
                    ],
                    "approx_decode_eviction_tokens": repeat["approx"][
                        "decode_eviction_tokens"
                    ],
                    "approx_quality_cached_tokens": repeat["approx"]["quality"][
                        "cached_tokens"
                    ],
                    "dense_quality_cached_tokens": repeat["dense"]["quality"][
                        "cached_tokens"
                    ],
                }
            )

    summary = {}
    keys = sorted({(row["candidate"], row["body_tokens"]) for row in rows})
    for candidate, body_tokens in keys:
        selected = [
            row
            for row in rows
            if row["candidate"] == candidate and row["body_tokens"] == body_tokens
        ]
        rates = [
            row["quality_token_consistency"]["token_consistency_rate"]
            for row in selected
        ]
        prefixes = [
            row["quality_token_consistency"]["longest_matching_prefix"]
            for row in selected
        ]
        summary[f"{candidate}:{body_tokens}"] = {
            "samples": len(selected),
            "first_token_match_rate": sum(row["first_token_match"] for row in selected)
            / len(selected),
            "quality_exact_match_rate": sum(
                row["quality_8_token_match"] for row in selected
            )
            / len(selected),
            "median_token_consistency_rate": statistics.median(rates),
            "mean_token_consistency_rate": statistics.fmean(rates),
            "median_longest_matching_prefix": statistics.median(prefixes),
            "fallback_evidence": sorted(
                {row["approx_fallback_evidence"] for row in selected}
            ),
        }

    overall_rates = [
        row["quality_token_consistency"]["token_consistency_rate"] for row in rows
    ]
    output = {
        "schema_version": 1,
        "derivation": "zero_gpu_from_cl1_raw",
        "source_artifact": str(args.input),
        "source_run_id": payload["run_id"],
        "source_raw_sha256": payload.get("raw_sha256"),
        "source_git_sha": payload.get("source_git_sha"),
        "guardrail_note": (
            "the frozen CL1 runner treats a missing "
            "sglang:approx_kv_dense_fallback_total series as zero fallback; "
            "this derivation reports it as indirectly_verified instead"
        ),
        "overall": {
            "samples": len(rows),
            "first_token_match_rate": sum(row["first_token_match"] for row in rows)
            / len(rows),
            "quality_exact_match_rate": sum(
                row["quality_8_token_match"] for row in rows
            )
            / len(rows),
            "median_token_consistency_rate": statistics.median(overall_rates),
            "mean_token_consistency_rate": statistics.fmean(overall_rates),
        },
        "by_candidate_body": summary,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["overall"], indent=1))
    for key, value in summary.items():
        print(
            f"{key:14} first={value['first_token_match_rate']:.3f} "
            f"exact8={value['quality_exact_match_rate']:.3f} "
            f"tokrate={value['median_token_consistency_rate']:.3f} "
            f"prefix={value['median_longest_matching_prefix']:.1f} "
            f"fallback={value['fallback_evidence']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
