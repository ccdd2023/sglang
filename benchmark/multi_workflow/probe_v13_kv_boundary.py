#!/usr/bin/env python3
"""Measure source/target K/V drift across shared-code boundary zones.

This is a development-only motivation probe.  It uses a hash-selected subset
of the already exposed full-225 cases and never reads evaluator truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from benchmark.multi_workflow.run_bridge_reuse_pilot import (
    sha256_file,
    write_json,
)
from benchmark.multi_workflow.run_coding_native_workload_v10 import (
    MODEL,
    PROJECT,
    read_json,
)
from sglang.srt.layers.rotary_embedding.utils import apply_rotary_emb


ARTIFACTS = Path("/home/gfy/CodeMAS_Project/kvflow-artifacts")
OLD_CASES = (
    ARTIFACTS
    / "impactkv_full225_accuracy_audit_20260724/FULL225_CASES.json"
)
DEFAULT_OUTPUT = ARTIFACTS / "impactkv_v13_kv_boundary_probe_20260727"
SELECTION_SEED = "impactkv-v13-kv-boundary-20260727"
CASES = 32
ZONE_TOKENS = 16
ROTARY_DIM = 128
ROPE_BASE = 1_000_000.0


def selection_key(case: dict[str, Any]) -> str:
    return hashlib.sha256(
        f"{SELECTION_SEED}:{case['original_case_id']}".encode()
    ).hexdigest()


def selected_cases() -> list[dict[str, Any]]:
    rows = sorted(
        read_json(OLD_CASES)["cases"],
        key=selection_key,
    )[:CASES]
    if len(rows) != CASES:
        raise ValueError(f"expected {CASES} cases")
    if len({str(row["original_case_id"]) for row in rows}) != CASES:
        raise ValueError("selected cases are not unique")
    return rows


def zone_slices(length: int, width: int = ZONE_TOKENS) -> dict[str, slice]:
    if length < 3:
        raise ValueError("shared span is too short for zones")
    edge = min(width, max(1, (length - 1) // 2))
    return {
        "head": slice(0, edge),
        "interior": slice(edge, length - edge),
        "tail": slice(length - edge, length),
    }


def register(output: Path) -> dict[str, Any]:
    path = output / "V13_KV_PROBE_REGISTRATION.json"
    if path.exists():
        value = read_json(path)
        if value["inputs"]["cases_sha256"] != sha256_file(OLD_CASES):
            raise ValueError("registered cases changed")
        return value
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    selected = selected_cases()
    selection_path = output / "V13_KV_PROBE_CASES.json"
    write_json(
        selection_path,
        {
            "cases": [
                {
                    "case_id": row["case_id"],
                    "original_case_id": row["original_case_id"],
                    "segment_tokens": row["segment_tokens"],
                    "source_start": row["source_start"],
                    "suite": row["suite"],
                    "target_start": row["target_start"],
                }
                for row in selected
            ]
        },
    )
    value = {
        "date": "2026-07-27",
        "experiment": "V13 source/target K/V shared-boundary drift probe",
        "registered_before_gpu": True,
        "model": MODEL,
        "protocol": {
            "cases": CASES,
            "selection": "SHA256 ordering without accuracy labels",
            "zone_tokens": ZONE_TOKENS,
            "key_metric": (
                "cosine distance after applying the exact source-to-target "
                "RoPE delta to source K"
            ),
            "value_metric": "cosine distance between source and target V",
            "truth_or_tests_read": False,
        },
        "frozen_decision_rule": {
            "candidate_guards": [
                "head16",
                "tail16",
                "head16_tail16",
            ],
            "primary_zone": (
                "zone with the largest mean of normalized K and V cosine "
                "distance; ties prefer tail, then head"
            ),
            "motivation_pass": (
                "largest boundary-zone normalized drift exceeds interior "
                "normalized drift by at least 5%"
            ),
        },
        "inputs": {
            "cases_sha256": sha256_file(OLD_CASES),
            "probe_source_sha256": sha256_file(Path(__file__)),
            "selection_sha256": sha256_file(selection_path),
        },
        "protected": {
            "existing_preregistration_thresholds_modified": False,
            "old_dirty_checkout_modified": False,
            "paper_modified": False,
            "prefetch": False,
        },
        "scope": (
            "Development-only mechanistic probe on already exposed prompts; "
            "no accuracy or holdout claim."
        ),
        "status": "REGISTERED_BEFORE_V13_KV_PROBE_GPU",
    }
    write_json(path, value)
    return value


def _rotated_source_keys(keys: torch.Tensor, delta: int) -> torch.Tensor:
    # HF cache: [batch, kv_heads, sequence, head_dim].
    value = keys[0].permute(1, 0, 2).contiguous().float()
    if delta == 0:
        return value
    inverse_frequency = 1.0 / (
        ROPE_BASE
        ** (
            torch.arange(
                0,
                ROTARY_DIM,
                2,
                dtype=torch.float32,
                device=value.device,
            )
            / ROTARY_DIM
        )
    )
    positions = torch.full(
        (value.shape[0],),
        delta,
        dtype=torch.float32,
        device=value.device,
    )
    frequencies = torch.einsum("i,j->ij", positions, inverse_frequency)
    return apply_rotary_emb(
        value,
        frequencies.cos(),
        frequencies.sin(),
        True,
    )


def _metric(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    cosine = F.cosine_similarity(left.float(), right.float(), dim=-1)
    return {
        "cosine_distance_mean": float((1.0 - cosine).mean().item()),
        "max_abs_mean": float(
            (left.float() - right.float()).abs().amax(dim=-1).mean().item()
        ),
    }


def measure(output: Path) -> dict[str, Any]:
    registration = register(output)
    destination = output / "V13_KV_PROBE_MEASUREMENTS.json"
    if destination.exists():
        return {"status": "already_complete"}
    registered_source = registration["inputs"]["probe_source_sha256"]
    current_source = sha256_file(Path(__file__))
    if registered_source != current_source:
        amendment = output / "V13_KV_PROBE_RUNTIME_AMENDMENT_001.json"
        if not amendment.exists():
            write_json(
                amendment,
                {
                    "date": "2026-07-27",
                    "trigger": (
                        "The first measurement attempt stopped after model "
                        "forward because the installed Transformers cache "
                        "layer contains metadata after its K/V tensors."
                    ),
                    "change": (
                        "Read cache_layer[0] and cache_layer[1] explicitly "
                        "instead of unpacking the complete cache layer."
                    ),
                    "registered_source_sha256": registered_source,
                    "corrected_source_sha256": current_source,
                    "measurements_written_before_amendment": False,
                    "unchanged": [
                        "selected cases",
                        "K/V metrics",
                        "zone definitions",
                        "decision rule",
                        "all thresholds",
                    ],
                    "status": "AMENDED_BEFORE_FIRST_KV_MEASUREMENT",
                },
            )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to("cuda")
    model.eval()
    rows = []
    try:
        for case in selected_cases():
            source_ids = torch.tensor(
                [case["source_input_ids"]],
                dtype=torch.long,
                device="cuda",
            )
            target_ids = torch.tensor(
                [case["target_input_ids"]],
                dtype=torch.long,
                device="cuda",
            )
            with torch.inference_mode():
                source = model(
                    input_ids=source_ids,
                    use_cache=True,
                    return_dict=True,
                )
                target = model(
                    input_ids=target_ids,
                    use_cache=True,
                    return_dict=True,
                )
            source_cache = source.past_key_values
            target_cache = target.past_key_values
            if hasattr(source_cache, "to_legacy_cache"):
                source_cache = source_cache.to_legacy_cache()
                target_cache = target_cache.to_legacy_cache()
            length = int(case["segment_tokens"])
            source_start = int(case["source_start"])
            target_start = int(case["target_start"])
            delta = target_start - source_start
            zones = zone_slices(length)
            for layer, (source_layer, target_layer) in enumerate(
                zip(source_cache, target_cache, strict=True)
            ):
                # Current Transformers cache layers may carry extra metadata
                # after the K/V tensors; only the first two entries are K/V.
                source_k, source_v = source_layer[0], source_layer[1]
                target_k, target_v = target_layer[0], target_layer[1]
                source_k = source_k[
                    :, :, source_start : source_start + length, :
                ]
                target_k = target_k[
                    :, :, target_start : target_start + length, :
                ]
                source_v = source_v[
                    :, :, source_start : source_start + length, :
                ][0].permute(1, 0, 2).contiguous()
                target_v = target_v[
                    :, :, target_start : target_start + length, :
                ][0].permute(1, 0, 2).contiguous()
                rotated_k = _rotated_source_keys(source_k, delta)
                target_k = target_k[0].permute(1, 0, 2).contiguous()
                for zone, span in zones.items():
                    rows.append(
                        {
                            "case_id": case["original_case_id"],
                            "component": "k",
                            "layer": layer,
                            "position_delta": delta,
                            "segment_tokens": length,
                            "suite": case["suite"],
                            "zone": zone,
                            **_metric(rotated_k[span], target_k[span]),
                        }
                    )
                    rows.append(
                        {
                            "case_id": case["original_case_id"],
                            "component": "v",
                            "layer": layer,
                            "position_delta": delta,
                            "segment_tokens": length,
                            "suite": case["suite"],
                            "zone": zone,
                            **_metric(source_v[span], target_v[span]),
                        }
                    )
            del source, target, source_cache, target_cache
    finally:
        del model
        torch.cuda.empty_cache()
    write_json(destination, {"rows": rows, "status": "complete"})
    return {"rows": len(rows), "status": "complete"}


def summarize(output: Path) -> dict[str, Any]:
    registration = register(output)
    rows = read_json(output / "V13_KV_PROBE_MEASUREMENTS.json")["rows"]
    means: dict[str, dict[str, float]] = {}
    for component in ("k", "v"):
        means[component] = {}
        for zone in ("head", "interior", "tail"):
            values = [
                float(row["cosine_distance_mean"])
                for row in rows
                if row["component"] == component and row["zone"] == zone
            ]
            means[component][zone] = statistics.mean(values)
    normalized = {
        zone: statistics.mean(
            means[component][zone] / max(means[component]["interior"], 1e-12)
            for component in ("k", "v")
        )
        for zone in ("head", "interior", "tail")
    }
    priority = {"tail": 2, "head": 1, "interior": 0}
    primary = max(normalized, key=lambda zone: (normalized[zone], priority[zone]))
    boundary = max(normalized["head"], normalized["tail"])
    value = {
        "component_zone_mean_cosine_distance": means,
        "normalized_joint_drift_vs_interior": normalized,
        "selected_primary_zone": primary,
        "motivation_passed": boundary >= 1.05,
        "recommended_guard": (
            "head16_tail16"
            if min(normalized["head"], normalized["tail"]) >= 1.05
            else f"{primary}16"
        ),
        "registration_rule": registration["frozen_decision_rule"],
        "status": "V13_KV_PROBE_COMPLETE",
    }
    write_json(output / "V13_KV_PROBE_RESULT.json", value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("register")
    sub.add_parser("measure")
    sub.add_parser("summarize")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.command == "register":
        value = register(output)
    elif args.command == "measure":
        value = measure(output)
    else:
        value = summarize(output)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
