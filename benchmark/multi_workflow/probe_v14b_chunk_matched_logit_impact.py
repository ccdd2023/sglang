#!/usr/bin/env python3
"""Rank KV repairs against a chunk-matched Dense replay reference."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM

from benchmark.multi_workflow.probe_v13_kv_boundary import selected_cases
from benchmark.multi_workflow.probe_v14_logit_impact_kv import (
    CONTINUATION_TOKENS,
    MODEL,
    VARIANTS,
    _legacy,
    _logit_metrics,
    build_cache,
    repair_fraction,
)
from benchmark.multi_workflow.run_bridge_reuse_pilot import (
    sha256_file,
    write_json,
)
from benchmark.multi_workflow.run_coding_native_workload_v10 import read_json


ARTIFACTS = Path("/home/gfy/CodeMAS_Project/kvflow-artifacts")
V14_OUTPUT = ARTIFACTS / "impactkv_v14_logit_impact_kv_20260727"
V14_CANARY = V14_OUTPUT / "canary/V14_LOGIT_MEASUREMENTS.json"
DEFAULT_OUTPUT = (
    ARTIFACTS / "impactkv_v14b_chunk_matched_logit_impact_20260727"
)
CANDIDATES = tuple(
    variant
    for variant in VARIANTS
    if variant not in ("full_copy", "dense_replay")
)


def register(output: Path) -> dict[str, Any]:
    path = output / "V14B_REGISTRATION.json"
    if path.exists():
        value = read_json(path)
        if value["inputs"]["v14_canary_sha256"] != sha256_file(V14_CANARY):
            raise ValueError("registered V14 canary changed")
        return value
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    value = {
        "date": "2026-07-27",
        "experiment": "V14b chunk-matched task-logit KV splice probe",
        "registered_before_gpu": True,
        "model": MODEL,
        "motivation": (
            "V14 one-case canary found 1.997e-4 KL between monolithic Dense "
            "and an all-target-KV suffix replay, exceeding V14's 1e-5 gate."
        ),
        "protocol": {
            "cases": len(selected_cases()),
            "continuation_tokens": CONTINUATION_TOKENS,
            "primary_reference": (
                "all-target-KV Dense replay using the identical cached-prefix "
                "and suffix execution shape as every treatment"
            ),
            "numerical_audit": (
                "monolithic Dense continuation logits versus Dense replay"
            ),
            "truth_or_tests_read": False,
            "prefetch": False,
        },
        "frozen_gates": {
            "monolithic_vs_replay_mean_kl_max": 5e-4,
            "monolithic_vs_replay_top1_agreement_min": 0.99,
            "candidate_kl_reduction_vs_full_copy_min": 0.20,
            "candidate_repair_fraction_max": 0.50,
            "candidate_top1_agreement_not_below_full_copy": True,
            "selection": (
                "maximize KL reduction per repair fraction among passing "
                "candidates; ties prefer lower repair fraction"
            ),
        },
        "inputs": {
            "probe_source_sha256": sha256_file(Path(__file__)),
            "v14_canary_sha256": sha256_file(V14_CANARY),
            "v14_registration_sha256": sha256_file(
                V14_OUTPUT / "V14_LOGIT_IMPACT_REGISTRATION.json"
            ),
        },
        "protected": {
            "existing_preregistration_thresholds_modified": False,
            "old_dirty_checkout_modified": False,
            "paper_modified": False,
            "prefetch": False,
        },
        "scope": (
            "Development-only causal logit probe; V14 remains stopped and its "
            "registration is unchanged."
        ),
        "status": "REGISTERED_BEFORE_V14B_GPU",
    }
    write_json(path, value)
    return value


def _replay(
    *,
    model: Any,
    variant: str,
    case: dict[str, Any],
    source_cache: Any,
    target_cache: Any,
    inputs: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    cache = build_cache(
        model=model,
        variant=variant,
        source_cache=source_cache,
        target_cache=target_cache,
        source_start=int(case["source_start"]),
        target_start=int(case["target_start"]),
        length=int(case["segment_tokens"]),
    )
    with torch.inference_mode():
        value = model(
            input_ids=inputs,
            past_key_values=cache,
            cache_position=positions,
            position_ids=positions.unsqueeze(0),
            use_cache=False,
            return_dict=True,
        )
    logits = value.logits[:, -CONTINUATION_TOKENS:, :]
    del cache, value
    return logits


def measure(output: Path, canary: bool) -> dict[str, Any]:
    register(output)
    destination = (
        output / "canary" / "V14B_MEASUREMENTS.json"
        if canary
        else output / "V14B_MEASUREMENTS.json"
    )
    if destination.exists():
        return {"status": "already_complete"}
    cases = selected_cases()[:1] if canary else selected_cases()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to("cuda")
    model.eval()
    rows = []
    audits = []
    try:
        for case in cases:
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
                generated = model.generate(
                    input_ids=target_ids,
                    attention_mask=torch.ones_like(target_ids),
                    do_sample=False,
                    max_new_tokens=CONTINUATION_TOKENS,
                    min_new_tokens=CONTINUATION_TOKENS,
                    pad_token_id=model.config.eos_token_id,
                )
                continuation = generated[:, target_ids.shape[1] :]
                dense_inputs = torch.cat(
                    (target_ids, continuation[:, :-1]), dim=1
                )
                monolithic = model(
                    input_ids=dense_inputs,
                    use_cache=False,
                    return_dict=True,
                )
            target_length = target_ids.shape[1]
            monolithic_logits = monolithic.logits[
                :,
                target_length - 1 : target_length - 1 + CONTINUATION_TOKENS,
                :,
            ]
            source_cache = _legacy(source.past_key_values)
            target_cache = _legacy(target.past_key_values)
            copy_end = int(case["target_start"]) + int(
                case["segment_tokens"]
            )
            replay_inputs = torch.cat(
                (target_ids[:, copy_end:], continuation[:, :-1]), dim=1
            )
            positions = torch.arange(
                copy_end,
                copy_end + replay_inputs.shape[1],
                dtype=torch.long,
                device="cuda",
            )
            dense_replay_logits = _replay(
                model=model,
                variant="dense_replay",
                case=case,
                source_cache=source_cache,
                target_cache=target_cache,
                inputs=replay_inputs,
                positions=positions,
            )
            audits.append(
                {
                    "case_id": case["original_case_id"],
                    **_logit_metrics(
                        monolithic_logits,
                        dense_replay_logits,
                        continuation,
                    ),
                }
            )
            for variant in ("full_copy", *CANDIDATES):
                logits = _replay(
                    model=model,
                    variant=variant,
                    case=case,
                    source_cache=source_cache,
                    target_cache=target_cache,
                    inputs=replay_inputs,
                    positions=positions,
                )
                rows.append(
                    {
                        "case_id": case["original_case_id"],
                        "repair_fraction": repair_fraction(
                            variant, int(case["segment_tokens"])
                        ),
                        "segment_tokens": int(case["segment_tokens"]),
                        "suite": case["suite"],
                        "variant": variant,
                        **_logit_metrics(
                            dense_replay_logits,
                            logits,
                            continuation,
                        ),
                    }
                )
                del logits
            del source, target, monolithic, source_cache, target_cache
    finally:
        del model
        torch.cuda.empty_cache()
    write_json(
        destination,
        {"numerical_audits": audits, "rows": rows, "status": "complete"},
    )
    return {
        "audits": len(audits),
        "rows": len(rows),
        "status": "complete",
    }


def summarize(output: Path) -> dict[str, Any]:
    registration = register(output)
    payload = read_json(output / "V14B_MEASUREMENTS.json")
    rows = payload["rows"]
    audits = payload["numerical_audits"]
    arms = {}
    for variant in ("full_copy", *CANDIDATES):
        values = [row for row in rows if row["variant"] == variant]
        arms[variant] = {
            "mean_kl": statistics.mean(row["kl_mean"] for row in values),
            "mean_nll": statistics.mean(row["nll"] for row in values),
            "mean_repair_fraction": statistics.mean(
                row["repair_fraction"] for row in values
            ),
            "mean_top1_agreement": statistics.mean(
                row["top1_agreement"] for row in values
            ),
        }
    audit = {
        "mean_kl": statistics.mean(row["kl_mean"] for row in audits),
        "mean_top1_agreement": statistics.mean(
            row["top1_agreement"] for row in audits
        ),
    }
    gates = registration["frozen_gates"]
    audit["passed"] = (
        audit["mean_kl"]
        <= gates["monolithic_vs_replay_mean_kl_max"]
        and audit["mean_top1_agreement"]
        >= gates["monolithic_vs_replay_top1_agreement_min"]
    )
    full = arms["full_copy"]
    candidates = {}
    for variant in CANDIDATES:
        value = arms[variant]
        reduction = (
            (full["mean_kl"] - value["mean_kl"])
            / max(full["mean_kl"], 1e-12)
        )
        passed = (
            reduction
            >= gates["candidate_kl_reduction_vs_full_copy_min"]
            and value["mean_repair_fraction"]
            <= gates["candidate_repair_fraction_max"]
            and value["mean_top1_agreement"]
            >= full["mean_top1_agreement"]
        )
        candidates[variant] = {
            "kl_reduction": reduction,
            "passed": passed,
            "score": reduction
            / max(value["mean_repair_fraction"], 1e-12),
        }
    eligible = [
        variant for variant in CANDIDATES if candidates[variant]["passed"]
    ]
    selected = (
        max(
            eligible,
            key=lambda variant: (
                candidates[variant]["score"],
                -arms[variant]["mean_repair_fraction"],
            ),
        )
        if audit["passed"] and eligible
        else None
    )
    value = {
        "arms": arms,
        "candidates": candidates,
        "numerical_audit": audit,
        "selected_candidate": selected,
        "status": "V14B_COMPLETE",
    }
    write_json(output / "V14B_RESULT.json", value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("register")
    measure_parser = sub.add_parser("measure")
    measure_parser.add_argument("--canary", action="store_true")
    sub.add_parser("summarize")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.command == "register":
        value = register(output)
    elif args.command == "measure":
        value = measure(output, args.canary)
    else:
        value = summarize(output)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
