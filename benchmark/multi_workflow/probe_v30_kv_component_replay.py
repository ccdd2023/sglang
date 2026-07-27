#!/usr/bin/env python3
"""Compare K-only, V-only, and layer-block repair on coding KV spans.

V14 compared cache replay with a separate full-forward Dense path, so its
one-case negative control retained execution-path numerical error.  V30 keeps
the already frozen coding cases and splice semantics, but compares every arm
with the full-target-KV replay produced by the exact same execution path.
Historical CacheBlend labels are used only to report damage/safe cohorts; they
never select a repair arm or enter a repair mask.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM

from benchmark.multi_workflow.probe_v14_logit_impact_kv import (
    CONTINUATION_TOKENS,
    _legacy,
    _logit_metrics,
    build_cache,
    repair_fraction,
)
from benchmark.multi_workflow.probe_v16_behavioral_contract_kv import (
    DEFAULT_OUTPUT as V16_OUTPUT,
)
from benchmark.multi_workflow.run_bridge_reuse_pilot import (
    sha256_file,
    write_json,
)
from benchmark.multi_workflow.run_coding_native_workload_v10 import (
    MODEL,
    read_json,
)


ARTIFACTS = Path("/home/gfy/CodeMAS_Project/kvflow-artifacts")
DEFAULT_OUTPUT = ARTIFACTS / "impactkv_v30_kv_component_replay_20260727"
V13_RESULT = (
    ARTIFACTS
    / "impactkv_v13_kv_boundary_probe_20260727"
    / "V13_KV_PROBE_RESULT.json"
)
V14_CANARY = (
    ARTIFACTS
    / "impactkv_v14_logit_impact_kv_20260727"
    / "canary"
    / "V14_LOGIT_MEASUREMENTS.json"
)
V16_CASES = V16_OUTPUT / "V16_CASES.json"
VARIANTS = (
    "full_copy",
    "target_k_source_v",
    "source_k_target_v",
    "repair_early12",
    "repair_middle12",
    "repair_late12",
    "dense_replay",
)


def register(output: Path) -> dict[str, Any]:
    path = output / "V30_REGISTRATION.json"
    if path.exists():
        value = read_json(path)
        for name, source in (
            ("v13_result_sha256", V13_RESULT),
            ("v14_canary_sha256", V14_CANARY),
            ("v16_cases_sha256", V16_CASES),
        ):
            if value["inputs"][name] != sha256_file(source):
                raise ValueError(f"registered V30 input changed: {name}")
        return value
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    cases = read_json(V16_CASES)["cases"]
    cohort_counts = {
        cohort: sum(row["cohort"] == cohort for row in cases)
        for cohort in ("damage", "matched_safe")
    }
    if cohort_counts != {"damage": 9, "matched_safe": 9}:
        raise ValueError(f"unexpected V30 cohort counts: {cohort_counts}")
    value = {
        "date": "2026-07-27",
        "experiment": "V30 same-path coding KV component replay",
        "registered_before_gpu": True,
        "motivation": (
            "V29 showed that history selection alone degenerates to General "
            "when it keeps the same copied-token budget. V13 found shared-"
            "span V drift substantially above K drift, while the V14 canary "
            "could not pass its cross-execution-path Dense control. Test the "
            "K/V and layer asymmetry with a same-cache-replay Dense control."
        ),
        "hypothesis": (
            "On shared coding spans, repairing V while reusing RoPE-corrected "
            "K (source_k_target_v) reduces Dense-reference KL more than the "
            "equal-cost opposite split; at least one 12-layer block also "
            "contains a reproducible concentration of repair value."
        ),
        "variants": list(VARIANTS),
        "protocol": {
            "cases": len(cases),
            "cohorts": cohort_counts,
            "continuation_tokens": CONTINUATION_TOKENS,
            "dense_reference": (
                "full target K/V replay through the identical cache-replay "
                "suffix path used by every candidate"
            ),
            "component_repairs": {
                "target_k_source_v": "repair K, reuse V",
                "source_k_target_v": "reuse RoPE-corrected K, repair V",
            },
            "layer_repairs": {
                "repair_early12": "repair K and V in layers 0-11",
                "repair_middle12": "repair K and V in layers 12-23",
                "repair_late12": "repair K and V in layers 24-35",
            },
            "historical_labels_used_only_for_diagnostic_cohort": True,
            "candidate_uses_outcome_labels": False,
            "truth_or_evaluator_tests_read": False,
            "prefetch": False,
        },
        "frozen_gates": {
            "dense_replay_mean_kl_max": 1e-12,
            "dense_replay_top1_agreement_min": 1.0,
            "v_repair_mean_kl_not_above_k_repair": True,
            "v_repair_kl_reduction_vs_full_copy_min": 0.20,
            "v_repair_top1_not_below_k_repair": True,
            "best_layer_block_kl_reduction_vs_full_copy_min": 0.20,
            "advance_only_to_online_delta_estimation_probe": True,
        },
        "inputs": {
            "probe_source_sha256": sha256_file(Path(__file__)),
            "v13_result_sha256": sha256_file(V13_RESULT),
            "v14_canary_sha256": sha256_file(V14_CANARY),
            "v16_cases_sha256": sha256_file(V16_CASES),
        },
        "protected": {
            "existing_preregistration_thresholds_modified": False,
            "old_dirty_checkout_modified": False,
            "paper_modified": False,
            "prefetch": False,
        },
        "scope": (
            "Development-only causal component probe on exposed coding cases. "
            "It cannot make a functional-accuracy or serving-speed claim."
        ),
        "status": "REGISTERED_BEFORE_V30_GPU",
    }
    write_json(path, value)
    return value


def _measure_cases(
    output: Path,
    cases: list[dict[str, Any]],
    destination: Path,
) -> dict[str, Any]:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to("cuda")
    model.eval()
    rows = []
    try:
        for case in cases:
            source_ids = torch.tensor(
                [case["source_input_ids"]], dtype=torch.long, device="cuda"
            )
            target_ids = torch.tensor(
                [case["target_input_ids"]], dtype=torch.long, device="cuda"
            )
            with torch.inference_mode():
                source = model(
                    input_ids=source_ids, use_cache=True, return_dict=True
                )
                target = model(
                    input_ids=target_ids, use_cache=True, return_dict=True
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
            source_cache = _legacy(source.past_key_values)
            target_cache = _legacy(target.past_key_values)
            target_start = int(case["target_start"])
            source_start = int(case["source_start"])
            length = int(case["segment_tokens"])
            copy_end = target_start + length
            replay_inputs = torch.cat(
                (target_ids[:, copy_end:], continuation[:, :-1]), dim=1
            )
            positions = torch.arange(
                copy_end,
                copy_end + replay_inputs.shape[1],
                dtype=torch.long,
                device="cuda",
            )
            logits: dict[str, torch.Tensor] = {}
            for variant in VARIANTS:
                cache = build_cache(
                    model=model,
                    variant=variant,
                    source_cache=source_cache,
                    target_cache=target_cache,
                    source_start=source_start,
                    target_start=target_start,
                    length=length,
                )
                with torch.inference_mode():
                    replay = model(
                        input_ids=replay_inputs,
                        past_key_values=cache,
                        cache_position=positions,
                        position_ids=positions.unsqueeze(0),
                        use_cache=False,
                        return_dict=True,
                    )
                logits[variant] = replay.logits[
                    :, -CONTINUATION_TOKENS:, :
                ].detach()
                del cache, replay
            dense_logits = logits["dense_replay"]
            for variant in VARIANTS:
                rows.append(
                    {
                        "case_id": case["original_case_id"],
                        "cohort": case["cohort"],
                        "repair_fraction": repair_fraction(variant, length),
                        "segment_tokens": length,
                        "suite": case["suite"],
                        "variant": variant,
                        **_logit_metrics(
                            dense_logits, logits[variant], continuation
                        ),
                    }
                )
            del source, target, source_cache, target_cache, logits
    finally:
        del model
        torch.cuda.empty_cache()
    write_json(destination, {"rows": rows, "status": "complete"})
    return {"cases": len(cases), "rows": len(rows), "status": "complete"}


def measure(output: Path, canary: bool) -> dict[str, Any]:
    register(output)
    destination = (
        output / "canary" / "V30_MEASUREMENTS.json"
        if canary
        else output / "V30_MEASUREMENTS.json"
    )
    if destination.exists():
        return {"status": "already_complete"}
    cases = read_json(V16_CASES)["cases"]
    if canary:
        cases = [
            next(row for row in cases if row["cohort"] == cohort)
            for cohort in ("damage", "matched_safe")
        ]
    return _measure_cases(output, cases, destination)


def _arm_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "mean_kl": statistics.mean(row["kl_mean"] for row in rows),
        "mean_nll": statistics.mean(row["nll"] for row in rows),
        "mean_repair_fraction": statistics.mean(
            row["repair_fraction"] for row in rows
        ),
        "mean_top1_agreement": statistics.mean(
            row["top1_agreement"] for row in rows
        ),
    }


def summarize(output: Path) -> dict[str, Any]:
    registration = register(output)
    rows = read_json(output / "V30_MEASUREMENTS.json")["rows"]
    arms = {}
    for cohort in ("damage", "matched_safe", "all"):
        subset = (
            rows
            if cohort == "all"
            else [row for row in rows if row["cohort"] == cohort]
        )
        arms[cohort] = {
            variant: _arm_summary(
                [row for row in subset if row["variant"] == variant]
            )
            for variant in VARIANTS
        }
    overall = arms["all"]
    full_kl = overall["full_copy"]["mean_kl"]
    v_repair = overall["source_k_target_v"]
    k_repair = overall["target_k_source_v"]
    layer_variants = (
        "repair_early12",
        "repair_middle12",
        "repair_late12",
    )
    best_layer = min(
        layer_variants,
        key=lambda variant: overall[variant]["mean_kl"],
    )
    v_reduction = (full_kl - v_repair["mean_kl"]) / max(full_kl, 1e-12)
    layer_reduction = (
        full_kl - overall[best_layer]["mean_kl"]
    ) / max(full_kl, 1e-12)
    gates = registration["frozen_gates"]
    outcomes = {
        "dense_replay_mean_kl": (
            overall["dense_replay"]["mean_kl"]
            <= gates["dense_replay_mean_kl_max"]
        ),
        "dense_replay_top1": (
            overall["dense_replay"]["mean_top1_agreement"]
            >= gates["dense_replay_top1_agreement_min"]
        ),
        "v_repair_mean_kl_not_above_k_repair": (
            v_repair["mean_kl"] <= k_repair["mean_kl"]
        ),
        "v_repair_kl_reduction_vs_full_copy": (
            v_reduction
            >= gates["v_repair_kl_reduction_vs_full_copy_min"]
        ),
        "v_repair_top1_not_below_k_repair": (
            v_repair["mean_top1_agreement"]
            >= k_repair["mean_top1_agreement"]
        ),
        "best_layer_block_kl_reduction_vs_full_copy": (
            layer_reduction
            >= gates["best_layer_block_kl_reduction_vs_full_copy_min"]
        ),
    }
    passed = all(outcomes.values())
    value = {
        "arms": arms,
        "best_layer_block": best_layer,
        "best_layer_kl_reduction_vs_full_copy": layer_reduction,
        "completed_at_utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
        "decision": (
            "Advance to online K/V-delta estimation motivation."
            if passed
            else "Do not implement component-aware serving from this probe."
        ),
        "gate_outcomes": outcomes,
        "registration_sha256": sha256_file(
            output / "V30_REGISTRATION.json"
        ),
        "status": (
            "PASS_V30_KV_COMPONENT_REPLAY"
            if passed
            else "FAIL_V30_KV_COMPONENT_REPLAY"
        ),
        "v_repair_kl_reduction_vs_full_copy": v_reduction,
    }
    write_json(output / "V30_RESULT.json", value)
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
    print(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
