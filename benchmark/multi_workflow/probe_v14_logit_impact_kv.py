#!/usr/bin/env python3
"""Causally rank K/V repairs by task-continuation logit impact."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, DynamicCache

from benchmark.multi_workflow.probe_v13_kv_boundary import (
    DEFAULT_OUTPUT as V13_PROBE_OUTPUT,
    _rotated_source_keys,
    selected_cases,
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
DEFAULT_OUTPUT = ARTIFACTS / "impactkv_v14_logit_impact_kv_20260727"
V13_RESULT = V13_PROBE_OUTPUT / "V13_KV_PROBE_RESULT.json"
CONTINUATION_TOKENS = 16
LAYERS = 36
VARIANTS = (
    "full_copy",
    "target_k_source_v",
    "source_k_target_v",
    "repair_early12",
    "repair_middle12",
    "repair_late12",
    "repair_head16",
    "repair_tail16",
    "repair_head16_tail16",
    "dense_replay",
)


def repair_fraction(
    variant: str, length: int, layers: int = LAYERS
) -> float:
    if variant == "full_copy":
        return 0.0
    if variant in ("target_k_source_v", "source_k_target_v"):
        return 0.5
    if variant.startswith("repair_") and variant.endswith("12"):
        return 12 / layers
    if variant == "repair_head16":
        return min(16, length) / length
    if variant == "repair_tail16":
        return min(16, length) / length
    if variant == "repair_head16_tail16":
        return min(32, length) / length
    if variant == "dense_replay":
        return 1.0
    raise ValueError(variant)


def register(output: Path) -> dict[str, Any]:
    path = output / "V14_LOGIT_IMPACT_REGISTRATION.json"
    if path.exists():
        value = read_json(path)
        if value["inputs"]["v13_probe_sha256"] != sha256_file(V13_RESULT):
            raise ValueError("registered V13 probe changed")
        return value
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    cases_path = V13_PROBE_OUTPUT / "V13_KV_PROBE_CASES.json"
    value = {
        "date": "2026-07-27",
        "experiment": "V14 task-aligned logit-impact KV splice probe",
        "registered_before_gpu": True,
        "model": MODEL,
        "variants": list(VARIANTS),
        "protocol": {
            "cases": len(selected_cases()),
            "continuation_tokens": CONTINUATION_TOKENS,
            "continuation": (
                "greedy Dense continuation generated from online-visible prompt"
            ),
            "splice": (
                "Dense target prefix + mixed shared K/V + normally forwarded "
                "target suffix and teacher-forced continuation"
            ),
            "metrics": [
                "continuation NLL",
                "KL from dense continuation logits",
                "top1 agreement",
            ],
            "truth_or_tests_read": False,
            "prefetch": False,
        },
        "frozen_gates": {
            "dense_replay_mean_kl_max": 1e-5,
            "dense_replay_top1_agreement_min": 1.0,
            "candidate_excess_nll_reduction_min": 0.20,
            "candidate_mean_kl_reduction_min": 0.20,
            "candidate_repair_fraction_max": 0.50,
            "selection": (
                "among passing candidates, maximize mean KL reduction per "
                "mean repair fraction; ties use lower repair fraction"
            ),
        },
        "inputs": {
            "cases_sha256": sha256_file(cases_path),
            "probe_source_sha256": sha256_file(Path(__file__)),
            "v13_probe_sha256": sha256_file(V13_RESULT),
        },
        "protected": {
            "existing_preregistration_thresholds_modified": False,
            "old_dirty_checkout_modified": False,
            "paper_modified": False,
            "prefetch": False,
        },
        "scope": (
            "Development-only causal logit probe on already exposed prompts; "
            "no functional accuracy claim."
        ),
        "status": "REGISTERED_BEFORE_V14_GPU",
    }
    write_json(path, value)
    return value


def _legacy(cache: Any) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if hasattr(cache, "to_legacy_cache"):
        cache = cache.to_legacy_cache()
    return [(layer[0], layer[1]) for layer in cache]


def _shared_mix(
    *,
    variant: str,
    layer: int,
    source_k: torch.Tensor,
    source_v: torch.Tensor,
    target_k: torch.Tensor,
    target_v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    length = source_k.shape[2]
    use_target_k = torch.zeros(
        length, dtype=torch.bool, device=source_k.device
    )
    use_target_v = torch.zeros_like(use_target_k)
    if variant == "target_k_source_v":
        use_target_k[:] = True
    elif variant == "source_k_target_v":
        use_target_v[:] = True
    elif variant in (
        "repair_early12",
        "repair_middle12",
        "repair_late12",
    ):
        selected = (
            (variant == "repair_early12" and layer < 12)
            or (variant == "repair_middle12" and 12 <= layer < 24)
            or (variant == "repair_late12" and layer >= 24)
        )
        if selected:
            use_target_k[:] = True
            use_target_v[:] = True
    elif variant in (
        "repair_head16",
        "repair_tail16",
        "repair_head16_tail16",
    ):
        if variant in ("repair_head16", "repair_head16_tail16"):
            use_target_k[: min(16, length)] = True
            use_target_v[: min(16, length)] = True
        if variant in ("repair_tail16", "repair_head16_tail16"):
            use_target_k[max(0, length - 16) :] = True
            use_target_v[max(0, length - 16) :] = True
    elif variant == "dense_replay":
        use_target_k[:] = True
        use_target_v[:] = True
    elif variant != "full_copy":
        raise ValueError(variant)
    mask = use_target_k.view(1, 1, length, 1)
    mixed_k = torch.where(mask, target_k, source_k)
    mask = use_target_v.view(1, 1, length, 1)
    mixed_v = torch.where(mask, target_v, source_v)
    return mixed_k, mixed_v


def build_cache(
    *,
    model: Any,
    variant: str,
    source_cache: list[tuple[torch.Tensor, torch.Tensor]],
    target_cache: list[tuple[torch.Tensor, torch.Tensor]],
    source_start: int,
    target_start: int,
    length: int,
) -> DynamicCache:
    rows = []
    delta = target_start - source_start
    for layer, ((source_k, source_v), (target_k, target_v)) in enumerate(
        zip(source_cache, target_cache, strict=True)
    ):
        source_shared_k = source_k[
            :, :, source_start : source_start + length, :
        ]
        rotated = _rotated_source_keys(source_shared_k, delta)
        rotated = (
            rotated.permute(1, 0, 2)
            .unsqueeze(0)
            .to(dtype=target_k.dtype)
        )
        source_shared_v = source_v[
            :, :, source_start : source_start + length, :
        ]
        target_shared_k = target_k[
            :, :, target_start : target_start + length, :
        ]
        target_shared_v = target_v[
            :, :, target_start : target_start + length, :
        ]
        mixed_k, mixed_v = _shared_mix(
            variant=variant,
            layer=layer,
            source_k=rotated,
            source_v=source_shared_v,
            target_k=target_shared_k,
            target_v=target_shared_v,
        )
        rows.append(
            (
                torch.cat((target_k[:, :, :target_start, :], mixed_k), dim=2),
                torch.cat((target_v[:, :, :target_start, :], mixed_v), dim=2),
            )
        )
    return DynamicCache(ddp_cache_data=rows, config=model.config)


def _logit_metrics(
    dense_logits: torch.Tensor,
    variant_logits: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, float]:
    dense_logp = F.log_softmax(dense_logits.float(), dim=-1)
    variant_logp = F.log_softmax(variant_logits.float(), dim=-1)
    dense_p = dense_logp.exp()
    kl = (dense_p * (dense_logp - variant_logp)).sum(dim=-1)
    nll = F.nll_loss(
        variant_logp.reshape(-1, variant_logp.shape[-1]),
        labels.reshape(-1),
        reduction="mean",
    )
    dense_nll = F.nll_loss(
        dense_logp.reshape(-1, dense_logp.shape[-1]),
        labels.reshape(-1),
        reduction="mean",
    )
    return {
        "dense_nll": float(dense_nll.item()),
        "kl_mean": float(kl.mean().item()),
        "nll": float(nll.item()),
        "top1_agreement": float(
            (
                dense_logits.argmax(dim=-1)
                == variant_logits.argmax(dim=-1)
            )
            .float()
            .mean()
            .item()
        ),
    }


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
                    do_sample=False,
                    max_new_tokens=CONTINUATION_TOKENS,
                    min_new_tokens=CONTINUATION_TOKENS,
                    pad_token_id=model.config.eos_token_id,
                )
                continuation = generated[:, target_ids.shape[1] :]
                dense_inputs = torch.cat(
                    (target_ids, continuation[:, :-1]), dim=1
                )
                dense = model(
                    input_ids=dense_inputs,
                    use_cache=False,
                    return_dict=True,
                )
            target_length = target_ids.shape[1]
            dense_logits = dense.logits[
                :, target_length - 1 : target_length - 1 + CONTINUATION_TOKENS, :
            ]
            source_cache = _legacy(source.past_key_values)
            target_cache = _legacy(target.past_key_values)
            length = int(case["segment_tokens"])
            target_start = int(case["target_start"])
            copy_end = target_start + length
            suffix_and_teacher = torch.cat(
                (target_ids[:, copy_end:], continuation[:, :-1]), dim=1
            )
            positions = torch.arange(
                copy_end,
                copy_end + suffix_and_teacher.shape[1],
                dtype=torch.long,
                device="cuda",
            )
            for variant in VARIANTS:
                cache = build_cache(
                    model=model,
                    variant=variant,
                    source_cache=source_cache,
                    target_cache=target_cache,
                    source_start=int(case["source_start"]),
                    target_start=target_start,
                    length=length,
                )
                with torch.inference_mode():
                    replay = model(
                        input_ids=suffix_and_teacher,
                        past_key_values=cache,
                        cache_position=positions,
                        position_ids=positions.unsqueeze(0),
                        use_cache=False,
                        return_dict=True,
                    )
                replay_logits = replay.logits[
                    :, -CONTINUATION_TOKENS:, :
                ]
                rows.append(
                    {
                        "case_id": case["original_case_id"],
                        "repair_fraction": repair_fraction(
                            variant, length
                        ),
                        "segment_tokens": length,
                        "suite": case["suite"],
                        "variant": variant,
                        **_logit_metrics(
                            dense_logits, replay_logits, continuation
                        ),
                    }
                )
                del cache, replay
            del source, target, dense, source_cache, target_cache
    finally:
        del model
        torch.cuda.empty_cache()
    write_json(destination, {"rows": rows, "status": "complete"})
    return {"rows": len(rows), "status": "complete"}


def measure(output: Path, canary: bool) -> dict[str, Any]:
    registration = register(output)
    destination = (
        output / "canary" / "V14_LOGIT_MEASUREMENTS.json"
        if canary
        else output / "V14_LOGIT_MEASUREMENTS.json"
    )
    if destination.exists():
        return {"status": "already_complete"}
    registered_source = registration["inputs"]["probe_source_sha256"]
    current_source = sha256_file(Path(__file__))
    if registered_source != current_source:
        amendment = output / "V14_RUNTIME_AMENDMENT_001.json"
        if not amendment.exists():
            write_json(
                amendment,
                {
                    "date": "2026-07-27",
                    "trigger": (
                        "The first one-case canary stopped before writing "
                        "metrics when a non-selected layer of repair_early12 "
                        "was treated as an unknown variant."
                    ),
                    "change": (
                        "Recognize every layer-block variant on every layer; "
                        "use source KV outside its selected 12-layer block."
                    ),
                    "canary_metrics_written_before_amendment": False,
                    "formal_metrics_written_before_amendment": False,
                    "registered_source_sha256": registered_source,
                    "corrected_source_sha256": current_source,
                    "unchanged": [
                        "cases",
                        "variants and splice semantics",
                        "continuation protocol",
                        "metrics",
                        "all thresholds",
                    ],
                    "status": "AMENDED_BEFORE_FIRST_V14_METRIC",
                },
            )
    cases = selected_cases()[:1] if canary else selected_cases()
    return _measure_cases(output, cases, destination)


def summarize(output: Path) -> dict[str, Any]:
    registration = register(output)
    rows = read_json(output / "V14_LOGIT_MEASUREMENTS.json")["rows"]
    by_variant = {
        variant: [row for row in rows if row["variant"] == variant]
        for variant in VARIANTS
    }
    arms = {
        variant: {
            "mean_dense_nll": statistics.mean(
                row["dense_nll"] for row in values
            ),
            "mean_excess_nll": statistics.mean(
                row["nll"] - row["dense_nll"] for row in values
            ),
            "mean_kl": statistics.mean(row["kl_mean"] for row in values),
            "mean_nll": statistics.mean(row["nll"] for row in values),
            "mean_repair_fraction": statistics.mean(
                row["repair_fraction"] for row in values
            ),
            "mean_top1_agreement": statistics.mean(
                row["top1_agreement"] for row in values
            ),
        }
        for variant, values in by_variant.items()
    }
    full = arms["full_copy"]
    candidates = {}
    gates = registration["frozen_gates"]
    for variant in VARIANTS:
        if variant in ("full_copy", "dense_replay"):
            continue
        value = arms[variant]
        excess_reduction = (
            (full["mean_excess_nll"] - value["mean_excess_nll"])
            / max(full["mean_excess_nll"], 1e-12)
        )
        kl_reduction = (
            (full["mean_kl"] - value["mean_kl"])
            / max(full["mean_kl"], 1e-12)
        )
        passed = (
            excess_reduction
            >= gates["candidate_excess_nll_reduction_min"]
            and kl_reduction >= gates["candidate_mean_kl_reduction_min"]
            and value["mean_repair_fraction"]
            <= gates["candidate_repair_fraction_max"]
        )
        candidates[variant] = {
            "excess_nll_reduction": excess_reduction,
            "kl_reduction": kl_reduction,
            "passed": passed,
            "score": kl_reduction
            / max(value["mean_repair_fraction"], 1e-12),
        }
    eligible = [
        variant for variant, value in candidates.items() if value["passed"]
    ]
    selected = (
        max(
            eligible,
            key=lambda variant: (
                candidates[variant]["score"],
                -arms[variant]["mean_repair_fraction"],
            ),
        )
        if eligible
        else None
    )
    dense_replay = arms["dense_replay"]
    negative_control = {
        "kl_passed": (
            dense_replay["mean_kl"]
            <= gates["dense_replay_mean_kl_max"]
        ),
        "top1_passed": (
            dense_replay["mean_top1_agreement"]
            >= gates["dense_replay_top1_agreement_min"]
        ),
    }
    value = {
        "arms": arms,
        "candidates": candidates,
        "negative_control": negative_control,
        "selected_candidate": (
            selected if all(negative_control.values()) else None
        ),
        "status": "V14_LOGIT_IMPACT_COMPLETE",
    }
    write_json(output / "V14_LOGIT_IMPACT_RESULT.json", value)
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
