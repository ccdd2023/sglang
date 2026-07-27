#!/usr/bin/env python3
"""Probe coding-semantic KV repair against equal-budget generic repairs.

V16 asks whether online-visible behavioral clauses are more causally useful
than head, tail, random, or function-signature repair.  Historical damage
labels select a development diagnosis set, but no label, evaluator truth, or
generated answer enters the repair mask.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import statistics
from pathlib import Path
from typing import Any, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from benchmark.multi_workflow.audit_cacheblend_dense_flips_v15 import (
    DEFAULT_OUTPUT as V15_OUTPUT,
    DENSE as CACHEBLEND_DENSE,
    REUSE as CACHEBLEND_REUSE,
    read_jsonl,
)
from benchmark.multi_workflow.probe_v14_logit_impact_kv import (
    CONTINUATION_TOKENS,
    MODEL,
    _legacy,
    _logit_metrics,
)
from benchmark.multi_workflow.probe_v13_kv_boundary import (
    _rotated_source_keys,
)
from benchmark.multi_workflow.run_bridge_reuse_pilot import (
    sha256_file,
    write_json,
)
from benchmark.multi_workflow.run_coding_native_workload_v10 import read_json


ARTIFACTS = Path("/home/gfy/CodeMAS_Project/kvflow-artifacts")
FULL225_CASES = (
    ARTIFACTS / "impactkv_full225_accuracy_audit_20260724/FULL225_CASES.json"
)
DEFAULT_OUTPUT = (
    ARTIFACTS / "impactkv_v16_behavioral_contract_kv_20260727"
)
REPAIR_BUDGET = 32
VARIANTS = (
    "full_copy",
    "random32",
    "head32",
    "tail32",
    "signature32",
    "behavior32",
    "dense_replay",
)

_BEHAVIOR_PATTERNS = (
    (re.compile(r"(?i)#\s*task\s*:"), 12),
    (
        re.compile(
            r"(?i)\b(?:if|when|whenever|unless|otherwise|where|while)\b"
        ),
        9,
    ),
    (re.compile(r"(?i)\b(?:not|never|without|no|must|only)\b"), 10),
    (
        re.compile(
            r"(?i)\b(?:return|raise|round|convert|count|find|remove|"
            r"reverse|sum|sort|replace|check)\b"
        ),
        7,
    ),
    (re.compile(r"(?i)\bexample\b|>>>"), 8),
    (re.compile(r"(?:==|!=|<=|>=|<|>|\+|-|\*|/|%)"), 5),
    (re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?![\w.])"), 4),
)
_SIGNATURE = re.compile(
    r"(?im)^(?:#\s*(?:required public interface\s*:)?\s*)?"
    r"(?:async\s+)?def\s+[A-Za-z_]\w*\s*\([^\n]*$"
)


def _output_index(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["case_id"]): row for row in read_jsonl(path)}


def select_case_ids() -> dict[str, list[str]]:
    dense = _output_index(CACHEBLEND_DENSE)
    reuse = _output_index(CACHEBLEND_REUSE)
    damage = sorted(
        case_id
        for case_id in dense
        if bool(dense[case_id]["passed"]) and not bool(reuse[case_id]["passed"])
    )
    if len(damage) != 9:
        raise ValueError(f"expected 9 stable CacheBlend damages, got {len(damage)}")
    safe_candidates = [
        case_id
        for case_id in dense
        if bool(dense[case_id]["passed"]) and bool(reuse[case_id]["passed"])
    ]
    safe = []
    used: set[str] = set()
    for damaged in damage:
        row = dense[damaged]
        candidates = [
            case_id
            for case_id in safe_candidates
            if case_id not in used and dense[case_id]["suite"] == row["suite"]
        ]
        selected = min(
            candidates,
            key=lambda case_id: (
                abs(
                    int(dense[case_id]["context_tokens"])
                    - int(row["context_tokens"])
                ),
                case_id,
            ),
        )
        safe.append(selected)
        used.add(selected)
    return {"damage": damage, "matched_safe": safe}


def _line_spans(text: str) -> list[tuple[int, int, str]]:
    spans = []
    cursor = 0
    for line in text.splitlines(keepends=True):
        spans.append((cursor, cursor + len(line), line))
        cursor += len(line)
    if cursor < len(text):
        spans.append((cursor, len(text), text[cursor:]))
    return spans


def _token_scores(
    text: str,
    offsets: Sequence[Sequence[int]],
    *,
    mode: str,
) -> list[int]:
    line_scores: list[tuple[int, int, int]] = []
    for start, end, line in _line_spans(text):
        if mode == "behavior":
            score = sum(
                weight * len(pattern.findall(line))
                for pattern, weight in _BEHAVIOR_PATTERNS
            )
            stripped = line.strip().lower()
            if stripped.startswith(("def ", "# def ")) or (
                "required public interface" in stripped
            ):
                score = max(0, score - 8)
        elif mode == "signature":
            score = 20 if _SIGNATURE.search(line) else 0
            if "required public interface" in line.lower():
                score += 10
        else:
            raise ValueError(mode)
        line_scores.append((start, end, score))
    scores = []
    for token_start, token_end in offsets:
        scores.append(
            max(
                (
                    score
                    for line_start, line_end, score in line_scores
                    if token_end > line_start and token_start < line_end
                ),
                default=0,
            )
        )
    return scores


def _top_budget(scores: Sequence[int], budget: int) -> list[int]:
    if budget <= 0:
        return []
    positive = [index for index, score in enumerate(scores) if score > 0]
    selected = sorted(
        positive,
        key=lambda index: (-scores[index], index),
    )[:budget]
    if len(selected) < budget:
        anchors = selected or [len(scores) // 2]
        remaining = sorted(
            (index for index in range(len(scores)) if index not in selected),
            key=lambda index: (
                min(abs(index - anchor) for anchor in anchors),
                index,
            ),
        )
        selected.extend(remaining[: budget - len(selected)])
    return sorted(selected)


def semantic_masks(
    *,
    case_id: str,
    segment_ids: list[int],
    tokenizer: Any,
    budget: int = REPAIR_BUDGET,
) -> dict[str, list[int]]:
    budget = min(budget, len(segment_ids))
    text = tokenizer.decode(
        segment_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    if list(encoded["input_ids"]) != segment_ids:
        raise ValueError(f"{case_id}: shared segment decode is not reversible")
    offsets = encoded["offset_mapping"]
    behavior = _top_budget(
        _token_scores(text, offsets, mode="behavior"), budget
    )
    signature = _top_budget(
        _token_scores(text, offsets, mode="signature"), budget
    )
    rng = random.Random(
        int(hashlib.sha256(case_id.encode()).hexdigest()[:16], 16)
    )
    random_mask = sorted(rng.sample(range(len(segment_ids)), budget))
    return {
        "behavior32": behavior,
        "head32": list(range(budget)),
        "random32": random_mask,
        "signature32": signature,
        "tail32": list(range(len(segment_ids) - budget, len(segment_ids))),
    }


def prepare_cases() -> dict[str, Any]:
    selection = select_case_ids()
    wanted = {
        case_id: cohort
        for cohort, case_ids in selection.items()
        for case_id in case_ids
    }
    cases = [
        row
        for row in read_json(FULL225_CASES)["cases"]
        if str(row["original_case_id"]) in wanted
    ]
    if len(cases) != 18:
        raise ValueError(f"expected 18 V16 cases, got {len(cases)}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    prepared = []
    for row in cases:
        start = int(row["target_start"])
        length = int(row["segment_tokens"])
        segment_ids = [
            int(value)
            for value in row["target_input_ids"][start : start + length]
        ]
        source_start = int(row["source_start"])
        source_segment = row["source_input_ids"][
            source_start : source_start + length
        ]
        if list(source_segment) != segment_ids:
            raise ValueError(f"{row['original_case_id']}: segment mismatch")
        prepared.append(
            {
                **row,
                "cohort": wanted[str(row["original_case_id"])],
                "repair_masks": semantic_masks(
                    case_id=str(row["original_case_id"]),
                    segment_ids=segment_ids,
                    tokenizer=tokenizer,
                ),
            }
        )
    return {"cases": prepared, "selection": selection}


def register(output: Path) -> dict[str, Any]:
    path = output / "V16_REGISTRATION.json"
    if path.exists():
        value = read_json(path)
        if value["inputs"]["full225_cases_sha256"] != sha256_file(
            FULL225_CASES
        ):
            raise ValueError("registered full-225 cases changed")
        return value
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    cases = prepare_cases()
    cases_path = output / "V16_CASES.json"
    write_json(cases_path, cases)
    value = {
        "date": "2026-07-27",
        "experiment": "V16 equal-budget behavioral-contract KV repair probe",
        "registered_before_gpu": True,
        "motivation": (
            "V15 found stable CacheBlend damage but shallow length/workflow "
            "features reached only about 12% damage precision. Test a causal "
            "coding-semantic mask instead of encoding that weak classifier."
        ),
        "hypothesis": (
            "At an identical 32-token target-KV repair budget, online-visible "
            "behavioral clauses and boundary conditions reduce Dense-reference "
            "KL more than head, tail, deterministic random, and function-"
            "signature repair."
        ),
        "variants": list(VARIANTS),
        "protocol": {
            "cases": 18,
            "cohorts": {"historical_damage": 9, "matched_safe": 9},
            "continuation_tokens": CONTINUATION_TOKENS,
            "repair_budget_tokens": REPAIR_BUDGET,
            "selection_labels_used_only_for_diagnostic_cohort": True,
            "repair_mask_uses_outcome_labels": False,
            "repair_mask_online_inputs": [
                "tokenized shared coding segment",
                "Task marker",
                "behavioral condition and negation clauses",
                "examples, operators, and boundary literals",
            ],
            "truth_or_evaluator_tests_read": False,
            "prefetch": False,
        },
        "frozen_gates": {
            "behavior_mean_kl_below_each_equal_budget_control": True,
            "behavior_damage_cohort_kl_reduction_vs_full_copy_min": 0.20,
            "behavior_top1_not_below_best_equal_budget_control": True,
            "signature_is_negative_control_not_selection_target": True,
            "advance_only_to_new_full225_preregistration": True,
        },
        "inputs": {
            "cacheblend_dense_sha256": sha256_file(CACHEBLEND_DENSE),
            "cacheblend_reuse_sha256": sha256_file(CACHEBLEND_REUSE),
            "full225_cases_sha256": sha256_file(FULL225_CASES),
            "probe_source_sha256": sha256_file(Path(__file__)),
            "v15_repeat_result_sha256": sha256_file(
                V15_OUTPUT / "V15_REPEAT_RESULT.json"
            ),
            "v16_cases_sha256": sha256_file(cases_path),
        },
        "protected": {
            "existing_preregistration_thresholds_modified": False,
            "old_dirty_checkout_modified": False,
            "paper_modified": False,
            "prefetch": False,
        },
        "scope": (
            "Development-only causal probe on exposed cases. It cannot make a "
            "functional-accuracy claim or select itself for the final test."
        ),
        "status": "REGISTERED_BEFORE_V16_GPU",
    }
    write_json(path, value)
    return value


def _masked_cache(
    *,
    model: Any,
    source_cache: list[tuple[torch.Tensor, torch.Tensor]],
    target_cache: list[tuple[torch.Tensor, torch.Tensor]],
    source_start: int,
    target_start: int,
    length: int,
    repair_indices: list[int],
    dense_replay: bool,
) -> DynamicCache:
    mask = torch.zeros(length, dtype=torch.bool, device="cuda")
    if dense_replay:
        mask[:] = True
    elif repair_indices:
        mask[torch.tensor(repair_indices, device="cuda")] = True
    mask = mask.view(1, 1, length, 1)
    rows = []
    delta = target_start - source_start
    for (source_k, source_v), (target_k, target_v) in zip(
        source_cache, target_cache, strict=True
    ):
        source_shared_k = source_k[
            :, :, source_start : source_start + length, :
        ]
        rotated = _rotated_source_keys(source_shared_k, delta)
        rotated = (
            rotated.permute(1, 0, 2).unsqueeze(0).to(dtype=target_k.dtype)
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
        mixed_k = torch.where(mask, target_shared_k, rotated)
        mixed_v = torch.where(mask, target_shared_v, source_shared_v)
        rows.append(
            (
                torch.cat((target_k[:, :, :target_start, :], mixed_k), dim=2),
                torch.cat((target_v[:, :, :target_start, :], mixed_v), dim=2),
            )
        )
    return DynamicCache(ddp_cache_data=rows, config=model.config)


def measure(output: Path, canary: bool) -> dict[str, Any]:
    register(output)
    destination = (
        output / "canary/V16_MEASUREMENTS.json"
        if canary
        else output / "V16_MEASUREMENTS.json"
    )
    if destination.exists():
        return {"status": "already_complete"}
    cases = read_json(output / "V16_CASES.json")["cases"]
    if canary:
        cases = [
            next(row for row in cases if row["cohort"] == cohort)
            for cohort in ("damage", "matched_safe")
        ]
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
                cache = _masked_cache(
                    model=model,
                    source_cache=source_cache,
                    target_cache=target_cache,
                    source_start=int(case["source_start"]),
                    target_start=target_start,
                    length=length,
                    repair_indices=case["repair_masks"].get(variant, []),
                    dense_replay=variant == "dense_replay",
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
                        "repair_fraction": (
                            1.0
                            if variant == "dense_replay"
                            else len(case["repair_masks"].get(variant, []))
                            / length
                        ),
                        "repair_tokens": (
                            length
                            if variant == "dense_replay"
                            else len(case["repair_masks"].get(variant, []))
                        ),
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


def summarize(output: Path) -> dict[str, Any]:
    registration = register(output)
    rows = read_json(output / "V16_MEASUREMENTS.json")["rows"]
    arms = {}
    for cohort in ("damage", "matched_safe", "all"):
        subset = rows if cohort == "all" else [
            row for row in rows if row["cohort"] == cohort
        ]
        arms[cohort] = {}
        for variant in VARIANTS:
            values = [row for row in subset if row["variant"] == variant]
            arms[cohort][variant] = {
                "mean_kl": statistics.mean(row["kl_mean"] for row in values),
                "mean_nll": statistics.mean(row["nll"] for row in values),
                "mean_repair_fraction": statistics.mean(
                    row["repair_fraction"] for row in values
                ),
                "mean_top1_agreement": statistics.mean(
                    row["top1_agreement"] for row in values
                ),
            }
    behavior = arms["all"]["behavior32"]
    controls = ("random32", "head32", "tail32", "signature32")
    damage_full = arms["damage"]["full_copy"]["mean_kl"]
    damage_behavior = arms["damage"]["behavior32"]["mean_kl"]
    reduction = (damage_full - damage_behavior) / max(damage_full, 1e-12)
    best_control_top1 = max(
        arms["all"][variant]["mean_top1_agreement"] for variant in controls
    )
    gates = registration["frozen_gates"]
    verdict = {
        "behavior_kl_below_each_control": all(
            behavior["mean_kl"] < arms["all"][variant]["mean_kl"]
            for variant in controls
        ),
        "damage_kl_reduction_passed": (
            reduction
            >= gates["behavior_damage_cohort_kl_reduction_vs_full_copy_min"]
        ),
        "top1_passed": (
            behavior["mean_top1_agreement"] >= best_control_top1
        ),
    }
    result = {
        "arms": arms,
        "damage_behavior_kl_reduction_vs_full_copy": reduction,
        "selected_for_full225_preregistration": all(verdict.values()),
        "status": "V16_COMPLETE",
        "verdict": verdict,
    }
    write_json(output / "V16_RESULT.json", result)
    return result


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
