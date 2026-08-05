#!/usr/bin/env python3
"""Test whether task-conditioned coding signals improve V46 island selection.

This is a bounded motivation experiment, not a V47 runtime implementation.
All reuse arms receive the same RepoBench-P target prompt and copy exactly
three 512-token middle islands.  Only the selector changes:

* ``v46_recency_m47``: V46's current length/recency rule;
* ``coding_symbol_overlap_m47``: answer-blind identifier overlap between the
  code immediately before the completion cursor and repository observations;
* ``matched_random_m47``: a frozen seeded random selector.

Dense is measured in the same run.  Ordinary Radix reuse and prefetch are off.
"""

from __future__ import annotations

import argparse
import collections
import difflib
import hashlib
import json
import math
import random
import re
import statistics
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from benchmark.multi_workflow.run_bridge_reuse_pilot import (
    manifest_case,
    write_json,
)
from benchmark.multi_workflow.run_v40_repobench_control import (
    DEFAULT_WORKLOAD,
    MODEL,
    SOURCE_PREFIX,
    SOURCE_SUFFIX,
    _prediction_line,
    _render,
    _text_span,
    run_arm,
)


ROOT = Path("/home/gfy/CodeMAS_Project")
DEFAULT_OUTPUT = (
    ROOT
    / "kvflow-artifacts/impactkv_m47_task_conditioned_pool_20260805"
    / "canary12"
)
ARMS = (
    "v46_recency_m47",
    "coding_symbol_overlap_m47",
    "matched_random_m47",
)
ISLANDS_PER_CASE = 3
TOKENS_PER_ISLAND = 512
RANDOM_SEED = 20260805
QUERY_TAIL_CHARS = 2048
IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
STOP_IDENTIFIERS = {
    "and",
    "args",
    "class",
    "context",
    "def",
    "dict",
    "elif",
    "else",
    "false",
    "for",
    "from",
    "get",
    "has",
    "import",
    "int",
    "kwargs",
    "list",
    "none",
    "not",
    "raise",
    "result",
    "return",
    "self",
    "set",
    "str",
    "that",
    "the",
    "this",
    "true",
    "value",
    "while",
    "with",
    "yield",
}


def _identifier_counts(text: str) -> collections.Counter[str]:
    return collections.Counter(
        token.lower()
        for token in IDENTIFIER_RE.findall(text)
        if token.lower() not in STOP_IDENTIFIERS
    )


def coding_symbol_scores(
    query_text: str,
    candidate_texts: list[str],
) -> list[float]:
    """Return answer-blind cursor-local identifier relevance scores.

    The scorer deliberately uses only text present before the missing line.
    IDF is computed within each case so common repository-wide identifiers do
    not dominate rarer function/class/API names near the completion cursor.
    """

    query = _identifier_counts(query_text[-QUERY_TAIL_CHARS:])
    documents = [_identifier_counts(text) for text in candidate_texts]
    document_frequency = collections.Counter(
        token for document in documents for token in document
    )
    count = len(documents)
    scores = []
    for document in documents:
        score = 0.0
        for token, query_frequency in query.items():
            if token not in document:
                continue
            inverse_frequency = (
                math.log((count + 1) / (document_frequency[token] + 1)) + 1
            )
            score += (
                (1 + math.log(query_frequency))
                * inverse_frequency
                * (1 + math.log(document[token]))
            )
        scores.append(score)
    return scores


def _stable_case_seed(case_id: str) -> int:
    digest = hashlib.sha256(
        f"{RANDOM_SEED}:{case_id}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], byteorder="little")


def _select_indices(
    arm: str,
    case_id: str,
    candidates: list[dict[str, Any]],
    query_text: str,
) -> tuple[list[int], list[float] | None]:
    if len(candidates) < ISLANDS_PER_CASE:
        raise ValueError(f"{case_id}: fewer than three matched candidates")
    if arm == "v46_recency_m47":
        selected = sorted(
            range(len(candidates)),
            key=lambda index: candidates[index]["context_index"],
            reverse=True,
        )[:ISLANDS_PER_CASE]
        return selected, None
    if arm == "coding_symbol_overlap_m47":
        scores = coding_symbol_scores(
            query_text,
            [str(candidate["text"]) for candidate in candidates],
        )
        selected = sorted(
            range(len(candidates)),
            key=lambda index: (
                scores[index],
                candidates[index]["context_index"],
            ),
            reverse=True,
        )[:ISLANDS_PER_CASE]
        return selected, scores
    if arm == "matched_random_m47":
        generator = random.Random(_stable_case_seed(case_id))
        return generator.sample(range(len(candidates)), ISLANDS_PER_CASE), None
    raise ValueError(arm)


def _candidate_spans(
    *,
    source_prompt: str,
    source_ids: list[int],
    source_offsets: list[tuple[int, int]],
    target_prompt: str,
    target_ids: list[int],
    target_offsets: list[tuple[int, int]],
    reusable: list[str],
) -> list[dict[str, Any]]:
    candidates = []
    for context_index, text in enumerate(reusable):
        source_span = _text_span(source_prompt, source_offsets, text)
        target_span = _text_span(target_prompt, target_offsets, text)
        if source_span is None or target_span is None:
            continue
        source_start, source_length = source_span
        target_start, target_length = target_span
        if source_length != target_length or source_length < TOKENS_PER_ISLAND:
            continue
        offset = source_length - TOKENS_PER_ISLAND
        span = {
            "source_start": source_start + offset,
            "target_start": target_start + offset,
            "length": TOKENS_PER_ISLAND,
            "context_index": context_index,
            "text": text,
        }
        if (
            source_ids[
                span["source_start"] : span["source_start"]
                + TOKENS_PER_ISLAND
            ]
            != target_ids[
                span["target_start"] : span["target_start"]
                + TOKENS_PER_ISLAND
            ]
        ):
            continue
        candidates.append(span)
    return candidates


def prepare_case(
    tokenizer: Any,
    case: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    reusable = [
        str(segment["text"])
        for segment in case["segments"]
        if bool(segment["reusable"])
    ]
    query_segments = [
        str(segment["text"])
        for segment in case["segments"]
        if not bool(segment["reusable"])
    ]
    if len(query_segments) != 1:
        raise ValueError(f"{case['case_id']}: expected one query segment")
    source_messages = [
        {
            "role": "system",
            "content": (
                "You inspect repository code returned by read-only tools. "
                "Return only a one-word acknowledgement."
            ),
        },
        {
            "role": "user",
            "content": SOURCE_PREFIX + "".join(reusable) + SOURCE_SUFFIX,
        },
    ]
    source_prompt, source_ids, source_offsets = _render(
        tokenizer, source_messages
    )
    target_prompt, target_ids, target_offsets = _render(
        tokenizer, case["messages"]
    )
    candidates = _candidate_spans(
        source_prompt=source_prompt,
        source_ids=source_ids,
        source_offsets=source_offsets,
        target_prompt=target_prompt,
        target_ids=target_ids,
        target_offsets=target_offsets,
        reusable=reusable,
    )
    selections = {}
    for arm in ARMS:
        selected_indices, scores = _select_indices(
            arm,
            str(case["case_id"]),
            candidates,
            query_segments[0],
        )
        selected = sorted(
            (candidates[index] for index in selected_indices),
            key=lambda row: row["target_start"],
        )
        selections[arm] = {
            "candidate_scores": (
                [
                    {
                        "context_index": candidate["context_index"],
                        "score": scores[index],
                    }
                    for index, candidate in enumerate(candidates)
                ]
                if scores is not None
                else None
            ),
            "selected_context_indices": [
                row["context_index"] for row in selected
            ],
            "selected_islands": [
                {
                    key: int(row[key])
                    for key in ("source_start", "target_start", "length")
                }
                for row in selected
            ],
            "selected_tokens": sum(row["length"] for row in selected),
        }
    base_case = {
        "answers": list(case["metadata"]["answers"]),
        "case_id": str(case["case_id"]),
        "max_new_tokens": int(case["max_new_tokens"]),
        "source_input_ids": source_ids,
        "target_input_ids": target_ids,
    }
    audit = {
        "case_id": str(case["case_id"]),
        "candidate_context_indices": [
            int(row["context_index"]) for row in candidates
        ],
        "selections": selections,
    }
    return base_case, audit


def prepare(workload_path: Path, output: Path, limit: int) -> dict[str, Any]:
    workload = json.loads(workload_path.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    source_cases = workload["cases"][:limit] if limit > 0 else workload["cases"]
    prepared = [prepare_case(tokenizer, case) for case in source_cases]
    cases = [row[0] for row in prepared]
    audits = [row[1] for row in prepared]
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "CASES.json", {"cases": cases})
    write_json(output / "SELECTION_AUDIT.json", {"cases": audits})

    for arm in ARMS:
        manifest_rows = []
        audit_by_id = {row["case_id"]: row for row in audits}
        for case in cases:
            group_id = f"m47-{arm}-{case['case_id']}"
            selection = audit_by_id[case["case_id"]]["selections"][arm]
            for island_index, span in enumerate(selection["selected_islands"]):
                row = manifest_case(
                    case_id=f"{case['case_id']}-i{island_index}",
                    policy_label=arm,
                    source_ids=case["source_input_ids"],
                    target_ids=case["target_input_ids"],
                    span=span,
                )
                row.update(
                    source_id=f"{case['case_id']}-source-i{island_index}",
                    target_group_id=group_id,
                    target_uses=1,
                )
                manifest_rows.append(row)
        write_json(
            output / "manifests" / f"{arm}.json",
            {
                "cache_dtype": "bfloat16",
                "cases": manifest_rows,
                "lease_ttl_s": 900,
                "ledger_path": str(
                    output / "server" / arm / "EXACT_LEDGER.jsonl"
                ),
                "model_id": MODEL,
                "ordinary_prefix_reuse_enabled": False,
                "rope": {
                    "base": 1_000_000,
                    "is_neox_style": True,
                    "rotary_dim": 128,
                },
                "version": 2,
            },
        )

    recency_differences = 0
    random_differences = 0
    recency_symbol_overlap = []
    for row in audits:
        selections = row["selections"]
        recency = set(
            selections["v46_recency_m47"]["selected_context_indices"]
        )
        symbol = set(
            selections["coding_symbol_overlap_m47"][
                "selected_context_indices"
            ]
        )
        matched_random = set(
            selections["matched_random_m47"]["selected_context_indices"]
        )
        recency_differences += recency != symbol
        random_differences += matched_random != symbol
        recency_symbol_overlap.append(len(recency & symbol))
    registration = {
        "status": "REGISTERED_BEFORE_GPU",
        "purpose": (
            "causally test whether cursor-local coding identifiers improve "
            "which V46 observation islands are reused"
        ),
        "dataset": "RepoBench-P",
        "model": MODEL,
        "cases": len(cases),
        "arms": ["dense", *ARMS],
        "frozen_controls": {
            "copied_tokens_per_target": (
                ISLANDS_PER_CASE * TOKENS_PER_ISLAND
            ),
            "islands_per_target": ISLANDS_PER_CASE,
            "ordinary_prefix_reuse": False,
            "prefetch": False,
            "random_seed": RANDOM_SEED,
            "same_target_prompt": True,
        },
        "answer_blind": (
            "selector reads only repository observations and the query text "
            "already visible before the missing line"
        ),
        "quality_scope": (
            "RepoBench next-line exact and code similarity; not functional "
            "task accuracy"
        ),
        "selection_separation": {
            "symbol_differs_from_recency_cases": recency_differences,
            "symbol_differs_from_random_cases": random_differences,
            "mean_symbol_recency_island_overlap": statistics.fmean(
                recency_symbol_overlap
            ),
        },
        "mechanism_gate": {
            "expected_copy_events_per_reuse_arm": (
                len(cases) * ISLANDS_PER_CASE
            ),
            "fallback_events_max": 0,
        },
        "expansion_rule": (
            "expand canary12 to full50 only if every reuse arm has all "
            "expected physical copies, zero fallback, and the symbol selector "
            "differs from recency on at least 8/12 cases"
        ),
    }
    write_json(output / "REGISTRATION.json", registration)
    return registration


def _mean(values: list[float]) -> float:
    return statistics.fmean(values)


def _bootstrap_mean_ci(
    values: list[float],
    *,
    samples: int = 20_000,
) -> list[float]:
    if not values:
        raise ValueError("cannot bootstrap an empty sample")
    generator = random.Random(RANDOM_SEED)
    means = sorted(
        statistics.fmean(generator.choice(values) for _ in values)
        for _ in range(samples)
    )
    return [means[int(0.025 * samples)], means[int(0.975 * samples)]]


def _arm_summary(
    *,
    arm: str,
    cases: dict[str, dict[str, Any]],
    dense: dict[str, dict[str, Any]],
    result: dict[str, Any],
    audit: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    targets = {row["case_id"]: row for row in result["targets"]}
    sources = {row["case_id"]: row for row in result["sources"]}
    rows = []
    for case_id, case in cases.items():
        answer = str(case["answers"][0])
        dense_line = _prediction_line(str(dense[case_id]["output_text"]))
        prediction = _prediction_line(str(targets[case_id]["output_text"]))
        selection = audit[case_id]["selections"][arm]
        selected_islands = selection["selected_islands"]
        prompt_tokens = len(case["target_input_ids"])
        causal_work = sum(
            island["length"]
            * (2 * island["target_start"] + island["length"] - 1)
            / 2
            for island in selected_islands
        )
        dense_causal_work = prompt_tokens * (prompt_tokens + 1) / 2
        dense_gap_tokens = sum(
            max(
                0,
                right["target_start"]
                - (left["target_start"] + left["length"]),
            )
            for left, right in zip(selected_islands, selected_islands[1:])
        )
        rows.append(
            {
                "arm": arm,
                "case_id": case_id,
                "answer": answer,
                "prediction_line": prediction,
                "dense_prediction_line": dense_line,
                "exact_line": prediction.strip() == answer.strip(),
                "dense_exact_line": dense_line.strip() == answer.strip(),
                "code_sim": difflib.SequenceMatcher(
                    None, prediction, answer
                ).ratio(),
                "dense_code_sim": difflib.SequenceMatcher(
                    None, dense_line, answer
                ).ratio(),
                "prediction_identical_to_dense": prediction == dense_line,
                "ttft_ms": float(targets[case_id]["ttft_ms"]),
                "dense_ttft_ms": float(dense[case_id]["ttft_ms"]),
                "source_build_ms": float(sources[case_id]["elapsed_ms"]),
                "selected_context_indices": selection[
                    "selected_context_indices"
                ],
                "selected_islands": selected_islands,
                "mean_selected_position_fraction": _mean(
                    [
                        island["target_start"] / prompt_tokens
                        for island in selected_islands
                    ]
                ),
                "causal_attention_work_proxy_fraction": (
                    causal_work / dense_causal_work
                ),
                "dense_gap_tokens_between_islands": dense_gap_tokens,
            }
        )
    copies = [
        row
        for row in result["ledger_rows"]
        if row.get("event") == "target_copied"
    ]
    fallbacks = [
        row
        for row in result["ledger_rows"]
        if row.get("event") == "target_fallback"
    ]
    dense_ttft = _mean([row["dense_ttft_ms"] for row in rows])
    reuse_ttft = _mean([row["ttft_ms"] for row in rows])
    build = _mean([row["source_build_ms"] for row in rows])
    expected_copies = len(rows) * ISLANDS_PER_CASE
    return (
        {
            "mechanism_ok": (
                len(copies) == expected_copies and len(fallbacks) == 0
            ),
            "copy_events": len(copies),
            "expected_copy_events": expected_copies,
            "fallback_events": len(fallbacks),
            "exact_line": sum(row["exact_line"] for row in rows),
            "code_sim_percent": 100 * _mean([row["code_sim"] for row in rows]),
            "identical_to_dense": sum(
                row["prediction_identical_to_dense"] for row in rows
            ),
            "rescue_vs_dense": sum(
                row["exact_line"] and not row["dense_exact_line"]
                for row in rows
            ),
            "damage_vs_dense": sum(
                row["dense_exact_line"] and not row["exact_line"]
                for row in rows
            ),
            "mean_ttft_ms": reuse_ttft,
            "cache_ready_speedup": dense_ttft / reuse_ttft,
            "mean_source_build_ms": build,
            "n4_including_build_speedup": dense_ttft / (reuse_ttft + build / 4),
            "mean_copied_tokens": ISLANDS_PER_CASE * TOKENS_PER_ISLAND,
            "mean_selected_context_index": _mean(
                [
                    float(index)
                    for row in rows
                    for index in row["selected_context_indices"]
                ]
            ),
            "mean_selected_position_fraction": _mean(
                [row["mean_selected_position_fraction"] for row in rows]
            ),
            "causal_attention_work_proxy_percent": 100
            * _mean(
                [
                    row["causal_attention_work_proxy_fraction"]
                    for row in rows
                ]
            ),
            "mean_dense_gap_tokens_between_islands": _mean(
                [row["dense_gap_tokens_between_islands"] for row in rows]
            ),
        },
        rows,
    )


def summarize(output: Path) -> dict[str, Any]:
    cases = {
        row["case_id"]: row
        for row in json.loads((output / "CASES.json").read_text())["cases"]
    }
    audit = {
        row["case_id"]: row
        for row in json.loads(
            (output / "SELECTION_AUDIT.json").read_text()
        )["cases"]
    }
    dense_value = json.loads((output / "dense.json").read_text())
    dense = {row["case_id"]: row for row in dense_value["targets"]}
    if set(cases) != set(dense):
        raise ValueError("Dense target coverage differs from frozen cases")
    dense_rows = []
    for case_id, case in cases.items():
        answer = str(case["answers"][0])
        prediction = _prediction_line(str(dense[case_id]["output_text"]))
        dense_rows.append(
            {
                "case_id": case_id,
                "exact_line": prediction.strip() == answer.strip(),
                "code_sim": difflib.SequenceMatcher(
                    None, prediction, answer
                ).ratio(),
                "ttft_ms": float(dense[case_id]["ttft_ms"]),
            }
        )
    arm_summaries = {
        "dense": {
            "exact_line": sum(row["exact_line"] for row in dense_rows),
            "code_sim_percent": 100
            * _mean([row["code_sim"] for row in dense_rows]),
            "mean_ttft_ms": _mean([row["ttft_ms"] for row in dense_rows]),
            "cache_ready_speedup": 1.0,
        }
    }
    all_rows = []
    for arm in ARMS:
        result = json.loads((output / f"{arm}.json").read_text())
        summary, rows = _arm_summary(
            arm=arm,
            cases=cases,
            dense=dense,
            result=result,
            audit=audit,
        )
        arm_summaries[arm] = summary
        all_rows.extend(rows)
    symbol = arm_summaries["coding_symbol_overlap_m47"]
    recency = arm_summaries["v46_recency_m47"]
    matched_random = arm_summaries["matched_random_m47"]
    rows_by_arm_case = {
        (row["arm"], row["case_id"]): row for row in all_rows
    }
    case_ids = sorted(cases)
    symbol_minus_recency_sim = [
        100
        * (
            rows_by_arm_case[("coding_symbol_overlap_m47", case_id)][
                "code_sim"
            ]
            - rows_by_arm_case[("v46_recency_m47", case_id)]["code_sim"]
        )
        for case_id in case_ids
    ]
    symbol_minus_random_sim = [
        100
        * (
            rows_by_arm_case[("coding_symbol_overlap_m47", case_id)][
                "code_sim"
            ]
            - rows_by_arm_case[("matched_random_m47", case_id)]["code_sim"]
        )
        for case_id in case_ids
    ]
    recency_minus_symbol_ttft = [
        rows_by_arm_case[("v46_recency_m47", case_id)]["ttft_ms"]
        - rows_by_arm_case[("coding_symbol_overlap_m47", case_id)]["ttft_ms"]
        for case_id in case_ids
    ]
    recency_minus_random_ttft = [
        rows_by_arm_case[("v46_recency_m47", case_id)]["ttft_ms"]
        - rows_by_arm_case[("matched_random_m47", case_id)]["ttft_ms"]
        for case_id in case_ids
    ]
    value = {
        "status": (
            "COMPLETE"
            if all(
                arm_summaries[arm]["mechanism_ok"] for arm in ARMS
            )
            else "MECHANISM_FAILURE"
        ),
        "dataset": "RepoBench-P",
        "samples": len(cases),
        "arms": arm_summaries,
        "selector_effect": {
            "symbol_minus_recency_code_sim_points": (
                symbol["code_sim_percent"] - recency["code_sim_percent"]
            ),
            "symbol_minus_random_code_sim_points": (
                symbol["code_sim_percent"]
                - matched_random["code_sim_percent"]
            ),
            "symbol_minus_recency_exact": (
                symbol["exact_line"] - recency["exact_line"]
            ),
            "symbol_minus_random_exact": (
                symbol["exact_line"] - matched_random["exact_line"]
            ),
            "paired_bootstrap_95_percent_ci": {
                "symbol_minus_recency_code_sim_points": _bootstrap_mean_ci(
                    symbol_minus_recency_sim
                ),
                "symbol_minus_random_code_sim_points": _bootstrap_mean_ci(
                    symbol_minus_random_sim
                ),
                "recency_minus_symbol_ttft_ms": _bootstrap_mean_ci(
                    recency_minus_symbol_ttft
                ),
                "recency_minus_random_ttft_ms": _bootstrap_mean_ci(
                    recency_minus_random_ttft
                ),
            },
            "recency_minus_symbol_ttft_ms": _mean(
                recency_minus_symbol_ttft
            ),
            "recency_minus_random_ttft_ms": _mean(
                recency_minus_random_ttft
            ),
        },
        "interpretation_scope": (
            "This isolates selection at equal 1536-token/three-island reuse. "
            "It is a next-line motivation result, not functional accuracy."
        ),
        "rows": all_rows,
    }
    write_json(output / "RESULT.json", value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--workload", type=Path, default=DEFAULT_WORKLOAD)
    prepare_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    prepare_parser.add_argument("--limit", type=int, default=12)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run_parser.add_argument("--arm", choices=("dense", *ARMS), required=True)
    run_parser.add_argument("--port", type=int, default=31100)
    summary_parser = subparsers.add_parser("summarize")
    summary_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.command == "prepare":
        value = prepare(args.workload, args.output, args.limit)
    elif args.command == "run":
        value = run_arm(
            args.output,
            args.arm,
            args.port,
            reuse_arm=args.arm,
        )
    else:
        value = summarize(args.output)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
